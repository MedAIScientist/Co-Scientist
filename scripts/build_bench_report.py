"""Generate docs/BENCH_RESULTS.md from data/co_scientist.db.

The bench DB tracks every cross-model comparison the user has run. This
script walks `bench_runs` + `bench_candidates` + `bench_matches`, joins
back to each bench's bench-session to pull the actual hypothesis records,
re-scores every hypothesis against every known gold set, and emits one
markdown file with an index + per-bench detail + file paths so a reader
can navigate from a one-line summary down to the raw JSON artifact.

Usage:
    python scripts/build_bench_report.py [--db data/co_scientist.db]
                                         [--out docs/BENCH_RESULTS.md]
                                         [--include-failed]

The output is committed-friendly: no timestamps inside row values, paths
are repo-relative.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from co_scientist.bench.goldset import (  # noqa: E402
    GOLDSETS,
    score_candidate_against_goldset,
)

# --------------------------------------------------------------------------- #
# Data loading

@dataclass
class BenchRow:
    id: str
    created_at: str
    status: str
    research_goal: str
    judge_provider: str
    judge_model: str
    goldset_label: str | None
    goldset_size: int | None
    artifact_path: str | None
    session_id: str | None


@dataclass
class CandRow:
    id: str
    label: str
    provider: str
    model: str
    mode: str
    n_hypotheses: int
    wins: int
    losses: int
    mean_elo: float | None
    top_elo: float | None
    total_cost_usd: float
    total_input_tok: int
    total_output_tok: int
    mean_latency_ms: int | None
    gold_hits: int
    gold_hit_names: list[str]
    error: str | None


def _load_benches(con: sqlite3.Connection) -> list[BenchRow]:
    rows = con.execute(
        """SELECT br.id, br.created_at, br.status, br.research_goal,
                  br.judge_provider, br.judge_model, br.goldset_label,
                  br.goldset_size, br.artifact_path,
                  (SELECT s.id FROM sessions s
                     WHERE json_extract(s.config_snapshot, '$.bench_id') = br.id
                     LIMIT 1) AS session_id
             FROM bench_runs br
            ORDER BY br.created_at"""
    ).fetchall()
    return [BenchRow(**dict(r)) for r in rows]


def _load_candidates(con: sqlite3.Connection, bench_id: str) -> list[CandRow]:
    rows = con.execute(
        """SELECT id, label, provider, model, mode,
                  n_hypotheses, wins, losses, mean_elo, top_elo,
                  total_cost_usd, total_input_tok, total_output_tok,
                  mean_latency_ms,
                  gold_hits, gold_hit_names, error
             FROM bench_candidates
            WHERE bench_id=?
            ORDER BY (mean_elo IS NULL), mean_elo DESC, label""",
        (bench_id,),
    ).fetchall()
    out: list[CandRow] = []
    for r in rows:
        d = dict(r)
        hit_names_json = d.pop("gold_hit_names", None) or "[]"
        try:
            d["gold_hit_names"] = json.loads(hit_names_json)
        except json.JSONDecodeError:
            d["gold_hit_names"] = []
        out.append(CandRow(**d))
    return out


def _load_session_hypotheses(session_id: str | None) -> list[dict]:
    """Read every hypothesis record artifact for a bench's session."""
    if session_id is None:
        return []
    pat = REPO_ROOT / "data" / "artifacts" / session_id / "hypotheses" / "*.json"
    out: list[dict] = []
    for p in sorted(glob.glob(str(pat))):
        try:
            with open(p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rec = d.get("record") or {}
        rec.setdefault("id", Path(p).stem)
        rec["_artifact_path"] = os.path.relpath(p, REPO_ROOT)
        rec["_mode"] = d.get("mode", "pipeline")
        out.append(rec)
    return out


def _rescore_all_goldsets(records: list[dict]) -> dict[str, list[str]]:
    """Returns {goldset_label: [matched_entity_names]}."""
    out: dict[str, list[str]] = {}
    for label, gs in GOLDSETS.items():
        agg = score_candidate_against_goldset(records, gs)
        out[label] = sorted(agg)
    return out


def _winning_candidate_for_hyp(
    con: sqlite3.Connection, bench_id: str, hyp_record: dict
) -> list[str]:
    """Find candidate labels associated with a hypothesis via the
    bench_matches table. `bench_matches.hyp_*_text` stores the hypothesis
    *summary* (not the title), truncated to 4000 chars, so we search by
    the statement prefix instead of the title."""
    statement = (hyp_record.get("statement") or hyp_record.get("summary")
                 or hyp_record.get("title") or "")
    if not statement:
        return []
    # Use the first ~50 chars of the statement as the search key. Too few
    # and we get false positives; too many and tokenizer differences (e.g.
    # the persistence layer mid-sentence-split) cause misses.
    needle = f"%{statement[:50].strip()}%"
    rows = con.execute(
        """SELECT DISTINCT bc_a.label AS label_a, bc_a.mode AS mode_a,
                  bc_b.label AS label_b, bc_b.mode AS mode_b,
                  CASE WHEN bm.hyp_a_text LIKE ? THEN 'a'
                       WHEN bm.hyp_b_text LIKE ? THEN 'b'
                       ELSE NULL END AS side
             FROM bench_matches bm
             JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a
             JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b
            WHERE bm.bench_id = ?
              AND (bm.hyp_a_text LIKE ? OR bm.hyp_b_text LIKE ?)""",
        (needle, needle, bench_id, needle, needle),
    ).fetchall()

    def _strip_mode_suffix(label: str) -> str:
        # Vs-raw presets append "[pipe]" / "[raw]" to the candidate label
        # for readability in the runtime table; here we render it cleanly.
        for suffix in ("[pipe]", "[raw]"):
            if label.endswith(suffix):
                return label[: -len(suffix)]
        return label

    out: set[str] = set()
    for r in rows:
        side = r["side"]
        if side == "a":
            lbl, mode = _strip_mode_suffix(r["label_a"]), r["mode_a"]
        elif side == "b":
            lbl, mode = _strip_mode_suffix(r["label_b"]), r["mode_b"]
        else:
            continue
        out.add(f"{lbl} ({mode})")
    return sorted(out)


# --------------------------------------------------------------------------- #
# Markdown rendering

def _fmt_usd(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:.4f}"


def _fmt_ms(x: int | None) -> str:
    if x is None:
        return "—"
    if x < 1000:
        return f"{x}ms"
    return f"{x/1000:.1f}s"


def _fmt_elo(x: float | None) -> str:
    return f"{x:.0f}" if x is not None else "—"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _bench_section(con: sqlite3.Connection, b: BenchRow) -> str:
    cands = _load_candidates(con, b.id)
    records = _load_session_hypotheses(b.session_id)
    rescore = _rescore_all_goldsets(records)
    total_cost = sum(c.total_cost_usd for c in cands)
    n_matches = con.execute(
        "SELECT COUNT(*) FROM bench_matches WHERE bench_id=?", (b.id,)
    ).fetchone()[0]

    lines: list[str] = []
    lines.append(f"## Bench `{b.id}`\n")
    lines.append(f"- **Created:** {b.created_at}")
    lines.append(f"- **Status:** {b.status}")
    lines.append(f"- **Judge:** `{b.judge_provider}:{b.judge_model}`")
    lines.append(f"- **Gold set at runtime:** `{b.goldset_label or '(none)'}`"
                 + (f" (size {b.goldset_size})" if b.goldset_size else ""))
    lines.append(f"- **Total cost:** {_fmt_usd(total_cost)}")
    lines.append(f"- **Matches played:** {n_matches}")
    if b.session_id:
        lines.append(f"- **Session:** `{b.session_id}`")
    if b.artifact_path:
        lines.append(f"- **Bench artifact:** `{b.artifact_path}`")
    lines.append("")
    lines.append(f"**Goal:**\n\n> {b.research_goal.replace(chr(10), ' ')[:600]}"
                 + ("…" if len(b.research_goal) > 600 else ""))
    lines.append("")

    # Per-candidate table
    if cands:
        lines.append("### Candidates")
        lines.append("")
        headers = ["label", "mode", "n_hyps", "W-L", "Elo",
                   "hits (runtime)", "$", "tokens (in / out)", "p50", "note"]
        rows = []
        for c in cands:
            note = (c.error or "")[:60]
            tokens_cell = (
                f"{c.total_input_tok:,} / {c.total_output_tok:,}"
                if (c.total_input_tok or c.total_output_tok) else "—"
            )
            rows.append([
                f"`{c.label}`",
                c.mode or "pipeline",
                str(c.n_hypotheses),
                f"{c.wins}-{c.losses}" if (c.wins or c.losses) else "—",
                _fmt_elo(c.mean_elo),
                f"{c.gold_hits}/{b.goldset_size or '—'}",
                _fmt_usd(c.total_cost_usd),
                tokens_cell,
                _fmt_ms(c.mean_latency_ms),
                note,
            ])
        lines.append(_md_table(headers, rows))
        lines.append("")

    # Hypotheses produced
    if records:
        lines.append(f"### Hypotheses surfaced ({len(records)} total)")
        lines.append("")
        for r in records:
            title = (r.get("title") or "(no title)")[:120]
            statement = (r.get("statement") or r.get("summary") or "")[:240]
            cands_for = _winning_candidate_for_hyp(con, b.id, r)
            who = ", ".join(f"`{c}`" for c in cands_for) if cands_for else "_(no match table entry)_"
            lines.append(f"- **{title}** — via {who}")
            if statement:
                lines.append(f"  - {statement}")
            mode = r.get("_mode") or "pipeline"
            art = r.get("_artifact_path", "")
            lines.append(f"  - mode: `{mode}` · artifact: [`{art}`]({art})")
        lines.append("")
    else:
        lines.append("_No hypotheses produced (every candidate failed)._")
        lines.append("")

    # Cross-goldset rescore
    if rescore:
        lines.append("### Recall across known gold sets (post-hoc rescore)")
        lines.append("")
        for gs_label, hits in rescore.items():
            gs_size = len(GOLDSETS[gs_label].entities)
            marker = "✅" if hits else "·"
            lines.append(f"- {marker} `{gs_label}` ({gs_size} entities): "
                         f"**{len(hits)}/{gs_size}** → "
                         f"{', '.join(hits) if hits else '_none_'}")
        lines.append("")

    # Files pointer
    if b.session_id:
        ses = b.session_id
        lines.append("### Files")
        lines.append("")
        lines.append(f"- Hypotheses (all `record_hypothesis` payloads): "
                     f"`data/artifacts/{ses}/hypotheses/`")
        lines.append(f"- LLM transcripts (request + response per call): "
                     f"`data/artifacts/{ses}/transcripts/generation/`")
        if b.artifact_path:
            lines.append(f"- Bench summary JSON (per-candidate "
                         f"`gold_hit_detail` with alias / field / hyp): "
                         f"`{b.artifact_path}`")
        lines.append("")
        lines.append("**SQL to inspect this bench:**")
        lines.append("")
        lines.append("```sql")
        lines.append("-- per-candidate detail")
        lines.append("SELECT label, mode, n_hypotheses, wins, losses,")
        lines.append("       round(mean_elo,0), gold_hits, gold_hit_names,")
        lines.append("       round(total_cost_usd, 4),")
        lines.append("       total_input_tok, total_output_tok")
        lines.append("  FROM bench_candidates")
        lines.append(f" WHERE bench_id='{b.id}';")
        lines.append("")
        lines.append("-- every match with judge rationale")
        lines.append("SELECT bc_a.label, bc_b.label, bm.winner,")
        lines.append("       round(bm.judge_cost_usd, 4),")
        lines.append("       substr(bm.rationale, 1, 200)")
        lines.append("  FROM bench_matches bm")
        lines.append("  JOIN bench_candidates bc_a ON bc_a.id = bm.cand_a")
        lines.append("  JOIN bench_candidates bc_b ON bc_b.id = bm.cand_b")
        lines.append(f" WHERE bm.bench_id='{b.id}';")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _index_section(con: sqlite3.Connection, benches: list[BenchRow]) -> str:
    """One-line summary of every recorded bench, linkable to the detail section."""
    lines = ["## Index of recorded benches", ""]
    headers = ["bench", "created", "preset / kind", "n_cand", "n_matches",
               "total $", "goldset", "hits"]
    rows = []
    for b in benches:
        cands = _load_candidates(con, b.id)
        records = _load_session_hypotheses(b.session_id)
        rescore = _rescore_all_goldsets(records)
        total = sum(c.total_cost_usd for c in cands)
        n_matches = con.execute(
            "SELECT COUNT(*) FROM bench_matches WHERE bench_id=?", (b.id,)
        ).fetchone()[0]
        # Pick the runtime gold set's hit count if available; else the best.
        gs_hits = "—"
        if b.goldset_label and b.goldset_label in rescore:
            n = len(rescore[b.goldset_label])
            size = len(GOLDSETS[b.goldset_label].entities)
            gs_hits = f"{n}/{size}"
        rows.append([
            f"[`{b.id[:24]}…`](#bench-{b.id.lower()})",
            b.created_at[:19] + "Z",
            _guess_preset(b),
            str(len(cands)),
            str(n_matches),
            _fmt_usd(total),
            f"`{b.goldset_label or '—'}`",
            gs_hits,
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def _headline_findings_section() -> str:
    """Static narrative summary of what the benches have shown so far.

    Kept in the generator (not appended manually to the doc) so a fresh
    regenerate doesn't lose the conclusions when the index re-renders.
    Update this block if a new bench changes the headline finding.
    """
    return "\n".join([
        "## Headline findings",
        "",
        "Across the AML drug-repurposing benches run on this codebase:",
        "",
        "### 1. The strict no-prior-evidence prompt is genuinely hard",
        "",
        "Models default to well-known AML repurposing candidates (Auranofin, "
        "Itraconazole, Venetoclax, Riluzole) that **violate** the "
        "no-prior-evidence constraint. Across 13 hypotheses produced under "
        "the strict prompt, only **one** matched the paper's broader 5-drug "
        "list (Pacritinib, via `claude-opus-4.7 (direct)` in "
        "`frontier-aml-vs-raw`) and **none** matched the strict top-3 "
        "(Nanvuranlat, KIRA6, Leflunomide).",
        "",
        "### 2. Pipeline-vs-raw: the harness's value-add depends on the model",
        "",
        "The `*-vs-raw` presets run each candidate model **twice** — once "
        "through the full Generation pipeline (literature tools + tool loop + "
        "dedup), once as a single forced-tool LM call. Across the paper "
        "baselines (`paper-aml-vs-raw`), **direct mode beat pipeline mode for "
        "every model that produced hypotheses in both modes**:",
        "",
        "| model | pipeline | direct | direct beats pipeline? |",
        "| --- | --- | --- | --- |",
        "| openai-o1 | 0-5 (Elo 1128) | 3-2 (Elo 1213) | yes |",
        "| gemini-2-pro | 2-3 (Elo 1183) | 5-0 (Elo 1274) | yes (decisive) |",
        "| gemini-2-flash-thinking | 0 hyps | 1-4 (Elo 1158) | yes — pipeline failed |",
        "| claude-haiku-4.5 | 0 hyps | 4-1 (Elo 1244) | yes — pipeline failed |",
        "",
        "Same pattern earlier on `gemini-3-flash-preview` (a one-model "
        "vs-raw run): pipeline went 4-2 vs raw 5-1 — within noise on Elo, "
        "but pipeline was **8x more expensive** ($0.0275 vs $0.0033) and "
        "**3x slower** (14.9s vs 5.0s).",
        "",
        "Smaller / older models therefore tend to be *hurt* by the harness "
        "on this task — the tool-loop adds cost and failure modes without "
        "improving the rated hypothesis. The two cases where pipeline did "
        "win were `gpt-4o` (3-3 vs 0-6 for its own direct) and the gemini "
        "near-tie noted above; both with stronger reasoning-tuned base "
        "models.",
        "",
        "### 3. Frontier models need looser caps to use the pipeline",
        "",
        "In `frontier-aml-vs-raw` **all 4 pipeline modes failed** "
        "(claude-opus-4.7 burned $0.83 of $1.50 cap on its first call; "
        "gpt-5, gemini-3-pro, gemini-3-flash all exhausted their tool "
        "loops). Only 2 of 4 direct modes produced usable output. The strict "
        "AML prompt + 8-iteration tool-loop cap + per-candidate budget cap "
        "is too tight for current frontier models — they want more headroom "
        "before they'll emit `record_hypothesis`.",
        "",
        "### Practical implications",
        "",
        "- On a hard, well-defined task, **the cheapest baseline is "
        "  `--candidate model@direct` with a small per-candidate budget**. "
        "  The pipeline is worth its cost only when the model is strong "
        "  enough to use the literature tools productively *within* the "
        "  iteration cap.",
        "- Reproducing the paper's specific picks (Nanvuranlat / KIRA6 / "
        "  Leflunomide, or the broader 5) needs **more breadth** — the "
        "  paper surfaced these after running 15 expert-curated goals + "
        "  the full system's iterative refinement, not from a single "
        "  Generation call.",
        "- Budget caps matter. Tight per-candidate caps mask quality "
        "  questions behind admission failures. For expensive models (Opus, "
        "  o1) `--budget-per-candidate 2.0` is the floor on this prompt; "
        "  `--n 5+` with multiple seeds is the floor for stable recall "
        "  numbers.",
        "",
        "",
    ])


def _guess_preset(b: BenchRow) -> str:
    """Heuristic: derive a short preset label from goldset + candidate count."""
    goal = (b.research_goal or "").lower()
    if "aml" in goal or "leukemia" in goal:
        return "AML repurposing"
    if "microbiome" in goal:
        return "microbiome smoke"
    return "custom"


def build_report(db_path: Path, out_path: Path) -> None:
    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    benches = _load_benches(con)
    sections = [
        "# Bench results",
        "",
        "Live results from every cross-model bench run on this codebase. "
        "See [`../README.md`](../README.md) for what the bench is and how to "
        "run it.",
        "",
        f"_Auto-generated from `{os.path.relpath(db_path, REPO_ROOT)}` by_ "
        "_`python scripts/build_bench_report.py`._ "
        "_Re-run after any new `co-scientist bench` to refresh._",
        "",
        "## How to read this doc",
        "",
        "1. **Index** below lists every bench ever run on this machine, "
        "one row per bench. Click a bench-id link to jump to its detail.",
        "2. **Per-bench detail** shows, for each bench:",
        "   - the goal it was given,",
        "   - the candidate result table (Elo, hits, $),",
        "   - **every hypothesis the bench produced** with its full statement,",
        "     attributed to the model that produced it (from the bench-match table),",
        "   - **post-hoc rescore** against every registered gold set — so a bench "
        "that ran with `aml-repurposing-paper-top3` at the time can still show "
        "whether any hypothesis would have hit the broader "
        "`aml-repurposing-paper-5` list, and vice versa,",
        "   - **file pointers** for the artifacts on disk + ready-to-run SQL "
        "for the raw DB rows.",
        "",
        f"**Total benches:** {len(benches)} · "
        f"**With gold-set scoring:** "
        f"{sum(1 for b in benches if b.goldset_label)}",
        "",
        _headline_findings_section(),
        _index_section(con, benches),
        "## Per-bench detail",
        "",
    ]
    for b in benches:
        # GitHub anchors lowercase the heading; add the explicit lowercased id
        # as a marker so the index links work.
        sections.append(f'<a id="bench-{b.id.lower()}"></a>')
        sections.append(_bench_section(con, b))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, {len(benches)} benches)")


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(REPO_ROOT / "data" / "co_scientist.db"))
    p.add_argument("--out", default=str(REPO_ROOT / "docs" / "BENCH_RESULTS.md"))
    args = p.parse_args()
    build_report(Path(args.db), Path(args.out))


if __name__ == "__main__":
    _main()
