"""One-shot backfill: split historical bench_candidates token totals.

The bench runner previously stored `total_input_tok = input + output` and
left `total_output_tok = 0`. The real per-candidate input/output split is
recoverable from the `transcripts` table by joining through tasks via
`tasks.idempotency_key = 'bench::<candidate_id>::*'`.

Run once after upgrading: `python scripts/backfill_bench_tokens.py`. Safe
to re-run — it's idempotent (overwrites with the reconstructed values).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def backfill(db_path: Path, dry_run: bool) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT bc.id AS cand_id, bc.label, bc.model,
               COALESCE(SUM(tr.input_tokens), 0) AS new_in,
               COALESCE(SUM(tr.output_tokens), 0) AS new_out,
               bc.total_input_tok AS old_in, bc.total_output_tok AS old_out
          FROM bench_candidates bc
          LEFT JOIN tasks tk
                 ON tk.idempotency_key LIKE ('bench::' || bc.id || '::%')
          LEFT JOIN transcripts tr ON tr.task_id = tk.id
         GROUP BY bc.id
        """
    ).fetchall()

    changed = 0
    for r in rows:
        new_in, new_out = int(r["new_in"]), int(r["new_out"])
        old_in, old_out = int(r["old_in"]), int(r["old_out"])
        combined_old = old_in + old_out
        combined_new = new_in + new_out
        # Sanity-check: the old "input_tok" stored input+output, so it should
        # equal the reconstructed combined total (modulo rare cases where a
        # transcript wasn't persisted or a budget settle reservation leaked).
        diff = abs(combined_old - combined_new)
        flag = "" if diff < 50 else f"  [diff={diff}]"
        if old_in == new_in and old_out == new_out:
            continue
        changed += 1
        print(
            f"{r['cand_id']}  {r['label']:<22} {r['model']:<48}  "
            f"OLD in={old_in:>7} out={old_out:>5}  →  "
            f"NEW in={new_in:>7} out={new_out:>5}{flag}"
        )
        if not dry_run:
            conn.execute(
                "UPDATE bench_candidates SET total_input_tok=?, total_output_tok=? WHERE id=?",
                (new_in, new_out, r["cand_id"]),
            )
    if not dry_run:
        conn.commit()
    print(f"\n{changed} candidate rows {'would be' if dry_run else ''}updated.")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/co_scientist.db", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    backfill(args.db, args.dry_run)
