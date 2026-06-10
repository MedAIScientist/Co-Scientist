"""Tests for the termination predicate + StabilityTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from co_scientist.config import Config
from co_scientist.models import ResearchPlan, Session
from co_scientist.orchestrator.termination import (
    EloSnapshot,
    StabilityTracker,
    StopReason,
    budget_exceeded,
    should_stop,
    wall_clock_exceeded,
)


def _session(
    *,
    used_usd: float = 0.0,
    budget_usd: float = 25.0,
    used_tokens: int = 0,
    budget_tokens: int = 1_000_000,
    deadline_in_s: float | None = None,
) -> Session:
    now = datetime.now(UTC)
    deadline = now + timedelta(seconds=deadline_in_s) if deadline_in_s is not None else None
    return Session(
        id="ses_t", created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="o"),
        config_snapshot={}, budget_tokens=budget_tokens, budget_usd=budget_usd,
        budget_used_tokens=used_tokens, budget_used_usd=used_usd,
        wall_deadline=deadline,
    )


# ----------------------------- budget / wall ----------------------------- #

def test_budget_exceeded_by_usd() -> None:
    assert not budget_exceeded(_session(used_usd=10, budget_usd=25))
    assert budget_exceeded(_session(used_usd=25, budget_usd=25))
    assert budget_exceeded(_session(used_usd=30, budget_usd=25))


def test_budget_exceeded_by_tokens() -> None:
    assert budget_exceeded(_session(used_tokens=2_000_000, budget_tokens=1_000_000))
    assert not budget_exceeded(_session(used_tokens=500_000, budget_tokens=1_000_000))


def test_wall_clock_exceeded() -> None:
    assert not wall_clock_exceeded(_session(deadline_in_s=60))
    assert wall_clock_exceeded(_session(deadline_in_s=-60))


def test_external_stop_wins_over_progress() -> None:
    cfg = Config()
    tracker = StabilityTracker(k=3, n=3, eps=25)
    assert should_stop(cfg, _session(), tracker, external_stop=True) is StopReason.EXTERNAL


# ----------------------------- stability tracker ----------------------------- #

def _snap(
    match_count: int,
    ids: list[str],
    elos: list[float],
    pool_size: int = 0,
) -> EloSnapshot:
    return EloSnapshot(
        match_count=match_count,
        top_ids=tuple(ids),
        top_elos=tuple(elos),
        pool_size=pool_size,
    )


def test_stability_requires_n_snapshots() -> None:
    tr = StabilityTracker(k=3, n=3, eps=25)
    tr.push(_snap(10, ["a", "b", "c"], [1500, 1400, 1300]))
    tr.push(_snap(20, ["a", "b", "c"], [1500, 1400, 1300]))
    assert not tr.is_stable()
    tr.push(_snap(30, ["a", "b", "c"], [1500, 1400, 1300]))
    assert tr.is_stable()


def test_stability_fails_when_top_k_changes() -> None:
    tr = StabilityTracker(k=3, n=3, eps=50)
    for mc in (10, 20):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300]))
    tr.push(_snap(30, ["a", "b", "d"], [1500, 1400, 1310]))    # c → d
    assert not tr.is_stable()


def test_stability_fails_when_elo_drifts_past_epsilon() -> None:
    tr = StabilityTracker(k=3, n=3, eps=25)
    tr.push(_snap(10, ["a", "b", "c"], [1500, 1400, 1300]))
    tr.push(_snap(20, ["a", "b", "c"], [1510, 1410, 1310]))
    tr.push(_snap(30, ["a", "b", "c"], [1540, 1390, 1305]))   # a moved by 40 > 25
    assert not tr.is_stable()


def test_stability_passes_when_within_epsilon() -> None:
    tr = StabilityTracker(k=3, n=3, eps=30)
    tr.push(_snap(10, ["a", "b", "c"], [1500, 1400, 1300]))
    tr.push(_snap(20, ["a", "b", "c"], [1510, 1410, 1310]))
    tr.push(_snap(30, ["a", "b", "c"], [1520, 1395, 1305]))   # all moves within 25
    assert tr.is_stable()


# ----------------------------- combined predicate ----------------------------- #

def test_should_stop_returns_budget_first() -> None:
    cfg = Config()
    tr = StabilityTracker(k=3, n=3, eps=25)
    s = _session(used_usd=100, budget_usd=10)
    assert should_stop(cfg, s, tr) is StopReason.BUDGET


def test_should_stop_returns_wall_clock_when_budget_ok() -> None:
    cfg = Config()
    tr = StabilityTracker(k=3, n=3, eps=25)
    s = _session(used_usd=1, budget_usd=10, deadline_in_s=-10)
    assert should_stop(cfg, s, tr) is StopReason.WALL_CLOCK


def test_should_stop_none_when_running() -> None:
    cfg = Config()
    tr = StabilityTracker(k=3, n=3, eps=25)
    s = _session(used_usd=1, budget_usd=10, deadline_in_s=60)
    assert should_stop(cfg, s, tr) is None


# ----------------------------- min_ideas guard ----------------------------- #

def test_min_ideas_guard_blocks_stable_on_small_pool() -> None:
    """elo_stable must not fire when pool_size < min_ideas."""
    tr = StabilityTracker(k=3, n=3, eps=25, min_ideas=10)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=3))
    assert not tr.is_stable()


def test_min_ideas_guard_allows_stable_once_pool_is_large_enough() -> None:
    tr = StabilityTracker(k=3, n=3, eps=25, min_ideas=3)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=3))
    assert tr.is_stable()


def test_min_ideas_guard_zero_is_disabled() -> None:
    """Default min_ideas=0 must never block stability (backward compat)."""
    tr = StabilityTracker(k=3, n=3, eps=25, min_ideas=0)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=1))
    assert tr.is_stable()


# ----------------------------- min_matches guard ----------------------------- #

def test_min_matches_guard_blocks_stable_below_threshold() -> None:
    """elo_stable must not fire when total match_count < min_matches."""
    tr = StabilityTracker(k=3, n=3, eps=25, min_matches=100)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=20))
    assert not tr.is_stable()


def test_min_matches_guard_allows_stable_once_enough_matches() -> None:
    tr = StabilityTracker(k=3, n=3, eps=25, min_matches=30)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=20))
    assert tr.is_stable()


def test_min_matches_guard_zero_is_disabled() -> None:
    tr = StabilityTracker(k=3, n=3, eps=25, min_matches=0)
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=1))
    assert tr.is_stable()


# ----------------------------- combined guards ----------------------------- #

def test_both_guards_must_pass_for_stability() -> None:
    """Stable only when both min_ideas AND min_matches are satisfied."""
    tr = StabilityTracker(k=3, n=3, eps=25, min_ideas=10, min_matches=100)
    # pool_size OK but matches too low → not stable
    for mc in (10, 20, 30):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=15))
    assert not tr.is_stable()
    # Now push with both satisfied
    for mc in (110, 120, 130):
        tr.push(_snap(mc, ["a", "b", "c"], [1500, 1400, 1300], pool_size=15))
    assert tr.is_stable()
