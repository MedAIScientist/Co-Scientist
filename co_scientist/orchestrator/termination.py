"""Termination predicate for the Supervisor's main loop.

`should_stop(session)` returns one of:
- BUDGET     — token/USD budget exhausted
- WALL_CLOCK — session time deadline crossed
- ELO_STABLE — top-K hypotheses unchanged for N snapshots within ε
- None       — keep running
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import aiosqlite

from ..config import Config
from ..models import Session


class StopReason(Enum):
    BUDGET = "budget"
    WALL_CLOCK = "wall_clock"
    ELO_STABLE = "elo_stable"
    EXTERNAL = "external"      # user pressed pause/abort or invoked /sessions/{id}/abort
    IDLE = "idle"              # queue drained and decide_next_steps returned 0


@dataclass
class EloSnapshot:
    """The top-K leaderboard at one point in time."""

    match_count: int
    top_ids: tuple[str, ...]
    top_elos: tuple[float, ...]
    pool_size: int = 0  # total hypotheses in_tournament or pinned at snapshot time


class StabilityTracker:
    """Owns the recent EloSnapshot history. One per session."""

    def __init__(
        self,
        k: int,
        n: int,
        eps: float,
        min_ideas: int = 0,
        min_matches: int = 0,
    ) -> None:
        self.k = k
        self.n = n
        self.eps = eps
        self.min_ideas = min_ideas
        self.min_matches = min_matches
        self._history: list[EloSnapshot] = []

    def push(self, snap: EloSnapshot) -> None:
        self._history.append(snap)
        # keep slightly more than needed so we can see drift
        if len(self._history) > self.n * 2:
            self._history = self._history[-self.n * 2 :]

    @property
    def history(self) -> list[EloSnapshot]:
        return list(self._history)

    def is_stable(self) -> bool:
        if len(self._history) < self.n:
            return False
        recent = self._history[-self.n :]
        # Guard: do not declare stability until the pool is large enough
        # and enough matches have been played.  Guards default to 0 (off),
        # preserving the original behaviour when not configured.
        if self.min_ideas > 0 and recent[-1].pool_size < self.min_ideas:
            return False
        if self.min_matches > 0 and recent[-1].match_count < self.min_matches:
            return False
        # All N snapshots must have the same top-K id set
        first_set = set(recent[0].top_ids)
        for s in recent[1:]:
            if set(s.top_ids) != first_set:
                return False
        # AND the max per-id Elo delta across the window must be < eps
        # Build {id: [elo across snapshots]}
        per_id: dict[str, list[float]] = {}
        for s in recent:
            for hid, elo in zip(s.top_ids, s.top_elos, strict=True):
                per_id.setdefault(hid, []).append(elo)
        return all(max(elos) - min(elos) < self.eps for elos in per_id.values())


async def snapshot_top_k(
    conn: aiosqlite.Connection, session_id: str, k: int
) -> EloSnapshot:
    """Read the current top-K leaderboard + count of completed matches + pool size."""
    async with conn.execute(
        """SELECT id, elo FROM hypotheses
              WHERE session_id=? AND state IN ('in_tournament','pinned')
                AND elo IS NOT NULL
              ORDER BY elo DESC LIMIT ?""",
        (session_id, k),
    ) as cur:
        rows = await cur.fetchall()
    async with conn.execute(
        "SELECT COUNT(*) AS n FROM tournament_matches WHERE session_id=? AND mode != 'invalid'",
        (session_id,),
    ) as cur:
        mc_row = await cur.fetchone()
    async with conn.execute(
        """SELECT COUNT(*) AS n FROM hypotheses
              WHERE session_id=? AND state IN ('in_tournament','pinned')""",
        (session_id,),
    ) as cur:
        pool_row = await cur.fetchone()
    return EloSnapshot(
        match_count=mc_row["n"] if mc_row else 0,
        top_ids=tuple(r["id"] for r in rows),
        top_elos=tuple(r["elo"] for r in rows),
        pool_size=pool_row["n"] if pool_row else 0,
    )


def budget_exceeded(session: Session) -> bool:
    return (
        (session.budget_usd > 0 and session.budget_used_usd >= session.budget_usd)
        or (session.budget_tokens > 0 and session.budget_used_tokens >= session.budget_tokens)
    )


def wall_clock_exceeded(session: Session) -> bool:
    if session.wall_deadline is None:
        return False
    now = datetime.now(UTC)
    deadline = session.wall_deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return now >= deadline


def should_stop(
    cfg: Config,
    session: Session,
    tracker: StabilityTracker,
    external_stop: bool = False,
) -> StopReason | None:
    if external_stop:
        return StopReason.EXTERNAL
    if budget_exceeded(session):
        return StopReason.BUDGET
    if wall_clock_exceeded(session):
        return StopReason.WALL_CLOCK
    del cfg  # reserved for future config-driven termination rules
    if tracker.is_stable():
        return StopReason.ELO_STABLE
    return None
