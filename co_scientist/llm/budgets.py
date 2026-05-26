"""Token + USD budgets, per-session and per-agent.

Concurrent agents share the same TokenBudget; admission is serialized by an
asyncio.Lock so two workers can't simultaneously over-reserve.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..config import Config


class BudgetExceeded(Exception):
    """Raised by `BudgetGuard.admit` when no headroom remains."""


@dataclass
class _Counter:
    used_tokens: int = 0                # input + output, kept for legacy callers
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    used_usd: float = 0.0
    reserved_tokens: int = 0
    reserved_usd: float = 0.0


@dataclass
class TokenBudget:
    """Total session budget. Per-agent shares are computed from cfg.budget_shares."""

    cfg: Config
    budget_tokens: int
    budget_usd: float
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _global: _Counter = field(default_factory=_Counter)
    _per_agent: dict[str, _Counter] = field(default_factory=dict)

    # ----------------------------- shares ------------------------------- #

    def share_tokens(self, agent: str) -> int:
        pct = self._agent_share_pct(agent)
        return int(self.budget_tokens * pct)

    def share_usd(self, agent: str) -> float:
        pct = self._agent_share_pct(agent)
        return self.budget_usd * pct

    def _agent_share_pct(self, agent: str) -> float:
        shares = self.cfg.budget_shares
        return {
            "generation": shares.generation,
            "reflection": shares.reflection,
            "ranking": shares.ranking,
            "evolution": shares.evolution,
            "metareview": shares.metareview,
            "proximity": shares.proximity,
        }.get(agent, 0.0)

    # ----------------------------- ops ---------------------------------- #

    async def admit(
        self, agent: str, *, est_tokens: int, est_usd: float
    ) -> None:
        """Block-style admission: raise BudgetExceeded if we can't afford this call."""
        async with self._lock:
            ctr = self._per_agent.setdefault(agent, _Counter())
            # Session-wide cap first (includes reserve)
            if (
                self._global.used_tokens + self._global.reserved_tokens + est_tokens
                > self.budget_tokens
            ) or (
                self._global.used_usd + self._global.reserved_usd + est_usd
                > self.budget_usd
            ):
                raise BudgetExceeded(
                    f"session budget exhausted (used_usd={self._global.used_usd:.2f},"
                    f" reserved={self._global.reserved_usd:.2f}, cap={self.budget_usd:.2f})"
                )
            # Per-agent share (skip for never-degrade agents — caller passes 'metareview_final'
            # via the same agent='metareview' key, but the reserve covers it).
            cap_usd = self.share_usd(agent) + self.cfg.budget_shares.reserve * self.budget_usd / 2
            if ctr.used_usd + ctr.reserved_usd + est_usd > cap_usd:
                raise BudgetExceeded(
                    f"agent {agent!r} share exhausted (used={ctr.used_usd:.2f},"
                    f" reserved={ctr.reserved_usd:.2f}, share={cap_usd:.2f})"
                )
            ctr.reserved_tokens += est_tokens
            ctr.reserved_usd += est_usd
            self._global.reserved_tokens += est_tokens
            self._global.reserved_usd += est_usd

    async def settle(
        self,
        agent: str,
        *,
        est_tokens: int,
        est_usd: float,
        actual_usd: float,
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
        actual_tokens: int | None = None,
    ) -> None:
        """Release the reservation and credit actual usage.

        Pass `actual_input_tokens` and `actual_output_tokens` separately so the
        bench (and any future per-input/output accounting) can read them from
        the snapshot. The legacy `actual_tokens` kwarg is treated as a combined
        total and credited only to `used_tokens` — its split is unknown so the
        per-input/output counters stay at 0 for that call.
        """
        if actual_tokens is None:
            actual_tokens = actual_input_tokens + actual_output_tokens
        async with self._lock:
            ctr = self._per_agent.setdefault(agent, _Counter())
            ctr.reserved_tokens = max(0, ctr.reserved_tokens - est_tokens)
            ctr.reserved_usd = max(0.0, ctr.reserved_usd - est_usd)
            ctr.used_tokens += actual_tokens
            ctr.used_input_tokens += actual_input_tokens
            ctr.used_output_tokens += actual_output_tokens
            ctr.used_usd += actual_usd
            self._global.reserved_tokens = max(0, self._global.reserved_tokens - est_tokens)
            self._global.reserved_usd = max(0.0, self._global.reserved_usd - est_usd)
            self._global.used_tokens += actual_tokens
            self._global.used_input_tokens += actual_input_tokens
            self._global.used_output_tokens += actual_output_tokens
            self._global.used_usd += actual_usd

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {
            "_global": {
                "used_tokens": self._global.used_tokens,
                "used_input_tokens": self._global.used_input_tokens,
                "used_output_tokens": self._global.used_output_tokens,
                "used_usd": self._global.used_usd,
                "reserved_tokens": self._global.reserved_tokens,
                "reserved_usd": self._global.reserved_usd,
                "budget_tokens": self.budget_tokens,
                "budget_usd": self.budget_usd,
            }
        }
        for agent, ctr in self._per_agent.items():
            out[agent] = {
                "used_tokens": ctr.used_tokens,
                "used_input_tokens": ctr.used_input_tokens,
                "used_output_tokens": ctr.used_output_tokens,
                "used_usd": ctr.used_usd,
                "reserved_tokens": ctr.reserved_tokens,
                "reserved_usd": ctr.reserved_usd,
                "share_usd": self.share_usd(agent),
            }
        return out
