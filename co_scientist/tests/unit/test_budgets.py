"""Tests for the budget guard."""

from __future__ import annotations

import pytest

from co_scientist.config import Config
from co_scientist.llm.budgets import BudgetExceeded, TokenBudget


@pytest.mark.asyncio
async def test_admit_then_settle_updates_counters() -> None:
    cfg = Config()
    b = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=10.0)

    await b.admit("generation", est_tokens=1000, est_usd=0.5)
    snap1 = b.snapshot()
    assert snap1["_global"]["reserved_usd"] == pytest.approx(0.5)

    await b.settle(
        "generation",
        est_tokens=1000, est_usd=0.5,
        actual_tokens=800, actual_usd=0.40,
    )
    snap2 = b.snapshot()
    assert snap2["_global"]["reserved_usd"] == pytest.approx(0.0)
    assert snap2["_global"]["used_usd"] == pytest.approx(0.40)
    assert snap2["generation"]["used_usd"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_global_cap_eventually_blocks_runaway_spend() -> None:
    """Repeated admits across all agents eventually hit either the per-agent share
    or the global cap; the system must refuse to keep admitting indefinitely."""
    cfg = Config()
    b = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=1.0)
    raised = False
    for _ in range(200):
        for agent in (
            "generation", "reflection", "ranking",
            "evolution", "metareview", "proximity",
        ):
            try:
                await b.admit(agent, est_tokens=10, est_usd=0.01)
            except BudgetExceeded:
                raised = True
                break
        if raised:
            break
    assert raised, "BudgetExceeded should fire long before 1200 admits"


@pytest.mark.asyncio
async def test_per_agent_share_is_enforced() -> None:
    cfg = Config()
    # share for generation is 20% by default → $0.20 of $1
    b = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=1.0)
    # under share
    await b.admit("generation", est_tokens=100, est_usd=0.10)
    # over share (plus the half-reserve allowance), should raise eventually
    with pytest.raises(BudgetExceeded):
        # 3x more should clearly exceed share+reserve
        for _ in range(20):
            await b.admit("generation", est_tokens=100, est_usd=0.30)


@pytest.mark.asyncio
async def test_share_percent_for_unknown_agent_is_zero() -> None:
    cfg = Config()
    b = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=10.0)
    # 'unknown' has 0% share; only the half-reserve allowance buffers us
    assert b.share_usd("unknown") == 0.0


@pytest.mark.asyncio
async def test_settle_with_zero_actual_releases_reservation() -> None:
    """If a call fails after admission (e.g. retry exhaustion), the caller must
    be able to release the reservation with actual=0 so the reserve doesn't leak."""
    cfg = Config()
    b = TokenBudget(cfg=cfg, budget_tokens=10_000, budget_usd=1.0)

    await b.admit("generation", est_tokens=500, est_usd=0.20)
    snap_mid = b.snapshot()
    assert snap_mid["_global"]["reserved_usd"] == pytest.approx(0.20)

    # Simulate failed call: settle with actual=0
    await b.settle(
        "generation",
        est_tokens=500, est_usd=0.20,
        actual_tokens=0, actual_usd=0.0,
    )
    snap_end = b.snapshot()
    assert snap_end["_global"]["reserved_usd"] == pytest.approx(0.0)
    assert snap_end["_global"]["used_usd"] == pytest.approx(0.0)
    # Subsequent admit should now succeed
    await b.admit("generation", est_tokens=500, est_usd=0.20)
