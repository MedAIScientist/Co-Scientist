"""Tests for the LLM retry policy."""

from __future__ import annotations

import httpx
import pytest

from co_scientist.llm.retry import RetryExhausted, RetryPolicy, with_retry


@pytest.mark.asyncio
async def test_returns_on_first_success() -> None:
    calls = {"n": 0}

    async def ok():
        calls["n"] += 1
        return "ok"

    result = await with_retry(ok, policy=RetryPolicy(base_ms=1, cap_ms=1))
    assert result == "ok"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_total_attempts_cap_trips_before_per_class_cap() -> None:
    """A connection that flaps timeouts must hit the total cap, not loop forever."""
    calls = {"n": 0}

    async def always_timeout():
        calls["n"] += 1
        raise httpx.TimeoutException("boom")

    policy = RetryPolicy(
        max_attempts_429=1000,
        max_attempts_529=1000,
        max_attempts_5xx=1000,
        max_attempts_timeout=1000,
        max_attempts_total=3,
        base_ms=1,
        cap_ms=1,
    )
    with pytest.raises(RetryExhausted) as ei:
        await with_retry(always_timeout, policy=policy)
    assert ei.value.attempts == 3
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_returns_on_eventual_success() -> None:
    calls = {"n": 0}

    async def flap():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.TimeoutException("not yet")
        return "done"

    policy = RetryPolicy(
        max_attempts_timeout=10,
        max_attempts_total=10,
        base_ms=1, cap_ms=1,
    )
    assert await with_retry(flap, policy=policy) == "done"
    assert calls["n"] == 3
