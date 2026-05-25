"""Anthropic API retry policy.

- 429: respect Retry-After, exp backoff
- 529 (overloaded): respect Retry-After, longer backoff
- 5xx, timeouts: standard exp backoff with jitter
- 4xx (except 429): never retry — propagate
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts_429: int = 6
    max_attempts_529: int = 8
    max_attempts_5xx: int = 5
    max_attempts_timeout: int = 3
    # Total cap across all error classes. Without this, a flapping connection
    # that cycles 429 → 529 → 5xx → timeout can retry up to
    # (6+8+5+3) = 22 times before any per-class counter trips.
    max_attempts_total: int = 12
    base_ms: int = 1000
    cap_ms: int = 60_000


class RetryExhausted(RuntimeError):
    def __init__(self, last_error: BaseException, attempts: int):
        super().__init__(f"retry exhausted after {attempts} attempts: {last_error!r}")
        self.last_error = last_error
        self.attempts = attempts


def _retry_after_seconds(err: APIStatusError) -> float | None:
    headers = getattr(getattr(err, "response", None), "headers", None)
    if not headers:
        return None
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra is None:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        pass
    # RFC 7231 also allows HTTP-date format.
    try:
        from datetime import UTC, datetime
        when = parsedate_to_datetime(ra)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delta = (when - datetime.now(UTC)).total_seconds()
        return max(0.0, delta) if delta < 3600 else None
    except (TypeError, ValueError):
        return None


def _backoff_ms(base_ms: int, cap_ms: int, attempt: int, *, full_jitter: bool = True) -> int:
    exp = min(cap_ms, base_ms * (2**attempt))
    return random.randint(0, exp) if full_jitter else exp


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
) -> T:
    """Run `fn` with the configured retry policy. Never retries 4xx (except 429)."""

    attempt_429 = 0
    attempt_529 = 0
    attempt_5xx = 0
    attempt_timeout = 0
    attempt_total = 0

    while True:
        try:
            return await fn()

        # 4xx auth / bad request: never retry
        except (AuthenticationError, PermissionDeniedError, BadRequestError):
            raise

        except RateLimitError as e:
            attempt_429 += 1
            attempt_total += 1
            if attempt_429 >= policy.max_attempts_429 or attempt_total >= policy.max_attempts_total:
                raise RetryExhausted(e, attempt_total) from e
            ra = _retry_after_seconds(e)
            delay_s = ra if ra is not None else _backoff_ms(policy.base_ms, policy.cap_ms, attempt_429) / 1000
            await asyncio.sleep(delay_s)

        except APIStatusError as e:
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 529:
                attempt_529 += 1
                attempt_total += 1
                if attempt_529 >= policy.max_attempts_529 or attempt_total >= policy.max_attempts_total:
                    raise RetryExhausted(e, attempt_total) from e
                ra = _retry_after_seconds(e)
                delay_s = (
                    ra if ra is not None else _backoff_ms(policy.base_ms * 2, policy.cap_ms * 2, attempt_529) / 1000
                )
                await asyncio.sleep(delay_s)
            elif status is not None and 500 <= status < 600:
                attempt_5xx += 1
                attempt_total += 1
                if attempt_5xx >= policy.max_attempts_5xx or attempt_total >= policy.max_attempts_total:
                    raise RetryExhausted(e, attempt_total) from e
                delay_s = _backoff_ms(policy.base_ms // 2 or 250, policy.cap_ms // 2, attempt_5xx) / 1000
                await asyncio.sleep(delay_s)
            else:
                # 4xx other than 429: do not retry
                raise

        except InternalServerError as e:
            attempt_5xx += 1
            attempt_total += 1
            if attempt_5xx >= policy.max_attempts_5xx or attempt_total >= policy.max_attempts_total:
                raise RetryExhausted(e, attempt_total) from e
            await asyncio.sleep(
                _backoff_ms(policy.base_ms // 2 or 250, policy.cap_ms // 2, attempt_5xx) / 1000
            )

        except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.NetworkError) as e:
            attempt_timeout += 1
            attempt_total += 1
            if attempt_timeout >= policy.max_attempts_timeout or attempt_total >= policy.max_attempts_total:
                raise RetryExhausted(e, attempt_total) from e
            await asyncio.sleep(
                _backoff_ms(policy.base_ms, policy.cap_ms // 4, attempt_timeout) / 1000
            )
