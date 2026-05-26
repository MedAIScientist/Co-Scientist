"""Thin wrapper around the Anthropic Messages API.

Adds:
- Retry policy (see retry.py).
- Token + USD accounting; settles with TokenBudget after each call.
- Persists a Transcript row + a raw artifact (full messages array) on disk.
- Helpers to build cache_control blocks at the canonical breakpoints.
- Convenience for the standard agent tool-use loop (in tool_loop.py).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from anthropic import AsyncAnthropic

from ..config import Config
from ..ids import transcript_id
from ..models import Transcript
from ..storage.artifacts import write_json
from ..storage.repos import sessions as sessions_repo
from ..storage.repos import transcripts as transcripts_repo
from .budgets import TokenBudget
from .retry import RetryPolicy, with_retry
from .routing import ModelRoute, estimate_cost_usd


@dataclass
class CallContext:
    """Per-call metadata for accounting and persistence."""

    session_id: str
    task_id: str | None
    agent: str
    action: str
    mode: str | None = None     # e.g. "literature", "verification"


@dataclass
class CachedBlock:
    """One text block with optional cache_control marker."""

    text: str
    cache: bool = False         # True → emit cache_control: ephemeral


@dataclass
class AgentCallSpec:
    """Inputs to one LLM call, before we serialize them into Anthropic's schema."""

    route: ModelRoute
    system_blocks: list[CachedBlock] = field(default_factory=list)
    user_blocks: list[CachedBlock] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: dict[str, Any] | None = None      # e.g. {"type": "auto"} or {"type": "tool", "name": "..."}
    max_output_tokens: int = 4096
    stop_sequences: list[str] | None = None
    extra_messages: list[dict[str, Any]] = field(default_factory=list)
    """Appended *after* the user_blocks message — used for tool_use/tool_result threads."""


@dataclass
class AnthropicResponse:
    """Lightweight wrapper around the raw API response, with accounting attached."""

    raw: Any                       # anthropic.types.Message
    transcript_id: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int


class AnthropicClient:
    """Async wrapper. One instance per session is fine."""

    def __init__(
        self,
        cfg: Config,
        *,
        db: aiosqlite.Connection,
        budget: TokenBudget,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._budget = budget
        self._retry = retry_policy or RetryPolicy(
            max_attempts_429=cfg.retry.max_attempts_429,
            max_attempts_529=cfg.retry.max_attempts_529,
            max_attempts_5xx=cfg.retry.max_attempts_5xx,
            max_attempts_timeout=cfg.retry.max_attempts_timeout,
            base_ms=cfg.retry.base_ms,
            cap_ms=cfg.retry.cap_ms,
        )
        api_key = cfg.secrets.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = AsyncAnthropic(api_key=api_key)

    # ----------------------------- main call ----------------------------- #

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
        """Issue one messages.create with cache breakpoints, retries, accounting."""
        # Build payload
        system = _build_blocks(spec.system_blocks) or None
        first_user = _build_blocks(spec.user_blocks)
        messages: list[dict[str, Any]] = []
        if first_user:
            messages.append({"role": "user", "content": first_user})
        messages.extend(spec.extra_messages)

        request: dict[str, Any] = {
            "model": spec.route.model,
            "max_tokens": spec.max_output_tokens,
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if spec.tools:
            request["tools"] = spec.tools
        if spec.tool_choice is not None:
            request["tool_choice"] = spec.tool_choice
        if spec.stop_sequences:
            request["stop_sequences"] = spec.stop_sequences
        # Anthropic constraint: extended thinking is incompatible with a forced
        # tool_choice (must be "auto" or "none"). Silently drop the budget when
        # caller forced a specific tool; tool-use loops with `auto` keep thinking.
        thinking_ok = (
            spec.route.thinking_tokens > 0
            and (spec.tool_choice is None or spec.tool_choice.get("type") in ("auto", "none"))
        )
        if thinking_ok:
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": spec.route.thinking_tokens,
            }

        # Estimate cost upfront and admit
        est_in = est_input_tokens or _rough_token_count(spec)
        est_out = spec.max_output_tokens
        est_cost = estimate_cost_usd(
            model=spec.route.model, input_tokens=est_in, output_tokens=est_out
        )
        await self._budget.admit(
            ctx.agent, est_tokens=est_in + est_out, est_usd=est_cost
        )

        started = datetime.now(UTC)
        t0 = time.monotonic()

        async def _do() -> Any:
            return await self._client.messages.create(**request)

        # The retry loop can raise after exhausting attempts; if we don't release
        # the reservation in that case it leaks for the remainder of the session.
        try:
            resp = await with_retry(_do, policy=self._retry)
        except BaseException:
            await self._budget.settle(
                ctx.agent,
                est_tokens=est_in + est_out,
                est_usd=est_cost,
                actual_input_tokens=0,
                actual_output_tokens=0,
                actual_usd=0.0,
            )
            raise
        finished = datetime.now(UTC)

        usage = resp.usage
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost_usd = estimate_cost_usd(
            model=spec.route.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=cache_read,
            cache_write=cache_write,
        )

        await self._budget.settle(
            ctx.agent,
            est_tokens=est_in + est_out,
            est_usd=est_cost,
            actual_input_tokens=in_tok,
            actual_output_tokens=out_tok,
            actual_usd=cost_usd,
        )

        # Persist artifact + transcript row
        trn_id = transcript_id()
        artifact = {
            "request": _redact(request),
            "response": _response_to_dict(resp),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
        artifact_path = await write_json(
            self._cfg, ctx.session_id, f"transcripts/{ctx.agent}", trn_id, artifact
        )

        t = Transcript(
            id=trn_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            agent=ctx.agent,
            action=ctx.action,
            model=spec.route.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=cache_read,
            cache_write=cache_write,
            cost_usd=cost_usd,
            started_at=started,
            finished_at=finished,
            artifact_path=artifact_path,
        )
        await transcripts_repo.insert(self._db, t)
        await sessions_repo.add_usage(self._db, ctx.session_id, in_tok + out_tok, cost_usd)

        return AnthropicResponse(
            raw=resp,
            transcript_id=trn_id,
            cost_usd=cost_usd,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=cache_read,
            cache_write=cache_write,
        )


# --------------------------------------------------------------------------- #
# helpers


def _build_blocks(blocks: Iterable[CachedBlock]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in blocks:
        block: dict[str, Any] = {"type": "text", "text": b.text}
        if b.cache:
            block["cache_control"] = {"type": "ephemeral"}
        out.append(block)
    return out


def _rough_token_count(spec: AgentCallSpec) -> int:
    """Cheap heuristic: 1 token ≈ 4 chars. Used only for budget admission."""
    n = 0
    for b in spec.system_blocks:
        n += len(b.text) // 4
    for b in spec.user_blocks:
        n += len(b.text) // 4
    for m in spec.extra_messages:
        n += len(json.dumps(m)) // 4
    # Tool schemas are sent on every call and can dominate input on tool-heavy
    # agents (generation, reflection). Skipping them systematically
    # under-reserves the budget.
    if spec.tools:
        n += len(json.dumps(spec.tools)) // 4
    return max(n, 32)


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip nothing for now; placeholder for future PII / secret redaction."""
    return payload


def _response_to_dict(resp: Any) -> dict[str, Any]:
    """Convert anthropic.types.Message → JSON-serializable dict."""
    try:
        return resp.model_dump()
    except Exception:
        return {"_unparseable": str(resp)}
