"""Tests for the tool-use loop, especially terminal-tool short-circuit.

The terminal-tool short-circuit matters for any provider whose model does NOT
reliably emit `stop_reason="end_turn"` after calling a recording tool — that
includes most OpenAI-compat models (Gemini, OpenAI o-series via tool_calls,
Llama through OpenRouter, etc.). Without the short-circuit they loop until
max_iters and ToolLoopExhausted, even though a perfectly valid record was
emitted on the first call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from co_scientist.llm.anthropic_client import (
    AgentCallSpec,
    AnthropicResponse,
    CachedBlock,
    CallContext,
)
from co_scientist.llm.routing import ModelRoute
from co_scientist.llm.tool_loop import run_tool_loop


def _fake_response(*, stop_reason: str, blocks: list[dict]) -> AnthropicResponse:
    """Build an AnthropicResponse whose .raw quacks like an anthropic Message."""
    content = []
    for b in blocks:
        # Each block must expose .type, .name, .input, .text
        content.append(SimpleNamespace(
            type=b.get("type", "text"),
            name=b.get("name", ""),
            input=b.get("input", {}),
            text=b.get("text", ""),
            id=b.get("id", ""),
        ))
    raw = SimpleNamespace(stop_reason=stop_reason, content=content)
    return AnthropicResponse(
        raw=raw, transcript_id="trn_x",
        cost_usd=0.0, input_tokens=0, output_tokens=0,
        cache_read=0, cache_write=0,
    )


def _spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[{"name": "search", "description": "", "input_schema": {}}],
        max_output_tokens=512,
    )


def _ctx() -> CallContext:
    return CallContext(session_id="s", task_id="t", agent="generation", action="a")


@pytest.mark.asyncio
async def test_loop_ends_on_record_hypothesis_even_when_stop_reason_is_tool_use() -> None:
    """The bug we hit on Gemini: model emits record_hypothesis but keeps
    stop_reason=tool_use, so without short-circuit the loop runs to
    max_iters."""
    client = MagicMock()
    client.call = AsyncMock(side_effect=[
        _fake_response(
            stop_reason="tool_use",
            blocks=[{
                "type": "tool_use",
                "id": "call_1",
                "name": "record_hypothesis",
                "input": {"title": "t", "statement": "s"},
            }],
        ),
    ])
    registry = MagicMock()

    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=registry,
        max_iters=8, parallel_cap=4, tool_timeout_s=1.0,
    )
    assert result.iterations == 1
    # The terminal tool_use is logged but never dispatched.
    assert client.call.await_count == 1
    assert result.tool_calls[0]["name"] == "record_hypothesis"


@pytest.mark.asyncio
async def test_loop_ends_normally_on_end_turn() -> None:
    client = MagicMock()
    client.call = AsyncMock(return_value=_fake_response(
        stop_reason="end_turn",
        blocks=[{"type": "text", "text": "all done"}],
    ))
    registry = MagicMock()
    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=registry,
        max_iters=8, parallel_cap=4, tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_dispatches_non_terminal_tools_then_continues() -> None:
    """Search tool calls should still be dispatched and the loop continues."""
    from co_scientist.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(side_effect=[
        _fake_response(
            stop_reason="tool_use",
            blocks=[{
                "type": "tool_use", "id": "call_search",
                "name": "search", "input": {"q": "foo"},
            }],
        ),
        _fake_response(
            stop_reason="tool_use",
            blocks=[{
                "type": "tool_use", "id": "call_record",
                "name": "record_hypothesis",
                "input": {"title": "t", "statement": "s"},
            }],
        ),
    ])

    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(
        is_error=False, content={"ok": True}, duration_ms=1,
    ))

    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=registry,
        max_iters=8, parallel_cap=4, tool_timeout_s=1.0,
    )
    assert result.iterations == 2
    # search was dispatched; record_hypothesis was NOT (terminal)
    assert registry.call.await_count == 1
    names = [tc["name"] for tc in result.tool_calls]
    assert names == ["search", "record_hypothesis"]


@pytest.mark.asyncio
async def test_loop_terminates_on_record_review() -> None:
    client = MagicMock()
    client.call = AsyncMock(return_value=_fake_response(
        stop_reason="tool_use",
        blocks=[{
            "type": "tool_use", "id": "c", "name": "record_review",
            "input": {"verdict": "accept"},
        }],
    ))
    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=MagicMock(),
        max_iters=8, parallel_cap=4, tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_terminates_on_custom_terminal_tool() -> None:
    """The terminal-tool set is configurable via kwarg."""
    client = MagicMock()
    client.call = AsyncMock(return_value=_fake_response(
        stop_reason="tool_use",
        blocks=[{"type": "tool_use", "id": "c", "name": "my_done_signal", "input": {}}],
    ))
    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=MagicMock(),
        max_iters=8, parallel_cap=4, tool_timeout_s=1.0,
        terminal_tool_names=("my_done_signal",),
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_force_terminal_tool_on_final_iteration() -> None:
    """A model that only ever searches should be forced to record on the last
    iteration when force_terminal_tool is set — instead of exhausting the loop."""
    from co_scientist.tools.base import ToolResult

    # The model keeps searching forever unless tool_choice forces a specific
    # tool — exactly how a real provider behaves under a forced tool_choice.
    def _respect_tool_choice(spec, *_a, **_k):
        tc = spec.tool_choice or {}
        if tc.get("type") == "tool" and tc.get("name") == "record_hypothesis":
            return _fake_response(
                stop_reason="tool_use",
                blocks=[{
                    "type": "tool_use", "id": "r", "name": "record_hypothesis",
                    "input": {"title": "t", "statement": "s"},
                }],
            )
        return _fake_response(
            stop_reason="tool_use",
            blocks=[{"type": "tool_use", "id": "s", "name": "search", "input": {"q": "x"}}],
        )

    client = MagicMock()
    client.call = AsyncMock(side_effect=_respect_tool_choice)
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(
        is_error=False, content={"n": 0, "results": []}, duration_ms=1,
    ))

    result = await run_tool_loop(
        client, spec=_spec(), ctx=_ctx(), registry=registry,
        max_iters=3, parallel_cap=4, tool_timeout_s=1.0,
        force_terminal_tool="record_hypothesis",
    )
    # The loop committed on the forced final iteration instead of exhausting.
    assert result.iterations == 3
    assert result.tool_calls[-1]["name"] == "record_hypothesis"
    # The final (3rd) call forced tool_choice to record_hypothesis.
    final_spec = client.call.await_args_list[-1].args[0]
    assert final_spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    # Earlier calls used the original auto tool_choice (no forcing).
    first_spec = client.call.await_args_list[0].args[0]
    assert first_spec.tool_choice != {"type": "tool", "name": "record_hypothesis"}
