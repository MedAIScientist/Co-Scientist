"""LLMProvider — vendor-agnostic LLM client interface.

The co-scientist began as an Anthropic-only system; the type hints and
intermediate request shapes (`AgentCallSpec`, `CachedBlock`) are
Anthropic-flavored. Rather than rewrite every agent we treat those types as
the canonical normalized form: each provider takes a normalized spec, calls
its vendor SDK, and returns an `AnthropicResponse` whose `.raw` exposes a
Message-like object with `.content`, `.stop_reason`, `.usage`.

Concretely:
- AnthropicProvider: passes through to anthropic.AsyncAnthropic.messages.create.
- OpenAIProvider: translates to openai.chat.completions.create (also supports
  arbitrary OpenAI-compatible base_urls: Groq, Together, OpenRouter, Mistral,
  Ollama, Gemini OpenAI-compat endpoint).

Provider-specific features:
- cache_control: honored only on Anthropic. Stripped before sending elsewhere.
- thinking / extended reasoning: Anthropic for Claude opus; on OpenAI we
  translate to `reasoning_effort` for o-series models, else drop.
- batch API: Anthropic only; the BatchPool still talks to Anthropic directly.

Users select a provider in `[llm] provider = "..."` and per-agent models in
`[models]`. Model strings are passed verbatim to the configured provider.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .anthropic_client import AgentCallSpec, AnthropicResponse, CallContext


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface every LLM client implements."""

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
        ...


# Provider names accepted in config.
KNOWN_PROVIDERS = frozenset({
    "anthropic",
    "openai",
    "openai_compatible",
})


def get_provider(
    cfg,
    *,
    db,
    budget,
    retry_policy=None,
) -> LLMProvider:
    """Construct the LLM provider configured in `cfg.llm.provider`.

    Selection is case-insensitive. Unknown values fall back to `anthropic`
    with a warning so older configs continue to work.
    """
    from ..logging import get_logger

    log = get_logger("llm.provider")

    name = (getattr(cfg.llm, "provider", "anthropic") or "anthropic").strip().lower()
    if name not in KNOWN_PROVIDERS:
        log.warning("unknown_llm_provider", configured=name, fallback="anthropic")
        name = "anthropic"

    if name == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg, db=db, budget=budget, retry_policy=retry_policy)

    if name in ("openai", "openai_compatible"):
        from .openai_client import OpenAIClient

        return OpenAIClient(
            cfg, db=db, budget=budget, retry_policy=retry_policy,
            compat_mode=(name == "openai_compatible"),
        )

    # Unreachable
    raise ValueError(f"unsupported LLM provider {name!r}")
