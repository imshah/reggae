"""Per-model pricing (USD per 1M tokens) for cost estimates.

Rates are best-effort and used only for pre-call estimates and the session
budget guard — they are not billing-authoritative. Cache-read is priced at
~0.1x input where a model supports prompt caching; cache-write at ~1.25x.
Unknown models fall back to a conservative default.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    input: float          # per 1M input tokens
    output: float         # per 1M output tokens
    cache_read: float = 0.0
    cache_write: float = 0.0
    supports_cache: bool = False


# Claude (Anthropic) — see claude-api pricing reference.
_CLAUDE: dict[str, Rate] = {
    "claude-opus-4-8": Rate(5.0, 25.0, 0.5, 6.25, supports_cache=True),
    "claude-sonnet-5": Rate(3.0, 15.0, 0.3, 3.75, supports_cache=True),
    "claude-haiku-4-5": Rate(1.0, 5.0, 0.1, 1.25, supports_cache=True),
}

# Kimi / Moonshot (OpenAI-compatible). Placeholder rates — adjust to the
# provider's published pricing for the exact model id in use.
_KIMI: dict[str, Rate] = {
    "kimi-k3-turbo-preview": Rate(0.60, 2.50),
    "kimi-k3": Rate(1.0, 3.0),
}

_DEFAULT = Rate(3.0, 15.0)


def rate_for(provider: str, model: str) -> Rate:
    table = _CLAUDE if provider == "claude" else _KIMI if provider == "kimi" else {}
    if model in table:
        return table[model]
    # prefix match (handles dated/variant ids)
    for key, r in table.items():
        if model.startswith(key):
            return r
    return _DEFAULT


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    r = rate_for(provider, model)
    fresh_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
    return (
        fresh_input * r.input
        + cache_read_tokens * r.cache_read
        + cache_write_tokens * r.cache_write
        + output_tokens * r.output
    ) / 1_000_000
