"""Remote-provider abstraction.

Tasks call remote LLMs only through the `Provider` protocol, so the escalation
vendor is swappable (Claude / Kimi / future) via config or a `--provider` flag.
Each provider owns its own token-counting, cost estimate, and (optional) prompt
caching so the cost guard stays accurate after a switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from docmind.config import Config


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class Completion:
    text: str
    usage: Usage
    model: str
    provider: str


class Provider(Protocol):
    name: str

    def model_for(self, heavy: bool) -> str: ...

    def count_tokens(self, system: str, user: str, *, heavy: bool = False) -> int: ...

    def estimate(
        self, input_tokens: int, output_tokens_guess: int, *, heavy: bool = False
    ) -> float: ...

    def complete(
        self,
        system: str,
        user: str,
        *,
        heavy: bool = False,
        cache_system: bool = True,
        max_tokens: int = 4096,
    ) -> Completion: ...

    def available(self) -> tuple[bool, str]: ...


def get_provider(cfg: Config, name: str | None = None) -> Provider:
    """Instantiate the active (or named) remote provider."""
    chosen = (name or cfg.remote_provider or "claude").lower()
    if chosen == "claude":
        from docmind.llm.claude import ClaudeProvider

        return ClaudeProvider(cfg)
    if chosen == "kimi":
        from docmind.llm.kimi import KimiProvider

        return KimiProvider(cfg)
    raise ValueError(f"Unknown provider: {chosen!r} (expected 'claude' or 'kimi')")


AVAILABLE_PROVIDERS = ("claude", "kimi")
