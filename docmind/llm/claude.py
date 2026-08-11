"""ClaudeProvider — Anthropic SDK.

Defaults to claude-opus-4-8. Uses prompt caching on the (large, stable) system
block so repeated questions over the same corpus pay ~0.1x on the shared
prefix. Adaptive thinking + effort are sent only for models that accept them
(so a downgrade to Haiku 4.5 stays safe).
"""
from __future__ import annotations

import os

from docmind.config import Config
from docmind.llm import pricing
from docmind.llm.provider import Completion, Usage

# models that accept adaptive thinking + output_config.effort
_EFFORT_OK = ("claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5", "claude-sonnet-4-6")


class ClaudeProvider:
    name = "claude"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None

    # --- lazy client (so local-only use needs no key) ---------------------
    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> tuple[bool, str]:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, ""
        # SDK may still resolve an `ant auth login` profile; try lazily.
        return (
            False,
            "No ANTHROPIC_API_KEY set. Export it, or run `ant auth login`. "
            "docmind still works fully local without it.",
        )

    def model_for(self, heavy: bool) -> str:
        return self.cfg.claude_heavy_model if heavy else self.cfg.claude_model

    def list_models(self) -> list[str]:
        """Model ids the account exposes; [] on any failure (no key/offline).

        The list endpoint auto-paginates when iterated directly.
        """
        try:
            return sorted(m.id for m in self._get_client().models.list())
        except Exception:
            return []

    def _supports_effort(self, model: str) -> bool:
        return any(model.startswith(m) for m in _EFFORT_OK)

    # --- token counting ---------------------------------------------------
    def count_tokens(
        self, system: str, user: str, *, heavy: bool = False, model: str | None = None
    ) -> int:
        model = model or self.model_for(heavy)
        try:
            client = self._get_client()
            resp = client.messages.count_tokens(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return int(resp.input_tokens)
        except Exception:
            # fall back to a rough word-based estimate if the API is unavailable
            return int((len(system.split()) + len(user.split())) * 1.3)

    def estimate(self, input_tokens: int, output_tokens_guess: int, *, heavy: bool = False,
                 model: str | None = None) -> float:
        return pricing.estimate_cost(
            self.name, model or self.model_for(heavy), input_tokens, output_tokens_guess
        )

    # --- completion -------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        *,
        heavy: bool = False,
        cache_system: bool = True,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> Completion:
        client = self._get_client()
        model = model or self.model_for(heavy)

        system_param = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache_system
            else system
        )

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": [{"role": "user", "content": user}],
        }
        if self._supports_effort(model):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.cfg.claude_effort}

        resp = client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0),
            output_tokens=getattr(u, "output_tokens", 0),
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return Completion(text=text, usage=usage, model=model, provider=self.name)
