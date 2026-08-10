"""KimiProvider — Moonshot's OpenAI-compatible API via the `openai` SDK.

Same system/user shape as ClaudeProvider. Moonshot exposes an OpenAI-compatible
endpoint, so we point the OpenAI client at kimi_base_url with KIMI_API_KEY.
Prompt caching is a no-op here (applied automatically server-side if/when the
provider supports it); token counting falls back to a word-based estimate.
"""
from __future__ import annotations

import os

from docmind.config import Config
from docmind.llm import pricing
from docmind.llm.provider import Completion, Usage


class KimiProvider:
    name = "kimi"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.cfg.kimi_base_url,
                api_key=os.environ.get("KIMI_API_KEY", ""),
            )
        return self._client

    def available(self) -> tuple[bool, str]:
        if os.environ.get("KIMI_API_KEY"):
            return True, ""
        return (
            False,
            "No KIMI_API_KEY set. Export it to use the Kimi provider. "
            "docmind still works fully local without it.",
        )

    def model_for(self, heavy: bool) -> str:
        return self.cfg.kimi_heavy_model if heavy else self.cfg.kimi_model

    def count_tokens(self, system: str, user: str, *, heavy: bool = False) -> int:
        # Moonshot has a token-count endpoint but it varies by version;
        # a word-based estimate is sufficient for the pre-call cost guard.
        return int((len(system.split()) + len(user.split())) * 1.3)

    def estimate(self, input_tokens: int, output_tokens_guess: int, *, heavy: bool = False) -> float:
        return pricing.estimate_cost(
            self.name, self.model_for(heavy), input_tokens, output_tokens_guess
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        heavy: bool = False,
        cache_system: bool = True,
        max_tokens: int = 4096,
    ) -> Completion:
        client = self._get_client()
        model = self.model_for(heavy)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        )
        return Completion(text=text, usage=usage, model=model, provider=self.name)
