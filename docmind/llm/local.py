"""Local chat model (qwen3:14b via Ollama) for routine Q&A."""
from __future__ import annotations

import ollama

from docmind.config import Config


class LocalLLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = ollama.Client(host=cfg.ollama_host)

    def _messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self.cfg.local_chat_model,
            messages=self._messages(system, user),
        )
        return (resp.get("message", {}) or {}).get("content", "").strip()

    def chat_stream(self, system: str, user: str):
        """Yield content chunks as they stream from Ollama."""
        for part in self._client.chat(
            model=self.cfg.local_chat_model,
            messages=self._messages(system, user),
            stream=True,
        ):
            piece = (part.get("message", {}) or {}).get("content", "")
            if piece:
                yield piece
