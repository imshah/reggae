"""Local chat model (qwen3:14b via Ollama) for routine Q&A."""
from __future__ import annotations

import ollama

from docmind.config import Config


def _base_name(tag: str) -> str:
    """Model name without its Ollama tag, e.g. 'qwen3-embedding:latest' → 'qwen3-embedding'."""
    return tag.split(":", 1)[0]


def chat_capable(names: list[str], cfg: Config) -> list[str]:
    """Keep only chat/reasoning models: drop the configured embedding + vision
    models and anything that looks like an embedding model."""
    exclude = {_base_name(cfg.embed_model), _base_name(cfg.vision_model)}
    return [
        n for n in names
        if _base_name(n) not in exclude and "embed" not in n.lower()
    ]


class LocalLLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = ollama.Client(host=cfg.ollama_host)

    def _messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def chat(self, system: str, user: str, model: str | None = None) -> str:
        resp = self._client.chat(
            model=model or self.cfg.local_chat_model,
            messages=self._messages(system, user),
        )
        return (resp.get("message", {}) or {}).get("content", "").strip()

    def chat_stream(self, system: str, user: str, model: str | None = None):
        """Yield content chunks as they stream from Ollama."""
        for part in self._client.chat(
            model=model or self.cfg.local_chat_model,
            messages=self._messages(system, user),
            stream=True,
        ):
            piece = (part.get("message", {}) or {}).get("content", "")
            if piece:
                yield piece

    def chat_models(self) -> list[str]:
        """Installed models usable for chat — excludes the embedding + vision models."""
        return chat_capable(self.list_models(), self.cfg)

    def list_models(self) -> list[str]:
        """Installed Ollama model tags; [] on any failure (daemon down/offline)."""
        try:
            resp = self._client.list()
            models = getattr(resp, "models", None)
            if models is None and isinstance(resp, dict):
                models = resp.get("models", [])
            out: list[str] = []
            for m in models or []:
                name = getattr(m, "model", None)
                if name is None and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    out.append(name)
            return sorted(out)
        except Exception:
            return []
