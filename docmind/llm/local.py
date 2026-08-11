"""Local chat model (qwen3:14b via Ollama) for routine Q&A."""
from __future__ import annotations

import ollama

from docmind.config import Config


# Explicit allowlist of local (Ollama) reasoning/chat model families. A model is
# offered in the UI only if its base name (the part before the ':' tag) matches an
# entry here EXACTLY — so 'qwen3' covers qwen3:14b and qwen3:1.7b, but not
# qwen3-embedding (embeddings) or qwen2.5vl (vision), which are simply not listed.
# Add a family here to expose its installed models.
CHAT_MODEL_FAMILIES = {
    "qwen3",
    "qwen2.5",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral",
    "mixtral",
    "gemma2",
    "gemma3",
    "phi3",
    "phi4",
    "deepseek-r1",
    "deepseek-v3",
}


def _base_name(tag: str) -> str:
    """Model name without its Ollama tag, e.g. 'qwen3:14b' → 'qwen3'."""
    return tag.split(":", 1)[0]


def chat_capable(names: list[str]) -> list[str]:
    """Keep only models whose family is on the CHAT_MODEL_FAMILIES allowlist
    (exact base-name match)."""
    return [n for n in names if _base_name(n) in CHAT_MODEL_FAMILIES]


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
        """Installed models on the chat-model allowlist (CHAT_MODEL_FAMILIES)."""
        return chat_capable(self.list_models())

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
