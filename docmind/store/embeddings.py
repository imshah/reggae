"""Local embeddings via Ollama (qwen3-embedding).

Uses the batched `embed` API (the older `embeddings(prompt=…)` is deprecated and
issues one request per chunk). Adds retries and a per-item fallback so a
transient runner hiccup (EOF/500) or a single problematic chunk doesn't abort a
whole ingest. `truncate=True` (Ollama default) clamps over-long inputs to the
model's context instead of crashing the runner.
"""
from __future__ import annotations

import time

import ollama

from docmind.config import Config

_BATCH = 16
_ATTEMPTS = 3


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = ollama.Client(host=cfg.ollama_host)
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning exactly one vector per input."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            vectors.extend(self._embed_batch(texts[i : i + _BATCH]))
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(_ATTEMPTS):
            try:
                resp = self._client.embed(model=self.cfg.embed_model, input=batch)
                embs = getattr(resp, "embeddings", None)
                if embs is None:
                    embs = resp["embeddings"]
                return [list(v) for v in embs]
            except Exception as e:  # noqa: BLE001 - retry any runner error
                last = e
                time.sleep(0.5 * (attempt + 1))

        # Isolate a bad chunk: fall back to embedding items one at a time.
        if len(batch) > 1:
            out: list[list[float]] = []
            for t in batch:
                out.extend(self._embed_batch([t]))
            return out

        raise RuntimeError(
            f"Ollama embedding failed after {_ATTEMPTS} attempts for a chunk "
            f"({len(batch[0])} chars) using model '{self.cfg.embed_model}'. "
            f"Last error: {last}. Is the Ollama server healthy and the model "
            f"pulled? Try: `ollama run {self.cfg.embed_model}` / restart Ollama."
        ) from last

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim
