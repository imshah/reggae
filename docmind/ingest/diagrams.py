"""Describe extracted diagram images with a local vision model (Ollama).

Descriptions are produced once at ingest time and stored as searchable chunks
(kind="diagram"), so diagram semantics are queryable later without re-running
vision.
"""
from __future__ import annotations

import ollama

from docmind.config import Config

_PROMPT = (
    "You are analysing an image extracted from a technical/process document. "
    "If it is a diagram (flowchart, architecture, sequence, ER, mind map, etc.), "
    "describe it precisely: its type, every node/component, the connections and "
    "their direction, any labels, and the overall flow or structure it conveys. "
    "If it is not a meaningful diagram (photo, logo, decorative image), reply "
    "with exactly: NOT_A_DIAGRAM"
)


def describe_image(cfg: Config, image_path: str) -> str | None:
    """Return a textual description, or None if it isn't a useful diagram."""
    client = ollama.Client(host=cfg.ollama_host)
    try:
        resp = client.chat(
            model=cfg.vision_model,
            messages=[{"role": "user", "content": _PROMPT, "images": [image_path]}],
        )
    except Exception:
        return None
    content = (resp.get("message", {}) or {}).get("content", "").strip()
    if not content or "NOT_A_DIAGRAM" in content.upper():
        return None
    return content
