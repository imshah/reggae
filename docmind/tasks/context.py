"""Build cited context blocks from retrieved chunks."""
from __future__ import annotations

from docmind.store.vectordb import Hit


def _cite(hit: Hit) -> str:
    where = hit.section or "?"
    if hit.page:
        where = f"{where} p{hit.page}"
    return f"{hit.title}:{where}"


def format_context(hits: list[Hit]) -> str:
    """Numbered, labelled excerpts the model can cite by [n] and source tag."""
    parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        tag = _cite(h)
        marker = "diagram" if h.kind == "diagram" else "text"
        parts.append(f"[{i}] ({marker}) {tag}\n{h.text}")
    return "\n\n".join(parts) if parts else "(no relevant context found)"


def sources_list(hits: list[Hit]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        c = _cite(h)
        if c not in seen:
            seen.append(c)
    return seen
