"""Q&A prompt: answer strictly from retrieved context, with citations."""
from __future__ import annotations

from docmind.store.vectordb import Hit
from docmind.tasks.context import format_context

SYSTEM = (
    "You are a precise document analyst for a senior technology leader. "
    "Answer the question using ONLY the numbered context excerpts provided. "
    "Cite the excerpts you rely on inline using their source tags, e.g. "
    "(Onboarding:Overview p2). If the context does not contain the answer, say "
    "so plainly and state what is missing — do not invent facts. Be concise and "
    "structured."
)


def build_prompt(question: str, hits: list[Hit]) -> tuple[str, str]:
    context = format_context(hits)
    user = f"Context excerpts:\n\n{context}\n\n---\nQuestion: {question}"
    return SYSTEM, user
