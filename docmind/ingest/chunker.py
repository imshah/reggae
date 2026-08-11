"""Heading-aware chunking with metadata.

Text blocks (already grouped by section/page from the parsers) are split into
~chunk_tokens-sized pieces with overlap. Diagram descriptions become their own
single chunks tagged kind="diagram". Token counts are approximated by words
(~1.3 tokens/word); exact counts aren't needed for chunk sizing.
"""
from __future__ import annotations

import uuid

from docmind.config import Config
from docmind.ingest.parsers import ParsedDoc
from docmind.store.vectordb import Chunk


def _split_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []
    out: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + max_words])
        if piece.strip():
            out.append(piece)
        if start + max_words >= len(words):
            break
    return out


def build_chunks(
    cfg: Config,
    doc_id: str,
    source_path: str,
    parsed: ParsedDoc,
    diagram_descriptions: list[tuple[str, int, str]],
    group: str = "default",
) -> list[Chunk]:
    """diagram_descriptions: list of (description, page, section)."""
    max_words = int(cfg.chunk_tokens / 1.3)
    overlap_words = int(cfg.chunk_overlap / 1.3)
    chunks: list[Chunk] = []

    for block in parsed.blocks:
        for piece in _split_words(block.text, max_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=doc_id,
                    source_path=source_path,
                    title=parsed.title,
                    section=block.section,
                    kind="text",
                    page=block.page,
                    text=piece,
                    vector=[],  # filled by the embedder before storage
                    group=group,
                )
            )

    for desc, page, section in diagram_descriptions:
        chunks.append(
            Chunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                source_path=source_path,
                title=parsed.title,
                section=section,
                kind="diagram",
                page=page,
                text=f"[Diagram] {desc}",
                vector=[],
                group=group,
            )
        )

    return chunks
