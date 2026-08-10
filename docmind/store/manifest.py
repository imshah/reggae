"""Document registry: which documents are indexed, their hash, chunk counts.

The manifest is the source of truth for `list`/`remove` and for hash-based
dedupe on `add` (re-adding a changed file replaces its chunks).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from docmind.config import MANIFEST_PATH, ensure_dirs


@dataclass
class DocRecord:
    doc_id: str
    source_path: str
    sha256: str
    added_at: float
    chunk_count: int
    diagram_count: int


def doc_id_for(path: Path) -> str:
    """Stable id from the absolute path (independent of content)."""
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self) -> None:
        self.docs: dict[str, DocRecord] = {}
        self._load()

    def _load(self) -> None:
        if MANIFEST_PATH.exists():
            try:
                data = json.loads(MANIFEST_PATH.read_text())
                self.docs = {k: DocRecord(**v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError, TypeError):
                self.docs = {}

    def save(self) -> None:
        ensure_dirs()
        MANIFEST_PATH.write_text(
            json.dumps({k: asdict(v) for k, v in self.docs.items()}, indent=2)
        )

    def upsert(self, rec: DocRecord) -> None:
        self.docs[rec.doc_id] = rec
        self.save()

    def remove(self, doc_id: str) -> DocRecord | None:
        rec = self.docs.pop(doc_id, None)
        if rec:
            self.save()
        return rec

    def get(self, doc_id: str) -> DocRecord | None:
        return self.docs.get(doc_id)

    def unchanged(self, path: Path) -> bool:
        """True if the file is already indexed with the same content hash."""
        rec = self.docs.get(doc_id_for(path))
        return bool(rec and rec.sha256 == file_sha256(path))

    def all(self) -> list[DocRecord]:
        return sorted(self.docs.values(), key=lambda r: r.added_at)


def new_record(path: Path, chunk_count: int, diagram_count: int) -> DocRecord:
    return DocRecord(
        doc_id=doc_id_for(path),
        source_path=str(path.resolve()),
        sha256=file_sha256(path),
        added_at=time.time(),
        chunk_count=chunk_count,
        diagram_count=diagram_count,
    )
