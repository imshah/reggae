"""Document registry: which documents are indexed, their hash, chunk counts.

The manifest is the source of truth for `list`/`remove` and for hash-based
dedupe on `add` (re-adding a changed file replaces its chunks).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from docmind.config import MANIFEST_PATH, ensure_dirs

DEFAULT_GROUP = "default"


@dataclass
class DocRecord:
    doc_id: str
    source_path: str
    sha256: str
    added_at: float
    chunk_count: int
    diagram_count: int
    groups: list[str] = field(default_factory=lambda: [DEFAULT_GROUP])

    @property
    def primary_group(self) -> str:
        """First group — used for the denormalized chunk column / displays."""
        return self.groups[0] if self.groups else DEFAULT_GROUP


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
                known = {f.name for f in fields(DocRecord)}
                self.docs = {
                    k: DocRecord(**_migrate_fields(v, known))
                    for k, v in data.items()
                }
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

    def all(self, group: str | None = None) -> list[DocRecord]:
        recs = self.docs.values()
        if group is not None:
            recs = [r for r in recs if group in r.groups]
        return sorted(recs, key=lambda r: r.added_at)

    def groups(self) -> list[str]:
        """Sorted distinct group names across all documents."""
        return sorted({g for r in self.docs.values() for g in r.groups})

    def add_to_group(self, doc_id: str, group: str) -> bool:
        """Add a group membership (no-op if already a member)."""
        rec = self.docs.get(doc_id)
        if not rec:
            return False
        if group not in rec.groups:
            rec.groups.append(group)
            self.save()
        return True

    def remove_from_group(self, doc_id: str, group: str) -> bool:
        """Drop a group membership; fall back to DEFAULT_GROUP if none remain."""
        rec = self.docs.get(doc_id)
        if not rec or group not in rec.groups:
            return False
        rec.groups = [g for g in rec.groups if g != group] or [DEFAULT_GROUP]
        self.save()
        return True


def _migrate_fields(raw: dict, known: set[str]) -> dict:
    """Coerce a stored record dict into current DocRecord fields.

    Handles the legacy single `group` scalar by mapping it to `groups`, and
    enforces the ≥1-group invariant.
    """
    data = {k: v for k, v in raw.items() if k in known}
    if "groups" not in data:
        legacy = raw.get("group")
        data["groups"] = [legacy] if legacy else [DEFAULT_GROUP]
    if not data.get("groups"):
        data["groups"] = [DEFAULT_GROUP]
    return data


def new_record(
    path: Path, chunk_count: int, diagram_count: int,
    groups: list[str] | None = None,
) -> DocRecord:
    return DocRecord(
        doc_id=doc_id_for(path),
        source_path=str(path.resolve()),
        sha256=file_sha256(path),
        added_at=time.time(),
        chunk_count=chunk_count,
        diagram_count=diagram_count,
        groups=list(groups) if groups else [DEFAULT_GROUP],
    )
