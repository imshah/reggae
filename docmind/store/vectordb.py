"""LanceDB-backed vector store.

One table holds vectors + chunk text + metadata. Per-document removal is a
single predicate delete (`doc_id = '...'`), which keeps the index clean when
documents are added and removed over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import lancedb
import pyarrow as pa

from docmind.config import Config

TABLE = "chunks"


DEFAULT_GROUP = "default"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    section: str
    kind: str          # "text" | "diagram"
    page: int
    text: str
    vector: list[float]
    group: str = DEFAULT_GROUP


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    section: str
    kind: str
    page: int
    text: str
    score: float
    group: str = DEFAULT_GROUP


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("section", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("group", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _q(value: str) -> str:
    """SQL-escape a string literal for a LanceDB predicate."""
    return value.replace("'", "''")


class VectorStore:
    def __init__(self, cfg: Config, dim: int | None = None):
        from docmind.config import LANCE_DIR, ensure_dirs

        self.cfg = cfg
        self.dim = dim
        ensure_dirs()
        self.db = lancedb.connect(str(LANCE_DIR))
        if TABLE in self.db.table_names():
            self.table = self.db.open_table(TABLE)
            self._migrate_group_column()
        elif dim is not None:
            self.table = self.db.create_table(TABLE, schema=_schema(dim))
        else:
            self.table = None  # created lazily on first add()

    def _migrate_group_column(self) -> None:
        """Backfill a `group` column on tables created before grouping existed."""
        try:
            if "group" in self.table.schema.names:
                return
            # add_columns takes SQL expressions; a literal backfills every row
            self.table.add_columns({"group": f"'{DEFAULT_GROUP}'"})
        except Exception:
            # older LanceDB without add_columns — group filters degrade to
            # "no match"; a `docmind add --force` re-index restores full support
            pass

    def _ensure_table(self, dim: int) -> None:
        if self.table is None:
            self.dim = dim
            self.table = self.db.create_table(TABLE, schema=_schema(dim))

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._ensure_table(len(chunks[0].vector))
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "title": c.title,
                "section": c.section,
                "kind": c.kind,
                "page": c.page,
                "text": c.text,
                "group": c.group,
                "vector": c.vector,
            }
            for c in chunks
        ]
        self.table.add(rows)

    def delete_doc(self, doc_id: str) -> None:
        if self.table is None:
            return
        self.table.delete(f"doc_id = '{_q(doc_id)}'")

    def search(
        self,
        query_vec: list[float],
        top_k: int,
        kind: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        """doc_ids: restrict to these documents; None = no restriction;
        empty list = no eligible docs (returns [])."""
        if self.table is None:
            return []
        if doc_ids is not None and not doc_ids:
            return []
        q = self.table.search(query_vec).limit(top_k)
        clauses: list[str] = []
        if kind:
            clauses.append(f"kind = '{_q(kind)}'")
        if doc_ids:
            ids = ", ".join(f"'{_q(d)}'" for d in doc_ids)
            clauses.append(f"doc_id IN ({ids})")
        if clauses:
            q = q.where(" AND ".join(clauses))
        results = q.to_list()
        hits: list[Hit] = []
        for r in results:
            # LanceDB returns L2 distance in _distance; convert to a similarity-ish score
            dist = float(r.get("_distance", 0.0))
            hits.append(
                Hit(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    source_path=r["source_path"],
                    title=r["title"],
                    section=r["section"],
                    kind=r["kind"],
                    page=int(r["page"]),
                    text=r["text"],
                    score=1.0 / (1.0 + dist),
                    group=r.get("group", DEFAULT_GROUP),
                )
            )
        return hits

    def count(self, doc_ids: list[str] | None = None) -> int:
        if self.table is None:
            return 0
        if doc_ids is not None and not doc_ids:
            return 0
        try:
            if doc_ids:
                ids = ", ".join(f"'{_q(d)}'" for d in doc_ids)
                return self.table.count_rows(f"doc_id IN ({ids})")
            return self.table.count_rows()
        except Exception:
            return 0
