"""Store and search vectors using PostgreSQL + pgvector.

One table holds every strategy; (strategy, chunk_id) is the primary key, so re-ingesting
the same chunks updates rows in place instead of duplicating them. Search is an exact
(sequential-scan) cosine search: no ANN index is created, so filters never reduce recall
the way they can with approximate indexes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .models import Chunk

TABLE = "day6_chunks"
# Metadata fields a caller may filter on -> SQL column. Anything else is rejected.
FILTERABLE = {"chapter": "chapter", "section": "section", "document_id": "document_id", "page": "page"}


def content_hash(chunk: Chunk, model_name: str) -> str:
    payload = json.dumps([chunk.text, chunk.page, chunk.metadata, model_name], sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions
        self.model_name = model_name

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url, connect_timeout=5)
        register_vector(conn)
        return conn

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy      text    NOT NULL,
                    chunk_id      text    NOT NULL,
                    document_id   text    NOT NULL,
                    source        text    NOT NULL,
                    page          integer NOT NULL,
                    chapter       integer,
                    section       text,
                    text          text    NOT NULL,
                    metadata      jsonb   NOT NULL,
                    model_name    text    NOT NULL,
                    content_hash  text    NOT NULL,
                    embedding     vector({self.dimensions}) NOT NULL,
                    updated_at    timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            # Refuse to mix vectors of different sizes in one table.
            row = conn.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,)).fetchone()
            if row and row[0] != f"vector({self.dimensions})":
                raise RuntimeError(f"{TABLE}.embedding is {row[0]}, model gives vector({self.dimensions})")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_filter_idx ON {TABLE} (strategy, chapter, section)")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Write chunks with stable IDs; skip unchanged rows; delete obsolete chunks.

        Obsolete = rows of this strategy for the same document(s) whose chunk_id is not in
        the new set (e.g. the source changed and now yields fewer chunks).
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk_id inside one batch")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector has {len(v)} dims, expected {self.dimensions}")

        stats = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}
        with self._connect() as conn:
            existing = dict(conn.execute(
                f"SELECT chunk_id, content_hash FROM {TABLE} WHERE strategy = %s", (strategy,)).fetchall())
            with conn.cursor() as cur:
                for chunk, vector in zip(chunks, vectors):
                    digest = content_hash(chunk, self.model_name)
                    if existing.get(chunk.chunk_id) == digest:
                        stats["unchanged"] += 1
                        continue
                    stats["updated" if chunk.chunk_id in existing else "inserted"] += 1
                    cur.execute(f"""
                        INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, chapter,
                            section, text, metadata, model_name, content_hash, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                            page = EXCLUDED.page, chapter = EXCLUDED.chapter,
                            section = EXCLUDED.section, text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata, model_name = EXCLUDED.model_name,
                            content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding,
                            updated_at = now()""",
                        (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
                         chunk.metadata.get("chapter"), chunk.metadata.get("section"), chunk.text,
                         json.dumps(chunk.metadata, ensure_ascii=False), self.model_name, digest,
                         np.asarray(vector, dtype=np.float32)))
                documents = sorted({c.document_id for c in chunks})
                cur.execute(f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = ANY(%s) "
                            f"AND NOT (chunk_id = ANY(%s))", (strategy, documents, ids))
                stats["deleted"] = cur.rowcount
        return stats

    def count(self, strategy: str) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {TABLE} WHERE strategy = %s",
                                (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter in SQL, rank by cosine distance (exact search), return top-k.

        Only whitelisted metadata fields can be filtered; values are always passed as
        query parameters, never formatted into SQL.
        """
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        where = ["strategy = %s", "model_name = %s"]
        params: list[Any] = [strategy, self.model_name]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"filter on '{key}' is not allowed")
            where.append(f"{FILTERABLE[key]} = %s")
            params.append(value)
        vector = np.asarray(query_vector, dtype=np.float32)
        sql = (f"SELECT chunk_id, document_id, source, page, chapter, section, text, metadata, "
               f"embedding <=> %s AS distance FROM {TABLE} WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s LIMIT %s")
        with self._connect() as conn:
            rows = conn.execute(sql, [vector, *params, vector, top_k]).fetchall()
        return [{
            "rank": i + 1, "chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
            "chapter": r[4], "section": r[5], "text": r[6], "metadata": r[7],
            "score": round(1.0 - float(r[8]), 6),
        } for i, r in enumerate(rows)]
