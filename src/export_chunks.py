"""Export the Day 6 chunks (index B) from pgvector to data/chunks.json (Day 7 schema).

The lexical retrievers (BM25) need the chunk texts in memory; dense retrieval always
queries pgvector directly. Run:  python -m src.export_chunks
"""
from __future__ import annotations

import hashlib
import json

import psycopg

from .common import CHUNKS_FILE, DAY6_STRATEGY, env, load_env, write_json
from .day6.store import TABLE


def main() -> None:
    load_env()
    model = env("EMBEDDING_MODEL")
    with psycopg.connect(env("DATABASE_URL"), connect_timeout=5) as conn:
        rows = conn.execute(
            f"SELECT chunk_id, document_id, source, page, chapter, section, text, metadata, model_name, "
            f"content_hash FROM {TABLE} WHERE strategy = %s ORDER BY chunk_id", (DAY6_STRATEGY,)).fetchall()
    if not rows:
        raise SystemExit(f"no rows for strategy {DAY6_STRATEGY} in {TABLE}; run the Day 6 pipeline first")
    if {r[8] for r in rows} != {model}:
        raise SystemExit(f"index model {sorted({r[8] for r in rows})} != EMBEDDING_MODEL {model}")
    chunks = [{
        "id": r[0], "text": r[6], "source_id": r[1], "source": r[2], "page": r[3],
        "metadata": {**r[7], "chapter": r[4], "section": r[5], "day6_strategy": DAY6_STRATEGY,
                     "content_hash": r[9]},
    } for r in rows]
    write_json(CHUNKS_FILE, chunks)
    digest = hashlib.sha256(json.dumps([c["metadata"]["content_hash"] for c in chunks]).encode()).hexdigest()
    print(f"exported {len(chunks)} chunks (strategy {DAY6_STRATEGY}, model {model}) -> {CHUNKS_FILE}")
    print(f"index fingerprint (sha256 of content hashes): {digest[:16]}")


if __name__ == "__main__":
    main()
