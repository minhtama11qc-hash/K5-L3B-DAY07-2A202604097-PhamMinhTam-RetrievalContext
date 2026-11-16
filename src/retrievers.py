"""Retrieval methods compared in Day 7. All return the same result shape:

    {"id", "text", "source_id", "source", "page", "metadata", "score", "rank", "method"}

Methods
  dense   : Day 6 pgvector index B, all-MiniLM-L6-v2 query embedding, exact cosine search.
  bm25    : Okapi BM25 over the same 330 chunks (English tokenizer + small stop-word list).
  hybrid  : Reciprocal Rank Fusion (RRF, k=60) of dense and bm25 candidate lists; RRF uses
            ranks only, so the two score scales never get added.
  rerank  : candidates from hybrid (top-N), re-scored by a local cross-encoder
            (cross-encoder/ms-marco-MiniLM-L-6-v2) on (query, chunk text).

Filters are user-provided metadata (chapter / section) and are applied in every method:
in SQL for dense, as a pre-filter on the chunk list for bm25. Errors propagate; there is no
silent fallback to another method or to mock results.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

from .common import DAY6_STRATEGY, match_form

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an and are as at be been being but by can could did do does for from had has have how
if in into is it its of on or should that the their them they this to was were what when where which
who whom why will with would""".split())
FILTER_KEYS = ("chapter", "section")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(match_form(text)) if t not in STOPWORDS]


def _passes(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    for key, value in (filters or {}).items():
        if key not in FILTER_KEYS:
            raise ValueError(f"filter on '{key}' is not allowed")
        if str(chunk["metadata"].get(key)) != str(value):
            return False
    return True


def _ranked(items: list[tuple[dict[str, Any], float]], method: str, top_k: int) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda x: (-x[1], x[0]["id"]))[:top_k]
    return [{**c, "score": round(float(s), 6), "rank": i + 1, "method": method} for i, (c, s) in enumerate(items)]


class DenseRetriever:
    """Day 6 vector store, unchanged: same model, same table, same cosine search."""
    name = "dense"

    def __init__(self, database_url: str, model_name: str, chunks: list[dict[str, Any]]):
        from .day6.embeddings import SentenceTransformerEmbedder
        from .day6.store import PgVectorStore
        self.embedder = SentenceTransformerEmbedder(model_name)
        self.store = PgVectorStore(database_url, self.embedder.dimensions, model_name)
        self.by_id = {c["id"]: c for c in chunks}
        self.timing: dict[str, float] = {}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        rows = self.store.search(DAY6_STRATEGY, qvec, top_k, filters)
        self.timing = {"embed_ms": (t1 - t0) * 1000, "search_ms": (time.perf_counter() - t1) * 1000}
        out = []
        for r in rows:
            chunk = self.by_id.get(r["chunk_id"])
            if chunk is None:  # store and exported chunks.json disagree -> refuse to guess
                raise RuntimeError(f"{r['chunk_id']} not in data/chunks.json; re-run python -m src.export_chunks")
            out.append((chunk, r["score"]))
        return _ranked(out, self.name, top_k)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(term for tf in self.tf for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, dl in zip(self.chunks, self.tf, self.len):
            if not _passes(chunk, filters):
                continue
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((chunk, s))
        return _ranked(scored, self.name, top_k)


class HybridRRF:
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, candidates: int = 20, k: int = 60):
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        lists = [self.dense.search(query, filters=filters, top_k=self.candidates),
                 self.bm25.search(query, filters=filters, top_k=self.candidates)]
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, dict[str, Any]] = {}
        for results in lists:
            for r in results:
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (self.k + r["rank"])
                chunk_by_id[r["id"]] = r
        items = [({k: v for k, v in chunk_by_id[i].items() if k not in ("score", "rank", "method")}, s)
                 for i, s in fused.items()]
        return _ranked(items, self.name, top_k)


class CrossEncoderRerank:
    name = "rerank"
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, base, candidates: int = 20, model_name: str = MODEL):
        from sentence_transformers import CrossEncoder
        self.base, self.candidates, self.model_name = base, candidates, model_name
        self.model = CrossEncoder(model_name, device="cpu")
        self.name = f"rerank({base.name})"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        items = [({k: v for k, v in c.items() if k not in ("score", "rank", "method")}, s)
                 for c, s in zip(cands, scores)]
        return _ranked(items, self.name, top_k)
