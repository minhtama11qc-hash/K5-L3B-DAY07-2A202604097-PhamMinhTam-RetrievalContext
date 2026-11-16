"""Benchmark retrieval methods (no LLM) on ONE split with the original question text.

    python -m src.eval_retrieval --split dev
    python -m src.eval_retrieval --split dev --methods dense,rerank_dense --top-k 5 --tag k5

Per method: Hit@1, Evidence Hit@3, Hit@10, MRR@10, Context Evidence Rate (top-k packed
under the char budget), latency. Unanswerable questions are not in the denominators.
Outputs: results/retrieval/<split>[-tag]/{summary.json, summary.md, traces/<method>.json}
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from .common import CHUNKS_FILE, RESULTS, env, is_hit, load_env, load_questions, read_json, write_json
from .context import pack

ALL_METHODS = ["dense", "bm25", "hybrid", "rerank_hybrid", "rerank_dense"]
DIAG_K = 10


def build(methods: list[str], chunks: list[dict[str, Any]], candidates: int = 20) -> dict[str, Any]:
    from .retrievers import BM25Retriever, CrossEncoderRerank, DenseRetriever, HybridRRF
    load_env()
    built: dict[str, Any] = {}
    need_dense = any(m in ("dense", "hybrid", "rerank_hybrid", "rerank_dense") for m in methods)
    need_bm25 = any(m in ("bm25", "hybrid", "rerank_hybrid") for m in methods)
    dense = DenseRetriever(env("DATABASE_URL"), env("EMBEDDING_MODEL"), chunks) if need_dense else None
    bm25 = BM25Retriever(chunks) if need_bm25 else None
    hybrid = HybridRRF(dense, bm25, candidates=candidates) if dense and bm25 else None
    for m in methods:
        if m == "dense":
            built[m] = dense
        elif m == "bm25":
            built[m] = bm25
        elif m == "hybrid":
            built[m] = hybrid
        elif m == "rerank_hybrid":
            built[m] = CrossEncoderRerank(hybrid, candidates=candidates)
        elif m == "rerank_dense":
            built[m] = CrossEncoderRerank(dense, candidates=candidates)
        else:
            raise SystemExit(f"unknown method {m}; choose from {ALL_METHODS}")
    return built


def run(retriever, questions, *, top_k: int, budget: int, use_filter: bool, neighbors: bool,
        by_id: dict[str, Any]) -> list[dict[str, Any]]:
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=max(top_k, DIAG_K))
        ms = (time.perf_counter() - t0) * 1000
        flags = [is_hit(r, q["gold"]) for r in results]
        context, ids, dropped = pack(results[:top_k], budget, by_id=by_id, neighbors=neighbors)
        ctx_hit = any(is_hit(by_id[i], q["gold"]) for i in ids)
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        traces.append({
            "query_id": q["id"], "split": q["split"], "question": q["question"], "filter": filters,
            "top_k": top_k, "budget_chars": budget, "neighbors": neighbors,
            "results": [{"rank": r["rank"], "id": r["id"], "page": r["page"],
                         "section": r["metadata"].get("section"), "score": r["score"], "hit": f,
                         "text": r["text"][:300]} for r, f in zip(results, flags)],
            "context_ids": ids, "dropped_by_budget": dropped, "context_chars": len(context),
            "hit1": bool(flags[:1] and flags[0]), "hit3": any(flags[:3]), "hit10": any(flags[:10]),
            "first_hit_rank": first, "rr10": 1 / first if first else 0.0, "context_hit": ctx_hit,
            "latency_ms": round(ms, 1),
        })
    return traces


def summarize(name: str, traces: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    n = len(traces)
    cnt = lambda k: sum(t[k] for t in traces)
    return {
        "method": name, **cfg, "N": n,
        "hit1": f"{cnt('hit1')}/{n}", "hit1_rate": round(cnt("hit1") / n, 3),
        "hit3": f"{cnt('hit3')}/{n}", "hit3_rate": round(cnt("hit3") / n, 3),
        "hit10": f"{cnt('hit10')}/{n}", "mrr10": round(statistics.mean(t["rr10"] for t in traces), 3),
        "context_evidence": f"{cnt('context_hit')}/{n}", "context_evidence_rate": round(cnt("context_hit") / n, 3),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
        "avg_context_tokens_est": round(statistics.mean(t["context_chars"] for t in traces) / 4),
        "median_latency_ms": round(statistics.median(t["latency_ms"] for t in traces), 1),
        "misses_at3": [t["query_id"] for t in traces if not t["hit3"]],
        "misses_at1": [t["query_id"] for t in traces if not t["hit1"]],
        "cost_usd": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=2600, help="context budget in characters")
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--no-filter", action="store_true", help="ignore user-provided filters")
    ap.add_argument("--neighbors", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    chunks = read_json(CHUNKS_FILE)
    by_id = {c["id"]: c for c in chunks}
    questions = load_questions(args.split)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    t0 = time.perf_counter()
    built = build(methods, chunks, args.candidates)
    load_s = time.perf_counter() - t0
    out = RESULTS / "retrieval" / (args.split + (f"-{args.tag}" if args.tag else ""))
    cfg = {"top_k": args.top_k, "budget_chars": args.budget, "candidates": args.candidates,
           "use_filter": not args.no_filter, "neighbors": args.neighbors, "split": args.split}
    summaries = []
    for m in methods:
        traces = run(built[m], questions, top_k=args.top_k, budget=args.budget, use_filter=not args.no_filter,
                     neighbors=args.neighbors, by_id=by_id)
        write_json(out / "traces" / f"{m}.json", traces)
        summaries.append(summarize(m, traces, cfg))
    write_json(out / "summary.json", {"model_load_seconds": round(load_s, 1), "methods": summaries})
    lines = ["| Method | Hit@1 | Evidence Hit@3 | Hit@10 | MRR@10 | Context Evidence | Avg ctx chars | Median ms | Misses@3 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['hit1']} ({s['hit1_rate']:.0%}) | {s['hit3']} ({s['hit3_rate']:.0%}) | "
                     f"{s['hit10']} | {s['mrr10']} | {s['context_evidence']} ({s['context_evidence_rate']:.0%}) | "
                     f"{s['avg_context_chars']} | {s['median_latency_ms']} | {', '.join(s['misses_at3']) or '-'} |")
    md = f"Split {args.split}, top_k={args.top_k}, budget={args.budget} chars, filter={'on' if not args.no_filter else 'off'}, neighbors={args.neighbors}\n\n" + "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
