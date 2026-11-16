"""Run the agent on one split with one named config and save traces + automatic metrics.

    python -m src.run_agent --split dev --config baseline_dense
    python -m src.run_agent --split dev --config baseline_dense --only q02,u01

The agent receives only {question, user filter}. Gold evidence is loaded AFTER each run,
only to compute Context Evidence Rate for the report. Manual labels go to
results/agent/<split>/<config>/reviews.json (template written if missing).
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

from .agent import Agent
from .common import CHUNKS_FILE, RESULTS, is_hit, load_cases, load_questions, read_json, write_json
from .configs import CONFIGS
from .eval_retrieval import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--only", default="", help="comma-separated case ids (debugging)")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    if args.split == "test":
        chosen = RESULTS / "agent" / "CHOSEN.json"
        if not chosen.exists():
            raise SystemExit("freeze the chosen config in results/agent/CHOSEN.json before running test")
        frozen = read_json(chosen)
        if args.config not in (frozen["chosen"], frozen["baseline"]):
            raise SystemExit(f"test may only run the frozen configs {frozen['chosen']} / {frozen['baseline']}")

    chunks = read_json(CHUNKS_FILE)
    by_id = {c["id"]: c for c in chunks}
    methods = sorted({cfg.method, "bm25"} if cfg.keyword_tool else {cfg.method})
    retrievers = build(methods, chunks)
    agent = Agent(cfg, retrievers, by_id)

    cases = load_cases(args.split)
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    out = RESULTS / "agent" / args.split / cfg.name
    traces = []
    for c in cases:
        t0 = time.perf_counter()
        try:
            r = agent.run(c["question"], c.get("filter"))   # only question + user filter
        except Exception as exc:  # provider / store error: recorded separately, not as a quality result
            r = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "answer": None, "citations": [],
                 "events": [], "context_ids": [], "searches": 0, "cost_usd": 0.0,
                 "elapsed_s": round(time.perf_counter() - t0, 2)}
        traces.append({"query_id": c["id"], "split": c["split"], "question": c["question"],
                       "filter": c.get("filter"), **r})
        print(f"{c['id']}: {r['status']:8s} searches={r['searches']} ${r['cost_usd']:.5f} "
              f"{r['elapsed_s']}s  {(r.get('answer') or r.get('error') or '')[:90]}", flush=True)

    # ---- automatic metrics (gold used only here, after the runs)
    gold = {q["id"]: q["gold"] for q in load_questions(args.split)}
    answerable = {c["id"]: c["answerable"] for c in load_cases(args.split)}
    for t in traces:
        if answerable[t["query_id"]] and t["status"] != "error":
            t["auto"] = {"context_has_evidence": any(is_hit(by_id[i], gold[t["query_id"]]) for i in t["context_ids"]),
                         "cited_evidence_chunk": any(is_hit(by_id[i], gold[t["query_id"]]) for i in t["citations"])}
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if answerable[t["query_id"]]]
    una = [t for t in ok if not answerable[t["query_id"]]]
    summary = {
        "config": cfg.name, "split": args.split, "model": cfg.model, "cases": len(traces),
        "errors": [t["query_id"] for t in traces if t["status"] == "error"],
        "tool_use": f"{sum(t['searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(Counter(t["searches"] for t in ok).items())),
        "tools_used": dict(Counter(e["tool"] for t in ok for e in t["events"] if e["event"] == "search")),
        "context_evidence": f"{sum(t['auto']['context_has_evidence'] for t in ans)}/{len(ans)}",
        "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
        "wrong_abstentions": f"{sum(t['status'] == 'abstain' for t in ans)}/{len(ans)}",
        "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in una)}/{len(una)}",
        "cited_evidence_chunk": f"{sum(t['auto']['cited_evidence_chunk'] for t in ans)}/{len(ans)}",
        "answers_with_citations": f"{sum(bool(t['citations']) for t in ok if t['status'] == 'answered')}/{sum(t['status'] == 'answered' for t in ok)}",
        "total_citations": sum(len(t["citations"]) for t in ok),
        "invalid_citation_events": sum(e["event"] == "invalid_citations" for t in ok for e in t["events"]),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else 0,
        "total_prompt_tokens": sum(t["prompt_tokens"] for t in ok),
        "total_completion_tokens": sum(t["completion_tokens"] for t in ok),
        "total_cost_usd": round(sum(t["cost_usd"] for t in traces), 5),
        "median_cost_usd": round(statistics.median(t["cost_usd"] for t in ok), 6) if ok else 0,
        "median_latency_s": round(statistics.median(t["elapsed_s"] for t in ok), 2) if ok else 0,
        "note": "auto metrics only; grounded answer, citation support and abstention correctness are manual labels in reviews.json",
    }
    if not args.only:
        write_json(out / "traces.json", traces)
        write_json(out / "summary_auto.json", summary)
        reviews = out / "reviews.json"
        if not reviews.exists():
            write_json(reviews, [{"query_id": t["query_id"], "answerable": answerable[t["query_id"]],
                                  "citation_support": [None] * len(t["citations"]),
                                  **({"answer_grade": None} if answerable[t["query_id"]] else {"abstention_correct": None}),
                                  "reason": ""} for t in traces if t["status"] != "error"])
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
