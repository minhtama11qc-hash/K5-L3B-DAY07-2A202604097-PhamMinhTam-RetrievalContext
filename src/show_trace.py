"""Print agent traces in a compact form for manual grading.

    python -m src.show_trace --split dev --config baseline_dense [--ids q02,q10] [--full]
"""
from __future__ import annotations

import argparse

from .common import CASES_FILE, CHUNKS_FILE, QUESTIONS_FILE, RESULTS, read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--full", action="store_true", help="print full text of cited chunks")
    args = ap.parse_args()
    by_id = {c["id"]: c for c in read_json(CHUNKS_FILE)}
    qs = {q["id"]: q for q in read_json(QUESTIONS_FILE)}
    cases = {c["id"]: c for c in read_json(CASES_FILE)}
    traces = read_json(RESULTS / "agent" / args.split / args.config / "traces.json")
    keep = set(args.ids.split(",")) if args.ids else None
    for t in traces:
        if keep and t["query_id"] not in keep:
            continue
        qid = t["query_id"]
        print(f"\n=== {qid} [{'answerable' if cases[qid]['answerable'] else 'UNANSWERABLE'}] {t['question']}")
        if qid in qs:
            print(f"  expected: {qs[qid]['expected_answer']}")
            print(f"  gold chunks: {[i for g in qs[qid]['gold'] for i in g['chunk_ids']]}")
        for e in t["events"]:
            if e["event"] == "search":
                print(f"  > {e['tool']}({e['query']!r}) -> {[r['id'].split(':', 2)[2] for r in e['results']]} admitted={len(e['admitted'])}")
            elif e["event"] not in ("llm_call",):
                print(f"  > {e['event']} {({k: v for k, v in e.items() if k != 'event'})}")
        print(f"  status={t['status']} auto={t.get('auto')}")
        print(f"  answer: {t['answer']}")
        for c in t["citations"]:
            text = by_id[c]["text"]
            print(f"    [{c.split(':', 2)[2]}] {text if args.full else text[:350]}")


if __name__ == "__main__":
    main()
