"""Combine automatic agent metrics with manual labels (starter/agent_metrics.aggregate).

    python -m src.report_tables --split dev
Writes results/agent/<split>/agent_table.md and metrics_<config>.json for every config
that has a fully labelled reviews.json.
"""
from __future__ import annotations

import argparse
import sys

from .common import CASES_FILE, RESULTS, ROOT, read_json, write_json

sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402  (starter script, unchanged)


def frac(d) -> str:
    return f"{d['n']}/{d['N']}" + (f" ({d['n'] / d['N']:.0%})" if d["N"] else " (n/a)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    args = ap.parse_args()
    cases = [c for c in read_json(CASES_FILE) if c["split"] == args.split]
    base = RESULTS / "agent" / args.split
    rows = ["| Config | Tool use | Searches (dist.) | Grounded 0/1/2 | Mean /2 | Citation valid | Answers w/ citation | Abstention | Wrong abstention | Context evidence | Cost total | Median s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        auto = read_json(d / "summary_auto.json")
        reviews = read_json(d / "reviews.json")
        if any(r.get("answer_grade", r.get("abstention_correct")) is None for r in reviews):
            print(f"{d.name}: reviews not complete, skipped")
            continue
        m = aggregate(cases, read_json(d / "traces.json"), reviews)
        write_json(d / "metrics.json", {"split": args.split, "config": d.name, **m, "auto": auto})
        g = m["grounded_answer"]
        rows.append(f"| {d.name} | {frac(m['tool_use'])} | {auto['searches_distribution']} | {g['0']}/{g['1']}/{g['2']} | "
                    f"{g['mean']:.2f} | {frac(m['citation_validity'])} | {frac(m['answer_citation_coverage'])} | "
                    f"{frac(m['abstention'])} | {frac(m['false_abstention'])} | {auto['context_evidence']} | "
                    f"${auto['total_cost_usd']:.4f} | {auto['median_latency_s']} |")
    md = "\n".join(rows) + "\n"
    (base / "agent_table.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
