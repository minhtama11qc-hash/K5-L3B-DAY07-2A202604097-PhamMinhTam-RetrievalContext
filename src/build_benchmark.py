"""Build the Day 7 benchmark from the Day 6 questions + questions written for Day 7.

Split (decided before any Day 7 run):
  dev  = all 24 Day 6 questions (q01-q24, already seen in Day 6) + 3 new unanswerable (u01-u03)
  test = 8 new answerable (t01-t08, pages no Day 6 question used) + 3 new unanswerable (u04-u06)

Validation: every gold evidence must lie wholly inside ONE Day 6 chunk on the gold page
(normalized match); otherwise the build fails. Run:
  python -m src.build_benchmark --day6-questions <Day6 repo>/benchmark/questions.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import BENCH, CASES_FILE, CHUNKS_FILE, QUESTIONS_FILE, is_hit, read_json, write_json

SOURCE_ID = "openstax-clinical-nursing-skills"
DEFAULT_DAY6 = Path("../../Day06/K5-L3B-DAY06-PhamMinhTam-2A202604097-DataIngestion/benchmark/questions.json")


def from_day6(q: dict) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    gold += [{"source_id": a["gold_source_id"], "page": a["gold_page"], "evidence": a["evidence_quote"]}
             for a in q.get("alternative_gold", [])]
    out = {"id": q["id"], "split": "dev", "day6_split": q["split"], "type": q["type"], "question": q["question"],
           "expected_answer": q["expected_answer"], "gold": gold, "origin": "day6"}
    if q.get("filter"):
        out["filter"] = q["filter"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6-questions", type=Path, default=DEFAULT_DAY6)
    args = ap.parse_args()
    chunks = read_json(CHUNKS_FILE)
    day6 = read_json(args.day6_questions)["questions"]
    new = read_json(BENCH / "day7_new_questions.json")

    questions = [from_day6(q) for q in day6]
    for q in new["answerable"]:
        questions.append({**q, "gold": [{"source_id": SOURCE_ID, **g} for g in q["gold"]], "origin": "day7"})

    problems = []
    for q in questions:
        for g in q["gold"]:
            holders = [c["id"] for c in chunks if is_hit(c, [g])]
            g["chunk_ids"] = holders  # informational only; never given to retrieval or the agent
            if not holders:
                problems.append(f"{q['id']}: evidence not inside a single chunk on p{g['page']}: {g['evidence'][:60]}")
    if problems:
        raise SystemExit("benchmark invalid:\n  " + "\n  ".join(problems))

    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
              "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if q.get("filter") else {})}
             for q in questions]
    cases += [{"id": u["id"], "split": u["split"], "question": u["question"], "answerable": False,
               "expected_behavior": "search_and_abstain", "absence_reason": u["absence_reason"],
               "absence_check": u["absence_check"]} for u in new["unanswerable"]]
    order = {"dev": 0, "test": 1}
    cases.sort(key=lambda c: (order[c["split"]], not c["answerable"], c["id"]))

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate ids"
    write_json(QUESTIONS_FILE, questions)
    write_json(CASES_FILE, cases)
    for split in ("dev", "test"):
        a = sum(c["split"] == split and c["answerable"] for c in cases)
        u = sum(c["split"] == split and not c["answerable"] for c in cases)
        print(f"{split}: {a} answerable + {u} unanswerable")
    print(f"all {sum(len(q['gold']) for q in questions)} gold evidences lie inside one chunk on the gold page")


if __name__ == "__main__":
    main()
