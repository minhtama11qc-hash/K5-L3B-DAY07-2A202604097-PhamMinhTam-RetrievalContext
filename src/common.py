"""Shared paths, env loading, text normalization and the evidence-hit definition.

The hit definition is the same as Day 6: a chunk is an evidence hit when it comes from a
gold (source_id, page) and contains the *whole* gold evidence after normalization
(NFKC, ASCII quotes/dashes, collapsed whitespace, lower-case).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BENCH = DATA / "benchmark"
RESULTS = ROOT / "results"
CHUNKS_FILE = DATA / "chunks.json"
QUESTIONS_FILE = BENCH / "questions.json"       # answerable questions (gold evidence)
CASES_FILE = BENCH / "agent_cases.json"         # all agent cases incl. unanswerable

DAY6_STRATEGY = "B"   # Day 6 chosen index: paragraph chunks (C = B + optional filter)

_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
          "−": "-", " ": " "}


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)  # project .env wins over stale system variables


def env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set (see .env.example)")
    return value


def match_form(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _PUNCT.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    text = match_form(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and match_form(g["evidence"]) in text for g in gold)


def load_questions(split: str) -> list[dict[str, Any]]:
    """Answerable questions of ONE split only (never load test gold while tuning on dev)."""
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    return [q for q in read_json(QUESTIONS_FILE) if q["split"] == split]


def load_cases(split: str) -> list[dict[str, Any]]:
    """Agent inputs: id, question, optional user filter. No gold, no expected answer."""
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    return [{k: c[k] for k in ("id", "split", "question", "answerable", "filter") if k in c}
            for c in read_json(CASES_FILE) if c["split"] == split]
