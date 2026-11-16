"""Named agent configurations. Each one is run on dev; the chosen one (and the baseline) on test.

SELECTION CRITERION (written 2026-11-16 before any agent run, applied on dev only):
  1. Highest mean grounded-answer grade (/2) on the 24 answerable dev questions (manual labels).
  2. Must abstain correctly on >= 2/3 unanswerable dev questions; otherwise excluded.
  3. If two configs are within 0.05 of each other on (1): higher Context Evidence Rate,
     then fewer wrong abstentions, then lower median cost, then lower median latency.
The chosen config is frozen in results/agent/CHOSEN.json before the first test run.
"""
from .agent import AgentConfig

CONFIGS = {c.name: c for c in [
    # Baseline = Day 6 choice (C): dense pgvector + user filter, top-3, one tool.
    AgentConfig("baseline_dense", method="dense", top_k=3, budget=2600, max_searches=3,
                note="Day 6 strategy C as a search tool"),
    # Method 2 / 3 behind the same agent (single search tool).
    AgentConfig("bm25", method="bm25", top_k=3, budget=2600, max_searches=3,
                note="lexical BM25 only"),
    AgentConfig("hybrid", method="hybrid", top_k=3, budget=2600, max_searches=3,
                note="RRF(dense, bm25), 20 candidates each"),
    # Improvement 1: rerank hybrid candidates with a cross-encoder (H: better top-3 ordering -> better context).
    AgentConfig("rerank_hybrid", method="rerank_hybrid", top_k=3, budget=2600, max_searches=3,
                note="cross-encoder rerank of 20 hybrid candidates"),
    # Improvement 2: two tools (semantic + keyword) and a larger budget (H: agent can pick the
    # right tool per query and keep more evidence for multi-part questions).
    AgentConfig("multitool_k5", method="rerank_hybrid", top_k=5, budget=4800, max_searches=4, keyword_tool=True,
                note="rerank_hybrid search + BM25 keyword_search, top-5, 4800-char budget, 4 searches"),
    # ---- Added after reading dev errors of the 5 configs above (same selection criterion).
    # Improvement 3: dev error q02 = lead-in chunk p005:c003 retrieved, list in neighbor c004 never retrieved.
    # H: adding same-page neighbors of each admitted chunk puts the continuation into context.
    # Only change vs baseline: neighbors on + budget 4800 so neighbors can fit.
    AgentConfig("neighbors", method="dense", top_k=3, budget=4800, max_searches=3, neighbors=True,
                note="baseline + same-page neighbor chunks, 4800-char budget"),
    # Improvement 4: dev errors = (a) bm25 q02 answered 'interpreting' (Tanner) for an NCSBN question with a
    # valid but non-supporting citation; (b) 4-9 rejected citation formats per config (extra LLM calls).
    # H: prompt v2 (format example + entity check) removes (a) and reduces (b).
    AgentConfig("neighbors_v2", method="dense", top_k=3, budget=4800, max_searches=3, neighbors=True, prompt="v2",
                note="neighbors + prompt v2"),
]}
