"""Context selection: dedup, ordering, character budget, optional neighbor expansion.

Budget unit = characters of the packed context INCLUDING the "[id | source | page]" headers.
Approximate tokens = chars / 4 (English) and are reported alongside.
"""
from __future__ import annotations

import re
from typing import Any

_CID = re.compile(r"^(?P<prefix>.+:p)(?P<page>\d{3}):c(?P<idx>\d{3})$")


def neighbor_ids(chunk_id: str) -> list[str]:
    """IDs of the previous/next chunk on the same page (Day 6 IDs are doc:strategy:pNNN:cNNN)."""
    m = _CID.match(chunk_id)
    if not m:
        return []
    i = int(m["idx"])
    return [f"{m['prefix']}{m['page']}:c{j:03d}" for j in (i - 1, i + 1) if j >= 0]


def block(c: dict[str, Any]) -> str:
    sec = c["metadata"].get("section")
    return f"[{c['id']} | {c['source']} | page {c['page']} | section {sec}]\n{c['text']}"


def pack(candidates: list[dict[str, Any]], budget: int, *, by_id: dict[str, dict[str, Any]] | None = None,
         neighbors: bool = False) -> tuple[str, list[str], list[str]]:
    """Keep candidates in the given order, skip duplicates and blocks that do not fit.

    With neighbors=True, each selected chunk is followed by its same-page neighbors when
    they fit (helps when a list/heading is split across chunks). Returns (context, ids, dropped).
    """
    blocks: list[str] = []
    ids: list[str] = []
    dropped: list[str] = []
    used = 0

    def add(c: dict[str, Any]) -> bool:
        nonlocal used
        if c["id"] in ids:
            return True
        b = block(c)
        extra = len(b) + (2 if blocks else 0)
        if used + extra > budget:
            return False
        blocks.append(b)
        ids.append(c["id"])
        used += extra
        return True

    for c in candidates:
        if c["id"] in ids:
            continue
        if not add(c):
            dropped.append(c["id"])
            continue
        if neighbors and by_id:
            for nid in neighbor_ids(c["id"]):
                if nid in by_id and nid not in ids:
                    add(by_id[nid])
    return "\n\n".join(blocks), ids, dropped
