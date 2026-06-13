"""Text metric: normalized similarity on whitespace-normalized text.

For born-digital pages, our text comes straight from the text layer, so this is
near-exact when present; the metric still flags dropped/garbled regions. Compares
the concatenated text of a page's text-class chunks (1.0 = identical)."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_similarity(pred: str, ref: str) -> float:
    a, b = _norm(pred), _norm(ref)
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    return 1.0 - _lev(a, b) / m if m else 1.0
