"""Text metric: word-token overlap (precision / recall / f1).

Goal (design §6): did we recover the page's text content? For born-digital that
should be near-exact. Whole-string edit distance is the wrong tool — it's wrecked
by chunking/class differences (spec text living in our tables vs ref's text),
markdown noise (### / ** / ---), and ordering (reading order is a separate
metric). So we compare the *multiset of word tokens* over ALL chunk text on the
page (tags stripped), which is agnostic to chunk boundaries, class, and order.

text_similarity (ordered char similarity) is kept as a secondary signal."""

from __future__ import annotations

import re
from collections import Counter

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_TOK = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> list[str]:
    """Lowercased word tokens with HTML/markdown tags stripped."""
    return _TOK.findall(_TAG.sub(" ", (s or "")).lower())


def token_overlap(pred: str, ref: str) -> tuple[int, int, int]:
    """(intersection, pred_total, ref_total) over token multisets."""
    cp, cr = Counter(tokens(pred)), Counter(tokens(ref))
    inter = sum((cp & cr).values())
    return inter, sum(cp.values()), sum(cr.values())


def token_prf(inter: int, pred_total: int, ref_total: int) -> dict[str, float]:
    if pred_total == 0 and ref_total == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}  # both empty = agree
    p = inter / pred_total if pred_total else 0.0
    r = inter / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


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
