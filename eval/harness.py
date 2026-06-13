"""Eval harness: score our pipeline output against reference (silver) per gold_id,
then aggregate per-component metrics overall and per stratum (design §6).

Reference is the Landing.ai silver map today; the same code consumes gold labels
later. Per-component scorecards are kept separate (not averaged) — each component
fails differently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.landing_map import load_and_map
from eval.metrics.layout import (
    area_prf, content_coverage_area, coverage_areas, iou, layout_scores, _prf,
)
from eval.metrics.table import teds
from eval.metrics.text import text_similarity
from eval.taxonomy import COARSE, coarse_ours


def _our_chunks(doc: dict) -> list[dict]:
    """Flatten our document.json (single page) to {coarse, type, bbox, text, html, ro}."""
    out = []
    for page in doc.get("pages", []):
        for c in page.get("chunks", []):
            content = c.get("content") or {}
            out.append({
                "coarse": coarse_ours(c["type"]),
                "type": c["type"],
                "bbox": c["bbox"],
                "text": content.get("text", ""),
                "html": content.get("html", ""),
                "ro": c.get("reading_order", 0),
            })
    return out


def _page_text(chunks: list[dict], by_ro: bool) -> str:
    text_chunks = [c for c in chunks if c["coarse"] == "text"]
    key = (lambda c: c["ro"]) if by_ro else (lambda c: (c["bbox"]["y0"], c["bbox"]["x0"]))
    return " ".join(c["text"] for c in sorted(text_chunks, key=key) if c["text"])


def _match_tables(pred: list[dict], ref: list[dict], thresh: float = 0.5):
    """Yield (pred_table, ref_table) pairs by greedy IoU, both with html."""
    pt = [c for c in pred if c["coarse"] == "table" and c["html"]]
    rt = [c for c in ref if c["coarse"] == "table" and c["html"]]
    pairs = sorted(
        ((iou(p["bbox"], r["bbox"]), pi, ri) for pi, p in enumerate(pt) for ri, r in enumerate(rt)),
        reverse=True,
    )
    used_p, used_r = set(), set()
    for v, pi, ri in pairs:
        if v < thresh or pi in used_p or ri in used_r:
            continue
        used_p.add(pi); used_r.add(ri)
        yield pt[pi], rt[ri]


def evaluate(version: str, pred_root: Path) -> dict[str, Any]:
    gold = Path("gold") / version
    manifest = json.loads((gold / "gold_manifest.json").read_text())
    silver = load_and_map(gold / "silver_landing_ai.json", gold / "gold_manifest.json")
    stratum_of = {p["gold_id"]: p["stratum"] for p in manifest["pages"]}

    # accumulators
    layout_acc = {cls: [0, 0, 0] for cls in COARSE}    # box-match tp, fp, fn (secondary)
    cover_acc = {cls: [0, 0, 0] for cls in COARSE}     # per-class area inter, pred, ref (primary)
    content_acc = [0, 0, 0]                             # class-agnostic localization
    teds_scores: list[float] = []
    text_scores: list[float] = []
    strat: dict[str, dict] = {}

    for rec in manifest["pages"]:
        gid = rec["gold_id"]
        st = stratum_of[gid]
        doc_path = pred_root / "jobs" / gid / "output" / "document.json"
        if not doc_path.exists():
            continue
        ours = _our_chunks(json.loads(doc_path.read_text()))
        ref = silver[gid]["chunks"]

        sb = strat.setdefault(st, {"n": 0, "cover": {c: [0, 0, 0] for c in COARSE},
                                   "content": [0, 0, 0], "teds": [], "text": []})
        sb["n"] += 1

        # localization — class-agnostic content coverage
        ci, cp, cr = content_coverage_area(ours, ref)
        content_acc[0] += ci; content_acc[1] += cp; content_acc[2] += cr
        sb["content"][0] += ci; sb["content"][1] += cp; sb["content"][2] += cr

        # layout — primary: area coverage (granularity-agnostic)
        for cls, (inter, pa, ra) in coverage_areas(ours, ref, COARSE).items():
            cover_acc[cls][0] += inter; cover_acc[cls][1] += pa; cover_acc[cls][2] += ra
            sc = sb["cover"][cls]
            sc[0] += inter; sc[1] += pa; sc[2] += ra

        # layout — secondary: box-count match (segmentation agreement)
        ls = layout_scores(ours, ref, COARSE)
        for cls, prf in ls["per_class"].items():
            for i, k in enumerate(("tp", "fp", "fn")):
                layout_acc[cls][i] += prf[k]

        # tables (TEDS on IoU-matched pairs)
        for p, r in _match_tables(ours, ref):
            score = teds(p["html"], r["html"])
            teds_scores.append(score); sb["teds"].append(score)

        # text
        ts = text_similarity(_page_text(ours, by_ro=True), _page_text(ref, by_ro=False))
        text_scores.append(ts); sb["text"].append(ts)

    def box_micro(acc):
        tot = [sum(acc[c][i] for c in acc) for i in range(3)]
        return _prf(*tot)

    def cover_micro(acc):
        tot = [sum(acc[c][i] for c in acc) for i in range(3)]
        return area_prf(*tot)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "version": version,
        "reference": "landing_ai_silver",
        "n_pages": sum(s["n"] for s in strat.values()),
        # Localization only (class-agnostic): did we put a region where content is.
        "content_coverage": area_prf(*content_acc),
        # Primary: area-coverage P/R (granularity-agnostic).
        "layout_coverage": {
            "per_class": {c: area_prf(*cover_acc[c]) for c in COARSE
                          if sum(cover_acc[c]) > 0},
            "micro": cover_micro(cover_acc),
        },
        # Secondary: box-count match (sensitive to segmentation granularity).
        "layout_boxmatch": {
            "per_class": {c: _prf(*layout_acc[c]) for c in COARSE
                          if sum(layout_acc[c]) > 0},
            "micro": box_micro(layout_acc),
        },
        "table": {"n_matched_pairs": len(teds_scores), "mean_teds": mean(teds_scores)},
        "text": {"n_pages": len(text_scores), "mean_similarity": mean(text_scores)},
        "by_stratum": {
            st: {
                "n_pages": sb["n"],
                "content_coverage_f1": area_prf(*sb["content"])["f1"],
                "layout_coverage_f1": cover_micro(sb["cover"])["f1"],
                "table_n": len(sb["teds"]),
                "table_mean_teds": mean(sb["teds"]),
                "text_mean": mean(sb["text"]),
            }
            for st, sb in sorted(strat.items())
        },
    }
