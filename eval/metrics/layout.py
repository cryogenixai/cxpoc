"""Layout detection metric: per-coarse-class precision / recall / F1.

Greedy IoU matching between predicted and reference boxes of the same coarse
class (IoU >= threshold = a true positive). Reference here is the silver labels;
swap in gold later with no change.
"""

from __future__ import annotations

from typing import Any


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / (area_a + area_b - inter)


def match_class(pred: list[dict], ref: list[dict], thresh: float = 0.5) -> tuple[int, int, int]:
    """(tp, fp, fn) for one class via greedy highest-IoU matching."""
    pairs = []
    for pi, p in enumerate(pred):
        for ri, r in enumerate(ref):
            v = iou(p["bbox"], r["bbox"])
            if v >= thresh:
                pairs.append((v, pi, ri))
    pairs.sort(reverse=True)
    used_p, used_r = set(), set()
    tp = 0
    for _, pi, ri in pairs:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi); used_r.add(ri); tp += 1
    fp = len(pred) - tp
    fn = len(ref) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def layout_scores(pred_chunks: list[dict], ref_chunks: list[dict],
                  classes: list[str], thresh: float = 0.5) -> dict[str, Any]:
    """pred/ref chunks each: {coarse, bbox}. Returns per-class + micro-avg P/R/F1.

    Box-count matching — sensitive to segmentation granularity (N small boxes vs
    one merged box don't match), so kept as a secondary 'segmentation-agreement'
    diagnostic. Use coverage_areas/area_prf for the primary granularity-agnostic
    layout metric."""
    per_class = {}
    tot = [0, 0, 0]
    for cls in classes:
        p = [c for c in pred_chunks if c["coarse"] == cls]
        r = [c for c in ref_chunks if c["coarse"] == cls]
        if not p and not r:
            continue
        tp, fp, fn = match_class(p, r, thresh)
        per_class[cls] = _prf(tp, fp, fn)
        tot[0] += tp; tot[1] += fp; tot[2] += fn
    return {"per_class": per_class, "micro": _prf(*tot)}


# -- Area-coverage layout metric (granularity-agnostic) ----------------------

def _rasterize(boxes: list[dict], grid: int):
    """Union mask of normalized 0-1 boxes on a grid x grid bool array."""
    import numpy as np

    mask = np.zeros((grid, grid), dtype=bool)
    for b in boxes:
        x0 = max(0, min(grid, int(b["x0"] * grid)))
        x1 = max(0, min(grid, int(round(b["x1"] * grid))))
        y0 = max(0, min(grid, int(b["y0"] * grid)))
        y1 = max(0, min(grid, int(round(b["y1"] * grid))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def coverage_areas(pred_chunks: list[dict], ref_chunks: list[dict],
                   classes: list[str], grid: int = 1000) -> dict[str, tuple[int, int, int]]:
    """Per class -> (intersection_px, pred_area_px, ref_area_px) of the union
    masks. Segmentation-agnostic: many small boxes covering the same area as one
    big box yield the same mask. Accumulate across pages, then call area_prf."""
    out = {}
    for cls in classes:
        p = [c for c in pred_chunks if c["coarse"] == cls]
        r = [c for c in ref_chunks if c["coarse"] == cls]
        if not p and not r:
            continue
        pm = _rasterize([c["bbox"] for c in p], grid)
        rm = _rasterize([c["bbox"] for c in r], grid)
        out[cls] = (int((pm & rm).sum()), int(pm.sum()), int(rm.sum()))
    return out


def area_prf(inter: int, pred_area: int, ref_area: int) -> dict[str, float]:
    p = inter / pred_area if pred_area else 0.0
    r = inter / ref_area if ref_area else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
