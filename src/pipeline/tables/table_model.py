"""Table recognizers. Each turns a table crop into HTML.

The crop is a sub-image of the page at the same resolution, so a TATR cell box
in crop pixel coords maps to page pixel coords by adding the region's origin —
that's how cells are filled from the page text layer (more accurate than OCRing
the crop, design §2).
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any, Protocol

_MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"


class TableRecognizer(Protocol):
    name: str

    def recognize(
        self, crop_png: bytes, region_bbox: dict[str, float], page_words: list[dict]
    ) -> dict[str, Any]:
        ...


# -- geometry / text helpers --------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _dedup_1d(items: list[tuple[list[float], float]], vertical: bool, thresh: float = 0.4):
    """Greedy suppression of overlapping row/column lines, keeping higher score."""
    def interval(box):
        return (box[1], box[3]) if vertical else (box[0], box[2])

    kept: list[tuple[list[float], float]] = []
    for box, score in sorted(items, key=lambda x: x[1], reverse=True):
        if all(_interval_iou(interval(box), interval(k[0])) < thresh for k in kept):
            kept.append((box, score))
    kept.sort(key=lambda x: interval(x[0])[0])
    return kept


def _words_in(box: tuple[float, float, float, float], words: list[dict]) -> str:
    """Text-layer words whose center falls in box (page pixel coords), in order."""
    hits = []
    for w in words:
        b = w["bbox"]
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            hits.append((round(cy), cx, w["text"]))
    hits.sort()
    return " ".join(t for _, _, t in hits)


def _cell(row_box: list[float], col_box: list[float]) -> list[float]:
    return [col_box[0], row_box[1], col_box[2], row_box[3]]


def _center_in(cell_box: list[float], span_box: list[float]) -> bool:
    cx, cy = (cell_box[0] + cell_box[2]) / 2, (cell_box[1] + cell_box[3]) / 2
    return span_box[0] <= cx <= span_box[2] and span_box[1] <= cy <= span_box[3]


def _to_page(box: list[float], ox: float, oy: float) -> tuple[float, float, float, float]:
    return (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)


def _row_is_header(row_box: list[float], headers: list[list[float]]) -> bool:
    ry0, ry1 = row_box[1], row_box[3]
    h = ry1 - ry0
    for hb in headers:
        overlap = max(0.0, min(ry1, hb[3]) - max(ry0, hb[1]))
        if h and overlap / h > 0.5:
            return True
    return False


def _build_html(rows, cols, headers, spans, region_bbox, words) -> tuple[str, int]:
    ox, oy = region_bbox["x0"], region_bbox["y0"]
    row_boxes = [b for b, _ in rows]
    col_boxes = [b for b, _ in cols]
    R, C = len(row_boxes), len(col_boxes)

    occupied = [[False] * C for _ in range(R)]
    anchors: dict[tuple[int, int], tuple[int, int, str]] = {}

    for span_box, _ in sorted(spans, key=lambda x: -((x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))):
        covered = [
            (i, j)
            for i in range(R)
            for j in range(C)
            if _center_in(_cell(row_boxes[i], col_boxes[j]), span_box)
        ]
        if not covered:
            continue
        i0, i1 = min(i for i, _ in covered), max(i for i, _ in covered)
        j0, j1 = min(j for _, j in covered), max(j for _, j in covered)
        if any(occupied[i][j] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)):
            continue
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                occupied[i][j] = True
        text = _words_in(_to_page(span_box, ox, oy), words)
        anchors[(i0, j0)] = (i1 - i0 + 1, j1 - j0 + 1, text)

    parts = ["<table>"]
    for i in range(R):
        parts.append("<tr>")
        tag = "th" if _row_is_header(row_boxes[i], headers) else "td"
        for j in range(C):
            if (i, j) in anchors:
                rs, cs, text = anchors[(i, j)]
                attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                parts.append(f"<{tag}{attrs}>{_esc(text)}</{tag}>")
            elif occupied[i][j]:
                continue
            else:
                text = _words_in(_to_page(_cell(row_boxes[i], col_boxes[j]), ox, oy), words)
                parts.append(f"<{tag}>{_esc(text)}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts), R * C


# -- recognizers --------------------------------------------------------------

class StubTableRecognizer:
    """No model: dump the text-layer words inside the region into one cell."""

    name = "stub-table"

    def recognize(self, crop_png, region_bbox, page_words):
        box = (region_bbox["x0"], region_bbox["y0"], region_bbox["x1"], region_bbox["y1"])
        text = _words_in(box, page_words)
        return {
            "html": f"<table><tr><td>{_esc(text)}</td></tr></table>",
            "confidence": 0.5,
            "n_rows": 1,
            "n_cols": 1,
            "source": "stub",
        }


class TATRRecognizer:
    """Table Transformer structure recognition. transformers/torch imported
    lazily; CPU by default (set CXPOC_DETECT_DEVICE to override)."""

    name = "tatr"

    def __init__(self, device: str | None = None, threshold: float = 0.5):
        self.device = device or os.environ.get("CXPOC_DETECT_DEVICE", "cpu")
        self.threshold = threshold
        self._model = None
        self._proc = None
        self._lock = threading.Lock()  # extract runs regions in a thread pool

    def _load(self):
        # Serialize load: the transformers lazy importer is not thread-safe, and
        # the model should be built exactly once and shared across worker threads.
        with self._lock:
            if self._model is None:
                from transformers import AutoImageProcessor, TableTransformerForObjectDetection

                self._proc = AutoImageProcessor.from_pretrained(_MODEL_ID)
                self._model = TableTransformerForObjectDetection.from_pretrained(_MODEL_ID)
                self._model.to(self.device).eval()
        return self._model, self._proc

    def recognize(self, crop_png, region_bbox, page_words):
        import torch
        from PIL import Image

        model, proc = self._load()
        with Image.open(io.BytesIO(crop_png)) as im:
            img = im.convert("RGB")
            cw, ch = img.size
            # The v1.1 preprocessor config ships only 'longest_edge'; DETR's
            # resize needs both. Supply a DETR-style size explicitly.
            inputs = proc(
                images=img,
                return_tensors="pt",
                size={"shortest_edge": 800, "longest_edge": 1000},
            ).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            res = proc.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=[(ch, cw)]
            )[0]

        id2label = model.config.id2label
        objs = [
            (id2label[int(l)], [float(x) for x in box], float(s))
            for l, box, s in zip(res["labels"], res["boxes"], res["scores"])
        ]
        rows = _dedup_1d([(b, s) for lab, b, s in objs if lab == "table row"], vertical=True)
        cols = _dedup_1d([(b, s) for lab, b, s in objs if lab == "table column"], vertical=False)
        headers = [b for lab, b, _ in objs if lab == "table column header"]
        spans = [(b, s) for lab, b, s in objs if lab == "table spanning cell"]

        if not rows or not cols:
            return {"html": "", "confidence": 0.0, "n_rows": len(rows), "n_cols": len(cols), "source": "tatr"}

        html, _ = _build_html(rows, cols, headers, spans, region_bbox, page_words)
        scores = [s for _, s in rows] + [s for _, s in cols]
        conf = sum(scores) / len(scores) if scores else 0.0
        return {
            "html": html,
            "confidence": conf,
            "n_rows": len(rows),
            "n_cols": len(cols),
            "source": "tatr",
        }


def make_table_recognizer(name: str | None = None) -> TableRecognizer:
    name = name or os.environ.get("CXPOC_TABLE_RECOGNIZER", "tatr")
    if name in ("stub", "stub-table"):
        return StubTableRecognizer()
    if name in ("tatr", "table-transformer"):
        return TATRRecognizer()
    raise ValueError(f"unknown table recognizer: {name!r}")
