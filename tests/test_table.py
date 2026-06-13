"""Table handler / recognizer tests.

- StubTableRecognizer: fast, model-free (text-layer words -> one cell).
- TATRRecognizer: golden-style check on a real 10-K financial table, skipped
  when the weights/corpus aren't present.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pipeline.detect.layout_model import DEFAULT_WEIGHTS
from pipeline.tables import StubTableRecognizer, TATRRecognizer

_TENK = Path("/Users/aasthana/work/data/10k-pdf/TSLA-2024.pdf")
_HAVE = Path(DEFAULT_WEIGHTS).exists() and _TENK.exists()


def test_stub_recognizer_single_cell():
    words = [
        {"text": "Hello", "bbox": {"x0": 10, "y0": 10, "x1": 40, "y1": 20}},
        {"text": "World", "bbox": {"x0": 45, "y0": 10, "x1": 80, "y1": 20}},
    ]
    region = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    res = StubTableRecognizer().recognize(b"", region, words)
    assert res["html"] == "<table><tr><td>Hello World</td></tr></table>"
    assert res["source"] == "stub"


def _table_crop_from_tenk(page_index: int):
    """Render a 10-K page, find its table via DocLayout-YOLO, return
    (crop_png, region_bbox, page_words)."""
    import fitz
    from PIL import Image

    from pipeline.detect import DocLayoutYOLODetector

    scale = 200 / 72
    doc = fitz.open(_TENK)
    page = doc.load_page(page_index)
    png = page.get_pixmap(dpi=200).tobytes("png")
    words = [
        {"text": t, "bbox": {"x0": x0 * scale, "y0": y0 * scale, "x1": x1 * scale, "y1": y1 * scale}}
        for x0, y0, x1, y1, t, *_ in page.get_text("words")
    ]
    doc.close()

    tables = [r for r in DocLayoutYOLODetector().detect(png, page_index) if r["type"] == "table"]
    assert tables, "expected a table region on this page"
    b = tables[0]["bbox"]
    crop = Image.open(io.BytesIO(png)).convert("RGB").crop(
        (int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"]))
    )
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue(), b, words


@pytest.mark.skipif(not _HAVE, reason="DocLayout-YOLO weights or 10-K corpus not present")
def test_tatr_extracts_financial_table():
    crop, bbox, words = _table_crop_from_tenk(40)
    res = TATRRecognizer().recognize(crop, bbox, words)

    assert res["source"] == "tatr"
    assert res["n_rows"] >= 5 and res["n_cols"] >= 3   # a real multi-col table
    assert res["confidence"] > 0.5
    html = res["html"]
    assert html.startswith("<table>") and html.endswith("</table>")
    assert "<th" in html                                # header row detected
    assert "colspan" in html or "rowspan" in html       # spanning header
    # cell values filled from the text layer (known figures on this page)
    assert "61,870" in html
