"""Stage 1 layout detection tests.

- StubLayoutDetector: fast contract checks (no model).
- DocLayoutYOLODetector: golden-style property checks on a real page, skipped
  if the weights aren't present so the suite stays green without the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.detect import DocLayoutYOLODetector, StubLayoutDetector
from pipeline.detect.layout_model import DEFAULT_WEIGHTS, _dedup
from tests.conftest import make_minimal_pdf  # noqa: F401 (ensures import path)

ALLOWED_TYPES = {
    "title", "section_header", "paragraph", "list", "caption", "table",
    "figure", "chart", "diagram", "logo", "photo", "page_header",
    "page_footer", "footnote", "formula",
}

_TENK = Path("/Users/aasthana/work/data/10k-pdf/TSLA-2024.pdf")
_HAVE_WEIGHTS = Path(DEFAULT_WEIGHTS).exists()


def _png_of(pdf_path, page_index, dpi=200) -> bytes:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        return doc.load_page(page_index).get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


# -- Stub detector (always runs) ---------------------------------------------

def test_stub_returns_five_bands():
    png = _png_of_minimal()
    regions = StubLayoutDetector().detect(png, 0)
    assert [r["type"] for r in regions] == ["title", "paragraph", "table", "chart", "figure"]
    for r in regions:
        b = r["bbox"]
        assert b["x0"] < b["x1"] and b["y0"] < b["y1"]


def _png_of_minimal() -> bytes:
    """Render the in-repo minimal PDF to a PNG via PyMuPDF."""
    import fitz

    doc = fitz.open(stream=make_minimal_pdf(), filetype="pdf")
    try:
        return doc.load_page(0).get_pixmap(dpi=200).tobytes("png")
    finally:
        doc.close()


# -- Dedup unit test ----------------------------------------------------------

def test_dedup_keeps_higher_confidence_on_overlap():
    box = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    regions = [
        {"type": "title", "bbox": box, "detector_confidence": 0.3},
        {"type": "paragraph", "bbox": box, "detector_confidence": 0.9},
        {"type": "table", "bbox": {"x0": 200, "y0": 200, "x1": 300, "y1": 300}, "detector_confidence": 0.5},
    ]
    kept = _dedup(regions)
    assert len(kept) == 2  # overlapping pair collapsed to one
    overlap = [r for r in kept if r["bbox"] == box][0]
    assert overlap["type"] == "paragraph"  # higher confidence won


# -- Real DocLayout-YOLO golden checks (skipped without weights/PDF) ----------

@pytest.mark.skipif(not _HAVE_WEIGHTS, reason="DocLayout-YOLO weights not present")
def test_doclayout_types_in_schema():
    png = _png_of_minimal()
    regions = DocLayoutYOLODetector().detect(png, 0)
    assert all(r["type"] in ALLOWED_TYPES for r in regions)
    for r in regions:
        assert 0.0 <= r["detector_confidence"] <= 1.0


@pytest.mark.skipif(
    not (_HAVE_WEIGHTS and _TENK.exists()),
    reason="weights or 10-K corpus not present",
)
def test_doclayout_finds_table_on_financial_page():
    # p40 of TSLA-2024 is a financial/prose page that contains a table.
    regions = DocLayoutYOLODetector().detect(_png_of(_TENK, 40), 40)
    types = {r["type"] for r in regions}
    assert "paragraph" in types
    assert "table" in types
    # no duplicate boxes survive dedup
    boxes = [tuple(r["bbox"].values()) for r in regions]
    assert len(boxes) == len(set(boxes))
