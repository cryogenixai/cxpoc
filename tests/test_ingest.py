"""Stage 0 (real PyMuPDF ingest) — property/golden-style tests.

Asserts *properties* of the parse rather than exact bytes (the render is a real
rasterization): page count, triage, plausible dimensions, word extraction with
in-bounds boxes, and the scanned-path branch.
"""

from __future__ import annotations

from pipeline import manifest as M
from pipeline.run import init_job
from pipeline.stages import IngestStage


def _ingest(storage, pdf_path):
    ctx = init_job(storage, pdf_path)
    IngestStage().run(ctx)
    return ctx


def test_digital_pdf_triage_and_dims(storage, sample_pdf):
    ctx = _ingest(storage, sample_pdf)
    pages = ctx.read_json("pages", "pages.json")
    assert len(pages) == 1
    page = pages[0]
    assert page["page_kind"] == "digital"
    assert page["dpi"] == 200
    # 612x792 pt MediaBox @ 200 DPI -> 1700x2200 px.
    assert page["width_px"] == 1700
    assert page["height_px"] == 2200
    assert M.load(ctx)["source"]["pages"] == 1


def test_words_extracted_in_bounds(storage, sample_pdf):
    ctx = _ingest(storage, sample_pdf)
    page = ctx.read_json("pages", "pages.json")[0]
    words = ctx.read_json("pages", "p0000.words.json")
    texts = [w["text"] for w in words]
    assert texts == ["Hello", "Cryogenic"]
    for w in words:
        b = w["bbox"]
        assert 0 <= b["x0"] < b["x1"] <= page["width_px"]
        assert 0 <= b["y0"] < b["y1"] <= page["height_px"]


def test_render_is_real_png(storage, sample_pdf):
    ctx = _ingest(storage, sample_pdf)
    png = ctx.read_bytes("pages", "p0000.png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    assert len(png) > 100  # not the 1x1 placeholder


def test_scanned_pdf_branch(storage, blank_pdf):
    ctx = _ingest(storage, blank_pdf)
    page = ctx.read_json("pages", "pages.json")[0]
    assert page["page_kind"] == "scanned"
    assert ctx.read_json("pages", "p0000.words.json") == []


def test_ingest_idempotent(storage, sample_pdf):
    ctx = _ingest(storage, sample_pdf)
    sha1 = M.load(ctx)["source"]["sha256"]
    IngestStage().run(ctx)  # second run is a no-op
    assert M.load(ctx)["source"]["sha256"] == sha1
    assert M.stage_status(M.load(ctx), "ingest") == M.DONE
