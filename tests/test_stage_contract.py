"""L1 — per-stage contract: schema-valid output, manifest transitions,
idempotency (run twice ≡ run once), and resume after a simulated crash.

Models are mocked (mock VLM), so these run fast on every change.
"""

from __future__ import annotations

from pipeline import manifest as M
from pipeline.run import build_stages, run_pipeline
from pipeline.schema import validate


def _run_stage(name, job, vlm):
    for stage in build_stages(vlm):
        if stage.name == name:
            stage.run(job)
            return stage
    raise AssertionError(name)


def test_ingest_outputs_and_manifest(job, mock_vlm):
    _run_stage("ingest", job, mock_vlm)
    m = M.load(job)
    assert M.stage_status(m, "ingest") == M.DONE
    assert job.exists("pages", "pages.json")
    assert job.exists("pages", "p0000.png")
    words = job.read_json("pages", "p0000.words.json")
    assert any(w["text"] == "Hello" for w in words)
    assert m["source"]["sha256"]  # digest recorded


def test_layout_covers_all_handler_routes(job, mock_vlm):
    _run_stage("ingest", job, mock_vlm)
    _run_stage("layout", job, mock_vlm)
    regions = job.read_json("layout", "p0000.regions.json")
    types = {r["type"] for r in regions}
    # figure was routed by the (mock) router to a concrete image type.
    assert "figure" not in types
    assert {"title", "paragraph", "table", "chart", "diagram"} <= types


def test_text_handler_echoes_clipped_words(job, mock_vlm):
    for name in ("ingest", "layout", "extract"):
        _run_stage(name, job, mock_vlm)
    title = job.read_json("extracted", "p0000_r00.json")
    assert title["type"] == "title"
    assert title["content"]["text"] == "Hello Cryogenic"


def test_full_pipeline_schema_valid(job, mock_vlm):
    doc = run_pipeline(job, vlm=mock_vlm)
    validate(doc)  # raises on failure
    assert doc["schema_version"] == "1.0"
    assert len(doc["pages"]) == 1
    chunks = doc["pages"][0]["chunks"]
    assert len(chunks) == 5
    # reading_order is dense and starts at 0.
    assert [c["reading_order"] for c in chunks] == [0, 1, 2, 3, 4]
    # bboxes normalised into 0..1.
    for c in chunks:
        for v in c["bbox"].values():
            assert 0.0 <= v <= 1.0


def test_idempotent_run_twice_equals_once(job, mock_vlm):
    doc1 = run_pipeline(job, vlm=mock_vlm)
    doc2 = run_pipeline(job, vlm=mock_vlm)  # all stages already done -> skip
    assert doc1 == doc2
    m = M.load(job)
    assert m["status"] == M.DONE


def test_resume_after_simulated_crash(job, mock_vlm):
    run_pipeline(job, vlm=mock_vlm)

    # Simulate a crash that lost the assemble output and left it not-done.
    m = M.load(job)
    M.set_stage(m, "assemble", M.PENDING)
    M.save(job, m)
    job.storage.write(job.key("output", "document.json"), b"{}")  # clobbered

    # Re-running resumes: earlier stages skip, assemble re-produces valid output.
    doc = run_pipeline(job, vlm=mock_vlm)
    validate(doc)
    assert len(doc["pages"][0]["chunks"]) == 5


def test_resume_partial_extract(job, mock_vlm):
    # Run through layout, then mark one region done with a sentinel artifact and
    # confirm extract leaves it untouched while filling the rest.
    for name in ("ingest", "layout"):
        _run_stage(name, job, mock_vlm)

    m = M.load(job)
    M.set_region(m, "p0000_r00", M.DONE, handler="text")
    M.save(job, m)
    job.write_json({"region_id": "p0000_r00", "page_index": 0, "type": "title",
                    "bbox": {"x0": 100, "y0": 100, "x1": 1554, "y1": 300},
                    "content": {"text": "SENTINEL"}, "source": "x",
                    "confidence": 1.0, "model": None, "timings": {}},
                   "extracted", "p0000_r00.json")

    _run_stage("extract", job, mock_vlm)
    # The pre-marked region was not re-extracted (sentinel survives).
    assert job.read_json("extracted", "p0000_r00.json")["content"]["text"] == "SENTINEL"
    # Others were filled.
    assert job.exists("extracted", "p0000_r03.json")
