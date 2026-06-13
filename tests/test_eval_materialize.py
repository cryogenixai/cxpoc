"""Integration test for gold-set materialization (stub pipeline, no models)."""

from __future__ import annotations

import csv
import json

import fitz

from eval.materialize import materialize
from pipeline import manifest as M
from pipeline.detect import StubLayoutDetector
from pipeline.jobctx import JobContext
from pipeline.run import init_job
from pipeline.stages import IngestStage, LayoutStage
from pipeline.storage import LocalFS


def _build_store(tmp_path, sample_pdf):
    """A mining store with one doc taken through ingest + (stub) layout."""
    store_root = tmp_path / "store"
    store = LocalFS(store_root)
    ctx = init_job(store, sample_pdf, job_id="sample")
    IngestStage().run(ctx)
    LayoutStage(detector=StubLayoutDetector()).run(ctx)
    return store_root


def _write_shortlist(path, gold_id="sample__p0000"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gold_id", "doc_id", "source", "page", "stratum"])
        w.writeheader()
        w.writerow({"gold_id": gold_id, "doc_id": "sample", "source": "test",
                    "page": 0, "stratum": "narrative"})


def test_materialize_freezes_versioned_units(tmp_path, sample_pdf):
    store_root = _build_store(tmp_path, sample_pdf)
    shortlist = tmp_path / "shortlist.csv"
    _write_shortlist(shortlist)

    m = materialize("v1", shortlist, tmp_path / "nope.json", store_root, tmp_path / "gold")

    assert m["version"] == "v1" and m["n_pages"] == 1
    gold = tmp_path / "gold" / "v1"
    unit = gold / "sample__p0000"
    assert (unit / "page.pdf").exists()
    assert (unit / "textlayer.json").exists()
    assert (unit / "page.png").exists()

    # manifest record carries provenance + checksum
    man = json.loads((gold / "gold_manifest.json").read_text())
    rec = man["pages"][0]
    assert rec["gold_id"] == "sample__p0000" and rec["source_page"] == 0
    assert len(rec["checksum"]) == 64

    # assembled PDF has exactly the selected page
    doc = fitz.open(gold / "gold_pages_v1.pdf")
    assert doc.page_count == 1
    doc.close()


def test_materialize_respects_curation_in_set(tmp_path, sample_pdf):
    store_root = _build_store(tmp_path, sample_pdf)
    shortlist = tmp_path / "shortlist.csv"
    _write_shortlist(shortlist)
    curation = tmp_path / "curation.json"
    curation.write_text(json.dumps({"sample__p0000": "out"}))  # excluded

    m = materialize("v1", shortlist, curation, store_root, tmp_path / "gold")
    # nothing marked "in" -> falls back to full shortlist (1 page)
    assert m["n_pages"] == 1
