"""L1 — manifest state-machine contract."""

from __future__ import annotations

from pipeline import manifest as M


def test_new_manifest_all_pending():
    m = M.new_manifest("job1", source={"filename": "x.pdf"})
    assert m["job_id"] == "job1"
    assert m["status"] == M.RUNNING
    for stage in M.STAGE_ORDER:
        assert M.stage_status(m, stage) == M.PENDING


def test_stage_transitions():
    m = M.new_manifest("j", source={})
    M.set_stage(m, "ingest", M.RUNNING)
    assert M.stage_status(m, "ingest") == M.RUNNING
    assert "started" in m["stages"]["ingest"]
    M.set_stage(m, "ingest", M.DONE, pages=3)
    assert M.stage_status(m, "ingest") == M.DONE
    assert m["stages"]["ingest"]["pages"] == 3
    assert "ended" in m["stages"]["ingest"]


def test_region_transitions():
    m = M.new_manifest("j", source={})
    assert M.region_status(m, "p0000_r00") == M.PENDING
    M.set_region(m, "p0000_r00", M.DONE, handler="text")
    assert M.region_status(m, "p0000_r00") == M.DONE
    assert m["stages"]["extract"]["regions"]["p0000_r00"]["handler"] == "text"


def test_save_load_roundtrip(job):
    m = M.load(job)
    M.set_stage(m, "ingest", M.DONE)
    M.save(job, m)
    reloaded = M.load(job)
    assert M.stage_status(reloaded, "ingest") == M.DONE
