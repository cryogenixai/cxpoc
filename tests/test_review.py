"""Review backend API tests (FastAPI TestClient, no running server, offline).

Builds a real job by running the pipeline with the mock VLM (no models, no
Stage 1), then points the app at that store. Validates the three API endpoints
plus the SPA being served.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pipeline.run import init_job, run_pipeline
from pipeline.storage import LocalFS
from review.app import create_app


@pytest.fixture
def client_and_job(tmp_path, sample_pdf, mock_vlm):
    store_root = tmp_path / "_jobs"
    store = LocalFS(store_root)
    ctx = init_job(store, sample_pdf)
    run_pipeline(ctx, vlm=mock_vlm)  # produces output/document.json + source.pdf
    app = create_app(store_root)
    return TestClient(app), ctx.job_id


def test_list_jobs(client_and_job):
    client, job_id = client_and_job
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert job_id in r.json()["jobs"]


def test_get_document(client_and_job):
    client, job_id = client_and_job
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    doc = r.json()
    assert doc["job_id"] == job_id
    assert len(doc["pages"]) == 1
    assert len(doc["pages"][0]["chunks"]) == 5


def test_get_pdf(client_and_job):
    client, job_id = client_and_job
    r = client.get(f"/api/jobs/{job_id}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_unknown_job_404(client_and_job):
    client, _ = client_and_job
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/pdf").status_code == 404


def test_serves_spa(client_and_job):
    client, _ = client_and_job
    r = client.get("/")
    assert r.status_code == 200
    assert "cxpoc review" in r.text


def test_empty_store_lists_nothing(tmp_path):
    app = create_app(tmp_path / "empty")
    client = TestClient(app)
    assert client.get("/api/jobs").json() == {"jobs": []}
