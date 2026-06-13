"""L3 — end-to-end smoke test through the CLI entrypoint.

One small PDF through the full pipeline -> schema-valid document.json. Catches
wiring breaks (storage, manifest, stage ordering, CLI plumbing).
"""

from __future__ import annotations

from pipeline.run import main
from pipeline.storage import LocalFS
from pipeline.jobctx import JobContext
from pipeline import manifest as M
from pipeline.schema import validate


def test_cli_end_to_end(tmp_path, sample_pdf, capsys):
    store = tmp_path / "store"
    rc = main(["--input", str(sample_pdf), "--store", f"file://{store}"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "complete" in out

    # Recover the job id from the printed line and validate the artifact.
    job_id = out.split("job ", 1)[1].split(" ", 1)[0]
    ctx = JobContext(job_id=job_id, storage=LocalFS(store))
    assert M.load(ctx)["status"] == M.DONE
    validate(ctx.read_json("output", "document.json"))


def test_cli_missing_input_returns_2(tmp_path):
    rc = main(["--input", str(tmp_path / "nope.pdf"), "--store", f"file://{tmp_path}/s"])
    assert rc == 2
