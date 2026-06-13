"""FastAPI backend for the review tool.

Read-only API over a local job store (the same ``jobs/{job_id}/`` layout the
pipeline writes, via pipeline.storage.LocalFS). Serves the static SPA and three
JSON/binary endpoints. No auth, single-user, localhost — a dev aid.

Run:  python -m review.app [--store ./_jobs] [--port 8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the pipeline package importable when run directly (src/ layout), without
# requiring an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.jobctx import JobContext
from pipeline.storage import LocalFS

_STATIC = Path(__file__).resolve().parent / "static"


def _list_job_ids(store: LocalFS) -> list[str]:
    """Distinct job ids that have a document.json under jobs/{id}/output/."""
    ids = set()
    for key in store.list("jobs"):
        # key looks like jobs/{id}/...; collect ids that have produced output.
        parts = key.split("/")
        if len(parts) >= 4 and parts[2] == "output" and parts[3] == "document.json":
            ids.add(parts[1])
    return sorted(ids)


def create_app(store_root: str | Path = "./_jobs") -> FastAPI:
    store = LocalFS(store_root)
    app = FastAPI(title="cxpoc review", version="0.1.0")

    def ctx(job_id: str) -> JobContext:
        return JobContext(job_id=job_id, storage=store)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": _list_job_ids(store)}

    @app.get("/api/jobs/{job_id}")
    def get_document(job_id: str):
        c = ctx(job_id)
        if not c.exists("output", "document.json"):
            raise HTTPException(status_code=404, detail=f"no document.json for job {job_id}")
        return JSONResponse(c.read_json("output", "document.json"))

    @app.get("/api/jobs/{job_id}/pdf")
    def get_pdf(job_id: str):
        c = ctx(job_id)
        if not c.exists("source.pdf"):
            raise HTTPException(status_code=404, detail=f"no source.pdf for job {job_id}")
        # LocalFS path -> stream directly with the right content type.
        path = store.root / c.key("source.pdf")
        return FileResponse(path, media_type="application/pdf", filename=f"{job_id}.pdf")

    # SPA last so it doesn't shadow the API routes.
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review", description="cxpoc review tool")
    parser.add_argument("--store", default="./_jobs", help="local job store root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"cxpoc review → http://{args.host}:{args.port}  (store: {args.store})")
    uvicorn.run(create_app(args.store), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
