"""CLI entrypoint + the sequential pipeline runner (design §6).

A plain Python runner — ingest -> layout -> extract -> assemble — each stage
taking only the JobContext. No workflow framework: the manifest provides
resumability. Re-running a job is safe and resumes from the last completed unit.

Usage:
    python -m pipeline.run --input path/to/doc.pdf [--store URI] [--job-id ID]

    --store defaults to file://./_jobs (local artifact store). On the VM it is
    an s3:// URI; the code path is identical.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from . import manifest as M
from .detect import LayoutDetector, make_detector
from .jobctx import JobContext
from .stages import AssembleStage, ExtractStage, IngestStage, LayoutStage
from .storage import get_storage
from .vlm import VLMClient


def build_stages(vlm: VLMClient | None = None, detector: LayoutDetector | None = None):
    vlm = vlm or VLMClient()
    detector = detector or make_detector()
    return [
        IngestStage(),
        LayoutStage(detector=detector, vlm=vlm),
        ExtractStage(vlm=vlm),
        AssembleStage(),
    ]


def run_pipeline(
    ctx: JobContext,
    vlm: VLMClient | None = None,
    detector: LayoutDetector | None = None,
) -> dict:
    """Run all stages in order against an already-initialised job. Idempotent."""
    for stage in build_stages(vlm, detector):
        stage.run(ctx)
    return ctx.read_json("output", "document.json")


def init_job(storage, source_path: Path, job_id: str | None = None) -> JobContext:
    """Create a new job: assign an id, copy the source in, write the manifest."""
    job_id = job_id or uuid.uuid4().hex
    ctx = JobContext(job_id=job_id, storage=storage)
    ctx.write_bytes(source_path.read_bytes(), "source.pdf")
    manifest = M.new_manifest(job_id, source={"filename": source_path.name})
    M.save(ctx, manifest)
    return ctx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cxpoc", description="Document parsing pipeline (PoC)")
    parser.add_argument("--input", required=True, help="path to a source PDF")
    parser.add_argument("--store", default="file://./_jobs", help="artifact store URI")
    parser.add_argument("--job-id", default=None, help="reuse/resume a specific job id")
    args = parser.parse_args(argv)

    storage = get_storage(args.store)
    source = Path(args.input)
    if not source.exists():
        print(f"input not found: {source}", file=sys.stderr)
        return 2

    vlm = VLMClient()

    # Resume an existing job if --job-id points at one; otherwise create fresh.
    if args.job_id:
        ctx = JobContext(job_id=args.job_id, storage=storage)
        if not M.exists(ctx):
            ctx = init_job(storage, source, job_id=args.job_id)
    else:
        ctx = init_job(storage, source)

    document = run_pipeline(ctx, vlm=vlm)
    out_key = ctx.key("output", "document.json")
    print(f"job {ctx.job_id} complete: {len(document['pages'])} page(s) -> {out_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
