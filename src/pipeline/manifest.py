"""Manifest — the single source of truth for job state (design §5).

The manifest is a JSON document at ``jobs/{job_id}/manifest.json``. It is the
state machine that makes stages idempotent and resumable: a stage consults it,
skips work already ``done``, and updates it atomically when finished. Per-region
status under the ``extract`` stage means a re-run resumes per region, not just
per stage.

Represented as a plain dict (easy to serialise, schema-flexible) with helper
functions rather than a class.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .jobctx import JobContext

MANIFEST_KEY = "manifest.json"

# Stage status values.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

STAGE_ORDER = ["ingest", "layout", "extract", "assemble"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_manifest(job_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": RUNNING,
        "source": source,
        "stages": {name: {"status": PENDING} for name in STAGE_ORDER},
    }


def load(ctx: JobContext) -> dict[str, Any]:
    return ctx.storage.read_json(ctx.key(MANIFEST_KEY))


def save(ctx: JobContext, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    # write_json -> atomic write in the storage layer.
    ctx.storage.write_json(ctx.key(MANIFEST_KEY), manifest)


def exists(ctx: JobContext) -> bool:
    return ctx.storage.exists(ctx.key(MANIFEST_KEY))


def stage_status(manifest: dict[str, Any], stage: str) -> str:
    return manifest["stages"].get(stage, {}).get("status", PENDING)


def set_stage(
    manifest: dict[str, Any],
    stage: str,
    status: str,
    **extra: Any,
) -> None:
    entry = manifest["stages"].setdefault(stage, {})
    entry["status"] = status
    if status == RUNNING:
        entry["started"] = _now()
    if status in (DONE, FAILED):
        entry["ended"] = _now()
    entry.update(extra)


# -- Per-region helpers for the extract stage --------------------------------

def region_status(manifest: dict[str, Any], region_id: str) -> str:
    regions = manifest["stages"].get("extract", {}).get("regions", {})
    return regions.get(region_id, {}).get("status", PENDING)


def set_region(
    manifest: dict[str, Any],
    region_id: str,
    status: str,
    **extra: Any,
) -> None:
    extract = manifest["stages"].setdefault("extract", {})
    regions = extract.setdefault("regions", {})
    entry = regions.setdefault(region_id, {})
    entry["status"] = status
    entry.update(extra)


def mark_complete(manifest: dict[str, Any]) -> None:
    manifest["status"] = DONE
