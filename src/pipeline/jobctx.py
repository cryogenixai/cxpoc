"""JobContext — the single argument every stage receives.

Generalises the design's ``run(self, job_dir)`` (§6) to work over the storage
abstraction: instead of a filesystem path, a stage gets a job id + a Storage
backend, and builds keys via ``ctx.key(...)``. All artifacts for a job live
under ``jobs/{job_id}/`` (§5), mapping 1:1 to S3 keys later.
"""

from __future__ import annotations

from dataclasses import dataclass

from .storage import Storage


@dataclass
class JobContext:
    job_id: str
    storage: Storage

    def key(self, *parts: str) -> str:
        """Build a storage key under this job's directory."""
        return "/".join(("jobs", self.job_id, *parts))

    # Thin pass-throughs scoped to the job (handy and intention-revealing).
    def read_json(self, *parts: str):
        return self.storage.read_json(self.key(*parts))

    def write_json(self, obj, *parts: str) -> None:
        self.storage.write_json(self.key(*parts), obj)

    def write_bytes(self, data: bytes, *parts: str) -> None:
        self.storage.write(self.key(*parts), data)

    def read_bytes(self, *parts: str) -> bytes:
        return self.storage.read(self.key(*parts))

    def exists(self, *parts: str) -> bool:
        return self.storage.exists(self.key(*parts))

    def list(self, *prefix_parts: str) -> list[str]:
        return self.storage.list(self.key(*prefix_parts))
