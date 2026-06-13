"""Storage abstraction (design §11.3).

A flat key->bytes store with two backends behind one interface:

  * LocalFS  — a directory on disk (laptop dev, and the default for tests).
  * S3       — an S3 bucket, optionally pointed at LocalStack via AWS_S3_ENDPOINT.

The job-directory layout from §5 maps 1:1 to keys: ``jobs/{job_id}/...``. Keeping
this seam thin from day one is what makes the later jump to real S3 an
orchestration swap rather than a refactor.

Writes are atomic: a partially written artifact must never be observable, so a
crashed stage always resumes from a clean state.
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Storage(ABC):
    """Key->bytes store. Keys are ``/``-separated, relative, no leading slash."""

    @abstractmethod
    def read(self, key: str) -> bytes: ...

    @abstractmethod
    def write(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return all keys under ``prefix``, sorted."""

    # -- JSON / text convenience helpers (shared by both backends) -----------

    def read_json(self, key: str) -> Any:
        return json.loads(self.read(key).decode("utf-8"))

    def write_json(self, key: str, obj: Any) -> None:
        data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
        self.write(key, data)

    def read_text(self, key: str) -> str:
        return self.read(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write(key, text.encode("utf-8"))


class LocalFS(Storage):
    """Filesystem backend rooted at ``root``."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write to a unique temp file, then rename into place.
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out = [
            str(p.relative_to(self.root)).replace(os.sep, "/")
            for p in base.rglob("*")
            if p.is_file()
        ]
        return sorted(out)


class S3Storage(Storage):
    """S3 backend. ``endpoint_url`` lets it target LocalStack for local dev."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
    ):
        import boto3  # imported lazily so dev/test never need it unless used

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("AWS_S3_ENDPOINT") or None,
        )
        # Create the bucket if missing (no-op on a real, pre-provisioned bucket).
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            self.client.create_bucket(Bucket=bucket)

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def read(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._full(key))
        return obj["Body"].read()

    def write(self, key: str, data: bytes) -> None:
        # S3 PutObject is atomic — no temp-file dance needed.
        self.client.put_object(Bucket=self.bucket, Key=self._full(key), Body=data)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list[str]:
        full = self._full(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"][strip:])
        return sorted(keys)


def get_storage(uri: str) -> Storage:
    """Build a backend from a URI.

    ``s3://bucket[/prefix]`` -> S3 (endpoint from AWS_S3_ENDPOINT if set).
    ``file://path`` or a bare path -> LocalFS.
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        return S3Storage(bucket, prefix)
    if uri.startswith("file://"):
        return LocalFS(uri[len("file://"):])
    return LocalFS(uri)
