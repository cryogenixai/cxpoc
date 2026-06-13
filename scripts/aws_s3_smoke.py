"""aws_s3_smoke — validate the real S3 backend via our own storage abstraction.

Exercises pipeline.storage.S3Storage against real AWS S3 (create bucket if
needed, write/read/list bytes + json, then clean up the smoke keys). This proves
the exact code path the pipeline uses on the VM, not just LocalStack.

Usage:
    python scripts/aws_s3_smoke.py [--bucket NAME] [--keep]

Default bucket is cxpoc-jobs-<account-id> (globally unique). Requires AWS
credentials configured (e.g. via `aws configure` as cxpoc-dev).
"""

from __future__ import annotations

import argparse

import boto3

from pipeline.storage import S3Storage

PREFIX = "jobs/_smoke"


def default_bucket() -> str:
    account = boto3.client("sts").get_caller_identity()["Account"]
    return f"cxpoc-jobs-{account}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aws_s3_smoke")
    parser.add_argument("--bucket", default=None, help="S3 bucket (default: cxpoc-jobs-<account>)")
    parser.add_argument("--keep", action="store_true", help="do not delete smoke keys")
    args = parser.parse_args(argv)

    bucket = args.bucket or default_bucket()
    s = S3Storage(bucket)  # creates the bucket if missing
    print(f"bucket ready : {bucket}")

    s.write(f"{PREFIX}/hello.txt", b"hi from cxpoc")
    s.write_json(f"{PREFIX}/meta.json", {"ok": True, "n": 3})

    print(f"read bytes   : {s.read(f'{PREFIX}/hello.txt')!r}")
    print(f"read json    : {s.read_json(f'{PREFIX}/meta.json')}")
    print(f"exists       : {s.exists(f'{PREFIX}/hello.txt')}")
    print(f"list         : {s.list(PREFIX)}")

    ok = (
        s.read(f"{PREFIX}/hello.txt") == b"hi from cxpoc"
        and s.read_json(f"{PREFIX}/meta.json") == {"ok": True, "n": 3}
        and s.exists(f"{PREFIX}/hello.txt")
    )

    if not args.keep:
        for key in (f"{PREFIX}/hello.txt", f"{PREFIX}/meta.json"):
            s.client.delete_object(Bucket=bucket, Key=key)
        print(f"cleaned up   : {s.list(PREFIX)}")

    print("RESULT       :", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
