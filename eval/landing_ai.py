"""Landing.ai ADE-parse client — fetch silver reference labels for the gold set.

Posts a PDF (typically the assembled gold_pages_{version}.pdf) to the ADE parse
endpoint with split=page, and saves the raw JSON response. The API key is read
from the LANDINGAI_API_KEY env var — never hardcoded or committed.

API ref: https://docs.landing.ai/api-reference/tools/ade-parse

Usage:
    LANDINGAI_API_KEY=... python -m eval.landing_ai \
        --pdf gold/v1/gold_pages_v1.pdf --out eval/out/landing_ai/v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

US_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"
EU_ENDPOINT = "https://api.va.eu-west-1.landing.ai/v1/ade/parse"


def parse_pdf(pdf: Path, api_key: str, model: str = "dpt-2-latest",
              split: str = "page", endpoint: str = US_ENDPOINT,
              timeout: float = 600.0) -> dict:
    import httpx

    with pdf.open("rb") as f:
        files = {"document": (pdf.name, f, "application/pdf")}
        data = {"model": model, "split": split}
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = httpx.post(endpoint, headers=headers, files=files, data=data, timeout=timeout)

    if resp.status_code not in (200, 206):
        raise SystemExit(f"Landing.ai error {resp.status_code}: {resp.text[:500]}")
    if resp.status_code == 206:
        print("WARNING: 206 partial success — some pages failed (see metadata.failed_pages)")
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.landing_ai")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="dpt-2-latest")
    parser.add_argument("--split", default="page")
    parser.add_argument("--eu", action="store_true", help="use the EU endpoint")
    args = parser.parse_args(argv)

    key = os.environ.get("LANDINGAI_API_KEY")
    if not key:
        print("LANDINGAI_API_KEY not set", file=sys.stderr)
        return 2

    result = parse_pdf(
        Path(args.pdf), key, model=args.model, split=args.split,
        endpoint=EU_ENDPOINT if args.eu else US_ENDPOINT,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    meta = result.get("metadata", {})
    print(f"saved -> {out}")
    print(f"  pages: {meta.get('page_count')}  chunks: {len(result.get('chunks', []))}  "
          f"credits: {meta.get('credit_usage')}  failed_pages: {meta.get('failed_pages')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
