"""Run our full pipeline on each gold page → per-gold_id document.json.

Each gold unit is a single-page PDF (gold/{version}/{gold_id}/page.pdf), so the
pipeline produces a 1-page document.json per gold_id, directly aligned to the
silver labels by gold_id. Idempotent (manifest-resumable); uses the real models
(DocLayout-YOLO + TATR). The figure/chart/image handlers stay on the mock VLM
until the GPU quota lands — chart/image eval is deferred anyway.

Usage:
    python -m eval.predict [--version v1] [--out eval/out/pred_v1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import manifest as M
from pipeline.jobctx import JobContext
from pipeline.run import run_pipeline
from pipeline.storage import LocalFS


def predict_all(version: str, out_root: Path) -> dict[str, dict]:
    gold_dir = Path("gold") / version
    manifest = json.loads((gold_dir / "gold_manifest.json").read_text())
    store = LocalFS(out_root)

    results: dict[str, dict] = {}
    for i, rec in enumerate(manifest["pages"], 1):
        gold_id = rec["gold_id"]
        page_pdf = gold_dir / gold_id / "page.pdf"
        ctx = JobContext(job_id=gold_id, storage=store)
        if not ctx.exists("output", "document.json"):
            if not M.exists(ctx):
                ctx.write_bytes(page_pdf.read_bytes(), "source.pdf")
                M.save(ctx, M.new_manifest(gold_id, source={"filename": f"{gold_id}.pdf"}))
            run_pipeline(ctx)
        results[gold_id] = ctx.read_json("output", "document.json")
        if i % 10 == 0 or i == len(manifest["pages"]):
            print(f"  [{i}/{len(manifest['pages'])}] predicted")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.predict")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", default=None, help="store root (default eval/out/pred_{version})")
    args = parser.parse_args(argv)

    out_root = Path(args.out or f"eval/out/pred_{args.version}")
    results = predict_all(args.version, out_root)
    print(f"predicted {len(results)} gold pages -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
