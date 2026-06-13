"""Step 1 (mining): run Stage 0/1 over the corpus and emit a per-page feature
table for stratified gold-set selection (design §4.1, §7.1).

Runs only ingest + layout (not extract/assemble) — features come from page
geometry + detected regions, so we skip the expensive TATR/VLM handlers. Per-doc
work is idempotent via the manifest, so reruns are cheap and the corpus can be
mined in chunks. One detector instance is shared across docs (no reload).

Usage:
    python -m eval.mine_candidates --inputs DIR_OR_PDF [DIR_OR_PDF ...] \
        [--store eval/out/mining_store] [--out eval/out/page_features.csv] \
        [--max-pages-per-doc N] [--limit-docs N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make pipeline importable when run directly (src/ layout), like review/app.py.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline import manifest as M
from pipeline.detect import make_detector
from pipeline.jobctx import JobContext
from pipeline.stages import IngestStage, LayoutStage
from pipeline.storage import LocalFS

from eval.features import page_features

FIELDNAMES = [
    "doc_id", "source", "page", "gold_id", "page_type_guess",
    "n_regions", "n_table_regions", "has_figure", "has_toc", "n_columns",
    "word_count", "text_density", "mean_detector_conf", "landscape",
    "is_borderless_hint", "page_kind",
]


def _sanitize(stem: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


def collect_pdfs(inputs: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdfs.append(p)
    return pdfs


def mine_doc(pdf: Path, store: LocalFS, detector, max_pages: int | None) -> list[dict]:
    doc_id = _sanitize(pdf.stem)
    source = pdf.parent.name  # e.g. "10k-pdf" / "dell-pdf"
    ctx = JobContext(job_id=doc_id, storage=store)

    if not M.exists(ctx):
        ctx.write_bytes(pdf.read_bytes(), "source.pdf")
        M.save(ctx, M.new_manifest(doc_id, source={"filename": pdf.name}))

    IngestStage().run(ctx)
    LayoutStage(detector=detector).run(ctx)

    pages = ctx.read_json("pages", "pages.json")
    rows = []
    for page_meta in pages:
        pi = page_meta["page_index"]
        if max_pages is not None and pi >= max_pages:
            break
        regions = ctx.read_json("layout", f"p{pi:04d}.regions.json")
        words = ctx.storage.read_json(ctx.key(page_meta["words_key"]))
        feat = page_features(doc_id, source, page_meta, regions, words)
        feat["gold_id"] = f"{doc_id}__p{pi:04d}"  # stable, order-independent (§4 cautions)
        rows.append(feat)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mine_candidates")
    parser.add_argument("--inputs", nargs="+", required=True, help="PDF files or directories")
    parser.add_argument("--store", default="eval/out/mining_store")
    parser.add_argument("--out", default="eval/out/page_features.csv")
    parser.add_argument("--max-pages-per-doc", type=int, default=None)
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args(argv)

    pdfs = collect_pdfs(args.inputs)
    if args.limit_docs:
        pdfs = pdfs[: args.limit_docs]
    if not pdfs:
        print("no PDFs found", file=sys.stderr)
        return 2

    store = LocalFS(args.store)
    detector = make_detector()  # one shared instance — load the model once
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, pdf in enumerate(pdfs, 1):
        rows = mine_doc(pdf, store, detector, args.max_pages_per_doc)
        all_rows.extend(rows)
        print(f"[{i}/{len(pdfs)}] {pdf.name}: {len(rows)} pages")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote {len(all_rows)} page rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
