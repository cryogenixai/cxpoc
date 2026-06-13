"""Phase 1.5 — table grid word-coverage (no labels, no GPU beyond CPU TATR).

A self-supervised structure-recall signal for TATR: of the text-layer words that
fall inside a detected table region, what fraction land inside TATR's cell grid?
Dropped words mean the grid missed rows/columns — a structure-recall gap we can
flag without any gold labels. (Cell *values* come from the text layer by
construction, so a value cross-check would be circular for our own pipeline;
that one is meaningful only against silver labels like Landing.ai — deferred.)

Operates on the shortlist's table pages, reusing the mining store's Stage 0/1
artifacts (layout regions + table crops + words).

Usage:
    python -m eval.crosscheck [--shortlist eval/out/shortlist.csv] \
        [--store eval/out/mining_store] [--out eval/out/table_coverage.csv] \
        [--max-tables N] [--flag-below 0.8]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.jobctx import JobContext
from pipeline.storage import LocalFS
from pipeline.tables import TATRRecognizer


def word_coverage(cell_boxes: list[list[float]], words: list[dict],
                  region_bbox: dict[str, float]) -> tuple[int, int]:
    """(captured, total): words whose center is in the table region, and how many
    of those fall inside some grid cell. cell_boxes are crop coords; words are
    page coords, so shift by the region origin to compare."""
    ox, oy = region_bbox["x0"], region_bbox["y0"]
    rx0, ry0, rx1, ry1 = region_bbox["x0"], region_bbox["y0"], region_bbox["x1"], region_bbox["y1"]
    captured = total = 0
    for w in words:
        b = w["bbox"]
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1):
            continue
        total += 1
        ccx, ccy = cx - ox, cy - oy
        if any(c[0] <= ccx <= c[2] and c[1] <= ccy <= c[3] for c in cell_boxes):
            captured += 1
    return captured, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crosscheck")
    parser.add_argument("--shortlist", default="eval/out/shortlist.csv")
    parser.add_argument("--store", default="eval/out/mining_store")
    parser.add_argument("--out", default="eval/out/table_coverage.csv")
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--flag-below", type=float, default=0.8)
    args = parser.parse_args(argv)

    store = LocalFS(args.store)
    tatr = TATRRecognizer()
    shortlist = [r for r in csv.DictReader(open(args.shortlist)) if int(r["n_table_regions"]) > 0]

    rows = []
    n_tables = 0
    for entry in shortlist:
        if args.max_tables and n_tables >= args.max_tables:
            break
        doc_id, page = entry["doc_id"], int(entry["page"])
        ctx = JobContext(job_id=doc_id, storage=store)
        regions = ctx.read_json("layout", f"p{page:04d}.regions.json")
        meta = next(p for p in ctx.read_json("pages", "pages.json") if p["page_index"] == page)
        words = ctx.storage.read_json(ctx.key(meta["words_key"]))

        for r in regions:
            if r["type"] != "table" or not r.get("crop_key"):
                continue
            crop = ctx.storage.read(ctx.key(r["crop_key"]))
            cells = tatr.cell_boxes(crop)
            captured, total = word_coverage(cells, words, r["bbox"])
            cov = captured / total if total else None
            rows.append({
                "gold_id": entry["gold_id"], "region_id": r["region_id"],
                "stratum": entry["stratum"], "n_cells": len(cells),
                "words_in_region": total, "words_captured": captured,
                "coverage": round(cov, 3) if cov is not None else "",
            })
            n_tables += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["gold_id", "region_id", "stratum", "n_cells",
                            "words_in_region", "words_captured", "coverage"])
        w.writeheader()
        w.writerows(rows)

    covs = [r["coverage"] for r in rows if r["coverage"] != ""]
    flagged = [r for r in rows if r["coverage"] != "" and r["coverage"] < args.flag_below]
    print(f"tables checked: {len(rows)}  ->  {out_path}")
    if covs:
        print(f"mean word-coverage: {sum(covs)/len(covs):.3f}  "
              f"(min {min(covs):.3f}, max {max(covs):.3f})")
        print(f"flagged < {args.flag_below}: {len(flagged)} tables (likely missed rows/cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
