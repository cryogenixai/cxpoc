"""Step 1 (stratify): rank mined pages into the gold-set matrix, enforce the
per-doc cap, and emit a ranked shortlist for human pick (design §3, §4.2-3).

Stratify-don't-randomize: deliberately over-sample the hard tail (borderless
financial tables, low-confidence pages) so each failure mode has enough
examples. The per-doc cap forces layout variety across many documents.

Outputs an oversampled (~2x target) shortlist; a human fills the final ~60 from
it (§4.3). Also prints a coverage table (stratum -> selected vs target).

Usage:
    python -m eval.stratify [--features eval/out/page_features.csv] \
        [--out eval/out/shortlist.csv] [--cap 3] [--oversample 2]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Target pages per stratum (sums to ~60). Financial tables weighted heavily —
# TATR worst case (§6 "weight effort by corpus value").
TARGETS = {
    "financial_table": 14,
    "dense_narrative": 8,    # MD&A / footnotes-notes (dense prose + nested tables)
    "spec_table": 8,
    "diagram_figure": 8,
    "narrative": 4,
    "landscape_table": 4,
    "very_dense": 4,
    "cover_toc": 3,
    "near_empty": 2,
}


def hardness(stratum: str, row: dict) -> float:
    """Higher = more desirable to include. Low detector confidence is the
    generic 'this is hard / informative' signal; strata add specific weights."""
    conf = float(row["mean_detector_conf"])
    n_table = int(row["n_table_regions"])
    score = (1.0 - conf)  # 0..1, harder pages first
    if stratum in ("financial_table", "spec_table", "landscape_table"):
        score += 0.5 * n_table
        score += 1.0 * int(row["is_borderless_hint"])
        score += 0.3 * (int(row["n_columns"]) - 1)
    elif stratum == "diagram_figure":
        score += 1.0 * int(row["has_figure"])
    elif stratum in ("dense_narrative", "very_dense"):
        score += float(row["text_density"])
    elif stratum == "near_empty":
        score += 1.0 - float(row["text_density"])  # prefer the emptiest
    return score


def stratify(rows: list[dict], cap: int, oversample: int):
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[r["page_type_guess"]].append(r)

    per_doc = defaultdict(int)        # global per-doc count across all strata
    coverage: dict[str, int] = defaultdict(int)
    chosen_ids: set[str] = set()
    selected: list[dict] = []

    # Pre-rank candidates per stratum by hardness once.
    ranked = {
        s: sorted(rows_s, key=lambda r: hardness(s, r), reverse=True)
        for s, rows_s in by_stratum.items()
    }

    def fill(stratum: str, limit: int):
        for r in ranked.get(stratum, []):
            if coverage[stratum] >= limit:
                break
            if r["gold_id"] in chosen_ids or per_doc[r["doc_id"]] >= cap:
                continue
            per_doc[r["doc_id"]] += 1
            chosen_ids.add(r["gold_id"])
            coverage[stratum] += 1
            selected.append({
                **r, "stratum": stratum, "score": round(hardness(stratum, r), 3),
                "selection_reason": f"{stratum}: conf={r['mean_detector_conf']}, "
                                    f"tables={r['n_table_regions']}, fig={r['has_figure']}, "
                                    f"borderless={r['is_borderless_hint']}",
            })

    # Pass 1 — guarantee each stratum its base target (every failure mode gets
    # examples before any stratum oversamples). Pass 2 — oversample toward
    # oversample x target with whatever per-doc budget remains, value-first.
    priority = sorted(TARGETS, key=lambda s: TARGETS[s], reverse=True)
    for stratum in priority:
        fill(stratum, TARGETS[stratum])
    for stratum in priority:
        fill(stratum, TARGETS[stratum] * oversample)
    return selected, dict(coverage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stratify")
    parser.add_argument("--features", default="eval/out/page_features.csv")
    parser.add_argument("--out", default="eval/out/shortlist.csv")
    parser.add_argument("--cap", type=int, default=3, help="max pages per source doc")
    parser.add_argument("--oversample", type=int, default=2, help="shortlist = oversample x target")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.features)))
    selected, coverage = stratify(rows, args.cap, args.oversample)

    cols = ["gold_id", "doc_id", "source", "page", "stratum", "score",
            "selection_reason", "page_type_guess", "n_table_regions", "has_figure",
            "n_columns", "text_density", "mean_detector_conf", "is_borderless_hint",
            "landscape", "page_kind"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)

    print(f"shortlist: {len(selected)} pages from {len({r['doc_id'] for r in selected})} docs "
          f"(cap={args.cap}, oversample={args.oversample}x) -> {out_path}\n")
    print(f"{'stratum':18} {'target':>7} {'short':>6}")
    for stratum in sorted(TARGETS, key=lambda s: -TARGETS[s]):
        print(f"{stratum:18} {TARGETS[stratum]:>7} {coverage.get(stratum, 0):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
