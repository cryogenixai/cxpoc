"""Run the eval harness and print + save the v1 scorecard.

Usage:
    python -m eval.run_eval [--version v1] [--pred eval/out/pred_v1]
        [--out eval/out/scorecard_v1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval.harness import evaluate


def _fmt(v):
    return "  —  " if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_eval")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--pred", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    pred_root = Path(args.pred or f"eval/out/pred_{args.version}")
    sc = evaluate(args.version, pred_root)

    out = Path(args.out or f"eval/out/scorecard_{args.version}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sc, indent=2))

    print(f"\n=== cxpoc {sc['version']} scorecard  (vs {sc['reference']}, {sc['n_pages']} pages) ===\n")

    cc = sc["content_coverage"]
    print(f"LOCALIZATION — content coverage (class-agnostic: did we find the content area)")
    print(f"  P {cc['precision']:.3f}  R {cc['recall']:.3f}  F1 {cc['f1']:.3f}\n")

    print("LAYOUT — area coverage (PRIMARY, granularity-agnostic; precision / recall / f1)")
    for cls, p in sc["layout_coverage"]["per_class"].items():
        print(f"  {cls:11} P {p['precision']:.3f}  R {p['recall']:.3f}  F1 {p['f1']:.3f}")
    m = sc["layout_coverage"]["micro"]
    print(f"  {'MICRO':11} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}\n")

    bm = sc["layout_boxmatch"]["micro"]
    print(f"LAYOUT — box-match (secondary, segmentation agreement): "
          f"P {bm['precision']:.3f}  R {bm['recall']:.3f}  F1 {bm['f1']:.3f}\n")

    t = sc["table"]
    print(f"TABLE   mean TEDS {_fmt(t['mean_teds'])}  over {t['n_matched_pairs']} IoU-matched table pairs")
    tx = sc["text"]
    print(f"TEXT    word-token  P {tx['precision']:.3f}  R {tx['recall']:.3f}  F1 {tx['f1']:.3f}\n")

    print("BY STRATUM  (locF1 = class-agnostic localization; covF1 = per-class layout)")
    print(f"  {'stratum':18} {'n':>3}  {'locF1':>6}  {'covF1':>6}  {'tables':>6}  {'TEDS':>6}  {'text':>6}")
    for st, s in sc["by_stratum"].items():
        print(f"  {st:18} {s['n_pages']:>3}  {s['content_coverage_f1']:>6.3f}  "
              f"{s['layout_coverage_f1']:>6.3f}  {s['table_n']:>6}  "
              f"{_fmt(s['table_mean_teds']):>6}  {_fmt(s['text_f1']):>6}")

    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
