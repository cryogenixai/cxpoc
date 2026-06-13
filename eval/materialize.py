"""Step 2 (materialize + assemble): freeze a versioned gold set (design §4.4-5).

Turns the selected candidate pages into self-contained, versioned units and one
assembled PDF for a single Landing.ai/Textract call:

    gold/{version}/
      gold_manifest.json        # frozen: version, per-page record + checksums
      gold_pages_{version}.pdf  # assembled; PDF page seq i <-> manifest pages[i]
      {gold_id}/ page.pdf  page.png  textlayer.json

Selection source: the curation.json "in" set if any pages are marked in, else the
full shortlist as-is (v1 default). gold_id is stable/order-independent (doc__pNNNN),
so labels attached later survive re-selection.

Usage:
    python -m eval.materialize [--version v1] [--shortlist eval/out/shortlist.csv]
        [--curation eval/out/curation.json] [--store eval/out/mining_store]
        [--out gold]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import fitz  # PyMuPDF

from pipeline.jobctx import JobContext
from pipeline.storage import LocalFS


def _selected(shortlist: Path, curation: Path, ignore_curation: bool = False) -> list[dict]:
    rows = list(csv.DictReader(shortlist.open()))
    if curation.exists() and not ignore_curation:
        sel = json.loads(curation.read_text())
        keep = {gid for gid, d in sel.items() if d == "in"}
        if keep:
            chosen = [r for r in rows if r["gold_id"] in keep]
            print(f"using curation.json: {len(chosen)} pages marked 'in'")
            return chosen
    print(f"no curation 'in' set — using full shortlist as-is ({len(rows)} pages)")
    return rows


def materialize(version: str, shortlist: Path, curation: Path,
                store_root: Path, out_base: Path, ignore_curation: bool = False) -> dict:
    rows = _selected(shortlist, curation, ignore_curation)
    out = out_base / version
    out.mkdir(parents=True, exist_ok=True)
    store = LocalFS(store_root)

    src_cache: dict[str, fitz.Document] = {}

    def source_doc(doc_id: str) -> fitz.Document:
        if doc_id not in src_cache:
            ctx = JobContext(job_id=doc_id, storage=store)
            src_cache[doc_id] = fitz.open(stream=ctx.read_bytes("source.pdf"), filetype="pdf")
        return src_cache[doc_id]

    assembled = fitz.open()
    records = []
    for seq, r in enumerate(rows):
        doc_id, page = r["doc_id"], int(r["page"])
        gold_id = r["gold_id"]
        unit = out / gold_id
        unit.mkdir(exist_ok=True)

        # single-page PDF
        one = fitz.open()
        one.insert_pdf(source_doc(doc_id), from_page=page, to_page=page)
        page_pdf = one.tobytes()
        (unit / "page.pdf").write_bytes(page_pdf)
        one.close()

        # render + text layer copied from the mining store (already 200 DPI)
        ctx = JobContext(job_id=doc_id, storage=store)
        (unit / "page.png").write_bytes(ctx.storage.read(ctx.key("pages", f"p{page:04d}.png")))
        (unit / "textlayer.json").write_text(
            json.dumps(ctx.storage.read_json(ctx.key("pages", f"p{page:04d}.words.json")), indent=2)
        )

        # append to the assembled PDF (seq = position there)
        assembled.insert_pdf(source_doc(doc_id), from_page=page, to_page=page)

        records.append({
            "seq": seq,
            "gold_id": gold_id,
            "doc_id": doc_id,
            "source": r["source"],
            "source_page": page,             # 0-based page_index in the original doc
            "stratum": r["stratum"],
            "checksum": hashlib.sha256(page_pdf).hexdigest(),
        })

    assembled_name = f"gold_pages_{version}.pdf"
    assembled.save(out / assembled_name)
    assembled.close()
    for d in src_cache.values():
        d.close()

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_pages": len(records),
        "assembled_pdf": assembled_name,
        "strata": {s: sum(r["stratum"] == s for r in records) for s in sorted({r["stratum"] for r in records})},
        "pages": records,
    }
    (out / "gold_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.materialize")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--shortlist", default="eval/out/shortlist.csv")
    parser.add_argument("--curation", default="eval/out/curation.json")
    parser.add_argument("--store", default="eval/out/mining_store")
    parser.add_argument("--out", default="gold")
    parser.add_argument("--ignore-curation", action="store_true",
                        help="materialize the full shortlist, ignoring curation.json")
    args = parser.parse_args(argv)

    m = materialize(args.version, Path(args.shortlist), Path(args.curation),
                    Path(args.store), Path(args.out), args.ignore_curation)
    print(f"\ngold set {m['version']}: {m['n_pages']} pages -> gold/{m['version']}/")
    print(f"  assembled: gold/{m['version']}/{m['assembled_pdf']}")
    print(f"  strata: {m['strata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
