"""Candidate viewer — browse the stratified shortlist before curating the gold set.

Left nav: each shortlist document with its candidate pages indented beneath it.
Click a page → the right pane shows that page's 200-DPI render (already in the
mining store, so no PDF.js / heavy PDF loading). Read-only, localhost.

Run:  python -m eval.viewer [--shortlist eval/out/shortlist.csv]
                            [--store eval/out/mining_store] [--port 8090]
"""

from __future__ import annotations

import argparse
import json
import csv
import os
import sys
import uuid
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_STATIC = Path(__file__).resolve().parent / "static"


class Decision(BaseModel):
    gold_id: str
    decision: str  # "in" | "out" | "none"


def _load_selections(path: Path) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_selections(path: Path, sel: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(sel, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _load_candidates(shortlist: Path):
    by_doc: dict[str, dict] = {}
    for r in csv.DictReader(shortlist.open()):
        doc = by_doc.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "source": r["source"], "pages": []})
        doc["pages"].append({
            "page": int(r["page"]),            # 0-based page_index
            "label": f"p{int(r['page']) + 1}",  # 1-based for display
            "stratum": r["stratum"],
            "gold_id": r["gold_id"],
        })
    out = []
    for doc in by_doc.values():
        doc["pages"].sort(key=lambda p: p["page"])
        out.append(doc)
    out.sort(key=lambda d: (d["source"], d["doc_id"]))
    return out


def create_app(shortlist="eval/out/shortlist.csv", store_root="eval/out/mining_store",
               selections="eval/out/curation.json") -> FastAPI:
    shortlist_path = Path(shortlist)
    store_root = Path(store_root)
    selections_path = Path(selections)
    app = FastAPI(title="cxpoc candidate viewer", version="0.1.0")

    @app.get("/api/candidates")
    def candidates():
        return JSONResponse(_load_candidates(shortlist_path))

    @app.get("/api/page/{doc_id}/{page}.png")
    def page_png(doc_id: str, page: int):
        png = store_root / "jobs" / doc_id / "pages" / f"p{page:04d}.png"
        if not png.exists():
            raise HTTPException(status_code=404, detail=f"no render for {doc_id} p{page}")
        return FileResponse(png, media_type="image/png")

    @app.get("/api/selections")
    def get_selections():
        return JSONResponse(_load_selections(selections_path))

    @app.post("/api/selections")
    def set_selection(d: Decision):
        sel = _load_selections(selections_path)
        if d.decision == "none":
            sel.pop(d.gold_id, None)
        elif d.decision in ("in", "out"):
            sel[d.gold_id] = d.decision
        else:
            raise HTTPException(status_code=400, detail="decision must be in|out|none")
        _save_selections(selections_path, sel)
        counts = {"in": sum(v == "in" for v in sel.values()),
                  "out": sum(v == "out" for v in sel.values())}
        return {"ok": True, "counts": counts}

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.viewer")
    parser.add_argument("--shortlist", default="eval/out/shortlist.csv")
    parser.add_argument("--store", default="eval/out/mining_store")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"candidate viewer → http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.shortlist, args.store), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
