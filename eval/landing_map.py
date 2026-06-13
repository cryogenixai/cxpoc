"""Map a Landing.ai ADE-parse response into our IR, keyed by gold_id.

The assembled PDF's page index (chunk.grounding.page) equals the manifest seq,
so we join each silver chunk back to its gold_id. Boxes are already normalized
0-1 (left/top/right/bottom -> x0/y0/x1/y1). Table chunks carry HTML in markdown;
text chunks carry markdown with anchor tags we strip.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eval.taxonomy import coarse_landing

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_ANCHOR_RE = re.compile(r"<a\s+id=['\"][^'\"]*['\"]\s*>\s*</a>", re.IGNORECASE)


def _box_to_bbox(box: dict) -> dict[str, float]:
    return {"x0": box["left"], "y0": box["top"], "x1": box["right"], "y1": box["bottom"]}


def _table_html(markdown: str) -> str:
    m = _TABLE_RE.search(markdown or "")
    return m.group(0) if m else ""


def _clean_text(markdown: str) -> str:
    return _ANCHOR_RE.sub("", markdown or "").strip()


def map_silver(silver: dict, manifest: dict) -> dict[str, dict[str, Any]]:
    """Return {gold_id: {"seq", "chunks": [{type, coarse, bbox, text, html}]}}."""
    seq_to_gold = {p["seq"]: p["gold_id"] for p in manifest["pages"]}
    out: dict[str, dict[str, Any]] = {
        p["gold_id"]: {"seq": p["seq"], "chunks": []} for p in manifest["pages"]
    }
    for c in silver.get("chunks", []):
        g = c.get("grounding") or {}
        page = g.get("page")
        gold_id = seq_to_gold.get(page)
        if gold_id is None or "box" not in g:
            continue
        ctype = c.get("type", "other")
        out[gold_id]["chunks"].append({
            "type": ctype,
            "coarse": coarse_landing(ctype),
            "bbox": _box_to_bbox(g["box"]),
            "text": _clean_text(c.get("markdown", "")),
            "html": _table_html(c.get("markdown", "")) if ctype == "table" else "",
        })
    return out


def load_and_map(silver_path: str | Path, manifest_path: str | Path) -> dict[str, dict]:
    silver = json.loads(Path(silver_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    return map_silver(silver, manifest)
