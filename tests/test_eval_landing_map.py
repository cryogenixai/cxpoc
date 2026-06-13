"""Mapper test against the frozen v1 silver (committed, deterministic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.landing_map import load_and_map, map_silver

_SILVER = Path("gold/v1/silver_landing_ai.json")
_MANIFEST = Path("gold/v1/gold_manifest.json")


def test_map_silver_synthetic():
    manifest = {"pages": [{"seq": 0, "gold_id": "doc__p0001"}]}
    silver = {"chunks": [
        {"type": "table", "markdown": "<a id='x'></a>\ncap\n<table><tr><td>1</td></tr></table>",
         "grounding": {"box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8}, "page": 0}},
        {"type": "text", "markdown": "<a id='y'></a>Hello",
         "grounding": {"box": {"left": 0, "top": 0, "right": 1, "bottom": 0.1}, "page": 0}},
    ]}
    m = map_silver(silver, manifest)
    chunks = m["doc__p0001"]["chunks"]
    assert len(chunks) == 2
    tbl = [c for c in chunks if c["coarse"] == "table"][0]
    assert tbl["html"] == "<table><tr><td>1</td></tr></table>"
    assert tbl["bbox"] == {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.8}
    txt = [c for c in chunks if c["coarse"] == "text"][0]
    assert txt["text"] == "Hello"  # anchor stripped


@pytest.mark.skipif(not _SILVER.exists(), reason="frozen v1 silver not present")
def test_map_frozen_v1_silver():
    m = load_and_map(_SILVER, _MANIFEST)
    assert len(m) == 71
    for v in m.values():
        for c in v["chunks"]:
            for k in ("x0", "y0", "x1", "y1"):
                assert 0.0 <= c["bbox"][k] <= 1.0
    # at least some real tables with HTML
    assert sum(1 for v in m.values() for c in v["chunks"] if c["coarse"] == "table" and c["html"]) > 10
