"""Per-page feature extraction for candidate mining (design §4.1).

Pure functions over Stage 0/1 artifacts (pages.json + layout regions + words),
kept separate from the I/O in mine_candidates.py so they're unit-testable.
"""

from __future__ import annotations

from typing import Any

FIGURE_TYPES = {"figure", "chart", "diagram", "logo", "photo"}
TEXT_TYPES = {
    "paragraph", "title", "section_header", "heading", "list", "caption",
    "footnote", "toc", "page_header", "page_footer", "page_number", "other",
}


def _area(b: dict[str, float]) -> float:
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def looks_like_toc(words: list[dict], page_w: int) -> bool:
    """Heuristic TOC detector from the text layer.

    A TOC entry is 'text ........ <page-number>' — the number sits at the far
    right after a big horizontal gap (the leader region). That right-gap test is
    what separates a TOC from a financial table, whose numbers sit in aligned
    columns with small gaps (and usually carry commas, so str.isdigit() is False).
    """
    from collections import defaultdict

    lines: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        b = w.get("bbox")
        if not b:
            continue
        lines[round((b["y0"] + b["y1"]) / 2 / 12)].append(w)

    page_ref_lines = dot_leaders = 0
    heading = False
    for lws in lines.values():
        lws.sort(key=lambda w: w["bbox"]["x0"])
        toks = [w["text"] for w in lws]
        text = " ".join(toks).lower()
        if "table of contents" in text or text.strip() in ("contents", "index"):
            heading = True
        for t in toks:
            if len(t) >= 2 and set(t) <= {"."}:
                dot_leaders += 1
        # right-aligned page-number line: ends in a short integer, preceded by a
        # big gap from descriptive (non-numeric) text.
        if len(toks) >= 2 and toks[-1].isdigit() and len(toks[-1]) <= 4:
            gap = lws[-1]["bbox"]["x0"] - lws[-2]["bbox"]["x1"]
            if gap > 0.12 * page_w and any(not t.isdigit() for t in toks[:-1]):
                page_ref_lines += 1

    return page_ref_lines >= 6 or dot_leaders >= 4 or (heading and page_ref_lines >= 2)


def estimate_columns(regions: list[dict], page_w: int) -> int:
    """1 or 2 columns from text-region geometry: a 2-column page has narrow
    body regions sitting on both the left and right halves."""
    mid = page_w / 2
    narrow_left = narrow_right = 0
    for r in regions:
        if r["type"] not in ("paragraph", "list"):
            continue
        b = r["bbox"]
        width = b["x1"] - b["x0"]
        if width > 0.55 * page_w:
            return 1  # a full-width body block => single column
        cx = (b["x0"] + b["x1"]) / 2
        if cx < mid:
            narrow_left += 1
        else:
            narrow_right += 1
    return 2 if (narrow_left and narrow_right) else 1


def guess_page_type(source: str, n_table: int, has_figure: bool,
                    has_toc: bool, word_count: int, n_regions: int,
                    text_density: float) -> str:
    """Coarse page-type guess feeding the stratification matrix (§3)."""
    if word_count < 30 and n_regions < 3:
        return "near_empty"
    if has_toc:
        return "cover_toc"
    if n_table >= 1:
        # financial corpus => statement-style tables; technical => spec tables.
        return "financial_table" if source == "10k-pdf" else "spec_table"
    if has_figure:
        return "diagram_figure"
    if text_density > 0.55 or n_regions >= 14:
        return "dense_narrative"
    return "narrative"


def page_features(doc_id: str, source: str, page_meta: dict,
                  regions: list[dict], words: list[dict]) -> dict[str, Any]:
    W, H = page_meta["width_px"], page_meta["height_px"]
    area = W * H or 1
    types = [r["type"] for r in regions]
    n_table = sum(t == "table" for t in types)
    has_figure = any(t in FIGURE_TYPES for t in types)
    has_toc = any(t == "toc" for t in types)
    # TOC: trust a detected toc region, else the text-layer heuristic — but not
    # on a table-dominated page (guards against a financial page misfiring).
    if not has_toc and n_table < 2 and looks_like_toc(words, W):
        has_toc = True
    confs = [r.get("detector_confidence", 1.0) for r in regions]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    text_area = sum(_area(r["bbox"]) for r in regions if r["type"] in TEXT_TYPES)
    text_density = text_area / area
    n_columns = estimate_columns(regions, W)
    word_count = len(words)
    landscape = W > H

    page_type = guess_page_type(
        source, n_table, has_figure, has_toc, word_count, len(regions), text_density
    )
    # Stratify-relevant edge flags. Only reclassify pure-text pages as
    # very_dense — never override a table/figure signal, which is more useful
    # for stratification (density is already its own column).
    if landscape and n_table:
        page_type = "landscape_table"
    elif page_type in ("narrative", "dense_narrative") and (text_density > 0.7 or len(regions) >= 20):
        page_type = "very_dense"

    # Coarse borderless hint: financial statement tables are typically borderless
    # (TATR worst case). A real ruling-line analysis can replace this later.
    is_borderless_hint = source == "10k-pdf" and n_table >= 1

    return {
        "doc_id": doc_id,
        "source": source,
        "page": page_meta["page_index"],
        "page_type_guess": page_type,
        "n_regions": len(regions),
        "n_table_regions": n_table,
        "has_figure": int(has_figure),
        "has_toc": int(has_toc),
        "n_columns": n_columns,
        "word_count": word_count,
        "text_density": round(text_density, 4),
        "mean_detector_conf": round(mean_conf, 4),
        "landscape": int(landscape),
        "is_borderless_hint": int(is_borderless_hint),
        "page_kind": page_meta.get("page_kind", "digital"),
    }
