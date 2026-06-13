"""Classification refinement — coarse layout class -> fine IR taxonomy (design §3, §4.1).

Pure, model-free helpers that upgrade the detector's coarse classes using *exact*
born-digital text-layer signals (font size, leading glyph, position, punctuation):

  * ``body_font_size`` / ``heading_level`` — section_header/title -> heading.level
  * ``list_ordered``                       — list -> ordered vs bulleted
  * ``is_page_number``                     — page_header/footer -> page_number
  * ``is_continuation``                    — paragraph/list -> continued flag (Stage 3)

These are deliberately the cheap, deterministic counterpart to a fine-grained
vision detector: for born-digital PDFs these signals are ground truth, so we are
strongest on exactly the classes a pure-vision model is weakest on. Everything
here is a pure function of its inputs — no I/O, no models — so it unit-tests in
isolation and slots into Stage 1 (type/attrs) and Stage 3 (continuation).
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import median

# --- Heading level (font-size relative to page body) -------------------------

# A region's representative size, divided by the page body size, must clear these
# ratios to count as H1/H2/H3. Tuned conservatively: clearly-larger text only.
_H1_RATIO = 1.45
_H2_RATIO = 1.20
_H3_RATIO = 1.07
# Bold same-size headings (common in 10-Ks: bold body-size section headers) count
# as H3. Tolerance below 1.0 absorbs float noise — real font sizes come back as
# e.g. 9.9975 vs a body rounded to 10.0, which an exact >= 1.0 test would miss.
_BOLD_RATIO = 0.97


def body_font_size(sizes: list[float]) -> float:
    """The page's body-text font size = the most common rounded size on the page.

    Modal rather than mean/median because body text dominates by word count while
    headings are rare; the mode is robust to a few large heading words. Falls back
    to the median, then 0.0, on sparse input.
    """
    rounded = [round(s, 1) for s in sizes if s and s > 0]
    if not rounded:
        return 0.0
    counts = Counter(rounded)
    top = max(counts.values())
    # Among sizes tied for most-frequent, take the smallest (body < incidental large).
    return min(s for s, c in counts.items() if c == top)


def heading_level(region_size: float, body_size: float, *, bold: bool = False) -> int | None:
    """Map a region's representative font size to a heading level, or None if body text.

    ``bold`` nudges a same-size-as-body run up to H3 (bold inline headings are common
    in manuals). Returns 1, 2, 3, or None.
    """
    if body_size <= 0 or region_size <= 0:
        return None
    ratio = region_size / body_size
    if ratio >= _H1_RATIO:
        return 1
    if ratio >= _H2_RATIO:
        return 2
    if ratio >= _H3_RATIO:
        return 3
    if bold and ratio >= _BOLD_RATIO:
        return 3
    return None


def representative_size(sizes: list[float]) -> float:
    """A region's representative font size = median of its word/span sizes."""
    vals = [s for s in sizes if s and s > 0]
    return median(vals) if vals else 0.0


# --- List subtype (ordered vs bulleted) --------------------------------------

_BULLET_GLYPHS = "•◦‣·▪–-*"  # leading marker glyphs for unordered lists
# Ordered markers: "1." "1)" "(1)" "a." "a)" "iv." "IV)" etc.
_ORDERED_RE = re.compile(r"^\(?\s*([0-9]{1,3}|[ivxlcdm]{1,5}|[a-z])\s*[.)]", re.IGNORECASE)


def list_ordered(text: str) -> bool | None:
    """For a region already classed ``list``: True if numbered, False if bulleted, None if neither.

    Looks only at the leading marker of the region text.
    """
    s = text.lstrip()
    if not s:
        return None
    if s[0] in _BULLET_GLYPHS:
        return False
    if _ORDERED_RE.match(s):
        return True
    return None


# --- Page number (margin band + short numeric/label pattern) -----------------

_PAGE_BAND = 0.08  # top/bottom 8% of page height
_PAGE_NUM_RE = re.compile(
    r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$"      # "12", "Page 12", "12 of 340", "12/340"
    r"|^[-–—]\s*\d{1,4}\s*[-–—]$"                            # "- 12 -"
    r"|^[ivxlcdm]{1,6}$",                                    # roman "iv"
    re.IGNORECASE,
)


def is_page_number(text: str, region_bbox: dict[str, float], page_height: float) -> bool:
    """True if a (header/footer) region is really a page number.

    Requires both a top/bottom margin-band position and a short numeric/label pattern,
    so a one-line footnote or running header is not misclassified. ``region_bbox`` and
    ``page_height`` are in the same (pixel) units.
    """
    s = text.strip()
    if not s or len(s) > 16:
        return False
    if page_height <= 0:
        return False
    cy = (region_bbox["y0"] + region_bbox["y1"]) / 2 / page_height
    in_band = cy <= _PAGE_BAND or cy >= (1 - _PAGE_BAND)
    return in_band and bool(_PAGE_NUM_RE.match(s))


# --- Continuation (cross-column / cross-page stitch, Stage 3) -----------------

_TERMINAL = ".!?:;\"')]}"          # sentence/clause terminators
_CONTINUABLE = {"paragraph", "list"}


def is_continuation(prev_type: str, prev_text: str, cur_type: str, cur_text: str) -> bool:
    """True if ``cur`` continues ``prev`` across a column/page break.

    Heuristic (born-digital): same continuable type, the previous block does NOT end
    on a terminator, and the current block starts lowercase (paragraph) or carries on a
    list. Conservative by design — a false negative just emits two adjacent blocks; a
    false positive wrongly welds them, so we require both signals.
    """
    if prev_type != cur_type or cur_type not in _CONTINUABLE:
        return False
    prev = prev_text.rstrip()
    cur = cur_text.lstrip()
    if not prev or not cur:
        return False
    prev_open = prev[-1] not in _TERMINAL
    if cur_type == "paragraph":
        return prev_open and cur[:1].islower()
    # list: a continued list fragment does not start with a fresh marker
    return prev_open and list_ordered(cur) is None and cur[:1] not in _BULLET_GLYPHS


# --- Region refinement (Stage 1: coarse class -> fine type + attributes) -----

# Boilerplate/navigation types the downstream chunker excludes by default.
BOILERPLATE_TYPES = {"page_header", "page_footer", "page_number", "toc"}


def clip_words(words: list[dict], bbox: dict[str, float]) -> list[dict]:
    """Words whose centre falls inside ``bbox`` (same pixel coords as the region)."""
    out = []
    for w in words:
        wb = w["bbox"]
        cx, cy = (wb["x0"] + wb["x1"]) / 2, (wb["y0"] + wb["y1"]) / 2
        if bbox["x0"] <= cx <= bbox["x1"] and bbox["y0"] <= cy <= bbox["y1"]:
            out.append(w)
    return out


def refine_region(
    rtype: str, words_in: list[dict], region_bbox: dict[str, float],
    body_size: float, page_height: float,
) -> tuple[str, dict]:
    """Upgrade a coarse detector class to the fine IR type + attributes (§4.1).

    Returns ``(type, attributes)``. Deterministic and model-free: heading level
    from font size, list ordering from the leading marker, page-number split by
    position+pattern. Unknown/already-fine types pass through with empty attrs.
    """
    text = " ".join(w.get("text", "") for w in words_in)
    rep = representative_size([w.get("size", 0.0) for w in words_in])
    bold = any(w.get("bold") for w in words_in)
    attrs: dict = {}

    if rtype == "section_header":
        # Detector already says this is a header; font sets the level (default 3).
        attrs["level"] = heading_level(rep, body_size, bold=bold) or 3
        return "heading", attrs
    if rtype == "title":
        # DocLayout-YOLO (DocStructBench) has no section_header class — it labels
        # section headings as `title`. Attach a font-derived level so hierarchy is
        # captured; keep the `title` type (cover-page doc-title detection is a
        # separate, deferred concern).
        lvl = heading_level(rep, body_size, bold=bold)
        if lvl:
            attrs["level"] = lvl
        return rtype, attrs
    if rtype == "list":
        ordered = list_ordered(text)
        if ordered is not None:
            attrs["ordered"] = ordered
        return rtype, attrs
    if rtype in ("page_header", "page_footer"):
        if is_page_number(text, region_bbox, page_height):
            return "page_number", attrs
    return rtype, attrs
