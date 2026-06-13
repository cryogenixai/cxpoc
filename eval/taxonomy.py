"""Common coarse taxonomy for cross-system layout comparison.

Our pipeline and Landing.ai use different (and finer/coarser) type sets, so a
per-class layout metric has to compare on a shared, coarse taxonomy. We collapse
both sides to: table | figure | logo | text | boilerplate | code | other.
"""

from __future__ import annotations

COARSE = ["table", "figure", "logo", "text", "boilerplate", "code", "other"]

_OURS = {
    "table": "table",
    "chart": "figure", "diagram": "figure", "photo": "figure", "figure": "figure",
    "logo": "logo",
    "title": "text", "section_header": "text", "heading": "text",
    "paragraph": "text", "list": "text", "caption": "text", "toc": "text",
    "page_header": "boilerplate", "page_footer": "boilerplate",
    "page_number": "boilerplate", "footnote": "boilerplate", "marginalia": "boilerplate",
    "formula": "other",
}

_LANDING = {
    "table": "table",
    "figure": "figure",
    "logo": "logo",
    "text": "text",
    "title": "text",
    "marginalia": "boilerplate",
    "page_header": "boilerplate", "page_footer": "boilerplate", "page_number": "boilerplate",
    "scan_code": "code",
    "form": "other", "key_value": "other", "card": "other", "attestation": "other",
}


def coarse_ours(t: str) -> str:
    return _OURS.get(t, "other")


def coarse_landing(t: str) -> str:
    return _LANDING.get(t, "other")
