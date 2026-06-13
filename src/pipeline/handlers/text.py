"""Text-like handler: title / section_header / paragraph / list / caption.

On the digital path this needs no model call (§2): clip the words whose boxes
fall inside the region bbox and join them in order.
"""

from __future__ import annotations

from ..schema import ChunkResult, PageContext, Region
from ..vlm import VLMClient


def _inside(word_bbox: dict[str, float], region_bbox: dict[str, float]) -> bool:
    """True if the word box center sits within the region box (pixel coords)."""
    cx = (word_bbox["x0"] + word_bbox["x1"]) / 2
    cy = (word_bbox["y0"] + word_bbox["y1"]) / 2
    return (
        region_bbox["x0"] <= cx <= region_bbox["x1"]
        and region_bbox["y0"] <= cy <= region_bbox["y1"]
    )


class TextHandler:
    # Coarse detector classes plus the fine types produced by classification
    # refinement (§4.1): section_header -> heading, page_header/footer ->
    # page_number, and the toc/other catch-alls. All extracted from the text layer.
    handles = {"title", "heading", "section_header", "paragraph", "list", "caption",
               "page_header", "page_footer", "page_number", "footnote", "toc", "other"}

    def extract(self, region: Region, page: PageContext, vlm: VLMClient,
                crop: bytes | None = None) -> ChunkResult:
        words = [w["text"] for w in page.words if _inside(w["bbox"], region.bbox)]
        return ChunkResult(
            type=region.type,
            content={"text": " ".join(words)},
            source="text-layer",
            confidence=1.0,
        )
