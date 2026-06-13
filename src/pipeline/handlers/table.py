"""Table handler (§R4).

Primary: a TableRecognizer (TATR) recovers the cell grid from the region crop and
fills cells from the text layer -> HTML. Fallback: when structure confidence is
low (or no grid found), send the crop to the VLM prompted to emit HTML directly.
The recognizer is swappable (stub for tests) via CXPOC_TABLE_RECOGNIZER.
"""

from __future__ import annotations

from ..schema import ChunkResult, PageContext, Region
from ..tables import TableRecognizer, make_table_recognizer
from ..vlm import VLMClient

# Below this structure confidence we don't trust the grid; fall back to the VLM.
_FALLBACK_THRESHOLD = 0.3


class TableHandler:
    handles = {"table"}

    def __init__(self, recognizer: TableRecognizer | None = None):
        self.recognizer = recognizer or make_table_recognizer()

    def extract(self, region: Region, page: PageContext, vlm: VLMClient,
                crop: bytes | None = None) -> ChunkResult:
        result = None
        if crop is not None:
            result = self.recognizer.recognize(crop, region.bbox, page.words)

        # Fallback to the VLM when there's no usable grid.
        if result is None or not result["html"] or result["confidence"] < _FALLBACK_THRESHOLD:
            html = vlm.extract_table(image_bytes=crop or b"")
            return ChunkResult(
                type="table",
                content={"html": html},
                source="vlm-fallback",
                confidence=0.5,
                model=vlm.model_id,
            )

        return ChunkResult(
            type="table",
            content={"html": result["html"]},
            source=result["source"],
            confidence=result["confidence"],
        )
