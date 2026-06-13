"""Chart handler (§R3). Calls the VLM (mock in the skeleton) for a strict-schema
extraction, and would additionally fold in text-layer data labels for born-digital
charts (§2). Exercises the VLM seam end-to-end.
"""

from __future__ import annotations

from ..schema import ChunkResult, PageContext, Region
from ..vlm import VLMClient


class ChartHandler:
    handles = {"chart"}

    def extract(self, region: Region, page: PageContext, vlm: VLMClient,
                crop: bytes | None = None) -> ChunkResult:
        words = [w for w in page.words]  # text-layer labels would be filtered to bbox here
        chart = vlm.extract_chart(image_bytes=crop or b"", words=words)
        confidence = float(chart.pop("confidence", 0.5))
        return ChunkResult(
            type="chart",
            content=chart,
            source="vlm",
            confidence=confidence,
            model=vlm.model_id,
        )
