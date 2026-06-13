"""Image handler (§R2): logo / diagram / photo. VLM captioning (mock in skeleton)."""

from __future__ import annotations

from ..schema import ChunkResult, PageContext, Region
from ..vlm import VLMClient


class ImageHandler:
    handles = {"diagram", "logo", "photo"}

    def extract(self, region: Region, page: PageContext, vlm: VLMClient) -> ChunkResult:
        description = vlm.describe_image(image_bytes=b"")
        return ChunkResult(
            type=region.type,
            content={"description": description},
            source="vlm",
            confidence=0.5,
            model=vlm.model_id,
        )
