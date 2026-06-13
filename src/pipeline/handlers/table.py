"""Table handler (§R4). Skeleton emits a placeholder HTML grid.

Real version: Table Transformer (TATR) structure recognition reconciled against
the text layer, with a VLM->HTML fallback on low structure confidence (§2). The
output contract — ``content.html`` — does not change when that lands.
"""

from __future__ import annotations

from ..schema import ChunkResult, PageContext, Region
from ..vlm import VLMClient


class TableHandler:
    handles = {"table"}

    def extract(self, region: Region, page: PageContext, vlm: VLMClient) -> ChunkResult:
        html = "<table><tr><td>[mock cell]</td></tr></table>"
        return ChunkResult(
            type="table",
            content={"html": html},
            source="stub",
            confidence=0.5,
        )
