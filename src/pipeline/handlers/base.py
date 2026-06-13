"""RegionHandler interface + the type->handler registry (design §6)."""

from __future__ import annotations

from typing import Protocol

from ..schema import ChunkResult, PageContext, Region
from ..vlm import VLMClient


class RegionHandler(Protocol):
    handles: set[str]

    def extract(self, region: Region, page: PageContext, vlm: VLMClient) -> ChunkResult: ...


def build_registry() -> dict[str, RegionHandler]:
    """Map each region type to the handler that owns it."""
    # Imported here to avoid a circular import at module load.
    from .chart import ChartHandler
    from .image import ImageHandler
    from .table import TableHandler
    from .text import TextHandler

    handlers: list[RegionHandler] = [
        TextHandler(),
        TableHandler(),
        ChartHandler(),
        ImageHandler(),
    ]
    registry: dict[str, RegionHandler] = {}
    for handler in handlers:
        for region_type in handler.handles:
            registry[region_type] = handler
    return registry
