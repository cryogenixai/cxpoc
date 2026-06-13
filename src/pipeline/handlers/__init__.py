"""Region handlers (design §2 stage 2). One interface, routed by region type.

The walking-skeleton handlers are deliberately trivial: text echoes the words
clipped from the text layer; table emits a placeholder HTML grid; chart and
image call the (mock) VLM. They prove the routing fabric and the ChunkResult
contract. Real implementations (TATR, schema-prompted chart extraction, VLM
captioning) replace these one vertical slice at a time (§7) with no change to
the interface or the rest of the pipeline.
"""

from .base import RegionHandler, build_registry

__all__ = ["RegionHandler", "build_registry"]
