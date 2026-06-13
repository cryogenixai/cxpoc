"""Layout detection (design §3 Stage 1).

A LayoutDetector turns a page image into typed regions. Two implementations
behind one interface (the design's swappable-component principle):

  * StubLayoutDetector    — fixed positional bands; no model. Fast tests.
  * DocLayoutYOLODetector — real DocLayout-YOLO detection (CPU-capable).

The figure-type router (figure -> chart|diagram|logo|photo) lives in the layout
*stage*, not here, because it calls the VLM; detectors stay model-of-vision only.
"""

from .layout_model import (
    DocLayoutYOLODetector,
    LayoutDetector,
    StubLayoutDetector,
    make_detector,
)

__all__ = [
    "LayoutDetector",
    "StubLayoutDetector",
    "DocLayoutYOLODetector",
    "make_detector",
]
