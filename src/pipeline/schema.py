"""Output schema + the small typed records passed between stages (design §4).

``OUTPUT_SCHEMA`` is the JSON Schema the final ``document.json`` is validated
against before a job is declared done (§3 step 3). ``content`` is polymorphic by
``type`` (text / html / chart-object / description), so the schema validates the
common envelope (id, type, bbox, reading_order, source, confidence) and leaves
``content`` open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jsonschema

SCHEMA_VERSION = "1.1"

_BBOX = {
    "type": "object",
    "required": ["x0", "y0", "x1", "y1"],
    "properties": {
        # Normalised 0..1 relative coordinates (resolution-independent, §3).
        "x0": {"type": "number", "minimum": 0, "maximum": 1},
        "y0": {"type": "number", "minimum": 0, "maximum": 1},
        "x1": {"type": "number", "minimum": 0, "maximum": 1},
        "y1": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_CHUNK = {
    "type": "object",
    "required": ["id", "type", "bbox", "reading_order", "content", "source", "confidence"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        # Modifiers per §4.1: heading level, list ordering, continuation, etc.
        # Open object ({} when none) — kept optional for backward compatibility.
        "attributes": {"type": "object"},
        # True for boilerplate/navigation types the chunker excludes by default.
        "is_boilerplate": {"type": "boolean"},
        "bbox": _BBOX,
        "reading_order": {"type": "integer", "minimum": 0},
        "content": {"type": "object"},  # polymorphic by type
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_PAGE = {
    "type": "object",
    "required": ["page_index", "width_px", "height_px", "dpi", "page_kind", "chunks"],
    "properties": {
        "page_index": {"type": "integer", "minimum": 0},
        "width_px": {"type": "integer", "minimum": 1},
        "height_px": {"type": "integer", "minimum": 1},
        "dpi": {"type": "integer", "minimum": 1},
        "page_kind": {"enum": ["digital", "scanned"]},
        "chunks": {"type": "array", "items": _CHUNK},
    },
}

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "job_id", "source", "pipeline", "pages"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "job_id": {"type": "string"},
        "source": {
            "type": "object",
            "required": ["filename", "pages"],
            "properties": {
                "filename": {"type": "string"},
                "pages": {"type": "integer", "minimum": 0},
                "sha256": {"type": "string"},
            },
        },
        "pipeline": {"type": "object"},
        "pages": {"type": "array", "items": _PAGE},
    },
}


def validate(document: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the document is not schema-valid."""
    jsonschema.validate(document, OUTPUT_SCHEMA)


# -- Inter-stage records ------------------------------------------------------

@dataclass
class Region:
    """A typed region detected on a page (Stage 1 output)."""

    region_id: str
    page_index: int
    type: str
    bbox: dict[str, float]          # pixel coords at this stage; normalised in assemble
    detector_confidence: float = 1.0
    crop_key: str | None = None     # regions/pNNNN_rNN.png, for image-needing handlers
    attributes: dict[str, Any] = field(default_factory=dict)  # §4.1 modifiers (level, ordered, …)


@dataclass
class PageContext:
    """Everything a handler needs about the page a region sits on."""

    page_index: int
    width_px: int
    height_px: int
    dpi: int
    page_kind: str
    words: list[dict[str, Any]] = field(default_factory=list)  # [{text, bbox}]
    page_png_key: str | None = None


@dataclass
class ChunkResult:
    """A handler's extraction output for one region (Stage 2 output)."""

    type: str
    content: dict[str, Any]
    source: str
    confidence: float
    model: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
