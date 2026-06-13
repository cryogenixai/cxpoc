"""Pipeline stages (design §3). Each stage is a pure function of the artifact
store: it reads inputs per the manifest, writes outputs, and updates the
manifest — idempotent and resumable. Skeleton implementations are trivial.
"""

from .assemble import AssembleStage
from .extract import ExtractStage
from .ingest import IngestStage
from .layout import LayoutStage

__all__ = ["IngestStage", "LayoutStage", "ExtractStage", "AssembleStage"]
