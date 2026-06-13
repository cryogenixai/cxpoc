"""Table structure recognition (design §2 table handler, R4).

Two implementations behind one interface:
  * StubTableRecognizer — no model; emits a single text-layer cell. Fast tests.
  * TATRRecognizer      — Table Transformer structure recognition: recovers the
                          row/column grid (+ spanning cells), fills each cell
                          from the text layer, emits HTML.
"""

from .table_model import (
    StubTableRecognizer,
    TableRecognizer,
    TATRRecognizer,
    make_table_recognizer,
)

__all__ = [
    "TableRecognizer",
    "StubTableRecognizer",
    "TATRRecognizer",
    "make_table_recognizer",
]
