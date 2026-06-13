"""Stage 0 — Ingest & Triage (design §3).

SKELETON: does not yet open the PDF with PyMuPDF. It emits one placeholder page
(a 1x1 PNG, a small fixed word list, ``digital`` page kind) so the rest of the
pipeline has a faithful Stage 0 contract to consume. The real PyMuPDF render +
text-layer extraction + digital/scanned triage is the first vertical slice (§7,
build step 2); it replaces this body without changing the output keys.

Outputs (per §0): pages/pNNNN.png, pages/pNNNN.words.json, plus a pages.json
index of page metadata for downstream stages.
"""

from __future__ import annotations

import base64
import hashlib

from .. import manifest as M
from ..jobctx import JobContext

# A minimal valid 1x1 PNG — placeholder page render for the skeleton.
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Placeholder page geometry (≈ US-Letter @ 200 DPI).
_W, _H, _DPI = 1654, 2339, 200

# Placeholder words, positioned so the layout stub's title/paragraph regions
# clip them meaningfully (see layout.py for the matching region boxes).
_WORDS = [
    {"text": "Hello", "bbox": {"x0": 120, "y0": 150, "x1": 300, "y1": 220}},
    {"text": "Cryogenic", "bbox": {"x0": 320, "y0": 150, "x1": 600, "y1": 220}},
    {"text": "Body", "bbox": {"x0": 120, "y0": 400, "x1": 300, "y1": 460}},
    {"text": "text", "bbox": {"x0": 320, "y0": 400, "x1": 500, "y1": 460}},
]


class IngestStage:
    name = "ingest"

    def run(self, ctx: JobContext) -> None:
        manifest = M.load(ctx)
        if M.stage_status(manifest, self.name) == M.DONE:
            return  # idempotent: already done, skip

        M.set_stage(manifest, self.name, M.RUNNING)
        M.save(ctx, manifest)

        # One placeholder page (skeleton). Real ingest loops over PDF pages here.
        page_meta = {
            "page_index": 0,
            "width_px": _W,
            "height_px": _H,
            "dpi": _DPI,
            "page_kind": "digital",
            "png_key": "pages/p0000.png",
            "words_key": "pages/p0000.words.json",
        }

        ctx.write_bytes(_PLACEHOLDER_PNG, "pages", "p0000.png")
        ctx.write_json(_WORDS, "pages", "p0000.words.json")
        ctx.write_json([page_meta], "pages", "pages.json")

        # Record source digest now that we've "read" it.
        source_bytes = ctx.read_bytes("source.pdf")
        sha = hashlib.sha256(source_bytes).hexdigest()
        manifest["source"]["sha256"] = sha
        manifest["source"]["pages"] = 1

        M.set_stage(manifest, self.name, M.DONE, pages=1)
        M.save(ctx, manifest)
