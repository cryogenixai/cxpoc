"""Stage 0 — Ingest & Triage (design §3).

Opens the PDF with PyMuPDF. For each page:

  1. Render to PNG at ~200 DPI (vision models need the image regardless of path).
  2. Triage digital vs. scanned via a text-coverage heuristic.
  3. digital path: extract words + pixel-space bboxes from the text layer
     (lossless ground truth, free). scanned path: left as a placeholder — emits
     an empty word list and tags the page ``scanned`` (real OCR is a later slice).

Word and page geometry are stored in **pixel coordinates** of the rendered image
(PyMuPDF text coords scaled by DPI/72), so they line up 1:1 with the PNG and with
the layout stage's region boxes. Assembly normalises to 0..1 at the end.

Outputs (per §0): pages/pNNNN.png, pages/pNNNN.words.json, plus a pages.json
index of page metadata for downstream stages.
"""

from __future__ import annotations

import hashlib

import fitz  # PyMuPDF

from .. import manifest as M
from ..jobctx import JobContext

DPI = 200
_SCALE = DPI / 72.0  # PDF points -> rendered pixels

# Triage heuristic: a page with at least this many text-layer characters is
# treated as born-digital; below it we fall back to the (placeholder) scanned
# path. Crude but sufficient for v1 clean corporate PDFs; the real ink-coverage
# ratio from §0 can replace it without changing the contract.
_DIGITAL_MIN_CHARS = 5


def _extract_words(page: "fitz.Page") -> list[dict]:
    """Text-layer words with bboxes scaled to rendered-image pixel coords.

    PyMuPDF ``get_text("words")`` yields (x0, y0, x1, y1, word, block, line, n)
    in PDF points with a top-left origin — same orientation as the image, so we
    only scale by DPI. Words arrive in reading order.
    """
    words = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        words.append({
            "text": text,
            "bbox": {
                "x0": x0 * _SCALE,
                "y0": y0 * _SCALE,
                "x1": x1 * _SCALE,
                "y1": y1 * _SCALE,
            },
        })
    return words


class IngestStage:
    name = "ingest"

    def run(self, ctx: JobContext) -> None:
        manifest = M.load(ctx)
        if M.stage_status(manifest, self.name) == M.DONE:
            return  # idempotent: already done, skip

        M.set_stage(manifest, self.name, M.RUNNING)
        M.save(ctx, manifest)

        source_bytes = ctx.read_bytes("source.pdf")
        doc = fitz.open(stream=source_bytes, filetype="pdf")

        pages_meta = []
        try:
            for pi in range(doc.page_count):
                page = doc.load_page(pi)

                pix = page.get_pixmap(dpi=DPI)
                png = pix.tobytes("png")

                char_count = len(page.get_text("text").strip())
                is_digital = char_count >= _DIGITAL_MIN_CHARS
                words = _extract_words(page) if is_digital else []

                png_key = f"pages/p{pi:04d}.png"
                words_key = f"pages/p{pi:04d}.words.json"
                ctx.write_bytes(png, png_key)
                ctx.write_json(words, words_key)

                pages_meta.append({
                    "page_index": pi,
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "dpi": DPI,
                    "page_kind": "digital" if is_digital else "scanned",
                    "png_key": png_key,
                    "words_key": words_key,
                })
        finally:
            doc.close()

        ctx.write_json(pages_meta, "pages", "pages.json")

        sha = hashlib.sha256(source_bytes).hexdigest()
        manifest["source"]["sha256"] = sha
        manifest["source"]["pages"] = len(pages_meta)

        M.set_stage(manifest, self.name, M.DONE, pages=len(pages_meta))
        M.save(ctx, manifest)
