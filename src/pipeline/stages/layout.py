"""Stage 1 — Layout Detection & Figure Routing (design §3).

Runs a LayoutDetector on each page image to get typed regions, then:
  * refines every ``figure`` region via the VLM figure router
    (figure -> chart|diagram|logo|photo); mock router returns "diagram" until
    the GPU VLM is live, so the seam is identical now and later.
  * writes a crop (regions/pNNNN_rNN.png) for any region a downstream handler
    needs as an image (tables, charts, images), recording crop_key.

The detector is swappable: a fast StubLayoutDetector for tests, real
DocLayout-YOLO otherwise (selected via CXPOC_LAYOUT_DETECTOR). Output key
(layout/pNNNN.regions.json) is unchanged from the skeleton.
"""

from __future__ import annotations

import io

from .. import manifest as M
from .. import refine
from ..detect import LayoutDetector, make_detector
from ..jobctx import JobContext
from ..vlm import VLMClient

# Region types whose handlers consume the pixel crop.
_IMAGE_TYPES = {"table", "chart", "diagram", "logo", "photo"}


def _crop_png(page_png: bytes, bbox: dict[str, float]) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(page_png)) as im:
        box = (int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"]))
        crop = im.convert("RGB").crop(box)
        out = io.BytesIO()
        crop.save(out, format="PNG")
        return out.getvalue()


class LayoutStage:
    name = "layout"

    def __init__(self, detector: LayoutDetector | None = None, vlm: VLMClient | None = None):
        self.detector = detector or make_detector()
        self.vlm = vlm or VLMClient()

    def run(self, ctx: JobContext) -> None:
        manifest = M.load(ctx)
        if M.stage_status(manifest, self.name) == M.DONE:
            return

        M.set_stage(manifest, self.name, M.RUNNING)
        M.save(ctx, manifest)

        pages = ctx.read_json("pages", "pages.json")
        for page in pages:
            pi = page["page_index"]
            page_png = ctx.storage.read(ctx.key(page["png_key"]))
            raw = self.detector.detect(page_png, pi)

            # Text-layer signals for classification refinement (§4.1). The page
            # body font size is the modal word size; word boxes share the region
            # pixel coords, so clipping is direct.
            words = ctx.storage.read_json(ctx.key(page["words_key"]))
            body_size = refine.body_font_size([w.get("size", 0.0) for w in words])
            page_h = page["height_px"]

            regions = []
            for i, r in enumerate(raw):
                rtype = r["type"]
                bbox = r["bbox"]
                region_id = f"p{pi:04d}_r{i:02d}"

                # Figure router: refine figure -> chart|diagram|logo|photo.
                if rtype == "figure":
                    crop = _crop_png(page_png, bbox)
                    rtype = self.vlm.classify_figure(image_bytes=crop)

                # Classification refinement: coarse class -> fine type + attributes
                # using exact text-layer signals (font size, leading glyph, position).
                words_in = refine.clip_words(words, bbox)
                rtype, attributes = refine.refine_region(rtype, words_in, bbox, body_size, page_h)

                crop_key = None
                if rtype in _IMAGE_TYPES:
                    crop_key = f"regions/{region_id}.png"
                    ctx.storage.write(ctx.key(crop_key), _crop_png(page_png, bbox))

                regions.append({
                    "region_id": region_id,
                    "page_index": pi,
                    "type": rtype,
                    "bbox": bbox,
                    "detector_confidence": r.get("detector_confidence", 1.0),
                    "crop_key": crop_key,
                    "attributes": attributes,
                })

            ctx.write_json(regions, "layout", f"p{pi:04d}.regions.json")

        M.set_stage(manifest, self.name, M.DONE, model=self.detector.name)
        M.save(ctx, manifest)
