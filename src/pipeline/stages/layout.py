"""Stage 1 — Layout Detection & Figure Routing (design §3).

SKELETON: emits a fixed set of typed regions per page instead of running
DocLayout-YOLO. The set deliberately covers every handler route (title,
paragraph, table, chart, figure) so the downstream routing fabric and the
figure-type router are both exercised end-to-end. The ``figure`` region is run
through the VLM figure router (mock -> "diagram") exactly as the real Stage 1
will, refining it to chart|diagram|logo|photo.

Real Stage 1 (vertical slice, §7 build step 3) swaps the fixed list for the
detector + router; the output key (layout/pNNNN.regions.json) is unchanged.
"""

from __future__ import annotations

from .. import manifest as M
from ..jobctx import JobContext
from ..vlm import VLMClient

# Fixed region template per page as *fractions* of the page, so the stub adapts
# to whatever real page dimensions Stage 0 produces. Stacked top-to-bottom so a
# simple y-then-x sort yields the natural reading order. The title band covers
# the top of the page, where a document's heading text typically sits.
_TEMPLATE = [
    ("title",     (0.00, 0.00, 1.00, 0.15)),
    ("paragraph", (0.00, 0.15, 1.00, 0.45)),
    ("table",     (0.00, 0.45, 1.00, 0.62)),
    ("chart",     (0.00, 0.62, 1.00, 0.85)),
    ("figure",    (0.00, 0.85, 1.00, 1.00)),
]


def _scale(frac: tuple[float, float, float, float], w: int, h: int) -> dict[str, float]:
    fx0, fy0, fx1, fy1 = frac
    return {"x0": fx0 * w, "y0": fy0 * h, "x1": fx1 * w, "y1": fy1 * h}


class LayoutStage:
    name = "layout"

    def __init__(self, vlm: VLMClient | None = None):
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
            w, h = page["width_px"], page["height_px"]
            regions = []
            for i, (rtype, frac) in enumerate(_TEMPLATE):
                resolved = rtype
                if rtype == "figure":
                    # Figure-type router refines the crop (mock -> diagram).
                    resolved = self.vlm.classify_figure(image_bytes=b"")
                regions.append({
                    "region_id": f"p{pi:04d}_r{i:02d}",
                    "page_index": pi,
                    "type": resolved,
                    "bbox": _scale(frac, w, h),
                    "detector_confidence": 1.0,
                })
            ctx.write_json(regions, "layout", f"p{pi:04d}.regions.json")

        M.set_stage(manifest, self.name, M.DONE, model="stub-layout")
        M.save(ctx, manifest)
