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

# Fixed region template per page (pixel bboxes), stacked top-to-bottom so a
# simple y-then-x sort yields the natural reading order.
_TEMPLATE = [
    ("title",     {"x0": 100, "y0": 100,  "x1": 1554, "y1": 300}),
    ("paragraph", {"x0": 100, "y0": 350,  "x1": 1554, "y1": 800}),
    ("table",     {"x0": 100, "y0": 850,  "x1": 1554, "y1": 1300}),
    ("chart",     {"x0": 100, "y0": 1350, "x1": 1554, "y1": 1900}),
    ("figure",    {"x0": 100, "y0": 1950, "x1": 1554, "y1": 2300}),
]


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
            regions = []
            for i, (rtype, bbox) in enumerate(_TEMPLATE):
                resolved = rtype
                if rtype == "figure":
                    # Figure-type router refines the crop (mock -> diagram).
                    resolved = self.vlm.classify_figure(image_bytes=b"")
                regions.append({
                    "region_id": f"p{pi:04d}_r{i:02d}",
                    "page_index": pi,
                    "type": resolved,
                    "bbox": bbox,
                    "detector_confidence": 1.0,
                })
            ctx.write_json(regions, "layout", f"p{pi:04d}.regions.json")

        M.set_stage(manifest, self.name, M.DONE, model="stub-layout")
        M.save(ctx, manifest)
