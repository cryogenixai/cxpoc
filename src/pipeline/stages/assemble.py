"""Stage 3 — Assembly (design §3).

Sort regions into reading order (XY-cut: y-then-x — sufficient for corporate
single/two-column layouts), normalise all bboxes to 0..1 relative coordinates,
merge into the final document.json, and validate against the output schema (§4)
before declaring the job done.
"""

from __future__ import annotations

from .. import manifest as M
from ..jobctx import JobContext
from ..schema import SCHEMA_VERSION, validate


def _normalise(bbox: dict[str, float], w: int, h: int) -> dict[str, float]:
    return {
        "x0": bbox["x0"] / w,
        "y0": bbox["y0"] / h,
        "x1": bbox["x1"] / w,
        "y1": bbox["y1"] / h,
    }


class AssembleStage:
    name = "assemble"

    def run(self, ctx: JobContext) -> None:
        manifest = M.load(ctx)
        if M.stage_status(manifest, self.name) == M.DONE:
            return

        M.set_stage(manifest, self.name, M.RUNNING)
        M.save(ctx, manifest)

        pages_meta = ctx.read_json("pages", "pages.json")

        out_pages = []
        for page in pages_meta:
            pi = page["page_index"]
            w, h = page["width_px"], page["height_px"]

            # Load every extracted region for this page.
            records = []
            for r in ctx.read_json("layout", f"p{pi:04d}.regions.json"):
                records.append(ctx.read_json("extracted", f"{r['region_id']}.json"))

            # XY-cut reading order: sort by top edge, then left edge (pixel space).
            records.sort(key=lambda rec: (rec["bbox"]["y0"], rec["bbox"]["x0"]))

            chunks = []
            for order, rec in enumerate(records):
                chunks.append({
                    "id": rec["region_id"],
                    "type": rec["type"],
                    "bbox": _normalise(rec["bbox"], w, h),
                    "reading_order": order,
                    "content": rec["content"],
                    "source": rec["source"],
                    "confidence": rec["confidence"],
                })

            out_pages.append({
                "page_index": pi,
                "width_px": w,
                "height_px": h,
                "dpi": page["dpi"],
                "page_kind": page["page_kind"],
                "chunks": chunks,
            })

        document = {
            "schema_version": SCHEMA_VERSION,
            "job_id": ctx.job_id,
            "source": manifest["source"],
            "pipeline": {
                "versions": {
                    "layout": manifest["stages"].get("layout", {}).get("model", "stub"),
                    "vlm": manifest["stages"].get("extract", {}).get("model", "mock"),
                }
            },
            "pages": out_pages,
        }

        # Validate before declaring done (§3 step 3).
        validate(document)
        ctx.write_json(document, "output", "document.json")

        M.set_stage(manifest, self.name, M.DONE)
        M.mark_complete(manifest)
        M.save(ctx, manifest)
