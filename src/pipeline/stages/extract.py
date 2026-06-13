"""Stage 2 — Region Handlers (design §3).

Each region is an independent task routed by type to a RegionHandler. Per-region
status lives in the manifest, so a re-run resumes per region (not per stage) —
a failed/slow VLM region doesn't redo the whole page. Regions run concurrently
via a thread pool (VLM calls are I/O-bound, §6); this is also the seam where
multi-doc concurrency plugs in later.

Output: one extracted/pNNNN_rNN.json per region.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .. import manifest as M
from ..handlers import build_registry
from ..jobctx import JobContext
from ..schema import PageContext, Region
from ..vlm import VLMClient


class ExtractStage:
    name = "extract"

    def __init__(self, vlm: VLMClient | None = None, max_workers: int = 4):
        self.vlm = vlm or VLMClient()
        self.registry = build_registry()
        self.max_workers = max_workers

    def run(self, ctx: JobContext) -> None:
        manifest = M.load(ctx)
        if M.stage_status(manifest, self.name) == M.DONE:
            return

        M.set_stage(manifest, self.name, M.RUNNING)
        M.save(ctx, manifest)

        pages = {p["page_index"]: p for p in ctx.read_json("pages", "pages.json")}
        words_by_page = {
            pi: ctx.storage.read_json(ctx.key(p["words_key"]))
            for pi, p in pages.items()
        }

        # Gather all regions that still need work (idempotent resume).
        todo: list[Region] = []
        for pi, page in pages.items():
            regions = ctx.read_json("layout", f"p{pi:04d}.regions.json")
            for r in regions:
                if M.region_status(manifest, r["region_id"]) == M.DONE:
                    continue
                todo.append(Region(**r))

        def handle(region: Region):
            page = pages[region.page_index]
            page_ctx = PageContext(
                page_index=region.page_index,
                width_px=page["width_px"],
                height_px=page["height_px"],
                dpi=page["dpi"],
                page_kind=page["page_kind"],
                words=words_by_page[region.page_index],
                page_png_key=page["png_key"],
            )
            handler = self.registry.get(region.type)
            if handler is None:
                raise KeyError(f"no handler for region type {region.type!r}")
            crop = None
            if region.crop_key:
                crop = ctx.storage.read(ctx.key(region.crop_key))
            result = handler.extract(region, page_ctx, self.vlm, crop=crop)
            return region, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for region, result in pool.map(handle, todo):
                record = {
                    "region_id": region.region_id,
                    "page_index": region.page_index,
                    "type": result.type,
                    "attributes": region.attributes,  # §4.1 modifiers set in layout
                    "bbox": region.bbox,
                    "content": result.content,
                    "source": result.source,
                    "confidence": result.confidence,
                    "model": result.model,
                    "timings": result.timings,
                }
                ctx.write_json(record, "extracted", f"{region.region_id}.json")
                # Reload-free incremental manifest update would race across
                # threads; we update after each completes in this loop (the
                # pool.map iterator is consumed serially here).
                M.set_region(manifest, region.region_id, M.DONE,
                             handler=result.type)
                M.save(ctx, manifest)

        M.set_stage(manifest, self.name, M.DONE)
        M.save(ctx, manifest)
