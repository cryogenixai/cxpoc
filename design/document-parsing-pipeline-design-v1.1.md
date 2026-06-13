# Document Parsing Pipeline — Design Document

**Status:** Draft v0.3 · **Date:** 2026-06-12
**Scope of this doc:** Extraction pipeline only. API design, serverless deployment, and multi-document concurrency are deliberately out of scope for v1 but the design keeps clean seams for them.

---

## 1. Goals and Requirements

Build a system that parses documents (born-digital PDFs and, later, scanned images) into a single structured JSON output.

### Functional requirements

| # | Requirement |
|---|-------------|
| R1 | Complete JSON output with bounding-box information for every extracted element |
| R2 | Images (logos, diagrams, photos) include a generated textual description |
| R3 | Charts reveal data points, x/y axis labels and ticks, title/heading, and legend |
| R4 | Tables identified and converted to HTML-style `<table><tr><td>` structure (incl. rowspan/colspan) |

### Scope

- **v1:** Clean, mostly born-digital corporate PDFs. One document processed at a time. Batch latency is acceptable.
- **Future (designed for, not built):** Messy scans, handwriting, stamps/signatures; multi-document concurrency; serverless API.

### Non-goals for v1

No API layer, no auth, no UI, no fine-tuning of models, no distributed orchestration.

---

## 2. Architecture Principle

**Hybrid pipeline, not a single end-to-end VLM.** Specialized detectors provide precise localization (bounding boxes); vision-language models provide interpretation (chart data, diagram descriptions, table fallback). Everything is grounded back to page coordinates.

Secondary principles:

1. **Every stage is a swappable component behind a common interface.** The page image and region crops are the universal currency between components.
2. **Stages are pure functions of files.** Input paths → output paths; no in-memory state crosses stage boundaries. This makes the later jump to distributed/serverless execution an orchestration swap, not a refactor.
3. **Confidence and provenance on every chunk** from day one, so low-confidence routing (heavier model, human review) can be added later without schema changes.
4. **Evidence-driven evolution.** Model swaps and fine-tunes are justified only by measured failures in the eval harness.

---

## 3. Pipeline Stages

```
source.pdf
   │
   ▼
[Stage 0: Ingest & Triage]──► page images + text-layer words (or OCR words)
   │
   ▼
[Stage 1: Layout Detection + Figure Routing]──► typed regions with bboxes
   │
   ▼
[Stage 2: Region Handlers]  (parallel per region)
   ├── text/title/list ──► text content (from text layer)
   ├── table ──► HTML <tr>/<td> structure
   ├── chart ──► structured data points JSON
   └── logo/diagram/photo ──► VLM description
   │
   ▼
[Stage 3: Assembly]──► reading order + final document.json
```

### Stage 0 — Ingest & Triage

Open the PDF with **PyMuPDF**. For each page:

1. Render the page to PNG at ~150–200 DPI (vision models need the image regardless of path).
2. Detect whether a reliable text layer exists (heuristic: extracted character coverage vs. rendered ink). Tag the page `digital` or `scanned`.
3. `digital` path: extract words + coordinates directly from the text layer (lossless ground truth, free).
4. `scanned` path (v1 placeholder): **PaddleOCR** or **Surya** to produce word boxes. Quality here matters less in v1 than having the routing skeleton in place.

**Output:** `pages/pNNNN.png`, `pages/pNNNN.words.json` (word text + bbox + source).

### Stage 1 — Layout Detection & Figure Routing

Run a layout detector (**DocLayout-YOLO**, or the Docling layout model) on each page image. Region classes: `title`, `section_header`, `paragraph`, `list`, `table`, `figure`, `caption`, `page_header`, `page_footer`, `footnote`.

Detectors trained on DocLayNet do not reliably distinguish chart vs. diagram vs. logo vs. photo — they lump them as `figure`. Therefore a **figure-type router** refines each `figure` crop into `chart | diagram | logo | photo` via either a tiny classifier or one cheap VLM call per crop. The router is the future extension point for new classes (stamps, signatures, form fields).

**Output:** `layout/pNNNN.regions.json` — list of `{region_id, type, bbox, detector_confidence}` — plus region crops `regions/pNNNN_rNN.png` for any region a downstream handler needs as an image.

### Stage 2 — Region Handlers (parallel per region)

Each region is an independent task routed by type. Handlers share one interface (see §6).

**Text / title / list / caption.** Clip words from the Stage 0 word list using the region bbox; join in order. No OCR/model call on the digital path. Keep word-level boxes internally; emit the region-level bbox in output.

**Table (R4).** Primary: **Table Transformer (TATR)** for structure recognition → cell grid with cell bboxes. Fill cell contents by reconciling cell bboxes against the text layer (more accurate than OCRing crops). Emit HTML `<table><tr><td>` including `rowspan`/`colspan`; optionally retain cell-level bboxes for auditability. Fallback: when TATR structure confidence is low, send the crop to a VLM prompted to emit HTML directly.

**Chart (R3).** Crop → VLM (e.g., Qwen2.5-VL or Claude API) with a strict JSON-schema prompt:

```json
{
  "chart_type": "bar|line|pie|scatter|other",
  "title": "...",
  "x_axis": {"label": "...", "ticks": ["..."]},
  "y_axis": {"label": "...", "ticks": ["..."]},
  "legend": ["series A", "series B"],
  "series": [{"name": "...", "points": [{"x": "...", "y": 0.0}]}],
  "confidence": 0.0
}
```

Additionally extract any text-layer words falling inside the chart bbox — born-digital charts often carry exact data labels as text, which beats visual estimation. Data points read off pixels are estimates and must carry a lower confidence.

**Logo / diagram / photo (R2).** Crop → VLM with a structured captioning prompt: what it depicts, components and relationships (for diagrams), any embedded text.

**Output:** one `extracted/pNNNN_rNN.json` per region: `{region_id, type, content, source, confidence, model, timings}`.

### Stage 3 — Assembly

1. Sort regions into logical reading order per page (**XY-cut**; sufficient for corporate single/two-column layouts; replaceable with a learned reading-order model later).
2. Merge into the final document JSON (schema in §4). Normalize all bboxes to 0–1 relative coordinates (resolution-independent) and attach the page index.
3. Validate against the output JSON schema before declaring the job done.

---

## 4. Output JSON Schema (v1)

```json
{
  "schema_version": "1.0",
  "job_id": "uuid-or-content-hash",
  "source": {"filename": "report.pdf", "pages": 12, "sha256": "..."},
  "pipeline": {"versions": {"layout": "doclayout-yolo-x.y", "table": "tatr-x.y", "vlm": "model-id"}},
  "pages": [
    {
      "page_index": 0,
      "width_px": 1654, "height_px": 2339, "dpi": 200,
      "page_kind": "digital",
      "chunks": [
        {
          "id": "p0000_r003",
          "type": "table",
          "bbox": {"x0": 0.07, "y0": 0.31, "x1": 0.93, "y1": 0.58},
          "reading_order": 3,
          "content": {"html": "<table><tr><td>...</td></tr></table>"},
          "source": "tatr+text-layer",
          "confidence": 0.94
        },
        {
          "id": "p0000_r004",
          "type": "chart",
          "bbox": {"x0": 0.10, "y0": 0.60, "x1": 0.90, "y1": 0.88},
          "reading_order": 4,
          "content": {"chart_type": "bar", "title": "...", "x_axis": {}, "y_axis": {}, "legend": [], "series": []},
          "source": "vlm",
          "confidence": 0.71
        }
      ]
    }
  ]
}
```

`content` is polymorphic by `type`: `{text}` for text-like chunks, `{html}` for tables, the chart schema for charts, `{description}` for logos/diagrams/photos. Every chunk always carries `bbox`, `source`, `confidence`.

---

## 5. Persistence & Job Layout

**Economic rationale ("parse once, query many"):** the expensive step is the single pass through the image/PDF modality; everything afterward runs on cheap structured data. Parsing once into an immutable `document.json`, then answering unlimited downstream questions from it without re-touching pixels, is what makes a document system scale — and it is only viable if the parse is information-complete (which L5 measures). This is the economic justification for the immutable-artifact design below, not merely an engineering nicety: it also delivers provenance/audit (every value links to a span, cell, and page), privacy (share structure, keep images private), and cheap schema changes (add fields without reprocessing pixels).

Filesystem artifact store, one directory per job, keyed by job ID (content hash gives free dedup):

```
jobs/{job_id}/
  source.pdf
  manifest.json            # state machine + pointers (single source of truth)
  pages/   p0001.png  p0001.words.json
  layout/  p0001.regions.json
  regions/ p0001_r03.png   # crops for model calls
  extracted/ p0001_r03.json
  output/  document.json
```

Rules:

1. **Append-only artifacts.** A stage never mutates another stage's output.
2. **Atomic manifest updates.** Write to temp file, then rename.
3. **Idempotent stages.** A stage consults the manifest and skips work already `done`; re-running a job is safe and resumes from the last completed unit (per region, not just per stage).
4. **Keep intermediates after success** (TTL/cleanup policy later). Crops and layout JSONs are the raw material for debugging and the eval/fine-tuning set.
5. **Thin storage abstraction** (`storage.read(key)` / `storage.write(key, bytes)`) from day one. v1 backs it with the local filesystem; later it maps 1:1 to S3 object keys. Manifest graduates from JSON file to DynamoDB/Postgres only when cross-job querying is needed.

### Manifest schema (sketch)

```json
{
  "job_id": "...",
  "created_at": "...", "updated_at": "...",
  "status": "running",
  "stages": {
    "ingest":  {"status": "done", "started": "...", "ended": "...", "pages": 12},
    "layout":  {"status": "done", "model": "doclayout-yolo-x.y"},
    "extract": {
      "status": "running",
      "regions": {
        "p0001_r01": {"status": "done", "handler": "text"},
        "p0001_r03": {"status": "failed", "handler": "chart", "attempts": 2, "error": "vlm timeout"}
      }
    },
    "assemble": {"status": "pending"}
  }
}
```

---

## 6. Stage & Handler Interfaces

```python
class Stage(Protocol):
    name: str
    def run(self, job_dir: Path) -> None:
        """Reads inputs per manifest, writes outputs, updates manifest.
        Must be idempotent and resumable."""

class RegionHandler(Protocol):
    handles: set[str]                      # e.g. {"table"}
    def extract(self, region: Region, page_ctx: PageContext) -> ChunkResult:
        ...
```

**Orchestration (v1):** a plain Python runner — `ingest → layout → extract → assemble` — each taking only `job_dir`. No workflow framework; the manifest provides resumability. Within Stage 2, regions are a list of independent tasks executed via `ThreadPoolExecutor` (VLM calls are I/O-bound), which is also the seam where multi-doc concurrency plugs in later (queue of `(job_id, stage)` messages; file contract unchanged).

---

## 7. Testing Strategy — Test As We Build

Four layers, cheapest first:

**L1 — Contract tests** (fast, no models, run on every change). Per stage: schema-valid output from fixture input; correct manifest transitions; idempotency (run twice ≡ run once); resume after simulated crash. Model calls are mocked. Classic TDD applies fully here.

**L2 — Golden-file tests** (real models, run on demand). 3–5 fixture PDFs: text-heavy page, table-heavy page, chart page, multi-column page, edge case (empty/blank page). Run a stage once, review, freeze output as golden. Assert *properties*, not exact equality: "≥1 table detected", "table has 4 columns", "bbox IoU vs. golden ≥ 0.8", "chart JSON parses with non-empty x_axis". Exact-match assertions on model output are brittle by design and are avoided.

**L3 — End-to-end smoke test.** One small PDF through the full pipeline → schema-valid `document.json`. Catches wiring breaks.

**L4 — Component eval harness** (per model-swap / weekly, not per commit). 30–50 representative annotated PDFs (bootstrap annotations with a frontier VLM, then hand-correct). Per-component metrics, because each requirement needs its own: layout mAP (detection), table **TEDS** (structure fidelity), text accuracy (OCR/extraction), chart-data **MAE** + axis/legend match (R3), reading-order accuracy. Every fine-tune or model-swap decision must cite a measured gap here. Public datasets for the hardest requirements: **ChartQA / PlotQA** for charts, a TEDS-scored table set (PubTabNet/FinTabNet) for tables — far more diagnostic for R2/R3 than text-heavy single-page sets.

**L5 — Extraction-completeness QA eval** (the landing.ai DocVQA method, adapted). The premise: *if a text-only LLM can answer questions about a document using only our `document.json` — with no access to the original image — then our parse preserved the information.* This measures information-completeness end to end without hand-labeling boxes, and it is largely self-supervising:

1. A frontier VLM that **does** see the page image generates Q&A pairs (factual, table-lookup, chart-reading, "what does this logo say").
2. A separate text-only LLM answers each question using **only** our JSON output.
3. Score by (case-insensitive) match; low scores localize what the parse dropped.

Make this the headline metric for the text + table paths. Feed bounding-box coordinates into the QA prompt — landing.ai's jump from ~95% (plain markdown) to ~99% came from adding spatial grounding to the structured output, so coordinates are an accuracy lever, not just the R1 deliverable.

### Error taxonomy (adapted from landing.ai's benchmark)

All eval failures (L4 and L5) are classified into buckets that point to *different fixes*, rather than a single accuracy number:

| Bucket | Meaning | Where the fix lives |
|---|---|---|
| **Missed parse** | Information present in the doc but never extracted (recall failure) | Stage 1 layout / Stage 0 OCR |
| **Incorrect parse** | Extracted but wrong (character confusion, table misread) (precision failure) | Stage 2 handler / OCR model |
| **Downstream/prompt miss** | Data correct in JSON but consumer reasons wrong | QA prompt, not the parser |
| **Out-of-scope** | Pure visual/spatial-layout question no parser targets | Documented limitation |
| **Dataset issue** | Ambiguous or wrong ground truth | Excluded from accuracy, kept visible |

Always keep a "dataset issue" bucket so bad labels don't masquerade as model failures, and maintain an always-visible **failure gallery** (image + prediction + ground truth + bucket) — the fastest way to spot systematic handler weaknesses.

### Build order — walking skeleton, vertical slices

1. **Skeleton:** all four stages wired with trivial implementations (layout returns one whole-page region; handlers echo text). L1 + L3 passing. Pipeline "works" end to end on day one.
2. Upgrade **Stage 0** to real PyMuPDF ingest + triage; add golden tests.
3. Upgrade **Stage 1** to DocLayout-YOLO + figure router; golden tests.
4. Upgrade **Stage 2 table handler** (TATR + text-layer reconciliation) — highest engineering risk, table-heavy corpus.
5. Upgrade **Stage 2 chart handler** (VLM + schema prompt + text-layer data labels).
6. Upgrade **Stage 2 image handler** (VLM descriptions).
7. **Stage 3** real reading order + final schema validation.
8. Build the **L4 + L5 eval sets** in parallel from steps 2–7's fixture outputs (L5 Q&A pairs auto-generated by a frontier VLM, spot-checked).

At every step the full pipeline remains runnable.

---

## 8. Accuracy via Agentic Verification (v1.5)

Two distinct things get called "RL agentic"; they have very different cost/benefit.

### 8.1 Agentic verification loops — no RL training (do this)

This is what landing.ai-style "agentic" extraction actually means: orchestration where the model checks and corrects its own work. Document extraction has *cheap verifiers*, which makes these loops unusually effective. Each verifier is a step inside a Stage 2 handler, and the chunk `confidence` becomes verifier-informed rather than self-reported.

- **Chart render-and-compare:** re-plot the extracted series with matplotlib and have a VLM compare the re-render against the original crop ("same trend/shape/values?"). Mismatch → retry with the discrepancy as feedback. The verification task (compare two images) is easier than generation, so it catches hallucinated data points well.
- **Table arithmetic checks:** "Total" rows/columns must equal the sum of their cells; a mismatch re-triggers extraction of that region.
- **Cross-source consistency:** visually-estimated chart values vs. data-label text from the PDF text layer; table cell contents vs. text-layer words inside the cell bbox. Disagreement lowers confidence or triggers a retry.
- **Generator–critic pass:** a second VLM call lists errors in the first extraction against the crop; errors feed a corrected attempt. Typically meaningful accuracy gains for ~2–3x inference cost — acceptable under batch.

Slots into the existing design with no schema change; targeted as v1.5 after the off-the-shelf baseline is measured.

### 8.2 RL training (RLVR / GRPO-style) — defer

Training a VLM with RL needs a verifiable reward; we already have those signals (TEDS for tables, MAE for chart points, IoU for boxes), so the **L4 eval harness is the reward function** if we ever go there. But it is research-grade effort: GPU training infra, thousands of documents, and reward-hacking pitfalls (degenerate gaming of TEDS). Expected return does not beat "better prompt + verifier loop + newer VLM" until those are exhausted. Sequencing: verifier loops (8.1) → measure → consider RL training only for a stubborn, measured gap that survives the cheaper fixes.

---

## 9. Model Choices (v1, all off-the-shelf — no fine-tuning)

| Component | Primary | Fallback / alternative |
|---|---|---|
| PDF ingest / text layer | PyMuPDF | pdfplumber |
| OCR (scanned path, placeholder) | PaddleOCR or Surya | VLM-as-OCR |
| Layout detection | DocLayout-YOLO | Docling layout model, DiT |
| Figure-type router | cheap VLM call per crop | tiny CNN classifier (MobileNet) |
| Table structure | Table Transformer (TATR) | VLM → HTML |
| Chart extraction | VLM w/ strict JSON schema | chart-tuned model (ChartGemma-class) |
| Image/diagram description | VLM captioning prompt | — |
| Reading order | XY-cut | learned reading-order model |

**Fine-tuning policy:** none in v1. Later candidates, in order of likelihood: (1) layout detector fine-tune for domain elements (stamps, signatures) — 200–500 annotated pages; (2) TATR, only if tables are unusual and VLM fallback is insufficient; (3) OCR recognizer, only when handwriting arrives. The VLM is never fine-tuned — prompts and few-shot examples instead. Every fine-tune requires a measured L4 failure, and prompt/model-swap fixes are tried first.

---

## 10. Evolution Path (designed for, not built)

**Handwriting & stamps:** swap Stage 0's OCR engine (TrOCR or VLM-OCR) and add classes to the Stage 1 router. Pipeline shape unchanged.

**Multi-document concurrency:** replace the sequential runner with a queue (SQS + Lambda / Step Functions, or Celery); each message is `(job_id, stage)` or `(job_id, region_id)`. File contracts and manifest unchanged.

**Serverless interface:** S3-backed storage abstraction + Step Functions orchestrating one Lambda (or container task for GPU stages) per stage; API Gateway front door submits jobs and polls manifest status. The v1 design maps onto this 1:1.

**Low-confidence routing / human review:** chunks below a confidence threshold get re-routed to a heavier model or review queue — enabled by the per-chunk `confidence` + `source` fields already in the schema.

---

## 11. Experimental Setup & Infrastructure (do first, before production design)

Goal: run the pipeline end-to-end on real PDFs with the least infrastructure, then harden. Portable and dockerized from the start; AWS VMs + S3 as the substrate.

### 11.1 Compute — two Spot VMs, one per model under test

**Settled decisions:** self-hosted VLM, model weights on a shared EBS volume (not baked into the image).

During the experimentation phase we run **two `g5.xlarge` Spot instances in parallel**, each serving a different VLM, so the same PDF can be processed by both and outputs compared directly.

```
EBS Snapshot — /models/
  qwen2.5-vl-7b-instruct/    (~14 GB, pulled once)
  gemma-4-12b-it/            (~24 GB, pulled once)
        │                           │
        │  restore to               │  restore to
        ▼                           ▼
 g5.xlarge Spot              g5.xlarge Spot
 MODEL=qwen2.5-vl-7b         MODEL=gemma-4-12b-it
 ~$0.35/hr                   ~$0.35/hr

 Same pipeline image          Same pipeline image
 Same S3 bucket               Same S3 bucket
 Different vLLM endpoint      Different vLLM endpoint
```

Instance sizing: `g5.xlarge` (1× A10G 24 GB VRAM, 4 vCPU, 16 GB RAM).
- VLM runs on GPU at 4-bit quantization (~7 GB for 12B, ~4 GB for 7B) — leaves headroom.
- DocLayout-YOLO and TATR run on CPU (acceptable for batch per §12.6).
- Spin VMs up/down independently — pay only while experimenting (~$0.70/hr for both).

The `MODEL` and `VLM_BASE_URL` env vars are the only difference between the two VMs. Pipeline code is identical.

### 11.2 Containerization — one codebase, two build targets

One small pipeline image (~3 GB); model weights live on a mounted EBS volume pulled once per VM. This keeps image rebuilds fast (seconds) and lets you switch the model under test by changing a single env var and restarting the vLLM container — no image rebuild required.

**Two build targets from the same Dockerfile and the same pipeline code**, because dev happens on a macOS laptop (no CUDA, no GPU) and production runs on a Linux GPU VM:

```
Dockerfile  (multi-stage)
  ├── target: dev   → python:3.12-slim (multi-arch: Apple Silicon + Intel)
  │                   pipeline code + deps, NO models, NO CUDA.
  │                   Mac laptop: skeleton + LocalStack + L1/L3 tests.
  │                   VLMClient points at a mock — no vLLM sidecar.
  │
  └── target: gpu   → nvidia/cuda runtime base (linux/amd64)
                      pipeline code + deps, models on mounted volume.
                      EC2 g5.xlarge: real detectors + vLLM sidecar.
```

Only the base image and the presence of the vLLM sidecar differ between targets. `storage.py`, `manifest.py`, the stages, and the handlers are byte-for-byte identical. The walking skeleton (trivial stages, mocked models) runs entirely under the `dev` target on a Mac — no GPU required until real models land in the vertical slices.

```
Dockerfile               # multi-stage: dev (slim, multi-arch) | gpu (CUDA)
docker-compose.yml        # dev: pipeline app + LocalStack
docker-compose.gpu.yml    # gpu: adds vLLM sidecar (model from volume)
pyproject.toml            # fully pinned deps; CUDA-specific deps gated to gpu extras
scripts/
  pull_model.sh          # downloads model weights to /models/ volume on first VM boot
src/pipeline/
  storage.py             # abstraction: LocalFS | S3 backends, read(key)/write(key)
  manifest.py
  vlm.py                 # VLMClient: hits VLM_BASE_URL (OpenAI-compatible) or mock; MODEL env var
  stages/{ingest,layout,extract,assemble}.py
  handlers/{text,table,chart,image}.py
  run.py                 # CLI: python -m pipeline.run --job s3://.../doc.pdf
tests/
```

`vlm.py` is the single seam for model experimentation: all handlers call `VLMClient`, which reads `VLM_BASE_URL` and `VLM_MODEL` from the environment (a `mock://` URL returns canned responses for local dev). Swapping models = change two env vars; swapping dev↔gpu = change the build target.

### 11.3 Storage — S3 from the start, via the abstraction

`storage.py` exposes `read(key)/write(key)` over two backends: `LocalFS` (laptop dev) and `S3`. Locally, **LocalStack** in docker-compose emulates S3 so dev and AWS paths are identical. The job-directory layout from §5 maps 1:1 to S3 keys: `s3://bucket/jobs/{job_id}/...`. The VM gets an **IAM role** for bucket access — no credentials baked into the image.

### 11.4 How an experimental job runs

`docker run` the image on the VM pointing at an input PDF in S3 → the sequential Python runner from §6 processes one doc → artifacts + `document.json` written back to S3 → container exits. Runs triggered by hand or a shell loop over a folder of test PDFs. No queue, no API, no orchestration. This exercises the whole pipeline and feeds the L4 eval harness.

### 11.5 Explicitly deferred to production (all wrap the core; none change pipeline code)

API Gateway / job-submission endpoint; Step Functions or queue (SQS/Celery); autoscaling; multi-doc concurrency; GPU model server (Triton / vLLM); monitoring & alerting; retry policy beyond the manifest's resume; secrets management beyond the IAM role.

### 11.6 Experimental build sequence

1. Walking skeleton (trivial stages) running **locally on a Mac laptop** via the `dev` build target against LocalStack — proves wiring + storage abstraction + manifest + VLMClient mock. No GPU/CUDA.
2. Same code via the `gpu` build target on **one EC2 VM** against real S3 + real vLLM — proves portability and the AWS path.
3. Spin up the **second EC2 VM** with a different model; run the same fixture PDFs on both.
4. Upgrade stages to real models one at a time (vertical-slice order from §7), validating on fixture PDFs.
5. Run the growing fixture set through both VMs; build the L4 eval harness; compare per-model metrics.

### 11.7 Settled decisions

| Decision | Choice | Rationale |
|---|---|---|
| VLM hosting | Self-hosted | Privacy; batch tolerance makes GPU cost acceptable |
| Model weights | EBS volume (pulled on first boot) | Fast image rebuilds; swap model via env var, no rebuild |
| VMs | Two `g5.xlarge` Spot instances | True parallel A/B comparison across models |
| VLM quantization | 4-bit (bitsandbytes) | Fits two models on a single A10G with headroom for detectors on CPU |
| Figure-type router | VLM call (reuses vLLM sidecar) | No extra model to manage |
| Job ID | UUID | Simple; avoids reprocessing-semantics edge cases of content hash |
| Walking skeleton scope | Trivial stubs first, then vertical slices | Full pipeline wired on day one; isolate failures per stage |
| Build targets | `dev` (slim, multi-arch) + `gpu` (CUDA) from one Dockerfile | Dev on macOS laptop without GPU; same code runs on the GPU VM |
| Local dev machine | macOS laptop (Apple Silicon or Intel) | Skeleton + LocalStack + tests need no GPU; models run only on EC2 |

---

## 12. Open Decisions & Risks

1. ~~**VLM provider**~~ — **settled:** self-hosted, two `g5.xlarge` Spot VMs (Qwen2.5-VL-7B and Gemma 4 12B-it), model weights on EBS volume.
2. **Chart data-point fidelity:** pixel-read values are estimates. Decide how to represent uncertainty to consumers (confidence per series? per point?).
3. ~~**Job ID scheme**~~ — **settled:** UUID.
4. **DPI / image size** trade-off: detection accuracy vs. VLM token cost.
5. **Intermediate retention policy** (keep-forever in dev; TTL in prod).
6. ~~**GPU requirements**~~ — **settled:** YOLO + TATR on CPU; VLM on GPU at 4-bit quantization.

---

## 13. Visual Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT                                       │
│                      source.pdf                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Ingest & Triage                      [PyMuPDF]          │
│                                                                     │
│  ┌─────────────┐    digital? ──► extract text layer (free, exact)  │
│  │  page.pdf   │◄──                                                 │
│  └─────────────┘    scanned? ──► PaddleOCR / Surya (placeholder)  │
│                                                                     │
│  output: pages/p0001.png  +  pages/p0001.words.json                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Layout Detection & Figure Routing   [DocLayout-YOLO]    │
│                                                                     │
│  page image ──► detector ──► typed regions                         │
│                                                                     │
│  title │ paragraph │ list │ table │ figure │ caption │ header ...  │
│                                                                     │
│                        figure crops                                 │
│                            │                                        │
│                            ▼                                        │
│                    ┌───────────────┐                                │
│                    │ Figure Router │  ◄── Qwen2.5-VL call          │
│                    └───┬───────────┘                                │
│                chart ──┤── diagram ──┤── logo ──┤── photo          │
│                                                                     │
│  output: layout/p0001.regions.json  +  regions/p0001_r03.png       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Region Handlers               [parallel per region]     │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐                       │
│   │  text / title /  │  │     TABLE        │                       │
│   │  list / caption  │  │                  │                       │
│   │                  │  │  TATR structure  │                       │
│   │  clip words from │  │  recognition     │                       │
│   │  text layer bbox │  │       +          │                       │
│   │  (no model call) │  │  text-layer fill │                       │
│   │                  │  │                  │                       │
│   │  → {text}        │  │  low confidence? │                       │
│   └──────────────────┘  │  → VLM fallback  │                       │
│                         │                  │                       │
│                         │  → {html table}  │                       │
│                         └──────────────────┘                       │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐                       │
│   │     CHART        │  │  logo / diagram  │                       │
│   │                  │  │  / photo         │                       │
│   │  crop + text     │  │                  │                       │
│   │  layer labels    │  │  crop            │                       │
│   │       │          │  │       │          │                       │
│   │       ▼          │  │       ▼          │                       │
│   │  Qwen2.5-VL      │  │  Qwen2.5-VL     │                       │
│   │  strict JSON     │  │  captioning      │                       │
│   │  schema prompt   │  │  prompt          │                       │
│   │                  │  │                  │                       │
│   │  → {chart JSON}  │  │  → {description} │                       │
│   └──────────────────┘  └──────────────────┘                       │
│                                                                     │
│  output: extracted/p0001_r03.json  (one per region)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Assembly                            [XY-cut order]      │
│                                                                     │
│  sort regions into reading order ──► merge all pages               │
│  normalize bboxes to 0–1 coords  ──► validate schema               │
│                                                                     │
│  output: output/document.json                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                       │
│                     document.json                                   │
│                                                                     │
│  { page_index, chunks: [                                            │
│      { id, type, bbox, reading_order, content, confidence }        │
│  ]}                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

```
INFRASTRUCTURE (docker-compose)

  ┌─────────────────────┐     ┌──────────────────────────┐
  │   pipeline app      │────►│  vLLM                    │
  │   (Python runner)   │     │  Qwen2.5-VL-7B           │
  │                     │◄────│  localhost:8000           │
  └────────┬────────────┘     │  (OpenAI-compatible API) │
           │                  └──────────────────────────┘
           │ read/write
           ▼
  ┌─────────────────────┐
  │  storage.py         │  LocalFS (dev)
  │  abstraction        │  S3 (prod)      jobs/{uuid}/
  └─────────────────────┘                  ├── manifest.json
                                           ├── pages/
                                           ├── layout/
                                           ├── regions/
                                           ├── extracted/
                                           └── output/
```

The manifest sits alongside all artifacts and is the single source of truth for job state — stages check it before doing work, update it atomically when done, giving free resume-on-crash.

---

## 14. Immediate Next Steps

1. Review/approve this design.
2. Settle the two §11.7 decisions: hosted vs. self-hosted VLM, baked vs. pulled models.
3. Generate the **experimental walking skeleton**: storage abstraction (LocalFS + S3), manifest module, stage interface, trivial stage implementations, Dockerfile + docker-compose (with LocalStack), fixture PDF, L1 contract tests, L3 smoke test — runnable locally against LocalStack.
4. Promote the same image to an EC2 VM against real S3 (proves portability).
5. First vertical upgrade: Stage 0 (real PyMuPDF ingest + triage) + golden tests.
