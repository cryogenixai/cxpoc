# cxpoc — Document Parsing Pipeline (PoC)

Hybrid PDF → structured-JSON pipeline. Specialized detectors localize (bounding
boxes); vision-language models interpret (charts, diagrams, table fallback).
Everything is grounded back to page coordinates.

Full design: [`design/document-parsing-pipeline-design-v1.1.md`](design/document-parsing-pipeline-design-v1.1.md).

## Status: walking skeleton

All four stages are wired with **trivial implementations** and a **mock VLM**, so
the whole pipeline runs end-to-end on a laptop before any real model lands. Real
models replace the stubs one vertical slice at a time (design §7).

```
ingest → layout → extract → assemble → output/document.json
```

| Stage | Skeleton behavior | Real (later) |
|-------|-------------------|--------------|
| ingest | one placeholder page + fixed words | PyMuPDF render + text-layer + triage |
| layout | fixed regions covering every route | DocLayout-YOLO + figure router |
| extract | text echoes words; table/chart/image stubbed/mock | TATR, schema-prompted VLM |
| assemble | XY-cut sort, normalize bbox, validate schema | learned reading order |

## Layout

```
src/pipeline/
  storage.py    # LocalFS | S3 backends, read(key)/write(key)  (§11.3)
  manifest.py   # job state machine; atomic, idempotent, resumable  (§5)
  jobctx.py     # JobContext: the single arg every stage receives
  schema.py     # output JSON schema (§4) + inter-stage records
  vlm.py        # VLMClient — the one seam for model swaps; mock:// in dev  (§11.2)
  stages/       # ingest, layout, extract, assemble  (§3)
  handlers/     # text, table, chart, image  (§2)
  run.py        # CLI + sequential runner  (§6)
tests/          # L1 contract + L3 smoke
```

## Run it (local, no GPU)

```bash
# In the project venv (boto3, jsonschema, pytest installed):
pip install -e ".[dev]"

# Tests (mock VLM, no models):
pytest -q

# End-to-end on the fixture PDF, artifacts to ./_jobs:
python -m pipeline.run --input tests/fixtures/sample.pdf --store file://./_jobs
```

Output lands at `_jobs/jobs/<job_id>/output/document.json` (schema-valid).

## Build targets (design §11.2)

One codebase, two Docker targets:

- **`dev`** — `python:3.12-slim`, multi-arch. Laptop + LocalStack + tests. No CUDA.
- **`gpu`** — CUDA base. EC2 `g5.xlarge` + vLLM sidecar; model weights mounted
  from an EBS volume. Swap models (Qwen ↔ Gemma 4) via the `VLM_MODEL` env var.

```bash
docker compose up -d localstack            # local S3 emulation
docker compose run --rm pipeline --input /work/tests/fixtures/sample.pdf --store s3://cxpoc-jobs
```

## Model experimentation

`vlm.py` is the only place model choice lives. Two env vars (`VLM_BASE_URL`,
`VLM_MODEL`) select the model; `mock://` returns canned responses for local dev.
Two `g5.xlarge` Spot VMs (one per model) enable parallel A/B comparison on the
same PDFs (design §11.1).
