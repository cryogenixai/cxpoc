# Evaluation Harness & Gold Set — Design Notes

**Status:** Draft v0.1 · **Date:** 2026-06-13
**Scope of this doc:** How we measure extraction accuracy — gold-set construction, annotation workflow, and the per-component + end-to-end eval design. Companion to `document-parsing-pipeline-design-v1.1.md`. Motivated by the open question of how accurate DocLayout-YOLO and TATR actually are on our corpus, which can't be answered without a measured baseline.

---

## 1. Why a gold set

We cannot decide whether DocLayout-YOLO / TATR are "good enough," nor compare alternatives (other table models, a VLM-first path, commercial APIs), without a measured accuracy number on **our** documents. The gold set is that yardstick. It is also the input that defines what "accurate" means and which pages we send to reference systems.

The gold set is a **living, frozen, versioned artifact**: committed to git, held out, never casually tuned against, and grown whenever a new failure mode is found in dev or production.

---

## 2. Silver vs. gold: how Landing.ai fits

Landing.ai's Agentic Document Extraction is accurate on our hardest cases (complex/borderless financial tables) and returns structured chunks + markdown with bounding-box grounding, which maps onto our Block IR.

**Caveat — it is a silver standard, not gold.** Scoring our pipeline directly against Landing.ai output measures *agreement with Landing.ai*, not accuracy. Consequences:

- We inherit its errors — a correct pipeline looks "wrong" wherever Landing.ai is wrong.
- Ceiling effect: our measured score can never exceed Landing.ai's true accuracy.
- Circularity if Landing.ai is ever also a production fallback.

**Resolution — use it as a labeling accelerator, not as truth:**

1. Treat Landing.ai output as **pre-annotation a human corrects** (model-assisted labeling). Editing near-correct silver → gold is far cheaper than annotating from scratch.
2. **Adjudicate the hardest ~10–15 pages by hand** to estimate Landing.ai's *own* error rate on our docs, so we know how far to trust the silver labels elsewhere.
3. **Cross-check table values against the text layer** (bbox intersection, same reconciliation Stage 2 already does) — free validation that auto-flags suspect cells.
4. **Tier trust:** silver-as-is where it reconciles cleanly; human-corrected gold on the hard / high-disagreement pages.

Bonus: run the same pages through Landing.ai **and** Textract → two commercial reference points bracketing the self-hosted pipeline, a stronger basis for the build-vs-buy call. A highly accurate Landing.ai is also a candidate for the escalation tier of a confidence-based fallback ladder.

> **Data-egress note:** routing docs through a third-party API is fine for the current public corpus (Tesla 10-Ks, Dell manuals). Revisit if sensitive docs enter the corpus.

---

## 3. Gold-set composition

Two reframes drive everything:

- **Pages, not documents.** The unit of evaluation is a page. Annotating a whole 100-page 10-K is wasteful (layouts repeat). Curate ~50–80 representative pages drawn from *many* documents.
- **Stratify by failure axis; do not randomly sample.** Random sampling over-represents the easy majority and under-samples the hard tail — exactly where DocLayout/TATR break. Deliberately over-sample hard cases relative to natural frequency, so each failure mode and each element type has enough examples for a stable metric.

### Selection matrix (mapped to corpus)

**From `10k-pdf` (financial — hardest table cases):**

| Cell | What it tests |
|---|---|
| Financial statements (balance sheet, income, cash flow) | Borderless tables, spanning headers, numeric columns — TATR worst case; weight heavily |
| Footnotes / notes pages | Dense nested tables + dense narrative |
| MD&A narrative | Multi-column-ish prose; layout + reading order |
| Cover / TOC | Simple regression baseline |

**From `dell-pdf` (technical manuals — figure/layout cases):**

| Cell | What it tests |
|---|---|
| Spec tables | Simpler bordered tables (TATR baseline) |
| Diagrams + callouts | Figure detection + caption association |
| Mixed text+image+list | Region separation + reading order |
| Dense procedural pages | List/step structure |

**Plus deliberate edge/adversarial pages:** rotated/landscape tables, near-empty pages, very dense pages.

**Skip for now:** scanned pages (born-digital today). Reserve 1–2 to sanity-check Stage 0 triage; no further budget until the scanned path is real.

### Sizing

**~50–80 pages**, stratified. A stratified 60 beats a random 300 — the random set is mostly near-identical easy pages.

### Variety constraint

**Cap pages per source document (max ~3–4).** This is the key mechanism that forces layout diversity — without it, ranking surfaces many near-identical pages from the one densest doc, testing a single template instead of variety. Spread across many companies/products.

---

## 4. Selection & extraction method

A **manifest-driven extraction pipeline** (mirrors the existing architecture: manifest = system of record, single-page artifacts derived/reproducible from it).

1. **Candidate generation (automated).** Run Stage 0/1 over the *entire* corpus; emit a per-page feature table:
   `doc_id, page, page_type_guess, n_table_regions, text_density, mean_detector_conf, has_figure, n_columns, is_borderless_hint`.
   Most of these signals already exist.
2. **Stratified shortlisting.** Rank pages into matrix cells; surface ~2× target per cell; **enforce the per-doc cap**. Filter out cross-page-table pages (see cautions).
3. **Human pick.** Person selects from the shortlist to fill cells evenly, tracking a coverage table (cell → count). Keep a candidate pool of ~100–120 to allow swaps.
4. **Materialize** each selected `(doc_id, page)` as a self-contained unit:
   ```
   gold/{doc_id}__p{page:04d}/
     page.pdf        # single-page PDF — ideal for per-page Landing.ai/Textract calls
     page.png        # 200-DPI Stage 0 render
     textlayer.json  # words + bbox, for the value cross-check
   ```
   (PyMuPDF/pikepdf — already in the stack.)
5. **Freeze.** Write the gold manifest, one record per page, commit to git:
   `{ gold_id, doc_id, source_page, stratum, selection_reason, checksum }`.
   Gold *labels* attach later by `gold_id`, decoupled from selection order.

### Cautions

- **`gold_id` must be stable and order-independent** (encode provenance, e.g. `doc__pNNNN`; never a sequential index) — re-running selection must not orphan attached labels. Same content-derived-ID principle used for chunks.
- **Cross-page tables and running headers/footers lose context** when a page is isolated; a continuing table looks truncated. Exclude such pages or tag `known_truncation` so the metric doesn't penalize correct behavior.

---

## 5. Annotation workflow

**Do not build a general-purpose annotation tool** for a one-off ~60-page set — it's a yak-shave. The task is *adjudicating pre-labels*, not annotating from scratch.

- **Default: Label Studio + pre-annotations.** Supports bbox + region labels + relations and imports model predictions as pre-annotations, so annotators correct Landing.ai output rather than starting blank. Table HTML for TEDS can be corrected in a side-by-side / spreadsheet view.
- **Build a thin viewer only if** (a) gold-set labeling becomes a continuous operation, or (b) inline text-layer reconciliation (highlight cells where model value ≠ text-layer digits) would meaningfully speed adjudication. If so: a **time-boxed ~1-day** Streamlit/Gradio viewer over existing page renders + structured output + reconciliation flags, with accept/reject/edit into the IR. Not a product.

**Annotate everything once.** Each gold page is labeled with all fields — layout regions, table HTML, reading order, chart values, captions — so the same set feeds every component eval and the e2e eval. (Reason the per-element stratification matters: each component metric needs enough examples to be stable.)

---

## 6. Eval design

Per-component evals are non-negotiable: a single aggregate "accuracy" number says the pipeline is broken without saying *where*. Each component fails differently and needs its own metric.

### Per-component metrics

| Component | Metric | Notes |
|---|---|---|
| Layout detection | Region precision/recall/mAP, **per class** | Table recall ≠ figure recall |
| Reading order (assembly) | Kendall-τ / permutation distance vs gold order | Cheap, often missed, breaks RAG silently |
| Text region | CER/WER (scanned); near-exact for born-digital | Clipping text-layer words ≈ free for digital |
| Table | **TEDS / GriTS** (structure) + **cell-value accuracy** | Split structure from values |
| Chart | Numeric value error + series/label match | Against gold JSON |
| Image/diagram | Caption quality (LLM-judge / embedding sim vs reference) | Inherently fuzzy; small sample |

Metrics are not comparable across components — keep separate scorecards, not one average.

### Isolated vs. in-situ (key refinement)

Eval each component **twice**:

- **Isolated** — fed the *gold* upstream input (e.g. TATR given the correct table crop) → the component's own error alone.
- **In-situ** — fed the *real* upstream output (table structure from the raw page) → includes the cascade (e.g. layout's table-detection recall).

The **gap between the two is the error the upstream stage injects** — this is what tells us whether to invest in layout vs. table structure.

### End-to-end eval (keep exactly one)

Component metrics can all look green while composition fails (scrambled reading order, mis-nested sections, a correct table assembled in the wrong place); conversely a component can fail its metric without mattering downstream. Pair the component suite with **one extrinsic eval tied to the real goal** — retrieval/answer quality over the resulting chunks. Diagnose with component evals; decide ship/no-ship with the e2e eval.

### Effort & maturity

- **Weight effort by corpus value.** 10-Ks are tables-and-text; build the table eval rigorously (TEDS + value cross-check), keep image-caption eval to a small LLM-judged spot-check. But deliberately seed chart/figure pages (Dell manuals) or those components get no signal at all.
- **Metric maturity varies.** Tables (TEDS) and layout (mAP) are well-defined and automatable; captions and chart-semantics are fuzzy — use LLM-judge / human spot-check on smaller samples and don't treat those numbers as equally solid.

---

## 7. Recommended sequence

1. Mine candidate pages from the corpus using Stage 0/1 confidence + region counts + per-doc cap → ranked shortlist CSV.
2. Curate a stratified ~60-page set across the matrix; materialize single-page units; freeze the manifest.
3. Run Landing.ai (and Textract) on those pages.
4. Adjudicate hard / high-disagreement pages to gold in Label Studio (pre-annotations + text-layer value cross-check).
5. Build the eval suite: per-component metrics (isolated + in-situ) + one e2e retrieval eval.

## 8. Open questions

- Exact Landing.ai ADE response schema → needed to spec the IR mapper and Label Studio pre-annotation import format.
- Embedding model choice (drives chunk token budgets) — interacts with the e2e retrieval eval.
- Final gold-set size and per-doc cap values.
- Whether the thin reconciliation viewer is warranted (deferred until after the first labeling pass).
