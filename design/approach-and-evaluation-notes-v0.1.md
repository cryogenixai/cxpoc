# Approach & Evaluation — Reference Notes

**Status:** Draft v0.1 · **Date:** 2026-06-13
**Scope of this doc:** What two reference sources tell us about our extraction approach and how to evaluate it — (1) the **HtmlRAG** paper (WWW '25), and (2) **Pryon's production "PDF Processing" deck** (GA, © 2024). Captures architectural takeaways, the metric definitions we'll use, and a realistic accuracy baseline. Companion to `document-parsing-pipeline-design-v1.1.md` and `eval-harness-and-gold-set-v0.1.md`.

---

## 1. Sources reviewed

| Source | What it is | Why it matters |
|---|---|---|
| HtmlRAG (Tan et al., WWW '25, `arXiv:2411.02959`) | RAG paper: HTML > plain text as the format of retrieved knowledge; clean + block-tree pruning | Validates structure-preserving representation and informs chunk granularity |
| Pryon "PDF Processing" deck (GA, 2024) | The existing in-house production pipeline | Prior art from our own shop: architecture confirmation + a real accuracy baseline + a mature label taxonomy |

---

## 2. Metric definitions (so numbers are comparable)

**mAP — mean Average Precision.** The standard accuracy metric for object/layout **detection**.

- **IoU (Intersection over Union):** overlap between a predicted box and ground-truth box. A detection counts as correct when IoU ≥ a threshold (commonly 0.5).
- **AP (Average Precision):** area under the precision–recall curve for one class (sweeping the confidence threshold). One number per class.
- **mAP:** mean of the per-class APs → one overall score, 0–1 (higher better).
- **IoU-threshold caveat:** "mAP@0.5" uses a single 0.5 cutoff; the stricter COCO "mAP@[.5:.95]" averages IoU 0.5→0.95 and yields lower numbers. **Not directly comparable across thresholds** — always note which when comparing.

**Detection ≠ structure.** mAP answers "did we find and localize the region?" It is *separate* from:

- **TEDS / GriTS** — table **structure** correctness (rows/cols/spans) *after* the table is found.
- A high table-detection mAP (e.g. 0.92) says nothing about whether the cells/spans were parsed correctly. Keep these metrics distinct in our eval (see `eval-harness-and-gold-set-v0.1.md` §6).

---

## 3. Pryon production pipeline — what it shows

**Architecture (Argo workflow, per-page, "high mem"):** `download_content → pdf_split (1 page) →` two parallel fan-outs — a **text path** (`pdf2text` ×N → in-memory capture) and an **image path** (`pdf2image` ×N → vision/section model (GPU) + vision text-detection model) — converging at `merge process (ocr) → predom`. Display layer: **Retrieval + Display View Creation**.

**Vision Segmentation label set (18 classes), trained on ~40,000 in-house annotated pages:**
Paragraph · Heading 1 · Heading 2 · Heading Other · Bulleted List · Numbered List · Page Number · Header/Footer · Title/Subtitle · Table of Contents · Table · Image · Caption (image/table) · Side box · Other · **Paragraph Continued · Bulleted Continued · Numbered Continued**.

**Accuracy — overall mAP = 51.45%.** Per-class AP:

| Strong | AP | Weak tail | AP |
|---|---|---|---|
| Table | 0.92 | Other | 0.29 |
| ToC | 0.91 | Footnote | 0.22 |
| List | 0.83 | Endnote | 0.12 |
| Heading | 0.81 | Copyright | ~0.01 |
| Paragraph | 0.74 | (side_box 0.41, header_footer 0.43, caption 0.45, references 0.45, icon 0.48, image 0.50) | |
| Figure | 0.68 | | |

**Stated next steps:** more data for weak classes; **HTML/Word/other formats**; **LayoutLM (joint visual + language)** for segmentation; finer-grained table/list analysis.

**Other notes:** text detection + OCR goes *inside* figures/scans; "coordinates of every text bit → answer highlighting" is called out as a well-loved user feature.

---

## 4. Implications for our approach

1. **The hybrid + bbox architecture is validated by production.** Pryon's text-path/vision-path/merge skeleton mirrors our Stage 0/1/2/3, and bbox-grounded answer highlighting is a proven user win — direct support for R1 (bbox on every element) and our provenance-first IR. Our manifest-driven "pure functions of files" is the cleaner expression of their Argo DAG.
2. **Enrich the Block IR taxonomy toward Pryon's label set** — especially:
   - **Continuation classes** (`Paragraph/Bulleted/Numbered Continued`): a first-class fix for the cross-page / cross-column continuation problem flagged in the gold-set doc.
   - **Boilerplate/structure classes** (`Header/Footer`, `Page Number`, `ToC`, `Caption`, `Side box`): needed for clean reading-order stitching and chunk hygiene.
3. **Plan for text+visual fusion, not pure-vision layout.** Pryon's pure-vision segmentation plateaus at ~51% mAP and their own next step is LayoutLM (multimodal). We already extract the text layer — fuse it into layout decisions, or at minimum extend reconciliation beyond tables to correct pure-vision output. This is the cheap version of their LayoutLM ambition.
4. **Our born-digital triage + text-layer-for-values is a deliberate improvement.** Pryon OCRs everything (`merge process (ocr)`, high mem). For born-digital financial PDFs that is heavier and risks digit errors; our triage preserves exact numbers (values-vs-structure principle).
5. **Structure-preserving representation (HtmlRAG):** keep structure-rich blocks as **HTML, not Markdown** (HtmlRAG shows HTML > Markdown, esp. for complex tables — our hardest 10-K content). Markdown is fine for prose. We already keep tables as HTML; make it a documented, deliberate choice.
6. **Query-time HTML pruning (HtmlRAG) is NOT needed for the PoC.** It solves long-context compression of whole retrieved web docs — a problem our pre-chunk-and-index design avoids. Note as a possible future post-retrieval refiner only.

## 5. Implications for evaluation

1. **Per-class detection metrics are the right frame** — Pryon evaluates exactly this way. Our per-component/per-class eval plan is confirmed.
2. **Use mAP 51.45% as the realistic production baseline / reference line.** A GA system trained on 40K annotated pages tops out here, with a brutal long tail. This:
   - sets expectations (don't expect off-the-shelf DocLayout-YOLO to beat it), and
   - **justifies the confidence-based fallback ladder / VLM escalation.**
3. **Keep detection (mAP) and structure (TEDS) metrics distinct.** Pryon's table 0.92 is region detection, not structure correctness. Our table eval must report both, on the gold set.
4. **Embedding granularity has a floor *and* a ceiling (HtmlRAG + our discussion).** Too-large chunks compress lossily; too-small blocks give vague embeddings. Target ~256 words (~300–350 tokens) for the embedding unit and use small-to-big expansion for generation context — refines the earlier "embed small" advice.
5. **End-to-end eval is the right top-level measure.** Both references evaluate downstream task quality (HtmlRAG: QA EM/Hit@1/ROUGE; Pryon: the highlight/display feature). Confirms keeping one end-to-end retrieval eval above the component evals.

## 6. Open items carried forward

- Confirm whether Pryon's mAP is @0.5 or @[.5:.95] before using it as a comparison baseline.
- Decide final enriched Block IR taxonomy (which Pryon classes to adopt vs. collapse).
- Whether to fuse text-layer into Stage 1 layout (cheap LayoutLM-style) or keep it to reconciliation only.
