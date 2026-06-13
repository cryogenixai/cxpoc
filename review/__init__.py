"""cxpoc review — a local, read-only human-in-the-loop tool.

Renders the original PDF (PDF.js) with the extracted regions from a job's
document.json overlaid as color-coded boxes, so a human can eyeball parse
quality against the source. Independent of how the regions were produced
(stub or real detector). See design §7 (failure gallery) and §10 (review queue).
"""
