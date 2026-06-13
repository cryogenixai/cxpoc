"""cxpoc eval — gold-set construction + evaluation harness.

See design/eval-harness-and-gold-set-v0.1.md. Separate from the pipeline package
(like review/); imports pipeline.* to reuse Stage 0/1 and storage. Builds the
stratified gold set (mine -> stratify -> materialize -> freeze) and the
per-component + e2e eval suite.
"""
