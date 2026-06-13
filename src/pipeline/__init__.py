"""cxpoc — document parsing pipeline (PoC walking skeleton).

See design/document-parsing-pipeline-design-v1.1.md. Stages are pure functions
of the artifact store: input keys -> output keys, state in the manifest. This
skeleton wires all four stages with trivial implementations and a mock VLM so
the whole pipeline runs end-to-end on a laptop before any real model lands.
"""

__version__ = "0.1.0"
