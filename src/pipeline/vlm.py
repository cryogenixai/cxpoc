"""VLMClient — the single seam for model experimentation (design §11.2).

Every handler that needs a vision-language model calls this client; it reads
``VLM_BASE_URL`` and ``VLM_MODEL`` from the environment. Swapping models (Qwen
vs. Gemma 4, hosted vs. self-hosted) is a config change, never a code change.

Two modes:
  * ``mock://``   — returns deterministic canned responses. Used on the laptop
                    dev target and in tests; no model server required.
  * everything else — an OpenAI-compatible endpoint (the vLLM sidecar on the
                    EC2 VM). The real-call paths are stubbed here with a clear
                    NotImplementedError; they are filled in during the Stage 2
                    vertical slices (§7), which is the first time a real model
                    actually runs.

The high-level methods (classify_figure / extract_chart / describe_image) are
what handlers call, so handler code never deals with prompts or transport.
"""

from __future__ import annotations

import os
from typing import Any


class VLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or os.environ.get("VLM_BASE_URL", "mock://")
        self.model = model or os.environ.get("VLM_MODEL", "mock")
        self.is_mock = self.base_url.startswith("mock://")

    # -- High-level tasks used by handlers / the figure router ---------------

    def classify_figure(self, image_bytes: bytes) -> str:
        """Figure-type router (§Stage 1): chart | diagram | logo | photo."""
        if self.is_mock:
            return "diagram"
        raise NotImplementedError("real figure router lands in the Stage 1 slice")

    def extract_chart(self, image_bytes: bytes, words: list[dict[str, Any]]) -> dict[str, Any]:
        """Chart handler (§R3): strict-schema chart extraction."""
        if self.is_mock:
            return {
                "chart_type": "bar",
                "title": "[mock chart]",
                "x_axis": {"label": "", "ticks": []},
                "y_axis": {"label": "", "ticks": []},
                "legend": [],
                "series": [],
                "confidence": 0.5,
            }
        raise NotImplementedError("real chart extraction lands in the Stage 2 slice")

    def describe_image(self, image_bytes: bytes) -> str:
        """Logo/diagram/photo handler (§R2): structured caption."""
        if self.is_mock:
            return "[mock description]"
        raise NotImplementedError("real image captioning lands in the Stage 2 slice")

    def extract_table(self, image_bytes: bytes) -> str:
        """Table handler VLM fallback (§R4): emit HTML directly from the crop."""
        if self.is_mock:
            return "<table><tr><td>[mock cell]</td></tr></table>"
        raise NotImplementedError("real table VLM fallback lands in the Stage 2 slice")

    @property
    def model_id(self) -> str:
        return self.model
