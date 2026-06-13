"""Layout detectors. Each returns *raw typed regions* for one page image:
``[{"type", "bbox", "detector_confidence"}, ...]`` in the image's pixel coords.
The layout stage assigns region ids, runs the figure router, and writes crops.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Protocol

DEFAULT_WEIGHTS = "models/doclayout_yolo_docstructbench_imgsz1024.pt"
_HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
_HF_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"

# DocLayout-YOLO (DocStructBench) class name -> our schema region type.
_CLASS_MAP = {
    "title": "title",
    "plain text": "paragraph",
    "abandon": "page_footer",      # headers/footers/page numbers
    "figure": "figure",            # refined by the router in the stage
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "table_footnote": "footnote",
    "isolate_formula": "formula",
    "formula_caption": "caption",
}


class LayoutDetector(Protocol):
    name: str

    def detect(self, image_bytes: bytes, page_index: int) -> list[dict[str, Any]]:
        ...


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        return im.size  # (w, h)


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / (area_a + area_b - inter)


def _dedup(regions: list[dict[str, Any]], iou_thresh: float = 0.5) -> list[dict[str, Any]]:
    """Class-agnostic greedy suppression: when two regions overlap heavily, keep
    the higher-confidence one. Removes the model's competing class hypotheses for
    the same area (e.g. paragraph vs. title on one block)."""
    ordered = sorted(regions, key=lambda r: r["detector_confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for r in ordered:
        if all(_iou(r["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(r)
    return kept


class StubLayoutDetector:
    """Fixed positional bands as fractions of the page — no model. The bands
    cover every handler route (title/paragraph/table/chart/figure) so the
    downstream fabric is exercised; the figure band is refined by the router.
    Used for fast, model-free contract tests.
    """

    name = "stub-layout"

    _TEMPLATE = [
        ("title",     (0.00, 0.00, 1.00, 0.15)),
        ("paragraph", (0.00, 0.15, 1.00, 0.45)),
        ("table",     (0.00, 0.45, 1.00, 0.62)),
        ("chart",     (0.00, 0.62, 1.00, 0.85)),
        ("figure",    (0.00, 0.85, 1.00, 1.00)),
    ]

    def detect(self, image_bytes: bytes, page_index: int) -> list[dict[str, Any]]:
        w, h = _image_size(image_bytes)
        regions = []
        for rtype, (fx0, fy0, fx1, fy1) in self._TEMPLATE:
            regions.append({
                "type": rtype,
                "bbox": {"x0": fx0 * w, "y0": fy0 * h, "x1": fx1 * w, "y1": fy1 * h},
                "detector_confidence": 1.0,
            })
        return regions


class DocLayoutYOLODetector:
    """Real DocLayout-YOLO detection. torch/doclayout_yolo are imported lazily so
    this module imports fine in a core-only environment. Runs on CPU by default
    (acceptable for batch, design §12.6); set CXPOC_DETECT_DEVICE to override.
    """

    name = "doclayout-yolo"

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: str | None = None,
        imgsz: int = 1024,
        conf: float = 0.2,
    ):
        self.weights = Path(weights)
        self.device = device or os.environ.get("CXPOC_DETECT_DEVICE", "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self._model = None  # lazy

    @staticmethod
    def ensure_weights(weights: str | Path = DEFAULT_WEIGHTS) -> Path:
        """Download the detector weights to ``weights`` if absent."""
        path = Path(weights)
        if path.exists():
            return path
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=_HF_REPO, filename=_HF_FILE, local_dir=str(path.parent)
        )
        return Path(downloaded)

    def _load(self):
        if self._model is None:
            from doclayout_yolo import YOLOv10

            self.ensure_weights(self.weights)
            self._model = YOLOv10(str(self.weights))
        return self._model

    def detect(self, image_bytes: bytes, page_index: int) -> list[dict[str, Any]]:
        from PIL import Image

        model = self._load()
        with Image.open(io.BytesIO(image_bytes)) as im:
            img = im.convert("RGB")
            results = model.predict(
                img, imgsz=self.imgsz, conf=self.conf,
                device=self.device, verbose=False,
            )

        res = results[0]
        names = res.names
        regions = []
        for box, cls, score in zip(
            res.boxes.xyxy.tolist(),
            res.boxes.cls.tolist(),
            res.boxes.conf.tolist(),
        ):
            raw_name = names[int(cls)]
            x0, y0, x1, y1 = box
            regions.append({
                "type": _CLASS_MAP.get(raw_name, raw_name),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "detector_confidence": float(score),
            })
        # Drop competing overlapping hypotheses, then order top-to-bottom,
        # left-to-right — a stable order for downstream region ids.
        regions = _dedup(regions)
        regions.sort(key=lambda r: (r["bbox"]["y0"], r["bbox"]["x0"]))
        return regions


def make_detector(name: str | None = None) -> LayoutDetector:
    """Resolve a detector by name (env CXPOC_LAYOUT_DETECTOR, default doclayout)."""
    name = name or os.environ.get("CXPOC_LAYOUT_DETECTOR", "doclayout")
    if name in ("stub", "stub-layout"):
        return StubLayoutDetector()
    if name in ("doclayout", "doclayout-yolo"):
        return DocLayoutYOLODetector()
    raise ValueError(f"unknown layout detector: {name!r}")
