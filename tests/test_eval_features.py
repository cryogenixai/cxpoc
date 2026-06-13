"""Unit tests for eval feature extraction + stratification (no models)."""

from __future__ import annotations

from eval.features import estimate_columns, guess_page_type, page_features
from eval.stratify import stratify


def _region(rtype, x0, y0, x1, y1, conf=0.9):
    return {"type": rtype, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "detector_confidence": conf}


def test_estimate_columns_single_full_width():
    regions = [_region("paragraph", 50, 100, 950, 300)]
    assert estimate_columns(regions, page_w=1000) == 1


def test_estimate_columns_two():
    regions = [
        _region("paragraph", 50, 100, 400, 300),    # narrow, left
        _region("paragraph", 550, 100, 900, 300),   # narrow, right
    ]
    assert estimate_columns(regions, page_w=1000) == 2


def test_guess_type_financial_vs_spec():
    assert guess_page_type("10k-pdf", n_table=1, has_figure=False, has_toc=False,
                           word_count=200, n_regions=8, text_density=0.3) == "financial_table"
    assert guess_page_type("dell-pdf", n_table=1, has_figure=False, has_toc=False,
                           word_count=200, n_regions=8, text_density=0.3) == "spec_table"


def test_guess_type_near_empty_and_toc():
    assert guess_page_type("10k-pdf", 0, False, False, 5, 1, 0.0) == "near_empty"
    assert guess_page_type("10k-pdf", 0, False, True, 200, 10, 0.2) == "cover_toc"


def test_features_table_signal_not_swallowed_by_density():
    # A dense page WITH a table stays financial_table (not very_dense).
    regions = [_region("table", 50, 400, 950, 800)] + [
        _region("paragraph", 50, 100 + i * 10, 950, 108 + i * 10) for i in range(22)
    ]
    page = {"page_index": 3, "width_px": 1000, "height_px": 2000, "page_kind": "digital"}
    feat = page_features("TSLA-2024", "10k-pdf", page, regions, words=[{"text": "x"}] * 300)
    assert feat["page_type_guess"] == "financial_table"
    assert feat["n_table_regions"] == 1
    assert feat["is_borderless_hint"] == 1


def test_stratify_respects_per_doc_cap():
    # 10 financial pages from each of 2 docs; cap=3 -> at most 3 per doc.
    rows = []
    for doc in ("A", "B"):
        for p in range(10):
            rows.append({
                "gold_id": f"{doc}__p{p:04d}", "doc_id": doc, "source": "10k-pdf",
                "page": p, "page_type_guess": "financial_table", "n_table_regions": "2",
                "has_figure": "0", "n_columns": "1", "text_density": "0.3",
                "mean_detector_conf": "0.6", "is_borderless_hint": "1",
                "landscape": "0", "page_kind": "digital",
            })
    selected, coverage = stratify(rows, cap=3, oversample=2)
    from collections import Counter
    per_doc = Counter(r["doc_id"] for r in selected)
    assert all(v <= 3 for v in per_doc.values())
    assert all(r["stratum"] == "financial_table" for r in selected)
