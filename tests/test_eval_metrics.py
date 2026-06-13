"""Unit tests for layout / TEDS / text metrics."""

from __future__ import annotations

from eval.metrics.layout import area_prf, content_coverage_area, coverage_areas, iou, layout_scores
from eval.metrics.table import teds
from eval.metrics.text import text_similarity


def _c(coarse, x0, y0, x1, y1):
    return {"coarse": coarse, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


def test_iou_basic():
    a = {"x0": 0, "y0": 0, "x1": 2, "y1": 2}
    b = {"x0": 1, "y0": 1, "x1": 3, "y1": 3}
    assert abs(iou(a, b) - (1 / 7)) < 1e-6


def test_layout_perfect_match():
    pred = [_c("table", 0, 0, 1, 1), _c("text", 0, 1, 1, 2)]
    ref = [_c("table", 0, 0, 1, 1), _c("text", 0, 1, 1, 2)]
    s = layout_scores(pred, ref, ["table", "text"])
    assert s["micro"]["f1"] == 1.0
    assert s["per_class"]["table"]["recall"] == 1.0


def test_layout_missed_table():
    pred = [_c("text", 0, 1, 1, 2)]                 # table not detected
    ref = [_c("table", 0, 0, 1, 1), _c("text", 0, 1, 1, 2)]
    s = layout_scores(pred, ref, ["table", "text"])
    assert s["per_class"]["table"]["recall"] == 0.0
    assert s["per_class"]["text"]["recall"] == 1.0


def test_coverage_agnostic_to_granularity():
    # One big ref box vs many small pred boxes tiling the same area -> coverage ~1,
    # even though box-match scores 0 (the exact granularity-mismatch case).
    ref = [_c("text", 0.0, 0.0, 1.0, 1.0)]
    pred = [_c("text", 0.0, i / 10, 1.0, (i + 1) / 10) for i in range(10)]
    areas = coverage_areas(pred, ref, ["text"])
    prf = area_prf(*areas["text"])
    assert prf["recall"] > 0.98 and prf["precision"] > 0.98

    # box-match, by contrast, fails on the same input
    assert layout_scores(pred, ref, ["text"])["per_class"]["text"]["f1"] < 0.3


def test_content_coverage_ignores_class_disagreement():
    # Spec-sheet case: we call it a table, reference calls it text, same area.
    # Per-class layout scores 0 on both classes, but content coverage ~1.0 —
    # localization is fine; only the class label differs.
    pred = [_c("table", 0.1, 0.1, 0.9, 0.9)]
    ref = [_c("text", 0.1, 0.1, 0.9, 0.9)]
    cc = area_prf(*content_coverage_area(pred, ref))
    assert cc["f1"] > 0.98
    ls = layout_scores(pred, ref, ["table", "text"])
    assert ls["micro"]["f1"] == 0.0


def test_teds_identical():
    h = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert teds(h, h) == 1.0


def test_teds_content_diff_less_than_one():
    a = "<table><tr><td>hello</td></tr></table>"
    b = "<table><tr><td>world</td></tr></table>"
    score = teds(a, b)
    assert 0.0 < score < 1.0  # same structure, different cell text


def test_teds_structure_diff():
    a = "<table><tr><td>a</td><td>b</td></tr></table>"
    b = "<table><tr><td>a</td></tr><tr><td>b</td></tr></table>"
    assert teds(a, b) < 1.0


def test_text_similarity():
    assert text_similarity("Hello  world", "Hello world") == 1.0  # whitespace-normalized
    assert text_similarity("", "") == 1.0
    assert 0.0 <= text_similarity("abc", "xyz") < 1.0
