"""Unit tests for layout / TEDS / text metrics."""

from __future__ import annotations

from eval.metrics.layout import iou, layout_scores
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
