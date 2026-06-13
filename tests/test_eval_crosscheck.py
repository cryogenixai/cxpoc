"""Unit test for the table grid word-coverage function (no model)."""

from __future__ import annotations

from eval.crosscheck import word_coverage


def _word(cx, cy):
    return {"bbox": {"x0": cx - 1, "y0": cy - 1, "x1": cx + 1, "y1": cy + 1}}


def test_coverage_counts_only_in_region_and_in_cell():
    region = {"x0": 100, "y0": 100, "x1": 300, "y1": 300}
    # Two cells in CROP coords (region origin = 100,100):
    #   cell A covers crop (0..100, 0..100) -> page (100..200, 100..200)
    #   cell B covers crop (100..200, 0..100) -> page (200..300, 100..200)
    cells = [[0, 0, 100, 100], [100, 0, 200, 100]]
    words = [
        _word(150, 150),   # in region, in cell A      -> captured
        _word(250, 150),   # in region, in cell B      -> captured
        _word(150, 250),   # in region, below both cells -> dropped (missed row)
        _word(500, 150),   # outside region            -> ignored
    ]
    captured, total = word_coverage(cells, words, region)
    assert total == 3          # the off-page word is excluded
    assert captured == 2       # the third word fell outside the grid
