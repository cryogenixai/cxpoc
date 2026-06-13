"""Unit tests for the classification-refinement helpers (pure, model-free)."""

from __future__ import annotations

import pytest

from pipeline.refine import (
    body_font_size,
    clip_words,
    heading_level,
    is_continuation,
    is_page_number,
    list_ordered,
    refine_region,
    representative_size,
)


def _word(text, x0, y0, x1, y1, size=10.0, bold=False):
    return {"text": text, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "size": size, "bold": bold}


# --- body_font_size / heading_level ------------------------------------------

def test_body_font_size_is_modal_body_not_headings():
    # 20 body words at 10pt, a few heading words at 18/14pt -> body is 10.
    sizes = [10.0] * 20 + [18.0, 18.0, 14.0]
    assert body_font_size(sizes) == 10.0


def test_body_font_size_ties_pick_smaller():
    assert body_font_size([10.0, 10.0, 16.0, 16.0]) == 10.0


def test_body_font_size_empty():
    assert body_font_size([]) == 0.0


@pytest.mark.parametrize("size,expected", [
    (20.0, 1),   # 2.0x -> H1
    (15.0, 1),   # 1.5x -> H1
    (12.5, 2),   # 1.25x -> H2
    (11.0, 3),   # 1.1x -> H3
    (10.0, None),  # body
    (9.0, None),   # smaller than body
])
def test_heading_level_ratios(size, expected):
    assert heading_level(size, body_size=10.0) == expected


def test_heading_level_bold_inline_promotes_to_h3():
    assert heading_level(10.0, body_size=10.0, bold=True) == 3
    assert heading_level(10.0, body_size=10.0, bold=False) is None


def test_heading_level_bold_same_size_float_noise():
    # Real 10-K case: bold section header reports 9.9975 against a body rounded to
    # 10.0 -> ratio just under 1.0. Must still be H3 (regression for the >= 1.0 bug).
    assert heading_level(9.9975, body_size=10.0, bold=True) == 3


def test_heading_level_bold_does_not_rescue_clearly_smaller():
    # Bold but genuinely smaller than body (e.g. bold footnote) is not a heading.
    assert heading_level(8.0, body_size=10.0, bold=True) is None


def test_heading_level_guards_zero():
    assert heading_level(12.0, body_size=0.0) is None
    assert heading_level(0.0, body_size=10.0) is None


def test_representative_size_median():
    assert representative_size([10.0, 12.0, 14.0]) == 12.0
    assert representative_size([]) == 0.0


# --- list_ordered ------------------------------------------------------------

@pytest.mark.parametrize("text", ["1. First item", "2) second", "(3) third", "a. alpha", "iv. roman", "IV) ROMAN"])
def test_list_ordered_true(text):
    assert list_ordered(text) is True


@pytest.mark.parametrize("text", ["• bullet", "- dash item", "* star", "◦ sub-bullet", "– en-dash"])
def test_list_ordered_false(text):
    assert list_ordered(text) is False


@pytest.mark.parametrize("text", ["Just a sentence.", "", "   "])
def test_list_ordered_none(text):
    assert list_ordered(text) is None


# --- is_page_number ----------------------------------------------------------

def _bbox(y0, y1):
    return {"x0": 0.4, "y0": y0, "x1": 0.6, "y1": y1}


@pytest.mark.parametrize("text", ["12", "Page 12", "12 of 340", "12/340", "- 7 -", "iv"])
def test_is_page_number_footer_band(text):
    # bottom band of a 1000px page
    assert is_page_number(text, _bbox(960, 980), page_height=1000) is True


def test_is_page_number_top_band():
    assert is_page_number("3", _bbox(10, 30), page_height=1000) is True


def test_is_page_number_rejects_body_band():
    # numeric but mid-page -> not a page number
    assert is_page_number("12", _bbox(500, 520), page_height=1000) is False


def test_is_page_number_rejects_long_text():
    assert is_page_number("Risk Factors and Forward Looking", _bbox(960, 980), 1000) is False


def test_is_page_number_rejects_non_pattern():
    assert is_page_number("Tesla, Inc.", _bbox(960, 980), 1000) is False


# --- is_continuation ---------------------------------------------------------

def test_continuation_paragraph_open_end_lowercase_start():
    assert is_continuation(
        "paragraph", "the company recognized revenue of",
        "paragraph", "approximately $1.2 billion in the period.",
    ) is True


def test_no_continuation_when_prev_ends_with_period():
    assert is_continuation(
        "paragraph", "The company recognized revenue.",
        "paragraph", "approximately $1.2 billion.",
    ) is False


def test_no_continuation_when_current_starts_uppercase():
    assert is_continuation(
        "paragraph", "the company recognized revenue of",
        "paragraph", "The board approved a buyback.",
    ) is False


def test_no_continuation_across_different_types():
    assert is_continuation("paragraph", "open clause", "table", "anything") is False


def test_no_continuation_for_non_continuable_type():
    assert is_continuation("heading", "open clause", "heading", "more") is False


def test_list_continuation_fragment_without_new_marker():
    assert is_continuation(
        "list", "1. install the battery and",
        "list", "connect the power adapter",
    ) is True


def test_no_list_continuation_when_new_marker_starts():
    assert is_continuation(
        "list", "1. install the battery and",
        "list", "2. connect the power adapter",
    ) is False


# --- clip_words --------------------------------------------------------------

def test_clip_words_by_center():
    words = [_word("in", 10, 10, 20, 20), _word("out", 200, 200, 210, 210)]
    region = {"x0": 0, "y0": 0, "x1": 100, "y1": 100}
    clipped = clip_words(words, region)
    assert [w["text"] for w in clipped] == ["in"]


# --- refine_region -----------------------------------------------------------

def test_refine_section_header_becomes_heading_with_level():
    words = [_word("Energy", 0, 0, 50, 12, size=10.0, bold=True),
             _word("Storage", 52, 0, 100, 12, size=10.0, bold=True)]
    rtype, attrs = refine_region("section_header", words, {"x0": 0, "y0": 0, "x1": 100, "y1": 12},
                                 body_size=10.0, page_height=1000)
    assert rtype == "heading"
    assert attrs == {"level": 3}  # bold same-size -> H3


def test_refine_section_header_defaults_level_when_no_font_signal():
    rtype, attrs = refine_region("section_header", [], {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                                 body_size=0.0, page_height=1000)
    assert rtype == "heading"
    assert attrs == {"level": 3}


def test_refine_title_gets_level_keeps_type():
    # DocLayout labels section headings as `title`; we attach a level but keep type.
    words = [_word("Solar", 0, 0, 40, 12, size=10.0, bold=True),
             _word("Energy", 42, 0, 90, 12, size=10.0, bold=True)]
    rtype, attrs = refine_region("title", words, {"x0": 0, "y0": 0, "x1": 90, "y1": 12},
                                 body_size=10.0, page_height=1000)
    assert rtype == "title"
    assert attrs == {"level": 3}


def test_refine_title_no_level_when_body_sized_non_bold():
    words = [_word("plain", 0, 0, 40, 10, size=10.0, bold=False)]
    rtype, attrs = refine_region("title", words, {"x0": 0, "y0": 0, "x1": 40, "y1": 10},
                                 body_size=10.0, page_height=1000)
    assert rtype == "title"
    assert attrs == {}


def test_refine_list_sets_ordered():
    words = [_word("1.", 0, 0, 10, 10), _word("Install", 12, 0, 60, 10)]
    rtype, attrs = refine_region("list", words, {"x0": 0, "y0": 0, "x1": 60, "y1": 10},
                                 body_size=10.0, page_height=1000)
    assert rtype == "list"
    assert attrs == {"ordered": True}


def test_refine_footer_becomes_page_number():
    words = [_word("12", 480, 965, 510, 980)]
    rtype, attrs = refine_region("page_footer", words, {"x0": 0.48 * 1000, "y0": 965, "x1": 0.51 * 1000, "y1": 980},
                                 body_size=10.0, page_height=1000)
    assert rtype == "page_number"


def test_refine_paragraph_passthrough():
    words = [_word("Some", 0, 0, 40, 10), _word("prose.", 42, 0, 90, 10)]
    rtype, attrs = refine_region("paragraph", words, {"x0": 0, "y0": 0, "x1": 90, "y1": 10},
                                 body_size=10.0, page_height=1000)
    assert rtype == "paragraph"
    assert attrs == {}
