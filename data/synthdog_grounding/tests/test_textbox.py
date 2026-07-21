"""Tests for elements/textbox.py.

Two goals:
  1. Prove fragility — document the silent failure modes in TextBox.generate().
  2. Prove the performance cost — show that character-by-character rendering
     is measurably slower than a single PIL draw.text() call.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elements.readers import LiteralTextCursor
from elements.textbox import TextBox, _extract_word_ratios

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FONT_PATH = str(next(Path(__file__).resolve().parents[1].glob("resources/font/**/*.ttf")))
FONT_SIZE = 32  # pixels — large enough to actually lay out glyphs
FONT_CFG = {"path": FONT_PATH, "bold": False}
BOX_SIZE = (600, FONT_SIZE)  # (width, height) in pixels


def _cursor(text: str) -> LiteralTextCursor:
    return LiteralTextCursor(text)


def _make_textbox() -> TextBox:
    return TextBox({"fill": [1.0, 1.0]})  # no fill randomness for test clarity


# ---------------------------------------------------------------------------
# _extract_word_ratios — pure function, fully unit-testable
# ---------------------------------------------------------------------------


class _FakeLayer:
    """Minimal stand-in for a synthtiger TextLayer."""

    def __init__(self, left: float, right: float):
        self.left = left
        self.right = right


def test_extract_word_ratios_two_words():
    chars = list("hello world")
    layers = [_FakeLayer(i * 10, i * 10 + 9) for i in range(len(chars))]
    result = _extract_word_ratios(chars, layers, line_width=110.0)

    assert len(result) == 2
    assert result[0]["text"] == "hello"
    assert result[1]["text"] == "world"
    # x1_ratio of "hello" starts at 0, x2_ratio of "world" approaches 1
    assert result[0]["x1_ratio"] == pytest.approx(0.0)
    assert result[1]["x2_ratio"] == pytest.approx(layers[-1].right / 110.0)


def test_extract_word_ratios_single_word():
    chars = list("word")
    layers = [_FakeLayer(i * 8, i * 8 + 7) for i in range(4)]
    result = _extract_word_ratios(chars, layers, line_width=40.0)
    assert len(result) == 1
    assert result[0]["text"] == "word"


def test_extract_word_ratios_zero_width_does_not_divide_by_zero():
    chars = list("ab")
    layers = [_FakeLayer(0, 5), _FakeLayer(5, 10)]
    # line_width=0 must not raise
    result = _extract_word_ratios(chars, layers, line_width=0.0)
    assert result[0]["x1_ratio"] == pytest.approx(0.0)
    assert result[0]["x2_ratio"] == pytest.approx(1.0)


def test_extract_word_ratios_trailing_space_ignored():
    # A trailing space in chars should not create a ghost empty word
    chars = list("hi ")
    layers = [_FakeLayer(i * 6, i * 6 + 5) for i in range(3)]
    result = _extract_word_ratios(chars, layers, line_width=18.0)
    assert len(result) == 1
    assert result[0]["text"] == "hi"


# ---------------------------------------------------------------------------
# TextBox.generate() — fragility proofs
# ---------------------------------------------------------------------------


def test_generate_normal_sentence_returns_layer():
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("hello world foo bar"), FONT_CFG)
    assert layer is not None
    assert len(text) > 0
    assert len(words) > 0


def test_generate_single_long_word_truncates_instead_of_dropping():
    """A long token with no spaces is now truncated at the overflow point.

    Previously this returned (None, None, None), silently dropping the cell.
    Now it returns whatever characters fit, so no cell is lost to a missing
    space boundary.
    """
    tb = _make_textbox()
    np.random.seed(0)
    narrow_box = (60, FONT_SIZE)  # narrow enough that any multi-char word overflows
    layer, text, words = tb.generate(narrow_box, _cursor("superlongwordwithoutanyspaces"), FONT_CFG)
    assert layer is not None
    assert len(text) > 0
    # text is a prefix of the original token
    assert "superlongwordwithoutanyspaces".startswith(text)


def test_generate_only_punctuation_returns_none():
    """Lines that reduce to only non-alphanumeric chars fail the alpha check.

    text_alpha_only strips [^\\w], so a line of e.g. dashes or dots renders
    fine but gets thrown away after layout work is already done.
    """
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("--- ... ~~~"), FONT_CFG)
    assert layer is None


def test_generate_empty_string_returns_none():
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor(""), FONT_CFG)
    assert layer is None


def test_generate_only_spaces_returns_none():
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("     "), FONT_CFG)
    assert layer is None


def test_generate_word_wider_than_box_returns_none():
    """A normal word that is wider than the box triggers full backtrack → None."""
    tb = _make_textbox()
    np.random.seed(0)
    tiny_box = (1, FONT_SIZE)  # 1 px wide — nothing fits
    layer, text, words = tb.generate(tiny_box, _cursor("hello world"), FONT_CFG)
    assert layer is None


# ---------------------------------------------------------------------------
# Performance proof — character-by-character vs single draw.text()
# ---------------------------------------------------------------------------

BENCH_TEXT = "The quick brown fox jumps over the lazy dog"
BENCH_REPEATS = 20


def _time_textbox_generate() -> float:
    """Time the current character-by-character TextBox.generate()."""
    tb = _make_textbox()
    np.random.seed(0)
    # Warm up
    tb.generate(BOX_SIZE, _cursor(BENCH_TEXT), FONT_CFG)

    t0 = time.perf_counter()
    for _ in range(BENCH_REPEATS):
        np.random.seed(0)
        tb.generate(BOX_SIZE, _cursor(BENCH_TEXT), FONT_CFG)
    return (time.perf_counter() - t0) / BENCH_REPEATS


def _time_single_draw_text() -> float:
    """Time a single PIL draw.text() call over the same text."""
    font = ImageFont.truetype(FONT_PATH, size=FONT_SIZE)
    # Warm up
    img = Image.new("RGBA", BOX_SIZE, (0, 0, 0, 0))
    ImageDraw.Draw(img).text((0, 0), BENCH_TEXT, font=font, fill=(0, 0, 0, 255))
    np.array(img, dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(BENCH_REPEATS):
        img = Image.new("RGBA", BOX_SIZE, (0, 0, 0, 0))
        ImageDraw.Draw(img).text((0, 0), BENCH_TEXT, font=font, fill=(0, 0, 0, 255))
        np.array(img, dtype=np.float32)
    return (time.perf_counter() - t0) / BENCH_REPEATS


# ---------------------------------------------------------------------------
# Null count tracking — prove the pipeline surfaces dropped cells
# ---------------------------------------------------------------------------


def test_render_cells_counts_nulls():
    """_render_cells must count every None return from textbox.generate().

    We feed a 1px-wide cell so every textbox call returns None, then verify
    the null/total counts come back correctly.
    """
    from elements.content import Content

    # Minimal config: one font dir, no zones, no sprinkle
    cfg = {
        "text": {"path": str(Path(__file__).resolve().parents[1] / "resources/corpus/enwiki.txt")},
        "font": {"paths": [str(Path(__file__).resolve().parents[1] / "resources/font/en")], "weights": [1]},
        "layout": {},
        "textbox": {"fill": [1.0, 1.0]},
        "textbox_color": {"prob": 0},
        "content_color": {"prob": 0},
        "text_sprinkle": {"prob": 0},
    }
    np.random.seed(0)
    content = Content(cfg)

    # 3 cells, each 1px wide → guaranteed None from every textbox call
    cells = [
        ([0.0, 0.0, 1.0, 32.0], "left", 0),
        ([0.0, 0.0, 1.0, 32.0], "left", 1),
        ([0.0, 0.0, 1.0, 32.0], "left", 2),
    ]
    text_layers, texts, block_ids, wpl = [], [], [], []
    block_region_types: dict = {}

    np.random.seed(0)
    _, null_count, total_count = content._render_cells(
        cells,
        content.reader,
        content.font.sample(),
        "body",
        0,
        block_region_types,
        text_layers,
        texts,
        block_ids,
        wpl,
    )

    assert total_count == 3
    assert null_count == 3
    assert len(text_layers) == 0


def test_render_cells_null_frac_zero_on_success():
    """When cells successfully render, null_count must be 0."""
    from elements.content import Content

    cfg = {
        "text": {"path": str(Path(__file__).resolve().parents[1] / "resources/corpus/enwiki.txt")},
        "font": {"paths": [str(Path(__file__).resolve().parents[1] / "resources/font/en")], "weights": [1]},
        "layout": {},
        "textbox": {"fill": [1.0, 1.0]},
        "textbox_color": {"prob": 0},
        "content_color": {"prob": 0},
        "text_sprinkle": {"prob": 0},
    }
    np.random.seed(0)
    content = Content(cfg)

    # Wide cell — should fit text
    cells = [([0.0, 0.0, 600.0, 32.0], "left", 0)]
    text_layers, texts, block_ids, wpl = [], [], [], []
    block_region_types: dict = {}

    np.random.seed(1)
    _, null_count, total_count = content._render_cells(
        cells,
        content.reader,
        content.font.sample(),
        "body",
        0,
        block_region_types,
        text_layers,
        texts,
        block_ids,
        wpl,
    )

    assert total_count == 1
    assert null_count == 0
    assert len(text_layers) == 1


# ---------------------------------------------------------------------------
# Performance proof — character-by-character vs single draw.text()
# ---------------------------------------------------------------------------


def test_single_draw_text_is_faster_than_char_by_char():
    """Prove that a single draw.text() is meaningfully faster than char-by-char.

    This test does not enforce a specific speedup ratio — it just asserts the
    direction. The printed output shows the actual numbers.
    """
    char_by_char_s = _time_textbox_generate()
    single_draw_s = _time_single_draw_text()
    speedup = char_by_char_s / single_draw_s

    print(f"\nchar-by-char : {char_by_char_s * 1000:.2f} ms/line")
    print(f"single draw  : {single_draw_s * 1000:.2f} ms/line")
    print(f"speedup      : {speedup:.1f}x")

    assert single_draw_s < char_by_char_s, (
        f"Expected single draw.text() to be faster, but "
        f"char-by-char={char_by_char_s * 1000:.2f}ms, single={single_draw_s * 1000:.2f}ms"
    )
