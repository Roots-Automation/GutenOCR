"""Tests for elements/textbox.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import ImageFont as PILImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthtiger import layers

from elements.readers import LiteralTextCursor
from elements.textbox import TextBox, _extract_word_ratios

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FONT_PATH = str(next(Path(__file__).resolve().parents[1].glob("resources/font/**/*.ttf")))
FONT_SIZE = 32
FONT_CFG = {"path": FONT_PATH, "bold": False}
BOX_SIZE = (600, FONT_SIZE)
BENCH_TEXT = "The quick brown fox jumps over the lazy dog"
BENCH_REPEATS = 20


def _cursor(text: str) -> LiteralTextCursor:
    return LiteralTextCursor(text)


def _make_textbox() -> TextBox:
    return TextBox({"fill": [1.0, 1.0]})


# ---------------------------------------------------------------------------
# _extract_word_ratios
# ---------------------------------------------------------------------------


def test_extract_word_ratios_two_words():
    chars = list("hello world")
    positions = [(i * 10, i * 10 + 9) for i in range(len(chars))]
    result = _extract_word_ratios(chars, positions, line_width=110.0)

    assert len(result) == 2
    assert result[0]["text"] == "hello"
    assert result[1]["text"] == "world"
    assert result[0]["x1_ratio"] == pytest.approx(0.0)
    assert result[1]["x2_ratio"] == pytest.approx(positions[-1][1] / 110.0)
    assert result[0]["x1_ratio"] <= result[0]["x2_ratio"]
    assert result[1]["x1_ratio"] <= result[1]["x2_ratio"]
    assert result[0]["x2_ratio"] <= result[1]["x1_ratio"]
    assert result[1]["x1_ratio"] > 0.0


def test_extract_word_ratios_single_word():
    chars = list("word")
    positions = [(i * 8, i * 8 + 7) for i in range(4)]
    result = _extract_word_ratios(chars, positions, line_width=40.0)
    assert len(result) == 1
    assert result[0]["text"] == "word"


def test_extract_word_ratios_zero_width_does_not_divide_by_zero():
    chars = list("ab")
    positions = [(0, 5), (5, 10)]
    result = _extract_word_ratios(chars, positions, line_width=0.0)
    assert result[0]["x1_ratio"] == pytest.approx(0.0)
    assert result[0]["x2_ratio"] == pytest.approx(1.0)


def test_extract_word_ratios_trailing_space_ignored():
    chars = list("hi ")
    positions = [(i * 6, i * 6 + 5) for i in range(3)]
    result = _extract_word_ratios(chars, positions, line_width=18.0)
    assert len(result) == 1
    assert result[0]["text"] == "hi"


# ---------------------------------------------------------------------------
# TextBox.generate()
# ---------------------------------------------------------------------------


def test_generate_normal_sentence_returns_layer():
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("hello world foo bar"), FONT_CFG)
    assert layer is not None
    assert len(text) > 0
    assert len(words) > 0


def test_generate_single_long_word_truncates_instead_of_dropping():
    """A long token with no spaces is truncated rather than dropped."""
    tb = _make_textbox()
    np.random.seed(0)
    narrow_box = (60, FONT_SIZE)
    layer, text, words = tb.generate(narrow_box, _cursor("superlongwordwithoutanyspaces"), FONT_CFG)
    assert layer is not None
    assert len(text) > 0
    assert "superlongwordwithoutanyspaces".startswith(text)


def test_generate_cursor_not_poisoned_no_space():
    """No-space backtrack: consecutive calls must not overlap in source."""
    tb = _make_textbox()
    source = "superlongwordwithoutanyspaces second"
    cursor = _cursor(source)
    narrow_box = (60, FONT_SIZE)

    np.random.seed(0)
    _, text1, _ = tb.generate(narrow_box, cursor, FONT_CFG)
    _, text2, _ = tb.generate(narrow_box, cursor, FONT_CFG)

    assert text1 is not None and text2 is not None
    pos1 = source.find(text1)
    pos2 = source.find(text2, pos1 + len(text1))
    assert pos2 >= pos1 + len(text1), f"{text2!r} overlaps {text1!r}"


def test_generate_cursor_continuity_normal_backtrack():
    """Normal last_space backtrack: two successive lines must not overlap in source."""
    tb = _make_textbox()
    source = "hello world foo bar baz"
    cursor = _cursor(source)
    half_box = (150, FONT_SIZE)

    np.random.seed(0)
    _, text1, _ = tb.generate(half_box, cursor, FONT_CFG)
    _, text2, _ = tb.generate(half_box, cursor, FONT_CFG)

    assert text1 is not None and text2 is not None
    pos1 = source.find(text1)
    pos2 = source.find(text2, pos1 + len(text1))
    assert pos2 >= pos1 + len(text1), f"{text2!r} overlaps {text1!r}"


def test_generate_only_punctuation_returns_none():
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("--- ... ~~~"), FONT_CFG)
    assert layer is None


def test_generate_empty_string_returns_none():
    assert _make_textbox().generate(BOX_SIZE, _cursor(""), FONT_CFG) == (None, None, None)


def test_generate_only_spaces_returns_none():
    assert _make_textbox().generate(BOX_SIZE, _cursor("     "), FONT_CFG) == (None, None, None)


def test_generate_word_wider_than_box_returns_none():
    assert _make_textbox().generate((1, FONT_SIZE), _cursor("hello world"), FONT_CFG) == (None, None, None)


def test_generate_word_ratios_bounded():
    """All word x-ratios must lie in [0, 1]."""
    tb = _make_textbox()
    np.random.seed(0)
    _, _, words = tb.generate(BOX_SIZE, _cursor(BENCH_TEXT), FONT_CFG)
    assert words is not None
    for w in words:
        assert 0.0 <= w["x1_ratio"] <= w["x2_ratio"] <= 1.0, w["text"]


def test_generate_first_word_x1_ratio_is_zero_after_backtrack():
    """After a last_space backtrack, the next line's first word must start at x1_ratio=0."""
    tb = _make_textbox()
    source = "hello world foo bar baz"
    cursor = _cursor(source)
    half_box = (150, FONT_SIZE)

    np.random.seed(0)
    tb.generate(half_box, cursor, FONT_CFG)  # consume first line, triggers backtrack
    _, _, words = tb.generate(half_box, cursor, FONT_CFG)  # second line has leading space

    assert words is not None and len(words) > 0
    assert words[0]["x1_ratio"] == pytest.approx(0.0, abs=1e-6), (
        f"First word '{words[0]['text']}' x1_ratio={words[0]['x1_ratio']:.4f}, expected 0.0"
    )


def test_generate_deterministic():
    """Same seed must produce identical text and word dicts."""
    tb = _make_textbox()
    source = BENCH_TEXT

    np.random.seed(42)
    _, text_a, words_a = tb.generate(BOX_SIZE, _cursor(source), FONT_CFG)

    np.random.seed(42)
    _, text_b, words_b = tb.generate(BOX_SIZE, _cursor(source), FONT_CFG)

    assert text_a == text_b
    assert len(words_a) == len(words_b)
    for wa, wb in zip(words_a, words_b):
        assert wa == wb


class _BufCursor:
    """Minimal cursor over a fixed string that preserves '\\n', for testing."""

    def __init__(self, s: str) -> None:
        self._s = s
        self._i = 0

    def __len__(self) -> int:
        return len(self._s)

    def __iter__(self) -> "_BufCursor":
        return self

    def __next__(self) -> str:
        if self._i >= len(self._s):
            raise StopIteration
        ch = self._s[self._i]
        self._i += 1
        return ch

    def move(self, i: int) -> None:
        self._i = i

    def next(self) -> None:
        self._i += 1

    def prev(self) -> None:
        self._i = max(0, self._i - 1)

    def get(self) -> str:
        return self._s[self._i % len(self._s)]


def test_walkback_accounts_for_newline_cursor_cost():
    """A newline consumed before an overflowing char must count as a cursor step.

    Without the fix, skipped=0 when overflow fires (\\n not counted), so
    n_restore undershoots by 1 and the cursor lands on '\\n' (pos 3) instead
    of ' ' (pos 2), causing the next generate() to start 1 char too late.
    """
    from pillow_compat import _cached_truetype

    tb = _make_textbox()  # fill=[1.0, 1.0]
    np.random.seed(0)

    font_obj = _cached_truetype(FONT_PATH, FONT_SIZE)
    ascent, descent = font_obj.getmetrics()
    char_scale = FONT_SIZE / (ascent + descent)

    # Compute a box width that fits "ab " but NOT "ab c"
    w_fits = font_obj.getlength("ab ") * char_scale
    w_overflow = font_obj.getlength("ab c") * char_scale
    box_width = (w_fits + w_overflow) / 2

    # Source positions: a=0  b=1  ' '=2  '\n'=3  c=4  d=5
    # Trace: 'a','b',' ' appended (last_space=2, cursor_costs=[1,1,1]);
    #        '\n'(3) consumed → fix: skipped=1, old: skipped=0;
    #        'c'(4) overflows → cursor.prev() (_i: 5→4); break (trailing skipped).
    # n_restore = cursor_costs[2] + skipped = 1 + 1(fix) = 2  →  _i: 4→3→2
    # n_restore = cursor_costs[2] + skipped = 1 + 0(old) = 1  →  _i: 4→3  (off by 1)
    cursor = _BufCursor("ab \ncd")
    tb.generate((box_width, FONT_SIZE), cursor, FONT_CFG)

    assert cursor._i == 2, (
        f"cursor should be at pos 2 (' ') after walkback, got {cursor._i}; "
        "the consumed '\\n' was not counted in cursor restore steps"
    )


def test_generate_skips_crlf():
    """CR and LF characters must be silently skipped."""
    tb = _make_textbox()

    np.random.seed(0)
    _, text_clean, words_clean = tb.generate(BOX_SIZE, _cursor("hello world"), FONT_CFG)

    np.random.seed(0)
    _, text_crlf, words_crlf = tb.generate(BOX_SIZE, _cursor("hel\rlo\n world"), FONT_CFG)

    assert text_clean == text_crlf
    assert words_clean == words_crlf


def test_generate_layer_height_matches_cell():
    """Returned layer height must equal the requested cell height."""
    tb = _make_textbox()
    np.random.seed(0)
    layer, _, _ = tb.generate(BOX_SIZE, _cursor(BENCH_TEXT), FONT_CFG)
    assert layer is not None
    assert abs(layer.size[1] - FONT_SIZE) < 2


def test_char_scale_getmetrics_matches_textlayer_probe():
    """char_scale from getmetrics() must agree with a TextLayer height measurement."""
    font_obj = PILImageFont.truetype(FONT_PATH, size=FONT_SIZE)
    ascent, descent = font_obj.getmetrics()
    pil_height = ascent + descent
    scale_metrics = FONT_SIZE / pil_height if pil_height > 0 else 1.0

    probe = layers.TextLayer("A", path=FONT_PATH, bold=False, size=FONT_SIZE)
    scale_probe = FONT_SIZE / probe.height if probe.height > 0 else 1.0

    assert scale_metrics == pytest.approx(scale_probe, rel=1e-3)


# ---------------------------------------------------------------------------
# Structural performance — TextLayer call count (deterministic, CI-safe)
# ---------------------------------------------------------------------------


def test_generate_makes_exactly_one_textlayer_call():
    """generate() must render the whole line in a single TextLayer call."""
    tb = _make_textbox()
    np.random.seed(0)
    with patch("elements.textbox.layers.TextLayer", wraps=layers.TextLayer) as mock_tl:
        tb.generate(BOX_SIZE, _cursor(BENCH_TEXT), FONT_CFG)
        assert mock_tl.call_count == 1


# ---------------------------------------------------------------------------
# Null count tracking
# ---------------------------------------------------------------------------


def test_render_cells_counts_nulls():
    """_render_cells must count every None return from generate()."""
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

    cells = [
        ([0.0, 0.0, 1.0, 32.0], "left", 0),
        ([0.0, 0.0, 1.0, 32.0], "left", 1),
        ([0.0, 0.0, 1.0, 32.0], "left", 2),
    ]
    text_layers, texts, block_ids, wpl = [], [], [], []

    np.random.seed(0)
    _, null_count, total_count = content._render_cells(
        cells, content.reader, content.font.sample(), "body", 0, {}, text_layers, texts, block_ids, wpl
    )

    assert total_count == 3
    assert null_count == 3
    assert len(text_layers) == 0


def test_render_cells_null_frac_zero_on_success():
    """When cells render successfully, null_count must be 0."""
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

    cells = [([0.0, 0.0, 600.0, 32.0], "left", 0)]
    text_layers, texts, block_ids, wpl = [], [], [], []

    np.random.seed(1)
    _, null_count, total_count = content._render_cells(
        cells, content.reader, content.font.sample(), "body", 0, {}, text_layers, texts, block_ids, wpl
    )

    assert total_count == 1
    assert null_count == 0
    assert len(text_layers) == 1


# ---------------------------------------------------------------------------
# Unrenderable character skipping
# ---------------------------------------------------------------------------


def test_generate_skips_unrenderable_chars_in_text():
    """Characters the font cannot render must be absent from the returned text."""
    tb = _make_textbox()
    np.random.seed(0)
    # CourierPrime cannot render CJK — the characters must be dropped silently.
    _, text, _ = tb.generate(BOX_SIZE, _cursor("Hello 中文 world"), FONT_CFG)
    assert text is not None
    assert "中" not in text
    assert "文" not in text
    assert "Hello" in text
    assert "world" in text


def test_generate_unrenderable_chars_absent_from_word_annotations():
    """Word annotations must not contain characters the font cannot render."""
    tb = _make_textbox()
    np.random.seed(0)
    _, _, words = tb.generate(BOX_SIZE, _cursor("Hello 中文 world"), FONT_CFG)
    all_word_text = " ".join(w["text"] for w in words)
    assert "中" not in all_word_text
    assert "文" not in all_word_text


def test_generate_all_unrenderable_returns_none():
    """A line composed entirely of unrenderable characters must return (None, None, None)."""
    tb = _make_textbox()
    np.random.seed(0)
    layer, text, words = tb.generate(BOX_SIZE, _cursor("中文한국어"), FONT_CFG)
    assert layer is None
    assert text is None
    assert words is None


def test_generate_mixed_line_has_correct_word_count():
    """After dropping unrenderable chars the remaining words must be annotated correctly."""
    tb = _make_textbox()
    np.random.seed(0)
    _, _, words = tb.generate(BOX_SIZE, _cursor("foo 中 bar 文 baz"), FONT_CFG)
    assert words is not None
    word_texts = [w["text"] for w in words]
    assert word_texts == ["foo", "bar", "baz"]


def test_extract_word_ratios_all_spaces_returns_empty():
    """A char list of only spaces must yield zero words — not crash or return empty-text words."""
    chars = list("   ")
    positions = [(i * 5, i * 5 + 4) for i in range(3)]
    assert _extract_word_ratios(chars, positions, line_width=15.0) == []


def test_extract_word_ratios_consecutive_spaces_still_yields_two_words():
    """Multiple consecutive spaces between words must not cause duplication or merging."""
    chars = list("hi  there")
    positions = [(i * 6, i * 6 + 5) for i in range(len(chars))]
    result = _extract_word_ratios(chars, positions, line_width=float(len(chars) * 6))
    assert len(result) == 2
    assert result[0]["text"] == "hi"
    assert result[1]["text"] == "there"


def test_extract_word_ratios_adjacent_words_non_overlapping():
    """For every adjacent word pair, word[i].x2_ratio must be ≤ word[i+1].x1_ratio.

    Overlapping word bboxes would break downstream geometry checks.
    """
    chars = list("foo bar baz")
    positions = [(i * 8, i * 8 + 7) for i in range(len(chars))]
    words = _extract_word_ratios(chars, positions, line_width=float(len(chars) * 8))
    assert len(words) == 3
    for i in range(len(words) - 1):
        assert words[i]["x2_ratio"] <= words[i + 1]["x1_ratio"], (
            f"word {i} x2={words[i]['x2_ratio']:.4f} overlaps word {i + 1} x1={words[i + 1]['x1_ratio']:.4f}"
        )


def test_generate_leading_unrenderable_char_not_in_output():
    """When the cursor starts with an unrenderable char, the char must not appear in
    the returned text or word annotations, and the line must still render normally."""
    tb = _make_textbox()
    np.random.seed(0)
    _, text, words = tb.generate(BOX_SIZE, _cursor("中foo bar"), FONT_CFG)
    assert text is not None
    assert "中" not in text
    assert text.startswith("foo")
    assert words is not None
    assert words[0]["text"] == "foo"


def test_generate_cursor_exhausted_no_overflow_returns_valid():
    """When the cursor is exhausted before the box is full (no overflow char),
    generate() must return a valid result — not (None, None, None)."""
    tb = _make_textbox()
    np.random.seed(0)
    # "A" fits easily in BOX_SIZE; cursor exhausts after 'A' + trailing space.
    _, text, words = tb.generate(BOX_SIZE, _cursor("A"), FONT_CFG)
    assert text == "A"
    assert words is not None and len(words) == 1
    assert words[0]["text"] == "A"


def test_walkback_crlf_each_char_counts_as_one_step():
    """\\r and \\n in a CRLF sequence must each add 1 to skipped, totalling 2.

    Source: "ab \\r\\ncd"  (indices: a=0 b=1 sp=2 \\r=3 \\n=4 c=5 d=6)
    Box: fits "ab " but not "ab c".

    After rendering 'a','b',' ' (last_space=2, cursor_costs=[1,1,1], skipped=0):
      '\\r'(3→4): skipped=1
      '\\n'(4→5): skipped=2
      'c'(5→6): overflow → cursor.prev() → _i=5

    n_restore = cursor_costs[2] + skipped = 1+2 = 3
    cursor.prev() ×3: 5→4→3→2 → cursor._i=2 (the space) ✓

    Without the fix (neither \\r nor \\n increments skipped), skipped=0 so
    n_restore=1, cursor.prev()×1: 5→4 → _i=4 (the \\n) — off by 2.
    """
    from pillow_compat import _cached_truetype

    tb = _make_textbox()
    np.random.seed(0)
    font_obj = _cached_truetype(FONT_PATH, FONT_SIZE)
    ascent, descent = font_obj.getmetrics()
    char_scale = FONT_SIZE / (ascent + descent)
    w_fits = font_obj.getlength("ab ") * char_scale
    w_overflow = font_obj.getlength("ab c") * char_scale
    box_width = (w_fits + w_overflow) / 2

    cursor = _BufCursor("ab \r\ncd")
    tb.generate((box_width, FONT_SIZE), cursor, FONT_CFG)

    assert cursor._i == 2, (
        f"cursor should be at pos 2 (the space) after CRLF walkback, got {cursor._i}; "
        "\\r and/or \\n were not counted in cursor restore steps"
    )


def test_walkback_double_newline_both_counted():
    """Two consecutive \\n chars must each increment skipped, giving n_restore=3.

    Source: "ab \\n\\ncd"  (a=0 b=1 sp=2 \\n=3 \\n=4 c=5 d=6)
    Box: fits "ab " but not "ab c".

    After ' '(2): last_space=2, cursor_costs=[1,1,1], skipped=0.
    '\\n'(3→4): skipped=1.  '\\n'(4→5): skipped=2.
    'c'(5→6): overflow → cursor.prev() → _i=5.
    n_restore = 1+2 = 3 → prev()×3: 5→4→3→2 → _i=2 ✓

    With only one \\n counted (partial fix), n_restore=2: _i=3 — still wrong.
    With neither counted (old code), n_restore=1: _i=4 — wrong by 2.
    """
    from pillow_compat import _cached_truetype

    tb = _make_textbox()
    np.random.seed(0)
    font_obj = _cached_truetype(FONT_PATH, FONT_SIZE)
    ascent, descent = font_obj.getmetrics()
    char_scale = FONT_SIZE / (ascent + descent)
    w_fits = font_obj.getlength("ab ") * char_scale
    w_overflow = font_obj.getlength("ab c") * char_scale
    box_width = (w_fits + w_overflow) / 2

    cursor = _BufCursor("ab \n\ncd")
    tb.generate((box_width, FONT_SIZE), cursor, FONT_CFG)

    assert cursor._i == 2, (
        f"cursor should be at pos 2 (the space) after double-\\n walkback, got {cursor._i}; "
        "one or both \\n chars were not counted in cursor restore steps"
    )


def test_walkback_trailing_unrenderable_restores_cursor_to_space():
    """Unrenderable chars at the END of the buffer (before StopIteration) must
    be included in n_restore so the cursor lands on the last-space, not past it.

    Source: "foo 中" — '中' is not renderable by the English test font.
    Box: wide enough to fit all visible content ("foo ").

    After rendering 'f','o','o',' ' (last_space=3, cursor_costs=[1,1,1,1]):
      '中'(4→5): unrenderable → skipped=1.
      StopIteration. cursor._i=5.

    n_restore = cursor_costs[3] + skipped = 1+1 = 2
    cursor.prev()×2: 5→4→3 → cursor._i=3 (the space) ✓

    Without trailing skipped in n_restore: n_restore=1, cursor._i=4 ('中').
    """
    tb = _make_textbox()
    np.random.seed(0)
    cursor = _BufCursor("foo 中")  # U+4E2D = '中', unrenderable in CourierPrime
    tb.generate(BOX_SIZE, cursor, FONT_CFG)

    assert cursor._i == 3, (
        f"cursor should be at pos 3 (the space) after trailing-unrenderable walkback, "
        f"got {cursor._i}; the unrenderable '\\u4e2d' was not counted in n_restore"
    )


def test_walkback_restores_cursor_past_skipped_chars_before_overflow():
    """Skipped chars between the last rendered char and the overflow must be
    included in the walkback step count, or the following renderable word is
    lost mid-character.

    Setup: box fits 'foo bar' but not 'foo barx'.  The cursor holds
    'foo bar中中x' — the two CJK chars are skipped, so 'x' is the char that
    triggers overflow.  After line 1 ('foo'), line 2 must start at 'bar' (the
    word right after the space), not in the middle of 'bar' or 'arx'.
    """
    # Box width: fits 'foo bar' (73 px) but not 'foo barx' (83 px).
    # Computed from CourierPrime at 32 px; 78 px sits between them.
    NARROW = (78, FONT_SIZE)
    tb = _make_textbox()

    cursor = _cursor("foo bar中中x next")
    np.random.seed(0)
    _, line1, _ = tb.generate(NARROW, cursor, FONT_CFG)
    np.random.seed(0)
    _, line2, words2 = tb.generate(NARROW, cursor, FONT_CFG)

    assert line1 == "foo", f"expected line1='foo', got {line1!r}"
    # 'bar' must be the first word on line 2 — not 'ar', 'r', or 'arx'
    assert words2 is not None and words2[0]["text"].startswith("bar"), (
        f"expected line2 to start with 'bar', got words={[w['text'] for w in words2]!r}"
    )
