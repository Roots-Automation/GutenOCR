"""Tests for elements/readers.py."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elements.readers import HuggingFaceTextReader, LiteralTextCursor, TextReader

# ---------------------------------------------------------------------------
# LiteralTextCursor
# ---------------------------------------------------------------------------


def _consume_all(cursor: LiteralTextCursor) -> str:
    return "".join(cursor)


def test_literal_emits_text_then_stops():
    cursor = LiteralTextCursor("hello")
    result = _consume_all(cursor)
    assert result == "hello "  # trailing space included


def test_literal_raises_stop_iteration_after_full_consumption():
    cursor = LiteralTextCursor("hi")
    _consume_all(cursor)
    with pytest.raises(StopIteration):
        next(cursor)


def test_literal_prev_backtracks_one_char():
    cursor = LiteralTextCursor("abc")
    next(cursor)  # 'a'
    next(cursor)  # 'b'
    cursor.prev()
    assert cursor.get() == "b"


def test_literal_prev_decrements_consumed_so_stop_iteration_is_correct():
    cursor = LiteralTextCursor("ab")  # buf = "ab ", len=3
    next(cursor)  # _consumed=1
    next(cursor)  # _consumed=2
    cursor.prev()  # _consumed=1
    # Should be able to consume 2 more chars before StopIteration
    chars = _consume_all(cursor)
    assert len(chars) == 2


def test_literal_move_resets_consumed_so_cursor_is_reusable():
    cursor = LiteralTextCursor("hi")
    _consume_all(cursor)  # exhaust it
    cursor.move(0)  # reset
    result = _consume_all(cursor)
    assert result == "hi "


def test_literal_move_mid_string_sets_consumed_consistently():
    cursor = LiteralTextCursor("abcde")  # buf = "abcde ", len=6
    cursor.move(2)
    # Should be able to consume len - 2 = 4 more chars
    chars = _consume_all(cursor)
    assert len(chars) == 4
    assert chars[0] == "c"


def test_literal_strips_cr_lf():
    cursor = LiteralTextCursor("line1\r\nline2")
    result = _consume_all(cursor)
    assert "\r" not in result
    assert "\n" not in result
    assert "line1line2" in result


def test_literal_crlf_does_not_burn_consumed_budget():
    clean = LiteralTextCursor("ab")
    crlf = LiteralTextCursor("a\nb")
    clean_chars = _consume_all(clean)
    crlf_chars = _consume_all(crlf)
    # Both should yield the same visible characters (plus trailing space)
    assert clean_chars == crlf_chars


# ---------------------------------------------------------------------------
# TextReader
# ---------------------------------------------------------------------------


def _make_text_reader(content: str) -> TextReader:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return TextReader(tmp.name)


def test_text_reader_reads_correct_characters():
    text = "abcdefgh"
    reader = _make_text_reader(text)
    chars = [reader.get()]
    for _ in range(len(text) - 1):
        reader.next()
        chars.append(reader.get())
    assert "".join(chars) == text


def test_text_reader_wraps_around():
    reader = _make_text_reader("abc")
    # advance past the end
    for _ in range(len("abc")):
        reader.next()
    assert reader.get() == "a"


def test_text_reader_prev_backtracks():
    reader = _make_text_reader("abc")
    reader.next()  # -> 'b'
    reader.prev()  # back to 'a'
    assert reader.get() == "a"


def test_text_reader_move_jumps_to_position():
    reader = _make_text_reader("abcdef")
    reader.move(3)
    assert reader.get() == "d"


def test_text_reader_length_matches_content():
    content = "hello world"
    reader = _make_text_reader(content)
    assert len(reader) == len(content)


def test_text_reader_context_manager_closes_file():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False)
    tmp.write("test")
    tmp.flush()
    tmp.close()
    with TextReader(tmp.name) as reader:
        assert reader.get() == "t"
    assert reader.fp.closed


# ---------------------------------------------------------------------------
# HuggingFaceTextReader
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal iterable that mimics a HuggingFace streaming dataset."""

    def __init__(self, samples):
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)


def _make_hf_reader(texts=None, **kwargs):
    """Instantiate HuggingFaceTextReader backed by a fake dataset."""
    if texts is None:
        # 30 identical docs → long joined text for threshold tests
        texts = ["The quick brown fox jumps over the lazy dog"] * 30
    samples = [{"text": t} for t in texts]
    fake_dataset = _FakeDataset(samples)
    mock_ds_mod = MagicMock()
    mock_ds_mod.load_dataset = MagicMock(return_value=fake_dataset)

    buf = kwargs.pop("buffer_size", len(texts))
    with patch.dict(sys.modules, {"datasets": mock_ds_mod}):
        reader = HuggingFaceTextReader(buffer_size=buf, **kwargs)
    return reader


def test_hf_reader_get_returns_a_character():
    reader = _make_hf_reader()
    char = reader.get()
    assert isinstance(char, str) and len(char) == 1


def test_hf_reader_next_advances_position():
    reader = _make_hf_reader()
    reader.next()
    assert reader.idx == 1


def test_hf_reader_prev_backtracks_position():
    reader = _make_hf_reader()
    reader.next()
    reader.next()
    reader.prev()
    assert reader.idx == 1


def test_hf_reader_move_jumps_to_position():
    reader = _make_hf_reader()
    reader.move(5)
    assert reader.idx == 5
    assert reader.get() == reader._get_current_text()[5]


def test_hf_reader_next_sets_needs_refresh_above_threshold():
    """next() must set _needs_refresh when idx crosses the 80% watermark."""
    reader = _make_hf_reader()
    text = reader._get_current_text()
    n = len(text)
    threshold = int(n * 0.8)

    reader.move(threshold)  # land just at the threshold
    reader._needs_refresh = False  # ensure clean slate
    reader.next()  # crosses into the >80% zone

    assert reader._needs_refresh


def test_hf_reader_prev_clears_needs_refresh_below_threshold():
    """The prev() fix: backtracking below 80% must clear _needs_refresh.

    Without the fix, prev() left a stale True flag that caused the next
    move() to spuriously refresh the buffer and corrupt the read position.
    """
    reader = _make_hf_reader()
    text = reader._get_current_text()
    n = len(text)
    # Start above the threshold with the flag set (as if next() got us here).
    reader.idx = int(n * 0.8) + 2
    reader._needs_refresh = True

    reader.prev()  # moves back to int(n*0.8)+1, which is still >80% → flag stays
    reader.prev()  # moves back to int(n*0.8), which is ≤80% → flag must clear

    assert not reader._needs_refresh


def test_hf_reader_charset_strips_non_ascii():
    """charset='ascii' must filter out non-ASCII characters."""
    reader = _make_hf_reader(texts=["café résumé"] * 5, charset="ascii")
    text = reader._get_current_text()
    assert all(ord(c) < 128 for c in text)
