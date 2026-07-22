"""Tests for elements/readers.py."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elements.readers import LiteralTextCursor, TextReader

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
