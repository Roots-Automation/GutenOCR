"""Unit tests for Pillow-native annotation builders and shared helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotations import (
    build_block_annotations,
    build_pillow_line_annotations,
    build_pillow_word_annotations,
    filter_degenerate,
)
from serialization import LineAnnotation, WordAnnotation

from elements.entry import RenderedLine


def _make_line(
    text="hello world",
    block_id=0,
    region_type="body",
    x1=10,
    y1=10,
    x2=90,
    y2=30,
    words=None,
    word_x1s=None,
    word_x2s=None,
):
    if words is None:
        words = text.split()
    if word_x1s is None:
        n = len(words)
        step = (x2 - x1) // max(n, 1)
        word_x1s = [x1 + i * step for i in range(n)]
        word_x2s = [x1 + (i + 1) * step for i in range(n)]
    return RenderedLine(
        text=text,
        block_id=block_id,
        region_type=region_type,
        x1_px=x1,
        y1_px=y1,
        x2_px=x2,
        y2_px=y2,
        words=words,
        word_x1s=word_x1s,
        word_x2s=word_x2s,
        font_family="Test",
        font_size_px=12,
        text_color_rgb=[0, 0, 0],
    )


class TestBuildPillowLineAnnotations:
    def test_normalizes_correctly(self):
        rl = _make_line(x1=10, y1=20, x2=90, y2=40)
        lines = build_pillow_line_annotations([rl], image_width=100, image_height=100)
        assert len(lines) == 1
        bbox = lines[0].bbox
        assert abs(bbox[0] - 0.1) < 1e-3
        assert abs(bbox[1] - 0.2) < 1e-3
        assert abs(bbox[2] - 0.9) < 1e-3
        assert abs(bbox[3] - 0.4) < 1e-3

    def test_bbox_x1_le_x2_y1_le_y2(self):
        rl = _make_line(x1=5, y1=5, x2=50, y2=25)
        lines = build_pillow_line_annotations([rl], 100, 100)
        b = lines[0].bbox
        assert b[0] <= b[2]
        assert b[1] <= b[3]

    def test_clamped_to_01(self):
        rl = _make_line(x1=0, y1=0, x2=200, y2=200)
        lines = build_pillow_line_annotations([rl], 100, 100)
        b = lines[0].bbox
        for v in b:
            assert 0.0 <= v <= 1.0

    def test_line_id_assigned_sequentially(self):
        rls = [_make_line(text=f"line {i}", block_id=i) for i in range(5)]
        lines = build_pillow_line_annotations(rls, 100, 100)
        for i, ln in enumerate(lines):
            assert ln.line_id == i

    def test_font_metadata_preserved(self):
        rl = _make_line()
        rl.font_family = "MyFont"
        rl.font_size_px = 14
        rl.text_color_rgb = [30, 30, 30]
        lines = build_pillow_line_annotations([rl], 100, 100)
        assert lines[0].font_family == "MyFont"
        assert lines[0].font_size_px == 14
        assert lines[0].text_color_rgb == [30, 30, 30]

    def test_empty_input(self):
        lines = build_pillow_line_annotations([], 100, 100)
        assert lines == []


class TestBuildPillowWordAnnotations:
    def test_word_x_range_within_line_x_range(self):
        rl = _make_line(x1=10, y1=10, x2=90, y2=30, words=["hello", "world"], word_x1s=[10, 50], word_x2s=[45, 88])
        words = build_pillow_word_annotations([rl], 100, 100)
        line_x1 = rl.x1_px / 100
        line_x2 = rl.x2_px / 100
        for w in words:
            assert w.bbox[0] >= line_x1 - 1e-3
            assert w.bbox[2] <= line_x2 + 1e-3

    def test_word_count_matches(self):
        rl = _make_line(words=["a", "b", "c"], word_x1s=[0, 30, 60], word_x2s=[25, 55, 85])
        words = build_pillow_word_annotations([rl], 100, 100)
        assert len(words) == 3

    def test_empty_words_no_crash(self):
        rl = _make_line(words=[], word_x1s=[], word_x2s=[])
        words = build_pillow_word_annotations([rl], 100, 100)
        assert words == []

    def test_word_ids_sequential(self):
        rl1 = _make_line(words=["a", "b"], word_x1s=[0, 50], word_x2s=[45, 95])
        rl2 = _make_line(words=["c"], word_x1s=[0], word_x2s=[40])
        words = build_pillow_word_annotations([rl1, rl2], 100, 100)
        assert [w.word_id for w in words] == [0, 1, 2]

    def test_word_bbox_normalized(self):
        rl = _make_line(words=["hello"], word_x1s=[10], word_x2s=[50])
        words = build_pillow_word_annotations([rl], 100, 100)
        for v in words[0].bbox:
            assert 0.0 <= v <= 1.0


class TestBuildBlockAnnotations:
    def test_groups_lines_by_block_id(self):
        block_ids = [0, 0, 1, 1, 1]
        bboxes = [
            [0.1, 0.1, 0.9, 0.2],
            [0.1, 0.2, 0.9, 0.3],
            [0.1, 0.4, 0.9, 0.5],
            [0.1, 0.5, 0.9, 0.6],
            [0.1, 0.6, 0.9, 0.7],
        ]
        texts = [f"line {i}" for i in range(5)]
        blocks = build_block_annotations(block_ids, bboxes, texts)
        assert len(blocks) == 2
        assert blocks[0].block_id == 0
        assert blocks[1].block_id == 1
        assert len(blocks[0].line_ids) == 2
        assert len(blocks[1].line_ids) == 3

    def test_block_bbox_is_union_of_line_bboxes(self):
        block_ids = [0, 0]
        bboxes = [[0.1, 0.1, 0.5, 0.2], [0.2, 0.2, 0.9, 0.4]]
        texts = ["line 0", "line 1"]
        blocks = build_block_annotations(block_ids, bboxes, texts)
        b = blocks[0].bbox
        assert abs(b[0] - 0.1) < 1e-3
        assert abs(b[1] - 0.1) < 1e-3
        assert abs(b[2] - 0.9) < 1e-3
        assert abs(b[3] - 0.4) < 1e-3

    def test_block_text_is_joined(self):
        block_ids = [0, 0]
        bboxes = [[0.0, 0.0, 0.5, 0.1], [0.0, 0.1, 0.5, 0.2]]
        texts = ["hello", "world"]
        blocks = build_block_annotations(block_ids, bboxes, texts)
        assert blocks[0].text == "hello world"


class TestFilterDegenerate:
    def _make_line_ann(self, bbox, block_id=0, line_id=0):
        return LineAnnotation(text="x", bbox=bbox, block_id=block_id, line_id=line_id)

    def _make_word_ann(self, bbox, line_id=0, word_id=0):
        return WordAnnotation(text="x", bbox=bbox, line_id=line_id, word_id=word_id)

    def test_keeps_large_lines(self):
        ln = self._make_line_ann([0.0, 0.0, 0.5, 0.5])
        wd = self._make_word_ann([0.0, 0.0, 0.4, 0.4], line_id=0)
        lines, words, dl, dw = filter_degenerate([ln], [wd], min_area=16.0, w=100, h=100)
        assert len(lines) == 1
        assert len(words) == 1
        assert dl == 0

    def test_drops_tiny_lines_and_their_words(self):
        ln = self._make_line_ann([0.0, 0.0, 0.01, 0.01])  # 1x1 px at 100x100
        wd = self._make_word_ann([0.0, 0.0, 0.01, 0.01], line_id=0)
        lines, words, dl, dw = filter_degenerate([ln], [wd], min_area=16.0, w=100, h=100)
        assert len(lines) == 0
        assert len(words) == 0
        assert dl == 1

    def test_reassigns_ids_after_drop(self):
        lns = [
            self._make_line_ann([0.0, 0.0, 0.5, 0.5], line_id=0),
            self._make_line_ann([0.0, 0.0, 0.01, 0.01], line_id=1),  # degenerate
            self._make_line_ann([0.5, 0.5, 1.0, 1.0], line_id=2),
        ]
        wds = [
            self._make_word_ann([0.0, 0.0, 0.4, 0.4], line_id=0, word_id=0),
            self._make_word_ann([0.5, 0.5, 0.9, 0.9], line_id=2, word_id=1),
        ]
        lines, words, dl, dw = filter_degenerate(lns, wds, min_area=16.0, w=100, h=100)
        assert len(lines) == 2
        assert [ln.line_id for ln in lines] == [0, 1]
        assert [wd.line_id for wd in words] == [0, 1]
