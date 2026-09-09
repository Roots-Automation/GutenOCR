"""Unit and integration tests for the index_definition generator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elements.entry import compute_word_positions, wrap_text
from elements.layout import ColumnCursor, ColumnRect, sample_layout

# ── wrap_text ────────────────────────────────────────────────────────────────


class MockFont:
    """Minimal font mock: each character is 10px wide."""

    def getlength(self, text):
        return float(len(text) * 10)

    def getmetrics(self):
        return 14, 2


class TestWrapText:
    def setup_method(self):
        self.font = MockFont()

    def test_single_word_fits(self):
        lines = wrap_text("hello", self.font, max_width_px=100)
        assert lines == ["hello"]

    def test_two_words_fit_on_one_line(self):
        # "hello world" = 5+1+5 = 11 chars * 10 = 110px <= 120
        lines = wrap_text("hello world", self.font, max_width_px=120)
        assert lines == ["hello world"]

    def test_forces_break_when_too_wide(self):
        # "hello world" = 110px > 60px → split
        lines = wrap_text("hello world", self.font, max_width_px=60)
        assert len(lines) == 2
        assert lines[0] == "hello"
        assert lines[1] == "world"

    def test_empty_string_returns_empty(self):
        lines = wrap_text("", self.font, max_width_px=100)
        assert lines == []

    def test_multiple_breaks(self):
        # Each word is 5 chars * 10px = 50px; width=60 fits one word
        lines = wrap_text("alpha beta gamma", self.font, max_width_px=60)
        assert lines == ["alpha", "beta", "gamma"]

    def test_long_single_word_is_truncated(self):
        # A word wider than max_width is truncated to fit (with ellipsis)
        lines = wrap_text("superlongword", self.font, max_width_px=50)
        assert len(lines) == 1
        assert len(lines[0]) < len("superlongword")
        assert lines[0].endswith("…")


# ── compute_word_positions ────────────────────────────────────────────────────


class TestComputeWordPositions:
    def setup_method(self):
        self.font = MockFont()

    def test_single_word(self):
        x1s, x2s = compute_word_positions("hello", self.font, line_x=0)
        assert len(x1s) == 1
        assert x1s[0] == 0
        assert x2s[0] == 50  # 5 chars * 10px

    def test_two_words_non_overlapping(self):
        x1s, x2s = compute_word_positions("ab cd", self.font, line_x=0)
        assert len(x1s) == 2
        assert x1s[0] < x2s[0]
        assert x1s[1] >= x2s[0]  # second word starts at or after first ends
        assert x1s[1] < x2s[1]

    def test_offset_by_line_x(self):
        x1s_0, _ = compute_word_positions("hello", self.font, line_x=0)
        x1s_100, _ = compute_word_positions("hello", self.font, line_x=100)
        assert x1s_100[0] == x1s_0[0] + 100

    def test_empty_line(self):
        x1s, x2s = compute_word_positions("", self.font, line_x=0)
        assert x1s == []
        assert x2s == []

    def test_x1_lt_x2_for_all_words(self):
        x1s, x2s = compute_word_positions("one two three", self.font, line_x=5)
        for x1, x2 in zip(x1s, x2s):
            assert x1 < x2


# ── ColumnCursor ──────────────────────────────────────────────────────────────


class TestColumnCursor:
    def _col(self, y1=0, y2=100):
        return ColumnRect(col_idx=0, x1=0, y1=y1, x2=100, y2=y2)

    def test_initial_y_equals_col_y1(self):
        cursor = ColumnCursor(self._col(y1=20))
        assert cursor.current_y == 20

    def test_advance_moves_y(self):
        cursor = ColumnCursor(self._col())
        cursor.advance(30)
        assert cursor.current_y == 30

    def test_remaining_height_decreases(self):
        cursor = ColumnCursor(self._col(y1=0, y2=100))
        assert cursor.remaining_height == 100
        cursor.advance(40)
        assert cursor.remaining_height == 60

    def test_not_full_initially(self):
        cursor = ColumnCursor(self._col())
        assert not cursor.is_full

    def test_is_full_when_exhausted(self):
        cursor = ColumnCursor(self._col(y1=0, y2=50))
        cursor.advance(50)
        assert cursor.is_full

    def test_is_full_when_over_exhausted(self):
        cursor = ColumnCursor(self._col(y1=0, y2=50))
        cursor.advance(80)
        assert cursor.is_full


# ── sample_layout ─────────────────────────────────────────────────────────────


class TestSampleLayout:
    def test_returns_correct_col_count(self):
        config = {
            "layout": {
                "num_cols_choices": [3],
                "num_cols_weights": [1],
                "margin": [0.05, 0.05],
                "gutter_frac": [0.02, 0.02],
            }
        }
        rng = np.random.default_rng(0)
        cols, _ = sample_layout(800, 1000, config, rng)
        assert len(cols) == 3

    def test_columns_do_not_overlap(self):
        config = {
            "layout": {
                "num_cols_choices": [4],
                "num_cols_weights": [1],
                "margin": [0.05, 0.05],
                "gutter_frac": [0.02, 0.02],
            }
        }
        rng = np.random.default_rng(42)
        cols, _ = sample_layout(800, 1000, config, rng)
        for i in range(len(cols) - 1):
            assert cols[i].x2 <= cols[i + 1].x1 + 1  # allow 1px rounding

    def test_single_column_fills_content_width(self):
        config = {
            "layout": {
                "num_cols_choices": [1],
                "num_cols_weights": [1],
                "margin": [0.0, 0.0],
                "gutter_frac": [0.02, 0.02],
            }
        }
        rng = np.random.default_rng(7)
        cols, (cx1, cy1, cx2, cy2) = sample_layout(800, 1000, config, rng)
        assert len(cols) == 1
        assert cols[0].x1 == cx1
        # x2 may differ by rounding
        assert abs(cols[0].x2 - cx2) <= 2


# ── DefinitionCorpus ──────────────────────────────────────────────────────────


class TestDefinitionCorpus:
    def test_load_returns_nonempty(self):
        from corpus.definitions import DefinitionCorpus

        corpus = DefinitionCorpus.load(min_definition_length=10)
        assert len(corpus) > 1000

    def test_shuffled_batch_returns_all_entries(self):
        from corpus.definitions import DefinitionCorpus

        corpus = DefinitionCorpus.load(min_definition_length=10)
        rng = np.random.default_rng(0)
        batch = corpus.shuffled_batch(rng)
        assert len(batch) == len(corpus)

    def test_shuffled_batch_is_seeded(self):
        from corpus.definitions import DefinitionCorpus

        corpus = DefinitionCorpus.load(min_definition_length=10)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        b1 = corpus.shuffled_batch(rng1)
        b2 = corpus.shuffled_batch(rng2)
        assert b1[0] == b2[0]
        assert b1[-1] == b2[-1]

    def test_shuffled_batch_different_seeds_differ(self):
        from corpus.definitions import DefinitionCorpus

        corpus = DefinitionCorpus.load(min_definition_length=10)
        b1 = corpus.shuffled_batch(np.random.default_rng(1))
        b2 = corpus.shuffled_batch(np.random.default_rng(2))
        assert b1[0] != b2[0]

    def test_all_definitions_meet_min_length(self):
        from corpus.definitions import DefinitionCorpus

        corpus = DefinitionCorpus.load(min_definition_length=10)
        for _, defn in corpus._entries:
            assert len(defn) >= 10


# ── Integration tests (marked slow) ───────────────────────────────────────────


@pytest.fixture(scope="module")
def integration_generator():
    import yaml
    from generate import IndexDefinitionGenerator

    config_path = Path(__file__).resolve().parent.parent / "config" / "config_base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return IndexDefinitionGenerator(config)


@pytest.mark.slow
class TestGenerateIntegration:
    def test_generate_returns_dict(self, integration_generator):
        data = integration_generator.generate(seed=42)
        assert data is not None
        assert "image" in data
        assert "lines" in data
        assert "words" in data
        assert "blocks" in data
        assert "metrics" in data
        assert "generation_params" in data

    def test_image_is_uint8_rgb(self, integration_generator):
        data = integration_generator.generate(seed=43)
        assert data is not None
        img = data["image"]
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_line_bboxes_normalized(self, integration_generator):
        data = integration_generator.generate(seed=44)
        assert data is not None
        for ln in data["lines"]:
            x1, y1, x2, y2 = ln.bbox
            assert 0.0 <= x1 <= 1.0
            assert 0.0 <= y1 <= 1.0
            assert 0.0 <= x2 <= 1.0
            assert 0.0 <= y2 <= 1.0
            assert x1 <= x2
            assert y1 <= y2

    def test_word_bboxes_within_line_bboxes(self, integration_generator):
        data = integration_generator.generate(seed=45)
        assert data is not None
        line_map = {ln.line_id: ln.bbox for ln in data["lines"]}
        tol = 2e-3
        for wd in data["words"]:
            if wd.line_id not in line_map:
                continue
            lb = line_map[wd.line_id]
            wb = wd.bbox
            assert wb[0] >= lb[0] - tol
            assert wb[2] <= lb[2] + tol

    def test_block_covers_member_lines(self, integration_generator):
        data = integration_generator.generate(seed=46)
        assert data is not None
        line_map = {ln.line_id: ln.bbox for ln in data["lines"]}
        for blk in data["blocks"]:
            bx1, by1, bx2, by2 = blk.bbox
            for lid in blk.line_ids:
                if lid not in line_map:
                    continue
                lx1, ly1, lx2, ly2 = line_map[lid]
                assert lx1 >= bx1 - 1e-3
                assert ly1 >= by1 - 1e-3
                assert lx2 <= bx2 + 1e-3
                assert ly2 <= by2 + 1e-3

    def test_same_seed_identical_structure(self, integration_generator):
        """Same seed should produce same layout structure and annotation content."""
        d1 = integration_generator.generate(seed=99)
        d2 = integration_generator.generate(seed=99)
        assert d1 is not None and d2 is not None
        assert d1["image"].shape == d2["image"].shape
        p1, p2 = d1["generation_params"], d2["generation_params"]
        assert p1["mode"] == p2["mode"]
        assert p1["num_cols"] == p2["num_cols"]
        assert p1["font_family"] == p2["font_family"]
        assert len(d1["lines"]) == len(d2["lines"])
        assert len(d1["words"]) == len(d2["words"])
        assert len(d1["blocks"]) == len(d2["blocks"])
        for l1, l2 in zip(d1["lines"], d2["lines"]):
            assert l1.text == l2.text
            assert l1.bbox == l2.bbox

    def test_different_seeds_different_images(self, integration_generator):
        d1 = integration_generator.generate(seed=100)
        d2 = integration_generator.generate(seed=200)
        assert d1 is not None and d2 is not None
        assert not np.array_equal(d1["image"], d2["image"])

    def test_quality_metrics_keys(self, integration_generator):
        data = integration_generator.generate(seed=47)
        assert data is not None
        metrics = data["metrics"]
        for key in ("word_count", "line_count", "sharpness", "image_size"):
            assert key in metrics

    def test_save_writes_jpeg_and_jsonl(self, integration_generator, tmp_path):
        for seed in range(100):
            data = integration_generator.generate(seed=seed)
            if data is None:
                continue
            ok = integration_generator.save(str(tmp_path), data, idx=0)
            if ok:
                break
        found_jpg = list(tmp_path.rglob("*.jpg"))
        found_jsonl = list(tmp_path.rglob("metadata.jsonl"))
        assert len(found_jpg) >= 1
        assert len(found_jsonl) >= 1
