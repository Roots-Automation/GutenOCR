"""Integration tests for generate.py.

Verifies the acceptance criterion: --num-samples 4 --seed 42 produces
4 images and 4 JSON sidecars with valid structure.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from content_distribution import ContentDistribution
from generate import generate_sample
from otsl import NL, OTSL_VOCAB, validate_otsl

OTSL_CELL_TOKENS = OTSL_VOCAB - {NL}  # fcel, ecel, lcel, ucel, xcel


def _make_sample(tmp_path: Path, seed: int, sample_id: int = 0) -> tuple[Path, Path]:
    rng = random.Random(seed)
    dist = ContentDistribution()
    paths = generate_sample(
        sample_id,
        rng,
        dist,
        min_rows=2,
        max_rows=6,
        min_cols=2,
        max_cols=5,
        span_prob=0.2,
        augment=False,
        augment_count=0,
        output_dir=tmp_path,
    )
    jpg = next(p for p in paths if p.suffix == ".jpg")
    jsn = next(p for p in paths if p.suffix == ".json")
    return jpg, jsn


class TestGenerateSampleAcceptance:
    def test_four_samples_seed_42(self, tmp_path):
        """Acceptance gate: 4 samples with seed 42 produce 4 jpg + 4 json."""
        rng = random.Random(42)
        dist = ContentDistribution()
        for i in range(4):
            generate_sample(
                i,
                rng,
                dist,
                min_rows=2,
                max_rows=6,
                min_cols=2,
                max_cols=5,
                span_prob=0.2,
                augment=False,
                augment_count=0,
                output_dir=tmp_path,
            )
        jpgs = list(tmp_path.glob("*.jpg"))
        jsons = list(tmp_path.glob("*.json"))
        assert len(jpgs) == 4
        assert len(jsons) == 4

    def test_sidecar_has_required_keys(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=1)
        data = json.loads(jsn.read_text())
        assert "image" in data
        assert "text" in data
        assert "table" in data

        assert {"path", "width", "height"} <= data["image"].keys()
        assert {"words", "lines"} <= data["text"].keys()
        assert {"otsl", "html", "rows", "cols"} <= data["table"].keys()

    def test_image_dimensions_match_sidecar(self, tmp_path):
        jpg, jsn = _make_sample(tmp_path, seed=2)
        data = json.loads(jsn.read_text())
        img = Image.open(jpg)
        assert img.width == data["image"]["width"]
        assert img.height == data["image"]["height"]

    def test_otsl_tokens_all_valid(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=3)
        data = json.loads(jsn.read_text())
        otsl = data["table"]["otsl"]
        for t in otsl.split():
            assert t in OTSL_VOCAB, f"Unknown token: {t!r}"

    def test_otsl_nl_count_matches_rows(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=4)
        data = json.loads(jsn.read_text())
        otsl = data["table"]["otsl"]
        rows = data["table"]["rows"]
        assert otsl.split().count(NL) == rows

    def test_otsl_validates(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=5)
        data = json.loads(jsn.read_text())
        validate_otsl(
            data["table"]["otsl"],
            rows=data["table"]["rows"],
            cols=data["table"]["cols"],
        )

    def test_word_boxes_are_lists_of_four(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=6)
        data = json.loads(jsn.read_text())
        for w in data["text"]["words"]:
            assert "text" in w
            assert "box" in w
            assert len(w["box"]) == 4

    def test_html_contains_table_tag(self, tmp_path):
        _, jsn = _make_sample(tmp_path, seed=7)
        data = json.loads(jsn.read_text())
        assert "<table>" in data["table"]["html"]
        assert "</table>" in data["table"]["html"]

    def test_deterministic_with_same_seed(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()

        def _run(out):
            rng = random.Random(42)
            dist = ContentDistribution()
            generate_sample(
                0,
                rng,
                dist,
                min_rows=2,
                max_rows=6,
                min_cols=2,
                max_cols=5,
                span_prob=0.2,
                augment=False,
                augment_count=0,
                output_dir=out,
            )

        _run(out1)
        _run(out2)

        j1 = json.loads((out1 / "00000000.json").read_text())
        j2 = json.loads((out2 / "00000000.json").read_text())
        assert j1["table"]["otsl"] == j2["table"]["otsl"]
        assert j1["image"]["width"] == j2["image"]["width"]
        assert j1["image"]["height"] == j2["image"]["height"]

    def test_augment_produces_extra_files(self, tmp_path):
        rng = random.Random(42)
        dist = ContentDistribution()
        paths = generate_sample(
            0,
            rng,
            dist,
            min_rows=2,
            max_rows=6,
            min_cols=2,
            max_cols=5,
            span_prob=0.2,
            augment=True,
            augment_count=2,
            output_dir=tmp_path,
        )
        jpgs = [p for p in paths if p.suffix == ".jpg"]
        jsons = [p for p in paths if p.suffix == ".json"]
        # 1 base + 2 augmented
        assert len(jpgs) == 3
        assert len(jsons) == 3
