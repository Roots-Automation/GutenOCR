"""Unit tests for packaging.py."""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from packaging import package_samples


def _write_fake_sample(directory: Path, idx: int) -> None:
    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    (directory / f"{idx:08d}.jpg").write_bytes(buf.getvalue())

    sidecar = {
        "image": {"path": f"{idx:08d}.jpg", "width": 32, "height": 32},
        "text": {"words": [], "lines": []},
        "table": {"otsl": "FCEL NL", "html": "<table></table>", "rows": 1, "cols": 1},
    }
    (directory / f"{idx:08d}.json").write_text(json.dumps(sidecar))


class TestPackageSamples:
    def test_empty_dir_returns_zero(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        out = tmp_path / "out"
        n = package_samples(raw, out)
        assert n == 0

    def test_single_shard(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(4):
            _write_fake_sample(raw, i)

        out = tmp_path / "out"
        n = package_samples(raw, out, samples_per_shard=10)
        assert n == 1
        shards = list(out.glob("train-*.tar"))
        assert len(shards) == 1

    def test_multiple_shards(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(5):
            _write_fake_sample(raw, i)

        out = tmp_path / "out"
        n = package_samples(raw, out, samples_per_shard=2)
        assert n == 3
        assert len(list(out.glob("train-*.tar"))) == 3

    def test_tar_contains_jpg_and_json(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(4):
            _write_fake_sample(raw, i)

        out = tmp_path / "out"
        package_samples(raw, out, samples_per_shard=10)

        shard = out / "train-00000.tar"
        with tarfile.open(shard) as tf:
            names = tf.getnames()
        assert len(names) == 8  # 4 jpg + 4 json
        assert sum(1 for n in names if n.endswith(".jpg")) == 4
        assert sum(1 for n in names if n.endswith(".json")) == 4

    def test_shard_naming(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(3):
            _write_fake_sample(raw, i)

        out = tmp_path / "out"
        package_samples(raw, out, samples_per_shard=1)
        names = sorted(p.name for p in out.glob("*.tar"))
        assert names == ["train-00000.tar", "train-00001.tar", "train-00002.tar"]

    def test_dry_run_writes_nothing(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for i in range(4):
            _write_fake_sample(raw, i)

        out = tmp_path / "out"
        n = package_samples(raw, out, dry_run=True)
        assert n == 0
        assert not out.exists() or not list(out.glob("*.tar"))

    def test_missing_json_skips_gracefully(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_fake_sample(raw, 0)
        # Write an image with no matching JSON
        img = Image.new("RGB", (32, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        (raw / "00000001.jpg").write_bytes(buf.getvalue())

        out = tmp_path / "out"
        package_samples(raw, out, samples_per_shard=10)
        shard = out / "train-00000.tar"
        with tarfile.open(shard) as tf:
            names = tf.getnames()
        # Only the paired sample should appear
        assert "00000000.jpg" in names
        assert "00000001.jpg" not in names
