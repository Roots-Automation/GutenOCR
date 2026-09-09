"""Package raw generated samples into WebDataset train-NNNNN.tar shards.

Input layout (from generate.py):
    raw_dir/
        train/
            image_0.jpg
            metadata.jsonl
        validation/
            ...
        test/
            ...

Output layout:
    output_dir/
        train-00000.tar
            image_0.jpg
            image_0.json
            ...
        train-00001.tar
        ...
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import tarfile
from math import ceil
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png"}
_SPLITS = ["train", "validation", "test"]


def _load_jsonl_index(jsonl_path: Path) -> dict[str, dict]:
    """Return {file_name: record} mapping from a metadata.jsonl file."""
    index: dict[str, dict] = {}
    if not jsonl_path.exists():
        return index
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            fname = record.get("file_name", "")
            if fname:
                index[fname] = record
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSONL line in %s", jsonl_path)
    return index


def package_split(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
    *,
    samples_per_shard: int = 1000,
    dry_run: bool = False,
) -> int:
    """Stream one split's image+metadata pairs into tar shards."""
    img_paths = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    total = len(img_paths)

    if total == 0:
        logger.warning("No images found in %s", split_dir)
        return 0

    jsonl_index = _load_jsonl_index(split_dir / "metadata.jsonl")
    n_shards = ceil(total / samples_per_shard)
    prefix = split_name if split_name != "train" else "train"

    logger.info("%s: %d samples → %d shard(s)", split_name, total, n_shards)

    if dry_run:
        logger.info("Dry run — no files written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_written = 0

    for shard_idx in range(n_shards):
        shard_path = output_dir / f"{prefix}-{shard_idx:05d}.tar"
        batch = img_paths[shard_idx * samples_per_shard : (shard_idx + 1) * samples_per_shard]

        with tarfile.open(shard_path, "w") as tf:
            for img_path in tqdm(batch, desc=f"Shard {shard_idx:05d}", leave=False):
                # Add image
                tf.add(img_path, arcname=img_path.name)

                # Build per-image JSON sidecar from JSONL index
                record = jsonl_index.get(img_path.name, {"file_name": img_path.name})
                json_bytes = json.dumps(record, ensure_ascii=False).encode("utf-8")
                info = tarfile.TarInfo(name=img_path.with_suffix(".json").name)
                info.size = len(json_bytes)
                tf.addfile(info, io.BytesIO(json_bytes))

        logger.info("Wrote %s (%d samples)", shard_path.name, len(batch))
        shards_written += 1

    return shards_written


def package_all(
    input_dir: Path,
    output_dir: Path,
    *,
    samples_per_shard: int = 1000,
    dry_run: bool = False,
) -> int:
    """Package all splits from a generate.py output directory."""
    total_shards = 0
    for split in _SPLITS:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        split_out = output_dir / split
        total_shards += package_split(
            split_dir,
            split_out,
            split_name=split,
            samples_per_shard=samples_per_shard,
            dry_run=dry_run,
        )
    return total_shards


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Package generate.py output into WebDataset tar shards.")
    parser.add_argument(
        "input_dir", type=Path, help="Directory produced by generate.py (contains train/validation/test)."
    )
    parser.add_argument("output_dir", type=Path, help="Destination for tar shards.")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = package_all(
        args.input_dir,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        logger.info("Done. %d shard(s) written.", n)


if __name__ == "__main__":
    main()
