"""Package raw generated samples into train-NNNNN.tar shards.

Follows the streaming-write pattern from tabmepp/standardize.py:
no global index is built — samples are written directly into the
current shard tar as they are processed.

Input layout:
    raw_dir/
        00000000.jpg
        00000000.json
        00000001.jpg
        ...

Output layout:
    output_dir/
        train-00000.tar
            00000000.jpg
            00000000.json
            ...
        train-00001.tar
        ...
"""

from __future__ import annotations

import argparse
import io
import logging
import tarfile
from math import ceil
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def package_samples(
    raw_dir: Path,
    output_dir: Path,
    *,
    samples_per_shard: int = 1000,
    dry_run: bool = False,
) -> int:
    """Stream raw sample pairs into tar shards.

    Args:
        raw_dir: Directory containing {id}.jpg + {id}.json pairs.
        output_dir: Destination directory for train-NNNNN.tar files.
        samples_per_shard: Number of image+JSON pairs per shard.
        dry_run: If True, report counts without writing any files.

    Returns:
        Number of shards written (0 for dry run).
    """
    img_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    total = len(img_paths)

    if total == 0:
        logger.warning("No images found in %s", raw_dir)
        return 0

    n_shards = ceil(total / samples_per_shard)
    logger.info("%d samples → %d shard(s) of up to %d", total, n_shards, samples_per_shard)

    if dry_run:
        logger.info("Dry run — no files written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_written = 0

    for shard_idx in range(n_shards):
        shard_path = output_dir / f"train-{shard_idx:05d}.tar"
        batch = img_paths[shard_idx * samples_per_shard : (shard_idx + 1) * samples_per_shard]

        with tarfile.open(shard_path, "w") as tf:
            for img_path in tqdm(batch, desc=f"Shard {shard_idx:05d}", leave=False):
                json_path = img_path.with_suffix(".json")
                if not json_path.exists():
                    logger.warning("Missing JSON for %s — skipping", img_path.name)
                    continue

                # Add image
                tf.add(img_path, arcname=img_path.name)

                # Add JSON via BytesIO to avoid path embedding
                json_bytes = json_path.read_bytes()
                info = tarfile.TarInfo(name=json_path.name)
                info.size = len(json_bytes)
                tf.addfile(info, io.BytesIO(json_bytes))

        logger.info("Wrote %s (%d samples)", shard_path.name, len(batch))
        shards_written += 1

    return shards_written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Package raw generated samples into tar shards.")
    parser.add_argument("raw_dir", type=Path, help="Directory containing .jpg + .json pairs.")
    parser.add_argument("output_dir", type=Path, help="Destination directory for tar shards.")
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = package_samples(
        args.raw_dir,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        logger.info("Done. %d shard(s) written.", n)


if __name__ == "__main__":
    main()
