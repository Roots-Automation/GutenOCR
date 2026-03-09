import argparse
import io
import logging
import os
import sys

import pandas as pd
import torch
from PIL import Image
from diversity.sampler import k_center_greedy
from diversity.visual_utils import compute_embeddings, get_image_files

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_directory_generator(image_dir: str):
    """Yields images from directory and tracks filenames."""
    files = get_image_files(image_dir)
    filenames = []

    def generator():
        for f in files:
            path = os.path.join(image_dir, f)
            try:
                img = Image.open(path)
                filenames.append(f)
                yield img
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}. Skipping.")

    return generator(), len(files), filenames


def load_parquet_generator(parquet_path: str):
    """Yields images from Parquet and tracks filenames."""
    df = pd.read_parquet(parquet_path)
    filenames = []

    def generator():
        for idx, row in df.iterrows():
            try:
                # Standardize image data (handles both raw bytes and dictionary formats)
                img_data = row.get("image")
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    logger.warning(f"Unknown image format at row {idx}")
                    continue

                img = Image.open(io.BytesIO(img_bytes))

                # Attempt to retrieve a meaningful filename from common columns
                fname = row.get("file_name", row.get("filename", f"row_{idx}.png"))
                filenames.append(fname)
                yield img
            except Exception as e:
                logger.warning(f"Error reading row {idx}: {e}. Skipping.")

    return generator(), len(df), filenames


def main():
    parser = argparse.ArgumentParser(description="Rank images by visual diversity.")
    parser.add_argument(
        "input_path", type=str, help="Directory of images OR path to Parquet file"
    )
    parser.add_argument("output_csv", type=str, help="Path for output CSV")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after ranking Top-N images. Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # 1. Detect Input Type & Setup Generator
    if os.path.isdir(args.input_path):
        logger.info(f"Processing directory: {args.input_path}")
        img_gen, count, filenames = load_directory_generator(args.input_path)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".parquet"):
        logger.info(f"Processing Parquet: {args.input_path}")
        img_gen, count, filenames = load_parquet_generator(args.input_path)
    else:
        logger.error("Input path must be a directory or a .parquet file.")
        sys.exit(1)

    if count == 0:
        logger.error("No input records found.")
        sys.exit(1)

    # 2. Compute Embeddings
    logger.info(f"Computing embeddings for ~{count} images...")
    embeddings = compute_embeddings(
        img_gen, total_count=count, batch_size=args.batch_size, device=args.device
    )

    # Note: filenames list is populated as generator is consumed in compute_embeddings
    if len(embeddings) != len(filenames):
        logger.warning(
            f"Mismatch: {len(embeddings)} embeddings vs {len(filenames)} filenames. "
            "Alignment may be compromised if validation skipped items during embedding computation."
        )

    # 3. Rank
    logger.info("Ranking images...")

    limit = args.limit
    if limit is None or limit > len(embeddings):
        limit = len(embeddings)

    # Perform k-center greedy selection without pre-existing indices
    ranking_stats = k_center_greedy(embeddings, k=limit)

    # 4. Save
    results = []
    for stat in ranking_stats:
        idx = stat["index"]
        # Map dataset index back to filename
        stat["filename"] = filenames[idx]
        del stat["index"]
        results.append(stat)

    df = pd.DataFrame(results)

    # Reorder columns for output consistency
    cols = ["rank", "filename", "radius_before", "radius_after", "radius_reduction"]
    df = df[cols]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved ranking to {args.output_csv}")


if __name__ == "__main__":
    main()
