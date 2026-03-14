#!/usr/bin/env python3
"""
Extract text from FinePDFs dataset and save to a .txt file.
This script loads the first N rows from the specified subset of FinePDFs.
"""

import argparse


def extract_finepdfs_text(
    output: str,
    target_samples: int,
    dataset: str,
    subset: str,
    split: str,
):
    """Extract text from FinePDFs dataset and save to txt file."""
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading {dataset} dataset ({subset} subset)...")

    ds = load_dataset(dataset, subset, streaming=True)
    train_ds = ds[split]

    print(f"Extracting {target_samples:,} ASCII-only samples to {output}...")

    valid_samples = 0
    processed_rows = 0

    pbar = tqdm(total=target_samples, desc="Valid samples", unit="samples")

    with open(output, "w", encoding="utf-8") as f:
        for sample in train_ds:
            processed_rows += 1

            text = sample.get("text", "")
            cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()

            if not cleaned_text:
                continue

            try:
                cleaned_text.encode("ascii")
            except UnicodeEncodeError:
                continue

            f.write(cleaned_text)
            f.write("\n")
            valid_samples += 1

            pbar.update(1)
            pbar.set_postfix(
                {
                    "processed_rows": f"{processed_rows:,}",
                    "success_rate": f"{valid_samples / processed_rows * 100:.1f}%",
                }
            )

            if valid_samples >= target_samples:
                break

    pbar.close()
    print(f"Extraction complete! Saved {valid_samples:,} ASCII-only samples to {output}")
    print(f"Total rows processed: {processed_rows:,}")
    print(f"Success rate: {valid_samples / processed_rows * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Extract text from a HuggingFace dataset into a .txt file")
    parser.add_argument("--output", default="finepdfs_eng_latn_1M.txt", help="Output text file path")
    parser.add_argument("--target-samples", type=int, default=1_000_000, help="Number of ASCII-only samples to extract")
    parser.add_argument("--dataset", default="HuggingFaceFW/finepdfs", help="HuggingFace dataset name")
    parser.add_argument("--subset", default="eng_Latn", help="Dataset subset/config name")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    args = parser.parse_args()

    extract_finepdfs_text(
        output=args.output,
        target_samples=args.target_samples,
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
    )


if __name__ == "__main__":
    main()
