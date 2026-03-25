#!/usr/bin/env python3
"""
Extract text from a HuggingFace dataset and save to a .txt corpus file.

This is a pre-generation tool for preparing text corpus files that SynthDoG
uses as input during document generation.

Characters that cannot be encoded in the target charset are stripped from each
line (not the whole line). Lines that become too short after stripping are
discarded.

Usage:
    # FinePDFs (default)
    uv run python extract_corpus.py

    # English Wikipedia
    uv run python extract_corpus.py \\
        --dataset wikimedia/wikipedia --subset 20231101.en \\
        --output resources/corpus/enwiki.txt

    # Custom dataset / charset
    uv run python extract_corpus.py \\
        --dataset allenai/c4 --subset en \\
        --charset latin-1 --output c4_latin1.txt
"""

import argparse


def extract_text(
    output: str,
    target_samples: int,
    dataset: str,
    subset: str,
    split: str,
    text_field: str,
    charset: str,
    min_length: int,
    max_chars: int | None,
):
    """Extract and charset-clean text from a HuggingFace dataset."""
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading {dataset}/{subset} ({split} split)...")
    ds = load_dataset(dataset, subset, streaming=True, trust_remote_code=False)
    rows = ds[split]

    print(f"Extracting up to {target_samples:,} samples → {output}  (charset={charset}, min_length={min_length})")

    valid_samples = 0
    processed_rows = 0

    pbar = tqdm(total=target_samples, desc="samples", unit="samples")

    with open(output, "w", encoding="utf-8") as f:
        for sample in rows:
            processed_rows += 1

            text = sample.get(text_field, "") or ""
            # Collapse all newlines to spaces; strip leading/trailing whitespace
            text = text.replace("\n", " ").replace("\r", " ").strip()

            if not text:
                continue

            # Strip individual characters that cannot be encoded in the target
            # charset rather than discarding the whole line.
            cleaned = text.encode(charset, errors="ignore").decode(charset)
            # Collapse any runs of whitespace left behind by stripped chars
            cleaned = " ".join(cleaned.split())
            # Truncate to max_chars if requested (at a word boundary)
            if max_chars and len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars].rsplit(" ", 1)[0]

            if len(cleaned) < min_length:
                continue

            f.write(cleaned)
            f.write("\n")
            valid_samples += 1

            pbar.update(1)
            pbar.set_postfix(
                processed=f"{processed_rows:,}",
                rate=f"{valid_samples / processed_rows * 100:.1f}%",
            )

            if valid_samples >= target_samples:
                break

    pbar.close()
    print(f"\nDone. {valid_samples:,} samples written to {output}")
    print(f"Rows processed : {processed_rows:,}")
    print(f"Acceptance rate: {valid_samples / max(processed_rows, 1) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Extract charset-filtered text from a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # FinePDFs (default)
  uv run python extract_corpus.py

  # English Wikipedia
  uv run python extract_corpus.py \\
      --dataset wikimedia/wikipedia --subset 20231101.en \\
      --output resources/corpus/enwiki.txt

  # C4 with Latin-1 charset
  uv run python extract_corpus.py \\
      --dataset allenai/c4 --subset en --charset latin-1
        """,
    )
    parser.add_argument("--output", default="finepdfs_eng_latn_1M.txt", help="Output .txt file path")
    parser.add_argument("--target-samples", type=int, default=1_000_000, help="Number of lines to write (default: 1M)")
    parser.add_argument("--dataset", default="HuggingFaceFW/finepdfs", help="HuggingFace dataset name")
    parser.add_argument("--subset", default="eng_Latn", help="Dataset config/subset name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--text-field", default="text", help="Field name containing text (default: text)")
    parser.add_argument(
        "--charset",
        default="ascii",
        help="Target character encoding. Characters outside this set are stripped per-line (default: ascii).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum line length after charset stripping; shorter lines are discarded (default: 20).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate each line to at most this many characters at a word boundary (default: no limit).",
    )
    args = parser.parse_args()

    extract_text(
        output=args.output,
        target_samples=args.target_samples,
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        text_field=args.text_field,
        charset=args.charset,
        min_length=args.min_length,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
