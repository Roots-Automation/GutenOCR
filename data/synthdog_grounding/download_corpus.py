"""Download a local text corpus from HuggingFaceFW/finepdfs for offline generation.

Usage:
    uv run python download_corpus.py                         # default: 50k docs → resources/corpus/finepdfs_en.txt
    uv run python download_corpus.py --docs 100000           # larger corpus
    uv run python download_corpus.py --out resources/corpus/custom.txt
"""

import argparse
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download finepdfs corpus to a local text file.")
    p.add_argument("--dataset", default="HuggingFaceFW/finepdfs", help="HuggingFace dataset name")
    p.add_argument("--split", default="train")
    p.add_argument("--docs", type=int, default=50_000, help="Number of documents to download")
    p.add_argument("--out", default="resources/corpus/finepdfs_en.txt", help="Output file path")
    p.add_argument("--charset", default="ascii", help="Strip chars not encodable in this charset (default: ascii)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ({args.split}) in streaming mode...")
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    saved = 0
    skipped = 0
    total_chars = 0

    with open(out_path, "w", encoding="utf-8") as fp:
        for sample in ds:
            text = sample.get("text") or sample.get("content") or ""
            if not text:
                skipped += 1
                continue

            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Filter to target charset
            if args.charset:
                text = text.encode(args.charset, errors="ignore").decode(args.charset)

            if not text:
                skipped += 1
                continue

            fp.write(text)
            fp.write("\n")
            total_chars += len(text) + 1
            saved += 1

            if saved % 5_000 == 0:
                mb = total_chars / 1_048_576
                print(f"  {saved:,}/{args.docs:,} docs  ({mb:.1f} MB)", flush=True)

            if saved >= args.docs:
                break

    mb = total_chars / 1_048_576
    print(f"\nDone. {saved:,} docs, {skipped:,} skipped, {mb:.1f} MB → {out_path}")
    print("\nTo use in config, replace the 'text' block with:")
    print("    text:")
    print("      type: file")
    print("      path: resources/corpus/finepdfs_en.txt")


if __name__ == "__main__":
    main()
