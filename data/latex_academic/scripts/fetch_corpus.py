"""Extract clean sentences from peS2o v2 into resources/corpus_sentences.txt.

peS2o is published by Allen AI under ODC-By (Open Database Commons Attribution).
Derived sentence extracts may be redistributed with this attribution notice:

    Sentences extracted from peS2o v2 (Soldaini & Lo, 2023).
    Source: https://huggingface.co/datasets/allenai/peS2o
    License: ODC-By 1.0

Usage:
    uv run python scripts/fetch_corpus.py
    uv run python scripts/fetch_corpus.py --shards 3 --target 100000
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_OUT = Path(__file__).parent.parent / "resources" / "corpus_sentences.txt"

# peS2o v2 shards — 20 total, first 10 are ~1.5 GB each (S2ORC-style)
_SHARD_URL_TEMPLATE = "hf://datasets/allenai/peS2o/data/v2/train-{n:05d}-of-00020.json.gz"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(\[])")
_GARBAGE = re.compile(
    r"[<>{}\\|@#^*=\[\]~`]"  # LaTeX/HTML/markdown artifacts
    r"|https?://"  # URLs
    r"|\d{4,}"  # long runs of digits (accession numbers etc.)
    r"|[^\x00-\x7F]",  # non-ASCII (keeps corpus latin-script only)
)


def _clean_sentence(s: str) -> str | None:
    s = s.strip()
    if len(s) < 30 or len(s) > 250:
        return None
    # Must start with capital letter or digit
    if not (s[0].isupper() or s[0].isdigit()):
        return None
    # Must end with terminal punctuation
    if s[-1] not in ".!?)":
        return None
    # Reject sentences with garbage patterns
    if _GARBAGE.search(s):
        return None
    # Reject if >30% digits (data tables leaking in)
    digits = sum(c.isdigit() for c in s)
    if digits / len(s) > 0.30:
        return None
    return s


def _extract_sentences(text: str) -> list[str]:
    # Strip leading title-like line (often the paper title repeated at the top)
    text = text.strip()
    # Remove ALLCAPS section headings (METHODS, RESULTS, CONCLUSIONS etc.)
    text = re.sub(r"\n[A-Z][A-Z\s/]{3,}\n", " ", text)
    parts = _SENTENCE_SPLIT.split(text)
    out = []
    for part in parts:
        s = _clean_sentence(part)
        if s:
            out.append(s)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch peS2o sentences into corpus_sentences.txt")
    parser.add_argument("--shards", type=int, default=2, help="Number of peS2o v2 shards to process (default: 2)")
    parser.add_argument(
        "--target", type=int, default=50_000, help="Stop after this many clean sentences (default: 50000)"
    )
    parser.add_argument("--output", type=Path, default=_OUT, help="Output path")
    args = parser.parse_args()

    try:
        import datasets
    except ImportError:
        logger.error("datasets library not installed. Run: uv add datasets huggingface-hub")
        sys.exit(1)

    sentences: list[str] = []
    seen: set[str] = set()

    for shard_idx in range(args.shards):
        if len(sentences) >= args.target:
            break

        url = _SHARD_URL_TEMPLATE.format(n=shard_idx)
        logger.info("Processing shard %d/%d: %s", shard_idx + 1, args.shards, url)

        try:
            ds = datasets.load_dataset(
                "json",
                data_files={"train": url},
                streaming=True,
                split="train",
            )
        except Exception as exc:
            logger.warning("Failed to load shard %d: %s", shard_idx, exc)
            continue

        doc_count = 0
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            for sent in _extract_sentences(text):
                if sent not in seen:
                    seen.add(sent)
                    sentences.append(sent)
            doc_count += 1
            if doc_count % 5000 == 0:
                logger.info("  shard %d: %d docs, %d sentences so far", shard_idx, doc_count, len(sentences))
            if len(sentences) >= args.target:
                break

        logger.info("Shard %d done: %d docs, %d total sentences", shard_idx, doc_count, len(sentences))

    logger.info("Writing %d sentences to %s", len(sentences), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences), encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()
