"""CLI entry point for synthetic LaTeX math formula generation.

Produces a JSON object {"0": "<latex>", "1": "<latex>", ...} compatible with
GutenOCR/data/grounded_latex/generate_equations.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from roots_ocr.utils.cli import setup_logging
except ImportError:

    def setup_logging(**_kwargs: object) -> None:  # type: ignore[misc]
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


from .corpus import generate
from .domains import DEFAULT_WEIGHTS, DOMAIN_TAGS, GENERATORS

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for generating synthetic LaTeX math formulas."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic LaTeX mathematical formula strings for OCR training. "
            "Output is compatible with GutenOCR/data/grounded_latex/generate_equations.py."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100_000,
        metavar="N",
        help="Number of unique formulas to generate (default: 100000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_WEIGHTS.keys()),
        choices=list(GENERATORS.keys()),
        metavar="DOMAIN",
        help=(f"Domains to include. Default: all. Choices: {', '.join(GENERATORS.keys())}"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--display-fraction",
        type=float,
        default=0.20,
        metavar="F",
        help="Fraction of bare formulas wrapped in display-math environments (default: 0.20).",
    )
    parser.add_argument(
        "--inline-fraction",
        type=float,
        default=0.10,
        metavar="F",
        help="Fraction of bare formulas wrapped in inline $...$ delimiters (default: 0.10).",
    )
    all_tags = sorted({t for tags in DOMAIN_TAGS.values() for t in tags})
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help=(f"Include only domains with any of these tags. Available: {', '.join(all_tags)}."),
    )
    parser.add_argument(
        "--exclude-tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Exclude domains with any of these tags.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="Output metadata dicts (formula + domain) instead of bare strings.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.display_fraction <= 1.0:
        logger.error("--display-fraction must be in [0, 1]")
        sys.exit(1)

    if not 0.0 <= args.inline_fraction <= 1.0:
        logger.error("--inline-fraction must be in [0, 1]")
        sys.exit(1)

    logger.info(
        "Generating %d formulas | domains: %s | seed: %s",
        args.count,
        ", ".join(args.domains),
        args.seed,
    )

    formulas = generate(
        count=args.count,
        domains=args.domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=args.seed,
        display_fraction=args.display_fraction,
        inline_fraction=args.inline_fraction,
        tags=args.tags,
        exclude_tags=args.exclude_tags,
        include_metadata=args.metadata,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(formulas, fh, ensure_ascii=False)

    logger.info("Wrote %d formulas to %s", len(formulas), args.output)
