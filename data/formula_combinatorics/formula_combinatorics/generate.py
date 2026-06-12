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
from .domains import DEFAULT_WEIGHTS, DOMAIN_TAGS, GENERATORS, PACK_HASHES, TEMPLATES
from .engine.symbol_inventory import SYMBOL_STRATA

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
        required="--content-hash" not in sys.argv,
        metavar="PATH",
        help="Output JSON file path. Not required when --content-hash is used.",
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
        "--difficulty",
        nargs="+",
        default=None,
        choices=["elementary", "undergraduate", "graduate", "research"],
        metavar="LEVEL",
        help=(
            "Include only domains at the given difficulty level(s). "
            "Choices: elementary, undergraduate, graduate, research."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        metavar="N",
        help="Keep only templates whose LaTeX brace-nesting depth is ≤ N.",
    )
    parser.add_argument(
        "--length-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Keep only templates whose character-length proxy falls in [MIN, MAX].",
    )
    _stratum_choices = sorted(SYMBOL_STRATA.keys())
    parser.add_argument(
        "--symbol-tier",
        nargs="+",
        default=None,
        choices=_stratum_choices,
        metavar="TIER",
        help=(
            f"Keep only templates that exercise at least one of the given symbol strata. "
            f"Choices: {', '.join(_stratum_choices)}."
        ),
    )
    parser.add_argument(
        "--hold-out-domains",
        nargs="+",
        default=None,
        choices=list(GENERATORS.keys()),
        metavar="DOMAIN",
        help="Exclude these domains from generation (domain-level hold-out for split construction).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "JSON file mapping domain names to sampling weights. "
            "Values are merged over the defaults; unknown domains are ignored."
        ),
    )
    parser.add_argument(
        "--weight",
        nargs="+",
        default=None,
        metavar="DOMAIN=VALUE",
        help=(
            "Inline per-domain weight override(s) in 'domain=value' format "
            "(e.g. --weight algebra=10.0 calculus=0.5). "
            "Applied after --weights; both may be used together."
        ),
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="Output metadata dicts (formula + domain + template_name) instead of bare strings.",
    )

    render_group = parser.add_argument_group("render gate (post-generation)")
    render_group.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Run the render gate: compile every formula and keep only those that render cleanly.",
    )
    render_group.add_argument(
        "--render-engine",
        choices=["katex", "tex", "two-stage"],
        default="two-stage",
        metavar="ENGINE",
        help="Rendering engine: katex (fast pre-filter), tex (lualatex, authoritative), "
        "or two-stage (katex→lualatex, default).",
    )
    render_group.add_argument(
        "--render-output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for rendered PNG images (default: <output_stem>_images/).",
    )
    render_group.add_argument(
        "--render-reject-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSONL file for failed formulas (default: <output_stem>_rejects.jsonl).",
    )
    render_group.add_argument(
        "--render-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers for the TeX stage (default: 4).",
    )
    render_group.add_argument(
        "--render-dpi",
        type=int,
        default=150,
        metavar="N",
        help="PNG resolution for the TeX stage in DPI (default: 150).",
    )
    render_group.add_argument(
        "--katex-node-bin",
        default="node",
        metavar="PATH",
        help="Path to the node executable (default: node).",
    )
    render_group.add_argument(
        "--tex-bin",
        default="lualatex",
        metavar="PATH",
        help="Path to lualatex or xelatex (default: lualatex).",
    )
    parser.add_argument(
        "--content-hash",
        action="store_true",
        default=False,
        help=(
            "Print the aggregate SHA-256 content hash of all loaded TOML template packs "
            "and exit.  Useful for pinning corpus snapshots in benchmark reproducibility records."
        ),
    )
    args = parser.parse_args()

    if args.content_hash:
        import hashlib

        if not PACK_HASHES:
            print("(no TOML packs loaded — all domains are Python modules)")
        else:
            combined = hashlib.sha256("|".join(f"{k}:{v}" for k, v in sorted(PACK_HASHES.items())).encode()).hexdigest()
            for domain, sha in sorted(PACK_HASHES.items()):
                print(f"  {domain}: {sha}")
            print(f"aggregate: {combined}")
        sys.exit(0)

    if not 0.0 <= args.display_fraction <= 1.0:
        logger.error("--display-fraction must be in [0, 1]")
        sys.exit(1)

    if not 0.0 <= args.inline_fraction <= 1.0:
        logger.error("--inline-fraction must be in [0, 1]")
        sys.exit(1)

    if args.length_range is not None and args.length_range[0] > args.length_range[1]:
        logger.error("--length-range MIN must be ≤ MAX")
        sys.exit(1)

    # Build merged weight dict from defaults + file overrides + inline overrides.
    weight_overrides: dict[str, float] = {}
    if args.weights is not None:
        try:
            with open(args.weights, encoding="utf-8") as fh:
                weight_overrides = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load --weights file %s: %s", args.weights, exc)
            sys.exit(1)
        unknown = set(weight_overrides) - set(GENERATORS)
        if unknown:
            logger.warning("Unknown domains in --weights file (ignored): %s", sorted(unknown))
            weight_overrides = {k: v for k, v in weight_overrides.items() if k in GENERATORS}
    if args.weight is not None:
        for kv in args.weight:
            if "=" not in kv:
                logger.error("--weight entries must be in 'domain=value' format, got: %r", kv)
                sys.exit(1)
            k, v = kv.split("=", 1)
            k = k.strip()
            if k not in GENERATORS:
                logger.warning("Unknown domain in --weight (ignored): %r", k)
                continue
            try:
                weight_overrides[k] = float(v)
            except ValueError:
                logger.error("--weight value for %r is not a float: %r", k, v)
                sys.exit(1)
    merged_weights = {**DEFAULT_WEIGHTS, **weight_overrides}

    # Determine which params require per-template access.
    need_templates = any(f is not None for f in (args.max_depth, args.length_range, args.symbol_tier))

    logger.info(
        "Generating %d formulas | domains: %s | seed: %s",
        args.count,
        ", ".join(args.domains),
        args.seed,
    )

    # The render gate requires metadata (domain + template_name) for the reject log.
    need_metadata = args.metadata or args.render
    try:
        formulas = generate(
            count=args.count,
            domains=args.domains,
            generators=GENERATORS,
            weights=merged_weights,
            seed=args.seed,
            display_fraction=args.display_fraction,
            inline_fraction=args.inline_fraction,
            tags=args.tags,
            exclude_tags=args.exclude_tags,
            include_metadata=need_metadata,
            pack_hashes=PACK_HASHES if need_metadata else None,
            difficulty=args.difficulty,
            max_depth=args.max_depth,
            length_range=tuple(args.length_range) if args.length_range else None,
            symbol_tiers=set(args.symbol_tier) if args.symbol_tier else None,
            hold_out_domains=args.hold_out_domains,
            templates=TEMPLATES if need_templates else None,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.render:
        from .engine.render import render_corpus

        render_out = args.render_output or args.output.parent / (args.output.stem + "_images")
        reject_log = args.render_reject_log or args.output.parent / (args.output.stem + "_rejects.jsonl")

        formulas, _report = render_corpus(
            formulas=formulas,
            output_dir=render_out,
            engine=args.render_engine,
            reject_log=reject_log,
            workers=args.render_workers,
            dpi=args.render_dpi,
            katex_node_bin=args.katex_node_bin,
            tex_bin=args.tex_bin,
        )

        if not args.metadata:
            # Strip back to bare strings if the user did not request metadata.
            formulas = {k: v["formula"] for k, v in formulas.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(formulas, fh, ensure_ascii=False)

    logger.info("Wrote %d formulas to %s", len(formulas), args.output)
