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


from .corpus import (
    generate,
    partition_symbols,
    partition_templates,
    write_jsonl,
    write_manifest,
)
from .domains import DEFAULT_WEIGHTS, DOMAIN_TAGS, GENERATORS, PACK_HASHES, PACK_META, TEMPLATES
from .engine.symbol_inventory import MUST_COVER, SYMBOL_FREQUENCY_TIERS, SYMBOL_STRATA

logger = logging.getLogger(__name__)

# Combined symbol-tier choices: existing strata + frequency tiers
_STRATUM_CHOICES = sorted(SYMBOL_STRATA.keys())
_FREQ_TIER_CHOICES = sorted(SYMBOL_FREQUENCY_TIERS.keys())
_ALL_TIER_CHOICES = sorted(set(_STRATUM_CHOICES) | set(_FREQ_TIER_CHOICES))


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
        help="Output file path. Extension ignored when --output-format is given. "
        "Not required when --content-hash is used.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl"],
        default="json",
        dest="output_format",
        help="Output format: json (default, legacy dict) or jsonl (one Sample record per line).",
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
    parser.add_argument(
        "--style-rate",
        type=float,
        default=0.0,
        metavar="F",
        help=(
            "Probability of wrapping each non-environment formula with a uniformly drawn "
            "math style command (\\displaystyle, \\textstyle, \\scriptstyle, "
            "\\scriptscriptstyle). Default: 0.0 (disabled)."
        ),
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
    parser.add_argument(
        "--symbol-tier",
        nargs="+",
        default=None,
        choices=_ALL_TIER_CHOICES,
        metavar="TIER",
        help=(
            f"Keep only templates that exercise at least one of the given symbol strata or "
            f"frequency tiers. Stratum choices: {', '.join(_STRATUM_CHOICES)}. "
            f"Frequency-tier choices: {', '.join(_FREQ_TIER_CHOICES)}."
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
        "--hold-out-templates",
        nargs="+",
        default=None,
        metavar="FRAC_OR_NAME",
        help=(
            "Hold out a fraction or named list of templates. "
            "Pass a single float (e.g. 0.10) to hold out that fraction of templates per domain, "
            "or one or more template names to hold out by name. "
            "A companion <output>.held_out_templates.<ext> file is generated for the held-out partition."
        ),
    )
    parser.add_argument(
        "--hold-out-symbols",
        nargs="+",
        default=None,
        metavar="FRAC_OR_SYMBOL",
        help=(
            "Hold out a fraction or named list of MUST_COVER symbols. "
            "Pass a single float (e.g. 0.15) to hold out that fraction of symbols, "
            "or one or more symbol names (LaTeX tokens). "
            "Templates exercising any held-out symbol are excluded from train (strict semantics). "
            "A companion <output>.held_out_symbols.<ext> file is generated."
        ),
    )
    parser.add_argument(
        "--include-draws",
        action="store_true",
        default=False,
        dest="include_draws",
        help="Attach slot-name→value draws dict to each record (larger output; for semantic analysis).",
    )
    parser.add_argument(
        "--coverage-mode",
        nargs="?",
        const=5,
        type=int,
        default=None,
        dest="coverage_mode",
        metavar="N",
        help=(
            "Keep sampling until every MUST_COVER symbol appears at least N times "
            "(default N=5 when flag is given without a value). "
            "Generation continues beyond --count if needed, up to count×20 attempts."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the split manifest JSON file (default: <output>.manifest.json). "
        "Written whenever --hold-out-templates or --hold-out-symbols is used.",
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
        help="Output metadata dicts (formula + domain + template_name + provenance + structural metrics) "
        "instead of bare strings.",
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

    if not 0.0 <= args.style_rate <= 1.0:
        logger.error("--style-rate must be in [0, 1]")
        sys.exit(1)

    if args.length_range is not None and args.length_range[0] > args.length_range[1]:
        logger.error("--length-range MIN must be ≤ MAX")
        sys.exit(1)

    # Parse --hold-out-templates
    hold_out_templates_spec: list[str] | float | None = None
    if args.hold_out_templates is not None:
        if len(args.hold_out_templates) == 1:
            try:
                hold_out_templates_spec = float(args.hold_out_templates[0])
                if not 0.0 < hold_out_templates_spec < 1.0:
                    logger.error("--hold-out-templates fraction must be in (0, 1)")
                    sys.exit(1)
            except ValueError:
                hold_out_templates_spec = args.hold_out_templates
        else:
            hold_out_templates_spec = args.hold_out_templates

    # Parse --hold-out-symbols
    hold_out_symbols_spec: set[str] | float | None = None
    if args.hold_out_symbols is not None:
        if len(args.hold_out_symbols) == 1:
            try:
                hold_out_symbols_spec = float(args.hold_out_symbols[0])
                if not 0.0 < hold_out_symbols_spec < 1.0:
                    logger.error("--hold-out-symbols fraction must be in (0, 1)")
                    sys.exit(1)
            except ValueError:
                hold_out_symbols_spec = set(args.hold_out_symbols)
        else:
            hold_out_symbols_spec = set(args.hold_out_symbols)

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

    # Resolve symbol tiers: frequency-tier names expand to their constituent strata
    resolved_symbol_tiers: set[str] | None = None
    if args.symbol_tier:
        resolved_symbol_tiers = set()
        for tier_name in args.symbol_tier:
            if tier_name in SYMBOL_STRATA:
                resolved_symbol_tiers.add(tier_name)
            elif tier_name in SYMBOL_FREQUENCY_TIERS:
                # Frequency-tier: translate to the constituent SYMBOL_STRATA names
                freq_syms = SYMBOL_FREQUENCY_TIERS[tier_name]
                for stratum_name, stratum_syms in SYMBOL_STRATA.items():
                    if stratum_syms & freq_syms:
                        resolved_symbol_tiers.add(stratum_name)
            else:
                logger.error("Unknown symbol tier: %r", tier_name)
                sys.exit(1)

    # Determine which params require per-template access.
    need_templates = any(f is not None for f in (args.max_depth, args.length_range, args.symbol_tier))
    # Metadata (rich records) also benefits from templates for structural metrics.
    need_metadata = args.metadata or args.render

    logger.info(
        "Generating %d formulas | domains: %s | seed: %s",
        args.count,
        ", ".join(args.domains),
        args.seed,
    )

    # -----------------------------------------------------------------------
    # Hold-out template partitioning
    # -----------------------------------------------------------------------
    train_templates = TEMPLATES
    held_out_templates: dict = {}
    held_out_template_names: list[str] = []
    held_out_symbol_set: frozenset[str] = frozenset()
    held_out_symbol_names: list[str] = []

    if hold_out_templates_spec is not None:
        train_templates, held_out_templates, held_out_template_names = partition_templates(
            TEMPLATES,
            hold_out=hold_out_templates_spec,
            seed=args.seed,
        )
        logger.info("Hold-out templates: %d names", len(held_out_template_names))

    if hold_out_symbols_spec is not None:
        from .engine._template_dsl import filter_templates as _filter_templates

        _, held_out_symbol_set = partition_symbols(
            MUST_COVER,
            hold_out=hold_out_symbols_spec,
            seed=args.seed,
        )
        held_out_symbol_names = sorted(held_out_symbol_set)
        logger.info("Hold-out symbols: %d symbols", len(held_out_symbol_names))

        # Re-filter train_templates to exclude any template that uses a held-out symbol
        filtered_train: dict = {}
        for domain, tmpl_list in train_templates.items():
            kept = _filter_templates(tmpl_list, exclude_symbols=held_out_symbol_set)
            filtered_train[domain] = kept
        train_templates = filtered_train

    # -----------------------------------------------------------------------
    # Generate train partition
    # -----------------------------------------------------------------------
    # When hold-out is active, use train_templates; else use TEMPLATES for both
    # template pre-filtering and structural metric lookup.
    active_templates = (
        train_templates if (hold_out_templates_spec is not None or hold_out_symbols_spec is not None) else TEMPLATES
    )

    # Rebuild generators from filtered train templates when hold-out is active
    active_generators = dict(GENERATORS)
    if hold_out_templates_spec is not None or hold_out_symbols_spec is not None:
        for domain, tmpl_list in active_templates.items():
            if tmpl_list:
                active_generators[domain] = _make_generator_for_domain(domain, tmpl_list)

    try:
        formulas = generate(
            count=args.count,
            domains=args.domains,
            generators=active_generators,
            weights=merged_weights,
            seed=args.seed,
            display_fraction=args.display_fraction,
            inline_fraction=args.inline_fraction,
            tags=args.tags,
            exclude_tags=args.exclude_tags,
            include_metadata=need_metadata,
            pack_hashes=PACK_HASHES if need_metadata else None,
            pack_meta=PACK_META if need_metadata else None,
            difficulty=args.difficulty,
            max_depth=args.max_depth,
            length_range=tuple(args.length_range) if args.length_range else None,
            symbol_tiers=resolved_symbol_tiers,
            hold_out_domains=args.hold_out_domains,
            templates=active_templates if (need_templates or need_metadata) else None,
            include_draws=args.include_draws,
            split_tag="train",
            coverage_mode=args.coverage_mode is not None,
            coverage_n=args.coverage_mode if args.coverage_mode is not None else 5,
            style_rate=args.style_rate,
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
            formulas = {k: v["formula"] for k, v in formulas.items()}

    _write_output(formulas, args.output, args.output_format)
    logger.info("Wrote %d formulas to %s", len(formulas), args.output)

    # -----------------------------------------------------------------------
    # Generate held-out template partition (separate output file)
    # -----------------------------------------------------------------------
    if held_out_templates:
        held_out_generators = dict(GENERATORS)
        for domain, tmpl_list in held_out_templates.items():
            if tmpl_list:
                held_out_generators[domain] = _make_generator_for_domain(domain, tmpl_list)
        # Only generate from domains that have held-out templates
        held_domains = [d for d in args.domains if d in held_out_templates and held_out_templates[d]]

        if held_domains:
            try:
                held_formulas = generate(
                    count=args.count,
                    domains=held_domains,
                    generators=held_out_generators,
                    weights=merged_weights,
                    seed=args.seed,
                    display_fraction=args.display_fraction,
                    inline_fraction=args.inline_fraction,
                    include_metadata=need_metadata,
                    pack_hashes=PACK_HASHES if need_metadata else None,
                    pack_meta=PACK_META if need_metadata else None,
                    templates=held_out_templates if (need_templates or need_metadata) else None,
                    include_draws=args.include_draws,
                    split_tag="held_out_template",
                )
            except ValueError as exc:
                logger.warning("Held-out template generation failed: %s", exc)
                held_formulas = {}

            if held_formulas:
                held_path = args.output.parent / (
                    args.output.stem + ".held_out_templates" + _output_ext(args.output_format)
                )
                _write_output(held_formulas, held_path, args.output_format)
                logger.info("Wrote %d held-out template formulas to %s", len(held_formulas), held_path)

    # -----------------------------------------------------------------------
    # Generate held-out symbol partition
    # -----------------------------------------------------------------------
    if held_out_symbol_set:
        from .engine._template_dsl import filter_templates as _filter_templates

        # Templates that exercise held-out symbols (the held-out partition)
        held_sym_templates: dict = {}
        for domain, tmpl_list in TEMPLATES.items():
            sym_exercising = [t for t in tmpl_list if _any_template_uses_held_symbol(t, held_out_symbol_set)]
            if sym_exercising:
                held_sym_templates[domain] = sym_exercising

        if held_sym_templates:
            held_sym_generators = dict(GENERATORS)
            for domain, tmpl_list in held_sym_templates.items():
                held_sym_generators[domain] = _make_generator_for_domain(domain, tmpl_list)
            held_sym_domains = [d for d in args.domains if d in held_sym_templates]

            if held_sym_domains:
                try:
                    held_sym_formulas = generate(
                        count=args.count,
                        domains=held_sym_domains,
                        generators=held_sym_generators,
                        weights=merged_weights,
                        seed=args.seed,
                        display_fraction=args.display_fraction,
                        inline_fraction=args.inline_fraction,
                        include_metadata=need_metadata,
                        pack_hashes=PACK_HASHES if need_metadata else None,
                        pack_meta=PACK_META if need_metadata else None,
                        templates=held_sym_templates if (need_templates or need_metadata) else None,
                        include_draws=args.include_draws,
                        split_tag="held_out_symbol",
                    )
                except ValueError as exc:
                    logger.warning("Held-out symbol generation failed: %s", exc)
                    held_sym_formulas = {}

                if held_sym_formulas:
                    held_sym_path = args.output.parent / (
                        args.output.stem + ".held_out_symbols" + _output_ext(args.output_format)
                    )
                    _write_output(held_sym_formulas, held_sym_path, args.output_format)
                    logger.info(
                        "Wrote %d held-out symbol formulas to %s",
                        len(held_sym_formulas),
                        held_sym_path,
                    )

    # -----------------------------------------------------------------------
    # Write split manifest (when any hold-out was requested)
    # -----------------------------------------------------------------------
    if hold_out_templates_spec is not None or hold_out_symbols_spec is not None or args.hold_out_domains:
        manifest_path = args.manifest or args.output.parent / (args.output.stem + ".manifest.json")
        write_manifest(
            path=manifest_path,
            seed=args.seed,
            held_out_domains=args.hold_out_domains,
            held_out_template_names=held_out_template_names if held_out_template_names else None,
            held_out_symbols=held_out_symbol_names if held_out_symbol_names else None,
            train_count=len(formulas),
            held_out_template_count=sum(len(v) for v in held_out_templates.values()),
            held_out_symbol_count=len(held_out_symbol_set),
        )
        logger.info("Wrote split manifest to %s", manifest_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_ext(fmt: str) -> str:
    return ".jsonl" if fmt == "jsonl" else ".json"


def _write_output(records: dict, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        write_jsonl(records, path)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False)


def _make_generator_for_domain(domain: str, tmpl_list: list) -> object:
    from .engine._template_dsl import make_generator

    return make_generator(tmpl_list)


def _any_template_uses_held_symbol(t: object, held_symbols: frozenset[str]) -> bool:
    from .engine._template_dsl import _template_uses_any_symbol

    return _template_uses_any_symbol(t, held_symbols)  # type: ignore[arg-type]
