"""Core generation engine: produce a corpus of unique LaTeX formula strings."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .domains._config import DOMAIN_CONFIG
from .engine._pack_loader import PackMeta
from .engine._sample import make_semantic_key
from .engine._template_dsl import (
    Template,
    _last_draws,
    _last_template_name,
    filter_templates,
    make_generator,
    n_eff,
)
from .engine.symbol_inventory import MUST_COVER, SYMBOL_FREQUENCY_TIERS

logger = logging.getLogger(__name__)


_TAG_POOL = ["1", "2", "3", "4", "5", "6", "*", r"\dagger", "a", "b", "i", "ii"]

_STYLE_MODIFIERS = [r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\scriptscriptstyle"]


def _is_wrapped(formula: str) -> bool:
    return formula.startswith(r"\begin{") or formula.startswith(r"\[") or formula.startswith("$")


def _wrap_display(formula: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.65:
        return rf"\[{formula}\]"
    if r < 0.85:
        return rf"\begin{{equation}}{formula}\end{{equation}}"
    tag = rng.choice(_TAG_POOL)
    return rf"\begin{{equation}}{formula}\tag{{{tag}}}\end{{equation}}"


def _wrap_inline(formula: str) -> str:
    return f"${formula}$"


def _highest_frequency_tier(strata: frozenset[str]) -> str | None:
    """Return the highest-rarity frequency tier exercised by a template's strata.

    Priority: tail > body > head (most specialist wins).
    Returns None if the template exercises no MUST_COVER symbols.
    """
    for tier in ("tail", "body", "head"):
        if SYMBOL_FREQUENCY_TIERS[tier].intersection(sym for stratum in strata for sym in (stratum,)):
            return tier
    return None


def _symbol_tier_for_strata(strata: frozenset[str], templates_strata_syms: frozenset[str] | None = None) -> str | None:
    """Return the highest-rarity frequency tier present among a template's symbol pools."""
    # strata here is the frozenset of SYMBOL_STRATA names exercised by the template.
    # We translate stratum names into their constituent symbols, then find tier membership.
    from .engine.symbol_inventory import SYMBOL_STRATA, SYMBOL_TIER_OF

    tier_priority = {"tail": 2, "body": 1, "head": 0}
    best: str | None = None
    best_priority = -1
    for stratum_name in strata:
        if stratum_name in SYMBOL_STRATA:
            for sym in SYMBOL_STRATA[stratum_name]:
                sym_tier = SYMBOL_TIER_OF.get(sym)
                if sym_tier is not None:
                    p = tier_priority.get(sym_tier, -1)
                    if p > best_priority:
                        best_priority = p
                        best = sym_tier
    return best


def _build_template_index(
    templates: dict[str, list[Template]],
) -> dict[str, dict[str, Template]]:
    """Build a {domain → {template_name → Template}} lookup for O(1) access."""
    index: dict[str, dict[str, Template]] = {}
    for domain, tmpl_list in templates.items():
        index[domain] = {t.name: t for t in tmpl_list}
    return index


# ---------------------------------------------------------------------------
# Partition helpers (for hold-out splits)
# ---------------------------------------------------------------------------


def partition_templates(
    templates: dict[str, list[Template]],
    hold_out: list[str] | float | None,
    seed: int | None = None,
) -> tuple[dict[str, list[Template]], dict[str, list[Template]], list[str]]:
    """Partition templates into train and held-out sets.

    Args:
        templates: Per-domain template lists.
        hold_out: Either a list of template names to hold out, or a float fraction
            of templates to hold out per domain (seeded shuffle, deterministic).
        seed: Seed for the shuffle when ``hold_out`` is a fraction.

    Returns:
        ``(train_templates, held_out_templates, held_out_names)`` where
        ``held_out_names`` is the sorted list of all held-out template names.
    """
    if hold_out is None:
        return templates, {}, []

    rng = random.Random(seed)
    train: dict[str, list[Template]] = {}
    held: dict[str, list[Template]] = {}
    held_names: set[str] = set()

    if isinstance(hold_out, list):
        held_set = set(hold_out)
        for domain, tmpl_list in templates.items():
            t_list = [t for t in tmpl_list if t.name not in held_set]
            h_list = [t for t in tmpl_list if t.name in held_set]
            train[domain] = t_list
            if h_list:
                held[domain] = h_list
            held_names.update(t.name for t in h_list)
    else:
        # Float fraction: deterministic per-domain seeded shuffle
        fraction = float(hold_out)
        for domain, tmpl_list in templates.items():
            if not tmpl_list:
                train[domain] = []
                continue
            shuffled = list(tmpl_list)
            rng.shuffle(shuffled)
            n_held = max(1, math.ceil(len(shuffled) * fraction))
            train[domain] = shuffled[n_held:]
            held_domain = shuffled[:n_held]
            if held_domain:
                held[domain] = held_domain
            held_names.update(t.name for t in held_domain)

    return train, held, sorted(held_names)


def partition_symbols(
    symbol_set: frozenset[str],
    hold_out: set[str] | float | None,
    seed: int | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Partition a symbol set into train and held-out subsets.

    Args:
        symbol_set: Full symbol universe to partition.
        hold_out: Either an explicit set of symbols to hold out, or a float fraction.
            When a fraction, symbols are sorted alphabetically then seeded-shuffled.
        seed: Seed for the shuffle when ``hold_out`` is a fraction.

    Returns:
        ``(train_symbols, held_out_symbols)``.
    """
    if hold_out is None:
        return symbol_set, frozenset()

    if isinstance(hold_out, set):
        held = frozenset(hold_out)
        return symbol_set - held, held

    # Float fraction
    rng = random.Random(seed)
    all_syms = sorted(symbol_set)
    rng.shuffle(all_syms)
    n_held = max(1, math.ceil(len(all_syms) * float(hold_out)))
    held = frozenset(all_syms[:n_held])
    return symbol_set - held, held


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate(
    count: int,
    domains: list[str],
    generators: dict[str, Callable[[random.Random], str]],
    weights: dict[str, float],
    seed: int | None = None,
    display_fraction: float = 0.20,
    inline_fraction: float = 0.10,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    include_metadata: bool = False,
    strict: bool = False,
    pack_hashes: dict[str, str] | None = None,
    pack_meta: dict[str, PackMeta] | None = None,
    difficulty: list[str] | None = None,
    max_depth: int | None = None,
    length_range: tuple[int, int] | None = None,
    symbol_tiers: set[str] | None = None,
    hold_out_domains: list[str] | None = None,
    templates: dict[str, list[Template]] | None = None,
    # WU6 additions
    include_draws: bool = False,
    split_tag: str = "train",
    coverage_mode: bool = False,
    coverage_n: int = 5,
    style_rate: float = 0.0,
) -> dict[str, str] | dict[str, dict]:
    """Generate a corpus of unique LaTeX formula strings.

    Args:
        count: Target number of unique formulas.
        domains: Subset of generator keys to include.
        generators: Mapping of domain name → generator callable.
        weights: Per-domain sampling weights (need not be normalized).
        seed: Random seed for reproducibility.
        display_fraction: Fraction of bare formulas wrapped in display-math environments.
        inline_fraction: Fraction of bare formulas wrapped in inline $...$ delimiters.
        tags: If given, include only domains whose tags overlap with this list.
        exclude_tags: If given, exclude domains whose tags overlap with this list.
        include_metadata: If True, return rich Sample dicts instead of bare strings.
        strict: If True, raise ``RuntimeError`` when any domain's error rate exceeds
            the threshold (≥1% errors with ≥10 attempts); otherwise emit a WARNING.
        pack_hashes: Optional mapping of domain name → SHA-256 of the TOML pack.
            Kept for backward compatibility; superseded by ``pack_meta`` for full provenance.
        pack_meta: Optional mapping of domain name → full PackMeta.  When present,
            all PackMeta fields are included in per-sample records.
        difficulty: If given, keep only domains whose difficulty level is in this list.
        max_depth: If given, keep only templates whose brace-nesting depth is ≤ this value.
        length_range: If given as ``(min, max)``, keep only templates whose char_length falls
            within the range (inclusive).
        symbol_tiers: If given, keep only templates that exercise at least one of the named
            SYMBOL_STRATA tiers.
        hold_out_domains: Domains to exclude from generation (domain-level hold-out).
        templates: Per-domain template lists.  Required when any of ``max_depth``,
            ``length_range``, ``symbol_tiers``, or ``include_metadata`` is specified for
            full structural metrics.
        include_draws: If True, attach the slot-name→value ``draws`` dict to each record
            (larger output; use only when the semantic key isn't enough).
        split_tag: Label attached to every record's ``"split"`` field.
            Callers use ``"held_out_template"`` / ``"held_out_symbol"`` / ``"held_out_domain"``
            for the held-out partitions.
        coverage_mode: If True, continue sampling beyond ``count`` until every MUST_COVER
            symbol appears at least ``coverage_n`` times (up to ``count * 20`` attempts).
        coverage_n: Minimum appearances per MUST_COVER symbol when ``coverage_mode=True``.
        style_rate: Probability of wrapping each non-environment formula with a uniformly
            drawn math style command (``\\displaystyle``, ``\\textstyle``, ``\\scriptstyle``,
            or ``\\scriptscriptstyle``).  Skipped for ``\\begin{...}`` environment formulas.
            Default 0.0 (disabled).

    Returns:
        Dict mapping string index to LaTeX formula string, or to a rich Sample dict
        when ``include_metadata=True``.
    """
    if tags is not None:
        domains = [d for d in domains if d in DOMAIN_CONFIG and any(t in DOMAIN_CONFIG[d].tags for t in tags)]
    if exclude_tags is not None:
        domains = [
            d for d in domains if d not in DOMAIN_CONFIG or not any(t in DOMAIN_CONFIG[d].tags for t in exclude_tags)
        ]
    if difficulty is not None:
        domains = [d for d in domains if d in DOMAIN_CONFIG and DOMAIN_CONFIG[d].difficulty in difficulty]
    if hold_out_domains is not None:
        held = set(hold_out_domains)
        domains = [d for d in domains if d not in held]
    if not domains:
        _active_filters = [
            f
            for label, f in [
                ("tags", tags),
                ("exclude_tags", exclude_tags),
                ("difficulty", difficulty),
                ("hold_out_domains", hold_out_domains),
            ]
            if f is not None
        ]
        raise ValueError(
            f"No domains remaining after filtering. Active filters: {_active_filters}. "
            "Check that the combination of --domains, --tags, --difficulty, and --hold-out-domains "
            "leaves at least one domain active."
        )

    generators = dict(generators)  # local copy — do not mutate the caller's dict

    # Template-level pre-filtering.
    if templates is not None and any(f is not None for f in (max_depth, length_range, symbol_tiers)):
        empty_domains: list[str] = []
        for d in domains:
            if d not in templates:
                continue
            filtered = filter_templates(
                templates[d],
                max_depth=max_depth,
                length_range=length_range,
                symbol_tiers=symbol_tiers,
            )
            if not filtered:
                empty_domains.append(d)
            else:
                generators[d] = make_generator(filtered)
        if empty_domains:
            logger.warning("Domains with no templates matching structural filters (removed): %s", empty_domains)
            domain_set = set(empty_domains)
            domains = [d for d in domains if d not in domain_set]
        if not domains:
            raise ValueError(
                "No domains remain after template-level filtering "
                f"(max_depth={max_depth}, length_range={length_range}, symbol_tiers={symbol_tiers}). "
                "Relax the structural constraints."
            )

    # Build template name → Template index for structural metrics (used in rich records).
    tmpl_index: dict[str, dict[str, Template]] = {}
    if include_metadata and templates is not None:
        tmpl_index = _build_template_index(templates)

    rng = random.Random(seed)
    raw_weights = [weights[d] for d in domains]
    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]
    gens = [generators[d] for d in domains]

    results: dict[str, str | dict] = {}
    seen: set[str] = set()
    attempts = 0
    # coverage_mode may need more attempts than count * 10
    max_attempts = count * (20 if coverage_mode else 10)
    domain_attempts: Counter[str] = Counter()
    domain_errors: Counter[str] = Counter()

    # Per-symbol hit counters for coverage_mode
    coverage_counts: Counter[str] = Counter() if coverage_mode else Counter()

    def _coverage_satisfied() -> bool:
        if not coverage_mode:
            return True
        return all(coverage_counts[sym] >= coverage_n for sym in MUST_COVER)

    while (len(results) < count or not _coverage_satisfied()) and attempts < max_attempts:
        attempts += 1
        domain_name = "<unknown>"
        try:
            idx = rng.choices(range(len(gens)), weights=norm_weights, k=1)[0]
            domain_name = domains[idx]
            domain_attempts[domain_name] += 1
            formula = gens[idx](rng)
            template_name = _last_template_name.get()
            draws = _last_draws.get() if include_metadata else {}
            formula = formula.strip()
            if formula and style_rate > 0.0 and not formula.startswith(r"\begin{"):
                if rng.random() < style_rate:
                    formula = "{" + rng.choice(_STYLE_MODIFIERS) + " " + formula + "}"
            if formula and not _is_wrapped(formula):
                r = rng.random()
                if r < display_fraction:
                    formula = _wrap_display(formula, rng)
                elif r < display_fraction + inline_fraction:
                    formula = _wrap_inline(formula)
            if formula and formula not in seen:
                seen.add(formula)
                if coverage_mode:
                    for sym in MUST_COVER:
                        if sym in formula:
                            coverage_counts[sym] += 1
                key = str(len(results))
                if include_metadata:
                    record = _build_record(
                        formula=formula,
                        domain_name=domain_name,
                        template_name=template_name,
                        draws=draws,
                        include_draws=include_draws,
                        tmpl_index=tmpl_index,
                        pack_hashes=pack_hashes,
                        pack_meta=pack_meta,
                        split_tag=split_tag,
                    )
                    results[key] = record
                else:
                    results[key] = formula
        except Exception:
            domain_errors[domain_name] += 1
            logger.warning("Generator error in domain '%s' (skipping sample)", domain_name, exc_info=True)

    if len(results) < count:
        logger.warning(
            "Generated %d / %d unique formulas after %d attempts",
            len(results),
            count,
            attempts,
        )

    if coverage_mode:
        missing = [sym for sym in MUST_COVER if coverage_counts[sym] < coverage_n]
        if missing:
            logger.warning(
                "coverage_mode: %d MUST_COVER symbols appeared fewer than %d times: %s",
                len(missing),
                coverage_n,
                sorted(missing),
            )
        else:
            logger.info(
                "coverage_mode: all %d MUST_COVER symbols appeared ≥ %d times",
                len(MUST_COVER),
                coverage_n,
            )

    bad_domains = [
        d for d in domain_attempts if domain_attempts[d] >= 10 and domain_errors[d] / domain_attempts[d] >= 0.01
    ]
    if bad_domains:
        parts = [f"{d}: {domain_errors[d]}/{domain_attempts[d]} errors" for d in sorted(bad_domains)]
        msg = "High generator error rate — " + ", ".join(parts)
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    return results


def _build_record(
    formula: str,
    domain_name: str,
    template_name: str | None,
    draws: dict[str, str],
    include_draws: bool,
    tmpl_index: dict[str, dict[str, Template]],
    pack_hashes: dict[str, str] | None,
    pack_meta: dict[str, PackMeta] | None,
    split_tag: str,
) -> dict:
    """Assemble a rich Sample record for one generated formula."""
    record: dict = {
        "formula": formula,
        "domain": domain_name,
        "template_name": template_name,
        "split": split_tag,
    }

    # Semantic key — stable hash of (template_name, draws)
    record["semantic_key"] = make_semantic_key(template_name, draws)

    # Draws (opt-in — can be large)
    if include_draws:
        record["draws"] = draws

    # Structural metrics from Template object
    tmpl: Template | None = None
    if template_name and domain_name in tmpl_index:
        tmpl = tmpl_index[domain_name].get(template_name)

    if tmpl is not None:
        record["depth"] = tmpl.depth
        record["char_length"] = tmpl.char_length
        record["has_fraction"] = tmpl.has_fraction
        record["has_matrix"] = tmpl.has_matrix
        record["has_integral"] = tmpl.has_integral
        record["has_script_chain"] = tmpl.has_script_chain
        record["strata"] = sorted(tmpl.strata)
        record["n_eff"] = n_eff(tmpl)
        record["symbol_tier"] = _symbol_tier_for_strata(tmpl.strata)
    else:
        record["depth"] = None
        record["char_length"] = None
        record["has_fraction"] = None
        record["has_matrix"] = None
        record["has_integral"] = None
        record["has_script_chain"] = None
        record["strata"] = []
        record["n_eff"] = None
        record["symbol_tier"] = None

    # Domain difficulty
    if domain_name in DOMAIN_CONFIG:
        record["difficulty"] = DOMAIN_CONFIG[domain_name].difficulty
    else:
        record["difficulty"] = None

    # Render gate fields (populated later by render_corpus; initialize to None)
    record["render_ok"] = None
    record["image_path"] = None

    # Pack provenance
    meta = (pack_meta or {}).get(domain_name)
    if meta is not None:
        record["content_pack_hash"] = meta.sha256
        record["content_pack_name"] = meta.name
        record["content_pack_version"] = meta.version
        record["content_pack_author"] = meta.author
        record["content_pack_license"] = meta.license
    elif pack_hashes is not None:
        record["content_pack_hash"] = pack_hashes.get(domain_name)
        record["content_pack_name"] = None
        record["content_pack_version"] = None
        record["content_pack_author"] = None
        record["content_pack_license"] = None
    else:
        record["content_pack_hash"] = None
        record["content_pack_name"] = None
        record["content_pack_version"] = None
        record["content_pack_author"] = None
        record["content_pack_license"] = None

    return record


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def write_jsonl(records: dict[str, str | dict], path: Path) -> None:
    """Write corpus records to a JSONL file (one record per line).

    Each line is a JSON object.  For bare-string corpora the object is
    ``{"index": "0", "formula": "..."}``; for metadata corpora it is the full
    Sample dict with an added ``"index"`` field.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for idx, record in records.items():
            if isinstance(record, str):
                fh.write(json.dumps({"index": idx, "formula": record}, ensure_ascii=False) + "\n")
            else:
                row = {"index": idx, **record}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    seed: int | None,
    held_out_domains: list[str] | None,
    held_out_template_names: list[str] | None,
    held_out_symbols: list[str] | None,
    train_count: int,
    held_out_template_count: int,
    held_out_symbol_count: int,
) -> None:
    """Write a split manifest JSON file recording exactly what was held out."""
    manifest = {
        "seed": seed,
        "held_out_domains": sorted(held_out_domains) if held_out_domains else [],
        "held_out_template_names": sorted(held_out_template_names) if held_out_template_names else [],
        "held_out_symbols": sorted(held_out_symbols) if held_out_symbols else [],
        "train_count": train_count,
        "held_out_template_count": held_out_template_count,
        "held_out_symbol_count": held_out_symbol_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
