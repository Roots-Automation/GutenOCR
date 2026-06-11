"""Core generation engine: produce a corpus of unique LaTeX formula strings."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from .domains._config import DOMAIN_CONFIG

logger = logging.getLogger(__name__)


_TAG_POOL = ["1", "2", "3", "4", "5", "6", "*", r"\dagger", "a", "b", "i", "ii"]


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
        include_metadata: If True, return ``{"formula": ..., "domain": ...}`` dicts
            instead of bare strings.

    Returns:
        Dict mapping string index to LaTeX formula string, or to a metadata dict
        when ``include_metadata=True``.
    """
    if tags is not None:
        domains = [d for d in domains if d in DOMAIN_CONFIG and any(t in DOMAIN_CONFIG[d].tags for t in tags)]
    if exclude_tags is not None:
        domains = [
            d for d in domains if d not in DOMAIN_CONFIG or not any(t in DOMAIN_CONFIG[d].tags for t in exclude_tags)
        ]
    if not domains:
        logger.warning("No domains remaining after tag filtering; returning empty corpus.")
        return {}

    rng = random.Random(seed)
    raw_weights = [weights[d] for d in domains]
    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]
    gens = [generators[d] for d in domains]

    results: dict[str, str | dict] = {}
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 10

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        domain_name = "<unknown>"
        try:
            idx = rng.choices(range(len(gens)), weights=norm_weights, k=1)[0]
            domain_name = domains[idx]
            formula = gens[idx](rng)
            formula = formula.strip()
            if formula and not _is_wrapped(formula):
                r = rng.random()
                if r < display_fraction:
                    formula = _wrap_display(formula, rng)
                elif r < display_fraction + inline_fraction:
                    formula = _wrap_inline(formula)
            if formula and formula not in seen:
                seen.add(formula)
                key = str(len(results))
                if include_metadata:
                    results[key] = {"formula": formula, "domain": domain_name}
                else:
                    results[key] = formula
        except Exception:
            logger.warning("Generator error in domain '%s' (skipping sample)", domain_name, exc_info=True)

    if len(results) < count:
        logger.warning(
            "Generated %d / %d unique formulas after %d attempts",
            len(results),
            count,
            attempts,
        )
    return results
