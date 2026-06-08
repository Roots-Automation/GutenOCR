"""Core generation engine: produce a corpus of unique LaTeX formula strings."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from .align import _align

logger = logging.getLogger(__name__)


def _is_wrapped(formula: str) -> bool:
    return formula.startswith(r"\begin{") or formula.startswith(r"\[") or formula.startswith("$")


def _wrap_display(formula: str, rng: random.Random) -> str:
    if rng.random() < 0.70:
        return rf"\[{formula}\]"
    return rf"\begin{{equation}}{formula}\end{{equation}}"


def _wrap_inline(formula: str) -> str:
    return f"${formula}$"


def generate(
    count: int,
    domains: list[str],
    generators: dict[str, Callable[[random.Random], str]],
    weights: dict[str, float],
    seed: int | None = None,
    align_fraction: float = 0.15,
    display_fraction: float = 0.20,
    inline_fraction: float = 0.10,
) -> dict[str, str]:
    """Generate a corpus of unique LaTeX formula strings.

    Args:
        count: Target number of unique formulas.
        domains: Subset of generator keys to include.
        generators: Mapping of domain name → generator callable.
        weights: Per-domain sampling weights (need not be normalized).
        seed: Random seed for reproducibility.
        align_fraction: Fraction of output using multi-line align* environments.
        display_fraction: Fraction of bare formulas wrapped in display-math environments.
        inline_fraction: Fraction of bare formulas wrapped in inline $...$ delimiters.

    Returns:
        Dict mapping string index to LaTeX formula string.
    """
    rng = random.Random(seed)
    raw_weights = [weights[d] for d in domains]
    total_w = sum(raw_weights)
    norm_weights = [w / total_w for w in raw_weights]
    gens = [generators[d] for d in domains]

    results: dict[str, str] = {}
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 10

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        domain_name = "<align>"
        try:
            if rng.random() < align_fraction:
                formula = _align(rng)
            else:
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
                results[str(len(results))] = formula
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
