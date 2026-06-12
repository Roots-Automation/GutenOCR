"""Keyword-presence tests for the 31 domains with only smoke coverage.

For each domain, verifies that a distinctive LaTeX keyword appears in at least
one sample across 1000 draws with seed=0.  Catches cases where a domain
accidentally loses its core notation after a refactor.
"""

from __future__ import annotations

import random

import pytest
from formula_combinatorics.domains import GENERATORS

# Domain → keyword that MUST appear in at least 1 of 1000 samples.
# Only domains that lack domain-specific tests elsewhere are included;
# math_fonts and quantum_notation are covered by their dedicated test files.
_DOMAIN_KEYWORDS: dict[str, str] = {
    "calculus": r"\int",
    "algebra": r"\frac",
    "trigonometry": r"\sin",
    "linear_algebra": r"\begin{pmatrix}",
    "probability": r"\mathbb{P}",
    "geometry": r"\angle",
    "statistics": r"\sigma",
    "classical_mechanics": r"\frac",
    "electromagnetism": r"\nabla",
    "statistical_mechanics": r"\beta",
    "quantum_mechanics": r"\hbar",
    "field_theory": r"\partial",
    "chemistry": r"\rightarrow",
    "differential_equations": r"\frac{d",
    "fourier": r"\hat",
    "group_theory": r"\cong",
    "number_theory": r"\equiv",
    "combinatorics": r"\binom",
    "set_theory": r"\in",
    "logic": r"\forall",
    "proof_theory": r"\dfrac",
    "topology": r"\mathcal",
    "analysis": "psilon",  # matches both \epsilon and \varepsilon
    "complex_analysis": r"\mathbb{C}",
    "measure_theory": r"\int",
    "optimization": r"\min",
    "information_theory": r"\log",
    "graph_theory": r"\in E(",  # edge set membership
    "category_theory": r"\circ",
    "differential_geometry": r"\partial",
    "representation_theory": r"\oplus",
    "ring_field_theory": r"\cdot",
    "stochastic_processes": r"\mathbb{E}",
    "p_adic": r"\mathbb{Q}",
    "custom_operators": r"\operatorname",
    "align": r"\begin{align",
}

_N_SAMPLES = 1000
_SEED = 0


def _ids() -> list[str]:
    return [f"{d}:{kw}" for d, kw in _DOMAIN_KEYWORDS.items()]


@pytest.mark.parametrize("domain,keyword", list(_DOMAIN_KEYWORDS.items()), ids=_ids())
def test_keyword_appears_in_domain_samples(domain: str, keyword: str) -> None:
    gen = GENERATORS[domain]
    rng = random.Random(_SEED)
    for i in range(_N_SAMPLES):
        if keyword in gen(rng):
            return  # passed — keyword found
    pytest.fail(
        f"{domain}: keyword {keyword!r} not found in {_N_SAMPLES} samples "
        f"(seed={_SEED}). Domain may have lost its core notation."
    )
