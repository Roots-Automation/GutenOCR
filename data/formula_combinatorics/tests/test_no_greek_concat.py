"""Certify that no generator ever produces a Greek-macro/letter concatenation."""

import random

import pytest
from formula_combinatorics import GENERATORS
from formula_combinatorics.engine._template_dsl import _check_no_greek_concat


def test_check_fires_on_bad_latex() -> None:
    with pytest.raises(ValueError, match="Greek macro"):
        _check_no_greek_concat(r"e^{i\varepsilonY}", "test")
    with pytest.raises(ValueError, match="Greek macro"):
        _check_no_greek_concat(r",\quadN_{K}", "test")
    with pytest.raises(ValueError, match="double subscript"):
        _check_no_greek_concat(r"p_2_1^{a_1}", "test")
    with pytest.raises(ValueError, match="double subscript"):
        _check_no_greek_concat(r"p_2_{s}^{a_s}", "test")


def test_check_passes_on_good_latex() -> None:
    _check_no_greek_concat(r"e^{i\varepsilon Y}", "test")
    _check_no_greek_concat(r"\alpha^2 + \beta^2", "test")
    _check_no_greek_concat(r"\Delta z", "test")
    _check_no_greek_concat(r"\omega_1 + \omega_2", "test")
    _check_no_greek_concat(r"\quad N_{K}", "test")  # space after \quad is fine
    _check_no_greek_concat(r"\cdots + \cdots", "test")  # \cdots is fine
    _check_no_greek_concat(r"\top \neq \bot", "test")  # \top is fine
    _check_no_greek_concat(r"x_{n_1} + y", "test")  # braced nested subscript fine


@pytest.mark.parametrize("domain", list(GENERATORS))
def test_generator_no_greek_concat(domain: str) -> None:
    """Sample each domain 1000 times; the validator in sample() raises on any bug."""
    gen = GENERATORS[domain]
    rng = random.Random(42)
    for _ in range(1000):
        gen(rng)
