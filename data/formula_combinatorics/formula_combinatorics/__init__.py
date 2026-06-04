"""formula_combinatorics — synthetic LaTeX math formula generator."""

from .corpus import generate
from .domains import DEFAULT_WEIGHTS, GENERATORS

__all__ = ["generate", "GENERATORS", "DEFAULT_WEIGHTS"]
