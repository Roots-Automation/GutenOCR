"""Compatibility shim — re-exports from formula_combinatorics.engine._coverage."""

from formula_combinatorics.engine._coverage import CoverageReport, measure_coverage, measure_coverage_by_domain

__all__ = ["CoverageReport", "measure_coverage", "measure_coverage_by_domain"]
