# Backward-compat shim — source of truth is engine/symbol_inventory.py
from formula_combinatorics.engine.symbol_inventory import (  # noqa: F401
    COVERAGE_N,
    COVERAGE_SEED,
    MUST_COVER,
    SYMBOL_STRATA,
    collect_should_cover,
)
