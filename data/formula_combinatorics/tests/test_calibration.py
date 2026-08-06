"""CI calibration tests: empirical birthday horizon vs analytic n_eff.

Tests are marked @pytest.mark.slow and skipped in normal CI.

Run the full calibration sweep:
    pytest -m slow tests/test_calibration.py -v

Run a single domain:
    pytest -m slow tests/test_calibration.py -v -k algebra
"""

from __future__ import annotations

import pytest
from formula_combinatorics._calibration import (
    CALIBRATION_TOLERANCE,
    CalibrationResult,
    probe_domain,
    tuning_report,
)
from formula_combinatorics.domains import GENERATORS

_DOMAINS = sorted(GENERATORS.keys())


@pytest.mark.slow
@pytest.mark.parametrize("domain", _DOMAINS)
def test_n_eff_calibration(domain: str) -> None:
    """Empirical birthday horizon must be within 10x of the analytic prediction.

    Analytic horizon = sqrt(sum of template n_effs).
    Empirical horizon = median collision count over 5 trials.

    A ratio below CALIBRATION_TOLERANCE (0.1) means the analytic n_eff is
    inflated — likely due to Sub n_eff_estimate constants being too high.
    When all trials exhaust max_samples without a collision the domain's
    true n_eff is well above the probe budget; no failure is raised.
    """
    result = probe_domain(domain, seed=42, max_samples=50_000, n_trials=5)

    report_lines = tuning_report(result, tolerance=CALIBRATION_TOLERANCE)
    for line in report_lines:
        print(line)

    if result.n_uncapped == result.n_trials:
        return

    assert result.ratio >= CALIBRATION_TOLERANCE, (
        f"{domain}: empirical_horizon={result.empirical_horizon:.0f} is {result.ratio:.3f}x "
        f"analytic_horizon={result.analytic_horizon:.0f} "
        f"(analytic_n_eff={result.analytic_n_eff:,.0f}). "
        f"Sub n_eff_estimate constants may be inflated. "
        f"Top templates by analytic n_eff: " + str(sorted(result.per_template.items(), key=lambda kv: -kv[1])[:5])
    )


@pytest.mark.slow
def test_calibration_report_format() -> None:
    """Smoke test: probe_domain returns a well-formed CalibrationResult."""
    result = probe_domain("algebra", seed=0, max_samples=10_000, n_trials=2)
    assert isinstance(result, CalibrationResult)
    assert result.domain == "algebra"
    assert result.analytic_n_eff > 0
    assert result.analytic_horizon > 0
    assert result.empirical_horizon > 0
    assert len(result.trial_counts) == 2
    assert result.n_uncapped >= 0
    assert isinstance(result.per_template, dict)
    assert len(result.per_template) > 0
