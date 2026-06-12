"""n_eff calibration: compare analytic estimates to empirical birthday horizons.

The birthday problem predicts that a collision (repeated sample) occurs at
approximately sqrt(N) draws when N is the effective output-space size.
This module runs the collision probe across trials and compares the median
empirical horizon to the analytic prediction sqrt(sum_of_template_n_effs).

Intended use
------------
- Importable module consumed by tests/test_calibration.py.
- CLI delegation target for tools/collision_probe.py.

Tolerance
---------
If empirical_horizon < TOLERANCE * analytic_horizon, the analytic n_eff is
overestimated (likely due to inflated Sub n_eff_estimate constants). The tuning
report lists the largest-n_eff templates as candidates for adjustment.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass

from .domains import GENERATORS, TEMPLATES
from .engine._template_dsl import ExcludeParamSub, ParamSub, Sub, Template
from .engine._template_dsl import n_eff as _n_eff

_DEFAULT_MAX_SAMPLES: int = 50_000
_DEFAULT_N_TRIALS: int = 5
_DEFAULT_SEED: int = 42

# Empirical horizon must be at least this fraction of the analytic horizon.
# Values below this indicate inflated Sub n_eff_estimate constants.
CALIBRATION_TOLERANCE: float = 0.1


@dataclass
class TrialResult:
    collision: bool
    count: int  # samples drawn (including the colliding one)
    first_idx: int | None
    formula: str | None
    elapsed_s: float


@dataclass
class CalibrationResult:
    domain: str
    analytic_n_eff: float  # sum of template n_effs
    analytic_horizon: float  # sqrt(analytic_n_eff): birthday threshold
    empirical_horizon: float  # median collision count across trials
    ratio: float  # empirical_horizon / analytic_horizon (near 1.0 = well-calibrated)
    per_template: dict[str, float]  # template name → n_eff
    trial_counts: list[int]  # raw collision counts per trial
    n_trials: int  # number of trials requested
    n_uncapped: int  # trials that hit max_samples without collision


def probe_single(
    domain: str,
    seed: int,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> TrialResult:
    """Run one trial: sample until collision or max_samples is reached.

    Returns a TrialResult compatible with the collision_probe.py CLI output format.
    """
    gen = GENERATORS[domain]
    rng = random.Random(seed)
    seen: dict[str, int] = {}

    t0 = time.perf_counter()
    n_sampled = 0
    for _ in range(max_samples):
        try:
            formula = gen(rng)
        except Exception:
            continue  # skip erroring samples, consistent with corpus.generate()
        n_sampled += 1
        if formula in seen:
            return TrialResult(
                collision=True,
                count=n_sampled,
                first_idx=seen[formula] + 1,
                formula=formula,
                elapsed_s=time.perf_counter() - t0,
            )
        seen[formula] = n_sampled - 1

    return TrialResult(
        collision=False,
        count=max_samples,
        first_idx=None,
        formula=None,
        elapsed_s=time.perf_counter() - t0,
    )


def probe_domain(
    domain: str,
    seed: int = _DEFAULT_SEED,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    n_trials: int = _DEFAULT_N_TRIALS,
) -> CalibrationResult:
    """Run N trials and compare median empirical horizon to analytic n_eff.

    If all trials hit max_samples without a collision the domain's n_eff is
    well above the probe budget; n_uncapped == n_trials in that case.

    Args:
        domain: Registered domain name.
        seed: Base seed; trial i uses seed + i.
        max_samples: Per-trial cap.
        n_trials: Number of independent trials.

    Returns:
        CalibrationResult with empirical vs analytic comparison.
    """
    templates = TEMPLATES.get(domain, [])
    per_template: dict[str, float] = {t.name: _n_eff(t) for t in templates}
    analytic_n_eff = sum(per_template.values()) or 1.0
    analytic_horizon = math.sqrt(analytic_n_eff)

    trial_counts: list[int] = []
    n_uncapped = 0
    for i in range(n_trials):
        r = probe_single(domain, seed + i, max_samples)
        trial_counts.append(r.count)
        if not r.collision:
            n_uncapped += 1

    empirical_horizon = float(statistics.median(trial_counts))
    ratio = empirical_horizon / analytic_horizon if analytic_horizon > 0 else float("inf")

    return CalibrationResult(
        domain=domain,
        analytic_n_eff=analytic_n_eff,
        analytic_horizon=analytic_horizon,
        empirical_horizon=empirical_horizon,
        ratio=ratio,
        per_template=per_template,
        trial_counts=trial_counts,
        n_trials=n_trials,
        n_uncapped=n_uncapped,
    )


def _has_sub_slots(t: Template) -> bool:
    """Return True if t (or any of its variants) has a Sub/ParamSub/ExcludeParamSub slot."""
    if any(isinstance(s, (Sub, ParamSub, ExcludeParamSub)) for s in t.slots.values()):
        return True
    return any(_has_sub_slots(v) for v in t.variants)


def tuning_report(
    result: CalibrationResult,
    tolerance: float = CALIBRATION_TOLERANCE,
    top_n: int = 10,
) -> list[str]:
    """Return lines describing templates that may need n_eff_estimate tuning.

    For miscalibrated domains, each template line shows the current analytic n_eff,
    the suggested n_eff (current * correction_factor, where correction_factor = ratio²),
    and a [Sub] tag for templates whose n_eff contains adjustable estimates.

    The correction_factor is a domain-level approximation: it scales the whole domain's
    analytic n_eff to match the empirical birthday horizon.  Per-template corrections
    require per-template probes; this gives an actionable starting point.

    Returns an empty list when the domain is well-calibrated or entirely uncapped.
    """
    if result.n_uncapped == result.n_trials or result.ratio >= tolerance:
        return []

    correction_factor = result.ratio**2
    domain_templates = {t.name: t for t in TEMPLATES.get(result.domain, [])}

    lines = [
        f"[{result.domain}] MISCALIBRATED  "
        f"ratio={result.ratio:.3f} < {tolerance:.2f}  "
        f"correction_factor={correction_factor:.2e}  "
        f"empirical={result.empirical_horizon:.0f}  "
        f"analytic_horizon={result.analytic_horizon:.0f}",
    ]
    for name, ne in sorted(result.per_template.items(), key=lambda kv: -kv[1])[:top_n]:
        suggested = ne * correction_factor
        t = domain_templates.get(name)
        tag = "[Sub]  " if (t and _has_sub_slots(t)) else "[exact]"
        lines.append(f"  {name:<44}  current={ne:>18,.0f}  →  suggested={suggested:>14,.0f}  {tag}")
    return lines
