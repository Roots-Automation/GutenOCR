#!/usr/bin/env python3
"""
Collision Probe — birthday-problem analysis for formula domain generators.

For each domain, samples repeatedly until a formula is generated that exactly
matches one already seen. The sample count at first collision is the effective
"birthday horizon" for that domain — a proxy for output diversity.

Low collision counts reveal domains with small combinatorial spaces.
High counts confirm a domain produces meaningfully distinct output.

Usage:
    python3 collision_probe.py                       # all domains, 1 trial
    python3 collision_probe.py --domain calculus     # one domain
    python3 collision_probe.py --trials 20           # distribution over 20 seeds
    python3 collision_probe.py --seed 42             # fixed starting seed
    python3 collision_probe.py --max-samples 200000  # raise the cap
    python3 collision_probe.py --show-collision       # print the duplicate pair
    python3 collision_probe.py --n-eff               # analytical n_eff per template (no sampling)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formula_combinatorics._calibration import probe_single as _probe_single
from formula_combinatorics._template_dsl import n_eff as _n_eff
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS, TEMPLATES
from formula_combinatorics.engine._template_dsl import sample as _sample

# ── terminal colours (degraded gracefully if not a tty) ──────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def DIM(t: str) -> str:
    return _c("2", t)


def BOLD(t: str) -> str:
    return _c("1", t)


def GREEN(t: str) -> str:
    return _c("32", t)


def YELLOW(t: str) -> str:
    return _c("33", t)


def RED(t: str) -> str:
    return _c("31", t)


def CYAN(t: str) -> str:
    return _c("36", t)


def BLUE(t: str) -> str:
    return _c("34", t)


# ── core probe ────────────────────────────────────────────────────────────────


def probe_domain(
    domain: str,
    seed: int,
    max_samples: int,
) -> dict:
    """Sample from `domain` until a collision or `max_samples` is reached.

    Returns a dict with keys:
        collision    bool   — whether a collision was found
        count        int    — samples drawn (including the colliding one)
        first_idx    int    — index of the earlier match (0-based)
        formula      str    — the colliding formula (or last sample if capped)
        elapsed_s    float  — wall time

    Delegates to formula_combinatorics._calibration.probe_single.
    """
    r = _probe_single(domain, seed, max_samples)
    return {
        "collision": r.collision,
        "count": r.count,
        "first_idx": r.first_idx,
        "formula": r.formula,
        "elapsed_s": r.elapsed_s,
    }


# ── formatting helpers ────────────────────────────────────────────────────────


def _bar(frac: float, width: int = 24) -> str:
    """Tiny ASCII progress bar showing frac ∈ [0, 1]."""
    filled = round(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt_count(n: int, max_samples: int, capped: bool) -> str:
    suffix = "+" if capped else ""
    return f"{n:,}{suffix}"


def _summary_line(
    domain: str,
    weight: float,
    result: dict,
    max_samples: int,
) -> str:
    count = result["count"]
    capped = not result["collision"]
    frac = min(count / max_samples, 1.0)

    count_str = _fmt_count(count, max_samples, capped)

    if capped:
        bar = _bar(1.0)
        colour = GREEN
        status = "no collision"
    elif count < 50:
        bar = _bar(frac)
        colour = RED
        status = f"collision #{result['first_idx']}"
    elif count < 500:
        bar = _bar(frac)
        colour = YELLOW
        status = f"collision #{result['first_idx']}"
    else:
        bar = _bar(frac)
        colour = CYAN
        status = f"collision #{result['first_idx']}"

    elapsed = f"{result['elapsed_s'] * 1000:.0f}ms"
    return (
        f"  {colour(f'{domain:<22}')}"
        f" {DIM(f'{weight:>4.0%}')}"
        f"  {colour(bar)}"
        f"  {BOLD(f'{count_str:>9}')}"
        f"  {DIM(status)}"
        f"  {DIM(elapsed)}"
    )


def _stats(values: list[int]) -> str:
    if not values:
        return ""
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    med = s[n // 2]
    lo, hi = s[0], s[-1]
    return f"  min={lo:,}  median={med:,}  mean={mean:,.0f}  max={hi:,}"


# ── single trial ──────────────────────────────────────────────────────────────


def run_single(
    domains: list[str],
    seed: int,
    max_samples: int,
    show_collision: bool,
) -> None:
    print(f"\n{BOLD('Collision Probe')}  {DIM(f'seed={seed}  max={max_samples:,}')}\n")
    print(f"  {'domain':<22} {'wt':>4}  {'progress':^26}  {'samples':>9}  notes")
    print("  " + "─" * 80)

    for domain in domains:
        weight = DEFAULT_WEIGHTS.get(domain, 0.0)
        result = probe_domain(domain, seed, max_samples)
        print(_summary_line(domain, weight, result, max_samples))

        if show_collision and result["collision"]:
            f = result["formula"]
            preview = f if len(f) <= 90 else f[:87] + "…"
            print(f"    {DIM('↳ first seen at sample')} {result['first_idx']}: {DIM(preview)}")

    print()
    print(
        DIM(
            f"  Legend: {RED('red')} < 50  "
            f"{YELLOW('yellow')} < 500  "
            f"{CYAN('cyan')} ≥ 500  "
            f"{GREEN('green')} no collision within {max_samples:,}"
        )
    )
    print()


# ── multi-trial distribution ──────────────────────────────────────────────────


def run_trials(
    domains: list[str],
    base_seed: int,
    trials: int,
    max_samples: int,
) -> None:
    print(
        f"\n{BOLD('Collision Distribution')}  {DIM(f'{trials} trials  base_seed={base_seed}  max={max_samples:,}')}\n"
    )
    print(f"  {'domain':<22}  {'min':>8}  {'median':>8}  {'mean':>8}  {'max':>8}  {'no-collision':>12}")
    print("  " + "─" * 76)

    for domain in domains:
        counts = []
        uncapped = 0
        for t in range(trials):
            r = probe_domain(domain, base_seed + t, max_samples)
            if not r["collision"]:
                uncapped += 1
            counts.append(r["count"])

        s = sorted(counts)
        n = len(s)
        mean = sum(s) / n
        med = s[n // 2]
        lo, hi = s[0], s[-1]

        # Colour the median
        if uncapped == trials:
            col = GREEN
        elif med < 50:
            col = RED
        elif med < 500:
            col = YELLOW
        else:
            col = CYAN

        no_col_str = f"{uncapped}/{trials}"
        print(
            f"  {col(f'{domain:<22}')}"
            f"  {lo:>8,}"
            f"  {col(f'{med:>8,}')}"
            f"  {mean:>8,.0f}"
            f"  {hi:>8,}"
            f"  {DIM(f'{no_col_str:>12}')}"
        )

    print()


# ── batch uniqueness ──────────────────────────────────────────────────────────


def batch_unique(domain: str, seed: int, batch_size: int) -> float:
    """Sample batch_size formulas and return fraction that are unique."""
    gen = GENERATORS[domain]
    rng = random.Random(seed)
    seen = set()
    for _ in range(batch_size):
        seen.add(gen(rng))
    return len(seen) / batch_size


def run_batch(
    domains: list[str],
    seed: int,
    batch_size: int,
    trials: int,
) -> None:
    if trials == 1:
        print(f"\n{BOLD('Batch Uniqueness')}  {DIM(f'batch={batch_size:,}  seed={seed}')}\n")
        print(f"  {'domain':<22} {'wt':>4}  {'unique':>8}  {'total':>8}  {'%unique':>8}")
        print("  " + "─" * 60)
        for domain in domains:
            weight = DEFAULT_WEIGHTS.get(domain, 0.0)
            frac = batch_unique(domain, seed, batch_size)
            unique = round(frac * batch_size)
            col = GREEN if frac >= 0.99 else YELLOW if frac >= 0.90 else RED
            print(
                f"  {col(f'{domain:<22}')}"
                f" {DIM(f'{weight:>4.0%}')}"
                f"  {unique:>8,}"
                f"  {batch_size:>8,}"
                f"  {col(f'{frac:>7.1%}')}"
            )
        print()
    else:
        print(
            f"\n{BOLD('Batch Uniqueness Distribution')}  "
            f"{DIM(f'{trials} trials  batch={batch_size:,}  base_seed={seed}')}\n"
        )
        print(f"  {'domain':<22}  {'min%':>7}  {'median%':>8}  {'mean%':>8}  {'max%':>7}")
        print("  " + "─" * 62)
        for domain in domains:
            fracs = sorted(batch_unique(domain, seed + t, batch_size) for t in range(trials))
            n = len(fracs)
            med = fracs[n // 2]
            mean = sum(fracs) / n
            col = GREEN if med >= 0.99 else YELLOW if med >= 0.90 else RED
            print(f"  {col(f'{domain:<22}')}  {fracs[0]:>7.1%}  {col(f'{med:>8.1%}')}  {mean:>8.1%}  {fracs[-1]:>7.1%}")
        print()


# ── per-template batch uniqueness ────────────────────────────────────────────


def run_batch_per_template(domains: list[str], seed: int, batch_size: int) -> None:
    """For each template in each domain, draw batch_size samples and report % unique."""
    ported = [d for d in domains if d in TEMPLATES]
    unported = [d for d in domains if d not in TEMPLATES]

    print(f"\n{BOLD('Per-Template Batch Uniqueness')}  {DIM(f'batch={batch_size:,}  seed={seed}')}\n")

    if not ported:
        print(f"  {YELLOW('No ported domains found.')}\n")
    else:
        for domain in ported:
            templates = TEMPLATES[domain]
            weight = DEFAULT_WEIGHTS.get(domain, 0.0)
            print(f"  {BOLD(domain)}  {DIM(f'{weight:.0%}')}")
            print(f"    {'template':<36}  {'unique':>8}  {'%unique':>8}  {'n_eff':>14}")
            print("    " + "─" * 72)

            for t in templates:
                rng = random.Random(seed)
                seen: set[str] = set()
                for _ in range(batch_size):
                    seen.add(_sample(t, rng))
                frac = len(seen) / batch_size
                te = _n_eff(t)

                col = GREEN if frac >= 0.99 else YELLOW if frac >= 0.90 else RED
                print(f"    {col(f'{t.name:<36}')}  {len(seen):>8,}  {col(f'{frac:>7.1%}')}  {DIM(f'{te:>14,.0f}')}")
            print()

    if unported:
        print(f"  {DIM('Not yet ported:')} {DIM(', '.join(unported))}\n")


# ── analytical n_eff ──────────────────────────────────────────────────────────


def run_n_eff(domains: list[str]) -> None:
    """Print the analytically-computed n_eff for each template in each domain."""
    ported = [d for d in domains if d in TEMPLATES]
    unported = [d for d in domains if d not in TEMPLATES]

    print(f"\n{BOLD('Analytical n_eff')}  {DIM('(template DSL — no sampling required)')}\n")

    if not ported:
        print(f"  {YELLOW('No ported domains found.')}  Port domains to the template DSL to see analytical n_eff.\n")
    else:
        for domain in ported:
            templates = TEMPLATES[domain]
            domain_total = sum(_n_eff(t) for t in templates)
            weight = DEFAULT_WEIGHTS.get(domain, 0.0)

            print(
                f"  {BOLD(f'{domain:<22}')} {DIM(f'{weight:>4.0%}')}  "
                f"templates={len(templates)}  "
                f"total_n_eff={CYAN(f'{domain_total:>14,.0f}')}"
            )
            for t in templates:
                te = _n_eff(t)
                bar_frac = min(math.log10(max(te, 1)) / 7, 1.0)  # log scale 0–10M
                bar = _bar(bar_frac, width=16)
                print(f"    {DIM(f'{t.name:<34}')} {bar}  {te:>12,.0f}")
        print()

    if unported:
        print(f"  {DIM('Not yet ported:')} {DIM(', '.join(unported))}\n")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_WEIGHTS.keys()),
        choices=list(GENERATORS.keys()),
        metavar="DOMAIN",
        help="Domains to probe (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting random seed (default: 0)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent trials per domain (default: 1). "
        "When >1, prints a distribution table instead of per-domain bars.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        metavar="N",
        help="Sample cap per trial (default: 50000). "
        "Domains that reach this without collision are reported as 'no collision'.",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Print the colliding formula and the index of its first occurrence.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        metavar="N",
        help="If set, report %% unique in a batch of N samples instead of collision horizon.",
    )
    parser.add_argument(
        "--batch-per-template",
        type=int,
        default=None,
        metavar="N",
        help="For each template, draw N samples and report %% unique alongside n_eff.",
    )
    parser.add_argument(
        "--n-eff",
        action="store_true",
        help="Print analytically-computed n_eff per template for ported domains (no sampling).",
    )
    args = parser.parse_args()

    if args.n_eff:
        run_n_eff(args.domains)
    elif args.batch_per_template is not None:
        run_batch_per_template(args.domains, args.seed, args.batch_per_template)
    elif args.batch is not None:
        run_batch(args.domains, args.seed, args.batch, args.trials)
    elif args.trials == 1:
        run_single(args.domains, args.seed, args.max_samples, args.show_collision)
    else:
        run_trials(args.domains, args.seed, args.trials, args.max_samples)


if __name__ == "__main__":
    main()
