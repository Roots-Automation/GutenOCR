"""Declarative template DSL for formula domain generators.

Each branch in a domain generator is expressed as a named ``Template``
with typed ``Slot`` declarations.  Two functions are provided:

* ``sample(t, rng)``  — draw one formula string from the template
* ``n_eff(t)``        — compute the effective output-space size analytically

This replaces the ``c = rng.randint(0, N)`` + ``if c == K`` pattern with a
list of named objects whose combinatorics are visible without running any
empirical probe.

Slot types
----------
Slot        — draw one symbol from a fixed pool; optional _maybe_idx decoration
ExcludeSlot — like Slot, but excludes values already drawn by named sibling slots
Sub         — call a sub-generator (e.g. _expr, _atom); n_eff is an estimate
ParamSub    — sub-generator whose second arg is the value of another drawn slot
              (used for generators like _poly(rng, v))

Convenience constructors: ``S``, ``X``, ``E``, ``P`` (see module bottom).

Template latex convention
-------------------------
Identical to the existing ``rf"..."`` strings in domain files:
  ``{{`` and ``}}`` for literal LaTeX braces, ``{name}`` for slot substitution.
Templates are plain ``r"..."`` strings; ``.format(**draws)`` is called at
sample time — *not* at definition time.

Example
-------
>>> from formula_combinatorics._template_dsl import S, E, Template, sample, n_eff
>>> from formula_combinatorics._vocab import _VARS, _SCALARS, _expr
>>> import random
>>> T = Template(
...     "quadratic_formula",
...     r"{v} = \\frac{{-{q} \\pm \\sqrt{{{q}^2 - 4 {p} {r}}}}}{{2{p}}}",
...     slots={k: S(_VARS + _SCALARS, idx=0.35) for k in "vpqr"},
...     distinct=[list("vpqr")],
... )
>>> print(f"n_eff = {n_eff(T):,.0f}")
>>> print(sample(T, random.Random(0)))
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Index decoration pool (mirrors _maybe_idx in _vocab.py)
# ---------------------------------------------------------------------------

_IDX_POOL: tuple[str, ...] = ("0", "1", "2", "i", "j", "k", "n", "m")
_N_IDX = len(_IDX_POOL)  # 8


def _eff_mult(idx: float) -> float:
    """Effective pool size multiplier from _maybe_idx decoration.

    A slot decorated with probability ``idx`` has an effective pool size of
    ``|pool| * (1 + idx * (_N_IDX - 1))`` because each bare symbol can also
    appear as one of ``_N_IDX`` subscripted variants.
    """
    return 1.0 + idx * (_N_IDX - 1)


def _decorate(val: str, idx: float, rng: random.Random) -> str:
    if idx > 0.0 and rng.random() < idx:
        return f"{val}_{{{rng.choice(_IDX_POOL)}}}"
    return val


# ---------------------------------------------------------------------------
# Slot types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """Draw one symbol from a fixed pool, optionally decorated with a subscript."""

    pool: tuple[str, ...]
    idx: float = 0.0

    def draw(self, rng: random.Random, exclude: frozenset[str] = frozenset()) -> str:
        pool = [p for p in self.pool if p not in exclude]
        val = rng.choice(pool)
        return _decorate(val, self.idx, rng)

    def eff_size(self) -> float:
        return len(self.pool) * _eff_mult(self.idx)


@dataclass(frozen=True)
class ExcludeSlot:
    """Like ``Slot``, but excludes values already drawn by the named sibling slots."""

    pool: tuple[str, ...]
    exclude_from: tuple[str, ...]
    idx: float = 0.0

    def draw(self, rng: random.Random, exclude: frozenset[str] = frozenset()) -> str:
        pool = [p for p in self.pool if p not in exclude]
        val = rng.choice(pool)
        return _decorate(val, self.idx, rng)

    def eff_size(self, n_excluded: int = 0) -> float:
        remaining = max(1, len(self.pool) - n_excluded)
        return remaining * _eff_mult(self.idx)


@dataclass(frozen=True)
class Sub:
    """Call a sub-generator (e.g. ``_expr``, ``_atom``).  n_eff is an estimate."""

    gen: Callable[[random.Random], str]
    n_eff_estimate: float = 1e4

    def draw(self, rng: random.Random, _draws: dict[str, str] | None = None) -> str:
        return self.gen(rng)

    def eff_size(self) -> float:
        return self.n_eff_estimate


@dataclass(frozen=True)
class ParamSub:
    """Sub-generator whose second arg is the value of another already-drawn slot.

    Example: ``_poly(rng, v)`` where ``v`` is drawn by a ``Slot``::

        "poly": ParamSub(_poly, param_slot="v", n_eff_estimate=500)
    """

    gen: Callable[[random.Random, str], str]
    param_slot: str
    n_eff_estimate: float = 1e4

    def draw(self, rng: random.Random, draws: dict[str, str]) -> str:
        return self.gen(rng, draws[self.param_slot])

    def eff_size(self) -> float:
        return self.n_eff_estimate


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_SlotType = Slot | ExcludeSlot | Sub | ParamSub


@dataclass
class Template:
    """A named formula template with typed slot declarations.

    Parameters
    ----------
    name:
        Human-readable identifier used in ``--n-eff`` output and debug traces.
    latex:
        ``str.format()``-style string.  Use ``{{``/``}}`` for literal LaTeX
        braces, ``{slot_name}`` for substitution.
    slots:
        Mapping from slot name to a Slot/ExcludeSlot/Sub/ParamSub descriptor.
        Slots are drawn in declaration order; ``ExcludeSlot.exclude_from`` and
        ``ParamSub.param_slot`` must refer to names that appear *before* them
        in this dict (Python 3.7+ dict preserves insertion order).
    distinct:
        Groups of slot names that must all receive different base symbols.
        Each group is sampled as a single ``rng.sample()`` call from the
        union of the group's pools.
    variants:
        If non-empty, ``latex`` and ``slots`` are ignored; one variant is
        chosen at random and sampled recursively.
    """

    name: str
    latex: str
    slots: dict[str, _SlotType]
    distinct: list[list[str]] = field(default_factory=list)
    variants: list[Template] = field(default_factory=list)


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


def sample(t: Template, rng: random.Random) -> str:
    """Draw one formula string from *t*."""
    if t.variants:
        return sample(rng.choice(t.variants), rng)

    # Constant templates (no slots, no distinct groups) — return verbatim so that
    # LaTeX brace groups like \frac{a}{b} are never mistaken for format slots.
    if not t.slots and not t.distinct:
        return t.latex

    draws: dict[str, str] = {}
    constrained: set[str] = {name for group in t.distinct for name in group}

    # 1. Distinct groups — sample without replacement from the union pool
    for group in t.distinct:
        # Collect the union of raw (pre-decoration) pools for this group
        union: list[str] = []
        seen_union: set[str] = set()
        for name in group:
            s = t.slots[name]
            if isinstance(s, (Slot, ExcludeSlot)):
                for p in s.pool:
                    if p not in seen_union:
                        union.append(p)
                        seen_union.add(p)
        raw_vals = rng.sample(union, len(group))
        for name, raw in zip(group, raw_vals):
            s = t.slots[name]
            idx = s.idx if isinstance(s, (Slot, ExcludeSlot)) else 0.0
            draws[name] = _decorate(raw, idx, rng)

    # 2. Independent slots — draw in declaration order
    for name, s in t.slots.items():
        if name in constrained:
            continue
        if isinstance(s, Sub):
            draws[name] = s.draw(rng)
        elif isinstance(s, ParamSub):
            draws[name] = s.draw(rng, draws)
        elif isinstance(s, ExcludeSlot):
            exclude = frozenset(
                # strip decoration to compare base symbols
                draws[ex].split("_")[0]
                for ex in s.exclude_from
                if ex in draws
            )
            draws[name] = s.draw(rng, exclude)
        else:  # Slot
            draws[name] = s.draw(rng)

    return t.latex.format(**draws)


# ---------------------------------------------------------------------------
# n_eff()
# ---------------------------------------------------------------------------


def n_eff(t: Template) -> float:
    """Compute the effective output-space size of *t* analytically.

    For ``Sub`` / ``ParamSub`` slots the ``n_eff_estimate`` is used as-is.
    For ``ExcludeSlot`` the count is a conservative lower bound.
    For ``variants`` the result is the sum across all variants.
    """
    if t.variants:
        return sum(n_eff(v) for v in t.variants)

    constrained: set[str] = {name for group in t.distinct for name in group}
    total = 1.0

    # Independent slots
    for name, s in t.slots.items():
        if name in constrained:
            continue
        if isinstance(s, Slot):
            total *= s.eff_size()
        elif isinstance(s, ExcludeSlot):
            n_excluded = len(s.exclude_from)
            total *= s.eff_size(n_excluded)
        else:  # Sub or ParamSub
            total *= s.eff_size()

    # Distinct groups
    for group in t.distinct:
        # Union pool size
        union: set[str] = set()
        idx_sum = 0.0
        n_pool_slots = 0
        for name in group:
            s = t.slots[name]
            if isinstance(s, (Slot, ExcludeSlot)):
                union.update(s.pool)
                idx_sum += s.idx
                n_pool_slots += 1
        avg_idx = idx_sum / n_pool_slots if n_pool_slots else 0.0
        eff_size = len(union) * _eff_mult(avg_idx)
        total *= math.perm(round(eff_size), len(group))

    return total


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def S(pool: list[str] | tuple[str, ...], idx: float = 0.0) -> Slot:
    """Shorthand for ``Slot(tuple(pool), idx=idx)``."""
    return Slot(tuple(pool), idx=idx)


def X(
    pool: list[str] | tuple[str, ...],
    exclude: list[str] | tuple[str, ...],
    idx: float = 0.0,
) -> ExcludeSlot:
    """Shorthand for ``ExcludeSlot(tuple(pool), tuple(exclude), idx=idx)``."""
    return ExcludeSlot(tuple(pool), tuple(exclude), idx=idx)


def E(gen: Callable[[random.Random], str], n: float = 1e4) -> Sub:
    """Shorthand for ``Sub(gen, n_eff_estimate=n)``."""
    return Sub(gen, n_eff_estimate=n)


def P(
    gen: Callable[[random.Random, str], str],
    param: str,
    n: float = 1e4,
) -> ParamSub:
    """Shorthand for ``ParamSub(gen, param_slot=param, n_eff_estimate=n)``."""
    return ParamSub(gen, param_slot=param, n_eff_estimate=n)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

_N_EFF_CAP_DEFAULT: float = 1_000_000


def compute_weights(
    templates: list[Template],
    cap: float = _N_EFF_CAP_DEFAULT,
) -> list[float]:
    """Compute sqrt-compressed n_eff weights for weighted template sampling."""
    return [math.sqrt(min(n_eff(t), cap)) for t in templates]


def make_dispatcher(
    templates: list[Template],
    weights: list[float],
) -> Callable[[random.Random], str]:
    """Return a dispatch function that samples a template by weight and renders it."""

    def _dispatch(rng: random.Random) -> str:
        t = rng.choices(templates, weights=weights)[0]
        return sample(t, rng)

    return _dispatch
