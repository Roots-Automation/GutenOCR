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
Slot           — draw one symbol from a fixed pool; optional _maybe_idx decoration
ExcludeSlot    — like Slot, but excludes values already drawn by named sibling slots
Sub            — call a sub-generator (e.g. _expr, _atom); n_eff is an estimate
ParamSub       — sub-generator whose second arg is the value of another drawn slot
                 (used for generators like _poly(rng, v))
ExcludeParamSub — like ParamSub, but also passes a frozenset of already-drawn values
                 from named sibling slots so the generator can filter its pool
                 (signature: gen(rng, param_value, exclude: frozenset[str]) -> str)

Convenience constructors: ``S``, ``X``, ``E``, ``P``, ``EP`` (see module bottom).

Reusable slot constants (bottom of this module)
------------------------------------------------
Use these instead of spelling out the constructor every time::

    _LIM_MOD     — S(("", r"\\limits"))            optional \\limits modifier
    _VAR_SLOT    — S(_VARS, idx=0.35)               bound variable  x, y, z, …
    _SCALAR_SLOT — S(_SCALARS)                      scalar          a, b, c, k, …
    _GREEK_SLOT  — S(_GREEK)                        Greek letter    α, β, γ, …
    _INDEX_SLOT  — S(_INDICES)                      index           i, j, k, …
    _EXPR_SLOT   — E(_expr, n=5_000)                generic expression
    _ATOM_SLOT   — E(_atom, n=150)                  atomic symbol
    _FN_SLOT     — E(_fn_rich_nosub, n=100)         function name (no subscript)
    _FN_RICH_SLOT— E(_fn_rich, n=272)               function name with decoration

Slot naming convention
----------------------
Use these standard names in ``slots`` dicts so that template metadata is
consistent across domains (important for filtering / benchmarks)::

    var   — the primary bound variable  (x, y, z, …)
    fn    — a function name             (f, g, F, …); fn1/fn2 if multiple
    expr  — a generic sub-expression
    coef  — a scalar / coefficient      (a, b, k, …)
    idx   — an index or subscript       (i, j, k, …)
    mat   — a matrix name               (A, B, M, …)
    rel   — a relation symbol           (=, \\leq, \\sim, …)

Domain-specific parameter names are acceptable as extensions when they carry
clear semantic meaning (e.g. ``rv`` for random variable, ``lam`` for lambda,
``mu``, ``sig``, ``op`` for operator pool).  Avoid opaque abbreviations like
``kk``, ``nn``, ``vv`` — prefer the canonical names above with numeric suffixes
(``idx1``/``idx2``) when multiple slots of the same kind are needed.

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
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Index decoration pool (mirrors _maybe_idx in _vocab.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Greek-macro concatenation guard
# ---------------------------------------------------------------------------

_GREEK_CONCAT_RE = re.compile(
    r"\\(?:"
    # Greek letters (lowercase)
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu(?!lti)|nu|xi|pi(?!tchfork)|varpi|rho|varrho|sigma|varsigma|tau|"
    r"upsilon|phi|varphi|chi|psi|omega|"
    # Greek letters (uppercase)
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    # Other common single-purpose math commands.
    # Negative lookaheads prevent matching valid longer commands:
    #   \to -> exclude \top (?!p)
    #   \cdot -> exclude \cdots (?!s)
    r"partial|nabla|ell|neg|Box|Diamond|square|lozenge|arg|"
    r"to(?!p)|"
    r"cdot(?!s)|"
    r"quad|qquad"  # \quad/\qquad followed by a letter → \quadX (undefined command)
    r")[A-Za-z]"
)

# Double-subscript guard: catches p_2_1 style (unbraced single-char subscript
# followed immediately by another subscript operator).
_DOUBLE_SUB_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]_")


def _check_no_greek_concat(latex: str, name: str) -> None:
    m = _GREEK_CONCAT_RE.search(latex)
    if m:
        snippet = latex[max(0, m.start() - 5) : m.end() + 5]
        raise ValueError(
            f"Template '{name}': Greek macro directly concatenated with a letter "
            f"at position {m.start()}: ...{snippet!r}..."
        )
    m = _DOUBLE_SUB_RE.search(latex)
    if m:
        snippet = latex[max(0, m.start() - 5) : m.end() + 5]
        raise ValueError(f"Template '{name}': double subscript (unbraced) at position {m.start()}: ...{snippet!r}...")


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
        if not pool:
            raise ValueError(
                f"ExcludeSlot pool is empty after applying exclusions {sorted(exclude)!r}. "
                f"Original pool has {len(self.pool)} entries; check that exclude_from "
                f"doesn't exhaust the pool."
            )
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


@dataclass(frozen=True)
class ExcludeParamSub:
    """Like ``ParamSub``, but also passes a frozenset of already-drawn base symbols.

    The generator signature is ``gen(rng, param_value, exclude) -> str`` where
    ``exclude`` is a ``frozenset`` of base symbols drawn by ``exclude_slots``
    (decoration stripped).  Use this instead of encoding multiple exclusions as
    a pipe-separated string in a ``ParamSub``.

    Example: draw a coefficient excluding both n and c1::

        "c2": ExcludeParamSub(_lin_rec_b, param_slot="n",
                              exclude_slots=("n", "c1"), n_eff_estimate=7)
    """

    gen: Callable[[random.Random, str, frozenset[str]], str]
    param_slot: str
    exclude_slots: tuple[str, ...]
    n_eff_estimate: float = 1e4

    def draw(self, rng: random.Random, draws: dict[str, str]) -> str:
        param = draws[self.param_slot]
        exclude = frozenset(draws[s].split("_{")[0].split("_")[0] for s in self.exclude_slots if s in draws)
        return self.gen(rng, param, exclude)

    def eff_size(self) -> float:
        return self.n_eff_estimate


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_SlotType = Slot | ExcludeSlot | Sub | ParamSub | ExcludeParamSub


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
        elif isinstance(s, (ParamSub, ExcludeParamSub)):
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

    result = t.latex.format(**draws)
    _check_no_greek_concat(result, t.name)
    return result


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
        else:  # Sub, ParamSub, or ExcludeParamSub
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


def EP(
    gen: Callable[[random.Random, str, frozenset[str]], str],
    param: str,
    exclude: list[str] | tuple[str, ...],
    n: float = 1e4,
) -> ExcludeParamSub:
    """Shorthand for ``ExcludeParamSub(gen, param_slot=param, exclude_slots=..., n_eff_estimate=n)``."""
    return ExcludeParamSub(gen, param_slot=param, exclude_slots=tuple(exclude), n_eff_estimate=n)


# Reusable slot for the optional \limits modifier on \sum, \prod, \int, etc.
_LIM_MOD = S(("", r"\limits"))


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


def register_domain(
    name: str,
    templates: list[Template],
    weight: float,
    cap: float = _N_EFF_CAP_DEFAULT,
) -> tuple[
    dict[str, Callable[[random.Random], str]],
    dict[str, float],
    dict[str, list[Template]],
]:
    """Build the three public registry dicts for a single-key domain module.

    Returns (GENERATORS, WEIGHTS, TEMPLATES) ready for tuple-unpacking::

        GENERATORS, WEIGHTS, TEMPLATES = register_domain("algebra", _TEMPLATES, 0.09)

    Domain modules should call the wrapper in ``domains._config`` instead, which
    resolves weight and cap from the central ``DOMAIN_CONFIG`` dict.
    """
    w = compute_weights(templates, cap=cap)
    return {name: make_dispatcher(templates, w)}, {name: weight}, {name: templates}


# ---------------------------------------------------------------------------
# Reusable slot constants
# ---------------------------------------------------------------------------
# Import vocabulary here (not at module top) to keep _vocab.py independent of
# the template DSL — neither module imports the other at its own top level.
# ---------------------------------------------------------------------------

from ._vocab import (  # noqa: E402
    _GREEK,
    _INDICES,
    _SCALARS,
    _VARS,
    _atom,
    _expr,
    _fn_rich,
    _fn_rich_nosub,
)

# Fixed-pool variable / scalar / index slots
_VAR_SLOT: Slot = S(tuple(_VARS), idx=0.35)
_SCALAR_SLOT: Slot = S(tuple(_SCALARS))
_GREEK_SLOT: Slot = S(tuple(_GREEK))
_INDEX_SLOT: Slot = S(tuple(_INDICES))

# Sub-generator slots — n_eff estimates match the corpus-wide defaults so that
# weighting is consistent across all domains that adopt these constants.
_EXPR_SLOT: Sub = E(_expr, n=5_000)
_ATOM_SLOT: Sub = E(_atom, n=150)
_FN_SLOT: Sub = E(_fn_rich_nosub, n=100)
_FN_RICH_SLOT: Sub = E(_fn_rich, n=272)
