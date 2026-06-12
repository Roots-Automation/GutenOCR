"""Shared slot-pool constants for the logic and proof_theory domains."""

from __future__ import annotations

from ..engine._vocab import _BBOLD

_PROP_POOL = (
    "P",
    "Q",
    "R",
    "S",
    "A",
    "B",
    r"\phi",
    r"\psi",
    r"\chi",
    r"\varphi",
    r"\alpha",
    r"\beta",
)  # 12 — proposition letters
_PRED_POOL = ("P", "Q", "R", "F", "G", r"\phi", r"\psi", r"\chi", r"\Phi")  # 9 — predicate names
_VAR_POOL = ("x", "y", "z", "a", "b", "c", "u", "v", "w", "n", "m")  # 11 — individual vars
_BBOLD_POOL = tuple(_BBOLD)  # 9 — \mathbb{...}
_WORLD_POOL = ("w", "u", "v", "s", "t", r"\mathcal{W}", r"\mathcal{M}")  # 7 — Kripke worlds
_TYPE_POOL = (r"\sigma", r"\tau", r"\alpha", r"\beta", r"\gamma", "A", "B", "C")  # 8 — types
_TERM_POOL = ("t", "s", "r", "u", "v", "a", "b", "c")  # 8 — lambda terms
_CMD_POOL = ("C", "S", "T", "P", "Q")  # 5 — program commands
_NUM_POOL = ("0", "1", "2", "3", "n", "m")  # 6 — Church numeral indices
_LL_POOL = (r"\alpha", r"\beta", r"\gamma", r"\delta", "A", "B", "C", "D")  # 8 — linear logic formulas
_UNIV_POOL = ("0", "1", "2", r"\omega", "i", "j")  # 6 — universe levels
_PAIR_POOL = ("p", "q", "r", "s", "e", "d")  # 6 — pair / sigma terms
