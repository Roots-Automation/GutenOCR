"""Sample record schema for benchmark-grade corpus output.

A ``Sample`` is the rich per-sample record emitted when ``include_metadata=True``
and the full provenance fields are requested.  It is a superset of the legacy
``{"formula": ..., "domain": ..., "template_name": ...}`` dict: projecting to
those three keys gives the old format.

The ``semantic_key`` is a stable SHA-256 fingerprint of ``(template_name, draws)``
that identifies the sample's mathematical identity independent of surface formatting.
Two samples with the same key are semantically equivalent (same template, same slot
assignments); two with different keys may still produce the same LaTeX string by
coincidence but are treated as distinct instances.
"""

from __future__ import annotations

import hashlib
import json
from typing import TypedDict


class Sample(TypedDict, total=False):
    # --- core identity ---
    formula: str  # the LaTeX formula string (always present)
    domain: str  # e.g. "algebra"
    template_name: str | None  # leaf template name; None for Python-module domains with no DSL
    draws: dict[str, str]  # slot-name → drawn value (only when include_draws=True)
    semantic_key: str  # sha256(template_name + canonical draws); stable across runs

    # --- structural metrics (from Template, precomputed at registration) ---
    difficulty: str  # elementary | undergraduate | graduate | research
    depth: int  # max LaTeX brace-nesting depth
    char_length: int  # character-length proxy of the template string
    has_fraction: bool
    has_matrix: bool
    has_integral: bool
    has_script_chain: bool
    strata: list[str]  # sorted list of SYMBOL_STRATA names exercised
    n_eff: float  # analytic effective output-space size
    symbol_tier: str | None  # head | body | tail — highest-rarity tier exercised

    # --- render gate (WU2) ---
    render_ok: bool | None  # True = rendered cleanly; None = not rendered
    image_path: str | None  # path to PNG (only when rendered)

    # --- pack provenance (WU4) ---
    content_pack_hash: str | None
    content_pack_name: str | None
    content_pack_version: str | None
    content_pack_author: str | None
    content_pack_license: str | None

    # --- split tag ---
    split: str  # "train" | "held_out_domain" | "held_out_template" | "held_out_symbol"


def make_semantic_key(template_name: str | None, draws: dict[str, str]) -> str:
    """Return a stable SHA-256 fingerprint of (template_name, draws).

    The key is hex-encoded and truncated to 32 characters (128 bits of collision
    resistance — ample for corpus-scale dedup).
    """
    payload = f"{template_name or ''}\x00{json.dumps(sorted(draws.items()), ensure_ascii=False)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
