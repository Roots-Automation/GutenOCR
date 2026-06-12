"""TOML template pack loader.

Loads a ``.toml`` file that declares domain template content in a declarative,
serializable format and returns the same ``list[Template]`` that the equivalent
Python module would produce.

Pack file structure
-------------------
::

    [meta]
    name        = "quantum_notation"   # domain registry name
    version     = "1.0.0"
    author      = "GutenOCR"
    license     = "MIT"
    description = "Quantum notation — Dirac bra-ket coverage"

    [[pool]]
    name   = "STATE_POOL"
    values = ["\\\\psi", "\\\\phi", ...]   # TOML basic strings, or use literals

    [[template]]
    name  = "pure_ket"
    latex = '|{state}\\rangle'             # TOML literal string avoids extra escaping

    [template.slots.state]
    type = "S"
    pool = "STATE_POOL"          # named pool reference

    [[template]]
    name  = "inner_product_bk"
    latex = '\\langle{bra}|{ket}\\rangle'

    [template.slots.bra]
    type = "S"
    pool = "STATE_POOL"

    [template.slots.ket]
    type = "X"
    pool = "STATE_POOL"
    exclude_from = ["bra"]

Slot types
----------
- ``S``  — ``Slot(pool, idx=0.0)``; pool is a pool name or inline list of strings
- ``X``  — ``ExcludeSlot(pool, exclude_from, idx=0.0)``
- ``E``  — ``Sub(gen, n_eff_estimate)``; gen is a name in ``SUB_GENERATORS``
- ``P``  — ``ParamSub(gen, param_slot, n_eff_estimate)``
- ``EP`` — ``ExcludeParamSub(gen, param_slot, exclude_slots, n_eff_estimate)``

Pack hash
---------
``load_pack()`` returns a ``PackResult`` namedtuple that includes the SHA-256
hex digest of the raw TOML bytes.  This digest is used to hash-address corpus
snapshots for benchmark reproducibility.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Python < 3.11 requires the 'tomli' package: pip install tomli") from exc

from ._sub_registry import SUB_GENERATORS
from ._template_dsl import ExcludeParamSub, ExcludeSlot, ParamSub, Slot, Sub, Template


class PackMeta(NamedTuple):
    name: str
    version: str
    author: str
    license: str
    description: str
    sha256: str


class PackResult(NamedTuple):
    templates: list[Template]
    meta: PackMeta


def load_pack(path: Path) -> PackResult:
    """Load a TOML template pack and return templates plus provenance metadata.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.toml`` pack file.

    Returns
    -------
    PackResult
        ``templates`` — ordered list of ``Template`` objects, equivalent to what
        the corresponding Python module would produce.
        ``meta`` — provenance metadata including SHA-256 of the raw file bytes.
    """
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    data = tomllib.loads(raw.decode())

    meta_block = data.get("meta", {})
    meta = PackMeta(
        name=meta_block.get("name", ""),
        version=meta_block.get("version", ""),
        author=meta_block.get("author", ""),
        license=meta_block.get("license", ""),
        description=meta_block.get("description", ""),
        sha256=sha256,
    )

    # Build local pool registry: name → tuple[str, ...]
    pools: dict[str, tuple[str, ...]] = {}
    for pool_def in data.get("pool", []):
        pools[pool_def["name"]] = tuple(pool_def["values"])

    templates = [_load_template(t, pools) for t in data.get("template", [])]

    return PackResult(templates=templates, meta=meta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_pool(pool_ref: str | list[str], pools: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Resolve a pool reference to a concrete tuple of strings."""
    if isinstance(pool_ref, list):
        return tuple(pool_ref)
    if pool_ref not in pools:
        raise KeyError(f"Pool {pool_ref!r} not found in pack. Available pools: {sorted(pools)}")
    return pools[pool_ref]


def _load_slot(
    slot_data: dict, pools: dict[str, tuple[str, ...]]
) -> Slot | ExcludeSlot | Sub | ParamSub | ExcludeParamSub:
    """Deserialize one slot descriptor from its TOML dict."""
    slot_type = slot_data.get("type", "")
    match slot_type:
        case "S":
            return Slot(
                pool=_resolve_pool(slot_data["pool"], pools),
                idx=float(slot_data.get("idx", 0.0)),
            )
        case "X":
            return ExcludeSlot(
                pool=_resolve_pool(slot_data["pool"], pools),
                exclude_from=tuple(slot_data.get("exclude_from", [])),
                idx=float(slot_data.get("idx", 0.0)),
            )
        case "E":
            gen_name = slot_data["gen"]
            if gen_name not in SUB_GENERATORS:
                raise KeyError(
                    f"Sub-generator {gen_name!r} not found in SUB_GENERATORS. Available: {sorted(SUB_GENERATORS)}"
                )
            return Sub(
                gen=SUB_GENERATORS[gen_name],
                n_eff_estimate=float(slot_data.get("n", 1e4)),
            )
        case "P":
            gen_name = slot_data["gen"]
            if gen_name not in SUB_GENERATORS:
                raise KeyError(
                    f"Sub-generator {gen_name!r} not found in SUB_GENERATORS. Available: {sorted(SUB_GENERATORS)}"
                )
            return ParamSub(
                gen=SUB_GENERATORS[gen_name],
                param_slot=slot_data["param"],
                n_eff_estimate=float(slot_data.get("n", 1e4)),
            )
        case "EP":
            gen_name = slot_data["gen"]
            if gen_name not in SUB_GENERATORS:
                raise KeyError(
                    f"Sub-generator {gen_name!r} not found in SUB_GENERATORS. Available: {sorted(SUB_GENERATORS)}"
                )
            return ExcludeParamSub(
                gen=SUB_GENERATORS[gen_name],
                param_slot=slot_data["param"],
                exclude_slots=tuple(slot_data.get("exclude_slots", [])),
                n_eff_estimate=float(slot_data.get("n", 1e4)),
            )
        case _:
            raise ValueError(f"Unknown slot type {slot_type!r}. Expected one of: S, X, E, P, EP")


def _load_template(t_data: dict, pools: dict[str, tuple[str, ...]]) -> Template:
    """Deserialize one template from its TOML dict."""
    slots = {name: _load_slot(slot_data, pools) for name, slot_data in t_data.get("slots", {}).items()}
    variants = [_load_template(v, pools) for v in t_data.get("variants", [])]
    distinct = [list(g) for g in t_data.get("distinct", [])]
    return Template(
        name=t_data["name"],
        latex=t_data["latex"],
        slots=slots,
        distinct=distinct,
        variants=variants,
    )
