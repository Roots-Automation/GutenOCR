"""Tests for WU6: held-out splits — template hold-out, symbol hold-out, manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from formula_combinatorics.corpus import (
    generate,
    partition_symbols,
    partition_templates,
    write_manifest,
)
from formula_combinatorics.domains import DEFAULT_WEIGHTS, TEMPLATES
from formula_combinatorics.engine.symbol_inventory import MUST_COVER

# ---------------------------------------------------------------------------
# partition_templates
# ---------------------------------------------------------------------------


def test_partition_templates_fraction_coverage():
    train, held, names = partition_templates(TEMPLATES, hold_out=0.1, seed=0)
    total_train = sum(len(v) for v in train.values())
    total_held = sum(len(v) for v in held.values())
    total_orig = sum(len(v) for v in TEMPLATES.values())
    assert total_train + total_held == total_orig


def test_partition_templates_fraction_no_overlap():
    # Template names are unique only within a domain; use (domain, name) pairs.
    train, held, names = partition_templates(TEMPLATES, hold_out=0.1, seed=0)
    train_keys: set[tuple[str, str]] = set()
    held_keys: set[tuple[str, str]] = set()
    for domain, tmpl_list in train.items():
        train_keys.update((domain, t.name) for t in tmpl_list)
    for domain, tmpl_list in held.items():
        held_keys.update((domain, t.name) for t in tmpl_list)
    assert train_keys.isdisjoint(held_keys), "Train and held-out (domain, template_name) pairs overlap"


def test_partition_templates_names_list():
    all_names = [t.name for tmpl_list in TEMPLATES.values() for t in tmpl_list]
    to_hold = all_names[:5]
    train, held, names = partition_templates(TEMPLATES, hold_out=to_hold, seed=0)
    assert sorted(names) == sorted(to_hold)
    train_names = {t.name for tmpl_list in train.values() for t in tmpl_list}
    for n in to_hold:
        assert n not in train_names


def test_partition_templates_deterministic():
    _, _, names1 = partition_templates(TEMPLATES, hold_out=0.15, seed=7)
    _, _, names2 = partition_templates(TEMPLATES, hold_out=0.15, seed=7)
    assert names1 == names2


def test_partition_templates_different_seeds():
    _, _, names1 = partition_templates(TEMPLATES, hold_out=0.15, seed=1)
    _, _, names2 = partition_templates(TEMPLATES, hold_out=0.15, seed=2)
    # Different seeds produce different partitions (probabilistic but reliable with 33 domains)
    assert names1 != names2


# ---------------------------------------------------------------------------
# partition_symbols
# ---------------------------------------------------------------------------


def test_partition_symbols_fraction():
    train, held = partition_symbols(MUST_COVER, hold_out=0.15, seed=0)
    assert len(train) + len(held) == len(MUST_COVER)
    assert train.isdisjoint(held)


def test_partition_symbols_names():
    to_hold = {r"\alpha", r"\beta", r"\gamma"}
    train, held = partition_symbols(MUST_COVER, hold_out=to_hold)
    assert held == to_hold
    assert train.isdisjoint(held)


def test_partition_symbols_deterministic():
    _, held1 = partition_symbols(MUST_COVER, hold_out=0.1, seed=3)
    _, held2 = partition_symbols(MUST_COVER, hold_out=0.1, seed=3)
    assert held1 == held2


# ---------------------------------------------------------------------------
# Template hold-out: zero formula overlap
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_held_out_template_formulas_no_overlap():
    """Train and held-out-template partitions must produce disjoint formula sets."""
    train_t, held_t, held_names = partition_templates(TEMPLATES, hold_out=0.1, seed=42)

    from formula_combinatorics.engine._template_dsl import make_generator

    train_gens = {d: make_generator(tl) for d, tl in train_t.items() if tl}
    held_domains = [d for d in held_t if held_t[d]]
    held_gens = {d: make_generator(tl) for d, tl in held_t.items() if tl}

    train_formulas = generate(
        count=200,
        domains=list(train_gens),
        generators=train_gens,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    if not held_domains:
        pytest.skip("No held-out domains (template list too small)")

    held_formulas = generate(
        count=200,
        domains=held_domains,
        generators=held_gens,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    train_set = set(train_formulas.values())
    held_set = set(held_formulas.values())
    assert train_set.isdisjoint(held_set), (
        f"{len(train_set & held_set)} formulas appear in both train and held-out partitions"
    )


# ---------------------------------------------------------------------------
# Symbol hold-out: strict contamination check
# ---------------------------------------------------------------------------


def test_held_out_symbol_excluded_from_train_templates():
    """Templates exercising held-out symbols must not appear in train generators."""
    from formula_combinatorics.engine._template_dsl import _template_uses_any_symbol, filter_templates

    held_syms = frozenset({r"\alpha", r"\mathbb{R}"})
    for domain, tmpl_list in TEMPLATES.items():
        train = filter_templates(tmpl_list, exclude_symbols=held_syms)
        for t in train:
            assert not _template_uses_any_symbol(t, held_syms), (
                f"Template {t.name!r} in domain {domain!r} uses a held-out symbol but survived train filter"
            )


def test_partition_symbols_then_filter_disjoint():
    _, held_sym_set = partition_symbols(MUST_COVER, hold_out=0.1, seed=0)
    from formula_combinatorics.engine._template_dsl import filter_templates

    for domain, tmpl_list in TEMPLATES.items():
        train = filter_templates(tmpl_list, exclude_symbols=held_sym_set)
        held = [t for t in tmpl_list if t not in train]
        train_names = {t.name for t in train}
        held_names = {t.name for t in held}
        assert train_names.isdisjoint(held_names)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_write_manifest_round_trips(tmp_path: Path):
    _, _, held_names = partition_templates(TEMPLATES, hold_out=0.1, seed=5)
    manifest_path = tmp_path / "corpus.manifest.json"
    write_manifest(
        path=manifest_path,
        seed=5,
        held_out_domains=["physics"],
        held_out_template_names=held_names,
        held_out_symbols=[r"\alpha", r"\beta"],
        train_count=1000,
        held_out_template_count=50,
        held_out_symbol_count=2,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 5
    assert "physics" in manifest["held_out_domains"]
    assert sorted(manifest["held_out_template_names"]) == sorted(held_names)
    assert r"\alpha" in manifest["held_out_symbols"]
    assert manifest["train_count"] == 1000


def test_manifest_held_out_fields_sorted(tmp_path: Path):
    manifest_path = tmp_path / "corpus.manifest.json"
    write_manifest(
        path=manifest_path,
        seed=0,
        held_out_domains=["z_domain", "a_domain"],
        held_out_template_names=["z_tmpl", "a_tmpl"],
        held_out_symbols=[r"\zeta", r"\alpha"],
        train_count=0,
        held_out_template_count=0,
        held_out_symbol_count=0,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["held_out_domains"] == sorted(manifest["held_out_domains"])
    assert manifest["held_out_template_names"] == sorted(manifest["held_out_template_names"])
    assert manifest["held_out_symbols"] == sorted(manifest["held_out_symbols"])
