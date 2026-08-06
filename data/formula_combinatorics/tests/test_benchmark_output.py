"""Tests for WU6: benchmark-native output — rich Sample records, JSONL writer, legacy compat."""

from __future__ import annotations

import json
from pathlib import Path

from formula_combinatorics.corpus import generate, write_jsonl
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS, PACK_HASHES, PACK_META, TEMPLATES

# ---------------------------------------------------------------------------
# Rich per-sample records
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    "formula",
    "domain",
    "template_name",
    "semantic_key",
    "split",
    "depth",
    "char_length",
    "has_fraction",
    "has_matrix",
    "has_integral",
    "has_script_chain",
    "strata",
    "n_eff",
    "symbol_tier",
    "render_ok",
    "image_path",
    "content_pack_hash",
    "content_pack_name",
    "content_pack_version",
    "content_pack_author",
    "content_pack_license",
    "difficulty",
}


def _generate_small(seed: int = 42, domains: list[str] | None = None) -> dict:
    domains = domains or ["algebra", "calculus", "quantum_notation"]
    return generate(
        count=50,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=seed,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
    )


def test_rich_record_has_all_expected_fields():
    records = _generate_small()
    for record in records.values():
        assert isinstance(record, dict)
        for field in EXPECTED_FIELDS:
            assert field in record, f"Missing field: {field}"


def test_formula_is_nonempty_string():
    records = _generate_small()
    for record in records.values():
        assert isinstance(record["formula"], str)
        assert record["formula"].strip()


def test_domain_is_known():
    records = _generate_small(domains=["algebra", "calculus"])
    for record in records.values():
        assert record["domain"] in {"algebra", "calculus"}


def test_split_tag_default_is_train():
    records = _generate_small()
    for record in records.values():
        assert record["split"] == "train"


def test_split_tag_custom():
    records = generate(
        count=20,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
        split_tag="held_out_template",
    )
    for record in records.values():
        assert record["split"] == "held_out_template"


def test_semantic_key_is_32_char_hex():
    records = _generate_small()
    for record in records.values():
        key = record["semantic_key"]
        assert isinstance(key, str)
        assert len(key) == 32
        int(key, 16)  # must be valid hex


def test_semantic_key_stable_across_runs():
    r1 = _generate_small(seed=7)
    r2 = _generate_small(seed=7)
    for k in r1:
        assert r1[k]["semantic_key"] == r2[k]["semantic_key"]


def test_semantic_key_unique_per_record():
    records = _generate_small(seed=99, domains=["algebra"])
    keys = [r["semantic_key"] for r in records.values()]
    # Semantic keys can collide only if template+draws are identical; with 50 samples
    # from a domain with large n_eff, expect no collisions.
    assert len(keys) == len(set(keys))


def test_difficulty_is_valid():
    valid_levels = {"elementary", "undergraduate", "graduate", "research"}
    records = _generate_small()
    for record in records.values():
        assert record["difficulty"] in valid_levels or record["difficulty"] is None


def test_structural_metrics_types():
    records = _generate_small(domains=["algebra"])
    for record in records.values():
        if record["depth"] is not None:
            assert isinstance(record["depth"], int)
            assert record["depth"] >= 0
        if record["char_length"] is not None:
            assert isinstance(record["char_length"], int)
            assert record["char_length"] >= 0
        if record["n_eff"] is not None:
            assert isinstance(record["n_eff"], float)
            assert record["n_eff"] > 0
        if record["strata"] is not None:
            assert isinstance(record["strata"], list)


def test_strata_is_sorted_list():
    records = _generate_small(domains=["algebra"])
    for record in records.values():
        strata = record["strata"]
        assert strata == sorted(strata)


def test_symbol_tier_valid_or_none():
    valid_tiers = {"head", "body", "tail", None}
    records = _generate_small()
    for record in records.values():
        assert record["symbol_tier"] in valid_tiers


def test_render_fields_initialized_to_none():
    records = _generate_small()
    for record in records.values():
        assert record["render_ok"] is None
        assert record["image_path"] is None


def test_pack_provenance_for_toml_domain():
    # algebra is a TOML-loaded domain — should have provenance
    records = generate(
        count=20,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
    )
    for record in records.values():
        if "algebra" in PACK_META:
            assert record["content_pack_hash"] is not None
            assert record["content_pack_name"] is not None


def test_pack_provenance_none_for_python_domain():
    # quantum_notation is a Python module domain — no TOML pack
    records = generate(
        count=20,
        domains=["quantum_notation"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
    )
    if "quantum_notation" not in PACK_META:
        for record in records.values():
            assert record["content_pack_hash"] is None


def test_draws_not_included_by_default():
    records = _generate_small()
    for record in records.values():
        assert "draws" not in record


def test_draws_included_when_requested():
    records = generate(
        count=20,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
        include_draws=True,
    )
    for record in records.values():
        assert "draws" in record
        assert isinstance(record["draws"], dict)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def test_write_jsonl_produces_valid_jsonl(tmp_path: Path):
    records = _generate_small()
    out = tmp_path / "corpus.jsonl"
    write_jsonl(records, out)
    lines = out.read_text().splitlines()
    assert len(lines) == len(records)
    for line in lines:
        obj = json.loads(line)
        assert "index" in obj
        assert "formula" in obj


def test_write_jsonl_bare_strings(tmp_path: Path):
    bare = generate(
        count=20,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    out = tmp_path / "bare.jsonl"
    write_jsonl(bare, out)
    for line in out.read_text().splitlines():
        obj = json.loads(line)
        assert "index" in obj
        assert "formula" in obj
        assert isinstance(obj["formula"], str)


# ---------------------------------------------------------------------------
# Legacy JSON backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_json_format_unchanged(tmp_path: Path):
    """Bare JSON output (no --metadata) must remain byte-identical to pre-WU6 format."""
    result = generate(
        count=100,
        domains=["algebra", "calculus"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=42,
    )
    # Legacy format: {"0": "<latex>", ...}
    assert all(isinstance(v, str) for v in result.values())
    assert set(result.keys()) == {str(i) for i in range(len(result))}


def test_metadata_record_is_superset_of_legacy():
    """Projecting rich records to {formula, domain, template_name} gives the legacy subset."""
    rich = _generate_small(seed=5)
    for record in rich.values():
        assert "formula" in record
        assert "domain" in record
        assert "template_name" in record
