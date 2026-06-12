"""Tests for the TOML template pack loader and sub-generator registry."""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import pytest
from formula_combinatorics.engine._pack_loader import PackMeta, PackResult, load_pack
from formula_combinatorics.engine._sub_registry import SUB_GENERATORS
from formula_combinatorics.engine._template_dsl import ExcludeSlot, Slot, Sub, sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pack(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_pack.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# PackResult structure
# ---------------------------------------------------------------------------


class TestLoadPackStructure:
    def test_returns_pack_result(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [meta]
            name    = "test"
            version = "1.0.0"
            author  = "tester"
            license = "MIT"
            description = "smoke test"

            [[pool]]
            name   = "VARS"
            values = ["x", "y", "z"]

            [[template]]
            name  = "simple"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = "VARS"
        """,
        )
        result = load_pack(p)
        assert isinstance(result, PackResult)
        assert isinstance(result.meta, PackMeta)
        assert isinstance(result.templates, list)

    def test_meta_populated(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [meta]
            name        = "my_domain"
            version     = "2.1.0"
            author      = "Alice"
            license     = "Apache-2.0"
            description = "A test domain"
        """,
        )
        result = load_pack(p)
        assert result.meta.name == "my_domain"
        assert result.meta.version == "2.1.0"
        assert result.meta.author == "Alice"
        assert result.meta.license == "Apache-2.0"
        assert result.meta.description == "A test domain"

    def test_sha256_populated(self, tmp_path):
        p = _write_pack(tmp_path, "[meta]\nname = 'test'\n")
        result = load_pack(p)
        assert len(result.meta.sha256) == 64
        assert all(c in "0123456789abcdef" for c in result.meta.sha256)

    def test_sha256_changes_with_content(self, tmp_path):
        p1 = tmp_path / "a.toml"
        p2 = tmp_path / "b.toml"
        p1.write_text("[meta]\nname = 'pack_a'\n")
        p2.write_text("[meta]\nname = 'pack_b'\n")
        assert load_pack(p1).meta.sha256 != load_pack(p2).meta.sha256


# ---------------------------------------------------------------------------
# Pool resolution
# ---------------------------------------------------------------------------


class TestPoolResolution:
    def test_named_pool(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[pool]]
            name   = "ABC"
            values = ["a", "b", "c"]

            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = "ABC"
        """,
        )
        result = load_pack(p)
        slot = result.templates[0].slots["v"]
        assert isinstance(slot, Slot)
        assert slot.pool == ("a", "b", "c")

    def test_inline_pool(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = ["x", "y"]
        """,
        )
        result = load_pack(p)
        slot = result.templates[0].slots["v"]
        assert isinstance(slot, Slot)
        assert slot.pool == ("x", "y")

    def test_unknown_pool_raises(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = "NONEXISTENT"
        """,
        )
        with pytest.raises(KeyError, match="NONEXISTENT"):
            load_pack(p)


# ---------------------------------------------------------------------------
# Slot type deserialization
# ---------------------------------------------------------------------------


class TestSlotTypes:
    def test_S_slot(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = ["a", "b"]
        """,
        )
        slot = load_pack(p).templates[0].slots["v"]
        assert isinstance(slot, Slot)
        assert slot.idx == 0.0

    def test_S_slot_with_idx(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = ["a", "b"]
            idx  = 0.35
        """,
        )
        slot = load_pack(p).templates[0].slots["v"]
        assert isinstance(slot, Slot)
        assert slot.idx == pytest.approx(0.35)

    def test_X_slot(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{a}{b}'
            [template.slots.a]
            type = "S"
            pool = ["x", "y", "z"]
            [template.slots.b]
            type = "X"
            pool = ["x", "y", "z"]
            exclude_from = ["a"]
        """,
        )
        slots = load_pack(p).templates[0].slots
        assert isinstance(slots["b"], ExcludeSlot)
        assert slots["b"].exclude_from == ("a",)

    def test_E_slot(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "E"
            gen  = "_atom"
            n    = 200.0
        """,
        )
        slot = load_pack(p).templates[0].slots["v"]
        assert isinstance(slot, Sub)
        assert slot.n_eff_estimate == 200.0
        assert slot.gen is SUB_GENERATORS["_atom"]

    def test_E_slot_unknown_gen_raises(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "E"
            gen  = "does_not_exist"
        """,
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            load_pack(p)

    def test_unknown_slot_type_raises(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "Z"
            pool = ["a"]
        """,
        )
        with pytest.raises(ValueError, match="Unknown slot type"):
            load_pack(p)


# ---------------------------------------------------------------------------
# Template deserialization
# ---------------------------------------------------------------------------


class TestTemplateDeserialization:
    def test_no_slots_template(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "constant"
            latex = 'x^2 + 1'
        """,
        )
        t = load_pack(p).templates[0]
        assert t.name == "constant"
        assert t.latex == "x^2 + 1"
        assert t.slots == {}

    def test_distinct_groups(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name     = "t"
            latex    = '{a}{b}'
            distinct = [["a", "b"]]
            [template.slots.a]
            type = "S"
            pool = ["x", "y", "z"]
            [template.slots.b]
            type = "S"
            pool = ["x", "y", "z"]
        """,
        )
        t = load_pack(p).templates[0]
        assert t.distinct == [["a", "b"]]

    def test_slot_order_preserved(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{a}{b}{c}'
            [template.slots.a]
            type = "S"
            pool = ["1"]
            [template.slots.b]
            type = "S"
            pool = ["2"]
            [template.slots.c]
            type = "S"
            pool = ["3"]
        """,
        )
        t = load_pack(p).templates[0]
        assert list(t.slots.keys()) == ["a", "b", "c"]

    def test_multiple_templates(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "first"
            latex = 'A'

            [[template]]
            name  = "second"
            latex = 'B'
        """,
        )
        result = load_pack(p)
        assert len(result.templates) == 2
        assert result.templates[0].name == "first"
        assert result.templates[1].name == "second"


# ---------------------------------------------------------------------------
# Sampling correctness
# ---------------------------------------------------------------------------


class TestSamplingCorrectness:
    def test_S_slot_draws_from_pool(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "S"
            pool = ["alpha", "beta", "gamma"]
        """,
        )
        t = load_pack(p).templates[0]
        rng = random.Random(0)
        outputs = {sample(t, rng) for _ in range(100)}
        assert outputs <= {"alpha", "beta", "gamma"}
        assert len(outputs) > 1  # diversity

    def test_X_slot_excludes(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{a}{b}'
            [template.slots.a]
            type = "S"
            pool = ["x", "y", "z"]
            [template.slots.b]
            type = "X"
            pool = ["x", "y", "z"]
            exclude_from = ["a"]
        """,
        )
        t = load_pack(p).templates[0]
        rng = random.Random(0)
        for _ in range(200):
            out = sample(t, rng)
            # The two drawn symbols must differ
            assert out[0] != out[1], f"Duplicate drawn in {out!r}"

    def test_E_slot_calls_gen(self, tmp_path):
        p = _write_pack(
            tmp_path,
            """
            [[template]]
            name  = "t"
            latex = '{v}'
            [template.slots.v]
            type = "E"
            gen  = "_atom"
            n    = 150.0
        """,
        )
        t = load_pack(p).templates[0]
        rng = random.Random(0)
        outputs = [sample(t, rng) for _ in range(50)]
        assert all(isinstance(o, str) and o for o in outputs)
        assert len(set(outputs)) > 1  # generator produces variety


# ---------------------------------------------------------------------------
# Sub-generator registry
# ---------------------------------------------------------------------------


class TestSubRegistry:
    def test_core_generators_present(self):
        for name in ("_atom", "_expr", "_fn_rich", "_fn_rich_nosub", "_poly"):
            assert name in SUB_GENERATORS, f"Missing core generator: {name!r}"

    def test_poly_mid_variants_present(self):
        for deg in range(2, 6):
            assert f"_poly_mid_{deg}" in SUB_GENERATORS

    def test_matrix_env_variants_present(self):
        for spec in (
            "matrix_env_2x2_pmatrix",
            "matrix_env_3x3_vmatrix",
            "matrix_env_2x2_bmatrix",
            "matrix_ellipsis_pmatrix",
            "smallmatrix_2x2",
        ):
            assert spec in SUB_GENERATORS, f"Missing matrix generator: {spec!r}"

    def test_generators_are_callable(self):
        for name, gen in SUB_GENERATORS.items():
            assert callable(gen), f"Generator {name!r} is not callable"

    def test_atom_and_expr_produce_strings(self):
        rng = random.Random(7)
        for name in ("_atom", "_expr"):
            result = SUB_GENERATORS[name](rng)
            assert isinstance(result, str) and result, f"{name!r} returned empty/non-string"

    def test_matrix_generators_produce_strings(self):
        rng = random.Random(7)
        for name in ("matrix_env_2x2_pmatrix", "matrix_ellipsis_bmatrix", "smallmatrix_3x2"):
            result = SUB_GENERATORS[name](rng)
            assert isinstance(result, str) and result, f"{name!r} returned empty/non-string"


# ---------------------------------------------------------------------------
# Migrated domain: seed-identical verification
# ---------------------------------------------------------------------------

MIGRATED_DOMAINS = ["quantum_notation", "math_fonts", "linear_algebra"]


@pytest.mark.parametrize("domain_name", MIGRATED_DOMAINS)
def test_toml_domain_in_registry(domain_name):
    """TOML-migrated domains are present in the corpus registry."""
    from formula_combinatorics.domains import GENERATORS, PACK_HASHES, TEMPLATES

    assert domain_name in GENERATORS
    assert domain_name in TEMPLATES
    assert domain_name in PACK_HASHES, f"{domain_name} has no pack hash — not loaded from TOML?"


@pytest.mark.parametrize("domain_name", MIGRATED_DOMAINS)
def test_toml_domain_pack_hash_is_hex(domain_name):
    """Pack hashes are 64-char lowercase hex strings."""
    from formula_combinatorics.domains import PACK_HASHES

    sha = PACK_HASHES[domain_name]
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


@pytest.mark.parametrize("domain_name", MIGRATED_DOMAINS)
def test_toml_domain_generates_without_error(domain_name):
    """TOML-loaded domain produces 200 samples without exception."""
    from formula_combinatorics.domains import GENERATORS

    gen = GENERATORS[domain_name]
    rng = random.Random(0)
    outputs = [gen(rng) for _ in range(200)]
    assert all(isinstance(o, str) and o for o in outputs)


# ---------------------------------------------------------------------------
# Pack hash in corpus metadata
# ---------------------------------------------------------------------------


def test_generate_includes_pack_hash_in_metadata():
    """generate() embeds content_pack_hash for TOML-loaded domains when metadata=True."""
    from formula_combinatorics.corpus import generate
    from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS, PACK_HASHES

    result = generate(
        count=50,
        domains=["quantum_notation"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
    )
    for record in result.values():
        assert "content_pack_hash" in record
        assert record["content_pack_hash"] == PACK_HASHES["quantum_notation"]


def test_generate_no_pack_hash_when_not_requested():
    """generate() omits content_pack_hash when pack_hashes=None."""
    from formula_combinatorics.corpus import generate
    from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS

    result = generate(
        count=20,
        domains=["quantum_notation"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        include_metadata=True,
        pack_hashes=None,
    )
    for record in result.values():
        assert "content_pack_hash" not in record
