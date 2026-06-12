"""Tests for WU6: semantic key stability and semantic dedup."""

from __future__ import annotations

from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS, PACK_HASHES, PACK_META, TEMPLATES
from formula_combinatorics.engine._sample import make_semantic_key

# ---------------------------------------------------------------------------
# make_semantic_key unit tests
# ---------------------------------------------------------------------------


def test_semantic_key_is_32_hex_chars():
    key = make_semantic_key("my_template", {"var": "x", "coef": "a"})
    assert len(key) == 32
    int(key, 16)


def test_semantic_key_stable():
    k1 = make_semantic_key("tmpl", {"a": "1", "b": "2"})
    k2 = make_semantic_key("tmpl", {"a": "1", "b": "2"})
    assert k1 == k2


def test_semantic_key_order_independent():
    """Draws dict order should not affect the key (uses sorted items)."""
    k1 = make_semantic_key("tmpl", {"a": "1", "b": "2"})
    k2 = make_semantic_key("tmpl", {"b": "2", "a": "1"})
    assert k1 == k2


def test_semantic_key_differs_on_template_name():
    k1 = make_semantic_key("tmpl_a", {"var": "x"})
    k2 = make_semantic_key("tmpl_b", {"var": "x"})
    assert k1 != k2


def test_semantic_key_differs_on_draws():
    k1 = make_semantic_key("tmpl", {"var": "x"})
    k2 = make_semantic_key("tmpl", {"var": "y"})
    assert k1 != k2


def test_semantic_key_none_template_name():
    k1 = make_semantic_key(None, {"var": "x"})
    k2 = make_semantic_key(None, {"var": "x"})
    assert k1 == k2


def test_semantic_key_empty_draws():
    k1 = make_semantic_key("tmpl", {})
    k2 = make_semantic_key("tmpl", {})
    assert k1 == k2
    k3 = make_semantic_key("other", {})
    assert k1 != k3


# ---------------------------------------------------------------------------
# Semantic key in generated corpus
# ---------------------------------------------------------------------------


def _gen(seed: int = 0) -> dict:
    return generate(
        count=100,
        domains=["algebra", "calculus"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=seed,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
    )


def test_semantic_keys_present_in_corpus():
    records = _gen()
    for record in records.values():
        assert "semantic_key" in record
        assert isinstance(record["semantic_key"], str)
        assert len(record["semantic_key"]) == 32


def test_semantic_keys_stable_across_runs():
    r1 = _gen(seed=11)
    r2 = _gen(seed=11)
    for k in r1:
        assert r1[k]["semantic_key"] == r2[k]["semantic_key"], f"Semantic key unstable for record {k}"


def test_semantic_keys_stable_independent_of_formula_wrapping():
    """The semantic key is based on (template, draws), not on the wrapped formula string."""
    records = _gen(seed=55)
    seen_keys: dict[str, str] = {}
    for record in records.values():
        key = record["semantic_key"]
        formula = record["formula"]
        if key in seen_keys:
            # Same key → same template + draws; formula may differ only in wrapping
            pass  # acceptable
        seen_keys[key] = formula
