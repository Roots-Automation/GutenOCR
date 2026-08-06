"""Config integrity assertions for domains/_config.py and the domain registry."""

from __future__ import annotations

from formula_combinatorics.domains import DEFAULT_WEIGHTS, DOMAIN_DIFFICULTY, DOMAIN_TAGS, GENERATORS, TEMPLATES
from formula_combinatorics.domains._config import DOMAIN_CONFIG

_VALID_DIFFICULTIES = {"elementary", "undergraduate", "graduate", "research"}


# ---------------------------------------------------------------------------
# Weight sums
# ---------------------------------------------------------------------------


def test_default_weights_are_positive() -> None:
    """DEFAULT_WEIGHTS are unnormalized; generate() normalises them internally.
    All values must be strictly positive for meaningful corpus sampling."""
    for name, w in DEFAULT_WEIGHTS.items():
        assert w > 0, f"{name}: weight must be positive, got {w}"


def test_domain_config_weights_are_positive() -> None:
    for name, meta in DOMAIN_CONFIG.items():
        assert meta.weight > 0, f"{name}: weight must be positive, got {meta.weight}"


def test_domain_config_caps_are_positive() -> None:
    for name, meta in DOMAIN_CONFIG.items():
        assert meta.cap > 0, f"{name}: cap must be positive, got {meta.cap}"


# ---------------------------------------------------------------------------
# Key coverage
# ---------------------------------------------------------------------------


def test_every_default_weight_key_in_domain_config() -> None:
    missing = set(DEFAULT_WEIGHTS.keys()) - set(DOMAIN_CONFIG.keys())
    assert not missing, f"Domains in DEFAULT_WEIGHTS but not DOMAIN_CONFIG: {sorted(missing)}"


def test_every_domain_config_key_in_generators() -> None:
    missing = set(DOMAIN_CONFIG.keys()) - set(GENERATORS.keys())
    assert not missing, f"Domains in DOMAIN_CONFIG but not registered as generators: {sorted(missing)}"


def test_generators_and_default_weights_same_keys() -> None:
    assert set(GENERATORS.keys()) == set(DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Identifier validity
# ---------------------------------------------------------------------------


def test_domain_names_are_valid_identifiers() -> None:
    import re

    ident_re = re.compile(r"^[a-z][a-z0-9_]*$")
    bad = [name for name in DOMAIN_CONFIG if not ident_re.match(name)]
    assert not bad, f"Domain names that are not valid lowercase identifiers: {bad}"


# ---------------------------------------------------------------------------
# Difficulty values
# ---------------------------------------------------------------------------


def test_domain_config_difficulty_values_are_valid() -> None:
    for name, meta in DOMAIN_CONFIG.items():
        assert meta.difficulty in _VALID_DIFFICULTIES, (
            f"{name}: unexpected difficulty {meta.difficulty!r}; valid values: {sorted(_VALID_DIFFICULTIES)}"
        )


# ---------------------------------------------------------------------------
# Metadata indices cover all domains
# ---------------------------------------------------------------------------


def test_domain_tags_covers_all_domains() -> None:
    missing = set(DEFAULT_WEIGHTS.keys()) - set(DOMAIN_TAGS.keys())
    assert not missing, f"Domains missing from DOMAIN_TAGS: {sorted(missing)}"


def test_domain_difficulty_covers_all_domains() -> None:
    missing = set(DEFAULT_WEIGHTS.keys()) - set(DOMAIN_DIFFICULTY.keys())
    assert not missing, f"Domains missing from DOMAIN_DIFFICULTY: {sorted(missing)}"


def test_every_domain_has_at_least_one_tag() -> None:
    untagged = [name for name, tags in DOMAIN_TAGS.items() if not tags]
    assert not untagged, f"Domains with no tags: {sorted(untagged)}"


# ---------------------------------------------------------------------------
# Template registry coverage
# ---------------------------------------------------------------------------


def test_templates_registered_for_all_domains() -> None:
    missing = set(DEFAULT_WEIGHTS.keys()) - set(TEMPLATES.keys())
    assert not missing, f"Domains missing from TEMPLATES registry: {sorted(missing)}"


def test_every_domain_has_at_least_one_template() -> None:
    empty = [name for name, ts in TEMPLATES.items() if not ts]
    assert not empty, f"Domains with no templates: {sorted(empty)}"
