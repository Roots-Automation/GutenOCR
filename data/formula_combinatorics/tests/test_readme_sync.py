"""Tests that README.md stays in sync with the live registry and CLI surface."""

from __future__ import annotations

from pathlib import Path

from formula_combinatorics.domains import GENERATORS

_README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
_DOMAIN_COUNT = len(GENERATORS)


def test_readme_domain_count_matches_registry() -> None:
    """README must state the correct number of registered domains."""
    assert str(_DOMAIN_COUNT) in _README, (
        f"README does not contain the current domain count ({_DOMAIN_COUNT}). "
        "Update README.md to reflect the live registry."
    )


def test_readme_has_no_stale_align_fraction_param() -> None:
    """--align-fraction was removed; its presence in the README signals re-drift."""
    assert "--align-fraction" not in _README, (
        "README still references the removed --align-fraction parameter. "
        "The current CLI uses --display-fraction and --inline-fraction."
    )


def test_readme_has_no_stale_align_fraction_api_arg() -> None:
    """align_fraction= was removed from the Python API; presence signals re-drift."""
    assert "align_fraction" not in _README, "README still references the removed align_fraction= API argument."


def test_readme_documents_display_fraction() -> None:
    """README must document the current --display-fraction flag."""
    assert "--display-fraction" in _README, "README is missing documentation for --display-fraction."


def test_readme_documents_inline_fraction() -> None:
    """README must document the current --inline-fraction flag."""
    assert "--inline-fraction" in _README, "README is missing documentation for --inline-fraction."
