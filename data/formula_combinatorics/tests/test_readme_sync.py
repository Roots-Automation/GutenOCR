"""Tests that README.md stays in sync with the live registry and CLI surface."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from formula_combinatorics.domains import GENERATORS

_README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
_DOMAIN_COUNT = len(GENERATORS)


def _live_cli_flags() -> list[str]:
    """Return all optional flag strings from the live argparser (e.g. '--display-fraction')."""
    # Prevent main() from calling setup_logging or parse_args during import.
    import formula_combinatorics.generate as _gen_mod

    flags: list[str] = []
    captured: list[argparse.ArgumentParser] = []

    def _capture(self: argparse.ArgumentParser, args: object = None, namespace: object = None) -> argparse.Namespace:
        captured.append(self)
        raise SystemExit(0)

    with patch.object(argparse.ArgumentParser, "parse_args", _capture):
        try:
            _gen_mod.main()
        except SystemExit:
            pass

    if captured:
        for action in captured[0]._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            for opt in action.option_strings:
                if opt.startswith("--"):
                    flags.append(opt)

    return flags


def test_readme_domain_count_matches_registry() -> None:
    """README must state the correct number of registered domains."""
    assert str(_DOMAIN_COUNT) in _README, (
        f"README does not contain the current domain count ({_DOMAIN_COUNT}). "
        "Update README.md to reflect the live registry."
    )


def test_readme_has_no_stale_align_fraction_api_arg() -> None:
    """align_fraction= was removed from the Python API; presence signals re-drift."""
    assert "align_fraction" not in _README, "README still references the removed align_fraction= API argument."


def test_readme_documents_all_cli_flags() -> None:
    """Every --flag exposed by the live argparser must appear in README.md."""
    flags = _live_cli_flags()
    assert flags, "Could not extract CLI flags from generate.main() — check the parser capture logic."
    missing = [f for f in flags if f not in _README]
    assert not missing, (
        f"README is missing documentation for {len(missing)} CLI flag(s): {missing}. "
        "Add them to the Options table in README.md."
    )
