"""Integration tests for the formula-generate CLI (generate.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from formula_combinatorics.domains import GENERATORS
from formula_combinatorics.generate import main


def _run(argv: list[str]) -> None:
    with patch("sys.argv", argv):
        main()


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


def test_basic_invocation_creates_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "50", "--seed", "0"])
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_output_keys_are_sequential_string_ints(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "30", "--seed", "1"])
    data = json.load(open(out))
    assert set(data.keys()) == {str(i) for i in range(30)}


def test_output_values_are_nonempty_strings(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "20", "--seed", "2"])
    data = json.load(open(out))
    for k, v in data.items():
        assert isinstance(v, str) and len(v) > 0, f"Empty or non-string value at key {k!r}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility(tmp_path: Path) -> None:
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    _run(["formula-generate", "--output", str(out1), "--count", "40", "--seed", "7"])
    _run(["formula-generate", "--output", str(out2), "--count", "40", "--seed", "7"])
    assert json.load(open(out1)) == json.load(open(out2))


def test_different_seeds_differ(tmp_path: Path) -> None:
    out1 = tmp_path / "seed0.json"
    out2 = tmp_path / "seed1.json"
    _run(["formula-generate", "--output", str(out1), "--count", "40", "--seed", "0"])
    _run(["formula-generate", "--output", str(out2), "--count", "40", "--seed", "99"])
    assert json.load(open(out1)) != json.load(open(out2))


# ---------------------------------------------------------------------------
# --domains filtering
# ---------------------------------------------------------------------------


def test_domains_flag_restricts_output(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "30",
            "--seed",
            "0",
            "--domains",
            "algebra",
            "calculus",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    used_domains = {v["domain"] for v in data.values()}
    assert used_domains <= {"algebra", "calculus"}


def test_unknown_domain_causes_argparse_exit(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(
            [
                "formula-generate",
                "--output",
                str(out),
                "--count",
                "10",
                "--domains",
                "algebra",
                "definitely_not_a_real_domain_xyz",
            ]
        )
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --tags and --exclude-tags
# ---------------------------------------------------------------------------


def test_tags_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "20", "--seed", "0", "--tags", "foundational"])
    data = json.load(open(out))
    assert len(data) == 20


def test_exclude_tags_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--exclude-tags",
            "structural",
        ]
    )
    data = json.load(open(out))
    assert len(data) == 20


# ---------------------------------------------------------------------------
# --metadata flag
# ---------------------------------------------------------------------------


def test_metadata_flag_produces_formula_domain_dicts(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "30", "--seed", "0", "--metadata"])
    data = json.load(open(out))
    for k, v in data.items():
        assert isinstance(v, dict), f"Key {k!r}: expected dict, got {type(v)}"
        assert "formula" in v and "domain" in v
        assert isinstance(v["formula"], str) and len(v["formula"]) > 0
        assert v["domain"] in GENERATORS


# ---------------------------------------------------------------------------
# --display-fraction / --inline-fraction
# ---------------------------------------------------------------------------


def test_no_wrapping_when_fractions_zero(tmp_path: Path) -> None:
    """With fractions=0, the wrapping logic must NOT add any display/inline
    delimiters.  Use non-structural domains so no formula is pre-wrapped
    by the domain itself (structural domains like 'align' produce \begin{} natively)."""
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "50",
            "--seed",
            "0",
            "--display-fraction",
            "0.0",
            "--inline-fraction",
            "0.0",
            "--domains",
            "algebra",
            "calculus",
            "trigonometry",
        ]
    )
    data = json.load(open(out))
    for v in data.values():
        assert not v.startswith(r"\["), f"Unexpected display wrap: {v[:30]!r}"
        assert not v.startswith(r"\begin{"), f"Unexpected begin env: {v[:30]!r}"
        assert not v.startswith("$"), f"Unexpected inline wrap: {v[:30]!r}"


def test_display_fraction_one_wraps_all(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "30",
            "--seed",
            "0",
            "--display-fraction",
            "1.0",
            "--inline-fraction",
            "0.0",
        ]
    )
    data = json.load(open(out))
    for v in data.values():
        is_wrapped = v.startswith(r"\[") or v.startswith(r"\begin{")
        assert is_wrapped, f"Expected display wrap but got: {v[:40]!r}"


# ---------------------------------------------------------------------------
# Fraction out-of-range validation → sys.exit(1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--display-fraction", "-0.1"),
        ("--display-fraction", "1.1"),
        ("--inline-fraction", "-0.1"),
        ("--inline-fraction", "1.1"),
    ],
)
def test_fraction_out_of_range_exits_nonzero(flag: str, value: str, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(["formula-generate", "--output", str(out), "--count", "10", flag, value])
    assert exc_info.value.code != 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--display-fraction", "0.0"),
        ("--display-fraction", "1.0"),
        ("--inline-fraction", "0.0"),
        ("--inline-fraction", "1.0"),
    ],
)
def test_fraction_boundary_values_succeed(flag: str, value: str, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    # Other fraction must be set to 0 to avoid sum > 1 generating an error or weird behaviour
    other_flag = "--inline-fraction" if flag == "--display-fraction" else "--display-fraction"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "5",
            "--seed",
            "0",
            flag,
            value,
            other_flag,
            "0.0",
        ]
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# --count 0 edge case
# ---------------------------------------------------------------------------


def test_count_zero_produces_empty_output(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "0", "--seed", "0"])
    assert out.exists()
    data = json.load(open(out))
    assert isinstance(data, dict)
    assert len(data) == 0


# ---------------------------------------------------------------------------
# Parent directory auto-creation
# ---------------------------------------------------------------------------


def test_output_parent_dirs_autocreated(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "out.json"
    assert not nested.parent.exists()
    _run(["formula-generate", "--output", str(nested), "--count", "5", "--seed", "0"])
    assert nested.exists(), f"Output file not created at {nested}"
