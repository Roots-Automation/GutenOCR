"""Shared pytest configuration and fixtures."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (skipped in normal CI; run with -m slow)",
    )
