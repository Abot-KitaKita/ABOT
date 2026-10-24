"""Shared test helpers. Fixture data in tests/fixtures/ is SYNTHETIC."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


def call_or_skip(fn, *args, **kwargs):
    """Call fn; skip the test (not fail) while the owner has not implemented it yet."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


@pytest.fixture(scope="session")
def fx():
    """Load every synthetic fixture file once."""
    from abot.contracts import read_file

    names = {
        "grid": "grid.parquet",
        "grid_cells": "grid_cells.geojson",
        "facilities": "facilities.geojson",
        "candidates": "candidates.geojson",
        "reach_facilities": "reach_facilities.parquet",
        "reach_candidates": "reach_candidates.parquet",
    }
    missing = [n for n in names.values() if not (FIXTURES / n).exists()]
    if missing:
        pytest.fail(f"fixtures missing {missing}: run python scripts/make_fixtures.py")
    return {k: read_file(k, FIXTURES / v) for k, v in names.items()}
