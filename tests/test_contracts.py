"""Contract checks: fixtures obey them, broken data is caught, real files are checked once delivered."""
from __future__ import annotations

import pytest

from abot import contracts
from abot.contracts import SPECS, cross_checks, validate


@pytest.mark.parametrize("key", ["grid", "grid_cells", "facilities", "candidates", "reach_facilities", "reach_candidates"])
def test_fixtures_meet_contract(fx, key):
    assert validate(key, fx[key]) == []


def test_fixture_links(fx):
    assert cross_checks(fx) == []


def test_duplicate_ids_caught(fx):
    bad = fx["grid"].copy()
    bad.loc[1, "cell_id"] = bad.loc[0, "cell_id"]
    assert any("duplicate" in e for e in validate("grid", bad))


def test_missing_column_caught(fx):
    bad = fx["grid"].drop(columns=["adults_30_79"])
    assert any("missing columns" in e for e in validate("grid", bad))


def test_int_psgc_caught(fx):
    bad = fx["grid"].copy()
    bad["psgc"] = range(len(bad))  # PSGC must stay a string
    assert any(e.startswith("psgc") for e in validate("grid", bad))


def test_adults_above_pop_caught(fx):
    bad = fx["grid"].copy()
    bad.loc[0, "adults_30_79"] = bad.loc[0, "pop_total"] + 10
    assert any("adults_30_79 > pop_total" in e for e in validate("grid", bad))


def test_projected_coordinates_caught(fx):
    bad = fx["grid"].copy()
    bad["x"], bad["y"] = 289000.0, 1612000.0  # metres, not lon/lat
    assert any("outside the Mandaluyong box" in e for e in validate("grid", bad))


def test_reach_limit_caught(fx):
    bad = fx["reach_candidates"].copy()
    bad.loc[0, "walk_m"] = 5000
    assert any("above 1125" in e for e in validate("reach_candidates", bad))


def test_bad_facility_status_caught(fx):
    bad = fx["facilities"].copy()
    bad.loc[0, "status"] = "Verified"
    assert any("status" in e for e in validate("facilities", bad))


def test_broken_link_caught(fx):
    frames = dict(fx)
    frames["reach_candidates"] = fx["reach_candidates"].assign(cell_id="NOPE")
    assert any("not found" in e for e in cross_checks(frames))


@pytest.mark.parametrize("key", list(SPECS))
def test_delivered_real_files_meet_contract(key):
    """Runs on the real data/processed and results/ files once their owners deliver them."""
    spec = SPECS[key]
    if not spec.path.exists():
        pytest.skip(f"{spec.filename} not delivered yet (owner {spec.owner})")
    df = contracts.read_file(key)
    assert validate(key, df) == []
