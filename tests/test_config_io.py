"""Config safety and reach helpers (owner E)."""
from __future__ import annotations

import pytest

from abot import config
from abot.io import covered_cells, nearest_walk_m, reach_sets


def test_config_loads():
    cfg = config.load()
    assert cfg["walk_minutes"] == 15


def test_walk_limits():
    cfg = config.load()
    assert config.walk_limit_m(cfg) == pytest.approx(1125.0)
    assert config.walk_limit_m(cfg, speed_kmh=4.0) == pytest.approx(1000.0)
    assert config.walk_limit_m(cfg, minutes=30) == pytest.approx(2250.0)


def test_uncited_prevalence_refused():
    cfg = {"prevalence": {"diabetes": {"mid": 0.1, "mid_source": None}}}
    with pytest.raises(config.ConfigError):
        config.prevalence(cfg, "diabetes", "mid")


def test_cited_prevalence_returned():
    cfg = {"prevalence": {"hypertension": {"mid": 0.33, "mid_source": "cited source"}}}
    assert config.prevalence(cfg, "hypertension", "mid") == pytest.approx(0.33)


def test_bad_condition_refused():
    with pytest.raises(config.ConfigError):
        config.prevalence(config.load(), "cancer", "mid")


def test_reach_sets_respect_limit(fx):
    rc = fx["reach_candidates"]
    sets_1000 = reach_sets(rc, "cand_id", 1000)
    sets_1125 = reach_sets(rc, "cand_id", 1125)
    assert set(sets_1000) == set(rc["cand_id"])
    assert all(sets_1000[c] <= sets_1125[c] for c in sets_1000)


def test_uncounted_facilities_ignored(fx):
    rf, fac = fx["reach_facilities"], fx["facilities"]
    counted = fac.loc[fac["counted"], "fac_id"]
    all_ids = fac["fac_id"]
    assert covered_cells(rf, counted, 1125) <= covered_cells(rf, all_ids, 1125)
    nearest = nearest_walk_m(rf, counted)
    assert (nearest >= 0).all()
