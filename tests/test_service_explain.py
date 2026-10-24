"""Tests for the service-type rule and explanation sentences."""
from __future__ import annotations

import pandas as pd
import pytest

from abot import config
from abot.contracts import SERVICE_LABELS
from abot.explain import sentence
from abot.service_type import MOBILE, NEEDS_NETWORK, TELECONSULT, assign, site_cell


def _assign(fx, cand_id, grid=None, facilities=None, rf=None):
    return assign(cand_id, fx["grid"] if grid is None else grid,
                  fx["facilities"] if facilities is None else facilities,
                  fx["reach_facilities"] if rf is None else rf, config.load(), candidates=fx["candidates"])


def _cand(fx, cand_id):
    return fx["candidates"].set_index("cand_id").loc[cand_id]


def test_every_candidate_gets_one_valid_label(fx):
    for cand_id in fx["candidates"]["cand_id"]:
        assert _assign(fx, cand_id) in SERVICE_LABELS


def test_weak_network_gets_mobile_clinic(fx):
    grid = fx["grid"].assign(network_tercile=1)
    for cand_id in fx["candidates"]["cand_id"]:
        assert _assign(fx, cand_id, grid=grid) == MOBILE


def test_missing_network_is_flagged(fx):
    grid = fx["grid"].assign(network_tercile=pd.NA)
    assert _assign(fx, "CSYN1", grid=grid) == NEEDS_NETWORK


def test_good_network_with_referral_gets_teleconsult(fx):
    grid = fx["grid"].assign(network_tercile=3)
    cell = site_cell(grid, _cand(fx, "CSYN1").geometry.x, _cand(fx, "CSYN1").geometry.y)
    rf = pd.DataFrame({"fac_id": ["FSYN1"], "cell_id": [cell], "walk_m": [500.0]})
    assert _assign(fx, "CSYN1", grid=grid, rf=rf) == TELECONSULT


def test_good_network_without_referral_gets_mobile(fx):
    grid = fx["grid"].assign(network_tercile=3)
    cell = site_cell(grid, _cand(fx, "CSYN1").geometry.x, _cand(fx, "CSYN1").geometry.y)
    rf = pd.DataFrame({"fac_id": ["FSYN1"], "cell_id": [cell], "walk_m": [2400.0]})  # beyond 30 min
    assert _assign(fx, "CSYN1", grid=grid, rf=rf) == MOBILE


def test_uncounted_facility_is_not_a_referral(fx):
    grid = fx["grid"].assign(network_tercile=3)
    cell = site_cell(grid, _cand(fx, "CSYN1").geometry.x, _cand(fx, "CSYN1").geometry.y)
    rf = pd.DataFrame({"fac_id": ["FSYN3"], "cell_id": [cell], "walk_m": [100.0]})  # FSYN3 counted=False
    assert _assign(fx, "CSYN1", grid=grid, rf=rf) == MOBILE


def test_unknown_candidate_raises(fx):
    with pytest.raises(KeyError):
        _assign(fx, "NOPE")


def test_sentence_has_no_nan():
    text = sentence(1900.0, 2600.0, None, None)
    assert text == "About 1,900–2,600 likely patients."


def test_sentence_full_format():
    assert sentence(1934.0, 2611.0, 24.4, 1) == (
        "About 1,900–2,600 likely patients; nearest listed facility for referral is a 24-minute walk away; lowest wealth quintile.")


def test_sentence_nan_values_left_out():
    assert "nan" not in sentence(40.0, 60.0, float("nan"), float("nan")).lower()


def test_sentence_small_and_equal_counts():
    assert sentence(42.0, 42.4, 0.3, 5) == "About 42 likely patients; nearest listed facility for referral is a 1-minute walk away; highest wealth quintile."
    assert sentence(0.2, 0.4, None, None) == "Fewer than 1 likely patient."


def test_sentence_article():
    assert "an 11-minute walk" in sentence(10, 12, 11.2, None)
    assert "an 8-minute walk" in sentence(10, 12, 8.0, None)
    assert "a 26-minute walk" in sentence(10, 12, 26.0, None)
