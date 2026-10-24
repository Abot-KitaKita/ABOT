"""Toy tests for owner B's need module (need, unmet, equity_weight)."""
from __future__ import annotations

import pandas as pd
import pytest

from abot.need import equity_weight, need, unmet

GRID = pd.DataFrame({
    "cell_id": ["c1", "c2", "c3"],
    "adults_30_79": [100.0, 250.0, 0.0],
})


def test_need_scales_adults_by_share():
    n = need(GRID, 0.33)
    assert n.to_dict() == {"c1": 33.0, "c2": 82.5, "c3": 0.0}
    assert n.index.tolist() == ["c1", "c2", "c3"]


def test_need_refuses_share_outside_unit_interval():
    for bad in (0.0, 1.0, 1.2, -0.1):
        with pytest.raises(ValueError):
            need(GRID, bad)


def test_unmet_zeroes_covered_cells():
    n = need(GRID, 0.33)
    u = unmet(n, {"c1", "c3"})
    assert u.to_dict() == {"c1": 0.0, "c2": 82.5, "c3": 0.0}
    assert unmet(n, set()) .to_dict() == n.to_dict()
    assert (unmet(n, {"c1"}) <= n).all()  # unmet <= need always


def test_equity_weight_endpoints_and_missing():
    q = pd.Series({"c1": 1, "c2": 5, "c3": 3, "c4": None}, dtype="Float64")
    w = equity_weight(q, lam=0.5)
    assert w["c1"] == pytest.approx(1.5)   # poorest: 1 + 0.5*4/4
    assert w["c2"] == pytest.approx(1.0)   # richest: 1 + 0.5*0/4
    assert w["c3"] == pytest.approx(1.25)  # middle:  1 + 0.5*2/4
    assert w["c4"] == pytest.approx(1.0)   # missing -> 1.0


def test_equity_weight_lambda_zero_is_flat():
    q = pd.Series({"c1": 1, "c2": 5})
    assert equity_weight(q, lam=0.0).tolist() == [1.0, 1.0]
