"""Toy tests for owner B's placement baselines and helpers (greedy itself: test_place.py).

The equity-weighted greedy toy (5 cells, 3 sites, known answer):
  unmet:   c1=10 (poor, weight 3), c2=10, c3=10, c4=5, c5=1
  weights: c1=3, others=1
  reach:   A -> {c1, c2},  B -> {c2, c3, c4},  C -> {c5}
  rank 1:  A = 10*3 + 10*1 = 40   vs  B = 10*1 + 10*1 + 5*1 = 25  -> A wins
  rank 2:  B adds {c3, c4} = 15   vs  C adds {c5} = 1            -> B wins
"""
from __future__ import annotations

import pandas as pd
import pytest

from abot.place import (
    baseline_most_populous_barangay,
    baseline_nearest_city_hall,
    baseline_random,
    greedy,
    people_reached,
)

UNMET = pd.Series({"c1": 10.0, "c2": 10.0, "c3": 10.0, "c4": 5.0, "c5": 1.0})
WEIGHT = pd.Series({"c1": 3.0, "c2": 1.0, "c3": 1.0, "c4": 1.0, "c5": 1.0})
REACH = {"A": {"c1", "c2"}, "B": {"c2", "c3", "c4"}, "C": {"c5"}}


def test_greedy_equity_weight_flips_the_pick():
    picks = greedy(UNMET, WEIGHT, REACH, 2)
    assert [p["cand_id"] for p in picks] == ["A", "B"]
    assert picks[0]["added_people"] == pytest.approx(20.0)     # unweighted
    assert picks[0]["added_weighted"] == pytest.approx(40.0)   # 10*3 + 10*1
    assert picks[1]["added_people"] == pytest.approx(15.0)


def test_people_reached_is_the_union():
    assert people_reached(UNMET, REACH, ["A", "B"]) == pytest.approx(35.0)
    assert people_reached(UNMET, REACH, ["A", "A"]) == pytest.approx(20.0)  # no double counting
    assert people_reached(UNMET, REACH, []) == 0.0


def test_random_baseline_is_reproducible():
    draws1 = baseline_random(["S1", "S2", "S3", "S4"], k=2, draws=50, seed=42)
    draws2 = baseline_random(["S4", "S2", "S1", "S3"], k=2, draws=50, seed=42)  # order-independent
    assert draws1 == draws2
    assert len(draws1) == 50 and all(len(d) == 2 for d in draws1)
    assert all(len(set(d)) == 2 for d in draws1)  # no repeats inside a draw
    assert baseline_random(["S1", "S2"], k=2, draws=5, seed=1) != baseline_random(["S1", "S2"], k=2, draws=5, seed=2)


def test_nearest_city_hall_orders_and_ties_by_id():
    s = pd.Series({"C2": 500.0, "C1": 500.0, "C3": 100.0, "C4": 900.0})  # C1/C2 tie
    assert baseline_nearest_city_hall(s, 2) == ["C3", "C1"]
    assert baseline_nearest_city_hall(s, 0) == []


def test_most_populous_barangay_picks_nearest_candidate():
    grid = pd.DataFrame({
        "cell_id": ["g1", "g2", "g3"],
        "x": [121.030, 121.060, 121.061],
        "y": [14.570, 14.590, 14.591],
        "pop_total": [100.0, 50.0, 50.0],
        "psgc": ["B-big", "B-small", "B-small"],
    })
    # candidate ids chosen so C1 is nearest the B-big centroid, C2 nearest the B-small centroid
    candidates = pd.DataFrame({
        "cand_id": ["C1", "C2"],
        "x": [121.030, 121.0605],
        "y": [14.570, 14.5905],
    })
    picked = baseline_most_populous_barangay(grid, candidates, k=2)
    assert picked == ["C1", "C2"]
