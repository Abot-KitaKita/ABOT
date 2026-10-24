"""Tests for abot.place.greedy (owner B). They skip until greedy() is implemented.

Toy example (unmet need per cell, all weights 1):
  cells: c1=10, c2=10, c3=10, c4=5, c5=1
  sites: A -> {c1, c2},  B -> {c2, c3, c4},  C -> {c5},  D -> {} (reaches nothing)
  correct greedy order: B (25), then A (adds only c1 = 10), then C (1). D is never picked.
"""
from __future__ import annotations

import pandas as pd
import pytest

from abot.place import greedy
from tests.conftest import call_or_skip

UNMET = pd.Series({"c1": 10.0, "c2": 10.0, "c3": 10.0, "c4": 5.0, "c5": 1.0})
ONES = pd.Series(1.0, index=UNMET.index)
REACH = {"A": {"c1", "c2"}, "B": {"c2", "c3", "c4"}, "C": {"c5"}, "D": set()}


def ids(picks):
    return [p["cand_id"] for p in picks]


def test_k_zero_returns_nothing():
    assert call_or_skip(greedy, UNMET, ONES, REACH, 0) == []


def test_order_and_no_double_counting():
    picks = call_or_skip(greedy, UNMET, ONES, REACH, 3)
    assert ids(picks) == ["B", "A", "C"]
    assert [p["added_people"] for p in picks] == pytest.approx([25.0, 10.0, 1.0])
    assert [p["rank"] for p in picks] == [1, 2, 3]


def test_total_equals_union_of_covered_cells():
    picks = call_or_skip(greedy, UNMET, ONES, REACH, 3)
    union = set().union(*(REACH[p["cand_id"]] for p in picks))
    assert sum(p["added_people"] for p in picks) == pytest.approx(UNMET[list(union)].sum())


def test_stops_when_nothing_left_to_add():
    picks = call_or_skip(greedy, UNMET, ONES, REACH, 10)
    assert len(picks) == 3 and "D" not in ids(picks)


def test_equity_weight_changes_choice_not_count():
    weight = ONES.copy()
    weight["c1"] = 3.0  # c1 is in a poorer area
    picks = call_or_skip(greedy, UNMET, weight, REACH, 1)
    assert picks[0]["cand_id"] == "A"
    assert picks[0]["added_people"] == pytest.approx(20.0)    # unweighted people
    assert picks[0]["added_weighted"] == pytest.approx(40.0)  # 10*3 + 10*1


def test_tie_broken_by_cand_id():
    reach = {"S2": {"c1"}, "S1": {"c2"}}  # both add 10
    picks = call_or_skip(greedy, UNMET, ONES, reach, 1)
    assert picks[0]["cand_id"] == "S1"


def test_poorest2_counts():
    poorest = pd.Series({"c1": True, "c2": False, "c3": True, "c4": False, "c5": False})
    picks = call_or_skip(greedy, UNMET, ONES, REACH, 2, poorest2=poorest)
    assert [p["added_poorest2"] for p in picks] == pytest.approx([10.0, 10.0])  # B: c3 ; A: c1
    assert call_or_skip(greedy, UNMET, ONES, REACH, 1)[0]["added_poorest2"] is None


def test_cumulative_reach_never_decreases():
    picks = call_or_skip(greedy, UNMET, ONES, REACH, 4)
    assert all(p["added_people"] >= 0 for p in picks)
