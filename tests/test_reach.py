"""Toy tests for owner B's reach.compute_reach (no OSM download needed).

Toy network: a straight line of 5 nodes 0..4, 100 m apart.
  node i: lon = 121.000 + 0.001*i, lat = 14.570  (edges i<->i+1, length 100 m)
  site S1 at node 1; site S2 at node 4; site S3 far away from every cell.
  cells: c2 at node 2, c3 at node 3, c4 at node 4.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point

from abot.reach import compute_reach

MAX_M = 250.0


def toy_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph(crs="EPSG:4326")
    for i in range(5):
        G.add_node(i, x=121.000 + 0.001 * i, y=14.570)
    for i in range(4):
        G.add_edge(i, i + 1, length=100.0)
        G.add_edge(i + 1, i, length=100.0)
    return G


SITES = gpd.GeoDataFrame(
    {"site_id": ["S1", "S2", "S3"]},
    geometry=[Point(121.001, 14.570), Point(121.004, 14.570), Point(121.000, 14.590)],
    crs="EPSG:4326",
)
GRID = pd.DataFrame({
    "cell_id": ["c2", "c3", "c4"],
    "x": [121.002, 121.003, 121.004],
    "y": [14.570, 14.570, 14.570],
})


def test_pairs_and_distances_within_cutoff():
    out = compute_reach(toy_graph(), SITES, "site_id", GRID, MAX_M)
    got = {(r.site_id, r.cell_id, r.walk_m) for r in out.itertuples()}
    # S1 at node 1: c2=100, c3=200 (c4=300 > cutoff). S2 at node 4: c4=0, c3=100, c2=200.
    # S3 snaps to node 0: only c2 (node 2) is within 250 m, at 200 m.
    assert got == {
        ("S1", "c2", 100.0), ("S1", "c3", 200.0),
        ("S2", "c2", 200.0), ("S2", "c3", 100.0), ("S2", "c4", 0.0),
        ("S3", "c2", 200.0),
    }


def test_site_far_from_all_cells_reaches_nothing():
    out = compute_reach(toy_graph(), SITES.iloc[[2]], "site_id", GRID, 50.0)  # S3: nearest cell 200 m
    assert out.empty


def test_site_own_cell_is_covered_at_zero_metres():
    out = compute_reach(toy_graph(), SITES.iloc[[1]], "site_id", GRID, MAX_M)
    row = out[out["cell_id"] == "c4"]
    assert len(row) == 1 and float(row["walk_m"].iloc[0]) == 0.0


def test_output_is_sorted_and_matches_contract_columns():
    out = compute_reach(toy_graph(), SITES, "site_id", GRID, MAX_M)
    assert list(out.columns) == ["site_id", "cell_id", "walk_m"]
    assert out.equals(out.sort_values(["site_id", "cell_id"], kind="mergesort").reset_index(drop=True))


def test_missing_id_column_raises():
    with pytest.raises(KeyError):
        compute_reach(toy_graph(), SITES, "nope", GRID, MAX_M)
