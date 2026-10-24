"""Site placement: greedy equity-weighted maximal covering + three baselines.

OWNER: B. Contract: CONTEXT.md section 4 and tests/test_place.py.
Run `pytest tests/test_place.py -q` while implementing; the tests skip until greedy exists.

Determinism: candidates are always considered in sorted id order and ties go to the
smallest cand_id; randomness only via numpy RandomState(seed) in baseline_random.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def greedy(
    unmet: pd.Series,
    weight: pd.Series,
    reach: dict[str, set[str]],
    k: int,
    poorest2: pd.Series | None = None,
) -> list[dict]:
    """Pick up to k candidate sites that add the most weighted unmet need.

    unmet, weight, poorest2: indexed by cell_id. poorest2 is boolean (wealth_quintile <= 2).
    reach: cand_id -> set of cell_ids within the walking limit.
    Repeat k times: choose the candidate whose NOT-YET-COVERED cells have the largest
    sum(unmet * weight); break ties by cand_id ascending; mark its cells covered.
    Stop early when no candidate adds anything (> 0).

    Returns a list of dicts in pick order with keys:
      rank (1..k), cand_id, added_people (unweighted), added_weighted,
      added_poorest2 (unmet newly covered in poorest2 cells; None if poorest2 is None).
    """
    if k <= 0:
        return []
    unmet = unmet.astype(float)
    weighted = unmet * weight.reindex(unmet.index).fillna(1.0)
    covered: set[str] = set()
    picks: list[dict] = []
    for rank in range(1, k + 1):
        best_id, best_cells, best_score = None, set(), 0.0
        for cand_id in sorted(reach):  # sorted -> deterministic tie-break by cand_id
            new_cells = sorted(c for c in reach[cand_id] if c in unmet.index and c not in covered)
            if not new_cells:
                continue
            score = float(weighted[new_cells].sum())  # sorted order: FP-sum is reproducible
            if score > best_score:  # strictly greater: first (smallest) id wins ties
                best_id, best_cells, best_score = cand_id, new_cells, score
        if best_id is None or best_score <= 0:
            break  # nothing more can be added
        covered |= set(best_cells)
        picks.append({
            "rank": rank,
            "cand_id": best_id,
            "added_people": float(unmet[best_cells].sum()),
            "added_weighted": best_score,
            "added_poorest2": (
                float(unmet[[c for c in best_cells if bool(poorest2.get(c, False))]].sum())
                if poorest2 is not None
                else None
            ),
        })
    return picks


def baseline_most_populous_barangay(grid: pd.DataFrame, candidates: pd.DataFrame, k: int) -> list[str]:
    """For the k most populous barangays, the candidate nearest each barangay's population-weighted centroid.

    Centroids and distances are straight-line in EPSG:32651 (documented simplification of
    the "nearest" rule for this baseline). Ties: smallest psgc, then smallest cand_id.
    """
    if k <= 0:
        return []
    pts = gpd_points(grid[["x", "y"]], "EPSG:4326", "EPSG:32651")
    work = grid[["psgc", "pop_total"]].copy()
    work["px"], work["py"] = pts.x, pts.y
    bars = []
    for psgc, g in work.groupby("psgc", sort=True):
        w = g["pop_total"].clip(lower=0)
        tot = float(w.sum())
        if tot <= 0:
            continue
        bars.append((psgc, tot, float((g["px"] * w).sum() / tot), float((g["py"] * w).sum() / tot)))
    bars.sort(key=lambda r: (-r[1], r[0]))  # most populous first, tie by psgc
    cand_pts = gpd_points(candidates[["geometry"]], "EPSG:4326", "EPSG:32651") \
        if "geometry" in candidates.columns else gpd_points(candidates, "EPSG:4326", "EPSG:32651")
    cx = pd.Series(cand_pts.x.values, index=candidates["cand_id"].astype(str))
    cy = pd.Series(cand_pts.y.values, index=candidates["cand_id"].astype(str))
    picks: list[str] = []
    for _, _, bx, by in bars[:k]:
        d2 = ((cx - bx) ** 2 + (cy - by) ** 2).to_dict()
        picks.append(sorted(d2.items(), key=lambda kv: (kv[1], kv[0]))[0][0])  # ties: cand_id ascending
    return picks


def baseline_nearest_city_hall(candidate_walk_to_city_hall_m: pd.Series, k: int) -> list[str]:
    """The k candidates with the shortest walk to City Hall (Series indexed by cand_id)."""
    if k <= 0:
        return []
    s = candidate_walk_to_city_hall_m.dropna().astype(float)
    s.index = s.index.astype(str)
    best = sorted(s.items(), key=lambda kv: (kv[1], kv[0]))  # distance, then cand_id ascending
    return [cid for cid, _ in best[:k]]


def baseline_random(cand_ids: list[str], k: int, draws: int, seed: int) -> list[list[str]]:
    """`draws` random k-sets of candidate ids, reproducible with `seed`."""
    if k <= 0 or draws <= 0:
        return []
    ids = sorted(map(str, cand_ids))  # sort so the rng draws depend on order, not input shuffles
    rng = np.random.RandomState(seed)
    n = min(k, len(ids))
    return [[ids[j] for j in rng.choice(len(ids), size=n, replace=False)] for _ in range(draws)]


def people_reached(unmet: pd.Series, reach: dict[str, set[str]], chosen: list[str]) -> float:
    """Unweighted unmet need covered by the union of the chosen sites (each cell counted once)."""
    cells: set[str] = set()
    for cand_id in chosen:
        cells |= reach.get(str(cand_id), set())
    cells &= set(unmet.index)
    return float(unmet[sorted(cells)].sum()) if cells else 0.0  # sorted: reproducible FP-sum


def gpd_points(df: pd.DataFrame, from_crs: str, to_crs: str):
    """Project a frame's x/y (or geometry) to `to_crs` and return a GeoSeries of points."""
    import geopandas as gpd

    if "x" in df.columns:
        g = gpd.GeoSeries(gpd.points_from_xy(df["x"], df["y"]), crs=from_crs)
    else:
        g = gpd.GeoSeries(df["geometry"].values, crs=from_crs)
    return g.to_crs(to_crs)
