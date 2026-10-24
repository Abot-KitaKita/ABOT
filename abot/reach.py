"""Walking reach from facilities and candidate sites to grid cells.

OWNER: B. Outputs: data/processed/reach_facilities.parquet (<= 2,250 m)
and data/processed/reach_candidates.parquet (<= 1,125 m). Contract: CONTEXT.md section 4.
Compute on the OSM walking network in EPSG:32651; snap points to the nearest network node.

Method: each site and cell centroid is snapped to its nearest graph node; walk_m is the
shortest-path distance along the walking network between the snapped nodes ("length"
edge attribute, metres). Snap residuals (point -> node) are not added; this is stated
in docs/methods.md as a known approximation.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from abot import paths
from abot.contracts import REACH_CAND_MAX_M, REACH_FAC_MAX_M

GRAPH_PATH = paths.RAW / "walk.graphml"


def _snap(graph, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Nearest graph node for each (lon, lat) point, as a plain numpy haversine search.

    No osmnx/scikit-learn dependency: the graph is small (~4k nodes) so a vectorised
    brute-force search is fast and needs no optional libraries (CONTEXT.md rule 9).
    """
    from pyproj import CRS

    node_ids = np.asarray(list(graph.nodes))
    xs = np.array([graph.nodes[n]["x"] for n in node_ids], dtype=float)
    ys = np.array([graph.nodes[n]["y"] for n in node_ids], dtype=float)
    crs = CRS.from_user_input(graph.graph["crs"]) if graph.graph.get("crs") else None
    geographic = crs is None or crs.is_geographic
    if geographic:
        xs, ys = np.radians(xs), np.radians(ys)

    out = np.empty(len(lon), dtype=node_ids.dtype)
    for start in range(0, len(lon), 512):  # batch to bound memory
        lo = np.asarray(lon[start:start + 512], dtype=float)[:, None]
        la = np.asarray(lat[start:start + 512], dtype=float)[:, None]
        if geographic:
            lo, la = np.radians(lo), np.radians(la)
            dlat = ys[None, :] - la
            dlon = xs[None, :] - lo
            a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(ys)[None, :] * np.sin(dlon / 2) ** 2
            d = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # mean Earth radius, metres
        else:
            d = np.hypot(xs[None, :] - lo, ys[None, :] - la)
        out[start:start + 512] = node_ids[d.argmin(axis=1)]
    return out


def compute_reach(graph, sites: pd.DataFrame, id_col: str, grid: pd.DataFrame, max_m: float) -> pd.DataFrame:
    """Return columns [id_col, 'cell_id', 'walk_m'] for every site-cell pair within max_m."""
    if id_col not in sites.columns:
        raise KeyError(f"{id_col!r} not in sites columns {list(sites.columns)}")
    site_lon = sites.geometry.x.to_numpy(dtype=float)
    site_lat = sites.geometry.y.to_numpy(dtype=float)
    site_nodes = _snap(graph, site_lon, site_lat)

    cell_lon = grid["x"].to_numpy(dtype=float)
    cell_lat = grid["y"].to_numpy(dtype=float)
    cell_nodes = _snap(graph, cell_lon, cell_lat)
    cell_ids = grid["cell_id"].astype(str).to_numpy()
    cells_by_node: dict[int, list[str]] = {}
    for node, cell_id in zip(cell_nodes, cell_ids):
        cells_by_node.setdefault(int(node), []).append(cell_id)

    rows: list[dict] = []
    for site_id, node in zip(sites[id_col].astype(str), site_nodes):
        dists = nx.single_source_dijkstra_path_length(graph, int(node), cutoff=float(max_m), weight="length")
        for node2, dist in dists.items():  # includes the site's own node at 0 m: its cell is covered
            for cell_id in cells_by_node.get(node2, ()):
                rows.append({id_col: site_id, "cell_id": cell_id, "walk_m": round(float(dist), 1)})
    out = pd.DataFrame(rows, columns=[id_col, "cell_id", "walk_m"])
    return out.sort_values([id_col, "cell_id"], kind="mergesort").reset_index(drop=True)


def city_hall_walk_m(graph, candidates: pd.DataFrame, city_hall: dict) -> pd.Series:
    """Walk distance (m) from City Hall to each candidate site, indexed by cand_id.

    One Dijkstra from the snapped City Hall node; candidates not reached within the
    referral limit (2,250 m) get NaN and are excluded from the nearest-City-Hall baseline.
    """
    hall_nodes = _snap(graph, np.array([float(city_hall["lon"])]), np.array([float(city_hall["lat"])]))
    dists = nx.single_source_dijkstra_path_length(
        graph, int(hall_nodes[0]), cutoff=REACH_FAC_MAX_M, weight="length"
    )
    cand_nodes = _snap(graph, candidates.geometry.x.to_numpy(float), candidates.geometry.y.to_numpy(float))
    out = {
        str(cid): (float(dists[int(node)]) if int(node) in dists else np.nan)
        for cid, node in zip(candidates["cand_id"].astype(str), cand_nodes)
    }
    return pd.Series(out, dtype=float).sort_index()


def load_graph(path: Path | str = GRAPH_PATH):
    """Load the saved OSM walking network (raises with a clear message if missing)."""
    import osmnx as ox

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Walking network not found: {path}. Run scripts/download_walk.py first (owner B)."
        )
    return ox.load_graphml(path)


def main() -> None:
    """Build both reach files from data/raw walking network + processed grid/facilities/candidates."""
    if paths.USING_FIXTURES:
        raise SystemExit("ABOT_DATA=fixtures: fixtures are read-only inputs. Run on real data.")
    from abot import io

    graph = load_graph()
    grid = io.read("grid")
    facilities = io.read("facilities")
    candidates = io.read("candidates")

    rf = compute_reach(graph, facilities, "fac_id", grid, REACH_FAC_MAX_M)
    rc = compute_reach(graph, candidates, "cand_id", grid, REACH_CAND_MAX_M)
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    rf.to_parquet(paths.PROCESSED / "reach_facilities.parquet", index=False)
    rc.to_parquet(paths.PROCESSED / "reach_candidates.parquet", index=False)

    empty_f = sorted(set(facilities["fac_id"].astype(str)) - set(rf["fac_id"]))
    empty_c = sorted(set(candidates["cand_id"].astype(str)) - set(rc["cand_id"]))
    print(f"reach_facilities: {len(rf):,} pairs, {rf['fac_id'].nunique()} facilities")
    print(f"reach_candidates: {len(rc):,} pairs, {rc['cand_id'].nunique()} candidates")
    if empty_f:
        print(f"WARNING: {len(empty_f)} facilities reach no cell within {REACH_FAC_MAX_M:.0f} m: {empty_f[:5]}")
    if empty_c:
        print(f"WARNING: {len(empty_c)} candidates reach no cell within {REACH_CAND_MAX_M:.0f} m: {empty_c[:5]}")


if __name__ == "__main__":
    main()
