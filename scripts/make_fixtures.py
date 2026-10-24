"""Generate the SYNTHETIC development dataset in tests/fixtures/.

Everything here is made up (random numbers, straight-line distances x 1.3). It exists
only so the model, app and tests can be built before real data arrives. It must never
appear in the deck, the video or any reported number.

    python scripts/make_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abot.contracts import REACH_CAND_MAX_M, REACH_FAC_MAX_M, validate  # noqa: E402

OUT = ROOT / "tests" / "fixtures"
UTM = "EPSG:32651"
WGS = "EPSG:4326"
NX, NY, STEP = 20, 15, 100.0  # 20 x 15 cells of 100 m = 2.0 km x 1.5 km
DETOUR = 1.3  # synthetic stand-in for network distance


def main() -> None:
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)

    # Origin: a point inside the Mandaluyong bounding box, projected to metres.
    origin = gpd.GeoSeries([Point(121.025, 14.570)], crs=WGS).to_crs(UTM).iloc[0]
    rows, squares = [], []
    for j in range(NY):
        for i in range(NX):
            x0, y0 = origin.x + i * STEP, origin.y + j * STEP
            brgy = 1 + (i >= NX // 2) + 2 * (j >= NY // 2)  # four synthetic barangays by quadrant
            rows.append({"cell_id": f"SYN{j:02d}{i:02d}", "i": i, "j": j, "brgy": brgy,
                         "px": x0 + STEP / 2, "py": y0 + STEP / 2})
            squares.append(box(x0, y0, x0 + STEP, y0 + STEP))
    cells = gpd.GeoDataFrame(rows, geometry=squares, crs=UTM)

    pop = rng.gamma(shape=2.0, scale=60.0, size=len(cells)).round(1)
    adults = (pop * rng.uniform(0.35, 0.50, size=len(cells))).round(1)
    wealth = {1: 1, 2: 3, 3: 2, 4: 5}  # synthetic quintile per barangay
    mbps = {1: 8.0, 2: 25.0, 3: 15.0, 4: 40.0}  # synthetic network speed per barangay
    tercile = {1: 1, 2: 2, 3: 2, 4: 3}

    centroids = gpd.GeoSeries(gpd.points_from_xy(cells["px"], cells["py"]), crs=UTM).to_crs(WGS)
    grid = pd.DataFrame({
        "cell_id": cells.cell_id,
        "x": centroids.x.round(6),
        "y": centroids.y.round(6),
        "pop_total": pop,
        "adults_30_79": adults,
        "psgc": cells.brgy.map(lambda b: f"SYN-{b}"),
        "barangay": cells.brgy.map(lambda b: f"Synthetic barangay {b}"),
        "wealth_quintile": cells.brgy.map(wealth).astype("Int64"),
        "mobile_dl_mbps": cells.brgy.map(mbps).astype(float),
        "network_tercile": cells.brgy.map(tercile).astype("Int64"),
        "source_flags": "SYNTHETIC",
    })
    grid_cells = gpd.GeoDataFrame({"cell_id": cells.cell_id}, geometry=cells.geometry, crs=UTM).to_crs(WGS)

    def pts(spec: list[tuple[int, int]]) -> gpd.GeoSeries:
        return gpd.GeoSeries(
            [Point(origin.x + (i + 0.5) * STEP, origin.y + (j + 0.5) * STEP) for i, j in spec], crs=UTM
        )

    fac_xy = pts([(3, 3), (15, 11), (10, 1)])
    facilities = gpd.GeoDataFrame({
        "fac_id": ["FSYN1", "FSYN2", "FSYN3"],
        "name": ["Synthetic health centre 1", "Synthetic health station 2", "Synthetic hospital 3"],
        "type": ["Health center", "Barangay health station", "Hospital"],
        "status": "Listed, not verified",
        "counted": [True, True, False],
        "source": "SYNTHETIC",
    }, geometry=fac_xy).to_crs(WGS)

    cand_spec = [(1, 12), (6, 9), (9, 6), (12, 13), (17, 3), (18, 8), (4, 0), (14, 6)]
    kinds = ["barangay_hall", "school", "community_centre", "court"]
    cand_xy = pts(cand_spec)
    candidates = gpd.GeoDataFrame({
        "cand_id": [f"CSYN{n + 1}" for n in range(len(cand_spec))],
        "name": [f"Synthetic site {n + 1}" for n in range(len(cand_spec))],
        "kind": [kinds[n % 4] for n in range(len(cand_spec))],
        "psgc": [f"SYN-{1 + (i >= NX // 2) + 2 * (j >= NY // 2)}" for i, j in cand_spec],
    }, geometry=cand_xy).to_crs(WGS)

    cell_pts = gpd.GeoSeries(gpd.points_from_xy(cells["px"], cells["py"]), crs=UTM)

    def reach(site_xy: gpd.GeoSeries, ids: list[str], id_col: str, max_m: float) -> pd.DataFrame:
        out = []
        for sid, p in zip(ids, site_xy):
            d = cell_pts.distance(p) * DETOUR
            keep = d <= max_m
            out.append(pd.DataFrame({id_col: sid, "cell_id": cells.cell_id[keep].values,
                                     "walk_m": d[keep].round(1).values}))
        return pd.concat(out, ignore_index=True)

    reach_fac = reach(fac_xy, list(facilities.fac_id), "fac_id", REACH_FAC_MAX_M)
    reach_cand = reach(cand_xy, list(candidates.cand_id), "cand_id", REACH_CAND_MAX_M)

    outputs = {
        "grid": (grid, "grid.parquet"),
        "grid_cells": (grid_cells, "grid_cells.geojson"),
        "facilities": (facilities, "facilities.geojson"),
        "candidates": (candidates, "candidates.geojson"),
        "reach_facilities": (reach_fac, "reach_facilities.parquet"),
        "reach_candidates": (reach_cand, "reach_candidates.parquet"),
    }
    for key, (df, name) in outputs.items():
        errs = validate(key, df)
        if errs:
            raise SystemExit(f"fixture {key} breaks its contract: {errs}")
        path = OUT / name
        if name.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_file(path, driver="GeoJSON")
        print(f"wrote {path.relative_to(ROOT)}  ({len(df)} rows)")
    print("All fixtures are SYNTHETIC: for development and tests only.")


if __name__ == "__main__":
    main()
