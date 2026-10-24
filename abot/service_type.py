"""Service-type rule for a chosen site.

Rule (CONTEXT.md section 3), checked in this order:
  network data missing for the site's barangay               -> "Service type needs network data"
  1. barangay network_tercile == 1 (weakest third)           -> "Mobile clinic stop"
  2. a counted facility within the referral walk (30 min)    -> "Assisted teleconsult point (BHW-staffed)"
  3. otherwise                                                -> "Mobile clinic stop"
"Extended health-station hours" needs capacity data we do not have, so it is never assigned.

The site's barangay comes from the candidate's `psgc`; the site's own grid cell (for the
referral check) is the grid cell whose centroid is nearest to the candidate point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from abot import config as abot_config
from abot.contracts import SERVICE_LABELS

MOBILE, TELECONSULT, NEEDS_NETWORK, _CAPACITY = SERVICE_LABELS
_CANDIDATES_CACHE: dict[str, pd.DataFrame] = {}


def _load_candidates() -> pd.DataFrame:
    from abot import io, paths

    key = str(paths.PROCESSED)
    if key not in _CANDIDATES_CACHE:
        _CANDIDATES_CACHE[key] = io.read("candidates")
    return _CANDIDATES_CACHE[key]


def barangay_tercile(grid: pd.DataFrame, psgc: str) -> int | None:
    """Most common network_tercile among the barangay's cells (None if unknown)."""
    t = pd.to_numeric(grid.loc[grid["psgc"].astype(str) == str(psgc), "network_tercile"], errors="coerce").dropna()
    if t.empty:
        return None
    return int(t.mode().min())  # tie -> the weaker tercile, the cautious choice


def site_cell(grid: pd.DataFrame, lon: float, lat: float) -> str:
    """cell_id of the grid cell whose centroid is nearest to (lon, lat)."""
    # Equirectangular distance is exact enough to pick the nearest 100 m cell.
    dx = (grid["x"].to_numpy() - lon) * np.cos(np.radians(lat))
    dy = grid["y"].to_numpy() - lat
    return str(grid["cell_id"].iloc[int(np.argmin(dx * dx + dy * dy))])


def assign(cand_id: str, grid, facilities, reach_facilities, config, candidates: pd.DataFrame | None = None) -> str:
    """Return exactly one label from SERVICE_LABELS for candidate `cand_id`.

    facilities must carry the current `counted` flags (the app passes the planner's toggles).
    candidates is optional; it is read from the processed data folder when not given.
    """
    cands = _load_candidates() if candidates is None else candidates
    row = cands.loc[cands["cand_id"].astype(str) == str(cand_id)]
    if row.empty:
        raise KeyError(f"candidate {cand_id!r} not found in candidates")
    row = row.iloc[0]

    tercile = barangay_tercile(grid, row["psgc"])
    if tercile is None:
        point = row.geometry
        cell = site_cell(grid, point.x, point.y)
        t = pd.to_numeric(grid.loc[grid["cell_id"].astype(str) == cell, "network_tercile"], errors="coerce").dropna()
        tercile = int(t.iloc[0]) if not t.empty else None
    if tercile is None:
        return NEEDS_NETWORK
    if tercile == 1:
        return MOBILE

    referral_m = abot_config.walk_limit_m(config, minutes=config.get("referral_walk_minutes", 30))
    counted = set(facilities.loc[facilities["counted"].astype(bool), "fac_id"].astype(str))
    cell = site_cell(grid, row.geometry.x, row.geometry.y)
    rf = reach_facilities
    near = rf[(rf["cell_id"].astype(str) == cell) & (rf["fac_id"].astype(str).isin(counted)) & (rf["walk_m"] <= referral_m)]
    return TELECONSULT if not near.empty else MOBILE
