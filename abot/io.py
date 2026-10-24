"""Read contract files safely and turn reach tables into sets.

Everyone (including the app) should load data through read(), so a broken file
fails loudly at load time instead of producing wrong numbers later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from abot.contracts import SPECS, read_file, validate


class ContractError(ValueError):
    """A file does not match its contract in abot/contracts.py."""


def read(key: str, path: Path | str | None = None, strict: bool = True) -> pd.DataFrame:
    """Load a contract file (e.g. read('grid')) and validate it.

    strict=True raises ContractError on any violation; strict=False only warns.
    """
    if key not in SPECS:
        raise KeyError(f"Unknown contract {key!r}. Known: {list(SPECS)}")
    df = read_file(key, path)
    errors = validate(key, df)
    if errors:
        msg = f"{key} breaks its contract:\n  - " + "\n  - ".join(errors)
        if strict:
            raise ContractError(msg)
        print(f"WARNING: {msg}")
    return df


def reach_sets(reach: pd.DataFrame, id_col: str, max_m: float) -> dict[str, set[str]]:
    """Map each site id to the set of cell_ids within max_m walking metres.

    reach: a reach_facilities or reach_candidates table; id_col: 'fac_id' or 'cand_id'.
    Sites with no reachable cell are kept with an empty set.
    """
    if id_col not in reach.columns:
        raise KeyError(f"{id_col!r} not in reach table columns {list(reach.columns)}")
    within = reach[reach["walk_m"] <= max_m]
    out = {str(k): set(map(str, g["cell_id"])) for k, g in within.groupby(id_col)}
    for site in reach[id_col].astype(str).unique():
        out.setdefault(site, set())
    return out


def covered_cells(reach_facilities: pd.DataFrame, counted_fac_ids: Iterable[str], max_m: float) -> set[str]:
    """Cells within max_m of at least one counted facility (i.e. already covered)."""
    counted = set(map(str, counted_fac_ids))
    rf = reach_facilities[reach_facilities["fac_id"].astype(str).isin(counted) & (reach_facilities["walk_m"] <= max_m)]
    return set(rf["cell_id"].astype(str))


def nearest_walk_m(reach_facilities: pd.DataFrame, counted_fac_ids: Iterable[str]) -> pd.Series:
    """Shortest walk (m) from each cell to any counted facility, for cells within the reach table limit."""
    counted = set(map(str, counted_fac_ids))
    rf = reach_facilities[reach_facilities["fac_id"].astype(str).isin(counted)]
    return rf.groupby("cell_id")["walk_m"].min()
