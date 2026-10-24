"""File contracts for every ABOT handoff, as code.

Each owner runs this before handing a file to the next person:

    python -m abot.contracts            # check every delivered file + cross-file links
    python -m abot.contracts grid       # check one file

Exit code 1 means a delivered file breaks its contract. Missing files are reported
as "not delivered yet" and do not fail the check.

Do not change a contract without telling the integrator (E), who updates CONTEXT.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---- Allowed values (shared by the whole codebase) -------------------------
CONDITIONS = ("hypertension", "diabetes")
SCENARIOS = ("low", "mid", "high")
FACILITY_STATUS = ("Listed, not verified", "Verified by CHO")  # only use "Verified by CHO" if it really happened
CANDIDATE_KINDS = ("barangay_hall", "school", "community_centre", "court")
FACILITY_SETS = ("default", "with_hospitals", "all_listed")  # default = public health centres only
BASELINE_METHODS = ("abot", "most_populous_barangay", "nearest_city_hall", "random")
SERVICE_LABELS = (
    "Mobile clinic stop",
    "Assisted teleconsult point (BHW-staffed)",
    "Service type needs network data",
    "Not assessable — needs CHO capacity data",
)
# Generous box around Mandaluyong (lon/lat, EPSG:4326) to catch swapped x/y or projected coordinates.
BBOX = (120.95, 14.52, 121.10, 14.63)
REACH_FAC_MAX_M = 2250.0
REACH_CAND_MAX_M = 1125.0
_TOL = 1e-6


@dataclass(frozen=True)
class Col:
    name: str
    kind: str  # "str", "float", "int", "bool"
    nullable: bool = False


@dataclass(frozen=True)
class Spec:
    key: str
    filename: str
    folder: str  # "processed" or "results"
    owner: str
    columns: tuple[Col, ...]
    unique: tuple[str, ...] = ()
    geometry: str | None = None  # "Point" or "Polygon"
    checks: tuple[Callable[[pd.DataFrame], list[str]], ...] = field(default=())

    @property
    def path(self) -> Path:
        from abot import paths

        return paths.folder(self.folder) / self.filename


# ---- Row-level checks -------------------------------------------------------
def _in_set(col: str, allowed: tuple) -> Callable[[pd.DataFrame], list[str]]:
    def check(df: pd.DataFrame) -> list[str]:
        bad = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        return [f"{col}: values not allowed {bad[:5]} (allowed: {list(allowed)})"] if bad else []

    return check


def _between(col: str, lo: float | None, hi: float | None) -> Callable[[pd.DataFrame], list[str]]:
    def check(df: pd.DataFrame) -> list[str]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        errs = []
        if lo is not None and (s < lo - _TOL).any():
            errs.append(f"{col}: {(s < lo - _TOL).sum()} values below {lo}")
        if hi is not None and (s > hi + _TOL).any():
            errs.append(f"{col}: {(s > hi + _TOL).sum()} values above {hi}")
        return errs

    return check


def _bbox_xy(df: pd.DataFrame) -> list[str]:
    x0, y0, x1, y1 = BBOX
    out = ~df["x"].between(x0, x1) | ~df["y"].between(y0, y1)
    return [f"x/y: {int(out.sum())} cells outside the Mandaluyong box {BBOX} (EPSG:4326 lon/lat?)"] if out.any() else []


def _bbox_geom(df: pd.DataFrame) -> list[str]:
    if "geometry" not in df:
        return []
    x0, y0, x1, y1 = BBOX
    c = df.geometry.representative_point()
    out = ~c.x.between(x0, x1) | ~c.y.between(y0, y1)
    return [f"geometry: {int(out.sum())} features outside the Mandaluyong box {BBOX} (EPSG:4326?)"] if out.any() else []


def _adults_le_pop(df: pd.DataFrame) -> list[str]:
    bad = df["adults_30_79"] > df["pop_total"] + _TOL
    return [f"adults_30_79 > pop_total in {int(bad.sum())} cells"] if bad.any() else []


def _unmet_le_need(df: pd.DataFrame) -> list[str]:
    bad = df["unmet"] > df["need"] + _TOL
    return [f"unmet > need in {int(bad.sum())} rows"] if bad.any() else []


def _rank_le_k(df: pd.DataFrame) -> list[str]:
    bad = df["rank"] > df["k"]
    return [f"rank > k in {int(bad.sum())} rows"] if bad.any() else []


def _p5_le_p95(df: pd.DataFrame) -> list[str]:
    r = df[df["method"] == "random"]
    errs = []
    if r[["p5", "p95"]].isna().any().any():
        errs.append("random baseline rows need p5 and p95")
    if (r["p5"] > r["p95"] + _TOL).any():
        errs.append("p5 > p95 in random baseline rows")
    return errs


# ---- The contracts ----------------------------------------------------------
SPECS: dict[str, Spec] = {
    s.key: s
    for s in [
        Spec(
            "grid", "grid.parquet", "processed", "A",
            (
                Col("cell_id", "str"), Col("x", "float"), Col("y", "float"),
                Col("pop_total", "float"), Col("adults_30_79", "float"),
                Col("psgc", "str"), Col("barangay", "str"),
                Col("wealth_quintile", "int", nullable=True), Col("mobile_dl_mbps", "float", nullable=True),
                Col("network_tercile", "int", nullable=True), Col("source_flags", "str", nullable=True),
            ),
            unique=("cell_id",),
            checks=(_between("pop_total", 0, None), _between("adults_30_79", 0, None), _adults_le_pop,
                    _between("wealth_quintile", 1, 5), _between("network_tercile", 1, 3),
                    _between("mobile_dl_mbps", 0, None), _bbox_xy),
        ),
        Spec(
            "grid_cells", "grid_cells.geojson", "processed", "A",
            (Col("cell_id", "str"),), unique=("cell_id",), geometry="Polygon", checks=(_bbox_geom,),
        ),
        Spec(
            "facilities", "facilities.geojson", "processed", "A",
            (Col("fac_id", "str"), Col("name", "str"), Col("type", "str"), Col("status", "str"),
             Col("counted", "bool"), Col("source", "str")),
            unique=("fac_id",), geometry="Point",
            checks=(_in_set("status", FACILITY_STATUS), _bbox_geom),
        ),
        Spec(
            "candidates", "candidates.geojson", "processed", "A",
            (Col("cand_id", "str"), Col("name", "str"), Col("kind", "str"), Col("psgc", "str")),
            unique=("cand_id",), geometry="Point",
            checks=(_in_set("kind", CANDIDATE_KINDS), _bbox_geom),
        ),
        Spec(
            "reach_facilities", "reach_facilities.parquet", "processed", "B",
            (Col("fac_id", "str"), Col("cell_id", "str"), Col("walk_m", "float")),
            unique=("fac_id", "cell_id"), checks=(_between("walk_m", 0, REACH_FAC_MAX_M),),
        ),
        Spec(
            "reach_candidates", "reach_candidates.parquet", "processed", "B",
            (Col("cand_id", "str"), Col("cell_id", "str"), Col("walk_m", "float")),
            unique=("cand_id", "cell_id"), checks=(_between("walk_m", 0, REACH_CAND_MAX_M),),
        ),
        Spec(
            "need", "need.parquet", "results", "B",
            (Col("cell_id", "str"), Col("condition", "str"), Col("scenario", "str"),
             Col("need", "float"), Col("unmet", "float")),
            unique=("cell_id", "condition", "scenario"),
            checks=(_in_set("condition", CONDITIONS), _in_set("scenario", SCENARIOS),
                    _between("need", 0, None), _between("unmet", 0, None), _unmet_le_need),
        ),
        Spec(
            "placements", "placements.csv", "results", "B",
            (Col("condition", "str"), Col("scenario", "str"), Col("walk_speed", "float"),
             Col("lambda", "float"), Col("facility_set", "str"), Col("k", "int"), Col("rank", "int"),
             Col("cand_id", "str"), Col("added_people", "float"), Col("added_poorest2", "float", nullable=True),
             Col("service_type", "str")),
            unique=("condition", "scenario", "walk_speed", "lambda", "facility_set", "k", "rank"),
            checks=(_in_set("condition", CONDITIONS), _in_set("scenario", SCENARIOS),
                    _in_set("facility_set", FACILITY_SETS), _in_set("service_type", SERVICE_LABELS),
                    _between("k", 1, 10), _between("rank", 1, 10), _rank_le_k,
                    _between("added_people", 0, None), _between("added_poorest2", 0, None)),
        ),
        Spec(
            "baselines", "baselines.csv", "results", "B",
            (Col("condition", "str"), Col("scenario", "str"), Col("method", "str"), Col("k", "int"),
             Col("people", "float"), Col("p5", "float", nullable=True), Col("p95", "float", nullable=True)),
            unique=("condition", "scenario", "method", "k"),
            checks=(_in_set("condition", CONDITIONS), _in_set("scenario", SCENARIOS),
                    _in_set("method", BASELINE_METHODS), _between("k", 1, 10),
                    _between("people", 0, None), _p5_le_p95),
        ),
        Spec(
            "sensitivity", "sensitivity.csv", "results", "B",
            (Col("case", "str"), Col("top5_ids", "str"), Col("overlap_with_default", "int")),
            unique=("case",), checks=(_between("overlap_with_default", 0, 5),),
        ),
    ]
}


# ---- Validation -------------------------------------------------------------
def _kind_ok(series: pd.Series, kind: str) -> bool:
    s = series.dropna()
    if s.empty:
        return True
    if kind == "str":
        return all(isinstance(v, str) for v in s)
    if kind == "float":
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if kind == "int":
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            return False
        return bool(np.all(np.mod(s.astype(float), 1) == 0))
    if kind == "bool":
        return pd.api.types.is_bool_dtype(s) or all(isinstance(v, (bool, np.bool_)) for v in s)
    raise ValueError(f"unknown column kind {kind}")


def validate(key: str, df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations for `df` (empty list = valid)."""
    spec = SPECS[key]
    errors: list[str] = []
    missing = [c.name for c in spec.columns if c.name not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        errors.append("file has no rows")
    for col in spec.columns:
        if not col.nullable and df[col.name].isna().any():
            errors.append(f"{col.name}: {int(df[col.name].isna().sum())} empty values (not allowed)")
        if not _kind_ok(df[col.name], col.kind):
            errors.append(f"{col.name}: expected {col.kind}, got dtype {df[col.name].dtype}")
    if spec.unique and df.duplicated(list(spec.unique)).any():
        errors.append(f"duplicate rows for key {list(spec.unique)}: {int(df.duplicated(list(spec.unique)).sum())}")
    if spec.geometry:
        if "geometry" not in df.columns:
            errors.append("no geometry column")
        else:
            gtypes = set(df.geometry.geom_type.dropna())
            allowed = {spec.geometry, f"Multi{spec.geometry}"}
            if not gtypes <= allowed:
                errors.append(f"geometry must be {spec.geometry}, found {sorted(gtypes)}")
            crs = getattr(df, "crs", None)
            if crs is not None and crs.to_epsg() != 4326:
                errors.append(f"CRS must be EPSG:4326, found {crs}")
    if not errors:  # value checks assume types are right
        for check in spec.checks:
            errors.extend(check(df))
    return errors


def read_file(key: str, path: Path | str | None = None) -> pd.DataFrame:
    """Read a contract file from disk without validating it."""
    spec = SPECS[key]
    path = Path(path) if path else spec.path
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"cand_id": str, "cell_id": str, "psgc": str})
    if path.suffix == ".geojson":
        import geopandas as gpd

        return gpd.read_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def cross_checks(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Check that ids used in one file exist in the file they refer to."""
    links = [
        ("reach_facilities", "cell_id", "grid", "cell_id"),
        ("reach_facilities", "fac_id", "facilities", "fac_id"),
        ("reach_candidates", "cell_id", "grid", "cell_id"),
        ("reach_candidates", "cand_id", "candidates", "cand_id"),
        ("grid_cells", "cell_id", "grid", "cell_id"),
        ("need", "cell_id", "grid", "cell_id"),
        ("placements", "cand_id", "candidates", "cand_id"),
    ]
    errors = []
    for src, col, dst, dcol in links:
        if src in frames and dst in frames:
            extra = set(frames[src][col].astype(str)) - set(frames[dst][dcol].astype(str))
            if extra:
                errors.append(f"{src}.{col}: {len(extra)} ids not found in {dst}.{dcol} (e.g. {sorted(extra)[:3]})")
    return errors


def main(argv: list[str] | None = None) -> int:
    from abot import paths

    argv = sys.argv[1:] if argv is None else argv
    keys = argv or list(SPECS)
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        print(f"Unknown contract(s): {unknown}. Known: {list(SPECS)}")
        return 2
    if paths.USING_FIXTURES:
        print("NOTE: ABOT_DATA=fixtures -> checking SYNTHETIC fixture files\n")
    failed = False
    frames: dict[str, pd.DataFrame] = {}
    for key in keys:
        spec = SPECS[key]
        rel = spec.path.relative_to(paths.ROOT) if spec.path.is_relative_to(paths.ROOT) else spec.path
        if not spec.path.exists():
            print(f"[ .... ] {key:<17} {rel}  (not delivered yet, owner {spec.owner})")
            continue
        df = read_file(key)
        errs = validate(key, df)
        frames[key] = df
        if errs:
            failed = True
            print(f"[ FAIL ] {key:<17} {rel}  (owner {spec.owner})")
            for e in errs:
                print(f"           - {e}")
        else:
            print(f"[  OK  ] {key:<17} {rel}  {len(df):,} rows")
    xerrs = cross_checks(frames)
    if xerrs:
        failed = True
        print("\n[ FAIL ] cross-file links")
        for e in xerrs:
            print(f"           - {e}")
    elif len(frames) > 1:
        print("\n[  OK  ] cross-file links")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
