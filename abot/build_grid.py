"""Raw data -> grid, facility and candidate-site layers.

OWNER: A. Outputs (contracts in CONTEXT.md section 4):
  data/processed/grid.parquet, grid_cells.geojson, facilities.geojson, candidates.geojson
Record every source in docs/data_sources.csv and every check in docs/qa_log.md.
Check before handoff: python -m abot.contracts

Inputs (gitignored, under data/raw/ or ABOT_RAW_DIR):
  worldpop/phl_ppp_2020.tif
  worldpop/phl_{m|f}_{30,35,...,75}_2020.tif
  cchain/location.csv, brgy_geography.csv, tm_relative_wealth_index.csv,
          ookla_internet_speed.csv, geoportal_doh_poi_health.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from abot import paths
from abot.contracts import FACILITY_STATUS, SPECS, cross_checks, read_file, validate

CITY_PCODE = "PH137401000"
CELL_M = 100
DEDUP_M = 50.0
# A full national WorldPop GeoTIFF is about 150 MB. Anything smaller is still downloading.
MIN_RASTER_BYTES = 100_000_000
METRIC = "EPSG:32651"
AGE_STARTS = (30, 35, 40, 45, 50, 55, 60, 65, 70, 75)
# PSA NCR, 2020 Census of Population and Housing, City of Mandaluyong, 1 May 2020.
# https://rssoncr.psa.gov.ph/content/2020-census-population-and-housing-results-city-mandaluyong
PSA_2020_TOTAL = 425_758
CITY_HALL_OSM_ID = "112549085"
KIND_PRIORITY = {"barangay_hall": 0, "school": 1, "community_centre": 2, "court": 3}
COURT_SPORTS = {"basketball", "volleyball", "multi"}
SKIP_SPORTS = {"soccer", "football", "tennis", "golf", "swimming", "baseball"}


def equal_count_bins(frame: pd.DataFrame, value_col: str, id_col: str, k: int) -> pd.Series:
    """Bin rows into k groups of nearly equal size. The lowest values get bin 1.

    Ties break by id_col ascending. The result is indexed by id_col.
    """
    cols = [id_col, value_col]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    data = frame[cols].dropna()
    if data[id_col].duplicated().any():
        raise ValueError(f"duplicate {id_col}")
    if len(data) < k:
        raise ValueError(f"need at least {k} rows to make {k} bins, got {len(data)}")
    data = data.sort_values([value_col, id_col], kind="mergesort")
    bins = np.asarray(pd.qcut(np.arange(len(data)), k, labels=False), dtype=int) + 1
    return pd.Series(bins, index=data[id_col].to_numpy())


def keep_spaced(ids: list[str], xs: list[float], ys: list[float], priority: list[int], metres: float = DEDUP_M) -> list[str]:
    """Keep points that are more than `metres` from an already kept point.

    Lower priority number is considered first. Equal priority breaks by id ascending.
    xs and ys are metres (EPSG:32651).
    """
    if not (len(ids) == len(xs) == len(ys) == len(priority)):
        raise ValueError("ids, xs, ys, and priority must be the same length")
    kept_xy: list[tuple[float, float]] = []
    kept_ids: list[str] = []
    limit = metres * metres
    for i in sorted(range(len(ids)), key=lambda n: (priority[n], ids[n])):
        x, y = float(xs[i]), float(ys[i])
        if any((x - kx) ** 2 + (y - ky) ** 2 <= limit for kx, ky in kept_xy):
            continue
        kept_xy.append((x, y))
        kept_ids.append(ids[i])
    return kept_ids


def _refuse_fixtures() -> None:
    if paths.USING_FIXTURES:
        raise RuntimeError("Refusing to write real layers while ABOT_DATA=fixtures")


def _psgc(code: str) -> str:
    text = str(code).strip()
    if text.startswith("PH"):
        text = text[2:]
    if not text.isdigit():
        raise ValueError(f"expected a PSGC code, got {code!r}")
    return text


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return None
    return text


def _require(path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing {path}. Put the raw download there, or run python scripts/download_grid_inputs.py"
        )


def _cchain(name: str) -> pd.DataFrame:
    path = paths.RAW / "cchain" / name
    _require(path)
    return pd.read_csv(path, dtype=str)


def _barangays() -> gpd.GeoDataFrame:
    loc = _cchain("location.csv")
    loc = loc[loc["adm3_pcode"] == CITY_PCODE].drop_duplicates("adm4_pcode")
    if len(loc) != 27:
        raise RuntimeError(f"expected 27 Mandaluyong barangays in location.csv, found {len(loc)}")
    geo = _cchain("brgy_geography.csv")
    geo = geo[geo["adm4_pcode"].isin(set(loc["adm4_pcode"]))]
    geo = geo.sort_values("date").drop_duplicates("adm4_pcode", keep="last")
    if len(geo) != 27:
        missing = sorted(set(loc["adm4_pcode"]) - set(geo["adm4_pcode"]))
        raise RuntimeError(f"barangay shapes missing for {missing}")
    frame = gpd.GeoDataFrame(geo, geometry=gpd.GeoSeries.from_wkt(geo["geometry"]), crs=4326)
    frame["geometry"] = frame.geometry.make_valid()
    names = loc.set_index("adm4_pcode")["adm4_en"]
    frame["barangay"] = frame["adm4_pcode"].map(names)
    frame["psgc"] = frame["adm4_pcode"].map(_psgc)
    if frame["barangay"].isna().any() or frame.geometry.isna().any():
        raise RuntimeError("a Mandaluyong barangay is missing a name or a shape")
    return frame[["adm4_pcode", "psgc", "barangay", "geometry"]].reset_index(drop=True)


def _city_shape(barangays: gpd.GeoDataFrame):
    shape = barangays.geometry.union_all()
    if not shape.is_valid:
        shape = shape.buffer(0)
    return shape


def assert_full_raster(path) -> None:
    """Refuse a WorldPop file that is still downloading.

    A partial file still starts with a TIFF header, so opening it is not proof that it is complete.
    """
    _require(path)
    size = path.stat().st_size
    if size < MIN_RASTER_BYTES:
        raise RuntimeError(
            f"{path.name} is {size} bytes. A full WorldPop raster is at least "
            f"{MIN_RASTER_BYTES} bytes. The download is still in progress."
        )


def _read_window(path, bounds):
    assert_full_raster(path)
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=window).astype("float64")
        transform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        data[np.isclose(data, float(nodata))] = np.nan
    data[data < 0] = np.nan
    return data, transform


def _same_grid(left, right) -> bool:
    left_data, left_transform = left
    right_data, right_transform = right
    if left_data.shape != right_data.shape:
        return False
    return bool(np.allclose(
        [left_transform.a, left_transform.b, left_transform.c, left_transform.d, left_transform.e, left_transform.f],
        [right_transform.a, right_transform.b, right_transform.c, right_transform.d, right_transform.e, right_transform.f],
        atol=1e-9,
    ))


def _population(bounds):
    folder = paths.RAW / "worldpop"
    total_path = folder / "phl_ppp_2020.tif"
    total, total_transform = _read_window(total_path, bounds)
    adults = np.zeros(total.shape, dtype="float64")
    seen = np.zeros(total.shape, dtype=bool)
    for sex in ("m", "f"):
        for age in AGE_STARTS:
            path = folder / f"phl_{sex}_{age}_2020.tif"
            band, transform = _read_window(path, bounds)
            if not _same_grid((total, total_transform), (band, transform)):
                raise RuntimeError(f"{path.name} is not on the same grid as {total_path.name}")
            present = ~np.isnan(band)
            adults[present] += band[present]
            seen |= present
    adults[~seen] = np.nan
    return total, adults, total_transform


def _attribute_tables(barangays: gpd.GeoDataFrame):
    codes = set(barangays["adm4_pcode"])
    wealth = _cchain("tm_relative_wealth_index.csv")
    wealth_rows, wealth_year = _one_year(wealth, codes, "rwi_mean")
    speed = _cchain("ookla_internet_speed.csv")
    speed_rows, speed_year = _one_year(speed, codes, "mobile_mean_avg_d_kbps_mean")
    wealth_rows = wealth_rows.copy()
    speed_rows = speed_rows.copy()
    wealth_rows["rwi_mean"] = pd.to_numeric(wealth_rows["rwi_mean"], errors="coerce")
    speed_rows["kbps"] = pd.to_numeric(speed_rows["mobile_mean_avg_d_kbps_mean"], errors="coerce")
    if speed_rows["kbps"].dropna().median() < 500:
        raise RuntimeError("Ookla mobile download does not look like kbps; refusing to convert to Mbps")
    wealth_rows["psgc"] = wealth_rows["adm4_pcode"].map(_psgc)
    speed_rows["psgc"] = speed_rows["adm4_pcode"].map(_psgc)
    quintile = equal_count_bins(wealth_rows, "rwi_mean", "psgc", 5)
    tercile = equal_count_bins(speed_rows, "kbps", "psgc", 3)
    mbps = speed_rows.drop_duplicates("psgc").set_index("psgc")["kbps"] / 1000.0
    print(f"Wealth year used: {wealth_year}. Ookla year used: {speed_year}.")
    print(f"Barangays with wealth: {quintile.notna().sum()}/27. With mobile speed: {tercile.notna().sum()}/27.")
    return quintile, tercile, mbps, wealth_year, speed_year


def _one_year(frame: pd.DataFrame, codes: set[str], value_col: str):
    """Pick the latest year that covers the most Mandaluyong barangays."""
    sub = frame[frame["adm4_pcode"].isin(codes)].copy()
    if sub.empty:
        raise RuntimeError(f"no CCHAIN rows for Mandaluyong in column {value_col}")
    sub["date"] = sub["date"].astype(str)
    counts = sub.groupby("date")["adm4_pcode"].nunique()
    best = int(counts.max())
    year = sorted(counts[counts == best].index)[-1]
    chosen = sub[sub["date"] == year]
    missing = codes - set(chosen["adm4_pcode"])
    if missing:
        extras = []
        for code in sorted(missing):
            rows = sub[sub["adm4_pcode"] == code].sort_values("date")
            if not rows.empty:
                extras.append(rows.iloc[-1])
        if extras:
            chosen = pd.concat([chosen, pd.DataFrame(extras)], ignore_index=True)
            print(f"{value_col}: {len(extras)} barangays were missing {year}; used each barangay's latest earlier year")
    if value_col not in chosen.columns:
        raise KeyError(value_col)
    return chosen, year[:4]


def _cells(total: np.ndarray, adults: np.ndarray, bounds_transform, city, barangays: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    transform = bounds_transform
    height, width = total.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    lon = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    lat = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    pop = np.nan_to_num(total, nan=0.0).ravel()
    adult = np.nan_to_num(adults, nan=0.0).ravel()
    points = gpd.GeoSeries(gpd.points_from_xy(lon.ravel(), lat.ravel()), crs=4326)
    keep = points.within(city).to_numpy() & (pop > 0)
    if int(keep.sum()) == 0:
        raise RuntimeError("no WorldPop pixels with people fall inside Mandaluyong")
    easting = points[keep].to_crs(METRIC)
    ix = np.floor(np.round(easting.x.to_numpy(), 3) / CELL_M).astype(np.int64)
    iy = np.floor(np.round(easting.y.to_numpy(), 3) / CELL_M).astype(np.int64)
    pixels = pd.DataFrame({"ix": ix, "iy": iy, "pop_total": pop[keep], "adults_30_79": adult[keep]})
    grouped = pixels.groupby(["ix", "iy"], as_index=False)[["pop_total", "adults_30_79"]].sum()
    if grouped["adults_30_79"].sum() > grouped["pop_total"].sum() * 1.02:
        raise RuntimeError("age-sex total exceeds the population raster by more than 2% inside the city")
    raised = grouped["adults_30_79"] > grouped["pop_total"] + 1e-6
    if raised.any():
        print(f"pop_total raised to adults_30_79 in {int(raised.sum())} cells so the contract holds")
        grouped.loc[raised, "pop_total"] = grouped.loc[raised, "adults_30_79"]
    squares = [box(r.ix * CELL_M, r.iy * CELL_M, (r.ix + 1) * CELL_M, (r.iy + 1) * CELL_M) for r in grouped.itertuples()]
    cells = gpd.GeoDataFrame(grouped, geometry=squares, crs=METRIC)
    if not np.allclose(cells.geometry.area, CELL_M * CELL_M, atol=0.01):
        raise RuntimeError("a grid cell is not a 100 m square")
    centres = gpd.GeoSeries(cells.geometry.centroid, crs=METRIC).to_crs(4326)
    cells["x"] = centres.x.to_numpy()
    cells["y"] = centres.y.to_numpy()
    cells["cell_id"] = cells["ix"].astype(str) + "_" + cells["iy"].astype(str)
    cells = cells.to_crs(4326)
    hit = gpd.sjoin(
        gpd.GeoDataFrame({"cell_id": cells["cell_id"]}, geometry=centres, crs=4326),
        barangays[["psgc", "barangay", "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates("cell_id", keep="first")
    cells = cells.merge(hit[["cell_id", "psgc", "barangay"]], on="cell_id", how="left")
    missing = cells["psgc"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(
            cells.loc[missing, ["cell_id", "geometry"]].to_crs(METRIC),
            barangays[["psgc", "barangay", "geometry"]].to_crs(METRIC),
            how="left",
            distance_col="dist_m",
        ).drop_duplicates("cell_id", keep="first")
        far = near["dist_m"] > 200
        if far.any():
            raise RuntimeError(f"{int(far.sum())} cells are more than 200 m from a barangay polygon")
        filled = near.set_index("cell_id")
        cells.loc[missing, "psgc"] = cells.loc[missing, "cell_id"].map(filled["psgc"]).to_numpy()
        cells.loc[missing, "barangay"] = cells.loc[missing, "cell_id"].map(filled["barangay"]).to_numpy()
        print(f"Assigned {int(missing.sum())} cell centroids to the nearest barangay")
    if cells["psgc"].isna().any():
        raise RuntimeError("some cells still have no PSGC")
    return cells


def build_grid() -> None:
    """WorldPop 2020 100 m clipped to Mandaluyong; adults 30-79; PSGC join; wealth + network per barangay."""
    _refuse_fixtures()
    barangays = _barangays()
    city = _city_shape(barangays)
    minx, miny, maxx, maxy = barangays.total_bounds
    pad = 0.005
    total, adults, transform = _population((minx - pad, miny - pad, maxx + pad, maxy + pad))
    cells = _cells(total, adults, transform, city, barangays)
    quintile, tercile, mbps, wealth_year, speed_year = _attribute_tables(barangays)
    cells["wealth_quintile"] = cells["psgc"].map(quintile).astype("Int64")
    cells["network_tercile"] = cells["psgc"].map(tercile).astype("Int64")
    cells["mobile_dl_mbps"] = cells["psgc"].map(mbps).astype("Float64")
    wealth_note = f"wealth:CCHAIN_RWI_mean_{wealth_year}_quintile_within_city"
    speed_note = f"network:CCHAIN_Ookla_mobile_dl_{speed_year}_tercile"
    cells["source_flags"] = "pop_total:WorldPop_ppp_2020; adults_30_79:WorldPop_age_sex_2020; " + wealth_note + "; " + speed_note
    missing_wealth = cells["wealth_quintile"].isna()
    missing_speed = cells["network_tercile"].isna()
    if missing_wealth.any() or missing_speed.any():
        cells.loc[missing_wealth, "source_flags"] = cells.loc[missing_wealth, "source_flags"].str.replace(
            wealth_note, "wealth:missing", regex=False
        )
        cells.loc[missing_speed, "source_flags"] = cells.loc[missing_speed, "source_flags"].str.replace(
            speed_note, "network:missing", regex=False
        )
    grid = cells.assign(
        cell_id=cells["cell_id"].astype(str),
        psgc=cells["psgc"].astype(str),
        barangay=cells["barangay"].astype(str),
        pop_total=cells["pop_total"].astype(float),
        adults_30_79=cells["adults_30_79"].astype(float),
        x=cells["x"].astype(float),
        y=cells["y"].astype(float),
    )
    _publish("grid", grid)
    squares = gpd.GeoDataFrame({"cell_id": grid["cell_id"].astype(str)}, geometry=cells.geometry, crs=4326)
    _publish("grid_cells", squares)
    total_pop = float(grid["pop_total"].sum())
    low, high = PSA_2020_TOTAL * 0.85, PSA_2020_TOTAL * 1.15
    inside = low <= total_pop <= high
    print(f"Grid population total: {total_pop:,.0f}")
    print(f"Adults 30-79 total: {float(grid['adults_30_79'].sum()):,.0f}")
    print(f"PSA 2020 census total: {PSA_2020_TOTAL:,} (accept {low:,.0f} to {high:,.0f})")
    print(f"Census check: {'PASS' if inside else 'FAIL'}")
    print(f"Cells: {len(grid):,}. Barangays present: {grid['psgc'].nunique()}.")
    _print_spot_check(grid, barangays)


def _print_spot_check(grid: pd.DataFrame, barangays: gpd.GeoDataFrame) -> None:
    """Print three barangay centroids so a person can check them on a map."""
    names = ["Addition Hills", "Hulo", "Wack-wack Greenhills"]
    present = [name for name in names if name in set(barangays["barangay"])]
    if len(present) < 3:
        present = sorted(barangays["barangay"])[:3]
    for name in present:
        row = barangays.loc[barangays["barangay"] == name].iloc[0]
        point = row.geometry.representative_point()
        cells = grid.loc[grid["barangay"] == name]
        print(
            f"Spot check {name} PSGC {row.psgc}: "
            f"shape point {point.x:.5f}, {point.y:.5f}; cells {len(cells)}"
        )


def _osm_points(polygon, tags: dict) -> gpd.GeoDataFrame:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = paths.ROOT / "cache"
    ox.settings.timeout = 180
    found = ox.features_from_polygon(polygon, tags)
    if found.empty:
        return found
    found = found.reset_index()
    # osmnx 2 indexes features by element type and OSM id.
    if "element" not in found.columns or "id" not in found.columns:
        raise RuntimeError(f"OpenStreetMap result is missing element/id columns: {list(found.columns)[:12]}")
    found = found.rename(columns={"element": "element_type", "id": "osmid"})
    found = found[found.geometry.notna() & ~found.geometry.is_empty].copy()
    found["geometry"] = found.geometry.make_valid().representative_point()
    found = found[found.geometry.within(polygon)].copy()
    return found


def _ensure_columns(frame: gpd.GeoDataFrame, columns: tuple[str, ...]) -> gpd.GeoDataFrame:
    for name in columns:
        if name not in frame.columns:
            frame[name] = None
    return frame


# Which listed facilities count as EXISTING CHRONIC CARE (team decision, Fri 23 Oct 8:20 PM).
# ABOT is about public primary care for hypertension and diabetes, so only public health
# centres and barangay health stations count. Private clinics, doctors' offices, dialysis,
# dermatology, hyperbaric and school clinics do not; nor do disease-specific public programmes
# (TB DOTS, social hygiene). Hospitals never count by default. The old "every clinic and doctor"
# list stays as a sensitivity case (facility_set = "all_listed" in results/).
PUBLIC_PRIMARY_CARE_NAME = re.compile(r"\bhealth\s+(center|centre|station)\b", re.IGNORECASE)


def public_primary_care(name: str, kind: str) -> bool:
    """True for a public health centre or barangay health station (counted as existing care)."""
    if str(kind).lower() == "hospital":
        return False
    if str(kind).lower() == "health_post":
        return True
    return bool(PUBLIC_PRIMARY_CARE_NAME.search(str(name or "")))


def recount_facilities() -> None:
    """Re-apply public_primary_care() to the processed facilities file (no raw data needed)."""
    path = SPECS["facilities"].path
    fac = read_file("facilities")
    before = int(fac["counted"].sum())
    fac["counted"] = [public_primary_care(n, t) for n, t in zip(fac["name"], fac["type"])]
    _publish("facilities", fac)
    print(f"facilities: counted {before} -> {int(fac['counted'].sum())} of {len(fac)} ({path.name})")
    print(fac.loc[fac["counted"], ["type", "name"]].sort_values(["type", "name"]).to_string(index=False))


def build_facilities() -> None:
    """Facility points from OpenStreetMap. CCHAIN only has barangay counts, not points."""
    _refuse_fixtures()
    barangays = _barangays()
    city = _city_shape(barangays)
    found = _osm_points(city, {
        "amenity": ["clinic", "hospital", "doctors", "health_post"],
        "healthcare": ["clinic", "hospital", "doctor", "doctors", "centre", "center"],
    })
    if found.empty:
        raise RuntimeError("OpenStreetMap returned no health facilities inside Mandaluyong")
    found = _ensure_columns(found, ("name", "amenity", "healthcare"))
    found = found.drop_duplicates(["element_type", "osmid"])
    found["type"] = [
        "hospital" if (
            _text(amenity) == "hospital" or _text(care) == "hospital" or "hospital" in (_text(name) or "").lower()
        ) else (_text(amenity) or _text(care) or "clinic")
        for amenity, care, name in zip(found["amenity"], found["healthcare"], found["name"])
    ]
    found["name"] = [
        _text(name) or f"Unnamed {kind}"
        for name, kind in zip(found["name"], found["type"])
    ]
    found["fac_id"] = found["element_type"].astype(str) + "/" + found["osmid"].astype(str)
    metric = found.to_crs(METRIC)
    priority = [0 if not str(name).startswith("Unnamed") else 1 for name in found["name"]]
    keep = set(keep_spaced(
        found["fac_id"].tolist(),
        metric.geometry.x.tolist(),
        metric.geometry.y.tolist(),
        priority,
    ))
    dropped = len(found) - len(keep)
    found = found[found["fac_id"].isin(keep)].copy()
    found["status"] = FACILITY_STATUS[0]
    found["counted"] = [public_primary_care(n, t) for n, t in zip(found["name"], found["type"])]
    found["source"] = "OpenStreetMap"
    print(f"Facilities: {len(found)} after dropping {dropped} within {DEDUP_M:.0f} m")
    print(found["type"].value_counts().to_string())
    _print_doh_cross_check(len(found), set(barangays["adm4_pcode"]))
    _publish("facilities", gpd.GeoDataFrame({
        "fac_id": found["fac_id"].astype(str),
        "name": found["name"].astype(str),
        "type": found["type"].astype(str),
        "status": found["status"].astype(str),
        "counted": found["counted"].astype(bool),
        "source": found["source"].astype(str),
        "geometry": found.geometry,
    }, crs=4326))


def _print_doh_cross_check(osm_count: int, codes: set[str]) -> None:
    doh = _cchain("geoportal_doh_poi_health.csv")
    doh = doh[doh["adm4_pcode"].isin(codes)].copy()
    if doh.empty:
        print("DOH cross-check: no CCHAIN rows for Mandaluyong")
        return
    doh["date"] = doh["date"].astype(str)
    doh["doh_pois_count"] = pd.to_numeric(doh["doh_pois_count"], errors="coerce").fillna(0)
    latest = doh.sort_values("date").groupby("adm4_pcode", as_index=False).tail(1)
    total = float(latest["doh_pois_count"].sum())
    year = sorted(latest["date"])[-1][:4]
    print(f"DOH geoportal count from CCHAIN ({year}, sum of barangay totals): {total:.0f}")
    print(f"OpenStreetMap facility points: {osm_count}. Difference: {osm_count - total:.0f}")


# Team decision, Fri 23 Oct 9:35 PM. These names are private schools.
# Mandaluyong City College (city-run) and Rizal Technological University (state university) stay.
PRIVATE_SCHOOL_NAMES = frozenset({
    "st. augustine school of nursing",
    "la salle green hills",
    "st. therese private school",
    "don bosco technical college",
    "international baptist college",
    "international baptists college",
})


def is_private_school(name: str) -> bool:
    """True when the site name is one of the private schools the team dropped."""
    return str(name or "").strip().casefold() in PRIVATE_SCHOOL_NAMES


def recount_candidates() -> None:
    """Drop the named private schools from the processed candidate file."""
    cand = read_file("candidates")
    drop = cand["name"].map(is_private_school)
    print(f"Candidates: dropping {int(drop.sum())} private schools")
    print(cand.loc[drop, "name"].to_string(index=False))
    _publish("candidates", cand.loc[~drop].copy())


def _candidate_kind(row) -> str | None:
    if str(row["osmid"]) == CITY_HALL_OSM_ID:
        return None
    name = (_text(row["name"]) or "").lower()
    if "city hall" in name:
        return None
    amenity = _text(row["amenity"])
    office = _text(row["office"])
    leisure = _text(row["leisure"])
    sport = (_text(row["sport"]) or "").lower()
    if amenity == "townhall" or (office == "government" and ("barangay" in name or "brgy" in name)):
        return "barangay_hall"
    if amenity in {"school", "college", "university"}:
        return "school"
    if amenity == "community_centre":
        return "community_centre"
    if leisure == "pitch":
        if sport in SKIP_SPORTS:
            return None
        if sport in COURT_SPORTS or sport == "" or "court" in name or "basketball" in name:
            return "court"
    return None


def build_candidates() -> None:
    """OSM barangay halls, schools, community centres, and courts inside the city."""
    _refuse_fixtures()
    barangays = _barangays()
    city = _city_shape(barangays)
    found = _osm_points(city, {
        "amenity": ["townhall", "school", "college", "university", "community_centre"],
        "office": "government",
        "leisure": "pitch",
    })
    if found.empty:
        raise RuntimeError("OpenStreetMap returned no candidate sites inside Mandaluyong")
    found = _ensure_columns(found, ("name", "amenity", "office", "leisure", "sport"))
    found = found.drop_duplicates(["element_type", "osmid"])
    found["kind"] = found.apply(_candidate_kind, axis=1)
    found = found[found["kind"].notna()].copy()
    if found.empty:
        raise RuntimeError("no candidate sites left after the kind rules")
    found["cand_id"] = found["element_type"].astype(str) + "/" + found["osmid"].astype(str)
    metric = found.to_crs(METRIC)
    before = len(found)
    keep = set(keep_spaced(
        found["cand_id"].tolist(),
        metric.geometry.x.tolist(),
        metric.geometry.y.tolist(),
        [KIND_PRIORITY[kind] for kind in found["kind"]],
    ))
    found = found[found["cand_id"].isin(keep)].copy()
    print(f"Candidates: {len(found)} after dropping {before - len(found)} within {DEDUP_M:.0f} m")
    hit = gpd.sjoin(
        found[["cand_id", "geometry"]],
        barangays[["psgc", "barangay", "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates("cand_id", keep="first")
    found = found.merge(hit[["cand_id", "psgc", "barangay"]], on="cand_id", how="left")
    missing = found["psgc"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(
            found.loc[missing, ["cand_id", "geometry"]].to_crs(METRIC),
            barangays[["psgc", "barangay", "geometry"]].to_crs(METRIC),
            how="left",
            distance_col="dist_m",
        ).drop_duplicates("cand_id", keep="first")
        far = near["dist_m"] > 200
        if far.any():
            raise RuntimeError(f"{int(far.sum())} candidates are more than 200 m from a barangay")
        filled = near.set_index("cand_id")
        found.loc[missing, "psgc"] = found.loc[missing, "cand_id"].map(filled["psgc"]).to_numpy()
        found.loc[missing, "barangay"] = found.loc[missing, "cand_id"].map(filled["barangay"]).to_numpy()
    found["name"] = [
        _text(name) or f"{kind.replace('_', ' ').title()} ({barangay})"
        for name, kind, barangay in zip(found["name"], found["kind"], found["barangay"])
    ]
    private = found["name"].map(is_private_school)
    if private.any():
        print(f"Candidates: dropping {int(private.sum())} private schools")
        print(found.loc[private, "name"].to_string(index=False))
        found = found.loc[~private].copy()
    print(found["kind"].value_counts().to_string())
    _publish("candidates", gpd.GeoDataFrame({
        "cand_id": found["cand_id"].astype(str),
        "name": found["name"].astype(str),
        "kind": found["kind"].astype(str),
        "psgc": found["psgc"].astype(str),
        "geometry": found.geometry,
    }, crs=4326))


def _publish(key: str, frame: pd.DataFrame) -> None:
    spec = SPECS[key]
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    columns = [col.name for col in spec.columns]
    if spec.geometry:
        out = gpd.GeoDataFrame(frame[columns], geometry=frame.geometry, crs=4326)
        out.to_file(spec.path, driver="GeoJSON")
    else:
        frame[columns].to_parquet(spec.path, index=False)
    got = read_file(key)
    errors = validate(key, got)
    if errors:
        raise RuntimeError(f"{key} breaks its contract:\n  - " + "\n  - ".join(errors))
    print(f"[  OK  ] {key}: {len(got):,} rows -> {spec.path.relative_to(paths.ROOT)}")


def main() -> None:
    import sys

    flags = sys.argv[1:]
    if "--recount" in flags or "--recount-candidates" in flags:
        if "--recount" in flags:
            recount_facilities()
        if "--recount-candidates" in flags:
            recount_candidates()
        return
    build_grid()
    build_facilities()
    build_candidates()
    frames = {key: read_file(key) for key in ("grid", "grid_cells", "facilities", "candidates") if SPECS[key].path.exists()}
    errors = cross_checks(frames)
    if errors:
        raise RuntimeError("cross-file links failed:\n  - " + "\n  - ".join(errors))
    print("[  OK  ] cross-file links")


if __name__ == "__main__":
    main()
