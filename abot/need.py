"""Modelled need, unmet need and equity weights.

OWNER: B. Contract: CONTEXT.md section 3. Prevalence must come from abot.config.prevalence().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abot import config, io, paths
from abot.contracts import BASELINE_METHODS, CONDITIONS, SCENARIOS
from abot.place import (
    baseline_most_populous_barangay,
    baseline_nearest_city_hall,
    baseline_random,
    greedy,
    people_reached,
)

SERVICE_PLACEHOLDER = "Service type needs network data"  # used only if service_type.py is unavailable


def need(grid: pd.DataFrame, prevalence_share: float) -> pd.Series:
    """adults_30_79 x prevalence, indexed by cell_id."""
    if not 0 < float(prevalence_share) < 1:
        raise ValueError(f"prevalence share must be between 0 and 1, got {prevalence_share}")
    out = grid["adults_30_79"].astype(float) * float(prevalence_share)
    out.index = grid["cell_id"].astype(str)
    return out


def unmet(need_by_cell: pd.Series, covered: set[str]) -> pd.Series:
    """need where the cell is NOT in `covered`, else 0. Indexed by cell_id."""
    return need_by_cell.where(~need_by_cell.index.isin(set(map(str, covered))), 0.0)


def equity_weight(wealth_quintile: pd.Series, lam: float) -> pd.Series:
    """1 + lam * (5 - quintile) / 4; missing quintile -> 1.0. Indexed like the input."""
    q = pd.to_numeric(wealth_quintile, errors="coerce")
    w = 1.0 + float(lam) * (5.0 - q) / 4.0
    return w.fillna(1.0).astype(float)


def _service_type(cand_id: str, grid, facilities, rf, cfg, cache: dict) -> str:
    """The service-type rule, with one honest placeholder label if it is unavailable."""
    if cand_id not in cache:
        try:
            from abot import service_type

            cache[cand_id] = service_type.assign(cand_id, grid, facilities, rf, cfg)
        except NotImplementedError:
            cache[cand_id] = SERVICE_PLACEHOLDER
    return cache[cand_id]


def _city_hall_walk(candidates: pd.DataFrame, cfg: dict) -> tuple[pd.Series, str]:
    """Walk from City Hall to each candidate; straight-line fallback if the graph is absent."""
    from abot.place import gpd_points

    hall = cfg.get("city_hall") or {}
    if hall.get("lon") is None or hall.get("lat") is None:
        raise config.ConfigError("config.yaml city_hall is not filled in (owner B)")
    graph_path = paths.RAW / "walk.graphml"
    cache = paths.PROCESSED / "city_hall_walk.csv"   # committed, so reruns without the graph match
    if graph_path.exists():
        from abot import reach

        walk = reach.city_hall_walk_m(reach.load_graph(graph_path), candidates, hall)
        if not paths.USING_FIXTURES:
            walk.rename("walk_m").rename_axis("cand_id").reset_index().to_csv(cache, index=False)
        return walk, "OSM walking network"
    if cache.exists() and not paths.USING_FIXTURES:
        c = pd.read_csv(cache, dtype={"cand_id": str})
        walk = c.set_index("cand_id")["walk_m"].reindex(candidates["cand_id"].astype(str))
        return walk, "OSM walking network (cached in data/processed/city_hall_walk.csv)"
    pts = gpd_points(candidates, "EPSG:4326", "EPSG:32651")
    hall_pt = gpd_points(pd.DataFrame({"x": [hall["lon"]], "y": [hall["lat"]]}), "EPSG:4326", "EPSG:32651").iloc[0]
    d = np.hypot(pts.x.values - hall_pt.x, pts.y.values - hall_pt.y)
    print("WARNING: walk.graphml missing -> nearest-City-Hall baseline uses straight-line distance")
    return pd.Series(d, index=candidates["cand_id"].astype(str)), "straight-line fallback"


def build_results() -> None:
    """Write results/need.parquet, placements.csv, baselines.csv, sensitivity.csv (all scenarios)."""
    if paths.USING_FIXTURES:
        print("NOTE: ABOT_DATA=fixtures -> SYNTHETIC results into results/_synthetic/ (never for the deck)")
    cfg = config.load()
    grid = io.read("grid")
    facilities = io.read("facilities")
    candidates = io.read("candidates")
    rf = io.read("reach_facilities")
    rc = io.read("reach_candidates")
    grid = grid.copy()
    grid["cell_id"] = grid["cell_id"].astype(str)

    conditions = config.available_conditions(cfg)
    if not conditions:
        raise config.ConfigError("No condition has cited prevalence for all scenarios (see config.yaml)")
    k_max = int(cfg["k_max"])
    lam0 = float(cfg["equity_lambda"])
    speed0 = float(cfg["walk_speed_kmh"])
    speed_alt = float(cfg["alt_walk_speed_kmh"])
    limit0 = config.walk_limit_m(cfg)                       # 1125 m, 15 min @ 4.5 km/h
    limit_alt = config.walk_limit_m(cfg, speed_kmh=speed_alt)
    default_set = set(facilities.loc[facilities["counted"].astype(bool), "fac_id"].astype(str))
    hosp_set = set(facilities.loc[
        facilities["counted"].astype(bool) | facilities["type"].str.lower().eq("hospital"), "fac_id"
    ].astype(str))
    # Sensitivity: every listed clinic and doctor counted (the pre-Fri-8 PM definition).
    all_listed_set = set(facilities.loc[~facilities["type"].str.lower().eq("hospital"), "fac_id"].astype(str))
    covered_by = {
        (speed0, "all_listed"): io.covered_cells(rf, all_listed_set, limit0),
        (speed0, "default"): io.covered_cells(rf, default_set, limit0),
        (speed_alt, "default"): io.covered_cells(rf, default_set, limit_alt),
        (speed0, "with_hospitals"): io.covered_cells(rf, hosp_set, limit0),
    }
    cand_reach_by = {
        speed0: io.reach_sets(rc, "cand_id", limit0),
        speed_alt: io.reach_sets(rc, "cand_id", limit_alt),
    }
    weight = equity_weight(grid.set_index("cell_id")["wealth_quintile"], lam0)
    poorest2 = pd.to_numeric(grid.set_index("cell_id")["wealth_quintile"], errors="coerce") <= 2
    poorest2 = poorest2.fillna(False)

    # ---- need.parquet -------------------------------------------------------
    shares = {(c, s): config.prevalence(cfg, c, s) for c in conditions for s in SCENARIOS}
    need_rows, unmet_cache = [], {}
    for (cond, scen), share in sorted(shares.items()):
        n = need(grid, share)
        u = unmet(n, covered_by[(speed0, "default")])
        unmet_cache[(cond, scen, speed0, "default")] = u
        need_rows.append(pd.DataFrame({
            "cell_id": n.index, "condition": cond, "scenario": scen,
            "need": n.values.astype(float), "unmet": u.values.astype(float),
        }))
    need_df = pd.concat(need_rows, ignore_index=True)

    # ---- placements.csv -----------------------------------------------------
    st_cache: dict = {}
    placement_rows = []
    # (walk_speed, lambda, facility_set, scenarios covered)
    cases = [((speed0, lam0, "default"), conditions, list(SCENARIOS)),
             ((speed_alt, lam0, "default"), conditions, ["mid"]),
             ((speed0, 0.0, "default"), conditions, ["mid"]),
             ((speed0, 1.0, "default"), conditions, ["mid"]),
             ((speed0, lam0, "with_hospitals"), conditions, ["mid"]),
             ((speed0, lam0, "all_listed"), conditions, ["mid"])]
    CASE_KEYS = {(speed0, lam0, "default"): ["default", f"lambda_{lam0}", f"walk_speed_{speed0}", "facility_set_default"],
                 (speed_alt, lam0, "default"): [f"walk_speed_{speed_alt}"],
                 (speed0, 0.0, "default"): ["lambda_0.0"],
                 (speed0, 1.0, "default"): ["lambda_1.0"],
                 (speed0, lam0, "with_hospitals"): ["facility_set_with_hospitals"],
                 (speed0, lam0, "all_listed"): ["facility_set_all_listed"]}
    top5_by_case = {}
    for (speed, lam, fset), conds, scens in cases:
        covered = covered_by[(speed, fset)]
        reach_sets = cand_reach_by[speed]
        w = equity_weight(grid.set_index("cell_id")["wealth_quintile"], lam)
        for cond in conds:
            # Picks are scale-invariant in prevalence: run greedy ONCE at the mid scenario
            # (agreed with C after the app saw FP tie-flips across scenarios), then report
            # each scenario's counts by re-summing THAT scenario's real unmet over the
            # incremental cells. Counts are never scaled; picks never flip.
            if (cond, "mid", speed, fset) not in unmet_cache:
                unmet_cache[(cond, "mid", speed, fset)] = unmet(need(grid, shares[(cond, "mid")]), covered)
            u_mid = unmet_cache[(cond, "mid", speed, fset)]
            picks = greedy(u_mid, w, reach_sets, k_max, poorest2=poorest2)
            increments, seen = [], set()
            for p in picks:
                new = sorted(c for c in reach_sets[p["cand_id"]] if c in u_mid.index and c not in seen)
                seen |= set(new)
                increments.append(new)
            if cond == conditions[0]:
                for key in CASE_KEYS[(speed, lam, fset)]:
                    top5_by_case[key] = [p["cand_id"] for p in picks[:5]]
            for scen in scens:
                if (cond, scen, speed, fset) not in unmet_cache:
                    unmet_cache[(cond, scen, speed, fset)] = unmet(need(grid, shares[(cond, scen)]), covered)
                u = unmet_cache[(cond, scen, speed, fset)]
                for k in range(1, k_max + 1):
                    for p, cells_new in zip(picks[:k], increments[:k]):
                        added = float(u[cells_new].sum()) if cells_new else 0.0
                        if scen == "mid":
                            assert abs(added - p["added_people"]) < 1e-6, "increment mismatch vs greedy"
                        placement_rows.append({
                            "condition": cond, "scenario": scen, "walk_speed": speed, "lambda": lam,
                            "facility_set": fset, "k": k, "rank": p["rank"], "cand_id": p["cand_id"],
                            "added_people": added,
                            "added_poorest2": (float(u[[c for c in cells_new if bool(poorest2.get(c, False))]].sum())
                                               if poorest2 is not None and cells_new else None),
                            "service_type": _service_type(p["cand_id"], grid, facilities, rf, cfg, st_cache),
                        })
    placements = pd.DataFrame(placement_rows)

    # ---- baselines.csv ------------------------------------------------------
    # Prevalence scenarios cannot change picks (uniform scaling), so their sensitivity
    # rows repeat the default top-5: overlap = 5 by construction, and that is the claim.
    for scen in ("low", "high"):
        top5_by_case[f"prevalence_{scen}"] = list(top5_by_case.get("default", []))
    hall_walk, hall_source = _city_hall_walk(candidates, cfg)
    print(f"nearest-City-Hall baseline distances from: {hall_source}")
    rng_seed = int(cfg["seed"])
    draws = int(cfg["random_draws"])
    baseline_rows = []
    for cond in conditions:
        for scen in SCENARIOS:
            u = unmet_cache[(cond, scen, speed0, "default")]
            reach_sets = cand_reach_by[speed0]
            for k in range(1, k_max + 1):
                abot_people = float(placements[(placements.condition == cond) & (placements.scenario == scen)
                                                & (placements.walk_speed == speed0) & (placements["lambda"] == lam0)
                                                & (placements.facility_set == "default") & (placements.k == k)]
                                    ["added_people"].sum())
                mpb = baseline_most_populous_barangay(grid, candidates, k)
                nch = baseline_nearest_city_hall(hall_walk, k)
                rnd = baseline_random(list(cand_reach_by[speed0]), k, draws, rng_seed)
                rnd_people = [people_reached(u, reach_sets, d) for d in rnd]
                for method, people, p5, p95 in [
                    ("abot", abot_people, None, None),
                    ("most_populous_barangay", people_reached(u, reach_sets, mpb), None, None),
                    ("nearest_city_hall", people_reached(u, reach_sets, nch), None, None),
                    ("random", float(np.mean(rnd_people)), float(np.percentile(rnd_people, 5)),
                     float(np.percentile(rnd_people, 95))),
                ]:
                    baseline_rows.append({
                        "condition": cond, "scenario": scen, "method": method, "k": k,
                        "people": people, "p5": p5, "p95": p95,
                    })
    baselines = pd.DataFrame(baseline_rows)

    # ---- sensitivity.csv ----------------------------------------------------
    default_top5 = top5_by_case.get("default", [])
    sens_rows = []
    for case in sorted(c for c in top5_by_case if c != "default"):
        ids = top5_by_case[case]
        sens_rows.append({
            "case": case, "top5_ids": ";".join(ids),
            "overlap_with_default": len(set(ids) & set(default_top5)),
        })
    sensitivity = pd.DataFrame(sens_rows)

    # ---- write --------------------------------------------------------------
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    need_df.to_parquet(paths.RESULTS / "need.parquet", index=False)
    placements.to_csv(paths.RESULTS / "placements.csv", index=False)
    baselines.to_csv(paths.RESULTS / "baselines.csv", index=False)
    sensitivity.to_csv(paths.RESULTS / "sensitivity.csv", index=False)
    print(f"need: {len(need_df):,} rows | placements: {len(placements):,} | "
          f"baselines: {len(baselines):,} | sensitivity: {len(sensitivity):,} cases")

    _write_figures(need_df, placements, baselines, sensitivity, conditions)
    fac_note = (f"public health centres and barangay health stations ({len(default_set)} of "
                f"{len(facilities)} listed facilities; listed, not verified)")
    _write_headline(cfg, need_df, placements, baselines, sensitivity, conditions, shares, hall_source, fac_note)


def _write_figures(need_df, placements, baselines, sensitivity, conditions) -> None:
    """One PNG per results CSV, 2x resolution (deck charts; owner B)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir = paths.FIGURES / "_synthetic" if paths.USING_FIXTURES else paths.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)  # synthetic charts must never sit next to deck charts

    # reach curve: cumulative people reached by greedy k = 1..10 (mid scenario)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conditions:
        base = placements[(placements.condition == cond) & (placements.scenario == "mid")
                          & (placements.facility_set == "default") & (placements.walk_speed == 4.5)
                          & (placements["lambda"] == 0.5)]
        cum = base[base.k == base.k.max()].sort_values("rank")["added_people"].cumsum()
        ax.plot(range(1, len(cum) + 1), cum.values, marker="o", label=cond)
    ax.set_xlabel("number of new service points (k)")
    ax.set_ylabel("additional likely patients reached")
    ax.set_title("Greedy placement: cumulative unweighted reach (mid scenario, modelled)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "placements.png", dpi=200)
    plt.close(fig)

    # baselines bar chart at k = 5
    k5 = baselines[(baselines.k == 5) & (baselines.condition == conditions[0]) & (baselines.scenario == "mid")]
    fig, ax = plt.subplots(figsize=(8, 5))
    order = [m for m in BASELINE_METHODS]
    labels, vals, errs_lo, errs_hi = [], [], [], []
    for m in order:
        row = k5[k5.method == m]
        if row.empty:
            continue
        labels.append(m.replace("_", " "))
        vals.append(float(row["people"].iloc[0]))
        errs_lo.append(vals[-1] - float(row["p5"].iloc[0]) if pd.notna(row["p5"].iloc[0]) else 0.0)
        errs_hi.append(float(row["p95"].iloc[0]) - vals[-1] if pd.notna(row["p95"].iloc[0]) else 0.0)
    ax.bar(labels, vals, yerr=[errs_lo, errs_hi], capsize=4, color=["#2a9d8f", "#8ecae6", "#c5a3ff", "#f4a261"])
    ax.set_ylabel("likely patients reached (k = 5)")
    ax.set_title(f"ABOT vs simple placement rules ({conditions[0]}, mid, modelled)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "baselines.png", dpi=200)
    plt.close(fig)

    # sensitivity overlap
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sensitivity["case"], sensitivity["overlap_with_default"], color="#e76f51")
    ax.set_ylim(0, 5)
    ax.set_ylabel("top-5 sites shared with the default case")
    ax.set_title("Sensitivity: which picks stay the same")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=200)
    plt.close(fig)
    print(f"figures written to {out_dir} (placements.png, baselines.png, sensitivity.png)")


def _fmt(x: float) -> str:
    return f"{round(float(x)):,}"


def _write_headline(cfg, need_df, placements, baselines, sensitivity, conditions, shares, hall_source,
                    fac_note: str = "default (listed, not verified)") -> None:
    """results/headline.md: every number the deck/video may use, with its CSV origin."""
    cond = conditions[0]  # headline condition (hypertension); others listed after
    mid = baselines[(baselines.condition == cond) & (baselines.scenario == "mid")]
    k5_abot = float(mid[(mid.method == "abot") & (mid.k == 5)]["people"].iloc[0])

    lines = ["# Headline numbers — every figure below comes straight from a results file.",
             "",
             f"Condition: **{cond}** (prevalence shares: low {shares[(cond,'low')]:.3f} / "
             f"mid {shares[(cond,'mid')]:.3f} / high {shares[(cond,'high')]:.3f}, from `config.yaml` with "
             "cited sources in `docs/prevalence_sources.md`).",
             f"Default assumptions: {cfg['walk_speed_kmh']} km/h, {cfg['walk_minutes']}-minute walk "
             f"({config.walk_limit_m(cfg):.0f} m), equity lambda {cfg['equity_lambda']}, "
             f"existing care = {fac_note}.",
             ""]

    nd = need_df[need_df.condition == cond]
    need_r = {s: float(nd[nd.scenario == s]["need"].sum()) for s in SCENARIOS}
    unmet_r = {s: float(nd[nd.scenario == s]["unmet"].sum()) for s in SCENARIOS}
    lines += [
        "## Modelled need and unmet need",
        "",
        f"- Total modelled need ({cond}): **{_fmt(need_r['mid'])}** adults 30-79 likely patients "
        f"(range {_fmt(need_r['low'])}-{_fmt(need_r['high'])}) — `results/need.parquet`, sum(need), "
        f"condition={cond}, scenario=low/mid/high.",
        f"- Beyond a {cfg['walk_minutes']}-minute walk of a counted facility: **{_fmt(unmet_r['mid'])}** "
        f"(range {_fmt(unmet_r['low'])}-{_fmt(unmet_r['high'])}) — `results/need.parquet`, sum(unmet).",
        ""]
    for c2 in conditions[1:]:
        nd2 = need_df[need_df.condition == c2]
        u2 = {s: float(nd2[nd2.scenario == s]["unmet"].sum()) for s in SCENARIOS}
        lines += [f"- Same for {c2}: unmet **{_fmt(u2['mid'])}** "
                  f"(range {_fmt(u2['low'])}-{_fmt(u2['high'])}) — `results/need.parquet`."]

    k5 = {s: float(baselines[(baselines.condition == cond) & (baselines.scenario == s)
                              & (baselines.method == "abot") & (baselines.k == 5)]["people"].iloc[0])
          for s in SCENARIOS}
    lines += [
        "",
        "## Five new service points",
        "",
        f"- k = 5 reaches **{_fmt(k5['mid'])}** more likely patients "
        f"(range {_fmt(k5['low'])}-{_fmt(k5['high'])}) — `results/baselines.csv`, method=abot, k=5.",
        ""]
    n_picks5 = len(placements[(placements.condition == cond) & (placements.scenario == "mid")
                              & (placements.k == 5) & (placements.facility_set == "default")
                              & (placements.walk_speed == 4.5) & (placements["lambda"] == 0.5)])
    if 0 < n_picks5 < 5:
        lines += [f"- Greedy recommended only **{n_picks5} of the 5** requested sites: after them no remaining "
                  f"candidate adds any reach — the rest of the unmet need lies beyond every candidate's "
                  f"15-minute walk — `results/placements.csv`, k=5 (ranks 1-{n_picks5} only).", ""]
    lines += ["## Comparison with baselines at k = 5", ""]
    for m in BASELINE_METHODS:
        row = mid[(mid.method == m) & (mid.k == 5)]
        if row.empty:
            continue
        v = float(row["people"].iloc[0])
        extra = ""
        if m == "random":
            extra = f" (5th-95th percentile {float(row['p5'].iloc[0]):.0f}-{float(row['p95'].iloc[0]):.0f})"
        # Quote counts, never a % advantage: a small baseline makes the % meaningless.
        lines.append(f"- {m}: **{v:,.0f}**{extra} — `results/baselines.csv`, method={m}, k=5.")
    lines += ["", f"Nearest-City-Hall distances: {hall_source}.", ""]
    lines += ["## Sensitivity: top-5 stability", ""]
    n_default = len(placements[(placements.condition == cond) & (placements.scenario == "mid")
                               & (placements.k == 5) & (placements.facility_set == "default")
                               & (placements.walk_speed == 4.5) & (placements["lambda"] == 0.5)])
    for r in sensitivity.itertuples():
        lines.append(f"- {r.case}: {r.overlap_with_default} of {n_default} default picks kept "
                     f"(case picks = {r.top5_ids}) — `results/sensitivity.csv`.")
    lines += ["", "## Service types for the default top-5", ""]
    top5 = placements[(placements.condition == cond) & (placements.scenario == "mid")
                      & (placements.k == 5) & (placements.facility_set == "default")
                      & (placements.walk_speed == 4.5) & (placements["lambda"] == 0.5)]
    for r in top5.sort_values("rank").itertuples():
        lines.append(f"- rank {r.rank}, {r.cand_id}: +{r.added_people:,.0f} likely patients; "
                     f"{r.service_type} — `results/placements.csv`, k=5, rank={r.rank}.")
    if (top5["service_type"] == SERVICE_PLACEHOLDER).all():
        lines += ["", f"**Note:** service types are the placeholder '{SERVICE_PLACEHOLDER}' until "
                      "owner A delivers `abot/service_type.py` (due Fri 4:30 PM); rerun "
                      "`python -m abot.need` afterwards."]
    (paths.RESULTS / "headline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"headline.md written to {paths.RESULTS / 'headline.md'}")


if __name__ == "__main__":
    build_results()
