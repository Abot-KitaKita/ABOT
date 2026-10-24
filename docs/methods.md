# Methods

One short paragraph per step; every assumption points to `config.yaml`. Every reported number comes
only from `results/` files (summarised in `results/headline.md`).

1. **Population and target group** — WorldPop 2020 100 m grid (`phl_ppp_2020.tif`) clipped to
   Mandaluyong, built by `abot/build_grid.py`. Adults aged 30–79 (`grid.parquet.adults_30_79`)
   are the sum of WorldPop's 2020 age-sex rasters for ages 30–34 through 75–79. Barangays and their
   PSGC codes, relative wealth (2022) and Ookla mobile download speed (2022) come from Project CCHAIN;
   wealth quintiles and network terciles are ranked within the city. Check: grid total 411,208 vs the
   PSA 2020 census total of 425,758 (−3.4%; `docs/qa_log.md`).
2. **Modelled need** — `need(cell) = adults_30_79 × prevalence(condition, scenario)`. Conditions
   (`hypertension`, `diabetes`) are separate layers and are never summed. Prevalence comes only
   from `config.yaml`, where every value carries a cited source (`docs/prevalence_sources.md`):
   hypertension mid 0.33 (Resolve to Save Lives 2025 fact sheet citing WHO 2025); its low/high
   are mid ±20%, labelled *sensitivity assumption* because the source gives no interval;
   diabetes mid 0.164 with low 0.119 / high 0.215 taken from the WHO Global Health Observatory's
   own 2022 uncertainty interval (crude prevalence, adults 30+ — an age band slightly wider than
   our 30–79, a small overstatement and a stated limitation).

3. **Walking reach** — OpenStreetMap walking network for Mandaluyong (`data/raw/walk.graphml`,
   downloaded by `scripts/download_walk.py`). Each facility, candidate site and cell centroid is
   snapped to its nearest network node; walk distance is the shortest path along the network in
   metres ("length" edge weights). Snap residuals (point-to-node straight line) are not added, so
   distances are approximate at the scale of one block. `reach_facilities.parquet` stores all
   pairs within 2,250 m (the 30-minute referral check), `reach_candidates.parquet` within
   1,125 m (the 15-minute limit).

4. **Unmet need** — `unmet(cell) = need(cell)` if no counted facility reaches the cell within
   `walk_minutes` × `walk_speed_kmh` (15 min × 4.5 km/h = 1,125 m), else 0. Facilities are the
   rows of `facilities.geojson` with `counted = True` in the *default* set; the `with_hospitals`
   set adds hospitals. Every facility is *Listed, not verified*.

5. **Equity weight** — `weight(cell) = 1 + λ × (5 − wealth_quintile) / 4` with `equity_lambda`
   (λ = 0.5). Quintile 1 (poorest) is weighted ×1.5, quintile 5 ×1.0. Cells with missing wealth
   get weight 1.0. The weight changes which site is chosen; reported counts stay unweighted.

6. **Placement** — greedy equity-weighted maximal covering (`abot/place.py:greedy`). Repeat k
   times: pick the candidate whose not-yet-covered cells have the largest Σ unmet × weight,
   ties broken by `cand_id` ascending, then mark its cells covered. Stop early when nothing
   adds a positive score. Each person (cell) is counted once. `added_people` is always the
   unweighted sum of newly covered unmet need. k = 1…10 in `results/placements.csv`.

7. **Baselines** — same coverage rule, same k, three simple rules plus ABOT in
   `results/baselines.csv`: (a) `most_populous_barangay`: for the k most populous barangays,
   the candidate nearest that barangay's population-weighted centroid (straight-line in
   EPSG:32651 — a documented simplification of "nearest"); (b) `nearest_city_hall`: the k
   candidates with the shortest walking distance to City Hall (OSM way 112549085; distances
   from one Dijkstra run on the walking network); (c) `random`: 200 reproducible draws
   (`seed` 42) of k candidates, reported as mean with 5th–95th percentile.

8. **Service-type rule** — labels come from `abot/service_type.py` (CONTEXT.md section 3), called
   through `need.py` with the planner's current `counted` flags: weakest network tercile → mobile
   clinic stop; otherwise a counted facility within a 30-minute walk → assisted teleconsult point;
   otherwise mobile clinic stop. Network data cover all 27 barangays.
9. **Sensitivity** — `results/sensitivity.csv` reruns the greedy top-5 under: walking speed
   4.0 km/h (1,000 m), λ = 0 and λ = 1, the `with_hospitals` facility set, and the low/high
   prevalence scenarios; it reports how many of the default top-5 survive in each case.
   Picks are *scenario-invariant by construction*: the pipeline runs greedy once per
   (condition, case) at the mid scenario and re-sums each scenario's real unmet over the
   incremental cell sets — never scaling gains, never re-picking — so the prevalence rows
   are 5/5 by design (an assert checks mid rows against greedy's own gains). This avoids
   floating-point tie-flips between scenarios (set iteration walks sorted cell lists, and
   greedy is not rerun per scenario). With ~1,100 populated cells the placement runs in seconds, so the app
   recomputes it live.

10. **Limitations** — need is modelled, not observed; facilities are listed, not verified;
    results are proximity-only (no capacity data); walking times are not route-validated
    (node-snapped network paths, not field-checked routes); Ookla 2019–2022 is a proxy for
    network quality; candidate sites depend on OpenStreetMap completeness; residents near the
    city edge may use facilities in neighbouring cities; the diabetes age band (30+) is wider
    than the modelled group (30–79); these are planning estimates, not predictions of
    individual need or health outcomes.

## Reproducibility

- Everything above is deterministic: `seed` 42 for random draws, sorted ids for every
  tie-break, and sorted cell order for all floating-point sums (the results are verified
  byte-identical across runs with different `PYTHONHASHSEED`).
- Clean rebuild: `python -m abot.build_grid` → `python -m abot.reach` → `python -m abot.need`,
  then `python -m abot.contracts` and `pytest -q`.
- Synthetic fixtures (`ABOT_DATA=fixtures`) exist only for development; they write to
  `results/_synthetic/` (git-ignored) so synthetic numbers can never mix with real results.
