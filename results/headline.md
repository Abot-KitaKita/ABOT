# Headline numbers — every figure below comes straight from a results file.

Condition: **hypertension** (prevalence shares: low 0.264 / mid 0.330 / high 0.396, from `config.yaml` with cited sources in `docs/prevalence_sources.md`).
Default assumptions: 4.5 km/h, 15-minute walk (1125 m), equity lambda 0.5, existing care = public health centres and barangay health stations (14 of 42 listed facilities; listed, not verified).

## Modelled need and unmet need

- Total modelled need (hypertension): **58,644** adults 30-79 likely patients (range 46,915-70,373) — `results/need.parquet`, sum(need), condition=hypertension, scenario=low/mid/high.
- Beyond a 15-minute walk of a counted facility: **3,286** (range 2,629-3,943) — `results/need.parquet`, sum(unmet).

- Same for diabetes: unmet **1,633** (range 1,185-2,141) — `results/need.parquet`.

## Five new service points

- k = 5 reaches **3,101** more likely patients (range 2,481-3,721) — `results/baselines.csv`, method=abot, k=5.

## Comparison with baselines at k = 5

- abot: **3,101** — `results/baselines.csv`, method=abot, k=5.
- most_populous_barangay: **300** — `results/baselines.csv`, method=most_populous_barangay, k=5.
- nearest_city_hall: **0** — `results/baselines.csv`, method=nearest_city_hall, k=5.
- random: **542** (5th-95th percentile 0-1565) — `results/baselines.csv`, method=random, k=5.

Nearest-City-Hall distances: OSM walking network (cached in data/processed/city_hall_walk.csv).

## Sensitivity: top-5 stability

- facility_set_all_listed: 3 of 5 default picks kept (case picks = way/264398781;way/790263475;node/9776218417) — `results/sensitivity.csv`.
- facility_set_default: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- facility_set_with_hospitals: 4 of 5 default picks kept (case picks = node/9776218417;way/264398781;way/790263475;node/13810839431;relation/13360320) — `results/sensitivity.csv`.
- lambda_0.0: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- lambda_0.5: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- lambda_1.0: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- prevalence_high: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- prevalence_low: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.
- walk_speed_4.0: 4 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;way/1180298503) — `results/sensitivity.csv`.
- walk_speed_4.5: 5 of 5 default picks kept (case picks = node/4196499444;node/9776218417;way/790263475;way/264398781;node/13810839431) — `results/sensitivity.csv`.

## Service types for the default top-5

- rank 1, node/4196499444: +1,560 likely patients; Mobile clinic stop — `results/placements.csv`, k=5, rank=1.
- rank 2, node/9776218417: +694 likely patients; Mobile clinic stop — `results/placements.csv`, k=5, rank=2.
- rank 3, way/790263475: +459 likely patients; Mobile clinic stop — `results/placements.csv`, k=5, rank=3.
- rank 4, way/264398781: +327 likely patients; Mobile clinic stop — `results/placements.csv`, k=5, rank=4.
- rank 5, node/13810839431: +61 likely patients; Mobile clinic stop — `results/placements.csv`, k=5, rank=5.
