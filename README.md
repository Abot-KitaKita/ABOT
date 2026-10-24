# ABOT — chronic-care access planner (Mandaluyong City prototype)

ABOT is a decision-support prototype for placing additional **hypertension and diabetes** service points.
For Mandaluyong City it estimates where adults likely to need chronic care live, measures how much of that need
lies **beyond a 15-minute walk** of the city's public health centres, and suggests where **1–10 new service
points** would bring the most people within walking reach, giving extra weight to poorer areas. Every number is a
modelled planning estimate, shown as a low–high range, with its assumptions stated.

## Results at a glance

Hypertension, middle prevalence estimate, 15-minute walk at 4.5 km/h, 14 public health centres counted as existing
care (full list with sources in [`results/headline.md`](results/headline.md)):

| | Likely patients (low–high range) |
| --- | --- |
| Modelled need, adults 30–79 | 58,644 (46,915–70,373) |
| Beyond a 15-minute walk today | 3,286 (2,629–3,943) |
| Newly within reach with 5 suggested sites | 3,101 (2,481–3,721) |

The same five sites, compared with simple placement rules at the same settings:

| Way of picking 5 sites | People newly within reach |
| --- | --- |
| **ABOT** (equity-weighted maximal covering) | **3,101** |
| Random sites (mean of 200 draws; 5th–95th percentile 0–1,565) | 542 |
| Most populous barangays | 300 |
| Closest to City Hall | 0 |

The five suggested sites stay the same when the equity weight is switched off or doubled, and under the low and
high prevalence estimates; 4 of 5 stay with a slower walking speed (4.0 km/h). Diabetes: 1,633 (1,185–2,141) likely
patients beyond a 15-minute walk.

## What ABOT does

1. **Need:** adults aged 30–79 on a 100 m population grid × cited prevalence, under low / middle / high estimates.
2. **Unmet need:** modelled need beyond a 15-minute walk (1,125 m of walking network at 4.5 km/h) of the counted
   facilities: public health centres and barangay health stations.
3. **Placement:** greedy equity-weighted maximal covering over public candidate sites (barangay halls, schools,
   covered courts, community centres); each person is counted once.
4. **Comparison:** the same number of sites placed by three simple rules: random sites, most populous barangays,
   closest to City Hall.
5. **Service type:** a simple rule suggests a *mobile clinic stop* (weak mobile network, or no health centre within a
   30-minute walk for referral) or an *assisted teleconsult point*.
6. **Placement brief:** a one-page plain-language summary a health office can review and download.

## Running the app

Requires Python 3.11. All processed data and results are included, so no downloads are needed.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser (an internet connection is needed for the base map). In the app:

- **Plan settings** (top left): condition, how common the condition is (low / middle / high), walking speed,
  number of new care spots, *Poorer areas first*, and **Clinics we count**, where a planner can untick a facility
  that does not offer chronic care and see the results update.
- **View** menu (top right): **Map** (where people live far from care and the suggested spots), **Compare** (ABOT
  next to the simple rules), **Placement brief** (download), **Assumptions & sources**.
- **Back to the starting plan** restores the settings above.

Checks:

```bash
pytest -q                    # tests (models, rules, data contracts, brief)
python -m abot.contracts     # every data and result file against its contract
```

## Rebuilding the data from scratch

Raw downloads are not included. Download each source listed in [`docs/data_sources.csv`](docs/data_sources.csv)
into `data/raw/`, then run:

```bash
python scripts/download_walk.py   # OpenStreetMap walking network -> data/raw/walk.graphml
python -m abot.build_grid         # grid, facilities, candidate sites -> data/processed/
python -m abot.reach              # walking reach tables -> data/processed/
python -m abot.need               # need, unmet need, placements, baselines, sensitivity -> results/, figures/
python -m abot.contracts          # check every file against its contract
```

The pipeline is deterministic (fixed seed, sorted tie-breaks). Re-running `python -m abot.need` on the included
data reproduces the files in `results/`; the City Hall walking distances are cached in
`data/processed/city_hall_walk.csv`, so the walking network is not needed for that step.

## Data sources

All sources are open data; each field is labelled **observed** or **modelled** in
[`docs/data_sources.csv`](docs/data_sources.csv):

- **Population:** WorldPop 2020, 100 m grid, total and age-sex (CC BY 4.0).
- **Barangays, relative wealth (2022) and mobile network speed (Ookla, 2022):** Project CCHAIN (CC BY 4.0).
- **Health facilities, candidate sites, walking network, City Hall, city boundary:** OpenStreetMap (ODbL).
- **Hypertension prevalence:** 33% of adults 30–79 (WHO Global report on hypertension 2025, via a Resolve to Save
  Lives fact sheet).
- **Diabetes prevalence:** 16.4% [11.9–21.5] of adults 30+ (WHO Global Health Observatory, 2022).

Citations, values and checks: [`docs/prevalence_sources.md`](docs/prevalence_sources.md),
[`docs/qa_log.md`](docs/qa_log.md). All assumptions are in [`config.yaml`](config.yaml); the method is in
[`docs/methods.md`](docs/methods.md).

## Limitations

- Need is **modelled** from published prevalence and 2020 population estimates, not observed patient counts; it is
  shown as a range.
- Diabetes prevalence is the WHO crude estimate for adults **30 and over** (including 80+), applied to adults 30–79,
  which slightly overstates diabetes need. Hypertension low/high estimates are ±20% sensitivity assumptions because
  the source gives no interval.
- Facilities are **listed, not verified** for chronic-care services, staff, hours or capacity; results are
  **proximity-based only**. Being within walking reach is not the same as receiving care.
- Only public health centres and barangay health stations count as existing care; private clinics, specialist
  centres and hospitals do not (a sensitivity case counting every listed clinic is in `results/sensitivity.csv`).
- Walking times come from the OpenStreetMap network at an assumed speed and are **not validated on real routes**.
- Wealth groups rank barangays within the city only, over a narrow range of index values; mobile network quality uses
  Ookla speed tests as a proxy.
- Candidate sites depend on OpenStreetMap completeness. Private schools are removed by name; others whose names do
  not show it may remain. Suggested sites are not checked for permission, space or readiness.
- Residents near the city boundary may use facilities in neighbouring cities.
- Results are planning estimates, not predictions of individual need or health outcomes.

## Repository layout

```
app.py, app.css        Streamlit app
abot/                  pipeline: grid, reach, need, placement, service type, sentences, brief, data contracts
config.yaml            every assumption, with its source
data/processed/        grid, facilities, candidate sites, walking reach tables, city boundary
results/               need, placements, baselines, sensitivity, headline numbers
figures/               charts made from results/
docs/                  methods, data sources, prevalence sources, QA log
scripts/               download helpers and synthetic-fixture generator
tests/                 pytest suite and small synthetic fixtures (used for tests only)
```
