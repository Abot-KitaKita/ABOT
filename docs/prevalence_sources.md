# Prevalence sources

Every prevalence value in `config.yaml` must appear here with a source you opened yourself.
Every value was read on the source page itself; no figure is taken second-hand.

| Condition | Scenario | Value (share) | Population / age | Source (full citation) | URL | Page/table | Opened by | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hypertension | mid | 0.33 | Adults 30-79, Philippines | Resolve to Save Lives (2025). *How to increase hypertension control 6-fold through primary health care — lessons from scaling the Philippines Healthy Hearts Programme* [citing WHO Global report on hypertension 2025] | https://resolvetosavelives.org/wp-content/uploads/2025/09/873_PRTN_Philippines-Executive-Summary_Fact-Sheet_0925_Rev-A_v5.pdf | p.1, Context bullet 1: "affecting 33% of adults aged 30-79 (16.8M people)" | B | Confirmed on page (PDF text extracted 2026-10-23). Same page: only 1 in 3 treated, >4 in 5 uncontrolled (slide 2 claims) |
| hypertension | low | 0.264 | Adults 30-79, Philippines | derived from the cited mid value | — | — | B | source gives no interval; mid x 0.8, labelled "sensitivity assumption (+/-20%)" |
| hypertension | high | 0.396 | Adults 30-79, Philippines | derived from the cited mid value | — | — | B | source gives no interval; mid x 1.2, labelled "sensitivity assumption (+/-20%)" |
| diabetes | mid | 0.164 | Adults 30+ (incl. 80+), Philippines | WHO Global Health Observatory indicator "Prevalence of diabetes, crude" (NCD_DIABETES_PREVALENCE_CRUDE), Philippines, both sexes, age 30+, year 2022 | https://www.who.int/data/gho/data/indicators/indicator-details/GHO/prevalence-of-diabetes-crude (data via https://ghoapi.azureedge.net/api/NCD_DIABETES_PREVALENCE_CRUDE) | 2022 row, SEX_BTSX, AGEGROUP_YEARS30-PLUS: 16.4 [11.9-21.5] | B | API query opened 2026-10-23. Crude (survey-based), not age-standardised. Age band 30+ vs our 30-79 grid: slight overstatement, listed as limitation |
| diabetes | low | 0.119 | Adults 30+, Philippines | same WHO GHO indicator, 2022 uncertainty interval | same | lower bound 11.9 | B | source's own interval, not a derived assumption |
| diabetes | high | 0.215 | Adults 30+, Philippines | same WHO GHO indicator, 2022 uncertainty interval | same | upper bound 21.5 | B | source's own interval, not a derived assumption |
