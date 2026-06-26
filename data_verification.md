# Data Verification Log
## Disease Burden Among Aging Canadians — Multi-Source Analysis Dashboard

This document traces every key figure in the dashboard and paper back to its exact
source file, extraction method, and verification notebook. It exists so that any
number shown in the dashboard or cited in the paper can be independently verified
against the original raw data.

---

## How to verify independently

1. **Run the notebooks** in this repository against the original source files
2. **Query GBD directly** at [vizhub.healthdata.org/gbd-results](https://vizhub.healthdata.org/gbd-results) using: Location = Canada, Age = 60+ years, Sex = Both, the cause names below, Measure = DALYs or Deaths, Metric = Number or Rate
3. **Download CIHI data** at [cihi.ca](https://www.cihi.ca) — Pharmaceutical Data Tool and HMDB/OMHRS 2024–2025
4. **Download StatCan data** at [statcan.gc.ca](https://www150.statcan.gc.ca) — Table 17-10-0057-01

---

## GBD 2023 — DALYs, Canadians 60+, 2023
**Source file:** Category-specific GBD exports (original 7) + `IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx` (new 5)
**Notebook:** `01_gbd_trend_analysis_12category.ipynb`
**Filter:** cause_name = [disease], metric_name = 'Number', measure_name = 'DALYs (Disability-Adjusted Life Years)', age_name = '60+ years', sex_name = 'Both', year = 2023

| Disease Category | DALYs 2023 | Source File | Verified |
|---|---|---|---|
| Neoplasms | 1,557,828 | IHME-GBD_2023_DATA-1c9f46d3-1_neo.csv | ✅ |
| Cardiovascular diseases | 1,333,532 | IHME-GBD_2023_DATA-141409a9-1_cardio.xlsx | ✅ |
| Neurological disorders | 582,179 | IHME-GBD_2023_DATA-cd6c1a03-1_neuro.csv | ✅ |
| Musculoskeletal disorders | 552,267 | IHME-GBD_2023_DATA-100da0be-1_musculo.csv | ✅ |
| Chronic respiratory diseases | 415,842 | IHME-GBD_2023_DATA-5da954e9-1_respiratory.csv | ✅ |
| Diabetes and kidney diseases | 374,726 | IHME-GBD_2023_DATA-dd10953b-1_diabetics.csv | ✅ |
| Digestive diseases | 274,238 | IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx | ✅ |
| Sense organ diseases | 274,025 | IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx | ✅ |
| Other non-communicable diseases | 207,574 | IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx | ✅ |
| Mental disorders | 194,075 | IHME-GBD_2023_DATA-4eb8814a-1_mental.csv | ✅ |
| Substance use disorders | 61,873 | IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx | ✅ |
| Skin and subcutaneous diseases | 48,238 | IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx | ✅ |
| **Total (12 categories)** | **5,876,397** | Computed sum | ✅ |
| All-cause total | 6,895,367 | IHME-GBD_2023_DATA-fa7b4ff0-1.csv | ✅ |
| 12-cat as % of all-cause | 85.2% | Computed: 5,876,397 / 6,895,367 | ✅ |

---

## GBD 2023 — DALYs, Canadians 60+, All Years (1995–2023)
**Notebook:** `01_gbd_trend_analysis_12category.ipynb`
**Years extracted:** 1995, 2000, 2005, 2010, 2015, 2019, 2023

| Disease | % Change 1995–2023 | 1995 value | 2023 value |
|---|---|---|---|
| Substance use disorders | +223.6% | 19,119 | 61,873 |
| Skin and subcutaneous diseases | +164.1% | 18,263 | 48,238 |
| Mental disorders | +160.7% | 74,447 | 194,075 |
| Neurological disorders | +135.7% | 246,959 | 582,179 |
| Other non-communicable diseases | +135.7% | 88,070 | 207,574 |
| Sense organ diseases | +117.0% | 126,257 | 274,025 |
| Diabetes and kidney diseases | +116.7% | 172,889 | 374,726 |
| Musculoskeletal disorders | +112.9% | 259,358 | 552,267 |
| Digestive diseases | +112.5% | 129,061 | 274,238 |
| Chronic respiratory diseases | +89.4% | 219,540 | 415,842 |
| Neoplasms | +60.1% | 973,032 | 1,557,828 |
| Cardiovascular diseases | +11.5% | 1,196,259 | 1,333,532 |

---

## GBD 2023 — Age-Standardized DALY Rates, Canadians 60+
**Notebook:** `01_gbd_trend_analysis_12category.ipynb`
**Filter:** metric_name = 'Rate', measure_name = 'DALYs', age_name = '60+ years'

| Disease | Rate 1995 | Rate 2023 | % Change |
|---|---|---|---|
| Substance use disorders | 405.9 | 605.8 | +49.2% |
| Skin and subcutaneous diseases | 387.7 | 472.3 | +21.8% |
| Mental disorders | 1,580.4 | 1,900.1 | +20.2% |
| Neurological disorders | 5,242.5 | 5,699.7 | +8.7% |
| Other non-communicable diseases | 1,869.6 | 2,032.2 | +8.7% |
| Sense organ diseases | 2,680.2 | 2,682.8 | +0.1% |
| Diabetes and kidney diseases | 3,670.2 | 3,668.7 | −0.0% |
| Musculoskeletal disorders | 5,505.8 | 5,406.9 | −1.8% |
| Digestive diseases | 2,739.8 | 2,684.9 | −2.0% |
| Chronic respiratory diseases | 4,660.5 | 4,071.2 | −12.6% |
| Neoplasms | 20,655.9 | 15,251.7 | −26.2% |
| Cardiovascular diseases | 25,394.7 | 13,055.7 | −48.6% |

---

## GBD 2023 — All-Cause Totals, Canadians 60+
**Source file:** `IHME-GBD_2023_DATA-fa7b4ff0-1.csv`
**Filter:** cause_name = 'All causes', age_name = '60+ years', sex_name = 'Both'

| Metric | 1995 | 2023 | % Change |
|---|---|---|---|
| DALYs | 3,948,183 | 6,895,367 | +74.6% |
| Deaths | 172,766 | 280,718 | +62.5% |

---

## Forecasting — Polynomial Regression (degree=2), 2024–2040
**Notebook:** `02_forecasting_polynomial_regression_12category.ipynb`
**Method:** Cubic spline interpolation → polynomial regression (degree=2), trained on 29 annual points (1995–2023), forecast 2024–2040
**Model performance:** R² > 0.96, MAPE < 3.1%, LOOCV over 7 observed time points

| Disease | 2023 | 2025 | 2030 | 2035 | 2040 | Growth |
|---|---|---|---|---|---|---|
| Substance use disorders | 61,873 | 68,221 | 88,187 | 111,554 | 138,321 | +123.6% |
| Digestive diseases | 274,238 | 288,880 | 348,959 | 418,734 | 498,204 | +81.7% |
| Mental disorders | 194,075 | 198,269 | 242,364 | 293,231 | 350,870 | +80.8% |
| Musculoskeletal disorders | 552,267 | 604,804 | 713,547 | 836,954 | 975,023 | +76.5% |
| Skin and subcutaneous diseases | 48,238 | 51,512 | 61,398 | 72,548 | 84,963 | +76.1% |
| Sense organ diseases | 274,025 | 294,072 | 348,353 | 410,351 | 480,065 | +75.2% |
| Chronic respiratory diseases | 415,842 | 446,406 | 519,949 | 603,947 | 698,401 | +67.9% |
| Cardiovascular diseases | 1,333,532 | 1,387,908 | 1,600,147 | 1,865,612 | 2,184,304 | +63.8% |
| Neurological disorders | 582,179 | 619,538 | 716,951 | 824,541 | 942,307 | +61.9% |
| Neoplasms | 1,557,828 | 1,612,132 | 1,817,064 | 2,050,653 | 2,312,898 | +48.5% |
| Other non-communicable diseases | 207,574 | 212,140 | 238,064 | 265,343 | 293,976 | +41.6% |
| Diabetes & kidney diseases | 374,726 | 376,104 | 422,952 | 473,994 | 529,228 | +41.2% |
| **Total** | **5,876,397** | **6,159,986** | **7,117,935** | **8,227,462** | **9,488,560** | **+61.5%** |

---

## CIHI Pharmaceutical Data Tool — Provincial Senior Population & Antidepressant Rx
**Source file:** `prescribed-drug-use-seniors-pharm-data-tool-data-tables-en.xlsx`
**Filter:** Sex = Total, Age group = Total (65+), Drug = Antidepressants, Year = 2020–2024
**Notebook:** Not separately notebooked — values extracted and verified manually against the raw Excel file

| Province | Senior Pop (2024) | Growth % (2020–24) | Antidepressant Rx | Rx per 1,000 |
|---|---|---|---|---|
| Ontario | 2,954,128 | +14.2% | 640,981 | 217.0 |
| Quebec | 1,908,944 | +14.1% | 448,342 | 234.9 |
| British Columbia | 1,127,346 | +14.7% | 205,795 | 182.5 |
| Alberta | 740,710 | +22.0% | 153,543 | 207.3 |
| Manitoba | 251,515 | +13.0% | 53,462 | 212.6 |
| Saskatchewan | 217,401 | +13.2% | 47,626 | 219.1 |
| Nova Scotia | 238,698 | +13.6% | 47,925 | 200.8 |
| New Brunswick | 196,181 | +13.9% | 29,561 | 150.7 |
| Newfoundland and Labrador | 134,324 | +13.4% | 21,244 | 158.2 |
| Prince Edward Island | 36,848 | +15.2% | 9,458 | 256.7 |

---

## CIHI HMDB/OMHRS — Hospitalization Data, 2024–2025
**Source file:** `dad-hmdb-childbirth-2024-2025-data-tables-en.xlsx`
**Released:** February 19, 2026
**Note:** Quebec excluded from ALC comparison — CIHI's ALC definition for Quebec is structurally narrower than for the other nine provinces (see paper Section 2.4)

### National hospitalization rate trend
| Fiscal Year | Rate per 100,000 | Avg LOS (days) |
|---|---|---|
| 2020–2021 | 7,227.7 | 5.86 |
| 2021–2022 | 7,536.6 | 6.02 |
| 2022–2023 | 7,565.8 | 6.14 |
| 2023–2024 | 7,565.3 | 6.12 |
| 2024–2025 | 7,560.3 | 6.15 |

### Top 10 inpatient diagnoses, age 65+, 2024–2025
| Rank | Diagnosis | Cases | Avg LOS |
|---|---|---|---|
| 1 | COPD and bronchitis | 68,321 | 7.3 |
| 2 | Heart failure | 61,591 | 9.7 |
| 3 | Neurocognitive disorders | 49,996 | 17.1 |
| 4 | Pneumonia | 49,060 | 7.9 |
| 5 | Osteoarthritis of the knee | 45,585 | 2.1 |
| 6 | Other medical care (palliative) | 40,829 | 9.1 |
| 7 | Acute myocardial infarction | 40,284 | 5.6 |
| 8 | Fracture of femur | 39,700 | 11.4 |
| 9 | Cerebral infarction | 32,662 | 10.4 |
| 10 | Other diseases of the urinary system (UTI) | 26,443 | 7.8 |

### ALC indicators by province, 2024–2025
| Province | Hospitalizations | Patient Days in ALC (%) |
|---|---|---|
| Prince Edward Island | 13,245 | 28.0 |
| Newfoundland and Labrador | 47,277 | 23.8 |
| New Brunswick | 71,992 | 21.2 |
| Nova Scotia | 90,121 | 20.3 |
| Ontario | 1,202,528 | 17.9 |
| Manitoba | 106,998 | 17.3 |
| British Columbia | 433,239 | 15.8 |
| Alberta | 355,251 | 15.1 |
| Saskatchewan | 119,312 | 11.9 |
| Canada (excl. Quebec) | 2,449,865 | 17.2 |

---

## Statistics Canada — Population Projections, 65+, M2 Scenario
**Source files:** `1710005701-{province}-eng.csv` (11 files: Canada + 10 provinces)
**Table:** Statistics Canada Table 17-10-0057-01 (formerly CANSIM 052-0005)
**Released:** January 27, 2026
**Scenario:** M2 (medium-growth), recommended for policy analysis
**Notebook:** `02_statcan_population_integration.ipynb`

| Province | 2025 (k) | 2030 (k) | 2035 (k) | 2040 (k) | Growth 2025–2040 |
|---|---|---|---|---|---|
| Canada | 8,108.4 | 9,318.5 | 10,053.5 | 10,557.7 | +30.2% |
| Ontario | 3,069.7 | 3,567.0 | 3,897.3 | 4,122.5 | +34.3% |
| Quebec | 1,964.4 | 2,211.0 | 2,321.8 | 2,374.4 | +20.9% |
| British Columbia | 1,167.2 | 1,336.5 | 1,449.4 | 1,528.5 | +31.0% |
| Alberta | 780.0 | 934.0 | 1,046.8 | 1,152.9 | +47.8% |
| Manitoba | 260.4 | 295.6 | 315.2 | 330.7 | +27.0% |
| Saskatchewan | 225.8 | 255.5 | 270.4 | 283.9 | +25.7% |
| Nova Scotia | 246.8 | 277.4 | 288.8 | 292.5 | +18.5% |
| New Brunswick | 202.7 | 226.9 | 236.3 | 240.2 | +18.5% |
| Newfoundland and Labrador | 138.5 | 152.8 | 160.6 | 161.7 | +16.8% |
| Prince Edward Island | 38.1 | 43.7 | 46.6 | 48.5 | +27.3% |

*Provincial totals do not sum to Canada row — the national figure includes Yukon, NWT, and Nunavut.*

---

## CLSA — Tracking Cohort (Wave 3–4), Canadians Aged 65+
**Source file:** `clsa_Baseline_StratifiedFrequencies_Amala_Jolly.xlsx`
**Obtained under:** Formal institutional data access agreement
**Notebook:** `03_clsa_analysis.py`
**Sample:** n = 21,220 (Overall Frequency column); stratified subset n = 8,830

| Condition / Indicator | N | N_Total | Prevalence (%) | Variable | Verified |
|---|---|---|---|---|---|
| Hypertension | 8,102 | 21,213 | 38.2 | CCT_HBP_TRM | ✅ |
| Back problems | 5,196 | 21,205 | 24.5 | CCT_BCKP_TRM | ✅ |
| Depression positive screen | 3,598 | 21,136 | 17.0 | DEP_DPSFD_TRM | ✅ |
| Diabetes | 3,553 | 21,230 | 16.7 | CCT_DIAB_TRM | ✅ |
| OA of knee | 3,437 | 21,183 | 16.2 | CCT_OAKNEE_TRM | ✅ |
| Asthma | 2,355 | 21,224 | 11.1 | CCT_ASTHM_TRM | ✅ |
| Heart disease | 2,186 | 21,205 | 10.3 | CCT_HEART_TRM | ✅ |
| Loneliness (agree/strongly agree) | 1,651 | 18,864 | 8.8 | ENV_FLLNLY_MCQ | ✅ |
| Anxiety | 1,560 | 21,216 | 7.4 | CCT_ANXI_TRM | ✅ |
| COPD | 1,436 | 21,222 | 6.8 | CCT_COPD_TRM | ✅ |
| Self-rated health good or better | 18,362 | 21,220 | 86.5 | GEN_HLTH_TRM | ✅ |
| Volunteering at least monthly | 8,161 | 21,211 | 38.5 | SPA_VOLUN_TRM | ✅ |
| Other activities at least monthly | 11,346 | 21,168 | 53.6 | SPA_OTACT_TRM | ✅ |

*CLSA microdata cannot be shared publicly per data access agreement terms.
All prevalence estimates computed from the Overall Frequency column of the stratified
frequency table. Raw microdata file not included in this repository.*

---

## Verification Status Summary

| Data Source | Numbers Verified | Method |
|---|---|---|
| GBD 2023 — original 7 categories | ✅ All | Run `01_gbd_trend_analysis_12category.ipynb` against raw CSV/XLSX files |
| GBD 2023 — new 5 categories | ✅ All | Same notebook, reads `IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx` |
| GBD 2023 — all-cause | ✅ All | `IHME-GBD_2023_DATA-fa7b4ff0-1.csv`, col `val` |
| Forecasting (2024–2040) | ✅ All | Run `02_forecasting_polynomial_regression_12category.ipynb` |
| CIHI Pharmaceutical Data | ✅ All | Verified against raw Excel file manually |
| CIHI HMDB/OMHRS | ✅ All | Verified against raw Excel file, released Feb 19 2026 |
| Statistics Canada M2 | ✅ All | Run `02_statcan_population_integration.ipynb` against 11 provincial CSVs |
| CLSA Tracking cohort | ✅ All | Run `03_clsa_analysis.py` against raw stratified frequency file |

---

*Last updated: June 2026*
*Contact: Amala K J, University of Windsor / SRM Institute of Science and Technology*
