# GBD 2023 & Age-Friendly University (AFU) Canada Analysis

[![Streamlit App](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://gbd-and-afu-faje8xk4uapsox4nuqmbnz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Misalignment Between Elderly Disease Burden and Age-Friendly University Program Priorities in Canada: A GBD 2023 Analysis**

This repository contains the full analysis pipeline, data processing notebooks, and an interactive dashboard for a study examining whether Canadian Age-Friendly University (AFU) program priorities align with the health conditions causing the greatest and fastest-growing burden among Canadians aged 60+.

🔗 **Live Dashboard:** [gbd-and-afu-faje8xk4uapsox4nuqmbnz.streamlit.app](https://gbd-and-afu-faje8xk4uapsox4nuqmbnz.streamlit.app/)

---

## 📋 Overview

Canada's population aged 65+ is the fastest-growing age group, projected to keep growing through 2073. This study combines four independent data sources to evaluate whether Canada's 11 AFU institutions are addressing the right health priorities:

| Data Source | What It Provides |
|---|---|
| **GBD 2023** (IHME) | DALYs and age-standardized rates for 7 disease categories, 1995–2023, Canadians 60+ |
| **Statistics Canada** | Projected 65+ population by province, 2025–2040 (Table 17-10-0057-01) |
| **CIHI** | Hospitalization burden, length of stay, and Alternate Level of Care (ALC) days, 2024–2025 |
| **CLSA** | Individual-level corroboration of chronic disease prevalence and social participation outcomes |

---

## 🔑 Key Findings

- Total DALYs among Canadians 60+ grew **74.6%** from 1995–2023 (3.95M → 6.90M)
- **Mental disorders** show the steepest growth: +160.7% absolute, +20.2% age-standardized rate — yet receive the **least AFU program attention** (Critical Gap)
- **Cardiovascular disease** rates fell **48.6%** — strong evidence that sustained intervention works
- **Alberta** has the fastest-growing 65+ population (+22.1% since 2020, projected **+47.8% by 2040**) but only **1 AFU institution**
- **COPD/bronchitis** is the #1 hospitalization cause for Canadians 65+ (68,321 cases, 2024–25) despite near-zero AFU respiratory programming
- Forecasts project total DALYs reaching **7.99M by 2040** (+59.5%)

---

## 📁 Repository Structure

```
GBD-and-AFU/
├── app.py                              # Main Streamlit dashboard
├── .streamlit/                         # Dashboard configuration
├── data/
│   ├── gbd/                             # GBD 2023 raw data (CSV/XLSX)
│   ├── statcan/                         # Statistics Canada population projections
│   ├── cihi/                            # CIHI hospitalization & pharmaceutical data
│   └── afu/                             # AFU program audit data
├── notebooks/
│   ├── GBD_AFU_Trend_Analysis.ipynb     # Main GBD trend & forecasting analysis
│   ├── statcan_population_integration.ipynb
│   ├── cihi_hospitalization_integration.ipynb
│   └── afu_program_audit.ipynb
├── figures/                             # Generated figures (PNG)
├── output/                              # Exported CSV results
└── requirements.txt
```

---

## 🖥️ Dashboard

The interactive Streamlit dashboard provides 10 analytical views:

1. **Overview** — Disease burden summary, 2023
2. **Trend Analysis** — DALY trends, 1995–2023
3. **Rate Analysis** — Age-standardized rates (controls for population growth)
4. **Provincial Analysis** — AFU coverage vs. elderly burden by province
5. **AFU Alignment** — Gap analysis between burden and program priorities
6. **Forecasting 2040** — Polynomial regression projections
7. **Population Projections** — Statistics Canada 65+ projections, 2025–2040
8. **Hospitalization Burden** — CIHI top diagnoses, length of stay, ALC days
9. **Integrated Analysis** — Combined view across all four data sources
10. **Data Explorer** — Browse and download underlying data

### Run locally

```bash
git clone https://github.com/amalaajkamal/GBD-and-AFU.git
cd GBD-and-AFU
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Methodology

- **Trend analysis**: Absolute DALYs and age-standardized rates (per 100,000) for 7 disease categories, 1995–2023
- **Forecasting**: Cubic spline interpolation to annual resolution, polynomial regression (degree=2) for 2024–2040 projections (R² > 0.96, MAPE < 3.1%)
- **Provincial analysis**: AFU institution-to-senior-population ratios across Ontario, British Columbia, Alberta, and Manitoba
- **AFU program mapping**: Systematic review of all 11 Canadian AFU member institution websites, coded against a 7-category disease focus taxonomy

Full methodology is described in the accompanying manuscript (see `/manuscript`).

---

## 📚 Data Sources & Citations

- **GBD 2023**: Institute for Health Metrics and Evaluation (IHME). [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/)
- **Statistics Canada**: Table 17-10-0057-01, *Projected population, by projection scenario, age and gender*
- **CIHI**: [Hospital Morbidity Database (HMDB)](https://www.cihi.ca), Pharmaceutical Data Tool
- **CLSA**: Raina et al. (2019), *Cohort Profile: The Canadian Longitudinal Study on Aging*, IJE
- **AFU Program Data**: Johnson & Natarajan (2026), *A Review of the Age-Friendly University Global Network in Canada*, Educational Gerontology

---

## 👥 Authors

- **Amala K J ** — PhD Candidate, SRM Institute of Science and Technology / Mitacs GRA, University of Windsor
- **Dr. Shanthi Johnson** — Vice President, Research and Innovation, University of Windsor
- **Dr. Parthiban Natarajan** — Office of VP-Research and Innovation, University of Windsor
- **Dr. D. Rajeswari ** — Professor, SRM Institute of Science and Technology / Mitacs GRA, University of Windsor
- 

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Data from GBD 2023, Statistics Canada, and CIHI are subject to their respective terms of use.

---

## 🙏 Acknowledgments

This research was conducted as part of doctoral and Mitacs Graduate Research Assistant work at the University of Windsor, in collaboration with SRM Institute of Science and Technology.
