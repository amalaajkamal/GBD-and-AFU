# ============================================================
# Disease Burden Among Aging Canadians — Interactive Dashboard
# Companion to: "Disease Burden Among Aging Canadians: A Multi-
# Source Analysis and Forecast Using GBD 2023, CIHI, Statistics
# Canada, and CLSA Data"
# Run: streamlit run gbd_canada_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Disease Burden Among Aging Canadians",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #4D9FE8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4D9FE8;
        border-bottom: 2px solid #4D9FE8;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background: rgba(77, 159, 232, 0.15);
        border-left: 4px solid #4D9FE8;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(220, 80, 80, 0.15);
        border-left: 4px solid #E05555;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background: rgba(50, 160, 80, 0.15);
        border-left: 4px solid #32A050;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──
DISEASE_COLORS = {
    'Neoplasms':                    '#185FA5',
    'Cardiovascular diseases':      '#993C1D',
    'Neurological disorders':       '#0F6E56',
    'Musculoskeletal disorders':    '#534AB7',
    'Chronic respiratory diseases': '#854F0B',
    'Diabetes and kidney diseases': '#A32D2D',
    'Mental disorders':             '#3B6D11',
}
DISEASES = list(DISEASE_COLORS.keys())
OBS_YEARS = [1995, 2000, 2005, 2010, 2015, 2019, 2023]

PROVINCES = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba',
             'Saskatchewan', 'Nova Scotia', 'New Brunswick',
             'Newfoundland and Labrador', 'Prince Edward Island']

PROVINCE_COLORS = {
    'Ontario':                    '#185FA5',
    'Quebec':                     '#993C1D',
    'British Columbia':           '#0F6E56',
    'Alberta':                    '#F0A500',
    'Manitoba':                   '#534AB7',
    'Saskatchewan':               '#854F0B',
    'Nova Scotia':                '#A32D2D',
    'New Brunswick':              '#3B6D11',
    'Newfoundland and Labrador':  '#6B5B95',
    'Prince Edward Island':       '#D81B60',
}

ALC_PROVINCES = [p for p in PROVINCES if p != 'Quebec']  # Quebec excluded from ALC; see paper Section 2.4


# ── DATA LOADING: GBD (optional real files, else verified fallback) ──
@st.cache_data
def load_gbd_data():
    """Optionally load real GBD CSV/XLSX files if present alongside this script."""
    file_map = {
        'Chronic respiratory diseases': 'IHME-GBD_2023_DATA-5da954e9-1_respiratory.csv',
        'Mental disorders':             'IHME-GBD_2023_DATA-4eb8814a-1_mental.csv',
        'Diabetes and kidney diseases': 'IHME-GBD_2023_DATA-dd10953b-1_diabetics.csv',
        'Musculoskeletal disorders':    'IHME-GBD_2023_DATA-100da0be-1_musculo.csv',
        'Neurological disorders':       'IHME-GBD_2023_DATA-cd6c1a03-1_neuro.csv',
        'Neoplasms':                    'IHME-GBD_2023_DATA-1c9f46d3-1_neo.csv',
        'Cardiovascular diseases':      'IHME-GBD_2023_DATA-141409a9-1_cardio.xlsx',
    }
    dfs, missing = [], []
    for disease, fname in file_map.items():
        if os.path.exists(fname):
            try:
                df = pd.read_excel(fname) if fname.endswith('.xlsx') else pd.read_csv(fname)
                dfs.append(df)
            except Exception:
                missing.append(disease)
        else:
            missing.append(disease)
    if dfs:
        return pd.concat(dfs, ignore_index=True), missing
    return None, missing


@st.cache_data
def get_gbd_fallback():
    """Verified GBD 2023 data, Canadians 60+, seven disease categories. Matches paper Tables 1-3."""
    daly_data = {
        'Neoplasms':                    [973032, 1023542, 1080966, 1163133, 1294164, 1394261, 1557828],
        'Cardiovascular diseases':      [1196259, 1144752, 1078819, 1051908, 1126587, 1200679, 1333532],
        'Neurological disorders':       [246959, 286371, 331147, 387131, 456361, 516731, 582179],
        'Musculoskeletal disorders':    [259358, 282590, 313597, 364370, 433771, 500108, 552267],
        'Chronic respiratory diseases': [219540, 239279, 256933, 282544, 334880, 374449, 415842],
        'Diabetes and kidney diseases': [172889, 207633, 243272, 257874, 282879, 322291, 374726],
        'Mental disorders':             [74447, 79391, 90221, 108909, 129190, 148278, 194075],
    }
    rate_data = {
        'Cardiovascular diseases':      [25394.7, 22312.9, 18586.5, 15338.9, 13962.4, 13109.0, 13055.7],
        'Neoplasms':                    [20655.9, 19950.3, 18623.5, 16960.8, 16039.2, 15222.5, 15251.7],
        'Neurological disorders':       [5242.5,  5581.8,  5705.2,  5645.1,  5655.9,  5641.7,  5699.7],
        'Musculoskeletal disorders':    [5505.8,  5508.1,  5402.8,  5313.2,  5375.9,  5460.2,  5406.9],
        'Chronic respiratory diseases': [4660.5,  4663.9,  4426.6,  4120.1,  4150.3,  4088.2,  4071.2],
        'Diabetes and kidney diseases': [3670.2,  4047.1,  4191.2,  3760.3,  3505.9,  3518.8,  3668.7],
        'Mental disorders':             [1580.4,  1547.4,  1554.4,  1588.1,  1601.1,  1618.9,  1900.1],
    }
    deaths_2023 = {
        'Neoplasms': 88335, 'Cardiovascular diseases': 78305,
        'Neurological disorders': 28739, 'Chronic respiratory diseases': 19734,
        'Diabetes and kidney diseases': 15384, 'Musculoskeletal disorders': 1142,
        'Mental disorders': None,
    }
    return daly_data, rate_data, deaths_2023


@st.cache_data
def get_allcause_gbd():
    """All-cause GBD 2023 totals for Canadians 60+ — the broader denominator described in
    paper Section 3.1, distinct from the seven categories analyzed in depth elsewhere."""
    return {
        1995: {'dalys': 3_948_183, 'deaths': 172_766},
        2023: {'dalys': 6_895_367, 'deaths': 280_718},
    }


@st.cache_data
def compute_forecasts(daly_data):
    """Polynomial regression (degree=2) forecasts, 2024-2040. Matches paper Table 8."""
    annual_years = np.arange(1995, 2024)
    forecast_years = np.arange(2024, 2041)
    forecasts, metrics = {}, {}
    for disease, obs_vals in daly_data.items():
        cs = CubicSpline(OBS_YEARS, obs_vals)
        annual_vals = np.maximum(cs(annual_years), 0)
        X = annual_years.reshape(-1, 1)
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, annual_vals)
        forecast = np.maximum(model.predict(forecast_years.reshape(-1, 1)), 0)
        forecasts[disease] = forecast
        growth = (forecast[-1] - obs_vals[-1]) / obs_vals[-1] * 100
        metrics[disease] = {'growth_2040': round(growth, 1), 'val_2040': forecast[-1]}
    return forecasts, forecast_years, metrics


@st.cache_data
def get_provincial_pharma():
    """CIHI Pharmaceutical Data Tool — senior population, growth, antidepressant Rx, all ten
    provinces. Matches paper Table 4."""
    return pd.DataFrame([
        {'Province': 'Ontario',                   'Senior Pop (2024)': 2954128, 'Growth % (2020-24)': 14.2, 'Antidepressant Rx (2024)': 640981, 'Rx per 1,000 Seniors': 217.0},
        {'Province': 'Quebec',                    'Senior Pop (2024)': 1908944, 'Growth % (2020-24)': 14.1, 'Antidepressant Rx (2024)': 448342, 'Rx per 1,000 Seniors': 234.9},
        {'Province': 'British Columbia',          'Senior Pop (2024)': 1127346, 'Growth % (2020-24)': 14.7, 'Antidepressant Rx (2024)': 205795, 'Rx per 1,000 Seniors': 182.5},
        {'Province': 'Alberta',                   'Senior Pop (2024)': 740710,  'Growth % (2020-24)': 22.0, 'Antidepressant Rx (2024)': 153543, 'Rx per 1,000 Seniors': 207.3},
        {'Province': 'Manitoba',                  'Senior Pop (2024)': 251515,  'Growth % (2020-24)': 13.0, 'Antidepressant Rx (2024)': 53462,  'Rx per 1,000 Seniors': 212.6},
        {'Province': 'Saskatchewan',              'Senior Pop (2024)': 217401,  'Growth % (2020-24)': 13.2, 'Antidepressant Rx (2024)': 47626,  'Rx per 1,000 Seniors': 219.1},
        {'Province': 'Nova Scotia',               'Senior Pop (2024)': 238698,  'Growth % (2020-24)': 13.6, 'Antidepressant Rx (2024)': 47925,  'Rx per 1,000 Seniors': 200.8},
        {'Province': 'New Brunswick',             'Senior Pop (2024)': 196181,  'Growth % (2020-24)': 13.9, 'Antidepressant Rx (2024)': 29561,  'Rx per 1,000 Seniors': 150.7},
        {'Province': 'Newfoundland and Labrador', 'Senior Pop (2024)': 134324,  'Growth % (2020-24)': 13.4, 'Antidepressant Rx (2024)': 21244,  'Rx per 1,000 Seniors': 158.2},
        {'Province': 'Prince Edward Island',      'Senior Pop (2024)': 36848,   'Growth % (2020-24)': 15.2, 'Antidepressant Rx (2024)': 9458,   'Rx per 1,000 Seniors': 256.7},
    ])


@st.cache_data
def get_statcan_population():
    """Statistics Canada Table 17-10-0057-01 — M2 scenario — 65+ population (thousands),
    annual 2025-2040, Canada and all ten provinces. Matches paper Table 5."""
    years = list(range(2025, 2041))
    data = {
        'Canada':                    [8108.4, 8365.3, 8610.2, 8861.3, 9100.4, 9318.5, 9498.2, 9646.7, 9784.8, 9920.2, 10053.5, 10183.0, 10288.2, 10379.6, 10466.5, 10557.7],
        'Ontario':                   [3069.7, 3171.4, 3270.4, 3373.7, 3474.0, 3567.0, 3646.2, 3712.4, 3775.0, 3836.7, 3897.3, 3956.0, 4004.5, 4046.3, 4084.4, 4122.5],
        'Quebec':                    [1964.4, 2019.8, 2070.2, 2120.8, 2167.9, 2211.0, 2243.5, 2268.2, 2288.0, 2305.2, 2321.8, 2336.3, 2345.6, 2353.2, 2362.1, 2374.4],
        'British Columbia':          [1167.2, 1202.9, 1236.7, 1273.0, 1306.9, 1336.5, 1361.4, 1382.7, 1404.5, 1427.0, 1449.4, 1470.6, 1487.1, 1501.7, 1515.2, 1528.5],
        'Alberta':                   [780.0, 812.9, 844.4, 875.9, 906.2, 934.0, 958.2, 980.2, 1002.0, 1024.3, 1046.8, 1070.3, 1091.6, 1112.0, 1132.1, 1152.9],
        'Manitoba':                  [260.4, 268.2, 275.6, 282.7, 289.5, 295.6, 300.6, 304.4, 307.9, 311.4, 315.2, 318.9, 322.3, 325.1, 327.7, 330.7],
        'Saskatchewan':              [225.8, 232.1, 238.3, 244.7, 250.6, 255.5, 259.2, 262.2, 264.9, 267.8, 270.4, 273.3, 276.0, 278.6, 281.0, 283.9],
        'Nova Scotia':               [246.8, 253.6, 260.2, 266.4, 272.3, 277.4, 280.9, 283.6, 285.5, 287.2, 288.8, 290.4, 291.4, 292.0, 292.3, 292.5],
        'New Brunswick':             [202.7, 207.9, 213.1, 218.0, 222.6, 226.9, 230.1, 232.2, 233.8, 235.1, 236.3, 237.7, 238.8, 239.4, 239.8, 240.2],
        'Newfoundland and Labrador': [138.5, 141.7, 144.7, 147.5, 150.2, 152.8, 155.0, 156.8, 158.4, 159.6, 160.6, 161.5, 162.0, 162.0, 161.9, 161.7],
        'Prince Edward Island':      [38.1, 39.3, 40.4, 41.6, 42.7, 43.7, 44.4, 45.0, 45.5, 46.0, 46.6, 47.1, 47.5, 47.9, 48.2, 48.5],
    }
    df = pd.DataFrame(data, index=years)
    df.index.name = 'Year'
    return df


@st.cache_data
def get_cihi_hosp_65plus():
    """CIHI HMDB/OMHRS 2024-2025 — top 10 inpatient hospitalizations, age 65+. Matches paper Table 6."""
    return pd.DataFrame([
        {'rank': 1,  'diagnosis': 'COPD and bronchitis',                          'n_hosp': 68321, 'avg_los': 7.3,  'category': 'Respiratory'},
        {'rank': 2,  'diagnosis': 'Heart failure',                                 'n_hosp': 61591, 'avg_los': 9.7,  'category': 'Cardiovascular'},
        {'rank': 3,  'diagnosis': 'Neurocognitive disorders',                      'n_hosp': 49996, 'avg_los': 17.1, 'category': 'Neurological'},
        {'rank': 4,  'diagnosis': 'Pneumonia',                                     'n_hosp': 49060, 'avg_los': 7.9,  'category': 'Respiratory'},
        {'rank': 5,  'diagnosis': 'Osteoarthritis of the knee',                   'n_hosp': 45585, 'avg_los': 2.1,  'category': 'Musculoskeletal'},
        {'rank': 6,  'diagnosis': 'Other medical care (palliative)',               'n_hosp': 40829, 'avg_los': 9.1,  'category': 'Other'},
        {'rank': 7,  'diagnosis': 'Acute myocardial infarction',                   'n_hosp': 40284, 'avg_los': 5.6,  'category': 'Cardiovascular'},
        {'rank': 8,  'diagnosis': 'Fracture of femur',                             'n_hosp': 39700, 'avg_los': 11.4, 'category': 'Musculoskeletal'},
        {'rank': 9,  'diagnosis': 'Cerebral infarction',                           'n_hosp': 32662, 'avg_los': 10.4, 'category': 'Neurological'},
        {'rank': 10, 'diagnosis': 'Other diseases of the urinary system (UTI)',   'n_hosp': 26443, 'avg_los': 7.8,  'category': 'Other'},
    ])


@st.cache_data
def get_national_hosp_rate_trend():
    """CIHI HMDB Table 1 — national age-sex-standardized hospitalization rate, 2020-21 to 2024-25."""
    return pd.DataFrame([
        {'Fiscal Year': '2020–2021', 'Rate per 100,000': 7227.7, 'Avg LOS (days)': 5.86},
        {'Fiscal Year': '2021–2022', 'Rate per 100,000': 7536.6, 'Avg LOS (days)': 6.02},
        {'Fiscal Year': '2022–2023', 'Rate per 100,000': 7565.8, 'Avg LOS (days)': 6.14},
        {'Fiscal Year': '2023–2024', 'Rate per 100,000': 7565.3, 'Avg LOS (days)': 6.12},
        {'Fiscal Year': '2024–2025', 'Rate per 100,000': 7560.3, 'Avg LOS (days)': 6.15},
    ])


@st.cache_data
def get_cihi_alc():
    """CIHI HMDB/OMHRS Table 7 — ALC patient-day proportions, 2024-2025, nine provinces
    (Quebec excluded; see paper Section 2.4) plus the Canada total. Matches paper Table 7."""
    rows = pd.DataFrame([
        {'Province': 'Prince Edward Island',        'Hospitalizations (2024-25)': 13245,   'Patient Days in ALC (%)': 28.0},
        {'Province': 'Newfoundland and Labrador',   'Hospitalizations (2024-25)': 47277,   'Patient Days in ALC (%)': 23.8},
        {'Province': 'New Brunswick',                'Hospitalizations (2024-25)': 71992,   'Patient Days in ALC (%)': 21.2},
        {'Province': 'Nova Scotia',                  'Hospitalizations (2024-25)': 90121,   'Patient Days in ALC (%)': 20.3},
        {'Province': 'Ontario',                      'Hospitalizations (2024-25)': 1202528, 'Patient Days in ALC (%)': 17.9},
        {'Province': 'Manitoba',                     'Hospitalizations (2024-25)': 106998,  'Patient Days in ALC (%)': 17.3},
        {'Province': 'British Columbia',             'Hospitalizations (2024-25)': 433239,  'Patient Days in ALC (%)': 15.8},
        {'Province': 'Alberta',                      'Hospitalizations (2024-25)': 355251,  'Patient Days in ALC (%)': 15.1},
        {'Province': 'Saskatchewan',                 'Hospitalizations (2024-25)': 119312,  'Patient Days in ALC (%)': 11.9},
    ])
    canada_total = pd.DataFrame([
        {'Province': 'Canada (excl. Quebec)', 'Hospitalizations (2024-25)': 2449865, 'Patient Days in ALC (%)': 17.2}
    ])
    return rows, canada_total


CLSA_FINDINGS = {
    'multimorbidity_pct': 65,
    'no_chronic_pct': 15,
    'hypertension_pct': 37,
    'arthritis_pct': 30,
    'respiratory_pct': 16,
    'par_male_cardiometabolic': 21.3,
    'par_female_musculoskeletal': 22.7,
    'recreational_uplift_pct': 15,
    'volunteer_uplift_pct': 17,
    'sustained_aging_pct': 72,
}


# ── INITIAL DATA LOADING EXECUTION ──
gbd_raw, missing_files = load_gbd_data()
daly_data, rate_data, deaths_2023 = get_gbd_fallback()
allcause = get_allcause_gbd()

if gbd_raw is not None:
    daly_pivot = gbd_raw[
        (gbd_raw['age_name'] == '60+ years') &
        (gbd_raw['metric_name'] == 'Number') &
        (gbd_raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')
    ].pivot_table(index='cause_name', columns='year', values='val')
    for disease in DISEASES:
        if disease in daly_pivot.index:
            daly_data[disease] = [daly_pivot.loc[disease, y] for y in OBS_YEARS if y in daly_pivot.columns]

forecasts, forecast_years, forecast_metrics = compute_forecasts(daly_data)
prov_pharma_df = get_provincial_pharma()
pop_df = get_statcan_population()
hosp_df = get_cihi_hosp_65plus()
hosp_rate_trend_df = get_national_hosp_rate_trend()
alc_df, alc_canada_df = get_cihi_alc()


# ── SIDEBAR INTERFACES & FILTER GLOBALS (MIGRATED UPWARD) ──
st.sidebar.markdown("## 🏥 Disease Burden Dashboard")
st.sidebar.markdown("**GBD 2023 · CIHI · Statistics Canada · CLSA**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["📊 Overall Disease Burden", "📈 Absolute DALY Trends", "📉 Age-Standardized Rate Trends",
     "🗺️ Provincial Burden & Demographics", "🔮 Forecasting Through 2040",
     "👥 Population Projections (2025–2040)", "🏥 Hospitalization Burden & ALC",
     "🔗 Integrated Multi-Source Analysis", "📋 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

# CRITICAL FIX: These filters must compute before any visual sections evaluate down-script!
selected_diseases = st.sidebar.multiselect("Select diseases", DISEASES, default=DISEASES)
filtered_diseases = [d for d in selected_diseases if d in DISEASES] or DISEASES

selected_provinces = st.sidebar.multiselect("Select provinces", PROVINCES, default=PROVINCES)
filtered_provinces = [p for p in selected_provinces if p in PROVINCES] or PROVINCES

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Files (optional)")
st.sidebar.info("If present, real GBD CSV/XLSX files in this folder will be used instead of the embedded, verified dataset.")
if missing_files:
    st.sidebar.caption(f"Using embedded data for: {', '.join(missing_files[:3])}{'…' if len(missing_files) > 3 else ''}")
else:
    st.sidebar.success("✅ Real GBD files loaded")


# ============================================================
# PAGE 1: OVERALL DISEASE BURDEN
# ============================================================
if page == "📊 Overall Disease Burden":
    st.markdown('<p class="main-title">Disease Burden Among Aging Canadians</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A multi-source analysis using GBD 2023, CIHI, Statistics Canada, and CLSA data</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total DALYs, all-cause (2023)", f"{allcause[2023]['dalys']/1e6:.2f}M",
                   f"+{(allcause[2023]['dalys']-allcause[1995]['dalys'])/allcause[1995]['dalys']*100:.1f}% since 1995")
    with col2:
        st.metric("Deaths, all-cause (2023)", f"{allcause[2023]['deaths']:,}",
                   f"+{(allcause[2023]['deaths']-allcause[1995]['deaths'])/allcause[1995]['deaths']*100:.1f}% since 1995")
    with col3:
        st.metric("Fastest growing (7 categories)", "Mental disorders", "+160.7%")
    with col4:
        st.metric("Best improvement (rate)", "Cardiovascular", "-48.6%")
    with col5:
        st.metric("Provinces analyzed", "10", "All of Canada")
    with col6:
        st.metric("Projected 2040 (7 categories)", "7.99M DALYs", "+59.5%")

    st.markdown("""
    <div class="highlight-box">
    The 6.90 million all-cause figure above spans every disease and injury category tracked by
    GBD 2023. This study analyzes seven major disease categories in depth (DALYs: 5.01 million in
    2023, 72.7% of the all-cause total) — it is this seven-category subset that powers the trend,
    rate, and forecasting analyses throughout this dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-header">Disease Burden by Category — 2023</p>', unsafe_allow_html=True)
        dalys_2023 = {d: daly_data[d][-1] for d in filtered_diseases}
        df_plot = pd.DataFrame({'Disease': list(dalys_2023.keys()), 'DALYs': list(dalys_2023.values())})
        df_plot = df_plot.sort_values('DALYs', ascending=True)
        
        fig = px.bar(df_plot, x='DALYs', y='Disease', orientation='h',
                     color='Disease', color_discrete_map=DISEASE_COLORS,
                     labels={'DALYs': 'DALYs (Absolute Count)'})
        fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside', cliponaxis=False)
        fig.update_layout(showlegend=False, height=400,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                          yaxis=dict(showgrid=False),
                          margin=dict(l=10, r=100, t=20, b=20))
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        > **Figure Notes:** DALY absolute counts represent the total combined years of healthy life lost due to premature 
        > mortality and disability. Subsetting down to these seven key chronic disease frameworks captures **72.7%** > ($5.01\text{M}$ of $6.90\text{M}$) of all-cause aging health burdens across Canada in 2023.
        """)

    with col_right:
        st.markdown('<p class="section-header">Key Research Insights</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: rgba(220, 80, 80, 0.08); border-left: 5px solid #A32D2D; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
            <strong style="color: #A32D2D;">⚠️ Mental Health Velocity:</strong><br>
            Mental disorders showed the highest absolute DALY growth (+160.7%) and an alarming <strong>+20.2% increase in age-standardized rates</strong>, signifying a true structural escalation beyond raw population growth.
        </div>
        <div style="background-color: rgba(240, 165, 0, 0.08); border-left: 5px solid #F0A500; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
            <strong style="color: #B57C00;">⚠️ Geographic Strain Patterns:</strong><br>
            <strong>Alberta</strong> exhibits the highest senior expansion velocity (+22.0%), while <strong>Prince Edward Island</strong> registers the heaviest antidepressant volume and Alternate Level of Care (ALC) bed dependency.
        </div>
        <div style="background-color: rgba(15, 110, 86, 0.08); border-left: 5px solid #0F6E56; padding: 12px; border-radius: 4px; margin-bottom: 15px;">
            <strong style="color: #0F6E56;">✅ Cardiovascular Policy Success:</strong><br>
            Age-standardized rates contracted by <strong>-48.6%</strong>, demonstrating the massive efficacy of multi-decade vascular preventative strategies across the country.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">Proportional Share of Analyzed Burden (2023)</p>', unsafe_allow_html=True)
        dalys_pie = {d: daly_data[d][-1] for d in DISEASES}
        
        fig_donut = px.pie(values=list(dalys_pie.values()), names=list(dalys_pie.keys()),
                           color=list(dalys_pie.keys()), color_discrete_map=DISEASE_COLORS, hole=0.5)
        fig_donut.update_layout(showlegend=True, 
                               legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
                               height=280, margin=dict(l=10, r=10, t=10, b=10))
        fig_donut.update_traces(textposition='inside', textinfo='percent', textfont_size=11)
        st.plotly_chart(fig_donut, width='stretch')


# ============================================================
# PAGE 2: ABSOLUTE DALY TRENDS
# ============================================================
elif page == "📈 Absolute DALY Trends":
    st.markdown('<p class="main-title">Absolute DALY Trends, 1995–2023</p>', unsafe_allow_html=True)

    metric_type = st.radio("View metric as:", ["Absolute numbers", "% Growth from 1995 baseline"], horizontal=True)

    fig = go.Figure()
    for disease in filtered_diseases:
        vals = daly_data[disease]
        y_vals = vals if metric_type == "Absolute numbers" else [(v - vals[0]) / vals[0] * 100 for v in vals]
        fig.add_trace(go.Scatter(
            x=OBS_YEARS, y=y_vals, mode='lines+markers', name=disease,
            line=dict(color=DISEASE_COLORS[disease], width=2.5), marker=dict(size=7)
        ))

    ylabel = "DALYs (number)" if metric_type == "Absolute numbers" else "% Growth from 1995 baseline"
    fig.update_layout(height=450, xaxis_title="Year", yaxis_title=ylabel,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                       yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                       legend=dict(orientation='v', x=1.01, y=1),
                       margin=dict(l=0, r=150, t=20, b=40))
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown('<p class="section-header">Year-by-Year DALY Table</p>', unsafe_allow_html=True)
    df_table = pd.DataFrame({d: daly_data[d] for d in filtered_diseases}, index=OBS_YEARS)
    df_table.index.name = 'Year'
    df_table['% Change 1995-2023'] = df_table.apply(
        lambda col: f"+{(col.iloc[-1]-col.iloc[0])/col.iloc[0]*100:.1f}%" if col.name != '% Change 1995-2023' else '', axis=0
    )
    st.dataframe(df_table.style.format("{:,.0f}"), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="warning-box"><b>Fastest growing</b><br>Mental disorders: +160.7%<br>Neurological: +135.7%</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="highlight-box"><b>Moderate growth</b><br>Diabetes: +116.7%<br>Musculoskeletal: +112.9%</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="success-box"><b>Slowest growth</b><br>Cardiovascular: +11.5%<br>(Interventions effective)</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 3: AGE-STANDARDIZED RATE TRENDS
# ============================================================
elif page == "📉 Age-Standardized Rate Trends":
    st.markdown('<p class="main-title">Age-Standardized Rate Trends, 1995–2023</p>', unsafe_allow_html=True)
    st.markdown("Controls for population growth — shows the true burden trajectory")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">Rate trends, 1995–2023</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for disease in filtered_diseases:
            if disease in rate_data:
                fig.add_trace(go.Scatter(
                    x=OBS_YEARS, y=rate_data[disease], mode='lines+markers',
                    name=disease, line=dict(color=DISEASE_COLORS[disease], width=2.5),
                    marker=dict(size=6)
                ))
        fig.update_layout(height=380, xaxis_title="Year",
                          yaxis_title="Age-standardized rate per 100,000",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                          yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                          legend=dict(orientation='v', x=1.01, y=1),
                          margin=dict(l=0, r=150, t=10, b=40))
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.markdown('<p class="section-header">% Change in rate, 1995→2023</p>', unsafe_allow_html=True)
        rate_changes = {d: (rate_data[d][-1] - rate_data[d][0]) / rate_data[d][0] * 100
                         for d in filtered_diseases if d in rate_data}
        df_rate = pd.DataFrame({'Disease': list(rate_changes.keys()), 'Rate Change (%)': list(rate_changes.values())})
        df_rate = df_rate.sort_values('Rate Change (%)')
        colors = ['#0F6E56' if v < 0 else '#A32D2D' for v in df_rate['Rate Change (%)']]
        fig2 = px.bar(df_rate, x='Rate Change (%)', y='Disease', orientation='h',
                      color='Disease', color_discrete_map={d: c for d, c in zip(df_rate['Disease'], colors)})
        fig2.add_vline(x=0, line_color='gray', line_width=1)
        fig2.update_layout(showlegend=False, height=380,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', ticksuffix='%'),
                           yaxis=dict(showgrid=False),
                           margin=dict(l=0, r=20, t=10, b=40))
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    st.markdown('<p class="section-header">Age-Standardized Rate Table</p>', unsafe_allow_html=True)
    df_rates_table = pd.DataFrame(rate_data, index=OBS_YEARS).T
    df_rates_table.index.name = 'Disease'
    df_rates_table['Rate Change'] = df_rates_table.apply(
        lambda row: f"{(row.iloc[-1]-row.iloc[0])/row.iloc[0]*100:+.1f}%", axis=1
    )
    df_rates_table['Interpretation'] = df_rates_table['Rate Change'].apply(
        lambda x: '↓ Declining (interventions effective)' if float(x[:-1]) < -10
        else '↓ Slightly declining' if float(x[:-1]) < 0
        else '→ Stable' if float(x[:-1]) < 5
        else '↑ Rising (genuine increase) ⚠️'
    )
    st.dataframe(df_rates_table.style.format({c: "{:.1f}" for c in OBS_YEARS}), use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Key finding:</b> Cardiovascular disease rate fell -48.6% — the strongest evidence that targeted
    intervention works. Mental disorders rate rose +20.2% even after controlling for population growth
    — a genuine worsening that existing interventions have not yet addressed.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 4: PROVINCIAL BURDEN & DEMOGRAPHICS
# ============================================================
elif page == "🗺️ Provincial Burden & Demographics":
    st.markdown('<p class="main-title">Provincial Senior Population and Mental Health Burden Proxy</p>', unsafe_allow_html=True)
    st.markdown("CIHI Pharmaceutical Data Tool, 2020–2024 | All ten Canadian provinces")

    df_prov = prov_pharma_df[prov_pharma_df['Province'].isin(filtered_provinces)].copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Largest senior population", "Ontario", "2.95M seniors")
    with col2:
        st.metric("Fastest-growing", "Alberta", "+22.0% since 2020")
    with col3:
        st.metric("Highest antidepressant Rx", "Prince Edward Island", "256.7 per 1,000")
    with col4:
        st.metric("Lowest antidepressant Rx", "New Brunswick", "150.7 per 1,000")

    st.markdown("---")
    metric_choice = st.selectbox("Select metric to visualize:", [
        'Senior Pop (2024)', 'Growth % (2020-24)', 'Antidepressant Rx (2024)', 'Rx per 1,000 Seniors'
    ])

    col_left, col_right = st.columns(2)

    with col_left:
        df_sorted = df_prov.sort_values(metric_choice, ascending=True)
        fig = px.bar(df_sorted, x=metric_choice, y='Province', orientation='h',
                     color='Province', color_discrete_map=PROVINCE_COLORS, text=metric_choice)
        fig.update_traces(texttemplate='%{text:,.1f}', textposition='outside')
        fig.update_layout(showlegend=False, height=420, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=60, t=10, b=40),
                          yaxis=dict(showgrid=False), xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'))
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.markdown("**Two pressure points: growth velocity vs. mental health burden**")
        fig2 = px.scatter(df_prov, x='Growth % (2020-24)', y='Rx per 1,000 Seniors',
                          color='Province', color_discrete_map=PROVINCE_COLORS,
                          size='Senior Pop (2024)', text='Province',
                          labels={'Growth % (2020-24)': 'Senior population growth, 2020–2024 (%)',
                                  'Rx per 1,000 Seniors': 'Antidepressant Rx per 1,000 seniors'})
        fig2.update_traces(textposition='top center')
        fig2.update_layout(showlegend=False, height=420, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=20, t=10, b=40))
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
