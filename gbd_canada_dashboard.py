# ============================================================
# Disease Burden Among Aging Canadians — Interactive Dashboard
# Companion to: "Disease Burden Among Aging Canadians: A Multi-
# Source Analysis and Forecast Using GBD 2023, CIHI, Statistics
# Canada, and CLSA Data"
#
# Updated for the 12-category NCD scope (Non-Communicable Diseases
# Level 2 classification), extended from the original 7 categories
# per supervisor direction.
#
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Hide Streamlit default header ── */
    header[data-testid="stHeader"] {
        background: rgba(15,27,45,0) !important;
        height: 0rem !important;
        min-height: 0rem !important;
    }
    header[data-testid="stHeader"] * {
        display: none !important;
    }
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #0F1B2D 0%, #1A2942 50%, #0F1B2D 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #162035 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #CBD5E0 !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0rem;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.28rem 0.5rem;
        font-size: 0.84rem;
        line-height: 1.3;
        color: #A0AEC0 !important;
        border-radius: 4px;
        transition: background 0.15s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 179, 237, 0.08);
        color: #63B3ED !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        margin-bottom: 0.2rem;
        font-size: 0.84rem;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        font-size: 0.95rem;
        color: #63B3ED !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.5rem 0;
        border-color: rgba(99, 179, 237, 0.15);
    }
    section[data-testid="stSidebar"] .stMultiSelect label {
        font-size: 0.82rem;
        margin-bottom: 0.1rem;
        color: #A0AEC0 !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        background: rgba(15, 27, 45, 0.0);
        color: #E2E8F0;
        padding-top: 1.5rem;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* ── Typography ── */
    .main-title {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63B3ED 0%, #4FD1C5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #718096;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #63B3ED;
        border-bottom: 1px solid rgba(99, 179, 237, 0.3);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        letter-spacing: 0.01em;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(26, 41, 66, 0.9) 0%, rgba(22, 32, 53, 0.9) 100%);
        border: 1px solid rgba(99, 179, 237, 0.15);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] label {
        color: #718096 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E2E8F0 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* ── Callout boxes ── */
    .highlight-box {
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.08) 0%, rgba(79, 209, 197, 0.05) 100%);
        border-left: 3px solid #63B3ED;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #CBD5E0;
        font-size: 0.88rem;
        line-height: 1.6;
        height: 100%;
        text-align: left;
        vertical-align: top;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(252, 129, 74, 0.08) 0%, rgba(229, 62, 62, 0.05) 100%);
        border-left: 3px solid #FC814A;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #CBD5E0;
        font-size: 0.88rem;
        line-height: 1.6;
        height: 100%;
        text-align: left;
        vertical-align: top;
    }
    .success-box {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.08) 0%, rgba(56, 178, 172, 0.05) 100%);
        border-left: 3px solid #48BB78;
        padding: 0.85rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #CBD5E0;
        font-size: 0.88rem;
        line-height: 1.6;
        height: 100%;
        text-align: left;
        vertical-align: top;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        border: 1px solid rgba(99, 179, 237, 0.12);
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(15,27,45,0.95) !important;
    }
    .stDataFrame table {
        background: rgba(15,27,45,0.95) !important;
        color: #CBD5E0 !important;
    }
    .stDataFrame th {
        background: rgba(26,41,66,0.95) !important;
        color: #63B3ED !important;
        border-bottom: 1px solid rgba(99,179,237,0.2) !important;
    }
    .stDataFrame td {
        background: rgba(15,27,45,0.95) !important;
        color: #CBD5E0 !important;
        border-color: rgba(99,179,237,0.08) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 27, 45, 0.5);
        border-radius: 8px;
        padding: 6px 8px;
        gap: 8px;
        border: 1px solid rgba(99,179,237,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(99,179,237,0.04);
        color: #718096;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 8px 18px;
        border: 1px solid rgba(99,179,237,0.08);
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #A0AEC0;
        background: rgba(99,179,237,0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1A3A5C 0%, #1A4A5C 100%) !important;
        color: #63B3ED !important;
        border-color: rgba(99,179,237,0.3) !important;
    }

    /* ── Info / warning banners ── */
    .stInfo {
        background: rgba(99, 179, 237, 0.08);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 8px;
        color: #CBD5E0;
    }

    /* ── Dividers ── */
    hr {
        border-color: rgba(99, 179, 237, 0.1);
    }

    /* ── General text ── */
    p, li, span {
        color: #CBD5E0;
    }
    h1, h2, h3, h4 {
        color: #E2E8F0;
    }
    .metric-card-custom {
        background: linear-gradient(135deg, rgba(26,41,66,0.95) 0%, rgba(22,32,53,0.95) 100%);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .metric-card-custom .mc-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #718096;
        margin: 0;
    }
    .metric-card-custom .mc-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card-custom .mc-sublabel {
        font-size: 0.78rem;
        color: #718096;
        margin: 0;
    }
    .metric-card-custom .mc-delta-pos {
        font-size: 0.82rem;
        color: #48BB78;
        font-weight: 600;
        margin: 0;
    }
    .metric-card-custom .mc-delta-neg {
        font-size: 0.82rem;
        color: #FC814A;
        font-weight: 600;
        margin: 0;
    }
    /* ── Selectbox / dropdown dark styling ── */
    .stSelectbox > div > div {
        background: rgba(26,41,66,0.95) !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    .stSelectbox label {
        color: #CBD5E0 !important;
        font-size: 0.9rem !important;
    }

    /* ── Multiselect dark styling ── */
    .stMultiSelect > div > div {
        background: rgba(15,27,45,0.95) !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 8px !important;
        min-height: 42px !important;
    }
    .stMultiSelect label {
        color: #A0AEC0 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    /* Selected tag pills */
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, #1A3A5C 0%, #1A4A5C 100%) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 6px !important;
        color: #63B3ED !important;
        font-size: 0.78rem !important;
        padding: 2px 8px !important;
        max-width: 150px !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #63B3ED !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    /* Tag close button */
    .stMultiSelect [data-baseweb="tag"] [role="presentation"] {
        color: rgba(99,179,237,0.6) !important;
    }
    /* Dropdown list background */
    .stMultiSelect [data-baseweb="popover"] {
        background: rgba(15,27,45,0.98) !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
        border-radius: 8px !important;
    }
    .stMultiSelect [role="option"] {
        background: rgba(15,27,45,0.98) !important;
        color: #CBD5E0 !important;
        font-size: 0.85rem !important;
    }
    .stMultiSelect [role="option"]:hover {
        background: rgba(26,41,66,0.95) !important;
        color: #63B3ED !important;
    }
    /* Input text inside multiselect */
    .stMultiSelect input {
        color: #CBD5E0 !important;
        background: transparent !important;
    }

    .stCaption {
        color: #718096 !important;
        font-size: 0.78rem !important;
    }

    /* ── Section dividers ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(99,179,237,0.12);
        margin: 1.8rem 0 1.2rem 0;
    }
    .section-plane {
        background: rgba(99,179,237,0.03);
        border: 1px solid rgba(99,179,237,0.10);
        border-radius: 10px;
        padding: 1.2rem 1.2rem 0.5rem 1.2rem;
        margin-bottom: 1.2rem;
    }

</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──
# 12-category palette — same hex values used in the paper's rebuilt Figures 1-4,
# so colors are consistent between the manuscript and this dashboard.
DISEASE_COLORS = {
    'Neoplasms':                        '#185FA5',
    'Cardiovascular diseases':          '#993C1D',
    'Neurological disorders':           '#0F6E56',
    'Musculoskeletal disorders':        '#534AB7',
    'Chronic respiratory diseases':     '#854F0B',
    'Diabetes and kidney diseases':     '#A32D2D',
    'Digestive diseases':               '#3B6D11',
    'Sense organ diseases':             '#6B5B95',
    'Other non-communicable diseases':  '#B8742E',
    'Mental disorders':                 '#1E7A8C',
    'Substance use disorders':          '#D81B60',
    'Skin and subcutaneous diseases':   '#7A8450',
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
@st.cache_data(show_spinner=False, ttl=3600)
def load_gbd_data():
    """Optionally load real GBD CSV/XLSX files if present alongside this script."""
    file_map = {
        'Chronic respiratory diseases':     'IHME-GBD_2023_DATA-5da954e9-1_respiratory.csv',
        'Mental disorders':                 'IHME-GBD_2023_DATA-4eb8814a-1_mental.csv',
        'Diabetes and kidney diseases':     'IHME-GBD_2023_DATA-dd10953b-1_diabetics.csv',
        'Musculoskeletal disorders':        'IHME-GBD_2023_DATA-100da0be-1_musculo.csv',
        'Neurological disorders':           'IHME-GBD_2023_DATA-cd6c1a03-1_neuro.csv',
        'Neoplasms':                        'IHME-GBD_2023_DATA-1c9f46d3-1_neo.csv',
        'Cardiovascular diseases':          'IHME-GBD_2023_DATA-141409a9-1_cardio.xlsx',
        # 5 categories added for the 12-category extension — all loaded from one combined export
        'Digestive diseases':               'IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx',
        'Substance use disorders':          'IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx',
        'Skin and subcutaneous diseases':   'IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx',
        'Sense organ diseases':             'IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx',
        'Other non-communicable diseases':  'IHME-GBD_2023_DATA-13912a7c-1_new_level2_csv.xlsx',
    }
    dfs, missing = [], []
    seen_files = set()
    for disease, fname in file_map.items():
        if fname in seen_files:
            continue
        if os.path.exists(fname):
            try:
                df = pd.read_excel(fname) if fname.endswith('.xlsx') else pd.read_csv(fname)
                dfs.append(df)
                seen_files.add(fname)
            except Exception:
                missing.append(disease)
        else:
            missing.append(disease)
    if dfs:
        return pd.concat(dfs, ignore_index=True), missing
    return None, missing


@st.cache_data(show_spinner=False, ttl=3600)
def get_gbd_fallback():
    """Verified GBD 2023 data, Canadians 60+, twelve NCD Level 2 categories.
    Matches the paper's Tables 1-3 (12-category extension). rate_data holds TRUE
    age-standardized rates (WHO World Standard Population, direct method) -- not
    crude 60+ rates; see paper Section 2.2 / Table 3."""
    daly_data = {
        'Neoplasms':                        [973032, 1023542, 1080966, 1163133, 1294164, 1394261, 1557828],
        'Cardiovascular diseases':          [1196259, 1144752, 1078819, 1051908, 1126587, 1200679, 1333532],
        'Neurological disorders':           [246959, 286371, 331147, 387131, 456361, 516731, 582179],
        'Musculoskeletal disorders':        [259358, 282590, 313597, 364370, 433771, 500108, 552267],
        'Chronic respiratory diseases':     [219540, 239279, 256933, 282544, 334880, 374449, 415842],
        'Diabetes and kidney diseases':     [172889, 207633, 243272, 257874, 282879, 322291, 374726],
        'Digestive diseases':               [129061, 135067, 147968, 165586, 196872, 226955, 274238],
        'Sense organ diseases':             [126257, 139698, 161003, 174447, 202071, 242485, 274025],
        'Other non-communicable diseases':  [88070, 100156, 124373, 145828, 159825, 180552, 207574],
        'Mental disorders':                 [74447, 79391, 90221, 108909, 129190, 148278, 194075],
        'Substance use disorders':          [19119, 19818, 22299, 28417, 38850, 48394, 61873],
        'Skin and subcutaneous diseases':   [18263, 21343, 24451, 29243, 35866, 41111, 48238],
    }
    rate_data = {
        'Cardiovascular diseases':          [23274.5, 19571.0, 15973.9, 13251.2, 12134.9, 11433.8, 11409.2],
        'Neoplasms':                        [20190.7, 19249.0, 17923.7, 16395.3, 15467.4, 14555.1, 14499.3],
        'Neurological disorders':           [4671.4, 4747.2, 4721.9, 4705.9, 4779.4, 4791.7, 4843.8],
        'Musculoskeletal disorders':        [5495.4, 5506.8, 5435.2, 5376.9, 5439.0, 5513.8, 5448.4],
        'Sense organ diseases':             [2511.5, 2488.5, 2519.1, 2353.1, 2330.7, 2462.9, 2476.6],
        'Other non-communicable diseases':  [1813.1, 1859.6, 2023.4, 2014.8, 1874.8, 1868.3, 1920.0],
        'Digestive diseases':               [2617.3, 2453.3, 2357.5, 2248.7, 2286.5, 2320.2, 2522.2],
        'Chronic respiratory diseases':     [4314.3, 4169.9, 3925.5, 3721.7, 3792.9, 3743.1, 3720.7],
        'Diabetes and kidney diseases':     [3437.1, 3673.1, 3763.0, 3407.1, 3198.6, 3206.8, 3333.5],
        'Mental disorders':                 [1613.0, 1593.3, 1600.7, 1622.6, 1632.3, 1651.9, 1942.0],
        'Substance use disorders':          [420.3, 406.5, 405.4, 431.1, 500.7, 554.5, 649.9],
        'Skin and subcutaneous diseases':   [376.1, 397.2, 400.9, 408.0, 424.1, 427.6, 448.8],
    }
    deaths_2023 = {
        'Neoplasms': 88335, 'Cardiovascular diseases': 78305,
        'Neurological disorders': 28739, 'Chronic respiratory diseases': 19734,
        'Diabetes and kidney diseases': 15384, 'Musculoskeletal disorders': 1142,
        'Digestive diseases': 13783, 'Other non-communicable diseases': 5904,
        'Substance use disorders': 1684, 'Skin and subcutaneous diseases': 998,
        'Mental disorders': None,     # GBD reports no death data for this primarily-disability category
        'Sense organ diseases': None,  # GBD reports no death data for this primarily-disability category
    }
    return daly_data, rate_data, deaths_2023


@st.cache_data(show_spinner=False)
def get_allcause_gbd():
    """All-cause GBD 2023 totals for Canadians 60+ — the broader denominator described in
    paper Section 3.1, distinct from the twelve categories analyzed in depth elsewhere."""
    return {
        1995: {'dalys': 3_948_183, 'deaths': 172_766},
        2023: {'dalys': 6_895_367, 'deaths': 280_718},
    }


@st.cache_data(show_spinner=False, ttl=3600)
def compute_forecasts(daly_data):
    """Polynomial regression (degree=2) forecasts, 2024-2040. Matches paper Table 8
    (12-category extension)."""
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


@st.cache_data(show_spinner=False)
def get_provincial_pharma():
    """CIHI Pharmaceutical Data Tool — senior population, growth, antidepressant Rx, all ten
    provinces. Matches paper Table 4. (Unaffected by the 7->12 disease-category expansion.)"""
    return pd.DataFrame([
        {'Province': 'Ontario',                   'Senior Pop (2024)': 2954128, 'Growth % (2020-24)': 14.2, 'Antidepressant Rx (2024)': 640981, 'Rx per 1,000 Seniors': 217.0},
        {'Province': 'Quebec',                     'Senior Pop (2024)': 1908944, 'Growth % (2020-24)': 14.1, 'Antidepressant Rx (2024)': 448342, 'Rx per 1,000 Seniors': 234.9},
        {'Province': 'British Columbia',           'Senior Pop (2024)': 1127346, 'Growth % (2020-24)': 14.7, 'Antidepressant Rx (2024)': 205795, 'Rx per 1,000 Seniors': 182.5},
        {'Province': 'Alberta',                    'Senior Pop (2024)': 740710,  'Growth % (2020-24)': 22.0, 'Antidepressant Rx (2024)': 153543, 'Rx per 1,000 Seniors': 207.3},
        {'Province': 'Manitoba',                   'Senior Pop (2024)': 251515,  'Growth % (2020-24)': 13.0, 'Antidepressant Rx (2024)': 53462,  'Rx per 1,000 Seniors': 212.6},
        {'Province': 'Saskatchewan',                'Senior Pop (2024)': 217401,  'Growth % (2020-24)': 13.2, 'Antidepressant Rx (2024)': 47626,  'Rx per 1,000 Seniors': 219.1},
        {'Province': 'Nova Scotia',                 'Senior Pop (2024)': 238698,  'Growth % (2020-24)': 13.6, 'Antidepressant Rx (2024)': 47925,  'Rx per 1,000 Seniors': 200.8},
        {'Province': 'New Brunswick',               'Senior Pop (2024)': 196181,  'Growth % (2020-24)': 13.9, 'Antidepressant Rx (2024)': 29561,  'Rx per 1,000 Seniors': 150.7},
        {'Province': 'Newfoundland and Labrador',   'Senior Pop (2024)': 134324,  'Growth % (2020-24)': 13.4, 'Antidepressant Rx (2024)': 21244,  'Rx per 1,000 Seniors': 158.2},
        {'Province': 'Prince Edward Island',        'Senior Pop (2024)': 36848,   'Growth % (2020-24)': 15.2, 'Antidepressant Rx (2024)': 9458,   'Rx per 1,000 Seniors': 256.7},
    ])


@st.cache_data(show_spinner=False)
def get_statcan_population():
    """Statistics Canada Table 17-10-0057-01 — M2 scenario — 65+ population (thousands),
    annual 2025-2040, Canada and all ten provinces. Matches paper Table 5.
    (Unaffected by the 7->12 disease-category expansion.)"""
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


@st.cache_data(show_spinner=False)
def get_cihi_hosp_65plus():
    """CIHI HMDB/OMHRS 2024-2025 — top 10 inpatient hospitalizations, age 65+. Matches paper
    Table 6. (Unaffected by the 7->12 disease-category expansion.)"""
    return pd.DataFrame([
        {'rank': 1,  'diagnosis': 'COPD and bronchitis',                          'n_hosp': 68321, 'avg_los': 7.3,  'category': 'Respiratory'},
        {'rank': 2,  'diagnosis': 'Heart failure',                                'n_hosp': 61591, 'avg_los': 9.7,  'category': 'Cardiovascular'},
        {'rank': 3,  'diagnosis': 'Neurocognitive disorders',                     'n_hosp': 49996, 'avg_los': 17.1, 'category': 'Neurological'},
        {'rank': 4,  'diagnosis': 'Pneumonia',                                    'n_hosp': 49060, 'avg_los': 7.9,  'category': 'Respiratory'},
        {'rank': 5,  'diagnosis': 'Osteoarthritis of the knee',                   'n_hosp': 45585, 'avg_los': 2.1,  'category': 'Musculoskeletal'},
        {'rank': 6,  'diagnosis': 'Other medical care (palliative)',              'n_hosp': 40829, 'avg_los': 9.1,  'category': 'Other'},
        {'rank': 7,  'diagnosis': 'Acute myocardial infarction',                  'n_hosp': 40284, 'avg_los': 5.6,  'category': 'Cardiovascular'},
        {'rank': 8,  'diagnosis': 'Fracture of femur',                            'n_hosp': 39700, 'avg_los': 11.4, 'category': 'Musculoskeletal'},
        {'rank': 9,  'diagnosis': 'Cerebral infarction',                          'n_hosp': 32662, 'avg_los': 10.4, 'category': 'Neurological'},
        {'rank': 10, 'diagnosis': 'Other diseases of the urinary system (UTI)',   'n_hosp': 26443, 'avg_los': 7.8,  'category': 'Other'},
    ])


@st.cache_data(show_spinner=False)
def get_national_hosp_rate_trend():
    """CIHI HMDB Table 1 — national age-sex-standardized hospitalization rate, 2020-21 to
    2024-25. (Unaffected by the 7->12 disease-category expansion.)"""
    return pd.DataFrame([
        {'Fiscal Year': '2020–2021', 'Rate per 100,000': 7227.7, 'Avg LOS (days)': 5.86},
        {'Fiscal Year': '2021–2022', 'Rate per 100,000': 7536.6, 'Avg LOS (days)': 6.02},
        {'Fiscal Year': '2022–2023', 'Rate per 100,000': 7565.8, 'Avg LOS (days)': 6.14},
        {'Fiscal Year': '2023–2024', 'Rate per 100,000': 7565.3, 'Avg LOS (days)': 6.12},
        {'Fiscal Year': '2024–2025', 'Rate per 100,000': 7560.3, 'Avg LOS (days)': 6.15},
    ])


@st.cache_data(show_spinner=False)
def get_cihi_alc():
    """CIHI HMDB/OMHRS Table 7 — ALC patient-day proportions, 2024-2025, nine provinces
    (Quebec excluded; see paper Section 2.4) plus the Canada total. Matches paper Table 7.
    (Unaffected by the 7->12 disease-category expansion.)"""
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
    # All values computed directly from CLSA Tracking cohort (wave 3-4),
    # n=21,220 participants aged 65+. Source: clsa_Baseline_StratifiedFrequencies_Amala_Jolly.xlsx
    # Verified by 03_clsa_analysis.py
    'total_n': 21220,
    'stratified_n': 8830,
    'hypertension_pct': 38.2,
    'back_problems_pct': 24.5,
    'diabetes_pct': 16.7,
    'oa_knee_pct': 16.2,
    'asthma_pct': 11.1,
    'heart_disease_pct': 10.3,
    'anxiety_pct': 7.4,
    'copd_pct': 6.8,
    'depression_screen_pct': 17.0,
    'loneliness_pct': 8.8,
    'volunteering_monthly_pct': 38.5,
    'other_activities_monthly_pct': 53.6,
    'self_rated_health_good_pct': 86.5,
    # Published-paper findings retained only for the social engagement protective role
    # (longitudinal outcome, not computable from frequency data alone)
    'volunteer_uplift_pct': 17,   # Ho et al. 2023 IJERPH
    'recreational_uplift_pct': 15, # Ho et al. 2023 IJERPH
}


# ── LOAD DATA ──
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

# 12-category total DALYs for 2023, used in several headline metrics below
TOTAL_2023 = sum(v[-1] for v in daly_data.values())
TOTAL_1995 = sum(v[0] for v in daly_data.values())
TOTAL_GROWTH_PCT = (TOTAL_2023 - TOTAL_1995) / TOTAL_1995 * 100

# ── SIDEBAR ──
st.sidebar.markdown("## 🏥 Disease Burden Dashboard")
st.sidebar.markdown("**GBD 2023 · CIHI · Statistics Canada · CLSA**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["📊 Overall Disease Burden", "📈 Absolute DALY Trends", "📉 Age-Standardized Rate Trends",
     "🗺️ Provincial Burden & Demographics",
     "👥 Population Projections (2025–2040)", "🏥 Hospitalization Burden & ALC",
     "🔗 Integrated Multi-Source Analysis", "📋 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
selected_diseases = st.sidebar.multiselect("Select diseases", DISEASES, default=DISEASES)
filtered_diseases = [d for d in selected_diseases if d in DISEASES] or DISEASES

# Pages where the province filter actually affects the output
PROVINCE_FILTER_PAGES = [
    "🗺️ Provincial Burden & Demographics",
    "👥 Population Projections (2025–2040)",
    "🏥 Hospitalization Burden & ALC",
    "🔗 Integrated Multi-Source Analysis",
]

if page in PROVINCE_FILTER_PAGES:
    selected_provinces = st.sidebar.multiselect("Select provinces", PROVINCES, default=PROVINCES)
    filtered_provinces = [p for p in selected_provinces if p in PROVINCES] or PROVINCES
else:
    filtered_provinces = PROVINCES  # use all provinces silently on pages where it doesn't apply

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Files (optional)")
st.sidebar.info("If present, real GBD CSV/XLSX files in this folder will be used instead of the embedded, verified dataset.")
if missing_files:
    st.sidebar.caption(f"Using embedded data for: {', '.join(sorted(set(missing_files))[:3])}{'…' if len(set(missing_files)) > 3 else ''}")
else:
    st.sidebar.success("✅ Real GBD files loaded")


# ============================================================
# PAGE 1: OVERALL DISEASE BURDEN
# ============================================================
if page == "📊 Overall Disease Burden":
    st.markdown('<p class="main-title">Disease Burden Among Aging Canadians</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A multi-source analysis using GBD 2023, CIHI, Statistics Canada, and CLSA data</p>', unsafe_allow_html=True)

    # ── Metric cards — 2 rows x 3 columns ──

    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    with r1c1:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">All-cause Disease Burden</p>
            <p class="mc-value">6.90M DALYs</p>
            <p class="mc-delta-pos">▲ +74.6% since 1995 baseline</p>
            <p class="mc-sublabel">Canadians aged 60+ · GBD 2023</p>
        </div>""", unsafe_allow_html=True)
    with r1c2:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">All-cause Mortality</p>
            <p class="mc-value">280,718 Deaths</p>
            <p class="mc-delta-pos">▲ +62.5% since 1995 baseline</p>
            <p class="mc-sublabel">Canadians aged 60+ · GBD 2023</p>
        </div>""", unsafe_allow_html=True)
    with r1c3:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">Fastest Growing Category</p>
            <p class="mc-value">Substance Use Disorders</p>
            <p class="mc-delta-pos">▲ +223.6% absolute · +54.6% rate</p>
            <p class="mc-sublabel">Highest by both measures · 1995–2023</p>
        </div>""", unsafe_allow_html=True)
    with r2c1:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">Greatest Rate Improvement</p>
            <p class="mc-value">Cardiovascular Diseases</p>
            <p class="mc-delta-neg">▼ −51.0% age-standardized rate</p>
            <p class="mc-sublabel">Evidence that sustained intervention works</p>
        </div>""", unsafe_allow_html=True)
    with r2c2:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">Provinces Analyzed</p>
            <p class="mc-value">10 Provinces</p>
            <p class="mc-delta-pos">▲ Full national coverage</p>
            <p class="mc-sublabel">All Canadian provinces · CIHI & StatCan</p>
        </div>""", unsafe_allow_html=True)
    with r2c3:
        st.markdown("""
        <div class="metric-card-custom">
            <p class="mc-label">Projected Burden by 2040</p>
            <p class="mc-value">9.49M DALYs</p>
            <p class="mc-delta-pos">▲ +61.5% from 2023 · 12 NCD categories</p>
            <p class="mc-sublabel">Polynomial regression forecast · 2024–2040</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="highlight-box">
    The 6.90 million all-cause figure above spans every disease and injury category tracked by
    GBD 2023. This study analyzes twelve major disease categories within the Non-Communicable
    Diseases classification in depth (DALYs: {TOTAL_2023/1e6:.2f} million in 2023,
    {TOTAL_2023/allcause[2023]['dalys']*100:.1f}% of the all-cause total) — it is this
    twelve-category subset that powers the trend, rate, and forecasting analyses throughout
    this dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-header">Disease Burden by Category — 2023</p>', unsafe_allow_html=True)
        st.caption("Source: GBD 2023 (IHME) | Canadians aged 60+ | Measure: DALYs (Disability-Adjusted Life Years) | Single year snapshot: 2023 only")
        dalys_2023 = {d: daly_data[d][-1] for d in filtered_diseases}
        df_plot = pd.DataFrame({'Disease': list(dalys_2023.keys()), 'DALYs': list(dalys_2023.values())})
        df_plot = df_plot.sort_values('DALYs', ascending=True)
        fig = px.bar(df_plot, x='DALYs', y='Disease', orientation='h',
                     color='Disease', color_discrete_map=DISEASE_COLORS,
                     labels={'DALYs': 'DALYs (number)'})
        fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=480,
                           plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                           xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)',
                                      showline=False, zeroline=False),
                           yaxis=dict(showgrid=False, showline=False),
                           margin=dict(l=220, r=90, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Burden distribution, 2023</p>', unsafe_allow_html=True)
        dalys_pie = {d: daly_data[d][-1] for d in DISEASES}
        fig_pie = px.pie(values=list(dalys_pie.values()), names=list(dalys_pie.keys()),
                          color=list(dalys_pie.keys()), color_discrete_map=DISEASE_COLORS,
                          hole=0.45)
        fig_pie.update_layout(
            showlegend=True,
            height=480,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                font=dict(color='#CBD5E0', size=8),
                bgcolor='rgba(0,0,0,0)',
                orientation='v',
                x=1.0,
                y=0.5
            )
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont=dict(size=9, color='white'),
            marker=dict(line=dict(color='rgba(15,27,45,1)', width=1.5))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Key Findings as a bottom row ──
    st.markdown("---")
    st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    kf1, kf2, kf3, kf4, kf5 = st.columns(5)
    with kf1:
        st.markdown("""
        <div class="warning-box">
        <b>📈 Substance use disorders</b><br>
        Highest absolute DALY growth (+223.6%) AND rate increase (+54.6%) — more than double
        mental disorders. Still smallest burden today — window for early intervention.
        </div>""", unsafe_allow_html=True)
    with kf2:
        st.markdown("""
        <div class="warning-box">
        <b>📈 Mental disorders</b><br>
        +160.7% absolute growth, +20.4% rate increase, far larger existing burden (194,075 DALYs).
        Two related but distinct priorities alongside substance use disorders.
        </div>""", unsafe_allow_html=True)
    with kf3:
        st.markdown("""
        <div class="warning-box">
        <b>📈 Two geographic pressure points</b><br>
        Alberta: fastest-growing senior population (+22.0% since 2020). Prince Edward Island:
        highest antidepressant Rx rate and highest ALC burden of any province.
        </div>""", unsafe_allow_html=True)
    with kf4:
        st.markdown("""
        <div class="success-box">
        <b>✅ Cardiovascular disease</b><br>
        Age-standardized rate fell -51.0% — the strongest evidence in this dataset that
        sustained, targeted intervention works over decades.
        </div>""", unsafe_allow_html=True)
    with kf5:
        st.markdown("""
        <div class="highlight-box">
        <b>🔮 2040 forecast</b><br>
        Substance use disorders projected +123.6% growth by 2040 — by far the highest,
        more than 40 percentage points ahead of the next-highest category.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 2: ABSOLUTE DALY TRENDS
# ============================================================
elif page == "📈 Absolute DALY Trends":
    st.markdown('<p class="main-title">Absolute DALY Trends, 1995–2023</p>', unsafe_allow_html=True)
    st.caption("Source: GBD 2023 (IHME) | Canadians aged 60+ | Raw DALY counts — affected by both disease risk AND population growth")

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
    fig.update_layout(height=480, xaxis_title="Year", yaxis_title=ylabel,
                       plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                       xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)'),
                       yaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)'),
                       legend=dict(orientation='v', x=1.01, y=1, font=dict(color='#CBD5E0', size=10), bgcolor='rgba(15,27,45,0.6)', bordercolor='rgba(99,179,237,0.15)', borderwidth=1),
                       margin=dict(l=0, r=180, t=5, b=30))
    st.plotly_chart(fig, use_container_width=True)

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
        st.markdown('<div class="warning-box"><b>Fastest growing</b><br>Substance use disorders: +223.6%<br>Skin and subcutaneous diseases: +164.1%<br>Mental disorders: +160.7%</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="highlight-box"><b>Moderate growth</b><br>Neurological disorders: +135.7%<br>Other non-comm. diseases: +135.7%<br>Sense organ diseases: +117.0%</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="success-box"><b>Slowest growth</b><br>Cardiovascular disease: +11.5%<br>(Interventions effective)</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 3: AGE-STANDARDIZED RATE TRENDS
# ============================================================
elif page == "📉 Age-Standardized Rate Trends":
    st.markdown('<p class="main-title">Age-Standardized Rate Trends, 1995–2023</p>', unsafe_allow_html=True)
    st.caption("Source: GBD 2023 (IHME) | Canadians aged 60+ | Rates per 100,000 — population-adjusted, shows genuine disease risk change independent of demographics")
    st.markdown("Controls for population growth — shows the true burden trajectory")

    col_left, col_right = st.columns([3, 2])

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
        fig.update_layout(height=500, xaxis_title="Year",
                           yaxis_title="Age-standardized rate per 100,000",
                           plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                           xaxis=dict(showgrid=False, showline=False),
                           yaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)', showline=False),
                           legend=dict(orientation='h', x=0, y=-0.25,
                               font=dict(color='#CBD5E0', size=9), bgcolor='rgba(0,0,0,0)'),
                           margin=dict(l=0, r=10, t=5, b=90))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">% Change in rate, 1995→2023</p>', unsafe_allow_html=True)
        rate_changes = {d: (rate_data[d][-1] - rate_data[d][0]) / rate_data[d][0] * 100
                         for d in filtered_diseases if d in rate_data}
        df_rate = pd.DataFrame({'Disease': list(rate_changes.keys()), 'Rate Change (%)': list(rate_changes.values())})
        df_rate = df_rate.sort_values('Rate Change (%)')
        # Shorten disease names for the bar chart y-axis so bars get more width
        name_map = {
            'Cardiovascular diseases': 'Cardiovascular',
            'Chronic respiratory diseases': 'Chr. respiratory',
            'Diabetes and kidney diseases': 'Diabetes & kidney',
            'Musculoskeletal disorders': 'Musculoskeletal',
            'Neurological disorders': 'Neurological',
            'Other non-communicable diseases': 'Other NCDs',
            'Sense organ diseases': 'Sense organ',
            'Skin and subcutaneous diseases': 'Skin & subcut.',
            'Substance use disorders': 'Substance use',
            'Mental disorders': 'Mental disorders',
            'Digestive diseases': 'Digestive',
            'Neoplasms': 'Neoplasms',
        }
        df_rate['Disease_short'] = df_rate['Disease'].map(lambda x: name_map.get(x, x))
        colors = ['#0F6E56' if v < 0 else '#D81B60' for v in df_rate['Rate Change (%)']]
        fig2 = px.bar(df_rate, x='Rate Change (%)', y='Disease_short', orientation='h',
                      color='Disease_short', color_discrete_map={name_map.get(d,d): c for d,c in zip(df_rate['Disease'], colors)},
                      text='Rate Change (%)')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.add_vline(x=0, line_color='rgba(99,179,237,0.3)', line_width=1)
        x_min = df_rate['Rate Change (%)'].min()
        x_max = df_rate['Rate Change (%)'].max()
        padding = (x_max - x_min) * 0.25
        fig2.update_layout(showlegend=False, height=600,
                            plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                            xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)',
                                       ticksuffix='%', dtick=15, tick0=0,
                                       range=[x_min - padding, x_max + padding]),
                            yaxis=dict(showgrid=False),
                            margin=dict(l=0, r=70, t=5, b=40))
        st.plotly_chart(fig2, use_container_width=True)

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
    <b>Key finding:</b> Cardiovascular disease rate fell -51.0% — the strongest evidence that
    targeted intervention works. Substance use disorders rate rose +54.6% even after
    controlling for population growth — more than double mental disorders' +20.4% rate
    increase — confirming substance use disorders as the most rapidly worsening category by
    this measure, with mental disorders a close and substantial second.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 4: PROVINCIAL BURDEN & DEMOGRAPHICS
# ============================================================
elif page == "🗺️ Provincial Burden & Demographics":
    st.markdown('<p class="main-title">Provincial Senior Population and Mental Health Burden Proxy</p>', unsafe_allow_html=True)
    st.markdown("CIHI Pharmaceutical Data Tool, 2020–2024 | All ten Canadian provinces")

    df_prov = prov_pharma_df[prov_pharma_df['Province'].isin(filtered_provinces)].copy()

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Largest Senior Population</p>
            <p class="mc-value">Ontario</p>
            <p class="mc-sublabel">2.95M seniors · 2024</p>
        </div>""", unsafe_allow_html=True)
    with pc2:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Fastest-Growing Province</p>
            <p class="mc-value">Alberta</p>
            <p class="mc-delta-pos">▲ +22.0% since 2020</p>
        </div>""", unsafe_allow_html=True)
    with pc3:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Highest Antidepressant Rx</p>
            <p class="mc-value">Prince Edward Island</p>
            <p class="mc-sublabel">256.7 per 1,000 seniors</p>
        </div>""", unsafe_allow_html=True)
    with pc4:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Lowest Antidepressant Rx</p>
            <p class="mc-value">New Brunswick</p>
            <p class="mc-sublabel">150.7 per 1,000 seniors</p>
        </div>""", unsafe_allow_html=True)

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
                           yaxis=dict(showgrid=False), xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)'))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Two pressure points: growth velocity vs. mental health burden**")
        fig2 = px.scatter(df_prov, x='Growth % (2020-24)', y='Rx per 1,000 Seniors',
                           color='Province', color_discrete_map=PROVINCE_COLORS,
                           size='Senior Pop (2024)', text='Province',
                           labels={'Growth % (2020-24)': 'Senior population growth, 2020–2024 (%)',
                                   'Rx per 1,000 Seniors': 'Antidepressant Rx per 1,000 seniors'})
        fig2.update_traces(textposition='top center')
        fig2.update_layout(showlegend=False, height=420, plot_bgcolor='rgba(15,27,45,0)',
                            paper_bgcolor='rgba(15,27,45,0)', margin=dict(l=0, r=30, t=10, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Growth Velocity vs. Mental Health Burden — The Scissor Effect</p>', unsafe_allow_html=True)
    st.caption("Bars: senior population growth rate 2020–2024 (%). Line: antidepressant Rx per 1,000 seniors (right axis). Provinces where the line sits high but the bar is short reveal capacity strain independent of demographic growth.")

    df_scissor = df_prov[df_prov['Province'].isin(filtered_provinces)].copy()
    df_scissor = df_scissor.sort_values('Growth % (2020-24)', ascending=True)

    fig_scissor = go.Figure()

    # Background bars — population growth rate
    fig_scissor.add_trace(go.Bar(
        x=df_scissor['Province'],
        y=df_scissor['Growth % (2020-24)'],
        name='Senior Pop. Growth 2020–24 (%)',
        marker_color=[PROVINCE_COLORS.get(p, '#4A5568') for p in df_scissor['Province']],
        opacity=0.6,
        yaxis='y1',
    ))

    # Overlay line — antidepressant Rx per 1,000
    fig_scissor.add_trace(go.Scatter(
        x=df_scissor['Province'],
        y=df_scissor['Rx per 1,000 Seniors'],
        name='Antidepressant Rx per 1,000 seniors',
        mode='lines+markers+text',
        line=dict(color='#D81B60', width=3),
        marker=dict(size=9, color='#D81B60', symbol='circle'),
        text=[f"{v:.0f}" for v in df_scissor['Rx per 1,000 Seniors']],
        textposition='top center',
        textfont=dict(color='#D81B60', size=9),
        yaxis='y2',
    ))

    fig_scissor.update_layout(
        height=420,
        plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
        font=dict(color='#CBD5E0'),
        xaxis=dict(showgrid=False, showline=False, tickangle=-30),
        yaxis=dict(
            title=dict(text='Senior Pop. Growth 2020–24 (%)', font=dict(color='#63B3ED')),
            tickfont=dict(color='#63B3ED'),
            showgrid=True, gridcolor='rgba(99,179,237,0.06)',
            showline=False,
        ),
        yaxis2=dict(
            title=dict(text='Antidepressant Rx per 1,000 Seniors', font=dict(color='#D81B60')),
            tickfont=dict(color='#D81B60'),
            overlaying='y', side='right',
            showgrid=False, showline=False,
            range=[100, 300],
        ),
        legend=dict(
            orientation='h', x=0, y=1.08,
            font=dict(color='#CBD5E0', size=10),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=80, t=40, b=80),
        barmode='group',
    )
    st.plotly_chart(fig_scissor, use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Reading the scissor effect:</b> Alberta (far right) shows a tall bar (fastest growth +22.0%) but a mid-range line (207.3 Rx per 1,000) — its challenge is <i>demographic velocity</i>.
    Prince Edward Island shows a short bar (modest +15.2% growth) but the highest line of any province (256.7 Rx per 1,000) — its challenge is existing <i>community-care capacity strain</i>, independent of growth.
    The two curves "scissor apart" at these two provinces, revealing two fundamentally different policy problems within the same national picture.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Complete Provincial Summary Table</p>', unsafe_allow_html=True)
    st.dataframe(df_prov.set_index('Province'), use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    <b>📈 Alberta — demographic growth velocity</b><br>
    740,710 seniors growing at +22.0% since 2020 — 6.8 percentage points ahead of the next-fastest
    province (Prince Edward Island). Alberta's antidepressant Rx rate (207.3 per 1,000) sits in the
    middle of the national distribution, indicating its primary pressure point is the speed of growth
    itself, not an unusually high baseline mental health burden.
    </div>
    <div class="warning-box">
    <b>📈 Prince Edward Island — community capacity strain</b><br>
    Combines the highest antidepressant Rx rate of any province (256.7 per 1,000) with the highest
    proportion of hospital patient-days in alternate level of care (28.0%, see Hospitalization Burden
    page) — despite comparatively modest population growth (+15.2%). This is a distinct pressure point
    from Alberta's, concentrated in the smaller Atlantic provinces.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 6: POPULATION PROJECTIONS (2025-2040)
# ============================================================
elif page == "👥 Population Projections (2025–2040)":
    st.markdown('<p class="main-title">Population Projections — Statistics Canada</p>', unsafe_allow_html=True)
    st.markdown("**Source:** Table 17-10-0057-01 | M2 medium-growth scenario | Age 65+ | 2025–2040")

    display_provinces = filtered_provinces

    st.markdown("### Key Metrics (2025 → 2040, M2 Scenario)")
    regions = ['Canada'] + display_provinces
    cols = st.columns(min(len(regions), 6))
    for i, region in enumerate(regions[:6]):
        base = pop_df.loc[2025, region]
        end = pop_df.loc[2040, region]
        growth = (end - base) / base * 100
        with cols[i]:
            st.metric(label=region, value=f"{end/1000:.2f}M" if region == 'Canada' else f"{end:,.0f}k",
                      delta=f"+{growth:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Absolute 65+ population (thousands)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pop_df.index, y=pop_df['Canada'], name='Canada',
                                  line=dict(color='#2563EB', width=3), mode='lines+markers', marker=dict(size=5)))
        for p in display_provinces:
            fig.add_trace(go.Scatter(x=pop_df.index, y=pop_df[p], name=p,
                                      line=dict(color=PROVINCE_COLORS[p], width=2, dash='dash'),
                                      mode='lines+markers', marker=dict(size=4)))
        fig.update_layout(xaxis_title='Year', yaxis_title='Population (thousands)',
                           plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                           legend=dict(orientation='h', x=0, y=-0.3,
                               font=dict(color='#CBD5E0', size=8), bgcolor='rgba(0,0,0,0)'),
                           height=420, margin=dict(l=40, r=20, t=5, b=90),
                           xaxis=dict(showgrid=False, showline=False),
                           yaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)', showline=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Growth from 2025 baseline (%)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pop_df.index, y=(pop_df['Canada'] - pop_df.loc[2025, 'Canada']) / pop_df.loc[2025, 'Canada'] * 100,
                                   name='Canada', line=dict(color='#2563EB', width=3), mode='lines+markers', marker=dict(size=5)))
        for p in display_provinces:
            base = pop_df.loc[2025, p]
            growth_pct = (pop_df[p] - base) / base * 100
            fig2.add_trace(go.Scatter(x=pop_df.index, y=growth_pct, name=p,
                                       line=dict(color=PROVINCE_COLORS[p], width=2, dash='dash'),
                                       mode='lines+markers', marker=dict(size=4)))
        fig2.update_layout(xaxis_title='Year', yaxis_title='Growth from 2025 (%)',
                            plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                            legend=dict(orientation='h', x=0, y=-0.3,
                                font=dict(color='#CBD5E0', size=8), bgcolor='rgba(0,0,0,0)'),
                            height=420, margin=dict(l=40, r=20, t=5, b=90),
                            xaxis=dict(showgrid=False, showline=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)', showline=False))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Full Data Table (thousands, M2 scenario)")
    key_years = [2025, 2030, 2035, 2040]
    display_df = pop_df.loc[key_years, ['Canada'] + display_provinces].copy()
    display_df.index.name = 'Year'
    st.dataframe(display_df.style.format("{:,.1f}"), use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Key Finding:</b> Alberta's 65+ population is projected to grow by <b>+47.8%</b> by 2040 — the
    fastest rate of any of the ten provinces, nearly double the next-fastest, Prince Edward Island
    (+27.3%) and Saskatchewan (+25.7%). The four Atlantic provinces show the slowest projected growth,
    ranging from +16.8% to +27.3%.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 7: HOSPITALIZATION BURDEN & ALC
# ============================================================
elif page == "🏥 Hospitalization Burden & ALC":
    st.markdown('<p class="main-title">Hospitalization Burden — CIHI</p>', unsafe_allow_html=True)
    st.markdown("**Source:** CIHI Hospital Morbidity Database (HMDB)/OMHRS, 2024–2025 | Released February 19, 2026")

    cat_colors = {'Cardiovascular': '#EF4444', 'Respiratory': '#3B82F6', 'Neurological': '#8B5CF6',
                  'Musculoskeletal': '#F59E0B', 'Other': '#6B7280'}

    st.markdown("### National Hospitalization Rate, Canadians Aged 65+ (2020–21 to 2024–25)")
    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Hospitalization Rate (2024–25)</p>
            <p class="mc-value">7,560 per 100k</p>
            <p class="mc-delta-pos">▲ +4.6% vs 2020–21</p>
        </div>""", unsafe_allow_html=True)
    with hc2:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Average Length of Stay</p>
            <p class="mc-value">6.1 days</p>
            <p class="mc-sublabel">+0.3 days vs 2020–21</p>
        </div>""", unsafe_allow_html=True)
    with hc3:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Top Cause (65+)</p>
            <p class="mc-value">COPD & Bronchitis</p>
            <p class="mc-sublabel">68,321 cases · #1 by volume</p>
        </div>""", unsafe_allow_html=True)
    with hc4:
        st.markdown("""<div class="metric-card-custom">
            <p class="mc-label">Longest Hospital Stay</p>
            <p class="mc-value">Neurocognitive</p>
            <p class="mc-sublabel">17.1 days avg · Highest care intensity</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Top 10 Diagnoses (65+)", "National Rate Trend", "ALC Days by Province"])

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(
                x=hosp_df['n_hosp'][::-1] / 1000, y=hosp_df['diagnosis'][::-1], orientation='h',
                marker_color=[cat_colors[c] for c in hosp_df['category'][::-1]],
                text=[f"{v/1000:.1f}k" for v in hosp_df['n_hosp'][::-1]], textposition='outside'
            ))
            fig.update_layout(title='Hospitalizations by diagnosis (thousands)',
                               xaxis_title='Hospitalizations (thousands)', height=420,
                               plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                               xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)', showline=False),
                               yaxis=dict(showgrid=False, showline=False),
                               font=dict(color='#CBD5E0'),
                               title_font=dict(color='#CBD5E0', size=13),
                               margin=dict(l=260, r=100, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Category breakdown**")
            cat_totals = hosp_df.groupby('category')['n_hosp'].sum().reset_index()
            fig_pie = go.Figure(go.Pie(labels=cat_totals['category'], values=cat_totals['n_hosp'],
                                        marker_colors=[cat_colors.get(c, '#999') for c in cat_totals['category']], hole=0.4))
            fig_pie.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#CBD5E0', size=10), bgcolor='rgba(0,0,0,0)')
            )
            fig_pie.update_traces(textfont=dict(color='white', size=10),
                                   marker=dict(line=dict(color='rgba(15,27,45,1)', width=1.5)))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("**Note:** chronic respiratory disease ranks only 5th of 12 categories by DALYs (Table 1),"
                        " yet COPD and bronchitis ranks 1st in hospitalizations — see the Integrated Analysis page.")

    with tab2:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=hosp_rate_trend_df['Fiscal Year'], y=hosp_rate_trend_df['Rate per 100,000'],
                                        mode='lines+markers', name='Rate per 100,000', line=dict(color='#2563EB', width=3)))
        fig_trend.update_layout(title='National Hospitalization Rate, Canadians Aged 65+ (2020–21 to 2024–25)',
                                 yaxis_title='Rate per 100,000', height=380, margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.dataframe(hosp_rate_trend_df.set_index('Fiscal Year'), use_container_width=True)
        st.markdown("""
        <div class="warning-box">
        <b>Key Finding:</b> Neurocognitive disorders have an average length of stay of <b>17.1 days</b> —
        nearly three times the overall average of 6.1 days — reflecting the high care intensity of
        dementia-related hospitalizations and the need for expanded community-based dementia support.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        alc_display = alc_df[alc_df['Province'].isin([p for p in filtered_provinces if p != 'Quebec'])].copy()
        alc_display = alc_display.sort_values('Patient Days in ALC (%)', ascending=True)
        highlight = ['#F0A500' if p == 'Prince Edward Island' else '#185FA5' for p in alc_display['Province']]
        fig_alc = go.Figure(go.Bar(
            x=alc_display['Patient Days in ALC (%)'], y=alc_display['Province'], orientation='h',
            marker_color=highlight, text=[f"{v:.1f}%" for v in alc_display['Patient Days in ALC (%)']],
            textposition='outside'
        ))
        fig_alc.update_layout(title='Patient days in ALC (%), 2024–2025', xaxis_title='Patient Days in ALC (%)',
                               height=420, margin=dict(l=160, r=80, t=40, b=50),
                               plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                               xaxis=dict(showgrid=True, gridcolor='rgba(99,179,237,0.06)', showline=False),
                               yaxis=dict(showgrid=False, showline=False),
                               font=dict(color='#CBD5E0'),
                               title_font=dict(color='#CBD5E0', size=13))
        st.plotly_chart(fig_alc, use_container_width=True)

        st.markdown(f"Canada (excl. Quebec) overall: **{alc_canada_df['Patient Days in ALC (%)'].iloc[0]:.1f}%** "
                    f"across {alc_canada_df['Hospitalizations (2024-25)'].iloc[0]:,} hospitalizations. "
                    "Quebec is excluded — its ALC definition is structurally narrower than the one used "
                    "in the other nine provinces (see paper Section 2.4). The Canada total also includes "
                    "Yukon, Northwest Territories, and Nunavut, which are not shown as separate rows.")

        st.markdown("""
        <div class="warning-box">
        <b>Key Finding:</b> Prince Edward Island recorded the highest ALC burden of any province (28.0%),
        more than double Saskatchewan's 11.9% (the lowest). The highest-ALC provinces — PEI, Newfoundland
        and Labrador, New Brunswick, and Nova Scotia — are concentrated in Atlantic Canada, a pattern
        invisible in a smaller provincial comparison set.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 8: INTEGRATED MULTI-SOURCE ANALYSIS
# ============================================================
elif page == "🔗 Integrated Multi-Source Analysis":
    st.markdown('<p class="main-title">Integrated Analysis — All Four Data Sources</p>', unsafe_allow_html=True)
    st.markdown("Combining GBD 2023 burden data, Statistics Canada population projections, "
                "CIHI hospitalization and pharmaceutical data, and CLSA individual-level evidence.")

    tab1, tab2, tab3 = st.tabs(["GBD–CIHI Divergence", "Two Geographic Pressure Points", "CLSA Corroboration"])

    with tab1:
        st.markdown("#### DALY rank vs. hospitalization rank, by category")
        st.markdown("Comparing the four disease categories present in both GBD (Table 1, ranked out of "
                    "12) and CIHI's top-10 hospitalization diagnoses (Table 6).")
        daly_rank = {'Cardiovascular diseases': 2, 'Neurological disorders': 3,
                     'Musculoskeletal disorders': 4, 'Chronic respiratory diseases': 5}
        hosp_cat_map = {'Cardiovascular diseases': 'Cardiovascular', 'Neurological disorders': 'Neurological',
                        'Musculoskeletal disorders': 'Musculoskeletal', 'Chronic respiratory diseases': 'Respiratory'}
        hosp_totals = hosp_df.groupby('category')['n_hosp'].sum()
        hosp_rank_order = hosp_totals.sort_values(ascending=False).index.tolist()

        rows = []
        for disease, cat in hosp_cat_map.items():
            rows.append({
                'Disease Category': disease,
                'DALY Rank (of 12)': daly_rank[disease],
                'Hospitalizations (top 10 sum)': int(hosp_totals[cat]),
                'Hospitalization Rank (of 5)': hosp_rank_order.index(cat) + 1,
            })
        df_div = pd.DataFrame(rows).sort_values('DALY Rank (of 12)')
        st.dataframe(df_div.set_index('Disease Category'), use_container_width=True)

        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(name='DALY Rank (of 12)', x=df_div['Disease Category'], y=df_div['DALY Rank (of 12)'],
                                  marker_color='#854F0B'))
        fig_div.add_trace(go.Bar(name='Hospitalization Rank (of 5)', x=df_div['Disease Category'], y=df_div['Hospitalization Rank (of 5)'],
                                  marker_color='#3B82F6'))
        fig_div.update_layout(barmode='group', yaxis_title='Rank (1 = highest)',
                               yaxis=dict(autorange='reversed', gridcolor='rgba(99,179,237,0.06)', showline=False),
                               xaxis=dict(showgrid=False, showline=False),
                               plot_bgcolor='rgba(15,27,45,0)', paper_bgcolor='rgba(15,27,45,0)',
                               font=dict(color='#CBD5E0'),
                               legend=dict(font=dict(color='#CBD5E0', size=10), bgcolor='rgba(15,27,45,0.6)'),
                               height=380, margin=dict(l=20, r=20, t=20, b=60))
        st.plotly_chart(fig_div, use_container_width=True)

        st.markdown("""
        <div class="warning-box">
        <b>Key Finding:</b> Chronic respiratory disease ranks only 5th of 12 categories by DALYs, yet COPD
        and bronchitis is the #1 hospitalization cause for Canadians 65+ — a disconnect between disability
        burden and acute-care utilization, consistent with the episodic, exacerbation-driven nature of
        COPD. Cardiovascular disease shows the opposite pattern: large DALY burden, but a hospitalization
        share more proportionate to that burden, consistent with effective prevention and management.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Alberta (growth velocity) vs. the smaller Atlantic provinces (capacity strain)")
        merged = prov_pharma_df.merge(
            pd.concat([alc_df, pd.DataFrame([{'Province': 'Quebec', 'Hospitalizations (2024-25)': None, 'Patient Days in ALC (%)': None}])],
                      ignore_index=True),
            on='Province', how='left'
        )
        merged = merged[merged['Province'].isin(filtered_provinces)]

        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(name='Senior pop. growth 2020–24 (%)', x=merged['Province'],
                                    y=merged['Growth % (2020-24)'], marker_color='#F0A500', yaxis='y1'))
        fig_combo.add_trace(go.Scatter(name='Patient days in ALC (%)', x=merged['Province'],
                                        y=merged['Patient Days in ALC (%)'], mode='markers+lines',
                                        marker=dict(size=10, color='#A32D2D'), yaxis='y2'))
        fig_combo.update_layout(
            height=420, margin=dict(l=40, r=60, t=40, b=80),
            yaxis=dict(title='Senior population growth, 2020–2024 (%)'),
            yaxis2=dict(title='Patient days in ALC (%)', overlaying='y', side='right'),
            legend=dict(x=0.01, y=1.15, orientation='h', font=dict(color='#CBD5E0', size=10), bgcolor='rgba(15,27,45,0.6)')
        )
        st.plotly_chart(fig_combo, use_container_width=True)

        st.markdown("""
        <div class="highlight-box">
        <b>Integrated Insight:</b> Alberta combines by far the fastest senior population growth (+22.0%
        since 2020, projected +47.8% by 2040) with a mid-range antidepressant Rx rate and below-average
        ALC burden — its challenge is demographic velocity, not an undersized system today. Prince Edward
        Island shows the opposite pattern: comparatively modest population growth (+15.2%) paired with the
        highest antidepressant Rx rate and the highest ALC burden of any province — a community-care
        capacity shortfall relative to its current, more slowly growing population. These are two distinct
        policy problems, not one "highest burden" province.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Individual-level corroboration from the CLSA")
        st.markdown(
            f"CLSA Tracking cohort (wave 3–4), n = {CLSA_FINDINGS['total_n']:,} participants aged 65 and older. "
            f"Stratified frequency data available for {CLSA_FINDINGS['stratified_n']:,} participants "
            f"(males 65–74: 2,292; females 65–74: 2,336; males 75+: 2,102; females 75+: 2,100). "
            f"All prevalence estimates below are from the full sample of {CLSA_FINDINGS['total_n']:,}. "
            f"Source: clsa_Baseline_StratifiedFrequencies_Amala_Jolly.xlsx — analysed under formal data access agreement."
        )

        st.markdown("**Chronic condition prevalence**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Hypertension", f"{CLSA_FINDINGS['hypertension_pct']}%", "Most prevalent condition")
            st.metric("Back problems", f"{CLSA_FINDINGS['back_problems_pct']}%", "Musculoskeletal")
            st.metric("Diabetes", f"{CLSA_FINDINGS['diabetes_pct']}%", "Diabetes & kidney")
        with c2:
            st.metric("OA of knee", f"{CLSA_FINDINGS['oa_knee_pct']}%", "Musculoskeletal")
            st.metric("Heart disease", f"{CLSA_FINDINGS['heart_disease_pct']}%", "Cardiovascular")
            st.metric("COPD", f"{CLSA_FINDINGS['copd_pct']}%", "Respiratory")
        with c3:
            st.metric("Depression (positive screen)", f"{CLSA_FINDINGS['depression_screen_pct']}%", "Mental health")
            st.metric("Anxiety", f"{CLSA_FINDINGS['anxiety_pct']}%", "Mental health")
            st.metric("Loneliness", f"{CLSA_FINDINGS['loneliness_pct']}%", "Social determinant")

        st.markdown("**Social participation and general health**")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("Self-rated health 'good or better'", f"{CLSA_FINDINGS['self_rated_health_good_pct']}%", "Despite chronic burden")
        with c5:
            st.metric("Volunteering (at least monthly)", f"{CLSA_FINDINGS['volunteering_monthly_pct']}%", "Protective factor")
        with c6:
            st.metric("Other activities (at least monthly)", f"{CLSA_FINDINGS['other_activities_monthly_pct']}%", "Protective factor")

        st.markdown(f"""
        <div class="success-box">
        <b>CLSA corroboration of GBD burden patterns:</b><br>
        Hypertension ({CLSA_FINDINGS['hypertension_pct']}%) → GBD cardiovascular diseases (#2 by DALYs) ✅<br>
        Back problems ({CLSA_FINDINGS['back_problems_pct']}%) + OA knee ({CLSA_FINDINGS['oa_knee_pct']}%) → GBD musculoskeletal disorders (#4) ✅<br>
        Diabetes ({CLSA_FINDINGS['diabetes_pct']}%) → GBD diabetes & kidney diseases (#6) ✅<br>
        COPD ({CLSA_FINDINGS['copd_pct']}%) → CIHI #1 inpatient hospitalization cause ✅<br>
        Depression screen ({CLSA_FINDINGS['depression_screen_pct']}%) + Anxiety ({CLSA_FINDINGS['anxiety_pct']}%) → GBD mental disorders (fastest worsening by rate) ✅<br><br>
        Despite this chronic disease burden, {CLSA_FINDINGS['self_rated_health_good_pct']}% of participants
        rated their general health as good, very good, or excellent. Social participation was widespread:
        {CLSA_FINDINGS['volunteering_monthly_pct']}% volunteered at least monthly and
        {CLSA_FINDINGS['other_activities_monthly_pct']}% engaged in other recreational activities at least
        monthly, consistent with published evidence that social engagement is protective of healthy aging
        outcomes (Ho et al., 2023 IJERPH).
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 9: DATA EXPLORER
# ============================================================
elif page == "📋 Data Explorer":
    st.markdown('<p class="main-title">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown("Explore and download the underlying data from all four sources.")

    st.info(
        "**About the data in this dashboard** — All figures are embedded as verified constants "
        "extracted programmatically from the original source files using the Python notebooks "
        "in the accompanying GitHub repository. The dashboard runs on Streamlit Cloud, which "
        "does not have access to raw files at runtime — constants are used for reliability and "
        "speed. Every number traces back to a specific source file and notebook cell, documented "
        "in `data_verification.md` in the GitHub repository. To verify any figure independently: "
        "(1) run the notebooks in the repo against the original source files, or "
        "(2) query the GBD Results Tool directly at vizhub.healthdata.org/gbd-results using "
        "the parameters in paper Methods Section 2.1, or "
        "(3) consult the data verification log at github.com/amalaajkamal/GBD-and-AFU."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["GBD: DALYs & Rates", "CIHI: Provincial & Hospitalization", "StatCan: Population", "Download"]
    )

    with tab1:
        st.markdown("**Absolute DALYs — Canadians 60+, 1995–2023 (12 categories)**")
        df_dalys = pd.DataFrame(daly_data, index=OBS_YEARS).T
        df_dalys.index.name = 'Disease'
        df_dalys['% Change 1995-2023'] = df_dalys.apply(lambda row: f"+{(row.iloc[-1]-row.iloc[0])/row.iloc[0]*100:.1f}%", axis=1)
        st.dataframe(df_dalys.style.format({c: "{:,.0f}" for c in OBS_YEARS}), use_container_width=True)

        st.markdown("**Age-standardized DALY rates per 100,000 — Canadians 60+, 1995–2023 (12 categories)**")
        df_rates = pd.DataFrame(rate_data, index=OBS_YEARS).T
        df_rates.index.name = 'Disease'
        df_rates['Rate Change'] = df_rates.apply(lambda row: f"{(row.iloc[-1]-row.iloc[0])/row.iloc[0]*100:+.1f}%", axis=1)
        st.dataframe(df_rates.style.format({c: "{:.1f}" for c in OBS_YEARS}), use_container_width=True)

    with tab2:
        st.markdown("**Senior population and mental health burden proxy by province, 2024**")
        st.dataframe(prov_pharma_df.set_index('Province'), use_container_width=True)
        st.markdown("**Top 10 inpatient hospitalizations, age 65+, 2024–2025**")
        st.dataframe(hosp_df.set_index('rank'), use_container_width=True)
        st.markdown("**ALC indicators by province, 2024–2025**")
        st.dataframe(pd.concat([alc_df, alc_canada_df], ignore_index=True).set_index('Province'), use_container_width=True)

    with tab3:
        st.markdown("**Projected 65+ population by province, Statistics Canada M2 scenario**")
        key_years = [2025, 2030, 2035, 2040]
        st.dataframe(pop_df.loc[key_years].style.format("{:,.1f}"), use_container_width=True)
        st.markdown("**Download Statistics Canada population data (CSV)**")
        csv_pop = pop_df.to_csv().encode('utf-8')
        st.download_button("⬇️ Download population projections (CSV)", data=csv_pop,
                            file_name='StatCan_65plus_Population_M2_2025_2040.csv', mime='text/csv')

    with tab4:
        st.markdown("**Download analysis data as CSV**")
        df_download = pd.DataFrame(daly_data, index=OBS_YEARS).T
        df_download.index.name = 'Disease'
        st.download_button("⬇️ Download DALYs data (CSV)", data=df_download.to_csv().encode('utf-8'),
                            file_name='GBD_2023_Canada_DALYs_12cat.csv', mime='text/csv')

        df_rate_download = pd.DataFrame(rate_data, index=OBS_YEARS).T
        df_rate_download.index.name = 'Disease'
        st.download_button("⬇️ Download rate data (CSV)", data=df_rate_download.to_csv().encode('utf-8'),
                            file_name='GBD_2023_Canada_Rates_12cat.csv', mime='text/csv')

        st.download_button("⬇️ Download provincial data (CSV)", data=prov_pharma_df.to_csv(index=False).encode('utf-8'),
                            file_name='CIHI_Provincial_Senior_Population_2024.csv', mime='text/csv')

        st.download_button("⬇️ Download hospitalization data (CSV)", data=hosp_df.to_csv(index=False).encode('utf-8'),
                            file_name='CIHI_Top10_Hospitalizations_65plus_2024_2025.csv', mime='text/csv')

        st.download_button("⬇️ Download ALC data (CSV)", data=pd.concat([alc_df, alc_canada_df], ignore_index=True).to_csv(index=False).encode('utf-8'),
                            file_name='CIHI_ALC_Indicators_2024_2025.csv', mime='text/csv')


# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:12px; color:#718096; padding:1rem 0;'>
Disease Burden Among Aging Canadians — Multi-Source Analysis Dashboard |
GBD 2023 (IHME) · CIHI · Statistics Canada · CLSA | Twelve NCD Level 2 disease categories |
University of Windsor, Windsor, Ontario, Canada | June 2026
</div>
""", unsafe_allow_html=True)
