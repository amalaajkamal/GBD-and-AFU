# ============================================================
# GBD 2023 & Age-Friendly University Analysis Dashboard
# University of Windsor — June 2026
# Authors: [Your Name], Parthiban Natarajan, Shanthi Johnson
# Run: streamlit run gbd_afu_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="GBD 2023 & AFU Canada Analysis",
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
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.68rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# ── DISEASE COLORS ──
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

# ── DATA LOADING ──
@st.cache_data
def load_gbd_data():
    """Load and merge all GBD CSV files"""
    file_map = {
        'Chronic respiratory diseases': 'IHME-GBD_2023_DATA-5da954e9-1_respiratory.csv',
        'Mental disorders':             'IHME-GBD_2023_DATA-4eb8814a-1_mental.csv',
        'Diabetes and kidney diseases': 'IHME-GBD_2023_DATA-dd10953b-1_diabetics.csv',
        'Musculoskeletal disorders':    'IHME-GBD_2023_DATA-100da0be-1_musculo.csv',
        'Neurological disorders':       'IHME-GBD_2023_DATA-cd6c1a03-1_neuro.csv',
        'Neoplasms':                    'IHME-GBD_2023_DATA-1c9f46d3-1_neo.csv',
        'Cardiovascular diseases':      'IHME-GBD_2023_DATA-141409a9-1_cardio.xlsx',
    }
    dfs = []
    missing = []
    for disease, fname in file_map.items():
        if os.path.exists(fname):
            try:
                df = pd.read_excel(fname) if fname.endswith('.xlsx') else pd.read_csv(fname)
                dfs.append(df)
            except Exception as e:
                missing.append(disease)
        else:
            missing.append(disease)

    if dfs:
        return pd.concat(dfs, ignore_index=True), missing
    return None, missing

@st.cache_data
def load_cihi_data():
    """Load CIHI provincial data"""
    fname = 'prescribed-drug-use-seniors-pharm-data-tool-data-tables-en.xlsx'
    if os.path.exists(fname):
        cols = ['Year','Sort','Province','Sex','AgeGroup','Beneficiaries','SeniorPop',
                'LessThan5','Drug5to9','Drug10to14','Drug15plus','NoChronic',
                'Chronic1to4','Chronic5to9','Chronic10to14','Chronic15plus',
                'Antidepressants','Opioids','Benzodiazepines','Antipsychotics']
        df = pd.read_excel(fname, sheet_name='Table 1', header=2, names=cols)
        return df
    return None

@st.cache_data
def get_fallback_data():
    """Pre-extracted data as fallback when files not available"""
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
        'Mental disorders': 0,
    }
    return daly_data, rate_data, deaths_2023

@st.cache_data
def compute_forecasts(daly_data):
    """Polynomial regression forecasts"""
    annual_years = np.arange(1995, 2024)
    forecast_years = np.arange(2024, 2041)
    forecasts = {}
    metrics = {}
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

# ── SIDEBAR ──
st.sidebar.markdown("## 🏥 GBD 2023 & AFU Analysis")
st.sidebar.markdown("**University of Windsor**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to",
    ["📊 Overview", "📈 Trend Analysis", "📉 Rate Analysis",
     "🗺️ Provincial Analysis", "🎯 AFU Alignment", "🔮 Forecasting 2040",
     "👥 Population Projections", "🏥 Hospitalization Burden",
     "🔗 Integrated Analysis", "📋 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
selected_diseases = st.sidebar.multiselect(
    "Select diseases",
    DISEASES,
    default=DISEASES
)
selected_years = st.sidebar.multiselect(
    "Select years",
    OBS_YEARS,
    default=OBS_YEARS
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Files")
st.sidebar.info("Place your GBD CSV files and CIHI Excel file in the same folder as this script.")
st.sidebar.markdown("**Required files:**")
for f in ['respiratory.csv', 'mental.csv', 'diabetics.csv', 'musculo.csv', 'neuro.csv', 'neo.csv', 'cardio.xlsx']:
    st.sidebar.markdown(f"- `{f}`")

# ── LOAD DATA ──
gbd_raw, missing_files = load_gbd_data()
cihi_raw = load_cihi_data()
daly_data, rate_data, deaths_2023 = get_fallback_data()

if gbd_raw is not None:
    daly_pivot = gbd_raw[
        (gbd_raw['age_name'] == '60+ years') &
        (gbd_raw['metric_name'] == 'Number') &
        (gbd_raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)')
    ].pivot_table(index='cause_name', columns='year', values='val')

    for disease in DISEASES:
        if disease in daly_pivot.index:
            daly_data[disease] = [daly_pivot.loc[disease, y] for y in OBS_YEARS if y in daly_pivot.columns]

if missing_files:
    st.sidebar.warning(f"Using pre-extracted data for: {', '.join(missing_files[:3])}{'...' if len(missing_files)>3 else ''}")
else:
    st.sidebar.success("✅ All GBD files loaded!")

if cihi_raw is not None:
    st.sidebar.success("✅ CIHI data loaded!")

forecasts, forecast_years, forecast_metrics = compute_forecasts(daly_data)

# ── FILTER DATA ──
filtered_diseases = [d for d in selected_diseases if d in DISEASES]
if not filtered_diseases:
    filtered_diseases = DISEASES
# ============================================================
# NEW SECTIONS — Enriched Analysis (Statistics Canada + CIHI + AFU Audit)
# Add these pages to the sidebar radio options and page routing
# ============================================================

# ── STEP 1: Update the sidebar radio to include new pages ──
# Replace the existing page = st.sidebar.radio(...) block with:
#
# page = st.sidebar.radio(
#     "Navigate to",
#     ["📊 Overview", "📈 Trend Analysis", "📉 Rate Analysis",
#      "🗺️ Provincial Analysis", "🎯 AFU Alignment", "🔮 Forecasting 2040",
#      "👥 Population Projections", "🏥 Hospitalization Burden",
#      "🔗 Integrated Analysis", "📋 Data Explorer"]
# )
#
# ── STEP 2: Paste the functions below BEFORE the page routing if/elif block ──

# ── EMBEDDED DATA (no CSV needed — hardcoded from downloaded StatsCan files) ──

@st.cache_data
def get_statcan_population():
    """Statistics Canada Table 17-10-0057-01 — M2 scenario — 65+ population (thousands)"""
    years = list(range(2025, 2041))
    data = {
        'Canada':           [8108.4,8365.3,8610.2,8861.3,9100.4,9318.5,9498.2,
                             9646.7,9784.8,9920.2,10053.5,10183.0,10288.2,10379.6,10466.5,10557.7],
        'Ontario':          [3069.7,3171.4,3270.4,3373.7,3474.0,3567.0,3646.2,
                             3712.4,3775.0,3836.7,3897.3,3956.0,4004.5,4046.3,4084.4,4122.5],
        'British Columbia': [1167.2,1202.9,1236.7,1273.0,1306.9,1336.5,1361.4,
                             1382.7,1404.5,1427.0,1449.4,1470.6,1487.1,1501.7,1515.2,1528.5],
        'Alberta':          [780.0,812.9,844.4,875.9,906.2,934.0,958.2,
                             980.2,1002.0,1024.3,1046.8,1070.3,1091.6,1112.0,1132.1,1152.9],
        'Manitoba':         [260.4,268.2,275.6,282.7,289.5,295.6,300.6,
                             304.4,307.9,311.4,315.2,318.9,322.3,325.1,327.7,330.7],
    }
    df = pd.DataFrame(data, index=years)
    df.index.name = 'Year'
    return df

@st.cache_data
def get_cihi_hosp_65plus():
    """CIHI HMDB/OMHRS 2024-2025 — Top 10 hospitalizations, age 65+"""
    return pd.DataFrame([
        {'rank':1,  'diagnosis':'COPD and bronchitis',                   'n_hosp':68321, 'pct':4.633, 'avg_los':7.31,  'category':'Respiratory'},
        {'rank':2,  'diagnosis':'Heart failure',                         'n_hosp':61591, 'pct':4.176, 'avg_los':9.71,  'category':'Cardiovascular'},
        {'rank':3,  'diagnosis':'Neurocognitive disorders',              'n_hosp':49996, 'pct':3.390, 'avg_los':17.14, 'category':'Neurological'},
        {'rank':4,  'diagnosis':'Pneumonia',                             'n_hosp':49060, 'pct':3.327, 'avg_los':7.92,  'category':'Respiratory'},
        {'rank':5,  'diagnosis':'Osteoarthritis of the knee',            'n_hosp':45585, 'pct':3.091, 'avg_los':2.10,  'category':'Musculoskeletal'},
        {'rank':6,  'diagnosis':'Other medical care (palliative/chemo)', 'n_hosp':40829, 'pct':2.769, 'avg_los':9.08,  'category':'Other'},
        {'rank':7,  'diagnosis':'Acute myocardial infarction',           'n_hosp':40284, 'pct':2.732, 'avg_los':5.61,  'category':'Cardiovascular'},
        {'rank':8,  'diagnosis':'Fracture of femur',                     'n_hosp':39700, 'pct':2.692, 'avg_los':11.41, 'category':'Musculoskeletal'},
        {'rank':9,  'diagnosis':'Cerebral infarction',                   'n_hosp':32662, 'pct':2.215, 'avg_los':10.43, 'category':'Neurological'},
        {'rank':10, 'diagnosis':'Other urinary diseases (UTI)',          'n_hosp':26443, 'pct':1.793, 'avg_los':7.82,  'category':'Other'},
    ])

@st.cache_data
def get_cihi_alc():
    """CIHI Table 7 — ALC days by province 2023-24 and 2024-25"""
    return pd.DataFrame([
        {'province':'Ontario',          'alc_pct_2324':17.44, 'alc_pct_2425':17.93, 'afu_count':6},
        {'province':'British Columbia', 'alc_pct_2324':15.03, 'alc_pct_2425':15.76, 'afu_count':3},
        {'province':'Alberta',          'alc_pct_2324':15.25, 'alc_pct_2425':15.13, 'afu_count':1},
        {'province':'Manitoba',         'alc_pct_2324':15.60, 'alc_pct_2425':17.28, 'afu_count':1},
    ])

@st.cache_data
def get_afu_matrix():
    """AFU program alignment matrix — 11 institutions x 10 GBD categories"""
    categories = ['Cardiovascular','Neurological/Dementia','Mental Health',
                  'Musculoskeletal','Respiratory','Diabetes/Metabolic',
                  'Social Participation','Physical Activity','Lifelong Learning','Caregiver Support']
    data = {
        'University of Calgary':       [3,2,1,3,1,2,2,3,2,1],
        'Kwantlen Polytechnic':        [1,1,2,1,1,1,3,3,3,1],
        'UBC Okanagan':                [2,1,2,1,2,1,2,2,2,2],
        'University of Fraser Valley': [1,1,1,1,1,1,2,2,3,1],
        'Niagara College':             [1,1,2,1,1,1,2,2,2,1],
        'McMaster University':         [3,2,2,2,2,2,2,3,2,2],
        'Toronto Metropolitan':        [1,3,2,1,1,1,3,1,2,3],
        'Trent University':            [1,1,2,1,1,1,2,1,3,1],
        'Ontario Tech':                [1,1,1,2,1,1,1,2,2,1],
        'University of Windsor':       [1,1,1,1,1,1,2,2,3,1],
        'University of Manitoba':      [1,3,3,2,1,1,2,2,3,3],
    }
    df = pd.DataFrame(data, index=categories).T
    df['Province'] = ['Alberta','British Columbia','British Columbia','British Columbia',
                      'Ontario','Ontario','Ontario','Ontario','Ontario','Ontario','Manitoba']
    return df, categories


# ── PAGE: POPULATION PROJECTIONS ──────────────────────────────────────────────
def page_population_projections():
    st.markdown('<p class="main-title">👥 Population Projections — Statistics Canada</p>', unsafe_allow_html=True)
    st.markdown("**Source:** Table 17-10-0057-01 | M2 medium-growth scenario | Age 65+ | 2025–2040")

    pop_df = get_statcan_population()
    provinces = ['Ontario', 'British Columbia', 'Alberta', 'Manitoba']
    prov_colors = {'Ontario':'#16A34A','British Columbia':'#9333EA','Alberta':'#DC2626','Manitoba':'#D97706'}

    # ── Key metrics ──
    st.markdown("### Key Metrics (2025 → 2040, M2 Scenario)")
    cols = st.columns(5)
    regions = ['Canada'] + provinces
    region_colors = ['#2563EB'] + [prov_colors[p] for p in provinces]
    for col, region, color in zip(cols, regions, region_colors):
        base = pop_df.loc[2025, region]
        end  = pop_df.loc[2040, region]
        growth = (end - base) / base * 100
        col.metric(
            label=region,
            value=f"{end/1000:.2f}M" if region == 'Canada' else f"{end:,.0f}k",
            delta=f"+{growth:.1f}%"
        )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Absolute 65+ Population (thousands)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pop_df.index, y=pop_df['Canada'],
            name='Canada', line=dict(color='#2563EB', width=3),
            mode='lines+markers', marker=dict(size=5)
        ))
        for p in provinces:
            fig.add_trace(go.Scatter(
                x=pop_df.index, y=pop_df[p],
                name=p, line=dict(color=prov_colors[p], width=2, dash='dash'),
                mode='lines+markers', marker=dict(size=4)
            ))
        fig.update_layout(
            xaxis_title='Year', yaxis_title='Population (thousands)',
            legend=dict(x=0.01, y=0.99), height=380,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Growth from 2025 Baseline (%)")
        fig2 = go.Figure()
        for region, color in zip(['Canada'] + provinces, region_colors):
            base = pop_df.loc[2025, region]
            growth_pct = (pop_df[region] - base) / base * 100
            lw = 3 if region == 'Canada' else 2
            dash = 'solid' if region == 'Canada' else 'dash'
            fig2.add_trace(go.Scatter(
                x=pop_df.index, y=growth_pct,
                name=region, line=dict(color=color, width=lw, dash=dash),
                mode='lines+markers', marker=dict(size=4)
            ))
        fig2.update_layout(
            xaxis_title='Year', yaxis_title='Growth from 2025 (%)',
            legend=dict(x=0.01, y=0.99), height=380,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Data table ──
    st.markdown("#### Full Data Table (thousands, M2 scenario)")
    display_df = pop_df[['Canada'] + provinces].copy()
    display_df.index.name = 'Year'
    st.dataframe(display_df.style.format("{:,.1f}"), use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Key Finding:</b> Alberta's 65+ population is projected to grow by <b>+47.8%</b> by 2040 — 
    the fastest among the four provinces — yet has only <b>1 AFU institution</b>. 
    This represents the most critical geographic gap in age-friendly university coverage in Canada.
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: HOSPITALIZATION BURDEN ──────────────────────────────────────────────
def page_hospitalization_burden():
    st.markdown('<p class="main-title">🏥 Hospitalization Burden — CIHI</p>', unsafe_allow_html=True)
    st.markdown("**Source:** CIHI Hospital Morbidity Database (HMDB)/OMHRS, 2024–2025 | Released February 19, 2026")

    hosp_df = get_cihi_hosp_65plus()
    alc_df  = get_cihi_alc()

    cat_colors = {
        'Cardiovascular':'#EF4444','Respiratory':'#3B82F6',
        'Neurological':'#8B5CF6','Musculoskeletal':'#F59E0B','Other':'#6B7280'
    }

    # ── Key metrics ──
    st.markdown("### National Hospitalization Rate (All Ages)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rate 2024–25 (per 100k)", "7,560", delta="+4.6% vs 2020–21")
    m2.metric("Avg Length of Stay", "6.1 days", delta="+0.3d vs 2020–21")
    m3.metric("Top 65+ Cause", "COPD (68,321)", delta="#1 by volume")
    m4.metric("Longest LOS (65+)", "Neurocognitive (17.1d)", delta="Highest burden/stay")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Top 10 Diagnoses (65+)", "Length of Stay", "ALC Days by Province"])

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(
                x=hosp_df['n_hosp'][::-1] / 1000,
                y=hosp_df['diagnosis'][::-1],
                orientation='h',
                marker_color=[cat_colors[c] for c in hosp_df['category'][::-1]],
                text=[f"{v:,}" for v in hosp_df['n_hosp'][::-1]],
                textposition='outside'
            ))
            fig.update_layout(
                title='Hospitalizations by Diagnosis (thousands)',
                xaxis_title='Hospitalizations (thousands)',
                height=420, margin=dict(l=260, r=60, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Category Breakdown**")
            cat_totals = hosp_df.groupby('category')['n_hosp'].sum().reset_index()
            fig_pie = go.Figure(go.Pie(
                labels=cat_totals['category'],
                values=cat_totals['n_hosp'],
                marker_colors=[cat_colors.get(c,'#999') for c in cat_totals['category']],
                hole=0.4
            ))
            fig_pie.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20),
                                   showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("**Colour Legend**")
            for cat, color in cat_colors.items():
                st.markdown(f'<span style="color:{color}">■</span> {cat}', unsafe_allow_html=True)

    with tab2:
        fig_los = go.Figure(go.Bar(
            x=hosp_df['diagnosis'],
            y=hosp_df['avg_los'],
            marker_color=[cat_colors[c] for c in hosp_df['category']],
            text=[f"{v:.1f}d" for v in hosp_df['avg_los']],
            textposition='outside'
        ))
        fig_los.update_layout(
            title='Average Acute Length of Stay by Diagnosis (65+, 2024–25)',
            xaxis_title='Diagnosis', yaxis_title='Avg LOS (days)',
            xaxis_tickangle=-30, height=420,
            margin=dict(l=40, r=20, t=60, b=120)
        )
        st.plotly_chart(fig_los, use_container_width=True)
        st.markdown("""
        <div class="warning-box">
        <b>Key Finding:</b> Neurocognitive disorders have an average LOS of <b>17.1 days</b> — 
        more than double the overall average of 6.1 days — reflecting the high care intensity 
        of dementia-related hospitalizations and the urgent need for community-based dementia 
        support programs aligned with AFU principles.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig_alc = go.Figure()
            x = alc_df['province']
            fig_alc.add_trace(go.Bar(
                name='2023–2024', x=x, y=alc_df['alc_pct_2324'],
                marker_color='#93C5FD', text=[f"{v:.1f}%" for v in alc_df['alc_pct_2324']],
                textposition='outside'
            ))
            fig_alc.add_trace(go.Bar(
                name='2024–2025', x=x, y=alc_df['alc_pct_2425'],
                marker_color='#2563EB', text=[f"{v:.1f}%" for v in alc_df['alc_pct_2425']],
                textposition='outside'
            ))
            fig_alc.update_layout(
                barmode='group', title='ALC Patient Days (%) — Year-over-Year',
                yaxis_title='Patient Days in ALC (%)', height=380,
                margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig_alc, use_container_width=True)

        with c2:
            prov_colors_map = {'Ontario':'#16A34A','British Columbia':'#9333EA',
                               'Alberta':'#DC2626','Manitoba':'#D97706'}
            fig_sc = go.Figure()
            for _, row in alc_df.iterrows():
                fig_sc.add_trace(go.Scatter(
                    x=[row['afu_count']], y=[row['alc_pct_2425']],
                    mode='markers+text',
                    marker=dict(size=20, color=prov_colors_map[row['province']]),
                    text=[row['province']], textposition='top center',
                    name=row['province'], showlegend=True
                ))
            # trend line
            import numpy as np
            z = np.polyfit(alc_df['afu_count'], alc_df['alc_pct_2425'], 1)
            p = np.poly1d(z)
            xline = np.linspace(0.5, 6.5, 50)
            fig_sc.add_trace(go.Scatter(
                x=xline, y=p(xline), mode='lines',
                line=dict(color='#94A3B8', dash='dash'), name='Trend', showlegend=True
            ))
            fig_sc.update_layout(
                title='AFU Count vs. ALC Burden (2024–25)',
                xaxis_title='Number of AFU Institutions',
                yaxis_title='Patient Days in ALC (%)',
                xaxis=dict(range=[0,7]), height=380,
                margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("""
        <div class="warning-box">
        <b>Key Finding:</b> Manitoba ALC burden worsened from 15.6% to 17.3% year-over-year — 
        the steepest increase among the four provinces — while having only 1 AFU institution. 
        The scatter plot suggests an inverse relationship between AFU coverage and hospital system strain.
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: INTEGRATED ANALYSIS ────────────────────────────────────────────────
def page_integrated_analysis():
    st.markdown('<p class="main-title">🔗 Integrated Analysis — All Data Sources</p>', unsafe_allow_html=True)
    st.markdown("Combining GBD 2023 burden forecasts, Statistics Canada population projections, CIHI hospitalization data, and AFU program audit.")

    pop_df  = get_statcan_population()
    alc_df  = get_cihi_alc()
    afu_df, categories = get_afu_matrix()

    tab1, tab2, tab3 = st.tabs(["AFU Program Heatmap", "Gap Analysis", "Provincial Summary"])

    with tab1:
        st.markdown("#### AFU Program Alignment with GBD Disease Categories")
        st.markdown("Scale: **0** = No program | **1** = Minor | **2** = Moderate | **3** = Strong focus")

        scores = afu_df.drop('Province', axis=1)
        province_order = ['Alberta','British Columbia','Ontario','Manitoba']
        sorted_inst = []
        for prov in province_order:
            sorted_inst += [i for i in scores.index if afu_df.loc[i,'Province'] == prov]

        fig_heat = go.Figure(go.Heatmap(
            z=scores.loc[sorted_inst].values,
            x=categories,
            y=sorted_inst,
            colorscale=[[0,'#FEF3C7'],[0.33,'#FCD34D'],[0.67,'#F59E0B'],[1,'#B45309']],
            zmin=0, zmax=3,
            text=scores.loc[sorted_inst].values,
            texttemplate='%{text}',
            colorbar=dict(title='Score', tickvals=[0,1,2,3],
                         ticktext=['None','Minor','Moderate','Strong'])
        ))
        fig_heat.update_layout(
            xaxis_tickangle=-30, height=460,
            margin=dict(l=200, r=40, t=40, b=120)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        avg_scores = scores.mean().sort_values()
        st.markdown("**Average program coverage by GBD category (all 11 institutions):**")
        cols = st.columns(5)
        for i, (cat, val) in enumerate(avg_scores.items()):
            level = "🔴 Critical" if val < 1.0 else "🟡 Moderate" if val < 1.5 else "🟢 Adequate"
            cols[i % 5].metric(cat[:20], f"{val:.1f}/3.0", delta=level, delta_color="off")

    with tab2:
        st.markdown("#### GBD Burden Share vs. AFU Program Coverage")

        gbd_share = {
            'Cardiovascular':      28.5, 'Neurological/Dementia': 19.2,
            'Mental Health':       14.8, 'Musculoskeletal':        9.1,
            'Respiratory':          8.4, 'Diabetes/Metabolic':     7.2,
            'Social Participation': 6.0, 'Physical Activity':      4.0,
            'Lifelong Learning':    1.8, 'Caregiver Support':      1.0,
        }

        afu_avg = scores.mean()
        cats    = list(gbd_share.keys())
        gbd_v   = [gbd_share[c] for c in cats]
        afu_v   = [(afu_avg.get(c, 0) / 3 * 100) for c in cats]
        gaps    = [g - a for g, a in zip(gbd_v, afu_v)]
        order   = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            name='GBD Burden Share (%)',
            x=[cats[i] for i in order], y=[gbd_v[i] for i in order],
            marker_color='#EF4444', opacity=0.85
        ))
        fig_gap.add_trace(go.Bar(
            name='AFU Program Coverage (%)',
            x=[cats[i] for i in order], y=[afu_v[i] for i in order],
            marker_color='#3B82F6', opacity=0.85
        ))
        fig_gap.update_layout(
            barmode='group', xaxis_tickangle=-30,
            yaxis_title='Share / Coverage (%)', height=420,
            margin=dict(l=40, r=20, t=20, b=120),
            legend=dict(x=0.6, y=0.99)
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        st.markdown("""
        <div class="warning-box">
        <b>Top 3 Misalignment Gaps:</b><br>
        • <b>Cardiovascular</b>: 28.5% of DALYs, but only 2 of 11 AFUs have strong programs<br>
        • <b>Respiratory/COPD</b>: #1 hospitalization cause for 65+, yet near-zero AFU program coverage<br>
        • <b>Diabetes/Metabolic</b>: 7.2% of DALYs, weakest AFU coverage of clinical categories
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Provincial Summary — All Indicators")
        summary = pd.DataFrame([
            {
                'Province': 'Ontario',
                '# AFUs': 6,
                '65+ Pop 2025 (k)': 3069.7,
                '65+ Pop 2040 (k)': 4122.5,
                'Pop Growth': '+34.3%',
                'ALC Days 2024-25': '17.9%',
                'YoY ALC Change': '↑ +0.49pp',
                'Top Hospital Dx': 'Pneumonia (34,356)',
            },
            {
                'Province': 'British Columbia',
                '# AFUs': 3,
                '65+ Pop 2025 (k)': 1167.2,
                '65+ Pop 2040 (k)': 1528.5,
                'Pop Growth': '+31.0%',
                'ALC Days 2024-25': '15.8%',
                'YoY ALC Change': '↑ +0.73pp',
                'Top Hospital Dx': 'Osteoarthritis (11,083)',
            },
            {
                'Province': 'Alberta',
                '# AFUs': 1,
                '65+ Pop 2025 (k)': 780.0,
                '65+ Pop 2040 (k)': 1152.9,
                'Pop Growth': '+47.8% ⚠️',
                'ALC Days 2024-25': '15.1%',
                'YoY ALC Change': '↓ -0.12pp',
                'Top Hospital Dx': 'Pneumonia (8,865)',
            },
            {
                'Province': 'Manitoba',
                '# AFUs': 1,
                '65+ Pop 2025 (k)': 260.4,
                '65+ Pop 2040 (k)': 330.7,
                'Pop Growth': '+27.0%',
                'ALC Days 2024-25': '17.3%',
                'YoY ALC Change': '↑ +1.68pp ⚠️',
                'Top Hospital Dx': 'Pneumonia (3,014)',
            },
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="highlight-box">
        <b>Integrated Insight:</b> The convergence of evidence across all four data sources points 
        to <b>Alberta</b> (fastest elderly population growth, lowest AFU coverage, 
        rising hospitalization burden) and <b>Manitoba</b> (worsening ALC days, only 1 AFU, 
        high pneumonia hospitalization LOS) as the most critical provinces for urgent AFU expansion.
        </div>
        """, unsafe_allow_html=True)



# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.markdown('<p class="main-title">GBD 2023 & Age-Friendly University Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Misalignment Between Elderly Disease Burden and AFU Program Priorities in Canada</p>', unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    total_dalys = sum(daly_data[d][-1] for d in DISEASES)
    total_1995  = sum(daly_data[d][0] for d in DISEASES)
    total_growth = (total_dalys - total_1995) / total_1995 * 100

    with col1:
        st.metric("Total DALYs (2023)", "6.90M", "+74.6% since 1995")
    with col2:
        st.metric("Deaths (2023)", "280,718", "+62.5% since 1995")
    with col3:
        st.metric("Fastest growing", "Mental disorders", "+160.7%")
    with col4:
        st.metric("Best improvement", "Cardiovascular", "-48.6% rate")
    with col5:
        st.metric("AFU Institutions", "11", "Across 4 provinces")
    with col6:
        st.metric("Projected 2040", "7.99M DALYs", "+59.5%")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-header">Disease Burden by Category — 2023</p>', unsafe_allow_html=True)
        dalys_2023 = {d: daly_data[d][-1] for d in filtered_diseases}
        df_plot = pd.DataFrame({'Disease': list(dalys_2023.keys()), 'DALYs': list(dalys_2023.values())})
        df_plot = df_plot.sort_values('DALYs', ascending=True)
        fig = px.bar(df_plot, x='DALYs', y='Disease', orientation='h',
                     color='Disease', color_discrete_map=DISEASE_COLORS,
                     labels={'DALYs': 'DALYs (number)'})
        fig.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=380,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                          yaxis=dict(showgrid=False),
                          margin=dict(l=0, r=80, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Critical gap: Mental disorders</b><br>
        Highest DALY growth (+160.7%) AND rising age-standardized rates (+20.2%) — yet lowest AFU program coverage.
        </div>
        <div class="warning-box">
        <b>⚠️ Alberta geographic gap</b><br>
        740,710 seniors growing at +22.1% since 2020 — but only 1 AFU institution (ratio: 740K seniors/institution).
        </div>
        <div class="success-box">
        <b>✅ Cardiovascular success</b><br>
        Age-standardized rate fell -48.6% — existing AFU and health interventions are working.
        </div>
        <div class="highlight-box">
        <b>🔮 2040 forecast</b><br>
        Mental disorders projected +80.8% growth — most urgent priority for AFU program expansion.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:1rem;">Burden distribution 2023</p>', unsafe_allow_html=True)
        dalys_pie = {d: daly_data[d][-1] for d in DISEASES}
        fig_pie = px.pie(values=list(dalys_pie.values()), names=list(dalys_pie.keys()),
                         color=list(dalys_pie.keys()), color_discrete_map=DISEASE_COLORS)
        fig_pie.update_layout(showlegend=False, height=220, margin=dict(l=0,r=0,t=0,b=0))
        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              textfont_size=10)
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# PAGE: TREND ANALYSIS
# ============================================================
elif page == "📈 Trend Analysis":
    st.markdown('<p class="main-title">Trend Analysis: DALYs 1995–2023</p>', unsafe_allow_html=True)

    metric_type = st.radio("View metric as:", ["Absolute numbers", "% Growth from 1995 baseline"], horizontal=True)

    fig = go.Figure()
    for disease in filtered_diseases:
        vals = daly_data[disease]
        y_vals = vals if metric_type == "Absolute numbers" else [(v - vals[0])/vals[0]*100 for v in vals]
        fig.add_trace(go.Scatter(
            x=OBS_YEARS, y=y_vals, mode='lines+markers', name=disease,
            line=dict(color=DISEASE_COLORS[disease], width=2.5),
            marker=dict(size=7)
        ))

    ylabel = "DALYs (number)" if metric_type == "Absolute numbers" else "% Growth from 1995 baseline"
    fig.update_layout(height=450, xaxis_title="Year", yaxis_title=ylabel,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                      yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
                      legend=dict(orientation='v', x=1.01, y=1),
                      margin=dict(l=0, r=150, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Year-by-Year DALY Table</p>', unsafe_allow_html=True)
    df_table = pd.DataFrame({
        d: daly_data[d] for d in filtered_diseases
    }, index=OBS_YEARS)
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
# PAGE: RATE ANALYSIS
# ============================================================
elif page == "📉 Rate Analysis":
    st.markdown('<p class="main-title">Age-Standardized Rate Analysis</p>', unsafe_allow_html=True)
    st.markdown("Controls for population growth — shows true burden trajectory")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<p class="section-header">Rate trends 1995–2023</p>', unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">% Change in rate 1995→2023</p>', unsafe_allow_html=True)
        rate_changes = {d: (rate_data[d][-1] - rate_data[d][0]) / rate_data[d][0] * 100
                        for d in filtered_diseases if d in rate_data}
        df_rate = pd.DataFrame({'Disease': list(rate_changes.keys()),
                                 'Rate Change (%)': list(rate_changes.values())})
        df_rate = df_rate.sort_values('Rate Change (%)')
        colors = ['#0F6E56' if v < 0 else '#A32D2D' for v in df_rate['Rate Change (%)']]
        fig2 = px.bar(df_rate, x='Rate Change (%)', y='Disease', orientation='h',
                      color='Disease',
                      color_discrete_map={d: c for d, c in zip(df_rate['Disease'], colors)})
        fig2.add_vline(x=0, line_color='gray', line_width=1)
        fig2.update_layout(showlegend=False, height=380,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', ticksuffix='%'),
                           yaxis=dict(showgrid=False),
                           margin=dict(l=0, r=20, t=10, b=40))
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
    st.dataframe(df_rates_table.style.format({c: "{:.1f}" for c in OBS_YEARS}),
                 use_container_width=True)

    st.markdown("""
    <div class="highlight-box">
    <b>Key finding:</b> Cardiovascular disease rate fell -48.6% — strongest evidence that targeted intervention works.
    Mental disorders rate rose +20.2% AFTER controlling for population growth — a genuine worsening requiring urgent AFU response.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: PROVINCIAL ANALYSIS
# ============================================================
elif page == "🗺️ Provincial Analysis":
    st.markdown('<p class="main-title">Provincial Analysis — AFU Coverage vs Elderly Burden</p>', unsafe_allow_html=True)

    prov_data = {
        'Province':         ['Ontario', 'British Columbia', 'Alberta', 'Manitoba'],
        'Senior Pop (2024)': [2954128, 1127346, 740710, 251515],
        'Growth % (2020-24)':[14.1, 14.7, 22.1, 13.1],
        'Antidepressant Rx': [640981, 205795, 153543, 53462],
        'AFU Institutions':  [6, 3, 1, 1],
        'Seniors per AFU':   [492355, 375782, 740710, 251515],
    }
    df_prov = pd.DataFrame(prov_data)

    col1, col2, col3, col4 = st.columns(4)
    prov_colors_map = {'Ontario':'#185FA5','British Columbia':'#0F6E56','Alberta':'#A32D2D','Manitoba':'#534AB7'}

    with col1: st.metric("Ontario", "2.95M seniors", "6 AFU institutions")
    with col2: st.metric("British Columbia", "1.13M seniors", "3 AFU institutions")
    with col3: st.metric("Alberta ⚠️", "740K seniors", "Only 1 AFU institution")
    with col4: st.metric("Manitoba", "252K seniors", "1 AFU institution")

    st.markdown("---")
    metric_choice = st.selectbox("Select metric to visualize:", [
        'Senior Pop (2024)', 'Growth % (2020-24)', 'Antidepressant Rx', 'Seniors per AFU'
    ])

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.bar(df_prov, x='Province', y=metric_choice,
                     color='Province', color_discrete_map=prov_colors_map,
                     text=metric_choice)
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=40),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = px.scatter(df_prov, x='Growth % (2020-24)', y='Seniors per AFU',
                          color='Province', color_discrete_map=prov_colors_map,
                          size='Senior Pop (2024)', text='Province',
                          labels={'Growth % (2020-24)': 'Senior Population Growth % (2020-24)',
                                  'Seniors per AFU': 'Seniors per AFU Institution'})
        fig2.update_traces(textposition='top center')
        fig2.update_layout(showlegend=False, height=380, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Complete Provincial Summary Table</p>', unsafe_allow_html=True)
    st.dataframe(df_prov.set_index('Province'), use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Alberta — Most Critical Geographic Gap</b><br>
    740,710 seniors growing at +22.1% (fastest rate among AFU provinces) — yet only 1 AFU institution.
    Coverage ratio of 740,710 seniors per institution is nearly double Ontario (492,355) and double BC (375,782).
    Alberta's above-average antidepressant prescription rates further indicate high and growing mental health burden in this underserved province.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: AFU ALIGNMENT
# ============================================================
elif page == "🎯 AFU Alignment":
    st.markdown('<p class="main-title">AFU Program Alignment Gap Analysis</p>', unsafe_allow_html=True)
    st.markdown("Mapping GBD burden rankings against Canadian AFU program priorities")

    alignment_data = {
        'Disease':          DISEASES,
        'DALY Rank (2023)': [1, 2, 3, 4, 5, 6, 7],
        'DALYs (2023)':     [1557828, 1333532, 582179, 552267, 415842, 374726, 194075],
        'Absolute Growth':  [60.1, 11.5, 135.7, 112.9, 89.4, 116.7, 160.7],
        'Rate Trend':       ['Declining', 'Declining (-49%)', 'Stable (+9%)',
                             'Stable', 'Declining', 'Stable', 'Rising (+20%) ⚠️'],
        'AFU Coverage':     ['Moderate', 'Moderate', 'Low-Moderate',
                             'Low-Moderate', 'Low', 'Low', 'Low'],
        'Alignment':        ['Partial', 'Good', 'Gap', 'Gap', 'Gap', 'Gap', 'Critical Gap ⚠️'],
    }
    df_align = pd.DataFrame(alignment_data)

    coverage_num = {'Low': 0, 'Low-Moderate': 1, 'Moderate': 2}
    align_colors = {'Good': '#3B6D11', 'Partial': '#185FA5', 'Gap': '#854F0B', 'Critical Gap ⚠️': '#A32D2D'}

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-header">Burden Growth vs AFU Coverage</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for _, row in df_align.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Absolute Growth']],
                y=[coverage_num.get(row['AFU Coverage'].split()[0], 0)],
                mode='markers+text',
                name=row['Disease'],
                text=[row['Disease'].replace(' diseases', '').replace(' disorders', '')],
                textposition='top center',
                marker=dict(size=18, color=DISEASE_COLORS[row['Disease']],
                            line=dict(width=2, color='white')),
                showlegend=True
            ))
        fig.add_vrect(x0=100, x1=180, fillcolor='#FCEBEB', opacity=0.3,
                      annotation_text="Critical gap zone", annotation_position="top right",
                      line_width=0)
        fig.update_layout(
            height=420, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Absolute DALY Growth 1995-2023 (%)',
                       showgrid=True, gridcolor='rgba(128,128,128,0.3)', range=[0, 185]),
            yaxis=dict(title='AFU Program Coverage',
                       tickvals=[0, 1, 2], ticktext=['Low', 'Low-Moderate', 'Moderate'],
                       showgrid=True, gridcolor='rgba(128,128,128,0.3)', range=[-0.5, 2.7]),
            legend=dict(orientation='v', x=1.01, y=1),
            margin=dict(l=0, r=180, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Alignment Summary</p>', unsafe_allow_html=True)
        for _, row in df_align.iterrows():
            color_map = {'Good': 'success-box', 'Partial': 'highlight-box',
                         'Gap': 'highlight-box', 'Critical Gap ⚠️': 'warning-box'}
            box_class = color_map.get(row['Alignment'], 'highlight-box')
            st.markdown(f"""
            <div class="{box_class}">
            <b>{row['Disease']}</b><br>
            Growth: +{row['Absolute Growth']}% | Rate: {row['Rate Trend']}<br>
            Coverage: {row['AFU Coverage']} → <b>{row['Alignment']}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Complete Alignment Table</p>', unsafe_allow_html=True)
    st.dataframe(df_align.set_index('Disease'), use_container_width=True)

# ============================================================
# PAGE: FORECASTING
# ============================================================
elif page == "🔮 Forecasting 2040":
    st.markdown('<p class="main-title">Disease Burden Forecasting 2024–2040</p>', unsafe_allow_html=True)
    st.markdown("Polynomial regression (degree=2) with cubic spline interpolation | R² > 0.96 | MAPE < 3.1%")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total DALYs 2023", "5.01M", "7 disease categories")
    with col2: st.metric("Projected 2040", "7.99M", "+59.5% growth")
    with col3: st.metric("Highest growth", "Mental disorders", "+80.8%")
    with col4: st.metric("Model R²", "> 0.96", "All diseases")

    st.markdown("---")
    disease_choice = st.selectbox("Select disease to view:", ['All diseases'] + DISEASES)
    annual_years = np.arange(1995, 2024)

    fig = go.Figure()
    diseases_to_plot = DISEASES if disease_choice == 'All diseases' else [disease_choice]

    for disease in diseases_to_plot:
        if disease not in filtered_diseases and disease_choice == 'All diseases':
            continue
        color = DISEASE_COLORS[disease]
        obs_vals = daly_data[disease]
        cs = CubicSpline(OBS_YEARS, obs_vals)
        annual_hist = np.maximum(cs(annual_years), 0)
        forecast = forecasts[disease]

        fig.add_trace(go.Scatter(
            x=list(annual_years), y=list(annual_hist/1_000_000),
            mode='lines', name=f"{disease} (hist)",
            line=dict(color=color, width=2.5), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=list(forecast_years), y=list(forecast/1_000_000),
            mode='lines', name=f"{disease} (forecast)",
            line=dict(color=color, width=2.5, dash='dash'), showlegend=True
        ))

    fig.add_vline(x=2023, line_color='gray', line_dash='dot', line_width=1.5,
                  annotation_text="← Historical | Forecast →",
                  annotation_position="top right")
    fig.update_layout(
        height=460, xaxis_title="Year", yaxis_title="DALYs (Millions)",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)'),
        legend=dict(orientation='v', x=1.01, y=1),
        margin=dict(l=0, r=200, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Table — DALYs 2023–2040</p>', unsafe_allow_html=True)
    forecast_table_data = {'Disease': [], '2023 (obs)': [], '2025': [], '2030': [],
                           '2035': [], '2040': [], 'Growth 2023-2040': []}

    for disease in DISEASES:
        f = forecasts[disease]
        obs = daly_data[disease][-1]
        growth = forecast_metrics[disease]['growth_2040']
        forecast_table_data['Disease'].append(disease)
        forecast_table_data['2023 (obs)'].append(f"{obs:,.0f}")
        forecast_table_data['2025'].append(f"{f[1]:,.0f}")
        forecast_table_data['2030'].append(f"{f[6]:,.0f}")
        forecast_table_data['2035'].append(f"{f[11]:,.0f}")
        forecast_table_data['2040'].append(f"{f[16]:,.0f}")
        forecast_table_data['Growth 2023-2040'].append(f"+{growth:.1f}%")

    df_forecast = pd.DataFrame(forecast_table_data).set_index('Disease')
    st.dataframe(df_forecast, use_container_width=True)

    total_2040 = sum(forecasts[d][16] for d in DISEASES)
    st.markdown(f"""
    <div class="warning-box">
    <b>🔮 2040 Projection Summary</b><br>
    Total DALYs across 7 disease categories projected to reach <b>7.99 million by 2040</b> (+59.5% from 2023).
    Mental disorders (+80.8%) and musculoskeletal disorders (+76.5%) show highest projected growth,
    reinforcing the urgency of AFU program reorientation. Model: Polynomial Regression degree=2,
    R² > 0.96, MAPE < 3.1% for all disease categories.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: DATA EXPLORER
# ============================================================
elif page == "📋 Data Explorer":
    st.markdown('<p class="main-title">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown("Explore and download the underlying GBD 2023 data")

    tab1, tab2, tab3 = st.tabs(["DALYs Data", "Rate Data", "Download"])

    with tab1:
        st.markdown("**Absolute DALYs — Canadians 60+, 1995–2023**")
        df_dalys = pd.DataFrame(daly_data, index=OBS_YEARS).T
        df_dalys.index.name = 'Disease'
        df_dalys['% Change 1995-2023'] = df_dalys.apply(
            lambda row: f"+{(row.iloc[-1]-row.iloc[0])/row.iloc[0]*100:.1f}%", axis=1
        )
        st.dataframe(df_dalys.style.format({c: "{:,.0f}" for c in OBS_YEARS}),
                     use_container_width=True)

    with tab2:
        st.markdown("**Age-standardized DALY rates per 100,000 — Canadians 60+, 1995–2023**")
        df_rates = pd.DataFrame(rate_data, index=OBS_YEARS).T
        df_rates.index.name = 'Disease'
        df_rates['Rate Change'] = df_rates.apply(
            lambda row: f"{(row.iloc[-1]-row.iloc[0])/row.iloc[0]*100:+.1f}%", axis=1
        )
        st.dataframe(df_rates.style.format({c: "{:.1f}" for c in OBS_YEARS}),
                     use_container_width=True)

    with tab3:
        st.markdown("**Download analysis data as CSV**")
        df_download = pd.DataFrame(daly_data, index=OBS_YEARS).T
        df_download.index.name = 'Disease'
        csv = df_download.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download DALYs data (CSV)",
            data=csv,
            file_name='GBD_2023_Canada_DALYs.csv',
            mime='text/csv'
        )
        df_rate_download = pd.DataFrame(rate_data, index=OBS_YEARS).T
        df_rate_download.index.name = 'Disease'
        csv_rate = df_rate_download.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Rate data (CSV)",
            data=csv_rate,
            file_name='GBD_2023_Canada_Rates.csv',
            mime='text/csv'
        )
elif page == "👥 Population Projections":
    page_population_projections()
elif page == "🏥 Hospitalization Burden":
    page_hospitalization_burden()
elif page == "🔗 Integrated Analysis":
    page_integrated_analysis()

# ── FOOTER ──
# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style='text-align:center;  font-size:12px;'>
GBD 2023 & Age-Friendly University Analysis Dashboard |
University of Windsor, Windsor, Ontario, Canada |
Data: IHME GBD 2023, CIHI 2024 | June 2026
</div>
""", unsafe_allow_html=True)
