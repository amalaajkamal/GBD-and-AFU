import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from collections import Counter

st.set_page_config(
    page_title="AFU Global Network Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS for Impact Map page ─────────────────────────────────────
st.markdown("""
<style>
.impact-header {
    background: #0d1b2a;
    color: #00d4ff;
    padding: 8px 16px;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 4px;
    margin-bottom: 8px;
}
.stat-card {
    background: #1a2744;
    border: 1px solid #2e4a8a;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
}
.stat-label {
    font-size: 0.75rem;
    color: #8899bb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.region-btn {
    background: #1a2744;
    color: #cce4ff;
    border: 1px solid #2e4a8a;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 3px 0;
    width: 100%;
    text-align: left;
    font-size: 0.85rem;
    cursor: pointer;
}
.region-btn-active {
    background: #2e4a8a;
    color: #ffffff;
    border: 1px solid #00d4ff;
}
.country-item {
    color: #FF9800;
    font-size: 0.82rem;
    padding: 4px 0 4px 12px;
    border-left: 2px solid #FF9800;
    margin: 3px 0;
    cursor: pointer;
}
.section-dark {
    background: #0d1b2a;
    border-radius: 8px;
    padding: 12px;
    margin: 4px 0;
}
.overview-title {
    background: #1a2744;
    color: #ffffff;
    text-align: center;
    padding: 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_country_data():
    return pd.DataFrame([
        ("United States","North America",109,37.09,-95.71),
        ("Canada","North America",11,56.13,-106.35),
        ("Ireland","Europe",9,53.41,-8.24),
        ("United Kingdom","Europe",2,55.37,-3.43),
        ("Portugal","Europe",2,39.39,-8.22),
        ("Spain","Europe",2,40.46,-3.74),
        ("Croatia","Europe",1,45.10,15.20),
        ("Czech Republic","Europe",1,49.81,15.47),
        ("Hungary","Europe",1,47.16,19.50),
        ("Israel","Europe",1,31.04,34.85),
        ("Slovakia","Europe",1,48.66,19.69),
        ("Slovenia","Europe",1,46.15,14.99),
        ("Switzerland","Europe",1,46.81,8.22),
        ("South Korea","Asia",4,35.90,127.76),
        ("China","Asia",1,35.86,104.19),
        ("Philippines","Asia",1,12.87,121.77),
        ("Hong Kong SAR","Asia",1,22.39,114.10),
        ("Australia","Oceania",2,-25.27,133.77),
        ("Brazil","South America",3,-14.23,-51.92),
        ("Chile","South America",2,-35.67,-71.54),
    ], columns=["Country","Region","AFU_Members","Latitude","Longitude"])

@st.cache_data
def load_regional_data():
    return pd.DataFrame([
        ("North America",2,23,120),
        ("Europe",13,44,22),
        ("Asia",4,48,7),
        ("Oceania",1,14,2),
        ("South America",2,12,5),
    ], columns=["Region","Countries_in_AFU","Total_Countries","AFU_Institutions"])

@st.cache_data
def load_principles_data():
    return pd.DataFrame([
        (1,"P1: Encourage participation\nof older adults",18,72.0,"Well Implemented"),
        (2,"P2: Personal & career\ndevelopment",8,32.0,"Moderately Implemented"),
        (3,"P3: Recognize educational\nneeds",7,28.0,"Moderately Implemented"),
        (4,"P4: Intergenerational\nlearning",14,56.0,"Well Implemented"),
        (5,"P5: Online access for\nolder adults",4,16.0,"Underimplemented"),
        (6,"P6: Research agenda\ninformed by aging",12,48.0,"Well Implemented"),
        (7,"P7: Student understanding\nof longevity",4,16.0,"Underimplemented"),
        (8,"P8: Health, wellness &\ncultural access",12,48.0,"Well Implemented"),
        (9,"P9: Engage retired\ncommunity",7,28.0,"Moderately Implemented"),
        (10,"P10: Dialogue with aging\norganizations",5,20.0,"Underimplemented"),
    ], columns=["Principle_Number","Short_Label","Mentions","Pct","Gap_Flag"])

@st.cache_data
def load_best_practices():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Form_Data_Entry-Grid_view.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

# Institution list by country
INSTITUTIONS = {
    "United States": ["University of Minnesota","University of Massachusetts Boston","Arizona State University","Duke University","Rochester Institute of Technology","University of North Carolina Wilmington","University of Michigan","Middle Tennessee State University","University of South Florida","University of New Hampshire","University of Arizona","California State University San Bernardino","California State University Fullerton","California State University Long Beach","Dominican University of California","Fielding Graduate University","Los Angeles Pierce College","Palo Alto University","San Diego State University","Santa Monica College","UCLA","UC Berkeley","University of San Francisco","University of Southern California","University of the Pacific","Colorado State University","University of Colorado Denver","University of Colorado Anschutz","University of Colorado Colorado Springs","Central Connecticut State University","Goodwin University","Quinnipiac University","University of Bridgeport","University of Connecticut","University of Hartford","Florida Atlantic University","Florida State University","Eckerd College","St. Thomas University","Georgia State University","University of North Georgia","University of Hawaii at Manoa","Northeastern Illinois University","University of Illinois Urbana-Champaign","Concordia University Chicago","Purdue University","University of Indianapolis","Wichita State University","Frontier Nursing University","Northern Kentucky University","Western Kentucky University","Franciscan Missionaries of Our Lady University","University of Maine","University of New England","Towson University","University of Maryland Baltimore","University of Maryland Baltimore County","Lasell University","UMass Amherst","UMass Dartmouth","UMass Lowell","UMass Medical School","Springfield College","William James College","Eastern Michigan University","Michigan State University","Wayne State University","University of Minnesota Duluth","University of St Thomas","St Catherine University","St Cloud State University","Mississippi State University","University of Mississippi","Missouri State University","Washington University in St Louis","University of Montana","University of Nebraska at Omaha","Fairleigh Dickinson University","Stockton University","Hofstra University","Hunter College CUNY","Ithaca College","Purchase College SUNY","Cleveland State University","Miami University","University of Akron","University of Cincinnati","University of Central Oklahoma","Portland State University","Southern Oregon University","Western Oregon University","Drexel University","Pennsylvania State University","University of Rhode Island","East Tennessee State University","Tennessee State University","University of Texas at Austin","University of Utah","University of Vermont","Virginia Commonwealth University","Shepherd University","West Virginia University","University of Wisconsin La Crosse","University of Wisconsin Green Bay","University of Wisconsin Superior"],
    "Canada": ["University of Calgary","Kwantlen Polytechnic University","UBC Okanagan","University of the Fraser Valley","Niagara College","McMaster University","Toronto Metropolitan University","Trent University","Ontario Tech University","University of Windsor","University of Manitoba"],
    "Ireland": ["Atlantic Technological University","Dublin City University","Mary Immaculate College","Munster Technological University","National College of Ireland","Royal College of Surgeons Ireland","Trinity College Dublin","University College Dublin","University of Limerick"],
    "United Kingdom": ["University of Strathclyde","Ulster University"],
    "Portugal": ["ISEG — Lisbon School of Economics","Escola Superior de Saude de Santa Maria"],
    "Spain": ["University of Murcia","Universitat Internacional de Catalunya"],
    "Croatia": ["University of Rijeka"],
    "Czech Republic": ["Masaryk University"],
    "Hungary": ["John von Neumann University"],
    "Israel": ["University of Haifa"],
    "Slovakia": ["Comenius University Bratislava"],
    "Slovenia": ["University of Maribor"],
    "Switzerland": ["University of Zurich"],
    "South Korea": ["Chosun University","Paichai University","Yonsei University","Kyung Hee University"],
    "China": ["Open University of China (Seniors University of China)"],
    "Philippines": ["University of the Philippines"],
    "Hong Kong SAR": ["Chinese University of Hong Kong"],
    "Australia": ["University of Queensland","University of the Sunshine Coast"],
    "Brazil": ["Pontifical Catholic University of Campinas","Federal University of Technology Parana","Universidade Federal de Vicosa"],
    "Chile": ["Instituto Profesional AIEP","University of Talca"],
}

REGION_COLORS = {
    "North America": "#E63946",
    "Europe": "#2196F3",
    "Asia": "#FF9800",
    "Oceania": "#9C27B0",
    "South America": "#00BCD4",
}
GAP_COLORS = {
    "Well Implemented": "#27AE60",
    "Moderately Implemented": "#F39C12",
    "Underimplemented": "#E74C3C",
}

# ── Session state ──────────────────────────────────────────────────────────
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 AFU GN Dashboard")
    page = st.radio("Navigate", [
        "🌍 Global Overview",
        "📐 Principle Gap Analysis",
        "🗺️ Regional Equity",
        "📋 Best Practices Explorer",
        "🌐 Impact Map",
    ])
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- AFU GN Website (June 2026)\n- AFU Best Practices Database\n- UNESCO UIS 2024\n- National HEI Registries")
    st.markdown("---")
    st.caption("Paper: *Implementation Gap Analysis of the AFU Global Network*\nGenerations at Work, DCU, Oct 2026")

df_country = load_country_data()
df_regional = load_regional_data()
df_regional["Countries_Missing"] = df_regional["Total_Countries"] - df_regional["Countries_in_AFU"]
df_regional["Country_Coverage_Pct"] = (df_regional["Countries_in_AFU"] / df_regional["Total_Countries"] * 100).round(1)
df_principles = load_principles_data()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🌍 Global Overview":

    # ── GeoPulse-style header ──────────────────────────────────────────────
    st.markdown("""
    <div style="background:#050d1a; padding:8px 0 4px 0;">
        <span style="color:#4FC3F7; font-size:1.15rem; font-weight:800; letter-spacing:0.06em;">
            🌍 AFU GLOBAL NETWORK — IMPLEMENTATION GAP ANALYSIS
        </span>
        <span style="color:#37474F; font-size:0.8rem; margin-left:12px;">
            Geographic & Thematic Analysis • June 2026
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Compact KPI row ────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; gap:8px; margin:6px 0; flex-wrap:nowrap;">
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #4FC3F7;">
            <div style="color:#4FC3F7; font-size:1.5rem; font-weight:800;">156</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">Member Institutions</div>
        </div>
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #E63946;">
            <div style="color:#E63946; font-size:1.5rem; font-weight:800;">77%</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">North America Share</div>
        </div>
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #27AE60;">
            <div style="color:#27AE60; font-size:1.5rem; font-weight:800;">20</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">Countries</div>
        </div>
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #FF9800;">
            <div style="color:#FF9800; font-size:1.5rem; font-weight:800;">25</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">Best Practices</div>
        </div>
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #EF5350;">
            <div style="color:#EF5350; font-size:1.5rem; font-weight:800;">16%</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">P5 & P7 Rate</div>
        </div>
        <div style="background:#0a1628; border:1px solid #0d2137; border-radius:6px;
                    padding:8px 16px; flex:1; text-align:center; border-top:2px solid #9C27B0;">
            <div style="color:#9C27B0; font-size:1.5rem; font-weight:800;">11%</div>
            <div style="color:#546E7A; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;">Submission Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Region filter state ────────────────────────────────────────────────
    if "ov_region" not in st.session_state:
        st.session_state.ov_region = "Global View"

    region_tabs = {
        "Global View": (156, "#4FC3F7"),
        "North America": (120, "#E63946"),
        "Europe": (22, "#2196F3"),
        "Asia": (7, "#FF9800"),
        "South America": (5, "#00BCD4"),
        "Oceania": (2, "#9C27B0"),
    }

    # ── Map ────────────────────────────────────────────────────────────────
    sel = st.session_state.ov_region
    if sel == "Global View":
        map_df = df_country.copy()
        map_df["opacity"] = 1.0
    else:
        map_df = df_country.copy()
        map_df["opacity"] = map_df["Region"].apply(lambda x: 1.0 if x == sel else 0.12)

    fig_ov = go.Figure()

    for region in df_country["Region"].unique():
        rdf = map_df[map_df["Region"] == region]
        color = REGION_COLORS.get(region, "#888")
        opacity = float(rdf["opacity"].mean()) if len(rdf) > 0 else 1.0

        # Glow ring
        fig_ov.add_trace(go.Scattergeo(
            lat=rdf["Latitude"], lon=rdf["Longitude"],
            mode="markers", showlegend=False,
            marker=dict(size=rdf["AFU_Members"].apply(lambda x: max(10, min(55, x/2.2))),
                        color=color, opacity=opacity*0.2, line=dict(width=0)),
            hoverinfo="skip",
        ))

        # Main dot with count label
        fig_ov.add_trace(go.Scattergeo(
            lat=rdf["Latitude"], lon=rdf["Longitude"],
            mode="markers+text", name=region,
            marker=dict(size=rdf["AFU_Members"].apply(lambda x: max(8, min(45, x/2.5))),
                        color=color, opacity=opacity,
                        line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
            text=rdf["AFU_Members"].astype(str),
            textfont=dict(size=7, color="white", family="Arial Black"),
            textposition="middle center",
            customdata=rdf[["Country","AFU_Members","Penetration_Pct","Total_Universities"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>AFU Members: %{customdata[1]}<br>Penetration: %{customdata[2]}%<br>Total Universities: %{customdata[3]}<extra></extra>",
        ))

    fig_ov.update_layout(
        height=460, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
        geo=dict(
            showframe=False,
            showcoastlines=True, coastlinecolor="#0d2137",
            showland=True, landcolor="#0a1628",
            showocean=True, oceancolor="#050d1a",
            showlakes=False,
            showcountries=True, countrycolor="#0d2137", countrywidth=0.4,
            bgcolor="#050d1a", projection_type="natural earth",
            lataxis=dict(range=[-55, 80]),
        ),
        legend=dict(
            orientation="h", y=1.01, x=0.5, xanchor="center",
            font=dict(color="#546E7A", size=10),
            bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        font=dict(color="#546E7A"),
    )

    st.plotly_chart(fig_ov, use_container_width=True, config={"displayModeBar": False})

    # ── Region tabs at bottom (GeoPulse style) ─────────────────────────────
    tab_cols = st.columns(len(region_tabs))
    for i, (region, (count, color)) in enumerate(region_tabs.items()):
        with tab_cols[i]:
            is_active = st.session_state.ov_region == region
            bg = f"{color}33" if is_active else "#0a1628"
            border = f"2px solid {color}" if is_active else f"1px solid #0d2137"
            txt = color if is_active else "#546E7A"
            if st.button(
                f"{'🔵 ' if region=='Global View' else ''}{region}   {count}",
                key=f"ov_tab_{region}",
                use_container_width=True
            ):
                st.session_state.ov_region = region
                st.rerun()

    # ── Bottom row: donut + bar ────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="color:#4FC3F7; font-size:0.8rem; font-weight:700; letter-spacing:0.08em; margin-bottom:4px;">REGIONAL SHARE</div>', unsafe_allow_html=True)
        fig_donut = px.pie(df_regional, values="AFU_Institutions", names="Region",
                           color="Region", color_discrete_map=REGION_COLORS, hole=0.5)
        fig_donut.update_traces(textinfo="percent+label", textposition="outside",
                                textfont=dict(size=10))
        fig_donut.update_layout(height=300, showlegend=False,
                                paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
                                margin=dict(l=20,r=20,t=10,b=10),
                                font=dict(color="#90A4AE"))
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown('<div style="color:#4FC3F7; font-size:0.8rem; font-weight:700; letter-spacing:0.08em; margin-bottom:4px;">INSTITUTIONS PER REGION</div>', unsafe_allow_html=True)
        fig_bar = px.bar(df_regional.sort_values("AFU_Institutions"),
                         x="AFU_Institutions", y="Region",
                         color="Region", color_discrete_map=REGION_COLORS,
                         orientation="h", text="AFU_Institutions")
        fig_bar.update_traces(textposition="outside", textfont=dict(color="#90A4AE"))
        fig_bar.update_layout(height=300, showlegend=False,
                              paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
                              xaxis=dict(title="", color="#37474F", gridcolor="#0d2137"),
                              yaxis=dict(title="", color="#90A4AE"),
                              font=dict(color="#90A4AE"),
                              margin=dict(l=10,r=50,t=10,b=10))
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRINCIPLE GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📐 Principle Gap Analysis":
    st.title("📐 AFU Principle Implementation Gap Analysis")
    st.markdown("*Based on 25 Best Practice submissions from 17 institutions*")
    st.divider()

    c1, c2, c3 = st.columns(3)
    well = df_principles[df_principles["Gap_Flag"]=="Well Implemented"].shape[0]
    mod  = df_principles[df_principles["Gap_Flag"]=="Moderately Implemented"].shape[0]
    unde = df_principles[df_principles["Gap_Flag"]=="Underimplemented"].shape[0]
    c1.metric("✅ Well Implemented", f"{well} principles")
    c2.metric("⚠️ Moderately Implemented", f"{mod} principles")
    c3.metric("🔴 Underimplemented", f"{unde} principles")

    st.divider()
    st.subheader("Principle Citation Frequency (% of 25 submissions)")
    fig_p = px.bar(df_principles.sort_values("Pct"),
                   x="Pct", y="Short_Label", color="Gap_Flag",
                   color_discrete_map=GAP_COLORS, orientation="h", text="Pct",
                   labels={"Pct": "% of Submissions", "Short_Label": ""})
    fig_p.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_p.add_vline(x=50, line_dash="dot", line_color="gray",
                    annotation_text="50% threshold", annotation_position="top")
    fig_p.update_layout(height=480, xaxis=dict(range=[0,85]),
                        legend_title="Implementation Status",
                        margin=dict(l=10,r=60,t=20,b=20),
                        legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_p, use_container_width=True)
    st.info("💡 **Key Finding:** Principle 5 and Principle 7 are cited in only **16% of submissions** — the lowest of all 10 principles.")

    st.divider()
    st.subheader("Who Do Best Practice Activities Target?")
    try:
        df_bp = load_best_practices()
        audience_col = [c for c in df_bp.columns if "aimed at" in c.lower()][0]
        aud_counter = Counter()
        for val in df_bp[audience_col].dropna():
            for a in val.split(","):
                aud_counter[a.strip()] += 1
        df_aud = pd.DataFrame(aud_counter.items(), columns=["Audience","Count"]).sort_values("Count", ascending=True)
        fig_aud = px.bar(df_aud, x="Count", y="Audience", orientation="h",
                         color="Count", color_continuous_scale="Blues", text="Count")
        fig_aud.update_traces(textposition="outside")
        fig_aud.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=10,r=40,t=20,b=20))
        st.plotly_chart(fig_aud, use_container_width=True)
        st.warning("⚠️ Despite Principle 7 calling for increased student understanding of aging, dedicated student-only programming is rare.")
    except Exception:
        st.warning("Upload Form_Data_Entry-Grid_view.csv to the repo to enable audience analysis.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — REGIONAL EQUITY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Regional Equity":
    st.title("🗺️ Geographic Equity & Penetration Analysis")
    st.markdown("*Country coverage gaps and institutional adoption rates across AFU regions*")
    st.divider()

    tab1, tab2 = st.tabs(["Country Coverage Gap", "Penetration Rate by Country"])
    with tab1:
        st.subheader("Countries Represented vs. Total Countries Per Region")
        df_melt = df_regional.melt(id_vars="Region",
                                   value_vars=["Countries_in_AFU","Countries_Missing"],
                                   var_name="Type", value_name="Count")
        df_melt["Type"] = df_melt["Type"].map({"Countries_in_AFU":"In AFU GN","Countries_Missing":"Not in AFU GN"})
        fig_cov = px.bar(df_melt, x="Count", y="Region", color="Type", orientation="h",
                         color_discrete_map={"In AFU GN":"#2E6DA4","Not in AFU GN":"#D5E8F5"},
                         barmode="stack", text="Count")
        fig_cov.update_traces(textposition="inside")
        fig_cov.update_layout(height=400, xaxis_title="Number of Countries",
                              margin=dict(l=10,r=20,t=20,b=20),
                              legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_cov, use_container_width=True)
        st.info("💡 **Asia:** 48 countries, only 4 represented (8.3%). **Oceania:** 14 countries, only Australia represented (7.1% coverage).")
        df_display = df_regional[["Region","Countries_in_AFU","Total_Countries","Countries_Missing","Country_Coverage_Pct"]].copy()
        df_display.columns = ["Region","In AFU GN","Total Countries","Not Represented","Coverage %"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("AFU Adoption as % of National University System")
        fig_pen = px.bar(df_country.sort_values("Penetration_Pct", ascending=False),
                         x="Country", y="Penetration_Pct", color="Region",
                         color_discrete_map=REGION_COLORS,
                         text=df_country.sort_values("Penetration_Pct", ascending=False)["Penetration_Pct"].apply(lambda x: f"{x}%"),
                         hover_data={"AFU_Members": True})
        fig_pen.update_traces(textposition="outside", textfont_size=10)
        fig_pen.update_layout(height=500, xaxis_tickangle=-45,
                              yaxis_title="Penetration Rate (%)", xaxis_title="",
                              margin=dict(l=10,r=10,t=20,b=140))
        st.plotly_chart(fig_pen, use_container_width=True)
        st.info("💡 **Ireland (34.6%)** highest penetration. **China (0.03%)** lowest.")
        st.subheader("AFU Members vs. National University System Size")
        fig_sc = px.scatter(df_country, x="Total_Universities", y="AFU_Members",
                            color="Region", size="Penetration_Pct", hover_name="Country",
                            color_discrete_map=REGION_COLORS, log_x=True, size_max=30,
                            labels={"Total_Universities":"Total Universities (log scale)",
                                    "AFU_Members":"AFU Member Institutions"})
        fig_sc.update_layout(height=420, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — BEST PRACTICES EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 Best Practices Explorer":
    st.title("📋 Best Practices Explorer")
    st.markdown("*All 25 submissions from the AFU GN Best Practices Database*")
    st.divider()
    try:
        df_bp = load_best_practices()
        pcol    = [c for c in df_bp.columns if "Principle(s)" in c][0]
        ucol    = [c for c in df_bp.columns if "Submitting University" in c][0]
        tcol    = [c for c in df_bp.columns if "Title" in c][0]
        typecol = [c for c in df_bp.columns if "Type of Activity" in c][0]

        def extract_principles(val):
            if pd.isna(val): return []
            nums = re.findall(r'Principle\s*(\d+)', str(val))
            return sorted(set(int(n) for n in nums))

        df_bp["Principles"] = df_bp[pcol].apply(extract_principles)
        df_bp["Principles_str"] = df_bp["Principles"].apply(lambda x: ", ".join(f"P{p}" for p in x))

        col1, col2 = st.columns(2)
        with col1:
            principle_filter = st.multiselect("Filter by Principle",
                options=list(range(1,11)), format_func=lambda x: f"Principle {x}", default=[])
        with col2:
            uni_filter = st.multiselect("Filter by University",
                options=sorted(df_bp[ucol].dropna().unique()), default=[])

        df_filtered = df_bp.copy()
        if principle_filter:
            df_filtered = df_filtered[df_filtered["Principles"].apply(
                lambda ps: any(p in ps for p in principle_filter))]
        if uni_filter:
            df_filtered = df_filtered[df_filtered[ucol].isin(uni_filter)]

        st.markdown(f"**Showing {len(df_filtered)} of 25 submissions**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ongoing Activities", int((df_filtered[typecol].str.contains("Ongoing", na=False)).sum()))
        c2.metric("One-Time Activities", int((df_filtered[typecol].str.contains("One-Time", na=False)).sum()))
        c3.metric("Unique Universities", int(df_filtered[ucol].nunique()))
        st.divider()

        if len(df_filtered) > 0:
            filtered_counter = Counter()
            for ps in df_filtered["Principles"]:
                for p in ps:
                    filtered_counter[p] += 1
            df_filt_p = pd.DataFrame([(f"P{p}", filtered_counter.get(p,0)) for p in range(1,11)],
                                     columns=["Principle","Count"])
            fig_fp = px.bar(df_filt_p, x="Principle", y="Count",
                            color="Count", color_continuous_scale="Blues",
                            text="Count", title="Principle Frequency in Selected Submissions")
            fig_fp.update_traces(textposition="outside")
            fig_fp.update_layout(height=300, showlegend=False,
                                 coloraxis_showscale=False, margin=dict(l=10,r=10,t=40,b=20))
            st.plotly_chart(fig_fp, use_container_width=True)

        purpose_col = [c for c in df_bp.columns if "primary purpose" in c.lower()][0]
        outcome_col = [c for c in df_bp.columns if "outcomes" in c.lower()][0]
        unique_col  = [c for c in df_bp.columns if "unique" in c.lower()][0]
        duration_col= [c for c in df_bp.columns if "How long" in c][0]

        for _, row in df_filtered.iterrows():
            with st.expander(f"📌 {row[tcol]} — *{row[ucol]}*"):
                c1, c2 = st.columns([2,1])
                with c1:
                    st.markdown(f"**Purpose:** {row.get(purpose_col,'N/A')}")
                    st.markdown(f"**Outcomes:** {row.get(outcome_col,'N/A')}")
                    st.markdown(f"**What makes it unique:** {row.get(unique_col,'N/A')}")
                with c2:
                    st.markdown(f"**Principles:** {row['Principles_str']}")
                    st.markdown(f"**Type:** {row.get(typecol,'N/A')}")
                    st.markdown(f"**Duration:** {row.get(duration_col,'N/A')}")

    except FileNotFoundError:
        st.error("⚠️ Form_Data_Entry-Grid_view.csv not found.")
    except Exception as e:
        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — IMPACT MAP (City Cancer Challenge Style)
# ══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Impact Map":

    st.markdown("""
    <div style="background:#0d1b2a; padding:10px 16px; border-radius:6px; margin-bottom:12px;">
        <span style="color:#00d4ff; font-size:1.2rem; font-weight:800;">🌐 AFU Global Network</span>
        <span style="color:#8899bb; font-size:0.9rem; margin-left:12px;">Impact Map — Select a Region, then a Country</span>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 3.5, 1.5])

    regions = ["All Regions"] + sorted(df_country["Region"].unique().tolist())

    with left:
        st.markdown('<div style="background:#0d1b2a; padding:10px; border-radius:8px;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#00d4ff; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; margin-bottom:8px;">SELECT REGION</div>', unsafe_allow_html=True)

        for region in [r for r in regions if r != "All Regions"]:
            color = REGION_COLORS.get(region, "#888")
            count = df_regional[df_regional["Region"]==region]["AFU_Institutions"].sum() if region in df_regional["Region"].values else 0
            is_active = st.session_state.selected_region == region
            if st.button(f"● {region}  ({count})", key=f"reg_{region}", use_container_width=True):
                if st.session_state.selected_region == region:
                    st.session_state.selected_region = None
                    st.session_state.selected_country = None
                else:
                    st.session_state.selected_region = region
                    st.session_state.selected_country = None
                st.rerun()

        st.markdown("---")

        if st.session_state.selected_region:
            st.markdown(f'<div style="color:#00d4ff; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; margin-bottom:8px;">COUNTRIES IN {st.session_state.selected_region.upper()}</div>', unsafe_allow_html=True)
            region_countries = df_country[df_country["Region"]==st.session_state.selected_region]["Country"].tolist()
            for country in region_countries:
                n = df_country[df_country["Country"]==country]["AFU_Members"].values[0]
                is_sel = st.session_state.selected_country == country
                label = f"{"▶ " if is_sel else "○ "}{country} ({n})"
                if st.button(label, key=f"cty_{country}", use_container_width=True):
                    if st.session_state.selected_country == country:
                        st.session_state.selected_country = None
                    else:
                        st.session_state.selected_country = country
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        if st.session_state.selected_country:
            map_df = df_country.copy()
            map_df["opacity"] = map_df["Country"].apply(lambda x: 1.0 if x == st.session_state.selected_country else 0.2)
        elif st.session_state.selected_region:
            map_df = df_country.copy()
            map_df["opacity"] = map_df["Region"].apply(lambda x: 1.0 if x == st.session_state.selected_region else 0.2)
        else:
            map_df = df_country.copy()
            map_df["opacity"] = 1.0

        fig_impact = go.Figure()

        for region in df_country["Region"].unique():
            rdf = map_df[map_df["Region"]==region]
            opacity_val = float(rdf["opacity"].mean()) if len(rdf) > 0 else 1.0
            color = REGION_COLORS.get(region, "#888888")

            fig_impact.add_trace(go.Scattergeo(
                lat=rdf["Latitude"], lon=rdf["Longitude"],
                mode="markers", name=region,
                marker=dict(
                    size=rdf["AFU_Members"].apply(lambda x: max(8, min(40, x/2.5))),
                    color=color,
                    opacity=opacity_val,
                    line=dict(width=1.5, color="white"),
                ),
                text=rdf["Country"],
                customdata=rdf[["AFU_Members"]].values,
                hovertemplate="<b>%{text}</b><br>AFU Members: %{customdata[0]}<extra></extra>",
            ))

        fig_impact.update_layout(
            height=480, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
            geo=dict(
                showframe=False,
                showcoastlines=True, coastlinecolor="#2e4a8a",
                showland=True, landcolor="#1a2744",
                showocean=True, oceancolor="#0d1b2a",
                showlakes=True, lakecolor="#0d1b2a",
                showcountries=True, countrycolor="#2e4a8a",
                countrywidth=0.5, bgcolor="#0d1b2a",
                projection_type="natural earth",
            ),
            legend=dict(
                orientation="h", y=-0.05,
                font=dict(color="#8899bb", size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(color="#8899bb"),
        )
        st.plotly_chart(fig_impact, use_container_width=True)

        if st.session_state.selected_region or st.session_state.selected_country:
            if st.button("↩ Return to Global View", use_container_width=False):
                st.session_state.selected_region = None
                st.session_state.selected_country = None
                st.rerun()

    with right:
        st.markdown('<div style="background:#0d1b2a; padding:12px; border-radius:8px;">', unsafe_allow_html=True)

        if st.session_state.selected_country:
            cdata = df_country[df_country["Country"]==st.session_state.selected_country].iloc[0]
            institutions = INSTITUTIONS.get(st.session_state.selected_country, [])

            st.markdown(f'<div class="overview-title">{st.session_state.selected_country.upper()}</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{int(cdata["AFU_Members"])}</div><div class="stat-label">AFU Members</div></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="stat-card"><div class="stat-number" style="font-size:1.1rem;">{cdata["Region"]}</div><div class="stat-label">Region</div></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f'<div style="color:#00d4ff; font-size:0.75rem; font-weight:700; letter-spacing:0.1em;">INSTITUTIONS ({len(institutions)})</div>', unsafe_allow_html=True)
            for inst in institutions:
                st.markdown(f'<div style="color:#cce4ff; font-size:0.72rem; padding:3px 0 3px 8px; border-left:2px solid #2196F3; margin:2px 0;">• {inst}</div>', unsafe_allow_html=True)

        elif st.session_state.selected_region:
            rdata = df_regional[df_regional["Region"]==st.session_state.selected_region]
            rcountries = df_country[df_country["Region"]==st.session_state.selected_region]
            region = st.session_state.selected_region
            total_inst = int(rdata["AFU_Institutions"].values[0]) if len(rdata) > 0 else 0
            countries_in = int(rdata["Countries_in_AFU"].values[0]) if len(rdata) > 0 else 0
            total_countries = int(rdata["Total_Countries"].values[0]) if len(rdata) > 0 else 0
            coverage = round(countries_in/total_countries*100, 1) if total_countries > 0 else 0

            st.markdown(f'<div class="overview-title">{region.upper()}</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{total_inst}</div><div class="stat-label">Institutions</div></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{countries_in}</div><div class="stat-label">Countries</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-number">{coverage}%</div><div class="stat-label">Country Coverage</div></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f'<div style="color:#00d4ff; font-size:0.75rem; font-weight:700; letter-spacing:0.1em; margin-bottom:6px;">COUNTRIES IN NETWORK</div>', unsafe_allow_html=True)
            for _, row in rcountries.iterrows():
                color = REGION_COLORS.get(region, "#888")
                st.markdown(f'<div style="color:#cce4ff; font-size:0.78rem; padding:4px 0 4px 8px; border-left:2px solid {color}; margin:3px 0;">● {row["Country"]} — {int(row["AFU_Members"])} members</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="overview-title">GLOBAL OVERVIEW</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="stat-card"><div class="stat-number">156</div><div class="stat-label">Institutions</div></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="stat-card"><div class="stat-number">20</div><div class="stat-label">Countries</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-card"><div class="stat-number">5</div><div class="stat-label">Regions</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-card"><div class="stat-number">11%</div><div class="stat-label">Best Practice Rate</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-card"><div class="stat-number">77%</div><div class="stat-label">North America Share</div></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="color:#00d4ff; font-size:0.75rem; font-weight:700; letter-spacing:0.1em;">KEY GAPS</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#FF9800; font-size:0.78rem; padding:3px 0;">• P5: Online access — 16%</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#FF9800; font-size:0.78rem; padding:3px 0;">• P7: Longevity dividend — 16%</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#E74C3C; font-size:0.78rem; padding:3px 0;">• Asia: 4 of 48 countries</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
