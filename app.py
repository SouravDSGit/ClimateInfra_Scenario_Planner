"""
ClimateInfra Scenario Planner
==============================
Generative AI-powered climate resilience scenario planning for
stormwater and road infrastructure. Combines real NOAA precipitation
data with local LLM inference (Ollama) to generate structured
engineering scenario reports.

Author: [Your Name]
Affiliation: [Your Institution]
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from dotenv import load_dotenv

from utils.noaa_client import NOAAClient
from utils.llm_client import OllamaClient
from utils.prompts import (
    build_scenario_prompt,
    SYSTEM_PROMPT,
    INFRA_OPTIONS,
    SCENARIO_TYPES,
    HORIZONS,
    OLLAMA_MODELS,
)

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateInfra Scenario Planner",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a4d7c;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #f0f6ff;
        border-left: 4px solid #1a4d7c;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .scenario-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .optimistic { background: #d4edda; color: #155724; }
    .moderate   { background: #fff3cd; color: #856404; }
    .severe     { background: #f8d7da; color: #721c24; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🔑 NOAA API Token")
    noaa_token = st.text_input(
        "Token",
        value=os.getenv("NOAA_CDO_TOKEN", ""),
        type="password",
        help="Free token from https://www.ncdc.noaa.gov/cdo-web/token",
    )
    if not noaa_token:
        st.warning("Enter your free NOAA CDO token to fetch real data.")

    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")
    model_choice = st.selectbox(
        "Ollama Model",
        OLLAMA_MODELS,
        help="Run `ollama pull mistral` before use",
    )
    temperature = st.slider("Temperature", 0.1, 1.0, 0.65, 0.05,
                            help="Higher = more creative scenarios")

    st.markdown("---")
    st.markdown("### 📍 Location & Infrastructure")
    location_input = st.text_input(
        "Location",
        placeholder="e.g. Raleigh, NC",
        help="City, address, or region",
    )
    data_start_year = st.slider(
        "Historical data from", 1960, 2010, 1990,
        help="Start year for NOAA data fetch",
    )
    infra_type = st.selectbox("Infrastructure Type", INFRA_OPTIONS)

    st.markdown("---")
    st.markdown("### 🗓️ Scenario Parameters")
    selected_horizons = st.multiselect(
        "Planning Horizons",
        HORIZONS,
        default=["2050"],
        help="Select one or more future years",
    )
    selected_scenarios = st.multiselect(
        "Climate Scenarios",
        SCENARIO_TYPES,
        default=["Moderate (RCP 6.0)"],
    )

    run_button = st.button("🚀 Fetch Data & Generate Scenarios", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">🌊 ClimateInfra Scenario Planner</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Generative AI scenario planning for climate-resilient infrastructure '
    '· Powered by NOAA CDO data + local LLM inference</p>',
    unsafe_allow_html=True,
)

if not run_button:
    st.info(
        "👈 **Configure your analysis in the sidebar**, then click **Fetch Data & Generate Scenarios**.\n\n"
        "This tool pulls real historical precipitation records from NOAA's Climate Data Online API "
        "and uses a local large language model (via Ollama) to generate structured climate resilience "
        "scenario reports — grounded in observed hydrology, not generic templates."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📡 Real Data**\n\nPulls actual daily precipitation records from NOAA GHCND stations near your location.")
    with col2:
        st.markdown("**🤖 Local AI**\n\nRuns Mistral or Llama3 locally via Ollama — no data leaves your machine.")
    with col3:
        st.markdown("**📋 Structured Output**\n\nEngineering-grade scenario reports with risk assessment, adaptation strategies, and monitoring triggers.")
    st.stop()


# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
errors = []
if not noaa_token:
    errors.append("NOAA CDO API token is required.")
if not location_input.strip():
    errors.append("Please enter a location.")
if not selected_horizons:
    errors.append("Select at least one planning horizon.")
if not selected_scenarios:
    errors.append("Select at least one climate scenario.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()


# ─────────────────────────────────────────────
# Step 1: Check Ollama availability
# ─────────────────────────────────────────────
st.markdown("## 📡 Step 1 — Data Acquisition")

llm = OllamaClient(model=model_choice)
with st.spinner(f"Checking Ollama ({model_choice})..."):
    ok, msg = llm.is_available()

if not ok:
    st.error(f"**Ollama unavailable:** {msg}")
    st.stop()

st.success(f"✅ Ollama running — model `{model_choice}` ready")


# ─────────────────────────────────────────────
# Step 2: Geocode location
# ─────────────────────────────────────────────
noaa = NOAAClient(token=noaa_token)

with st.spinner(f"Geocoding '{location_input}'..."):
    try:
        geo = noaa.geocode(location_input)
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        st.stop()

st.success(f"📍 Location resolved: **{geo['display_name'][:80]}**  (lat={geo['lat']:.4f}, lon={geo['lon']:.4f})")


# ─────────────────────────────────────────────
# Step 3: Find NOAA stations
# ─────────────────────────────────────────────
with st.spinner("Finding nearby NOAA GHCND stations..."):
    try:
        stations = noaa.find_stations(geo["lat"], geo["lon"])
    except Exception as e:
        st.error(f"Station lookup failed: {e}")
        st.stop()

station_options = {f"{s['name']} ({s['id']}) — {s.get('datacoverage', 0)*100:.0f}% coverage": s for s in stations}
chosen_label = st.selectbox(
    "Select NOAA Station",
    list(station_options.keys()),
    help="Choose the station with highest data coverage and closest to your site",
)
chosen_station = station_options[chosen_label]
station_id = chosen_station["id"]
station_name = chosen_station["name"]


# ─────────────────────────────────────────────
# Step 4: Fetch precipitation data
# ─────────────────────────────────────────────
with st.spinner(f"Fetching daily precipitation records from {data_start_year}… (this may take 20–60s)"):
    try:
        df = noaa.fetch_daily_precip(station_id, start_year=data_start_year)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

stats = noaa.compute_stats(df)

st.success(f"✅ Fetched **{len(df):,} daily records** from {stats['year_range']} ({stats['n_years']} years)")


# ─────────────────────────────────────────────
# Step 5: Display observed data
# ─────────────────────────────────────────────
st.markdown("## 📊 Step 2 — Observed Precipitation Record")
st.caption(f"Station: **{station_name}** ({station_id})")

# Key stats row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Annual", f"{stats['mean_annual_precip_in']} in")
c2.metric("Record Daily Event", f"{stats['record_daily_in']} in", help=stats['record_daily_date'])
c3.metric("99th Pctile Daily", f"{stats['p99_daily_in']} in")
c4.metric("Avg Wet Days/yr", f"{stats['mean_wet_days_per_year']}")
trend_val = stats.get("trend_in_per_decade")
trend_display = f"{trend_val:+.2f} in/decade" if trend_val is not None else "N/A"
c5.metric("Observed Trend", trend_display)

# Annual precipitation time series
annual_df = pd.DataFrame(
    [(yr, val) for yr, val in stats["annual_series"].items()],
    columns=["Year", "Annual Precip (in)"],
)
fig_annual = go.Figure()
fig_annual.add_trace(go.Bar(
    x=annual_df["Year"], y=annual_df["Annual Precip (in)"],
    name="Annual Total", marker_color="#1a4d7c", opacity=0.75,
))
# Trend line
if trend_val is not None:
    import numpy as np
    years = annual_df["Year"].values.astype(int)
    fit = np.poly1d(np.polyfit(years, annual_df["Annual Precip (in)"].values, 1))
    fig_annual.add_trace(go.Scatter(
        x=years, y=fit(years),
        name=f"Trend ({trend_display})",
        line=dict(color="#e05c2a", width=2, dash="dash"),
    ))
fig_annual.update_layout(
    title=f"Annual Precipitation — {station_name}",
    xaxis_title="Year",
    yaxis_title="Precipitation (inches)",
    legend=dict(x=0, y=1),
    height=320,
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig_annual, use_container_width=True)

# Extreme events distribution
with st.expander("📈 Extreme Event Distribution"):
    fig_hist = px.histogram(
        df[df["precip_in"] > 0],
        x="precip_in",
        nbins=60,
        title="Distribution of Wet-Day Precipitation",
        labels={"precip_in": "Daily Precipitation (in)"},
        color_discrete_sequence=["#1a4d7c"],
    )
    fig_hist.add_vline(x=stats["p95_daily_in"], line_dash="dash", line_color="orange",
                       annotation_text="95th pctile")
    fig_hist.add_vline(x=stats["p99_daily_in"], line_dash="dash", line_color="red",
                       annotation_text="99th pctile")
    fig_hist.update_layout(height=280, margin=dict(t=50, b=30))
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────
# Step 6: Generate AI scenarios
# ─────────────────────────────────────────────
st.markdown("## 🤖 Step 3 — Generative Scenario Reports")
st.caption(f"Generating {len(selected_horizons) * len(selected_scenarios)} scenario(s) with `{model_choice}` · temperature={temperature}")

total = len(selected_horizons) * len(selected_scenarios)
scenario_count = 0

for horizon in selected_horizons:
    for scenario_type in selected_scenarios:
        scenario_count += 1
        tag_class = "optimistic" if "4.5" in scenario_type else ("severe" if "8.5" in scenario_type else "moderate")
        label = f"**{horizon}** · {scenario_type}"

        with st.expander(f"📋 Scenario {scenario_count}/{total}: {label}", expanded=(scenario_count == 1)):
            st.markdown(
                f'<span class="scenario-tag {tag_class}">{scenario_type}</span> '
                f'&nbsp; Planning horizon: **{horizon}**',
                unsafe_allow_html=True,
            )

            user_prompt = build_scenario_prompt(
                location=geo["display_name"],
                station_name=station_name,
                infrastructure_type=infra_type,
                stats=stats,
                horizon=horizon,
                scenario_type=scenario_type,
            )

            output_area = st.empty()
            full_text = ""

            try:
                with st.spinner("Generating scenario..."):
                    for token in llm.stream_chat(SYSTEM_PROMPT, user_prompt, temperature=temperature):
                        full_text += token
                        output_area.markdown(full_text + "▌")
                output_area.markdown(full_text)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")

            # Download button for each scenario
            st.download_button(
                label="⬇️ Download this scenario (Markdown)",
                data=f"# ClimateInfra Scenario Report\n\n{label}\n\nLocation: {geo['display_name']}\n"
                     f"Station: {station_name}\nInfrastructure: {infra_type}\n\n---\n\n{full_text}",
                file_name=f"scenario_{horizon}_{scenario_type.split()[0].lower()}.md",
                mime="text/markdown",
                key=f"dl_{horizon}_{scenario_type}",
            )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ClimateInfra Scenario Planner · Data: NOAA Climate Data Online (CDO) GHCND · "
    "AI: Local inference via Ollama · "
    "Built for climate-resilient infrastructure decision-making."
)