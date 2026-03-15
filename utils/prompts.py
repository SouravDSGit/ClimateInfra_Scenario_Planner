"""
Prompt engineering for climate-resilient infrastructure scenario planning.
Designed for civil/environmental engineering context.
"""

SYSTEM_PROMPT = """You are a senior climate resilience engineer specializing in
water resources, stormwater infrastructure, and hydrological risk assessment.
You combine expertise in hydrology, climate science, and civil infrastructure design.

Your role is to generate structured, technically rigorous climate scenario plans
that help engineers make infrastructure investment decisions under uncertainty.

When generating scenarios:
- Ground projections in observed precipitation trends and published climate science
- Be specific about infrastructure failure mechanisms (hydraulic capacity exceedance,
  scour, erosion, embankment saturation, pipe surcharge, etc.)
- Reference design standards where relevant (ASCE 7, HEC-18, HDS-5, FHWA guidelines)
- Provide actionable, costed adaptation strategies
- Distinguish between epistemic uncertainty (model uncertainty) and aleatory uncertainty
  (natural variability)
- Always flag if observed data suggests non-stationarity

Output format: Use clear markdown with headers, tables, and bullet points.
"""


def build_scenario_prompt(
    location: str,
    station_name: str,
    infrastructure_type: str,
    stats: dict,
    horizon: str,
    scenario_type: str,
) -> str:
    """
    Build a structured scenario planning prompt from real NOAA statistics.

    Parameters
    ----------
    location : str
        Human-readable location name
    station_name : str
        NOAA station name
    infrastructure_type : str
        e.g. "culvert", "stormwater detention basin", "road embankment"
    stats : dict
        Output from NOAAClient.compute_stats()
    horizon : str
        "2030", "2050", or "2080"
    scenario_type : str
        "Optimistic (RCP 4.5)", "Moderate (RCP 6.0)", or "Severe (RCP 8.5)"
    """

    trend_str = (
        f"{stats['trend_in_per_decade']:+.2f} in/decade"
        if stats.get("trend_in_per_decade") is not None
        else "insufficient data for trend"
    )

    years_ahead = int(horizon) - 2024

    return f"""
## Climate Resilience Scenario Planning Request

**Location:** {location}
**NOAA Station:** {station_name}
**Infrastructure Type:** {infrastructure_type}
**Planning Horizon:** {horizon} ({years_ahead} years from present)
**Scenario:** {scenario_type}

---

## Observed Precipitation Record ({stats['year_range']}, {stats['n_years']} years)

| Metric | Value |
|--------|-------|
| Mean annual precipitation | {stats['mean_annual_precip_in']} in/yr |
| Std dev (annual) | {stats['std_annual_precip_in']} in/yr |
| Maximum annual on record | {stats['max_annual_precip_in']} in/yr |
| Minimum annual on record | {stats['min_annual_precip_in']} in/yr |
| Mean annual max daily event | {stats['mean_max_daily_in']} in/day |
| Record single-day event | {stats['record_daily_in']} in ({stats['record_daily_date']}) |
| 95th percentile daily event | {stats['p95_daily_in']} in |
| 99th percentile daily event | {stats['p99_daily_in']} in |
| Mean wet days per year | {stats['mean_wet_days_per_year']} days |
| Observed precipitation trend | {trend_str} |

---

Please generate a **{scenario_type}** climate scenario report for the year **{horizon}**
for the specified {infrastructure_type} infrastructure. Structure your response as follows:

### 1. Scenario Overview
Brief narrative of the climate trajectory under this scenario, including
projected changes in annual precipitation, extreme event frequency and intensity,
and seasonal shifts relevant to this region.

### 2. Projected Precipitation Statistics ({horizon})
Provide a table comparing current observed values to projected values under
this scenario, including:
- Annual precipitation (mean, extreme years)
- Design storm magnitudes (2-yr, 10-yr, 100-yr return periods)
- Changes in extreme event frequency
- Seasonal distribution shifts

### 3. Infrastructure Risk Assessment — {infrastructure_type}
Specific failure modes and risk levels for this infrastructure type under
projected conditions. Include:
- Hydraulic/structural failure mechanisms
- Risk rating (Low / Moderate / High / Critical) with justification
- Estimated time to risk threshold exceedance
- Key vulnerability drivers

### 4. Adaptation Strategies
Prioritized list of adaptation measures with:
- Engineering intervention description
- Approximate cost range (order of magnitude)
- Implementation timeline
- Effectiveness under this scenario
- Co-benefits (environmental, community, cost)

### 5. Decision Triggers & Monitoring
What early warning signs should trigger infrastructure review or upgrade?
What monitoring should be implemented now?

### 6. Key Uncertainties
What are the primary sources of uncertainty in this scenario, and how
should they influence the decision-making process?
"""


INFRA_OPTIONS = [
    "Culvert (roadway drainage)",
    "Stormwater detention basin",
    "Road embankment (low-water crossing)",
    "Storm sewer network",
    "Bridge (scour risk)",
    "Green infrastructure (bioretention / bioswale)",
    "Coastal/tidal drainage structure",
    "Retention pond / regional stormwater facility",
]

SCENARIO_TYPES = [
    "Optimistic (RCP 4.5)",
    "Moderate (RCP 6.0)",
    "Severe (RCP 8.5)",
]

HORIZONS = ["2030", "2050", "2080"]

OLLAMA_MODELS = ["mistral:7b-instruct-q4_0", "mistral", "llama3", "llama3.1", "llama3.2", "gemma2", "phi3"]