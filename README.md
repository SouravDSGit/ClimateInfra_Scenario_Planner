# 🌊 ClimateInfra Scenario Planner

> **Generative AI scenario planning for climate-resilient stormwater and road infrastructure.**  
> Combines real NOAA precipitation records with local LLM inference to produce structured engineering scenario reports — grounded in observed hydrology, not generic templates.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Mistral%20%7C%20Llama3-purple)](https://ollama.com)
[![Data: NOAA CDO](https://img.shields.io/badge/data-NOAA%20CDO%20GHCND-0067a5)](https://www.ncdc.noaa.gov/cdo-web/)

---

[![Demo](https://img.youtube.com/vi/q67r80-7SyI/maxresdefault.jpg)](https://youtu.be/q67r80-7SyI)

## What It Does

Engineers and planners face a core challenge: **design standards assume stationarity, but climate is changing the precipitation statistics that infrastructure depends on.**

This tool addresses that gap by:

1. **Fetching real historical precipitation data** from the nearest NOAA GHCND weather station to any location in the US
2. **Computing observed trends** — is extreme rainfall already increasing at your site?
3. **Using a local LLM (Mistral / Llama3 via Ollama)** to generate structured climate scenario reports across three RCP pathways and three planning horizons
4. **Tailoring outputs to specific infrastructure types** — culverts, detention basins, road embankments, storm sewers, bridges, and more

Output reports include: projected precipitation statistics, infrastructure-specific failure mode analysis, prioritized adaptation strategies with cost ranges, monitoring trigger thresholds, and key uncertainties.

---

## Screenshots

| Data Dashboard                                          | Scenario Report                              |
| ------------------------------------------------------- | -------------------------------------------- |
| Annual precipitation trend + extreme event distribution | Structured AI-generated engineering scenario |

_(Add screenshots here after running the app)_

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A free [NOAA CDO API token](https://www.ncdc.noaa.gov/cdo-web/token)

### 1. Clone and install

```bash
git clone https://github.com/yourusername/climate-scenario-planner.git
cd climate-scenario-planner
pip install -r requirements.txt
```

### 2. Pull a language model

```bash
# Recommended: Mistral 7B (fast, strong reasoning)
ollama pull mistral

# Alternatives
ollama pull llama3        # Meta Llama 3 8B
ollama pull llama3.1      # Meta Llama 3.1 8B
ollama pull gemma2        # Google Gemma 2 9B
```

### 3. Set your NOAA token

```bash
cp .env.example .env
# Edit .env and add your NOAA_CDO_TOKEN
```

Or enter it directly in the app sidebar.

### 4. Run

```bash
streamlit run app.py
```

---

## Architecture

```
climate-scenario-planner/
├── app.py                  # Streamlit UI and workflow orchestration
├── utils/
│   ├── noaa_client.py      # NOAA CDO API wrapper + precipitation statistics
│   ├── llm_client.py       # Ollama streaming inference client
│   └── prompts.py          # Prompt engineering for scenario generation
├── requirements.txt
└── .env.example
```

### Data Pipeline

```
User Location Input
        │
        ▼
  OpenStreetMap Nominatim (geocoding — no API key needed)
        │
        ▼
  NOAA CDO API → Nearest GHCND Station Discovery
        │
        ▼
  Daily PRCP Records (1990–present, real observations)
        │
        ▼
  Statistical Analysis (annual totals, extremes, trends, percentiles)
        │
        ▼
  Structured Prompt → Ollama (Mistral/Llama3, runs locally)
        │
        ▼
  Scenario Report (Markdown, downloadable)
```

### Prompt Engineering

The system prompt establishes the LLM as a climate resilience engineer familiar with:

- HEC-18, HDS-5, ASCE 7, FHWA design standards
- RCP 4.5 / 6.0 / 8.5 emission pathways
- Infrastructure failure mechanisms (scour, surcharge, embankment saturation)
- Adaptation cost estimation

The user prompt injects all real NOAA statistics as a structured table, then requests output in a fixed schema covering: scenario overview, projected statistics table, failure mode risk matrix, adaptation strategies with costs, monitoring triggers, and uncertainty quantification.

---

## Climate Scenarios Supported

| Scenario   | Pathway | Description                                       |
| ---------- | ------- | ------------------------------------------------- |
| Optimistic | RCP 4.5 | Significant emissions reduction; moderate warming |
| Moderate   | RCP 6.0 | Partial mitigation; intermediate warming          |
| Severe     | RCP 8.5 | Business-as-usual; high-end warming               |

Planning horizons: **2030**, **2050**, **2080**

---

## Infrastructure Types

- Culvert (roadway drainage)
- Stormwater detention basin
- Road embankment / low-water crossing
- Storm sewer network
- Bridge (scour risk)
- Green infrastructure (bioretention / bioswale)
- Coastal/tidal drainage structure
- Retention pond / regional stormwater facility

---

## Research Context

This tool is part of a broader research program on **AI-driven decision support for climate-resilient infrastructure**, intersecting:

- **Water Resources Engineering**: non-stationary flood frequency analysis, design storm recalibration
- **Transportation Infrastructure Resilience**: culvert failure risk under changing hydrology (see [culvert-at-risk.org](https://culvert-at-risk.org))
- **Generative AI for Engineering**: structured scenario generation with domain-constrained prompting
- **Decision-Making Under Deep Uncertainty**: scenario planning across RCP pathways

### Related Work

- [culvert-at-risk.org](https://culvert-at-risk.org) — National-scale culvert failure risk assessment
- _[Add your publications here]_

---

## Limitations & Future Work

- NOAA CDO data availability varies by station (some have gaps or short records)
- LLM projections are based on general climate science patterns, not site-specific downscaled models
- Future work: integrate CMIP6 downscaled projections, NOAA Atlas 14 design storm data, and optimization-based adaptation planning

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

If you use this tool in research, please cite:

```bibtex
@software{climateinfra_scenario_planner,
  author = {[Your Name]},
  title = {ClimateInfra Scenario Planner: Generative AI for Climate-Resilient Infrastructure},
  year = {2024},
  url = {https://github.com/yourusername/climate-scenario-planner}
}
```
