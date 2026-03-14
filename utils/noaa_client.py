"""
NOAA Climate Data Online (CDO) API client.
Fetches real precipitation data from GHCND stations.
API token: https://www.ncdc.noaa.gov/cdo-web/token
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time


NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"


class NOAAClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {"token": token}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    # ------------------------------------------------------------------
    # Geocoding (OpenStreetMap Nominatim — no key required)
    # ------------------------------------------------------------------

    def geocode(self, location: str) -> dict:
        """Convert a city/address string to lat/lon using Nominatim."""
        resp = requests.get(
            f"{NOMINATIM_BASE}/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "ClimateInfraScenarioPlanner/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Location '{location}' not found. Try a more specific address.")
        r = results[0]
        return {
            "display_name": r["display_name"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
        }

    # ------------------------------------------------------------------
    # Station discovery
    # ------------------------------------------------------------------

    def find_stations(self, lat: float, lon: float, radius_deg: float = 1.0, limit: int = 5) -> list[dict]:
        """Find GHCND stations near a lat/lon with precipitation data."""
        extent = f"{lat - radius_deg},{lon - radius_deg},{lat + radius_deg},{lon + radius_deg}"
        resp = self.session.get(
            f"{NOAA_BASE}/stations",
            params={
                "datasetid": "GHCND",
                "datatypeid": "PRCP",
                "extent": extent,
                "limit": limit,
                "sortfield": "name",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise ValueError(
                f"No GHCND precipitation stations found within {radius_deg}° of ({lat:.2f}, {lon:.2f}). "
                "Try a larger area or different location."
            )
        return results

    # ------------------------------------------------------------------
    # Precipitation data
    # ------------------------------------------------------------------

    def fetch_daily_precip(
        self,
        station_id: str,
        start_year: int = 1990,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily PRCP records for a station. Returns a DataFrame with
        columns: date, precip_in (precipitation in inches).

        NOAA returns PRCP in tenths of mm; we convert to inches.
        """
        if end_year is None:
            end_year = datetime.now().year - 1  # last full year

        all_records = []
        # NOAA CDO API limits to 1 year per request for daily data
        for year in range(start_year, end_year + 1):
            try:
                resp = self.session.get(
                    f"{NOAA_BASE}/data",
                    params={
                        "datasetid": "GHCND",
                        "stationid": station_id,
                        "datatypeid": "PRCP",
                        "startdate": f"{year}-01-01",
                        "enddate": f"{year}-12-31",
                        "limit": 1000,
                        "units": "standard",  # tenths of mm
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                records = resp.json().get("results", [])
                all_records.extend(records)
                time.sleep(0.3)  # respect rate limits
            except requests.HTTPError as e:
                if e.response.status_code == 400:
                    continue  # no data for that year
                raise

        if not all_records:
            raise ValueError(
                f"No precipitation records returned for station {station_id} "
                f"between {start_year}–{end_year}."
            )

        df = pd.DataFrame(all_records)
        df["date"] = pd.to_datetime(df["date"])
        # NOAA standard units for PRCP = tenths of mm → convert to inches
        df["precip_in"] = df["value"] / 254.0  # tenths of mm → inches
        df = df[df["precip_in"] >= 0].sort_values("date").reset_index(drop=True)
        return df[["date", "precip_in"]]

    # ------------------------------------------------------------------
    # Summary statistics (no synthetic data — computed from real records)
    # ------------------------------------------------------------------

    def compute_stats(self, df: pd.DataFrame) -> dict:
        """Compute summary statistics from real daily precipitation records."""
        df = df.copy()
        df["year"] = df["date"].dt.year

        annual = df.groupby("year")["precip_in"].sum()
        wet_days = df[df["precip_in"] > 0.1].groupby("year").size()
        extremes = df.groupby("year")["precip_in"].max()

        # Simple linear trend in annual totals (inches/decade)
        if len(annual) >= 5:
            import numpy as np
            years = annual.index.values
            vals = annual.values
            slope, _ = np.polyfit(years, vals, 1)
            trend_per_decade = round(slope * 10, 2)
        else:
            trend_per_decade = None

        return {
            "n_years": int(len(annual)),
            "year_range": f"{int(annual.index.min())}–{int(annual.index.max())}",
            "mean_annual_precip_in": round(float(annual.mean()), 2),
            "std_annual_precip_in": round(float(annual.std()), 2),
            "max_annual_precip_in": round(float(annual.max()), 2),
            "min_annual_precip_in": round(float(annual.min()), 2),
            "mean_max_daily_in": round(float(extremes.mean()), 2),
            "record_daily_in": round(float(df["precip_in"].max()), 2),
            "record_daily_date": str(df.loc[df["precip_in"].idxmax(), "date"].date()),
            "p95_daily_in": round(float(df["precip_in"].quantile(0.95)), 2),
            "p99_daily_in": round(float(df["precip_in"].quantile(0.99)), 2),
            "mean_wet_days_per_year": round(float(wet_days.mean()), 1),
            "trend_in_per_decade": trend_per_decade,
            "annual_series": annual.to_dict(),
        }