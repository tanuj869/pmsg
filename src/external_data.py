"""Small, provenance-preserving connectors for public external data APIs."""
from __future__ import annotations

import os
import time
from typing import Any
import requests

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
OGD_RESOURCE_BASE = "https://api.data.gov.in/resource"

# NASA POWER point queries use the state capital as a transparent location proxy.
# This is a representative point series, not a spatial mean over a state.
STATE_POINTS = {
    "Andhra Pradesh": (16.5062, 80.6480, "Amaravati"),
    "Arunachal Pradesh": (27.0844, 93.6053, "Itanagar"),
    "Assam": (26.1445, 91.7362, "Dispur"),
    "Bihar": (25.5941, 85.1376, "Patna"),
    "Chhattisgarh": (21.2514, 81.6296, "Raipur"),
    "Goa": (15.4909, 73.8278, "Panaji"),
    "Gujarat": (23.2156, 72.6369, "Gandhinagar"),
    "Haryana": (30.7333, 76.7794, "Chandigarh"),
    "Himachal Pradesh": (31.1048, 77.1734, "Shimla"),
    "Jammu and Kashmir": (34.0837, 74.7973, "Srinagar"),
    "Jharkhand": (23.3441, 85.3096, "Ranchi"),
    "Karnataka": (12.9716, 77.5946, "Bengaluru"),
    "Kerala": (8.5241, 76.9366, "Thiruvananthapuram"),
    "Madhya Pradesh": (23.2599, 77.4126, "Bhopal"),
    "Maharashtra": (19.0760, 72.8777, "Mumbai"),
    "Manipur": (24.8170, 93.9368, "Imphal"),
    "Meghalaya": (25.5788, 91.8933, "Shillong"),
    "Mizoram": (23.7271, 92.7176, "Aizawl"),
    "Nagaland": (25.6751, 94.1086, "Kohima"),
    "Odisha": (20.2961, 85.8245, "Bhubaneswar"),
    "Punjab": (30.7333, 76.7794, "Chandigarh"),
    "Rajasthan": (26.9124, 75.7873, "Jaipur"),
    "Sikkim": (27.3389, 88.6065, "Gangtok"),
    "Tamil Nadu": (13.0827, 80.2707, "Chennai"),
    "Telangana": (17.3850, 78.4867, "Hyderabad"),
    "Tripura": (23.8315, 91.2868, "Agartala"),
    "Uttar Pradesh": (26.8467, 80.9462, "Lucknow"),
    "Uttarakhand": (30.3165, 78.0322, "Dehradun"),
    "West Bengal": (22.5726, 88.3639, "Kolkata"),
    "Andaman and Nicobar Islands": (11.6234, 92.7265, "Port Blair"),
    "Chandigarh": (30.7333, 76.7794, "Chandigarh"),
    "Dadra and Nagar Haveli and Daman and Diu": (20.3974, 72.8328, "Daman"),
    "NCT of Delhi": (28.6139, 77.2090, "New Delhi"),
    "Ladakh": (34.1526, 77.5771, "Leh"),
    "Lakshadweep": (10.5667, 72.6417, "Kavaratti"),
    "Puducherry": (11.9416, 79.8083, "Puducherry"),
}

_CACHE: dict[tuple, tuple[float, Any]] = {}
CACHE_SECONDS = 6 * 60 * 60


def _cached_get(key: tuple, url: str, params: dict, timeout: int = 20) -> dict:
    now = time.time()
    entry = _CACHE.get(key)
    if entry and now - entry[0] < CACHE_SECONDS:
        return entry[1]
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    _CACHE[key] = (now, payload)
    return payload


def fetch_nasa_power_state(state: str, start_year: int = 2023, end_year: int | None = None) -> dict:
    """Fetch monthly solar irradiance and air temperature for a state-capital point."""
    if state not in STATE_POINTS:
        raise ValueError(f"Unknown state/UT: {state}")
    lat, lon, location = STATE_POINTS[state]
    end_year = end_year or time.gmtime().tm_year
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": int(start_year),
        "end": int(end_year),
        "format": "JSON",
    }
    payload = _cached_get(("nasa", state, start_year, end_year), NASA_POWER_URL, params)
    properties = payload.get("properties", {})
    parameter_data = properties.get("parameter", {})
    radiation = parameter_data.get("ALLSKY_SFC_SW_DWN", {})
    temperature = parameter_data.get("T2M", {})
    # NASA labels the annual aggregate YYYY13 in its monthly response. Keep
    # actual calendar months here so downstream charts do not plot it as a month.
    months = sorted(
        month for month in (set(radiation) | set(temperature))
        if str(month)[-2:].isdigit() and 1 <= int(str(month)[-2:]) <= 12
    )
    series = [
        {
            "month": month,
            "solar_irradiance_kwh_m2_day": radiation.get(month),
            "temperature_c": temperature.get(month),
        }
        for month in months
    ]
    return {
        "state": state,
        "location_proxy": location,
        "latitude": lat,
        "longitude": lon,
        "series": series,
        "source": "NASA POWER Monthly API",
        "source_url": NASA_POWER_URL,
        "parameters": {"ALLSKY_SFC_SW_DWN": "kWh/m2/day", "T2M": "degC"},
        "api_metadata": payload.get("parameters", {}),
        "caveat": "State-capital point data are not a state-wide spatial average.",
    }


def fetch_ogd_resource(limit: int = 100, offset: int = 0) -> dict:
    """Fetch records from a configured data.gov.in resource; key and ID are user configured."""
    resource_id = os.getenv("PMSG_OGD_RESOURCE_ID", "").strip()
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not resource_id or not api_key:
        return {
            "configured": False,
            "records": [],
            "message": "Set PMSG_OGD_RESOURCE_ID and DATA_GOV_IN_API_KEY to enable this resource.",
        }
    params = {"api-key": api_key, "format": "json", "limit": limit, "offset": offset}
    payload = _cached_get(("ogd", resource_id, limit, offset), f"{OGD_RESOURCE_BASE}/{resource_id}", params)
    return {
        "configured": True,
        "records": payload.get("records", []),
        "total": payload.get("total", 0),
        "resource_id": resource_id,
        "source": "Open Government Data Platform India API",
        "source_url": f"https://data.gov.in/resource/{resource_id}",
    }


def source_status() -> dict:
    return {
        "nasa_power": {"configured": True, "source": "NASA POWER Monthly API"},
        "data_gov_in": {
            "configured": bool(os.getenv("DATA_GOV_IN_API_KEY") and os.getenv("PMSG_OGD_RESOURCE_ID")),
            "needs": ["DATA_GOV_IN_API_KEY", "PMSG_OGD_RESOURCE_ID"],
        },
        "niti_iced": {
            "configured": False,
            "source": "NITI Aayog India Climate & Energy Dashboard",
            "note": "Project covariate workbook is the retained snapshot; the dashboard does not publish a documented open API in this project.",
        },
    }
