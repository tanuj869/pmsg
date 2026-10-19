"""Generate reproducible dataset-register and data-dictionary CSVs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

DATASET_META = {
    "3__District-wise_beneficiaries__Maharashtra__11-12-2024.xlsx": (
        "District-wise PM Surya Ghar beneficiary observations for Maharashtra",
        "Open Government Data Platform India / source workbook",
        "https://www.data.gov.in/resource/district-wise-number-beneficiaries-under-pm-surya-ghar-muft-bijli-yojana-state-maharashtra",
        "2024-12-11",
    ),
    "District_and_discom_info-PM_surya_ghar.xlsx": (
        "District and DISCOM reference information used for geographic and utility context",
        "PM Surya Ghar portal / source workbook",
        "https://solarrooftop.pmsuryaghar.gov.in/",
        "Workbook reference date not recorded",
    ),
    "Installations.xlsx": (
        "State/UT installations under PM Surya Ghar",
        "Open Government Data Platform India / Rajya Sabha",
        "https://ap.data.gov.in/resource/stateut-wise-number-installations-under-pm-surya-ghar-muft-bijli-yojana-9-12-2024",
        "2024-12-09",
    ),
    "PMSG_Tariff_-_2024.xlsx": (
        "State utility domestic and industrial electricity tariff data for 2024",
        "Central Electricity Authority",
        "https://cea.nic.in/",
        "2024",
    ),
    "PMSG_Tariff_-_2025.xlsx": (
        "State utility domestic and industrial electricity tariff data for 2025",
        "Central Electricity Authority",
        "https://cea.nic.in/",
        "2025",
    ),
    "PMSG_ToD_Regime_by_State.xlsx": (
        "State-level time-of-day tariff regime indicator",
        "Project source workbook; coverage limitations are documented in README",
        "",
        "Workbook reference date not recorded",
    ),
    "Registrations__applications__households_benefited.xlsx": (
        "State/UT registrations, applications and households benefited under PM Surya Ghar",
        "Open Government Data Platform India / Rajya Sabha",
        "https://www.data.gov.in/resource/state-ut-wise-number-registrations-applications-and-households-benefitted-under-pm-surya",
        "Multiple dated releases; see workbook",
    ),
    "Solar_potential.xlsx": (
        "State-level solar potential estimates",
        "NITI Aayog India Climate & Energy Dashboard / MoSPI source references",
        "https://icedapi.niti.gov.in/",
        "Workbook reference year; see workbook",
    ),
    "state-wise_RE_and_non-RE_installed_capacity_at_two_points.xlsx": (
        "State-wise renewable and non-renewable installed capacity at two reference points",
        "Ministry of New and Renewable Energy",
        "https://mnre.gov.in/en/physical-progress/",
        "2018 and 2025 columns",
    ),
    "States_nasa.xlsx": (
        "State-level NASA POWER solar irradiance summary used in the empirical model",
        "NASA POWER; retained project extraction",
        "https://power.larc.nasa.gov/",
        "Long-term climatology; exact extraction period should be confirmed in workbook",
    ),
    "state_population_mohfw_2021.csv": (
        "State/UT population estimates as of 1 July 2021",
        "Ministry of Health and Family Welfare, Government of India",
        "https://www.mohfw.gov.in/sites/default/files/rhs20-21_2.pdf",
        "2021-07-01",
    ),
    "pmsg_snapshots_long.csv": (
        "Long-form official PM Surya Ghar state snapshot observations",
        "Rajya Sabha / data.gov.in / PM Surya Ghar portal",
        "https://www.data.gov.in/keywords/PMSG",
        "Multiple dated releases; see source column",
    ),
    "state_covariates_wide.csv": (
        "State-level tariff, solar potential, irradiance and renewable capacity covariates",
        "CEA, NITI Aayog ICeD, NASA POWER and MNRE source files",
        "https://icedapi.niti.gov.in/",
        "Mixed annual reference years; see column names and source workbooks",
    ),
    "modeling_table.csv": (
        "State-level modeling cross-section for application-to-installation conversion",
        "Derived from project source datasets",
        "",
        "2026-08-23 snapshot",
    ),
    "national_installations_series.csv": (
        "Dated national cumulative installation milestones used for forecasting",
        "Lok Sabha, Rajya Sabha and PIB/MNRE releases",
        "https://www.data.gov.in/keywords/PMSG",
        "2024-11-27 to 2026-08-05",
    ),
}

FIELD_META = {
    "state": ("Canonical State/UT name", "category"),
    "date": ("Observation or release date", "date"),
    "metric": ("Measure name for a long-form observation", "category"),
    "value": ("Reported measure value; units depend on metric", "varies by metric"),
    "source": ("Source record or publication attribution", "text"),
    "target_conversion_ratio": ("Installations divided by applications", "ratio, 0-1"),
    "applications_aug2026": ("Applications reported in the August 2026 cross-section", "applications"),
    "households_aug2026": ("Households benefited in the August 2026 cross-section", "households"),
    "installations_aug2026": ("Installations in the August 2026 cross-section", "installations"),
    "population_2021": ("Official estimated mid-year population as of 1 July 2021", "persons"),
    "installations_per_lakh_population": ("Installations per 100,000 residents; 2026 installations divided by 2021 population", "installations per 100,000 persons"),
    "applications_per_lakh_population": ("Applications per 100,000 residents; 2026 applications divided by 2021 population", "applications per 100,000 persons"),
    "adoption_data_available": ("Whether the August 2026 scheme snapshot contains adoption observations for this State/UT", "boolean"),
    "irradiance_annual_kwh_m2_day": ("Long-run annual average solar irradiance", "kWh/m2/day"),
    "solar_potential_6.69pct_wasteland_gw": ("Estimated state solar potential under the 6.69 percent wasteland assumption", "GW"),
    "mnre_re_capacity_2025_mw": ("Renewable energy installed capacity for 2025", "MW"),
    "tariff_domestic4kw_2025_paise_per_kwh": ("Domestic tariff for a 4 kW connection, 2025 source year", "paise/kWh"),
    "forecast_date": ("Target date for the forecast horizon", "date"),
    "point_forecast": ("Forecast national cumulative installations", "installations"),
    "lower_95": ("Lower bound of the reported 95 percent prediction interval", "installations"),
    "upper_95": ("Upper bound of the reported 95 percent prediction interval", "installations"),
}


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=100_000)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=100_000)
    except Exception:
        return None
    return None


def build_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    register, dictionary = [], []
    for folder, layer in ((RAW, "raw"), (PROCESSED, "processed")):
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            if path.name in {"dataset_register.csv", "data_dictionary.csv"}:
                continue
            table = read_table(path)
            meta = DATASET_META.get(path.name, (path.stem.replace("_", " "), "Owner/source needs confirmation", "", "Not recorded"))
            register.append({
                "dataset_id": path.stem,
                "filename": path.name,
                "layer": layer,
                "format": path.suffix.lower().lstrip("."),
                "description": meta[0],
                "source_owner": meta[1],
                "source_url": meta[2],
                "data_as_of": meta[3],
                "access_requirements": "Local source file",
                "rows_inspected": len(table) if table is not None else None,
                "columns": len(table.columns) if table is not None else None,
                "size_bytes": path.stat().st_size,
            })
            if table is None:
                continue
            for column in table.columns:
                series = table[column]
                meaning, unit = FIELD_META.get(column, (f"Field '{column}' from {path.name}; confirm against the source workbook or notebook.", "not specified"))
                dictionary.append({
                    "dataset_id": path.stem,
                    "field": str(column),
                    "meaning": meaning,
                    "unit": unit,
                    "dtype": str(series.dtype),
                    "missing_pct": round(float(series.isna().mean() * 100), 2),
                    "example": str(series.dropna().iloc[0])[:120] if series.notna().any() else "",
                })
    register.extend([
        {
            "dataset_id": "nasa_power_monthly_api",
            "filename": "on-demand API response",
            "layer": "external connector",
            "format": "JSON API",
            "description": "Monthly point solar radiation and temperature series; state capital coordinates are location proxies.",
            "source_owner": "NASA Langley Research Center POWER Project",
            "source_url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
            "data_as_of": "On demand; API response includes requested years",
            "access_requirements": "Public endpoint; network access required",
        },
        {
            "dataset_id": "data_gov_in_configured_resource",
            "filename": "on-demand API response",
            "layer": "external connector",
            "format": "JSON API",
            "description": "Configured Government of India open-data resource records for current scheme questions.",
            "source_owner": "Open Government Data Platform India",
            "source_url": "https://api.data.gov.in/resource/{resource-id}",
            "data_as_of": "On demand; resource publication cadence applies",
            "access_requirements": "DATA_GOV_IN_API_KEY and PMSG_OGD_RESOURCE_ID environment variables",
        },
    ])
    reg_df = pd.DataFrame(register)
    dict_df = pd.DataFrame(dictionary)
    reg_df.to_csv(PROCESSED / "dataset_register.csv", index=False)
    dict_df.to_csv(PROCESSED / "data_dictionary.csv", index=False)
    return reg_df, dict_df


if __name__ == "__main__":
    reg, fields = build_catalog()
    print(f"Catalogued {len(reg)} files and {len(fields)} fields.")
