import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SCENARIO_D_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "scenario_d_tariff_shock.csv")
MODELING_TABLE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "modeling_table.csv")
FORECAST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "installations_forecast.csv")

# Fitted coefficients from 03_modeling.ipynb:
# target_conversion_ratio = -0.707 + 1.607 * tariff_growth_pct + 0.250 * irradiance
INTERCEPT = -0.707
COEF_TARIFF = 1.607
COEF_IRRADIANCE = 0.250

# Pre-computed official log-linear forecast table from 04_forecasting.ipynb
OFFICIAL_FORECASTS = {
    3: {
        "horizon_months": 3,
        "forecast_date": "2026-11-03",
        "point_forecast": 5244104,
        "lower_95": 4624684,
        "upper_95": 5946487,
        "formatted_point": "~5.24M",
        "formatted_interval": "4.62M – 5.95M"
    },
    6: {
        "horizon_months": 6,
        "forecast_date": "2027-02-01",
        "point_forecast": 6835164,
        "lower_95": 6027813,
        "upper_95": 7750651,
        "formatted_point": "~6.84M",
        "formatted_interval": "6.03M – 7.75M"
    },
    12: {
        "horizon_months": 12,
        "forecast_date": "2027-07-31",
        "point_forecast": 11611929,
        "lower_95": 10240359,
        "upper_95": 13167205,
        "formatted_point": "~11.61M",
        "formatted_interval": "10.24M – 13.17M"
    }
}

def predict_conversion_ratio(tariff_growth: float, irradiance: float) -> float:
    """
    Predict application-to-installation conversion ratio using the fitted linear model:
    conversion_ratio = -0.707 + 1.607 * tariff_growth + 0.250 * irradiance
    """
    raw_pred = INTERCEPT + COEF_TARIFF * tariff_growth + COEF_IRRADIANCE * irradiance
    # Clip between 0 and 1
    return float(np.clip(raw_pred, 0.0, 1.0))

def forecast_installations(horizon_months: int) -> dict:
    """
    Returns point forecast and 95% confidence interval for 3, 6, or 12 month horizons,
    derived from the log-linear extrapolation in 04_forecasting.ipynb.
    """
    # Round to closest standard horizon if near 3, 6, or 12
    if horizon_months in OFFICIAL_FORECASTS:
        return OFFICIAL_FORECASTS[horizon_months]
    
    if abs(horizon_months - 3) <= 1:
        return OFFICIAL_FORECASTS[3]
    elif abs(horizon_months - 6) <= 2:
        return OFFICIAL_FORECASTS[6]
    elif abs(horizon_months - 12) <= 3:
        return OFFICIAL_FORECASTS[12]
        
    # Fallback to direct calculation using log-linear slope
    # t_days = 616 + horizon_months * 30
    future_t = 616 + horizon_months * 30
    # log_coeffs estimated: slope=0.003112, intercept=13.3444, resid_std=0.06649
    slope = 0.003112
    intercept = 13.3444
    resid_std = 0.06649
    log_pred = intercept + slope * future_t
    point = int(round(np.exp(log_pred)))
    lower = int(round(np.exp(log_pred - 1.96 * resid_std)))
    upper = int(round(np.exp(log_pred + 1.96 * resid_std)))
    return {
        "horizon_months": horizon_months,
        "point_forecast": point,
        "lower_95": lower,
        "upper_95": upper,
        "formatted_point": f"~{point/1e6:.2f}M",
        "formatted_interval": f"{lower/1e6:.2f}M – {upper/1e6:.2f}M"
    }

def _find_state_row(df: pd.DataFrame, state: str):
    """Return the first row matching a state name, case-insensitive."""
    if df is None or df.empty or "state" not in df.columns or not state:
        return None
    match = df[df["state"].astype(str).str.lower() == str(state).strip().lower()]
    if match.empty:
        return None
    return match.iloc[0]


def _finite(value) -> bool:
    return value is not None and pd.notna(value) and np.isfinite(float(value))


def simulate_tariff_shock(state: str = None, shock_pp: float = 0.10) -> dict:
    """
    Simulates a percentage-point shock in tariff growth.
    Since tariff_growth coefficient is 1.607, a +10pp (+0.10) shock produces
    a uniform +16.07pp (+0.1607) lift in conversion ratio across all states.

    States / UTs missing irradiance or tariff covariates (outside the n=28
    modeling sample) still return observed conversion ratios, with
    simulation_available=False instead of omitting keys.
    """
    mean_lift = COEF_TARIFF * shock_pp
    result = {
        "tariff_growth_coefficient": COEF_TARIFF,
        "shock_percentage_points": round(shock_pp * 100, 1),
        "mean_conversion_ratio_lift": round(mean_lift, 4),
        "mean_lift_percentage_points": round(mean_lift * 100, 1),
        "simulation_available": False,
    }

    if not state:
        return result

    result["state"] = state.strip()
    missing_covariates = []

    # Observed conversion for every state in the empirical panel
    if os.path.exists(MODELING_TABLE_PATH):
        try:
            mod = pd.read_csv(MODELING_TABLE_PATH)
            mod_row = _find_state_row(mod, state)
            if mod_row is not None:
                result["state"] = str(mod_row["state"])
                if _finite(mod_row.get("target_conversion_ratio")):
                    result["actual_conversion_ratio"] = round(
                        float(mod_row["target_conversion_ratio"]), 4
                    )
                t24 = mod_row.get("tariff_domestic4kw_2024_paise_per_kwh")
                t25 = mod_row.get("tariff_domestic4kw_2025_paise_per_kwh")
                irr = mod_row.get("irradiance_annual_kwh_m2_day")
                if not (_finite(t24) and _finite(t25) and float(t24) != 0):
                    missing_covariates.append("tariff growth")
                if not _finite(irr):
                    missing_covariates.append("solar irradiance")
        except Exception as e:
            result["lookup_error"] = str(e)

    # Pre-computed Scenario D predictions (complete-covariate sample)
    if os.path.exists(SCENARIO_D_PATH):
        try:
            df = pd.read_csv(SCENARIO_D_PATH)
            row = _find_state_row(df, state)
            if row is not None and _finite(row.get("predicted_base")):
                if "actual_conversion_ratio" not in result and _finite(row.get("target_conversion_ratio")):
                    result["actual_conversion_ratio"] = round(
                        float(row["target_conversion_ratio"]), 4
                    )
                base_pred = float(row["predicted_base"])
                shocked_pred = min(1.0, max(0.0, base_pred + mean_lift))
                result["state"] = str(row["state"])
                result["predicted_base_conversion_ratio"] = round(base_pred, 4)
                result["predicted_shocked_conversion_ratio"] = round(shocked_pred, 4)
                result["state_specific_lift_pp"] = round((shocked_pred - base_pred) * 100, 1)
                result["simulation_available"] = True
                return result
        except Exception as e:
            result["lookup_error"] = str(e)

    # Compute on the fly if both model features exist
    if os.path.exists(MODELING_TABLE_PATH) and not missing_covariates:
        try:
            mod = pd.read_csv(MODELING_TABLE_PATH)
            mod_row = _find_state_row(mod, state)
            if mod_row is not None:
                t24 = float(mod_row["tariff_domestic4kw_2024_paise_per_kwh"])
                t25 = float(mod_row["tariff_domestic4kw_2025_paise_per_kwh"])
                irr = float(mod_row["irradiance_annual_kwh_m2_day"])
                tariff_growth = (t25 - t24) / t24
                base_pred = predict_conversion_ratio(tariff_growth, irr)
                shocked_pred = min(1.0, max(0.0, base_pred + mean_lift))
                result["predicted_base_conversion_ratio"] = round(base_pred, 4)
                result["predicted_shocked_conversion_ratio"] = round(shocked_pred, 4)
                result["state_specific_lift_pp"] = round((shocked_pred - base_pred) * 100, 1)
                result["simulation_available"] = True
                return result
        except Exception as e:
            result["lookup_error"] = str(e)

    if missing_covariates:
        cov = " and ".join(missing_covariates)
        verb = "are" if len(missing_covariates) > 1 else "is"
        result["unavailable_reason"] = (
            f"{result.get('state', state)} is outside the n=28 econometric sample "
            f"because {cov} {verb} missing. Observed conversion is shown; the linear "
            f"tariff-shock prediction cannot be computed."
        )
    elif "actual_conversion_ratio" not in result:
        result["unavailable_reason"] = (
            f"No panel record found for '{state}'. Check the state name."
        )
    else:
        result["unavailable_reason"] = (
            "Linear tariff-shock prediction is not available for this state."
        )

    return result

if __name__ == "__main__":
    print("Testing predict_conversion_ratio (tariff_growth=0.15, irradiance=5.0):",
          predict_conversion_ratio(0.15, 5.0))
    print("Testing forecast_installations (6 months):", forecast_installations(6))
    print("Testing simulate_tariff_shock ('Gujarat', 0.10):", simulate_tariff_shock("Gujarat", 0.10))
