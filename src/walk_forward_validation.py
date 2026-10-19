"""Expanding-window, one-step-ahead validation for the sparse national series.

With only six published observations, seasonal and high-parameter models are
not estimable defensibly. This script compares transparent baselines over all
available expanding-window origins and reports the small fold count explicitly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MIN_TRAIN = 3


def forecast_values(train: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    t = np.arange(len(train), dtype=float)
    future = np.arange(len(train), len(train) + steps, dtype=float)
    linear = np.polyval(np.polyfit(t, train, 1), future)
    log_linear = np.exp(np.polyval(np.polyfit(t, np.log(train), 1), future))
    naive = np.repeat(train[-1], steps)
    return {"Naive last value": naive, "Linear trend": linear, "Log-linear trend": log_linear}


def run_walk_forward_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    series = pd.read_csv(PROCESSED / "national_installations_series.csv", parse_dates=["date"])
    series = series.sort_values("date").dropna(subset=["installations"])
    y = series["installations"].astype(float).to_numpy()
    if len(y) <= MIN_TRAIN:
        raise ValueError(f"Need more than {MIN_TRAIN} dated observations; found {len(y)}")
    if np.any(y <= 0):
        raise ValueError("Cumulative installations must be positive for log-linear validation")

    rows = []
    for origin in range(MIN_TRAIN, len(y)):
        train = y[:origin]
        actual = y[origin]
        target = series.iloc[origin]
        for name, predictions in forecast_values(train, 1).items():
            pred = float(predictions[0])
            rows.append({
                "model": name,
                "train_points": origin,
                "forecast_origin": series.iloc[origin - 1]["date"].date().isoformat(),
                "target_date": target["date"].date().isoformat(),
                "actual_installations": int(actual),
                "predicted_installations": round(pred),
                "absolute_error": abs(actual - pred),
                "absolute_percentage_error": abs(actual - pred) / actual * 100,
            })

    folds = pd.DataFrame(rows)
    baseline = folds[folds["model"] == "Naive last value"].set_index("target_date")["absolute_percentage_error"]
    summary_rows = []
    for name, group in folds.groupby("model", sort=False):
        mape = float(group["absolute_percentage_error"].mean())
        baseline_mape = float(baseline.mean())
        summary_rows.append({
            "model": name,
            "origins": int(group["target_date"].nunique()),
            "mae_installations": round(float(group["absolute_error"].mean()), 1),
            "rmse_installations": round(float(np.sqrt(np.mean(group["absolute_error"] ** 2))), 1),
            "mape_pct": round(mape, 3),
            "naive_baseline_mape_pct": round(baseline_mape, 3),
            "mape_change_vs_naive_pct": round((mape / baseline_mape - 1) * 100, 2) if baseline_mape else 0.0,
            "validation": "expanding-window one-step ahead",
            "warning": "Only six dated observations; three validation origins. Ranking is highly uncertain.",
        })
    folds_path = PROCESSED / "forecast_walk_forward_folds.csv"
    summary_path = PROCESSED / "forecast_walk_forward_summary.csv"
    folds.to_csv(folds_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    return folds, summary


if __name__ == "__main__":
    _, result = run_walk_forward_validation()
    print(result.to_string(index=False))
