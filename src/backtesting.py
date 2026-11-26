from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORIGINS = ("2023-09", "2025-03")
HORIZON_MONTHS = 19
SERIES_KEYS = ["SKU_ID", "Region"]


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "features").is_dir():
            return candidate
    raise FileNotFoundError("Run notebook 04 before backtesting.")


def _metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=False):
        actual = group["Actual"].to_numpy(dtype=float)
        prediction = group["Prediction"].to_numpy(dtype=float)
        denominator = float(np.abs(actual).sum())
        error = prediction - actual
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        rows.append({
            **dict(zip(keys, values)),
            "Rows": int(len(group)),
            "WAPE": float(np.abs(error).sum() / denominator) if denominator else np.nan,
            "Bias": float(error.sum() / denominator) if denominator else np.nan,
            "Actual_Units": float(actual.sum()),
            "Forecast_Units": float(prediction.sum()),
        })
    return pd.DataFrame(rows)


def run_baseline_backtest(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    demand_path = root / "outputs" / "demand_reconstruction" / "train_with_reconstructed_demand.csv"
    output_dir = root / "outputs" / "backtesting"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(demand_path, low_memory=False)
    data["Unconstrained_Demand"] = pd.to_numeric(data["Unconstrained_Demand"], errors="coerce")
    data = data.loc[
        data["Date"].between("2019-01", "2026-10")
        & data[["Date", "SKU_ID", "Region"]].notna().all(axis=1)
        & data["Unconstrained_Demand"].ge(0)
    ].copy()
    data = data.drop_duplicates(["Date", "SKU_ID", "Region"])
    data["Period"] = pd.PeriodIndex(data["Date"], freq="M")

    prediction_rows: list[dict[str, Any]] = []
    origin_summary = []
    for origin_text in ORIGINS:
        origin = pd.Period(origin_text, freq="M")
        horizon_end = origin + HORIZON_MONTHS
        history = data.loc[data["Period"].le(origin)].copy()
        holdout = data.loc[data["Period"].gt(origin) & data["Period"].le(horizon_end)].copy()
        observed_horizons = holdout["Period"].nunique()
        if observed_horizons != HORIZON_MONTHS:
            raise ValueError(
                f"Origin {origin}: expected {HORIZON_MONTHS} holdout months, found {observed_horizons}."
            )

        history_groups = {
            key: group.sort_values("Period")
            for key, group in history.groupby(SERIES_KEYS, dropna=False, sort=False)
        }
        family_region_mean = history.groupby(
            ["Product_Family", "Region"], dropna=False
        )["Unconstrained_Demand"].mean()
        family_mean = history.groupby("Product_Family", dropna=False)[
            "Unconstrained_Demand"
        ].mean()
        global_mean = float(history["Unconstrained_Demand"].mean())
        origin_fallbacks = 0

        for series_key, future in holdout.groupby(SERIES_KEYS, dropna=False, sort=False):
            series_history = history_groups.get(series_key)
            history_values = {}
            recent_values = []
            if series_history is not None:
                history_values = dict(zip(series_history["Period"], series_history["Unconstrained_Demand"]))
                recent_values = (
                    series_history["Unconstrained_Demand"].dropna().tail(3).astype(float).tolist()
                )
            sample = future.iloc[0]
            family = sample.get("Product_Family", "")
            region = sample.get("Region", "")
            fallback = family_region_mean.get((family, region), np.nan)
            if pd.isna(fallback):
                fallback = family_mean.get(family, np.nan)
            if pd.isna(fallback):
                fallback = global_mean
            if not recent_values:
                origin_fallbacks += len(future)
                recent_prediction = float(fallback)
            else:
                recent_prediction = float(np.mean(recent_values))

            seasonal_predictions = {}
            for _, row in future.sort_values("Period").iterrows():
                target_period = row["Period"]
                source_period = target_period - 12
                seasonal_value = (
                    seasonal_predictions.get(source_period)
                    if source_period > origin
                    else history_values.get(source_period, np.nan)
                )
                if pd.isna(seasonal_value):
                    seasonal_value = recent_prediction
                    seasonal_fallback = True
                else:
                    seasonal_fallback = False
                seasonal_value = max(0.0, float(seasonal_value))
                seasonal_predictions[target_period] = seasonal_value
                horizon = int(target_period.ordinal - origin.ordinal)
                actual = float(row["Unconstrained_Demand"])
                for model_name, forecast in (
                    ("seasonal_naive_12", seasonal_value),
                    ("recent_mean_3", max(0.0, recent_prediction)),
                ):
                    prediction_rows.append({
                        "Origin": origin_text,
                        "Date": str(target_period),
                        "Horizon": horizon,
                        "Horizon_Bucket": (
                            "1-6" if horizon <= 6 else "7-12" if horizon <= 12 else "13-19"
                        ),
                        "SKU_ID": row["SKU_ID"],
                        "Region": row["Region"],
                        "Product_Family": row.get("Product_Family", ""),
                        "Model": model_name,
                        "Actual": actual,
                        "Prediction": forecast,
                        "Seasonal_Fallback_Used": seasonal_fallback
                        if model_name == "seasonal_naive_12" else False,
                    })
        origin_summary.append({
            "Origin": origin_text,
            "Holdout_Start": str(origin + 1),
            "Holdout_End": str(horizon_end),
            "Holdout_Months": int(observed_horizons),
            "Rows": int(len(holdout)),
            "Cold_Start_Fallback_Rows": int(origin_fallbacks),
        })

    predictions = pd.DataFrame(prediction_rows)
    overall = _metrics(predictions, ["Origin", "Model"])
    by_horizon = _metrics(predictions, ["Origin", "Model", "Horizon_Bucket", "Horizon"])
    by_family = _metrics(
        predictions, ["Origin", "Model", "Product_Family", "Horizon_Bucket"]
    )
    by_region = _metrics(
        predictions, ["Origin", "Model", "Region", "Horizon_Bucket"]
    )
    by_family_region = _metrics(
        predictions, ["Origin", "Model", "Product_Family", "Region", "Horizon_Bucket"]
    )
    predictions.to_csv(output_dir / "rolling_origin_predictions.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "accuracy_overall.csv", index=False, encoding="utf-8-sig")
    by_horizon.to_csv(output_dir / "accuracy_by_horizon.csv", index=False, encoding="utf-8-sig")
    by_family.to_csv(output_dir / "accuracy_by_family.csv", index=False, encoding="utf-8-sig")
    by_region.to_csv(output_dir / "accuracy_by_region.csv", index=False, encoding="utf-8-sig")
    by_family_region.to_csv(
        output_dir / "accuracy_by_family_region_horizon.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "origins": origin_summary,
        "prediction_rows": int(len(predictions)),
        "overall_metrics": overall.to_dict("records"),
        "fallback_rows_seasonal_naive": int(predictions.loc[
            predictions["Model"].eq("seasonal_naive_12"), "Seasonal_Fallback_Used"
        ].sum()),
        "outputs": str(output_dir),
        "metric_definitions": {
            "WAPE": "sum(abs(prediction - actual)) / sum(abs(actual)); lower is better.",
            "Bias": "sum(prediction - actual) / sum(abs(actual)); positive means over-forecast.",
        },
        "assumptions": [
            "Validation origins are Sep 2023 and Mar 2025; each has the full 19-month holdout available.",
            "Seasonal-naive forecasts recurse from prior predictions when the 12-month source falls inside the holdout.",
            "Recent-mean baseline repeats the mean of the last three observed nonmissing months at the origin.",
            "Missing targets are excluded from metric denominators, not treated as zero.",
            "Missing seasonal history falls back to recent series mean, then family-region, family, and global mean.",
        ],
    }
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    return summary
