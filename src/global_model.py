from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ORIGINS = ("2023-09", "2025-03")
HORIZON_MONTHS = 19
SERIES_KEYS = ["SKU_ID", "Region"]
CATEGORICAL_FEATURES = ["Product_Family", "Region"]
DEMAND_LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
NUMERIC_FEATURES = [
    *(f"lag_{lag}" for lag in DEMAND_LAGS),
    *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
    *(f"rolling_observed_count_{window}" for window in ROLLING_WINDOWS),
    "zero_share_prior_12",
    "Month",
    "Month_Sin",
    "Month_Cos",
    "Known_Scoped_Event_Count",
    "Months_Since_Launch",
    "Months_To_EOL",
    "Is_Launched",
    "Is_Before_EOL",
    *(
        f"{column}_lag_{lag}"
        for column in (
            "ISM_Manufacturing_PMI",
            "Auto_Production_YoY_Pct",
            "Vehicle_Inventory_Days",
            "Raw_Material_Cost_Index",
            "New_Vehicle_Orders_Index",
            "Industrial_Util_Rate_Pct",
            "Fuel_Price_Index",
            "EV_Adoption_Index",
        )
        for lag in (1, 3)
    ),
]


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "features").is_dir():
            return candidate
    raise FileNotFoundError("Run notebooks 02 and 04 before global-model backtesting.")


def _series_key(row: pd.Series) -> tuple[str, str]:
    return str(row["SKU_ID"]), str(row["Region"])


def _design_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    numeric = frame.reindex(columns=NUMERIC_FEATURES).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    categories = frame.reindex(columns=CATEGORICAL_FEATURES).fillna("Unknown").astype(str)
    encoded = pd.get_dummies(categories, columns=CATEGORICAL_FEATURES, dtype=float)
    matrix = pd.concat([numeric, encoded], axis=1)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0.0)
    return matrix


def _fit_model(training: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, list[str]]:
    usable = training.loc[
        pd.to_numeric(training["Unconstrained_Demand"], errors="coerce").ge(0)
    ].copy()
    if usable.empty:
        raise ValueError("No nonnegative demand rows are available to fit the global model.")
    matrix = _design_matrix(usable)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(matrix, usable["Unconstrained_Demand"].astype(float))
    return model, list(matrix.columns)


def _intermittency_class(values: pd.Series) -> str:
    valid = values.dropna()
    positive = valid.loc[valid.gt(0)]
    if positive.empty or len(positive) < 2:
        return "insufficient_positive_history"
    adi = len(valid) / len(positive)
    mean_positive = float(positive.mean())
    cv2 = float((positive.std(ddof=1) / mean_positive) ** 2) if mean_positive else np.nan
    if pd.isna(cv2):
        return "insufficient_positive_history"
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi < 1.32:
        return "erratic"
    if cv2 < 0.49:
        return "intermittent"
    return "lumpy"


def _croston_sba(values: pd.Series, alpha: float = 0.1) -> float:
    known = values.dropna().sort_index()
    if known.empty:
        return 0.0
    positive = known.loc[known.gt(0)]
    if positive.empty:
        return 0.0

    first_period = known.index[0]
    last_positive_period = positive.index[0]
    smoothed_size = float(positive.iloc[0])
    smoothed_interval = max(1, last_positive_period.ordinal - first_period.ordinal + 1)
    for period, demand in known.loc[known.index > last_positive_period].items():
        if demand > 0:
            interval = max(1, period.ordinal - last_positive_period.ordinal)
            smoothed_size = alpha * float(demand) + (1 - alpha) * smoothed_size
            smoothed_interval = alpha * interval + (1 - alpha) * smoothed_interval
            last_positive_period = period
    return max(0.0, (1 - alpha / 2) * smoothed_size / smoothed_interval)


def _dynamic_features(row: pd.Series, state: dict[pd.Period, float]) -> dict[str, Any]:
    period = row["Date_Period"]
    result = row.to_dict()
    for lag in DEMAND_LAGS:
        result[f"lag_{lag}"] = state.get(period - lag, np.nan)
    for window in ROLLING_WINDOWS:
        values = [state.get(period - offset, np.nan) for offset in range(1, window + 1)]
        known = [value for value in values if pd.notna(value)]
        result[f"rolling_mean_{window}"] = float(np.mean(known)) if known else np.nan
        result[f"rolling_observed_count_{window}"] = len(known)
    prior_year = [state.get(period - offset, np.nan) for offset in range(1, 13)]
    result["zero_share_prior_12"] = float(np.mean([value == 0 for value in prior_year]))
    return result


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


def run_global_model_backtest(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    feature_path = root / "outputs" / "features" / "train_features.csv"
    demand_path = root / "outputs" / "demand_reconstruction" / "train_with_reconstructed_demand.csv"
    output_dir = root / "outputs" / "global_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(feature_path, low_memory=False)
    demand = pd.read_csv(demand_path, low_memory=False)
    features["Date_Period"] = pd.PeriodIndex(features["Date"], freq="M")
    features["Unconstrained_Demand"] = pd.to_numeric(
        features["Unconstrained_Demand"], errors="coerce"
    )
    demand["Unconstrained_Demand"] = pd.to_numeric(
        demand["Unconstrained_Demand"], errors="coerce"
    )
    demand = demand.loc[
        demand["Date"].between("2019-01", "2026-10")
        & demand[["Date", *SERIES_KEYS]].notna().all(axis=1)
    ].drop_duplicates(["Date", *SERIES_KEYS]).copy()
    demand["Date_Period"] = pd.PeriodIndex(demand["Date"], freq="M")

    outputs: list[pd.DataFrame] = []
    origin_summary = []
    for origin_text in ORIGINS:
        origin = pd.Period(origin_text, freq="M")
        horizon_end = origin + HORIZON_MONTHS
        training = features.loc[
            features["Date_Period"].le(origin)
            & features["Unconstrained_Demand"].ge(0)
        ].copy()
        if training.empty:
            raise ValueError(f"No training rows are available by origin {origin_text}.")
        model, design_columns = _fit_model(training)

        historical = demand.loc[demand["Date_Period"].le(origin)].copy()
        series_states: dict[tuple[str, str], dict[pd.Period, float]] = {}
        croston_values: dict[tuple[str, str], float] = {}
        classes: dict[tuple[str, str], str] = {}
        for key_values, group in historical.groupby(SERIES_KEYS, dropna=False, sort=False):
            key = tuple(str(value) for value in key_values)
            ordered = group.sort_values("Date_Period")
            first_period = ordered["Date_Period"].min()
            state = {
                period: np.nan
                for period in pd.period_range(first_period, origin, freq="M")
            }
            observed = ordered.set_index("Date_Period")["Unconstrained_Demand"]
            state.update(observed.to_dict())
            series_states[key] = state
            series = observed.copy()
            croston_values[key] = _croston_sba(series)
            classes[key] = _intermittency_class(series)

        holdout = features.loc[
            features["Date_Period"].gt(origin)
            & features["Date_Period"].le(horizon_end)
        ].copy()
        observed_months = holdout["Date_Period"].nunique()
        if observed_months != HORIZON_MONTHS:
            raise ValueError(
                f"Origin {origin_text}: expected {HORIZON_MONTHS} holdout months, "
                f"found {observed_months}."
            )

        origin_predictions = []
        for period, month_rows in holdout.groupby("Date_Period", sort=True):
            dynamic_rows = []
            keys = []
            for _, row in month_rows.iterrows():
                key = _series_key(row)
                state = series_states.setdefault(key, {})
                dynamic_rows.append(_dynamic_features(row, state))
                keys.append(key)
            dynamic_frame = pd.DataFrame(dynamic_rows)
            matrix = _design_matrix(dynamic_frame, design_columns)
            global_predictions = np.maximum(0.0, model.predict(matrix))

            for row_index, (_, row) in enumerate(month_rows.iterrows()):
                key = keys[row_index]
                class_name = classes.get(key, "insufficient_positive_history")
                actual = float(row["Unconstrained_Demand"])
                horizon = int(period.ordinal - origin.ordinal)
                for model_name, prediction in (
                    ("global_poisson_hgbr", float(global_predictions[row_index])),
                    ("croston_sba", croston_values.get(key, 0.0)),
                ):
                    origin_predictions.append({
                        "Origin": origin_text,
                        "Date": str(period),
                        "Horizon": horizon,
                        "Horizon_Bucket": (
                            "1-6" if horizon <= 6 else "7-12" if horizon <= 12 else "13-19"
                        ),
                        "SKU_ID": row["SKU_ID"],
                        "Region": row["Region"],
                        "Product_Family": row["Product_Family"],
                        "Intermittency_Class": class_name,
                        "Model": model_name,
                        "Actual": actual,
                        "Prediction": prediction,
                    })
                series_states[key][period] = float(global_predictions[row_index])
        origin_frame = pd.DataFrame(origin_predictions)
        outputs.append(origin_frame)
        origin_summary.append({
            "Origin": origin_text,
            "Holdout_Start": str(origin + 1),
            "Holdout_End": str(horizon_end),
            "Holdout_Months": int(observed_months),
            "Training_Rows": int(len(training)),
            "Evaluation_Rows": int(len(holdout)),
        })

    predictions = pd.concat(outputs, ignore_index=True)
    overall = _metrics(predictions, ["Origin", "Model"])
    by_horizon = _metrics(predictions, ["Origin", "Model", "Horizon_Bucket", "Horizon"])
    by_class = _metrics(
        predictions, ["Origin", "Model", "Intermittency_Class", "Horizon_Bucket"]
    )
    predictions.to_csv(
        output_dir / "rolling_origin_predictions.csv", index=False, encoding="utf-8-sig"
    )
    overall.to_csv(output_dir / "accuracy_overall.csv", index=False, encoding="utf-8-sig")
    by_horizon.to_csv(
        output_dir / "accuracy_by_horizon.csv", index=False, encoding="utf-8-sig"
    )
    by_class.to_csv(
        output_dir / "accuracy_by_intermittency.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "origins": origin_summary,
        "prediction_rows": int(len(predictions)),
        "models": ["global_poisson_hgbr", "croston_sba"],
        "metric_definitions": {
            "WAPE": "sum(abs(prediction - actual)) / sum(abs(actual)); lower is better.",
            "Bias": "sum(prediction - actual) / sum(abs(actual)); positive means over-forecast.",
        },
        "global_model_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "leakage_controls": [
            "Each model is trained only on rows dated on or before its validation origin.",
            "The global model recursively uses its own prior forecasts during the 19-month holdout.",
            "Target-derived contemporaneous fields and full-history intermittency labels are excluded from predictors.",
            "Intermittency classes for slicing are recalculated using only demand observed by each origin.",
            "Missing demand remains unknown and is excluded from evaluation rather than set to zero.",
        ],
        "outputs": str(output_dir),
    }
    with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    return summary
