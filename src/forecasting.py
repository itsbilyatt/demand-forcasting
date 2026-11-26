from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.global_model import (
    _design_matrix,
    _dynamic_features,
    _fit_model,
    _series_key,
)
from src.model_selection import run_model_selection

ORIGIN = pd.Period("2026-10", freq="M")
SERIES_KEYS = ["SKU_ID", "Region"]
SUBMISSION_COLUMNS = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]


def _flag(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "features").is_dir():
            return candidate
    raise FileNotFoundError("Run notebooks 02 and 04 before creating the final forecast.")


def _family_region_mean(
    means: pd.Series, family: Any, region: Any
) -> float:
    value = means.get((family, region), np.nan)
    return float(value) if pd.notna(value) else np.nan


def create_final_submission(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    history_path = (
        root / "outputs" / "demand_reconstruction" / "train_with_reconstructed_demand.csv"
    )
    test_path = root / "outputs" / "cleaned" / "test_cleaned.csv"
    feature_path = root / "outputs" / "features" / "test_features.csv"
    output_dir = root / "outputs" / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    features = pd.read_csv(feature_path, low_memory=False)
    master = pd.read_csv(root / "outputs" / "cleaned" / "sku_master_cleaned.csv", low_memory=False)
    events = pd.read_csv(root / "outputs" / "cleaned" / "events_calendar_cleaned.csv", low_memory=False)
    prices = pd.read_csv(root / "outputs" / "cleaned" / "train_cleaned.csv", low_memory=False)
    selection_path = root / "outputs" / "model_selection" / "model_selection_summary.json"
    if not selection_path.is_file():
        run_model_selection(root)
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    global_weight = float(selection["global_model_weight"])
    seasonal_weight = 1.0 - global_weight
    history["Unconstrained_Demand"] = pd.to_numeric(
        history["Unconstrained_Demand"], errors="coerce"
    )
    history = history.loc[
        history["Date"].between("2019-01", "2026-10")
        & history[["Date", *SERIES_KEYS]].notna().all(axis=1)
        & history["Unconstrained_Demand"].ge(0)
    ].drop_duplicates(["Date", *SERIES_KEYS]).copy()
    history["Date_Period"] = pd.PeriodIndex(history["Date"], freq="M")
    features["Date_Period"] = pd.PeriodIndex(features["Date"], freq="M")
    features = features.loc[features["Date_Period"].gt(ORIGIN)].copy()
    required_lifecycle = {"Is_Launched", "Is_Before_EOL"}
    if not required_lifecycle.issubset(features.columns):
        raise ValueError(f"Test features are missing lifecycle columns: {sorted(required_lifecycle - set(features.columns))}")

    expected_keys = set(
        map(
            tuple,
            test[["Date", *SERIES_KEYS]].drop_duplicates().itertuples(index=False, name=None),
        )
    )
    feature_keys = set(
        map(
            tuple,
            features[["Date", *SERIES_KEYS]].drop_duplicates().itertuples(index=False, name=None),
        )
    )
    if expected_keys != feature_keys:
        raise ValueError(
            f"Test feature keys do not match test data: "
            f"{len(expected_keys - feature_keys)} missing, "
            f"{len(feature_keys - expected_keys)} unexpected."
        )
    if features.duplicated(["Date", *SERIES_KEYS]).any():
        raise ValueError("Test features contain duplicate forecast keys.")

    family_region_means = history.groupby(
        ["Product_Family", "Region"], dropna=False
    )["Unconstrained_Demand"].mean()
    family_means = history.groupby("Product_Family", dropna=False)[
        "Unconstrained_Demand"
    ].mean()
    global_mean = float(history["Unconstrained_Demand"].mean())
    if not np.isfinite(global_mean):
        raise ValueError("No valid nonnegative historical demand is available for fallback.")

    training_features = pd.read_csv(
        root / "outputs" / "features" / "train_features.csv", low_memory=False
    )
    training_features["Unconstrained_Demand"] = pd.to_numeric(
        training_features["Unconstrained_Demand"], errors="coerce"
    )
    model, design_columns = _fit_model(training_features)
    replacement_by_sku = (
        master.dropna(subset=["Replaces_SKU"])
        .set_index("SKU_ID")["Replaces_SKU"]
        .astype(str)
        .to_dict()
    )
    price_rows = prices.loc[:, ["Date", "SKU_ID", "Unit_Price_USD"]].copy()
    price_rows["Unit_Price_USD"] = pd.to_numeric(price_rows["Unit_Price_USD"], errors="coerce")
    price_rows = price_rows.loc[
        price_rows["Date"].between("2019-01", str(ORIGIN))
        & price_rows["Unit_Price_USD"].gt(0)
    ]
    unit_price_by_sku = price_rows.sort_values("Date").groupby("SKU_ID")[
        "Unit_Price_USD"
    ].last().to_dict()

    price_events = []
    for event in events.itertuples(index=False):
        if "price change" not in str(getattr(event, "Event_Type", "")).casefold():
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(getattr(event, "Description", "")))
        start = str(getattr(event, "Start_Date", ""))[:7]
        if match and start:
            price_events.append((pd.Period(start, freq="M"), float(match.group(1)) / 100.0))

    global_states: dict[tuple[str, str], dict[pd.Period, float]] = {}
    history_values_by_series: dict[tuple[str, str], dict[pd.Period, float]] = {}
    recent_by_series: dict[tuple[str, str], list[float]] = {}
    for key, group in history.groupby(SERIES_KEYS, dropna=False, sort=False):
        normalized_key = tuple(str(value) for value in key)
        values = dict(zip(group["Date_Period"], group["Unconstrained_Demand"]))
        history_values_by_series[normalized_key] = values
        global_states[normalized_key] = values.copy()
        recent_by_series[normalized_key] = (
            group["Unconstrained_Demand"].dropna().tail(3).astype(float).tolist()
        )

    audit_rows: list[dict[str, Any]] = []
    seasonal_predictions: dict[tuple[tuple[str, str], pd.Period], float] = {}
    for period, month_rows in features.groupby("Date_Period", sort=True):
        dynamic_rows = []
        keys = []
        for _, row in month_rows.iterrows():
            key = _series_key(row)
            keys.append(key)
            state = global_states.setdefault(key, {})
            dynamic_rows.append(_dynamic_features(row, state))
        global_raw = np.maximum(
            0.0,
            model.predict(_design_matrix(pd.DataFrame(dynamic_rows), design_columns)),
        )
        for row_index, (_, row) in enumerate(month_rows.iterrows()):
            key = keys[row_index]
            history_values = history_values_by_series.get(key, {})
            recent_values = recent_by_series.get(key, [])
            source_period = period - 12
            seasonal_value = seasonal_predictions.get((key, source_period), np.nan)
            seasonal_source = "recursive_seasonal_naive_12" if source_period > ORIGIN else "seasonal_naive_12"
            if pd.isna(seasonal_value):
                seasonal_value = history_values.get(source_period, np.nan)
            replacement_sku = replacement_by_sku.get(str(row["SKU_ID"]))
            replacement_key = (replacement_sku, str(row["Region"])) if replacement_sku else None
            is_cold_start = not recent_values
            if is_cold_start and replacement_key is not None and _flag(row["Is_Launched"]):
                replacement_value = seasonal_predictions.get((replacement_key, source_period), np.nan)
                if pd.isna(replacement_value):
                    replacement_value = history_values_by_series.get(replacement_key, {}).get(source_period, np.nan)
                if pd.isna(replacement_value):
                    replacement_recent = recent_by_series.get(replacement_key, [])
                    replacement_value = float(np.mean(replacement_recent)) if replacement_recent else np.nan
                if pd.notna(replacement_value):
                    seasonal_value = float(replacement_value)
                    seasonal_source = f"replacement_{replacement_sku}_transfer_1.0"
            if pd.isna(seasonal_value):
                fallback = (
                    float(np.mean(recent_values))
                    if recent_values else _family_region_mean(
                        family_region_means, row.get("Product_Family"), row.get("Region")
                    )
                )
                if pd.isna(fallback):
                    fallback = family_means.get(row.get("Product_Family"), global_mean)
                seasonal_value = float(fallback)
                seasonal_source = "recent_series_mean_3" if recent_values else "hierarchy_mean_fallback"

            global_value = float(global_raw[row_index])
            if is_cold_start and replacement_key is not None and _flag(row["Is_Launched"]):
                if seasonal_source.startswith("replacement_"):
                    global_value = float(seasonal_value)

            active = _flag(row["Is_Launched"]) and _flag(row["Is_Before_EOL"])
            point = max(
                0.0,
                seasonal_weight * float(seasonal_value) + global_weight * global_value,
            ) if active else 0.0
            seasonal_predictions[(key, period)] = point
            global_states[key][period] = point
            recent_scale_values = [
                seasonal_predictions.get((key, period - offset),
                                         history_values.get(period - offset))
                for offset in range(1, 13)
            ]
            recent_scale_values = [
                float(item) for item in recent_scale_values if pd.notna(item)
            ]
            forecast_scale = (
                max(1.0, float(np.mean(np.abs(recent_scale_values))))
                if recent_scale_values else max(1.0, global_mean)
            )
            own_price = unit_price_by_sku.get(str(row["SKU_ID"]))
            price_source = "latest_same_sku_price"
            if own_price is None and replacement_sku:
                own_price = unit_price_by_sku.get(replacement_sku)
                price_source = f"replaced_sku_{replacement_sku}_price_proxy"
            if own_price is None:
                price_source = "unavailable"
            price = float(own_price) if own_price is not None else np.nan
            for event_period, increase in price_events:
                if event_period > ORIGIN and event_period <= period and np.isfinite(price):
                    price *= 1.0 + increase
            audit_rows.append({
                "Date": row["Date"],
                "SKU_ID": row["SKU_ID"],
                "Region": row["Region"],
                "Product_Family": row.get("Product_Family", ""),
                "Forecast_Raw": point,
                "Forecast_Scale": forecast_scale,
                "Units_Sold_Forecast": int(np.rint(point)),
                "Forecast_Method": f"blend_{seasonal_source}_global_hgbr",
                "Seasonal_Prediction": float(seasonal_value),
                "Global_Prediction": global_value,
                "Is_Launched": _flag(row["Is_Launched"]),
                "Is_Before_EOL": _flag(row["Is_Before_EOL"]),
                "Known_Scoped_Event_Count": int(row["Known_Scoped_Event_Count"]),
                "Unit_Price_Estimate_USD": price,
                "Unit_Price_Source": price_source,
                "Revenue_P50_Estimate_USD": point * price if np.isfinite(price) else np.nan,
            })

    audit = pd.DataFrame(audit_rows)
    submission = audit[SUBMISSION_COLUMNS].copy()
    if len(submission) != len(test):
        raise ValueError(
            f"Forecast row count {len(submission)} does not match test row count {len(test)}."
        )
    if submission.duplicated(["Date", *SERIES_KEYS]).any():
        raise ValueError("Submission contains duplicate forecast keys.")
    if submission[SUBMISSION_COLUMNS].isna().any().any():
        raise ValueError("Submission contains blank required fields.")
    if not pd.api.types.is_integer_dtype(submission["Units_Sold_Forecast"]):
        raise TypeError("Forecast values must be integers.")
    if submission["Units_Sold_Forecast"].lt(0).any():
        raise ValueError("Forecast values must be nonnegative.")

    expected_submission = test[["Date", *SERIES_KEYS]].sort_values(
        ["Date", *SERIES_KEYS]
    ).reset_index(drop=True)
    actual_submission = submission[["Date", *SERIES_KEYS]].sort_values(
        ["Date", *SERIES_KEYS]
    ).reset_index(drop=True)
    if not expected_submission.equals(actual_submission):
        raise ValueError("Submission keys do not exactly match the cleaned test keys.")

    submission = submission[SUBMISSION_COLUMNS]
    submission_path = output_dir / "final_submission.csv"
    audit_path = output_dir / "forecast_audit.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    method_counts = audit["Forecast_Method"].value_counts().to_dict()
    summary = {
        "selected_method": "validated_blend",
        "seasonal_naive_weight": seasonal_weight,
        "global_model_weight": global_weight,
        "selection_basis": selection["selection_method"],
        "history_origin": str(ORIGIN),
        "forecast_rows": int(len(submission)),
        "unique_forecast_keys": int(submission[["Date", *SERIES_KEYS]].drop_duplicates().shape[0]),
        "forecast_start": str(submission["Date"].min()),
        "forecast_end": str(submission["Date"].max()),
        "method_counts": method_counts,
        "replacement_transfer_assumption": (
            "For a launched replacement SKU with no own demand history, transfer the replaced "
            "SKU-region seasonal forecast at a 1.0 ratio until the successor has its own history."
        ),
        "price_revenue_assumption": (
            "Revenue estimates use the latest positive observed SKU price through Oct 2026 and "
            "apply explicitly listed future global price-change events; no demand elasticity is assumed."
        ),
        "submission_path": str(submission_path),
        "audit_path": str(audit_path),
    }
    with (output_dir / "submission_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary
