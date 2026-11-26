from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ORIGIN = pd.Period("2026-10", freq="M")
GROUP_KEYS = ["Product_Family", "Horizon_Bucket"]
TEST_FEATURE_COLUMNS = [
    "Date",
    "SKU_ID",
    "Region",
    "Product_Family",
    "Known_Scoped_Event_Count",
    "Is_Launched",
    "Is_Before_EOL",
    "Months_Since_Launch",
    "Months_To_EOL",
]


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "submission").is_dir():
            return candidate
    raise FileNotFoundError("Run notebook 07 before uncertainty and business checks.")


def _quantile_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, dropna=False)["Normalized_Error"]
        .agg(
            Residual_Count="count",
            Residual_Q10=lambda values: values.quantile(0.10),
            Residual_Q90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )


def _bounds(point: np.ndarray, scale: np.ndarray, q10: np.ndarray, q90: np.ndarray):
    lower = np.maximum(0.0, point + scale * q10)
    upper = np.maximum(point, point + scale * q90)
    lower = np.minimum(lower, point)
    return lower, upper


def run_business_checks(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    output_dir = root / "outputs" / "business_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(root / "outputs" / "submission" / "forecast_audit.csv", low_memory=False)
    features = pd.read_csv(
        root / "outputs" / "features" / "test_features.csv", low_memory=False
    )[TEST_FEATURE_COLUMNS]
    feature_columns_to_add = [
        column for column in TEST_FEATURE_COLUMNS
        if column not in audit.columns
        and column not in {"Date", "SKU_ID", "Region", "Product_Family"}
    ]
    audit = audit.merge(
        features[["Date", "SKU_ID", "Region", "Product_Family", *feature_columns_to_add]],
        on=["Date", "SKU_ID", "Region", "Product_Family"],
        how="left",
        validate="one_to_one",
    )
    if audit["Known_Scoped_Event_Count"].isna().any():
        raise ValueError("A forecast row is missing known event or lifecycle features.")
    audit["Date_Period"] = pd.PeriodIndex(audit["Date"], freq="M")
    audit["Horizon"] = audit["Date_Period"].map(lambda period: period.ordinal - ORIGIN.ordinal)
    audit["Horizon_Bucket"] = np.select(
        [audit["Horizon"].le(6), audit["Horizon"].le(12)],
        ["1-6", "7-12"],
        default="13-19",
    )

    residuals = pd.read_csv(
        root / "outputs" / "model_selection" / "rolling_origin_predictions.csv",
        low_memory=False,
    )
    residuals["Forecast_Scale"] = np.maximum(
        residuals["Prediction"].abs().to_numpy(dtype=float), 1.0
    )
    residuals["Normalized_Error"] = (
        residuals["Actual"] - residuals["Prediction"]
    ) / residuals["Forecast_Scale"]

    calibrations = _quantile_table(residuals, GROUP_KEYS)
    global_calibrations = _quantile_table(residuals, ["Horizon_Bucket"]).rename(
        columns={
            "Residual_Count": "Global_Residual_Count",
            "Residual_Q10": "Global_Residual_Q10",
            "Residual_Q90": "Global_Residual_Q90",
        }
    )
    calibrations.to_csv(
        output_dir / "interval_calibration_by_family_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    family_quantiles = {
        (row.Product_Family, row.Horizon_Bucket): (
            float(row.Residual_Q10),
            float(row.Residual_Q90),
            int(row.Residual_Count),
        )
        for row in calibrations.itertuples(index=False)
    }
    global_quantiles = {
        row.Horizon_Bucket: (
            float(row.Global_Residual_Q10),
            float(row.Global_Residual_Q90),
            int(row.Global_Residual_Count),
        )
        for row in global_calibrations.itertuples(index=False)
    }

    bounds = []
    for row in audit.itertuples(index=False):
        quantiles = family_quantiles.get((row.Product_Family, row.Horizon_Bucket))
        if quantiles is None or quantiles[2] < 30:
            quantiles = global_quantiles[row.Horizon_Bucket][:2]
        else:
            quantiles = quantiles[:2]
        lower, upper = _bounds(
            np.array([row.Forecast_Raw], dtype=float),
            np.array([row.Forecast_Scale], dtype=float),
            np.array([quantiles[0]], dtype=float),
            np.array([quantiles[1]], dtype=float),
        )
        bounds.append((float(lower[0]), float(row.Forecast_Raw), float(upper[0])))
    audit[["P10", "P50", "P90"]] = pd.DataFrame(bounds, index=audit.index)
    if not (audit["P10"].le(audit["P50"]) & audit["P50"].le(audit["P90"])).all():
        raise ValueError("Forecast intervals must satisfy P10 <= P50 <= P90.")
    inactive = ~audit["Is_Launched"].fillna(False).astype(bool) | ~audit[
        "Is_Before_EOL"
    ].fillna(False).astype(bool)
    audit.loc[inactive, ["Forecast_Raw", "Units_Sold_Forecast", "P10", "P50", "P90"]] = 0
    inactive_values = audit.loc[
        inactive, ["Forecast_Raw", "Units_Sold_Forecast", "P10", "P50", "P90"]
    ].apply(pd.to_numeric, errors="coerce")
    if inactive_values.gt(0).any().any():
        raise ValueError("Pre-launch or post-EOL forecasts and intervals must all be zero.")

    interval_validation = []
    for origin in residuals["Origin"].unique():
        calibration = residuals.loc[residuals["Origin"].ne(origin)]
        validation = residuals.loc[residuals["Origin"].eq(origin)].copy()
        family_cal = _quantile_table(calibration, GROUP_KEYS)
        global_cal = _quantile_table(calibration, ["Horizon_Bucket"])
        family_lookup = {
            (row.Product_Family, row.Horizon_Bucket): (
                float(row.Residual_Q10), float(row.Residual_Q90), int(row.Residual_Count)
            )
            for row in family_cal.itertuples(index=False)
        }
        global_lookup = {
            row.Horizon_Bucket: (float(row.Residual_Q10), float(row.Residual_Q90))
            for row in global_cal.itertuples(index=False)
        }
        q10_values, q90_values = [], []
        for row in validation.itertuples(index=False):
            quantiles = family_lookup.get((row.Product_Family, row.Horizon_Bucket))
            if quantiles is None or quantiles[2] < 30:
                quantiles = global_lookup[row.Horizon_Bucket]
            else:
                quantiles = quantiles[:2]
            q10_values.append(quantiles[0])
            q90_values.append(quantiles[1])
        points = validation["Prediction"].to_numpy(dtype=float)
        scales = validation["Forecast_Scale"].to_numpy(dtype=float)
        lower, upper = _bounds(
            points, scales, np.asarray(q10_values), np.asarray(q90_values)
        )
        actuals = validation["Actual"].to_numpy(dtype=float)
        interval_validation.append({
            "Origin": origin,
            "Rows": int(len(validation)),
            "Empirical_Coverage": float(((actuals >= lower) & (actuals <= upper)).mean()),
            "Mean_Interval_Width": float(np.mean(upper - lower)),
        })
    interval_validation_frame = pd.DataFrame(interval_validation)
    interval_validation_frame.to_csv(
        output_dir / "interval_leave_one_origin_out_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    monthly_total = (
        audit.groupby("Date", as_index=False)
        .agg(
            Sum_SKU_P10=("P10", "sum"),
            P50=("P50", "sum"),
            Sum_SKU_P90=("P90", "sum"),
            SKU_Region_Rows=("SKU_ID", "size"),
            Event_Exposed_Rows=("Known_Scoped_Event_Count", lambda values: int(values.gt(0).sum())),
        )
    )
    monthly_family = (
        audit.groupby(["Date", "Product_Family"], as_index=False)
        .agg(
            Sum_SKU_P10=("P10", "sum"),
            P50=("P50", "sum"),
            Sum_SKU_P90=("P90", "sum"),
        )
    )
    monthly_region = (
        audit.groupby(["Date", "Region"], as_index=False)
        .agg(
            Sum_SKU_P10=("P10", "sum"),
            P50=("P50", "sum"),
            Sum_SKU_P90=("P90", "sum"),
        )
    )
    for table, filename in (
        (monthly_total, "monthly_total_forecast.csv"),
        (monthly_family, "monthly_forecast_by_family.csv"),
        (monthly_region, "monthly_forecast_by_region.csv"),
    ):
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    family_sums = monthly_family.groupby("Date")["P50"].sum().sort_index()
    region_sums = monthly_region.groupby("Date")["P50"].sum().sort_index()
    total_sums = monthly_total.set_index("Date")["P50"].sort_index()
    hierarchy_reconciles = bool(
        np.allclose(family_sums.to_numpy(), total_sums.to_numpy())
        and np.allclose(region_sums.to_numpy(), total_sums.to_numpy())
    )
    if not hierarchy_reconciles:
        raise ValueError("Family/region aggregates do not reconcile to the total forecast.")

    forecast_units = pd.to_numeric(audit["Units_Sold_Forecast"], errors="raise")
    prelaunch = ~audit["Is_Launched"].fillna(False).astype(bool)
    post_eol = ~audit["Is_Before_EOL"].fillna(True).astype(bool)
    event_exposed = pd.to_numeric(audit["Known_Scoped_Event_Count"], errors="coerce").gt(0)
    summary = {
        "forecast_rows": int(len(audit)),
        "forecast_months": int(audit["Date"].nunique()),
        "total_p50_units": float(audit["P50"].sum()),
        "sum_of_sku_p10_units": float(audit["P10"].sum()),
        "sum_of_sku_p90_units": float(audit["P90"].sum()),
        "zero_forecast_rows": int(forecast_units.eq(0).sum()),
        "prelaunch_rows": int(prelaunch.sum()),
        "prelaunch_forecast_units": int(forecast_units.loc[prelaunch].sum()),
        "post_eol_rows": int(post_eol.sum()),
        "post_eol_forecast_units": int(forecast_units.loc[post_eol].sum()),
        "event_exposed_rows": int(event_exposed.sum()),
        "event_exposed_forecast_units": int(forecast_units.loc[event_exposed].sum()),
        "hierarchy_reconciles": hierarchy_reconciles,
        "leave_one_origin_out_interval_coverage": interval_validation_frame.to_dict("records"),
        "procurement_note": (
            "No on-hand inventory, open purchase orders, lead times, or service-level inputs "
            "are present; reorder quantities cannot be responsibly calculated from demand alone."
        ),
        "assumptions": [
            "P50 is the selected rolling-origin-validated seasonal-naive/global-model blend.",
            "P10/P90 are approximate empirical intervals from normalized rolling-origin residual quantiles by family and horizon bucket.",
            "Leave-one-origin-out coverage is reported to avoid evaluating each origin on its own calibration residuals.",
            "Monthly/family/region P10 and P90 columns are sums of SKU-level interval bounds, not aggregate-level quantiles.",
            "Forecasts and intervals are forced to zero before launch and after the inclusive EOL month.",
            "Known events are exposed for review; demand effects are not applied without validated causal evidence.",
        ],
        "outputs": str(output_dir),
    }
    audit.drop(columns=["Date_Period"], inplace=True)
    audit.to_csv(
        output_dir / "forecast_quantiles.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "business_check_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    return summary
