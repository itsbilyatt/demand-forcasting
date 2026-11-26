from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DASHBOARD_COLUMNS = [
    "Date",
    "SKU_ID",
    "Region",
    "Product_Family",
    "Forecast_Method",
    "P10",
    "P50",
    "P90",
    "Known_Scoped_Event_Count",
    "Is_Launched",
    "Is_Before_EOL",
    "Unit_Price_Estimate_USD",
    "Revenue_P50_Estimate_USD",
]
REQUIRED_FORECAST_COLUMNS = [
    "Date",
    "SKU_ID",
    "Region",
    "Product_Family",
    "Forecast_Method",
    "P10",
    "P50",
    "P90",
    "Known_Scoped_Event_Count",
    "Is_Launched",
    "Is_Before_EOL",
]


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "business_checks").is_dir():
            return candidate
    raise FileNotFoundError("Run notebook 08 before building the local dashboard.")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def _pack_records(
    frame: pd.DataFrame,
    columns: list[str],
    categories: set[str],
    dictionaries: dict[str, dict[str, int]],
) -> list[list[Any]]:
    packed = []
    for values in frame[columns].itertuples(index=False, name=None):
        row = []
        for column, value in zip(columns, values):
            if pd.isna(value):
                row.append(-1 if column in categories else None)
            elif column in categories:
                row.append(dictionaries[column][str(value)])
            elif isinstance(value, np.generic):
                row.append(value.item())
            else:
                row.append(value)
        packed.append(row)
    return packed


def build_local_dashboard(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    template_path = root / "app" / "index.html"
    forecast_path = root / "outputs" / "business_checks" / "forecast_quantiles.csv"
    summary_path = root / "outputs" / "business_checks" / "business_check_summary.json"
    coverage_path = (
        root / "outputs" / "business_checks" / "interval_leave_one_origin_out_coverage.csv"
    )
    history_path = (
        root / "outputs" / "demand_reconstruction" / "train_with_reconstructed_demand.csv"
    )
    anomaly_path = root / "outputs" / "demand_exploration" / "anomaly_register.csv"
    issue_path = root / "outputs" / "data_quality" / "data_quality_issues.csv"
    correction_path = root / "outputs" / "data_quality" / "data_corrections.csv"
    price_log_path = root / "outputs" / "data_quality" / "price_correction_log.csv"
    if not template_path.is_file():
        raise FileNotFoundError(f"Dashboard template not found: {template_path}")

    forecasts = pd.read_csv(forecast_path, low_memory=False)
    missing_columns = sorted(set(DASHBOARD_COLUMNS) - set(forecasts.columns))
    if missing_columns:
        raise ValueError(f"Forecast dashboard data is missing columns: {missing_columns}")
    if forecasts[REQUIRED_FORECAST_COLUMNS].isna().any().any():
        raise ValueError("Dashboard forecast data contains blank required values.")
    if forecasts.duplicated(["Date", "SKU_ID", "Region"]).any():
        raise ValueError("Dashboard forecast data contains duplicate SKU-region-month keys.")

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    coverage = pd.read_csv(coverage_path, low_memory=False)
    with (root / "outputs" / "data_quality" / "data_quality_summary.json").open(
        encoding="utf-8"
    ) as handle:
        quality_summary = json.load(handle)
    history = pd.read_csv(history_path, low_memory=False)
    for column in ("Unconstrained_Demand", "Unit_Price_USD"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.loc[
        history["Date"].between("2019-01", "2026-10")
        & history["Unconstrained_Demand"].notna()
    ].copy()
    history["Estimated_Revenue_USD"] = (
        history["Unconstrained_Demand"] * history["Unit_Price_USD"]
    )
    history_columns = [
        "Date",
        "SKU_ID",
        "Region",
        "Product_Family",
        "Sales_Channel",
        "Customer_Name",
        "Unconstrained_Demand",
        "Unit_Price_USD",
        "Estimated_Revenue_USD",
        "Demand_Estimate_Source",
    ]
    history = history[[column for column in history_columns if column in history]]

    accuracy = {}
    for name in ("overall", "family", "region", "horizon", "family_region"):
        accuracy_path = root / "outputs" / "model_selection" / f"accuracy_{name}.csv"
        accuracy[name] = _json_records(pd.read_csv(accuracy_path, low_memory=False))
    anomalies = pd.read_csv(anomaly_path, low_memory=False)
    issues = pd.read_csv(issue_path, low_memory=False)
    corrections = pd.read_csv(correction_path, low_memory=False)
    price_corrections = (
        pd.read_csv(price_log_path, low_memory=False).head(500)
        if price_log_path.is_file()
        else pd.DataFrame()
    )
    lifecycle = pd.read_csv(
        root / "outputs" / "cleaned" / "sku_master_cleaned.csv", low_memory=False
    )[["SKU_ID", "Launch_Date", "EOL_Date", "Replaces_SKU", "Lifecycle_Status"]]
    forecast_columns = DASHBOARD_COLUMNS
    category_columns = {
        "Date",
        "SKU_ID",
        "Region",
        "Product_Family",
        "Sales_Channel",
        "Customer_Name",
        "Demand_Estimate_Source",
        "Forecast_Method",
    }
    dimensions: dict[str, list[str]] = {}
    for column in category_columns:
        source_frames = [
            frame[column]
            for frame in (history, forecasts)
            if column in frame.columns
        ]
        values = pd.concat(source_frames, ignore_index=True).dropna().astype(str).unique()
        dimensions[column] = sorted(values.tolist())
    dictionaries = {
        column: {value: index for index, value in enumerate(values)}
        for column, values in dimensions.items()
    }
    payload = {
        "summary": summary,
        "coverage": _json_records(coverage),
        "dimensions": dimensions,
        "forecast_columns": forecast_columns,
        "history_columns": history.columns.tolist(),
        "rows": _pack_records(forecasts, forecast_columns, category_columns, dictionaries),
        "history": _pack_records(history, history.columns.tolist(), category_columns, dictionaries),
        "anomalies": _json_records(anomalies),
        "issues": _json_records(issues),
        "file_metrics": quality_summary.get("file_metrics", {}),
        "corrections": _json_records(corrections),
        "price_corrections": _json_records(price_corrections),
        "lifecycle": _json_records(lifecycle),
        "accuracy": accuracy,
    }
    data = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    template = template_path.read_text(encoding="utf-8")
    if template.count("__DASHBOARD_DATA__") != 1:
        raise ValueError("Dashboard template must contain exactly one data placeholder.")
    rendered = template.replace("__DASHBOARD_DATA__", data)

    output_dir = root / "outputs" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index.html"
    output_path.write_text(rendered, encoding="utf-8")
    return {
        "dashboard_path": str(output_path),
        "forecast_rows_embedded": int(len(forecasts)),
        "forecast_months": int(forecasts["Date"].nunique()),
        "history_rows_embedded": int(len(history)),
        "anomaly_rows_embedded": int(len(anomalies)),
        "price_corrections_previewed": int(len(price_corrections)),
        "shared_filters": ["Product_Family", "SKU_ID", "Region", "Sales_Channel", "Customer_Name"],
        "views": [
            "executive_overview",
            "data_health",
            "demand_explorer",
            "anomaly_monitor",
            "forecast_accuracy",
        ],
        "bytes": int(output_path.stat().st_size),
        "external_dependencies": 0,
    }


if __name__ == "__main__":
    print(json.dumps(build_local_dashboard(), indent=2))
