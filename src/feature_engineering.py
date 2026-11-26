from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MACRO_COLUMNS = (
    "ISM_Manufacturing_PMI",
    "Auto_Production_YoY_Pct",
    "Vehicle_Inventory_Days",
    "Raw_Material_Cost_Index",
    "New_Vehicle_Orders_Index",
    "Industrial_Util_Rate_Pct",
    "Fuel_Price_Index",
    "EV_Adoption_Index",
)


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "demand_reconstruction").is_dir():
            return candidate
    raise FileNotFoundError("Run notebooks 01 and 02 before feature engineering.")


def _periods(series: pd.Series) -> pd.Series:
    def parse(value):
        if pd.isna(value) or not str(value).strip():
            return pd.NaT
        return pd.Period(str(value)[:7], freq="M")

    return series.map(parse)


def run_feature_engineering(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    clean_dir = root / "outputs" / "cleaned"
    recon_dir = root / "outputs" / "demand_reconstruction"
    exploration_dir = root / "outputs" / "demand_exploration"
    output_dir = root / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(
        recon_dir / "train_with_reconstructed_demand.csv", low_memory=False
    )
    test = pd.read_csv(clean_dir / "test_cleaned.csv", low_memory=False)
    macro = pd.read_csv(clean_dir / "macro_indicators_cleaned.csv", low_memory=False)
    events = pd.read_csv(clean_dir / "events_calendar_cleaned.csv", low_memory=False)
    master = pd.read_csv(clean_dir / "sku_master_cleaned.csv", low_memory=False)
    profiles = pd.read_csv(exploration_dir / "demand_profile_by_series.csv", low_memory=False)

    history["Unconstrained_Demand"] = pd.to_numeric(history["Unconstrained_Demand"], errors="coerce")
    history_window = history["Date"].between("2019-01", "2026-10")
    history_valid = history.loc[
        history_window & history[["Date", "SKU_ID", "Region"]].notna().all(axis=1)
    ].copy()
    history_valid = history_valid.drop_duplicates(["Date", "SKU_ID", "Region"])
    test_valid = test.dropna(subset=["Date", "SKU_ID", "Region"]).copy()
    series_keys = pd.concat(
        [history_valid[["SKU_ID", "Region"]], test_valid[["SKU_ID", "Region"]]],
        ignore_index=True,
    ).drop_duplicates()
    periods = pd.period_range("2019-01", "2028-05", freq="M")
    calendar = pd.DataFrame({"Date_Period": periods})
    calendar["Date"] = calendar["Date_Period"].astype(str)
    grid = series_keys.merge(calendar, how="cross")

    history_columns = [
        "Date", "SKU_ID", "Region", "Unconstrained_Demand", "Product_Family",
        "Sales_Channel", "Customer_Name", "Stockout_Record_Found",
        "Demand_Confidence",
    ]
    history_columns = [column for column in history_columns if column in history_valid]
    grid = grid.merge(
        history_valid[history_columns],
        on=["Date", "SKU_ID", "Region"],
        how="left",
        validate="one_to_one",
    )
    test_columns = [
        "Date", "SKU_ID", "Region", "Product_Family", "Business_Unit",
        "SKU_Description", "Customer_Name", "Sales_Channel",
    ]
    grid = grid.merge(
        test_valid[test_columns].drop_duplicates(["Date", "SKU_ID", "Region"]),
        on=["Date", "SKU_ID", "Region"],
        how="left",
        suffixes=("", "_test"),
        validate="one_to_one",
    )
    for column in ("Product_Family", "Customer_Name", "Sales_Channel"):
        test_column = f"{column}_test"
        if test_column in grid:
            grid[column] = grid[test_column].combine_first(grid.get(column))
            grid.drop(columns=[test_column], inplace=True)
    if "Business_Unit" not in grid and "Business_Unit_test" in grid:
        grid["Business_Unit"] = grid["Business_Unit_test"]
    grid.drop(columns=["Business_Unit_test", "SKU_Description_test"], errors="ignore", inplace=True)

    grid = grid.sort_values(["SKU_ID", "Region", "Date_Period"]).reset_index(drop=True)
    group = grid.groupby(["SKU_ID", "Region"], dropna=False, sort=False)["Unconstrained_Demand"]
    for lag in (1, 2, 3, 6, 12):
        grid[f"lag_{lag}"] = group.shift(lag)
    for window in (3, 6, 12):
        grid[f"rolling_mean_{window}"] = group.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
        grid[f"rolling_observed_count_{window}"] = group.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).count()
        )
    grid["zero_share_prior_12"] = group.transform(
        lambda values: values.shift(1).eq(0).rolling(12, min_periods=1).mean()
    )
    grid["Month"] = grid["Date_Period"].dt.month
    grid["Year"] = grid["Date_Period"].dt.year
    grid["Month_Sin"] = np.sin(2 * math.pi * grid["Month"] / 12)
    grid["Month_Cos"] = np.cos(2 * math.pi * grid["Month"] / 12)
    origin = pd.Period("2026-10", freq="M")
    grid["Forecast_Horizon"] = grid["Date_Period"].map(
        lambda period: period.ordinal - origin.ordinal
    )
    grid["Is_Forecast_Period"] = grid["Forecast_Horizon"].gt(0)

    event_counts: dict[tuple[pd.Period, str, str], int] = {}
    for event in events.itertuples(index=False):
        start = _periods(pd.Series([getattr(event, "Start_Date", "")])).iat[0]
        end = _periods(pd.Series([getattr(event, "End_Date", "")])).iat[0]
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = pd.Period("2028-05", freq="M")
        scope_type = str(getattr(event, "Scope_Type", "")).strip().casefold()
        scope_value = str(getattr(event, "Scope_Value", "")).strip().casefold()
        for period in pd.period_range(start, end, freq="M"):
            key = (period, scope_type, scope_value)
            event_counts[key] = event_counts.get(key, 0) + 1

    def scoped_event_count(row) -> int:
        period = row["Date_Period"]
        counts = event_counts.get((period, "global", ""), 0)
        for scope, column in (
            ("region", "Region"),
            ("product_family", "Product_Family"),
            ("channel", "Sales_Channel"),
        ):
            value = str(row.get(column, "")).strip().casefold()
            counts += event_counts.get((period, scope, value), 0)
        customer = str(row.get("Customer_Name", "")).strip().casefold()
        region = str(row.get("Region", "")).strip().casefold()
        counts += event_counts.get((period, "customer", customer), 0)
        counts += event_counts.get((period, "customer", f"{customer}|{region}"), 0)
        return counts

    grid["Known_Scoped_Event_Count"] = [
        scoped_event_count(row) for row in grid[
            ["Date_Period", "Region", "Product_Family", "Sales_Channel", "Customer_Name"]
        ].to_dict("records")
    ]

    macro["Date_Period"] = _periods(macro["Date"])
    for column in MACRO_COLUMNS:
        macro[column] = pd.to_numeric(macro[column], errors="coerce")
    # Conflicting same-month macro rows are averaged for feature use and remain flagged in the quality report.
    macro_month = macro.groupby("Date_Period")[list(MACRO_COLUMNS)].mean()
    for lag in (1, 3):
        target_period = grid["Date_Period"] - lag
        for column in MACRO_COLUMNS:
            lookup = macro_month[column]
            grid[f"{column}_lag_{lag}"] = target_period.map(lookup)

    master["Launch_Period"] = _periods(master["Launch_Date"])
    master["EOL_Period"] = _periods(master["EOL_Date"])
    lifecycle = master.set_index("SKU_ID")
    grid["Launch_Period"] = grid["SKU_ID"].map(lifecycle["Launch_Period"])
    grid["EOL_Period"] = grid["SKU_ID"].map(lifecycle["EOL_Period"])
    grid["Months_Since_Launch"] = [
        (period.ordinal - launch.ordinal) if pd.notna(launch) else np.nan
        for period, launch in zip(grid["Date_Period"], grid["Launch_Period"])
    ]
    grid["Months_To_EOL"] = [
        (eol.ordinal - period.ordinal) if pd.notna(eol) else np.nan
        for period, eol in zip(grid["Date_Period"], grid["EOL_Period"])
    ]
    grid["Is_Launched"] = grid["Months_Since_Launch"].ge(0)
    grid["Is_Before_EOL"] = grid["Months_To_EOL"].isna() | grid["Months_To_EOL"].ge(0)
    grid.drop(columns=["Launch_Period", "EOL_Period"], inplace=True)

    profile_columns = [
        "SKU_ID", "Region", "ADI", "CV2_Positive_Demand", "Intermittency_Class",
    ]
    grid = grid.merge(
        profiles[profile_columns].drop_duplicates(["SKU_ID", "Region"]),
        on=["SKU_ID", "Region"],
        how="left",
        validate="many_to_one",
    )
    grid["Stockout_Record_Found"] = grid["Stockout_Record_Found"].fillna(False).astype(bool)
    grid["Demand_Confidence"] = grid["Demand_Confidence"].fillna("future_or_unobserved")

    train_features = grid.loc[
        grid["Date"].between("2019-01", "2026-10")
        & grid["Unconstrained_Demand"].notna()
    ].copy()
    test_features = grid.loc[
        grid["Date"].between("2026-11", "2028-05")
    ].merge(
        test_valid[["Date", "SKU_ID", "Region"]].drop_duplicates(),
        on=["Date", "SKU_ID", "Region"],
        how="inner",
        validate="one_to_one",
    )
    if test_features.duplicated(["Date", "SKU_ID", "Region"]).any():
        raise ValueError("Feature output contains duplicate forecast keys.")
    if len(test_features) != len(test_valid):
        raise ValueError(
            f"Feature rows ({len(test_features)}) do not match test keys ({len(test_valid)})."
        )

    dropped_history_rows = int(len(history) - len(history_valid))
    train_features.to_csv(output_dir / "train_features.csv", index=False, encoding="utf-8-sig")
    test_features.to_csv(output_dir / "test_features.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "train_feature_rows": int(len(train_features)),
        "test_feature_rows": int(len(test_features)),
        "dropped_history_rows_missing_key_or_outside_window": dropped_history_rows,
        "feature_columns": list(test_features.columns),
        "lag_features": [f"lag_{lag}" for lag in (1, 2, 3, 6, 12)],
        "macro_features": [
            f"{column}_lag_{lag}" for column in MACRO_COLUMNS for lag in (1, 3)
        ],
        "leakage_controls": [
            "Demand lags and rolling features are shifted by one calendar month on a complete series-month grid.",
            "Missing months and missing target values remain missing; they are not converted to zero.",
            "Macro features use only 1- and 3-month lags, not same-month realized indicators.",
            "Future forecast features use history available through Oct 2026 and known test metadata.",
            "Forecast horizon is measured from the Oct 2026 forecast origin.",
        ],
        "outputs": {
            "train_features": str(output_dir / "train_features.csv"),
            "test_features": str(output_dir / "test_features.csv"),
        },
    }
    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True, default=str)
    return metadata
