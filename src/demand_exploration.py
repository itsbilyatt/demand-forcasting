from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "demand_reconstruction").is_dir():
            return candidate
    raise FileNotFoundError("Run notebook 02 first; demand reconstruction output was not found.")


def _month(value: Any) -> pd.Period | None:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "nat", "none"}:
        return None
    try:
        return pd.Period(text[:7], freq="M")
    except (TypeError, ValueError):
        return None


def _build_event_index(events: pd.DataFrame, max_period: pd.Period) -> dict:
    event_index: dict[tuple[pd.Period, str, str], list[str]] = {}
    for event in events.itertuples(index=False):
        start = _month(getattr(event, "Start_Date", ""))
        end = _month(getattr(event, "End_Date", "")) or max_period
        if start is None or end < start:
            continue
        scope_type = str(getattr(event, "Scope_Type", "")).strip().casefold()
        scope_value = str(getattr(event, "Scope_Value", "")).strip().casefold()
        event_id = str(getattr(event, "Event_ID", ""))
        for period in pd.period_range(start, end, freq="M"):
            event_index.setdefault((period, scope_type, scope_value), []).append(event_id)
    return event_index


def _attach_events(row: dict[str, Any], event_index: dict) -> list[str]:
    period = _month(row.get("Date"))
    if period is None:
        return []
    matched = list(event_index.get((period, "global", ""), []))
    for scope_type, field in (
        ("region", "Region"),
        ("product_family", "Product_Family"),
        ("channel", "Sales_Channel"),
    ):
        value = str(row.get(field, "")).strip().casefold()
        matched.extend(event_index.get((period, scope_type, value), []))
    customer = str(row.get("Customer_Name", "")).strip().casefold()
    region = str(row.get("Region", "")).strip().casefold()
    matched.extend(event_index.get((period, "customer", customer), []))
    matched.extend(event_index.get((period, "customer", f"{customer}|{region}"), []))
    return matched


def run_demand_exploration(project_root: str | Path | None = None) -> dict[str, Any]:
    started = perf_counter()
    root = _project_root(project_root)
    demand_dir = root / "outputs" / "demand_reconstruction"
    clean_dir = root / "outputs" / "cleaned"
    output_dir = root / "outputs" / "demand_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        demand_dir / "train_with_reconstructed_demand.csv", low_memory=False
    )
    events = pd.read_csv(clean_dir / "events_calendar_cleaned.csv", low_memory=False)
    master = pd.read_csv(clean_dir / "sku_master_cleaned.csv", low_memory=False)
    event_index = _build_event_index(events, pd.Period("2028-05", freq="M"))
    print(f"Loaded inputs and indexed events in {perf_counter() - started:.1f}s", flush=True)
    data["Unconstrained_Demand"] = pd.to_numeric(data["Unconstrained_Demand"], errors="coerce")
    data["Date_Period"] = data["Date"].map(_month)
    expected_window = data["Date"].between("2019-01", "2026-10")
    history = data.loc[expected_window].copy()

    profiles: list[dict[str, Any]] = []
    for (sku_id, region), group in history.groupby(
        ["SKU_ID", "Region"], dropna=False, sort=False
    ):
        valid = group["Unconstrained_Demand"].dropna()
        positive = valid.loc[valid.gt(0)]
        intervals = group.loc[group["Unconstrained_Demand"].gt(0), "Date_Period"].sort_values()
        adi = (
            float(len(valid) / len(positive))
            if len(positive)
            else np.nan
        )
        mean_positive = float(positive.mean()) if len(positive) else np.nan
        cv2 = (
            float((positive.std(ddof=1) / mean_positive) ** 2)
            if len(positive) > 1 and mean_positive
            else np.nan
        )
        if pd.isna(adi) or pd.isna(cv2):
            demand_class = "insufficient_positive_history"
        elif adi < 1.32 and cv2 < 0.49:
            demand_class = "smooth"
        elif adi < 1.32:
            demand_class = "erratic"
        elif cv2 < 0.49:
            demand_class = "intermittent"
        else:
            demand_class = "lumpy"

        sku = master.loc[master["SKU_ID"].eq(sku_id)]
        profile = {
            "SKU_ID": sku_id,
            "Region": region,
            "Product_Family": group["Product_Family"].dropna().mode().iat[0]
            if group["Product_Family"].notna().any() else "",
            "Customer_Count": int(group["Customer_Name"].dropna().nunique()),
            "Channel_Count": int(group["Sales_Channel"].dropna().nunique()),
            "Months_With_Observed_Demand": int(len(valid)),
            "Positive_Demand_Months": int(len(positive)),
            "Zero_Demand_Rate": float(valid.eq(0).mean()) if len(valid) else np.nan,
            "Mean_Positive_Demand": mean_positive,
            "ADI": adi,
            "CV2_Positive_Demand": cv2,
            "Intermittency_Class": demand_class,
            "First_History_Month": str(group["Date"].min()),
            "Last_History_Month": str(group["Date"].max()),
            "Lifecycle_Status": sku["Lifecycle_Status"].iat[0] if len(sku) else "",
            "Launch_Date": sku["Launch_Date"].iat[0] if len(sku) else "",
            "EOL_Date": sku["EOL_Date"].iat[0] if len(sku) else "",
        }
        profiles.append(profile)
    profile_frame = pd.DataFrame(profiles)
    print(f"Built {len(profile_frame):,} SKU-region profiles in {perf_counter() - started:.1f}s", flush=True)

    monthly = (
        history.groupby(["Date", "Product_Family"], dropna=False)
        .agg(
            Demand=("Unconstrained_Demand", lambda values: values.sum(min_count=1)),
            Rows=("SKU_ID", "size"),
            Missing_Demand=("Unconstrained_Demand", lambda values: int(values.isna().sum())),
            Zero_Demand_Rate=(
                "Unconstrained_Demand",
                lambda values: float(values.eq(0).sum() / values.notna().sum())
                if values.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    monthly["Month_Number"] = pd.to_numeric(
        monthly["Date"].str[-2:], errors="coerce"
    ).astype("Int64")
    seasonality = (
        monthly.groupby(["Product_Family", "Month_Number"], dropna=False)
        .agg(
            Mean_Monthly_Demand=("Demand", "mean"),
            Demand_Observations=("Demand", "count"),
        )
        .reset_index()
    )
    print(f"Calculated monthly and seasonal summaries in {perf_counter() - started:.1f}s", flush=True)

    switches: list[dict[str, Any]] = []
    for (sku_id, region), group in history.sort_values("Date").groupby(
        ["SKU_ID", "Region"], dropna=False, sort=False
    ):
        for field in ("Customer_Name", "Sales_Channel"):
            sequence = group.loc[group[field].notna() & group[field].ne(""), ["Date", field]]
            sequence = sequence.drop_duplicates("Date").sort_values("Date")
            previous = None
            for record in sequence.itertuples(index=False, name=None):
                current_date, current_value = record
                if previous is not None and current_value != previous[1]:
                    switches.append({
                        "SKU_ID": sku_id,
                        "Region": region,
                        "Changed_Field": field,
                        "Date": current_date,
                        "Previous_Value": previous[1],
                        "New_Value": current_value,
                        "Classification": "real account/channel transition; verify business context",
                    })
                previous = (current_date, current_value)
    switches_frame = pd.DataFrame(switches)
    print(f"Detected {len(switches_frame):,} customer/channel switches in {perf_counter() - started:.1f}s", flush=True)

    anomalies: list[dict[str, Any]] = []

    def add_anomaly(row: pd.Series, anomaly_type: str, classification: str,
                    reason: str, action: str) -> None:
        record = {
            "Date": row.get("Date", ""),
            "SKU_ID": row.get("SKU_ID", ""),
            "Region": row.get("Region", ""),
            "Product_Family": row.get("Product_Family", ""),
            "Customer_Name": row.get("Customer_Name", ""),
            "Sales_Channel": row.get("Sales_Channel", ""),
            "Observed_Shipments": row.get("Units_Sold_Observed", np.nan),
            "Unconstrained_Demand": row.get("Unconstrained_Demand", np.nan),
            "Anomaly_Type": anomaly_type,
            "Classification": classification,
            "Reason": reason,
            "Recommended_Action": action,
        }
        record["Related_Event_IDs"] = "|".join(_attach_events(record, event_index))
        anomalies.append(record)

    stockout_rows = history["Stockout_Record_Found"].fillna(False).astype(bool)
    for _, row in history.loc[stockout_rows].iterrows():
        add_anomaly(
            row, "listed_stockout", "real_supply_constraint",
            str(row.get("Reason", "Stockout recorded")),
            "Use Requested_Units as current-month demand; keep shipments as the observed target.",
        )

    outside_rows = data.loc[~expected_window]
    for _, row in outside_rows.iterrows():
        add_anomaly(
            row, "outside_history_window", "likely_data_error",
            "Date is outside the problem's Jan 2019-Oct 2026 historical window.",
            "Exclude from training unless the source period is verified and corrected.",
        )

    raw_units = pd.to_numeric(data.get("Units_Sold_Raw"), errors="coerce")
    negative_rows = data.loc[raw_units.lt(0)]
    for _, row in negative_rows.iterrows():
        add_anomaly(
            row, "negative_raw_units", "likely_data_error",
            f"Raw Units_Sold is {row.get('Units_Sold_Raw')}; shipped units cannot be negative.",
            "Use stockout Shipped_Units when present; otherwise exclude target pending source review.",
        )

    missing_observed = history["Units_Sold_Observed"].isna()
    for _, row in history.loc[missing_observed].iterrows():
        add_anomaly(
            row, "missing_observed_shipments", "data_gap",
            "No observed shipment value is available after stockout reconciliation.",
            "Keep missing; do not convert the row to zero demand.",
        )

    for (sku_id, region), group in history.groupby(
        ["SKU_ID", "Region"], dropna=False, sort=False
    ):
        values = group["Unconstrained_Demand"]
        positive = values.loc[values.gt(0)]
        if len(positive) < 12:
            continue
        median = float(positive.median())
        mad = float((positive - median).abs().median())
        threshold = max(median * 4, median + 6 * mad)
        outlier_rows = group.loc[values.gt(threshold)]
        for _, row in outlier_rows.iterrows():
            add_anomaly(
                row, "high_demand_spike", "needs_review",
                f"Demand exceeds the series robust threshold ({threshold:.2f}); may be a real order or event.",
                "Check event/customer history and retain unless evidence supports an error.",
            )

    anomaly_frame = pd.DataFrame(anomalies)
    print(f"Built {len(anomaly_frame):,} anomaly rows in {perf_counter() - started:.1f}s", flush=True)
    profile_frame.to_csv(output_dir / "demand_profile_by_series.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "monthly_demand_by_family.csv", index=False, encoding="utf-8-sig")
    seasonality.to_csv(output_dir / "family_seasonality.csv", index=False, encoding="utf-8-sig")
    switches_frame.to_csv(output_dir / "customer_channel_switches.csv", index=False, encoding="utf-8-sig")
    anomaly_frame.to_csv(output_dir / "anomaly_register.csv", index=False, encoding="utf-8-sig")

    summary = {
        "history_rows_in_expected_window": int(len(history)),
        "history_rows_outside_expected_window": int(len(outside_rows)),
        "sku_region_series": int(len(profile_frame)),
        "series_with_zero_positive_demand_months": int(profile_frame["Positive_Demand_Months"].eq(0).sum()),
        "demand_class_counts": profile_frame["Intermittency_Class"].value_counts().to_dict(),
        "customer_or_channel_switches": int(len(switches_frame)),
        "anomaly_register_rows": int(len(anomaly_frame)),
        "anomaly_counts": anomaly_frame["Anomaly_Type"].value_counts().to_dict()
        if len(anomaly_frame) else {},
        "outputs": str(output_dir),
    }
    with (output_dir / "demand_exploration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    print(f"Saved exploration outputs in {perf_counter() - started:.1f}s", flush=True)
    return summary
