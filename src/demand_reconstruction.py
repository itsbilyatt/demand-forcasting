from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


STOCKOUT_NUMERIC_COLUMNS = (
    "Requested_Units",
    "Carried_In_Units",
    "Shipped_Units",
    "Backordered_Units",
    "Lost_Units",
    "Fill_Rate_Pct",
    "Stockout_Days",
)


def _find_project_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "cleaned").is_dir():
            return candidate
    raise FileNotFoundError("Cannot locate outputs/cleaned from the current directory.")


def run_demand_reconstruction(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_project_root(project_root)
    cleaned_dir = root / "outputs" / "cleaned"
    output_dir = root / "outputs" / "demand_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(cleaned_dir / "train_cleaned.csv", low_memory=False)
    stockouts = pd.read_csv(cleaned_dir / "inventory_stockouts_cleaned.csv", low_memory=False)
    master = pd.read_csv(cleaned_dir / "sku_master_cleaned.csv", low_memory=False)
    keys = ["Date", "SKU_ID", "Region"]

    for column in STOCKOUT_NUMERIC_COLUMNS:
        stockouts[column] = pd.to_numeric(stockouts[column], errors="coerce")
    train["Units_Sold"] = pd.to_numeric(train["Units_Sold"], errors="coerce")
    for column in ("Unit_Price_USD", "Revenue_USD"):
        train[column] = pd.to_numeric(train[column], errors="coerce")

    valid_stockout_keys = stockouts[keys].notna().all(axis=1)
    if not valid_stockout_keys.all():
        raise ValueError("Stockout file contains incomplete Date/SKU_ID/Region keys.")
    if stockouts.duplicated(keys).any():
        raise ValueError("Stockout file contains duplicate Date/SKU_ID/Region keys.")

    train_key_valid = (
        train["Date"].fillna("").astype(str).str.strip().ne("")
        & train["SKU_ID"].fillna("").astype(str).str.strip().ne("")
        & train["Region"].fillna("").astype(str).str.strip().ne("")
    )
    train_with_valid_keys = train.loc[train_key_valid].copy()
    if train_with_valid_keys.duplicated(keys).any():
        raise ValueError("Cleaned train has duplicate complete keys; resolve before joining stockouts.")

    sales_for_join = train_with_valid_keys[
        keys + ["Units_Sold", "Units_Sold_Raw", "Record_ID"]
    ]
    join_audit = stockouts.merge(
        sales_for_join,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    join_audit["Train_Row_Found"] = join_audit["_merge"].eq("both")
    join_audit["Train_Target_Missing"] = (
        join_audit["Train_Row_Found"] & join_audit["Units_Sold"].isna()
    )
    join_audit["Train_Target_Disagrees_With_Shipped"] = (
        join_audit["Train_Row_Found"]
        & join_audit["Units_Sold"].notna()
        & join_audit["Units_Sold"].ne(join_audit["Shipped_Units"])
    )
    join_audit["Train_Target_Needs_Shipment_Backfill"] = (
        join_audit["Train_Target_Missing"]
        | join_audit["Train_Target_Disagrees_With_Shipped"]
    )

    observed_column = "Units_Sold_Observed"
    train[observed_column] = train["Units_Sold"]
    stockout_values = stockouts[
        keys + list(STOCKOUT_NUMERIC_COLUMNS) + ["Reason"]
    ].copy()
    train = train.merge(
        stockout_values,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator="_stockout_merge",
    )
    train["Stockout_Record_Found"] = train["_stockout_merge"].eq("both")
    train["Stockout_Row_Added"] = False
    has_shipped = train["Stockout_Record_Found"] & train["Shipped_Units"].notna()
    train.loc[has_shipped, observed_column] = train.loc[has_shipped, "Shipped_Units"]
    train["Observed_Shipments_Source"] = "cleaned_train_Units_Sold"
    train.loc[has_shipped, "Observed_Shipments_Source"] = "inventory_stockouts_Shipped_Units"
    unsupported_negative_shipments = (
        ~train["Stockout_Record_Found"]
        & train[observed_column].notna()
        & train[observed_column].lt(0)
    )
    train.loc[unsupported_negative_shipments, observed_column] = pd.NA

    has_requested = train["Stockout_Record_Found"] & train["Requested_Units"].notna()
    train["Unconstrained_Demand"] = train[observed_column]
    train.loc[has_requested, "Unconstrained_Demand"] = train.loc[has_requested, "Requested_Units"]
    train["Demand_Estimate_Source"] = "observed_shipments_proxy"
    train.loc[has_requested, "Demand_Estimate_Source"] = "stockout_Requested_Units"
    train["Demand_Confidence"] = "low_stockout_file_is_incomplete"
    train.loc[has_requested, "Demand_Confidence"] = "high_request_observed_in_stockout_file"
    train["Demand_Adjustment_Units"] = train["Unconstrained_Demand"] - train[observed_column]
    train["Demand_Adjustment_Units"] = train["Demand_Adjustment_Units"].where(
        train["Unconstrained_Demand"].notna() & train[observed_column].notna()
    )
    train.drop(columns=["_stockout_merge"], inplace=True)

    unmatched = join_audit.loc[~join_audit["Train_Row_Found"]].copy()
    inserted_rows: list[dict[str, Any]] = []
    master_by_sku = master.set_index("SKU_ID", drop=False)
    for _, stockout_row in unmatched.iterrows():
        sku_id = stockout_row["SKU_ID"]
        sku = master_by_sku.loc[sku_id] if sku_id in master_by_sku.index else None
        row = {column: pd.NA for column in train.columns}
        row.update({
            "Date": stockout_row["Date"],
            "Date_Raw": stockout_row.get("Date_Raw", stockout_row["Date"]),
            "Year": int(str(stockout_row["Date"])[:4]),
            "Month": int(str(stockout_row["Date"])[5:7]),
            "SKU_ID": sku_id,
            "SKU_ID_Raw": stockout_row.get("SKU_ID_Raw", sku_id),
            "Region": stockout_row["Region"],
            "Region_Raw": stockout_row.get("Region_Raw", stockout_row["Region"]),
            "Record_ID": "",
            "Units_Sold": pd.NA,
            "Units_Sold_Raw": "",
            "Units_Sold_Observed": stockout_row["Shipped_Units"],
            "Observed_Shipments_Source": "inventory_stockouts_Shipped_Units",
            "Stockout_Record_Found": True,
            "Stockout_Row_Added": True,
            "Requested_Units": stockout_row["Requested_Units"],
            "Carried_In_Units": stockout_row["Carried_In_Units"],
            "Shipped_Units": stockout_row["Shipped_Units"],
            "Backordered_Units": stockout_row["Backordered_Units"],
            "Lost_Units": stockout_row["Lost_Units"],
            "Fill_Rate_Pct": stockout_row["Fill_Rate_Pct"],
            "Stockout_Days": stockout_row["Stockout_Days"],
            "Reason": stockout_row["Reason"],
            "Unconstrained_Demand": stockout_row["Requested_Units"],
            "Demand_Estimate_Source": "stockout_Requested_Units",
            "Demand_Confidence": "high_request_observed_in_stockout_file",
            "Demand_Adjustment_Units": (
                stockout_row["Requested_Units"] - stockout_row["Shipped_Units"]
            ),
            "Composite_Key_Valid": True,
        })
        if sku is not None:
            for column in ("Product_Family", "Business_Unit", "SKU_Description"):
                if column in row:
                    row[column] = sku.get(column, pd.NA)
        inserted_rows.append(row)

    if inserted_rows:
        train = pd.concat([train, pd.DataFrame(inserted_rows)], ignore_index=True, sort=False)

    complete_keys = train.loc[train[keys].notna().all(axis=1)]
    duplicate_keys = complete_keys.duplicated(keys, keep=False)
    if duplicate_keys.any():
        raise ValueError(
            f"Demand panel still has {int(duplicate_keys.sum())} duplicate complete keys."
        )

    # A missing stockout record is not proof of unconstrained supply.
    train["Stockout_Coverage_Caveat"] = (
        "No listed stockout; observed shipments used as a proxy because the stockout file is incomplete."
    )
    train.loc[train["Stockout_Record_Found"], "Stockout_Coverage_Caveat"] = (
        "Listed stockout; Requested_Units is the current-month demand, separate from carried-in units."
    )

    formula_lost = (
        stockouts["Requested_Units"] + stockouts["Carried_In_Units"]
        - stockouts["Shipped_Units"] - stockouts["Backordered_Units"]
    )
    lost_formula_mismatch = (
        formula_lost.notna() & stockouts["Lost_Units"].notna()
        & formula_lost.sub(stockouts["Lost_Units"]).abs().gt(0.01)
    )
    shipped_mismatch = int(join_audit["Train_Target_Disagrees_With_Shipped"].sum())
    target_backfills = int(join_audit["Train_Target_Missing"].sum())
    report_rows = [
        {
            "issue": "stockout keys absent from train",
            "rows": int(len(unmatched)),
            "action": "Added rows with observed shipments from the stockout file; left unknown customer/channel blank.",
        },
        {
            "issue": "train Units_Sold disagrees with stockout Shipped_Units",
            "rows": shipped_mismatch,
            "action": "Used Shipped_Units as the observed shipment for these stockout keys; original Units_Sold retained.",
        },
        {
            "issue": "train Units_Sold missing for stockout key",
            "rows": target_backfills,
            "action": "Backfilled observed shipments from Shipped_Units; this is evidence-backed, not zero imputation.",
        },
        {
            "issue": "stockout lost-units arithmetic mismatch",
            "rows": int(lost_formula_mismatch.sum()),
            "action": "Retained source fields and flagged; expected requested + carried-in - shipped - backordered.",
        },
        {
            "issue": "requested units below shipped units",
            "rows": int(stockouts["Requested_Units"].lt(stockouts["Shipped_Units"]).sum()),
            "action": "Not treated as an error: shipments may include carried-in units from the prior month.",
        },
        {
            "issue": "negative train target without stockout shipment evidence",
            "rows": int(unsupported_negative_shipments.sum()),
            "action": "Excluded from observed demand proxy pending anomaly review; did not convert to zero.",
        },
    ]
    audit_columns = [
        *keys, "Requested_Units", "Carried_In_Units", "Shipped_Units",
        "Backordered_Units", "Lost_Units", "Reason", "Train_Row_Found",
        "Train_Target_Missing", "Train_Target_Disagrees_With_Shipped",
    ]
    join_audit[audit_columns].to_csv(
        output_dir / "stockout_join_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(report_rows).to_csv(
        output_dir / "stockout_quality_report.csv", index=False, encoding="utf-8-sig"
    )
    train.to_csv(output_dir / "train_with_reconstructed_demand.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_train_rows": int(len(train) - len(inserted_rows)),
        "stockout_records": int(len(stockouts)),
        "stockout_records_matched_to_train": int(join_audit["Train_Row_Found"].sum()),
        "stockout_records_without_train_row": int(len(unmatched)),
        "inserted_stockout_rows": int(len(inserted_rows)),
        "train_targets_missing_for_stockout_keys": target_backfills,
        "train_targets_disagreeing_with_shipped_units": shipped_mismatch,
        "negative_train_targets_without_stockout_evidence": int(
            unsupported_negative_shipments.sum()
        ),
        "valid_unique_period_sku_region_keys_after_reconstruction": int(len(complete_keys)),
        "lost_units_formula_mismatches": int(lost_formula_mismatch.sum()),
        "requested_less_than_shipped_rows_due_possible_carry_in": int(
            stockouts["Requested_Units"].lt(stockouts["Shipped_Units"]).sum()
        ),
        "stockout_demand_rows": int(train["Demand_Estimate_Source"].eq("stockout_Requested_Units").sum()),
        "shipment_proxy_rows": int(train["Demand_Estimate_Source"].eq("observed_shipments_proxy").sum()),
        "observed_shipments_total": float(train["Units_Sold_Observed"].sum(skipna=True)),
        "unconstrained_demand_total": float(train["Unconstrained_Demand"].sum(skipna=True)),
        "outputs": {
            "demand_panel": str(output_dir / "train_with_reconstructed_demand.csv"),
            "stockout_join_audit": str(output_dir / "stockout_join_audit.csv"),
            "stockout_quality_report": str(output_dir / "stockout_quality_report.csv"),
        },
        "assumptions": [
            "Requested_Units is current-month customer demand; Carried_In_Units is prior-month demand and is not added to current demand.",
            "Shipped_Units is the authoritative observed shipment for listed stockout keys and should match train Units_Sold.",
            "For unlisted keys, observed shipments are only a demand proxy; the stockout file is incomplete, so no-stockout status is not inferred.",
            "Rows found in stockout data but absent from train are added with known key, request, and shipment values; missing customer/channel are not fabricated.",
            "Raw/cleaned Units_Sold remains alongside Units_Sold_Observed and Unconstrained_Demand.",
        ],
    }
    with (output_dir / "demand_reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary
