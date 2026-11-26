from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_ROWS = {
    "train.csv": 248_248,
    "test.csv": 58_219,
    "macro_indicators.csv": 114,
    "sku_master.csv": 601,
    "events_calendar.csv": 68,
    "inventory_stockouts.csv": 5_830,
    "sample_submission.csv": 5,
}

EXPECTED_PERIODS = {
    "train.csv": ("2019-01", "2026-10"),
    "test.csv": ("2026-11", "2028-05"),
    "macro_indicators.csv": ("2019-01", "2028-05"),
}

CHANNEL_ALIASES = {
    "directoem": "Direct OEM",
    "oemdirect": "Direct OEM",
    "tier1distributor": "Tier-1 Distributor",
    "t1distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "export": "Export",
}

REGION_ALIASES = {
    "northamerica": "North America",
    "namerica": "North America",
    "na": "North America",
    "europe": "Europe",
    "eu": "Europe",
    "asiapacific": "Asia Pacific",
    "apac": "Asia Pacific",
    "india": "India",
    "latinamerica": "Latin America",
    "latam": "Latin America",
    "middleeastafrica": "Middle East & Africa",
    "middleeastandafrica": "Middle East & Africa",
    "mea": "Middle East & Africa",
}

DATE_FORMATS = (
    "%Y-%m",
    "%Y/%m",
    "%b %Y",
    "%B %Y",
    "%b-%Y",
    "%B-%Y",
    "%m-%Y",
    "%m/%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
)


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _parse_date(value: Any) -> datetime | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    try:
        return datetime.combine(date.fromisoformat(text), datetime.min.time())
    except ValueError:
        return None


def _parsed_date_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str)
    parsed = {value: _parse_date(value) for value in values.unique()}
    return pd.to_datetime(values.map(parsed), errors="coerce")


def _canonicalize(series: pd.Series, canonical_values=(), aliases=None) -> tuple[pd.Series, pd.Series]:
    lookup = {_token(value): value for value in canonical_values if str(value).strip()}
    lookup.update(aliases or {})

    def convert(value: Any) -> str:
        text = "" if value is None else str(value).strip()
        return lookup.get(_token(text), text) if text else ""

    cleaned = series.map(convert)
    changed = cleaned.ne(series.fillna("").astype(str))
    return cleaned, changed


def _read_csv(data_dir: Path, file_name: str) -> pd.DataFrame:
    path = data_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def run_data_cleaning(project_root: str | Path | None = None) -> dict[str, Any]:
    if project_root is None:
        candidates = (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        root = next(
            (candidate for candidate in candidates if (candidate / "data" / "student_data").is_dir()),
            None,
        )
        if root is None:
            raise FileNotFoundError("Cannot locate data/student_data from the current directory.")
    else:
        root = Path(project_root).resolve()

    data_dir = root / "data" / "student_data"
    cleaned_dir = root / "outputs" / "cleaned"
    quality_dir = root / "outputs" / "data_quality"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    price_correction_rows: list[dict[str, Any]] = []
    file_metrics: dict[str, dict[str, Any]] = {}

    def record(target: list[dict[str, Any]], file_name: str, kind: str, count: int,
               action: str, severity: str, examples=None) -> None:
        if count <= 0:
            return
        target.append({
            "file": file_name,
            "type": kind,
            "affected_rows": int(count),
            "severity": severity,
            "action": action,
            "examples": json.dumps(examples or [], ensure_ascii=True, default=str),
        })

    def note_missing(file_name: str, frame: pd.DataFrame) -> None:
        for column in frame.columns:
            if column.startswith("_"):
                continue
            missing = frame[column].fillna("").astype(str).str.strip().eq("")
            record(
                issues, file_name, f"missing:{column}", int(missing.sum()),
                "Preserved as missing; do not infer zero or fabricate a value.",
                "review",
            )

    def standardize(frame: pd.DataFrame, column: str, file_name: str,
                    canonical=(), aliases=None) -> None:
        if column not in frame:
            return
        raw = frame[column].fillna("").astype(str)
        cleaned, changed = _canonicalize(raw, canonical, aliases)
        record(
            corrections, file_name, f"standardized:{column}", int(changed.sum()),
            "Applied a case/punctuation-insensitive match to an established canonical label.",
            "corrected",
        )
        frame[column] = cleaned

    def clean_period(frame: pd.DataFrame, file_name: str, column: str = "Date") -> pd.Series:
        if column not in frame:
            return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        raw_name = f"{column}_Raw"
        if raw_name not in frame:
            frame[raw_name] = frame[column]
        parsed = _parsed_date_series(frame[raw_name])
        invalid = parsed.isna() & frame[raw_name].fillna("").astype(str).str.strip().ne("")
        record(
            issues, file_name, f"unparseable_date:{column}", int(invalid.sum()),
            "Preserved the source value in the *_Raw column; investigate before modeling.",
            "error",
        )
        nonblank = frame[raw_name].fillna("").astype(str).str.strip().ne("")
        canonical = parsed.dt.strftime("%Y-%m")
        noncanonical = nonblank & parsed.notna() & frame[raw_name].astype(str).ne(canonical)
        record(
            corrections, file_name, f"standardized_date:{column}", int(noncanonical.sum()),
            "Converted parseable dates to YYYY-MM for monthly data; retained the raw value.",
            "corrected",
        )
        frame[column] = canonical.fillna("")
        return parsed

    def numeric_column(frame: pd.DataFrame, column: str, file_name: str,
                       allow_dash: bool = True) -> tuple[pd.Series, pd.Series]:
        raw = frame[column].fillna("").astype(str).str.strip()
        normalized = raw.str.replace(",", "", regex=False)
        missing = normalized.eq("") | (normalized.eq("-") if allow_dash else False)
        values = pd.to_numeric(normalized.mask(missing), errors="coerce")
        invalid = ~missing & values.isna()
        record(
            issues, file_name, f"invalid_numeric:{column}", int(invalid.sum()),
            "Preserved the raw value and left the cleaned numeric value missing.",
            "error",
            frame.loc[invalid, [column]].head(5).to_dict("records"),
        )
        return values, invalid

    master = _read_csv(data_dir, "sku_master.csv")
    note_missing("sku_master.csv", master)
    master["_source_row_number"] = np.arange(2, len(master) + 2)
    master_maps = {
        "Product_Family": sorted(master["Product_Family"].dropna().unique()),
        "Business_Unit": sorted(master["Business_Unit"].dropna().unique()),
        "Lifecycle_Status": sorted(master["Lifecycle_Status"].dropna().unique()),
        "Powertrain_Application": sorted(master["Powertrain_Application"].dropna().unique()),
    }
    for column in ("Product_Family", "Business_Unit", "Lifecycle_Status", "Powertrain_Application"):
        standardize(master, column, "sku_master.csv", master_maps[column])
    for column in ("Launch_Date", "EOL_Date"):
        if column in master:
            parsed = clean_period(master, "sku_master.csv", column)
            master[column] = parsed.dt.strftime("%Y-%m").fillna("")
    master_dups = master["SKU_ID"].duplicated(keep=False)
    record(
        issues, "sku_master.csv", "duplicate:SKU_ID", int(master_dups.sum()),
        "Retained duplicate reference rows for review; no arbitrary winner selected.",
        "error",
        master.loc[master_dups, ["SKU_ID"]].head(5).to_dict("records"),
    )
    if master["SKU_ID"].duplicated().any():
        raise ValueError("sku_master.csv contains duplicate SKU_ID values; resolve before joining.")
    master_index = master.set_index("SKU_ID", drop=False)
    known_sku_ids = set(master_index.index)

    family_values = master_maps["Product_Family"]
    business_values = master_maps["Business_Unit"]
    canonical_channels = ["Direct OEM", "Tier-1 Distributor", "Aftermarket", "Export"]
    canonical_regions = [
        "North America", "Europe", "Asia Pacific", "India", "Latin America",
        "Middle East & Africa",
    ]

    def clean_sales_metadata(frame: pd.DataFrame, file_name: str) -> None:
        normalize_sku_ids(frame, file_name)
        standardize(frame, "Product_Family", file_name, family_values)
        standardize(frame, "Business_Unit", file_name, business_values)
        standardize(frame, "Sales_Channel", file_name, canonical_channels, CHANNEL_ALIASES)
        standardize(frame, "Region", file_name, canonical_regions, REGION_ALIASES)
        if "SKU_Description" in frame:
            raw = frame["SKU_Description"].fillna("").astype(str)
            collapsed = raw.map(lambda value: " ".join(value.split()))
            record(
                corrections, file_name, "standardized:SKU_Description",
                int(collapsed.ne(raw).sum()),
                "Collapsed repeated/leading/trailing whitespace only.",
                "corrected",
            )
            frame["SKU_Description"] = collapsed

        for column in ("Product_Family", "Business_Unit", "SKU_Description"):
            if column not in frame:
                continue
            expected = frame["SKU_ID"].map(master_index[column]).fillna("")
            current = frame[column].fillna("").astype(str)
            current_token = current.map(_token)
            expected_token = expected.map(_token)
            known_sku = expected.ne("")
            conflict = current.ne("") & known_sku & current_token.ne(expected_token)
            record(
                issues, file_name, f"master_conflict:{column}", int(conflict.sum()),
                "Compared with sku_master.csv; conflicts are preserved or corrected only under the documented rule.",
                "review",
                frame.loc[conflict, ["SKU_ID", column]].head(5).to_dict("records"),
            )
            missing = current.eq("") & known_sku
            if missing.any():
                frame.loc[missing, column] = expected[missing]
                record(
                    corrections, file_name, f"filled_from_sku_master:{column}", int(missing.sum()),
                    "Filled missing product attributes from the matching SKU_ID in sku_master.csv.",
                    "corrected",
                )
            if column == "Business_Unit" and conflict.any():
                frame.loc[conflict, column] = expected[conflict]
                record(
                    corrections, file_name, "corrected_from_sku_master:Business_Unit",
                    int(conflict.sum()),
                    "Replaced conflicting Business_Unit with the SKU master value; original retained in *_Raw.",
                    "corrected",
                )

        unknown_sku = ~frame["SKU_ID"].isin(master_index.index)
        record(
            issues, file_name, "unknown_sku_id", int(unknown_sku.sum()),
            "Kept rows; verify SKU_ID against the product master.",
            "error",
            frame.loc[unknown_sku, ["SKU_ID"]].head(5).to_dict("records"),
        )

    def normalize_sku_ids(frame: pd.DataFrame, file_name: str) -> None:
        if "SKU_ID" not in frame:
            return
        if "SKU_ID_Raw" not in frame:
            frame["SKU_ID_Raw"] = frame["SKU_ID"]
        raw = frame["SKU_ID"].fillna("").astype(str).str.strip()

        def candidate(value: str) -> str:
            match = re.fullmatch(r"([A-Za-z]{3})[-_ ]?(\d{1,3})", value)
            if not match:
                return value
            proposed = f"{match.group(1).upper()}-{int(match.group(2)):03d}"
            return proposed if proposed in known_sku_ids else value

        normalized = raw.map(candidate)
        changed = normalized.ne(raw)
        record(
            corrections, file_name, "standardized:SKU_ID", int(changed.sum()),
            "Normalized separators/casing/zero-padding only when the result exists in sku_master.csv.",
            "corrected",
        )
        frame["SKU_ID"] = normalized

    def add_raw_columns(frame: pd.DataFrame, columns: list[str]) -> None:
        for column in columns:
            if column in frame and f"{column}_Raw" not in frame:
                frame[f"{column}_Raw"] = frame[column]

    def clean_train() -> pd.DataFrame:
        frame = _read_csv(data_dir, "train.csv")
        input_count = len(frame)
        note_missing("train.csv", frame)
        frame["_source_row_number"] = np.arange(2, len(frame) + 2)
        add_raw_columns(frame, [
            "Date", "Year", "Month", "Product_Family", "Business_Unit",
            "SKU_Description", "Sales_Channel", "Region", "Units_Sold",
            "Unit_Price_USD", "Revenue_USD",
        ])

        id_text = frame["Record_ID"].fillna("").astype(str).str.strip()
        id_counts = id_text[id_text.ne("")].value_counts()
        frame["Record_ID_Duplicate_Count"] = id_text.map(id_counts).fillna(1).astype("int64")
        duplicate_rows = frame["Record_ID_Duplicate_Count"].gt(1)
        duplicate_groups = int(id_counts.gt(1).sum())
        duplicate_extras = int((id_counts - 1).clip(lower=0).sum())
        conflict_fields = [
            "Date", "SKU_ID", "Region", "Units_Sold", "Unit_Price_USD", "Revenue_USD",
        ]
        duplicate_conflict_groups = 0
        if duplicate_groups:
            dup_frame = frame.loc[duplicate_rows & id_text.ne("")]
            for _, group in dup_frame.groupby("Record_ID", sort=False):
                if any(group[field].nunique(dropna=False) > 1 for field in conflict_fields):
                    duplicate_conflict_groups += 1
        record(
            issues, "train.csv", "duplicate:Record_ID_rows", int(duplicate_rows.sum()),
            "Kept only the row with the latest Load_Timestamp for each nonblank Record_ID.",
            "review",
        )
        record(
            issues, "train.csv", "duplicate:Record_ID_conflicting_groups", duplicate_conflict_groups,
            "Latest timestamp is selected; inspect the retained raw values for material corrections.",
            "review",
        )

        frame["_load_timestamp_sort"] = pd.to_datetime(
            frame["Load_Timestamp"], format="mixed", errors="coerce"
        )
        nonblank = frame.loc[id_text.ne("")].sort_values(
            ["_load_timestamp_sort", "_source_row_number"], kind="stable", na_position="first"
        )
        selected = nonblank.drop_duplicates("Record_ID", keep="last")
        blank_ids = frame.loc[id_text.eq("")]
        frame = pd.concat([selected, blank_ids], ignore_index=True)
        frame = frame.sort_values("_source_row_number", kind="stable").reset_index(drop=True)
        removed = input_count - len(frame)
        record(
            corrections, "train.csv", "deduplicated:Record_ID", removed,
            "Retained the latest Load_Timestamp; source row number breaks timestamp ties.",
            "corrected",
        )

        parsed_date = clean_period(frame, "train.csv", "Date")
        frame["Date_Parsed"] = parsed_date
        raw_year = pd.to_numeric(frame["Year_Raw"], errors="coerce")
        raw_month = pd.to_numeric(frame["Month_Raw"], errors="coerce")
        mismatch = parsed_date.notna() & (
            raw_year.ne(parsed_date.dt.year) | raw_month.ne(parsed_date.dt.month)
        )
        record(
            issues, "train.csv", "date_year_month_mismatch", int(mismatch.sum()),
            "Date is authoritative; Year and Month are recomputed from parsed Date.",
            "corrected",
            frame.loc[mismatch, ["Date_Raw", "Year_Raw", "Month_Raw", "Record_ID"]]
            .head(5).to_dict("records"),
        )
        record(
            corrections, "train.csv", "recomputed_year_month_from_Date", int(mismatch.sum()),
            "Recomputed Year and Month from the authoritative Date field.",
            "corrected",
        )
        low, high = EXPECTED_PERIODS["train.csv"]
        outside = parsed_date.notna() & (
            parsed_date.dt.strftime("%Y-%m").lt(low) | parsed_date.dt.strftime("%Y-%m").gt(high)
        )
        record(
            issues, "train.csv", "date_outside_expected_window", int(outside.sum()),
            "Preserved Date; exclude from historical modeling only after review against source records.",
            "error",
            frame.loc[outside, ["Date_Raw", "Year_Raw", "Month_Raw", "Record_ID"]]
            .head(5).to_dict("records"),
        )
        frame["Year"] = parsed_date.dt.year.astype("Int64")
        frame["Month"] = parsed_date.dt.month.astype("Int64")
        frame["Date_Out_Of_Expected_Window"] = outside
        frame["Date_Year_Month_Mismatch"] = mismatch

        clean_sales_metadata(frame, "train.csv")

        quantity, invalid_quantity = numeric_column(frame, "Units_Sold_Raw", "train.csv")
        price, invalid_price = numeric_column(frame, "Unit_Price_USD_Raw", "train.csv")
        revenue_raw, invalid_revenue = numeric_column(frame, "Revenue_USD_Raw", "train.csv")
        quantity_missing = quantity.isna()
        record(
            issues, "train.csv", "missing_or_dash:Units_Sold", int(quantity_missing.sum()),
            "Kept as missing; never replaced with zero because zero means no order.",
            "error",
        )
        fractional_quantity = quantity.notna() & (quantity.sub(quantity.round()).abs().gt(1e-9))
        negative_quantity = quantity.lt(0)
        record(
            issues, "train.csv", "fractional_Units_Sold", int(fractional_quantity.sum()),
            "Preserved the parsed value and flagged for unit review.",
            "review",
        )
        record(
            issues, "train.csv", "negative_Units_Sold", int(negative_quantity.sum()),
            "Preserved the parsed value and flagged for review.",
            "error",
        )
        negative_price = price.lt(0)
        record(
            issues, "train.csv", "negative_Unit_Price_USD", int(negative_price.sum()),
            "Preserved the parsed value and flagged for review.",
            "error",
        )

        frame["Date"] = frame["Date"].fillna("")
        period_key = frame["Date"].replace("", "__MISSING_DATE__")
        price_reference = price.where(price.gt(0))
        group_keys = [frame["SKU_ID"], period_key]
        peer_median = price_reference.groupby(group_keys, dropna=False).transform("median")
        peer_count = price_reference.groupby(group_keys, dropna=False).transform("count")
        peer_min = price_reference.groupby(group_keys, dropna=False).transform("min")
        peer_max = price_reference.groupby(group_keys, dropna=False).transform("max")

        zero_price = price.eq(0)
        price_fill = zero_price & peer_count.ge(1) & peer_median.gt(0)
        price_clean = price.copy()
        price_clean.loc[price_fill] = peer_median.loc[price_fill]
        record(
            corrections, "train.csv", "imputed_zero_price_from_same_sku_month",
            int(price_fill.sum()),
            "Replaced zero price only when a positive same-SKU/month peer exists; used the peer median.",
            "corrected",
        )
        for index in frame.index[price_fill]:
            price_correction_rows.append({
                "Record_ID": frame.at[index, "Record_ID"],
                "Date": frame.at[index, "Date"],
                "SKU_ID": frame.at[index, "SKU_ID"],
                "Region": frame.at[index, "Region"],
                "Original_Unit_Price_USD": float(price.at[index]),
                "Corrected_Unit_Price_USD": float(peer_median.at[index]),
                "Positive_Same_SKU_Month_Peers": int(peer_count.at[index]),
                "Correction_Type": "zero_from_same_sku_month_peer_median",
            })

        ratio = price.div(peer_median.where(peer_median.gt(0)))
        multipliers = np.array([0.001, 0.01, 0.1, 10.0, 100.0, 1000.0])
        ratio_values = ratio.to_numpy(dtype=float, na_value=np.nan)
        multiplier_match = np.zeros(len(frame), dtype=bool)
        for factor in multipliers:
            multiplier_match |= np.isfinite(ratio_values) & (
                np.abs(np.log10(np.maximum(ratio_values, 1e-300) / factor)) <= 0.02
            )
        factor_error = (
            price.gt(0) & peer_count.ge(2) & peer_median.gt(0)
            & (ratio.lt(0.5) | ratio.gt(2.0)) & multiplier_match
        )
        price_clean.loc[factor_error] = peer_median.loc[factor_error]
        record(
            corrections, "train.csv", "corrected_price_factor_error",
            int(factor_error.sum()),
            "Corrected only clear 10x/100x/1000x or reciprocal outliers using the same-SKU/month peer median.",
            "corrected",
        )
        for index in frame.index[factor_error]:
            price_correction_rows.append({
                "Record_ID": frame.at[index, "Record_ID"],
                "Date": frame.at[index, "Date"],
                "SKU_ID": frame.at[index, "SKU_ID"],
                "Region": frame.at[index, "Region"],
                "Original_Unit_Price_USD": float(price.at[index]),
                "Corrected_Unit_Price_USD": float(peer_median.at[index]),
                "Positive_Same_SKU_Month_Peers": int(peer_count.at[index]),
                "Correction_Type": "clear_factor_error_from_same_sku_month_peer_median",
            })

        unresolved_zero_price = zero_price & ~price_fill
        price_clean.loc[unresolved_zero_price] = np.nan
        record(
            issues, "train.csv", "unresolved_zero_price", int(unresolved_zero_price.sum()),
            "Changed invalid zero price to missing; requires source or event review.",
            "error",
        )
        price_ratio_clean = price_clean.div(peer_median.where(peer_median.gt(0)))
        price_outlier = (
            price_clean.gt(0) & peer_count.ge(2) & peer_median.gt(0)
            & (price_ratio_clean.lt(0.5) | price_ratio_clean.gt(2.0))
            & ~factor_error
        )
        record(
            issues, "train.csv", "price_deviation_from_same_sku_month", int(price_outlier.sum()),
            "Not auto-corrected; review against event-calendar surcharges and source pricing.",
            "review",
            frame.loc[price_outlier, [
                "Record_ID", "Date", "SKU_ID", "Region", "Unit_Price_USD_Raw",
            ]].head(5).to_dict("records"),
        )

        revenue_expected_before = quantity * price
        revenue_difference = (revenue_raw - revenue_expected_before).abs()
        revenue_tolerance = np.maximum(0.01, revenue_expected_before.abs() * 0.0001)
        revenue_mismatch = (
            quantity.notna() & price.notna() & revenue_raw.notna()
            & revenue_difference.gt(revenue_tolerance)
        )
        record(
            issues, "train.csv", "revenue_not_equal_units_times_price",
            int(revenue_mismatch.sum()),
            "Recomputed Revenue_USD where Units_Sold and cleaned Unit_Price_USD are available; raw value retained.",
            "corrected",
            frame.loc[revenue_mismatch, [
                "Record_ID", "Units_Sold_Raw", "Unit_Price_USD_Raw", "Revenue_USD_Raw",
            ]].head(5).to_dict("records"),
        )
        missing_revenue_filled = revenue_raw.isna() & quantity.notna() & price_clean.notna()
        record(
            corrections, "train.csv", "filled_revenue_from_units_times_price",
            int(missing_revenue_filled.sum()),
            "Calculated revenue only when both units and cleaned unit price are present.",
            "corrected",
        )
        record(
            corrections, "train.csv", "reconciled_revenue",
            int(revenue_mismatch.sum()),
            "Recomputed mismatched revenue from cleaned Units_Sold and Unit_Price_USD; raw revenue retained.",
            "corrected",
        )

        revenue_clean = revenue_raw.copy()
        can_reconcile = quantity.notna() & price_clean.notna()
        revenue_clean.loc[can_reconcile] = (
            quantity.loc[can_reconcile] * price_clean.loc[can_reconcile]
        ).round(2)
        frame["Units_Sold"] = quantity.astype("Float64")
        if not fractional_quantity.any():
            frame["Units_Sold"] = quantity.astype("Int64")
        frame["Unit_Price_USD"] = price_clean.astype("Float64")
        frame["Revenue_USD"] = revenue_clean.astype("Float64")
        frame["Units_Sold_Invalid"] = invalid_quantity
        frame["Unit_Price_Invalid"] = invalid_price
        frame["Revenue_USD_Invalid"] = invalid_revenue
        frame["Unit_Price_Review_Flag"] = price_outlier
        frame["Composite_Key_Valid"] = (
            frame["Date"].ne("") & frame["SKU_ID"].fillna("").astype(str).str.strip().ne("")
            & frame["Region"].fillna("").astype(str).str.strip().ne("")
        )
        composite_dup = frame["Composite_Key_Valid"] & frame.duplicated(
            ["Date", "SKU_ID", "Region"], keep=False
        )
        record(
            issues, "train.csv", "duplicate_valid_Date_SKU_Region",
            int(composite_dup.sum()),
            "Retained; investigate rather than summing rows with potentially different customers/channels.",
            "review",
            frame.loc[composite_dup, [
                "Date", "SKU_ID", "Region", "Customer_Name", "Sales_Channel", "Record_ID",
            ]].head(5).to_dict("records"),
        )

        file_metrics["train.csv"] = {
            "input_rows": input_count,
            "output_rows": len(frame),
            "record_id_duplicate_groups": duplicate_groups,
            "record_id_duplicate_rows_removed": removed,
            "record_id_conflicting_groups": duplicate_conflict_groups,
            "date_year_month_mismatches": int(mismatch.sum()),
            "outside_expected_period_rows": int(outside.sum()),
            "rows_with_missing_units": int(quantity_missing.sum()),
            "valid_composite_key_rows": int(frame["Composite_Key_Valid"].sum()),
            "valid_composite_key_duplicates": int(composite_dup.sum()),
        }
        return frame.drop(columns=["Date_Parsed"], errors="ignore")

    def clean_test() -> pd.DataFrame:
        frame = _read_csv(data_dir, "test.csv")
        note_missing("test.csv", frame)
        add_raw_columns(frame, [
            "Date", "Year", "Month", "Product_Family", "Business_Unit",
            "SKU_Description", "Sales_Channel", "Region",
        ])
        parsed = clean_period(frame, "test.csv")
        raw_year = pd.to_numeric(frame["Year_Raw"], errors="coerce")
        raw_month = pd.to_numeric(frame["Month_Raw"], errors="coerce")
        mismatch = parsed.notna() & (
            raw_year.ne(parsed.dt.year) | raw_month.ne(parsed.dt.month)
        )
        record(
            issues, "test.csv", "date_year_month_mismatch", int(mismatch.sum()),
            "Date is authoritative; Year and Month are recomputed from parsed Date.",
            "corrected",
        )
        record(
            corrections, "test.csv", "recomputed_year_month_from_Date", int(mismatch.sum()),
            "Recomputed Year and Month from the authoritative Date field.",
            "corrected",
        )
        low, high = EXPECTED_PERIODS["test.csv"]
        outside = parsed.notna() & (
            parsed.dt.strftime("%Y-%m").lt(low) | parsed.dt.strftime("%Y-%m").gt(high)
        )
        record(
            issues, "test.csv", "date_outside_expected_window", int(outside.sum()),
            "Preserved Date and flagged out-of-window keys.",
            "error",
        )
        frame["Year"] = parsed.dt.year.astype("Int64")
        frame["Month"] = parsed.dt.month.astype("Int64")
        clean_sales_metadata(frame, "test.csv")
        dup = frame.duplicated(["Date", "SKU_ID", "Region"], keep=False)
        record(
            issues, "test.csv", "duplicate_Date_SKU_Region", int(dup.sum()),
            "Retained duplicate forecast keys for review; no forecast row is silently discarded.",
            "error",
        )
        file_metrics["test.csv"] = {
            "input_rows": len(frame),
            "output_rows": len(frame),
            "outside_expected_period_rows": int(outside.sum()),
            "duplicate_forecast_key_rows": int(dup.sum()),
        }
        return frame

    train_clean = clean_train()
    test_clean = clean_test()
    sku_clean = master.drop(columns=["_source_row_number"], errors="ignore")
    file_metrics["sku_master.csv"] = {
        "input_rows": len(master),
        "output_rows": len(sku_clean),
        "duplicate_sku_id_rows": int(master["SKU_ID"].duplicated(keep=False).sum()),
        "expected_rows": EXPECTED_ROWS["sku_master.csv"],
    }

    def clean_reference(file_name: str, date_columns=(), category_columns=(),
                        numeric_columns=()) -> pd.DataFrame:
        frame = _read_csv(data_dir, file_name)
        note_missing(file_name, frame)
        normalize_sku_ids(frame, file_name)
        if "SKU_ID" in frame:
            unknown_sku = ~frame["SKU_ID"].isin(known_sku_ids)
            record(
                issues, file_name, "unknown_sku_id", int(unknown_sku.sum()),
                "Kept rows; verify SKU_ID against sku_master.csv.",
                "error",
                frame.loc[unknown_sku, ["SKU_ID"]].head(5).to_dict("records"),
            )
        input_rows = len(frame)
        parsed_date = None
        for column in date_columns:
            parsed = clean_period(frame, file_name, column)
            if column == "Date":
                parsed_date = parsed
                frame[column] = parsed.dt.strftime("%Y-%m").fillna("")
            else:
                raw_values = frame[f"{column}_Raw"].fillna("").astype(str)
                frame[column] = [
                    parsed_value.strftime("%Y-%m") if len(raw_value.strip()) == 7
                    and pd.notna(parsed_value)
                    else parsed_value.strftime("%Y-%m-%d") if pd.notna(parsed_value) else ""
                    for raw_value, parsed_value in zip(raw_values, parsed)
                ]
        for column in category_columns:
            if column == "Region":
                standardize(frame, column, file_name, canonical_regions, REGION_ALIASES)
            elif column == "Sales_Channel":
                standardize(frame, column, file_name, canonical_channels, CHANNEL_ALIASES)
        for column in numeric_columns:
            if column in frame:
                values, _ = numeric_column(frame, column, file_name)
                frame[column] = values
        if parsed_date is not None and {"Year", "Month"}.issubset(frame.columns):
            year = pd.to_numeric(frame["Year"], errors="coerce")
            month = pd.to_numeric(frame["Month"], errors="coerce")
            mismatch = parsed_date.notna() & (
                year.ne(parsed_date.dt.year) | month.ne(parsed_date.dt.month)
            )
            record(
                issues, file_name, "date_year_month_mismatch", int(mismatch.sum()),
                "Date is authoritative; Year and Month are recomputed from parsed Date.",
                "corrected",
            )
            record(
                corrections, file_name, "recomputed_year_month_from_Date", int(mismatch.sum()),
                "Recomputed Year and Month from the authoritative Date field.",
                "corrected",
            )
            frame["Year"] = parsed_date.dt.year.astype("Int64")
            frame["Month"] = parsed_date.dt.month.astype("Int64")
        if file_name == "macro_indicators.csv" and "Date" in frame:
            dup = frame.duplicated("Date", keep=False)
            record(
                issues, file_name, "duplicate:Date", int(dup.sum()),
                "Exact duplicate rows are removed; conflicting month rows are retained and flagged.",
                "review",
                frame.loc[dup, ["Date"]].head(5).to_dict("records"),
            )
            exact = frame.duplicated(keep="first")
            if exact.any():
                frame = frame.loc[~exact].copy()
                record(
                    corrections, file_name, "removed_exact_duplicate_rows", int(exact.sum()),
                    "Removed fully identical repeated rows only.",
                    "corrected",
                )
        if file_name == "inventory_stockouts.csv":
            fields = [
                "Requested_Units", "Carried_In_Units", "Shipped_Units",
                "Backordered_Units", "Lost_Units", "Fill_Rate_Pct", "Stockout_Days",
            ]
            numeric = {}
            for column in fields:
                numeric[column], _ = numeric_column(frame, column, file_name)
                frame[column] = numeric[column]
            expected_lost = (
                numeric["Requested_Units"] + numeric["Carried_In_Units"]
                - numeric["Shipped_Units"] - numeric["Backordered_Units"]
            )
            loss_mismatch = expected_lost.notna() & numeric["Lost_Units"].notna() & (
                expected_lost.sub(numeric["Lost_Units"]).abs().gt(0.01)
            )
            record(
                issues, file_name, "stockout_lost_units_formula_mismatch",
                int(loss_mismatch.sum()),
                "Preserved source values; review requested/shipped/backorder arithmetic.",
                "error",
            )
            denominator = numeric["Requested_Units"] + numeric["Carried_In_Units"]
            computed_fill = numeric["Shipped_Units"].div(denominator.where(denominator.ne(0))) * 100
            fill_mismatch = computed_fill.notna() & numeric["Fill_Rate_Pct"].notna() & (
                computed_fill.sub(numeric["Fill_Rate_Pct"]).abs().gt(0.1)
            )
            record(
                issues, file_name, "stockout_fill_rate_mismatch",
                int(fill_mismatch.sum()),
                "Preserved source fill rate; compare with shipped divided by requested plus carried-in.",
                "review",
            )
        file_metrics[file_name] = {
            "input_rows": input_rows,
            "output_rows": len(frame),
            "expected_rows": EXPECTED_ROWS.get(file_name),
        }
        return frame

    macro_clean = clean_reference(
        "macro_indicators.csv", date_columns=("Date",),
        numeric_columns=(
            "Year", "Month", "ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct",
            "Vehicle_Inventory_Days", "Raw_Material_Cost_Index",
            "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
            "Fuel_Price_Index", "EV_Adoption_Index",
        ),
    )
    events_clean = clean_reference(
        "events_calendar.csv", date_columns=("Start_Date", "End_Date"),
    )
    stockouts_clean = clean_reference(
        "inventory_stockouts.csv", date_columns=("Date",),
        category_columns=("Region",),
    )
    sample_clean = clean_reference(
        "sample_submission.csv", date_columns=("Date",),
        category_columns=("Region",),
        numeric_columns=("Units_Sold_Forecast",),
    )

    output_frames = {
        "train_cleaned.csv": train_clean,
        "test_cleaned.csv": test_clean,
        "sku_master_cleaned.csv": sku_clean,
        "macro_indicators_cleaned.csv": macro_clean,
        "events_calendar_cleaned.csv": events_clean,
        "inventory_stockouts_cleaned.csv": stockouts_clean,
        "sample_submission_cleaned.csv": sample_clean,
    }
    for file_name, frame in output_frames.items():
        frame.to_csv(cleaned_dir / file_name, index=False, encoding="utf-8-sig")

    issue_frame = pd.DataFrame(issues)
    correction_frame = pd.DataFrame(corrections)
    issue_frame.to_csv(quality_dir / "data_quality_issues.csv", index=False, encoding="utf-8-sig")
    correction_frame.to_csv(
        quality_dir / "data_corrections.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(price_correction_rows).to_csv(
        quality_dir / "price_correction_log.csv", index=False, encoding="utf-8-sig"
    )
    for file_name, metrics in file_metrics.items():
        metrics["expected_rows"] = EXPECTED_ROWS.get(file_name, metrics.get("expected_rows"))

    summary = {
        "project_root": str(root),
        "input_directory": str(data_dir),
        "cleaned_output_directory": str(cleaned_dir),
        "quality_report_directory": str(quality_dir),
        "price_correction_log": str(quality_dir / "price_correction_log.csv"),
        "file_metrics": file_metrics,
        "issues": issues,
        "corrections": corrections,
        "assumptions": [
            "Date is authoritative; Year and Month are recomputed from parseable Date values.",
            "Duplicate Record_ID rows keep the latest Load_Timestamp; source row breaks timestamp ties.",
            "Missing Units_Sold is not zero and remains missing; literal zero continues to mean no order.",
            "Missing family, business unit, and SKU description may be filled from the matching SKU master row.",
            "Conflicting Business_Unit values are corrected to sku_master.csv; original values are retained in *_Raw columns.",
            "Price corrections are limited to same-SKU/month evidence and clear powers-of-ten errors; uncertain outliers are flagged.",
            "Revenue is recomputed as Units_Sold times cleaned Unit_Price_USD when both are available; source value is retained in Revenue_USD_Raw.",
            "Rows outside the stated history window are retained and flagged, not silently deleted.",
            "Rows with unresolved composite-key duplicates are retained for review, not arbitrarily aggregated.",
        ],
    }
    with (quality_dir / "data_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    return summary
