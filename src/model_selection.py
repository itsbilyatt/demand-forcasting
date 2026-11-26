from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["Origin", "Date", "SKU_ID", "Region"]
WEIGHT_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def _find_root(project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "outputs" / "backtesting").is_dir():
            return candidate
    raise FileNotFoundError("Run baseline and global-model backtests before model selection.")


def _wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float(np.abs(prediction - actual).sum() / denominator) if denominator else np.nan


def run_model_selection(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _find_root(project_root)
    baseline_path = root / "outputs" / "backtesting" / "rolling_origin_predictions.csv"
    global_path = root / "outputs" / "global_model" / "rolling_origin_predictions.csv"
    output_dir = root / "outputs" / "model_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_path, low_memory=False)
    global_predictions = pd.read_csv(global_path, low_memory=False)
    seasonal = baseline.loc[
        baseline["Model"].eq("seasonal_naive_12"),
        KEYS + ["Horizon", "Horizon_Bucket", "Product_Family", "Actual", "Prediction"],
    ].rename(columns={"Prediction": "Seasonal_Prediction"})
    global_model = global_predictions.loc[
        global_predictions["Model"].eq("global_poisson_hgbr"),
        KEYS + ["Actual", "Prediction"],
    ].rename(columns={"Actual": "Global_Actual", "Prediction": "Global_Prediction"})
    joined = seasonal.merge(global_model, on=KEYS, how="inner", validate="one_to_one")
    expected = len(seasonal)
    if len(joined) != expected or joined.empty:
        raise ValueError(f"Global and seasonal predictions align on {len(joined)} of {expected} rows.")
    if not np.allclose(
        joined["Actual"].to_numpy(dtype=float),
        joined["Global_Actual"].to_numpy(dtype=float),
    ):
        raise ValueError("Global and seasonal validation targets do not align.")

    selection_rows = []
    selected_weights: dict[str, float] = {}
    origins = sorted(joined["Origin"].unique())
    if len(origins) < 2:
        raise ValueError("At least two independent rolling origins are required for blend validation.")
    for held_out_origin in origins:
        calibration = joined.loc[joined["Origin"].ne(held_out_origin)]
        validation = joined.loc[joined["Origin"].eq(held_out_origin)]
        actual_train = calibration["Actual"].to_numpy(dtype=float)
        candidates = [
            (
                _wape(
                    actual_train,
                    (1.0 - weight) * calibration["Seasonal_Prediction"].to_numpy(dtype=float)
                    + weight * calibration["Global_Prediction"].to_numpy(dtype=float),
                ),
                float(weight),
            )
            for weight in WEIGHT_GRID
        ]
        best_score = min(score for score, _ in candidates)
        best_weight = min(
            weight for score, weight in candidates if np.isclose(score, best_score)
        )
        selected_weights[held_out_origin] = best_weight
        actual = validation["Actual"].to_numpy(dtype=float)
        prediction = (
            (1.0 - best_weight) * validation["Seasonal_Prediction"].to_numpy(dtype=float)
            + best_weight * validation["Global_Prediction"].to_numpy(dtype=float)
        )
        selection_rows.append({
            "Origin": held_out_origin,
            "Calibration_Origins": ",".join(calibration["Origin"].unique()),
            "Selected_Global_Weight": best_weight,
            "Selected_Seasonal_Weight": 1.0 - best_weight,
            "Held_Out_WAPE": _wape(actual, prediction),
            "Held_Out_Bias": float((prediction - actual).sum() / np.abs(actual).sum()),
            "Seasonal_WAPE": _wape(
                actual, validation["Seasonal_Prediction"].to_numpy(dtype=float)
            ),
            "Global_WAPE": _wape(
                actual, validation["Global_Prediction"].to_numpy(dtype=float)
            ),
        })

    production_weight = float(np.mean(list(selected_weights.values())))
    joined["Model"] = "validated_blend"
    joined["Prediction"] = (
        (1.0 - production_weight) * joined["Seasonal_Prediction"]
        + production_weight * joined["Global_Prediction"]
    )
    blended = joined[
        KEYS + ["Horizon", "Horizon_Bucket", "Product_Family", "Actual", "Prediction", "Model"]
    ].copy()
    joined.drop(columns=["Global_Actual"], inplace=True)
    blended["WAPE_Contribution"] = (blended["Prediction"] - blended["Actual"]).abs()
    blended["Bias_Contribution"] = blended["Prediction"] - blended["Actual"]
    blended.to_csv(
        output_dir / "rolling_origin_predictions.csv", index=False, encoding="utf-8-sig"
    )

    metric_tables = {}
    for name, keys in (
        ("overall", ["Origin", "Model"]),
        ("horizon", ["Origin", "Model", "Horizon_Bucket", "Horizon"]),
        ("family", ["Origin", "Model", "Product_Family", "Horizon_Bucket"]),
        ("region", ["Origin", "Model", "Region", "Horizon_Bucket"]),
        ("family_region", [
            "Origin", "Model", "Product_Family", "Region", "Horizon_Bucket"
        ]),
    ):
        rows = []
        for group_values, group in blended.groupby(keys, dropna=False, sort=False):
            values = group_values if isinstance(group_values, tuple) else (group_values,)
            actual = group["Actual"].to_numpy(dtype=float)
            prediction = group["Prediction"].to_numpy(dtype=float)
            rows.append({
                **dict(zip(keys, values)),
                "Rows": int(len(group)),
                "WAPE": _wape(actual, prediction),
                "Bias": float((prediction - actual).sum() / np.abs(actual).sum()),
                "Actual_Units": float(actual.sum()),
                "Forecast_Units": float(prediction.sum()),
            })
        table = pd.DataFrame(rows)
        metric_tables[name] = table
        table.to_csv(output_dir / f"accuracy_{name}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(selection_rows).to_csv(
        output_dir / "leave_one_origin_out_blend_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "selected_method": "validated_blend",
        "global_model_weight": production_weight,
        "seasonal_naive_weight": 1.0 - production_weight,
        "selection_method": (
            "For each held-out origin, choose the WAPE-minimizing weight on all other origins; "
            "production weight is the mean of those out-of-origin selected weights."
        ),
        "leave_one_origin_out_results": selection_rows,
        "pooled_metrics": metric_tables["overall"].to_dict("records"),
        "prediction_rows": int(len(blended)),
        "outputs": str(output_dir),
    }
    with (output_dir / "model_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary
