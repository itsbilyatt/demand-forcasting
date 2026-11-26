from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from src.business_checks import run_business_checks
from src.dashboard import build_local_dashboard
from src.data_cleaning import run_data_cleaning
from src.demand_exploration import run_demand_exploration
from src.demand_reconstruction import run_demand_reconstruction
from src.feature_engineering import run_feature_engineering
from src.forecasting import create_final_submission
from src.global_model import run_global_model_backtest
from src.backtesting import run_baseline_backtest
from src.model_selection import run_model_selection


def run_pipeline(project_root: str | Path | None = None) -> dict[str, Any]:
    if project_root is None:
        root = Path.cwd().resolve()
    else:
        root = Path(project_root).resolve()
    if not (root / "data" / "student_data").is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root / 'data' / 'student_data'}")

    stages: list[tuple[str, Callable[[str | Path], dict[str, Any]]]] = [
        ("data_cleaning", run_data_cleaning),
        ("demand_reconstruction", run_demand_reconstruction),
        ("demand_exploration", run_demand_exploration),
        ("feature_engineering", run_feature_engineering),
        ("baseline_backtesting", run_baseline_backtest),
        ("global_model_backtesting", run_global_model_backtest),
        ("model_selection", run_model_selection),
        ("final_forecast", create_final_submission),
        ("business_checks", run_business_checks),
        ("dashboard", build_local_dashboard),
    ]
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for name, stage in stages:
        print(f"Running {name}...", flush=True)
        started = perf_counter()
        results[name] = stage(root)
        timings[name] = round(perf_counter() - started, 2)
        print(f"Completed {name} in {timings[name]:.2f}s", flush=True)

    summary = {
        "project_root": str(root),
        "stage_seconds": timings,
        "results": results,
    }
    output_dir = root / "outputs" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)
    print(f"Pipeline complete. Summary: {output_dir / 'pipeline_summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    run_pipeline()
