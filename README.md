# Demand forecasting

Reproducible demand-forecasting workflow for the Transcend hackathon dataset.

## Set up

Use Python 3.12 and the project environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the notebook interpreter. The complete reproducible workflow, including backtests, final submission, business checks, and dashboard, can be run with one command from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

Failures stop the pipeline and show the failing stage. Individual notebooks remain available for exploration and are run in numeric order when you want to inspect a step.

## Main deliverables

- `outputs\submission\final_submission.csv` — exact required submission schema and one nonnegative integer forecast per test key.
- `outputs\submission\forecast_audit.csv` — forecast method, source, and recursive seasonal-naive trace.
- `outputs\business_checks\forecast_quantiles.csv` — point forecasts and approximate empirical P10/P50/P90 intervals.
- `outputs\model_selection\leave_one_origin_out_blend_validation.csv` — blend weights tuned on one temporal origin and evaluated on the other.
- `outputs\data_quality\price_correction_log.csv` — row-level price corrections with same-SKU/month evidence.
- `outputs\dashboard\index.html` — standalone offline dashboard with executive, data-health, demand-explorer, anomaly, and accuracy views.
- `outputs\pipeline\pipeline_summary.json` — stage timings and machine-readable run summary.

To regenerate only the dashboard after its source outputs are available:

```powershell
.\.venv\Scripts\python.exe -m src.dashboard
```

## Model and interpretation

The selected final method is a 55% seasonal-naive / 45% global Poisson HGB blend. The global weight is selected by leave-one-origin-out tuning on the two 19-month rolling origins. Both held-out WAPEs improved over the seasonal-naive and standalone global forecasts; the validation report retains the per-origin results and bias. Forecasts and uncertainty bounds are zero before launch and after the inclusive EOL month. Launched replacement SKUs without their own demand history use their predecessor's forecast as an explicit 1.0 transfer assumption.

Revenue projections use the latest observed SKU price and apply only explicit future global price-change events; no demand response to price or other events is assumed. The P10/P90 bands are approximate empirical intervals calibrated from blended rolling-origin residuals, not guaranteed service levels. Their SKU-level bounds are not additive aggregate quantiles. The source files do not provide inventory on hand, open purchase orders, lead times, or service-level targets, so purchase quantities are intentionally not calculated. Channel/customer filters scope forecast SKU-region combinations by historical association; they do not represent a demand allocation or customer-level forecast.
