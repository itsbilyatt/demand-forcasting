# Demand forecasting

Reproducible demand-forecasting workflow for the Celeris Automobili Industrial demand forecasting challenge.

## Problem-statement alignment

The implementation is aligned with the **Core tier** and with a substantial portion of the **Advanced and Expert requirements**.

| Workstream | Alignment | Evidence |
|---|---|---|
| 1. Cleaning | Core complete; Advanced largely complete; Expert reproducibility complete | Profiles all seven inputs, standardizes dates and labels, deduplicates ERP rows, reconciles quantity/price/revenue, preserves missing values and zeros as different states, and writes machine-readable quality reports. The one-command pipeline is reproducible. |
| 2. Anomalies | Core and substantial Advanced complete | Produces an anomaly register and separates stockouts, missing shipments, negative quantities, demand spikes, and history-window errors. Listed stockouts are reconstructed as requested demand, while uncertainty and incomplete coverage are retained. |
| 3. Features and drivers | Core and Advanced complete; part of Expert complete | Builds calendar, lag, rolling, intermittency, lifecycle, event, and macro features. Macro features use explicit 1- and 3-month lags. Lead-lag stability over time is not presented as a production-stable result. |
| 4. Forecasting | Core complete; Advanced complete; substantial Expert complete | Uses two baselines and a global HistGradientBoosting model, validates a blend over two 19-month rolling origins, and exports forecasts for all 58,219 test rows. P10/P50/P90, lifecycle/EOL, replacement-SKU transfer, event exposure, and hierarchy reconciliation are included. |
| 5. Dashboard | Five required views complete; optional Expert content is limited | A local standalone dashboard has executive overview, data health, demand explorer, anomaly monitor, and forecast/accuracy views, shared filters, and a continuous history-to-forecast timeline. It is not deployed as a public service and does not implement every optional view. |

Key expert and scope limitations are explicit rather than hidden:

- P10/P50/P90 are approximate empirical intervals from rolling-origin residuals, not guaranteed service levels.
- Aggregate P10/P90 values are sums of SKU-level bounds, not true aggregate quantiles.
- The source data does not contain inventory on hand, open purchase orders, lead times, or service-level targets, so a procurement plan cannot be responsibly generated.
- Known events are exposed; demand effects are not assumed without validated causal evidence.
- The forecast is unconstrained demand, not a promised shipment quantity.

## Set up

Use Python 3.12 and the project environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter. The complete reproducible workflow, including backtests, final submission, business checks, and dashboard, can be run with one command from the project root.

### Run the pipeline

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

On Linux or macOS:

```bash
.venv/bin/python -m src.pipeline
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

## Deploy to GitHub Pages

GitHub Pages is the recommended deployment for this dashboard because the generated
artifact is a standalone HTML file with no server-side dependencies.

1. Push this repository to GitHub.
2. Open **Settings > Pages**.
3. Set **Source** to **GitHub Actions**.
4. Push a commit to `main`, or run **Deploy dashboard to GitHub Pages** manually from
   the Actions tab.

The deployment workflow rebuilds the complete pipeline and publishes
`outputs/dashboard/index.html` automatically.

## Model and interpretation

The selected final method is a 55% seasonal-naive / 45% global Poisson HGB blend. The global weight is selected by leave-one-origin-out tuning on the two 19-month rolling origins. Both held-out WAPEs improved over the seasonal-naive and standalone global forecasts; the validation report retains the per-origin results and bias. Forecasts and uncertainty bounds are zero before launch and after the inclusive EOL month. Launched replacement SKUs without their own demand history use their predecessor's forecast as an explicit 1.0 transfer assumption.

Revenue projections use the latest observed SKU price and apply only explicit future global price-change events; no demand response to price or other events is assumed. The P10/P90 bands are approximate empirical intervals calibrated from blended rolling-origin residuals, not guaranteed service levels. Their SKU-level bounds are not additive aggregate quantiles. The source files do not provide inventory on hand, open purchase orders, lead times, or service-level targets, so purchase quantities are intentionally not calculated. Channel/customer filters scope forecast SKU-region combinations by historical association; they do not represent a demand allocation or customer-level forecast.
