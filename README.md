# Celeris Automobili — demand forecasting control tower

A phase-gated pipeline that audits the raw ERP extract, cleans it, explains every
anomaly, and forecasts **unconstrained** monthly demand for 2026-11 → 2028-05
(19 months), plus a Streamlit dashboard that reads only the pipeline's outputs.

## 1. Where the data goes

Put the seven provided CSV files in **`student_data/` at the project root**:

```
student_data/
  train.csv  test.csv  macro_indicators.csv  sku_master.csv
  events_calendar.csv  inventory_stockouts.csv  sample_submission.csv
```

That directory is git-ignored: the data is not part of this repository, so a
fresh clone must be handed the CSVs before it can run. Phase 0 fails fast with a
clear message if any file is missing.

## 2. Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt          # macOS / Linux
```

Versions are pinned in `requirements.txt` (Python 3.14). `run_pipeline.py`
re-executes itself under `.venv/`, so on Windows you can start it with any
`python` on your PATH and it will still use the pinned environment.

## 3. Run end to end

```bash
python run_pipeline.py                # every phase, in order (~5.5 minutes)
python run_pipeline.py --phase 0      # one phase only
python run_pipeline.py --data-dir path/to/student_data
```

Phases run as isolated subprocesses, so one failure cannot leave the next stage
half-executed, and each stage's exit status is preserved. The run ends with a
`PIPELINE SUMMARY` block.

| Phase | Module | What it does | Main outputs |
| --- | --- | --- | --- |
| 0 | `src/profile_data.py` | Forensic audit of the raw CSVs, read-only, untyped | `outputs/profile_report.{json,txt}`, `outputs/profile/*.csv` |
| 1 | `src/clean.py` | Repairs data errors, reconciles units, builds the demand panel | `outputs/clean_report.*`, `outputs/clean/*.parquet` |
| 2 | `src/anomalies.py` | Separates data errors from real demand events, unconstrains demand | `outputs/anomaly_report.*`, `outputs/anomalies/*` |
| 3 | `src/features.py` | Feature build + driver screening, leakage-classed dictionary | `outputs/features_report.*`, `outputs/features/*` |
| 4 | `src/forecast.py` | Hurdle model, inner-window blend, 19-month backtest, forecast + submission | `outputs/forecast_report.*`, `outputs/forecast/*`, `outputs/submission.csv` |
| 5 | `src/dashboard_launch.py` | Contract checks over the outputs (no server started) | `outputs/dashboard_report.*` |

Phase order matters: each phase consumes the previous phase's outputs. Running a
later phase alone on a stale `outputs/` tree will still work, but `--phase all`
is the supported path.

### Constants worth knowing

* `Date` is authoritative; duplicates are identified by `Record_ID` and resolved
  by keeping the latest `Load_Timestamp`.
* `Units_Sold = 0` means "no order placed". A missing row is **not** zero.
* Price is one USD price per SKU, identical across regions, changing only on 1 January.
* No sales exist outside a SKU's `Launch_Date` → `EOL_Date` window.
* The forecast is unconstrained demand, not shipments: stockouts suppress
  observed units, and Phase 2 restores them before modelling.
* Random seeds are fixed (`SEED = 20261125`) and the blend weight is chosen on an
  inner window that ends before the backtest origin, never on the scored months.

## 4. Open the dashboard

```bash
streamlit run src/dashboard_launch.py
```

The dashboard reads exclusively from `outputs/`; it never touches `student_data/`
and never re-runs the pipeline, so re-run the phase that produces a table before
expecting it to change.

The sidebar carries **one filter set** — family, SKU, region, channel, customer —
shared by every view, so the order-book side and the forecast side always show the
same slice. Pages:

| Page | What it answers |
| --- | --- |
| Overview | 19-month units and revenue, book-vs-forecast revenue, family / region / channel mix |
| Demand explorer | family → SKU → region → customer drill-down, unconstrained vs observed demand, anomalies marked on the time series |
| Forecast | units, band and revenue by slice, family × month, cell detail, CSV download |
| Validation | candidates on the holdout, and WAPE / bias by product family, region, business unit and months ahead (units and revenue) |
| Anomalies | the Phase 2 register by class, kind and severity, with the monthly flag rate |
| Data quality | the units ledger, Phase 1 diagnostics and the cleaning log |
| Assumptions | every rule applied, the limitations and the sources |
| Macro lead/lag (Advanced) | 12-month demand change against each macro driver at lags of ±6 months, and the leading lag per driver |
| Procurement plan (Advanced) | the P90 buy requirement per SKU × region for a coverage window the planner sets, with list value and CSV download |

The two Advanced views sit behind the sidebar **Advanced views** toggle, so the Core
seven stay the landing navigation; the threshold table and the P90 plan are still read
from the same `outputs/` files as everything else.

Two tables exist only for these views and are produced by the pipeline:
`clean/demand_customer_month.parquet` (order book at customer × channel grain, with
price and revenue) and `forecast/backtest_slices.csv` (holdout error per slice).
The channel and customer filters narrow the order-book views; the forecast itself is
SKU × region, and revenue is the units forecast at the list price Phase 3 carries
into the horizon.

### Hosted on Streamlit Community Cloud

Vercel cannot host this app: it runs Python only as framework-preset functions
(FastAPI/Flask/Django, 300 s cap), and `streamlit run` starts its own server that
Vercel has no way to launch. Community Cloud runs the server as-is.

The 27 files the dashboard declares (`REQUIRED` + `OPTIONAL` in
`src/dashboard_launch.py`) are committed even though `outputs/` is git-ignored, so
a Cloud build has data without running the pipeline first.

* Repository `g0dofbot/celeris-demand-forecast`, branch `main`
* Entrypoint `src/dashboard_launch.py`
* Advanced settings → Python **3.14** (the version the pins were verified on)

## 5. What to look at first

1. `outputs/profile_report.txt` — the raw-data defects, in the order a judge
   should care about them. Defects are labelled `[FINDING]`, not `[FAIL]`: Phase 0
   is read-only, so finding a defect is its job. Phase 1 repairs each one and
   re-checks it as a `[PASS]`.
2. `outputs/clean_report.txt` — the cleaning ledger: every rule, how many rows
   and units it moved, and the reconciliation that proves no unit was invented
   or lost.
3. `outputs/anomaly_report.txt` — which signals were data errors and which were
   real demand events, with the treatment applied to each.
4. `outputs/forecast_report.json` and `outputs/forecast/backtest_scores.csv` —
   WAPE and bias per model on the 19-month holdout, plus the chosen blend.
5. `outputs/submission.csv` — the deliverable: `Date, SKU_ID, Region, Units_Sold_Forecast`.
6. `streamlit run src/dashboard_launch.py` — the same results, explorable.

## 6. Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Contract tests assert each phase's published outputs keep their promised shape
and invariants; they skip cleanly when `outputs/` has not been generated yet.

## 7. Layout

```
run_pipeline.py     phase runner (re-execs into .venv, isolates subprocesses)
src/                one module per phase
tests/              per-phase contract tests
outputs/            generated artifacts (git-ignored, except the 27 the dashboard reads)
student_data/       input CSVs (git-ignored)
logs/               run logs (git-ignored)
```
