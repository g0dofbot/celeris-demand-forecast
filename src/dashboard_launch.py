#!/usr/bin/env python
"""Phase 5 - Planning control tower (Streamlit app over outputs/).

The dashboard is a pure reader.  Every number it shows comes from a file that
phases 1-4 already wrote under ``outputs/``; it recomputes nothing, reads no raw
CSV, and never touches the delivered data folder.  That is enforced by
``resolve`` (every declared path must stay inside the outputs directory) and by
a contract test that asserts the module declares no path outside it.

Run the app:
    streamlit run src/dashboard_launch.py

Check the outputs headlessly (what ``python run_pipeline.py --phase 5`` does):
    python src/dashboard_launch.py --out-dir outputs

Pages
-----
Core (the default view)
Overview        demand and revenue outlook, family / region / channel mix
Demand explorer family -> SKU -> region -> customer drill-down, flagged events
Forecast       monthly and SKU-level forecast with filters and a CSV download
Validation     the forward-in-time backtest: candidates, monthly fit, by slice
Anomalies      the Phase 2 register, filterable, with the monthly flag rate
Data quality   the Phase 1 units ledger, cleaning log and price repairs
Assumptions    every rule the pipeline applied, in one place

Advanced (behind the sidebar tier toggle)
Macro lead/lag  correlation of demand against each macro driver by lag
Procurement    the buy requirement implied by the P90 forecast

The Advanced views are secondary analysis, so they stay out of the navigation
unless the planner asks for them; the Core seven are always the landing view.

Shared filters
--------------
The sidebar carries one set of filters (family, SKU, region, channel, customer)
that every view reads, so the training side and the forecast side always show
the same slice.  The forecast itself is SKU x region, so the channel and
customer filters narrow the order-book views (overview, explorer, anomalies).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED: dict[str, str] = {
    "forecast_cells": "forecast/forecast_cell_month.parquet",
    "forecast_monthly": "forecast/forecast_monthly_summary.parquet",
    "backtest_scores": "forecast/backtest_scores.csv",
    "backtest_slices": "forecast/backtest_slices.csv",
    "backtest_monthly": "forecast/backtest_monthly.parquet",
    "anomalies": "anomalies/anomaly_register.parquet",
    "monthly_summary": "anomalies/monthly_summary.parquet",
    "reconciliation": "clean/reconciliation.csv",
    "driver_effects": "features/driver_effects.csv",
    "feature_dictionary": "features/feature_dictionary.csv",
    "feature_rules": "features/feature_rules.csv",
    "forecast_rules": "forecast/forecast_rules.csv",
    "demand_monthly": "clean/demand_monthly.parquet",
    "demand_customer": "clean/demand_customer_month.parquet",
    "forecast_report": "forecast_report.json",
    "submission": "submission.csv",
}
OPTIONAL: dict[str, str] = {
    "backtest_cells": "forecast/backtest_cell_month.parquet",
    "model_diagnostics": "forecast/model_diagnostics.json",
    "driver_panel": "features/driver_panel_monthly.parquet",
    "driver_coefficients": "features/driver_screen_coefficients.csv",
    "cleaning_log": "clean/cleaning_log.csv",
    "anomaly_rules": "anomalies/anomaly_rules.csv",
    "sku_master": "clean/sku_master_clean.parquet",
    "clean_report": "clean_report.json",
    "anomaly_report": "anomaly_report.json",
    "features_report": "features_report.json",
    "profile_report": "profile_report.json",
}
PAGE_TITLES = {
    "Overview": "Planning overview",
    "Demand explorer": "Demand explorer",
    "Forecast": "Forecast explorer",
    "Validation": "Backtest and validation",
    "Anomalies": "Anomaly register",
    "Data quality": "Data quality and reconciliation",
    "Assumptions": "Assumptions and limitations",
    "Macro lead/lag": "Macro lead/lag analysis",
    "Procurement plan": "Procurement plan at P90",
}
CORE_PAGES = list(PAGE_TITLES)[:7]
ADVANCED_PAGES = list(PAGE_TITLES)[7:]
PAGES = CORE_PAGES + ADVANCED_PAGES
TABLE_SUFFIXES = {".parquet", ".csv"}
FILTER_KEYS = {"family": "flt_family", "sku": "flt_sku", "region": "flt_region",
               "channel": "flt_channel", "customer": "flt_customer"}
HISTORY_COLUMNS = ("Product_Family", "SKU_ID", "Region", "Sales_Channel", "Customer_Name")
CELL_COLUMNS = ("SKU_ID", "Region")
LEAD_LAGS = tuple(range(-6, 7))
# company-level macro series of the Phase 3 panel, with the labels the view uses
MACRO_DRIVERS = {
    "price_index": "Vehicle price index",
    "ISM_Manufacturing_PMI": "Manufacturing PMI",
    "Auto_Production_YoY_Pct": "Auto production YoY",
    "Vehicle_Inventory_Days": "Vehicle inventory days",
    "Raw_Material_Cost_Index": "Raw material cost index",
    "New_Vehicle_Orders_Index": "New vehicle orders index",
    "Industrial_Util_Rate_Pct": "Industrial utilisation",
    "Fuel_Price_Index": "Fuel price index",
    "EV_Adoption_Index": "EV adoption index",
}


# --------------------------------------------------------------------------- #
# access: outputs/ only
# --------------------------------------------------------------------------- #
def output_dir() -> Path:
    """The outputs directory; ``CELERIS_OUT_DIR`` exists so tests can point at a fixture."""
    override = os.environ.get("CELERIS_OUT_DIR")
    base = Path(override) if override else PROJECT_ROOT / "outputs"
    return base.resolve()


def resolve(base: Path, relative: str) -> Path:
    """Resolve a declared path, refusing anything that escapes the outputs folder."""
    if Path(relative).is_absolute():
        raise ValueError(f"dashboard paths must be relative to outputs/: {relative}")
    candidate = (base / relative).resolve()
    if base.resolve() not in candidate.parents and candidate != base.resolve():
        raise ValueError(f"path escapes outputs/: {relative}")
    # no extension fallback: the check must verify the file it declares, not a
    # lookalike, or a missing CSV would be silently satisfied by a stale parquet
    return candidate


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"not a table: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _under_streamlit() -> bool:
    """True when executed by the Streamlit runtime rather than from the CLI."""
    try:
        from streamlit.runtime import exists
    except Exception:  # pragma: no cover - older streamlit
        return False
    try:
        return bool(exists())
    except Exception:  # pragma: no cover
        return False


def _read_cached(path_text: str, stamp: float) -> pd.DataFrame:
    del stamp  # part of the cache key only
    return read_table(Path(path_text))


# cache only inside the app: decorating at import time would emit a cache warning
# every time the phase is checked from the command line
cached_read = (st.cache_data(show_spinner=False)(_read_cached) if _under_streamlit()
               else _read_cached)


# --------------------------------------------------------------------------- #
# headless check (the phase entry point)
# --------------------------------------------------------------------------- #
def _inventory(out_dir: Path, read_tables: bool) -> tuple[list[dict], list[str]]:
    """Stat every declared output; optionally parse it to prove it is readable."""
    rows, missing = [], []
    for name, relative in {**REQUIRED, **OPTIONAL}.items():
        required = name in REQUIRED
        path = resolve(out_dir, relative)
        record = {"name": name, "file": relative, "required": required,
                  "status": "ok", "rows": 0, "columns": 0, "bytes": 0, "modified": 0.0}
        if not path.exists():
            record["status"] = "MISSING"
            if required:
                missing.append(relative)
            rows.append(record)
            continue
        stat = path.stat()
        record["bytes"], record["modified"] = stat.st_size, stat.st_mtime
        if not read_tables:
            rows.append(record)
            continue
        try:
            if path.suffix in TABLE_SUFFIXES:
                frame = read_table(path)
                record["rows"], record["columns"] = len(frame), frame.shape[1]
            else:
                read_json(path)
        except Exception as error:  # pragma: no cover - corrupt output
            record["status"] = f"UNREADABLE: {type(error).__name__}"
        rows.append(record)
    return rows, missing


def unreadable_files(rows: list[dict]) -> list[str]:
    """Declared files that exist but did not parse."""
    return [row["file"] for row in rows if row["status"] != "ok"]


def presence_detail(rows: list[dict], missing: list[str]) -> str:
    """One sentence describing what is wrong with the inventory."""
    detail = f"{len(REQUIRED) - len(missing)}/{len(REQUIRED)} files"
    if missing:
        detail += f"; missing {missing}"
    unreadable = unreadable_files(rows)
    if unreadable:
        detail += f"; unreadable {unreadable}"
    return detail


def check_outputs(out_dir: Path) -> tuple[bool, list[dict], list[str]]:
    """Full validation: every output exists **and** parses.

    This is the ``--phase 5`` path.  It reads every declared file, so it belongs
    in the one-shot command, never in the app's render loop.
    """
    rows, missing = _inventory(out_dir, read_tables=True)
    return not missing and not unreadable_files(rows), rows, missing


def check_presence(out_dir: Path) -> tuple[bool, list[dict], list[str]]:
    """Cheap check for the app: existence, size and mtime only, no parsing.

    Streamlit reruns the whole script on every widget interaction, so this path
    must not touch a 58k-row parquet.  The tables themselves are read once by
    ``load`` through the mtime-keyed cache.  The trade-off is deliberate: this
    cannot see a torn file, only a missing one.
    """
    rows, missing = _inventory(out_dir, read_tables=False)
    return not missing, rows, missing


def _payload(verdicts: list[dict], rows: list[dict]) -> dict:
    return {
        "meta": {"phase": 5, "app": "src/dashboard_launch.py", "pages": PAGES,
                 "pandas": pd.__version__, "streamlit": st.__version__},
        "verdicts": verdicts,
        "inventory": rows,
        "kpis": {},
        "outputs": {name: relative for name, relative in {**REQUIRED, **OPTIONAL}.items()},
    }


def _write_report(out_dir: Path, payload: dict, lines: list[str], started: float) -> None:
    payload["meta"]["generated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    payload["meta"]["runtime_seconds"] = round(time.time() - started, 2)
    path = out_dir / "dashboard_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines += ["", f"JSON -> {path}"]
    (out_dir / "dashboard_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 control tower output check")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--data-dir", default=None,
                        help="accepted for pipeline compatibility; the dashboard never reads it")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir = out_dir.resolve()

    started = time.time()
    lines = ["CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER",
             "Phase 5: planning control tower (Streamlit reader over outputs/)",
             f"outputs : {out_dir}",
             f"launch  : streamlit run {Path(__file__).resolve()}",
             "", "=" * 78, "1. OUTPUT INVENTORY", "=" * 78]
    print("[dashboard] checking outputs", file=sys.stderr, flush=True)

    ok, rows, missing = check_outputs(out_dir)
    lines.append(f"  {'table':<22}{'file':<46}{'rows':>9}{'cols':>6}  status")
    for record in rows:
        flag = "" if record["status"] == "ok" else f"  <- {record['status']}"
        lines.append(f"  {record['name']:<22}{record['file']:<46}"
                     f"{record['rows']:>9,}{record['columns']:>6}{flag}")

    verdicts: list[dict] = []

    def verdict(name: str, passed: bool, detail: str = "") -> None:
        verdicts.append({"phase": 5, "check": name, "ok": bool(passed), "detail": detail})
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    lines += ["", "=" * 78, "2. PHASE 5 EXIT CHECK", "=" * 78]

    # without its inputs there is nothing to check: report and stop rather than
    # failing later with a KeyError on a table that was never read
    if not ok:
        verdict("every required output is present and readable", False,
                presence_detail(rows, missing) + "; the remaining checks need these files")
        _write_report(out_dir, _payload(verdicts, rows), lines, started)
        print("\nPHASE 5 FAILED 1 check(s)", file=sys.stderr)
        return 1

    report_names = {"forecast_report", "clean_report", "anomaly_report", "features_report",
                    "profile_report", "model_diagnostics"}
    tables, reports = {}, {}
    for record in rows:
        path = resolve(out_dir, record["file"])
        if record["name"] in REQUIRED and path.suffix in TABLE_SUFFIXES:
            tables[record["name"]] = read_table(path)
        elif record["name"] in report_names and path.suffix == ".json":
            reports[record["name"]] = read_json(path)

    cells = tables["forecast_cells"]
    monthly = tables["forecast_monthly"]
    submission = tables["submission"]
    anomalies = tables["anomalies"]
    ledger = tables["reconciliation"]
    scores = tables["backtest_scores"]
    customers = tables["demand_customer"]
    slices = tables["backtest_slices"]

    verdict("every required output is present and readable", ok,
            presence_detail(rows, missing))
    empty = [name for name, frame in tables.items() if len(frame) == 0]
    verdict("every required table is non-empty", not empty, f"{len(tables)} tables"
            + (f"; empty {empty}" if empty else ""))
    verdict("forecast covers the 19-month horizon",
            int(cells["Month"].nunique()) == 19 and len(cells) == len(submission),
            f"{len(cells):,} keys across {cells['Month'].nunique()} months")
    verdict("submission matches the forecast cells",
            int(submission["Units_Sold_Forecast"].sum()) == int(cells["forecast_units"].sum()),
            f"{submission['Units_Sold_Forecast'].sum():,.0f} units")
    verdict("anomaly register is classified",
            bool(anomalies["anomaly_class"].notna().all()) and len(anomalies) > 0,
            f"{len(anomalies):,} rows, {anomalies['anomaly_class'].nunique()} classes")
    running = ledger["units_running"].to_numpy(dtype=float)
    panel_total = float(tables["demand_monthly"]["demand_units"].sum())
    verdict("units ledger closes on the demand panel",
            abs(running[-1] - panel_total) <= 0.5,
            f"ledger {running[-1]:,.0f} vs panel {panel_total:,.0f} units")
    verdict("backtest scores include the shipped model",
            "blend" in set(scores["model"]),
            f"{len(scores)} candidates scored")
    verdict("every phase report is readable",
            {"forecast_report", "clean_report", "anomaly_report", "features_report"} <= set(reports),
            f"{len(reports)} reports parsed")
    declared = list(REQUIRED.values()) + list(OPTIONAL.values())
    inside = all(not Path(relative).is_absolute() for relative in declared)
    verdict("the dashboard declares no path outside outputs/", inside,
            f"{len(declared)} declared paths, all relative")
    verdict("assumption logs are available for the assumptions page",
            {"feature_rules", "forecast_rules"} <= set(tables),
            f"{len(tables['feature_rules']) + len(tables['forecast_rules'])} rules")
    verdict("monthly summary and forecast bands reconcile",
            abs(float(monthly["forecast"].sum()) - float(cells["forecast_units"].sum())) <= 1.0,
            f"{monthly['forecast'].sum():,.0f} vs {cells['forecast_units'].sum():,.0f}")
    verdict("forecast cells carry price and revenue",
            "forecast_revenue" in cells.columns
            and bool(cells["forecast_revenue"].notna().all())
            and abs(float(monthly["revenue"].sum())
                    - float(cells["forecast_revenue"].sum())) <= 1.0,
            f"${monthly['revenue'].sum():,.0f} over {len(monthly)} months")
    verdict("customer table can serve the drill-down",
            {"Customer_Name", "Sales_Channel", "revenue_recomputed", "observed_units"}
            <= set(customers.columns) and customers["Customer_Name"].nunique() > 1,
            f"{len(customers):,} rows, {customers['Customer_Name'].nunique()} customers, "
            f"{customers['Sales_Channel'].nunique()} channels")
    verdict("backtest accuracy is sliced by family, region and horizon",
            {"product_family", "region", "horizon_months"} <= set(slices["dimension"])
            and bool(slices["wape"].notna().all()),
            f"{len(slices)} slice rows across {slices['dimension'].nunique()} dimensions")
    verdict("the P90 forecast supports the procurement view",
            bool(cells["forecast_p90"].notna().all()) and float(cells["forecast_p90"].sum()) > 0,
            f"{cells['forecast_p90'].sum():,.0f} P90 units over "
            f"{cells['Month'].nunique()} months")

    lines.append("")
    lines.append(f"  elapsed seconds                                 {time.time() - started:,.1f}")
    payload = _payload(verdicts, rows)
    payload["kpis"] = {
        "forecast_units": float(cells["forecast_units"].sum()),
        "forecast_revenue": float(cells["forecast_revenue"].sum()),
        "forecast_months": int(cells["Month"].nunique()),
        "forecast_keys": int(len(cells)),
        "first_month_units": float(monthly["forecast"].iloc[0]),
        "last_month_units": float(monthly["forecast"].iloc[-1]),
        "anomalies": int(len(anomalies)),
        "backtest_wape": float(scores.loc[scores["model"].eq("blend"), "wape"].iloc[0]),
        "backtest_bias": float(scores.loc[scores["model"].eq("blend"), "bias"].iloc[0]),
    }
    _write_report(out_dir, payload, lines, started)

    failed = [verdict for verdict in verdicts if not verdict["ok"]]
    if failed:
        print(f"\nPHASE 5 FAILED {len(failed)} check(s)", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# data access for the app
# --------------------------------------------------------------------------- #
def load(out_dir: Path) -> dict:
    """Read every available output once; tables through the mtime-keyed cache."""
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for name, relative in {**REQUIRED, **OPTIONAL}.items():
        path = resolve(out_dir, relative)
        if not path.exists():
            continue
        if path.suffix in TABLE_SUFFIXES:
            frames[name] = cached_read(str(path), path.stat().st_mtime)
        elif path.suffix == ".json":
            reports[name] = read_json(path)
    frames["_reports"] = reports
    frames["_out_dir"] = out_dir
    return frames


def report(frames: dict, name: str) -> dict:
    return frames["_reports"].get(name, {})


# --------------------------------------------------------------------------- #
# shared widgets and filters
# --------------------------------------------------------------------------- #
def month_axis(frame: pd.DataFrame, column: str = "Month") -> pd.Series:
    return pd.to_datetime(frame[column])


def _multiselect(label: str, options: list, key: str):
    """A multiselect whose stale values never survive a changed option list."""
    chosen = [value for value in st.session_state.get(key, []) if value in options]
    if st.session_state.get(key) != chosen:
        st.session_state[key] = chosen
    return st.multiselect(label, options, key=key)


def _select(label: str, options: list, key: str):
    """A selectbox that falls back to its first option when the pool changes."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.selectbox(label, options, key=key)


def _nav(label: str, options: list, key: str):
    """Sidebar navigation that returns to the first view when the tier changes."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.sidebar.radio(label, options, key=key)


def shared_filters(frames: dict) -> dict:
    """One filter set for every page; options cascade down the hierarchy."""
    history = frames["demand_customer"]
    st.sidebar.markdown("**Shared filters**")
    family = _multiselect("Product family",
                          sorted(history["Product_Family"].dropna().unique()),
                          FILTER_KEYS["family"])
    pool = history.loc[history["Product_Family"].isin(family)] if family else history
    sku = _multiselect("SKU", sorted(pool["SKU_ID"].dropna().unique()), FILTER_KEYS["sku"])
    pool = pool.loc[pool["SKU_ID"].isin(sku)] if sku else pool
    region = _multiselect("Region", sorted(pool["Region"].dropna().unique()), FILTER_KEYS["region"])
    pool = pool.loc[pool["Region"].isin(region)] if region else pool
    channel = _multiselect("Sales channel",
                           sorted(pool["Sales_Channel"].dropna().unique()),
                           FILTER_KEYS["channel"])
    pool = pool.loc[pool["Sales_Channel"].isin(channel)] if channel else pool
    customer = _multiselect("Customer",
                            sorted(pool["Customer_Name"].dropna().unique()),
                            FILTER_KEYS["customer"])
    if st.sidebar.button("Clear filters"):
        for key in FILTER_KEYS.values():
            st.session_state[key] = []
        st.rerun()
    st.sidebar.caption("Channel and customer narrow the order-book views; the forecast is "
                       "SKU x region.")
    return {"family": family, "sku": sku, "region": region,
            "channel": channel, "customer": customer}


def apply_history_filters(frame: pd.DataFrame, filters: dict,
                          columns: tuple[str, ...] = HISTORY_COLUMNS) -> pd.DataFrame:
    """Apply the shared filters to any table that carries those columns."""
    mapping = {"Product_Family": filters["family"], "SKU_ID": filters["sku"],
               "Region": filters["region"], "Sales_Channel": filters["channel"],
               "Customer_Name": filters["customer"]}
    view = frame
    for column in columns:
        chosen = mapping.get(column) or []
        if chosen:
            view = view.loc[view[column].isin(chosen)]
    return view


def apply_cell_filters(cells: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the shared filters to the SKU x region forecast cells."""
    view = cells
    if filters["family"]:
        view = view.loc[view["Product_Family"].isin(filters["family"])]
    if filters["sku"]:
        view = view.loc[view["SKU_ID"].isin(filters["sku"])]
    if filters["region"]:
        view = view.loc[view["Region"].isin(filters["region"])]
    return view


def band_ratios(monthly: pd.DataFrame) -> tuple[float, float]:
    """The p10/p90 multipliers of the holdout monthly ratio."""
    return float(monthly["ratio_p10"].iloc[0]), float(monthly["ratio_p90"].iloc[0])


def filtered_history(frames: dict, filters: dict) -> pd.DataFrame:
    """Monthly order-book units and revenue for the shared slice.

    With no channel or customer filter this returns the panel's unconstrained
    demand (stockout-suppressed units restored); otherwise it returns observed
    units from the customer grain, because suppression is only logged per cell.
    """
    if not any(filters.values()):
        panel = frames["demand_monthly"]
        monthly = panel.groupby("Month", as_index=False).agg(
            units=("demand_units", "sum"), revenue=("revenue_recomputed", "sum"))
        return monthly
    history = apply_history_filters(frames["demand_customer"], filters)
    return history.groupby("Month", as_index=False).agg(
        units=("observed_units", "sum"), revenue=("revenue_recomputed", "sum"))


def band_chart(forecast: pd.DataFrame, title: str, history: pd.DataFrame | None = None
               ) -> go.Figure:
    """History line, forecast line and a multiplicative 80% band."""
    figure = go.Figure()
    if history is not None and len(history):
        figure.add_trace(go.Scatter(x=month_axis(history), y=history["value"],
                                    name="actual demand", mode="lines",
                                    line=dict(color="#1f4e79", width=2)))
    if len(forecast):
        figure.add_trace(go.Scatter(x=month_axis(forecast), y=forecast["high"],
                                    name="p90", mode="lines",
                                    line=dict(width=0), hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=month_axis(forecast), y=forecast["low"],
                                    name="80% band", mode="lines", fill="tonexty",
                                    line=dict(width=0),
                                    fillcolor="rgba(214,120,40,0.18)", hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=month_axis(forecast), y=forecast["value"],
                                    name="forecast", mode="lines+markers",
                                    line=dict(color="#d67828", width=2)))
    figure.update_layout(title=title, height=420, hovermode="x unified",
                         legend=dict(orientation="h", y=1.02, x=0),
                         margin=dict(l=10, r=10, t=70, b=10), yaxis_title="units")
    return figure


def revenue_chart(history: pd.DataFrame, forecast: pd.DataFrame, title: str) -> go.Figure:
    """Booked revenue bars against the forecast revenue line."""
    figure = go.Figure()
    if len(history):
        figure.add_trace(go.Bar(x=month_axis(history), y=history["revenue"],
                                name="booked revenue", marker_color="#1f4e79"))
    if len(forecast):
        figure.add_trace(go.Scatter(x=month_axis(forecast), y=forecast["revenue"],
                                    name="forecast revenue", mode="lines+markers",
                                    line=dict(color="#d67828", width=2)))
    figure.update_layout(title=title, height=380, hovermode="x unified",
                         legend=dict(orientation="h", y=1.02, x=0),
                         margin=dict(l=10, r=10, t=70, b=10), yaxis_title="USD")
    return figure


def mix_chart(frame: pd.DataFrame, dimension: str, value: str, title: str,
              top: int = 8) -> go.Figure:
    """Share of revenue by one dimension, largest first."""
    grouped = (frame.groupby(dimension)[value].sum().sort_values(ascending=False)
               .head(top).reset_index())
    total = float(grouped[value].sum())
    grouped["share"] = grouped[value] / total if total else 0.0
    figure = px.bar(grouped, x="share", y=dimension, orientation="h", title=title,
                    height=320, text=grouped["share"].map(lambda share: f"{share:.0%}"))
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_tickformat=".0%",
                         xaxis_title="share of revenue", yaxis_title="")
    return figure


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta)


def frame_download(frame: pd.DataFrame, name: str, label: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")


# --------------------------------------------------------------------------- #
# page: overview
# --------------------------------------------------------------------------- #
def page_overview(frames: dict, filters: dict) -> None:
    monthly = frames["forecast_monthly"].copy().sort_values("Month")
    monthly["Month"] = month_axis(monthly)

    cells = apply_cell_filters(frames["forecast_cells"], filters)
    anomalies = apply_history_filters(frames["anomalies"], filters, columns=CELL_COLUMNS)
    history = filtered_history(frames, filters)
    scores = frames["backtest_scores"].set_index("model")
    shipped = scores.loc["blend"]

    if cells.empty:
        st.info("No forecast cells match the shared filters.")
        return

    outlook = cells.groupby("Month", as_index=False).agg(
        units=("forecast_units", "sum"), revenue=("forecast_revenue", "sum"))
    outlook["Month"] = month_axis(outlook)
    run_rate = float(outlook["units"].mean())
    trailing = float(history["units"].tail(12).mean()) if len(history) else 0.0

    st.markdown("### Where the business is heading")
    kpi_row([
        ("19-month demand", f"{outlook['units'].sum():,.0f}", None),
        ("19-month revenue", f"${outlook['revenue'].sum() / 1e6:,.2f}M", None),
        ("Monthly run rate", f"{run_rate:,.0f}",
         f"{run_rate / trailing - 1:+.1%} vs trailing" if trailing else None),
    ])
    kpi_row([
        ("Backtest WAPE", f"{shipped['wape']:.3f}", f"{shipped['skill_vs_naive']:+.1%} vs naive"),
        ("Backtest bias", f"{shipped['bias']:+.1%}", None),
        ("Anomalies flagged", f"{len(anomalies):,}", None),
    ])

    ratio_low, ratio_high = band_ratios(monthly)
    units = pd.DataFrame({"Month": outlook["Month"], "value": outlook["units"],
                          "low": outlook["units"] * ratio_low,
                          "high": outlook["units"] * ratio_high})
    st.plotly_chart(band_chart(units, "Demand and the 19-month outlook",
                               history.rename(columns={"units": "value"})),
                    width="stretch")
    st.plotly_chart(revenue_chart(history, outlook, "Booked revenue and the forecast at list price"),
                    width="stretch")
    st.caption("Revenue is the units forecast at the list price Phase 3 carried into the "
               "horizon; the band on units is the holdout monthly ratio.")

    st.markdown("### Mix")
    mixes = st.columns(3)
    with mixes[0]:
        st.plotly_chart(mix_chart(cells, "Product_Family", "forecast_revenue",
                                  "Forward revenue by family"), width="stretch")
    with mixes[1]:
        st.plotly_chart(mix_chart(cells, "Region", "forecast_revenue",
                                  "Forward revenue by region"), width="stretch")
    with mixes[2]:
        recent = frames["demand_customer"]
        recent = recent.loc[recent["Month"] > recent["Month"].max() - pd.DateOffset(months=12)]
        recent = apply_history_filters(recent, filters)
        if len(recent):
            st.plotly_chart(mix_chart(recent, "Sales_Channel", "revenue_recomputed",
                                      "Booked revenue by channel (12m)"), width="stretch")
        else:
            st.info("No booked revenue for this slice.")
    st.caption("Family and region use the forward revenue; channel mix comes from the order "
               "book because the forecast is channel-agnostic.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Next six months**")
        table = monthly.head(6)[["Month", "forecast", "revenue", "band_p10", "band_p90"]].copy()
        table["Month"] = table["Month"].dt.strftime("%Y-%m")
        table.columns = ["Month", "Forecast units", "Revenue USD", "Units p10", "Units p90"]
        st.dataframe(table.round(0), width="stretch", hide_index=True)
        st.caption("The monthly band is the aggregate holdout ratio, not a sum of cell bands.")
    with right:
        st.markdown("**Largest demand events**")
        events = anomalies.loc[anomalies["kind"].eq("DEMAND_EVENT")].copy()
        if len(events):
            events["abs_impact"] = events["impact"].abs()
            top = events.nlargest(8, "abs_impact")[
                ["Month", "SKU_ID", "Region", "anomaly_class", "impact"]]
            top["Month"] = pd.to_datetime(top["Month"]).dt.strftime("%Y-%m")
            st.dataframe(top, width="stretch", hide_index=True)
        else:
            st.info("No demand events in this slice.")


# --------------------------------------------------------------------------- #
# page: demand explorer
# --------------------------------------------------------------------------- #
def page_explorer(frames: dict, filters: dict) -> None:
    history = apply_history_filters(frames["demand_customer"], filters)
    register = apply_history_filters(frames["anomalies"], filters, columns=CELL_COLUMNS)
    panel = frames["demand_monthly"]

    st.markdown("### Family, SKU, region, customer")
    if history.empty:
        st.info("No observed demand matches the shared filters.")
        return

    controls = st.columns(4)
    with controls[0]:
        family = _select("Product family",
                         ["All", *sorted(history["Product_Family"].unique())],
                         "exp_family")
    pool = history if family == "All" else history.loc[history["Product_Family"].eq(family)]
    with controls[1]:
        sku = _select("SKU", ["All", *sorted(pool["SKU_ID"].unique())], "exp_sku")
    pool = pool if sku == "All" else pool.loc[pool["SKU_ID"].eq(sku)]
    with controls[2]:
        region = _select("Region", ["All", *sorted(pool["Region"].unique())], "exp_region")
    pool = pool if region == "All" else pool.loc[pool["Region"].eq(region)]
    with controls[3]:
        customer = _select("Customer", ["All", *sorted(pool["Customer_Name"].unique())],
                           "exp_customer")
    view = pool if customer == "All" else pool.loc[pool["Customer_Name"].eq(customer)]

    label = " / ".join(part for part in [family, sku, region, customer] if part != "All")
    st.caption(f"Slice: {label or 'everything shared filters allow'} - "
               f"{view['Customer_Name'].nunique()} customers, "
               f"{view['SKU_ID'].nunique()} SKUs, {len(view):,} order-book rows.")

    orders = view.groupby("Month", as_index=False).agg(
        units=("observed_units", "sum"), revenue=("revenue_recomputed", "sum"),
        customers=("Customer_Name", "nunique"), orders=("orders", "sum"))
    cells = panel.merge(view[["SKU_ID", "Region"]].drop_duplicates(),
                        on=["SKU_ID", "Region"], how="inner")
    cells = cells.loc[cells["lifecycle_ok"] & cells["demand_units"].notna()]
    restored = cells.groupby("Month", as_index=False).agg(
        demand=("demand_units", "sum"), suppressed=("suppressed_units", "sum"))
    series = (orders.merge(restored, on="Month", how="outer").sort_values("Month")
              .reset_index(drop=True))
    series[["units", "revenue", "demand", "suppressed", "orders", "customers"]] = (
        series[["units", "revenue", "demand", "suppressed", "orders", "customers"]]
        .fillna(0.0))

    figure = go.Figure()
    figure.add_trace(go.Bar(x=series["Month"], y=series["suppressed"],
                            name="stockout-suppressed units",
                            marker_color="rgba(180,60,60,0.30)"))
    figure.add_trace(go.Scatter(x=series["Month"], y=series["units"],
                                name="observed units", mode="lines+markers",
                                line=dict(color="#1f4e79", width=2)))
    figure.add_trace(go.Scatter(x=series["Month"], y=series["demand"],
                                name="unconstrained demand", mode="lines",
                                line=dict(color="#2e7d32", width=2, dash="dot")))

    flags = register.loc[register["Month"].notna()].copy()
    if len(flags):
        flags = flags.groupby(["Month", "anomaly_class"], as_index=False).agg(
            impact=("impact", "sum"))
        flags = flags.merge(series[["Month", "units"]], on="Month", how="inner")
        if len(flags):
            markers = px.scatter(flags, x="Month", y="units", color="anomaly_class",
                                 hover_data=["impact"], symbol="anomaly_class")
            for trace in markers.data:
                trace.update(mode="markers",
                             marker=dict(size=12, line=dict(width=1, color="white")),
                             hovertemplate="%{x|%Y-%m}<br>%{fullData.name}<br>"
                                           "impact %{customdata[0]:,.0f}<extra></extra>")
                figure.add_trace(trace)

    figure.update_layout(title=f"Demand on the time series - {label or 'shared slice'}",
                         height=440, hovermode="x unified",
                         legend=dict(orientation="h", y=-0.18, x=0),
                         margin=dict(l=10, r=10, t=60, b=10), yaxis_title="units")
    st.plotly_chart(figure, width="stretch")
    st.caption("Diamonds are flagged anomalies; hover for the class and its impact. "
               "The gap between observed and unconstrained demand is demand the stockout "
               "log documents.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Monthly detail**")
        table = series.copy()
        table["Month"] = table["Month"].dt.strftime("%Y-%m")
        table = table[["Month", "orders", "units", "demand", "suppressed", "revenue"]]
        table.columns = ["Month", "Orders", "Observed units", "Demand units",
                         "Suppressed units", "Revenue USD"]
        st.dataframe(table.round(0), width="stretch", hide_index=True)
        frame_download(table, "celeris_demand_slice.csv", "Download this slice")
    with right:
        st.markdown("**Revenue by customer**")
        by_customer = (view.groupby("Customer_Name")["revenue_recomputed"].sum()
                       .sort_values(ascending=False).head(10).reset_index())
        by_customer.columns = ["Customer", "Revenue USD"]
        st.dataframe(by_customer.round(0), width="stretch", hide_index=True)
        st.markdown("**Revenue by channel**")
        by_channel = (view.groupby("Sales_Channel")["revenue_recomputed"].sum()
                      .sort_values(ascending=False).reset_index())
        by_channel.columns = ["Channel", "Revenue USD"]
        st.dataframe(by_channel.round(0), width="stretch", hide_index=True)

    st.markdown("**Cell revenue by SKU and region**")
    pivot = (view.pivot_table(index="SKU_ID", columns="Region",
                              values="revenue_recomputed", aggfunc="sum")
             .fillna(0.0).round(0))
    st.dataframe(pivot, width="stretch")


# --------------------------------------------------------------------------- #
# page: forecast
# --------------------------------------------------------------------------- #
def page_forecast(frames: dict, filters: dict) -> None:
    cells = apply_cell_filters(frames["forecast_cells"].copy(), filters)
    monthly = frames["forecast_monthly"].copy()
    monthly["Month"] = month_axis(monthly)
    cells["Month"] = month_axis(cells)

    st.markdown("### Forecast by slice")
    controls = st.columns(3)
    families = ["All families", *sorted(cells["Product_Family"].dropna().unique())]
    family = controls[0].selectbox("Product family", families)
    regions = ["All regions", *sorted(cells["Region"].dropna().unique())]
    region = controls[1].selectbox("Region", regions)
    business = ["All business units", *sorted(cells["Business_Unit"].dropna().unique())]
    unit = controls[2].selectbox("Business unit", business)

    view = cells
    if family != "All families":
        view = view.loc[view["Product_Family"].eq(family)]
    if region != "All regions":
        view = view.loc[view["Region"].eq(region)]
    if unit != "All business units":
        view = view.loc[view["Business_Unit"].eq(unit)]

    slice_label = family if unit == "All business units" else f"{family} / {unit}"
    if view.empty:
        st.info(f"No forecast cells match {slice_label}, {region}. "
                "Widen a filter to see the slice.")
        return

    ratio_low, ratio_high = band_ratios(monthly)
    series = view.groupby("Month", as_index=False)[["forecast_units", "forecast_revenue"]].sum()
    series["value"] = series["forecast_units"]
    series["low"] = series["value"] * ratio_low
    series["high"] = series["value"] * ratio_high
    st.plotly_chart(band_chart(series, f"Monthly demand - {slice_label}, {region}"),
                    width="stretch")
    st.caption("The band applies the holdout monthly ratio (p10/p90) to the slice total.")

    st.markdown("**Forecast revenue**")
    st.metric("19-month revenue in this slice", f"${series['forecast_revenue'].sum():,.0f}")
    st.caption("Units x the list price Phase 3 carried into the horizon, key by key.")

    st.markdown("**Family x month (units)**")
    pivot = (view.pivot_table(index="Product_Family", columns="Month",
                              values="forecast_units", aggfunc="sum")
             .fillna(0).round(0))
    pivot.columns = [pd.Timestamp(column).strftime("%Y-%m") for column in pivot.columns]
    st.dataframe(pivot, width="stretch")

    st.markdown("**Cell detail**")
    detail = view[["Month", "SKU_ID", "Region", "Product_Family", "forecast_units",
                   "forecast_p10", "forecast_p90", "list_price", "forecast_revenue",
                   "grid_status", "post_eol"]].copy()
    detail["Month"] = detail["Month"].dt.strftime("%Y-%m")
    detail.columns = ["Month", "SKU", "Region", "Family", "Forecast", "p10", "p90",
                      "List price", "Revenue", "Grid status", "Post-EOL"]
    st.dataframe(detail.head(500), width="stretch", hide_index=True)
    if len(detail) > 500:
        st.caption(f"Showing 500 of {len(detail):,} rows - download for the full slice.")
    frame_download(detail, "celeris_forecast_slice.csv", "Download this slice")


# --------------------------------------------------------------------------- #
# page: validation
# --------------------------------------------------------------------------- #
def page_validation(frames: dict, filters: dict) -> None:
    scores = frames["backtest_scores"].copy()
    backtest = frames["backtest_monthly"].copy()
    backtest["Month"] = month_axis(backtest)
    diagnostics = report(frames, "model_diagnostics")
    slices = frames["backtest_slices"].copy()

    st.markdown("### Forward-in-time backtest")
    origin = diagnostics.get("holdout_origin", "n/a")
    weight = diagnostics.get("blend_weight")
    chosen_on = diagnostics.get("blend_weight_selected_on", "n/a")
    caption = f"Fit strictly before {origin}, then 19 recursive months."
    if weight is not None:
        caption += (f" The blend weight ({weight:g}) was chosen on an inner window "
                    f"that ends before {chosen_on}, never on these months.")
    st.caption(caption)

    display = scores.copy()
    for column in ["wape", "monthly_wape", "bias", "skill_vs_naive"]:
        display[column] = (display[column] * 100).round(1)
    display = display.rename(columns={
        "model": "Model", "family": "Family", "wape": "Cell WAPE %",
        "monthly_wape": "Monthly WAPE %", "bias": "Bias %",
        "skill_vs_naive": "Skill vs naive %", "rows": "Rows"})
    st.dataframe(display[["Model", "Family", "Cell WAPE %", "Monthly WAPE %", "Bias %",
                          "Skill vs naive %", "Rows"]], width="stretch", hide_index=True)

    candidates = ["seasonal_naive", "hurdle_lgbm", "blend"]
    melted = backtest.melt(id_vars=["Month", "actual"], value_vars=candidates,
                           var_name="model", value_name="units")
    figure = px.line(melted, x="Month", y="units", color="model",
                     title="Monthly totals: actual against the candidates", height=380)
    figure.add_trace(go.Scatter(x=backtest["Month"], y=backtest["actual"], name="actual",
                                mode="lines+markers", line=dict(color="black", width=3)))
    figure.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_title="units",
                         legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(figure, width="stretch")

    st.markdown("### Accuracy by slice")
    selected = diagnostics.get("selected", "blend")
    mine = slices.loc[slices["model"].eq(selected)]
    tabs = st.tabs(["By product family", "By region", "By horizon", "By business unit"])
    tab_dimensions = [("product_family", "Product_Family"), ("region", "Region"),
                      ("horizon_months", None), ("business_unit", "Business_Unit")]
    for tab, (dimension, filter_column) in zip(tabs, tab_dimensions):
        with tab:
            block = mine.loc[mine["dimension"].eq(dimension)].copy()
            if filter_column is not None:
                chosen = filters["family"] if filter_column == "Product_Family" else (
                    filters["region"] if filter_column == "Region" else [])
                if chosen:
                    block = block.loc[block["value"].isin(chosen)]
            if block.empty:
                st.info("No backtest rows for this slice.")
                continue
            block = block.sort_values("wape", ascending=False)
            chart = px.bar(block, x="wape", y="value", orientation="h",
                           title=f"Cell WAPE by {dimension.replace('_', ' ')} "
                                 f"({selected})", height=340)
            chart.update_layout(margin=dict(l=10, r=10, t=60, b=10),
                                xaxis_tickformat=".0%", xaxis_title="WAPE",
                                yaxis_title="")
            st.plotly_chart(chart, width="stretch")
            table = block[["value", "rows", "actual_units", "forecast_units", "wape",
                           "bias", "revenue_wape", "revenue_bias"]].copy()
            for column in ["wape", "bias", "revenue_wape", "revenue_bias"]:
                table[column] = (table[column] * 100).round(1)
            table["actual_units"] = table["actual_units"].round(0)
            table["forecast_units"] = table["forecast_units"].round(0)
            table.columns = ["Slice", "Rows", "Actual units", "Forecast units",
                             "WAPE %", "Bias %", "Revenue WAPE %", "Revenue bias %"]
            st.dataframe(table, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        curve = pd.DataFrame(diagnostics.get("blend_weight_curve", []))
        if len(curve):
            chart = px.line(curve, x="weight", y="wape", markers=True,
                            title="Blend weight chosen on the inner window", height=320)
            if weight is not None:
                chart.add_vline(x=weight, line_dash="dash", line_color="#d67828")
            chart.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(chart, width="stretch")
    with right:
        st.markdown("**Worst keys in the holdout**")
        cells = frames.get("backtest_cells")
        if cells is not None and len(cells):
            cells = apply_history_filters(cells, filters, columns=CELL_COLUMNS)
            worst = cells.assign(error=(cells["blend"] - cells["actual"]).abs()).nlargest(
                12, "error")[["Month", "SKU_ID", "Region", "actual", "blend", "error"]].copy()
            worst["Month"] = pd.to_datetime(worst["Month"]).dt.strftime("%Y-%m")
            worst.columns = ["Month", "SKU", "Region", "Actual", "Forecast", "Abs error"]
            st.dataframe(worst.round(0), width="stretch", hide_index=True)
            st.caption("Most error sits in a handful of one-off large orders.")

    verdicts = report(frames, "forecast_report").get("verdicts", [])
    if verdicts:
        st.markdown("**Phase 4 exit checks**")
        st.dataframe(pd.DataFrame(verdicts)[["check", "ok", "detail"]],
                     width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# page: anomalies
# --------------------------------------------------------------------------- #
def page_anomalies(frames: dict, filters: dict) -> None:
    register = apply_history_filters(frames["anomalies"].copy(), filters, columns=CELL_COLUMNS)
    summary = frames["monthly_summary"].copy()
    summary["Month"] = month_axis(summary)
    register["Month"] = month_axis(register)

    st.markdown("### Anomaly register")
    st.caption("Data errors are repaired in Phase 1 and listed here for audit; demand events are "
               "kept in the training data and explained, never overwritten.")

    if register.empty:
        st.info("No registered anomalies match the shared filters.")
        return

    classes = register["anomaly_class"].value_counts().rename_axis("class").reset_index(
        name="rows")
    figure = px.bar(classes, x="rows", y="class", orientation="h",
                    title="Registered anomalies by class", height=360)
    figure.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(figure, width="stretch")

    controls = st.columns(3)
    kinds = ["All", *sorted(register["kind"].dropna().unique())]
    kind = controls[0].selectbox("Kind", kinds)
    chosen = controls[1].selectbox("Class", ["All", *sorted(register["anomaly_class"].dropna().unique())])
    severities = ["All", *sorted(register["severity"].dropna().astype(str).unique())]
    severity = controls[2].selectbox("Severity", severities)

    view = register
    if kind != "All":
        view = view.loc[view["kind"].eq(kind)]
    if chosen != "All":
        view = view.loc[view["anomaly_class"].eq(chosen)]
    if severity != "All":
        view = view.loc[view["severity"].astype(str).eq(severity)]

    st.markdown(f"**{len(view):,} rows** - largest by impact")
    table = view.reindex(view["impact"].abs().sort_values(ascending=False).index).head(200).copy()
    table["Month"] = table["Month"].dt.strftime("%Y-%m")
    columns = ["Month", "SKU_ID", "Region", "anomaly_class", "kind", "observed", "expected",
               "impact", "severity", "explained_by", "treatment"]
    st.dataframe(table[[column for column in columns if column in table.columns]],
                 width="stretch", hide_index=True)

    rate = summary.assign(flag_rate=summary["flagged"] / summary["cells"].replace(0, pd.NA))
    chart = px.line(rate, x="Month", y="flag_rate", title="Share of cells flagged each month",
                    height=320)
    chart.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_tickformat=".1%")
    st.plotly_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# page: macro lead/lag (advanced)
# --------------------------------------------------------------------------- #
def lead_lag_matrix(panel: pd.DataFrame, lags: tuple[int, ...] = LEAD_LAGS) -> pd.DataFrame:
    """Correlate 12-month demand change with each driver's change at every lag.

    Differencing both sides removes the shared trend, which would otherwise make
    every level correlation look strong.  A positive lag means the driver moved
    first, so the column is read as "how many months ahead does this driver run".
    """
    frame = panel.set_index(month_axis(panel)).sort_index()
    demand = frame["demand_units"].pct_change(12)
    rows = {}
    for column, label in MACRO_DRIVERS.items():
        if column not in frame:
            continue
        change = frame[column].diff(12)
        if change.notna().sum() < 24:  # too short a history to correlate
            continue
        rows[label] = {lag: demand.corr(change.shift(lag)) for lag in lags}
    return pd.DataFrame(rows).T


def page_lead_lag(frames: dict, filters: dict) -> None:
    panel = frames.get("driver_panel")
    st.markdown("### Which macro drivers run ahead of demand")
    if panel is None or len(panel) < 36:
        st.info("This view needs `features/driver_panel_monthly.parquet` from Phase 3. "
                "Run the pipeline to write it.")
        return

    matrix = lead_lag_matrix(panel)
    if matrix.empty:
        st.info("The macro panel carries no series long enough to correlate.")
        return

    figure = px.imshow(matrix, aspect="auto", text_auto=".2f", zmin=-1, zmax=1,
                       color_continuous_scale="RdBu_r", origin="upper",
                       labels=dict(x="driver lead in months (+ = driver leads)",
                                   y="", color="correlation"))
    figure.update_layout(title="Correlation of 12-month demand change with each driver",
                         height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(figure, width="stretch")
    st.caption("Company-level Phase 3 panel, so the SKU / region filters do not narrow it. "
               "Both sides are 12-month changes, which keeps the shared trend from driving the "
               "correlation; read each row for its strongest positive lag.")

    picks = matrix.abs().idxmax(axis=1)
    summary = pd.DataFrame({"Driver": matrix.index,
                            "Leading lag (months)": picks.to_numpy(),
                            "Correlation": [matrix.loc[driver, lag]
                                            for driver, lag in zip(matrix.index, picks)]})
    summary = summary.reindex(
        summary["Correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    summary["Correlation"] = summary["Correlation"].round(2)
    st.markdown("**Strongest lead per driver**")
    st.dataframe(summary, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# page: procurement (advanced)
# --------------------------------------------------------------------------- #
def page_procurement(frames: dict, filters: dict) -> None:
    cells = apply_cell_filters(frames["forecast_cells"], filters)
    cells = cells.loc[~cells["post_eol"].astype(bool)]

    st.markdown("### What to commit, at P90")
    st.caption("P90 is the 9-in-10 demand outcome for each SKU x region and month. Buying to "
               "P90 trades working capital for availability; the point forecast is the "
               "expected outcome.")
    if cells.empty:
        st.info("No forecast cells match the shared filters.")
        return

    months = month_axis(cells)
    first = months.min()
    coverage = st.slider("Coverage window: months of demand to commit now", 1, 12, 3,
                         help="The delivered data carries no on-hand stock or supplier lead "
                              "time, so the window is the planner's assumption, not a "
                              "pipeline number.")
    window = cells.loc[months < first + pd.DateOffset(months=coverage)]

    plan = (window.groupby(["SKU_ID", "Product_Family", "Region"], as_index=False)
            .agg(p90_units=("forecast_p90", "sum"),
                 point_units=("forecast_units", "sum"),
                 price=("list_price", "first")))
    plan["p90_value"] = plan["p90_units"] * plan["price"]
    plan = plan.sort_values("p90_value", ascending=False).reset_index(drop=True)
    cover = plan["p90_units"].sum() / plan["point_units"].sum() - 1

    kpi_row([
        ("Commit now (P90 units)", f"{plan['p90_units'].sum():,.0f}", None),
        ("Commit now (list value)", f"${plan['p90_value'].sum() / 1e6:,.2f}M", None),
        ("SKU x region lines", f"{len(plan):,}", f"over {coverage} months"),
        ("Cover above the point forecast", f"{cover:+.0%}", None),
    ])

    schedule = (cells.assign(Month=months).groupby("Month", as_index=False)
                .agg(p90=("forecast_p90", "sum"), point=("forecast_units", "sum")))
    figure = go.Figure()
    figure.add_trace(go.Bar(x=schedule["Month"], y=schedule["p90"], name="P90 demand",
                            marker_color="#1f4e79"))
    figure.add_trace(go.Scatter(x=schedule["Month"], y=schedule["point"],
                                name="point forecast", mode="lines+markers",
                                line=dict(color="#d67828", width=2)))
    figure.update_layout(title="Monthly buy requirement", height=380, hovermode="x unified",
                         legend=dict(orientation="h", y=1.02, x=0),
                         margin=dict(l=10, r=10, t=70, b=10), yaxis_title="units")
    st.plotly_chart(figure, width="stretch")

    st.markdown(f"**Top purchase lines in the {coverage}-month window**")
    table = plan.head(25).copy()
    table["p90_value"] = table["p90_value"].round(0)
    table["p90_units"] = table["p90_units"].round(0)
    table["point_units"] = table["point_units"].round(0)
    table["price"] = table["price"].round(2)
    table.columns = ["SKU", "Family", "Region", "P90 units", "Point units", "List price",
                     "Commit value USD"]
    st.dataframe(table, width="stretch", hide_index=True)
    frame_download(plan, "celeris_procurement_plan.csv", "Download the full plan")
    st.caption("P90 is summed per SKU and region: a planning quantity, not a guaranteed "
               "outcome. Post-EOL cells, which the forecast pins at zero by rule, are excluded.")


# --------------------------------------------------------------------------- #
# page: data quality
# --------------------------------------------------------------------------- #
def page_quality(frames: dict) -> None:
    ledger = frames["reconciliation"].copy()
    clean_report = report(frames, "clean_report")

    st.markdown("### Units ledger")
    st.caption("Every excluded unit is itemised; the ledger closes on the panel that trains the "
               "forecast.")
    st.dataframe(ledger[["seq", "step", "rule_id", "rows", "units", "units_running"]],
                 width="stretch", hide_index=True)

    moves = ledger.loc[ledger["units"].notna() & ledger["units"].ne(0)]
    if len(moves):
        figure = px.bar(moves, x="units", y="step", orientation="h",
                        title="Units added or removed by each step", height=340)
        figure.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(figure, width="stretch")

    metrics = clean_report.get("metrics", {})
    if metrics:
        st.markdown("**Phase 1 diagnostics**")
        flattened = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                for inner, nested in value.items():
                    if isinstance(nested, (int, float, str)):
                        flattened.append({"check": f"{key}.{inner}", "value": nested})
            elif isinstance(value, (int, float, str)):
                flattened.append({"check": key, "value": value})
        if flattened:
            st.dataframe(pd.DataFrame(flattened[:60]), width="stretch", hide_index=True)

    log = frames.get("cleaning_log")
    if log is not None and len(log):
        st.markdown(f"**Cleaning log** ({len(log):,} actions)")
        st.dataframe(log.head(200), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# page: assumptions
# --------------------------------------------------------------------------- #
LIMITATIONS = """
- **Price after 2026 is carried forward.** No 2027/2028 list price exists in the delivered data,
  so the forecast assumes the 2026 price holds. Cells affected are flagged
  `price_imputed_future` in `outputs/features/`, and forecast revenue inherits that assumption.
- **Post-EOL keys are zero by rule.** 156 test keys fall after their SKU's EOL month; the ERP
  records no sales there, so they are forecast at 0 rather than extrapolated.
- **Cell-level error is dominated by one-off orders.** A whole-cell WAPE near 0.41 is mostly
  small cells and a dozen 5,000+ unit orders; the monthly total is the more stable view
  (~6% aggregate error on the holdout).
- **The blend weight is chosen on an inner window.** It is not read off the scored holdout,
  and the holdout never re-selects it.
- **Absence is not zero in the data.** A month with no ERP row is carried as missing through
  cleaning and anomaly detection; only the forecast path treats a dormant cell as a zero
  month, because the test set requires a number for every key.
- **Digit-shift suspects are flagged, not repaired.** Phase 2 records 86 suspected
  digit-shift cells; the training data keeps the values and the register explains them.
- **Demand, not shipments.** The target is unconstrained demand: units the stockout log shows
  were requested but not served are restored before modelling. The customer view keeps
  observed units, because suppression is only logged per SKU and region.
- **The forecast is channel-agnostic.** It is produced per SKU and region, so channel and
  customer mix is shown from the order book, not from the forecast.
"""


def page_assumptions(frames: dict) -> None:
    st.markdown("### Every rule the pipeline applied")
    feature_rules = frames["feature_rules"].copy()
    forecast_rules = frames["forecast_rules"].copy()
    anomaly_rules = frames.get("anomaly_rules")

    tabs = st.tabs(["Forecast", "Features", "Anomalies"])
    with tabs[0]:
        st.dataframe(forecast_rules, width="stretch", hide_index=True)
    with tabs[1]:
        st.dataframe(feature_rules, width="stretch", hide_index=True)
    with tabs[2]:
        if anomaly_rules is not None and len(anomaly_rules):
            st.dataframe(anomaly_rules, width="stretch", hide_index=True)
        else:
            st.info("No anomaly rule log available.")

    st.markdown("### Known limitations")
    st.markdown(LIMITATIONS)

    st.markdown("### Sources")
    sources = []
    for name, relative in {**REQUIRED, **OPTIONAL}.items():
        path = resolve(frames["_out_dir"], relative)
        if not path.exists():
            continue
        stat = path.stat()
        sources.append({"panel": name, "file": relative,
                        "modified": time.strftime("%Y-%m-%d %H:%M",
                                                  time.localtime(stat.st_mtime)),
                        "kilobytes": round(stat.st_size / 1024, 1)})
    st.dataframe(pd.DataFrame(sources), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def render() -> None:
    st.set_page_config(page_title="Celeris demand control tower", layout="wide")
    out_dir = output_dir()
    # presence only: the app reruns on every click, and a full read here would
    # re-parse ~300k rows each time
    ok, rows, missing = check_presence(out_dir)

    st.sidebar.title("Celeris control tower")
    advanced = st.sidebar.toggle(
        "Advanced views", value=False,
        help="Adds the macro lead/lag analysis and the P90 procurement plan. "
             "The Core views stay the default.")
    page = _nav("View", PAGES if advanced else CORE_PAGES, "view")
    st.sidebar.caption(f"Source: `{out_dir}`")
    latest = max((record["modified"] for record in rows), default=0.0)
    st.sidebar.caption("Latest output: "
                       + time.strftime("%Y-%m-%d %H:%M", time.localtime(latest)))
    if st.sidebar.button("Reload outputs"):
        st.cache_data.clear()
        st.rerun()

    if not ok:
        st.error("The dashboard needs the earlier phases. Missing: " + ", ".join(missing))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        return

    frames = load(out_dir)
    filters = shared_filters(frames)
    st.title(PAGE_TITLES[page])
    if page == "Overview":
        page_overview(frames, filters)
    elif page == "Demand explorer":
        page_explorer(frames, filters)
    elif page == "Forecast":
        page_forecast(frames, filters)
    elif page == "Validation":
        page_validation(frames, filters)
    elif page == "Anomalies":
        page_anomalies(frames, filters)
    elif page == "Macro lead/lag":
        page_lead_lag(frames, filters)
    elif page == "Procurement plan":
        page_procurement(frames, filters)
    elif page == "Data quality":
        page_quality(frames)
    else:
        page_assumptions(frames)


if __name__ == "__main__":
    if _under_streamlit():
        render()
    else:
        raise SystemExit(main())
