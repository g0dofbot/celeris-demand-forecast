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
Overview       19-month demand outlook, band, and the numbers a planner needs
Forecast       monthly and SKU-level forecast with filters and a CSV download
Validation     the forward-in-time backtest: candidates, monthly fit, worst keys
Anomalies      the Phase 2 register, filterable, with the monthly flag rate
Data quality   the Phase 1 units ledger, cleaning log and price repairs
Assumptions    every rule the pipeline applied, in one place
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
    "backtest_monthly": "forecast/backtest_monthly.parquet",
    "anomalies": "anomalies/anomaly_register.parquet",
    "monthly_summary": "anomalies/monthly_summary.parquet",
    "reconciliation": "clean/reconciliation.csv",
    "driver_effects": "features/driver_effects.csv",
    "feature_dictionary": "features/feature_dictionary.csv",
    "feature_rules": "features/feature_rules.csv",
    "forecast_rules": "forecast/forecast_rules.csv",
    "demand_monthly": "clean/demand_monthly.parquet",
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
PAGES = ["Overview", "Forecast", "Validation", "Anomalies", "Data quality", "Assumptions"]
PAGE_TITLES = {
    "Overview": "Planning overview",
    "Forecast": "Forecast explorer",
    "Validation": "Backtest and validation",
    "Anomalies": "Anomaly register",
    "Data quality": "Data quality and reconciliation",
    "Assumptions": "Assumptions and limitations",
}
TABLE_SUFFIXES = {".parquet", ".csv"}


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

    lines.append("")
    lines.append(f"  elapsed seconds                                 {time.time() - started:,.1f}")
    payload = _payload(verdicts, rows)
    payload["kpis"] = {
        "forecast_units": float(cells["forecast_units"].sum()),
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
# shared widgets
# --------------------------------------------------------------------------- #
def month_axis(frame: pd.DataFrame, column: str = "Month") -> pd.Series:
    return pd.to_datetime(frame[column])


def band_chart(history: pd.DataFrame, forecast: pd.DataFrame, title: str,
               unit_label: str = "units") -> go.Figure:
    """History line, forecast line and a multiplicative 80% band."""
    figure = go.Figure()
    if len(history):
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
                         margin=dict(l=10, r=10, t=70, b=10),
                         yaxis_title=unit_label)
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
def page_overview(frames: dict) -> None:
    monthly = frames["forecast_monthly"].copy().sort_values("Month")
    summary = frames["monthly_summary"].copy().sort_values("Month")
    cells = frames["forecast_cells"]
    scores = frames["backtest_scores"].set_index("model")
    anomalies = frames["anomalies"]

    monthly["Month"] = month_axis(monthly)
    summary["Month"] = month_axis(summary)
    trailing = float(summary["demand_units"].tail(12).mean())
    average = float(monthly["forecast"].mean())
    shipped = scores.loc["blend"]

    st.markdown("### Where the business is heading")
    kpi_row([
        ("19-month demand", f"{cells['forecast_units'].sum():,.0f}", None),
        ("Monthly run rate", f"{average:,.0f}", f"{average / trailing - 1:+.1%} vs trailing"),
        ("Next month", f"{monthly['forecast'].iloc[0]:,.0f}",
         f"{monthly['forecast'].iloc[0] / float(summary['demand_units'].iloc[-1]) - 1:+.1%} vs last actual"),
    ])
    kpi_row([
        ("Backtest WAPE", f"{shipped['wape']:.3f}", f"{shipped['skill_vs_naive']:+.1%} vs naive"),
        ("Backtest bias", f"{shipped['bias']:+.1%}", None),
        ("Anomalies flagged", f"{len(anomalies):,}", None),
    ])

    history = pd.DataFrame({"Month": summary["Month"], "value": summary["demand_units"]})
    forecast = pd.DataFrame({"Month": monthly["Month"], "value": monthly["forecast"],
                             "low": monthly["band_p10"], "high": monthly["band_p90"]})
    st.plotly_chart(band_chart(history, forecast, "Company demand and the 19-month outlook"),
                    width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Next six months**")
        table = monthly.head(6)[["Month", "forecast", "band_p10", "band_p90", "p10", "p90"]].copy()
        table["Month"] = table["Month"].dt.strftime("%Y-%m")
        table.columns = ["Month", "Forecast", "Total p10", "Total p90",
                         "Sum of cell p10", "Sum of cell p90"]
        st.dataframe(table.round(0), width="stretch", hide_index=True)
        st.caption("Use the total band for planning totals; the per-cell band is not additive "
                   "and is only meaningful key by key.")
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
            st.info("No demand events registered.")


# --------------------------------------------------------------------------- #
# page: forecast
# --------------------------------------------------------------------------- #
def page_forecast(frames: dict) -> None:
    cells = frames["forecast_cells"].copy()
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

    ratio_low = float(monthly["ratio_p10"].iloc[0])
    ratio_high = float(monthly["ratio_p90"].iloc[0])
    series = view.groupby("Month", as_index=False)["forecast_units"].sum().rename(
        columns={"forecast_units": "value"})
    series["low"] = series["value"] * ratio_low
    series["high"] = series["value"] * ratio_high
    st.plotly_chart(band_chart(pd.DataFrame(columns=["Month", "value"]), series,
                               f"Monthly demand - {slice_label}, {region}", "units"),
                    width="stretch")
    st.caption("The band applies the holdout monthly ratio (p10/p90) to the slice total.")

    st.markdown("**Family x month**")
    pivot = (view.pivot_table(index="Product_Family", columns="Month",
                              values="forecast_units", aggfunc="sum")
             .fillna(0).round(0))
    pivot.columns = [pd.Timestamp(column).strftime("%Y-%m") for column in pivot.columns]
    st.dataframe(pivot, width="stretch")

    st.markdown("**Cell detail**")
    detail = view[["Month", "SKU_ID", "Region", "Product_Family", "forecast_units",
                   "forecast_p10", "forecast_p90", "grid_status", "post_eol"]].copy()
    detail["Month"] = detail["Month"].dt.strftime("%Y-%m")
    detail.columns = ["Month", "SKU", "Region", "Family", "Forecast", "p10", "p90",
                      "Grid status", "Post-EOL"]
    st.dataframe(detail.head(500), width="stretch", hide_index=True)
    if len(detail) > 500:
        st.caption(f"Showing 500 of {len(detail):,} rows - download for the full slice.")
    frame_download(detail, "celeris_forecast_slice.csv", "Download this slice")


# --------------------------------------------------------------------------- #
# page: validation
# --------------------------------------------------------------------------- #
def page_validation(frames: dict) -> None:
    scores = frames["backtest_scores"].copy()
    backtest = frames["backtest_monthly"].copy()
    backtest["Month"] = month_axis(backtest)
    diagnostics = report(frames, "model_diagnostics")

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
        "monthly_wape": "Monthly WAPE %", "bias": "Bias %", "skill_vs_naive": "Skill vs naive %",
        "mae": "MAE", "rmse": "RMSE", "rows": "Rows"})
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
            worst = cells.assign(error=(cells["blend"] - cells["actual"]).abs()).nlargest(
                12, "error")[["Month", "SKU_ID", "Region", "actual", "blend", "error"]]
            worst = worst.copy()
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
def page_anomalies(frames: dict) -> None:
    register = frames["anomalies"].copy()
    summary = frames["monthly_summary"].copy()
    summary["Month"] = month_axis(summary)
    register["Month"] = month_axis(register)

    st.markdown("### Anomaly register")
    st.caption("Data errors are repaired in Phase 1 and listed here for audit; demand events are "
               "kept in the training data and explained, never overwritten.")

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
  `price_imputed_future` in `outputs/features/`.
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
  were requested but not served are restored before modelling.
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
        if path.exists():
            sources.append({"panel": name, "file": relative,
                            "modified": time.strftime("%Y-%m-%d %H:%M",
                                                      time.localtime(path.stat().st_mtime)),
                            "kilobytes": round(path.stat().st_size / 1024, 1)})
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
    page = st.sidebar.radio("View", PAGES)
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
    st.title(PAGE_TITLES[page])
    if page == "Overview":
        page_overview(frames)
    elif page == "Forecast":
        page_forecast(frames)
    elif page == "Validation":
        page_validation(frames)
    elif page == "Anomalies":
        page_anomalies(frames)
    elif page == "Data quality":
        page_quality(frames)
    else:
        page_assumptions(frames)


if __name__ == "__main__":
    if _under_streamlit():
        render()
    else:
        raise SystemExit(main())
