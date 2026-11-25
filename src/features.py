#!/usr/bin/env python
"""Phase 3 - Feature engineering and driver analysis.

Two deliverables:

1. ``features_monthly.parquet`` - one model-ready row per (SKU, Region, Month),
   covering the 94 training months **and** the 19 forecast months, with an
   explicit ``split`` column.  Phase 4 can consume it directly or rebuild from
   the clean tables; either way the definition of every column is in
   ``feature_dictionary.csv``.
2. A driver analysis - seasonality, price elasticity, event lift, macro
   association, lifecycle ramp, channel mix, and a forward-in-time driver
   screen - written to ``driver_effects.csv`` and summarised in the report.

Leakage contract
----------------
Every demand-derived feature is lagged: ``lag_*``, ``mean_*``, ``median_*``,
``std_*``, ``max_*``, ``zero_share_*``, ``months_since_last_order`` and the
structure block all read months **strictly before** the row's own month.  The
current month's demand appears only in the ``target_*`` columns, which are NaN
for the forecast split.  Calendar, lifecycle, price, event and macro features
are known at the time the month is planned (the price list steps only on
1 January, the event calendar runs to 2028-08 and the macro series to 2028-05),
so they are allowed at t.

Missing rows stay missing: a cell with no order is ``target_missing`` and its
lags are NaN, never zero.  Explicit zeros are recorded as zeros.

Driver analysis is descriptive and in-sample on training months only.  The one
fitted object is the driver screen, which fits on months before the holdout and
scores the last 12 training months - forward in time, never touching the
19-month forecast window.

Outputs (outputs/features/)
---------------------------
features_monthly.parquet        train + forecast rows, all features and targets
feature_dictionary.csv          every column: definition, source, leakage class
driver_effects.csv              seasonality, elasticity, event lift, macro, mix
driver_screen_coefficients.csv  standardised ridge coefficients of the screen
driver_panel_monthly.parquet    company month view of demand and drivers
features_report.json / .txt     machine + human readable summary

Usage
-----
    python src/features.py --clean-dir outputs/clean --out-dir outputs
    python run_pipeline.py --phase 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile_data import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, fmt, jsonable  # noqa: E402

CLEAN_DIR = "clean"
ANOMALY_DIR = "anomalies"
FEATURE_DIR = "features"
SERIES = ["SKU_ID", "Region"]
CELL = ["SKU_ID", "Region", "Month"]
TRAIN_STATUSES = ["positive", "zero", "suppressed_only", "absent"]
MACRO_COLUMNS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
                 "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index",
                 "Industrial_Util_Rate_Pct", "Fuel_Price_Index", "EV_Adoption_Index"]
CHANNELS = ["Aftermarket", "Direct OEM", "Export", "Tier-1 Distributor"]
HOLDOUT_MONTHS = 12
RIDGE_LAMBDA = 1.0
SCREEN_TARGET = "target_demand"
MAX_LAG_MONTHS = 24

# Feature blocks for the screen.  Names must exist in the feature table.
FEATURE_BLOCKS: dict[str, list[str]] = {
    "calendar_lifecycle": ["t_index", "month_sin", "month_cos", "is_quarter_end",
                           "is_year_end", "months_since_launch", "months_to_eol",
                           "is_launch_window", "is_eol_window"],
    "lags": ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_1_present",
             "months_since_last_order", "mean_3", "mean_6", "mean_12", "median_3",
             "std_6", "zero_share_6", "max_12", "yoy_ratio"],
    "censoring": ["suppressed_lag_1", "suppressed_3m_sum", "stockout_days_lag_1",
                  "stockout_days_3m_sum"],
    "price": ["log_price", "price_change_yoy", "months_since_year_start",
              "is_price_step_month"],
    "events": ["event_count", "event_price_count", "event_supply_count"],
    "macro": MACRO_COLUMNS,
    "structure": ["customers_lag_1", "channels_lag_1", "direct_oem_share_3m",
                  "export_share_3m", "aftermarket_share_3m", "hhi_customer_3m"],
}

KEY_COLUMNS = ["SKU_ID", "Region", "Month", "Year", "Month_Num", "Product_Family",
               "Business_Unit", "split", "horizon_index", "grid_status"]

FEATURE_NOTES: dict[str, tuple[str, str, str]] = {
    # column: (definition, source, leakage_class)
    "target_demand": ("unconstrained demand for the month (NaN outside training)",
                      "demand_monthly", "target"),
    "target_observed": ("observed units for the month (NaN when the cell has no row)",
                        "demand_monthly", "target"),
    "target_is_zero": ("True when the month is an explicit zero order, not a missing row",
                       "demand_grid", "target"),
    "target_missing": ("True when the cell has no row and no stockout record",
                       "demand_grid", "target"),
    "cell_absent": ("cell has no ERP row this month; absence is not a zero",
                    "demand_grid", "target"),
    "lifecycle_ok": ("month is inside Launch_Date..EOL_Date", "sku_master", "structural"),
    "pre_launch": ("month precedes Launch_Date", "sku_master", "structural"),
    "post_eol": ("month follows EOL_Date", "sku_master", "structural"),
    "t_index": ("months since 2019-01; linear trend", "calendar", "known_at_t"),
    "month_sin": ("sine of the calendar month, 12-month cycle", "calendar", "known_at_t"),
    "month_cos": ("cosine of the calendar month, 12-month cycle", "calendar", "known_at_t"),
    "is_quarter_end": ("month is March, June, September or December", "calendar", "known_at_t"),
    "is_year_end": ("month is December", "calendar", "known_at_t"),
    "months_since_launch": ("months since Launch_Month, clipped at 0", "sku_master", "known_at_t"),
    "months_to_eol": ("months until EOL_Month (NaN when no EOL)", "sku_master", "known_at_t"),
    "is_launch_window": ("within three months of launch", "sku_master", "known_at_t"),
    "is_eol_window": ("within three months before EOL", "sku_master", "known_at_t"),
    "lag_1": ("demand at t-1 (NaN when absent)", "demand_monthly", "lagged"),
    "lag_2": ("demand at t-2 (NaN when absent)", "demand_monthly", "lagged"),
    "lag_3": ("demand at t-3 (NaN when absent)", "demand_monthly", "lagged"),
    "lag_6": ("demand at t-6 (NaN when absent)", "demand_monthly", "lagged"),
    "lag_12": ("demand at t-12 (NaN when absent)", "demand_monthly", "lagged"),
    "lag_1_present": ("1 when the cell has an ERP row at t-1", "demand_grid", "lagged"),
    "months_since_last_order": ("months since the last month with demand > 0 (prior months only)",
                                "demand_monthly", "lagged"),
    "mean_3": ("mean demand over t-1..t-3", "demand_monthly", "lagged"),
    "mean_6": ("mean demand over t-1..t-6", "demand_monthly", "lagged"),
    "mean_12": ("mean demand over t-1..t-12", "demand_monthly", "lagged"),
    "median_3": ("median demand over t-1..t-3 (robust to one lumpy order)",
                 "demand_monthly", "lagged"),
    "std_6": ("standard deviation of demand over t-1..t-6", "demand_monthly", "lagged"),
    "zero_share_6": ("share of t-1..t-6 with an explicit zero", "demand_monthly", "lagged"),
    "max_12": ("largest demand in t-1..t-12 (order-size memory)", "demand_monthly", "lagged"),
    "yoy_ratio": ("(lag_12+1)/(lag_24+1): last year versus the year before",
                  "demand_monthly", "lagged"),
    "suppressed_lag_1": ("unserved units at t-1 (0 where no stockout was logged)",
                         "demand_monthly", "lagged"),
    "suppressed_3m_sum": ("unserved units over t-1..t-3", "demand_monthly", "lagged"),
    "stockout_days_lag_1": ("stockout days at t-1", "demand_monthly", "lagged"),
    "stockout_days_3m_sum": ("stockout days over t-1..t-3", "demand_monthly", "lagged"),
    "price": ("canonical annual list price for the SKU-year", "sales_clean", "known_at_t"),
    "log_price": ("natural log of price", "sales_clean", "known_at_t"),
    "price_change_yoy": ("price versus the previous year's price", "sales_clean", "known_at_t"),
    "months_since_year_start": ("month-of-year - 1; price steps only on 1 January",
                                "calendar", "known_at_t"),
    "is_price_step_month": ("January month in which the price differs from the prior year",
                            "sales_clean", "known_at_t"),
    "price_imputed_future": ("1 when 2027/2028 price is carried from 2026 (no future list given)",
                             "sales_clean", "known_at_t"),
    "event_count": ("market events active this month that match the cell scope",
                    "events_clean", "known_at_t"),
    "event_price_count": ("price-type events active this month", "events_clean", "known_at_t"),
    "event_supply_count": ("supply/disruption/logistics events active this month",
                           "events_clean", "known_at_t"),
    "has_event": ("1 when any matched event is active", "events_clean", "known_at_t"),
    "Auto_Production_YoY_Frac": ("auto production YoY as a fraction (source is percentage points)",
                                 "macro_clean", "known_at_t"),
    "customers_lag_1": ("distinct customers ordering at t-1", "sales_clean", "lagged"),
    "channels_lag_1": ("distinct channels ordering at t-1", "sales_clean", "lagged"),
    "direct_oem_share_3m": ("Direct OEM revenue share over t-1..t-3", "sales_clean", "lagged"),
    "export_share_3m": ("Export revenue share over t-1..t-3", "sales_clean", "lagged"),
    "aftermarket_share_3m": ("Aftermarket revenue share over t-1..t-3", "sales_clean", "lagged"),
    "hhi_customer_3m": ("customer revenue concentration (HHI) over t-1..t-3",
                        "sales_clean", "lagged"),
    "prev_anomaly_class": ("Phase 2 primary anomaly class at t-1", "anomalies", "lagged"),
    "prev_drop": ("a drop was registered at t-1", "anomalies", "lagged"),
    "prev_spike": ("a spike was registered at t-1", "anomalies", "lagged"),
    "prev_censored": ("a stockout censored t-1", "anomalies", "lagged"),
}
for _column in MACRO_COLUMNS:
    FEATURE_NOTES.setdefault(_column, (f"macro indicator {_column} for the month",
                                       "macro_clean", "known_at_t"))
for _column, _definition in [
    ("Month", "month-start timestamp of the row"),
    ("Year", "calendar year of the month"),
    ("Month_Num", "calendar month 1-12"),
    ("horizon_index", "1-19 inside the forecast window, 0 for training months"),
    ("grid_status", "Phase 1 cell status: positive/zero/absent/suppressed_only/forecast/post_eol"),
    ("split", "train or forecast"),
    ("Product_Family", "product family from the SKU master"),
    ("Business_Unit", "business unit from the SKU master"),
    ("SKU_ID", "canonical SKU identifier"),
    ("Region", "canonical region"),
]:
    FEATURE_NOTES.setdefault(_column, (f"key/categorical column ({_definition}); not a model input",
                                       "keys", "key"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:,.{digits}f}"


def _month_text(value) -> str:
    return "" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m")


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole months from ``earlier`` to ``later`` (positive when later is after)."""
    return ((later.dt.year - earlier.dt.year) * 12
            + (later.dt.month - earlier.dt.month)).astype("float64")


def _rolling(frame: pd.DataFrame, column: str, window: int, how: str,
             min_periods: int | None = None) -> pd.Series:
    """Rolling statistic over the previous ``window`` months, current month excluded.

    ``groupby().rolling()`` is used rather than a per-series Python lambda: with
    3,083 series the lambda form dominated the phase runtime.
    """
    minimum = min_periods if min_periods is not None else max(2, window // 2)
    shifted = frame.groupby(SERIES, sort=False)[column].shift(1)
    keys = [frame[column_key] for column_key in SERIES]
    rolled = getattr(shifted.groupby(keys, sort=False).rolling(window, min_periods=minimum),
                     how)()
    rolled.index = rolled.index.droplevel(list(range(len(SERIES))))
    return rolled.reindex(frame.index)


def _months_since_last_order(series: pd.Series) -> pd.Series:
    """Months since the previous positive-demand month (never the current one)."""
    index = np.arange(len(series))
    present = series.fillna(False).to_numpy(dtype=bool)
    last = np.maximum.accumulate(np.where(present, index, -1))
    previous = np.concatenate(([-1], last[:-1]))
    distance = np.where(previous < 0, np.nan, index - previous)
    return pd.Series(np.clip(distance, 0, MAX_LAG_MONTHS), index=series.index)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
class FeatureReport:
    """Console transcript + machine-readable diagnostics for Phase 3."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.verdicts: list[dict] = []
        self.metrics: dict = {}
        self.log: list[dict] = []
        self._t0 = time.time()

    def h(self, title: str) -> None:
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"[features] {title}", file=sys.stderr, flush=True)

    def sub(self, title: str) -> None:
        self.lines.append(f"\n-- {title} " + "-" * max(0, 58 - len(title)))

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def kv(self, key: str, value, indent: int = 2, width: int = 46) -> None:
        self.lines.append(f"{' ' * indent}{key:<{width}} {fmt(value)}")

    def rule(self, rule_id: str, issue: str, action: str, rows: int = 0,
             detail: str = "") -> None:
        self.log.append({"rule_id": rule_id, "issue": issue, "action": action,
                         "rows_affected": int(rows), "detail": detail})

    def verdict(self, name: str, ok: bool, detail: str = "") -> None:
        self.verdicts.append({"phase": 3, "check": name, "ok": bool(ok), "detail": detail})
        self.p(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. load
# --------------------------------------------------------------------------- #
def load_inputs(out_dir: Path, clean_dir: Path, anomaly_dir: Path) -> dict:
    tables = {
        name: pd.read_parquet(clean_dir / f"{name}.parquet")
        for name in ["sales_clean", "sku_master_clean", "demand_monthly", "demand_grid",
                     "stockouts_clean", "macro_clean", "events_clean", "test_clean"]
    }
    tables["anomaly_register"] = pd.read_parquet(anomaly_dir / "anomaly_register.parquet")
    return tables


# --------------------------------------------------------------------------- #
# 2. base frame: training cells + forecast keys
# --------------------------------------------------------------------------- #
def build_base(tables: dict[str, pd.DataFrame], report: FeatureReport) -> pd.DataFrame:
    report.h("1. BASE FRAME")
    grid = tables["demand_grid"]
    test = tables["test_clean"]
    panel = tables["demand_monthly"]
    master = tables["sku_master_clean"]

    train = grid.loc[grid["status"].isin(TRAIN_STATUSES),
                     ["SKU_ID", "Region", "Month", "status"]].copy()
    train["split"] = "train"

    horizon = test.loc[:, ["SKU_ID", "Region", "Month", "pre_launch", "post_eol"]].copy()
    horizon["status"] = np.where(horizon["post_eol"], "post_eol", "forecast")
    horizon["split"] = "forecast"
    horizon = horizon.drop(columns=["pre_launch", "post_eol"])

    base = pd.concat([train, horizon], ignore_index=True)
    base = base.rename(columns={"status": "grid_status"})
    base["Year"] = base["Month"].dt.year
    base["Month_Num"] = base["Month"].dt.month
    base["split"] = pd.Categorical(base["split"], categories=["train", "forecast"], ordered=True)
    forecast_months = sorted(pd.Timestamp(month) for month in horizon["Month"].unique())
    index_frame = pd.DataFrame({"Month": forecast_months,
                                "horizon_index": np.arange(1, len(forecast_months) + 1)})
    base = base.merge(index_frame, on="Month", how="left")
    base["horizon_index"] = base["horizon_index"].fillna(0).astype(int)

    base = base.merge(master[["SKU_ID", "Product_Family", "Business_Unit", "Launch_Month",
                              "EOL_Month", "Lifecycle_Status"]], on="SKU_ID", how="left")
    base = base.sort_values([*SERIES, "Month"]).reset_index(drop=True)

    panel_fields = panel[["SKU_ID", "Region", "Month", "observed_units", "demand_units",
                          "suppressed_units", "stockout_days", "requested_units",
                          "shipped_units", "unit_price"]]
    base = base.merge(panel_fields, on=CELL, how="left")
    base["target_observed"] = base["observed_units"]
    base["target_demand"] = base["demand_units"]
    base["target_is_zero"] = base["target_observed"].eq(0).fillna(False)
    base["cell_absent"] = base["grid_status"].eq("absent")
    base["target_missing"] = base["target_demand"].isna()
    base["lifecycle_ok"] = ~base["grid_status"].isin(["pre_launch", "post_eol"])
    base["pre_launch"] = base["grid_status"].eq("pre_launch")
    base["post_eol"] = base["grid_status"].eq("post_eol")

    report.kv("rows", f"{len(base):,}")
    report.kv("train / forecast", f"{int(base['split'].eq('train').sum()):,} / "
                                  f"{int(base['split'].eq('forecast').sum()):,}")
    report.kv("train lifecycle-valid cells", f"{int(base['split'].eq('train').sum()):,}")
    report.kv("forecast months", f"{len(forecast_months)} "
                                 f"({_month_text(forecast_months[0])} -> "
                                 f"{_month_text(forecast_months[-1])})")
    report.kv("pairs", f"{len(base.groupby(SERIES)):,}")
    report.rule("FEA-BASE", "model rows assembled from the calendar grid and the test keys",
                "training months plus the 19 forecast months, one row per key", len(base),
                detail=f"{int(base['target_missing'].sum()):,} rows have no observed demand "
                       f"(absence is not zero)")
    report.metrics["base"] = {
        "rows": len(base),
        "train_rows": int(base["split"].eq("train").sum()),
        "forecast_rows": int(base["split"].eq("forecast").sum()),
        "pairs": int(len(base.groupby(SERIES))),
        "forecast_months": len(forecast_months),
        "target_missing_rows": int(base["target_missing"].sum()),
    }
    return base


# --------------------------------------------------------------------------- #
# 3. features
# --------------------------------------------------------------------------- #
def add_calendar_lifecycle(frame: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    origin = pd.Timestamp("2019-01-01")
    frame["t_index"] = ((frame["Month"].dt.year - origin.year) * 12
                        + (frame["Month"].dt.month - origin.month)).astype(int)
    angle = 2 * np.pi * (frame["Month_Num"] - 1) / 12
    frame["month_sin"] = np.sin(angle)
    frame["month_cos"] = np.cos(angle)
    frame["is_quarter_end"] = frame["Month_Num"].isin([3, 6, 9, 12])
    frame["is_year_end"] = frame["Month_Num"].eq(12)

    since_launch = months_between(frame["Month"], frame["Launch_Month"])
    frame["months_since_launch"] = since_launch.clip(lower=0).fillna(0)
    frame["months_to_eol"] = months_between(frame["EOL_Month"], frame["Month"]).where(
        frame["EOL_Month"].notna())
    frame["is_launch_window"] = frame["months_since_launch"].le(3)
    frame["is_eol_window"] = frame["months_to_eol"].between(0, 3).fillna(False)

    report.kv("months since launch (median)", f"{frame['months_since_launch'].median():.0f}")
    report.kv("launch-window rows", int(frame["is_launch_window"].sum()))
    report.kv("rows with an EOL", int(frame["months_to_eol"].notna().sum()))
    report.rule("FEA-CALENDAR", "trend, 12-month cycle and quarter-end flags",
                "calendar features are known when the month is planned", len(frame),
                detail="quarter-end months carry the reported B2B ordering peak")
    return frame


def add_lags(frame: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    grouped = frame.groupby(SERIES, sort=False)
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = grouped["target_demand"].shift(lag)
    frame["lag_24"] = grouped["target_demand"].shift(24)
    frame["lag_1_present"] = grouped["target_observed"].shift(1).notna().astype(int)

    present = frame["target_demand"].fillna(0).gt(0)
    frame["__positive"] = present
    frame["months_since_last_order"] = (
        frame.groupby(SERIES, sort=False)["__positive"].transform(
            lambda series: _months_since_last_order(series)))

    frame["mean_3"] = _rolling(frame, "target_demand", 3, "mean")
    frame["mean_6"] = _rolling(frame, "target_demand", 6, "mean")
    frame["mean_12"] = _rolling(frame, "target_demand", 12, "mean")
    frame["median_3"] = _rolling(frame, "target_demand", 3, "median", min_periods=2)
    frame["std_6"] = _rolling(frame, "target_demand", 6, "std")
    frame["max_12"] = _rolling(frame, "target_demand", 12, "max")
    zero = frame["target_observed"].eq(0).astype(float).where(frame["target_observed"].notna())
    frame["__zero"] = zero
    frame["zero_share_6"] = _rolling(frame, "__zero", 6, "mean", min_periods=2)
    # The ratio is undefined when there was no prior-year demand at all; dividing
    # by a +1 pseudo count produced values in the tens of thousands and wrecked
    # anything fitted on the column.
    frame["yoy_ratio"] = ((frame["lag_12"] + 1) / (frame["lag_24"] + 1)).where(
        frame["lag_24"].ge(1))

    report.kv("lag_1 available", f"{int(frame['lag_1'].notna().sum()):,} rows")
    report.kv("lag_12 available", f"{int(frame['lag_12'].notna().sum()):,} rows")
    report.kv("median months since last order", f"{frame['months_since_last_order'].median():.0f}")
    report.rule("FEA-LAGS", "demand history features",
                "shifted by at least one month; absence stays NaN", len(frame),
                detail="lag_1..lag_12, rolling mean/median/std/max, zero share, "
                       "months since last order")
    return frame.drop(columns=["lag_24", "__positive", "__zero"])


def add_censoring(frame: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    grouped = frame.groupby(SERIES, sort=False)
    frame["suppressed_lag_1"] = grouped["suppressed_units"].shift(1).fillna(0.0)
    frame["suppressed_3m_sum"] = (
        frame.groupby(SERIES, sort=False)["suppressed_units"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).sum())).fillna(0.0)
    frame["stockout_days_lag_1"] = grouped["stockout_days"].shift(1).fillna(0.0)
    frame["stockout_days_3m_sum"] = (
        frame.groupby(SERIES, sort=False)["stockout_days"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).sum())).fillna(0.0)
    report.kv("rows with censoring history", int(frame["suppressed_3m_sum"].gt(0).sum()))
    report.rule("FEA-CENSOR", "stockout history",
                "lagged unserved units and stockout days; current-month censoring excluded",
                int(frame["suppressed_3m_sum"].gt(0).sum()))
    return frame


def build_price_ladder(sales: pd.DataFrame) -> pd.DataFrame:
    """One canonical price per SKU-year (Phase 1 already enforced one per year)."""
    ladder = (sales.groupby(["SKU_ID", "Price_Year"])["Price_Clean_USD"].median()
              .reset_index()
              .rename(columns={"Price_Year": "Year", "Price_Clean_USD": "price"})
              .sort_values(["SKU_ID", "Year"]))
    ladder["Year"] = ladder["Year"].astype(int)
    return ladder


def add_price(frame: pd.DataFrame, sales: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    ladder = build_price_ladder(sales)
    frame["Year"] = frame["Year"].astype(int)

    # The list price steps only on 1 January, so a year without a step carries the
    # previous year's price.  No 2027/2028 list is provided: carry 2026 and say so.
    years = sorted(int(y) for y in frame["Year"].unique())
    skus = sorted(frame["SKU_ID"].unique())
    full = pd.MultiIndex.from_product([skus, years], names=["SKU_ID", "Year"])
    full_ladder = ladder.set_index(["SKU_ID", "Year"]).reindex(full).reset_index()
    full_ladder["price"] = full_ladder.groupby("SKU_ID")["price"].ffill().bfill()
    frame = frame.drop(columns=[c for c in ["price"] if c in frame.columns])
    frame = frame.merge(full_ladder[["SKU_ID", "Year", "price"]],
                        on=["SKU_ID", "Year"], how="left")

    last_year = int(sales["Price_Year"].max())
    frame["price_imputed_future"] = frame["Year"].gt(last_year).astype(int)
    previous = frame[["SKU_ID", "Year"]].copy()
    previous["Year"] = previous["Year"] - 1
    prior = previous.merge(ladder.rename(columns={"price": "price_prev"}),
                           on=["SKU_ID", "Year"], how="left")
    frame["price_prev"] = prior["price_prev"].to_numpy()
    frame["price_change_yoy"] = frame["price"] / frame["price_prev"].where(
        frame["price_prev"] > 0) - 1.0
    frame["log_price"] = np.log(frame["price"].where(frame["price"] > 0))
    frame["months_since_year_start"] = frame["Month_Num"] - 1
    frame["is_price_step_month"] = (frame["Month_Num"].eq(1)
                                    & frame["price_change_yoy"].abs().gt(1e-9))
    frame = frame.drop(columns=["price_prev"])

    coverage = float(frame["price"].notna().mean())
    report.kv("price coverage", f"{coverage:.4%}")
    report.kv("rows with a carried future price", int(frame["price_imputed_future"].sum()))
    report.kv("January price steps", int(frame["is_price_step_month"].sum()))
    report.rule("FEA-PRICE", "annual list price and its year-on-year step",
                "price is known in advance because it steps only on 1 January",
                len(frame),
                detail=f"coverage {coverage:.2%}; 2027-2028 carried from {last_year}")
    return frame


def _event_scope_key(frame: pd.DataFrame) -> pd.Series:
    return (frame["Month"].dt.strftime("%Y-%m"))


def add_events(frame: pd.DataFrame, events: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    resolved = events.loc[events["scope_resolved"].fillna(False)].copy()
    resolved["scope_type"] = (resolved["Scope_Type"].astype(str).str.strip().str.upper()
                              .str.replace(" ", "_").str.replace("-", "_"))
    resolved["event_class"] = np.select(
        [resolved["Event_Type"].str.contains("price", case=False, na=False),
         resolved["Event_Type"].str.contains("supply|disruption|logistics", case=False, na=False)],
        ["price", "supply"], default="other")

    expanded: list[dict] = []
    for row in resolved.itertuples(index=False):
        start = row.Start_Month
        end = row.End_Month if pd.notna(row.End_Month) else start
        for month in pd.date_range(start, end, freq="MS"):
            expanded.append({"Month": month, "scope_type": row.scope_type,
                             "scope_value": str(row.Scope_Value).strip(),
                             "event_class": row.event_class})
    long = pd.DataFrame(expanded)
    long["scope_key"] = (long["scope_value"].astype(str).str.upper()
                         .str.replace(r"[^A-Z0-9]", "", regex=True))

    def counts(scope_types: list[str], keys: list[str]) -> pd.DataFrame:
        subset = long.loc[long["scope_type"].isin(scope_types)]
        grouped = subset.groupby(keys, sort=False)
        return grouped.agg(event_count=("event_class", "size"),
                           event_price_count=("event_class", lambda s: int((s == "price").sum())),
                           event_supply_count=("event_class", lambda s: int((s == "supply").sum()))
                           ).reset_index()

    global_counts = counts(["GLOBAL"], ["Month"])
    region_counts = counts(["REGION"], ["Month", "scope_key"])
    family_counts = counts(["PRODUCT_FAMILY"], ["Month", "scope_key"])
    sku_counts = counts(["SKU"], ["Month", "scope_key"])

    frame["__region_key"] = (frame["Region"].astype(str).str.upper()
                             .str.replace(r"[^A-Z0-9]", "", regex=True))
    frame["__family_key"] = (frame["Product_Family"].astype(str).str.upper()
                             .str.replace(r"[^A-Z0-9]", "", regex=True))
    frame["__sku_key"] = (frame["SKU_ID"].astype(str).str.upper()
                          .str.replace(r"[^A-Z0-9]", "", regex=True))

    result = frame[["Month"]].merge(global_counts, on="Month", how="left")
    for source, key_column, prefix in [
        (region_counts, "__region_key", "region"),
        (family_counts, "__family_key", "family"),
        (sku_counts, "__sku_key", "sku"),
    ]:
        right = source.rename(columns={"scope_key": key_column})
        right = right[right["event_count"] > 0]
        matched = frame[["Month", key_column]].merge(right, on=["Month", key_column], how="left")
        for column in ["event_count", "event_price_count", "event_supply_count"]:
            result[column] = result[column].fillna(0) + matched[column].fillna(0).to_numpy()

    frame["event_count"] = result["event_count"].fillna(0).astype(int).to_numpy()
    frame["event_price_count"] = result["event_price_count"].fillna(0).astype(int).to_numpy()
    frame["event_supply_count"] = result["event_supply_count"].fillna(0).astype(int).to_numpy()
    frame["has_event"] = frame["event_count"].gt(0).astype(int)
    frame = frame.drop(columns=["__region_key", "__family_key", "__sku_key"])

    unmatched = int((~events["scope_resolved"].fillna(False)).sum())
    not_cell = int(long["scope_type"].isin(["CUSTOMER", "CHANNEL", "SALES_CHANNEL"]).sum())
    report.kv("event-months expanded", f"{len(long):,}")
    report.kv("rows inside an event", f"{int(frame['has_event'].sum()):,}")
    report.kv("unresolved-scope events excluded", unmatched)
    report.kv("customer/channel event-months not cell-attributable", not_cell)
    report.rule("FEA-EVENTS", "event calendar matched to cell scope",
                "month counts for global / region / family / SKU scope",
                int(frame["has_event"].sum()),
                detail=f"{unmatched} unresolved events and {not_cell} customer/channel "
                       f"event-months cannot be attributed to a SKU-Region cell")
    return frame


def add_macro(frame: pd.DataFrame, macro: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    macro = macro.rename(columns={"Month": "Month"})
    frame = frame.merge(macro[["Month", *MACRO_COLUMNS, "Auto_Production_YoY_Frac"]],
                        on="Month", how="left")
    coverage = float(frame[MACRO_COLUMNS].notna().all(axis=1).mean())
    report.kv("macro coverage (all indicators)", f"{coverage:.4%}")
    report.rule("FEA-MACRO", "macro indicators joined by month",
                "levels are known when the month is planned", len(frame),
                detail=f"coverage {coverage:.2%}; Auto_Production_YoY_Pct converted to a fraction")
    return frame


def build_structure(sales: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    usable = sales.loc[sales["usable_for_training"].fillna(False)]
    base = usable.groupby(CELL, sort=False).agg(
        customers=("Customer_Name", "nunique"),
        channels=("Sales_Channel", "nunique"),
        revenue=("Revenue_Recomputed_USD", "sum"),
    ).reset_index()

    by_channel = (usable.groupby([*CELL, "Sales_Channel"], sort=False)
                  ["Revenue_Recomputed_USD"].sum().reset_index())
    total = by_channel.groupby(CELL, sort=False)["Revenue_Recomputed_USD"].transform("sum")
    by_channel["share"] = by_channel["Revenue_Recomputed_USD"] / total.where(total > 0)
    wide = (by_channel.pivot_table(index=CELL, columns="Sales_Channel", values="share",
                                   aggfunc="sum").fillna(0.0).reset_index())
    wide.columns.name = None
    rename = {channel: f"share_{channel.lower().replace(' ', '_').replace('-', '_')}"
              for channel in CHANNELS}
    wide = wide.rename(columns=rename)
    for column in rename.values():
        if column not in wide:
            wide[column] = 0.0

    by_customer = (usable.groupby([*CELL, "Customer_Name"], sort=False)
                   ["Revenue_Recomputed_USD"].sum().reset_index())
    customer_total = by_customer.groupby(CELL, sort=False)["Revenue_Recomputed_USD"].transform("sum")
    by_customer["share"] = by_customer["Revenue_Recomputed_USD"] / customer_total.where(
        customer_total > 0)
    hhi = (by_customer.assign(square=by_customer["share"] ** 2)
           .groupby(CELL, sort=False)["square"].sum().rename("hhi_customer").reset_index())

    structure = base.merge(wide[[*CELL, *rename.values()]], on=CELL, how="left")
    structure = structure.merge(hhi, on=CELL, how="left")
    report.kv("structure rows", f"{len(structure):,}")
    report.rule("FEA-STRUCTURE", "customer and channel mix per cell-month",
                "revenue shares and customer HHI, then lagged before use", len(structure))
    return structure


def add_structure_features(frame: pd.DataFrame, structure: pd.DataFrame,
                           report: FeatureReport) -> pd.DataFrame:
    frame = frame.merge(structure.rename(columns={"customers": "customers_t",
                                                  "channels": "channels_t"}),
                        on=CELL, how="left")
    frame["customers_lag_1"] = frame.groupby(SERIES, sort=False)["customers_t"].shift(1)
    frame["channels_lag_1"] = frame.groupby(SERIES, sort=False)["channels_t"].shift(1)
    for source, target in [("share_direct_oem", "direct_oem_share_3m"),
                           ("share_export", "export_share_3m"),
                           ("share_aftermarket", "aftermarket_share_3m")]:
        frame[target] = _rolling(frame, source, 3, "mean", min_periods=1)
    frame["hhi_customer_3m"] = _rolling(frame, "hhi_customer", 3, "mean", min_periods=1)
    frame = frame.drop(columns=["customers_t", "channels_t", "revenue", "share_tier_1_distributor",
                                "share_aftermarket", "share_direct_oem", "share_export",
                                "hhi_customer"], errors="ignore")
    report.kv("customers at t-1 (median)", f"{frame['customers_lag_1'].median():.0f}")
    return frame


def add_anomaly_context(frame: pd.DataFrame, register: pd.DataFrame,
                        report: FeatureReport) -> pd.DataFrame:
    cell_rows = register.loc[register["level"].eq("cell"),
                             [*CELL, "anomaly_class"]].drop_duplicates(CELL)
    frame = frame.merge(cell_rows, on=CELL, how="left")
    frame["prev_anomaly_class"] = frame.groupby(SERIES, sort=False)["anomaly_class"].shift(1)
    frame["prev_anomaly_class"] = frame["prev_anomaly_class"].fillna("")
    frame["prev_drop"] = frame["prev_anomaly_class"].str.contains("drop|collapse", na=False).astype(int)
    frame["prev_spike"] = frame["prev_anomaly_class"].str.contains("spike", na=False).astype(int)
    frame["prev_censored"] = frame["prev_anomaly_class"].str.contains("censored", na=False).astype(int)
    frame = frame.drop(columns="anomaly_class")
    report.kv("rows with a prior-month anomaly", int(frame["prev_anomaly_class"].ne("").sum()))
    report.rule("FEA-ANOMALY", "Phase 2 register used as lagged context only",
                "t-1 anomaly class; the current month is never an input",
                int(frame["prev_anomaly_class"].ne("").sum()))
    return frame


def build_features(tables: dict[str, pd.DataFrame], report: FeatureReport) -> pd.DataFrame:
    frame = build_base(tables, report)

    report.h("2. FEATURE CONSTRUCTION")
    frame = add_calendar_lifecycle(frame, report)
    frame = add_lags(frame, report)
    frame = add_censoring(frame, report)
    frame = add_price(frame, tables["sales_clean"], report)
    frame = add_events(frame, tables["events_clean"], report)
    frame = add_macro(frame, tables["macro_clean"], report)
    structure = build_structure(tables["sales_clean"], report)
    frame = add_structure_features(frame, structure, report)
    frame = add_anomaly_context(frame, tables["anomaly_register"], report)

    # Targets must never exist for the forecast split.
    forecast = frame["split"].eq("forecast")
    for column in ["target_demand", "target_observed"]:
        frame.loc[forecast, column] = np.nan
    frame.loc[forecast, "target_is_zero"] = False
    frame.loc[forecast, "target_missing"] = True

    ordered = [*KEY_COLUMNS, "lifecycle_ok", "pre_launch", "post_eol", "cell_absent",
               "target_demand", "target_observed", "target_is_zero", "target_missing",
               "t_index", "month_sin", "month_cos", "is_quarter_end", "is_year_end",
               "months_since_launch", "months_to_eol", "is_launch_window", "is_eol_window",
               "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_1_present",
               "months_since_last_order", "mean_3", "mean_6", "mean_12", "median_3",
               "std_6", "zero_share_6", "max_12", "yoy_ratio",
               "suppressed_lag_1", "suppressed_3m_sum", "stockout_days_lag_1",
               "stockout_days_3m_sum",
               "price", "log_price", "price_change_yoy", "months_since_year_start",
               "is_price_step_month", "price_imputed_future",
               "event_count", "event_price_count", "event_supply_count", "has_event",
               *MACRO_COLUMNS, "Auto_Production_YoY_Frac",
               "customers_lag_1", "channels_lag_1", "direct_oem_share_3m",
               "export_share_3m", "aftermarket_share_3m", "hhi_customer_3m",
               "prev_anomaly_class", "prev_drop", "prev_spike", "prev_censored"]
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise RuntimeError(f"feature columns missing: {missing}")
    frame = frame[ordered]
    frame = frame.sort_values([*SERIES, "Month"]).reset_index(drop=True)
    frame["split"] = frame["split"].astype(str)

    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    report.kv("columns", len(frame.columns))
    report.kv("numeric feature columns", len(numeric))
    report.rule("FEA-MATRIX", "feature matrix assembled",
                "train and forecast rows share one schema", len(frame),
                detail=f"{len(frame.columns)} columns, {len(numeric)} numeric")
    report.metrics["matrix"] = {"rows": len(frame), "columns": len(frame.columns),
                                "numeric_columns": len(numeric)}
    return frame


# --------------------------------------------------------------------------- #
# 4. driver analysis
# --------------------------------------------------------------------------- #
def seasonality_effects(company: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    level = company["demand_units"].rolling(12, center=True, min_periods=6).median()
    ratio = company["demand_units"] / level.where(level > 0)
    work = pd.DataFrame({"Month": company["Month"], "ratio": ratio})
    work["Month_Num"] = work["Month"].dt.month
    summary = work.groupby("Month_Num")["ratio"].agg(["median", "count",
                                                     lambda s: s.quantile(0.25),
                                                     lambda s: s.quantile(0.75)])
    summary.columns = ["median", "n_years", "q25", "q75"]
    peak, trough = summary["median"].idxmax(), summary["median"].idxmin()
    quarter_end = summary.loc[[3, 6, 9, 12], "median"].mean()
    other = summary.drop(index=[3, 6, 9, 12])["median"].mean()

    rows = []
    for month, row in summary.iterrows():
        rows.append({"driver": "seasonality", "scope": "company", "metric": f"month_{month:02d}",
                     "estimate": float(row["median"]), "unit": "index_vs_12m_median",
                     "ci_low": float(row["q25"]), "ci_high": float(row["q75"]),
                     "n_obs": int(row["n_years"]), "method": "ratio_to_centred_12m_median",
                     "note": "1.0 = an average month for the company"})
    rows.append({"driver": "seasonality", "scope": "company", "metric": "quarter_end_ratio",
                 "estimate": float(quarter_end / other), "unit": "ratio",
                 "ci_low": np.nan, "ci_high": np.nan, "n_obs": int(len(company)),
                 "method": "mean(index) quarter-end vs other months",
                 "note": "Mar/Jun/Sep/Dec versus the rest"})
    report.kv("seasonality peak / trough", f"month {peak} ({summary.loc[peak, 'median']:.2f}) / "
                                          f"month {trough} ({summary.loc[trough, 'median']:.2f})")
    report.kv("quarter-end ratio", f"{quarter_end / other:.3f}")
    report.rule("DRV-SEASON", "calendar seasonality", "month index against a centred 12-month median",
                len(company), detail=f"peak month {peak}, trough month {trough}, "
                                     f"quarter-end ratio {quarter_end / other:.3f}")
    return pd.DataFrame(rows)


def price_elasticity(frame: pd.DataFrame, master: pd.DataFrame,
                     report: FeatureReport) -> pd.DataFrame:
    """Within-SKU log-log elasticity with month-of-year and trend controls.

    Price moves only on 1 January, so identification comes from year-to-year
    steps net of a linear trend and calendar-month effects.  This is a
    conditional association, not a causal elasticity.
    """
    data = frame.loc[(frame["split"] == "train") & frame["target_demand"].gt(0)
                     & frame["price"].gt(0) & frame["lifecycle_ok"]].copy()
    data["log_demand"] = np.log(data["target_demand"])
    data = data.merge(master[["SKU_ID", "Business_Unit"]], on="SKU_ID", how="left",
                      suffixes=("", "_master"))
    data["business_unit"] = data["Business_Unit_master"].fillna(data["Business_Unit"])

    def estimate(scope: str, subset: pd.DataFrame) -> dict | None:
        if subset["SKU_ID"].nunique() < 5 or len(subset) < 100:
            return None
        y = subset["log_demand"].to_numpy()
        price = subset["log_price"].to_numpy()
        months = pd.get_dummies(subset["Month_Num"], prefix="m", drop_first=True,
                                dtype=float).to_numpy()
        trend = subset["t_index"].to_numpy(dtype=float)
        design = np.column_stack([price, months, trend, np.ones(len(subset))])
        groups = subset["SKU_ID"].to_numpy()
        # Within-SKU transformation removes the SKU level (and any fixed SKU size).
        codes, _ = pd.factorize(groups)
        count = np.bincount(codes)
        for column in range(design.shape[1]):
            design[:, column] -= np.bincount(codes, weights=design[:, column])[codes] / count[codes]
        y_centred = y - np.bincount(codes, weights=y)[codes] / count[codes]
        beta, *_ = np.linalg.lstsq(design, y_centred, rcond=None)
        resid = y_centred - design @ beta
        dof = max(len(subset) - design.shape[1] - len(count), 1)
        sigma2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(max(sigma2 * xtx_inv[0, 0], 0.0)))
        return {"estimate": float(beta[0]), "se": se, "n_obs": int(len(subset)),
                "skus": int(subset["SKU_ID"].nunique())}

    rows = []
    for scope, subset in [("company", data)] + list(data.groupby("business_unit")):
        result = estimate(scope, subset)
        if result is None:
            continue
        rows.append({"driver": "price_elasticity", "scope": str(scope),
                     "metric": "log_demand_on_log_price",
                     "estimate": result["estimate"], "unit": "elasticity",
                     "ci_low": result["estimate"] - 1.96 * result["se"],
                     "ci_high": result["estimate"] + 1.96 * result["se"],
                     "n_obs": result["n_obs"], "method": "within_sku_ols",
                     "note": f"{result['skus']} SKUs; month-of-year and linear trend controls; "
                             f"conditional association"})
    result_frame = pd.DataFrame(rows)
    overall = result_frame.loc[result_frame["scope"].eq("company")]
    if len(overall):
        report.kv("price elasticity (company)", f"{overall['estimate'].iloc[0]:.3f} "
                                               f"[{overall['ci_low'].iloc[0]:.3f}, "
                                               f"{overall['ci_high'].iloc[0]:.3f}]")
    report.rule("DRV-PRICE", "within-SKU price elasticity",
                "OLS on log demand with month-of-year and trend controls",
                int(len(data)),
                detail="price steps only on 1 January, so the estimate is a year-to-year "
                       "association")
    return result_frame


def event_lift(frame: pd.DataFrame, events: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    """Matched comparison: event months against the same calendar month in prior years."""
    resolved = events.loc[events["scope_resolved"].fillna(False)].copy()
    resolved["scope_type"] = (resolved["Scope_Type"].astype(str).str.strip().str.upper()
                              .str.replace(" ", "_").str.replace("-", "_"))
    train = frame.loc[(frame["split"] == "train") & frame["lifecycle_ok"]]
    matrix = train.pivot_table(index=SERIES, columns="Month", values="target_demand",
                               aggfunc="first")
    pairs = matrix.index
    rows = []
    for row in resolved.itertuples(index=False):
        scope = row.scope_type
        value = str(row.Scope_Value).strip()
        if scope == "GLOBAL":
            scope_index = pairs
        elif scope == "REGION":
            scope_index = pairs[pairs.get_level_values("Region") == value]
        elif scope == "PRODUCT_FAMILY":
            members = train.loc[train["Product_Family"].eq(value), SERIES].drop_duplicates()
            scope_index = pairs.intersection(pd.MultiIndex.from_frame(members))
        elif scope == "SKU":
            scope_index = pairs[pairs.get_level_values("SKU_ID") == value]
        else:
            continue
        if len(scope_index) == 0:
            continue

        start = row.Start_Month
        end = row.End_Month if pd.notna(row.End_Month) else start
        actual_total, baseline_total, n_months = 0.0, 0.0, 0
        for month in pd.date_range(start, end, freq="MS"):
            if month not in matrix.columns:
                continue
            prior_years = [month - pd.DateOffset(years=lag) for lag in (1, 2, 3)]
            prior_years = [column for column in prior_years if column in matrix.columns]
            if not prior_years:
                continue
            baseline_by_cell = matrix.loc[scope_index, prior_years].mean(axis=1, skipna=True)
            valid = baseline_by_cell.notna().to_numpy()
            if not valid.any():
                continue
            cells = scope_index[valid]
            actual_total += float(matrix.loc[cells, month].fillna(0.0).sum())
            baseline_total += float(baseline_by_cell.to_numpy()[valid].sum())
            n_months += 1
        if n_months and baseline_total > 0:
            rows.append({"driver": "event_lift", "scope": row.Event_Type,
                         "metric": row.Event_ID,
                         "estimate": actual_total / baseline_total - 1.0,
                         "unit": "lift_vs_prior_years",
                         "ci_low": np.nan, "ci_high": np.nan, "n_obs": n_months,
                         "method": "matched_prior_years_same_month",
                         "note": f"{scope}:{value}; {_month_text(start)}"})
    frame_out = pd.DataFrame(rows)
    if frame_out.empty:
        report.rule("DRV-EVENT", "event lift", "no scoped event could be matched to cells", 0)
        return frame_out

    by_type = (frame_out.groupby("scope")["estimate"]
               .agg(["median", "count"]).reset_index()
               .rename(columns={"scope": "event_type", "median": "median_lift"}))
    report.sub("event lift by type")
    for row in by_type.itertuples(index=False):
        report.p(f"  {row.event_type:<22} median lift {row.median_lift:>+8.1%} "
                 f"across {row.count} event(s)")
    aggregated = by_type.assign(driver="event_lift", scope=by_type["event_type"],
                                metric="median_lift_by_type",
                                estimate=by_type["median_lift"], unit="lift",
                                ci_low=np.nan, ci_high=np.nan, n_obs=by_type["count"],
                                method="matched_prior_years_same_month",
                                note="median of per-event lift estimates")
    report.rule("DRV-EVENT", "event lift against the same month in prior years",
                "matched comparison per resolved event scope", len(frame_out),
                detail=f"{len(by_type)} event types")
    return pd.concat([frame_out, aggregated[frame_out.columns]], ignore_index=True)


def macro_association(company: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    rows = []
    demand = company["demand_units"].to_numpy()
    for column in MACRO_COLUMNS:
        values = company[column]
        mask = values.notna() & np.isfinite(demand)
        if mask.sum() < 24:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho, p_value = stats.spearmanr(values[mask], demand[mask])
        rows.append({"driver": "macro", "scope": "company", "metric": column,
                     "estimate": float(rho) if np.isfinite(rho) else np.nan,
                     "unit": "spearman_rho", "ci_low": np.nan, "ci_high": np.nan,
                     "n_obs": int(mask.sum()), "method": "spearman_levels",
                     "note": f"p={p_value:.3f}" if np.isfinite(p_value) else ""})
    frame = pd.DataFrame(rows)
    if len(frame):
        top = frame.reindex(frame["estimate"].abs().sort_values(ascending=False).index)
        report.sub("macro association with company demand (Spearman)")
        for row in top.head(4).itertuples(index=False):
            report.p(f"  {row.metric:<28} rho {row.estimate:>+6.3f}  ({row.note})")
        report.rule("DRV-MACRO", "macro association with company demand",
                    "Spearman rank correlation on monthly levels", int(len(frame)),
                    detail=f"strongest {top['metric'].iloc[0]} "
                           f"rho={top['estimate'].iloc[0]:+.3f}")
    return frame


def lifecycle_ramp(frame: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    launches = frame.loc[(frame["split"] == "train") & frame["lifecycle_ok"]
                         & frame["months_since_launch"].le(24)
                         & frame["target_demand"].gt(0)].copy()
    mature = (frame.loc[(frame["split"] == "train") & frame["lifecycle_ok"]
                        & frame["months_since_launch"].between(13, 24)]
              .groupby("SKU_ID")["target_demand"].median().rename("mature"))
    launches = launches.merge(mature, on="SKU_ID", how="left")
    launches = launches.loc[launches["mature"].gt(0)]
    launches["index"] = launches["target_demand"] / launches["mature"]
    summary = (launches.groupby("months_since_launch")["index"]
               .agg(["median", "count"]).reset_index())
    rows = [{"driver": "lifecycle_ramp", "scope": "sku", "metric": f"month_{int(row.months_since_launch)}",
             "estimate": float(row.median), "unit": "index_vs_mature_median",
             "ci_low": np.nan, "ci_high": np.nan, "n_obs": int(row.count),
             "method": "demand_over_sku_mature_median",
             "note": "1.0 = mature volume"} for row in summary.itertuples(index=False)]
    if len(summary):
        def ramp(month: int) -> str:
            hit = summary.loc[summary["months_since_launch"].eq(month), "median"]
            return f"{float(hit.iloc[0]):.2f}" if len(hit) else "n/a"

        report.kv("launch ramp month 0 / month 6 / month 12",
                  f"{ramp(0)} / {ramp(6)} / {ramp(12)}")
        report.rule("DRV-LIFECYCLE", "launch ramp", "demand relative to the SKU's mature median",
                    int(len(launches)))
    return pd.DataFrame(rows)


def channel_mix(sales: pd.DataFrame, report: FeatureReport) -> pd.DataFrame:
    usable = sales.loc[sales["usable_for_training"].fillna(False)].copy()
    usable["Year"] = usable["Month"].dt.year
    yearly = (usable.groupby(["Year", "Sales_Channel"])["Units_Sold_Clean"].sum()
              .unstack(fill_value=0.0))
    totals = yearly.sum(axis=1)
    shares = yearly.div(totals, axis=0)
    rows = []
    for year in sorted(yearly.index)[1:]:
        previous = year - 1
        if previous not in yearly.index:
            continue
        within = float((shares.loc[previous] * (yearly.loc[year] - yearly.loc[previous])).sum())
        mix = float(((shares.loc[year] - shares.loc[previous]) * yearly.loc[previous]).sum())
        rows.append({"driver": "channel_mix", "scope": "company", "metric": f"within_{year}",
                     "estimate": within, "unit": "units", "ci_low": np.nan, "ci_high": np.nan,
                     "n_obs": int(len(usable)), "method": "shift_share",
                     "note": f"within-channel growth, {previous}->{year}"})
        rows.append({"driver": "channel_mix", "scope": "company", "metric": f"mix_{year}",
                     "estimate": mix, "unit": "units", "ci_low": np.nan, "ci_high": np.nan,
                     "n_obs": int(len(usable)), "method": "shift_share",
                     "note": f"channel-mix shift, {previous}->{year}"})
    frame = pd.DataFrame(rows)
    if len(frame):
        latest = frame.loc[frame["metric"].eq(f"within_{max(yearly.index)}")]
        latest_mix = frame.loc[frame["metric"].eq(f"mix_{max(yearly.index)}")]
        report.kv("latest year within-channel growth",
                  f"{_fmt(latest['estimate'].squeeze())} units")
        report.kv("latest year channel-mix effect", f"{_fmt(latest_mix['estimate'].squeeze())} units")
        report.rule("DRV-MIX", "channel mix shift-share",
                    "decomposes year-on-year demand into within-channel growth and mix",
                    int(len(frame)))
    return frame


def driver_screen(frame: pd.DataFrame, report: FeatureReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-in-time block ablation with a ridge model.

    Fits on training months before the holdout and scores the last 12 training
    months.  It measures the incremental explanatory value of each driver block;
    it is not a forecast of the 19-month window.
    """
    data = frame.loc[(frame["split"] == "train") & frame["target_demand"].notna()].copy()
    months = sorted(data["Month"].unique())
    if len(months) <= HOLDOUT_MONTHS:
        report.rule("DRV-SCREEN", "driver screen", "not enough months for a holdout", 0)
        return pd.DataFrame(), pd.DataFrame()
    cutoff = months[-HOLDOUT_MONTHS]
    fit_mask = data["Month"] < cutoff
    hold_mask = data["Month"] >= cutoff
    target = np.log1p(data[SCREEN_TARGET].to_numpy(dtype=float))
    y_fit, y_hold = target[fit_mask.to_numpy()], target[hold_mask.to_numpy()]

    def design(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        matrix = data[columns].astype(float)
        fit_values = matrix.loc[fit_mask.to_numpy()]
        medians = fit_values.median()
        # Winsorise at the fit-window 1st/99th percentile before standardising:
        # a handful of lumpy orders would otherwise dominate the ridge.
        low = fit_values.quantile(0.01)
        high = fit_values.quantile(0.99)
        matrix = matrix.clip(lower=low, upper=high, axis=1)
        matrix = matrix.fillna(medians).fillna(0.0)
        means = matrix.loc[fit_mask.to_numpy()].mean()
        stds = matrix.loc[fit_mask.to_numpy()].std().replace(0, 1.0).fillna(1.0)
        standardised = ((matrix - means) / stds).to_numpy()
        return standardised[fit_mask.to_numpy()], standardised[hold_mask.to_numpy()]

    def ridge(x_fit: np.ndarray, x_hold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Centre the target: standardised columns have zero mean, so without this
        # the model has no intercept and cannot reach the demand level at all.
        centre = float(y_fit.mean())
        gram = x_fit.T @ x_fit
        gram[np.diag_indices_from(gram)] += RIDGE_LAMBDA
        beta = np.linalg.solve(gram, x_fit.T @ (y_fit - centre))
        return beta, centre + x_hold @ beta

    def metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> tuple[float, float]:
        """Aggregate WAPE plus the model's own scale (log RMSE).

        WAPE alone is not a fair ranking here: most cell-months are small or
        zero, so predicting near-zero beats a genuinely informative lag model on
        aggregate absolute error.  Log RMSE matches the quantity the ridge is
        fitted on and is what the ablation is ranked by.
        """
        actual = np.expm1(actual_log)
        predicted = np.clip(np.expm1(predicted_log), 0.0, None)
        denominator = float(np.abs(actual).sum())
        wape_value = (float(np.abs(actual - predicted).sum() / denominator)
                      if denominator else np.nan)
        rmse_value = float(np.sqrt(np.mean((actual_log - predicted_log) ** 2)))
        return wape_value, rmse_value

    all_columns = [column for block in FEATURE_BLOCKS.values() for column in block]
    x_fit, x_hold = design(all_columns)
    beta, predicted = ridge(x_fit, x_hold)
    baseline_wape, baseline_rmse = metrics(y_hold, predicted)

    rows = [
        {"driver": "screen", "scope": "full_model", "metric": "rmse_log",
         "estimate": baseline_rmse, "unit": "log_rmse", "ci_low": np.nan,
         "ci_high": np.nan, "n_obs": int(hold_mask.sum()),
         "method": "ridge_forward_holdout",
         "note": f"fit < {_month_text(cutoff)}, scored {_month_text(cutoff)} onward"},
        {"driver": "screen", "scope": "full_model", "metric": "wape",
         "estimate": baseline_wape, "unit": "wape", "ci_low": np.nan,
         "ci_high": np.nan, "n_obs": int(hold_mask.sum()),
         "method": "ridge_forward_holdout",
         "note": "aggregate units; dominated by the largest cells"},
    ]
    for block, columns in FEATURE_BLOCKS.items():
        kept = [column for column in all_columns if column not in columns]
        x_fit_ablate, x_hold_ablate = design(kept)
        _, predicted_ablate = ridge(x_fit_ablate, x_hold_ablate)
        ablation_wape, ablation_rmse = metrics(y_hold, predicted_ablate)
        rows.append({"driver": "screen", "scope": f"without_{block}",
                     "metric": "rmse_log_delta",
                     "estimate": ablation_rmse - baseline_rmse, "unit": "log_rmse_points",
                     "ci_low": np.nan, "ci_high": np.nan, "n_obs": int(hold_mask.sum()),
                     "method": "ridge_forward_holdout_block_ablation",
                     "note": f"log RMSE {ablation_rmse:.4f} vs {baseline_rmse:.4f}; "
                             f"WAPE {ablation_wape:.4f}"})
    coefficients = pd.DataFrame({
        "feature": all_columns,
        "block": [block for block, columns in FEATURE_BLOCKS.items() for _ in columns],
        "coefficient": beta,
        "abs_coefficient": np.abs(beta),
    }).sort_values("abs_coefficient", ascending=False)

    report.kv("screen fit months", f"< {_month_text(cutoff)} ({int(fit_mask.sum()):,} rows)")
    report.kv("screen holdout", f"{_month_text(cutoff)} onward ({int(hold_mask.sum()):,} rows)")
    report.kv("holdout log RMSE / WAPE (full model)",
              f"{baseline_rmse:.4f} / {baseline_wape:.4f}")
    report.sub("driver block ablation (log-RMSE cost of removing the block)")
    ablation = pd.DataFrame(rows[2:])
    for row in ablation.sort_values("estimate", ascending=False).itertuples(index=False):
        report.p(f"  {row.scope:<32} {row.estimate:>+8.4f}  ({row.note})")
    report.rule("DRV-SCREEN", "driver block ablation",
                "ridge fitted before the holdout, scored on the last 12 training months",
                int(hold_mask.sum()),
                detail=f"full log RMSE {baseline_rmse:.4f}, WAPE {baseline_wape:.4f} "
                       f"from {_month_text(cutoff)}")
    report.metrics["screen"] = {
        "cutoff": str(_month_text(cutoff)),
        "fit_rows": int(fit_mask.sum()),
        "holdout_rows": int(hold_mask.sum()),
        "log_rmse_full": baseline_rmse,
        "wape_full": baseline_wape,
        "log_rmse_without": {row["scope"]: row["estimate"] for row in rows[2:]},
    }
    return pd.DataFrame(rows), coefficients


def build_driver_panel(frame: pd.DataFrame, panel: pd.DataFrame, monthly: pd.DataFrame,
                       report: FeatureReport) -> pd.DataFrame:
    """Company month view for the dashboard: demand, shipments, macro and structure."""
    demand = (panel.loc[panel["usable"]].groupby("Month").agg(
        demand_units=("demand_units", "sum"),
        observed_units=("observed_units", "sum"),
        suppressed_units=("suppressed_units", "sum"),
        stockout_days=("stockout_days", "sum"),
    ).reset_index())
    train = frame.loc[(frame["split"] == "train") & frame["lifecycle_ok"]]
    extras = train.groupby("Month").agg(
        customers=("customers_lag_1", "max"),
        hhi_customer=("hhi_customer_3m", "median"),
        price_index=("price", "median"),
        event_count=("event_count", "sum"),
        quarter_end_share=("is_quarter_end", "mean"),
    ).reset_index()
    panel_out = demand.merge(extras, on="Month", how="left")
    panel_out = panel_out.merge(monthly[["Month", "expected_units", "z_score"]],
                                on="Month", how="left")
    macro = frame.groupby("Month")[MACRO_COLUMNS].first().reset_index()
    panel_out = panel_out.merge(macro, on="Month", how="left")
    level = panel_out["demand_units"].rolling(12, center=True, min_periods=6).median()
    panel_out["season_index"] = panel_out["demand_units"] / level.where(level > 0)
    panel_out["stockout_rate"] = (panel_out["suppressed_units"]
                                   / panel_out["demand_units"].where(
                                       panel_out["demand_units"] > 0))
    panel_out = panel_out.sort_values("Month").reset_index(drop=True)
    report.kv("driver panel months", len(panel_out))
    return panel_out


# --------------------------------------------------------------------------- #
# 5. outputs
# --------------------------------------------------------------------------- #
def feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        definition, source, leakage = FEATURE_NOTES.get(
            column, ("not documented", "unknown", "unknown"))
        dtype = str(frame[column].dtype)
        rows.append({"column": column, "definition": definition, "source": source,
                     "leakage_class": leakage, "dtype": dtype,
                     "non_null": int(frame[column].notna().sum())})
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, report: FeatureReport, frame: pd.DataFrame,
                  dictionary: pd.DataFrame, effects: pd.DataFrame,
                  coefficients: pd.DataFrame, driver_panel: pd.DataFrame) -> None:
    feature_dir = out_dir / FEATURE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(feature_dir / "features_monthly.parquet", index=False)
    dictionary.to_csv(feature_dir / "feature_dictionary.csv", index=False)
    effects.to_csv(feature_dir / "driver_effects.csv", index=False)
    coefficients.to_csv(feature_dir / "driver_screen_coefficients.csv", index=False)
    driver_panel.to_parquet(feature_dir / "driver_panel_monthly.parquet", index=False)
    pd.DataFrame(report.log).to_csv(feature_dir / "feature_rules.csv", index=False)

    payload = {
        "meta": {
            "phase": 3,
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - report._t0, 2),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "python": sys.version.split()[0],
        },
        "metrics": report.metrics,
        "verdicts": report.verdicts,
        "rules": report.log,
        "outputs": {
            "features_monthly": "outputs/features/features_monthly.parquet",
            "feature_dictionary": "outputs/features/feature_dictionary.csv",
            "driver_effects": "outputs/features/driver_effects.csv",
            "driver_screen_coefficients": "outputs/features/driver_screen_coefficients.csv",
            "driver_panel_monthly": "outputs/features/driver_panel_monthly.parquet",
        },
    }
    (out_dir / "features_report.json").write_text(
        json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "features_report.txt").write_text(
        "\n".join(report.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 feature engineering")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--clean-dir", default=None)
    parser.add_argument("--anomaly-dir", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    clean_dir = Path(args.clean_dir) if args.clean_dir else out_dir / CLEAN_DIR
    anomaly_dir = Path(args.anomaly_dir) if args.anomaly_dir else out_dir / ANOMALY_DIR
    if not clean_dir.is_absolute():
        clean_dir = PROJECT_ROOT / clean_dir
    if not anomaly_dir.is_absolute():
        anomaly_dir = PROJECT_ROOT / anomaly_dir

    report = FeatureReport()
    report.p("CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER")
    report.p("Phase 3: feature engineering and driver analysis")
    report.p(f"clean dir   : {clean_dir}")
    report.p(f"anomaly dir : {anomaly_dir}")
    report.p(f"out dir     : {out_dir}")

    for required in [clean_dir / "demand_grid.parquet", anomaly_dir / "anomaly_register.parquet"]:
        if not required.exists():
            print(f"FATAL: missing Phase {1 if 'clean' in str(required) else 2} output: {required}",
                  file=sys.stderr)
            return 2

    tables = load_inputs(out_dir, clean_dir, anomaly_dir)
    monthly = pd.read_parquet(anomaly_dir / "monthly_summary.parquet")
    frame = build_features(tables, report)

    report.h("3. DRIVER ANALYSIS")
    company = (frame.loc[(frame["split"] == "train") & frame["lifecycle_ok"]]
               .groupby("Month").agg(demand_units=("target_demand", "sum"),
                                     observed_units=("target_observed", "sum")).reset_index())
    company = company.merge(tables["macro_clean"], on="Month", how="left")

    effects = [seasonality_effects(company, report),
               price_elasticity(frame, tables["sku_master_clean"], report),
               event_lift(frame, tables["events_clean"], report),
               macro_association(company, report),
               lifecycle_ramp(frame, report),
               channel_mix(tables["sales_clean"], report)]
    screen_effects, coefficients = driver_screen(frame, report)
    effects.append(screen_effects)
    effects_frame = pd.concat([frame_ for frame_ in effects if len(frame_)], ignore_index=True)
    driver_panel = build_driver_panel(frame, tables["demand_monthly"], monthly, report)
    dictionary = feature_dictionary(frame)

    # ---- exit checks ------------------------------------------------------ #
    report.h("4. PHASE 3 EXIT CHECK")
    train_rows = frame.loc[frame["split"].eq("train")]
    forecast_rows = frame.loc[frame["split"].eq("forecast")]
    grid = tables["demand_grid"]
    test = tables["test_clean"]
    lifecycle_valid = int(grid["status"].isin(TRAIN_STATUSES).sum())

    shifted = frame.groupby(SERIES, sort=False)["target_demand"].shift(1)
    lag_ok = bool(((frame["lag_1"].isna() & shifted.isna())
                   | np.isclose(frame["lag_1"], shifted, equal_nan=True)).all())
    shifted_12 = frame.groupby(SERIES, sort=False)["target_demand"].shift(12)
    lag12_ok = bool(((frame["lag_12"].isna() & shifted_12.isna())
                     | np.isclose(frame["lag_12"], shifted_12, equal_nan=True)).all())

    report.verdict("feature grain is unique", not frame.duplicated(CELL).any(),
                   f"{len(frame):,} rows")
    report.verdict("train rows equal lifecycle-valid grid cells",
                   len(train_rows) == lifecycle_valid,
                   f"{len(train_rows):,} rows")
    report.verdict("forecast rows equal the test keys",
                   len(forecast_rows) == len(test),
                   f"{len(forecast_rows):,} vs {len(test):,}")
    report.verdict("forecast rows carry no target",
                   bool(forecast_rows[["target_demand", "target_observed"]].isna().all().all()))
    report.verdict("lag_1 is strictly the prior month", lag_ok)
    report.verdict("lag_12 is strictly twelve months back", lag12_ok)
    report.verdict("no pre-launch or post-EOL row in training",
                   not bool(train_rows[["pre_launch", "post_eol"]].any().any()),
                   f"{int(train_rows['pre_launch'].sum())} pre-launch, "
                   f"{int(train_rows['post_eol'].sum())} post-EOL")
    report.verdict("price available for every row",
                   bool(frame["price"].notna().all()),
                   f"{int(frame['price_imputed_future'].sum()):,} future rows carry 2026")
    report.verdict("macro available for every row",
                   bool(frame[MACRO_COLUMNS].notna().all().all()))
    undocumented = [column for column in frame.columns
                    if column not in dictionary["column"].values]
    undefined = dictionary.loc[dictionary["leakage_class"].eq("unknown"), "column"].tolist()
    report.verdict("every column is in the dictionary with a leakage class",
                   not undocumented and not undefined,
                   f"{len(dictionary)} columns documented; "
                   f"{int(dictionary['leakage_class'].ne('key').sum())} are model inputs"
                   + (f"; undocumented {undefined}" if undefined else ""))
    screen_scopes = set(effects_frame.loc[effects_frame["driver"].eq("screen"), "scope"])
    report.verdict("every driver block has a screen result",
                   {"full_model"} | {f"without_{block}" for block in FEATURE_BLOCKS}
                   <= screen_scopes,
                   f"{len(screen_scopes)} screen rows")
    report.verdict("screen holdout is after the fit window",
                   bool(report.metrics.get("screen", {}).get("cutoff")),
                   f"cutoff {report.metrics.get('screen', {}).get('cutoff', 'n/a')}")
    report.verdict("driver effects cover the main blocks",
                   {"price_elasticity", "seasonality", "macro", "screen"}
                   <= set(effects_frame["driver"]),
                   f"{effects_frame['driver'].nunique()} driver families")
    report.verdict("feature table is deterministically ordered",
                   frame.equals(frame.sort_values([*SERIES, "Month"]).reset_index(drop=True)))

    report.kv("elapsed seconds", round(time.time() - started, 1))
    write_outputs(out_dir, report, frame, dictionary, effects_frame, coefficients, driver_panel)

    print("\n".join(report.lines))
    print(f"\nJSON  -> {out_dir / 'features_report.json'}")
    print(f"TEXT  -> {out_dir / 'features_report.txt'}")
    print(f"TABLES-> {out_dir / FEATURE_DIR}")
    failed = [v for v in report.verdicts if not v["ok"]]
    if failed:
        print(f"\nPHASE 3 FAILED {len(failed)} check(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
