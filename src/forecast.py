#!/usr/bin/env python
"""Phase 4 - Forecasting, forward-in-time validation and submission.

What is forecast
----------------
Unconstrained **demand** in units, per (SKU, Region, Month), for the 19 months
2026-11 -> 2028-05.  Demand, not shipments: the Phase 2 panel already restores
units the stockout log proves were requested but not served, so a stockout month
teaches the model real demand instead of teaching it the shortage.

Model
-----
A hurdle model, because these series are intermittent (a third of the cell-months
are explicit zeros and another slice has no order at all):

    E[units] = P(a positive order | features) x E[units | the order happens]

Both parts are global (one model for all 3,083 series, no per-series fitting) and
both are fitted on ``log1p`` / logit scale so a handful of 13,000-unit orders
cannot dominate.  Two engines are fitted and compared:

* ``hurdle_ridge`` - ridge-regularised logistic + linear ridge (closed form,
  deterministic, no library randomness);
* ``hurdle_lgbm``  - LightGBM classifier + regressor on the same design.

Two baselines are scored alongside them: seasonal naive (same month last year,
falling back to the trailing mean) and a flat 3-month moving average.

The model that ships is a convex blend of the seasonal baseline and the tree
hurdle.  Its weight is a free parameter, so it is chosen **before** the scored
holdout: an inner 19-month window inside the fit period sweeps 0.0-1.0 and the
winner is carried unchanged into the outer run.  The holdout therefore scores a
weight it had no hand in picking; the outer weight curve is reported as a
diagnostic only, never used to re-select.

Why the recursion
-----------------
Only the first forecast month has real lags.  Month 2 needs month 1's demand,
which does not exist, so the path is built **recursively**: each month's
prediction is written back into the demand history before the next month's lag
features are computed.  The same code path produces the backtest and the
submission, so the validation measures the model that actually ships.

Validation
----------
Forward in time only, and at the true horizon: fit on months before 2025-04,
forecast 19 months recursively, score against actuals for 2025-04..2026-10.  No
validation or test target is ever used for fitting; contributions of the horizon
months are excluded from every fit.  The final model is then refitted on all 94
training months and produces the submission.

Post-EOL keys
-------------
156 test keys belong to SKUs after their EOL month.  The Phase 1 ledger confirms
the ERP records no sales after EOL, so those keys are forecast at exactly 0
rather than extrapolated.  This is recorded as a rule, not a silent clip.

Outputs
-------
outputs/submission.csv                        Date,SKU_ID,Region,Units_Sold_Forecast
outputs/forecast/forecast_cell_month.parquet  every test key, every candidate, final + bands
                             and the list price with its revenue
outputs/forecast/backtest_scores.csv          candidate x metric
outputs/forecast/backtest_slices.csv          selected model error by family, region,
                             business unit and months ahead
outputs/forecast/backtest_cell_month.parquet  holdout predictions per key
outputs/forecast/backtest_monthly.parquet     company month: actual vs candidates
outputs/forecast/forecast_monthly_summary.parquet  company month forecast + band
outputs/forecast/model_diagnostics.json       weights, seeds, importances
outputs/forecast_report.json / .txt           machine + human readable summary

Usage
-----
    python src/forecast.py --features-dir outputs/features --out-dir outputs
    python run_pipeline.py --phase 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile_data import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, fmt, jsonable  # noqa: E402

FEATURE_DIR = "features"
FORECAST_DIR = "forecast"
SERIES = ["SKU_ID", "Region"]
CELL = ["SKU_ID", "Region", "Month"]
MONTH_ORIGIN = pd.Timestamp("2019-01-01")
HORIZON_MONTHS = 19
BACKTEST_MONTHS = 19
SEED = 20261125
RIDGE_LAMBDA = 3.0
LOGISTIC_ITERATIONS = 50
BIAS_TOLERANCE = 0.15
Z80 = 1.2816
MACRO_COLUMNS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
                 "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index",
                 "Industrial_Util_Rate_Pct", "Fuel_Price_Index", "EV_Adoption_Index"]

# Known when the month is planned - read straight from the Phase 3 table.
KNOWN_FEATURES = [
    "t_index", "month_sin", "month_cos", "is_quarter_end", "is_year_end",
    "months_since_launch", "months_to_eol", "is_launch_window", "is_eol_window",
    "log_price", "price_change_yoy", "months_since_year_start", "is_price_step_month",
    "event_count", "event_price_count", "event_supply_count",
    *MACRO_COLUMNS,
]
CATEGORICAL_FEATURES = ["Product_Family", "Business_Unit", "Region"]

# Recomputed recursively - never read from the feature table for horizon months.
LAG_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "mean_3", "mean_6", "mean_12", "median_3", "std_6", "max_12",
    "zero_share_6", "months_since_last_order", "yoy_ratio", "cell_mean",
]
PARITY_LAG_FEATURES = [column for column in LAG_FEATURES if column != "cell_mean"]

LGBM_PARAMS = dict(
    n_estimators=260, learning_rate=0.06, num_leaves=31, min_child_samples=40,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    max_bin=127, n_jobs=4, random_state=SEED, deterministic=True,
    force_row_wise=True, verbose=-1,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _month_text(value) -> str:
    return "" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|a-p| / sum(a), unit-weighted."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = np.abs(actual).sum()
    return _safe_divide(np.abs(actual - predicted).sum(), total)


def cohort_wape(frame: pd.DataFrame, column: str, by: str | None = None) -> pd.DataFrame:
    grouped = frame if by is None else frame.groupby(by, dropna=False)
    rows = []
    for key, block in ([(None, frame)] if by is None else grouped):
        rows.append({"segment": "ALL" if key is None else key,
                     "wape": wape(block["actual"], block[column]),
                     "bias": _safe_divide(block[column].sum(), block["actual"].sum()) - 1.0,
                     "actual": block["actual"].sum(), "forecast": block[column].sum()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
class ForecastReport:
    """Console transcript + machine-readable diagnostics for Phase 4."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.verdicts: list[dict] = []
        self.metrics: dict = {}
        self.log: list[dict] = []
        self._t0 = time.time()

    def h(self, title: str) -> None:
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"[forecast] {title}", file=sys.stderr, flush=True)

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
        self.verdicts.append({"phase": 4, "check": name, "ok": bool(ok), "detail": detail})
        self.p(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. load
# --------------------------------------------------------------------------- #
def load_inputs(out_dir: Path, feature_dir: Path, data_dir: Path | None = None) -> dict:
    clean_dir = out_dir / "clean"
    data_dir = Path(data_dir) if data_dir is not None else Path(DEFAULT_DATA_DIR)
    tables = {
        "frame": pd.read_parquet(feature_dir / "features_monthly.parquet"),
        "dictionary": pd.read_csv(feature_dir / "feature_dictionary.csv"),
        "demand_monthly": pd.read_parquet(clean_dir / "demand_monthly.parquet"),
        "test_clean": pd.read_parquet(clean_dir / "test_clean.parquet"),
        "sku_master": pd.read_parquet(clean_dir / "sku_master_clean.parquet"),
    }
    sample = data_dir / "sample_submission.csv"
    tables["submission_columns"] = (pd.read_csv(sample, nrows=1).columns.tolist()
                                    if sample.exists() else
                                    ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"])
    return tables


# --------------------------------------------------------------------------- #
# 2. panel: demand history matrix + known-at-t design
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Demand history on a pairs x months grid, plus the known-at-t design."""

    months: pd.DatetimeIndex
    pairs: pd.DataFrame
    demand: np.ndarray
    zero: np.ndarray
    known: pd.DataFrame
    design: np.ndarray
    feature_names: list[str]
    pair_code: np.ndarray
    position: np.ndarray
    lags: np.ndarray = None
    rows_by_position: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.known)


def encode_categoricals(known: pd.DataFrame, categories: dict[str, list]) -> tuple[np.ndarray, list[str]]:
    """Fixed-width one-hot block, built by hand so the column set never drifts."""
    blocks, names = [], []
    for column in CATEGORICAL_FEATURES:
        levels = categories[column]
        lookup = {level: index for index, level in enumerate(levels)}
        codes = known[column].map(lookup)
        block = np.zeros((len(known), len(levels)))
        valid = codes.notna().to_numpy()
        block[np.flatnonzero(valid), codes.to_numpy()[valid].astype(int)] = 1.0
        blocks.append(block)
        names.extend(f"{column}={level}" for level in levels)
    return np.hstack(blocks), names


def build_panel(tables: dict, report: ForecastReport) -> Panel:
    report.h("1. PANEL AND DEMAND HISTORY")
    frame = tables["frame"].copy()
    panel = tables["demand_monthly"]
    test = tables["test_clean"]

    frame["Month"] = pd.to_datetime(frame["Month"])
    train_months = sorted(frame.loc[frame["split"].eq("train"), "Month"].unique())
    horizon_months = sorted(frame.loc[frame["split"].eq("forecast"), "Month"].unique())
    months = pd.date_range(frame["Month"].min(), frame["Month"].max(), freq="MS")
    position_of = {month: index for index, month in enumerate(months)}

    pairs = frame[SERIES].drop_duplicates().sort_values(SERIES).reset_index(drop=True)
    pair_index = pd.MultiIndex.from_frame(pairs)
    frame["__pair"] = pair_index.get_indexer(pd.MultiIndex.from_frame(frame[SERIES]))
    frame["__pos"] = frame["Month"].map(position_of).astype(int)

    demand = np.full((len(pairs), len(months)), np.nan)
    zero = np.full((len(pairs), len(months)), np.nan)
    usable = panel.loc[panel["lifecycle_ok"] & panel["demand_units"].notna()]
    usable = usable.drop_duplicates(CELL)
    rows = pair_index.get_indexer(pd.MultiIndex.from_frame(usable[SERIES]))
    cols = usable["Month"].map(position_of).to_numpy()
    inside = (rows >= 0) & pd.notna(cols)
    demand[rows[inside], cols[inside].astype(int)] = usable["demand_units"].to_numpy()[inside]
    zero_values = usable["observed_units"].eq(0).astype(float).where(
        usable["observed_units"].notna()).to_numpy()
    zero[rows[inside], cols[inside].astype(int)] = zero_values[inside]

    missing_columns = [column for column in KNOWN_FEATURES + CATEGORICAL_FEATURES
                       if column not in frame.columns]
    if missing_columns:
        raise SystemExit(f"FATAL: Phase 3 table is missing {missing_columns}")

    # Region is both a key and a categorical feature; dict.fromkeys keeps one column.
    # The Phase 3 lag columns are carried along **only** so the exit check can prove
    # the recursive block reproduces them; they are never part of a design matrix.
    selected = list(dict.fromkeys([*CELL, *CATEGORICAL_FEATURES, "split", "grid_status",
                                   "post_eol", "target_demand", "target_observed",
                                   "target_is_zero", "cell_absent", "__pair", "__pos",
                                   *KNOWN_FEATURES, *PARITY_LAG_FEATURES, "price"]))
    known = frame[selected].copy()
    # the list price is carried per cell-month (Phase 3 carries 2026 forward), so
    # revenue can be shown beside the units without a second price lookup
    known["unit_price"] = known["price"].astype(float)
    known["__row"] = np.arange(len(known))

    categories = {column: sorted(frame[column].dropna().unique().tolist())
                  for column in CATEGORICAL_FEATURES}
    dummies, dummy_names = encode_categoricals(known, categories)
    numeric = known[KNOWN_FEATURES].astype(float).to_numpy()
    design = np.hstack([numeric, dummies])
    feature_names = [*KNOWN_FEATURES, *dummy_names]

    rows_by_position = {int(pos): block["__row"].to_numpy()
                        for pos, block in known.groupby("__pos")}

    report.kv("grid", f"{len(pairs):,} SKU-Region pairs x {len(months)} months")
    report.kv("training months", f"{len(train_months)} "
                                 f"({_month_text(train_months[0])} -> {_month_text(train_months[-1])})")
    report.kv("forecast months", f"{len(horizon_months)} "
                                 f"({_month_text(horizon_months[0])} -> {_month_text(horizon_months[-1])})")
    report.kv("recorded demand cells", f"{int(np.isfinite(demand).sum()):,}")
    report.kv("design columns", f"{design.shape[1]} ({len(KNOWN_FEATURES)} known + "
                                f"{len(dummy_names)} one-hot + {len(LAG_FEATURES)} recursive)")
    report.kv("test keys", f"{len(test):,} across {test['Month'].nunique()} months")
    report.rule("FC-PANEL", "demand history rebuilt on a pairs x months grid",
                "lags are recomputed from this grid, never read from the feature table",
                len(known),
                detail=f"{int(np.isfinite(demand).sum()):,} observed/demand cells, "
                       f"NaN elsewhere (absence is not zero)")
    report.metrics["panel"] = {
        "pairs": int(len(pairs)), "months": int(len(months)),
        "train_months": len(train_months), "horizon_months": len(horizon_months),
        "demand_cells": int(np.isfinite(demand).sum()),
        "design_columns": int(design.shape[1]),
    }
    panel = Panel(months=months, pairs=pairs, demand=demand, zero=zero, known=known,
                  design=design, feature_names=feature_names,
                  pair_code=known["__pair"].to_numpy(), position=known["__pos"].to_numpy(),
                  rows_by_position=rows_by_position)
    panel.lags = lag_lookup(panel)
    return panel


# --------------------------------------------------------------------------- #
# 3. recursive lag block
# --------------------------------------------------------------------------- #
def _window(values: np.ndarray, position: int, window: int, minimum: int,
            how: str = "mean") -> np.ndarray:
    """Statistic over positions ``[position-window, position)``; NaN below ``minimum``."""
    start = max(0, position - window)
    block = values[:, start:position]
    if block.shape[1] == 0:  # the first month has no history at all
        return np.full(values.shape[0], np.nan)
    count = np.isfinite(block).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "mean":
            total = np.nansum(block, axis=1)
            result = total / np.maximum(count, 1)
        elif how == "median":
            result = np.nanmedian(block, axis=1)
        elif how == "max":
            result = np.where(np.isfinite(block), block, -np.inf).max(axis=1)
            result = np.where(np.isfinite(result), result, np.nan)
        elif how == "std":
            total = np.nansum(block, axis=1)
            mean = total / np.maximum(count, 1)
            with np.errstate(invalid="ignore"):
                squared = np.where(np.isfinite(block), (block - mean[:, None]) ** 2, 0.0)
            result = np.sqrt(np.nansum(squared, axis=1) / np.maximum(count - 1, 1))
        else:  # pragma: no cover - guarded by callers
            raise ValueError(how)
    return np.where(count >= minimum, result, np.nan)


def lag_block(demand: np.ndarray, zero: np.ndarray, position: int) -> np.ndarray:
    """The recursive feature block for one month, identical in shape to LAG_FEATURES.

    Definitions match Phase 3 exactly (verified against the feature table for every
    training month), so the only new behaviour is what happens once the history
    stops: the caller writes predictions into ``demand``/``zero`` and the same
    formulas apply to them.
    """
    n = demand.shape[0]

    def lag(step: int) -> np.ndarray:
        return demand[:, position - step] if position - step >= 0 else np.full(n, np.nan)

    lag_12, lag_24 = lag(12), lag(24)
    with np.errstate(invalid="ignore", divide="ignore"):
        yoy = np.where(lag_24 >= 1, (lag_12 + 1) / (lag_24 + 1), np.nan)

    if position >= 1:
        prior = demand[:, :position] > 0
        has_order = prior.any(axis=1)
        # column j of the reversed block is month position-1-j, so the gap is j+1
        distance = prior[:, ::-1].argmax(axis=1).astype(float) + 1.0
        since = np.where(has_order, np.clip(distance, 0, 24), np.nan)
    else:
        since = np.full(n, np.nan)

    values = {
        "lag_1": lag(1), "lag_2": lag(2), "lag_3": lag(3), "lag_6": lag(6), "lag_12": lag_12,
        "mean_3": _window(demand, position, 3, 2),
        "mean_6": _window(demand, position, 6, 3),
        "mean_12": _window(demand, position, 12, 6),
        "median_3": _window(demand, position, 3, 2, "median"),
        "std_6": _window(demand, position, 6, 3, "std"),
        "max_12": _window(demand, position, 12, 6, "max"),
        "zero_share_6": _window(zero, position, 6, 2),
        "months_since_last_order": since,
        "yoy_ratio": yoy,
        "cell_mean": _window(demand, position, position, 1),
    }
    return np.column_stack([values[column] for column in LAG_FEATURES])


def lag_lookup(panel: Panel) -> np.ndarray:
    """Lag block for every known row, computed from the *actual* history."""
    values = np.full((panel.n_rows, len(LAG_FEATURES)), np.nan)
    for position, rows in panel.rows_by_position.items():
        block = lag_block(panel.demand, panel.zero, position)
        values[rows] = block[panel.pair_code[rows]]
    return values


# --------------------------------------------------------------------------- #
# 4. models
# --------------------------------------------------------------------------- #
class Standardiser:
    """Winsorise at the fit window, median-fill, then centre and scale."""

    def __init__(self) -> None:
        self.low = self.high = self.median = self.mean = self.std = None

    def fit(self, values: np.ndarray) -> "Standardiser":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(values, axis=0)
            self.low = np.nanquantile(values, 0.01, axis=0)
            self.high = np.nanquantile(values, 0.99, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.low = np.where(np.isfinite(self.low), self.low, self.median)
        self.high = np.where(np.isfinite(self.high), self.high, self.median)
        filled = self.transform_raw(values)
        self.mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def transform_raw(self, values: np.ndarray) -> np.ndarray:
        values = np.where(np.isfinite(values), values, self.median)
        return np.clip(values, self.low, self.high)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (self.transform_raw(values) - self.mean) / self.std

    def state(self) -> dict:
        return {"median": self.median.tolist(), "low": self.low.tolist(),
                "high": self.high.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist()}


def _penalty(size: int, lam: float) -> np.ndarray:
    penalty = np.full(size, lam)
    penalty[0] = 0.0  # the intercept column is never shrunk
    return penalty


def fit_logistic(x: np.ndarray, y: np.ndarray, lam: float,
                 iterations: int = LOGISTIC_ITERATIONS) -> np.ndarray:
    """Ridge-penalised logistic regression by L-BFGS.

    Newton steps would need an n x d x d Hessian pass (minutes on 260k rows);
    the penalised likelihood is strictly convex, so L-BFGS-B converges just as
    reliably from a zero start and is deterministic.
    """
    design = np.hstack([np.ones((len(x), 1)), x])
    penalty = _penalty(design.shape[1], lam)

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        likelihood = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        return -likelihood + 0.5 * float(np.sum(penalty * beta * beta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probability = _sigmoid(design @ beta)
        return design.T @ (probability - y) + penalty * beta

    result = minimize(objective, np.zeros(design.shape[1]), jac=gradient,
                      method="L-BFGS-B",
                      options={"maxiter": iterations, "ftol": 1e-12, "gtol": 1e-8})
    return result.x


def fit_linear(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with an unpenalised intercept; x is already centred by the standardiser."""
    design = np.hstack([np.ones((len(x), 1)), x])
    penalty = _penalty(design.shape[1], lam)
    gram = design.T @ design + np.diag(penalty)
    beta = np.linalg.solve(gram, design.T @ y)
    return beta


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class HurdleRidge:
    """Probability of an order x expected size, both ridge-regularised linear models."""

    name = "hurdle_ridge"
    family = "hurdle"

    def __init__(self, lam: float = RIDGE_LAMBDA) -> None:
        self.lam = lam
        self.scaler = Standardiser()
        self.classifier: np.ndarray | None = None
        self.magnitude: np.ndarray | None = None

    def fit(self, x: np.ndarray, demand: np.ndarray) -> "HurdleRidge":
        self.scaler.fit(x)
        z = self.scaler.transform(x)
        design = np.hstack([np.ones((len(z), 1)), z])
        present = np.isfinite(demand)
        positive = demand > 0
        target = np.where(present, demand, 0.0)
        self.classifier = fit_logistic(z, positive.astype(float), self.lam)
        log_size = np.log1p(target[positive])
        self.magnitude = fit_linear(z[positive], log_size, self.lam)
        # Duan smearing: a log-scale fit estimates the conditional median, so
        # rescale by the mean of exp(residual) before reading it as a mean.
        self.smear = float(np.mean(np.exp(log_size - design[positive] @ self.magnitude)))
        raw = self._probability(design) * self._size(design)
        self.level = float(np.clip(_safe_divide(float(target[present].sum()),
                                                float(raw[present].sum())), 0.5, 2.0))
        return self

    def _probability(self, design: np.ndarray) -> np.ndarray:
        return _sigmoid(design @ self.classifier)

    def _size(self, design: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(design @ self.magnitude) * self.smear - 1.0, 0.0, None)

    def predict_parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = np.hstack([np.ones((len(x), 1)), self.scaler.transform(x)])
        return self._probability(design), self._size(design) * self.level

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability, size = self.predict_parts(x)
        return np.clip(probability * size, 0.0, None)


class HurdleLgbm:
    """Same decomposition, trees instead of linear terms."""

    name = "hurdle_lgbm"
    family = "hurdle"

    def __init__(self) -> None:
        self.classifier = None
        self.magnitude = None

    def fit(self, x: np.ndarray, demand: np.ndarray) -> "HurdleLgbm":
        import lightgbm as lgb

        present = np.isfinite(demand)
        positive = demand > 0
        target = np.where(present, demand, 0.0)
        self.classifier = lgb.LGBMClassifier(**LGBM_PARAMS).fit(x, positive.astype(int))
        log_size = np.log1p(target[positive])
        self.magnitude = lgb.LGBMRegressor(**LGBM_PARAMS).fit(x[positive], log_size)
        self.smear = float(np.mean(np.exp(log_size - self.magnitude.predict(x[positive]))))
        # trees are fitted on log scale too, and the level drifts with the mix of
        # series, so the fit window sets one multiplicative correction
        raw = self._probability(x) * self._size(x)
        self.level = float(np.clip(_safe_divide(float(target[present].sum()),
                                                float(raw[present].sum())), 0.5, 2.0))
        return self

    def _probability(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(x)[:, 1]

    def _size(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.exp(self.magnitude.predict(x)) * self.smear - 1.0, 0.0, None)

    def predict_parts(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._probability(x), self._size(x) * self.level

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability, size = self.predict_parts(x)
        return np.clip(probability * size, 0.0, None)

    def importances(self, names: list[str], top: int = 25) -> list[dict]:
        gain = self.magnitude.booster_.feature_importance(importance_type="gain")
        order = np.argsort(gain)[::-1][:top]
        return [{"feature": names[i], "gain": float(gain[i])} for i in order if gain[i] > 0]


class Blend:
    """Convex combination of the seasonal baseline and the tree hurdle.

    The two models fail differently: the baseline carries the recurring account
    pattern, the tree carries the level and the drivers, so a combination beats
    either.  The weight is **not** a free parameter read off the scored holdout:
    it is chosen on an inner 19-month window that sits inside the fit period
    (see ``select_blend_weight``), and the outer holdout only ever sees the
    resulting fixed weight.  An earlier draft of this phase did read the weight
    off the holdout curve; that is exactly the leak this split removes.

    The lag block is the tail of the design, which is what the baseline reads.
    """

    name = "blend"
    family = "hurdle"

    def __init__(self, baseline, arm, weight: float = 0.5) -> None:
        self.baseline = baseline
        self.arm = arm
        self.weight = weight

    def predict(self, x: np.ndarray) -> np.ndarray:
        block = x[:, -len(LAG_FEATURES):]
        combined = (self.weight * self.baseline.predict_block(block)
                    + (1.0 - self.weight) * self.arm.predict(x))
        return np.clip(combined, 0.0, None)


class SeasonalNaive:
    """Same calendar month last year, falling back to the trailing mean, then history."""

    name = "seasonal_naive"
    family = "baseline"

    def predict_block(self, lags: np.ndarray) -> np.ndarray:
        index = {column: i for i, column in enumerate(LAG_FEATURES)}
        lag_12 = lags[:, index["lag_12"]]
        mean_3 = lags[:, index["mean_3"]]
        mean_12 = lags[:, index["mean_12"]]
        cell_mean = lags[:, index["cell_mean"]]
        value = np.where(np.isfinite(lag_12), lag_12,
                         np.where(np.isfinite(mean_3), mean_3,
                                  np.where(np.isfinite(mean_12), mean_12, cell_mean)))
        return np.clip(np.where(np.isfinite(value), value, 0.0), 0.0, None)


class MovingAverage:
    """Trailing three months, falling back to six, twelve and the full history."""

    name = "moving_average"
    family = "baseline"

    def predict_block(self, lags: np.ndarray) -> np.ndarray:
        index = {column: i for i, column in enumerate(LAG_FEATURES)}
        value = None
        for column in ["mean_3", "mean_6", "mean_12", "cell_mean"]:
            candidate = lags[:, index[column]]
            value = candidate if value is None else np.where(np.isfinite(value), value, candidate)
        return np.clip(np.where(np.isfinite(value), value, 0.0), 0.0, None)


def build_models() -> dict[str, object]:
    return {"seasonal_naive": SeasonalNaive(), "moving_average": MovingAverage()}


def fit_hurdles(panel: Panel, rows: np.ndarray, report: ForecastReport) -> tuple[dict, dict]:
    """Fit both hurdle engines on ``rows`` of the panel (training months only)."""
    design = np.hstack([panel.design[rows], panel.lags[rows]])
    demand = panel.known["target_demand"].to_numpy()[rows]  # NaN where the cell had no order

    ridge = HurdleRidge().fit(design, demand)
    report.kv("hurdle_ridge fitted", f"{len(rows):,} rows, {design.shape[1]} columns, "
                                     f"level x{ridge.level:.2f}")
    lgbm = HurdleLgbm().fit(design, demand)
    report.kv("hurdle_lgbm fitted", f"{len(rows):,} rows, "
                                   f"{int(np.nansum(demand > 0)):,} positive, "
                                   f"level x{lgbm.level:.2f}")
    return {"hurdle_ridge": ridge, "hurdle_lgbm": lgbm}, {"design": design, "demand": demand}


# --------------------------------------------------------------------------- #
# 5. recursive path
# --------------------------------------------------------------------------- #
def run_path(panel: Panel, model, positions: list[int],
             demand: np.ndarray, zero: np.ndarray) -> dict[int, np.ndarray]:
    """Walk the horizon month by month, writing each prediction back into history.

    ``demand``/``zero`` are owned by the caller, so every candidate gets its own
    path: the baseline's recursion uses the baseline's own predictions, not the
    winning model's.  Month ``p``'s lag features read columns before ``p`` only.
    """
    predictions: dict[int, np.ndarray] = {}
    for position in positions:
        rows = panel.rows_by_position.get(position)
        if rows is None or len(rows) == 0:
            continue
        block = lag_block(demand, zero, position)[panel.pair_code[rows]]
        if model.family == "baseline":
            predicted = model.predict_block(block)
        else:
            predicted = model.predict(np.hstack([panel.design[rows], block]))
        predicted = np.clip(np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0),
                            0.0, None)
        predictions[position] = predicted
        demand[panel.pair_code[rows], position] = predicted
        zero[panel.pair_code[rows], position] = (predicted < 0.5).astype(float)
    return predictions


def run_all(panel: Panel, candidates: dict, positions: list[int],
            blank_from: int | None = None) -> dict[str, dict[int, np.ndarray]]:
    """Run every candidate over its own recursive path."""
    output = {}
    for name, model in candidates.items():
        demand = panel.demand.copy()
        zero = panel.zero.copy()
        if blank_from is not None:
            demand[:, blank_from:] = np.nan
            zero[:, blank_from:] = np.nan
        output[name] = run_path(panel, model, positions, demand, zero)
    return output


def collect(panel: Panel, predictions: dict[str, dict[int, np.ndarray]],
            positions: list[int]) -> pd.DataFrame:
    frames = []
    for position in positions:
        rows = panel.rows_by_position.get(position)
        if rows is None or len(rows) == 0:
            continue
        block = pd.DataFrame({"__row": rows})
        for name, by_position in predictions.items():
            block[name] = by_position[position]
        frames.append(block)
    if not frames:
        return pd.DataFrame(columns=["__row", *predictions])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. backtest
# --------------------------------------------------------------------------- #
def select_blend_weight(panel: Panel, months: list, origin_position: int,
                        report: ForecastReport) -> tuple[float, pd.DataFrame, int]:
    """Choose the blend weight on an inner window, before the holdout is scored.

    The inner origin sits one holdout length earlier, so the inner window
    (19 months) ends exactly where the outer fit window ends.  Models are
    refitted on the inner fit period, the weight is swept there, and the winner
    is carried into the outer run unchanged - the scored months never inform it.
    """
    inner_position = origin_position - BACKTEST_MONTHS
    inner_month = months[inner_position]
    inner_rows = panel.known.index[panel.known["Month"] < inner_month].to_numpy()
    inner_candidates, _ = fit_hurdles(panel, inner_rows, report)
    inner_candidates.update(build_models())

    positions = list(range(inner_position, inner_position + BACKTEST_MONTHS))
    curve = []
    for weight in np.round(np.arange(0.0, 1.01, 0.1), 2):
        blend = Blend(inner_candidates["seasonal_naive"], inner_candidates["hurdle_lgbm"],
                      weight=float(weight))
        predictions = run_all(panel, {"blend": blend}, positions, blank_from=inner_position)
        scored = collect(panel, predictions, positions)
        scored = scored.merge(panel.known[["__row", "target_demand"]], on="__row", how="left")
        scored = scored.loc[scored["target_demand"].notna()]
        curve.append({"weight": float(weight),
                      "wape": wape(scored["target_demand"], scored["blend"]),
                      "bias": _safe_divide(scored["blend"].sum(),
                                           scored["target_demand"].sum()) - 1.0})

    frame = pd.DataFrame(curve)
    eligible = frame.loc[frame["bias"].abs() <= BIAS_TOLERANCE]
    chosen = (eligible if len(eligible) else frame).sort_values("wape").iloc[0]

    report.sub("blend weight chosen on the inner window")
    report.kv("inner window", f"{_month_text(inner_month)} -> "
                               f"{_month_text(months[inner_position + BACKTEST_MONTHS - 1])}"
                               f" ({len(positions)} months, fit < {_month_text(inner_month)})")
    for row in frame.itertuples(index=False):
        mark = "  <- chosen" if abs(row.weight - chosen["weight"]) < 1e-9 else ""
        report.p(f"  w={row.weight:.1f}  WAPE {row.wape:.4f}  bias {row.bias:+.1%}{mark}")
    report.kv("pre-registered blend weight", f"{chosen['weight']:.1f} "
                                             f"(inner WAPE {chosen['wape']:.4f})")
    report.rule("FC-BLEND", "blend weight is a free parameter",
                "selected on an inner 19-month window inside the fit period; "
                "the outer holdout only sees the resulting fixed weight",
                len(scored), detail=f"w={chosen['weight']:.1f} chosen at "
                                    f"{_month_text(inner_month)}; holdout never used")
    return float(chosen["weight"]), frame, inner_position


SLICE_DIMENSIONS = {"product_family": "Product_Family", "region": "Region",
                    "business_unit": "Business_Unit", "horizon_months": "months_ahead"}


def slice_scores(scored: pd.DataFrame, model: str) -> pd.DataFrame:
    """Same WAPE rule applied to family, region, business unit and months ahead.

    Revenue is scored too, because a planner reads a miss on a low-price, high-volume
    family differently from a miss on a high-price, low-volume one.
    """
    rows: list[dict] = []

    def record(dimension: str, value, block: pd.DataFrame) -> None:
        actual = block["actual"].to_numpy(float)
        predicted = block[model].to_numpy(float)
        actual_revenue = block["actual_revenue"].to_numpy(float)
        forecast_revenue = block["forecast_revenue"].to_numpy(float)
        rows.append({
            "model": model, "dimension": dimension, "value": str(value),
            "rows": int(len(block)),
            "actual_units": float(actual.sum()), "forecast_units": float(predicted.sum()),
            "wape": wape(actual, predicted),
            "bias": _safe_divide(predicted.sum(), actual.sum()) - 1.0,
            "actual_revenue": float(actual_revenue.sum()),
            "forecast_revenue": float(forecast_revenue.sum()),
            "revenue_wape": _safe_divide(np.abs(actual_revenue - forecast_revenue).sum(),
                                         np.abs(actual_revenue).sum()),
            "revenue_bias": _safe_divide(forecast_revenue.sum(), actual_revenue.sum()) - 1.0,
        })

    record("all", "ALL", scored)
    for dimension, column in SLICE_DIMENSIONS.items():
        for value, block in scored.groupby(column, dropna=False):
            record(dimension, value, block)
    return pd.DataFrame(rows)


def backtest(panel: Panel, tables: dict, report: ForecastReport) -> tuple[pd.DataFrame, dict]:
    report.h("2. FORWARD-IN-TIME BACKTEST")
    months = list(panel.months)
    train_months = sorted(panel.known.loc[panel.known["split"].eq("train"), "Month"].unique())
    origin_month = pd.Timestamp(train_months[-BACKTEST_MONTHS])
    origin_position = months.index(origin_month)

    blend_weight, weight_curve, inner_position = select_blend_weight(
        panel, months, origin_position, report)

    train_rows = panel.known.index[panel.known["Month"] < origin_month].to_numpy()
    candidates, fitted = fit_hurdles(panel, train_rows, report)
    candidates.update(build_models())
    candidates["blend"] = Blend(candidates["seasonal_naive"], candidates["hurdle_lgbm"],
                                weight=blend_weight)

    scored_positions = list(range(origin_position, origin_position + BACKTEST_MONTHS))
    predictions = run_all(panel, candidates, scored_positions, blank_from=origin_position)
    scored = collect(panel, predictions, scored_positions)
    scored = scored.merge(panel.known[["__row", *CELL, "Product_Family", "Business_Unit",
                                       "unit_price", "target_demand"]],
                          on="__row", how="left")
    scored = scored.rename(columns={"target_demand": "actual"})
    scored = scored.loc[scored["actual"].notna()].reset_index(drop=True)
    for name in predictions:
        scored[name] = scored[name].fillna(0.0)

    names = ["seasonal_naive", "moving_average", "hurdle_ridge", "hurdle_lgbm", "blend"]
    monthly = scored.groupby("Month").agg(actual=("actual", "sum"), **{
        name: (name, "sum") for name in names}).reset_index()
    for name in names:
        monthly[f"{name}_wape"] = [
            wape(block["actual"].to_numpy(), block[name].to_numpy())
            for _, block in scored.groupby("Month")]

    rows = []
    for name in names:
        rows.append({"model": name, "family": candidates[name].family,
                     "wape": wape(scored["actual"], scored[name]),
                     "monthly_wape": wape(monthly["actual"], monthly[name]),
                     "bias": _safe_divide(scored[name].sum(), scored["actual"].sum()) - 1.0,
                     "mae": float(np.abs(scored["actual"] - scored[name]).mean()),
                     "rmse": float(np.sqrt(np.mean((scored["actual"] - scored[name]) ** 2))),
                     "actual_units": float(scored["actual"].sum()),
                     "forecast_units": float(scored[name].sum()),
                     "rows": int(len(scored))})
    scores = pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
    naive = float(scores.loc[scores["model"].eq("seasonal_naive"), "wape"].iloc[0])
    scores["skill_vs_naive"] = 1.0 - scores["wape"] / naive

    report.sub("candidate scores on the 19-month holdout")
    report.p(f"  {'model':<18}{'family':<10}{'WAPE':>8}{'bias':>9}{'skill':>9}{'monthly':>9}")
    for row in scores.itertuples(index=False):
        report.p(f"  {row.model:<18}{row.family:<10}{row.wape:>8.4f}{row.bias:>+9.1%}"
                 f"{row.skill_vs_naive:>+9.1%}{row.monthly_wape:>9.4f}")
    report.kv("holdout months", f"{_month_text(months[origin_position])} -> "
                                f"{_month_text(months[-1])} ({len(scored_positions)})")
    report.kv("fit months", f"{_month_text(months[0])} -> "
                            f"{_month_text(months[origin_position - 1])}")
    report.kv("scored rows", f"{len(scored):,}")
    report.kv("holdout actual units", f"{scored['actual'].sum():,.0f}")

    eligible = scores.loc[scores["bias"].abs() <= BIAS_TOLERANCE]
    selected = (eligible if len(eligible) else scores).sort_values("wape").iloc[0]["model"]
    position_of = {month: index for index, month in enumerate(months)}
    scored["months_ahead"] = (scored["Month"].map(position_of).astype(int)
                              - origin_position + 1)
    scored["actual_revenue"] = scored["actual"] * scored["unit_price"].fillna(0.0)
    scored["forecast_revenue"] = scored[selected] * scored["unit_price"].fillna(0.0)
    slices = slice_scores(scored, selected)
    report.sub("selected model error by slice")
    for dimension in ["product_family", "region", "horizon_months"]:
        block = slices.loc[slices["dimension"].eq(dimension)]
        worst = block.sort_values("wape", ascending=False)
        report.p(f"  {dimension:<16}" + ", ".join(
            f"{row.value}: WAPE {row.wape:.3f}" for row in worst.head(4).itertuples()))
    report.kv("selected model", selected)
    report.rule("FC-BACKTEST", "candidates scored forward in time at the true horizon",
                "fit strictly before 2025-04, then 19 recursive months",
                len(scored),
                detail=f"selected {selected}; naive WAPE {naive:.4f}")
    report.metrics["backtest"] = {
        "origin": _month_text(months[origin_position]),
        "months": len(scored_positions), "rows": int(len(scored)),
        "scores": scores.to_dict(orient="records"), "selected": str(selected),
        "bias_tolerance": BIAS_TOLERANCE,
        "blend_weight": blend_weight,
        "blend_weight_curve": weight_curve.to_dict(orient="records"),
        "blend_weight_inner_origin": _month_text(months[inner_position]),
        "slices": slices.to_dict(orient="records"),
    }
    return scored, {"scores": scores, "monthly": monthly, "selected": selected,
                    "slices": slices,
                    "origin_position": origin_position, "fit_rows": train_rows,
                    "predictions": predictions, "candidates": candidates,
                    "design": fitted["design"], "blend_weight": blend_weight,
                    "weight_curve": weight_curve, "inner_position": inner_position}


# --------------------------------------------------------------------------- #
# 7. final forecast
# --------------------------------------------------------------------------- #
def final_forecast(panel: Panel, tables: dict, backtest_state: dict,
                   report: ForecastReport) -> pd.DataFrame:
    report.h("3. FINAL FORECAST AND SUBMISSION")
    months = list(panel.months)
    horizon_months = sorted(panel.known.loc[panel.known["split"].eq("forecast"), "Month"].unique())
    horizon_positions = [months.index(pd.Timestamp(month)) for month in horizon_months]

    all_train = panel.known.index[panel.known["split"].eq("train")].to_numpy()
    candidates, _ = fit_hurdles(panel, all_train, report)
    candidates.update(build_models())
    candidates["blend"] = Blend(candidates["seasonal_naive"], candidates["hurdle_lgbm"],
                                weight=backtest_state["blend_weight"])

    predictions = run_all(panel, candidates, horizon_positions)
    candidate_names = list(predictions)
    cells = collect(panel, predictions, horizon_positions)
    cells = cells.merge(panel.known[["__row", *CELL, "split", "grid_status", "post_eol",
                                     "cell_absent", "Product_Family", "Business_Unit",
                                     "unit_price"]],
                        on="__row", how="left")
    cells["Year"] = cells["Month"].dt.year
    cells["Month_Num"] = cells["Month"].dt.month

    selected = backtest_state["selected"]
    post_eol = cells["post_eol"].fillna(False).astype(bool)

    # determinism: the same fitted models and the same path must reproduce itself
    # (checked before the post-EOL override, which is a rule rather than a prediction)
    first_pass = cells[candidate_names].to_numpy()
    repeat = collect(panel, run_all(panel, candidates, horizon_positions), horizon_positions)
    first_rows = cells["__row"].to_numpy()
    repeat = repeat.set_index("__row").loc[first_rows]
    reproducible = bool(np.array_equal(first_pass, repeat[candidate_names].to_numpy()))

    cells.loc[post_eol, candidate_names] = 0.0
    cells["forecast_raw"] = cells[selected]
    cells.loc[post_eol, "forecast_raw"] = 0.0
    cells["forecast_units"] = np.rint(cells["forecast_raw"].clip(lower=0)).astype("int64")

    report.kv("selected model", selected)
    report.kv("post-EOL keys forced to zero", f"{int(post_eol.sum()):,}")
    report.kv("forecast keys", f"{len(cells):,}")
    report.kv("forecast units (19 months)", f"{cells['forecast_units'].sum():,.0f}")
    report.rule("FC-POST-EOL", "test keys after the SKU's EOL month",
                "forecast exactly zero - the ERP records no sales after EOL", int(post_eol.sum()),
                detail="not a clip: the Phase 1 ledger shows 0 post-EOL sales rows")
    return cells, {"reproducible": reproducible, "horizon_positions": horizon_positions,
                   "fit_rows": all_train, "candidates": candidates}


# --------------------------------------------------------------------------- #
# 8. uncertainty
# --------------------------------------------------------------------------- #
def add_uncertainty(cells: pd.DataFrame, scored: pd.DataFrame, selected: str,
                    report: ForecastReport) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report.h("4. UNCERTAINTY")
    predicted = np.clip(cells["forecast_raw"].to_numpy(), 0.0, None)
    # Multiplicative band: order sizes differ by four orders of magnitude across
    # the portfolio, so an additive count band is either meaningless for small
    # cells or absurd for large ones.  The spread of log((actual+1)/(pred+1)) on
    # the holdout is the scale-free equivalent.
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ratio = np.log((scored["actual"].to_numpy() + 1.0)
                           / (np.clip(scored[selected].to_numpy(), 0, None) + 1.0))
    sigma = float(np.nanstd(log_ratio)) if np.isfinite(log_ratio).any() else 0.5
    sigma = max(sigma, 1e-6)
    cells["forecast_p10"] = np.clip(np.rint((predicted + 1.0) * np.exp(-Z80 * sigma) - 1.0),
                                    0, None)
    cells["forecast_p50"] = cells["forecast_units"].astype(float)
    cells["forecast_p90"] = np.clip(np.rint((predicted + 1.0) * np.exp(Z80 * sigma) - 1.0),
                                    0, None)
    # a key that is zero by rule carries no band either
    inactive = cells["post_eol"].fillna(False).astype(bool)
    cells.loc[inactive, ["forecast_p10", "forecast_p90"]] = 0.0

    # revenue is the units forecast at the list price that Phase 3 carried into
    # the horizon, so the two can never drift apart
    price = cells["unit_price"].fillna(0.0)
    cells["list_price"] = price.round(2)
    cells["forecast_revenue"] = np.rint(cells["forecast_units"] * price).astype("int64")
    cells["revenue_p10"] = np.rint(cells["forecast_p10"] * price).astype("int64")
    cells["revenue_p90"] = np.rint(cells["forecast_p90"] * price).astype("int64")

    monthly = cells.groupby("Month").agg(
        forecast=("forecast_units", "sum"), p10=("forecast_p10", "sum"),
        p90=("forecast_p90", "sum"), revenue=("forecast_revenue", "sum"),
        cells=("forecast_units", "size"),
        post_eol=("post_eol", "sum")).reset_index()
    ratio_by_month = (scored.groupby("Month").apply(
        lambda block: _safe_divide(block["actual"].sum(), block[selected].sum()),
        include_groups=False).rename("ratio").reset_index())
    low, high = np.nanquantile(ratio_by_month["ratio"], [0.1, 0.9])
    monthly["ratio_p10"] = low
    monthly["ratio_p90"] = high
    monthly["band_p10"] = np.rint(monthly["forecast"] * low).astype("int64")
    monthly["band_p90"] = np.rint(monthly["forecast"] * high).astype("int64")
    # the revenue band uses the same aggregate ratio as the unit band
    monthly["revenue_p10"] = np.rint(monthly["revenue"] * low).astype("int64")
    monthly["revenue_p90"] = np.rint(monthly["revenue"] * high).astype("int64")
    monthly["year"] = monthly["Month"].dt.year
    monthly["month_num"] = monthly["Month"].dt.month

    report.kv("cell band sigma (log scale)", f"{sigma:.3f} "
                                             f"(x{np.exp(-Z80 * sigma):.2f} / "
                                             f"x{np.exp(Z80 * sigma):.2f})")
    report.kv("monthly ratio p10 / p90", f"{low:.2f} / {high:.2f}")
    report.kv("forecast revenue (19 months)", f"{cells['forecast_revenue'].sum():,.0f} USD")
    report.kv("forecast months with a band", len(monthly))
    report.rule("FC-UNCERTAINTY", "multiplicative error band on the cell forecast",
                "prediction x exp(+/- 1.28 x sigma), sigma from holdout log ratios",
                len(cells), detail=f"sigma {sigma:.3f}; aggregate band from monthly "
                                   f"holdout ratios p10 {low:.2f} / p90 {high:.2f}")
    report.metrics["uncertainty"] = {"sigma_log": sigma,
                                     "monthly_ratio_p10": float(low),
                                     "monthly_ratio_p90": float(high)}
    report.metrics["revenue"] = {"forecast_revenue": float(cells["forecast_revenue"].sum()),
                                 "price_source": "Phase 3 list price carried into the horizon",
                                 "unpriced_cells": int(price.eq(0.0).sum())}
    return cells, monthly, {"sigma_log": sigma, "low": float(low), "high": float(high)}


# --------------------------------------------------------------------------- #
# 9. outputs
# --------------------------------------------------------------------------- #
def build_submission(cells: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    submission = pd.DataFrame({
        "Date": cells["Month"].dt.strftime("%Y-%m"),
        "SKU_ID": cells["SKU_ID"],
        "Region": cells["Region"],
        "Units_Sold_Forecast": cells["forecast_units"].astype(int),
    })
    submission = submission[columns].sort_values(["Date", "SKU_ID", "Region"]).reset_index(drop=True)
    return submission


def write_outputs(out_dir: Path, report: ForecastReport, submission: pd.DataFrame,
                  cells: pd.DataFrame, monthly: pd.DataFrame, scores: pd.DataFrame,
                  backtest_cell: pd.DataFrame, backtest_monthly: pd.DataFrame,
                  slices: pd.DataFrame, diagnostics: dict) -> None:
    forecast_dir = out_dir / FORECAST_DIR
    forecast_dir.mkdir(parents=True, exist_ok=True)

    submission.to_csv(out_dir / "submission.csv", index=False)
    candidates = [column for column in cells.columns
                  if column in ("seasonal_naive", "moving_average", "hurdle_ridge",
                                "hurdle_lgbm", "blend")]
    keep = ["__row", *CELL, "Year", "Month_Num", "Product_Family", "Business_Unit",
            "grid_status", "post_eol", *candidates,
            "forecast_units", "forecast_raw", "forecast_p10", "forecast_p50", "forecast_p90",
            "unit_price", "list_price", "forecast_revenue", "revenue_p10", "revenue_p90"]
    cells[[column for column in keep if column in cells.columns]].to_parquet(
        forecast_dir / "forecast_cell_month.parquet", index=False)
    monthly.to_parquet(forecast_dir / "forecast_monthly_summary.parquet", index=False)
    scores.to_csv(forecast_dir / "backtest_scores.csv", index=False)
    backtest_cell.to_parquet(forecast_dir / "backtest_cell_month.parquet", index=False)
    backtest_monthly.to_parquet(forecast_dir / "backtest_monthly.parquet", index=False)
    slices.to_csv(forecast_dir / "backtest_slices.csv", index=False)
    pd.DataFrame(report.log).to_csv(forecast_dir / "forecast_rules.csv", index=False)
    (forecast_dir / "model_diagnostics.json").write_text(
        json.dumps(jsonable(diagnostics), indent=2), encoding="utf-8")

    payload = {
        "meta": {
            "phase": 4,
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - report._t0, 2),
            "pandas": pd.__version__, "numpy": np.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "python": sys.version.split()[0], "seed": SEED,
        },
        "metrics": report.metrics,
        "verdicts": report.verdicts,
        "rules": report.log,
        "outputs": {
            "submission": "outputs/submission.csv",
            "forecast_cell_month": "outputs/forecast/forecast_cell_month.parquet",
            "forecast_monthly_summary": "outputs/forecast/forecast_monthly_summary.parquet",
            "backtest_scores": "outputs/forecast/backtest_scores.csv",
            "backtest_cell_month": "outputs/forecast/backtest_cell_month.parquet",
            "backtest_monthly": "outputs/forecast/backtest_monthly.parquet",
            "backtest_slices": "outputs/forecast/backtest_slices.csv",
            "model_diagnostics": "outputs/forecast/model_diagnostics.json",
        },
    }
    (out_dir / "forecast_report.json").write_text(
        json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "forecast_report.txt").write_text("\n".join(report.lines) + "\n",
                                                 encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 forecasting and backtest")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--features-dir", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    feature_dir = Path(args.features_dir) if args.features_dir else out_dir / FEATURE_DIR
    if not feature_dir.is_absolute():
        feature_dir = PROJECT_ROOT / feature_dir

    report = ForecastReport()
    report.p("CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER")
    report.p("Phase 4: forecasting, forward-in-time validation and submission")
    report.p(f"feature dir : {feature_dir}")
    report.p(f"out dir     : {out_dir}")

    required = [feature_dir / "features_monthly.parquet", out_dir / "clean" / "demand_monthly.parquet",
                out_dir / "clean" / "test_clean.parquet"]
    for path in required:
        if not path.exists():
            print(f"FATAL: missing upstream output: {path}", file=sys.stderr)
            return 2

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    tables = load_inputs(out_dir, feature_dir, data_dir)
    panel = build_panel(tables, report)
    scored, backtest_state = backtest(panel, tables, report)
    cells, final_state = final_forecast(panel, tables, backtest_state, report)
    cells, monthly, band = add_uncertainty(cells, scored, backtest_state["selected"], report)
    submission = build_submission(cells, tables["submission_columns"])

    # ---- exit checks ------------------------------------------------------ #
    report.h("5. PHASE 4 EXIT CHECK")
    test = tables["test_clean"]
    dictionary = tables["dictionary"]

    expected_keys = test[["SKU_ID", "Region", "Month"]].copy()
    expected_keys["Month"] = pd.to_datetime(expected_keys["Month"]).astype("datetime64[ns]")
    submission_keys = pd.DataFrame({"SKU_ID": submission["SKU_ID"], "Region": submission["Region"],
                                    "Month": pd.to_datetime(submission["Date"] + "-01").astype("datetime64[ns]")})
    matched = expected_keys.merge(submission_keys.assign(__present=1), on=CELL, how="outer",
                                  indicator=True)
    key_match = (len(submission) == len(test)
                 and not submission.duplicated().any()
                 and bool(matched["_merge"].eq("both").all()))

    post_eol_rows = cells.loc[cells["post_eol"].fillna(False).astype(bool)]
    zero_post_eol = bool((post_eol_rows["forecast_units"] == 0).all())

    lag_values = panel.lags
    train_mask = (panel.known["split"].eq("train")).to_numpy()
    parity = {}
    for column in PARITY_LAG_FEATURES:
        expected_values = panel.known[column].to_numpy(float)[train_mask]
        actual_values = lag_values[train_mask, LAG_FEATURES.index(column)]
        same = ((np.isnan(expected_values) & np.isnan(actual_values))
                | np.isclose(expected_values, actual_values, rtol=1e-9, atol=1e-6))
        parity[column] = float(np.mean(same))

    first_horizon = final_state["horizon_positions"][0]
    last_observed = panel.demand[:, first_horizon - 1]
    first_forecast_rows = panel.rows_by_position[first_horizon]
    horizon_lag = lag_block(panel.demand, panel.zero, first_horizon)[
        panel.pair_code[first_forecast_rows], LAG_FEATURES.index("lag_1")]
    expected_lag = last_observed[panel.pair_code[first_forecast_rows]]
    lag_one_ok = bool(((np.isnan(horizon_lag) & np.isnan(expected_lag))
                       | np.isclose(horizon_lag, expected_lag, equal_nan=True)).all())

    lagged_used = [column for column in KNOWN_FEATURES
                   if column in dictionary["column"].values
                   and dictionary.set_index("column").loc[column, "leakage_class"] == "lagged"]
    target_used = [column for column in KNOWN_FEATURES if column.startswith("target_")]

    scores = backtest_state["scores"]
    eligible = scores.loc[scores["bias"].abs() <= BIAS_TOLERANCE]
    best = (eligible if len(eligible) else scores).sort_values("wape").iloc[0]["model"]
    ordered = bool((cells["forecast_p10"] <= cells["forecast_p50"]).all()
                   and (cells["forecast_p50"] <= cells["forecast_p90"]).all()
                   and (cells["forecast_p10"] >= 0).all())

    report.verdict("submission keys equal the test keys", key_match,
                   f"{len(submission):,} rows vs {len(test):,}")
    report.verdict("submission columns match the sample format",
                   list(submission.columns) == tables["submission_columns"],
                   ", ".join(submission.columns))
    report.verdict("submission is non-negative whole units",
                   bool((submission["Units_Sold_Forecast"] >= 0).all())
                   and submission["Units_Sold_Forecast"].dtype.kind in "iu",
                   f"{submission['Units_Sold_Forecast'].sum():,.0f} units total")
    report.verdict("post-EOL keys are forecast at exactly zero", zero_post_eol,
                   f"{len(post_eol_rows):,} keys")
    report.verdict("every test key has a forecast",
                   bool(cells["forecast_units"].notna().all() and len(cells) == len(test)),
                   f"{len(cells):,} rows")
    report.verdict("backtest fits strictly before the scored months",
                   bool((panel.known["Month"].to_numpy()[backtest_state["fit_rows"]]
                         < panel.months[backtest_state["origin_position"]]).all()),
                   f"fit < {_month_text(panel.months[backtest_state['origin_position']])}")
    report.verdict("backtest scores the full 19-month horizon",
                   int(scored["Month"].nunique()) == BACKTEST_MONTHS,
                   f"{scored['Month'].nunique()} months, {len(scored):,} rows")
    report.verdict("recursive lags reproduce Phase 3 on training months",
                   all(value > 0.999 for value in parity.values()),
                   f"min agreement {min(parity.values()):.5f} across "
                   f"{len(PARITY_LAG_FEATURES)} columns")
    report.verdict("horizon month 1 lag_1 is the last observed demand", lag_one_ok,
                   f"{int(np.isfinite(expected_lag).sum()):,} keys with history")
    report.verdict("only known-at-t columns are taken from the feature table",
                   not lagged_used and not target_used,
                   f"{len(KNOWN_FEATURES)} known columns, {len(lagged_used)} lagged, "
                   f"{len(target_used)} target")
    report.verdict("model selection follows the backtest rule",
                   str(best) == str(backtest_state["selected"]),
                   f"selected {backtest_state['selected']} by WAPE with |bias| <= "
                   f"{BIAS_TOLERANCE:.0%}")
    report.verdict("blend weight is chosen outside the scored months",
                   backtest_state["inner_position"] + BACKTEST_MONTHS
                   <= backtest_state["origin_position"],
                   f"w={backtest_state['blend_weight']:.1f} chosen on "
                   f"{_month_text(panel.months[backtest_state['inner_position']])} -> "
                   f"{_month_text(panel.months[backtest_state['inner_position']
                                              + BACKTEST_MONTHS - 1])}")
    report.verdict("forecast path is reproducible",
                   final_state["reproducible"], "identical predictions on a second pass")
    report.verdict("uncertainty bands are ordered and non-negative", ordered)
    report.verdict("every forecast cell is priced and revenue reconciles",
                   bool(cells["forecast_revenue"].notna().all())
                   and abs(float(monthly["revenue"].sum())
                           - float(cells["forecast_revenue"].sum())) <= 1.0
                   # revenue is rounded per cell, so the identity closes to half a
                   # dollar per key rather than to the cent on the grand total
                   and abs(float(cells["forecast_revenue"].sum())
                           - float((cells["forecast_units"]
                                    * cells["unit_price"].fillna(0.0)).sum()))
                   <= 0.5 * len(cells),
                   f"{fmt(cells['forecast_revenue'].sum())} USD at the list price")
    report.verdict("backtest error is sliced by family, region and horizon",
                   {"product_family", "region", "horizon_months"}
                   <= set(backtest_state["slices"]["dimension"])
                   and bool(backtest_state["slices"]["wape"].notna().all()),
                   f"{len(backtest_state['slices'])} slice rows")
    report.verdict("submission grain is unique", not submission.duplicated().any())

    naive_wape = float(scores.loc[scores["model"].eq("seasonal_naive"), "wape"].iloc[0])
    chosen_wape = float(scores.loc[scores["model"].eq(backtest_state["selected"]), "wape"].iloc[0])
    trailing = (panel.known.loc[(panel.known["split"].eq("train"))
                                & (panel.known["Month"] > panel.months[first_horizon - 12])]
                ["target_demand"].sum()) / 12
    monthly_total = monthly["forecast"].sum() / len(monthly)
    plausible = 0.5 <= _safe_divide(monthly_total, trailing) <= 2.0
    report.verdict("forecast level is plausible against the trailing year", plausible,
                   f"{monthly_total:,.0f}/month vs trailing {trailing:,.0f}/month")
    report.kv("backtest WAPE selected / naive", f"{chosen_wape:.4f} / {naive_wape:.4f} "
                                               f"({1 - chosen_wape / naive_wape:+.1%})")

    report.kv("elapsed seconds", round(time.time() - started, 1))

    diagnostics = {
        "seed": SEED, "ridge_lambda": RIDGE_LAMBDA, "lightgbm": LGBM_PARAMS,
        "selected": backtest_state["selected"],
        "blend_weight": backtest_state["blend_weight"],
        "blend_weight_curve": backtest_state["weight_curve"].to_dict(orient="records"),
        "blend_weight_selected_on": _month_text(
            panel.months[backtest_state["inner_position"]]),
        "holdout_origin": _month_text(panel.months[backtest_state["origin_position"]]),
        "features": {"known": KNOWN_FEATURES, "recursive": LAG_FEATURES,
                     "categorical": CATEGORICAL_FEATURES},
        "lag_parity": parity,
        "scores": backtest_state["scores"].to_dict(orient="records"),
        "uncertainty": band,
        "post_eol_keys": int(len(post_eol_rows)),
        "fit_rows_final": int(len(final_state["fit_rows"])),
        "importances": final_state["candidates"]["hurdle_lgbm"].importances(
            panel.feature_names + LAG_FEATURES),
    }

    write_outputs(out_dir, report, submission, cells, monthly, scores, scored,
                  backtest_state["monthly"], backtest_state["slices"], diagnostics)

    print("\n".join(report.lines))
    print(f"\nSUBMISSION -> {out_dir / 'submission.csv'}")
    print(f"TABLES     -> {out_dir / FORECAST_DIR}")
    failed = [v for v in report.verdicts if not v["ok"]]
    if failed:
        print(f"\nPHASE 4 FAILED {len(failed)} check(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
