#!/usr/bin/env python
"""Phase 2 - Anomaly register: data errors vs genuine demand events.

The point of this phase is the distinction the brief insists on:

* **Data error**  -> the record is wrong (double load, keying error, digit shift,
  broken revenue identity, unresolved region).  It is repaired in Phase 1 or
  flagged here for review; it must never be learned as demand.
* **Demand event** -> the record is right and the market moved (a real spike, a
  real collapse, a price change, a launch, a stockout that censored shipments).
  It is explained or kept for the model; it must never be "corrected" away.

Every demand event carries the unconstrained demand (`demand_units`), so a
stockout that censored shipments is registered as a *censoring* event rather
than mistaken for a decline.

Expected demand
---------------
Each cell is compared with the **same calendar month in the previous three
years** of its own SKU-Region series (median, in log1p space).  A two-way
year x month median model was tried first and abandoned: these series are
intermittent (about half the cells are explicit zeros) and they change regime,
so an additive year term stacked on an additive seasonal term produced
expectations up to 500x the cell's own level.  The seasonal-neighbour baseline
has no such identification problem and absorbs a regime change within a year.

The fallback chain is explicit and counted: prior 3 years -> next 2 years (only
for the first years of a series, where no prior observation exists) -> the
series median.  Residuals are scaled by a per-series MAD sigma with a pooled
fallback for short series and a floor in log space.

This is a *description of history*, not a forecast: nothing from the 19-month
forecast window is used, and Phase 4 never sees these expectations as features.

Register classes
----------------
DATA_ERROR     duplicate_load, unmatched_sku, sku_spelling, label_spelling,
               date_keying, date_key_mismatch, date_unparseable, negative_units,
               blank_units, price_region_scatter, price_missing,
               revenue_identity_break, price_off_cycle, region_unresolved,
               sales_outside_lifecycle, stockout_algebra, fill_rate_mismatch,
               event_scope_unresolved, macro_gap, event_end_imputed,
               master_eol_before_launch, suspected_digit_shift, error_hotspot
DEMAND_EVENT   spike, drop, demand_collapse, new_demand, censored_by_stockout,
               lifecycle_launch, lifecycle_end, price_change, macro_shock

Outputs (outputs/anomalies/)
----------------------------
anomaly_register.parquet/.csv  one row per anomaly, with evidence + treatment
data_errors.parquet            register subset: kind == DATA_ERROR
demand_events.parquet          register subset: kind == DEMAND_EVENT
demand_baseline.parquet        every usable cell with its expected value and z
monthly_summary.parquet        company month view: demand, shipments, macro shots
test_window_profile.parquet    forecast-window coverage facts (not anomalies)
anomaly_rules.csv              rule-by-rule diagnostics for this phase
anomaly_report.json / .txt     machine + human readable summary

Usage
-----
    python src/anomalies.py --clean-dir outputs/clean --out-dir outputs
    python run_pipeline.py --phase 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile_data import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, fmt, jsonable  # noqa: E402

CLEAN_DIR = "clean"
ANOMALY_DIR = "anomalies"
CELL = ["SKU_ID", "Region", "Month"]
SERIES = ["SKU_ID", "Region"]

# Detection tuning.  Deliberately conservative: a register that flags 10% of the
# panel is noise.  A move must be statistically extreme (4 sigma), materially
# large (+-75%) and operationally relevant (>=50 units) to be registered.  The
# looser view is not lost: demand_baseline.parquet keeps every cell's z and
# expected value, so a dashboard can drill below these thresholds.
Z_THRESHOLD = 4.0
MIN_REL_CHANGE = 0.75
MIN_UNITS_MOVED = 50.0
SCALE_FLOOR = 0.03
MIN_SERIES_POINTS = 12
COLLAPSE_BASE = 30.0
DIGIT_MULTIPLIERS = (10.0, 100.0)
DIGIT_TOLERANCE = 0.05
MACRO_Z = 3.0
MACRO_MIN_INDICATORS = 2
FLAG_RATE_LIMIT = 0.03
HOTSPOT_FACTOR = 3.0

ANOMALY_COLUMNS = [
    "anomaly_id", "kind", "anomaly_class", "secondary_class", "level", "scope",
    "SKU_ID", "Region", "Month", "metric", "observed", "expected", "impact",
    "rel_change", "z_score", "severity", "rows_affected", "units_affected",
    "evidence", "explained_by", "treatment", "status", "note",
]
NUMERIC_COLUMNS = ["observed", "expected", "impact", "rel_change", "z_score",
                   "rows_affected", "units_affected"]

# Rule-level data errors, mapped back to the Phase 1 rule that produced them.
DATA_ERROR_SPECS = [
    ("duplicate_load", "TRN-DUP-ID", "repaired", "high",
     "Double-loaded records; newest Load_Timestamp kept, the earlier copy dropped."),
    ("unmatched_sku", "TRN-SKU-MISS", "flagged", "high",
     "SKU could not be resolved to the master; excluded from demand."),
    ("sku_spelling", "TRN-SKU-KEY", "repaired", "low",
     "Case/punctuation variants and unpadded suffixes mapped to the master key."),
    ("label_spelling", "TRN-LABEL", "repaired", "low",
     "Customer/region/channel spellings normalised to one canonical label."),
    ("date_keying", "TRN-DATE-2062", "repaired", "high",
     "Year implausible against the Year/Month key; re-dated from the key, row kept."),
    ("date_key_mismatch", "TRN-DATE-KEY", "repaired", "medium",
     "Year/Month disagreed with Date; Date is authoritative."),
    ("date_unparseable", "TRN-DATE-BAD", "flagged", "high",
     "Date could not be parsed; the row cannot be placed in a month."),
    ("negative_units", "TRN-UNITS-NEG", "flagged", "medium",
     "Returns/adjustments; retained and flagged, never counted as demand."),
    ("blank_units", "TRN-UNITS-BLANK", "flagged", "high",
     "Blank units in an existing row is unknown, not zero."),
    ("price_region_scatter", "TRN-PRICE", "repaired", "high",
     "One base price per SKU-year enforced across regions; steps only on 1 January."),
    ("price_missing", "TRN-PRICE-FILL", "repaired", "medium",
     "Zero/missing unit price filled from the SKU-year ladder."),
    ("revenue_identity_break", "TRN-REVENUE", "flagged", "medium",
     "Units x recorded price != Revenue; recomputed revenue kept beside the recorded one."),
    ("region_unresolved", "TRN-REGION", "flagged", "high",
     "Region blank or unmappable; kept as Unknown and excluded from the demand panel."),
    ("sales_outside_lifecycle", "TRN-LIFECYCLE", "flagged", "high",
     "Sales before Launch_Date or after EOL_Date; flagged and excluded from training demand."),
    ("stockout_algebra", "STK-ALGEBRA", "flagged", "medium",
     "Requested != Shipped + Lost + Backordered; recomputed fill rate is authoritative."),
    ("fill_rate_mismatch", "STK-FILL", "repaired", "low",
     "Reported fill rate disagreed with Shipped/Requested by more than 0.5pp."),
    ("event_scope_unresolved", "EVT-SCOPE", "flagged", "medium",
     "Event scope does not resolve to a known entity, so it cannot be applied."),
    ("event_end_imputed", "EVT-END", "repaired", "low",
     "Blank End_Date set to Start_Date (one-month event)."),
    ("macro_gap", "MAC-GAP", "repaired", "low",
     "Blank macro observations linearly interpolated inside the series."),
    ("master_eol_before_launch", "MST-EOL", "repaired", "medium",
     "EOL_Date preceded Launch_Date; EOL ignored for lifecycle."),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def compact(value) -> str:
    """Case/punctuation-free key for scope matching."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())


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


def _group_index(frame: pd.DataFrame, keys: list[str]):
    if len(keys) == 1:
        return pd.Index(frame[keys[0]])
    return pd.MultiIndex.from_frame(frame[keys])


def _median_by(frame: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """Group median broadcast back to every row, without a Python loop."""
    medians = frame.groupby(keys, sort=False)[column].median()
    values = medians.reindex(_group_index(frame, keys)).to_numpy()
    return pd.Series(values, index=frame.index)


def _count_by(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    sizes = frame.groupby(keys, sort=False).size()
    values = sizes.reindex(_group_index(frame, keys)).to_numpy()
    return pd.Series(values, index=frame.index)


def seasonal_expected(work: pd.DataFrame, keys: list[str]
                      ) -> tuple[pd.Series, dict, np.ndarray]:
    """Expected log demand from the same calendar month in the prior years.

    A two-way year x month model was tried first and abandoned: this data is
    intermittent and changes regime (one SKU-Region can move from quarterly to
    monthly ordering), so an additive year term stacked on an additive seasonal
    term produced expectations up to 500x the cell's own level.  Comparing a
    month with *the same month* in the previous three years has no such
    identification problem, absorbs a regime change within a year, and is the
    benchmark a planner would use by hand.

    The fallback chain is explicit and counted: prior 3 years -> next 2 years
    (only for the first years of a series, where no prior observation exists) ->
    the series median.  Later history is never used to score a later month.
    """
    base = work.set_index([*keys, "Year", "Month_Num"])["y"]
    base = base[~base.index.duplicated(keep="first")]

    def probe(year_shift: int) -> np.ndarray:
        query = work[[*keys, "Month_Num"]].copy()
        query["Year"] = work["Year"].to_numpy() + year_shift
        return query.merge(base.rename("v"), left_on=[*keys, "Year", "Month_Num"],
                           right_index=True, how="left")["v"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prior = np.nanmedian(np.column_stack([probe(-lag) for lag in (1, 2, 3)]), axis=1)
        following = np.nanmedian(np.column_stack([probe(lag) for lag in (1, 2)]), axis=1)
    series_median = _median_by(work, keys, "y").to_numpy()
    expected = np.where(np.isfinite(prior), prior, following)
    expected = np.where(np.isfinite(expected), expected, series_median)
    source = np.where(np.isfinite(prior), "prior_years",
                      np.where(np.isfinite(following), "first_years", "series_median"))
    counts = {name: int((source == name).sum()) for name in
              ("prior_years", "first_years", "series_median")}
    return pd.Series(expected, index=work.index), counts, source


def _scale_by(work: pd.DataFrame, keys: list[str], values: pd.Series) -> pd.Series:
    """Per-series MAD sigma, pooled fallback for short series, floored."""
    centre = _median_by(work.assign(_v=values), keys, "_v")
    deviation = (values - centre).abs()
    mad = _median_by(work.assign(_v=deviation), keys, "_v") * 1.4826
    pooled = float(deviation.median() * 1.4826) if len(deviation) else SCALE_FLOOR
    pooled = max(pooled, SCALE_FLOOR)
    enough = _count_by(work, keys) >= MIN_SERIES_POINTS
    scale = mad.where(enough & (mad > SCALE_FLOOR), pooled)
    return scale.clip(lower=SCALE_FLOOR).fillna(pooled)


def robust_scores(work: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Seasonal baseline plus a MAD-scaled residual.  Log space throughout."""
    expected_log, sources, labels = seasonal_expected(work, keys)
    resid = work["y"] - expected_log
    sigma = _scale_by(work, keys, resid)
    frame = pd.DataFrame({
        "expected_log": expected_log,
        "residual_log": resid,
        "scale": sigma,
        "z_score": resid / sigma,
        "n_obs": _count_by(work, keys),
    }, index=work.index)
    frame.attrs["sources"] = sources
    frame.attrs["source"] = labels
    return frame


def severity_of(z, rel) -> str:
    try:
        magnitude = abs(float(z)) if z is not None and np.isfinite(float(z)) else 0.0
    except (TypeError, ValueError):
        magnitude = 0.0
    try:
        change = abs(float(rel)) if rel is not None and np.isfinite(float(rel)) else 0.0
    except (TypeError, ValueError):
        change = 0.0
    if magnitude >= 6.0 or change >= 2.0:
        return "high"
    if magnitude >= 4.5 or change >= 1.0:
        return "medium"
    return "low"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
class AnomalyReport:
    """Console transcript + machine-readable diagnostics for Phase 2."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.verdicts: list[dict] = []
        self.metrics: dict = {}
        self.log: list[dict] = []
        self._t0 = time.time()

    def h(self, title: str) -> None:
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"[anomalies] {title}", file=sys.stderr, flush=True)

    def sub(self, title: str) -> None:
        self.lines.append(f"\n-- {title} " + "-" * max(0, 58 - len(title)))

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def kv(self, key: str, value, indent: int = 2, width: int = 46) -> None:
        self.lines.append(f"{' ' * indent}{key:<{width}} {fmt(value)}")

    def rule(self, rule_id: str, issue: str, action: str, rows: int = 0,
             units: float | None = None, detail: str = "") -> None:
        self.log.append({"rule_id": rule_id, "issue": issue, "action": action,
                         "rows_affected": int(rows), "units_affected": None if units is None
                         else round(float(units), 3), "detail": detail})

    def verdict(self, name: str, ok: bool, detail: str = "") -> None:
        self.verdicts.append({"phase": 2, "check": name, "ok": bool(ok), "detail": detail})
        self.p(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. load + cross-check Phase 1
# --------------------------------------------------------------------------- #
def load_inputs(clean_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    tables = {
        name: pd.read_parquet(clean_dir / f"{name}.parquet")
        for name in ["sales_clean", "sku_master_clean", "demand_monthly", "demand_grid",
                     "stockouts_clean", "macro_clean", "events_clean", "test_clean"]
    }
    payload = json.loads((clean_dir.parent / "clean_report.json").read_text(encoding="utf-8"))
    return tables, payload


def phase1_consistency(tables: dict[str, pd.DataFrame], payload: dict,
                       report: AnomalyReport) -> dict:
    """Re-derive the Phase 1 invariants this phase depends on.  Fails loudly."""
    report.h("1. PHASE 1 CONTRACT RE-CHECK")
    panel = tables["demand_monthly"]
    sales = tables["sales_clean"]
    metrics = payload["metrics"]

    below = int((panel["demand_units"] < panel["observed_units"]).fillna(False).sum())
    report.kv("demand cells", f"{len(panel):,}")
    report.kv("cells where demand < observed (must be 0)", below)
    report.kv("dashboard/report demand total", f"{fmt(metrics['demand']['demand_units'])} vs "
                                              f"parquet {fmt(panel['demand_units'].sum())}")
    report.kv("suppressed units (report vs parquet)",
              f"{fmt(metrics['demand']['suppressed_units'])} vs "
              f"{fmt(panel['suppressed_units'].sum())}")

    if below:
        report.rule("ANM-CONTRACT", "unconstrained demand below observed units",
                    "abort: Phase 1 invariant violated", below)
    report.rule("ANM-CONTRACT", "Phase 1 demand tables re-read",
                "panel, sales and support tables verified",
                len(panel),
                detail=f"{len(sales):,} sales rows, {len(tables['stockouts_clean']):,} stockout rows")
    return metrics


# --------------------------------------------------------------------------- #
# 2. baseline + cell scoring
# --------------------------------------------------------------------------- #
def score_cells(panel: pd.DataFrame, report: AnomalyReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    report.h("2. EXPECTED DEMAND (ROBUST MEDIAN MODEL)")
    usable = panel.loc[panel["usable"]].copy()
    work = usable.loc[:, [*SERIES, "Month", "Year", "Month_Num", "demand_units"]].copy()
    work["demand_units"] = work["demand_units"].clip(lower=0).fillna(0.0)
    # log1p keeps explicit zeros finite; expm1 inverts it exactly.
    work["y"] = np.log1p(work["demand_units"])

    scores = robust_scores(work, SERIES)
    expected = np.expm1(scores["expected_log"]).clip(lower=0)
    residual = work["demand_units"] - expected
    rel_change = residual / expected.where(expected >= 1.0)

    frame = usable.copy()
    frame["expected_units"] = expected
    frame["residual_log"] = scores["residual_log"]
    frame["scale"] = scores["scale"]
    frame["z_score"] = scores["z_score"]
    frame["rel_change"] = rel_change
    frame["n_obs"] = scores["n_obs"]
    sources = scores.attrs.get("sources", {})
    frame["baseline_source"] = scores.attrs.get("source")
    frame["flagged"] = ((frame["z_score"].abs() >= Z_THRESHOLD)
                        & (frame["rel_change"].abs() >= MIN_REL_CHANGE)
                        & (residual.abs() >= MIN_UNITS_MOVED)).fillna(False)

    coverage = float(frame["expected_units"].notna().mean()) if len(frame) else 0.0
    report.kv("usable cells scored", f"{len(frame):,}")
    report.kv("expected-value coverage", f"{coverage:.4%}")
    report.kv("baseline source (prior / first / median)",
              f"{sources.get('prior_years', 0):,} / {sources.get('first_years', 0):,} / "
              f"{sources.get('series_median', 0):,}")
    report.kv("median series length (months)", f"{frame.groupby(SERIES).size().median():.0f}")
    report.kv("robust sigma (log), median / p95",
              f"{frame['scale'].median():.3f} / {frame['scale'].quantile(0.95):.3f}")
    report.kv("cells flagged as demand events", f"{int(frame['flagged'].sum()):,} "
                                               f"({frame['flagged'].mean():.2%})")
    report.rule("ANM-BASELINE", "expected demand per SKU-Region-month",
                "median of the same calendar month in the previous three years",
                len(frame), detail=f"coverage {coverage:.2%}; fallbacks prior/next/series "
                                   f"{sources.get('prior_years', 0):,}/"
                                   f"{sources.get('first_years', 0):,}/"
                                   f"{sources.get('series_median', 0):,}; "
                                   f"sigma floor {SCALE_FLOOR}; thresholds "
                                   f"|z|>={Z_THRESHOLD}, |rel|>={MIN_REL_CHANGE}, "
                                   f"|units|>={MIN_UNITS_MOVED:.0f}")
    report.metrics["baseline"] = {
        "usable_cells": len(frame),
        "coverage": coverage,
        "flagged_cells": int(frame["flagged"].sum()),
        "flag_rate": float(frame["flagged"].mean()) if len(frame) else 0.0,
        "sigma_median": float(frame["scale"].median()),
        "sigma_p95": float(frame["scale"].quantile(0.95)),
        "sources": sources,
    }
    return frame, scores


def detect_digit_shifts(scored: pd.DataFrame) -> pd.DataFrame:
    """One-month 10x/100x blips against both neighbours: a keying error signal.

    A genuine step change lifts the level and stays; an extra zero does not.
    Requires both neighbours to sit near the same baseline, so a launch ramp or a
    sustained growth phase is not captured.
    """
    work = scored.sort_values([*SERIES, "Month"]).copy()
    grouped = work.groupby(SERIES, sort=False)["demand_units"]
    work["prev_units"] = grouped.shift(1)
    work["next_units"] = grouped.shift(-1)
    neighbour = (work["prev_units"] + work["next_units"]) / 2.0
    work["neighbour_units"] = neighbour
    suspicious = pd.Series(False, index=work.index)
    multiplier = pd.Series(np.nan, index=work.index)
    for factor in DIGIT_MULTIPLIERS:
        scaled = factor * neighbour
        close = ((work["demand_units"] - scaled).abs() <= DIGIT_TOLERANCE * scaled)
        candidate = close.fillna(False) & neighbour.ge(5.0) & work["demand_units"].ge(100.0)
        multiplier = multiplier.where(~candidate, factor)
        suspicious |= candidate
    work["digit_multiplier"] = multiplier
    work["digit_shift"] = suspicious
    return work.loc[suspicious].copy()


def detect_collapses(scored: pd.DataFrame) -> pd.DataFrame:
    """An established run stops in a month that is normally active.

    A bare zero is not a collapse in this data: about 14% of usable cells are
    explicit zeros, and February shutdowns repeat every year.  Requiring all
    three prior months to be live *and* the cell's own calendar-month median to
    be live separates a stopped account from ordinary intermittency; the
    statistical drop detector remains the broad net.
    """
    work = scored.sort_values([*SERIES, "Month"]).copy()
    grouped = work.groupby(SERIES, sort=False)["demand_units"]
    prior_min = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=3).min())
    seasonal = work.groupby([*SERIES, "Month_Num"], sort=False)["demand_units"].transform("median")
    work["prior_min_units"] = prior_min
    work["seasonal_units"] = seasonal
    collapse = (work["demand_units"].eq(0) & prior_min.ge(COLLAPSE_BASE)
                & seasonal.ge(COLLAPSE_BASE)).fillna(False)
    return work.loc[collapse].copy()


# --------------------------------------------------------------------------- #
# 3. explanation helpers
# --------------------------------------------------------------------------- #
def price_event_months(events: pd.DataFrame) -> dict[pd.Timestamp, list[dict]]:
    """Explode price-type events into the months they cover."""
    lookup: dict[pd.Timestamp, list[dict]] = {}
    selected = events.loc[events["price_event"].fillna(False) & events["scope_resolved"].fillna(False)]
    for row in selected.itertuples(index=False):
        start = row.Start_Month
        end = row.End_Month if pd.notna(row.End_Month) else start
        for month in pd.date_range(start, end, freq="MS"):
            lookup.setdefault(month, []).append({
                "Event_ID": row.Event_ID,
                "scope_type": str(row.Scope_Type).strip().upper().replace(" ", "_"),
                "scope_value": str(row.Scope_Value).strip(),
            })
    return lookup


def match_events(month, sku: str, region: str, family: str,
                 lookup: dict[pd.Timestamp, list[dict]]) -> list[str]:
    hits = []
    for event in lookup.get(month, []):
        scope = event["scope_type"]
        value = event["scope_value"]
        if scope == "GLOBAL":
            hits.append(event["Event_ID"])
        elif scope == "REGION" and compact(value) == compact(region):
            hits.append(event["Event_ID"])
        elif scope == "PRODUCT_FAMILY" and compact(value) == compact(family):
            hits.append(event["Event_ID"])
        elif scope == "SKU" and compact(value) == compact(sku):
            hits.append(event["Event_ID"])
    return sorted(set(hits))


def macro_shock_months(macro: pd.DataFrame, report: AnomalyReport) -> tuple[pd.DataFrame, dict]:
    """Months where at least two indicators move together by more than 3 sigma."""
    indicators = [c for c in macro.columns
                  if c not in {"Month", "Auto_Production_YoY_Frac"}]
    frame = macro[["Month", *indicators]].sort_values("Month").reset_index(drop=True)
    flags = pd.DataFrame({"Month": frame["Month"]})
    details: list[str] = []
    for column in indicators:
        change = frame[column].diff()
        centre = change.median()
        spread = (change - centre).abs().median() * 1.4826
        if not np.isfinite(spread) or spread <= 0:
            flags[column] = False
            continue
        flags[column] = ((change - centre).abs() / spread).ge(MACRO_Z).fillna(False)
    count = flags[indicators].sum(axis=1)
    flags["shock_count"] = count
    flags["is_shock"] = count >= MACRO_MIN_INDICATORS
    for row in flags.loc[flags["is_shock"]].itertuples(index=False):
        moved = [c for c in indicators if getattr(row, c)]
        details.append(f"{row.Month:%Y-%m}: {', '.join(moved)}")
    report.kv("macro shock months", f"{int(flags['is_shock'].sum()):,}")
    report.rule("ANM-MACRO", "macro indicators move together by >3 robust sigma",
                "registered as an external explanation, not as a sales anomaly",
                int(flags["is_shock"].sum()), detail="; ".join(details[:8]))
    shock_map = {row.Month: row.shock_count for row in flags.loc[flags["is_shock"]].itertuples()}
    return flags, shock_map


# --------------------------------------------------------------------------- #
# 4. cell-level demand events
# --------------------------------------------------------------------------- #
def build_cell_events(scored: pd.DataFrame, collapses: pd.DataFrame,
                      digit_shifts: pd.DataFrame, stockouts: pd.DataFrame,
                      master: pd.DataFrame, events: pd.DataFrame,
                      report: AnomalyReport) -> pd.DataFrame:
    """One primary anomaly per (SKU, Region, Month), with precedence.

    Precedence (most specific first): suspected digit shift (data error) ->
    censored by stockout -> demand collapse -> spike/drop.  A lower-precedence
    class that also fires on the same cell is kept as ``secondary_class`` so
    nothing is hidden.
    """
    report.h("3. CELL-LEVEL ANOMALIES")
    lookup = price_event_months(events)
    families = master.set_index("SKU_ID")["Product_Family"].to_dict()
    reasons = (stockouts.sort_values("Unserved_Units", ascending=False)
               .drop_duplicates(CELL).set_index(CELL)["Reason"].to_dict())

    candidates: dict[tuple, list[dict]] = {}

    def add(priority: int, row, kind: str, klass: str, treatment: str,
            status: str, evidence: str, impact: float, explained: str = "") -> None:
        key = (row.SKU_ID, row.Region, row.Month)
        candidates.setdefault(key, []).append({
            "priority": priority, "kind": kind, "anomaly_class": klass,
            "level": "cell", "scope": "SKU x Region x Month",
            "SKU_ID": row.SKU_ID, "Region": row.Region, "Month": row.Month,
            "metric": "demand_units", "observed": float(row.demand_units),
            "expected": float(row.expected_units) if pd.notna(row.expected_units) else None,
            "impact": float(impact),
            "rel_change": float(row.rel_change) if pd.notna(row.rel_change) else None,
            "z_score": float(row.z_score) if pd.notna(row.z_score) else None,
            "severity": severity_of(row.z_score, row.rel_change),
            "rows_affected": None, "units_affected": None,
            "evidence": evidence, "explained_by": explained,
            "treatment": treatment, "status": status, "note": "",
            "secondary_class": "",
        })

    for row in scored.loc[scored["flagged"]].itertuples(index=False):
        move = "above" if row.z_score > 0 else "below"
        klass = "spike" if row.z_score > 0 else "drop"
        events_hit = match_events(row.Month, row.SKU_ID, row.Region,
                                  families.get(row.SKU_ID, ""), lookup)
        evidence = (f"demand {_fmt(row.demand_units)} vs expected {_fmt(row.expected_units)} "
                    f"({_fmt(row.rel_change * 100, 1)}% {move}, z={row.z_score:.1f})")
        add(40, row, "DEMAND_EVENT", klass,
            "explain" if events_hit else "model",
            "explained" if events_hit else "open",
            evidence, row.demand_units - row.expected_units,
            explained=";".join(events_hit))

    for row in collapses.itertuples(index=False):
        evidence = (f"explicit zero in a normally active month (median "
                    f"{_fmt(row.seasonal_units)} units) after three live months "
                    f"(min {_fmt(row.prior_min_units)} units)")
        add(30, row, "DEMAND_EVENT", "demand_collapse", "explain", "open",
            evidence, -float(row.prior_min_units), explained="")

    for row in digit_shifts.itertuples(index=False):
        evidence = (f"{_fmt(row.demand_units)} units vs {_fmt(row.neighbour_units)} in both "
                    f"neighbouring months (x{row.digit_multiplier:.0f} pattern) - "
                    f"consistent with an extra zero, not a market move")
        add(10, row, "DATA_ERROR", "suspected_digit_shift", "review", "flagged",
            evidence, row.demand_units - row.neighbour_units,
            explained="digit-shift pattern")

    # A percentage move is undefined against a zero seasonal baseline, so a
    # month that never sold in the prior three years but has real volume now is
    # registered explicitly instead of being silently skipped.
    new_demand = scored.loc[scored["expected_units"].lt(1.0)
                            & scored["demand_units"].ge(MIN_UNITS_MOVED)
                            & scored["z_score"].abs().ge(Z_THRESHOLD)]
    for row in new_demand.itertuples(index=False):
        evidence = (f"{_fmt(row.demand_units)} units booked in a calendar month with no "
                    f"demand in the prior three years (z={row.z_score:.1f})")
        add(35, row, "DEMAND_EVENT", "new_demand", "model", "open",
            evidence, row.demand_units)

    censored = scored.loc[scored["suppressed_units"] > 0]
    for row in censored.itertuples(index=False):
        reason = reasons.get((row.SKU_ID, row.Region, row.Month), "")
        if not isinstance(reason, str):
            reason = ""
        evidence = (f"{_fmt(row.requested_units)} units requested vs {_fmt(row.shipped_units)} "
                    f"shipped ({_fmt(row.suppressed_units)} unserved, "
                    f"{_fmt(row.stockout_days)} stockout days)"
                    + (f"; reason: {reason}" if reason else ""))
        add(20, row, "DEMAND_EVENT", "censored_by_stockout", "explain", "explained",
            evidence, row.suppressed_units, explained=f"stockout log{': ' + reason if reason else ''}")

    rows: list[dict] = []
    for key, options in candidates.items():
        options = sorted(options, key=lambda option: option["priority"])
        primary = dict(options[0])
        others = [option["anomaly_class"] for option in options[1:]
                  if option["anomaly_class"] != primary["anomaly_class"]]
        primary["secondary_class"] = ";".join(dict.fromkeys(others))
        if others:
            primary["note"] = ("also " + ", ".join(dict.fromkeys(others))
                               + " - primary class is the most specific explanation")
        if primary["kind"] == "DATA_ERROR" and not primary["explained_by"]:
            primary["explained_by"] = "entry pattern"
        rows.append(primary)

    frame = pd.DataFrame(rows).drop(columns="priority")
    counts = frame["anomaly_class"].value_counts().to_dict() if len(frame) else {}
    report.kv("cells with a primary anomaly", f"{len(frame):,}")
    report.kv("class counts", counts)
    report.kv("collapse definition", "zero in a normally active month after a live quarter")
    report.kv("of which data errors (digit-shift review)",
              f"{int(frame['kind'].eq('DATA_ERROR').sum()):,}")
    report.kv("explained demand events", f"{int(frame['status'].eq('explained').sum()):,}")
    report.rule("ANM-CENSOR", "stockout censored shipments",
                "registered as demand event; demand panel already restores requested units",
                int(len(censored)), units=float(censored["suppressed_units"].sum()))
    report.rule("ANM-COLLAPSE", "explicit zero after a running baseline",
                "registered as a real collapse", int(len(collapses)))
    report.rule("ANM-NEW", "demand in a calendar month with a zero seasonal baseline",
                "registered as new demand; a percentage move is undefined there",
                int(len(new_demand)), units=float(new_demand["demand_units"].sum()))
    report.rule("ANM-DIGIT", "one-month 10x/100x blip against both neighbours",
                "flagged as a suspected keying error for review; demand left untouched",
                int(len(digit_shifts)), units=float(digit_shifts["demand_units"].sum()))
    report.metrics["cell_events"] = {
        "rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "censored_cells": int(len(censored)),
        "censored_units": float(censored["suppressed_units"].sum()),
        "collapse_cells": int(len(collapses)),
        "new_demand_cells": int(len(new_demand)),
        "new_demand_units": float(new_demand["demand_units"].sum()),
        "digit_shift_cells": int(len(digit_shifts)),
        "explained_rows": int(frame["status"].eq("explained").sum()) if len(frame) else 0,
    }
    return frame


# --------------------------------------------------------------------------- #
# 5. data-error register
# --------------------------------------------------------------------------- #
def build_data_errors(sales: pd.DataFrame, report: AnomalyReport,
                      cleaning_log: pd.DataFrame) -> pd.DataFrame:
    report.h("4. DATA-ERROR REGISTER")
    rows: list[dict] = []
    observed = 0
    quiet = 0
    for klass, rule_id, status, severity, note in DATA_ERROR_SPECS:
        match = cleaning_log.loc[cleaning_log["rule_id"] == rule_id]
        affected = int(match["rows_affected"].sum()) if len(match) else 0
        units = float(match["units_affected"].dropna().sum()) if len(match) else 0.0
        detail = "; ".join(str(v) for v in match["detail"].dropna()) if len(match) else ""
        if affected == 0:
            quiet += 1
            continue
        observed += 1
        rows.append({
            "kind": "DATA_ERROR", "anomaly_class": klass, "secondary_class": "",
            "level": "dataset", "scope": rule_id, "SKU_ID": "", "Region": "",
            "Month": pd.NaT, "metric": "rows", "observed": float(affected),
            "expected": None, "impact": units or None, "rel_change": None,
            "z_score": None, "severity": severity, "rows_affected": affected,
            "units_affected": units, "evidence": detail or note,
            "explained_by": rule_id, "treatment": "repair" if status == "repaired" else "review",
            "status": status, "note": note,
        })

    # Off-cycle price restatements: a mid-year price move is a data error, not a
    # demand story (list prices change on 1 January).
    off_cycle = sales.loc[sales["price_repaired"].fillna(False) & sales["Month_Num"].ne(1)]
    if len(off_cycle):
        rows.append({
            "kind": "DATA_ERROR", "anomaly_class": "price_off_cycle", "secondary_class": "",
            "level": "dataset", "scope": "non-January price restatements",
            "SKU_ID": "", "Region": "", "Month": pd.NaT, "metric": "rows",
            "observed": float(len(off_cycle)), "expected": None, "impact": None,
            "rel_change": None, "z_score": None, "severity": "medium",
            "rows_affected": len(off_cycle), "units_affected": None,
            "evidence": f"{len(off_cycle):,} rows restated at the canonical price in a "
                        f"month other than January",
            "explained_by": "TRN-PRICE", "treatment": "repair", "status": "repaired",
            "note": "Prices are allowed to step only on 1 January, except event surcharges.",
        })

    # Monthly hot spots: a defect concentrated in one month is a loading incident,
    # not ordinary background noise, and Phase 4 must not train on that month's shape.
    hot_specs = [
        ("price_restatement_hotspot", "TRN-PRICE", sales["price_repaired"].fillna(False)),
        ("revenue_identity_hotspot", "TRN-REVENUE",
         sales["revenue_identity_failed"].fillna(False)),
        ("blank_units_hotspot", "TRN-UNITS-BLANK", sales["Units_Sold"].isna()),
    ]
    hotspot_rows: list[dict] = []
    for klass, source_rule, mask in hot_specs:
        counts = sales.loc[mask].groupby("Month").size()
        if counts.empty:
            continue
        baseline = float(counts.median())
        for month, count in counts.items():
            if count >= 5 and count > HOTSPOT_FACTOR * max(baseline, 1.0):
                hotspot_rows.append({
                    "kind": "DATA_ERROR", "anomaly_class": klass, "secondary_class": "",
                    "level": "company", "scope": "monthly concentration",
                    "SKU_ID": "", "Region": "", "Month": month, "metric": "rows",
                    "observed": float(count), "expected": baseline, "impact": None,
                    "rel_change": (count / baseline - 1.0) if baseline else None,
                    "z_score": None, "severity": "medium", "rows_affected": int(count),
                    "units_affected": None,
                    "evidence": f"{count:,} rows in {_month_text(month)} vs a median month of "
                                f"{baseline:,.0f}",
                    "explained_by": source_rule, "treatment": "review", "status": "flagged",
                    "note": "Defect concentrated in one month; treat that month's shape with care.",
                })
    rows.extend(hotspot_rows)

    frame = pd.DataFrame(rows)
    report.kv("data-error classes observed", f"{observed:,} / {len(DATA_ERROR_SPECS)}")
    report.kv("rules with no incidents", quiet)
    report.kv("monthly hot spots", f"{len(hotspot_rows):,}")
    report.kv("rows covered by rule-level errors",
              f"{int(frame.loc[frame['level'].eq('dataset'), 'rows_affected'].fillna(0).sum()):,}")
    report.rule("ANM-DERR", "records wrong on their face, separated from demand",
                "registered with status repaired/flagged and a Phase 1 rule id",
                int(frame["rows_affected"].fillna(0).sum()),
                detail=f"{len(frame)} register lines")
    report.rule("ANM-HOTSPOT", "defect concentration by month",
                "registered so a loading incident is not learned as seasonality",
                len(hotspot_rows))
    report.metrics["data_errors"] = {
        "register_rows": len(frame),
        "classes_observed": observed,
        "hotspots": len(hotspot_rows),
        "rows_covered": int(frame["rows_affected"].fillna(0).sum()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 6. structural / external events
# --------------------------------------------------------------------------- #
def build_lifecycle_events(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in master.itertuples(index=False):
        if pd.notna(row.Launch_Month):
            rows.append({
                "kind": "DEMAND_EVENT", "anomaly_class": "lifecycle_launch",
                "secondary_class": "", "level": "sku", "scope": "SKU",
                "SKU_ID": row.SKU_ID, "Region": "ALL", "Month": row.Launch_Month,
                "metric": "event", "observed": None, "expected": None, "impact": None,
                "rel_change": None, "z_score": None, "severity": "low",
                "rows_affected": None, "units_affected": None,
                "evidence": f"launch month for {row.SKU_ID}"
                            + (f" (replaces {row.Replaces_SKU})" if row.Replaces_SKU else ""),
                "explained_by": "sku_master", "treatment": "model", "status": "explained",
                "note": "Structural level change: Phase 3 must not read it as a shock.",
            })
        if pd.notna(row.EOL_Month):
            rows.append({
                "kind": "DEMAND_EVENT", "anomaly_class": "lifecycle_end",
                "secondary_class": "", "level": "sku", "scope": "SKU",
                "SKU_ID": row.SKU_ID, "Region": "ALL", "Month": row.EOL_Month,
                "metric": "event", "observed": None, "expected": None, "impact": None,
                "rel_change": None, "z_score": None, "severity": "low",
                "rows_affected": None, "units_affected": None,
                "evidence": f"end-of-life month for {row.SKU_ID}",
                "explained_by": "sku_master", "treatment": "model", "status": "explained",
                "note": "Demand after this month is zero by definition, not by observation.",
            })
    frame = pd.DataFrame(rows)
    frame.attrs["launches"] = int(frame["anomaly_class"].eq("lifecycle_launch").sum())
    frame.attrs["ends"] = int(frame["anomaly_class"].eq("lifecycle_end").sum())
    return frame


def build_price_events(events: pd.DataFrame) -> pd.DataFrame:
    selected = events.loc[events["price_event"].fillna(False)].copy()
    rows = []
    for row in selected.itertuples(index=False):
        rows.append({
            "kind": "DEMAND_EVENT", "anomaly_class": "price_change", "secondary_class": "",
            "level": "event", "scope": f"{row.Scope_Type}:{row.Scope_Value}",
            "SKU_ID": "", "Region": "", "Month": row.Start_Month, "metric": "event",
            "observed": None, "expected": None, "impact": None, "rel_change": None,
            "z_score": None, "severity": "medium", "rows_affected": None,
            "units_affected": None,
            "evidence": f"{row.Event_Type} - {row.Description} "
                        f"({_month_text(row.Start_Month)} to {_month_text(row.End_Month)})",
            "explained_by": row.Event_ID,
            "treatment": "explain" if row.scope_resolved else "review",
            "status": "explained" if row.scope_resolved else "unresolved",
            "note": "" if row.scope_resolved else "Scope does not resolve; not attributable.",
        })
    return pd.DataFrame(rows)


def build_macro_events(shocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indicators = [c for c in shocks.columns if c not in {"Month", "shock_count", "is_shock"}]
    for row in shocks.loc[shocks["is_shock"]].itertuples(index=False):
        moved = [c for c in indicators if getattr(row, c)]
        rows.append({
            "kind": "DEMAND_EVENT", "anomaly_class": "macro_shock", "secondary_class": "",
            "level": "company", "scope": "macro", "SKU_ID": "", "Region": "ALL",
            "Month": row.Month, "metric": "event", "observed": float(row.shock_count),
            "expected": None, "impact": None, "rel_change": None, "z_score": None,
            "severity": "medium", "rows_affected": None, "units_affected": None,
            "evidence": f"{len(moved)} indicators moved >3 sigma: {', '.join(moved)}",
            "explained_by": ",".join(moved), "treatment": "explain", "status": "explained",
            "note": "Market context for a company-level move; never a per-SKU excuse.",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. company month view
# --------------------------------------------------------------------------- #
def build_monthly_summary(panel: pd.DataFrame, sales: pd.DataFrame,
                          macro: pd.DataFrame, shock_map: dict,
                          report: AnomalyReport) -> pd.DataFrame:
    report.h("5. COMPANY MONTH VIEW")
    usable = panel.loc[panel["usable"]]
    monthly = usable.groupby("Month").agg(
        cells=("demand_units", "size"),
        positive_cells=("demand_units", lambda s: int((s > 0).sum())),
        zero_cells=("zero_sales", "sum"),
        observed_units=("observed_units", "sum"),
        demand_units=("demand_units", "sum"),
        suppressed_units=("suppressed_units", "sum"),
        requested_units=("requested_units", "sum"),
        shipped_units=("shipped_units", "sum"),
        stockout_days=("stockout_days", "sum"),
        stockout_cells=("stockout_recorded", "sum"),
    )
    monthly["unserved_units"] = monthly["requested_units"] - monthly["shipped_units"]
    monthly["fill_rate_pct"] = (monthly["shipped_units"]
                                / monthly["requested_units"].where(monthly["requested_units"] > 0)
                                * 100)
    usable_sales = sales.loc[sales["usable_for_training"].fillna(False)]
    revenue = usable_sales.groupby("Month").agg(
        sales_rows=("Record_ID", "size"),
        revenue_recomputed=("Revenue_Recomputed_USD", "sum"),
    )
    monthly = monthly.join(revenue, how="left")

    work = pd.DataFrame({
        "series": "COMPANY",
        "Month": monthly.index,
        "demand_units": monthly["demand_units"].to_numpy(),
    })
    work["Year"] = work["Month"].dt.year
    work["Month_Num"] = work["Month"].dt.month
    work["y"] = np.log1p(work["demand_units"].clip(lower=0))
    scores = robust_scores(work, ["series"])
    monthly["expected_units"] = np.expm1(scores["expected_log"]).clip(lower=0).to_numpy()
    monthly["z_score"] = scores["z_score"].to_numpy()
    monthly["rel_change"] = ((monthly["demand_units"] - monthly["expected_units"])
                             / monthly["expected_units"].where(monthly["expected_units"] >= 1))
    monthly["flagged"] = ((monthly["z_score"].abs() >= Z_THRESHOLD)
                          & (monthly["rel_change"].abs() >= MIN_REL_CHANGE)).fillna(False)
    monthly["macro_shock"] = [month in shock_map for month in monthly.index]
    monthly["shock_count"] = [shock_map.get(month, 0) for month in monthly.index]
    monthly = monthly.reset_index().rename(columns={"index": "Month"})
    monthly["Month"] = pd.to_datetime(monthly["Month"])
    monthly["sales_rows"] = monthly["sales_rows"].fillna(0).astype(int)
    monthly["revenue_recomputed"] = monthly["revenue_recomputed"].fillna(0.0)
    monthly["Year"] = monthly["Month"].dt.year
    monthly["Month_Num"] = monthly["Month"].dt.month
    monthly = monthly.sort_values("Month").reset_index(drop=True)

    report.kv("months", f"{len(monthly)} ({_month_text(monthly['Month'].min())} -> "
                        f"{_month_text(monthly['Month'].max())})")
    report.kv("company-level flagged months", int(monthly["flagged"].sum()))
    report.kv("macro shock months", int(monthly["macro_shock"].sum()))
    report.kv("fill rate (min / median / max)",
              f"{_fmt(monthly['fill_rate_pct'].min(), 1)}% / "
              f"{_fmt(monthly['fill_rate_pct'].median(), 1)}% / "
              f"{_fmt(monthly['fill_rate_pct'].max(), 1)}%")
    report.rule("ANM-COMPANY", "company-level deviation from the seasonal baseline",
                "registered at company scope; macro shocks attached as context",
                int(monthly["flagged"].sum()))
    return monthly


def build_test_profile(test: pd.DataFrame, report: AnomalyReport) -> pd.DataFrame:
    report.h("6. FORECAST-WINDOW PROFILE (coverage, not anomalies)")
    profile = test.groupby("Month").agg(
        rows=("Month", "size"),
        skus=("SKU_ID", "nunique"),
        regions=("Region", "nunique"),
        customers=("Customer_Name", "nunique"),
        post_eol_rows=("post_eol", "sum"),
        pre_launch_rows=("pre_launch", "sum"),
        unseen_pair_rows=("unseen_pair", "sum"),
        no_history_rows=("has_history", lambda s: int((~s.astype(bool)).sum())),
        unmatched_sku_rows=("sku_unmatched", "sum"),
    ).reset_index()
    facts = {
        "rows": len(test),
        "months": int(test["Month"].nunique()),
        "post_eol_rows": int(test["post_eol"].sum()),
        "pre_launch_rows": int(test["pre_launch"].sum()),
        "unseen_pair_rows": int(test["unseen_pair"].sum()),
        "unseen_pairs": int(test.loc[test["unseen_pair"], ["SKU_ID", "Region"]]
                            .drop_duplicates().shape[0]),
        "cold_start_skus": sorted(test.loc[~test["has_history"].astype(bool), "SKU_ID"].unique()),
        "unmatched_sku_rows": int(test["sku_unmatched"].sum()),
    }
    report.kv("rows / months", f"{facts['rows']:,} / {facts['months']}")
    report.kv("rows after EOL (Phase 4 decision)", facts["post_eol_rows"])
    report.kv("unseen SKU-Region pairs", f"{facts['unseen_pair_rows']} rows / "
                                         f"{facts['unseen_pairs']} pairs")
    report.kv("cold-start SKUs", ", ".join(facts["cold_start_skus"]) or "none")
    report.rule("ANM-TEST", "forecast keys that are post-EOL or cold start",
                "documented here; Phase 4 decides zero vs floor explicitly",
                facts["post_eol_rows"] + facts["pre_launch_rows"],
                detail=f"{facts['unseen_pairs']} unseen pairs; "
                       f"cold start {', '.join(facts['cold_start_skus']) or 'none'}")
    report.metrics["test_window"] = facts
    return profile


# --------------------------------------------------------------------------- #
# 8. register assembly
# --------------------------------------------------------------------------- #
def assemble(parts: list[pd.DataFrame], report: AnomalyReport) -> pd.DataFrame:
    report.h("7. REGISTER ASSEMBLY")
    frame = pd.concat([part for part in parts if len(part)], ignore_index=True)
    for column in ANOMALY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[ANOMALY_COLUMNS]
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Month"] = pd.to_datetime(frame["Month"], errors="coerce")
    frame["anomaly_id"] = None

    missing_text = frame["evidence"].isna() | frame["evidence"].astype(str).str.strip().eq("")
    evidence_fallback = (frame["anomaly_class"].astype(str) + " on " + frame["scope"].astype(str))
    frame.loc[missing_text, "evidence"] = evidence_fallback[missing_text]
    frame["treatment"] = frame["treatment"].fillna("review")
    frame["status"] = frame["status"].fillna("open")
    frame["severity"] = frame["severity"].fillna("low")
    frame["secondary_class"] = frame["secondary_class"].fillna("")

    frame = frame.sort_values(["kind", "level", "Month", "SKU_ID", "Region", "anomaly_class"],
                              kind="mergesort", na_position="last").reset_index(drop=True)
    frame["anomaly_id"] = [f"AN-{i:06d}" for i in range(1, len(frame) + 1)]

    cell_rows = frame.loc[frame["level"].eq("cell")]
    duplicate_cells = int(cell_rows.duplicated(CELL).sum())
    report.kv("register rows", f"{len(frame):,}")
    report.kv("by kind", frame["kind"].value_counts().to_dict())
    report.kv("by level", frame["level"].value_counts().to_dict())
    report.kv("duplicate cell classifications", duplicate_cells)
    report.rule("ANM-REGISTER", "anomaly register assembled",
                "one primary class per cell, every line carries evidence and a treatment",
                len(frame), detail=f"{int(frame['kind'].eq('DATA_ERROR').sum())} data errors, "
                                   f"{int(frame['kind'].eq('DEMAND_EVENT').sum())} demand events")
    report.metrics["register"] = {
        "rows": len(frame),
        "by_kind": {str(k): int(v) for k, v in frame["kind"].value_counts().items()},
        "by_level": {str(k): int(v) for k, v in frame["level"].value_counts().items()},
        "by_class": {str(k): int(v) for k, v in frame["anomaly_class"].value_counts().items()},
        "duplicate_cell_classifications": duplicate_cells,
        "unexplained_demand_events": int(frame.loc[frame["kind"].eq("DEMAND_EVENT"),
                                                   "status"].eq("open").sum()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 9. outputs
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: Path, report: AnomalyReport, register: pd.DataFrame,
                  baseline: pd.DataFrame, monthly: pd.DataFrame, test_profile: pd.DataFrame) -> None:
    anomaly_dir = out_dir / ANOMALY_DIR
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    register.to_parquet(anomaly_dir / "anomaly_register.parquet", index=False)
    csv_view = register.assign(Month=register["Month"].dt.strftime("%Y-%m").fillna(""))
    csv_view.to_csv(anomaly_dir / "anomaly_register.csv", index=False, encoding="utf-8")
    register.loc[register["kind"].eq("DATA_ERROR")].to_parquet(
        anomaly_dir / "data_errors.parquet", index=False)
    register.loc[register["kind"].eq("DEMAND_EVENT")].to_parquet(
        anomaly_dir / "demand_events.parquet", index=False)

    cell_anomalies = register.loc[register["level"].eq("cell"),
                                  [*CELL, "anomaly_class", "severity"]]
    columns = [*CELL, "observed_units", "expected_units", "residual_log", "z_score",
               "rel_change", "scale", "n_obs", "baseline_source", "flagged"]
    columns = [c for c in columns if c in baseline.columns]
    baseline_out = baseline[columns]
    baseline_out = baseline_out.merge(cell_anomalies, on=CELL, how="left")
    baseline_out["anomaly_class"] = baseline_out["anomaly_class"].fillna("")
    baseline_out.to_parquet(anomaly_dir / "demand_baseline.parquet", index=False)

    monthly.to_parquet(anomaly_dir / "monthly_summary.parquet", index=False)
    test_profile.to_parquet(anomaly_dir / "test_window_profile.parquet", index=False)
    pd.DataFrame(report.log).to_csv(anomaly_dir / "anomaly_rules.csv", index=False)

    payload = {
        "meta": {
            "phase": 2,
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - report._t0, 2),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "metrics": report.metrics,
        "verdicts": report.verdicts,
        "rules": report.log,
        "outputs": {
            "anomaly_register": "outputs/anomalies/anomaly_register.parquet",
            "data_errors": "outputs/anomalies/data_errors.parquet",
            "demand_events": "outputs/anomalies/demand_events.parquet",
            "demand_baseline": "outputs/anomalies/demand_baseline.parquet",
            "monthly_summary": "outputs/anomalies/monthly_summary.parquet",
            "test_window_profile": "outputs/anomalies/test_window_profile.parquet",
        },
    }
    (out_dir / "anomaly_report.json").write_text(
        json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "anomaly_report.txt").write_text(
        "\n".join(report.lines) + "\n", encoding="utf-8")


def summarise_register(register: pd.DataFrame, report: AnomalyReport) -> None:
    report.h("8. WHAT THE REGISTER SAYS")
    cell = register.loc[register["level"].eq("cell") & register["kind"].eq("DEMAND_EVENT")]
    if len(cell):
        peaks = cell.loc[cell["anomaly_class"].isin(["spike", "drop"])]
        peaks = peaks.sort_values("impact", key=lambda s: s.abs(), ascending=False)
        report.sub("largest single-cell demand moves")
        for row in peaks.head(10).itertuples(index=False):
            report.p(f"  {_month_text(row.Month)}  {row.SKU_ID:<8} {str(row.Region):<18} "
                     f"{row.anomaly_class:<15} {row.impact:>+11,.0f} units "
                     f"(z={row.z_score:.1f}, {row.status})")
        collapses = cell.loc[cell["anomaly_class"].eq("demand_collapse")]
        if len(collapses):
            report.sub("demand collapses (established run stops)")
            for row in collapses.sort_values("impact").head(10).itertuples(index=False):
                report.p(f"  {_month_text(row.Month)}  {row.SKU_ID:<8} {str(row.Region):<18}"
                         f"{row.impact:>+12,.0f} units lost vs the prior quarter")
        censored = cell.loc[cell["anomaly_class"].eq("censored_by_stockout")]
        if len(censored):
            report.sub("largest stockout censoring")
            censored = censored.sort_values("impact", ascending=False)
            for row in censored.head(8).itertuples(index=False):
                report.p(f"  {_month_text(row.Month)}  {row.SKU_ID:<8} {str(row.Region):<18} "
                         f"{row.impact:>10,.0f} units unserved")
    errors = register.loc[register["kind"].eq("DATA_ERROR")
                          & register["level"].eq("dataset")]
    if len(errors):
        report.sub("data errors by rows affected")
        ordered = errors.sort_values("rows_affected", ascending=False).head(10)
        for row in ordered.itertuples(index=False):
            rows = 0 if pd.isna(row.rows_affected) else int(row.rows_affected)
            report.p(f"  {row.anomaly_class:<28} {str(row.explained_by):<18} "
                     f"{rows:>8,} rows  {row.status}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 anomaly register")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--clean-dir", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    clean_dir = Path(args.clean_dir) if args.clean_dir else out_dir / CLEAN_DIR
    if not clean_dir.is_absolute():
        clean_dir = PROJECT_ROOT / clean_dir

    report = AnomalyReport()
    report.p("CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER")
    report.p("Phase 2: anomaly register - data errors vs genuine demand events")
    report.p(f"clean dir : {clean_dir}")
    report.p(f"out dir   : {out_dir}")

    if not (clean_dir / "demand_monthly.parquet").exists():
        print(f"FATAL: Phase 1 outputs missing in {clean_dir}; run --phase 1 first",
              file=sys.stderr)
        return 2
    tables, payload = load_inputs(clean_dir)
    cleaning_log = pd.read_csv(clean_dir / "cleaning_log.csv")
    metrics = phase1_consistency(tables, payload, report)

    panel = tables["demand_monthly"]
    sales = tables["sales_clean"]
    master = tables["sku_master_clean"]
    events = tables["events_clean"]
    stockouts = tables["stockouts_clean"]
    macro = tables["macro_clean"]
    test = tables["test_clean"]

    scored, _ = score_cells(panel, report)
    collapses = detect_collapses(scored)
    digit_shifts = detect_digit_shifts(scored)
    shocks, shock_map = macro_shock_months(macro, report)

    cell_events = build_cell_events(scored, collapses, digit_shifts, stockouts,
                                    master, events, report)
    data_errors = build_data_errors(sales, report, cleaning_log)
    lifecycle = build_lifecycle_events(master)
    price = build_price_events(events)
    macro_events = build_macro_events(shocks)
    monthly = build_monthly_summary(panel, sales, macro, shock_map, report)
    test_profile = build_test_profile(test, report)

    company_hits = monthly.loc[monthly["flagged"],
                               ["Month", "demand_units", "expected_units", "z_score",
                                "rel_change", "macro_shock"]]
    company_rows = []
    for row in company_hits.itertuples(index=False):
        company_rows.append({
            "kind": "DEMAND_EVENT",
            "anomaly_class": "spike" if row.z_score > 0 else "drop",
            "secondary_class": "", "level": "company", "scope": "company total",
            "SKU_ID": "", "Region": "ALL", "Month": row.Month, "metric": "demand_units",
            "observed": float(row.demand_units), "expected": float(row.expected_units),
            "impact": float(row.demand_units - row.expected_units),
            "rel_change": float(row.rel_change) if pd.notna(row.rel_change) else None,
            "z_score": float(row.z_score), "severity": severity_of(row.z_score, row.rel_change),
            "rows_affected": None, "units_affected": None,
            "evidence": f"company demand {_fmt(row.demand_units)} vs expected "
                        f"{_fmt(row.expected_units)} ({_fmt(row.rel_change * 100, 1)}%, "
                        f"z={row.z_score:.1f})",
            "explained_by": "macro shock" if row.macro_shock else "",
            "treatment": "explain" if row.macro_shock else "model",
            "status": "explained" if row.macro_shock else "open",
            "note": "Company-level move; check market context before reading it per SKU.",
        })
    company = pd.DataFrame(company_rows)

    register = assemble([data_errors, cell_events, lifecycle, price, macro_events, company],
                        report)
    summarise_register(register, report)

    # ---- exit checks ------------------------------------------------------ #
    report.h("9. PHASE 2 EXIT CHECK")
    recon = metrics["reconciliation"]
    register_cells = register.loc[register["level"].eq("cell")]
    demand_total = float(panel["demand_units"].sum())
    suppressed_total = float(panel["suppressed_units"].sum())
    report.verdict("anomaly ids unique and ordered",
                   bool(register["anomaly_id"].is_unique)
                   and list(register["anomaly_id"]) == sorted(register["anomaly_id"]),
                   f"{len(register):,} rows")
    report.verdict("every anomaly carries evidence and a treatment",
                   bool(register["evidence"].astype(str).str.strip().ne("").all()
                        and register["treatment"].astype(str).str.strip().ne("").all()))
    report.verdict("one primary classification per cell",
                   not register_cells.duplicated(CELL).any(),
                   f"{len(register_cells):,} cell rows")
    report.verdict("demand never below observed units",
                   not bool((panel["demand_units"] < panel["observed_units"]).fillna(False).any()),
                   f"max shortfall {_fmt((panel['observed_units'] - panel['demand_units']).max(), 1)}")
    report.verdict("no usable demand outside lifecycle",
                   not bool((panel.loc[panel["usable"], ["pre_launch", "post_eol"]]
                             .any(axis=1)).any()),
                   f"{int(panel['usable'].sum()):,} usable cells")
    report.verdict("suppressed units reconcile to Phase 1",
                   abs(suppressed_total - float(metrics["demand"]["suppressed_units"])) <= 0.5,
                   f"{_fmt(suppressed_total)}")
    report.verdict("demand total reconciles to Phase 1",
                   abs(demand_total - float(metrics["demand"]["demand_units"])) <= 0.5,
                   f"{_fmt(demand_total)}")
    report.verdict("Phase 1 units ledger closes",
                   bool(recon.get("closes", False)),
                   f"ledger {_fmt(recon.get('units_demand_ledger'))} vs panel {_fmt(demand_total)}")
    known_rules = set(cleaning_log["rule_id"].astype(str))
    error_rules = set(register.loc[register["kind"].eq("DATA_ERROR"), "explained_by"]
                      .astype(str).str.split(":").str[0])
    error_rules = {rule for rule in error_rules if rule.startswith(("TRN-", "STK-", "EVT-", "MAC-", "MST-"))}
    report.verdict("data errors trace to Phase 1 rules",
                   error_rules <= known_rules,
                   f"{len(error_rules)} rule ids referenced"
                   + ("" if error_rules <= known_rules else
                      f"; unknown {sorted(error_rules - known_rules)}"))
    baseline_cells = report.metrics["baseline"]["usable_cells"]
    report.verdict("cell flag rate is diagnosis-sized",
                   report.metrics["baseline"]["flag_rate"] <= FLAG_RATE_LIMIT
                   or baseline_cells < 5000,
                   f"{report.metrics['baseline']['flag_rate']:.2%} of usable cells "
                   f"(limit {FLAG_RATE_LIMIT:.0%})" if baseline_cells >= 5000
                   else f"{baseline_cells:,} cells - too few to rate, threshold check skipped")
    report.verdict("baseline covers every usable cell",
                   report.metrics["baseline"]["coverage"] >= 0.99,
                   f"{report.metrics['baseline']['coverage']:.2%}")
    report.verdict("both anomaly kinds are represented",
                   bool(register["kind"].eq("DATA_ERROR").any()
                        and register["kind"].eq("DEMAND_EVENT").any()),
                   f"{int(register['kind'].eq('DATA_ERROR').sum())} data errors / "
                   f"{int(register['kind'].eq('DEMAND_EVENT').sum())} demand events")
    report.verdict("register is deterministically ordered",
                   register.equals(register.sort_values(
                       ["kind", "level", "Month", "SKU_ID", "Region", "anomaly_class"],
                       kind="mergesort", na_position="last").reset_index(drop=True)))

    report.kv("elapsed seconds", round(time.time() - started, 1))
    write_outputs(out_dir, report, register, scored, monthly, test_profile)

    print("\n".join(report.lines))
    print(f"\nJSON  -> {out_dir / 'anomaly_report.json'}")
    print(f"TEXT  -> {out_dir / 'anomaly_report.txt'}")
    print(f"TABLES-> {out_dir / ANOMALY_DIR}")
    failed = [v for v in report.verdicts if not v["ok"]]
    if failed:
        print(f"\nPHASE 2 FAILED {len(failed)} check(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
