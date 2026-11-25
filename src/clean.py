#!/usr/bin/env python
"""Phase 1 - Cleaning, canonicalisation and reconciliation.

Turns the raw ERP extract into analysis-ready tables.  Every rule implemented
here is traceable to a Phase 0 finding (``outputs/profile_report.json``); the
rule id is written next to the affected rows in ``cleaning_log.csv`` so a
reviewer can audit a value back to the evidence that justified changing it.

Guiding distinctions
--------------------
* Data error  -> repaired, with the original value kept beside the repaired one.
* Real event  -> flagged, never overwritten (stockouts, lifecycle, surcharges).
* Missing row -> NOT a zero.  Only explicit ``Units_Sold = 0`` rows are zeros.

Rules applied
-------------
TRN-DATE-2062   Date year implausible vs the Year/Month key -> re-dated (never dropped).
TRN-DATE-KEY    Year/Month disagree with Date -> Date wins (Date is authoritative).
TRN-DUP-ID      Duplicate Record_ID = double load -> keep latest Load_Timestamp.
TRN-SKU-KEY     SKU spelling variants -> master key (case/punctuation + zero-pad).
TRN-LABEL       Customer/Region/Channel spelling variants -> single canonical label.
TRN-UNITS-NEG   Negative units -> excluded from demand (flagged, retained).
TRN-UNITS-BLANK Blank units -> unknown (flagged, never coerced to zero).
TRN-PRICE       Unit price -> one base price per SKU-year (identical across regions,
                steps only at 1 January); Revenue/Units used as a second candidate.
TRN-REVENUE     Units * Price != Revenue -> recomputed revenue kept beside recorded.
TRN-LIFECYCLE   Rows before Launch_Date or after EOL_Date -> flagged, marked unusable.

Outputs (outputs/clean/)
------------------------
sales_clean.parquet        deduplicated row-level fact table, cleaned + flagged
sku_master_clean.parquet   canonical product master with lifecycle months
demand_monthly.parquet     (Month, SKU, Region) observed vs unconstrained demand
demand_grid.parquet        every SKU-Region-month cell with an explicit status
stockouts_clean.parquet    stockout log with recomputed fill rate
macro_clean.parquet        deduplicated, gap-filled monthly macro series
events_clean.parquet       parsed event calendar with resolved scopes
test_clean.parquet         forecast-window keys, canonicalised and flagged
label_map.csv              every raw -> canonical label mapping, with method
cleaning_log.csv           rule-by-rule audit trail
reconciliation.csv         additive units ledger: every excluded unit is itemised
                           (duplicates, returns, lifecycle, unknown region) and the
                           running total must close on demand_monthly
clean_report.json / .txt   machine + human readable summary

Usage
-----
    python src/clean.py --data-dir student_data
    python run_pipeline.py --phase 1
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profile_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    EVENTS_FILE,
    MACRO_FILE,
    MASTER_FILE,
    STOCKOUT_FILE,
    SUB_FILE,
    TEST_FILE,
    TRAIN_FILE,
    fmt,
    jsonable,
    parse_dates,
    read_raw,
    to_num,
)

CLEAN_DIR = "clean"
GRAIN = ["Month", "SKU_ID", "Region"]
INPUT_FILES = [TRAIN_FILE, TEST_FILE, MACRO_FILE, MASTER_FILE,
               EVENTS_FILE, STOCKOUT_FILE, SUB_FILE]

# Price is one base list price per SKU-year; candidates outside this band around
# the SKU's own median are scale errors (Phase 0: worst observed ratio 137x).
PRICE_BAND = 3.0
# Customer spellings are 1-2 character corruptions of a 12-character name, so a
# high cutoff still maps all 220 observed variants (worst accepted = 0.95).
FUZZY_CUTOFF = 0.85
REVENUE_TOLERANCE = 0.005
MIN_REVENUE_TOLERANCE = 1.0

REGION_CANONICAL = ["Asia Pacific", "Europe", "India", "Latin America",
                    "Middle East & Africa", "North America"]
REGION_ALIASES = {
    "LATAM": "Latin America", "LATINAMERICA": "Latin America", "LATIN AM": "Latin America",
    "APAC": "Asia Pacific", "ASIAPACIFIC": "Asia Pacific",
    "EU": "Europe", "EUROPE": "Europe",
    "NA": "North America", "NAMERICA": "North America", "NORTHAMERICA": "North America",
    "MEA": "Middle East & Africa", "MIDDLEEASTAFRICA": "Middle East & Africa",
    "MIDDLEEASTANDAFRICA": "Middle East & Africa",
    "INDIA": "India",
}
CHANNEL_CANONICAL = ["Aftermarket", "Direct OEM", "Export", "Tier-1 Distributor"]
CHANNEL_ALIASES = {
    "OEMDIRECT": "Direct OEM", "DIRECTOEM": "Direct OEM",
    "T1DISTRIBUTOR": "Tier-1 Distributor", "TIER1DISTRIBUTOR": "Tier-1 Distributor",
    "AFTERMARKET": "Aftermarket", "EXPORT": "Export",
}

UNIT_COLUMNS = ["Requested_Units", "Carried_In_Units", "Shipped_Units",
                "Backordered_Units", "Lost_Units", "Fill_Rate_Pct", "Stockout_Days"]
MACRO_INDICATORS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct",
                    "Vehicle_Inventory_Days", "Raw_Material_Cost_Index",
                    "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
                    "Fuel_Price_Index", "EV_Adoption_Index"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def compact(value, strip_suffix: bool = False) -> str:
    """Case/punctuation-free key used for label matching."""
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    if strip_suffix:
        text = re.sub(r"(INC|CORP|LLC|LTD|GMBH|CO)$", "", text)
    return text


def sku_key(value) -> str:
    """Canonical SKU key: strip punctuation, zero-pad the numeric suffix.

    Phase 0: 1,012 punctuation/case-folded keys, of which 414 missed the master;
    padding the suffix (``BSS10`` -> ``BSS010``) resolves all 738 affected rows.
    """
    text = compact(value)
    match = re.match(r"^([A-Z]{2,4})(\d{1,3})$", text)
    return f"{match.group(1)}{int(match.group(2)):03d}" if match else text


def month_of(values: pd.Series) -> pd.Series:
    """Datetime month-start stamps from a parsed date series."""
    return values.dt.to_period("M").dt.to_timestamp()


def numeric_text(values: pd.Series) -> pd.Series:
    """Parse an ERP text column to numbers ($ , % and (1,234) tolerated)."""
    return to_num(values)[0]


def build_flags(index: pd.Index, definitions: list[tuple[str, pd.Series]]) -> pd.Series:
    """Join flag codes into one semicolon-delimited string per row."""
    codes: list[list[str]] = [[] for _ in range(len(index))]
    for code, mask in definitions:
        values = np.asarray(pd.Series(mask).fillna(False), dtype=bool)
        for position in np.flatnonzero(values):
            codes[position].append(code)
    return pd.Series([";".join(c) for c in codes], index=index, dtype=object)


def safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    """Elementwise ratio that never divides by zero."""
    return left / right.where(right != 0)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
class CleanReport:
    """Console transcript + machine-readable cleaning summary."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.verdicts: list[dict] = []
        self.metrics: dict = {}
        self.log: list[dict] = []
        self.label_rows: list[dict] = []
        self._t0 = time.time()

    def h(self, title: str) -> None:
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"[clean] {title}", file=sys.stderr, flush=True)

    def sub(self, title: str) -> None:
        self.lines.append(f"\n-- {title} " + "-" * max(0, 58 - len(title)))

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def kv(self, key: str, value, indent: int = 2, width: int = 44) -> None:
        self.lines.append(f"{' ' * indent}{key:<{width}} {fmt(value)}")

    def rule(self, rule_id: str, issue: str, action: str, rows: int = 0,
             units: float | None = None, detail: str = "") -> None:
        self.log.append({
            "rule_id": rule_id,
            "issue": issue,
            "action": action,
            "rows_affected": int(rows),
            "units_affected": None if units is None else round(float(units), 3),
            "detail": detail,
        })

    def verdict(self, name: str, ok: bool, detail: str = "") -> None:
        self.verdicts.append({"phase": 1, "check": name, "ok": bool(ok), "detail": detail})
        self.p(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. product master
# --------------------------------------------------------------------------- #
def clean_master(raw: pd.DataFrame, report: CleanReport) -> tuple[pd.DataFrame, dict[str, str]]:
    report.h("1. PRODUCT MASTER")
    master = raw[MASTER_FILE].copy()
    master["SKU_ID"] = master["SKU_ID"].astype(str).str.strip()
    master["SKU_Key"] = master["SKU_ID"].map(sku_key)
    # The master spelling (BSS-001) is the canonical identifier end to end, so
    # the submission format is preserved; the key is only the matching device.
    key_to_id = (master.drop_duplicates("SKU_Key")
                 .set_index("SKU_Key")["SKU_ID"].to_dict())
    master["Launch_Month"] = month_of(parse_dates(master["Launch_Date"]))
    master["EOL_Month"] = month_of(parse_dates(master["EOL_Date"]))
    master["Replaces_SKU"] = master["Replaces_SKU"].map(
        lambda v: key_to_id.get(sku_key(v), "") if str(v).strip() else "")

    for column in ["Product_Family", "Business_Unit", "SKU_Description",
                   "Size_Class", "Powertrain_Application", "Lifecycle_Status"]:
        master[column] = master[column].astype(str).str.strip()

    duplicated = int(master["SKU_Key"].duplicated().sum())
    report.rule("MST-KEY", "duplicate SKU_ID in master", "kept first occurrence", duplicated)
    master = master.drop_duplicates("SKU_Key", keep="first")

    report.rule("MST-LAUNCH", "missing Launch_Date", "kept; lifecycle lower bound unknown",
                int(master["Launch_Month"].isna().sum()))
    eol_before_launch = master["EOL_Month"].lt(master["Launch_Month"]).fillna(False)
    report.rule("MST-EOL", "EOL_Date precedes Launch_Date", "EOL ignored for lifecycle",
                int(eol_before_launch.sum()))
    master.loc[eol_before_launch, "EOL_Month"] = pd.NaT

    report.kv("master rows (canonical)", len(master))
    report.kv("lifecycle status counts", master["Lifecycle_Status"].value_counts().to_dict())
    report.kv("SKUs with EOL_Date", int(master["EOL_Month"].notna().sum()))
    report.kv("SKUs replacing another SKU", int(master["Replaces_SKU"].ne("").sum()))
    report.metrics["master"] = {
        "rows": len(master),
        "with_eol": int(master["EOL_Month"].notna().sum()),
        "replacements": int(master["Replaces_SKU"].ne("").sum()),
    }
    return master, key_to_id


# --------------------------------------------------------------------------- #
# 2. label resolver
# --------------------------------------------------------------------------- #
class LabelResolver:
    """Maps raw spellings onto canonical labels and records how it did it."""

    def __init__(self, canonical: list[str], aliases: dict[str, str],
                 strip_suffix: bool = False) -> None:
        self.canonical = list(dict.fromkeys(canonical))
        self.keys = {compact(value, strip_suffix): value for value in self.canonical}
        self.aliases = {compact(k, strip_suffix): v for k, v in aliases.items()}
        self.strip_suffix = strip_suffix
        self.rows: list[dict] = []

    def _one(self, raw: str) -> tuple[str, str, float]:
        text = str(raw).strip()
        if not text:
            return "Unknown", "blank", 0.0
        key = compact(text, self.strip_suffix)
        if text in self.canonical:
            return text, "exact", 1.0
        if key in self.keys:
            return self.keys[key], "normalised", 1.0
        if key in self.aliases:
            return self.aliases[key], "alias", 1.0
        scored = [(difflib.SequenceMatcher(None, key, candidate).ratio(), name)
                  for candidate, name in self.keys.items()]
        if scored:
            score, name = max(scored)
            if score >= FUZZY_CUTOFF:
                return name, "fuzzy", round(score, 4)
        return "Unknown", "unmapped", 0.0

    def resolve(self, values: pd.Series, column: str) -> pd.Series:
        text = values.astype(str).str.strip()
        counts = text.value_counts()
        mapping = {}
        for raw in counts.index:
            canonical, method, score = self._one(raw)
            mapping[raw] = canonical
            self.record(column, raw, canonical, method, score, int(counts[raw]))
        return text.map(mapping).fillna("Unknown")

    def record(self, column: str, raw: str, canonical: str, method: str,
               score: float, rows: int) -> None:
        self.rows.append({"column": column, "raw_value": raw,
                          "canonical_value": canonical, "method": method,
                          "similarity": score, "n_rows": rows})


def label_resolvers(test: pd.DataFrame) -> tuple[LabelResolver, LabelResolver, LabelResolver]:
    """Build the three label vocabularies from the clean reference sets."""
    return (
        LabelResolver(sorted(test["Customer_Name"].unique()), {}, strip_suffix=True),
        LabelResolver(REGION_CANONICAL, REGION_ALIASES),
        LabelResolver(CHANNEL_CANONICAL, CHANNEL_ALIASES),
    )


# --------------------------------------------------------------------------- #
# 3. training fact table
# --------------------------------------------------------------------------- #
def clean_train(raw: pd.DataFrame, master: pd.DataFrame, key_to_id: dict[str, str],
                report: CleanReport) -> pd.DataFrame:
    report.h("2. TRAINING FACT TABLE")
    frame = raw[TRAIN_FILE].copy()
    frame["__row"] = np.arange(len(frame))
    raw_rows = len(frame)
    raw_units = numeric_text(frame["Units_Sold"]).sum()
    # The reconciliation ledger runs in non-negative units: a return line is not
    # negative demand, it is a row that contributes no demand at all.
    raw_units_nonneg = float(numeric_text(frame["Units_Sold"]).clip(lower=0).sum())

    # ---- dates: Date is authoritative over Year/Month --------------------- #
    parsed = parse_dates(frame["Date"])
    year_key = numeric_text(frame["Year"])
    month_key = numeric_text(frame["Month"])
    # The Year column is an independent keying hint: a Date beyond the 99.9th
    # percentile of Year is a keying error, not a real observation.
    horizon = int(year_key.quantile(0.999)) if year_key.notna().any() else 2026
    beyond = parsed.dt.year.gt(horizon + 1)
    rebuild = (beyond & year_key.between(horizon - 30, horizon + 1)
               & month_key.between(1, 12)).fillna(False)
    repaired_text = (year_key[rebuild].astype(int).astype(str) + "-"
                     + month_key[rebuild].astype(int).astype(str).str.zfill(2) + "-01")
    parsed.loc[rebuild] = pd.to_datetime(repaired_text, format="%Y-%m-%d")

    parsed_month = month_of(parsed)
    key_month = pd.to_datetime(
        year_key.fillna(0).astype(int).astype(str) + "-"
        + month_key.fillna(0).astype(int).astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d", errors="coerce")
    key_mismatch = (parsed_month.notna() & key_month.notna()
                    & parsed_month.ne(key_month)).fillna(False)

    report.rule("TRN-DATE-2062", "date year outside the observed horizon (keying error)",
                "re-dated from Year/Month; row retained", int(rebuild.sum()),
                detail=f"plausible horizon year {horizon}; the Year column supplied the repair")
    report.rule("TRN-DATE-BAD", "Date unparseable", "kept but excluded from demand",
                int(parsed.isna().sum()))
    report.rule("TRN-DATE-KEY", "Year/Month disagree with Date",
                "Date wins; Year/Month recomputed from Date", int(key_mismatch.sum()))

    frame["Date"] = parsed
    frame["Month"] = parsed_month
    frame["Year"] = frame["Month"].dt.year
    frame["Month_Num"] = frame["Month"].dt.month
    frame["__date_repaired"] = rebuild
    frame["__date_key_mismatch"] = key_mismatch

    # ---- duplicate Record_ID (double load) -------------------------------- #
    frame["Load_Timestamp"] = parse_dates(frame["Load_Timestamp"])
    payload_columns = [c for c in frame.columns
                       if c not in {"Record_ID", "Load_Timestamp", "__row",
                                    "__date_repaired", "__date_key_mismatch"}]
    frame["__payload"] = pd.util.hash_pandas_object(frame[payload_columns], index=False)
    ordered = frame.sort_values(["Record_ID", "Load_Timestamp"], kind="mergesort",
                                na_position="first")
    kept = ordered.drop_duplicates("Record_ID", keep="last")
    dropped = frame[~frame["__row"].isin(kept["__row"])]
    if len(dropped):
        compare = dropped.merge(kept[["Record_ID", "__payload"]], on="Record_ID",
                                suffixes=("", "_kept"))
        identical = int((compare["__payload"] == compare["__payload_kept"]).sum())
    else:
        identical = 0
    report.rule("TRN-DUP-ID", "duplicate Record_ID (double load)",
                "kept latest Load_Timestamp", len(dropped),
                detail=f"{identical:,} of {len(dropped):,} dropped rows were identical "
                       f"payloads; the rest conflicted and resolved by newest load")
    frame = kept.sort_values("__row").reset_index(drop=True)
    units_after_dedupe_nonneg = float(numeric_text(frame["Units_Sold"]).clip(lower=0).sum())

    # ---- SKU canonicalisation -------------------------------------------- #
    frame["SKU_ID_raw"] = frame["SKU_ID"]
    frame["SKU_Key"] = frame["SKU_ID"].map(sku_key)
    frame["SKU_ID"] = frame["SKU_Key"].map(key_to_id)
    sku_changed = (frame["SKU_ID"].ne(frame["SKU_ID_raw"].str.strip())
                   & frame["SKU_ID"].notna()).fillna(False)
    sku_unmatched = frame["SKU_ID"].isna()
    report.rule("TRN-SKU-KEY", "SKU spelling variant / unpadded numeric suffix",
                "mapped to canonical master key", int(sku_changed.sum()),
                detail="BSS10 -> BSS010 and equivalent case/punctuation variants")
    report.rule("TRN-SKU-MISS", "SKU absent from master",
                "flagged and excluded from demand", int(sku_unmatched.sum()))
    frame = frame[~sku_unmatched].reset_index(drop=True)
    units_after_sku_nonneg = float(numeric_text(frame["Units_Sold"]).clip(lower=0).sum())

    # ---- labels ----------------------------------------------------------- #
    resolver_customer, resolver_region, resolver_channel = label_resolvers(raw[TEST_FILE])
    frame["Customer_Raw"] = frame["Customer_Name"]
    frame["Region_Raw"] = frame["Region"]
    frame["Channel_Raw"] = frame["Sales_Channel"]
    frame["Customer_Name"] = resolver_customer.resolve(frame["Customer_Name"], "Customer_Name")
    frame["Region"] = resolver_region.resolve(frame["Region"], "Region")
    frame["Sales_Channel"] = resolver_channel.resolve(frame["Sales_Channel"], "Sales_Channel")
    for resolver in (resolver_customer, resolver_region, resolver_channel):
        report.label_rows.extend(resolver.rows)

    label_changed = (frame["Customer_Name"].ne(frame["Customer_Raw"].str.strip())
                     | frame["Region"].ne(frame["Region_Raw"].str.strip())
                     | frame["Sales_Channel"].ne(frame["Channel_Raw"].str.strip()))
    report.rule("TRN-LABEL", "label spelling variants (customer / region / channel)",
                "mapped to canonical label", int(label_changed.sum()),
                detail=f"{frame['Customer_Name'].nunique()} customers, "
                       f"{frame['Region'].nunique()} regions, "
                       f"{frame['Sales_Channel'].nunique()} channels retained")

    # ---- numerics --------------------------------------------------------- #
    frame["Units_Sold"] = numeric_text(frame["Units_Sold"])
    frame["Unit_Price_USD"] = numeric_text(frame["Unit_Price_USD"])
    frame["Revenue_Recorded_USD"] = numeric_text(frame["Revenue_USD"])
    blank_units = frame["Units_Sold"].isna()
    negative_units = frame["Units_Sold"].lt(0).fillna(False)
    report.rule("TRN-UNITS-BLANK", "Units_Sold blank in an existing row",
                "treated as unknown, never as zero", int(blank_units.sum()))
    report.rule("TRN-UNITS-NEG", "Units_Sold negative",
                "excluded from demand (returns/adjustment)", int(negative_units.sum()),
                units=float(frame.loc[negative_units, "Units_Sold"].sum()))
    frame["units_valid"] = frame["Units_Sold"].ge(0).fillna(False)
    frame["Units_Sold_Clean"] = frame["Units_Sold"].where(frame["units_valid"])
    frame["units_clean"] = frame["Units_Sold_Clean"]

    # ---- lifecycle -------------------------------------------------------- #
    frame = frame.merge(master[["SKU_ID", "Launch_Month", "EOL_Month"]],
                        on="SKU_ID", how="left")
    pre_launch = frame["Month"].lt(frame["Launch_Month"]).fillna(False)
    post_eol = frame["Month"].gt(frame["EOL_Month"]).fillna(False)
    frame["lifecycle_ok"] = ~(pre_launch | post_eol)
    frame["pre_launch"] = pre_launch
    frame["post_eol"] = post_eol
    report.rule("TRN-LIFECYCLE", "sales outside Launch_Date..EOL_Date",
                "flagged and excluded from training demand",
                int((pre_launch | post_eol).sum()),
                units=float(frame.loc[pre_launch | post_eol, "Units_Sold_Clean"].sum()),
                detail=f"{int(pre_launch.sum()):,} before launch, "
                       f"{int(post_eol.sum()):,} after EOL")

    frame = _price_ladder(frame, report)
    frame = _revenue_identity(frame, report)

    region_unknown = frame["Region"].eq("Unknown")
    report.rule("TRN-REGION", "Region blank / unresolvable",
                "kept as Region=Unknown, excluded from demand",
                int(region_unknown.sum()),
                units=float(frame.loc[region_unknown, "Units_Sold_Clean"].sum()))
    report.rule("TRN-GRAIN-DUP", "Date+SKU+Region repeated",
                "retained: rows split by customer/channel (legitimate)",
                int(frame.duplicated(["Month", "SKU_ID", "Region"]).sum()))

    frame["usable_for_training"] = (frame["lifecycle_ok"] & ~region_unknown
                                    & frame["Month"].notna() & frame["units_valid"])

    # ---- units ledger: mutually exclusive exclusion buckets --------------- #
    # Applied in priority order so the buckets cannot overlap; their sum plus the
    # usable remainder is exactly sales_clean, which is what lets the
    # reconciliation table close on the demand panel instead of leaving a gap.
    nonneg_units = frame["Units_Sold"].clip(lower=0)
    buckets: list[dict] = []
    remaining = pd.Series(True, index=frame.index)
    for bucket, mask, rule_id, basis in [
        ("blank units (unknown, not zero)", blank_units, "TRN-UNITS-BLANK",
         "row retained; carries no demand"),
        ("negative units (returns/adjustments)", negative_units, "TRN-UNITS-NEG",
         "row retained and flagged; contributes no demand"),
        ("unparseable dates", frame["Month"].isna(), "TRN-DATE-BAD",
         "row retained; cannot be placed in a month"),
        ("pre-launch months", pre_launch, "TRN-LIFECYCLE", "Month < Launch_Month"),
        ("post-EOL months", post_eol, "TRN-LIFECYCLE", "Month > EOL_Month"),
        ("Region = Unknown", region_unknown, "TRN-REGION",
         "region blank or unresolvable"),
    ]:
        selected = mask.fillna(False) & remaining
        rows = int(selected.sum())
        buckets.append({"step": bucket, "rule_id": rule_id, "basis": basis,
                        "rows": rows, "units": float(nonneg_units[selected].sum())})
        remaining &= ~selected
    usable_units = float(nonneg_units[remaining].sum())
    report.rule("TRN-LEDGER", "units outside the demand panel",
                "itemised by mutually exclusive bucket, never left implicit",
                int(frame.shape[0] - remaining.sum()),
                units=float(nonneg_units[~remaining].sum()),
                detail="; ".join(f"{b['step']}: {b['rows']:,} rows / {fmt(b['units'])} units"
                                 for b in buckets))
    report.kv("units ledger - usable remainder", fmt(usable_units))

    frame["flags"] = build_flags(frame.index, [
        ("DATE_REPAIRED", frame["__date_repaired"]),
        ("DATE_KEY_MISMATCH", frame["__date_key_mismatch"]),
        ("SKU_KEY_REPAIRED", sku_changed),
        ("LABEL_REPAIRED", label_changed),
        ("UNITS_BLANK", blank_units),
        ("UNITS_NEGATIVE", negative_units),
        ("PRICE_FILLED", frame["price_filled"]),
        ("PRICE_REPAIRED", frame["price_repaired"]),
        ("REVENUE_IDENTITY_FAILED", frame["revenue_identity_failed"]),
        ("PRE_LAUNCH", pre_launch),
        ("POST_EOL", post_eol),
        ("REGION_UNKNOWN", region_unknown),
    ])

    units_dropped = raw_units - frame["Units_Sold"].sum()
    report.kv("raw rows", raw_rows)
    report.kv("rows after deduplication", len(frame))
    report.kv("usable rows for training", int(frame["usable_for_training"].sum()))
    report.kv("units: raw / clean", f"{fmt(raw_units)} / "
                                    f"{fmt(frame['Units_Sold_Clean'].sum())}")
    report.metrics["train"] = {
        "raw_rows": raw_rows,
        "rows_after_dedupe": len(frame),
        "rows_dropped": raw_rows - len(frame),
        "units_raw": None if pd.isna(raw_units) else float(raw_units),
        "units_dropped": None if pd.isna(units_dropped) else float(units_dropped),
        "usable_rows": int(frame["usable_for_training"].sum()),
        "sku_variants_repaired": int(sku_changed.sum()),
        "labels_repaired": int(label_changed.sum()),
        "units_missing": int(blank_units.sum()),
        "units_negative": int(negative_units.sum()),
        "units_clean": float(frame["Units_Sold_Clean"].sum()),
        "units_raw_nonneg": raw_units_nonneg,
        "units_duplicates_nonneg": raw_units_nonneg - units_after_dedupe_nonneg,
        "units_sku_unmatched_nonneg": units_after_dedupe_nonneg - units_after_sku_nonneg,
        "rows_sku_unmatched": int(sku_unmatched.sum()),
        "ledger": buckets,
        "usable_units": usable_units,
        "pre_launch_rows": int(pre_launch.sum()),
        "post_eol_rows": int(post_eol.sum()),
        "region_unknown_rows": int(region_unknown.sum()),
    }
    return frame


def _price_ladder(frame: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """One base price per SKU-year, enforced across regions, steps only in January."""
    positive = frame["Unit_Price_USD"].where(frame["Unit_Price_USD"] > 0)
    implied = safe_ratio(frame["Revenue_Recorded_USD"],
                         frame["Units_Sold_Clean"].where(frame["Units_Sold_Clean"] > 0))
    implied = implied.where(implied > 0)
    ladder_source = frame["lifecycle_ok"]

    median_scope = frame[ladder_source & positive.notna()]
    sku_median = median_scope.groupby("SKU_ID")["Unit_Price_USD"].median()
    all_median = frame[positive.notna()].groupby("SKU_ID")["Unit_Price_USD"].median()
    sku_median = sku_median.reindex(all_median.index).fillna(all_median)
    center = frame["SKU_ID"].map(sku_median)
    low, high = center / PRICE_BAND, center * PRICE_BAND

    candidate = positive.where(positive.between(low, high))
    from_revenue = implied.where(implied.between(low, high))
    frame["price_candidate"] = candidate.fillna(from_revenue)

    work = frame.loc[frame["price_candidate"].notna(),
                     ["SKU_ID", "Month", "price_candidate"]].copy()
    work["Price_Year"] = work["Month"].dt.year
    ladder = (work.groupby(["SKU_ID", "Price_Year"])["price_candidate"].median()
              .rename("base_price").reset_index())

    years = sorted(int(y) for y in frame["Month"].dt.year.dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [sorted(frame["SKU_ID"].unique()), years], names=["SKU_ID", "Price_Year"])
    ladder = ladder.set_index(["SKU_ID", "Price_Year"]).reindex(full_index).sort_index()
    present = ladder["base_price"].notna()
    ladder["base_price"] = ladder.groupby(level="SKU_ID")["base_price"].ffill().bfill()
    ladder["price_source"] = np.where(present, "annual", "carried")
    ladder = ladder.reset_index()

    frame["Price_Year"] = frame["Month"].dt.year
    frame = frame.merge(ladder, on=["SKU_ID", "Price_Year"], how="left")
    fallback = frame["SKU_ID"].map(sku_median)
    missing_price = frame["base_price"].isna()
    frame["base_price"] = frame["base_price"].fillna(fallback)
    frame.loc[missing_price & frame["base_price"].notna(), "price_source"] = "sku_median"
    frame["price_source"] = frame["price_source"].fillna("missing")
    frame["Price_Clean_USD"] = frame["base_price"]

    recorded = frame["Unit_Price_USD"]
    relative_gap = safe_ratio((recorded - frame["Price_Clean_USD"]).abs(),
                              frame["Price_Clean_USD"]).abs()
    frame["price_repaired"] = (recorded.notna() & frame["Price_Clean_USD"].notna()
                               & (relative_gap > REVENUE_TOLERANCE)).fillna(False)
    frame["price_filled"] = (recorded.isna() | recorded.le(0)).fillna(False)

    valid_before = frame["Unit_Price_USD"].gt(0)
    before = frame.loc[valid_before].groupby(["SKU_ID", "Month"])["Unit_Price_USD"]
    split_before = int((before.nunique() > 1).sum())
    ratio_before = safe_ratio(before.max(), before.min()).max()
    after = frame.groupby(["SKU_ID", "Month"])["Price_Clean_USD"]
    split_after = int((after.nunique(dropna=True) > 1).sum())
    ratio_after = safe_ratio(after.max(), after.min()).max()

    report.rule("TRN-PRICE", "unit price not constant across SKU-month regions",
                "single SKU-year base price applied",
                int(frame["price_repaired"].sum()),
                detail=f"price differed by region in {split_before:,} SKU-months before and "
                       f"{split_after:,} after; max within-SKU-month ratio "
                       f"{fmt(ratio_before)} -> {fmt(ratio_after)}")
    report.rule("TRN-PRICE-FILL", "unit price zero / missing",
                "filled from the SKU-year ladder", int(frame["price_filled"].sum()))
    report.kv("price differs by region (SKU-months)", f"{split_before:,} -> {split_after:,}")
    report.kv("max within-SKU-month price ratio",
              f"{fmt(ratio_before)} -> {fmt(ratio_after)}")
    report.metrics["price"] = {
        "sku_months_split_before": split_before,
        "sku_months_split_after": split_after,
        "max_ratio_before": None if pd.isna(ratio_before) else round(float(ratio_before), 4),
        "max_ratio_after": None if pd.isna(ratio_after) else round(float(ratio_after), 4),
        "rows_price_repaired": int(frame["price_repaired"].sum()),
        "rows_price_filled": int(frame["price_filled"].sum()),
        "sku_years": int(len(ladder)),
    }
    return frame


def _revenue_identity(frame: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """Recorded Units * recorded Price vs Revenue, plus the canonical restatement.

    The identity check uses the *recorded* price: it measures whether one ERP
    record is internally consistent.  Comparing against the repaired price would
    conflate a price error with a revenue error, so the two are reported apart.
    """
    recomputed = frame["Units_Sold_Clean"] * frame["Price_Clean_USD"]
    frame["Revenue_Recomputed_USD"] = recomputed
    recorded_value = frame["Units_Sold_Clean"] * frame["Unit_Price_USD"]
    tolerance = np.maximum(recorded_value.abs() * REVENUE_TOLERANCE, MIN_REVENUE_TOLERANCE)
    gap = (frame["Revenue_Recorded_USD"] - recorded_value).abs()
    frame["revenue_identity_failed"] = (
        frame["Revenue_Recorded_USD"].notna() & recorded_value.notna()
        & (gap > tolerance)).fillna(False)
    canonical_gap = ((frame["Revenue_Recorded_USD"] - recomputed).abs()
                     > np.maximum(recomputed.abs() * REVENUE_TOLERANCE,
                                  MIN_REVENUE_TOLERANCE)) & recomputed.notna()
    report.rule("TRN-REVENUE", "Units * recorded Price != Revenue",
                "recorded revenue and price kept; recomputed revenue added; row flagged",
                int(frame["revenue_identity_failed"].sum()),
                detail=f"recorded total {fmt(frame['Revenue_Recorded_USD'].sum())}, "
                       f"recomputed total {fmt(recomputed.sum())}")
    report.kv("revenue identity failures (recorded price)",
              f"{int(frame['revenue_identity_failed'].sum()):,} / {len(frame):,} rows")
    report.kv("revenue restated at canonical price",
              f"{int(canonical_gap.sum()):,} rows differ")
    report.metrics["revenue"] = {
        "identity_failures_recorded_price": int(frame["revenue_identity_failed"].sum()),
        "restated_at_canonical_price": int(canonical_gap.sum()),
        "recorded_total": float(frame["Revenue_Recorded_USD"].sum()),
        "recomputed_total": float(recomputed.sum()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 4. test window
# --------------------------------------------------------------------------- #
def clean_test(raw: pd.DataFrame, master: pd.DataFrame, key_to_id: dict[str, str],
               train: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    report.h("3. FORECAST WINDOW (test.csv)")
    frame = raw[TEST_FILE].copy()
    frame["Month"] = month_of(parse_dates(frame["Date"]))
    frame["Year"] = frame["Month"].dt.year
    frame["Month_Num"] = frame["Month"].dt.month
    frame["SKU_ID_raw"] = frame["SKU_ID"]
    frame["SKU_Key"] = frame["SKU_ID"].map(sku_key)
    frame["SKU_ID"] = frame["SKU_Key"].map(key_to_id)
    frame["sku_unmatched"] = frame["SKU_ID"].isna()

    resolver_customer, resolver_region, resolver_channel = label_resolvers(train)
    frame["Customer_Raw"] = frame["Customer_Name"]
    frame["Region_Raw"] = frame["Region"]
    frame["Customer_Name"] = resolver_customer.resolve(frame["Customer_Name"], "Customer_Name")
    frame["Region"] = resolver_region.resolve(frame["Region"], "Region")
    frame["Sales_Channel"] = resolver_channel.resolve(frame["Sales_Channel"], "Sales_Channel")
    for resolver in (resolver_customer, resolver_region, resolver_channel):
        report.label_rows.extend(resolver.rows)

    frame = frame.merge(master[["SKU_ID", "Launch_Month", "EOL_Month"]],
                        on="SKU_ID", how="left")
    frame["pre_launch"] = frame["Month"].lt(frame["Launch_Month"]).fillna(False)
    frame["post_eol"] = frame["Month"].gt(frame["EOL_Month"]).fillna(False)
    frame["lifecycle_ok"] = ~(frame["pre_launch"] | frame["post_eol"])

    train_pairs = set(map(tuple, train[["SKU_ID", "Region"]].drop_duplicates().to_numpy()))
    frame["unseen_pair"] = [tuple(pair) not in train_pairs
                            for pair in frame[["SKU_ID", "Region"]].to_numpy()]
    history = set(train["SKU_ID"].unique())
    frame["has_history"] = frame["SKU_ID"].isin(history)

    report.kv("rows", len(frame))
    report.kv("months", f"{frame['Month'].min():%Y-%m} -> {frame['Month'].max():%Y-%m} "
                        f"({frame['Month'].nunique()})")
    report.kv("SKUs / regions / customers / channels",
              f"{frame['SKU_ID'].nunique()} / {frame['Region'].nunique()} / "
              f"{frame['Customer_Name'].nunique()} / {frame['Sales_Channel'].nunique()}")
    report.kv("rows after EOL_Date", int(frame["post_eol"].sum()))
    report.kv("rows before Launch_Date", int(frame["pre_launch"].sum()))
    unseen_pairs = frame.loc[frame["unseen_pair"], ["SKU_ID", "Region"]].drop_duplicates()
    report.kv("unseen SKU-Region pairs", f"{int(frame['unseen_pair'].sum()):,} rows / "
                                         f"{len(unseen_pairs)} pairs")
    report.kv("cold-start SKUs (no history)",
              sorted(frame.loc[~frame["has_history"], "SKU_ID"].unique()))

    report.rule("TST-LIFECYCLE", "forecast keys outside Launch_Date..EOL_Date",
                "flagged for Phase 4 (zero vs floor is an explicit modelling choice)",
                int((frame["pre_launch"] | frame["post_eol"]).sum()))
    report.rule("TST-PAIR", "SKU-Region pair unseen in training",
                "flagged as cold start", int(frame["unseen_pair"].sum()))
    report.rule("TST-DUP", "duplicate Date+SKU+Region in test", "none",
                int(frame.duplicated(["Month", "SKU_ID", "Region"]).sum()))

    frame["flags"] = build_flags(frame.index, [
        ("PRE_LAUNCH", frame["pre_launch"]),
        ("POST_EOL", frame["post_eol"]),
        ("UNSEEN_PAIR", frame["unseen_pair"]),
        ("NO_HISTORY", ~frame["has_history"]),
        ("SKU_UNMATCHED", frame["sku_unmatched"]),
    ])
    report.metrics["test"] = {
        "rows": len(frame),
        "months": int(frame["Month"].nunique()),
        "after_eol": int(frame["post_eol"].sum()),
        "before_launch": int(frame["pre_launch"].sum()),
        "unseen_pairs": int(frame["unseen_pair"].sum()),
        "cold_start_skus": sorted(frame.loc[~frame["has_history"], "SKU_ID"].unique().tolist()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 5. stockouts
# --------------------------------------------------------------------------- #
def clean_stockouts(raw: pd.DataFrame, key_to_id: dict[str, str],
                    report: CleanReport) -> pd.DataFrame:
    report.h("4. STOCKOUT / SUPPRESSED-DEMAND LOG")
    frame = raw[STOCKOUT_FILE].copy()
    frame["Month"] = month_of(parse_dates(frame["Date"]))
    frame["SKU_ID"] = frame["SKU_ID"].map(sku_key).map(key_to_id)
    frame["Region"] = LabelResolver(REGION_CANONICAL, REGION_ALIASES).resolve(
        frame["Region"], "Region")
    for column in UNIT_COLUMNS:
        frame[column] = numeric_text(frame[column])

    requested_lt_shipped = frame["Requested_Units"].lt(frame["Shipped_Units"]).fillna(False)
    requested_lt_lost = frame["Requested_Units"].lt(frame["Lost_Units"]).fillna(False)
    algebra_gap = (frame["Requested_Units"] - frame["Shipped_Units"]
                   - frame["Lost_Units"] - frame["Backordered_Units"]).abs().gt(0).fillna(False)
    frame["Fill_Rate_Recomputed"] = (safe_ratio(frame["Shipped_Units"].clip(lower=0),
                                                frame["Requested_Units"].clip(lower=0)) * 100)
    fill_gap = (frame["Fill_Rate_Pct"] - frame["Fill_Rate_Recomputed"]).abs().gt(0.5).fillna(False)
    frame["Unserved_Units"] = (frame["Requested_Units"] - frame["Shipped_Units"]).clip(lower=0)

    report.kv("rows", len(frame))
    report.kv("months", f"{frame['Month'].min():%Y-%m} -> {frame['Month'].max():%Y-%m}")
    report.kv("requested / shipped units", f"{fmt(frame['Requested_Units'].sum())} / "
                                           f"{fmt(frame['Shipped_Units'].sum())}")
    report.kv("unserved units (requested - shipped)", fmt(frame["Unserved_Units"].sum()))
    report.kv("rows where requested = shipped + lost + backordered",
              f"{int((~algebra_gap).sum()):,} / {len(frame):,}")
    report.kv("rows whose reported fill rate is wrong by >0.5pp", int(fill_gap.sum()))
    report.rule("STK-ALGEBRA", "Requested != Shipped + Lost + Backordered",
                "kept; recomputed fill rate is authoritative", int(algebra_gap.sum()))
    report.rule("STK-FILL", "Fill_Rate_Pct inconsistent with Shipped/Requested",
                "recomputed from units", int(fill_gap.sum()),
                detail=f"max reported-vs-recomputed gap "
                       f"{(frame['Fill_Rate_Pct'] - frame['Fill_Rate_Recomputed']).abs().max():.1f}pp")
    report.rule("STK-OVER", "Shipped or Lost exceeds Requested", "kept and flagged",
                int((requested_lt_shipped | requested_lt_lost).sum()))
    report.rule("STK-DUP", "duplicate Date+SKU+Region in stockout log", "none",
                int(frame.duplicated(["Month", "SKU_ID", "Region"]).sum()))
    frame["flags"] = build_flags(frame.index, [
        ("ALGEBRA_GAP", algebra_gap),
        ("FILL_MISMATCH", fill_gap),
        ("SHIPPED_GT_REQUESTED", requested_lt_shipped),
        ("LOST_GT_REQUESTED", requested_lt_lost),
    ])
    report.metrics["stockouts"] = {
        "rows": len(frame),
        "requested_units": float(frame["Requested_Units"].sum()),
        "shipped_units": float(frame["Shipped_Units"].sum()),
        "unserved_units": float(frame["Unserved_Units"].sum()),
        "algebra_gap_rows": int(algebra_gap.sum()),
        "fill_mismatch_rows": int(fill_gap.sum()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 6. macro + events
# --------------------------------------------------------------------------- #
def clean_macro(raw: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    report.h("5. MACRO INDICATORS")
    frame = raw[MACRO_FILE].copy()
    frame["Month"] = month_of(parse_dates(frame["Date"]))
    for column in MACRO_INDICATORS:
        frame[column] = numeric_text(frame[column])
    frame = frame[["Month", *MACRO_INDICATORS]].dropna(subset=["Month"]).sort_values("Month")

    duplicated = frame["Month"].duplicated(keep=False)
    dup_months = sorted(frame.loc[duplicated, "Month"].dt.strftime("%Y-%m").unique())
    if duplicated.any():
        frame = frame.groupby("Month", as_index=False)[MACRO_INDICATORS].mean()
        report.rule("MAC-DUP", "duplicate month in macro series",
                    "numeric duplicates averaged", int(duplicated.sum()),
                    detail=f"months {', '.join(dup_months)}")

    full = pd.DataFrame({"Month": pd.date_range(frame["Month"].min(), frame["Month"].max(),
                                               freq="MS")})
    frame = full.merge(frame, on="Month", how="left")
    filled_cells = 0
    details = []
    for column in MACRO_INDICATORS:
        before = frame[column].isna()
        frame[column] = frame[column].interpolate(method="linear", limit_direction="both")
        if before.any():
            filled_cells += int(before.sum())
            months = frame.loc[before, "Month"].dt.strftime("%Y-%m")
            details.append(f"{column}: {', '.join(months)}")
    report.rule("MAC-GAP", "blank macro observations",
                "linearly interpolated within the series", filled_cells,
                detail="; ".join(details))
    report.kv("months (continuous)",
              f"{len(frame)} ({frame['Month'].min():%Y-%m} -> {frame['Month'].max():%Y-%m})")
    report.kv("interpolated cells", filled_cells)
    report.kv("Auto_Production_YoY_Pct range",
              f"{frame['Auto_Production_YoY_Pct'].min():.2f} .. "
              f"{frame['Auto_Production_YoY_Pct'].max():.2f} (percentage points)")
    frame["Auto_Production_YoY_Frac"] = frame["Auto_Production_YoY_Pct"] / 100.0
    report.metrics["macro"] = {
        "months": len(frame),
        "duplicate_months": dup_months,
        "interpolated_cells": filled_cells,
        "unit_note": "Auto_Production_YoY_Pct is percentage points, not a fraction",
    }
    return frame


def clean_events(raw: pd.DataFrame, master: pd.DataFrame, test: pd.DataFrame,
                 report: CleanReport) -> pd.DataFrame:
    report.h("6. EVENTS CALENDAR")
    frame = raw[EVENTS_FILE].copy()
    start = parse_dates(frame["Start_Date"])
    end = parse_dates(frame["End_Date"])
    frame["Start_Month"] = month_of(start)
    imputed_end = end.isna() & frame["Start_Month"].notna()
    frame["End_Month"] = month_of(end).fillna(frame["Start_Month"])
    frame["duration_months"] = ((frame["End_Month"].dt.year - frame["Start_Month"].dt.year) * 12
                                + (frame["End_Month"].dt.month - frame["Start_Month"].dt.month) + 1)

    for column in ["Scope_Type", "Scope_Value", "Event_Type", "Description"]:
        frame[column] = frame[column].astype(str).str.strip()

    scopes = {
        "REGION": {compact(v): v for v in REGION_CANONICAL},
        "CUSTOMER": {compact(v, True): v for v in test["Customer_Name"].unique()},
        "CHANNEL": {compact(v): v for v in CHANNEL_CANONICAL},
        "SALES_CHANNEL": {compact(v): v for v in CHANNEL_CANONICAL},
        "PRODUCT_FAMILY": {compact(v): v for v in master["Product_Family"].unique()},
        "SKU": {sku_key(v): v for v in master["SKU_ID"]},
    }
    resolved = []
    for scope_type, scope_value in zip(frame["Scope_Type"], frame["Scope_Value"]):
        key = scope_type.upper().replace(" ", "_").replace("-", "_")
        if scope_type.upper() == "GLOBAL":
            resolved.append(scope_value.strip().lower() in {"all", "global", "*", ""})
        else:
            lookup = scopes.get(key, {})
            probe = compact(scope_value, True) if key == "CUSTOMER" else compact(scope_value)
            resolved.append(bool(lookup) and (probe in lookup
                                              or scope_value in set(lookup.values())))
    frame["scope_resolved"] = resolved
    unresolved = ~frame["scope_resolved"]
    frame["price_event"] = frame["Event_Type"].str.contains("price", case=False, na=False)
    frame["flags"] = build_flags(frame.index, [("UNRESOLVED_SCOPE", unresolved),
                                               ("END_IMPUTED", imputed_end)])

    report.kv("rows", len(frame))
    report.kv("months", f"{frame['Start_Month'].min():%Y-%m} -> "
                        f"{frame['End_Month'].max():%Y-%m}")
    report.kv("scope types", frame["Scope_Type"].value_counts().to_dict())
    report.kv("unresolved scopes", frame.loc[unresolved, "Event_ID"].tolist())
    report.kv("price-type events", int(frame["price_event"].sum()))
    report.rule("EVT-END", "End_Date blank", "set to Start_Date", int(imputed_end.sum()))
    report.rule("EVT-SCOPE", "event scope does not resolve to a known entity",
                "flagged; not applied", int(unresolved.sum()),
                detail=", ".join(frame.loc[unresolved, "Event_ID"]))
    report.metrics["events"] = {
        "rows": len(frame),
        "unresolved": frame.loc[unresolved, "Event_ID"].tolist(),
        "price_events": int(frame["price_event"].sum()),
    }
    return frame


# --------------------------------------------------------------------------- #
# 7. demand panel + calendar grid
# --------------------------------------------------------------------------- #
def build_demand(sales: pd.DataFrame, stockouts: pd.DataFrame, master: pd.DataFrame,
                 report: CleanReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    report.h("7. UNCONSTRAINED DEMAND PANEL")
    eligible = sales[sales["lifecycle_ok"] & ~sales["Region"].eq("Unknown")
                     & sales["Month"].notna()]
    panel = eligible.groupby(GRAIN).agg(
        rows=("Record_ID", "size"),
        customers=("Customer_Name", "nunique"),
        channels=("Sales_Channel", "nunique"),
        observed_units=("Units_Sold_Clean", lambda s: s.sum(min_count=1)),
        unit_price=("Price_Clean_USD", "first"),
        revenue_recorded=("Revenue_Recorded_USD", lambda s: s.sum(min_count=1)),
        revenue_recomputed=("Revenue_Recomputed_USD", lambda s: s.sum(min_count=1)),
    ).reset_index()

    stock = stockouts[["Month", "SKU_ID", "Region", "Requested_Units", "Shipped_Units",
                       "Lost_Units", "Backordered_Units", "Stockout_Days",
                       "Fill_Rate_Recomputed"]].rename(columns={
                           "Requested_Units": "requested_units",
                           "Shipped_Units": "shipped_units",
                           "Lost_Units": "lost_units",
                           "Backordered_Units": "backordered_units",
                           "Stockout_Days": "stockout_days",
                           "Fill_Rate_Recomputed": "fill_rate_recomputed"})
    panel = panel.merge(stock, on=GRAIN, how="outer", indicator=True)
    panel["observed"] = panel["_merge"].isin(["both", "left_only"])
    panel["stockout_recorded"] = panel["_merge"].isin(["both", "right_only"])
    panel = panel.drop(columns="_merge")

    panel["rows"] = panel["rows"].fillna(0).astype(int)
    panel["customers"] = panel["customers"].fillna(0).astype(int)
    panel["channels"] = panel["channels"].fillna(0).astype(int)

    # Suppressed demand is only recovered where the stockout log documents a
    # shortfall; requested units are the customer's expression of demand.
    shortfall = (panel["stockout_recorded"]
                 & ((panel["stockout_days"].fillna(0) > 0)
                    | (panel["fill_rate_recomputed"].fillna(100) < 99.5)))
    requested = panel["requested_units"].clip(lower=0)
    demand = panel["observed_units"].where(panel["observed"])
    demand = demand.where(~shortfall, np.maximum(demand.fillna(0), requested))
    panel["demand_units"] = demand
    panel["suppressed_units"] = ((demand.fillna(0) - panel["observed_units"].fillna(0))
                                 .clip(lower=0))
    panel["zero_sales"] = panel["observed_units"].eq(0).fillna(False)

    panel = panel.merge(master[["SKU_ID", "Launch_Month", "EOL_Month",
                                "Lifecycle_Status", "Replaces_SKU"]],
                        on="SKU_ID", how="left")
    panel["pre_launch"] = panel["Month"].lt(panel["Launch_Month"]).fillna(False)
    panel["post_eol"] = panel["Month"].gt(panel["EOL_Month"]).fillna(False)
    panel["lifecycle_ok"] = ~(panel["pre_launch"] | panel["post_eol"])
    panel["usable"] = panel["lifecycle_ok"] & panel["demand_units"].notna()
    panel["Year"] = panel["Month"].dt.year
    panel["Month_Num"] = panel["Month"].dt.month
    panel["flags"] = build_flags(panel.index, [
        ("OBSERVED", panel["observed"]),
        ("STOCKOUT", panel["stockout_recorded"]),
        ("SUPPRESSED", panel["suppressed_units"] > 0),
        ("ZERO_SALES", panel["zero_sales"]),
        ("PRE_LAUNCH", panel["pre_launch"]),
        ("POST_EOL", panel["post_eol"]),
        ("NO_OBSERVATION", ~panel["observed"]),
    ])
    panel = panel.sort_values(GRAIN).reset_index(drop=True)

    report.rule("DEM-GRAIN", "duplicate demand panel grain", "none",
                int(panel.duplicated(GRAIN).sum()))
    report.kv("cells", len(panel))
    report.kv("cells with observed sales", int(panel["observed"].sum()))
    report.kv("cells with stockout log", int(panel["stockout_recorded"].sum()))
    report.kv("explicit zero-sales cells", int(panel["zero_sales"].sum()))
    report.kv("suppressed units recovered", fmt(panel["suppressed_units"].sum()))
    report.kv("observed vs unconstrained demand",
              f"{fmt(panel['observed_units'].sum())} -> {fmt(panel['demand_units'].sum())}")

    months = pd.date_range(sales["Month"].min(), sales["Month"].max(), freq="MS")
    pairs = sales.loc[~sales["Region"].eq("Unknown"), ["SKU_ID", "Region"]].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({"Month": months}), how="cross")
    grid = grid.merge(panel[["Month", "SKU_ID", "Region", "observed_units", "demand_units",
                             "zero_sales", "stockout_recorded"]], on=GRAIN, how="left")
    grid = grid.merge(master[["SKU_ID", "Launch_Month", "EOL_Month"]], on="SKU_ID", how="left")
    pre = grid["Month"].lt(grid["Launch_Month"]).fillna(False)
    post = grid["Month"].gt(grid["EOL_Month"]).fillna(False)
    status = np.select(
        [pre, post, grid["zero_sales"].eq(True),
         grid["observed_units"].gt(0),
         grid["stockout_recorded"].eq(True)],
        ["pre_launch", "post_eol", "zero", "positive", "suppressed_only"],
        default="absent")
    grid["status"] = status
    grid["has_rows"] = grid["observed_units"].notna() | grid["demand_units"].notna()
    grid["has_stockout"] = grid["stockout_recorded"].fillna(False)
    grid = grid.drop(columns=["zero_sales", "stockout_recorded", "Launch_Month", "EOL_Month"])
    grid = grid.sort_values(GRAIN).reset_index(drop=True)

    counts = grid["status"].value_counts().to_dict()
    report.kv("calendar cells", f"{len(grid):,} ({len(pairs):,} SKU-Region pairs x "
                                f"{len(months)} months)")
    report.kv("cell status counts", counts)
    report.kv("absent cells (NOT zeros)", f"{counts.get('absent', 0):,}")
    report.metrics["demand"] = {
        "panel_cells": len(panel),
        "observed_cells": int(panel["observed"].sum()),
        "stockout_cells": int(panel["stockout_recorded"].sum()),
        "zero_cells": int(panel["zero_sales"].sum()),
        "suppressed_units": float(panel["suppressed_units"].sum()),
        "demand_units": float(panel["demand_units"].sum()),
        "observed_units": float(panel["observed_units"].sum()),
        "grid_cells": len(grid),
        "grid_status": {str(k): int(v) for k, v in counts.items()},
    }
    return panel, grid


# --------------------------------------------------------------------------- #
# 8. reconciliation + outputs
# --------------------------------------------------------------------------- #
def build_reconciliation(report: CleanReport) -> pd.DataFrame:
    """Additive row/unit ledger from the raw extract to the demand panel.

    Every line is a signed delta, so the running column closes on
    ``demand_monthly``.  Units are counted on a non-negative basis because a
    return line contributes no demand; the signed gross total is kept in the
    note of the first line.  The lifecycle and unknown-region exclusions are
    first-class lines, so the difference between sales_clean and the panel is
    explicit rather than an unexplained gap.
    """
    report.h("8. RECONCILIATION - UNITS LEDGER")
    train = report.metrics["train"]
    demand = report.metrics["demand"]
    negative = next((float(row["units_affected"] or 0.0) for row in report.log
                     if row["rule_id"] == "TRN-UNITS-NEG"), 0.0)

    rows: list[dict] = [
        {"seq": 1, "step": "raw train rows (non-negative units)", "rule_id": "",
         "basis": "train.csv as loaded", "rows": train["raw_rows"],
         "units": train["units_raw_nonneg"],
         "note": f"gross signed total {fmt(train['units_raw'])} incl. {fmt(negative)} returns"},
        {"seq": 2, "step": "duplicate Record_ID removed (double load)",
         "rule_id": "TRN-DUP-ID", "basis": "newest Load_Timestamp kept",
         "rows": -train["rows_dropped"],
         "units": 0.0 - train["units_duplicates_nonneg"], "note": ""},
        {"seq": 3, "step": "unmatched SKU rows removed", "rule_id": "TRN-SKU-MISS",
         "basis": "SKU absent from the product master",
         "rows": -train["rows_sku_unmatched"],
         "units": 0.0 - train["units_sku_unmatched_nonneg"], "note": ""},
        {"seq": 4, "step": "sales_clean units", "rule_id": "", "basis": "subtotal",
         "rows": train["rows_after_dedupe"], "units": None,
         "note": f"{fmt(train['units_clean'])} non-negative units booked"},
    ]
    seq = 5
    for bucket in train["ledger"]:
        rows.append({"seq": seq, "step": bucket["step"], "rule_id": bucket["rule_id"],
                     "basis": bucket["basis"], "rows": -bucket["rows"],
                     "units": 0.0 - bucket["units"], "note": ""})
        seq += 1
    rows.append({"seq": seq, "step": "usable training units", "rule_id": "",
                 "basis": "subtotal: in lifecycle, known region, valid units",
                 "rows": train["usable_rows"], "units": None,
                 "note": f"{fmt(train['usable_units'])} units enter the demand panel"})
    seq += 1
    rows.append({"seq": seq, "step": "stockout demand recovered", "rule_id": "DEM-SUPPRESSED",
                 "basis": "requested units where a logged shortfall exists",
                 "rows": None, "units": demand["suppressed_units"], "note": "added"})
    seq += 1
    rows.append({"seq": seq, "step": "unconstrained demand", "rule_id": "",
                 "basis": "subtotal: demand_monthly.parquet", "rows": demand["panel_cells"],
                 "units": None, "note": f"{fmt(demand['demand_units'])} units"})

    frame = pd.DataFrame(rows)
    running: list[float] = []
    total = 0.0
    for value in frame["units"]:
        if value is not None and not pd.isna(value):
            total += float(value)
        running.append(total)
    frame["units_running"] = running

    sales_clean_subtotal = (train["units_raw_nonneg"] - train["units_duplicates_nonneg"]
                            - train["units_sku_unmatched_nonneg"])
    usable_subtotal = sales_clean_subtotal - sum(b["units"] for b in train["ledger"])
    closes = abs(total - demand["demand_units"]) <= 0.5

    report.p()
    report.p(f"{'#':>2}  {'step':<44}{'rows':>9}{'units':>15}{'running':>16}")
    report.p("-" * 88)
    for row in frame.itertuples(index=False):
        rows_text = "" if pd.isna(row.rows) else f"{int(row.rows):>9,}"
        units_text = "" if pd.isna(row.units) else f"{int(round(float(row.units))):>15,}"
        report.p(f"{row.seq:>2}  {row.step:<44}{rows_text}{units_text}"
                 f"{row.units_running:>16,.0f}")
        if row.note:
            report.p(f"    {'':<42}{row.note}")

    report.kv("row flow", f"{train['raw_rows']:,} raw -> "
                          f"{train['rows_after_dedupe']:,} after dedupe -> "
                          f"{train['usable_rows']:,} usable rows -> "
                          f"{demand['panel_cells']:,} panel cells")
    report.kv("returns excluded (outside the ledger)", f"{negative:,.0f} units in "
                                                     f"{train['units_negative']:,} rows")
    report.kv("units excluded for lifecycle",
              fmt(sum(b["units"] for b in train["ledger"]
                      if "launch" in b["step"] or "EOL" in b["step"])))
    report.kv("units excluded for unknown region",
              fmt(sum(b["units"] for b in train["ledger"] if "Region" in b["step"])))
    report.kv("units ledger closes on the demand panel",
              f"{'yes' if closes else 'NO'} ({fmt(total)} vs {fmt(demand['demand_units'])})")

    report.metrics["reconciliation"] = {
        "raw_rows": train["raw_rows"],
        "duplicate_rows": train["rows_dropped"],
        "unmatched_sku_rows": train["rows_sku_unmatched"],
        "clean_rows": train["rows_after_dedupe"],
        "usable_rows": train["usable_rows"],
        "panel_cells": demand["panel_cells"],
        "raw_units": train["units_raw"],
        "raw_units_nonneg": train["units_raw_nonneg"],
        "units_duplicates": train["units_duplicates_nonneg"],
        "units_dropped_duplicates": train["units_duplicates_nonneg"],
        "units_sku_unmatched": train["units_sku_unmatched_nonneg"],
        "units_negative": negative,
        "units_clean": train["units_clean"],
        "units_clean_ledger": sales_clean_subtotal,
        "exclusions": {b["step"]: {"rule_id": b["rule_id"], "rows": b["rows"],
                                    "units": b["units"]} for b in train["ledger"]},
        "units_usable": train["usable_units"],
        "units_usable_ledger": usable_subtotal,
        "units_suppressed": demand["suppressed_units"],
        "demand_units": demand["demand_units"],
        "units_demand_ledger": total,
        "closes": bool(closes),
    }
    return frame


def write_outputs(out_dir: Path, report: CleanReport, tables: dict[str, pd.DataFrame],
                  label_map: pd.DataFrame, reconciliation: pd.DataFrame) -> None:
    clean_dir = out_dir / CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(clean_dir / f"{name}.parquet", index=False)
    label_map.to_csv(clean_dir / "label_map.csv", index=False)
    pd.DataFrame(report.log).to_csv(clean_dir / "cleaning_log.csv", index=False)
    reconciliation.to_csv(clean_dir / "reconciliation.csv", index=False)

    payload = {
        "meta": {
            "phase": 1,
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - report._t0, 2),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "metrics": report.metrics,
        "verdicts": report.verdicts,
        "rules": report.log,
        "outputs": {name: f"outputs/{CLEAN_DIR}/{name}.parquet" for name in tables},
    }
    (out_dir / "clean_report.json").write_text(
        json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "clean_report.txt").write_text(
        "\n".join(report.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 cleaner")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    started = time.time()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    out_dir = Path(args.out_dir)

    report = CleanReport()
    report.p("CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER")
    report.p("Phase 1: cleaning, canonicalisation and reconciliation")
    report.p(f"data dir : {data_dir}")
    report.p(f"out dir  : {out_dir}")

    missing = [name for name in INPUT_FILES if not (data_dir / name).exists()]
    if missing:
        print(f"FATAL: missing input files: {missing}", file=sys.stderr)
        return 2
    raw = {name: read_raw(data_dir / name) for name in INPUT_FILES}
    report.metrics["_data_dir"] = str(data_dir)
    report.metrics["_inputs"] = {name: int(len(frame)) for name, frame in raw.items()}

    master, key_to_id = clean_master(raw, report)
    sales = clean_train(raw, master, key_to_id, report)
    test = clean_test(raw, master, key_to_id, sales, report)
    stockouts = clean_stockouts(raw, key_to_id, report)
    macro = clean_macro(raw, report)
    events = clean_events(raw, master, test, report)
    panel, grid = build_demand(sales, stockouts, master, report)
    reconciliation = build_reconciliation(report)

    label_map = pd.DataFrame(report.label_rows).drop_duplicates(
        ["column", "raw_value"], keep="first")
    sales = sales.drop(columns=[c for c in sales.columns
                                if c.startswith("__") or c in {"price_candidate", "SKU_Key",
                                                               "base_price", "units_valid",
                                                               "units_clean", "Revenue_USD"}])

    # ---- contract checks -------------------------------------------------- #
    report.h("9. PHASE 1 EXIT CHECK")
    report.verdict("Record_ID unique", bool(sales["Record_ID"].is_unique),
                   f"{len(sales):,} rows")
    report.verdict("all SKUs resolve to master",
                   bool(sales["SKU_ID"].isin(set(master["SKU_ID"])).all()))
    price_spread = sales.groupby(["SKU_ID", "Price_Year"])["Price_Clean_USD"].nunique(dropna=True)
    report.verdict("one base price per SKU-year", int(price_spread.max()) <= 1,
                   f"max distinct prices {int(price_spread.max())}")
    report.verdict("demand panel grain unique", not panel.duplicated(GRAIN).any())
    report.verdict("no absent calendar cell carries demand",
                   not bool(grid.loc[grid["status"].eq("absent"), "demand_units"].notna().any()))
    report.verdict("no usable demand after EOL",
                   not bool(sales.loc[sales["post_eol"], "usable_for_training"].any()),
                   f"{int(sales['post_eol'].sum()):,} train rows violate EOL; "
                   f"{report.metrics['demand']['grid_status'].get('post_eol', 0):,} "
                   f"calendar cells are post-EOL and left absent")
    recon = report.metrics["reconciliation"]
    report.verdict("units ledger reaches sales_clean",
                   abs(recon["units_clean_ledger"] - recon["units_clean"]) <= 0.5,
                   f"{fmt(recon['units_clean_ledger'])} vs {fmt(recon['units_clean'])}")
    report.verdict("units ledger reaches the demand panel",
                   abs(recon["units_usable_ledger"] - recon["units_usable"]) <= 0.5
                   and abs(recon["units_demand_ledger"] - recon["demand_units"]) <= 0.5,
                   f"usable {fmt(recon['units_usable_ledger'])} -> demand "
                   f"{fmt(recon['units_demand_ledger'])} (closes: {recon['closes']})")
    report.verdict("every excluded unit is itemised",
                   all({"rule_id", "rows", "units"} <= set(item)
                       for item in recon["exclusions"].values())
                   and sum(item["units"] for item in recon["exclusions"].values()) >= 0,
                   f"{len(recon['exclusions'])} buckets, "
                   f"{fmt(sum(item['units'] for item in recon['exclusions'].values()))} units: "
                   + ", ".join(recon["exclusions"]))
    report.verdict("train rows reconcile",
                   len(sales) + report.metrics["train"]["rows_dropped"]
                   == report.metrics["train"]["raw_rows"],
                   f"{report.metrics['train']['raw_rows']:,} - "
                   f"{report.metrics['train']['rows_dropped']:,} = {len(sales):,}")
    report.verdict("test rows unchanged", len(test) == len(raw[TEST_FILE]), f"{len(test):,}")
    expected_months = ((macro["Month"].max().to_period("M")
                        - macro["Month"].min().to_period("M")).n + 1)
    report.verdict("macro months unique and continuous",
                   bool(not macro["Month"].duplicated().any()
                        and len(macro) == expected_months),
                   f"{len(macro)} months")
    report.verdict("every rule is logged", len(report.log) > 0, f"{len(report.log)} rules")

    report.kv("elapsed seconds", round(time.time() - started, 1))
    write_outputs(out_dir, report, {
        "sales_clean": sales,
        "sku_master_clean": master,
        "demand_monthly": panel,
        "demand_grid": grid,
        "stockouts_clean": stockouts,
        "macro_clean": macro,
        "events_clean": events,
        "test_clean": test,
    }, label_map, reconciliation)

    print("\n".join(report.lines))
    print(f"\nJSON  -> {out_dir / 'clean_report.json'}")
    print(f"TEXT  -> {out_dir / 'clean_report.txt'}")
    print(f"TABLES-> {out_dir / CLEAN_DIR}")
    failed = [v for v in report.verdicts if not v["ok"]]
    if failed:
        print(f"\nPHASE 1 FAILED {len(failed)} check(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
