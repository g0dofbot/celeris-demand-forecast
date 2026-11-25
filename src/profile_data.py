#!/usr/bin/env python
"""Phase 0 - Forensic profiling of the Celeris Automobili demand dataset.

Reads all seven raw CSVs as *untyped text* (``dtype=str``, no NA coercion) and
reports what is actually inside them.  Nothing is cleaned here: no rule below
alters a value.  The purpose is to let the cleaning phase be written against
observed evidence instead of assumptions.

Covered per file: schema and inference, blanks/nulls, string-format variants,
date-format variants, duplicate grains, price dispersion, Units * Price vs
Revenue, calendar gaps, zero vs missing rows, master-data contradictions,
macro anomalies, event scope validity, stockout algebra, and forecast-window
integrity.

Outputs
-------
outputs/profile_report.json   machine-readable findings (all sections)
outputs/profile_report.txt    the console transcript
outputs/profile/*.csv         row-level evidence tables

Usage
-----
    python src/profile_data.py --data-dir student_data
    python run_pipeline.py --phase 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "student_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs"

MAX_LIST = 10
NA_TOKENS = {"na", "n/a", "nan", "null", "none", "-", "?", "--", "unknown", "n.a."}

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
MACRO_FILE = "macro_indicators.csv"
MASTER_FILE = "sku_master.csv"
EVENTS_FILE = "events_calendar.csv"
STOCKOUT_FILE = "inventory_stockouts.csv"
SUB_FILE = "sample_submission.csv"
ALL_FILES = [TRAIN_FILE, TEST_FILE, MACRO_FILE, MASTER_FILE, EVENTS_FILE, STOCKOUT_FILE, SUB_FILE]

# Shared grain of the demand fact table.
GRAIN = ["Date", "SKU_ID", "Region"]
GRAIN_CUSTOMER = ["Date", "SKU_ID", "Region", "Customer_Name"]
GRAIN_CHANNEL = ["Date", "SKU_ID", "Region", "Sales_Channel"]

DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$"), "YYYY-MM-DD HH:MM[:SS]"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "YYYY-MM-DD"),
    (re.compile(r"^\d{4}-\d{2}$"), "YYYY-MM (month)"),
    (re.compile(r"^\d{4}-\d{1}$|^\d{4}-\d{1,2}-\d{1,2}$"), "unpadded YYYY-M[-D]"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "YYYY/MM/DD"),
    (re.compile(r"^\d{2}[/-]\d{2}[/-]\d{4}$"), "DD/MM/YYYY or MM/DD/YYYY (ambiguous)"),
    (re.compile(r"^\d{1,2}-[A-Za-z]{3,9}-\d{4}$"), "DD-Mon-YYYY"),
    (re.compile(r"^[A-Za-z]{3,9}[ ,-]+\d{4}$"), "Mon YYYY"),
    (re.compile(r"^\d{4}$"), "YYYY (year only)"),
    (re.compile(r"^\d{5}$"), "Excel serial (5-digit day count)"),
    (re.compile(r"^\d{10,13}$"), "epoch integer"),
]
DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%b %Y",
    "%Y",
)


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def fmt(value, digits: int = 4) -> str:
    """Human readable scalar for the console transcript."""
    if value is None:
        return "n/a"
    if isinstance(value, (np.bool_, bool)):
        return "yes" if value else "no"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return str(value)
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.{digits}f}".rstrip("0").rstrip(".")
    if isinstance(value, (pd.Period, pd.Timestamp)):
        return str(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v) for v in value) if value else "-"
    return str(value)


def jsonable(obj):
    """Recursively convert numpy / pandas scalars into JSON-safe objects."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Period, pd.Timestamp)):
        return str(obj)
    if isinstance(obj, pd.Interval):
        return str(obj)
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class Report:
    """Collects console lines, metrics and findings; writes both artifacts."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.findings: list[dict] = []
        self.metrics: dict = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.verdicts: list[dict] = []
        self.profile_dir: Path = DEFAULT_OUT_DIR / "profile"
        self._t0 = time.time()
        self._section_t0 = self._t0

    # -- output primitives -------------------------------------------------- #
    def h(self, title: str) -> None:
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"[profile] {title}", file=sys.stderr, flush=True)
        self._section_t0 = time.time()

    def sub(self, title: str) -> None:
        self.lines.append(f"\n-- {title} " + "-" * max(0, 60 - len(title)))

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def kv(self, key: str, value, indent: int = 2, width: int = 42) -> None:
        self.lines.append(f"{' ' * indent}{key:<{width}} {fmt(value)}")

    def note(self, text: str, indent: int = 2) -> None:
        for line in textwrap.wrap(text, 96 - indent):
            self.lines.append(" " * indent + line)

    def table(self, frame: pd.DataFrame, indent: int = 2, max_rows: int = MAX_LIST) -> None:
        if frame is None or frame.empty:
            self.lines.append(" " * indent + "(empty)")
            return
        view = frame.head(max_rows)
        body = view.to_string(index=False, max_colwidth=34, float_format=lambda v: f"{v:,.4g}")
        for line in body.splitlines():
            self.lines.append(" " * indent + line)
        if len(frame) > max_rows:
            self.lines.append(" " * indent + f"... {len(frame) - max_rows:,} more row(s)")

    def finding(self, fid: str, severity: str, issue: str, count=None, evidence=None,
                section: str = "") -> None:
        self.findings.append({
            "id": fid,
            "severity": severity,
            "section": section,
            "issue": issue,
            "count": jsonable(count),
            "evidence": jsonable(evidence),
        })

    def verdict(self, phase: int, name: str, ok: bool, detail: str = "") -> None:
        self.verdicts.append({"phase": phase, "check": name, "ok": bool(ok), "detail": detail})
        self.p(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    # -- persistence -------------------------------------------------------- #
    def save_table(self, name: str, frame: pd.DataFrame, out_dir: Path) -> None:
        if frame is None:
            return
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        self.tables[name] = frame

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "phase": 0,
                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "runtime_seconds": round(time.time() - self._t0, 2),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
            "findings": self.findings,
            "verdicts": self.verdicts,
            "metrics": self.metrics,
            "evidence_tables": {k: f"outputs/profile/{k}.csv" for k in self.tables},
        }
        json_path = out_dir / "profile_report.json"
        json_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
        txt_path = out_dir / "profile_report.txt"
        txt_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return json_path, txt_path


# --------------------------------------------------------------------------- #
# parsing helpers (inspection only -- never mutate the input frames)
# --------------------------------------------------------------------------- #
def read_raw(path: Path) -> pd.DataFrame:
    """Read a CSV with zero type inference and zero NA coercion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def blank_mask(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip()
    return s.eq("") | s.str.lower().isin(NA_TOKENS)


def to_num(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a raw text column to numbers, tolerating $ , % and (1,234) forms."""
    s = raw.astype(str).str.strip()
    cleaned = s.str.replace(r"[,$%\s]", "", regex=True)
    neg = cleaned.str.match(r"^\(.*\)$", na=False)
    cleaned = cleaned.mask(neg, "-" + cleaned.str.strip("()"))
    return pd.to_numeric(cleaned, errors="coerce"), s


# Cheap single-pass patterns for type sniffing (never used for parsing).
NUMERIC_RE = r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
DATE_ANY_RE = "(?:" + ")|(?:".join(p.pattern for p, _ in DATE_PATTERNS) + ")"


def date_format_labels(s: pd.Series) -> pd.Series:
    """Vectorised classification of observed date spellings (first match wins)."""
    labels = pd.Series("unrecognised", index=s.index, dtype="object")
    blank = s.eq("") | s.str.lower().isin(NA_TOKENS)
    labels[blank] = "blank/NA"
    remaining = ~blank
    for pattern, label in DATE_PATTERNS:
        if not remaining.any():
            break
        hit = remaining & s.str.match(pattern, na=False)
        if hit.any():
            labels[hit] = label
            remaining &= ~hit
    return labels


MIN_YEAR, MAX_YEAR = 1900, 2100


def coerce_dates(text: pd.Series, date_format: str | None) -> pd.Series:
    """Parse a text slice to ns datetimes, sending implausible years to NaT.

    A single corrupt year (e.g. 1064) would otherwise raise OutOfBoundsDatetime
    and abort the whole audit, so out-of-range values are masked rather than
    allowed to poison the column.
    """
    try:
        parsed = (pd.to_datetime(text, format=date_format, errors="coerce")
                  if date_format else
                  pd.to_datetime(text, errors="coerce", format="mixed"))
    except (TypeError, ValueError):
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    parsed = pd.Series(parsed, index=text.index)
    if not parsed.empty:
        years = parsed.dt.year
        implausible = years.notna() & ((years < MIN_YEAR) | (years > MAX_YEAR))
        if implausible.any():
            parsed = parsed.mask(implausible)
    return parsed.astype("datetime64[ns]")


def parse_dates(raw: pd.Series) -> pd.Series:
    """Format-aware parse: fast paths per observed pattern, generic fallback last."""
    s = raw.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for pattern, date_format in (
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S"),
        (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", "%Y-%m-%dT%H:%M:%S"),
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{4}-\d{2}$", "%Y-%m"),
        (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),
        (r"^\d{1,2}-[A-Za-z]{3}-\d{4}$", "%d-%b-%Y"),
        (r"^[A-Za-z]{3,9}[ ,-]+\d{4}$", "%b %Y"),
        (r"^\d{4}$", "%Y"),
    ):
        todo = out.isna() & s.str.match(pattern, na=False)
        if todo.any():
            out.loc[todo] = coerce_dates(s[todo], date_format)
    todo = out.isna() & s.ne("") & ~s.str.lower().isin(NA_TOKENS)
    if todo.any():
        out.loc[todo] = coerce_dates(s[todo], None)
    return out


def normalise_key(value: str) -> str:
    """Lowercase, strip punctuation/whitespace -- used to spot label variants."""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 &\-/.]", "", text)
    return text.strip()


def variant_groups(series: pd.Series, max_group: int = MAX_LIST) -> pd.DataFrame:
    """Raw spellings that collapse to the same normalised key."""
    raw = series.astype(str)
    keys = (raw.str.strip().str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^a-z0-9 &\-/.]", "", regex=True)
            .str.strip())
    frame = pd.DataFrame({"raw": raw, "key": keys})
    frame = frame[frame["key"].ne("")]
    grouped = frame.groupby("key")["raw"].agg(lambda values: sorted(set(values)))
    multi = grouped[grouped.map(len) > 1]
    counts = frame["key"].value_counts()
    rows = []
    for key, variants in multi.items():
        rows.append({
            "normalised": key,
            "n_variants": len(variants),
            "variants": " | ".join(variants[:max_group]),
            "n_rows": int(counts.get(key, 0)),
        })
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False) if rows else pd.DataFrame()


def near_duplicates(series: pd.Series, cutoff: float = 0.92) -> pd.DataFrame:
    """Fuzzy near-duplicate labels (small-cardinality columns only)."""
    import difflib

    values = sorted({str(v).strip() for v in series.unique() if str(v).strip()})
    if not (1 < len(values) <= 150):
        return pd.DataFrame()
    rows = []
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            if normalise_key(left) == normalise_key(right):
                continue
            ratio = difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()
            if ratio >= cutoff:
                rows.append({"value_a": left, "value_b": right, "similarity": round(ratio, 3)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("similarity", ascending=False)


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Section 1-2: files, schema, in-the-wild column types
# --------------------------------------------------------------------------- #
def resolve_data_dir(requested: str) -> Path:
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if (candidate / TRAIN_FILE).exists():
        return candidate
    for found in sorted(PROJECT_ROOT.glob("student_data*/student_data")):
        if (found / TRAIN_FILE).exists():
            return found
    raise FileNotFoundError(f"Could not find {TRAIN_FILE} under {candidate}")


def sniff_column(raw: pd.Series, sample_size: int = 20_000) -> dict:
    """Infer what the column really contains, from the values themselves.

    Counts are computed on the whole column (cheap, C-level); shape tests run on
    a deterministic head sample so profiling stays fast on 250k-row files.
    """
    s = raw.astype(str).str.strip()
    blanks = blank_mask(raw)
    n = int(len(s))
    nonblank = s[~blanks]
    info = {"n": n, "blank": int(blanks.sum()), "unique": int(nonblank.nunique())}
    if len(nonblank) == 0:
        info.update(kind="empty", top=[])
        return info

    sample = nonblank.head(sample_size)
    num_ok = float(sample.str.fullmatch(NUMERIC_RE, na=False).mean())
    if num_ok >= 0.95:
        info["kind"] = "numeric"
        numeric, _ = to_num(sample)
        info["range"] = [float(numeric.min()), float(numeric.max())]
    else:
        date_ok = float(sample.str.match(DATE_ANY_RE, na=False).mean())
        if date_ok >= 0.9:
            parsed = parse_dates(sample)
            info["kind"] = "date" if parsed.notna().mean() >= 0.9 else "categorical"
            if info["kind"] == "date":
                info["range"] = [str(parsed.min()), str(parsed.max())]
        elif info["unique"] == len(nonblank) and len(nonblank) > 50:
            info["kind"] = "key/text"
        else:
            info["kind"] = "categorical"

    counts = nonblank.value_counts().head(6)
    info["top"] = [f"{idx} ({cnt:,})" for idx, cnt in counts.items()]
    return info


def section_files(report: Report, data_dir: Path, raw: dict[str, pd.DataFrame]) -> None:
    report.h("1. FILE INVENTORY & FINGERPRINTS")
    rows = []
    for name in ALL_FILES:
        path = data_dir / name
        frame = raw[name]
        rows.append({
            "file": name,
            "rows": len(frame),
            "cols": frame.shape[1],
            "size_MB": round(path.stat().st_size / 1e6, 2),
            "md5": md5_of(path)[:12],
        })
    report.table(pd.DataFrame(rows))
    report.metrics["files"] = rows
    report.p("\n  md5 prefixes are included so every later phase can prove it read the same bytes.")


def section_schema(report: Report, raw: dict[str, pd.DataFrame]) -> None:
    report.h("2. SCHEMA & IN-THE-WILD COLUMN TYPES")
    schema: dict[str, dict] = {}
    for name in ALL_FILES:
        frame = raw[name]
        report.sub(f"{name}  ({len(frame):,} rows x {frame.shape[1]} cols)")
        per_col = {}
        rows = []
        for col in frame.columns:
            info = sniff_column(frame[col])
            per_col[col] = info
            extra = ""
            if info["kind"] == "numeric" and "range" in info:
                extra = f"[{fmt(info['range'][0])}, {fmt(info['range'][1])}]"
            rows.append({
                "column": col,
                "kind": info["kind"],
                "blank": info["blank"],
                "unique": info["unique"],
                "range/notes": extra,
                "examples": "; ".join(info["top"][:3]),
            })
        report.table(pd.DataFrame(rows), max_rows=frame.shape[1])
        schema[name] = per_col
    report.metrics["schema"] = schema


# --------------------------------------------------------------------------- #
# Section 3-4: label variants and date formats
# --------------------------------------------------------------------------- #
VARIANT_COLUMNS = {
    TRAIN_FILE: ["Product_Family", "Business_Unit", "SKU_ID", "SKU_Description", "Customer_Name",
                 "Sales_Channel", "Region"],
    TEST_FILE: ["Product_Family", "Business_Unit", "SKU_ID", "SKU_Description", "Customer_Name",
                "Sales_Channel", "Region"],
    MASTER_FILE: ["Product_Family", "Business_Unit", "SKU_ID", "Size_Class",
                  "Powertrain_Application", "Lifecycle_Status", "Replaces_SKU"],
    EVENTS_FILE: ["Scope_Type", "Scope_Value", "Event_Type"],
    STOCKOUT_FILE: ["SKU_ID", "Region", "Reason"],
}


def section_labels(report: Report, raw: dict[str, pd.DataFrame], out_dir: Path) -> None:
    report.h("3. LABEL & STRING-VARIANT AUDIT")
    report.note("Each column is normalised (case, whitespace, punctuation). Any normalised key "
                "that maps to more than one raw spelling is a variant cluster.")
    report.metrics["variants"] = {}
    for name, columns in VARIANT_COLUMNS.items():
        frame = raw[name]
        found_any = False
        for col in columns:
            if col not in frame.columns:
                continue
            groups = variant_groups(frame[col])
            if not groups.empty:
                found_any = True
                report.sub(f"{name}.{col}")
                report.table(groups, max_rows=8)
                report.save_table(f"variants_{name.replace('.csv', '')}_{col}", groups, out_dir)
                report.metrics["variants"][f"{name}.{col}"] = jsonable(groups.to_dict("records"))
                report.finding(f"LBL-{name.replace('.csv', '')}-{col}", "medium",
                               f"Multiple raw spellings collapse to one normalised value in {name}.{col}",
                               int(groups["n_rows"].sum()), groups.head(5).to_dict("records"),
                               "labels")
            fuzzy = near_duplicates(frame[col])
            if not fuzzy.empty:
                report.sub(f"{name}.{col} (near-duplicate spellings, no exact normalisation match)")
                report.table(fuzzy, max_rows=6)
                report.save_table(f"fuzzy_{name.replace('.csv', '')}_{col}", fuzzy, out_dir)
                report.finding(f"FUZ-{name.replace('.csv', '')}-{col}", "low",
                               f"Near-duplicate (fuzzy) spellings in {name}.{col}",
                               len(fuzzy), fuzzy.head(5).to_dict("records"), "labels")
        if not found_any:
            report.sub(f"{name}: no case/whitespace/punctuation variant clusters found")
            report.metrics["variants"][name] = "clean"


DATE_COLUMNS = {
    TRAIN_FILE: ["Date", "Load_Timestamp"],
    TEST_FILE: ["Date"],
    MACRO_FILE: ["Date"],
    MASTER_FILE: ["Launch_Date", "EOL_Date"],
    EVENTS_FILE: ["Start_Date", "End_Date"],
    STOCKOUT_FILE: ["Date"],
}


def section_dates(report: Report, raw: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, pd.DataFrame]:
    report.h("4. DATE-FORMAT AUDIT")
    parsed: dict[str, pd.DataFrame] = {}
    for name, columns in DATE_COLUMNS.items():
        frame = raw[name].copy()
        for col in columns:
            if col not in frame.columns:
                continue
            labels = date_format_labels(frame[col].astype(str).str.strip())
            values = parse_dates(frame[col])
            counts = labels.value_counts()
            frame[f"{col}__parsed"] = values
            report.sub(f"{name}.{col}")
            report.kv("recognised formats (n distinct)", int((counts > 0).sum()))
            report.table(counts.rename_axis("format").reset_index(name="n_rows"), max_rows=10)
            blanks = int(blank_mask(frame[col]).sum())
            unparseable = int((~blank_mask(frame[col]) & values.isna()).sum())
            report.kv("blank", blanks)
            report.kv("unparseable non-blank", unparseable)

            # a leading 4-digit year outside 1900..2100 is corruption, not a real date
            leading_year = pd.to_numeric(
                frame[col].astype(str).str.strip().str.extract(r"^(\d{4})")[0], errors="coerce")
            implausible = leading_year.notna() & ((leading_year < MIN_YEAR) | (leading_year > MAX_YEAR))
            report.kv("implausible year (<1900 or >2100)", int(implausible.sum()))
            if implausible.any():
                examples = frame.loc[implausible, col].unique()[:MAX_LIST]
                report.kv("implausible examples", list(examples))
                report.finding(f"YEAR-{name}-{col}", "high",
                               f"{int(implausible.sum()):,} values in {name}.{col} carry an impossible year",
                               int(implausible.sum()), list(examples), "dates")
            if values.notna().any():
                report.kv("min / max", f"{values.min()} / {values.max()}")
                report.kv("distinct timestamps", int(values.nunique()))
            bad = frame.loc[values.isna() & ~blank_mask(frame[col]), col].unique()[:MAX_LIST]
            if len(bad):
                report.kv("unparseable examples", list(bad))
            report.metrics[f"dates.{name}.{col}"] = {
                "formats": {k: int(v) for k, v in counts.items()},
                "blank": blanks,
                "unparseable": unparseable,
                "min": str(values.min()) if values.notna().any() else None,
                "max": str(values.max()) if values.notna().any() else None,
            }
            if unparseable:
                report.finding(f"DATE-{name}-{col}", "high",
                               f"{unparseable:,} non-blank values in {name}.{col} cannot be parsed as a date",
                               unparseable, list(bad), "dates")
            if len(counts) > 1 and "month" in name:
                pass
            if len(counts) > 1:
                report.finding(f"FMT-{name}-{col}", "medium",
                               f"{len(counts)} different date formats coexist in {name}.{col}",
                               int(counts.sum()), {k: int(v) for k, v in counts.items()}, "dates")
        parsed[name] = frame
        if columns:
            bad_rows = frame[frame[[c for c in columns if f"{c}__parsed" in frame]].isna().any(axis=1)]
            if not bad_rows.empty:
                report.save_table(f"bad_dates_{name.replace('.csv', '')}",
                                  bad_rows.head(200)[columns], out_dir)
    return parsed


# --------------------------------------------------------------------------- #
# Section 5: train.csv deep dive
# --------------------------------------------------------------------------- #
def grain_audit(frame: pd.DataFrame, keys: list[str], label: str) -> dict | None:
    if not all(key in frame.columns for key in keys):
        return None
    duplicates = frame.duplicated(subset=keys, keep=False)
    n_rows = int(duplicates.sum())
    if n_rows == 0:
        return {"grain": label, "rows_in_dup_groups": 0, "dup_groups": 0, "extra_rows": 0,
                "conflicts": {}}
    grouped = frame.loc[duplicates].groupby(keys, dropna=False)
    n_groups = int(grouped.ngroups)
    conflicts = {}
    for col in ["Units_Sold", "Unit_Price_USD", "Revenue_USD", "Customer_Name", "Sales_Channel",
                "Record_ID", "Load_Timestamp"]:
        if col in frame.columns:
            distinct = grouped[col].nunique(dropna=False)
            conflicts[col] = int((distinct > 1).sum())
    return {
        "grain": label,
        "rows_in_dup_groups": n_rows,
        "dup_groups": n_groups,
        "extra_rows": n_rows - n_groups,
        "conflicts": conflicts,
    }


def section_train(report: Report, raw: dict[str, pd.DataFrame], master: pd.DataFrame,
                  stockouts: pd.DataFrame, out_dir: Path) -> dict:
    report.h("5. TRAIN.CSV DEEP DIVE")
    frame = raw[TRAIN_FILE].copy()
    n = len(frame)
    report.kv("rows x cols", f"{n:,} x {frame.shape[1]}")

    frame["Date__dt"] = parse_dates(frame["Date"])
    frame["__ym"] = frame["Date__dt"].dt.to_period("M")
    units, _ = to_num(frame["Units_Sold"])
    price, _ = to_num(frame["Unit_Price_USD"])
    revenue, _ = to_num(frame["Revenue_USD"])
    frame["__units"] = units
    frame["__price"] = price
    frame["__revenue"] = revenue
    frame["__load"] = parse_dates(frame["Load_Timestamp"]) if "Load_Timestamp" in frame else pd.NaT

    # --- out-of-window dates (year typos) ---------------------------------- #
    # History must stop where the scored window starts (test.csv begins 2026-11).
    # A handful of rows carry year 2062 while their Year column says 2026: that is
    # a keying error, not future demand, and it would otherwise wreck every
    # calendar/gap/cadence statistic in this section.
    hist_end = pd.Period("2026-10", "M")
    future_mask = frame["__ym"].notna() & (frame["__ym"] > hist_end)
    report.sub("Date window sanity (history must end 2026-10; scored window starts 2026-11)")
    report.kv("rows dated after 2026-10", int(future_mask.sum()))
    report.metrics["train.future_dated_rows"] = int(future_mask.sum())
    if future_mask.any():
        typo = frame.loc[future_mask]
        typo_years = sorted({int(y) for y in typo["Date__dt"].dt.year})
        typo_year_num, _ = to_num(typo["Year"])
        report.kv("typo'd years present", typo_years)
        report.kv("rows whose Year column disagrees with the typo'd year",
                  int((typo_year_num != typo["Date__dt"].dt.year).sum()))
        report.kv("bogus months", ", ".join(str(m) for m in sorted(set(typo["__ym"]))))
        report.table(typo[["Record_ID", "Date", "Year", "Month", "SKU_ID", "Region",
                           "Units_Sold"]].head(8))
        report.save_table("train_future_dated_rows", typo[
            ["Record_ID", "Date", "Year", "Month", "SKU_ID", "Region", "Sales_Channel",
             "Customer_Name", "Units_Sold", "Unit_Price_USD", "Revenue_USD"]] , out_dir)
        report.finding("TRN-DATE-FUTURE", "high",
                       "Date values after the last history month (year typo, e.g. 2062 for 2026); "
                       "the Year column proves the intended year",
                       int(future_mask.sum()), {"typo_years": typo_years}, "train")
        # analysis-only repair: keep month/day, take the year from the Year column
        broken = frame.loc[future_mask, "Date__dt"]
        fixed = pd.to_datetime({
            "year": typo_year_num.astype("int64").to_numpy(),
            "month": broken.dt.month.to_numpy(),
            "day": broken.dt.day.to_numpy(),
        }, errors="coerce")
        frame.loc[future_mask, "Date__dt"] = np.asarray(fixed, dtype="datetime64[us]")
        # rebuild the whole period column from the repaired dates (cheap, cannot half-align)
        frame["__ym"] = frame["Date__dt"].dt.to_period("M")
        report.kv("rows still undated after repair", int(frame["__ym"].isna().sum()))
        report.kv("repaired months (analysis view)",
                  ", ".join(str(m) for m in sorted(set(frame.loc[future_mask, "__ym"]))))
        report.note("Repair applies to the analysis view only; raw files are untouched. Phase 1 must "
                    "apply the same rule so the rows are re-dated rather than dropped or double-counted.")

    # --- coverage ---------------------------------------------------------- #
    report.sub("Calendar coverage")
    months = frame["__ym"].dropna()
    month_counts = months.value_counts().sort_index()
    expected = pd.period_range(months.min(), months.max(), freq="M")
    missing_months = [str(m) for m in expected if m not in month_counts.index]
    report.kv("first / last month", f"{months.min()} / {months.max()}")
    report.kv("months present (expected)", f"{month_counts.size} ({len(expected)})")
    report.kv("missing calendar months", missing_months if missing_months else "none")
    report.kv("rows per month (min / median / max)",
              f"{month_counts.min():,} / {int(month_counts.median()):,} / {month_counts.max():,}")
    report.metrics["train.coverage"] = {
        "first": str(months.min()), "last": str(months.max()),
        "months_present": int(month_counts.size), "months_expected": int(len(expected)),
        "missing_months": missing_months,
        "rows_per_month": {str(k): int(v) for k, v in month_counts.items()},
    }
    if missing_months:
        report.finding("TRN-MONTH-GAPS", "high", "Calendar months absent from train.csv",
                       len(missing_months), missing_months, "train")

    # monthly row-count drift is a strong signal of dropped loads
    drift = month_counts.pct_change().abs()
    if drift.notna().any():
        worst = drift.sort_values(ascending=False).head(5)
        report.kv("largest month-over-month row-count jump",
                  ", ".join(f"{k}: {v:.1%}" for k, v in worst.items()))

    report.sub("Dimensions")
    for col in ["Product_Family", "Business_Unit", "Region", "Sales_Channel", "Customer_Name",
                "SKU_ID", "SKU_Description"]:
        if col in frame.columns:
            report.kv(f"distinct {col}", int(frame[col].nunique()))
    report.metrics["train.dimensions"] = {
        col: int(frame[col].nunique())
        for col in ["Product_Family", "Business_Unit", "Region", "Sales_Channel", "Customer_Name", "SKU_ID"]
        if col in frame.columns
    }
    report.sub("Region presence (first observed month)")
    region_first = frame.groupby("Region")["__ym"].min().sort_values()
    region_last = frame.groupby("Region")["__ym"].max().sort_values()
    region_rows = frame["Region"].value_counts()
    report.table(pd.DataFrame({
        "Region": region_first.index,
        "first_month": [str(region_first[r]) for r in region_first.index],
        "last_month": [str(region_last[r]) for r in region_first.index],
        "rows": [int(region_rows[r]) for r in region_first.index],
    }))
    for region, first in region_first.items():
        if str(first) > str(months.min()):
            report.finding(f"TRN-REGION-OPEN-{region}", "medium",
                           f"Region '{region}' first appears in {first}, not at the start of history",
                           int(region_rows[region]), str(first), "train")

    # --- duplicates -------------------------------------------------------- #
    report.sub("Duplicate / grain audit")
    audits = [grain_audit(frame, GRAIN, "Date+SKU+Region"),
              grain_audit(frame, GRAIN_CUSTOMER, "Date+SKU+Region+Customer"),
              grain_audit(frame, GRAIN_CHANNEL, "Date+SKU+Region+Channel"),
              grain_audit(frame, ["Record_ID"], "Record_ID")]
    audits = [a for a in audits if a]
    report.table(pd.DataFrame([{
        "grain": a["grain"],
        "rows_in_dup_groups": a["rows_in_dup_groups"],
        "dup_groups": a["dup_groups"],
        "extra_rows": a["extra_rows"],
        "conflicting_fields": a["conflicts"],
    } for a in audits]), max_rows=10)
    report.metrics["train.grains"] = audits

    # Which grain is the natural fact grain?
    natural = next((a for a in audits if a["extra_rows"] == 0 and a["grain"] != "Record_ID"), None)
    report.kv("natural unique grain",
              natural["grain"] if natural else "none of the tested grains is unique")
    report.finding("TRN-GRAIN", "info",
                   "Unique fact grain of train.csv is "
                   + (natural["grain"] if natural else "not one of the tested key sets"),
                   None, [a for a in audits if a["grain"] != "Record_ID"], "train")

    base = next((a for a in audits if a["grain"] == "Date+SKU+Region"), None)
    if base and base["extra_rows"]:
        dup_rows = frame[frame.duplicated(subset=GRAIN, keep=False)].sort_values(GRAIN + ["Record_ID"])
        report.sub("Date+SKU+Region duplicates (sample)")
        report.table(dup_rows[GRAIN + ["Customer_Name", "Units_Sold", "Unit_Price_USD",
                                       "Revenue_USD", "Record_ID", "Load_Timestamp"]].head(8))
        report.save_table("train_dup_grain_sample",
                          dup_rows[GRAIN + ["Customer_Name", "Sales_Channel", "Units_Sold",
                                            "Unit_Price_USD", "Revenue_USD", "Record_ID",
                                            "Load_Timestamp"]].head(500), out_dir)
        report.finding("TRN-DUP-GRAIN", "high",
                       "Repeated Date+SKU+Region rows in train.csv (customer/order level rows)",
                       base["extra_rows"], base["conflicts"], "train")

    record_dupes = frame[frame.duplicated(subset=["Record_ID"], keep=False)]
    if not record_dupes.empty:
        report.sub("Record_ID duplicates: payload comparison")
        grouped = record_dupes.sort_values(["Record_ID", "__load"]).groupby("Record_ID")
        same_payload = 0
        conflicting = 0
        stale_newer = 0
        for _, block in grouped:
            compare = block.drop(columns=["__load", "Date__dt", "__ym", "__units", "__price",
                                          "__revenue", "Load_Timestamp"], errors="ignore")
            identical = compare.nunique(dropna=False).max() == 1
            same_payload += int(identical)
            conflicting += int(not identical)
            loads = block["__load"].dropna()
            if len(loads) and loads.nunique() > 1 and block["Load_Timestamp"].iloc[-1] != str(loads.max()):
                stale_newer += 1
        report.kv("Record_ID values duplicated", int(record_dupes["Record_ID"].nunique()))
        report.kv("rows involved", len(record_dupes))
        report.kv("groups with identical payload", same_payload)
        report.kv("groups with conflicting payload", conflicting)
        report.kv("groups where latest Load_Timestamp is not last row", stale_newer)
        report.table(record_dupes.sort_values(["Record_ID", "Load_Timestamp"])
                     .head(8)[["Record_ID", "Date", "SKU_ID", "Region", "Units_Sold",
                               "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"]])
        report.save_table("train_record_dupes",
                          record_dupes.sort_values(["Record_ID", "Load_Timestamp"])
                          .head(500)[["Record_ID", "Date", "SKU_ID", "Region", "Units_Sold",
                                      "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"]], out_dir)
        report.metrics["train.record_id_dupes"] = {
            "duplicated_ids": int(record_dupes["Record_ID"].nunique()),
            "rows": len(record_dupes),
            "identical_payload_groups": same_payload,
            "conflicting_payload_groups": conflicting,
        }
        report.finding("TRN-DUP-ID", "high",
                       "Record_ID appears more than once (double load); needs latest-Load_Timestamp dedupe",
                       len(record_dupes), {"ident": same_payload, "conflict": conflicting}, "train")

    # --- Year / Month contradiction --------------------------------------- #
    report.sub("Year / Month vs Date")
    year_num, _ = to_num(frame["Year"])
    month_num, _ = to_num(frame["Month"])
    parsed_ok = frame["__ym"].notna()
    bad_year = parsed_ok & (year_num != frame["Date__dt"].dt.year) & year_num.notna()
    bad_month = parsed_ok & (month_num != frame["Date__dt"].dt.month) & month_num.notna()
    report.kv("Year != Date.year", int(bad_year.sum()))
    report.kv("Month != Date.month", int(bad_month.sum()))
    report.kv("Date unparseable", int((~parsed_ok).sum()))
    mismatch = frame.loc[bad_year | bad_month,
                         ["Date", "Year", "Month", "SKU_ID", "Region", "Units_Sold"]]
    report.table(mismatch.head(8))
    if len(mismatch):
        report.save_table("train_year_month_mismatch", mismatch.head(500), out_dir)
        report.finding("TRN-YM", "high", "Year/Month columns contradict the Date column",
                       len(mismatch), mismatch.head(8).to_dict("records"), "train")
    else:
        report.p("  no Year/Month contradictions")

    # --- Units ------------------------------------------------------------- #
    report.sub("Units_Sold")
    units_stats = {
        "blank": int(blank_mask(frame["Units_Sold"]).sum()),
        "unparseable": int(((~blank_mask(frame["Units_Sold"])) & units.isna()).sum()),
        "decimal": int(((units.dropna() % 1) != 0).sum()),
        "negative": int((units < 0).sum()),
        "zero": int((units == 0).sum()),
        "p50": float(units.median()) if units.notna().any() else None,
        "p95": float(units.quantile(0.95)) if units.notna().any() else None,
        "p99": float(units.quantile(0.99)) if units.notna().any() else None,
        "max": float(units.max()) if units.notna().any() else None,
    }
    report.table(pd.DataFrame([units_stats]).T.rename(columns={0: "value"}).reset_index()
                 .rename(columns={"index": "metric"}).assign(value=lambda d: d["value"].map(fmt)))
    report.kv("zero share of rows", f"{(units == 0).mean():.3%}")
    report.metrics["train.units"] = units_stats
    if units_stats["negative"]:
        report.finding("TRN-UNITS-NEG", "high", "Negative Units_Sold values",
                       units_stats["negative"], None, "units")
    if units_stats["blank"]:
        report.finding("TRN-UNITS-BLANK", "high", "Blank Units_Sold in train",
                       units_stats["blank"], None, "units")

    top_units = frame.nlargest(8, "__units")[GRAIN + ["Customer_Name", "Units_Sold", "Unit_Price_USD"]]
    report.sub("Largest unit observations")
    report.table(top_units)

    # --- Price audit ------------------------------------------------------ #
    report.sub("Unit price audit")
    price_stats = {
        "blank": int(blank_mask(frame["Unit_Price_USD"]).sum()),
        "unparseable": int(((~blank_mask(frame["Unit_Price_USD"])) & price.isna()).sum()),
        "zero": int((price == 0).sum()),
        "negative": int((price < 0).sum()),
        "distinct across all rows": int(price.nunique()),
        "min": float(price.min()) if price.notna().any() else None,
        "median": float(price.median()) if price.notna().any() else None,
        "max": float(price.max()) if price.notna().any() else None,
    }
    report.table(pd.DataFrame([price_stats]).T.rename(columns={0: "value"}).reset_index()
                 .rename(columns={"index": "metric"}).assign(value=lambda d: d["value"].map(fmt)))
    report.metrics["train.price"] = price_stats

    per_sku = frame.groupby("SKU_ID")["__price"].agg(
        n_price="nunique", n_rows="size", p_min="min", p_max="max", p_median="median")
    per_sku["ratio"] = per_sku["p_max"] / per_sku["p_min"].replace(0, np.nan)
    report.kv("SKUs with >1 distinct price", int((per_sku["n_price"] > 1).sum()))
    report.kv("SKUs with max/min price ratio > 1.5", int((per_sku["ratio"] > 1.5).sum()))
    report.kv("max within-SKU price ratio", float(per_sku["ratio"].max()))
    report.table(per_sku.sort_values("ratio", ascending=False).head(8).reset_index()
                 .assign(**{c: lambda d, c=c: d[c].map(lambda v: round(float(v), 2)) for c in
                            ["p_min", "p_median", "p_max", "ratio"]}))

    sku_month_price = frame.groupby(["SKU_ID", "__ym"])["__price"].nunique()
    cross_region = sku_month_price[sku_month_price > 1]
    report.kv("SKU-months where price differs across regions", int(cross_region.size))
    if cross_region.size:
        sample = cross_region.reset_index(name="n_prices").head(8)
        report.table(sample)
        report.finding("TRN-PRICE-REGION", "high",
                       "Price is not identical across regions within SKU-month",
                       int(cross_region.size), sample.to_dict("records"), "price")

    # price changes and their month-of-year
    pivot = (frame.dropna(subset=["__ym", "__price"])
             .groupby(["SKU_ID", "__ym"])["__price"].median().sort_index())
    deltas = pivot.groupby(level=0).diff()
    mask = deltas.abs() > 1e-9
    changes = deltas[mask].reset_index()
    if not changes.empty:
        changes["month_of_year"] = changes["__ym"].astype(str).str[-2:].astype(int)
        previous = pivot.shift(1)[mask].to_numpy(dtype=float)
        previous = np.where(previous == 0, np.nan, previous)
        changes["pct_change"] = deltas[mask].to_numpy(dtype=float) / previous - 1
        by_month = changes["month_of_year"].value_counts().sort_index()
        report.kv("SKU-level price change events", len(changes))
        report.kv("share occurring in January", f"{(changes['month_of_year'] == 1).mean():.1%}")
        report.kv("share outside January", f"{(changes['month_of_year'] != 1).mean():.1%}")
        report.kv("median |price change|", f"{changes['pct_change'].abs().median():.2%}")
        report.table(by_month.rename_axis("month_of_year").reset_index(name="n_price_changes"))
        report.save_table("price_change_calendar", changes, out_dir)
        report.metrics["train.price_changes"] = {
            "n": int(len(changes)),
            "january_share": float((changes["month_of_year"] == 1).mean()),
            "by_month": {int(k): int(v) for k, v in by_month.items()},
        }
        non_jan = changes[changes["month_of_year"] != 1]
        if len(non_jan):
            report.finding("TRN-PRICE-NONJAN", "medium",
                           "SKU price changes outside 1 January (need event-calendar justification)",
                           len(non_jan), non_jan.head(8).to_dict("records"), "price")
    else:
        report.p("  no within-SKU price changes detected")

    # currency / unit scale errors
    median_price = frame.groupby("SKU_ID")["__price"].transform("median")
    ratio_to_sku = frame["__price"] / median_price.replace(0, np.nan)
    scaled = frame[(ratio_to_sku > 20) | (ratio_to_sku < 0.05)]
    report.kv("rows with price >20x or <1/20 of SKU median", len(scaled))
    if len(scaled):
        report.table(scaled.head(8)[GRAIN + ["Unit_Price_USD", "Units_Sold", "Revenue_USD"]])
        report.save_table("price_scale_outliers", scaled.head(500), out_dir)
        report.finding("TRN-PRICE-SCALE", "high",
                       "Price scale outliers consistent with unit/currency errors",
                       len(scaled), scaled.head(8).to_dict("records"), "price")
    report.metrics["train.price_scale_outliers"] = int(len(scaled))

    # --- Revenue identity -------------------------------------------------- #
    report.sub("Revenue identity: Units * Price vs Revenue_USD")
    work = frame.dropna(subset=["__units", "__price", "__revenue"]).copy()
    work = work[(work["__units"] > 0) & (work["__price"] > 0)]
    expected_rev = work["__units"] * work["__price"]
    diff = work["__revenue"] - expected_rev
    tolerance = np.maximum(0.05, expected_rev.abs() * 5e-4)
    bad = diff.abs() > tolerance
    ratio = work["__revenue"] / expected_rev.replace(0, np.nan)
    buckets = pd.cut(ratio, bins=[-np.inf, 0, 0.02, 0.2, 0.8, 1.2, 5, 50, 500, np.inf],
                     labels=["<=0", "0-0.02", "0.02-0.2", "0.2-0.8", "0.8-1.2", "1.2-5",
                             "5-50", "50-500", ">500"])
    report.kv("rows tested", len(work))
    report.kv("rows failing identity", int(bad.sum()))
    report.kv("failure share", f"{bad.mean():.3%}" if len(work) else "n/a")
    report.kv("median |diff| on failures", f"{diff[bad].abs().median():,.2f}" if bad.any() else "n/a")
    report.table(buckets.value_counts().sort_index().rename_axis("revenue/expected").reset_index(name="n_rows"))
    if bad.any():
        cols = GRAIN + ["Units_Sold", "Unit_Price_USD", "Revenue_USD"]
        sample = work.loc[bad, cols].head(10)
        report.table(sample)
        report.save_table("revenue_identity_failures",
                          work.loc[bad, cols].head(2000), out_dir)
        report.metrics["train.revenue_identity"] = {
            "rows_tested": int(len(work)),
            "failures": int(bad.sum()),
            "share": float(bad.mean()),
            "ratio_median": float(ratio[bad].median()) if bad.any() else None,
        }
        report.finding("TRN-REV", "high", "Units_Sold * Unit_Price_USD != Revenue_USD beyond rounding",
                       int(bad.sum()), sample.to_dict("records"), "revenue")
    else:
        report.metrics["train.revenue_identity"] = {"rows_tested": int(len(work)), "failures": 0}
        report.p("  identity holds within tolerance for every priced row")

    # --- zero vs missing --------------------------------------------------- #
    report.sub("Zero vs missing rows (grain completeness)")
    observed = (frame.dropna(subset=["__ym"])
                .groupby(["SKU_ID", "Region", "__ym"]).size().rename("n_rows").reset_index())
    keys = observed[["SKU_ID", "Region"]].drop_duplicates()
    months_df = pd.DataFrame({"__ym": expected})
    grid = keys.merge(months_df, how="cross")
    grid = grid.merge(observed, on=["SKU_ID", "Region", "__ym"], how="left")
    grid["present"] = grid["n_rows"].notna()
    # a cell is an "explicit zero" when it has rows and every one of them booked 0 units
    all_zero = (frame.groupby(["SKU_ID", "Region", "__ym"])["__units"]
                .agg(lambda values: bool((values == 0).all())).rename("all_zero").reset_index())
    grid = grid.merge(all_zero, on=["SKU_ID", "Region", "__ym"], how="left")
    grid["zero_units"] = grid["all_zero"].fillna(False).astype(bool)
    report.kv("SKU-Region pairs", f"{len(keys):,}")
    report.kv("grid cells (pairs x months)", f"{len(grid):,}")
    report.kv("cells with rows", f"{grid['present'].sum():,}")
    report.kv("cells with no row (implicit missing)", f"{(~grid['present']).sum():,}")
    report.kv("of present cells, explicit zeros", f"{grid['zero_units'].sum():,}")

    # classify missing cells against lifecycle windows
    launch_map, eol_map = {}, {}
    if not master.empty:
        launch_map = dict(zip(master["SKU_ID"], parse_dates(master["Launch_Date"])))
        eol_map = dict(zip(master["SKU_ID"], parse_dates(master["EOL_Date"])))
    grid["launch"] = grid["SKU_ID"].map(launch_map)
    grid["eol"] = grid["SKU_ID"].map(eol_map)
    cell_start = grid["__ym"].dt.start_time
    before_launch = (~grid["present"]) & grid["launch"].notna() & (cell_start < grid["launch"])
    after_eol = (~grid["present"]) & grid["eol"].notna() & (cell_start > grid["eol"])
    inside_gap = (~grid["present"]) & ~before_launch & ~after_eol
    report.kv("missing: before Launch_Date (legitimate)", int(before_launch.sum()))
    report.kv("missing: after EOL_Date (legitimate)", int(after_eol.sum()))
    report.kv("missing: inside SKU lifetime (candidate data gap)", int(inside_gap.sum()))

    # per pair: internal gaps between first and last observation
    span = (observed.groupby(["SKU_ID", "Region"])["__ym"]
            .agg(["min", "max", "count"]).reset_index())
    span["months_in_span"] = span.apply(
        lambda r: len(pd.period_range(r["min"], r["max"], freq="M")), axis=1)
    span["internal_missing"] = span["months_in_span"] - span["count"]
    report.kv("pairs with internal gaps", int((span["internal_missing"] > 0).sum()))
    report.kv("total internal missing cells", int(span["internal_missing"].sum()))
    report.table(span.sort_values("internal_missing", ascending=False).head(6)
                 [["SKU_ID", "Region", "min", "max", "count", "months_in_span", "internal_missing"]])
    report.save_table("train_span_gaps", span, out_dir)

    # are missing cells explained by the stockout log?
    if not stockouts.empty and "__ym" in stockouts.columns:
        so_keys = stockouts[["SKU_ID", "Region", "__ym"]].drop_duplicates()
        so_keys["logged_stockout"] = True
        gap_cells = grid.loc[inside_gap, ["SKU_ID", "Region", "__ym"]]
        gap_with_log = gap_cells.merge(so_keys, on=["SKU_ID", "Region", "__ym"], how="left")
        n_logged = int(gap_with_log["logged_stockout"].notna().sum())
        report.kv("missing cells that have a stockout log entry", n_logged)
        report.metrics["train.missing_vs_stockout"] = {
            "missing_inside_life": int(inside_gap.sum()),
            "with_stockout_log": n_logged,
        }
        if n_logged:
            report.finding("TRN-GAP-STOCKOUT", "high",
                           "Rows are entirely missing in months that the stockout log records as supplied",
                           n_logged, None, "train")

    report.metrics["train.grid"] = {
        "pairs": int(len(keys)),
        "cells": int(len(grid)),
        "present": int(grid["present"].sum()),
        "missing": int((~grid["present"]).sum()),
        "missing_before_launch": int(before_launch.sum()),
        "missing_after_eol": int(after_eol.sum()),
        "missing_inside_life": int(inside_gap.sum()),
        "explicit_zeros": int(grid["zero_units"].sum()),
    }
    report.finding("TRN-GAP", "high",
                   "Implicit missing SKU-Region-month rows inside SKU lifetimes (a missing row is not a zero)",
                   int(inside_gap.sum()),
                   {"pairs_with_internal_gaps": int((span["internal_missing"] > 0).sum())}, "train")

    # --- ordering cadence / customer switches ------------------------------ #
    report.sub("Customer ordering cadence")
    cust = (frame.dropna(subset=["__ym"])
            .groupby("Customer_Name")["__ym"].agg(["nunique", "size", "min", "max"]))
    cust["months_between_orders"] = (cust["max"].apply(lambda p: p.ordinal)
                                     - cust["min"].apply(lambda p: p.ordinal)) / cust["nunique"]
    cadence = pd.cut(cust["months_between_orders"], bins=[0, 1.5, 2.5, 3.5, 6.5, 12.5, np.inf],
                     labels=["monthly", "bi-monthly", "quarterly", "4-6 months",
                             "7-12 months", ">12 months"])
    report.kv("distinct customers", int(cust.shape[0]))
    report.table(cadence.value_counts().sort_index().rename_axis("approx cadence").reset_index(name="customers"))
    report.metrics["train.cadence"] = {str(k): int(v) for k, v in cadence.value_counts().sort_index().items()}
    cust_pairs = frame.groupby("Customer_Name")["SKU_ID"].nunique()
    report.kv("customers ordering >1 SKU", int((cust_pairs > 1).sum()))
    report.kv("customers with a single order month", int((cust["nunique"] == 1).sum()))
    report.save_table("customer_cadence", cust.reset_index(), out_dir)

    # customer churn / last order before the end of history
    last_month = months.max()
    dormant = cust[cust["max"] < last_month]
    report.kv("customers with no order in the final month", int(len(dormant)))
    report.kv("median months since their last order",
              float((last_month.ordinal - dormant["max"].apply(lambda p: p.ordinal)).median())
              if len(dormant) else 0)
    report.metrics["train.customers"] = {
        "n": int(cust.shape[0]),
        "single_month": int((cust["nunique"] == 1).sum()),
        "dormant_at_end": int(len(dormant)),
    }

    # --- per-SKU activity -------------------------------------------------- #
    report.sub("SKU activity window")
    sku_active = (frame.dropna(subset=["__ym"]).groupby("SKU_ID")["__ym"]
                  .agg(first_obs="min", last_obs="max", n_months="nunique", n_rows="size"))
    report.kv("SKUs with rows", int(len(sku_active)))
    report.kv("SKUs observed in all months", int((sku_active["n_months"] == len(expected)).sum()))
    report.table(sku_active.sort_values("n_months").head(6).reset_index())
    report.metrics["train.sku_activity"] = {
        "skus": int(len(sku_active)),
        "full_history": int((sku_active["n_months"] == len(expected)).sum()),
    }

    report.verdict(0, "train.csv parsed without type errors",
                   bool(parsed_ok.all() or parsed_ok.mean() > 0.999),
                   f"{int((~parsed_ok).sum()):,} unparseable Date values")
    report.verdict(0, "Units_Sold numeric", bool(units_stats["blank"] == 0 and units_stats["unparseable"] == 0),
                   f"{units_stats['blank']:,} blank, {units_stats['unparseable']:,} unparseable")
    report.verdict(0, "No negative units", bool(units_stats["negative"] == 0),
                   f"{units_stats['negative']:,} negatives")

    return {"frame": frame, "grid": grid, "months": list(expected)}


# --------------------------------------------------------------------------- #
# Section 6: master data
# --------------------------------------------------------------------------- #
def section_master(report: Report, raw: dict[str, pd.DataFrame], train: dict,
                   test: pd.DataFrame) -> None:
    report.h("8. SKU MASTER INTEGRITY")
    master = raw[MASTER_FILE].copy()
    report.kv("rows", len(master))

    report.sub("Keys")
    dup_ids = master["SKU_ID"][master["SKU_ID"].duplicated(keep=False)]
    report.kv("duplicate SKU_IDs", int(dup_ids.nunique()))
    if len(dup_ids):
        report.table(master[master["SKU_ID"].isin(dup_ids)].sort_values("SKU_ID").head(8))
        report.finding("MST-DUP", "high", "Duplicate SKU_ID in sku_master", int(dup_ids.nunique()), None, "master")

    train_frame = train["frame"]
    train_skus = set(train_frame["SKU_ID"].unique())
    test_skus = set(test["SKU_ID"].unique()) if "SKU_ID" in test.columns else set()
    master_skus = set(master["SKU_ID"].unique())
    report.kv("SKUs in train", f"{len(train_skus):,}")
    report.kv("SKUs in test", f"{len(test_skus):,}")
    report.kv("SKUs in master", f"{len(master_skus):,}")
    report.kv("in train, missing from master", len(train_skus - master_skus))
    report.kv("in master, never sold in train", len(master_skus - train_skus))
    report.kv("in test, missing from master", len(test_skus - master_skus))
    missing_master = sorted(train_skus - master_skus)[:MAX_LIST]
    if train_skus - master_skus:
        report.kv("examples", missing_master)
    report.metrics["master.keys"] = {
        "train": len(train_skus), "test": len(test_skus), "master": len(master_skus),
        "train_not_master": sorted(train_skus - master_skus),
        "master_not_train": sorted(master_skus - train_skus),
        "test_not_master": sorted(test_skus - master_skus),
    }
    if train_skus - master_skus:
        report.finding("MST-COVER-TRAIN", "high", "SKUs sold in train with no master record",
                       len(train_skus - master_skus), missing_master, "master")
    if test_skus - master_skus:
        report.finding("MST-COVER-TEST", "high", "SKUs in the forecast window with no master record",
                       len(test_skus - master_skus), sorted(test_skus - master_skus)[:MAX_LIST], "master")

    report.sub("Lifecycle dates")
    launch = parse_dates(master["Launch_Date"])
    eol = parse_dates(master["EOL_Date"])
    launch_blank = int(blank_mask(master["Launch_Date"]).sum())
    eol_blank = int(blank_mask(master["EOL_Date"]).sum())
    report.kv("Launch_Date blank", launch_blank)
    report.kv("Launch_Date unparseable", int((~blank_mask(master["Launch_Date"]) & launch.isna()).sum()))
    report.kv("EOL_Date blank (still active)", eol_blank)
    report.kv("EOL_Date populated", int(launch.size - eol_blank))
    report.kv("Launch_Date min / max", f"{launch.min()} / {launch.max()}")
    report.kv("EOL_Date min / max", f"{eol.min()} / {eol.max()}")
    inverted = launch.notna() & eol.notna() & (eol < launch)
    report.kv("EOL_Date before Launch_Date", int(inverted.sum()))
    if inverted.any():
        report.table(master.loc[inverted, ["SKU_ID", "Launch_Date", "EOL_Date", "Lifecycle_Status"]].head(8))
        report.finding("MST-EOL-BEFORE-LAUNCH", "high", "EOL_Date precedes Launch_Date",
                       int(inverted.sum()), None, "master")

    report.sub("Lifecycle_Status values")
    report.table(master["Lifecycle_Status"].value_counts().rename_axis("status").reset_index(name="n_skus"))
    cadence = pd.DataFrame({
        "status": master["Lifecycle_Status"],
        "has_eol": eol.notna(),
        "launch_in_forecast_window": launch >= pd.Timestamp("2026-11-01"),
        "launch_after_history": launch > pd.Timestamp("2026-10-31"),
    }).groupby(["status", "has_eol"]).size().rename("n_skus").reset_index()
    report.table(cadence)
    report.metrics["master.lifecycle"] = jsonable(
        master.assign(launch=launch, eol=eol)
        .groupby("Lifecycle_Status")
        .agg(n=("SKU_ID", "size"), has_eol=("eol", lambda s: int(s.notna().sum())),
             launch_min=("launch", "min"), launch_max=("launch", "max"))
        .reset_index().to_dict("records"))

    report.sub("Powertrain / size class")
    report.table(master["Powertrain_Application"].value_counts().rename_axis("powertrain").reset_index(name="n_skus"))
    report.table(master["Size_Class"].value_counts().head(10).rename_axis("size_class").reset_index(name="n_skus"))

    report.sub("Master vs transaction contradictions")
    tx = train_frame[["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description"]].drop_duplicates()
    joined = tx.merge(master[["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description"]],
                      on="SKU_ID", how="inner", suffixes=("_train", "_master"))
    contradictions = {}
    for col in ["Product_Family", "Business_Unit", "SKU_Description"]:
        left, right = f"{col}_train", f"{col}_master"
        mismatch = joined[left].map(normalise_key) != joined[right].map(normalise_key)
        contradictions[col] = int(joined.loc[mismatch, "SKU_ID"].nunique())
        report.kv(f"SKUs where {col} disagrees", contradictions[col])
        if contradictions[col]:
            report.table(joined.loc[mismatch, ["SKU_ID", left, right]].drop_duplicates().head(6))
    report.metrics["master.contradictions"] = contradictions
    if any(contradictions.values()):
        report.finding("MST-CONTRADICTION", "high",
                       "SKU attributes differ between transaction rows and the master",
                       contradictions, None, "master")

    report.sub("Replaces_SKU graph")
    replaces = master.loc[master["Replaces_SKU"].str.strip().ne(""), ["SKU_ID", "Replaces_SKU"]]
    report.kv("SKUs declaring a predecessor", len(replaces))
    dangling = replaces[~replaces["Replaces_SKU"].isin(master_skus)]
    self_ref = replaces[replaces["SKU_ID"] == replaces["Replaces_SKU"]]
    shared = replaces["Replaces_SKU"].value_counts()
    report.kv("dangling predecessor refs", len(dangling))
    report.kv("self references", len(self_ref))
    report.kv("predecessors reused by >1 SKU", int((shared > 1).sum()))
    if len(dangling):
        report.table(dangling.head(6))
        report.finding("MST-REPLACE-DANGLING", "medium", "Replaces_SKU points to unknown SKU_ID",
                       len(dangling), dangling.head(6).to_dict("records"), "master")
    report.metrics["master.replaces"] = {
        "declared": int(len(replaces)),
        "dangling": int(len(dangling)),
        "self": int(len(self_ref)),
        "shared": int((shared > 1).sum()),
    }

    report.sub("Sales outside lifecycle window")
    span = (train_frame.dropna(subset=["__ym"]).groupby("SKU_ID")["__ym"].agg(["min", "max"]).reset_index())
    span = span.merge(master.assign(launch=launch, eol=eol)[["SKU_ID", "launch", "eol"]],
                      on="SKU_ID", how="left")
    before = span[span["launch"].notna() & (span["min"].dt.start_time < span["launch"])]
    after = span[span["eol"].notna() & (span["max"].dt.start_time > span["eol"])]
    report.kv("SKUs with sales before Launch_Date", len(before))
    report.kv("SKUs with sales after EOL_Date", len(after))
    if len(before):
        report.table(before.head(6))
        report.finding("MST-BEFORE-LAUNCH", "high", "Sales recorded before the SKU's Launch_Date",
                       len(before), before.head(6).to_dict("records"), "master")
    if len(after):
        report.table(after.head(6))
        report.finding("MST-AFTER-EOL", "high", "Sales recorded after the SKU's EOL_Date",
                       len(after), after.head(6).to_dict("records"), "master")

    report.sub("Cold-start candidates (launch inside the forecast window)")
    cold = master.loc[launch > pd.Timestamp("2026-10-31"),
                      ["SKU_ID", "Product_Family", "Powertrain_Application", "Launch_Date",
                       "Replaces_SKU", "Lifecycle_Status"]]
    report.kv("SKUs launching in 2026-11 .. 2028-05", len(cold))
    report.kv("of those, with a Replaces_SKU predecessor", int(cold["Replaces_SKU"].str.strip().ne("").sum()))
    report.kv("of those, present in test.csv", int(cold["SKU_ID"].isin(test_skus).sum()))
    if len(cold):
        report.table(cold.head(10))
        report.save_table("cold_start_skus", cold, report.profile_dir)
    report.metrics["master.cold_start"] = {
        "n": int(len(cold)),
        "with_predecessor": int(cold["Replaces_SKU"].str.strip().ne("").sum()),
        "in_test": int(cold["SKU_ID"].isin(test_skus).sum()),
    }


# --------------------------------------------------------------------------- #
# Section 7: macro indicators
# --------------------------------------------------------------------------- #
def section_macro(report: Report, raw: dict[str, pd.DataFrame]) -> None:
    report.h("9. MACRO INDICATORS AUDIT")
    frame = raw[MACRO_FILE].copy()
    report.kv("rows x cols", f"{len(frame):,} x {frame.shape[1]}")
    frame["__date"] = parse_dates(frame["Date"])
    frame["__ym"] = frame["__date"].dt.to_period("M")
    months = frame["__ym"].dropna()
    expected = pd.period_range("2019-01", "2028-05", freq="M")
    dup_months = months[months.duplicated(keep=False)]
    report.kv("first / last month", f"{months.min()} / {months.max()}")
    report.kv("distinct months", int(months.nunique()))
    report.kv("expected months (2019-01..2028-05)", len(expected))
    report.kv("duplicate months", int(dup_months.nunique()))
    report.kv("missing months", [str(m) for m in expected if m not in set(months)][:12])
    if dup_months.nunique():
        report.table(frame.loc[months.duplicated(keep=False), ["Date", "Year", "Month"]].head(8))
        dup_frame = frame.loc[months.duplicated(keep=False)].sort_values(["__ym", "Date"])
        report.save_table("macro_duplicate_months", dup_frame,     report.profile_dir)
        report.finding("MAC-DUP", "high", "macro_indicators has duplicate month rows",
                       int(dup_months.nunique()), None, "macro")

    year_num, _ = to_num(frame["Year"])
    month_num, _ = to_num(frame["Month"])
    report.kv("Year != Date.year", int((year_num != frame["__date"].dt.year).sum()))
    report.kv("Month != Date.month", int((month_num != frame["__date"].dt.month).sum()))

    indicators = [c for c in frame.columns if c not in {"Date", "Year", "Month", "__date", "__ym"}]
    report.sub("Indicator quality")
    rows = []
    audit = {}
    for col in indicators:
        values, s = to_num(frame[col])
        blanks = int(blank_mask(frame[col]).sum())
        bad = int((~blank_mask(frame[col]) & values.isna()).sum())
        clean = values.dropna()
        med = clean.median()
        mad = (clean - med).abs().median()
        if len(clean) == 0:
            robust_z = pd.Series(dtype="float64")
        elif np.isfinite(mad) and mad > 0:
            robust_z = 0.6745 * (clean - med) / mad
        else:
            robust_z = pd.Series(0.0, index=clean.index)
        outliers = int((robust_z.abs() > 3.5).sum())
        drift = float((clean.max() - clean.min()) / abs(med)) if med not in (0, None) and med == med else None
        rows.append({
            "indicator": col,
            "blank": blanks,
            "non_numeric": bad,
            "min": round(float(clean.min()), 4) if len(clean) else None,
            "median": round(float(med), 4) if len(clean) else None,
            "max": round(float(clean.max()), 4) if len(clean) else None,
            "#MAD>3.5": outliers,
            "range/|median|": round(drift, 2) if drift is not None else None,
        })
        audit[col] = {
            "blank": blanks, "non_numeric": bad, "n_unique": int(clean.nunique()),
            "min": float(clean.min()) if len(clean) else None,
            "median": float(med) if len(clean) else None,
            "max": float(clean.max()) if len(clean) else None,
            "mad_outliers": outliers,
            "apparent_unit": ("fraction (needs x100 for %)" if col.endswith("_Pct") and
                              clean.abs().max() <= 1.5 else "percent points"),
        }
        if outliers:
            # robust_z is computed on the non-null subset, so re-align it to frame.index
            z_full = pd.Series(0.0, index=frame.index, dtype="float64")
            z_full.loc[clean.index] = robust_z.to_numpy()
            extreme = frame.loc[z_full.abs() > 3.5, ["Date", col]]
            report.save_table(f"macro_outliers_{col}", extreme,     report.profile_dir)
            report.finding(f"MAC-OUT-{col}", "medium",
                           f"{outliers} MAD-based outliers in {col}", outliers,
                           extreme.head(6).to_dict("records"), "macro")
        if blanks or bad:
            report.finding(f"MAC-NA-{col}", "high",
                           f"{col} has {blanks} blanks and {bad} non-numeric values",
                           blanks + bad, None, "macro")
    report.table(pd.DataFrame(rows), max_rows=len(rows))
    report.metrics["macro"] = audit

    report.sub("Scale / unit plausibility")
    for col in indicators:
        values, _ = to_num(frame[col])
        clean = values.dropna()
        if not len(clean):
            continue
        if col.endswith("_Pct") and clean.abs().max() <= 1.5:
            report.kv(f"{col}: stored as fraction", f"range [{clean.min():.4f}, {clean.max():.4f}] -> multiply by 100")
        if clean.eq(clean.iloc[0]).all():
            report.kv(f"{col}: constant", f"always {clean.iloc[0]}")
            report.finding(f"MAC-CONST-{col}", "medium", f"{col} never changes", None, None, "macro")

    report.sub("Forecast-window coverage")
    forecast = frame[frame["__ym"] >= pd.Period("2026-11", "M")]
    report.kv("rows in 2026-11..2028-05", len(forecast))
    report.kv("months covered", f"{forecast['__ym'].nunique()} of 19")
    report.metrics["macro.forecast_rows"] = int(len(forecast))

    # simple lag scan: does any indicator correlate with total monthly units?
    report.sub("Quick lag screen vs total monthly units (diagnostic only)")
    train = raw[TRAIN_FILE].copy()
    train["__ym"] = parse_dates(train["Date"]).dt.to_period("M")
    units, _ = to_num(train["Units_Sold"])
    train["__units"] = units
    monthly = train.groupby("__ym")["__units"].sum()
    series = frame.dropna(subset=["__ym"]).drop_duplicates("__ym").set_index("__ym")
    results = []
    for col in indicators:
        values, _ = to_num(series[col])
        for lag in range(0, 13):
            corr = monthly.corr(values.shift(lag))
            if pd.notna(corr):
                results.append({"indicator": col, "lag_months": -lag, "corr": round(float(corr), 3)})
    if results:
        corr_frame = pd.DataFrame(results)
        best = (corr_frame.assign(abs_corr=corr_frame["corr"].abs())
                .sort_values("abs_corr", ascending=False).groupby("indicator").head(1)
                .sort_values("abs_corr", ascending=False).drop(columns="abs_corr"))
        report.table(best, max_rows=8)
        report.save_table("macro_lag_screen", corr_frame,     report.profile_dir)
        report.metrics["macro.lag_screen"] = jsonable(best.to_dict("records"))
        report.note("Negative lag = indicator leads demand by that many months. Small-sample "
                    "correlations are shown only to prioritise Phase 3 lag-stability work.")


# --------------------------------------------------------------------------- #
# Section 8: events calendar
# --------------------------------------------------------------------------- #
def section_events(report: Report, raw: dict[str, pd.DataFrame], train: dict,
                   test: pd.DataFrame) -> None:
    report.h("10. EVENTS CALENDAR AUDIT")
    frame = raw[EVENTS_FILE].copy()
    report.kv("rows", len(frame))
    start = parse_dates(frame["Start_Date"])
    end = parse_dates(frame["End_Date"])
    report.kv("Start_Date unparseable", int((~blank_mask(frame["Start_Date"]) & start.isna()).sum()))
    report.kv("End_Date unparseable", int((~blank_mask(frame["End_Date"]) & end.isna()).sum()))
    inverted = start.notna() & end.notna() & (end < start)
    report.kv("End_Date before Start_Date", int(inverted.sum()))
    report.kv("duplicate Event_ID", int(frame["Event_ID"].duplicated().sum()))
    report.kv("date range", f"{start.min()} .. {end.max()}")
    report.sub("Event types")
    report.table(frame["Event_Type"].value_counts().rename_axis("event_type").reset_index(name="n"))
    report.sub("Scope types")
    report.table(frame["Scope_Type"].value_counts().rename_axis("scope_type").reset_index(name="n"))

    report.sub("Scope_Value resolution")
    train_frame = train["frame"]
    universe = set()
    universe |= set(train_frame["Region"].dropna().unique())
    universe |= set(train_frame["Product_Family"].dropna().unique())
    universe |= set(train_frame["SKU_ID"].dropna().unique())
    universe |= set(test.get("Region", pd.Series(dtype=str)).dropna().unique())
    universe |= set(test.get("Product_Family", pd.Series(dtype=str)).dropna().unique())
    universe |= set(test.get("SKU_ID", pd.Series(dtype=str)).dropna().unique())
    universe |= {"All", "Global", "ALL", "all"}
    unresolved = frame[~frame["Scope_Value"].isin(universe)]
    report.kv("Scope_Value not matching any region/family/SKU/All", int(len(unresolved)))
    if len(unresolved):
        report.table(unresolved[["Event_ID", "Scope_Type", "Scope_Value", "Event_Type"]].head(10))
        report.save_table("events_unresolved_scope", unresolved,     report.profile_dir)
        report.finding("EVT-SCOPE", "medium", "Event Scope_Value does not resolve to a known dimension value",
                       len(unresolved), unresolved.head(8).to_dict("records"), "events")

    report.sub("Temporal coverage")
    history_end = train["months"][-1] if train["months"] else pd.Period("2026-10", "M")
    in_history = start <= history_end.to_timestamp()
    in_window = start > history_end.to_timestamp()
    report.kv("events starting in history (<= 2026-10)", int(in_history.sum()))
    report.kv("events starting in forecast window", int(in_window.sum()))
    report.kv("events with no End_Date (blank)", int(blank_mask(frame["End_Date"]).sum()))
    span_days = (end - start).dt.days
    report.kv("duration days: median / max", f"{span_days.median():.0f} / {span_days.max():.0f}")
    report.table(frame.loc[in_window, ["Event_ID", "Start_Date", "End_Date", "Scope_Type",
                                       "Scope_Value", "Event_Type"]].head(10))
    report.metrics["events"] = {
        "n": int(len(frame)),
        "in_history": int(in_history.sum()),
        "in_forecast": int(in_window.sum()),
        "types": {k: int(v) for k, v in frame["Event_Type"].value_counts().items()},
        "scopes": {k: int(v) for k, v in frame["Scope_Type"].value_counts().items()},
    }
    if inverted.any():
        report.finding("EVT-INVERTED", "high", "Event End_Date precedes Start_Date",
                       int(inverted.sum()), None, "events")

    # do event types look one-off or recurring?
    report.sub("Recurrence pattern by type")
    tmp = frame.assign(start=start, year=start.dt.year)
    recurrence = (tmp.groupby(["Event_Type", "year"]).size().rename("n").reset_index()
                  .pivot(index="Event_Type", columns="year", values="n").fillna(0).astype(int))
    report.table(recurrence.reset_index(), max_rows=12)


# --------------------------------------------------------------------------- #
# Section 9: stockouts
# --------------------------------------------------------------------------- #
def section_stockouts(report: Report, raw: dict[str, pd.DataFrame], train: dict) -> None:
    report.h("11. STOCKOUT LOG AUDIT")
    frame = raw[STOCKOUT_FILE].copy()
    report.kv("rows", len(frame))
    frame["__ym"] = parse_dates(frame["Date"]).dt.to_period("M")
    report.kv("months covered", f"{frame['__ym'].min()} .. {frame['__ym'].max()}")

    report.sub("Grain")
    for keys in (["Date", "SKU_ID", "Region"], ["Date", "SKU_ID", "Region", "Reason"]):
        audit = grain_audit(frame, keys, "+".join(keys))
        if audit:
            report.kv(audit["grain"], f"{audit['extra_rows']:,} extra rows in "
                                      f"{audit['dup_groups']:,} groups; conflicts={audit['conflicts']}")
            report.metrics[f"stockout.grain.{audit['grain']}"] = audit

    numeric_cols = ["Requested_Units", "Carried_In_Units", "Shipped_Units", "Backordered_Units",
                    "Lost_Units", "Fill_Rate_Pct", "Stockout_Days"]
    report.sub("Value ranges")
    rows = []
    cols = {}
    for col in numeric_cols:
        values, _ = to_num(frame[col])
        cols[col] = values
        rows.append({
            "column": col,
            "blank": int(blank_mask(frame[col]).sum()),
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
            "negative": int((values < 0).sum()),
        })
    report.table(pd.DataFrame(rows), max_rows=len(rows))
    report.metrics["stockout.ranges"] = rows

    report.sub("Algebraic consistency")
    requested, shipped = cols["Requested_Units"], cols["Shipped_Units"]
    lost, back, carried = cols["Lost_Units"], cols["Backordered_Units"], cols["Carried_In_Units"]
    fill = cols["Fill_Rate_Pct"]
    candidates = {
        "Shipped + Lost + Backordered": shipped + lost + back,
        "Shipped + Backordered": shipped + back,
        "Shipped": shipped,
        "Lost + Backordered": lost + back,
    }
    for label, total in candidates.items():
        diff = (requested - total).abs()
        report.kv(f"Requested == {label}", f"{int((diff <= 0.5).sum()):,} / {len(frame):,} "
                                           f"({(diff <= 0.5).mean():.1%})")
    report.kv("Requested > Shipped (lost units exist)", int((requested > shipped).sum()))
    report.kv("Shipped > Requested", int((shipped > requested).sum()))
    report.kv("Lost > Requested", int((lost > requested).sum()))
    report.kv("Stockout_Days > 31", int((cols["Stockout_Days"] > 31).sum()))
    report.kv("Carried_In_Units == 0", f"{int((carried == 0).sum()):,} ({(carried == 0).mean():.1%})")

    implied = 100 * shipped / requested.replace(0, np.nan)
    gap = (fill - implied).abs()
    report.kv("Fill_Rate_Pct vs 100*Shipped/Requested",
              f"{int((gap <= 0.5).sum()):,} rows within 0.5pp; "
              f"median |gap| {gap.median():.2f}pp; max {gap.max():.2f}pp")
    report.metrics["stockout.algebra"] = {
        "identity_match": {k: float((requested - v).abs().le(0.5).mean()) for k, v in candidates.items()},
        "fill_rate_within_0_5pp": float((gap <= 0.5).mean()),
        "fill_rate_max_gap": float(gap.max()),
    }
    if (gap > 0.5).mean() > 0.01:
        report.finding("SO-FILL", "high",
                       "Fill_Rate_Pct does not equal 100*Shipped/Requested",
                       int((gap > 0.5).sum()), None, "stockouts")

    report.sub("Reasons")
    report.table(frame["Reason"].value_counts().rename_axis("reason").reset_index(name="n"))

    report.sub("Join coverage against train")
    train_frame = train["frame"]
    train_keys = train_frame[["SKU_ID", "Region", "__ym"]].drop_duplicates()
    stock_keys = frame[["SKU_ID", "Region", "__ym"]].drop_duplicates()
    matched = stock_keys.merge(train_keys, on=["SKU_ID", "Region", "__ym"], how="inner")
    report.kv("stockout keys (SKU-Region-month)", len(stock_keys))
    report.kv("keys with a train row", len(matched))
    report.kv("keys with no train row", len(stock_keys) - len(matched))
    report.kv("share of train cells flagged as stockout",
              f"{len(matched) / len(train_keys):.2%}" if len(train_keys) else "n/a")
    report.metrics["stockout.coverage"] = {
        "keys": int(len(stock_keys)),
        "matched": int(len(matched)),
        "unmatched": int(len(stock_keys) - len(matched)),
        "share_of_train_cells": float(len(matched) / len(train_keys)) if len(train_keys) else None,
    }

    # compare logged shipments with train units
    train_month = (train_frame.groupby(["SKU_ID", "Region", "__ym"])["__units"]
                   .sum().rename("train_units").reset_index())
    compare = (frame.groupby(["SKU_ID", "Region", "__ym"], as_index=False)
               .agg(requested=("Requested_Units", lambda s: to_num(s)[0].sum()),
                    shipped=("Shipped_Units", lambda s: to_num(s)[0].sum()),
                    lost=("Lost_Units", lambda s: to_num(s)[0].sum())))
    compare = compare.merge(train_month, on=["SKU_ID", "Region", "__ym"], how="left")
    compare = compare.dropna(subset=["train_units"])
    if len(compare):
        report.sub("Stockout log vs booked units")
        report.kv("rows compared", len(compare))
        report.kv("train units == shipped", f"{int((compare['train_units'] == compare['shipped']).sum()):,}")
        report.kv("train units > shipped", int((compare["train_units"] > compare["shipped"]).sum()))
        report.kv("train units < shipped", int((compare["train_units"] < compare["shipped"]).sum()))
        report.kv("train units == requested",
                  f"{int((compare['train_units'] == compare['requested']).sum()):,}")
        report.kv("train units < requested (suppressed demand)",
                  int((compare["train_units"] < compare["requested"]).sum()))
        report.kv("total requested vs booked vs shipped",
                  f"{compare['requested'].sum():,.0f} / {compare['train_units'].sum():,.0f} / "
                  f"{compare['shipped'].sum():,.0f}")
        report.kv("unserved units implied by log", f"{compare['lost'].sum():,.0f}")
        report.table(compare.sort_values("lost", ascending=False).head(8))
        report.save_table("stockout_vs_train", compare,     report.profile_dir)
        report.metrics["stockout.vs_train"] = {
            "rows": int(len(compare)),
            "units_equal_shipped": int((compare["train_units"] == compare["shipped"]).sum()),
            "units_below_requested": int((compare["train_units"] < compare["requested"]).sum()),
            "requested_sum": float(compare["requested"].sum()),
            "booked_sum": float(compare["train_units"].sum()),
            "shipped_sum": float(compare["shipped"].sum()),
            "lost_sum": float(compare["lost"].sum()),
        }
        report.finding("SO-VS-TRAIN", "info",
                       "Booked Units_Sold is supply-constrained in stockout months; unconstrained demand "
                       "must be reconstructed (requested/shipped/lost)",
                       int(len(compare)), {"below_requested": int((compare["train_units"] < compare["requested"]).sum())},
                       "stockouts")


# --------------------------------------------------------------------------- #
# Section 10: test + submission
# --------------------------------------------------------------------------- #
def section_test(report: Report, raw: dict[str, pd.DataFrame], train: dict,
                 master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    report.h("6. TEST.CSV (FORECAST WINDOW) INTEGRITY")
    frame = raw[TEST_FILE].copy()
    report.kv("rows x cols", f"{len(frame):,} x {frame.shape[1]}")
    frame["__dt"] = parse_dates(frame["Date"])
    frame["__ym"] = frame["__dt"].dt.to_period("M")
    months = frame["__ym"].dropna()
    expected = pd.period_range("2026-11", "2028-05", freq="M")
    report.kv("first / last month", f"{months.min()} / {months.max()}")
    report.kv("distinct months (expected)", f"{months.nunique()} ({len(expected)})")
    report.kv("months missing from window", [str(m) for m in expected if m not in set(months)] or "none")
    report.kv("rows per month (min / max)", f"{frame.groupby('__ym').size().min():,} / "
                                            f"{frame.groupby('__ym').size().max():,}")
    report.kv("blank cells (any column)",
              int((frame.drop(columns=["__dt", "__ym"]).astype(str).apply(lambda c: blank_mask(c)).sum().sum())))

    report.sub("Year / Month vs Date")
    year_num, _ = to_num(frame["Year"])
    month_num, _ = to_num(frame["Month"])
    report.kv("Year mismatch", int((year_num != frame["__dt"].dt.year).sum()))
    report.kv("Month mismatch", int((month_num != frame["__dt"].dt.month).sum()))

    report.sub("Duplicate grains")
    audits = [grain_audit(frame, GRAIN, "Date+SKU+Region"),
              grain_audit(frame, GRAIN_CUSTOMER, "Date+SKU+Region+Customer"),
              grain_audit(frame, GRAIN_CHANNEL, "Date+SKU+Region+Channel")]
    audits = [a for a in audits if a]
    report.table(pd.DataFrame([{
        "grain": a["grain"], "rows_in_dup_groups": a["rows_in_dup_groups"],
        "dup_groups": a["dup_groups"], "extra_rows": a["extra_rows"],
    } for a in audits]))
    natural = next((a for a in audits if a["extra_rows"] == 0), None)
    report.kv("unique key", natural["grain"] if natural else "none of the tested grains")
    report.metrics["test.grains"] = audits
    if any(a["extra_rows"] for a in audits):
        report.finding("TST-DUP", "high", "Duplicate keys in test.csv",
                       sum(a["extra_rows"] for a in audits), audits, "test")

    train_frame = train["frame"]
    report.sub("Key coverage vs history")
    def unseen(col):
        hist = set(train_frame[col].dropna().unique())
        cur = set(frame[col].dropna().unique())
        return sorted(cur - hist), sorted(hist - cur)

    new_skus, gone_skus = unseen("SKU_ID")
    new_regions, gone_regions = unseen("Region")
    new_customers, gone_customers = unseen("Customer_Name")
    new_channels, gone_channels = unseen("Sales_Channel")
    new_families, gone_families = unseen("Product_Family")
    for label, new, gone in [("SKU_ID", new_skus, gone_skus), ("Region", new_regions, gone_regions),
                             ("Customer_Name", new_customers, gone_customers),
                             ("Sales_Channel", new_channels, gone_channels),
                             ("Product_Family", new_families, gone_families)]:
        report.kv(f"{label}: new in test / absent from test",
                  f"{len(new):,} / {len(gone):,}")
        if len(new) and len(new) <= 12:
            report.kv("  new values", new)

    train_pairs = set(map(tuple, train_frame[["SKU_ID", "Region"]].drop_duplicates().to_numpy()))
    test_pairs = frame[["SKU_ID", "Region"]].drop_duplicates()
    test_pairs["seen_in_train"] = [tuple(row) in train_pairs for row in test_pairs.to_numpy()]
    unseen_pairs = test_pairs[~test_pairs["seen_in_train"]]
    unseen_rows = frame.merge(unseen_pairs[["SKU_ID", "Region"]], on=["SKU_ID", "Region"], how="inner")
    report.kv("SKU-Region pairs in test", len(test_pairs))
    report.kv("pairs never seen in train", len(unseen_pairs))
    report.kv("test rows on unseen pairs", f"{len(unseen_rows):,} ({len(unseen_rows) / len(frame):.2%})")
    report.metrics["test.coverage"] = {
        "new_skus": new_skus, "gone_skus": gone_skus,
        "new_regions": new_regions, "gone_regions": gone_regions,
        "new_customers": new_customers, "gone_customers": gone_customers,
        "new_channels": new_channels, "gone_channels": gone_channels,
        "unseen_pairs": int(len(unseen_pairs)),
        "unseen_pair_rows": int(len(unseen_rows)),
    }
    if len(unseen_rows):
        report.table(unseen_pairs.head(8))
        report.save_table("test_unseen_pairs", unseen_pairs, out_dir)
        report.finding("TST-UNSEEN-PAIRS", "high",
                       "Test rows for SKU-Region combinations with no history (cold start)",
                       len(unseen_rows), unseen_pairs.head(8).to_dict("records"), "test")

    report.sub("Lifecycle rules inside the forecast window")
    if not master.empty:
        launch = dict(zip(master["SKU_ID"], parse_dates(master["Launch_Date"])))
        eol = dict(zip(master["SKU_ID"], parse_dates(master["EOL_Date"])))
        row_launch = frame["SKU_ID"].map(launch)
        row_eol = frame["SKU_ID"].map(eol)
        before_launch = frame["__dt"].notna() & row_launch.notna() & (frame["__dt"] < row_launch)
        after_eol = frame["__dt"].notna() & row_eol.notna() & (frame["__dt"] > row_eol)
        report.kv("test rows before Launch_Date", int(before_launch.sum()))
        report.kv("test rows after EOL_Date", int(after_eol.sum()))
        report.kv("SKUs in test whose EOL is before window", int((row_eol < pd.Timestamp("2026-11-01")).sum()))
        report.metrics["test.lifecycle"] = {
            "before_launch": int(before_launch.sum()),
            "after_eol": int(after_eol.sum()),
        }
        if before_launch.any() or after_eol.any():
            report.finding("TST-LIFECYCLE", "high",
                           "Test keys violate Launch_Date/EOL_Date rules",
                           int(before_launch.sum() + after_eol.sum()), None, "test")

    report.sub("Month-over-month key stability")
    sizes = frame.groupby("__ym").size()
    report.kv("test rows per month (median)", int(sizes.median()))
    report.kv("months where key count deviates from median",
              int((sizes != sizes.median()).sum()))
    report.metrics["test.rows_per_month"] = {str(k): int(v) for k, v in sizes.items()}

    # submission
    report.h("7. SAMPLE SUBMISSION FORMAT")
    sub = raw[SUB_FILE]
    report.kv("rows x cols", f"{len(sub):,} x {sub.shape[1]}")
    report.kv("columns", list(sub.columns))
    report.table(sub.head())
    expected_cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    report.verdict(0, "submission columns match expected",
                   list(sub.columns) == expected_cols, f"got {list(sub.columns)}")
    sub_keys = set(map(tuple, sub[["Date", "SKU_ID", "Region"]].to_numpy()))
    report.kv("submission keys present in test",
              f"{len(sub_keys & set(map(tuple, frame[['Date', 'SKU_ID', 'Region']].to_numpy()))):,}"
              f" / {len(sub_keys):,}")
    report.kv("expected submission rows (= test rows)", f"{len(frame):,}")
    report.metrics["submission"] = {
        "rows": len(sub), "columns": list(sub.columns),
        "keys_in_test": int(len(sub_keys & set(map(tuple, frame[['Date', 'SKU_ID', 'Region']].to_numpy())))),
    }

    report.sub("Test rows per SKU-Region-month structure")
    report.kv("distinct SKUs / Regions / Customers / Channels",
              f"{frame['SKU_ID'].nunique()} / {frame['Region'].nunique()} / "
              f"{frame['Customer_Name'].nunique()} / {frame['Sales_Channel'].nunique()}")
    report.kv("rows per SKU-Region pair (median)",
              float(frame.groupby(['SKU_ID', 'Region']).size().median()))
    return frame


# --------------------------------------------------------------------------- #
# Section 12: candidate cleaning backlog
# --------------------------------------------------------------------------- #
def section_backlog(report: Report, data_dir: Path) -> None:
    report.h("12. EVIDENCE-BASED CLEANING BACKLOG (for Phase 1)")
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    findings = sorted(report.findings, key=lambda f: (order.get(f["severity"], 9), f["id"]))
    if not findings:
        report.p("  no defects detected")
        return
    rows = [{
        "severity": f["severity"],
        "id": f["id"],
        "issue": f["issue"],
        "count": f["count"],
    } for f in findings]
    report.table(pd.DataFrame(rows), max_rows=60)
    by_sev = pd.Series([f["severity"] for f in findings]).value_counts()
    report.kv("findings by severity", ", ".join(f"{k}: {v}" for k, v in by_sev.items()))

    counts = pd.DataFrame(rows)
    counts.to_csv(    report.profile_dir / "cleaning_backlog.csv", index=False)
    report.metrics["backlog"] = {"n_findings": len(findings),
                                 "by_severity": {k: int(v) for k, v in by_sev.items()}}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 forensic profiler")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)

    started = time.time()
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir)
    profile_dir = out_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    report = Report()
    report.profile_dir = profile_dir
    report.p("CELERIS AUTOMOBILI - DEMAND FORECASTING CONTROL TOWER")
    report.p("Phase 0: forensic profiling (read-only; no cleaning applied)")
    report.p(f"data dir : {data_dir}")
    report.p(f"out dir  : {out_dir}")

    missing = [name for name in ALL_FILES if not (data_dir / name).exists()]
    if missing:
        print(f"FATAL: missing input files: {missing}", file=sys.stderr)
        return 2
    raw = {name: read_raw(data_dir / name) for name in ALL_FILES}
    report.metrics["_data_dir"] = str(data_dir)

    section_files(report, data_dir, raw)
    section_schema(report, raw)
    section_labels(report, raw, profile_dir)
    section_dates(report, raw, profile_dir)

    master = raw[MASTER_FILE].copy()
    master["Launch_Date__parsed"] = parse_dates(master["Launch_Date"])
    master["EOL_Date__parsed"] = parse_dates(master["EOL_Date"])

    stockouts = raw[STOCKOUT_FILE].copy()
    stockouts["__ym"] = parse_dates(stockouts["Date"]).dt.to_period("M")

    train = section_train(report, raw, master, stockouts, profile_dir)
    test = section_test(report, raw, train, master, profile_dir)
    section_master(report, raw, train, test)
    section_macro(report, raw)
    section_events(report, raw, train, test)
    section_stockouts(report, raw, train)
    section_backlog(report, data_dir)

    report.h("13. PHASE 0 EXIT CHECK")
    report.kv("elapsed seconds", round(time.time() - started, 1))
    report.kv("findings logged", len(report.findings))
    json_path, txt_path = report.write(out_dir)

    # console: everything except the largest schema block
    print("\n".join(report.lines))
    print(f"\nJSON  -> {json_path}")
    print(f"TEXT  -> {txt_path}")
    print(f"TABLES-> {profile_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
