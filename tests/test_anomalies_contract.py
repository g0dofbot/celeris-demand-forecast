"""Phase 2 contract tests: data errors and genuine demand events must stay apart."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anomalies, clean  # noqa: E402

MONTHS = pd.date_range("2024-01-01", "2026-10-01", freq="MS")
SKUS = ["BSS-001", "HYD-018"]
REGIONS = ["North America", "Asia Pacific", "Europe"]
BASE_UNITS = 100

# (month, sku, region) -> deliberate deviation from the flat baseline.
SPIKE = ("2026-06", "BSS-001", "North America")        # 9x, a real demand event
DIGIT = ("2026-07", "HYD-018", "Asia Pacific")         # exactly 10x both neighbours
CENSORED = ("2026-08", "BSS-001", "North America")     # shipments short of demand
COLLAPSE = ("2026-09", "HYD-018", "North America")     # established run stops
DROP = ("2026-05", "BSS-001", "Europe")                # a large but genuine decline


def _row(record: str, month: pd.Timestamp, sku: str, region: str, units: int) -> dict:
    return {
        "Record_ID": record,
        "Date": month.strftime("%Y-%m"),
        "Year": month.year,
        "Month": month.month,
        "Product_Family": ("Brake & Safety Systems" if sku.startswith("BSS")
                           else "Hydraulic Systems"),
        "Business_Unit": "Safety & Compliance",
        "SKU_ID": sku,
        "SKU_Description": f"Part {sku}",
        "Customer_Name": "Solaris Automotive",
        "Sales_Channel": "Tier-1 Distributor",
        "Region": region,
        "Units_Sold": units,
        "Unit_Price_USD": 100.0,
        "Revenue_USD": units * 100.0,
        "Load_Timestamp": f"{month:%Y-%m}-01 01:00:00",
    }


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    overrides = {SPIKE: 900, DIGIT: 1000, CENSORED: 150, COLLAPSE: 0, DROP: 10}
    rows, index = [], 0
    for month in MONTHS:
        for sku in SKUS:
            for region in REGIONS:
                index += 1
                key = (month.strftime("%Y-%m"), sku, region)
                units = overrides.get(key, BASE_UNITS)
                rows.append(_row(f"CEL-{index:05d}", month, sku, region, units))
    pd.DataFrame(rows).to_csv(directory / "train.csv", index=False)

    test_rows = []
    for month in pd.date_range("2026-11-01", "2026-12-01", freq="MS"):
        for sku in SKUS:
            for region in REGIONS:
                test_rows.append({
                    "Date": month.strftime("%Y-%m"), "Year": month.year, "Month": month.month,
                    "Product_Family": ("Brake & Safety Systems" if sku.startswith("BSS")
                                       else "Hydraulic Systems"),
                    "Business_Unit": "Safety & Compliance", "SKU_ID": sku,
                    "SKU_Description": f"Part {sku}", "Customer_Name": "Solaris Automotive",
                    "Sales_Channel": "Tier-1 Distributor", "Region": region,
                })
    pd.DataFrame(test_rows).to_csv(directory / "test.csv", index=False)

    macro = pd.DataFrame({
        "Date": [m.strftime("%Y-%m") for m in MONTHS],
        "Year": [m.year for m in MONTHS],
        "Month": [m.month for m in MONTHS],
        "ISM_Manufacturing_PMI": [50 + i % 4 for i in range(len(MONTHS))],
        "Auto_Production_YoY_Pct": [1.5] * len(MONTHS),
        "Vehicle_Inventory_Days": [60.0] * len(MONTHS),
        "Raw_Material_Cost_Index": [100 + i % 3 for i in range(len(MONTHS))],
        "New_Vehicle_Orders_Index": [100.0] * len(MONTHS),
        "Industrial_Util_Rate_Pct": [79.0] * len(MONTHS),
        "Fuel_Price_Index": [90.0] * len(MONTHS),
        "EV_Adoption_Index": [40.0] * len(MONTHS),
    })
    macro.to_csv(directory / "macro_indicators.csv", index=False)

    pd.DataFrame({
        "SKU_ID": SKUS,
        "Product_Family": ["Brake & Safety Systems", "Hydraulic Systems"],
        "Business_Unit": ["Safety & Compliance", "Fluid Control Group"],
        "SKU_Description": [f"Part {s}" for s in SKUS],
        "Size_Class": ["Industrial 100", "Heavy-Duty 70"],
        "Powertrain_Application": ["Universal", "ICE"],
        "Launch_Date": ["2018-05", "2018-05"],
        "EOL_Date": ["", ""],
        "Replaces_SKU": ["", ""],
        "Lifecycle_Status": ["Active", "Active"],
    }).to_csv(directory / "sku_master.csv", index=False)

    pd.DataFrame({
        "Event_ID": ["EV-001"],
        "Start_Date": ["2026-11"],
        "End_Date": [""],
        "Scope_Type": ["Global"],
        "Scope_Value": ["All"],
        "Event_Type": ["Price Change"],
        "Description": ["Annual list-price increase"],
    }).to_csv(directory / "events_calendar.csv", index=False)

    pd.DataFrame({
        "Date": [CENSORED[0]],
        "SKU_ID": [CENSORED[1]],
        "Region": [CENSORED[2]],
        "Requested_Units": [400],
        "Carried_In_Units": [150],
        "Shipped_Units": [150],
        "Backordered_Units": [50],
        "Lost_Units": [200],
        "Fill_Rate_Pct": [37.5],
        "Stockout_Days": [12],
        "Reason": ["Capacity constraint"],
    }).to_csv(directory / "inventory_stockouts.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-11"], "SKU_ID": ["BSS-001"], "Region": ["North America"],
        "Units_Sold_Forecast": [100],
    }).to_csv(directory / "sample_submission.csv", index=False)


def digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.parquet"))}


def run_phase2(tmp_path: Path) -> tuple[Path, Path, dict, pd.DataFrame]:
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert anomalies.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    payload = json.loads((out_dir / "anomaly_report.json").read_text(encoding="utf-8"))
    register = pd.read_parquet(out_dir / "anomalies" / "anomaly_register.parquet")
    return data_dir, out_dir, payload, register


def cell_row(register: pd.DataFrame, planted: tuple[str, str, str]) -> pd.Series:
    month, sku, region = planted
    match = register.loc[(register["level"] == "cell") & (register["SKU_ID"] == sku)
                         & (register["Region"] == region)
                         & (register["Month"] == pd.Timestamp(month))]
    assert len(match) == 1, f"expected one register row for {planted}, got {len(match)}"
    return match.iloc[0]


def test_anomalies_write_artifacts_and_are_read_only(tmp_path: Path):
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    before = digests(out_dir / "clean")
    assert anomalies.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert digests(out_dir / "clean") == before, "Phase 2 must not rewrite Phase 1 outputs"
    for name in ["anomaly_register", "data_errors", "demand_events", "demand_baseline",
                 "monthly_summary", "test_window_profile"]:
        assert (out_dir / "anomalies" / f"{name}.parquet").exists(), name
    assert (out_dir / "anomalies" / "anomaly_register.csv").exists()
    assert (out_dir / "anomalies" / "anomaly_rules.csv").exists()


def test_data_errors_are_separated_from_demand_events(tmp_path: Path):
    _, out_dir, payload, register = run_phase2(tmp_path)
    panel = pd.read_parquet(out_dir / "clean" / "demand_monthly.parquet")

    # both halves of the distinction exist, and the ids are unique
    assert set(register["kind"]) == {"DATA_ERROR", "DEMAND_EVENT"}
    assert register["anomaly_id"].is_unique
    assert not register.loc[register["level"] == "cell"].duplicated(
        ["SKU_ID", "Region", "Month"]).any()

    # a real 9x month is a demand event, kept for the model
    spike = cell_row(register, SPIKE)
    assert spike["anomaly_class"] == "spike" and spike["kind"] == "DEMAND_EVENT"
    assert spike["impact"] == 800

    # a 10x-against-both-neighbours blip is a data error, not a spike
    digit = cell_row(register, DIGIT)
    assert digit["kind"] == "DATA_ERROR"
    assert digit["anomaly_class"] == "suspected_digit_shift"
    assert digit["status"] == "flagged"
    assert "spike" in digit["secondary_class"]

    # a stockout censors shipments: demand is restored, the cell is not a decline
    censored = cell_row(register, CENSORED)
    assert censored["anomaly_class"] == "censored_by_stockout"
    assert censored["observed"] == 400
    assert censored["impact"] == 250

    # an established run stopping is a collapse, not intermittency
    collapse = cell_row(register, COLLAPSE)
    assert collapse["anomaly_class"] == "demand_collapse"

    # a large genuine decline is a plain drop
    drop = cell_row(register, DROP)
    assert drop["anomaly_class"] == "drop" and drop["kind"] == "DEMAND_EVENT"

    # the panel keeps unconstrained demand and never falls below observed units
    assert (panel["demand_units"] >= panel["observed_units"]).all()
    assert not panel.loc[panel["usable"], ["pre_launch", "post_eol"]].any(axis=1).any()

    # every registered line is auditable
    assert register["evidence"].astype(str).str.strip().ne("").all()
    assert register["treatment"].astype(str).str.strip().ne("").all()
    errors = register.loc[register["kind"] == "DATA_ERROR"]
    assert errors["explained_by"].astype(str).str.strip().ne("").all()

    assert all(v["ok"] for v in payload["verdicts"]), payload["verdicts"]
