"""Phase 0 contract tests: the profiler must run, write artifacts, and never mutate inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import profile_data  # noqa: E402

MONTHS = ["2026-08", "2026-09", "2026-10"]
REGIONS = ["North America", "Asia Pacific"]
SKUS = ["BSS-001", "HYD-018"]


def _train_rows() -> list[dict]:
    rows = []
    for month in MONTHS:
        year, mon = month.split("-")
        for sku in SKUS:
            for region in REGIONS:
                rows.append({
                    "Record_ID": f"CEL-{len(rows):07d}",
                    "Date": month,
                    "Year": year,
                    "Month": int(mon),
                    "Product_Family": "Brake & Safety Systems" if sku.startswith("BSS") else "Hydraulic Systems",
                    "Business_Unit": "Safety & Compliance",
                    "SKU_ID": sku,
                    "SKU_Description": f"Part {sku}",
                    "Customer_Name": "Solaris Automotive",
                    "Sales_Channel": "Tier-1 Distributor",
                    "Region": region,
                    "Units_Sold": 10 + len(rows),
                    "Unit_Price_USD": 100.0,
                    "Revenue_USD": (10 + len(rows)) * 100.0,
                    "Load_Timestamp": f"{year}-{mon}-01 01:00:00",
                })
    return rows


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(_train_rows())
    # one double load, newer timestamp, identical payload
    duplicate = train.iloc[[0]].copy()
    duplicate["Load_Timestamp"] = "2026-11-05 09:00:00"
    train = pd.concat([train, duplicate], ignore_index=True)
    train.to_csv(directory / "train.csv", index=False)

    test = (train.drop_duplicates(subset=["Date", "SKU_ID", "Region"])
            .drop(columns=["Record_ID", "Units_Sold", "Unit_Price_USD", "Revenue_USD", "Load_Timestamp"]))
    test = test.assign(Date="2026-11", Year="2026", Month=11)
    test.to_csv(directory / "test.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-08", "2026-09", "2026-10", "2026-11"],
        "Year": [2026] * 4,
        "Month": [8, 9, 10, 11],
        "ISM_Manufacturing_PMI": [51.0, 52.0, 51.5, 52.2],
        "EV_Adoption_Index": [40.0, 40.5, 41.0, 41.5],
    }).to_csv(directory / "macro_indicators.csv", index=False)

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
        "End_Date": ["2026-11"],
        "Scope_Type": ["Global"],
        "Scope_Value": ["All"],
        "Event_Type": ["Price Change"],
        "Description": ["Annual list-price increase"],
    }).to_csv(directory / "events_calendar.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-09"],
        "SKU_ID": ["BSS-001"],
        "Region": ["North America"],
        "Requested_Units": [60],
        "Carried_In_Units": [40],
        "Shipped_Units": [40],
        "Backordered_Units": [12],
        "Lost_Units": [8],
        "Fill_Rate_Pct": [66.7],
        "Stockout_Days": [9],
        "Reason": ["Logistics delay"],
    }).to_csv(directory / "inventory_stockouts.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-11"],
        "SKU_ID": ["BSS-001"],
        "Region": ["North America"],
        "Units_Sold_Forecast": [50],
    }).to_csv(directory / "sample_submission.csv", index=False)


def file_digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.csv"))}


def test_profiler_writes_artifacts_and_is_read_only(tmp_path: Path):
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    before = file_digests(data_dir)

    exit_code = profile_data.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)])
    assert exit_code == 0

    json_path = out_dir / "profile_report.json"
    txt_path = out_dir / "profile_report.txt"
    assert json_path.exists() and txt_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) >= {"meta", "findings", "verdicts", "metrics"}
    assert payload["meta"]["phase"] == 0
    assert len(payload["metrics"]["files"]) == 7
    assert payload["findings"], "the planted double load should be detected"
    assert isinstance(payload["verdicts"], list) and payload["verdicts"]

    # read-only guarantee
    assert file_digests(data_dir) == before


def test_profiler_detects_planted_defects(tmp_path: Path):
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    profile_data.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)])
    payload = json.loads((out_dir / "profile_report.json").read_text(encoding="utf-8"))
    ids = {finding["id"] for finding in payload["findings"]}
    assert "TRN-DUP-ID" in ids, "duplicate Record_ID must be reported"
    assert "TRN-GRAIN" in ids, "fact-grain audit must be reported"
    assert (out_dir / "profile").is_dir()
