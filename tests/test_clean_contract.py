"""Phase 1 contract tests: cleaning must be deterministic, auditable and lossless."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import clean  # noqa: E402

MONTHS = ["2026-08", "2026-09", "2026-10"]
REGIONS = ["North America", "Asia Pacific"]
SKUS = ["BSS-001", "HYD-018"]


def _row(record: str, month: str, sku: str, region: str, units: int,
         price: float, customer: str = "Solaris Automotive",
         channel: str = "Tier-1 Distributor") -> dict:
    year, mon = month.split("-")
    return {
        "Record_ID": record,
        "Date": month,
        "Year": year,
        "Month": int(mon),
        "Product_Family": "Brake & Safety Systems" if sku.startswith("BSS") else "Hydraulic Systems",
        "Business_Unit": "Safety & Compliance",
        "SKU_ID": sku,
        "SKU_Description": f"Part {sku}",
        "Customer_Name": customer,
        "Sales_Channel": channel,
        "Region": region,
        "Units_Sold": units,
        "Unit_Price_USD": price,
        "Revenue_USD": units * price,
        "Load_Timestamp": f"{year}-{mon}-01 01:00:00",
    }


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for month in MONTHS:
        for sku in SKUS:
            for region in REGIONS:
                index += 1
                rows.append(_row(f"CEL-{index:05d}", month, sku, region, 10 + index, 100.0))
    # duplicate Record_ID: newer load, same payload
    duplicate = dict(rows[0], Load_Timestamp="2026-11-05 09:00:00")
    rows.append(duplicate)
    # keying error: year 2062 but Year/Month say 2026-10
    rows.append(_row("CEL-90001", "2062-10", "BSS-001", "North America", 5, 100.0))
    rows[-1]["Year"], rows[-1]["Month"], rows[-1]["Date"] = "2026", "10", "2062-10"
    # zero price with a valid revenue: price must be recovered from Revenue/Units
    zero_price = _row("CEL-90002", "2026-09", "HYD-018", "Asia Pacific", 7, 0.0)
    zero_price["Revenue_USD"] = 700.0
    rows.append(zero_price)
    # negative units and a blank region row: retained but excluded from demand
    rows.append(_row("CEL-90003", "2026-09", "BSS-001", "North America", -3, 100.0))
    blank_region = _row("CEL-90004", "2026-08", "HYD-018", "", 4, 100.0)
    rows.append(blank_region)
    # SKU spelling variant that needs zero-padding
    variant = _row("CEL-90005", "2026-08", "BSS1", "Europe", 6, 100.0)
    rows.append(variant)

    pd.DataFrame(rows).to_csv(directory / "train.csv", index=False)

    test = pd.DataFrame({
        "Date": ["2026-11"] * 4,
        "Year": [2026] * 4,
        "Month": [11] * 4,
        "Product_Family": ["Brake & Safety Systems"] * 4,
        "Business_Unit": ["Safety & Compliance"] * 4,
        "SKU_ID": [SKUS[0], SKUS[0], SKUS[1], SKUS[1]],
        "SKU_Description": ["Part", "Part", "Part", "Part"],
        "Customer_Name": ["Solaris Automotive"] * 4,
        "Sales_Channel": ["Tier-1 Distributor"] * 4,
        "Region": REGIONS + REGIONS,
    })
    test.to_csv(directory / "test.csv", index=False)

    pd.DataFrame({
        "Date": MONTHS,
        "Year": [2026] * 3,
        "Month": [8, 9, 10],
        "ISM_Manufacturing_PMI": [51.0, "", 52.0],
        "Auto_Production_YoY_Pct": [1.0, 2.0, 3.0],
        "Vehicle_Inventory_Days": [60.0, 61.0, 62.0],
        "Raw_Material_Cost_Index": [100.0, 101.0, 102.0],
        "New_Vehicle_Orders_Index": [100.0, 101.0, 102.0],
        "Industrial_Util_Rate_Pct": [79.0, 79.5, 80.0],
        "Fuel_Price_Index": [90.0, 91.0, 92.0],
        "EV_Adoption_Index": [40.0, 41.0, 42.0],
    }).to_csv(directory / "macro_indicators.csv", index=False)

    pd.DataFrame({
        "SKU_ID": SKUS + ["BSS-001"],
        "Product_Family": ["Brake & Safety Systems", "Hydraulic Systems", "Brake & Safety Systems"],
        "Business_Unit": ["Safety & Compliance", "Fluid Control Group", "Safety & Compliance"],
        "SKU_Description": [f"Part {s}" for s in SKUS + ["BSS-001"]],
        "Size_Class": ["Industrial 100", "Heavy-Duty 70", "Industrial 100"],
        "Powertrain_Application": ["Universal", "ICE", "Universal"],
        "Launch_Date": ["2018-05", "2018-05", "2018-05"],
        "EOL_Date": ["", "", ""],
        "Replaces_SKU": ["", "", ""],
        "Lifecycle_Status": ["Active", "Active", "Active"],
    }).drop_duplicates("SKU_ID").to_csv(directory / "sku_master.csv", index=False)

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


def digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.csv"))}


def run_cleaner(tmp_path: Path) -> tuple[Path, Path, dict]:
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    payload = json.loads((out_dir / "clean_report.json").read_text(encoding="utf-8"))
    return data_dir, out_dir, payload


def test_cleaning_writes_artifacts_and_is_read_only(tmp_path: Path):
    data_dir = tmp_path / "student_data"
    out_dir = tmp_path / "outputs"
    write_fixture(data_dir)
    before = digests(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert digests(data_dir) == before, "cleaning must never mutate raw inputs"
    for name in ["sales_clean", "demand_monthly", "demand_grid", "sku_master_clean",
                 "stockouts_clean", "macro_clean", "events_clean", "test_clean"]:
        assert (out_dir / "clean" / f"{name}.parquet").exists(), name
    assert (out_dir / "clean" / "cleaning_log.csv").exists()
    assert (out_dir / "clean" / "reconciliation.csv").exists()
    assert (out_dir / "clean" / "label_map.csv").exists()


def test_planted_defects_are_repaired(tmp_path: Path):
    data_dir, out_dir, payload = run_cleaner(tmp_path)
    sales = pd.read_parquet(out_dir / "clean" / "sales_clean.parquet")
    panel = pd.read_parquet(out_dir / "clean" / "demand_monthly.parquet")
    grid = pd.read_parquet(out_dir / "clean" / "demand_grid.parquet")
    raw_rows = len(pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False))

    # every row is accounted for; only the double load is dropped
    assert len(sales) == raw_rows - 1
    assert sales["Record_ID"].is_unique
    assert payload["metrics"]["train"]["rows_dropped"] == 1

    # 2062 keying error re-dated, never dropped
    repaired = sales.loc[sales["Record_ID"] == "CEL-90001"]
    assert len(repaired) == 1 and repaired["Year"].iloc[0] == 2026
    assert "DATE_REPAIRED" in repaired["flags"].iloc[0]

    # zero price recovered from Revenue / Units
    recovered = sales.loc[sales["Record_ID"] == "CEL-90002", "Price_Clean_USD"].iloc[0]
    assert recovered == 100.0

    # one base price per SKU-year, so no regional price scatter survives
    spread = sales.groupby(["SKU_ID", "Price_Year"])["Price_Clean_USD"].nunique()
    assert spread.max() == 1

    # SKU variant resolves to the master key instead of being dropped
    assert "BSS-001" in set(sales["SKU_ID"])

    # blank region retained in the fact table but absent from the demand panel
    assert sales["Region"].eq("Unknown").any()
    assert not panel["Region"].eq("Unknown").any()

    # negative units are visible but never counted as demand
    negative = sales.loc[sales["Units_Sold"].lt(0)]
    assert len(negative) == 1 and negative["Units_Sold_Clean"].isna().all()

    # missing calendar cells are absent, never zero-filled
    absent = grid.loc[grid["status"] == "absent"]
    assert not absent["demand_units"].notna().any()

    # stockout suppression raises demand above shipped units for that cell
    cell = panel.loc[(panel["SKU_ID"] == "BSS-001")
                     & (panel["Region"] == "North America")
                     & (panel["Month"] == pd.Timestamp("2026-09"))]
    assert cell["suppressed_units"].iloc[0] > 0
    assert cell["demand_units"].iloc[0] == 60

    # the audit trail covers the planted rules
    rules = set(pd.read_csv(out_dir / "clean" / "cleaning_log.csv")["rule_id"])
    assert {"TRN-DATE-2062", "TRN-DUP-ID", "TRN-PRICE", "TRN-UNITS-NEG",
            "TRN-REGION", "TRN-LIFECYCLE"} <= rules

    assert all(v["ok"] for v in payload["verdicts"])
