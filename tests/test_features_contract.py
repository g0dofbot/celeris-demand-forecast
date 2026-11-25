"""Phase 3 contract tests: features must be lagged, complete, and horizon-ready."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anomalies, clean, features  # noqa: E402

TRAIN_MONTHS = pd.date_range("2024-01-01", "2026-06-01", freq="MS")
FORECAST_MONTHS = pd.date_range("2026-07-01", "2026-08-01", freq="MS")
SKUS = ["BSS-001", "BSS-002", "BSS-003", "HYD-018", "HYD-019", "HYD-020"]
REGIONS = ["North America", "Europe"]
BASE_UNITS = 100
PRICE = 100.0


def _family(sku: str) -> str:
    return "Brake & Safety Systems" if sku.startswith("BSS") else "Hydraulic Systems"


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows, index = [], 0
    for month in TRAIN_MONTHS:
        for sku in SKUS:
            for region in REGIONS:
                index += 1
                rows.append({
                    "Record_ID": f"CEL-{index:05d}",
                    "Date": month.strftime("%Y-%m"),
                    "Year": month.year,
                    "Month": month.month,
                    "Product_Family": _family(sku),
                    "Business_Unit": "Safety & Compliance",
                    "SKU_ID": sku,
                    "SKU_Description": f"Part {sku}",
                    "Customer_Name": "Solaris Automotive",
                    "Sales_Channel": "Tier-1 Distributor",
                    "Region": region,
                    "Units_Sold": BASE_UNITS,
                    "Unit_Price_USD": PRICE,
                    "Revenue_USD": BASE_UNITS * PRICE,
                    "Load_Timestamp": f"{month:%Y-%m}-01 01:00:00",
                })
    pd.DataFrame(rows).to_csv(directory / "train.csv", index=False)

    test_rows = []
    for month in FORECAST_MONTHS:
        for sku in SKUS:
            for region in REGIONS:
                test_rows.append({
                    "Date": month.strftime("%Y-%m"), "Year": month.year, "Month": month.month,
                    "Product_Family": _family(sku), "Business_Unit": "Safety & Compliance",
                    "SKU_ID": sku, "SKU_Description": f"Part {sku}",
                    "Customer_Name": "Solaris Automotive",
                    "Sales_Channel": "Tier-1 Distributor", "Region": region,
                })
    pd.DataFrame(test_rows).to_csv(directory / "test.csv", index=False)

    macro_months = list(TRAIN_MONTHS) + list(FORECAST_MONTHS)
    pd.DataFrame({
        "Date": [m.strftime("%Y-%m") for m in macro_months],
        "Year": [m.year for m in macro_months],
        "Month": [m.month for m in macro_months],
        "ISM_Manufacturing_PMI": [50 + i % 4 for i in range(len(macro_months))],
        "Auto_Production_YoY_Pct": [1.5] * len(macro_months),
        "Vehicle_Inventory_Days": [60.0] * len(macro_months),
        "Raw_Material_Cost_Index": [100 + i % 3 for i in range(len(macro_months))],
        "New_Vehicle_Orders_Index": [100.0] * len(macro_months),
        "Industrial_Util_Rate_Pct": [79.0] * len(macro_months),
        "Fuel_Price_Index": [90.0] * len(macro_months),
        "EV_Adoption_Index": [40.0 + i * 0.1 for i in range(len(macro_months))],
    }).to_csv(directory / "macro_indicators.csv", index=False)

    pd.DataFrame({
        "SKU_ID": SKUS,
        "Product_Family": [_family(s) for s in SKUS],
        "Business_Unit": ["Safety & Compliance"] * len(SKUS),
        "SKU_Description": [f"Part {s}" for s in SKUS],
        "Size_Class": ["Industrial 100"] * len(SKUS),
        "Powertrain_Application": ["Universal"] * len(SKUS),
        "Launch_Date": ["2018-05"] * len(SKUS),
        "EOL_Date": [""] * len(SKUS),
        "Replaces_SKU": [""] * len(SKUS),
        "Lifecycle_Status": ["Active"] * len(SKUS),
    }).to_csv(directory / "sku_master.csv", index=False)

    pd.DataFrame({
        "Event_ID": ["EV-001", "EV-002"],
        "Start_Date": ["2025-01", "2025-09"],
        "End_Date": ["", "2025-10"],
        "Scope_Type": ["Global", "Region"],
        "Scope_Value": ["All", "Europe"],
        "Event_Type": ["Price Change", "Disruption"],
        "Description": ["Annual list-price increase", "Port congestion"],
    }).to_csv(directory / "events_calendar.csv", index=False)

    pd.DataFrame({
        "Date": ["2025-03"], "SKU_ID": ["BSS-001"], "Region": ["North America"],
        "Requested_Units": [180], "Carried_In_Units": [100], "Shipped_Units": [100],
        "Backordered_Units": [40], "Lost_Units": [40], "Fill_Rate_Pct": [55.6],
        "Stockout_Days": [8], "Reason": ["Capacity constraint"],
    }).to_csv(directory / "inventory_stockouts.csv", index=False)

    pd.DataFrame({
        "Date": ["2026-07"], "SKU_ID": ["BSS-001"], "Region": ["North America"],
        "Units_Sold_Forecast": [100],
    }).to_csv(directory / "sample_submission.csv", index=False)


def digests(out_dir: Path) -> dict[str, str]:
    """Digest the upstream tables only - Phase 3 writes its own directory."""
    result = {}
    for name in ["clean", "anomalies"]:
        for path in sorted((out_dir / name).glob("*.parquet")):
            result[f"{name}/{path.name}"] = hashlib.md5(path.read_bytes()).hexdigest()
    return result


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("phase3")
    data_dir, out_dir = root / "student_data", root / "outputs"
    write_fixture(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert anomalies.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    before = digests(out_dir)
    assert features.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    payload = json.loads((out_dir / "features_report.json").read_text(encoding="utf-8"))
    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "before": before,
        "payload": payload,
        "frame": pd.read_parquet(out_dir / "features" / "features_monthly.parquet"),
    }


def test_feature_matrix_is_complete_and_read_only(pipeline: dict):
    frame = pipeline["frame"]
    dictionary = pd.read_csv(pipeline["out_dir"] / "features" / "feature_dictionary.csv")

    assert not frame.duplicated(["SKU_ID", "Region", "Month"]).any()
    assert set(frame["split"]) == {"train", "forecast"}
    assert int(frame["split"].eq("train").sum()) == len(SKUS) * len(REGIONS) * len(TRAIN_MONTHS)
    assert int(frame["split"].eq("forecast").sum()) == len(SKUS) * len(REGIONS) * len(FORECAST_MONTHS)
    assert frame["Month"].max() == FORECAST_MONTHS[-1]

    # every column is documented, and Phase 3 never rewrites an upstream table
    assert set(frame.columns) == set(dictionary["column"])
    assert digests(pipeline["out_dir"]) == pipeline["before"]

    for name in ["driver_effects.csv", "driver_screen_coefficients.csv",
                 "driver_panel_monthly.parquet"]:
        assert (pipeline["out_dir"] / "features" / name).exists(), name
    assert all(v["ok"] for v in pipeline["payload"]["verdicts"]), pipeline["payload"]["verdicts"]


def test_targets_and_lags_never_look_forward(pipeline: dict):
    frame = pipeline["frame"]
    one_cell = frame.loc[(frame["SKU_ID"] == "BSS-001") & (frame["Region"] == "North America")]
    one_cell = one_cell.sort_values("Month").set_index("Month")

    # the horizon carries no target at all
    forecast = frame.loc[frame["split"] == "forecast"]
    assert forecast[["target_demand", "target_observed"]].isna().all().all()

    # lag_1 is exactly the previous month's target, lag_3 three months back
    shifted = frame.groupby(["SKU_ID", "Region"], sort=False)["target_demand"].shift(1)
    assert np.array_equal(frame["lag_1"].fillna(-1).to_numpy(), shifted.fillna(-1).to_numpy())

    month = pd.Timestamp("2025-06-01")
    previous = pd.Timestamp("2025-05-01")
    assert one_cell.loc[month, "lag_1"] == one_cell.loc[previous, "target_demand"]

    # rolling features exclude the current month
    expected_mean_3 = one_cell.loc[[pd.Timestamp("2025-05-01"), pd.Timestamp("2025-04-01"),
                                    pd.Timestamp("2025-03-01")], "target_demand"].mean()
    assert np.isclose(one_cell.loc[month, "mean_3"], expected_mean_3)

    # a first-ever month has no prior order to count back to
    first = one_cell.index.min()
    assert pd.isna(one_cell.loc[first, "months_since_last_order"])

    # the calendar is matched to the cell scope: global reaches North America,
    # the Europe-only disruption does not
    assert one_cell.loc[pd.Timestamp("2025-01-01"), "event_count"] >= 1
    assert one_cell.loc[pd.Timestamp("2025-09-01"), "event_count"] == 0
    europe = frame.loc[(frame["SKU_ID"] == "BSS-001") & (frame["Region"] == "Europe")]
    europe_september = europe.loc[europe["Month"].eq(pd.Timestamp("2025-09-01"))]
    assert europe_september["event_supply_count"].iloc[0] >= 1

    # a future month without a price list carries the last known price
    future = frame.loc[frame["split"] == "forecast"]
    assert future["price"].notna().all()
    assert (future["price"] == PRICE).all()

    # structure features are lagged, so the first month cannot have them
    assert pd.isna(one_cell.loc[first, "customers_lag_1"])
    assert one_cell.loc[month, "customers_lag_1"] == 1
