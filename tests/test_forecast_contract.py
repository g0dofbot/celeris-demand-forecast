"""Phase 4 contract tests: submission shape, the post-EOL rule, recursive honesty."""
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

from src import anomalies, clean, features, forecast  # noqa: E402

TRAIN_MONTHS = pd.date_range("2023-01-01", "2026-08-01", freq="MS")
FORECAST_MONTHS = pd.date_range("2026-09-01", "2028-03-01", freq="MS")
REGIONS = {"North America": 1.0, "Europe": 0.6}
LEVELS = {"BSS-001": 200, "BSS-002": 120, "BSS-003": 60, "HYD-018": 300,
          "HYD-019": 90, "HYD-020": 40, "THM-026": 150}
EOL_SKU, EOL_MONTH = "THM-026", pd.Timestamp("2025-06-01")
SEASON = {1: 1.00, 2: 0.95, 3: 1.05, 4: 1.10, 5: 1.00, 6: 0.90,
          7: 0.85, 8: 0.80, 9: 1.00, 10: 1.05, 11: 1.10, 12: 1.15}
PRICE = 100.0


def family(sku: str) -> str:
    return {"BSS": "Brake & Safety Systems", "HYD": "Hydraulic Systems"}.get(
        sku[:3], "Thermal Management")


def units_for(sku: str, month: pd.Timestamp, region: str) -> int:
    """A learnable series: level x season x growth, with a dormant SKU and zeros."""
    if sku == "HYD-019" and month.month % 2 == 1:
        return -1  # no ERP row at all: absence, not a zero
    if sku == "BSS-003" and month.month in (7, 8, 9):
        return 0  # explicit zero order
    growth = 1.0 + 0.05 * (month.year - 2023)
    return int(round(LEVELS[sku] * SEASON[month.month] * growth * REGIONS[region]))


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows, index = [], 0
    for month in TRAIN_MONTHS:
        for sku in LEVELS:
            if sku == EOL_SKU and month > EOL_MONTH:
                continue
            for region in REGIONS:
                units = units_for(sku, month, region)
                if units < 0:
                    continue
                index += 1
                rows.append({
                    "Record_ID": f"CEL-{index:05d}", "Date": month.strftime("%Y-%m"),
                    "Year": month.year, "Month": month.month,
                    "Product_Family": family(sku), "Business_Unit": "Safety & Compliance",
                    "SKU_ID": sku, "SKU_Description": f"Part {sku}",
                    "Customer_Name": "Solaris Automotive", "Sales_Channel": "Tier-1 Distributor",
                    "Region": region, "Units_Sold": units, "Unit_Price_USD": PRICE,
                    "Revenue_USD": round(units * PRICE, 2),
                    "Load_Timestamp": f"{month:%Y-%m}-01 01:00:00",
                })
    pd.DataFrame(rows).to_csv(directory / "train.csv", index=False)

    test_rows = []
    for month in FORECAST_MONTHS:
        for sku in LEVELS:
            for region in REGIONS:
                test_rows.append({
                    "Date": month.strftime("%Y-%m"), "Year": month.year, "Month": month.month,
                    "Product_Family": family(sku), "Business_Unit": "Safety & Compliance",
                    "SKU_ID": sku, "SKU_Description": f"Part {sku}",
                    "Customer_Name": "Solaris Automotive",
                    "Sales_Channel": "Tier-1 Distributor", "Region": region,
                })
    test = pd.DataFrame(test_rows)
    test.to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"Date": test["Date"].head(2), "SKU_ID": test["SKU_ID"].head(2),
                  "Region": test["Region"].head(2), "Units_Sold_Forecast": [50, 50]}
                 ).to_csv(directory / "sample_submission.csv", index=False)

    macro_months = list(TRAIN_MONTHS) + list(FORECAST_MONTHS)
    pd.DataFrame({
        "Date": [m.strftime("%Y-%m") for m in macro_months],
        "Year": [m.year for m in macro_months], "Month": [m.month for m in macro_months],
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
        "SKU_ID": list(LEVELS), "Product_Family": [family(s) for s in LEVELS],
        "Business_Unit": ["Safety & Compliance"] * len(LEVELS),
        "SKU_Description": [f"Part {s}" for s in LEVELS],
        "Size_Class": ["Industrial 100"] * len(LEVELS),
        "Powertrain_Application": ["Universal"] * len(LEVELS),
        "Launch_Date": ["2018-05"] * len(LEVELS),
        "EOL_Date": [EOL_MONTH.strftime("%Y-%m") if s == EOL_SKU else "" for s in LEVELS],
        "Replaces_SKU": [""] * len(LEVELS), "Lifecycle_Status": ["Active"] * len(LEVELS),
    }).to_csv(directory / "sku_master.csv", index=False)

    pd.DataFrame({
        "Event_ID": ["EV-001"], "Start_Date": ["2025-01"], "End_Date": [""],
        "Scope_Type": ["Global"], "Scope_Value": ["All"], "Event_Type": ["Price Change"],
        "Description": ["Annual list-price increase"],
    }).to_csv(directory / "events_calendar.csv", index=False)

    pd.DataFrame({
        "Date": ["2025-03"], "SKU_ID": ["BSS-001"], "Region": ["North America"],
        "Requested_Units": [260], "Carried_In_Units": [200], "Shipped_Units": [200],
        "Backordered_Units": [60], "Lost_Units": [60], "Fill_Rate_Pct": [76.9],
        "Stockout_Days": [6], "Reason": ["Capacity constraint"],
    }).to_csv(directory / "inventory_stockouts.csv", index=False)


def digests(out_dir: Path) -> dict[str, str]:
    """Digest every upstream table written by phases 0-3."""
    result = {}
    for name in ["clean", "anomalies", "features"]:
        for path in sorted((out_dir / name).glob("*")):
            if path.suffix in {".parquet", ".csv"}:
                result[f"{name}/{path.name}"] = hashlib.md5(path.read_bytes()).hexdigest()
    return result


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("phase4")
    data_dir, out_dir = root / "student_data", root / "outputs"
    write_fixture(data_dir)
    assert clean.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert anomalies.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    assert features.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0

    before = digests(out_dir)
    assert forecast.main(["--data-dir", str(data_dir), "--out-dir", str(out_dir)]) == 0
    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "before": before,
        "submission": pd.read_csv(out_dir / "submission.csv"),
        "cells": pd.read_parquet(out_dir / "forecast" / "forecast_cell_month.parquet"),
        "scores": pd.read_csv(out_dir / "forecast" / "backtest_scores.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
        "payload": json.loads((out_dir / "forecast_report.json").read_text(encoding="utf-8")),
        "diagnostics": json.loads(
            (out_dir / "forecast" / "model_diagnostics.json").read_text(encoding="utf-8")),
    }


def test_submission_matches_the_test_keys(pipeline: dict):
    submission = pipeline["submission"]
    test = pipeline["test"]

    assert list(submission.columns) == ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert len(submission) == len(test) == len(LEVELS) * len(REGIONS) * len(FORECAST_MONTHS)
    assert not submission.duplicated().any()

    expected = (test[["Date", "SKU_ID", "Region"]]
                .sort_values(["Date", "SKU_ID", "Region"]).reset_index(drop=True))
    actual = submission[["Date", "SKU_ID", "Region"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)

    assert set(submission["Date"]) == {month.strftime("%Y-%m") for month in FORECAST_MONTHS}
    assert submission["Units_Sold_Forecast"].dtype.kind in "iu"
    assert (submission["Units_Sold_Forecast"] >= 0).all()
    # an all-zero submission would satisfy the shape checks, so pin the level too
    assert submission["Units_Sold_Forecast"].sum() > 0


def test_post_eol_keys_are_zero_and_upstream_is_untouched(pipeline: dict):
    cells = pipeline["cells"]
    post_eol = cells.loc[cells["post_eol"].astype(bool)]

    assert len(cells) == len(pipeline["test"])
    assert len(post_eol) == len(REGIONS) * len(FORECAST_MONTHS)
    assert set(post_eol["SKU_ID"]) == {EOL_SKU}
    assert (post_eol["forecast_units"] == 0).all()
    assert (post_eol["forecast_p90"] == 0).all()

    # a dormant month may legitimately be forecast at zero, but no live series may
    # be zero across the whole horizon
    live = cells.loc[~cells["post_eol"].astype(bool)]
    totals = live.groupby(["SKU_ID", "Region"])["forecast_units"].sum()
    assert len(totals) == len(LEVELS) * len(REGIONS) - len(REGIONS)
    assert (totals > 0).all(), totals[totals <= 0]

    assert digests(pipeline["out_dir"]) == pipeline["before"]


def test_bands_are_ordered_and_backtest_is_forward_in_time(pipeline: dict):
    cells = pipeline["cells"]
    assert (cells["forecast_p10"] <= cells["forecast_p50"]).all()
    assert (cells["forecast_p50"] <= cells["forecast_p90"]).all()
    assert (cells["forecast_p10"] >= 0).all()

    payload = pipeline["payload"]
    assert all(verdict["ok"] for verdict in payload["verdicts"]), payload["verdicts"]

    backtest = payload["metrics"]["backtest"]
    assert backtest["months"] == 19
    assert backtest["origin"] == "2025-02"
    diagnostics = pipeline["diagnostics"]
    assert all(value > 0.999 for value in diagnostics["lag_parity"].values())
    assert diagnostics["seed"] == forecast.SEED

    scores = pipeline["scores"].set_index("model")
    assert {"seasonal_naive", "moving_average", "hurdle_ridge", "hurdle_lgbm", "blend"} <= set(
        scores.index)
    assert diagnostics["selected"] in scores.index
    # the selection rule is the one the report states
    eligible = scores.loc[scores["bias"].abs() <= forecast.BIAS_TOLERANCE]
    assert eligible["wape"].idxmin() == diagnostics["selected"]

    # the submission and the cell table cannot disagree
    assert int(cells["forecast_units"].sum()) == int(
        pipeline["submission"]["Units_Sold_Forecast"].sum())
