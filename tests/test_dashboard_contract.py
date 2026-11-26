"""Phase 5 contract tests: the dashboard reads outputs/ only, and it renders.

The reader is tested against the outputs the pipeline actually produced, because
that is the contract: no recomputation, no raw CSV, no writes anywhere except its
own report.  If the phases have not been run, these tests skip rather than fail
with a misleading missing-file error.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dashboard_launch  # noqa: E402

OUTPUTS = ROOT / "outputs"
pytestmark = pytest.mark.skipif(
    not (OUTPUTS / "forecast_report.json").exists(),
    reason="run the pipeline first (outputs/ has no forecast_report.json)")


def digests(directory: Path) -> dict[str, str]:
    """Every output except the dashboard's own report."""
    return {str(path.relative_to(directory)): hashlib.md5(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and not path.name.startswith("dashboard_report")}


def test_declared_paths_stay_inside_outputs(tmp_path: Path):
    declared = list(dashboard_launch.REQUIRED.values()) + list(dashboard_launch.OPTIONAL.values())
    assert declared, "the dashboard must declare its sources"
    assert all(not Path(relative).is_absolute() for relative in declared)

    for relative in declared:
        resolved = dashboard_launch.resolve(tmp_path, relative)
        assert tmp_path in resolved.parents

    with pytest.raises(ValueError):
        dashboard_launch.resolve(tmp_path, "../student_data/train.csv")
    with pytest.raises(ValueError):
        dashboard_launch.resolve(tmp_path, "/etc/passwd")

    source = (ROOT / "src" / "dashboard_launch.py").read_text(encoding="utf-8")
    assert "student_data" not in source, "the dashboard must never touch the raw delivery"
    assert "read_csv(DEFAULT_DATA_DIR" not in source
    # the render loop takes the cheap path; the validating one belongs to the CLI
    assert "ok, rows, missing = check_presence(out_dir)" in source
    assert "check_outputs(out_dir)" in source  # ...and the CLI still uses it
    render_body = source[source.index("def render("):]
    assert "check_outputs" not in render_body


def test_app_check_parses_nothing(monkeypatch):
    """The per-interaction check must be stat-only: no parquet, no JSON parse."""

    def boom(*args, **kwargs):
        raise AssertionError("the app must not parse outputs on every interaction")

    monkeypatch.setattr(dashboard_launch, "read_table", boom)
    monkeypatch.setattr(dashboard_launch, "read_json", boom)

    ok, rows, missing = dashboard_launch.check_presence(OUTPUTS)
    assert ok and not missing
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["rows"] == 0 and row["columns"] == 0 for row in rows)
    assert all(row["bytes"] > 0 and row["modified"] > 0 for row in rows)

    # the same trap must catch the *other* path, which proves the assertions above
    # are not vacuous: check_outputs really does open every file
    full_ok, full_rows, _ = dashboard_launch.check_outputs(OUTPUTS)
    assert not full_ok
    assert any(row["status"].startswith("UNREADABLE") for row in full_rows)


def test_every_declared_output_is_present_and_readable():
    ok, rows, missing = dashboard_launch.check_outputs(OUTPUTS)
    assert ok, f"missing outputs: {missing}"
    required_rows = [row for row in rows if row["required"] and row["columns"]]
    assert required_rows and all(row["rows"] > 0 for row in required_rows)
    assert len(rows) == len(dashboard_launch.REQUIRED) + len(dashboard_launch.OPTIONAL)
    broken = [row for row in rows if row["status"] != "ok"]
    assert not broken, broken

    payload = json.loads((OUTPUTS / "dashboard_report.json").read_text(encoding="utf-8"))
    assert payload["meta"]["phase"] == 5
    assert all(verdict["ok"] for verdict in payload["verdicts"]), payload["verdicts"]
    assert payload["kpis"]["forecast_keys"] > 0


def test_corrupt_table_is_caught_by_full_check_only(tmp_path: Path):
    """The two paths differ on purpose, and the CLI reports instead of crashing."""
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "forecast_cell_month.parquet").write_bytes(b"not a parquet file")

    ok, rows, missing = dashboard_launch.check_outputs(tmp_path)
    record = next(row for row in rows if row["name"] == "forecast_cells")
    assert record["status"].startswith("UNREADABLE")
    assert not ok and "forecast/forecast_cell_month.parquet" in dashboard_launch.unreadable_files(rows)

    # presence only stats, so it cannot see the corruption - the documented trade-off
    cheap_ok, cheap_rows, _ = dashboard_launch.check_presence(tmp_path)
    cheap_record = next(row for row in cheap_rows if row["name"] == "forecast_cells")
    assert cheap_record["status"] == "ok"
    assert not cheap_ok  # still fails, because the other required files are absent

    # a missing input produces a FAIL report, not a KeyError traceback
    assert dashboard_launch.main(["--out-dir", str(tmp_path)]) == 1
    payload = json.loads((tmp_path / "dashboard_report.json").read_text(encoding="utf-8"))
    assert payload["verdicts"] and not any(verdict["ok"] for verdict in payload["verdicts"])
    assert payload["kpis"] == {}


def test_phase_run_is_read_only_apart_from_its_report():
    before = digests(OUTPUTS)
    assert dashboard_launch.main(["--out-dir", str(OUTPUTS)]) == 0
    after = digests(OUTPUTS)
    assert before == after, sorted(set(before) ^ set(after))


def test_every_page_renders_without_an_exception():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src" / "dashboard_launch.py"), default_timeout=180)
    app.run()
    assert not list(app.exception), [str(error) for error in app.exception]

    # the landing navigation is the Core tier; the optional views stay out of it
    assert list(app.sidebar.radio[0].options) == dashboard_launch.CORE_PAGES
    for page in dashboard_launch.CORE_PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not list(app.exception), (page, [str(error) for error in app.exception])
        # proves the navigation happened rather than re-rendering page one
        assert app.title[0].value == dashboard_launch.PAGE_TITLES[page], page

    app.sidebar.toggle[0].set_value(True)
    app.run()
    assert list(app.sidebar.radio[0].options) == dashboard_launch.PAGES
    for page in dashboard_launch.ADVANCED_PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not list(app.exception), (page, [str(error) for error in app.exception])
        assert app.title[0].value == dashboard_launch.PAGE_TITLES[page], page

    # switching the tier off must return to a Core view, not leave a dead page selected
    app.sidebar.toggle[0].set_value(False)
    app.run()
    assert not list(app.exception), [str(error) for error in app.exception]
    assert app.title[0].value == dashboard_launch.PAGE_TITLES["Overview"]

    # the overview must actually show numbers, not just an empty shell
    app.sidebar.radio[0].set_value("Overview")
    app.run()
    assert len(app.metric) >= 6
    assert app.title[0].value == "Planning overview"


def test_empty_forecast_slice_explains_itself_and_drops_the_download():
    """An impossible filter combination must say so, not paint blank panels and a header-only CSV."""
    from streamlit.testing.v1 import AppTest

    cells = dashboard_launch.read_table(OUTPUTS / "forecast" / "forecast_cell_month.parquet")
    combos = set(zip(cells["Product_Family"], cells["Business_Unit"]))
    family = sorted(cells["Product_Family"].dropna().unique())[0]
    unit = next((candidate for candidate in sorted(cells["Business_Unit"].dropna().unique())
                 if (family, candidate) not in combos), None)
    assert unit is not None, "no incompatible family/business-unit pair to filter with"

    app = AppTest.from_file(str(ROOT / "src" / "dashboard_launch.py"), default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value("Forecast")
    app.run()

    # positive control: a real slice does offer the file, so its absence below means something
    assert app.title[0].value == "Forecast explorer"
    assert len(app.get("download_button")) == 1

    app.selectbox[0].set_value(family)
    app.selectbox[2].set_value(unit)
    app.run()
    assert not list(app.exception), [str(error) for error in app.exception]

    assert not len(app.dataframe), "an empty slice must not render blank tables"
    assert not len(app.get("plotly_chart"))
    assert not len(app.get("download_button")), "an empty slice must not offer a header-only CSV"
    assert [element.value for element in app.info] == [
        f"No forecast cells match {family} / {unit}, All regions. "
        "Widen a filter to see the slice."]
