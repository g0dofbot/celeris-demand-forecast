#!/usr/bin/env python
"""Single entry point for the Celeris demand forecasting pipeline.

    python run_pipeline.py                 # every implemented phase, in order
    python run_pipeline.py --phase 0       # forensic profiling only
    python run_pipeline.py --phase 0 --data-dir student_data

Phases are run as isolated subprocesses so a failure in one cannot leave the
others half-executed, and so the exit status of each stage is preserved.
Phases that are not implemented yet are reported explicitly instead of being
silently skipped.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

PHASES: dict[str, tuple[str, str]] = {
    "0": ("Forensic profiling", "src.profile_data"),
    "1": ("Cleaning & reconciliation", "src.clean"),
    "2": ("Anomaly register & unconstrained demand", "src.anomalies"),
    "3": ("Feature engineering & driver analysis", "src.features"),
    "4": ("Forecasting, backtest & submission", "src.forecast"),
    "5": ("Planning control tower dashboard", "src.dashboard_launch"),
}


def module_path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def run_phase(key: str, passthrough: list[str]) -> tuple[bool, str]:
    name, module = PHASES[key]
    path = module_path(module)
    if not path.exists():
        return False, f"phase {key} ({name}) not implemented yet: {path.name} is absent"

    banner = f"PHASE {key} - {name}"
    print(f"\n{'=' * 78}\n{banner}\n{'=' * 78}", flush=True)
    started = time.time()
    completed = subprocess.run([sys.executable, str(path), *passthrough], cwd=ROOT)
    elapsed = time.time() - started
    ok = completed.returncode == 0
    print(f"\n{banner}: {'OK' if ok else f'FAILED (exit {completed.returncode})'} "
          f"in {elapsed:.1f}s", flush=True)
    return ok, f"phase {key} exit {completed.returncode}"


def reexec_in_venv(argv: list[str]) -> int | None:
    """Re-run this script under .venv; returns its exit code, or None if already there.

    Uses subprocess rather than os.execv: on Windows, execv joins the argument list
    into a command line without quoting, which mangles this project's paths (they
    contain spaces) and silently drops the child's exit status.
    """
    if not VENV_PYTHON.exists():
        return None
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return None
    print(f"[run_pipeline] using project environment: {VENV_PYTHON}", flush=True)
    child = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve()), *argv], cwd=ROOT)
    return child.returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    forwarded = reexec_in_venv(argv)
    if forwarded is not None:
        return forwarded
    parser = argparse.ArgumentParser(description="Celeris demand forecasting pipeline")
    parser.add_argument("--phase", default="all",
                        choices=[*PHASES.keys(), "all"],
                        help="phase to execute (default: all)")
    parser.add_argument("--data-dir", default="student_data")
    args, passthrough = parser.parse_known_args(argv)

    if not (ROOT / args.data_dir).exists() and not Path(args.data_dir).exists():
        print(f"FATAL: data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    requested = list(PHASES) if args.phase == "all" else [args.phase]
    results = []
    for key in requested:
        ok, detail = run_phase(key, ["--data-dir", args.data_dir, *passthrough])
        results.append((key, ok, detail))

    print(f"\n{'=' * 78}\nPIPELINE SUMMARY\n{'=' * 78}")
    for key, ok, detail in results:
        print(f"  phase {key}: {'PASS' if ok else 'NOT RUN/FAIL'} - {detail}")

    executed = [ok for _, ok, _ in results]
    return 0 if all(executed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
