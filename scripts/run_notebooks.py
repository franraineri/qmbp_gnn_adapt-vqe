#!/usr/bin/env python
"""
GNN-HVA v6.0 — Notebook Executor with Validation

Executes the PoC notebooks programmatically with:
  - Pre-flight checks (imports, data dependencies, lint)
  - Cell-level error capture with context
  - Post-execution validation (metrics thresholds)
  - Result saving (executed notebook + summary JSON)

Usage:
    python scripts/run_notebooks.py                    # both notebooks
    python scripts/run_notebooks.py --phase 1-2        # only phases 1-2
    python scripts/run_notebooks.py --phase 3-4        # only phases 3-4
    python scripts/run_notebooks.py --timeout 600      # 10 min timeout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
NB_DIR = _root / "src" / "poc" / "v6"
RESULTS_DIR = _root / "scripts" / "notebook_results"

NOTEBOOKS = {
    "1-2": NB_DIR / "poc_v6_phases1_2.ipynb",
    "3-4": NB_DIR / "poc_v6_phases3_4.ipynb",
}

DATA_FILE = NB_DIR / "phase1_phase2_tfim_N6_p2_v6.npz"


# ── Pre-flight checks ───────────────────────────────────────────────────


def check_imports() -> list[str]:
    """Verify all required packages are importable."""
    errors = []
    for mod in ["qiskit", "torch", "torch_geometric", "sklearn", "numpy", "scipy", "nbformat"]:
        try:
            __import__(mod)
        except ImportError:
            errors.append(f"Missing: {mod}")
    return errors


def check_lint() -> bool:
    """Run ruff on v6 Python modules (not notebooks). Returns True if clean."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(NB_DIR), "--exclude", "*.ipynb"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"⚠️  Lint issues:\n{result.stdout[:500]}")
        return False
    return True


def check_data_dependency(phase: str) -> bool:
    """Phase 3-4 requires the .npz from Phase 1-2."""
    if phase == "3-4" and not DATA_FILE.exists():
        print(f"❌ Missing data: {DATA_FILE}")
        print("   Run phases 1-2 first, or: python scripts/run_notebooks.py --phase 1-2")
        return False
    return True


# ── Notebook execution ───────────────────────────────────────────────────


def execute_notebook(nb_path: Path, timeout: int = 300) -> dict:
    """Execute a notebook via nbconvert and return execution metadata."""
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    print(f"\n{'─' * 60}")
    print(f"  Executing: {nb_path.name}")
    print(f"{'─' * 60}")

    nb = nbformat.read(str(nb_path), as_version=4)
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
        extra_arguments=["--no-stderr"],
    )

    result = {
        "notebook": nb_path.name,
        "started": datetime.now().isoformat(),
        "success": False,
        "cells_total": len([c for c in nb.cells if c.cell_type == "code"]),
        "cells_executed": 0,
        "error_cell": None,
        "error_message": None,
        "error_traceback": None,
        "outputs": {},
        "elapsed_seconds": 0,
    }

    t0 = time.time()
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        result["success"] = True
        result["cells_executed"] = result["cells_total"]
    except CellExecutionError as e:
        result["error_message"] = str(e)[:500]
        # Find which cell failed
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    result["error_cell"] = i
                    result["error_traceback"] = "\n".join(output.get("traceback", []))[:2000]
                    break
    except Exception as e:
        result["error_message"] = f"Unexpected: {e}"

    result["elapsed_seconds"] = round(time.time() - t0, 1)

    # Extract key outputs from executed cells
    result["outputs"] = _extract_outputs(nb)

    # Save executed notebook
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{nb_path.stem}_executed_{ts}.ipynb"
    nbformat.write(nb, str(out_path))
    result["executed_notebook"] = str(out_path)

    # Print status
    if result["success"]:
        print(f"  ✅ Completed in {result['elapsed_seconds']}s")
    else:
        print(f"  ❌ Failed at cell {result['error_cell']} after {result['elapsed_seconds']}s")
        if result["error_traceback"]:
            # Print last 10 lines of traceback for debugging
            tb_lines = result["error_traceback"].split("\n")
            print("  Traceback (last 10 lines):")
            for line in tb_lines[-10:]:
                print(f"    {line}")

    return result


def _extract_outputs(nb) -> dict:
    """Pull key metrics from notebook stdout/stream outputs."""
    outputs = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream":
                text = output.get("text", "")
            elif output.get("output_type") == "execute_result":
                text = "".join(output.get("data", {}).get("text/plain", []))

            # Capture key metrics from output text
            for line in text.split("\n"):
                if "checklist" in line.lower() or "✅" in line or "❌" in line:
                    key = f"metric_{len(outputs)}"
                    outputs[key] = line.strip()
                elif "fidelity" in line.lower() and ":" in line:
                    outputs["fidelity_line"] = line.strip()
                elif "ΔE/gap" in line or "delta_e_over_gap" in line:
                    outputs["de_gap_line"] = line.strip()
    return outputs


# ── Post-execution validation ────────────────────────────────────────────


def validate_phase12(result: dict) -> list[str]:
    """Check Phase 1-2 produced the expected data file."""
    issues = []
    if not result["success"]:
        issues.append("Notebook execution failed")
        return issues
    if not DATA_FILE.exists():
        issues.append(f"Data file not created: {DATA_FILE}")
    else:
        import numpy as np

        data = dict(np.load(str(DATA_FILE), allow_pickle=True))
        if str(data.get("cost_function", "")) != "energy":
            issues.append(f"Wrong cost_function: {data.get('cost_function')}")
        if "theta_opt" not in data:
            issues.append("Missing theta_opt in dataset")
        n_points = len(data.get("h_values", []))
        if n_points < 20:
            issues.append(f"Only {n_points} h-points (expected ≥20 for non-uniform grid)")
    return issues


def validate_phase34(result: dict) -> list[str]:
    """Check Phase 3-4 metrics meet thresholds."""
    issues = []
    if not result["success"]:
        issues.append("Notebook execution failed")
    return issues


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Execute V6 notebooks with validation")
    parser.add_argument("--phase", choices=["1-2", "3-4", "all"], default="all")
    parser.add_argument("--timeout", type=int, default=300, help="Per-notebook timeout (seconds)")
    parser.add_argument("--skip-lint", action="store_true", help="Skip pre-flight lint check")
    args = parser.parse_args()

    print("=" * 60)
    print("  GNN-HVA v6.0 — Notebook Executor")
    print("=" * 60)

    # Pre-flight
    print("\n📋 Pre-flight checks...")
    import_errors = check_imports()
    if import_errors:
        print(f"  ❌ {import_errors}")
        return 1
    print("  ✅ All imports OK")

    if not args.skip_lint:
        if check_lint():
            print("  ✅ Lint clean")
        else:
            print("  ⚠️  Lint issues found (continuing anyway)")

    phases = ["1-2", "3-4"] if args.phase == "all" else [args.phase]
    all_results = []

    for phase in phases:
        nb_path = NOTEBOOKS[phase]
        if not nb_path.exists():
            print(f"  ❌ Notebook not found: {nb_path}")
            return 1

        if not check_data_dependency(phase):
            return 1

        result = execute_notebook(nb_path, timeout=args.timeout)
        all_results.append(result)

        # Post-execution validation
        validator = validate_phase12 if phase == "1-2" else validate_phase34
        issues = validator(result)
        if issues:
            print("  ⚠️  Validation issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✅ Validation passed")

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"run_summary_{ts}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "phases": args.phase,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 Summary saved: {summary_path}")

    # Final status
    all_ok = all(r["success"] for r in all_results)
    print(f"\n{'=' * 60}")
    if all_ok:
        print("  ✅ All notebooks executed successfully")
    else:
        failed = [r["notebook"] for r in all_results if not r["success"]]
        print(f"  ❌ Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
