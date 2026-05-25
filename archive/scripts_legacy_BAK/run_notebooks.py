#!/usr/bin/env python
"""
GNN-HVA v6.x — Notebook Executor with Auto-Registry

Executes the PoC notebooks programmatically with:
  - Pre-flight checks (imports, data dependencies, lint)
  - Wall-clock timeout guard (prevents runaway VQE from blocking pipeline)
  - Cell-level error capture with context and per-cell timing
  - Post-execution validation (metrics thresholds)
  - Peak memory tracking (critical for N=10 scaling)
  - Auto-registry: saves executed notebook + structured JSON summary
    with full metrics, timing, environment info, and git state
  - Binnacle-ready output: generates markdown summary that can be
    directly appended to documentation/binnacles/ without AI assistance

Exit codes:
    0 — All notebooks passed execution and validation
    1 — Notebook execution failure (cell error or timeout)
    2 — Validation failure (notebook ran but metrics out of spec)
    3 — Pre-flight failure (missing dependencies or data)

Usage:
    python scripts/run_notebooks.py                    # both notebooks
    python scripts/run_notebooks.py --phase 1-2        # only phases 1-2
    python scripts/run_notebooks.py --phase 3-4        # only phases 3-4
    python scripts/run_notebooks.py --timeout 600      # 10 min wall-clock timeout
    python scripts/run_notebooks.py --binnacle         # also write binnacle entry
    python scripts/run_notebooks.py --label "N10 test" # label for the run
    python scripts/run_notebooks.py --binnacle-file binnacle-N10.md
    python scripts/run_notebooks.py --dry-run          # pre-flight only, no execution
    python scripts/run_notebooks.py --keep-last 10     # prune old results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
NB_DIR = _root / "src" / "poc" / "v6"
RESULTS_DIR = _root / "scripts" / "notebook_results"
BINNACLE_DIR = _root / "documentation" / "binnacles"

NOTEBOOKS = {
    "1-2": NB_DIR / "poc_v6_phases1_2.ipynb",
    "3-4": NB_DIR / "poc_v6_phases3_4.ipynb",
}

DATA_FILE = NB_DIR / "phase1_phase2_tfim_N6_p2_v6.npz"

# Maximum total size for all_outputs registry (bytes of JSON text)
_MAX_OUTPUTS_SIZE = 512_000  # 512 KB cap


# ── Exit codes ───────────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_EXECUTION_FAILURE = 1
EXIT_VALIDATION_FAILURE = 2
EXIT_PREFLIGHT_FAILURE = 3


# ── Timeout infrastructure ───────────────────────────────────────────────


class NotebookTimeout(Exception):
    """Raised when a notebook exceeds its wall-clock timeout."""


def _alarm_handler(signum, frame):
    raise NotebookTimeout("Notebook exceeded wall-clock timeout")


# ── Safe float conversion ────────────────────────────────────────────────


def _safe_float(s: str) -> float | None:
    """Convert string to float, returning None for non-finite or invalid values."""
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, OverflowError):
        return None


# ── Environment capture ──────────────────────────────────────────────────


def capture_environment() -> dict:
    """Capture full environment info for reproducibility."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
    }

    # Git state
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_root),
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=str(_root),
        ).stdout.strip()
        git_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(_root),
        ).stdout.strip()
        env["git_commit"] = git_hash
        env["git_branch"] = git_branch
        env["git_dirty"] = bool(git_dirty)
    except Exception:
        env["git_commit"] = "unknown"
        env["git_branch"] = "unknown"
        env["git_dirty"] = None

    # Key package versions
    for pkg in ["qiskit", "torch", "torch_geometric", "sklearn", "numpy", "scipy"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not installed"

    return env


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
    """Execute a notebook via nbconvert and return execution metadata.

    The timeout is enforced as a wall-clock limit using SIGALRM, not
    per-cell (which is nbconvert's default behavior). This prevents
    runaway VQE optimizations from blocking the entire pipeline.
    """
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    print(f"\n{'─' * 60}")
    print(f"  Executing: {nb_path.name} (wall-clock limit: {timeout}s)")
    print(f"{'─' * 60}")

    nb = nbformat.read(str(nb_path), as_version=4)
    # Per-cell timeout set to wall-clock limit (SIGALRM is the real guard)
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
    )

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    result = {
        "notebook": nb_path.name,
        "notebook_path": str(nb_path),
        "started": datetime.now().isoformat(),
        "success": False,
        "cells_total": len(code_cells),
        "cells_executed": 0,
        "error_cell": None,
        "error_cell_source": None,
        "error_message": None,
        "error_traceback": None,
        "metrics": {},
        "all_outputs": [],
        "elapsed_seconds": 0,
        "peak_memory_mb": None,
        "slowest_cell_seconds": None,
    }

    t0 = time.time()

    # Wall-clock timeout via SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        result["success"] = True
        result["cells_executed"] = result["cells_total"]
    except NotebookTimeout:
        elapsed = round(time.time() - t0, 1)
        result["error_message"] = f"Wall-clock timeout after {elapsed}s (limit: {timeout}s)"
    except CellExecutionError as e:
        result["error_message"] = str(e)[:1000]
        # Find which cell failed and count cells executed before it
        code_idx = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            has_error = False
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    result["error_cell"] = code_idx
                    result["error_cell_source"] = cell.source[:500]
                    result["error_traceback"] = "\n".join(output.get("traceback", []))[:3000]
                    has_error = True
                    break
            if has_error:
                break
            code_idx += 1
        result["cells_executed"] = code_idx
    except Exception as e:
        result["error_message"] = f"Unexpected: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)

    result["elapsed_seconds"] = round(time.time() - t0, 1)

    # Peak memory (macOS reports bytes, Linux reports KB)
    try:
        rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        if sys.platform == "darwin":
            result["peak_memory_mb"] = round(rusage.ru_maxrss / (1024 * 1024), 1)
        else:
            result["peak_memory_mb"] = round(rusage.ru_maxrss / 1024, 1)
    except Exception:
        pass

    # Per-cell timing from execution metadata
    result["slowest_cell_seconds"] = _extract_slowest_cell(nb)

    # Extract outputs and metrics
    result["metrics"] = _extract_metrics(nb)
    result["all_outputs"] = _extract_all_outputs(nb)

    # Save executed notebook
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{nb_path.stem}_executed_{ts}.ipynb"
    nbformat.write(nb, str(out_path))
    result["executed_notebook"] = str(out_path)

    # Print status
    if result["success"]:
        mem_str = f", {result['peak_memory_mb']}MB" if result["peak_memory_mb"] else ""
        print(
            f"  ✅ Completed in {result['elapsed_seconds']}s "
            f"({result['cells_executed']} cells{mem_str})"
        )
    else:
        print(f"  ❌ Failed at cell {result['error_cell']} after {result['elapsed_seconds']}s")
        if result["error_cell_source"]:
            print("  Cell source (first 200 chars):")
            print(f"    {result['error_cell_source'][:200]}")
        if result["error_traceback"]:
            tb_lines = result["error_traceback"].split("\n")
            print("  Traceback (last 10 lines):")
            for line in tb_lines[-10:]:
                print(f"    {line}")

    # Print extracted metrics
    if result["metrics"]:
        print("\n  📊 Extracted metrics:")
        for key, value in result["metrics"].items():
            print(f"    {key}: {value}")

    if result["slowest_cell_seconds"]:
        print(f"  ⏱️  Slowest cell: {result['slowest_cell_seconds']:.1f}s")

    return result


def _extract_slowest_cell(nb) -> float | None:
    """Extract the slowest cell execution time from notebook metadata."""
    max_duration = 0.0
    found_any = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        meta = cell.get("metadata", {})
        # nbconvert stores execution timing in metadata.execution
        execution = meta.get("execution", {})
        start = execution.get("iopub.execute_input")
        end = execution.get("shell.execute_reply")
        if start and end:
            from datetime import datetime as dt

            try:
                # Parse ISO timestamps (may have timezone)
                t_start = dt.fromisoformat(start.replace("Z", "+00:00"))
                t_end = dt.fromisoformat(end.replace("Z", "+00:00"))
                duration = (t_end - t_start).total_seconds()
                if duration > max_duration:
                    max_duration = duration
                    found_any = True
            except (ValueError, TypeError):
                continue
    return round(max_duration, 1) if found_any else None


def _extract_metrics(nb) -> dict:
    """Pull structured metrics from notebook outputs.

    Looks for specific patterns in cell outputs to extract quantitative
    metrics automatically — no AI assistance needed to interpret results.
    Uses _safe_float() to handle malformed output gracefully.
    """
    import re

    metrics = {}

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = ""
            if output.get("output_type") == "stream":
                text = output.get("text", "")
            elif output.get("output_type") == "execute_result":
                text = "".join(output.get("data", {}).get("text/plain", []))

            for line in text.split("\n"):
                # Fidelity patterns
                m = re.search(r"[Aa]vg\s*fidelity[:\s]+(\d+\.?\d*)%?", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["avg_fidelity"] = v

                m = re.search(r"fid\s*[≥>=]+\s*(\d+\.?\d*)%?\s*:\s*(\d+)/(\d+)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["fid_threshold"] = v
                        metrics["fid_pass_count"] = int(m.group(2))
                        metrics["fid_total_count"] = int(m.group(3))

                # MSE patterns
                m = re.search(r"[Ff]inal\s*MSE[:\s]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["final_mse"] = v

                # ΔE/gap patterns
                m = re.search(r"[ΔD]E/gap[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["delta_e_over_gap"] = v

                # Checklist patterns
                m = re.search(r"[Cc]hecklist[:\s]+(\d+)/(\d+)", line)
                if m:
                    metrics["checklist_pass"] = int(m.group(1))
                    metrics["checklist_total"] = int(m.group(2))

                # Phase classification
                m = re.search(r"[Pp]hase[:\s]+(paramagnetic|ferromagnetic|indeterminate)", line)
                if m:
                    metrics["phase_label"] = m.group(1)

                # Training points
                m = re.search(r"[Tt]raining\s*(?:graphs|points)[:\s]+(\d+)/(\d+)", line)
                if m:
                    metrics["training_points"] = int(m.group(1))
                    metrics["total_points"] = int(m.group(2))

                # Energy
                m = re.search(r"E0\s*range[:\s]+\[([^,]+),\s*([^\]]+)\]", line)
                if m:
                    v1, v2 = _safe_float(m.group(1)), _safe_float(m.group(2))
                    if v1 is not None and v2 is not None:
                        metrics["e0_min"] = v1
                        metrics["e0_max"] = v2

                # Gradient analysis peaks
                m = re.search(r"[Pp]eaks?\s*detected[:\s]+(\d+)", line)
                if m:
                    metrics["gradient_peaks"] = int(m.group(1))

                m = re.search(r"[Cc]ritical\s*region\s*detected[:\s]+(True|False)", line)
                if m:
                    metrics["critical_region_detected"] = m.group(1) == "True"

                # ── V6.1 Hardware Deployer metrics ──

                # ZNE R² quality
                m = re.search(r"R[²2][:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["zne_r_squared"] = v

                # CES values
                m = re.search(r"CES\s*range[:\s]+\[([^,]+),\s*([^\]]+)\]", line)
                if m:
                    v1, v2 = _safe_float(m.group(1)), _safe_float(m.group(2))
                    if v1 is not None and v2 is not None:
                        metrics["ces_min"] = v1
                        metrics["ces_max"] = v2

                # Sigma (statistical uncertainty) — matches both "σ:" and "sigma:"
                m = re.search(r"σ[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if not m:
                    m = re.search(r"[Ss]igma[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["sigma"] = v

                # Shot budget
                m = re.search(r"[Ss]hots?[:\s=]+(\d{4,})", line)
                if m:
                    metrics["shots"] = int(m.group(1))

                # Extrapolation method
                m = re.search(r"[Ee]xtrapolation[:\s]+(linear|nn|none)", line)
                if m:
                    metrics["extrapolation_method"] = m.group(1)

                # Per-parameter head losses
                m = re.search(r"ZZ-head[^:]*[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["zz_head_loss"] = v

                m = re.search(r"X-head[^:]*[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["x_head_loss"] = v

                # Gradient norm range
                m = re.search(r"[Gg]radient\s*norm\s*range[:\s]+\[([^,]+),\s*([^\]]+)\]", line)
                if m:
                    v1, v2 = _safe_float(m.group(1)), _safe_float(m.group(2))
                    if v1 is not None and v2 is not None:
                        metrics["gradient_norm_min"] = v1
                        metrics["gradient_norm_max"] = v2

                # Peak h-values from gradient analysis
                m = re.search(r"[Pp]eak\s*(?:at\s*)?h[=:\s]+(\d+\.?\d*)", line)
                if m and "gradient_peak_h" not in metrics:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["gradient_peak_h"] = v

                # Observable errors
                m = re.search(r"⟨X⟩[:\s=]+([+-]?\d+\.?\d*)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["mag_x"] = v

                m = re.search(r"⟨ZZ⟩[:\s=]+([+-]?\d+\.?\d*)", line)
                if m:
                    v = _safe_float(m.group(1))
                    if v is not None:
                        metrics["corr_zz"] = v

                # Mode (simulation/hardware)
                m = re.search(r"[Mm]ode[:\s]+(simulation|hardware)", line)
                if m:
                    metrics["deploy_mode"] = m.group(1)

    return metrics


def _extract_all_outputs(nb) -> list[dict]:
    """Extract all cell outputs for full registry (debugging/audit trail).

    Enforces a total size cap (_MAX_OUTPUTS_SIZE) to prevent multi-MB
    registry files from notebooks with many verbose cells.
    """
    outputs = []
    total_size = 0
    code_idx = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell_outputs = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                cell_outputs.append(
                    {
                        "type": "stream",
                        "name": output.get("name", "stdout"),
                        "text": output.get("text", "")[:2000],
                    }
                )
            elif output.get("output_type") == "execute_result":
                text = "".join(output.get("data", {}).get("text/plain", []))
                cell_outputs.append(
                    {
                        "type": "result",
                        "text": text[:2000],
                    }
                )
            elif output.get("output_type") == "error":
                cell_outputs.append(
                    {
                        "type": "error",
                        "ename": output.get("ename", ""),
                        "evalue": output.get("evalue", "")[:500],
                    }
                )
        if cell_outputs:
            entry = {
                "cell_index": code_idx,
                "source_preview": cell.source[:200],
                "outputs": cell_outputs,
            }
            entry_size = len(json.dumps(entry, default=str))
            if total_size + entry_size > _MAX_OUTPUTS_SIZE:
                outputs.append(
                    {
                        "cell_index": code_idx,
                        "source_preview": "[TRUNCATED — output cap reached]",
                        "outputs": [],
                    }
                )
                break
            outputs.append(entry)
            total_size += entry_size
        code_idx += 1
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
        # Additional metrics from the data file
        result["metrics"]["dataset_n_points"] = n_points
        result["metrics"]["dataset_cost_function"] = str(data.get("cost_function", ""))
        if "fidelities" in data:
            fids = data["fidelities"]
            result["metrics"]["dataset_avg_fidelity"] = float(np.mean(fids))
            result["metrics"]["dataset_min_fidelity"] = float(np.min(fids))
            result["metrics"]["dataset_fid_ge_93pct"] = int(np.sum(fids >= 0.93))
    return issues


def validate_phase34(result: dict) -> list[str]:
    """Check Phase 3-4 metrics meet thresholds."""
    issues = []
    if not result["success"]:
        issues.append("Notebook execution failed")
    # Check extracted metrics against thresholds
    metrics = result.get("metrics", {})
    if "delta_e_over_gap" in metrics and metrics["delta_e_over_gap"] > 0.10:
        issues.append(f"ΔE/gap = {metrics['delta_e_over_gap']:.4f} > 10% (poor)")
    if "final_mse" in metrics and metrics["final_mse"] > 0.1:
        issues.append(f"Final MSE = {metrics['final_mse']:.4f} > 0.1 (MPNN not converged)")
    return issues


# ── Binnacle generation ──────────────────────────────────────────────────


def _auto_observations(all_results: list[dict]) -> list[str]:
    """Generate automatic observations from metrics — no AI needed.

    Compares metrics against known thresholds and produces human-readable
    bullet points summarizing the run quality.
    """
    obs = []
    for result in all_results:
        metrics = result.get("metrics", {})
        nb = result["notebook"]

        if not result["success"]:
            obs.append(f"❌ {nb} failed at cell {result.get('error_cell', '?')}")
            continue

        # Fidelity assessment
        avg_fid = metrics.get("avg_fidelity") or metrics.get("dataset_avg_fidelity")
        if avg_fid is not None:
            if avg_fid >= 99.0:
                obs.append(f"✅ {nb}: Excellent VQE fidelity ({avg_fid:.1f}%)")
            elif avg_fid >= 93.0:
                obs.append(f"✅ {nb}: Good VQE fidelity ({avg_fid:.1f}%)")
            else:
                obs.append(f"⚠️ {nb}: Low VQE fidelity ({avg_fid:.1f}%) — check VQE config")

        # MPNN convergence
        mse = metrics.get("final_mse")
        if mse is not None:
            if mse < 0.005:
                obs.append(f"✅ {nb}: MPNN well-converged (MSE={mse:.4f})")
            elif mse < 0.05:
                obs.append(f"✅ {nb}: MPNN converged (MSE={mse:.4f})")
            else:
                obs.append(f"⚠️ {nb}: MPNN not fully converged (MSE={mse:.4f}) — increase epochs")

        # Deployment quality
        de_gap = metrics.get("delta_e_over_gap")
        if de_gap is not None:
            if de_gap < 0.05:
                obs.append(f"✅ {nb}: ΔE/gap={de_gap:.4f} < 5% — PASS")
            elif de_gap < 0.10:
                obs.append(f"⚠️ {nb}: ΔE/gap={de_gap:.4f} — marginal (5-10%)")
            else:
                obs.append(f"❌ {nb}: ΔE/gap={de_gap:.4f} > 10% — FAIL")

        # Checklist
        ck_pass = metrics.get("checklist_pass")
        ck_total = metrics.get("checklist_total")
        if ck_pass is not None and ck_total is not None:
            if ck_pass == ck_total:
                obs.append(f"✅ {nb}: Full checklist {ck_pass}/{ck_total}")
            elif ck_pass >= ck_total - 1:
                obs.append(f"✅ {nb}: Checklist {ck_pass}/{ck_total} (near-perfect)")
            else:
                obs.append(f"⚠️ {nb}: Checklist {ck_pass}/{ck_total}")

        # Gradient analysis
        if metrics.get("critical_region_detected"):
            h_peak = metrics.get("gradient_peak_h", "?")
            obs.append(f"🔬 {nb}: Phase transition detected in weight space at h≈{h_peak}")

    return obs


def _load_previous_run() -> dict | None:
    """Load the most recent previous run summary for comparison."""
    if not RESULTS_DIR.exists():
        return None
    jsons = sorted(RESULTS_DIR.glob("run_summary_*.json"))
    if not jsons:
        return None
    # Load the last saved run (the current run hasn't been written yet)
    with open(jsons[-1]) as f:
        return json.load(f)


def _resolve_binnacle_path(label: str, binnacle_file: str | None) -> Path:
    """Determine the correct binnacle file based on label and explicit override.

    Priority:
    1. Explicit --binnacle-file argument
    2. Auto-detect from label (e.g., "N10" in label → binnacle-N10.md)
    3. Default to binnacle-N6.md (or create binnacle-notebook-runs.md)
    """
    if binnacle_file:
        return BINNACLE_DIR / binnacle_file

    # Auto-detect from label
    import re

    m = re.search(r"N(\d+)", label, re.IGNORECASE)
    if m:
        target = BINNACLE_DIR / f"binnacle-N{m.group(1)}.md"
        return target  # will be created if it doesn't exist

    # Default fallback
    default = BINNACLE_DIR / "binnacle-N6.md"
    if default.exists():
        return default
    return BINNACLE_DIR / "binnacle-notebook-runs.md"


def generate_binnacle_entry(all_results: list[dict], env: dict, label: str = "") -> str:
    """Generate a markdown binnacle entry from execution results.

    This is the key "auto-registry" feature: the output is a complete,
    human-readable record that can be appended to a binnacle file without
    any AI interpretation or rewriting.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"Notebook Execution — {label}" if label else "Notebook Execution"

    lines = [
        f"\n---\n\n## {now} — {title}\n\n",
        "### Environment\n\n",
        f"- Git: `{env.get('git_branch', '?')}` @ `{env.get('git_commit', '?')}`"
        f"{' (dirty)' if env.get('git_dirty') else ''}\n",
        f"- Python: {env.get('python_version', '?').split()[0]}\n",
        f"- Qiskit: {env.get('qiskit_version', '?')}, "
        f"PyTorch: {env.get('torch_version', '?')}, "
        f"PyG: {env.get('torch_geometric_version', '?')}\n",
        f"- Platform: {env.get('platform', '?')}\n\n",
    ]

    for result in all_results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        lines.append(f"### {result['notebook']} — {status}\n\n")
        lines.append(f"- Elapsed: {result['elapsed_seconds']}s\n")
        lines.append(f"- Cells: {result['cells_executed']}/{result['cells_total']}\n")
        if result.get("peak_memory_mb"):
            lines.append(f"- Peak memory: {result['peak_memory_mb']} MB\n")
        if result.get("slowest_cell_seconds"):
            lines.append(f"- Slowest cell: {result['slowest_cell_seconds']}s\n")

        if not result["success"]:
            lines.append(
                f"- Error at cell {result['error_cell']}: "
                f"{result.get('error_message', 'unknown')[:200]}\n"
            )

        metrics = result.get("metrics", {})
        if metrics:
            lines.append("\n**Metrics:**\n\n")
            lines.append("| Metric | Value |\n|--------|-------|\n")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.6g} |\n")
                else:
                    lines.append(f"| {key} | {value} |\n")
            lines.append("\n")

    # Auto-generated observations
    observations = _auto_observations(all_results)
    if observations:
        lines.append("### Observations (auto-generated)\n\n")
        for obs in observations:
            lines.append(f"- {obs}\n")
        lines.append("\n")

    # Comparison with previous run
    prev = _load_previous_run()
    if prev and prev.get("results"):
        lines.append("### Comparison with Previous Run\n\n")
        prev_metrics = {}
        for pr in prev["results"]:
            prev_metrics.update(pr.get("metrics", {}))
        curr_metrics = {}
        for cr in all_results:
            curr_metrics.update(cr.get("metrics", {}))

        compare_keys = ["final_mse", "delta_e_over_gap", "avg_fidelity", "checklist_pass"]
        lines.append("| Metric | Previous | Current | Change |\n")
        lines.append("|--------|----------|---------|--------|\n")
        for key in compare_keys:
            pv = prev_metrics.get(key)
            cv = curr_metrics.get(key)
            if (
                pv is not None
                and cv is not None
                and isinstance(pv, int | float)
                and isinstance(cv, int | float)
            ):
                diff = cv - pv
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                lines.append(f"| {key} | {pv:.4g} | {cv:.4g} | {arrow} {abs(diff):.4g} |\n")
        lines.append("\n")

    return "".join(lines)


# ── Results pruning ──────────────────────────────────────────────────────


def prune_results(keep_last: int) -> None:
    """Remove old result files, keeping only the N most recent summaries
    and their associated executed notebooks."""
    if not RESULTS_DIR.exists():
        return

    # Prune summaries
    summaries = sorted(RESULTS_DIR.glob("run_summary_*.json"))
    if len(summaries) > keep_last:
        to_remove = summaries[: len(summaries) - keep_last]
        for f in to_remove:
            f.unlink()
            print(f"  🗑️  Pruned: {f.name}")

    # Prune executed notebooks (keep same count)
    notebooks = sorted(RESULTS_DIR.glob("*_executed_*.ipynb"))
    if len(notebooks) > keep_last * 2:  # 2 notebooks per run (phases 1-2, 3-4)
        to_remove = notebooks[: len(notebooks) - keep_last * 2]
        for f in to_remove:
            f.unlink()
            print(f"  🗑️  Pruned: {f.name}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Execute V6 notebooks with validation and auto-registry"
    )
    parser.add_argument("--phase", choices=["1-2", "3-4", "all"], default="all")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Wall-clock timeout per notebook in seconds (default: 300)",
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip pre-flight lint check")
    parser.add_argument(
        "--binnacle",
        action="store_true",
        help="Write binnacle entry to documentation/binnacles/",
    )
    parser.add_argument(
        "--binnacle-file",
        type=str,
        default=None,
        help="Explicit binnacle filename (default: auto-detect from --label)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Label for this run (appears in binnacle, used for file routing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pre-flight checks only, do not execute notebooks",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="Prune old results, keeping only the N most recent runs",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  GNN-HVA v6.x — Notebook Executor with Auto-Registry")
    print("=" * 60)

    # Capture environment FIRST (before any execution)
    env = capture_environment()
    print(
        f"\n🔧 Environment: Python {env['python_version'].split()[0]}, "
        f"git {env.get('git_branch', '?')}@{env.get('git_commit', '?')}"
    )

    # Pre-flight
    print("\n📋 Pre-flight checks...")
    import_errors = check_imports()
    if import_errors:
        print(f"  ❌ {import_errors}")
        return EXIT_PREFLIGHT_FAILURE
    print("  ✅ All imports OK")

    if not args.skip_lint:
        if check_lint():
            print("  ✅ Lint clean")
        else:
            print("  ⚠️  Lint issues found (continuing anyway)")

    phases = ["1-2", "3-4"] if args.phase == "all" else [args.phase]

    # Check data dependencies upfront
    for phase in phases:
        nb_path = NOTEBOOKS[phase]
        if not nb_path.exists():
            print(f"  ❌ Notebook not found: {nb_path}")
            return EXIT_PREFLIGHT_FAILURE
        if not check_data_dependency(phase):
            return EXIT_PREFLIGHT_FAILURE

    print("  ✅ All data dependencies OK")

    # Dry-run: stop here
    if args.dry_run:
        print("\n🏁 Dry-run complete. Would execute:")
        for phase in phases:
            print(f"    - {NOTEBOOKS[phase].name} (timeout: {args.timeout}s)")
        return EXIT_OK

    # Execute notebooks
    all_results = []
    has_validation_failure = False

    for phase in phases:
        nb_path = NOTEBOOKS[phase]
        result = execute_notebook(nb_path, timeout=args.timeout)
        all_results.append(result)

        # Post-execution validation
        validator = validate_phase12 if phase == "1-2" else validate_phase34
        issues = validator(result)
        if issues:
            print("  ⚠️  Validation issues:")
            for issue in issues:
                print(f"    - {issue}")
            if result["success"]:
                has_validation_failure = True
        else:
            if result["success"]:
                print("  ✅ Validation passed")

        # Early abort: if phase 1-2 fails, skip 3-4 (it depends on the .npz)
        if not result["success"] and phase == "1-2" and "3-4" in phases:
            print("  ⛔ Skipping phase 3-4 (phase 1-2 failed)")
            break

    # Save structured JSON summary (the auto-registry)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(
        f"{args.phase}:{env.get('git_commit', '')}:{args.label}:{ts}".encode()
    ).hexdigest()[:8]
    summary_path = RESULTS_DIR / f"run_summary_{ts}_{run_id}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "label": args.label,
        "phases": args.phase,
        "environment": env,
        "results": all_results,
        "total_elapsed": sum(r["elapsed_seconds"] for r in all_results),
        "all_passed": all(r["success"] for r in all_results),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 JSON registry saved: {summary_path}")

    # Generate and optionally save binnacle entry
    binnacle_md = generate_binnacle_entry(all_results, env, args.label)

    if args.binnacle:
        binnacle_path = _resolve_binnacle_path(args.label, args.binnacle_file)
        BINNACLE_DIR.mkdir(parents=True, exist_ok=True)
        with open(binnacle_path, "a") as f:
            f.write(binnacle_md)
        print(f"📓 Binnacle entry appended to: {binnacle_path}")
    else:
        print("\n📓 Binnacle-ready output (use --binnacle to save to file):")
        print(binnacle_md)

    # Prune old results if requested
    if args.keep_last is not None:
        print(f"\n🧹 Pruning results (keeping last {args.keep_last})...")
        prune_results(args.keep_last)

    # Final status with structured exit codes
    all_ok = all(r["success"] for r in all_results)
    print(f"\n{'=' * 60}")
    if all_ok and not has_validation_failure:
        total_time = sum(r["elapsed_seconds"] for r in all_results)
        print(f"  ✅ All notebooks executed successfully ({total_time:.0f}s total)")
        print(f"{'=' * 60}")
        return EXIT_OK
    elif all_ok and has_validation_failure:
        print("  ⚠️  Notebooks ran but validation failed (metrics out of spec)")
        print(f"{'=' * 60}")
        return EXIT_VALIDATION_FAILURE
    else:
        failed = [r["notebook"] for r in all_results if not r["success"]]
        print(f"  ❌ Failed: {', '.join(failed)}")
        print(f"{'=' * 60}")
        return EXIT_EXECUTION_FAILURE


if __name__ == "__main__":
    sys.exit(main())
