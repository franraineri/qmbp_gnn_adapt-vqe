#!/usr/bin/env python3
"""Full experiment orchestration for cross-topology transfer pipeline.

Sequences all experiments with timeout handling and caching:
    1. Data generation (heavy_hex N=6, N=16) — skip if existing
    2. Within-topology cross-N validation (triangular, heavy_hex)
    3. Cross-topology transfer (tri→hex, hex→tri)
    4. Ablation (GNN vs MLP vs Scipy + norm_type comparison)

Each step has a 30-min timeout. Total budget: 180 min.
Produces a summary JSON with all results, timings, and verdict table.

Usage:
    python scripts/experiment_runners/cross_topology/run_orchestrator.py \\
        --output-dir results/scaling/cross_topology \\
        --seeds 42,43,44 \\
        --timeout 1800

    # Skip ablation for faster iteration:
    python scripts/experiment_runners/cross_topology/run_orchestrator.py \\
        --skip-ablation --timeout 1800

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts" / "experiment_runners") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "experiment_runners"))

from cross_topology.helpers import (
    build_experiment_envelope,
)
from qmbp_simulation.utils.helpers import json_dump

# Script paths for subprocess execution
_SCRIPTS_DIR = _ROOT / "scripts" / "experiment_runners" / "cross_topology"

logger = logging.getLogger(__name__)

# Total budget in seconds (180 min)
TOTAL_BUDGET_S = 180 * 60
# Warning threshold: 90% of budget consumed
BUDGET_WARNING_THRESHOLD = 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# Step execution with timeout
# ═══════════════════════════════════════════════════════════════════════════════


def run_step(
    step_name: str,
    command: list[str],
    timeout_s: int,
    cwd: Path | None = None,
) -> dict:
    """Execute a single experiment step as a subprocess with timeout.

    Parameters
    ----------
    step_name : str
        Human-readable name of the step (for logging).
    command : list[str]
        Command to execute (e.g. [sys.executable, "script.py", "--arg"]).
    timeout_s : int
        Maximum seconds to allow for this step.
    cwd : Path | None
        Working directory for the subprocess.

    Returns
    -------
    dict
        Step result with status, timing, returncode, and any error info.
    """
    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Step: {step_name}")
    logger.info(f"  Command: {' '.join(command)}")
    logger.info(f"  Timeout: {timeout_s}s ({timeout_s / 60:.0f}m)")
    logger.info(f"{'─' * 60}")

    t_start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
        elapsed = time.perf_counter() - t_start

        if result.returncode == 0:
            logger.info(f"  ✅ {step_name} completed in {elapsed:.1f}s")
            status = "completed"
        else:
            logger.warning(f"  ⚠️ {step_name} returned code {result.returncode} in {elapsed:.1f}s")
            status = "error"
            if result.stderr:
                # Log last few lines of stderr
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-5:]:
                    logger.warning(f"    stderr: {line}")

        return {
            "step": step_name,
            "status": status,
            "returncode": result.returncode,
            "time_s": elapsed,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t_start
        logger.warning(
            f"  ⚠️ {step_name} TIMED OUT after {elapsed:.1f}s "
            f"(limit: {timeout_s}s). Continuing to next step."
        )
        return {
            "step": step_name,
            "status": "timeout",
            "returncode": -1,
            "time_s": elapsed,
            "stdout_tail": "",
            "stderr_tail": f"TimeoutExpired after {timeout_s}s",
        }

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        logger.error(f"  ❌ {step_name} failed with exception: {e}")
        return {
            "step": step_name,
            "status": "exception",
            "returncode": -1,
            "time_s": elapsed,
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Data file existence check
# ═══════════════════════════════════════════════════════════════════════════════


def check_data_exists(
    data_dir: Path,
    topology: str,
    n: int,
    p: int = 1,
) -> bool:
    """Check if VQE data already exists for a given topology/size/p combo.

    Returns True if at least one matching file is found.
    """
    pattern = f"vqe_{topology}_N{n}_p{p}_*.json"
    matches = list(data_dir.glob(pattern))
    return len(matches) > 0


def find_result_files(output_dir: Path, prefix: str) -> list[Path]:
    """Find result files matching a prefix in the output directory."""
    pattern = f"{prefix}*.json"
    return sorted(output_dir.glob(pattern))


# ═══════════════════════════════════════════════════════════════════════════════
# Budget monitoring
# ═══════════════════════════════════════════════════════════════════════════════


def check_budget(
    t_start: float,
    total_budget_s: int = TOTAL_BUDGET_S,
) -> tuple[float, float, bool]:
    """Check elapsed time against total budget.

    Returns
    -------
    tuple of (elapsed_s, remaining_s, over_budget)
    """
    elapsed = time.perf_counter() - t_start
    remaining = total_budget_s - elapsed
    over_budget = elapsed >= total_budget_s

    if elapsed >= total_budget_s * BUDGET_WARNING_THRESHOLD:
        pct = elapsed / total_budget_s * 100
        logger.warning(
            f"  ⚠️ Budget warning: {pct:.0f}% consumed "
            f"({elapsed / 60:.1f}m / {total_budget_s / 60:.0f}m)"
        )

    return elapsed, remaining, over_budget


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator core
# ═══════════════════════════════════════════════════════════════════════════════


def run_orchestrator(
    data_dir: Path = Path("results/scaling"),
    output_dir: Path = Path("results/scaling/cross_topology"),
    seeds: tuple[int, ...] = (42, 43, 44),
    timeout_per_experiment: int = 1800,
    skip_existing: bool = True,
    skip_ablation: bool = False,
) -> dict:
    """Execute all experiments in sequence with timeout and caching.

    Steps:
    1. Check for existing heavy_hex N=6, N=16 data → generate if missing
    2. Run within-topology cross-N validation (triangular, heavy_hex)
    3. Run cross-topology transfer (tri→hex, hex→tri)
    4. Run ablation (GNN vs MLP vs Scipy + norm_type comparison)
    5. Produce summary JSON with all results and verdict table

    Parameters
    ----------
    data_dir : Path
        Base directory for data discovery.
    output_dir : Path
        Output directory for results.
    seeds : tuple[int, ...]
        Random seeds for reproducibility.
    timeout_per_experiment : int
        Maximum seconds per experiment step (default 1800 = 30 min).
    skip_existing : bool
        Skip data generation if valid result files exist.
    skip_ablation : bool
        Skip the ablation step (for faster iteration).

    Returns
    -------
    dict
        Summary dictionary with all results, timings, and verdict table.
    """
    t_orchestrator_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds_str = ",".join(str(s) for s in seeds)
    python = sys.executable

    step_results: list[dict] = []

    # ── Step 1: Data Generation ───────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  STEP 1: VQE Data Generation")
    logger.info("═" * 60)

    data_gen_needed = False
    topologies_to_gen: list[tuple[str, int]] = []

    # Check for heavy_hex N=6 and N=16
    for n_size in [6, 16]:
        if skip_existing and check_data_exists(output_dir, "heavy_hex", n_size):
            logger.info(f"  SKIP: heavy_hex N={n_size} — data exists")
        else:
            topologies_to_gen.append(("heavy_hex", n_size))
            data_gen_needed = True

    # Also check triangular N=6 and N=16
    for n_size in [6, 16]:
        if skip_existing and check_data_exists(output_dir, "triangular", n_size):
            logger.info(f"  SKIP: triangular N={n_size} — data exists")
        else:
            topologies_to_gen.append(("triangular", n_size))
            data_gen_needed = True

    if data_gen_needed:
        for topology, n_size in topologies_to_gen:
            cmd = [
                python,
                str(_SCRIPTS_DIR / "run_vqe_data_gen.py"),
                "--topology",
                topology,
                "--n",
                str(n_size),
                "--p",
                "1",
                "--seeds",
                seeds_str,
                "--output-dir",
                str(output_dir),
            ]
            if skip_existing:
                cmd.append("--skip-existing")

            result = run_step(
                step_name=f"data_gen_{topology}_N{n_size}",
                command=cmd,
                timeout_s=timeout_per_experiment,
            )
            step_results.append(result)

            # Check budget
            _, _, over = check_budget(t_orchestrator_start)
            if over:
                logger.warning("  ⚠️ Total budget exceeded — aborting remaining steps")
                break
    else:
        logger.info("  All data files exist — skipping data generation")
        step_results.append(
            {
                "step": "data_gen",
                "status": "skipped",
                "returncode": 0,
                "time_s": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )

    # Budget check after step 1
    _, remaining, over_budget = check_budget(t_orchestrator_start)
    if over_budget:
        logger.warning("  Total budget exceeded after data generation")

    # ── Step 2: Within-Topology Cross-N Validation ────────────────────
    if not over_budget:
        logger.info("\n" + "═" * 60)
        logger.info("  STEP 2: Within-Topology Cross-N Validation")
        logger.info("═" * 60)

        cmd = [
            python,
            str(_SCRIPTS_DIR / "run_cross_n_validation.py"),
            "--topology",
            "triangular",
            "heavy_hex",
            "--train-sizes",
            "6",
            "16",
            "--target-n",
            "10",
            "--norm-type",
            "none",
            "--seeds",
            seeds_str,
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(output_dir),
        ]
        result = run_step(
            step_name="cross_n_validation",
            command=cmd,
            timeout_s=timeout_per_experiment,
        )
        step_results.append(result)

        _, remaining, over_budget = check_budget(t_orchestrator_start)

    # ── Step 3: Cross-Topology Transfer ───────────────────────────────
    if not over_budget:
        logger.info("\n" + "═" * 60)
        logger.info("  STEP 3: Cross-Topology Transfer")
        logger.info("═" * 60)

        cmd = [
            python,
            str(_SCRIPTS_DIR / "run_cross_topology.py"),
            "--source-topologies",
            "triangular",
            "heavy_hex",
            "--target-n",
            "10",
            "--seeds",
            seeds_str,
            "--output-dir",
            str(output_dir),
        ]
        result = run_step(
            step_name="cross_topology_transfer",
            command=cmd,
            timeout_s=timeout_per_experiment,
        )
        step_results.append(result)

        _, remaining, over_budget = check_budget(t_orchestrator_start)

    # ── Step 4: Ablation ──────────────────────────────────────────────
    if not over_budget and not skip_ablation:
        logger.info("\n" + "═" * 60)
        logger.info("  STEP 4: Ablation (GNN vs MLP vs Scipy + norm_type)")
        logger.info("═" * 60)

        cmd = [
            python,
            str(_SCRIPTS_DIR / "run_ablation.py"),
            "--topologies",
            "triangular",
            "heavy_hex",
            "--target-n",
            "10",
            "--seeds",
            seeds_str,
            "--output-dir",
            str(output_dir),
        ]
        result = run_step(
            step_name="ablation",
            command=cmd,
            timeout_s=timeout_per_experiment,
        )
        step_results.append(result)
    elif skip_ablation:
        logger.info("\n  SKIP: Ablation (--skip-ablation flag)")
        step_results.append(
            {
                "step": "ablation",
                "status": "skipped",
                "returncode": 0,
                "time_s": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )

    # ── Summary ───────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_orchestrator_start

    logger.info("\n" + "═" * 60)
    logger.info("  ORCHESTRATOR SUMMARY")
    logger.info("═" * 60)

    # Build verdict table from step results
    verdict_table: list[dict] = []
    for step in step_results:
        verdict_table.append(
            {
                "step": step["step"],
                "status": step["status"],
                "time_s": step["time_s"],
            }
        )
        logger.info(f"  {step['step']:<30} {step['status']:<12} {step['time_s']:>7.1f}s")

    # Determine overall verdict
    non_skipped = [s for s in step_results if s["status"] != "skipped"]
    all_completed = all(s["status"] == "completed" for s in non_skipped)
    any_timeout = any(s["status"] == "timeout" for s in step_results)
    any_error = any(s["status"] == "error" for s in step_results)

    if all_completed and not any_timeout:
        overall_verdict = "PASS"
    elif any_timeout and not any_error:
        overall_verdict = "PARTIAL (timeouts)"
    elif any_error:
        overall_verdict = "PARTIAL (errors)"
    else:
        overall_verdict = "INCOMPLETE"

    logger.info(f"\n  Overall verdict: {overall_verdict}")
    logger.info(f"  Total time: {t_total:.1f}s ({t_total / 60:.1f}m)")
    if t_total > TOTAL_BUDGET_S:
        logger.warning(f"  ⚠️ Exceeded 180-min budget: {t_total / 60:.1f}m")

    # ── Build summary JSON ────────────────────────────────────────────
    # Collect source files from output_dir
    source_files = [str(p) for p in sorted(output_dir.glob("*.json"))]

    summary = build_experiment_envelope(
        experiment_name="cross_topology_orchestrator",
        source_files=source_files,
        seeds=list(seeds),
        total_time_s=t_total,
        timeout_per_experiment_s=timeout_per_experiment,
        total_budget_s=TOTAL_BUDGET_S,
        skip_existing=skip_existing,
        skip_ablation=skip_ablation,
    )

    summary["steps"] = step_results
    summary["verdict_table"] = verdict_table
    summary["verdict"] = {
        "overall": overall_verdict,
        "all_completed": all_completed,
        "any_timeout": any_timeout,
        "any_error": any_error,
        "total_time_s": t_total,
        "within_budget": t_total <= TOTAL_BUDGET_S,
    }

    # Try to load individual result files to aggregate verdicts
    cross_n_results = find_result_files(output_dir, "cross_n_validation_")
    cross_topo_results = find_result_files(output_dir, "cross_topology_transfer_")
    ablation_results = find_result_files(output_dir, "ablation_")

    experiment_verdicts = {}
    if cross_n_results:
        import json

        try:
            with open(cross_n_results[-1]) as f:
                cn_data = json.load(f)
            experiment_verdicts["within_topology_pass"] = cn_data.get("summary", {}).get(
                "all_topologies_pass", False
            )
        except (json.JSONDecodeError, KeyError):
            experiment_verdicts["within_topology_pass"] = None

    if cross_topo_results:
        import json

        try:
            with open(cross_topo_results[-1]) as f:
                ct_data = json.load(f)
            experiment_verdicts["cross_topology_pass"] = ct_data.get("summary", {}).get(
                "overall_pass", False
            )
        except (json.JSONDecodeError, KeyError):
            experiment_verdicts["cross_topology_pass"] = None

    if ablation_results:
        import json

        try:
            with open(ablation_results[-1]) as f:
                ab_data = json.load(f)
            experiment_verdicts["graph_structure_essential"] = ab_data.get("summary", {}).get(
                "graph_structure_essential", False
            )
        except (json.JSONDecodeError, KeyError):
            experiment_verdicts["graph_structure_essential"] = None

    summary["experiment_verdicts"] = experiment_verdicts

    # Save summary with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"orchestrator_summary_{timestamp}.json"
    json_dump(summary, out_path)
    logger.info(f"\n  Summary saved: {out_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full experiment orchestration for cross-topology transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/cross_topology",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/scaling",
        help="Base data directory for source file discovery",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per experiment step in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip data generation when valid result files exist",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force regeneration even if data files exist",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation step for faster iteration",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse seeds
    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    skip_existing = not args.no_skip_existing

    # Log all parameters at INFO level (Requirement 8.4)
    logger.info("═" * 60)
    logger.info("  Cross-Topology Transfer — Full Orchestration")
    logger.info("═" * 60)
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Timeout per step: {args.timeout}s ({args.timeout / 60:.0f}m)")
    logger.info(f"  Total budget: {TOTAL_BUDGET_S}s ({TOTAL_BUDGET_S / 60:.0f}m)")
    logger.info(f"  Skip existing: {skip_existing}")
    logger.info(f"  Skip ablation: {args.skip_ablation}")
    logger.info(f"  Python: {sys.executable}")
    logger.info("═" * 60)

    summary = run_orchestrator(
        data_dir=data_dir,
        output_dir=output_dir,
        seeds=seeds,
        timeout_per_experiment=args.timeout,
        skip_existing=skip_existing,
        skip_ablation=args.skip_ablation,
    )

    # Return 0 if all steps completed successfully
    verdict = summary.get("verdict", {})
    if verdict.get("all_completed", False):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
