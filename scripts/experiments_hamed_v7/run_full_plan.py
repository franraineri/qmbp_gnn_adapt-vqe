"""
Master Runner — Hamed V7 Full Experiment Plan.

Orchestrates all 22 sub-experiments across 5 execution phases (A→E),
with CLI control over which phases/techniques to run.

Usage:
    # Run all Phase A experiments
    python scripts/experiments_hamed_v7/run_full_plan.py --phase A

    # Run only Technique 3 (MPS) experiments across all phases
    python scripts/experiments_hamed_v7/run_full_plan.py --phase all --technique 3

    # Dry run to see what would execute
    python scripts/experiments_hamed_v7/run_full_plan.py --phase all --dry-run

    # Force re-run of completed experiments
    python scripts/experiments_hamed_v7/run_full_plan.py --phase A --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# ── Phase Plan ───────────────────────────────────────────────────────────────

PHASE_PLAN: dict[str, list[dict]] = {
    "A": [  # Quick wins (~20 min)
        {
            "experiment_id": "4A",
            "technique": 4,
            "script": "experiment_spsa_hardware.py",
            "args": ["--sub-experiment", "4A"],
            "timeout": 10,
            "description": "SPSA hyperparameter grid search (N=6, h=1.5)",
        },
        {
            "experiment_id": "4B",
            "technique": 4,
            "script": "experiment_spsa_hardware.py",
            "args": ["--sub-experiment", "4B"],
            "timeout": 10,
            "description": "SPSA with MPNN warm-start",
        },
        {
            "experiment_id": "1A",
            "technique": 1,
            "script": "experiment_nevergrad.py",
            "args": ["--sub-experiment", "1A"],
            "timeout": 15,
            "description": "Nevergrad fair budget comparison (N=6)",
        },
        {
            "experiment_id": "3A",
            "technique": 3,
            "script": "experiment_mps_simulation.py",
            "args": ["--sub-experiment", "3A"],
            "timeout": 10,
            "description": "MPS accuracy validation (N=6)",
        },
    ],
    "B": [  # Scaling tests (~60 min)
        {
            "experiment_id": "3B",
            "technique": 3,
            "script": "experiment_mps_simulation.py",
            "args": ["--sub-experiment", "3B"],
            "timeout": 15,
            "description": "MPS accuracy validation (N=10)",
        },
        {
            "experiment_id": "3C",
            "technique": 3,
            "script": "experiment_mps_simulation.py",
            "args": ["--sub-experiment", "3C"],
            "timeout": 20,
            "description": "MPS-only VQE (N=20)",
        },
        {
            "experiment_id": "1C",
            "technique": 1,
            "script": "experiment_nevergrad.py",
            "args": ["--sub-experiment", "1C"],
            "timeout": 20,
            "description": "Nevergrad scaling test (N=10)",
        },
        {
            "experiment_id": "2B",
            "technique": 2,
            "script": "experiment_qrc_warmstart.py",
            "args": ["--sub-experiment", "2B"],
            "timeout": 25,
            "description": "QRC vs MPNN comparison (N=10)",
        },
    ],
    "C": [  # Noise-aware (~60 min)
        {
            "experiment_id": "5A",
            "technique": 5,
            "script": "experiment_noise_aware_training.py",
            "args": ["--sub-experiment", "5A", "--use-fake-torino"],
            "timeout": 25,
            "description": "FakeTorino VQE data generation (N=6)",
        },
        {
            "experiment_id": "4C",
            "technique": 4,
            "script": "experiment_spsa_hardware.py",
            "args": ["--sub-experiment", "4C", "--use-fake-torino"],
            "timeout": 15,
            "description": "FakeTorino SPSA vs COBYLA (N=6)",
        },
        {
            "experiment_id": "5B",
            "technique": 5,
            "script": "experiment_noise_aware_training.py",
            "args": ["--sub-experiment", "5B", "--use-fake-torino"],
            "timeout": 20,
            "description": "Noise-aware vs noiseless MPNN training",
        },
        {
            "experiment_id": "4D",
            "technique": 4,
            "script": "experiment_spsa_hardware.py",
            "args": ["--sub-experiment", "4D", "--use-fake-torino"],
            "timeout": 20,
            "description": "FakeTorino SPSA (N=10)",
        },
    ],
    "D": [  # Advanced / novel (~90 min)
        {
            "experiment_id": "2A",
            "technique": 2,
            "script": "experiment_qrc_warmstart.py",
            "args": ["--sub-experiment", "2A"],
            "timeout": 20,
            "description": "4 reservoir designs comparison (N=6)",
        },
        {
            "experiment_id": "2D",
            "technique": 2,
            "script": "experiment_qrc_warmstart.py",
            "args": ["--sub-experiment", "2D"],
            "timeout": 25,
            "description": "Hybrid QRC+MPNN concatenated features",
        },
        {
            "experiment_id": "5C",
            "technique": 5,
            "script": "experiment_noise_aware_training.py",
            "args": ["--sub-experiment", "5C", "--use-fake-torino"],
            "timeout": 20,
            "description": "Mixed training (noiseless + noisy)",
        },
        {
            "experiment_id": "4E",
            "technique": 4,
            "script": "experiment_spsa_hardware.py",
            "args": ["--sub-experiment", "4E", "--use-fake-torino", "--use-zne"],
            "timeout": 15,
            "description": "SPSA + ZNE integration (3 layouts)",
        },
        {
            "experiment_id": "5E",
            "technique": 5,
            "script": "experiment_noise_aware_training.py",
            "args": ["--sub-experiment", "5E", "--use-fake-torino"],
            "timeout": 35,
            "description": "Iterative refinement (3 rounds)",
        },
    ],
    "E": [  # Stretch goals (~105 min)
        {
            "experiment_id": "3D",
            "technique": 3,
            "script": "experiment_mps_simulation.py",
            "args": ["--sub-experiment", "3D"],
            "timeout": 35,
            "description": "MPS VQE at N=30 (stretch)",
        },
        {
            "experiment_id": "3E",
            "technique": 3,
            "script": "experiment_mps_simulation.py",
            "args": ["--sub-experiment", "3E"],
            "timeout": 25,
            "description": "Critical region stress test (N=20)",
        },
        {
            "experiment_id": "1B",
            "technique": 1,
            "script": "experiment_nevergrad.py",
            "args": ["--sub-experiment", "1B"],
            "timeout": 15,
            "description": "Nevergrad warm-start scenario",
        },
        {
            "experiment_id": "5D",
            "technique": 5,
            "script": "experiment_noise_aware_training.py",
            "args": ["--sub-experiment", "5D", "--use-fake-torino", "--N", "10"],
            "timeout": 35,
            "description": "Noise-aware training at N=10",
        },
        {
            "experiment_id": "2C",
            "technique": 2,
            "script": "experiment_qrc_warmstart.py",
            "args": ["--sub-experiment", "2C"],
            "timeout": 20,
            "description": "QRC data efficiency (vary training size)",
        },
    ],
}

TECHNIQUE_NAMES = {
    1: "Nevergrad (gradient-free VQE)",
    2: "QRC warm-start",
    3: "MPS simulation",
    4: "SPSA hardware optimizer",
    5: "Noise-aware training",
}


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the master runner."""
    parser = argparse.ArgumentParser(
        description="Master runner for Hamed V7 full experiment plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase order (recommended):
  A: Quick wins (4A, 4B, 1A, 3A) — ~20 min
  B: Scaling tests (3B, 3C, 1C, 2B) — ~60 min
  C: Noise-aware (5A, 4C, 5B, 4D) — ~60 min
  D: Advanced (2A, 2D, 5C, 4E, 5E) — ~90 min
  E: Stretch goals (3D, 3E, 1B, 5D, 2C) — ~105 min
""",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["A", "B", "C", "D", "E", "all"],
        help="Execution phase to run (default: all)",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default="all",
        help="Technique number to filter (1-5 or 'all')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without running experiments",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override per-experiment timeout in minutes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of already-completed experiments",
    )
    return parser.parse_args()


# ── Execution Logic ──────────────────────────────────────────────────────────


def find_existing_result(experiment_id: str, results_dir: Path = RESULTS_DIR) -> Path | None:
    """Check if a result file already exists for the given experiment ID.

    Searches for any JSON file in results_dir containing the experiment_id
    in its filename. Also checks inside JSON content for experiment_id field
    to handle legacy naming conventions.

    Returns
    -------
    Path | None
        Path to existing result file, or None if not found.
    """
    if not results_dir.exists():
        return None

    for f in results_dir.glob("*.json"):
        if f.name == "summary_full_plan.json":
            continue
        # Match files like "nevergrad_1A_20260520_143000.json"
        # or "spsa_4A_20260520_143000.json"
        if f"_{experiment_id}_" in f.name or f.name.startswith(f"{experiment_id}_"):
            return f
        # Also check inside JSON for experiment_id field (handles new format)
        try:
            with open(f) as fp:
                data = json.load(fp)
            if data.get("experiment_id") == experiment_id:
                return f
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    return None


def run_sub_experiment(
    script: str,
    args: list[str],
    timeout_min: int = 60,
) -> tuple[bool, str]:
    """Execute a sub-experiment script as subprocess with timeout protection.

    Parameters
    ----------
    script : str
        Script filename (relative to experiments directory).
    args : list[str]
        CLI arguments to pass to the script.
    timeout_min : int
        Timeout in minutes.

    Returns
    -------
    tuple[bool, str]
        (success, output) where output is stdout on success or stderr on failure.
    """
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    cmd = [sys.executable, str(script_path)] + args
    timeout_s = timeout_min * 60

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout_min} minutes"
    except Exception as e:
        return False, f"Execution error: {e}"


def run_phase(
    phase_id: str,
    techniques: list[int] | None = None,
    dry_run: bool = False,
    timeout_override: int | None = None,
    force: bool = False,
) -> list[dict]:
    """Run all experiments in a phase, skipping already-completed ones.

    Parameters
    ----------
    phase_id : str
        Phase identifier (A-E).
    techniques : list[int] | None
        Filter by technique numbers. None means all.
    dry_run : bool
        If True, print plan without executing.
    timeout_override : int | None
        Override per-experiment timeout (minutes).
    force : bool
        If True, re-run even if results exist.

    Returns
    -------
    list[dict]
        List of result dicts with keys: experiment_id, success, output, wall_time_s.
    """
    experiments = PHASE_PLAN.get(phase_id, [])
    results = []

    print(f"\n{'=' * 60}")
    print(f"  Phase {phase_id}: {len(experiments)} experiments")
    print(f"{'=' * 60}")

    for exp in experiments:
        exp_id = exp["experiment_id"]
        technique = exp["technique"]
        description = exp.get("description", "")

        # Filter by technique
        if techniques is not None and technique not in techniques:
            continue

        # Check if already completed
        if not force:
            existing = find_existing_result(exp_id)
            if existing is not None:
                print(f"  [{exp_id}] SKIP (result exists: {existing.name})")
                results.append(
                    {
                        "experiment_id": exp_id,
                        "technique": technique,
                        "status": "skipped",
                        "reason": f"Result exists: {existing.name}",
                    }
                )
                continue

        # Determine timeout
        timeout = timeout_override if timeout_override is not None else exp.get("timeout", 60)

        if dry_run:
            print(f"  [{exp_id}] WOULD RUN: {exp['script']} {' '.join(exp['args'])}")
            print(f"           Technique {technique}: {description}")
            print(f"           Timeout: {timeout} min")
            results.append(
                {
                    "experiment_id": exp_id,
                    "technique": technique,
                    "status": "dry_run",
                }
            )
            continue

        # Execute
        print(f"  [{exp_id}] RUNNING: {description}")
        print(f"           Script: {exp['script']} {' '.join(exp['args'])}")
        print(f"           Timeout: {timeout} min")

        start_time = time.time()
        success, output = run_sub_experiment(
            script=exp["script"],
            args=exp["args"],
            timeout_min=timeout,
        )
        wall_time = time.time() - start_time

        status = "success" if success else "failed"
        print(f"           Result: {status} ({wall_time:.1f}s)")
        if not success:
            # Print first few lines of error
            error_lines = output.strip().split("\n")[:5]
            for line in error_lines:
                print(f"           ERROR: {line}")

        results.append(
            {
                "experiment_id": exp_id,
                "technique": technique,
                "status": status,
                "wall_time_s": wall_time,
                "output": output[:500] if output else "",
            }
        )

    return results


def generate_summary(results_dir: Path = RESULTS_DIR) -> dict:
    """Aggregate all results into a single summary JSON.

    Reads all JSON result files in the results directory and produces
    a consolidated summary with per-experiment status and key metrics.

    Parameters
    ----------
    results_dir : Path
        Directory containing individual result JSON files.

    Returns
    -------
    dict
        Consolidated summary.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {"experiments": [], "timestamp": datetime.now().isoformat()}

    all_results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary_full_plan.json":
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            all_results.append(
                {
                    "file": f.name,
                    "experiment_id": data.get("experiment_id", "unknown"),
                    "technique": data.get("technique", 0),
                    "success": data.get("success", False),
                    "summary": data.get("summary", {}),
                }
            )
        except (json.JSONDecodeError, KeyError) as e:
            all_results.append(
                {
                    "file": f.name,
                    "experiment_id": "parse_error",
                    "error": str(e),
                }
            )

    # Compute overall statistics
    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r.get("success", False))
    n_failed = n_total - n_success

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": n_total,
        "successful": n_success,
        "failed": n_failed,
        "experiments": all_results,
        "by_technique": {},
    }

    # Group by technique
    for tech_id, tech_name in TECHNIQUE_NAMES.items():
        tech_results = [r for r in all_results if r.get("technique") == tech_id]
        summary["by_technique"][str(tech_id)] = {
            "name": tech_name,
            "total": len(tech_results),
            "successful": sum(1 for r in tech_results if r.get("success", False)),
        }

    # Save summary
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary_full_plan.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Total: {n_total} | Success: {n_success} | Failed: {n_failed}")

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    """Main entry point for the master runner."""
    args = parse_args()

    # Determine phases to run
    phases = ["A", "B", "C", "D", "E"] if args.phase == "all" else [args.phase]

    # Determine technique filter
    if args.technique == "all":
        techniques = None  # No filter
    else:
        try:
            techniques = [int(args.technique)]
        except ValueError:
            print(f"Error: --technique must be 1-5 or 'all', got '{args.technique}'")
            sys.exit(1)

    print("=" * 60)
    print("  Hamed V7 Full Experiment Plan — Master Runner")
    print("=" * 60)
    print(f"  Phases: {', '.join(phases)}")
    print(f"  Techniques: {'all' if techniques is None else techniques}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Force: {args.force}")
    print(f"  Timeout override: {args.timeout or 'per-experiment default'}")
    print(f"  Results dir: {RESULTS_DIR}")

    # Execute phases
    all_results = []
    total_start = time.time()

    for phase_id in phases:
        phase_results = run_phase(
            phase_id=phase_id,
            techniques=techniques,
            dry_run=args.dry_run,
            timeout_override=args.timeout,
            force=args.force,
        )
        all_results.extend(phase_results)

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}")
    print("  Execution Complete")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    if not args.dry_run:
        n_run = sum(1 for r in all_results if r.get("status") == "success")
        n_failed = sum(1 for r in all_results if r.get("status") == "failed")
        n_skipped = sum(1 for r in all_results if r.get("status") == "skipped")
        print(f"  Run: {n_run} | Failed: {n_failed} | Skipped: {n_skipped}")

        # Generate consolidated summary
        generate_summary(RESULTS_DIR)


if __name__ == "__main__":
    main()
