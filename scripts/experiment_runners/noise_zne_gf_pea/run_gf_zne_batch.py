#!/usr/bin/env python3
"""Batch runner: Gate-Folding ZNE comparison across topologies.

Runs run_gf_zne_comparison.py for multiple (topology, N, p) configs
and collects results for cross-topology comparison via project_health.

Configurations tested:
  1. chain_1d  N=6  p=1  — baseline (CES spread exists, both methods viable)
  2. heavy_hex N=10 p=1  — critical case (CES uniform, GF-ZNE should win)
  3. ladder    N=6  p=1  — intermediate (moderate CES spread)

After running, use:
    python project_health/compare.py --exp GF_ZNE_CMP
    python -m project_health --json | python -c "import json,sys; d=json.load(sys.stdin); ..."

Usage:
    python scripts/experiment_runners/run_gf_zne_batch.py
    python scripts/experiment_runners/run_gf_zne_batch.py --configs chain_1d heavy_hex
    python scripts/experiment_runners/run_gf_zne_batch.py --dry-run
    python scripts/experiment_runners/run_gf_zne_batch.py --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent.parent
COMPARISON_SCRIPT = ROOT / "scripts" / "experiment_runners" / "run_gf_zne_comparison.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
RESULTS_DIR = ROOT / "results" / "experiments" / "exp_gf_zne_cmp"


@dataclass
class RunConfig:
    """Single experiment configuration."""

    topology: str
    n_qubits: int
    p_layers: int = 1
    extrapolator: str = "linear"
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.topology}_N{self.n_qubits}_p{self.p_layers}"


# Default batch: the three critical topology comparisons
DEFAULT_CONFIGS = [
    RunConfig("chain_1d", 6, 1, label="chain_N6_p1"),
    RunConfig("heavy_hex", 10, 1, label="heavy_hex_N10_p1"),
    RunConfig("ladder", 6, 1, label="ladder_N6_p1"),
]

# Reduced set for quick testing
QUICK_CONFIGS = [
    RunConfig("chain_1d", 6, 1, label="chain_N6_p1"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_config(config: RunConfig, dry_run: bool = False) -> dict:
    """Run a single GF-ZNE comparison experiment.

    Returns
    -------
    dict
        Result summary: label, topology, exit_code, elapsed, result_file.
    """
    cmd = [
        str(PYTHON),
        str(COMPARISON_SCRIPT),
        "--topology",
        config.topology,
        "--n-qubits",
        str(config.n_qubits),
        "--p-layers",
        str(config.p_layers),
        "--extrapolator",
        config.extrapolator,
        "--skip-preflight",
    ]

    logger.info(f"\n{'=' * 65}")
    logger.info(f"  Running: {config.label}")
    logger.info(f"  Command: {' '.join(cmd[1:])}")
    logger.info(f"{'=' * 65}")

    if dry_run:
        return {"label": config.label, "status": "dry_run", "elapsed_s": 0}

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=600,  # 10 min max per config
    )
    elapsed = time.time() - t0

    # Extract result file from output (last line with "Results:")
    result_file = None
    for line in result.stdout.split("\n"):
        if "Results:" in line:
            path_str = line.split("Results:")[-1].strip()
            result_file = path_str

    status = "pass" if result.returncode == 0 else "fail"
    logger.info(f"  Status: {status} ({elapsed:.1f}s)")
    if result.returncode != 0:
        # Show last 10 lines of stderr for diagnosis
        stderr_lines = result.stderr.strip().split("\n")[-10:]
        for line in stderr_lines:
            logger.error(f"    {line}")

    return {
        "label": config.label,
        "topology": config.topology,
        "n_qubits": config.n_qubits,
        "p_layers": config.p_layers,
        "status": status,
        "exit_code": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "result_file": result_file,
    }


def run_comparison_analysis() -> None:
    """Run project_health comparison on GF_ZNE_CMP results.

    Uses the project_health/compare.py tool to analyze results across runs.
    Also shows how to load results programmatically for custom analysis.
    """
    logger.info("\n" + "=" * 65)
    logger.info("  POST-RUN ANALYSIS")
    logger.info("=" * 65)

    # 1. Use compare.py CLI (experiment-level verdict)
    logger.info("\n  --- Experiment Verdict (via compare.py) ---")
    cmd = [str(PYTHON), str(ROOT / "project_health" / "compare.py"), "--exp", "GF_ZNE_CMP"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode == 0 and result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            logger.info(f"  {line}")
    else:
        logger.info("  (No results found via compare.py — first run)")

    # 2. Direct JSON analysis of latest results
    logger.info("\n  --- Cross-Topology Summary (direct JSON parse) ---")
    if not RESULTS_DIR.exists():
        logger.info("  No results directory found yet.")
        return

    result_files = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    if not result_files:
        logger.info("  No result files found.")
        return

    summaries = []
    for rf in result_files[:10]:  # Last 10 runs
        try:
            data = json.loads(rf.read_text())
            section_4 = data.get("results", {}).get("section_4", {})
            s4_data = section_4.get("data", {})
            summary = s4_data.get("summary", {})
            if summary:
                summaries.append(
                    {
                        "file": rf.name,
                        "topology": summary.get("topology", "?"),
                        "n_qubits": summary.get("n_qubits", "?"),
                        "gf_wins": summary.get("gf_wins", 0),
                        "n_points": summary.get("n_points", 0),
                        "mean_ces_gain": summary.get("mean_ces_zne_gain", 0),
                        "mean_gf_gain": summary.get("mean_gf_zne_gain", 0),
                        "mean_gf_r2": summary.get("mean_gf_zne_r2", 0),
                    }
                )
        except (json.JSONDecodeError, KeyError):
            continue

    if not summaries:
        logger.info("  Could not parse results.")
        return

    # Print comparison table
    logger.info(
        f"  {'Topology':<12} {'N':>3} {'GF wins':>8} "
        f"{'CES gain':>9} {'GF gain':>8} {'GF R²':>6} | File"
    )
    logger.info("  " + "-" * 70)
    for s in summaries:
        logger.info(
            f"  {s['topology']:<12} {s['n_qubits']:>3} "
            f"{s['gf_wins']}/{s['n_points']:>6} "
            f"{s['mean_ces_gain']:>+8.1%} {s['mean_gf_gain']:>+7.1%} "
            f"{s['mean_gf_r2']:>5.3f} | {s['file']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch GF-ZNE comparison across topologies",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["chain_1d", "heavy_hex", "ladder", "triangular"],
        default=None,
        help="Run only specific topologies (default: all three)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only chain_1d N=6 (fastest, ~40s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Only run post-analysis (no new experiments)",
    )
    parser.add_argument(
        "--extrapolator",
        default="linear",
        choices=["linear", "exponential"],
        help="Extrapolation method for GF-ZNE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Compare-only mode
    if args.compare:
        run_comparison_analysis()
        return

    # Select configs
    if args.quick:
        configs = QUICK_CONFIGS
    elif args.configs:
        configs = [RunConfig(t, 10 if t == "heavy_hex" else 6, 1) for t in args.configs]
    else:
        configs = DEFAULT_CONFIGS

    # Apply extrapolator override
    for c in configs:
        c.extrapolator = args.extrapolator

    # Dry-run
    if args.dry_run:
        logger.info("DRY RUN — would execute:")
        for c in configs:
            logger.info(f"  {c.label}: {c.topology} N={c.n_qubits} p={c.p_layers}")
        return

    # Execute batch
    logger.info(f"\nBatch GF-ZNE Comparison: {len(configs)} configurations")
    logger.info(f"Extrapolator: {args.extrapolator}")
    logger.info("")

    t_total = time.time()
    results = []
    for config in configs:
        try:
            result = run_single_config(config)
            results.append(result)
        except subprocess.TimeoutExpired:
            logger.error(f"  TIMEOUT: {config.label} exceeded 10 min limit")
            results.append({"label": config.label, "status": "timeout", "elapsed_s": 600})
        except Exception as e:
            logger.error(f"  ERROR: {config.label}: {e}")
            results.append({"label": config.label, "status": "error", "elapsed_s": 0})

    total_elapsed = time.time() - t_total

    # Final summary
    logger.info(f"\n{'=' * 65}")
    logger.info("  BATCH SUMMARY")
    logger.info(f"{'=' * 65}")
    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_fail = sum(1 for r in results if r.get("status") != "pass")
    logger.info(f"  Total: {len(results)} configs, {n_pass} passed, {n_fail} failed")
    logger.info(f"  Time: {total_elapsed:.1f}s")
    logger.info("")
    for r in results:
        status_icon = "✅" if r.get("status") == "pass" else "❌"
        logger.info(
            f"  {status_icon} {r['label']:<25} {r.get('elapsed_s', 0):.1f}s  "
            f"{r.get('result_file', '')}"
        )

    # Run comparison analysis
    logger.info("")
    run_comparison_analysis()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
