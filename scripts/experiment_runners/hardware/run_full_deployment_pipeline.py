#!/usr/bin/env python3
"""Full deployment pipeline: V3 rehearsal → IBM Torino deployment.

Orchestrates the two-step flow:
  1. Run V3 rehearsal with --use-flow-warmstart to generate σ_flow data
  2. Find the most recent rehearsal result JSON
  3. Pass it to run_ibm_deployment.py --sigma-flow-results <path>

Usage:
    python scripts/experiment_runners/hardware/run_full_deployment_pipeline.py
    python scripts/experiment_runners/hardware/run_full_deployment_pipeline.py --skip-rehearsal
    python scripts/experiment_runners/hardware/run_full_deployment_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REHEARSAL_DIR = Path("results/experiments/exp_hw_rehearsal_v3")
SCRIPT_DIR = Path(__file__).parent


def find_latest_rehearsal_json() -> Path | None:
    """Find the most recent run_*.json in the V3 rehearsal results."""
    if not REHEARSAL_DIR.exists():
        return None
    candidates = sorted(REHEARSAL_DIR.glob("run_*.json"), reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Full rehearsal → deployment pipeline")
    parser.add_argument(
        "--skip-rehearsal",
        action="store_true",
        help="Skip V3 rehearsal (use existing results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to deployment (no QPU usage)",
    )
    parser.add_argument(
        "--sigma-flow-threshold",
        type=float,
        default=0.5,
        help="σ_flow threshold for boost (default: 0.5)",
    )
    args = parser.parse_args()

    python = sys.executable

    # ── Step 1: Run V3 rehearsal (unless skipped) ─────────────────────────
    if not args.skip_rehearsal:
        print("\n═══ Step 1: Running V3 rehearsal with flow warmstart ═══\n")
        rehearsal_script = "scripts/experiment_runners/run_hardware_rehearsal_v3.py"
        cmd = [python, rehearsal_script, "--use-flow-warmstart"]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\n❌ Rehearsal failed (exit {result.returncode}). Aborting.")
            sys.exit(1)
        print("\n✅ Rehearsal complete.\n")
    else:
        print("\n⏭️  Skipping rehearsal (--skip-rehearsal)\n")

    # ── Step 2: Find latest rehearsal JSON ────────────────────────────────
    latest = find_latest_rehearsal_json()
    if latest is None:
        print(f"❌ No run_*.json found in {REHEARSAL_DIR}/")
        print("   Run without --skip-rehearsal first.")
        sys.exit(1)
    print(f"📄 Using rehearsal result: {latest.name}")

    # ── Step 3: Launch deployment ─────────────────────────────────────────
    print("\n═══ Step 3: Launching IBM Torino deployment ═══\n")
    deploy_script = str(SCRIPT_DIR / "run_ibm_deployment.py")
    cmd = [
        python,
        deploy_script,
        "--sigma-flow-results",
        str(latest),
        "--sigma-flow-threshold",
        str(args.sigma_flow_threshold),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
