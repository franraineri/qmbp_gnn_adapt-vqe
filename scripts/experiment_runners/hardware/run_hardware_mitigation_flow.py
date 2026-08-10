#!/usr/bin/env python3
"""Hardware Mitigation Flow — Orchestrated 3-step deployment.

Executes the validated hardware deployment flow:
  Step 1: Local rehearsal (MPNN quality check, 0 QPU cost)
  Step 2: Smoke test (C0_raw at h=4.0 only, validates infrastructure)
  Step 3: Full benchmark (C0,C1,C5,C3,C16 × 4 h-points)

Each step gates the next: if a step fails, execution aborts with
a clear diagnostic message. No QPU time is wasted on misconfigured runs.

Usage:
    # Full automated flow (recommended):
    python scripts/experiment_runners/hardware/run_hardware_mitigation_flow.py

    # Skip rehearsal (if already passed recently):
    python scripts/experiment_runners/hardware/run_hardware_mitigation_flow.py --skip-rehearsal

    # Dry-run (print what would execute, no QPU):
    python scripts/experiment_runners/hardware/run_hardware_mitigation_flow.py --dry-run

    # Custom parameters:
    python scripts/experiment_runners/hardware/run_hardware_mitigation_flow.py \
        --h-values 3.5,3.75,4.0 --shots 8192 --seed 43 --backend ibm_boston

    # Multi-seed (Tier 2):
    python scripts/experiment_runners/hardware/run_hardware_mitigation_flow.py \
        --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
BENCHMARK_SCRIPT = ROOT / "scripts/experiment_runners/hardware/run_mitigation_benchmark.py"
REHEARSAL_SCRIPT = ROOT / "scripts/experiment_runners/run_hardware_rehearsal_v3.py"
RESULTS_BASE = ROOT / "results/mitigation_benchmark"
FLOW_LOG = RESULTS_BASE / "flow_log.json"

# Smoke test thresholds
SMOKE_DE_GAP_MAX = 1.00  # C0_raw at h=4.0 should be ~40%, abort if >100%
SMOKE_DE_GAP_EXPECTED = 0.43  # Expected from FakeTorino simulation

# Rehearsal thresholds (from validated results)
REHEARSAL_SPEEDUP_MIN = 1.5
REHEARSAL_LOO_PASS_MIN = 0.80
REHEARSAL_INIT_DE_GAP_MAX = 0.05

# Hardware configs for each step
SMOKE_CONFIGS = "C0"
FULL_CONFIGS = "C0,C1,C5,C3,C16"

DEFAULT_H_VALUES = "4.0,3.75,3.5,3.25"
DEFAULT_SHOTS = 16384
DEFAULT_SEED = 42
DEFAULT_BACKEND = "ibm_kingston"


# ─── Utilities ────────────────────────────────────────────────────────────


def log_step(step_name: str, status: str, details: dict | None = None) -> None:
    """Append a step result to the flow log for post-hoc analysis."""
    FLOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "step": step_name,
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "details": details or {},
    }
    entries = []
    if FLOW_LOG.exists():
        try:
            entries = json.loads(FLOW_LOG.read_text())
        except (json.JSONDecodeError, ValueError):
            entries = []
    entries.append(entry)
    FLOW_LOG.write_text(json.dumps(entries, indent=2, default=str))
    print(f"  [{status}] {step_name}" + (f" — {details}" if details else ""))


def run_command(cmd: list[str], timeout_s: int = 3600, dry_run: bool = False) -> tuple[int, str]:
    """Execute a subprocess command with timeout. Returns (returncode, stdout)."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] Would execute: {cmd_str}")
        return 0, ""
    print(f"  [EXEC] {cmd_str}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            # Print last 30 lines of stderr for debugging
            stderr_lines = result.stderr.strip().split("\n")
            print("  [STDERR] (last 30 lines):")
            for line in stderr_lines[-30:]:
                print(f"    {line}")
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return -2, str(e)


def check_credentials() -> tuple[bool, str]:
    """Verify IBM credentials are set in environment."""
    ibm_key = os.environ.get("IBM_KEY")
    ibm_crn = os.environ.get("IBM_INSTANCE_CRN")
    if not ibm_key:
        return False, "IBM_KEY not set. Export with: export IBM_KEY=<your_token>"
    if not ibm_crn:
        return False, "IBM_INSTANCE_CRN not set. Export with: export IBM_INSTANCE_CRN=<your_crn>"
    return True, f"IBM_KEY={'*' * 8}...{ibm_key[-4:]}, CRN set"


def check_dependencies() -> tuple[bool, str]:
    """Verify critical Python packages are importable."""
    missing = []
    for pkg in ["qiskit", "qiskit_ibm_runtime", "qiskit_aer", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All dependencies available"


def parse_smoke_results(seed: int) -> tuple[bool, float | None, str]:
    """Parse the smoke test result (C0_raw at h=4.0) from the manifest."""
    manifest_path = RESULTS_BASE / "manifest.json"
    if not manifest_path.exists():
        return False, None, "No manifest.json found after smoke test"
    try:
        entries = json.loads(manifest_path.read_text())
        # Find C0_raw at h=4.0 with matching seed
        smoke_entries = [
            e
            for e in entries
            if e.get("config_id") == "C0_raw"
            and abs(e.get("h_value", 0) - 4.0) < 0.01
            and e.get("seed", 42) == seed
        ]
        if not smoke_entries:
            return False, None, "C0_raw h=4.0 entry not found in manifest"
        latest = smoke_entries[-1]
        de_gap = latest.get("delta_e_gap")
        if de_gap is None:
            return False, None, "delta_e_gap is None in smoke test result"
        return True, de_gap, f"C0_raw h=4.0: ΔE/gap={de_gap * 100:.1f}%"
    except Exception as e:
        return False, None, f"Failed to parse manifest: {e}"


# ─── Step Implementations ─────────────────────────────────────────────────


def step_0_preflight(args: argparse.Namespace) -> bool:
    """Step 0: Environment and dependency checks (no QPU, instant)."""
    print("\n" + "=" * 70)
    print("  STEP 0: PREFLIGHT CHECKS")
    print("=" * 70)

    # Check dependencies
    ok, msg = check_dependencies()
    if not ok:
        log_step("preflight_deps", "FAIL", {"error": msg})
        return False
    print(f"  ✓ Dependencies: {msg}")

    # Check credentials (only for hardware mode)
    if not args.dry_run:
        ok, msg = check_credentials()
        if not ok:
            log_step("preflight_creds", "FAIL", {"error": msg})
            return False
        print(f"  ✓ Credentials: {msg}")

    # Check scripts exist
    if not BENCHMARK_SCRIPT.exists():
        log_step("preflight_scripts", "FAIL", {"error": f"{BENCHMARK_SCRIPT} not found"})
        return False
    if not args.skip_rehearsal and not REHEARSAL_SCRIPT.exists():
        log_step("preflight_scripts", "FAIL", {"error": f"{REHEARSAL_SCRIPT} not found"})
        return False
    print("  ✓ Scripts: benchmark + rehearsal found")

    # Check disk space (results dir)
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Results dir: {RESULTS_BASE}")

    # Quality prediction check — warn before spending QPU time on ABORT configs
    try:
        from qmbp_simulation.analysis.quality_predictor import QualityPredictor

        predictor = QualityPredictor()
        topology = getattr(args, "topology", "heavy_hex")
        n_qubits = getattr(args, "n_qubits", 10)
        p_layers = getattr(args, "p_layers", 2)
        model = getattr(args, "model", "tfim")
        report = predictor.predict(
            model=model, topology=topology,
            n_qubits=n_qubits, p_layers=p_layers,
        )
        if not report.should_run:
            print(
                f"  ⚠️  QUALITY WARNING: Historical pass rate = "
                f"{report.pass_probability:.0%} (ABORT recommended). "
                f"This config may waste QPU time."
            )
            print(f"      Reasons: {'; '.join(report.reasons)}")
            if not args.dry_run:
                print("      Proceeding anyway (use --dry-run to preview).")
        else:
            print(
                f"  ✓ Quality prediction: {report.recommendation} "
                f"({report.pass_probability:.0%})"
            )
    except (ImportError, Exception):
        pass  # Non-blocking

    log_step("preflight", "PASS")
    return True


def step_1_rehearsal(args: argparse.Namespace) -> bool:
    """Step 1: Local MPNN rehearsal (0 QPU cost, ~5 min)."""
    print("\n" + "=" * 70)
    print("  STEP 1: LOCAL REHEARSAL (MPNN quality check)")
    print("  Expected: speedup≥1.5x, LOO≥80%, init ΔE/gap<5%")
    print("=" * 70)

    if args.skip_rehearsal:
        print("  [SKIP] --skip-rehearsal flag set")
        log_step("rehearsal", "SKIPPED")
        return True

    cmd = [
        PYTHON,
        str(REHEARSAL_SCRIPT),
        "--skip-hardware-sections",
        "--section",
        "10",
        "--section",
        "11",
        "--n-qubits",
        "10",
        "--topology",
        "heavy_hex",
        "--p-layers",
        "1",
        "--h-train",
        "4.5",
        "4.25",
        "4.0",
        "3.75",
        "3.5",
        "3.25",
        "3.0",
        "--h-test",
        "4.0",
        "3.25",
        "--mpnn-epochs",
        "3000",
        "--vqe-restarts",
        "1",
    ]

    returncode, stdout = run_command(cmd, timeout_s=600, dry_run=args.dry_run)
    if args.dry_run:
        log_step("rehearsal", "DRY-RUN")
        return True

    if returncode != 0:
        log_step("rehearsal", "FAIL", {"returncode": returncode})
        print("\n  ✗ Rehearsal FAILED. Fix MPNN quality before proceeding.")
        print("    Common fixes:")
        print("    - Increase --mpnn-epochs to 6000")
        print("    - Add more h-train points near boundary")
        print("    - Check VQE convergence at h_test values")
        return False

    log_step("rehearsal", "PASS", {"note": "MPNN quality validated"})
    print("  ✓ Rehearsal PASSED — MPNN ready for hardware")
    return True


def step_2_smoke_test(args: argparse.Namespace) -> bool:
    """Step 2: Single-point smoke test (C0_raw, h=4.0, ~3 min QPU)."""
    print("\n" + "=" * 70)
    print("  STEP 2: SMOKE TEST (C0_raw, h=4.0)")
    print(
        f"  Expected: ΔE/gap ≈ {SMOKE_DE_GAP_EXPECTED * 100:.0f}%, abort if >{SMOKE_DE_GAP_MAX * 100:.0f}%"
    )
    print("=" * 70)

    seed = args.seeds[0] if args.seeds else DEFAULT_SEED
    cmd = [
        PYTHON,
        str(BENCHMARK_SCRIPT),
        "--mode",
        "hardware",
        "--configs",
        SMOKE_CONFIGS,
        "--h-values",
        "4.0",
        "--shots",
        str(args.shots),
        "--seed",
        str(seed),
        "--backend",
        args.backend,
    ]

    returncode, stdout = run_command(cmd, timeout_s=600, dry_run=args.dry_run)
    if args.dry_run:
        log_step("smoke_test", "DRY-RUN")
        return True

    if returncode != 0:
        log_step("smoke_test", "FAIL", {"returncode": returncode, "reason": "script crashed"})
        print("\n  ✗ Smoke test CRASHED. Debug with --mode fake_backend first.")
        return False

    # Parse result
    ok, de_gap, msg = parse_smoke_results(seed)
    if not ok:
        log_step("smoke_test", "FAIL", {"reason": msg})
        print(f"\n  ✗ {msg}")
        return False

    print(f"  Result: {msg}")

    if de_gap > SMOKE_DE_GAP_MAX:
        log_step(
            "smoke_test",
            "FAIL",
            {
                "delta_e_gap": de_gap,
                "threshold": SMOKE_DE_GAP_MAX,
                "reason": "ΔE/gap exceeds 100% — hardware or pipeline broken",
            },
        )
        print(f"\n  ✗ ΔE/gap={de_gap * 100:.1f}% > {SMOKE_DE_GAP_MAX * 100:.0f}% — ABORTING")
        print("    Possible causes:")
        print("    - Backend calibration degraded (check mean_2q_error)")
        print("    - Transpilation produced wrong circuit (check n_2q=18)")
        print("    - IBM Runtime API changed (check qiskit-ibm-runtime version)")
        return False

    # Warn if significantly different from simulation prediction
    deviation = abs(de_gap - SMOKE_DE_GAP_EXPECTED) / SMOKE_DE_GAP_EXPECTED
    if deviation > 0.50:
        print(
            f"  ⚠ ΔE/gap={de_gap * 100:.1f}% deviates {deviation * 100:.0f}% from simulation "
            f"prediction ({SMOKE_DE_GAP_EXPECTED * 100:.0f}%). PEA may perform differently."
        )

    log_step("smoke_test", "PASS", {"delta_e_gap": de_gap})
    print(f"  ✓ Smoke test PASSED — infrastructure works, noise floor={de_gap * 100:.1f}%")
    return True


def step_3_full_benchmark(args: argparse.Namespace) -> bool:
    """Step 3: Full mitigation benchmark (5 configs × 4 h-points per seed)."""
    print("\n" + "=" * 70)
    print("  STEP 3: FULL MITIGATION BENCHMARK")
    print(f"  Configs: {FULL_CONFIGS}")
    print(f"  h-values: {args.h_values}")
    print(f"  Shots: {args.shots}")
    print(f"  Seeds: {args.seeds}")
    print("  Estimated QPU time: ~25 min per seed")
    print("=" * 70)

    seeds = args.seeds if args.seeds else [DEFAULT_SEED]
    all_passed = True

    for i, seed in enumerate(seeds):
        print(f"\n  --- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        cmd = [
            PYTHON,
            str(BENCHMARK_SCRIPT),
            "--mode",
            "hardware",
            "--configs",
            FULL_CONFIGS,
            "--h-values",
            args.h_values,
            "--shots",
            str(args.shots),
            "--seed",
            str(seed),
            "--backend",
            args.backend,
        ]

        returncode, stdout = run_command(cmd, timeout_s=3600, dry_run=args.dry_run)
        if args.dry_run:
            continue

        if returncode != 0:
            log_step(f"benchmark_seed_{seed}", "FAIL", {"returncode": returncode})
            print(f"\n  ✗ Benchmark FAILED for seed {seed}")
            all_passed = False
            # Continue with next seed (don't abort entire run for one seed failure)
            continue

        log_step(f"benchmark_seed_{seed}", "PASS")
        print(f"  ✓ Seed {seed} completed")

    if args.dry_run:
        log_step("full_benchmark", "DRY-RUN")
        return True

    if all_passed:
        log_step("full_benchmark", "PASS", {"seeds": seeds, "configs": FULL_CONFIGS})
        return True
    else:
        log_step("full_benchmark", "PARTIAL", {"seeds": seeds})
        print("\n  ⚠ Some seeds failed — partial results available")
        return True  # Don't block analysis for partial failures


# ─── Summary & Analysis ───────────────────────────────────────────────────


def step_4_analysis(args: argparse.Namespace) -> bool:
    """Step 4: Run analyzer on collected results (local, instant)."""
    print("\n" + "=" * 70)
    print("  STEP 4: ANALYSIS")
    print("=" * 70)

    cmd = [
        PYTHON,
        "-m",
        "project_health.analysis.mitigation_benchmark_analyzer",
        "--thesis-table",
    ]

    returncode, stdout = run_command(cmd, timeout_s=120, dry_run=args.dry_run)
    if returncode != 0 and not args.dry_run:
        log_step("analysis", "FAIL", {"returncode": returncode})
        print("  ⚠ Analyzer failed — results still available in manifest.json")
        return True  # Non-blocking

    log_step("analysis", "PASS")
    print("  ✓ Analysis complete — check results/mitigation_benchmark/analysis/")
    return True


def print_summary(start_time: float, args: argparse.Namespace) -> None:
    """Print final execution summary."""
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Total time:  {elapsed / 60:.1f} min")
    print(f"  Mode:        {'DRY-RUN' if args.dry_run else 'HARDWARE'}")
    print(f"  Configs:     {FULL_CONFIGS}")
    print(f"  h-values:    {args.h_values}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Flow log:    {FLOW_LOG}")
    print(f"  Results:     {RESULTS_BASE}/")
    print(f"  Manifest:    {RESULTS_BASE}/manifest.json")
    print("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hardware Mitigation Flow — orchestrated 3-step deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full automated flow:
    python run_hardware_mitigation_flow.py

    # Skip rehearsal (already passed):
    python run_hardware_mitigation_flow.py --skip-rehearsal

    # Dry-run (no QPU):
    python run_hardware_mitigation_flow.py --dry-run

    # Custom h-values and multi-seed:
    python run_hardware_mitigation_flow.py --h-values 3.5,3.75,4.0 --seeds 42,43,44

    # Different backend:
    python run_hardware_mitigation_flow.py --backend ibm_boston
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing (no QPU cost)",
    )
    parser.add_argument(
        "--skip-rehearsal",
        action="store_true",
        help="Skip Step 1 (use if rehearsal already passed recently)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip Step 2 smoke test (use if C0_raw already validated on this session)",
    )
    parser.add_argument(
        "--h-values",
        type=str,
        default=DEFAULT_H_VALUES,
        help=f"CSV of h-values for the benchmark (default: {DEFAULT_H_VALUES})",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots per circuit (default: {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(DEFAULT_SEED),
        help="CSV of seeds (default: 42). Use '42,43,44' for Tier 2.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        help=f"IBM backend name (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--smoke-threshold",
        type=float,
        default=SMOKE_DE_GAP_MAX,
        help=f"Max ΔE/gap for smoke test pass (default: {SMOKE_DE_GAP_MAX})",
    )

    args = parser.parse_args()
    # Parse seeds from CSV
    args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    return args


def main() -> int:
    args = parse_args()
    start_time = time.time()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  HARDWARE MITIGATION FLOW — Automated Deployment Pipeline         ║")
    print("║  System: N=10, p=1, heavy_hex, TFIM                               ║")
    print(f"║  Backend: {args.backend:<57s}║")
    print("╚" + "═" * 68 + "╝")

    # Step 0: Preflight
    if not step_0_preflight(args):
        print("\n  ✗ ABORTED at Step 0 (preflight). Fix issues above.")
        return 1

    # Step 1: Rehearsal
    if not step_1_rehearsal(args):
        print("\n  ✗ ABORTED at Step 1 (rehearsal). MPNN not ready.")
        return 1

    # Step 2: Smoke test
    if args.skip_smoke:
        print("\n  [SKIP] Step 2 (smoke test) — --skip-smoke flag set")
        log_step("smoke_test", "SKIPPED")
    elif not step_2_smoke_test(args):
        print("\n  ✗ ABORTED at Step 2 (smoke test). Infrastructure issue.")
        return 1

    # Step 3: Full benchmark
    if not step_3_full_benchmark(args):
        print("\n  ✗ ABORTED at Step 3 (benchmark). See errors above.")
        return 1

    # Step 4: Analysis
    step_4_analysis(args)

    # Summary
    print_summary(start_time, args)
    print("\n  ✓ FLOW COMPLETE — thesis data ready for Table 5.23\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
