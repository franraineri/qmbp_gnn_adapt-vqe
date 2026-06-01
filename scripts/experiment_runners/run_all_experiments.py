#!/usr/bin/env python3
"""Run ALL registered BaseExperiment scripts sequentially.

Executes every experiment in the EXPERIMENT_REGISTRY from run_experiment.py,
tracking pass/fail status, timing, and producing a summary log.

Usage:
    # Run all experiments with default configs
    python scripts/run_all_experiments.py

    # Dry run (list what would be executed)
    python scripts/run_all_experiments.py --dry-run

    # Run only specific categories
    python scripts/run_all_experiments.py --categories B F G

    # Run only specific experiments
    python scripts/run_all_experiments.py --exp A3 B1 F3

    # Exclude slow experiments
    python scripts/run_all_experiments.py --exclude G3 A3

    # Skip experiments that already have results
    python scripts/run_all_experiments.py --skip-existing

    # Override system size for all experiments
    python scripts/run_all_experiments.py --n-qubits 6

    # Verbose output
    python scripts/run_all_experiments.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# Experiment registry (mirrors scripts/run_experiment.py)
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENT_REGISTRY: dict[str, tuple[str, str]] = {
    "A3": ("experiments.scaling.exp_a3_scaling_law", "ExperimentA3"),
    "B1": ("experiments.optimization.exp_b1_analytical", "ExperimentB1"),
    "B2": ("experiments.optimization.exp_b2_freezing", "ExperimentB2"),
    "B4": ("experiments.optimization.exp_b4_hessian", "ExperimentB4"),
    "C1": ("experiments.predictor.exp_c1_physics_loss", "ExperimentC1"),
    "C3": ("experiments.optimization.exp_c3_sign", "ExperimentC3"),
    "D1": ("experiments.predictor.exp_d1_weight_space", "ExperimentD1"),
    "E3": ("experiments.predictor.exp_e3_active", "ExperimentE3"),
    "E4": ("experiments.generalization.exp_e4_longitudinal", "ExperimentE4"),
    "F1": ("experiments.landscape.exp_f1_dypp", "ExperimentF1"),
    "F3": ("experiments.landscape.exp_f3_fluctuation", "ExperimentF3"),
    "G1": ("experiments.predictor.exp_g1_data_efficiency", "ExperimentG1"),
    "G2": ("experiments.predictor.exp_g2_ensemble_calibration", "ExperimentG2"),
    "G3": ("experiments.scaling.exp_g3_n20_optimized", "ExperimentG3"),
    "G4": ("experiments.optimization.exp_g4_condition_restarts", "ExperimentG4"),
    "G5": ("experiments.predictor.exp_g5_cross_seed", "ExperimentG5"),
    "S1": ("experiments.scaling.exp_s1_entanglement_scaling", "ExperimentS1"),
    "S2": ("experiments.predictor.exp_s2_cross_topology", "ExperimentS2"),
    "S3": ("experiments.landscape.exp_s3_landscape_n20", "ExperimentS3"),
    "S4": ("experiments.predictor.exp_s4_data_efficiency_n10", "ExperimentS4"),
    "S5": ("experiments.scaling.exp_s5_n20_p1_pipeline", "ExperimentS5"),
    "S6": ("experiments.predictor.exp_s6_mc_dropout_uq", "ExperimentS6"),
    "S8": ("experiments.scaling.exp_s8_d1_finite_size_scaling", "ExperimentS8"),
    "S8b": ("experiments.scaling.exp_s8b_mpnn_finite_size_scaling", "ExperimentS8b"),
}

CATEGORY_NAMES = {
    "A": "Ground Truth Enhancement",
    "B": "VQE Optimization Enhancement",
    "C": "MPNN Predictor Enhancement",
    "D": "Landscape & Phase Transition Analysis",
    "E": "Scaling & Generalization",
    "F": "Novel Methodological Contributions",
    "G": "Pipeline Characterization & Validation",
}

# Base directory where BaseExperiment saves results
_RESULTS_BASE = Path("results/experiments")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""

    exp_id: str
    description: str
    success: bool
    elapsed_s: float
    error_msg: str = ""
    summary: dict | None = None

    @property
    def pass_rate(self) -> float | None:
        """Extract pass_rate from summary if available."""
        if self.summary and "pass_rate" in self.summary:
            return self.summary["pass_rate"]
        return None

    @property
    def mean_de_gap(self) -> float | None:
        """Extract mean ΔE/gap from summary if available."""
        if self.summary and "mean_de_gap" in self.summary:
            return self.summary["mean_de_gap"]
        return None

    @property
    def verdict(self) -> str:
        """Human-readable verdict based on experiment outcome.

        Uses pass_rate when available, but also checks for experiment-specific
        success indicators in the summary (e.g., 'hypothesis_confirmed').
        """
        if not self.success:
            return "ERROR"
        if self.summary is None:
            return "OK"
        # Check for explicit hypothesis confirmation (set by some experiments)
        if self.summary.get("hypothesis_confirmed") is True:
            return "CONFIRMED"
        if self.summary.get("hypothesis_confirmed") is False:
            return "REJECTED"
        # Fall back to pass_rate
        pr = self.pass_rate
        if pr is None:
            return "OK"
        if pr >= 0.9:
            return "CONFIRMED"
        if pr >= 0.5:
            return "PARTIAL"
        return "REJECTED"


# ═══════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════


def load_experiment_class(exp_id: str):
    """Load experiment class by ID from the registry.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the class doesn't exist in the module.
    """
    module_path, class_name = EXPERIMENT_REGISTRY[exp_id]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_experiment_description(exp_id: str) -> str:
    """Get the description from an experiment's default_config.

    Returns the experiment ID as fallback if loading fails.
    """
    try:
        cls = load_experiment_class(exp_id)
        if hasattr(cls, "default_config"):
            cfg = cls.default_config()
            return cfg.description or exp_id
    except Exception:
        pass
    return exp_id


def has_existing_results(exp_id: str) -> bool:
    """Check if an experiment already has saved results.

    Looks in results/experiments/exp_<id>/ for run_*.json files,
    matching the BaseExperiment save convention.
    """
    results_dir = _RESULTS_BASE / f"exp_{exp_id.lower()}"
    if not results_dir.exists():
        return False
    return any(results_dir.glob("run_*.json"))


def validate_imports(exp_ids: list[str]) -> list[str]:
    """Pre-validate that all experiment modules can be imported.

    Returns list of experiment IDs that failed to import.
    Prints warnings for failures but does not abort.
    """
    failures = []
    for exp_id in exp_ids:
        try:
            load_experiment_class(exp_id)
        except Exception as e:
            print(f"  ⚠️  {exp_id}: import failed — {e}")
            failures.append(exp_id)
    return failures


def _reset_logging() -> None:
    """Reset root logger handlers and level between experiments.

    Each BaseExperiment.setup() calls logging.basicConfig() which
    can accumulate handlers. This prevents duplicate log lines and
    ensures one experiment's DEBUG level doesn't leak to the next.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)


def _sanitize_summary(summary: Any) -> dict | None:
    """Ensure summary dict is JSON-serializable (handle numpy types)."""
    if summary is None:
        return None
    if not isinstance(summary, dict):
        return None
    try:
        # Round-trip through JSON to catch non-serializable types
        json.dumps(summary)
        return summary
    except (TypeError, ValueError):
        # Fallback: convert numpy types manually
        import numpy as np

        def _convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list | tuple):
                return [_convert(v) for v in obj]
            return obj

        try:
            return _convert(summary)
        except Exception:
            return {"error": "summary not serializable"}


def run_single_experiment(
    exp_id: str,
    n_qubits: int | None = None,
    p_layers: int | None = None,
    topology: str | None = None,
    seeds: list[int] | None = None,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a single experiment and return the result.

    Parameters
    ----------
    exp_id : str
        Experiment ID (e.g., "A3", "B1").
    n_qubits : int | None
        Override number of qubits.
    p_layers : int | None
        Override HVA layers.
    topology : str | None
        Override lattice topology.
    seeds : list[int] | None
        Override seeds.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    ExperimentResult
        Execution result with timing and status.
    """
    description = get_experiment_description(exp_id)

    # Reset logging to avoid handler accumulation
    _reset_logging()

    t0 = time.time()
    try:
        cls = load_experiment_class(exp_id)

        # Get default config and apply overrides
        config = cls.default_config()
        if n_qubits is not None:
            config.system.n_qubits = n_qubits
        if p_layers is not None:
            config.system.p_layers = p_layers
        if topology is not None:
            config.system.topology = topology
        if seeds is not None:
            config.seeds = seeds
        if verbose:
            config.verbose = True

        # Execute the experiment lifecycle
        experiment = cls(config)
        analysis = experiment.execute()
        elapsed = time.time() - t0

        # Extract summary if available
        summary = analysis.get("summary", {}) if isinstance(analysis, dict) else {}

        return ExperimentResult(
            exp_id=exp_id,
            description=description,
            success=True,
            elapsed_s=elapsed,
            summary=_sanitize_summary(summary),
        )

    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        # Keep last 5 lines of traceback for context
        error_lines = tb.strip().split("\n")[-5:]
        error_msg = "\n".join(error_lines)

        return ExperimentResult(
            exp_id=exp_id,
            description=description,
            success=False,
            elapsed_s=elapsed,
            error_msg=error_msg,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CLI and main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALL registered BaseExperiment scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Run all 16 experiments
    %(prog)s --dry-run                    # List what would run
    %(prog)s --categories B F             # Only optimization + landscape
    %(prog)s --exp A3 B1 F3              # Only specific experiments
    %(prog)s --exclude G3 A3             # Skip slow experiments
    %(prog)s --skip-existing              # Skip experiments with results
    %(prog)s --n-qubits 6 --verbose      # Override system size

Available experiments: """
        + ", ".join(sorted(EXPERIMENT_REGISTRY.keys())),
    )

    # Selection
    selection = parser.add_argument_group("experiment selection")
    selection.add_argument(
        "--exp",
        nargs="+",
        type=str,
        default=None,
        help="Run only specific experiment IDs",
    )
    selection.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=None,
        help="Run only experiments in these categories (A, B, C, D, E, F, G)",
    )
    selection.add_argument(
        "--exclude",
        nargs="+",
        type=str,
        default=None,
        help="Exclude specific experiment IDs (useful to skip slow ones like G3, A3)",
    )
    selection.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results in results/experiments/exp_<id>/",
    )

    # Overrides
    overrides = parser.add_argument_group("config overrides (applied to all experiments)")
    overrides.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Override N for all experiments",
    )
    overrides.add_argument(
        "--p",
        type=int,
        default=None,
        choices=[1, 2],
        help="Override p layers (must be ≤ 2)",
    )
    overrides.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=["chain_1d", "ladder", "triangular", "kagome"],
        help="Override topology for all experiments",
    )
    overrides.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds (e.g., --seeds 42 43)",
    )
    overrides.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging for each experiment",
    )

    # Execution control
    control = parser.add_argument_group("execution control")
    control.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments without running them",
    )
    control.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop execution on first failure",
    )
    control.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate imports (check all experiments can be loaded)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which experiments to run
    if args.exp:
        exp_ids = [e.upper() for e in args.exp]
        # Validate IDs
        for eid in exp_ids:
            if eid not in EXPERIMENT_REGISTRY:
                print(
                    f"ERROR: Unknown experiment '{eid}'. "
                    f"Available: {sorted(EXPERIMENT_REGISTRY.keys())}"
                )
                sys.exit(1)
    elif args.categories:
        cats = [c.upper() for c in args.categories]
        invalid_cats = [c for c in cats if c not in CATEGORY_NAMES]
        if invalid_cats:
            print(
                f"ERROR: Unknown categories: {invalid_cats}. "
                f"Available: {sorted(CATEGORY_NAMES.keys())}"
            )
            sys.exit(1)
        exp_ids = [eid for eid in sorted(EXPERIMENT_REGISTRY.keys()) if eid[0] in cats]
    else:
        exp_ids = sorted(EXPERIMENT_REGISTRY.keys())

    # Apply exclusions
    if args.exclude:
        excludes = {e.upper() for e in args.exclude}
        exp_ids = [eid for eid in exp_ids if eid not in excludes]

    # Skip existing if requested
    if args.skip_existing:
        before = len(exp_ids)
        exp_ids = [eid for eid in exp_ids if not has_existing_results(eid)]
        skipped = before - len(exp_ids)
        if skipped > 0:
            print(f"  Skipping {skipped} experiment(s) with existing results.\n")

    if not exp_ids:
        print("  No experiments to run.")
        return

    # Validate-only mode
    if args.validate_only:
        print(f"  Validating imports for {len(exp_ids)} experiments...")
        failures = validate_imports(exp_ids)
        if failures:
            print(f"\n  ❌ {len(failures)} experiment(s) failed to import: {failures}")
            sys.exit(1)
        else:
            print(f"  ✅ All {len(exp_ids)} experiments import successfully.")
        return

    # Header
    print("=" * 70)
    print("  RUN ALL BASEEXPERIMENT SCRIPTS")
    print("=" * 70)
    print(f"\n  Experiments:  {len(exp_ids)}")
    print(f"  Mode:         {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    if args.n_qubits:
        print(f"  N override:   {args.n_qubits}")
    if args.p:
        print(f"  p override:   {args.p}")
    if args.topology:
        print(f"  Topology:     {args.topology}")
    if args.seeds:
        print(f"  Seeds:        {args.seeds}")
    if args.exclude:
        print(f"  Excluded:     {', '.join(e.upper() for e in args.exclude)}")
    print()

    # Group by category for display
    by_cat: dict[str, list[str]] = {}
    for eid in exp_ids:
        cat = eid[0]
        by_cat.setdefault(cat, []).append(eid)

    for cat in sorted(by_cat.keys()):
        cat_name = CATEGORY_NAMES.get(cat, "Unknown")
        exps = by_cat[cat]
        print(f"  [{cat}] {cat_name}: {', '.join(exps)}")
    print()

    # Pre-validate imports before starting execution
    print("  Validating imports...", end=" ")
    failures = validate_imports(exp_ids)
    if failures:
        print(f"\n  ⚠️  {len(failures)} experiment(s) cannot be imported and will be skipped.")
        exp_ids = [eid for eid in exp_ids if eid not in failures]
        if not exp_ids:
            print("  No valid experiments remaining.")
            sys.exit(1)
    else:
        print("✅")
    print()

    # Dry run mode
    if args.dry_run:
        print("  Experiments that would be executed:")
        print(f"  {'ID':<6} {'Cat':<4} {'Description'}")
        print(f"  {'-' * 64}")
        for eid in exp_ids:
            desc = get_experiment_description(eid)
            cat = eid[0]
            existing = "📁" if has_existing_results(eid) else "  "
            print(f"  {eid:<6} [{cat}]  {existing} {desc[:55]}")
        print(f"\n  Total: {len(exp_ids)} experiments")
        print("  📁 = has existing results (use --skip-existing to skip)")
        return

    # Execute
    t_total = time.time()
    results: list[ExperimentResult] = []

    for i, exp_id in enumerate(exp_ids, 1):
        desc = get_experiment_description(exp_id)
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(exp_ids)}] {exp_id}: {desc}")
        print(f"{'─' * 70}")

        result = run_single_experiment(
            exp_id,
            n_qubits=args.n_qubits,
            p_layers=args.p,
            topology=args.topology,
            seeds=args.seeds,
            verbose=args.verbose,
        )
        results.append(result)

        status = "✅" if result.success else "❌"
        print(f"\n  {status} {exp_id}: {result.elapsed_s:.1f}s", end="")
        if result.success and result.summary:
            # Show key metrics inline
            pr = result.pass_rate
            de = result.mean_de_gap
            parts = []
            if pr is not None:
                parts.append(f"pass_rate={pr:.0%}")
            if de is not None:
                parts.append(f"mean_ΔE/gap={de:.4f}")
            if parts:
                print(f"  ({', '.join(parts)})", end="")
            print(f"  [{result.verdict}]")
        elif result.success:
            print(f"  [{result.verdict}]")
        else:
            print()
            print(f"     Error: {result.error_msg[:200]}")

        if not result.success and args.stop_on_failure:
            print("\n  ⛔ Stopping on first failure (--stop-on-failure)")
            break

    # Final summary
    total_elapsed = time.time() - t_total
    n_pass = sum(1 for r in results if r.success)
    n_fail = sum(1 for r in results if not r.success)
    n_confirmed = sum(1 for r in results if r.verdict == "CONFIRMED")
    n_rejected = sum(1 for r in results if r.verdict == "REJECTED")

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Total experiments: {len(results)}")
    print(f"  Executed OK:       {n_pass} ✅")
    print(f"  Execution errors:  {n_fail} ❌")
    print(f"  Total time:        {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print()
    print("  Verdicts:")
    print(f"    CONFIRMED (pass_rate ≥ 90%): {n_confirmed}")
    print(f"    REJECTED  (pass_rate < 50%): {n_rejected}")
    print(f"    PARTIAL/OK:                  {n_pass - n_confirmed - n_rejected}")
    print()

    # Per-experiment summary table with metrics
    print(f"  {'ID':<6} {'Verdict':<12} {'Time':<8} {'ΔE/gap':<10} {'Pass%':<8} {'Description'}")
    print(f"  {'-' * 68}")
    for r in results:
        verdict = r.verdict
        time_str = f"{r.elapsed_s:.1f}s"
        de_str = f"{r.mean_de_gap:.4f}" if r.mean_de_gap is not None else "—"
        pr_str = f"{r.pass_rate:.0%}" if r.pass_rate is not None else "—"
        desc = r.description[:30]
        print(f"  {r.exp_id:<6} {verdict:<12} {time_str:<8} {de_str:<10} {pr_str:<8} {desc}")

    if n_fail > 0:
        print("\n  FAILURES:")
        for r in results:
            if not r.success:
                # Show first line of error only in table
                first_line = r.error_msg.split("\n")[-1][:80]
                print(f"    ❌ {r.exp_id}: {first_line}")

    # Save execution log
    log_dir = _RESULTS_BASE / "all_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"all_experiments_log_{timestamp}.json"

    log_data = {
        "timestamp": timestamp,
        "total_experiments": len(results),
        "executed_ok": n_pass,
        "execution_errors": n_fail,
        "verdicts": {
            "confirmed": n_confirmed,
            "rejected": n_rejected,
            "partial_or_ok": n_pass - n_confirmed - n_rejected,
        },
        "total_elapsed_s": round(total_elapsed, 2),
        "overrides": {
            "n_qubits": args.n_qubits,
            "p_layers": args.p,
            "topology": args.topology,
            "seeds": args.seeds,
        },
        "excluded": [e.upper() for e in (args.exclude or [])],
        "results": [
            {
                "exp_id": r.exp_id,
                "description": r.description,
                "success": r.success,
                "verdict": r.verdict,
                "elapsed_s": round(r.elapsed_s, 2),
                "mean_de_gap": r.mean_de_gap,
                "pass_rate": r.pass_rate,
                "error_msg": r.error_msg if not r.success else "",
                "summary": r.summary if r.success else None,
            }
            for r in results
        ],
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n  Execution log: {log_path}")

    print(f"{'=' * 70}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
