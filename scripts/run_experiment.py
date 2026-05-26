#!/usr/bin/env python3
"""Unified CLI runner for experiments.

Usage:
    python scripts/run_experiment.py --exp F3
    python scripts/run_experiment.py --exp A3 --seeds 42 43 --verbose
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py --exp B1 --n-qubits 10 --p 1

All experiments follow the same lifecycle:
    setup() → run() → analyze() → report() → save()
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Ensure project root is in path for experiment imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Experiment registry: ID → (module_path, class_name)
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
}

_PLANNED_NOT_IMPLEMENTED: dict[str, str] = {
    "A1": "High effort — excluded from final plan",
    "A2": "High effort — excluded from final plan",
    "B3": "Excluded — needs external libs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --exp F3                    # Run F3 with defaults
    %(prog)s --exp A3 --verbose          # Run A3 with INFO logging
    %(prog)s --exp B1 --n-qubits 10      # Override system size
    %(prog)s --list                      # Show available experiments
    %(prog)s --exp B1 B4 F3              # Run multiple experiments
        """,
    )

    parser.add_argument(
        "--exp",
        nargs="+",
        type=str,
        help="Experiment ID(s) to run (e.g., A3, B4, F3)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )

    # Override options
    parser.add_argument("--n-qubits", type=int, help="Override number of qubits")
    parser.add_argument("--p", type=int, help="Override HVA layers")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seeds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    return parser.parse_args()


def _load_experiment_class(exp_id: str):
    """Load experiment class by ID from the registry."""
    exp_id = exp_id.upper()
    if exp_id not in EXPERIMENT_REGISTRY:
        if exp_id in _PLANNED_NOT_IMPLEMENTED:
            raise ValueError(
                f"Experiment {exp_id} is planned but not yet implemented: "
                f"{_PLANNED_NOT_IMPLEMENTED[exp_id]}. "
                f"Available: {sorted(EXPERIMENT_REGISTRY.keys())}"
            )
        raise ValueError(
            f"Unknown experiment: {exp_id}. Available: {sorted(EXPERIMENT_REGISTRY.keys())}"
        )

    module_path, class_name = EXPERIMENT_REGISTRY[exp_id]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_experiments() -> None:
    """Print available experiments with descriptions."""
    print("\nAvailable Experiments:")
    print("=" * 60)

    categories: dict[str, list[str]] = {}
    for exp_id in sorted(EXPERIMENT_REGISTRY.keys()):
        cat = exp_id[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp_id)

    category_names = {
        "A": "Ground Truth Enhancement",
        "B": "VQE Optimization Enhancement",
        "C": "MPNN Predictor Enhancement",
        "D": "Landscape & Phase Transition Analysis",
        "E": "Scaling & Generalization",
        "F": "Novel Methodological Contributions",
        "G": "Pipeline Characterization & Validation",
    }

    for cat in sorted(categories.keys()):
        print(f"\n  Category {cat}: {category_names.get(cat, 'Unknown')}")
        print(f"  {'-' * 50}")
        for exp_id in categories[cat]:
            try:
                cls = _load_experiment_class(exp_id)
                if hasattr(cls, "default_config"):
                    cfg = cls.default_config()
                    desc = cfg.description[:50]
                else:
                    desc = "(no description)"
            except Exception:
                desc = "(not yet implemented)"
            print(f"    {exp_id}: {desc}")

    print(f"\n{'=' * 60}")
    print("Run with: python scripts/run_experiment.py --exp <ID>")


def run_single_experiment(exp_id: str, args: argparse.Namespace) -> dict:
    """Run a single experiment with CLI overrides applied."""
    from qmbp_simulation.framework import ExperimentConfig

    cls = _load_experiment_class(exp_id)

    # Get default config
    if hasattr(cls, "default_config"):
        config = cls.default_config()
    else:
        config = ExperimentConfig(experiment_id=exp_id)

    # Apply CLI overrides
    if args.n_qubits:
        config.system.n_qubits = args.n_qubits
    if args.p:
        config.system.p_layers = args.p
    if args.seeds:
        config.seeds = args.seeds
    if args.verbose:
        config.verbose = True
    if args.debug:
        config.debug = True
        config.verbose = True

    # Execute
    experiment = cls(config)
    return experiment.execute()


def main() -> None:
    args = parse_args()

    if args.list:
        list_experiments()
        return

    if not args.exp:
        print("Error: specify --exp <ID> or --list")
        sys.exit(1)

    # Run each experiment
    results = {}
    for exp_id in args.exp:
        print(f"\n{'#' * 60}")
        print(f"# Running experiment: {exp_id.upper()}")
        print(f"{'#' * 60}\n")

        try:
            analysis = run_single_experiment(exp_id.upper(), args)
            results[exp_id.upper()] = analysis
        except Exception as e:
            print(f"\nERROR in {exp_id}: {e}")
            if args.debug:
                raise
            results[exp_id.upper()] = {"error": str(e)}

    # Summary if multiple experiments
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for exp_id, analysis in results.items():
            if "error" in analysis:
                print(f"  {exp_id}: FAILED — {analysis['error']}")
            else:
                summary = analysis.get("summary", {})
                de_gap = summary.get("mean_de_gap", "?")
                print(f"  {exp_id}: mean ΔE/gap = {de_gap}")


if __name__ == "__main__":
    main()
