#!/usr/bin/env python3
"""Auto-generate config presets from successful ResultIndex entries.

Scans the result index for configurations with >80% pass rate and
generates YAML preset files in configs/presets/noiseless/.

Usage:
    python scripts/generate_presets_from_index.py
    python scripts/generate_presets_from_index.py --min-pass-rate 0.9
    python scripts/generate_presets_from_index.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.framework.result_index import ResultIndex

logger = logging.getLogger(__name__)
PRESETS_DIR = ROOT / "configs" / "presets"


def _group_entries(index: ResultIndex) -> dict[tuple, list[dict]]:
    """Group valid index entries by (model, topology, n_qubits, p_layers)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for entry in index.valid_entries:
        model = entry.get("model", "")
        topology = entry.get("topology", "")
        n_qubits = entry.get("n_qubits", 0)
        p_layers = entry.get("p_layers", 0)

        key = (model, topology, n_qubits, p_layers)
        groups[key].append(entry)
    return groups


def _compute_group_stats(runs: list[dict]) -> dict:
    """Compute aggregate stats for a group of runs."""
    n_runs = len(runs)
    n_passed = sum(1 for r in runs if r.get("passed", False))
    pass_rate = n_passed / n_runs if n_runs > 0 else 0.0
    best_run = max(runs, key=lambda r: (r.get("pass_rate", 0), r.get("timestamp", "")))
    return {
        "n_runs": n_runs,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
        "best_pass_rate": best_run.get("pass_rate", 0.0),
        "best_file": best_run.get("_file", ""),
    }


def _infer_h_range(model: str, n_qubits: int) -> tuple[float, float, int]:
    """Infer reasonable h-range defaults based on known findings.

    Returns (h_min, h_max, h_points).
    """
    # From project findings: h_min=1.25-1.3 eliminates critical-region failures
    if model in ("tfim", "tfim_longitudinal"):
        if n_qubits >= 16:
            return 1.3, 3.0, 40
        elif n_qubits >= 10:
            return 1.25, 3.0, 30
        else:
            return 0.5, 2.0, 15
    elif model == "tfim_frustrated":
        return 0.5, 3.0, 20
    else:
        return 0.5, 2.0, 15


def _infer_vqe_params(n_qubits: int, p_layers: int) -> tuple[int, int]:
    """Infer reasonable maxiter and n_restarts.

    Returns (maxiter, n_restarts).
    """
    if n_qubits >= 16 and p_layers >= 3:
        return 1000, 7
    elif n_qubits >= 10:
        return 800, 5
    else:
        return 500, 5


def generate_preset_yaml(
    model: str,
    topology: str,
    n_qubits: int,
    p_layers: int,
    stats: dict,
) -> str:
    """Generate YAML content for a preset."""
    h_min, h_max, h_points = _infer_h_range(model, n_qubits)
    maxiter, n_restarts = _infer_vqe_params(n_qubits, p_layers)

    lines = [
        f"# {model} {topology} N={n_qubits} p={p_layers}",
        f"# Pass rate: {stats['best_pass_rate']:.0%} "
        f"({stats['n_passed']}/{stats['n_runs']} runs passed)",
        "# Auto-generated from ResultIndex",
        "runner: noiseless_pipeline",
        f"model: {model}",
        f"topology: {topology}",
        f"n_qubits: {n_qubits}",
        f"p_layers: {p_layers}",
        f"h_min: {h_min}",
        f"h_max: {h_max}",
        f"h_points: {h_points}",
        f"maxiter: {maxiter}",
        f"n_restarts: {n_restarts}",
        "seeds: [42, 43, 44]",
        f'description: "{model} {topology} N={n_qubits} p={p_layers} '
        f'— {stats["best_pass_rate"]:.0%} validated"',
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate presets from ResultIndex")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.8,
        help="Minimum pass rate to generate a preset (default: 0.8)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="Minimum number of runs required (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing preset files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load index
    index = ResultIndex()
    logger.info(f"Loaded index: {len(index)} entries")

    # Load dashboard for cross-check (fresher signal than ResultIndex alone)
    dashboard_configs: dict[tuple, dict] = {}
    try:
        import json
        dashboard_path = ROOT / "data" / "model_quality_dashboard.json"
        if dashboard_path.exists():
            with open(dashboard_path) as f:
                dashboard = json.load(f)
            for dc in dashboard.get("configs", []):
                key = (dc.get("model", ""), dc.get("topology", ""),
                       dc.get("n_qubits", 0), dc.get("p_layers", 0))
                dashboard_configs[key] = dc
            logger.info(f"Dashboard: {len(dashboard_configs)} configs loaded for cross-check")
    except (json.JSONDecodeError, OSError):
        pass

    # Group and filter
    groups = _group_entries(index)
    logger.info(f"Found {len(groups)} unique (model, topology, N, p) configurations")

    generated = 0
    skipped_rate = 0
    skipped_runs = 0
    skipped_exists = 0

    for (model, topology, n_qubits, p_layers), runs in sorted(groups.items()):
        stats = _compute_group_stats(runs)

        # Filter by pass rate
        if stats["best_pass_rate"] < args.min_pass_rate:
            skipped_rate += 1
            continue

        # Filter by minimum runs
        if stats["n_runs"] < args.min_runs:
            skipped_runs += 1
            continue

        # Cross-check with dashboard: skip if NPZ data shows < 50% pass
        # (the dashboard is updated every run and reflects actual θ quality)
        db_entry = dashboard_configs.get((model, topology, n_qubits, p_layers))
        if db_entry and db_entry.get("pass_rate_dual_criterion", db_entry.get("pass_rate_5pct", 1.0)) < 0.50:
            logger.debug(
                "  Skip %s/%s N=%d p=%d: dashboard pass_rate_dual=%.0f%% < 50%%",
                model, topology, n_qubits, p_layers,
                db_entry.get("pass_rate_dual_criterion", db_entry.get("pass_rate_5pct", 0)) * 100,
            )
            skipped_rate += 1
            continue

        # Generate filename
        preset_name = f"{model}_{topology}_n{n_qubits}_p{p_layers}"
        category = "noiseless"  # All indexed runs are noiseless for now
        preset_path = PRESETS_DIR / category / f"{preset_name}.yaml"

        # Check if already exists
        if preset_path.exists() and not args.force:
            skipped_exists += 1
            continue

        # Generate YAML
        yaml_content = generate_preset_yaml(model, topology, n_qubits, p_layers, stats)

        if args.dry_run:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"Would create: {preset_path.relative_to(ROOT)}")
            logger.info(yaml_content)
        else:
            preset_path.parent.mkdir(parents=True, exist_ok=True)
            preset_path.write_text(yaml_content)
            logger.info(
                f"  ✅ {preset_name} — {stats['best_pass_rate']:.0%} ({stats['n_runs']} runs)"
            )

        generated += 1

    # Summary
    logger.info(f"\n{'═' * 60}")
    logger.info(f"Generated: {generated} presets")
    logger.info(f"Skipped (pass_rate < {args.min_pass_rate:.0%}): {skipped_rate}")
    logger.info(f"Skipped (< {args.min_runs} runs): {skipped_runs}")
    logger.info(f"Skipped (already exists): {skipped_exists}")

    if not args.dry_run and generated > 0:
        logger.info(f"\nPresets saved to: {PRESETS_DIR / 'noiseless'}")
        logger.info("Use with: --preset noiseless/<name>")


if __name__ == "__main__":
    main()
