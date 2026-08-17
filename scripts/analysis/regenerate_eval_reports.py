#!/usr/bin/env python3
"""Regenerate evaluation reports from existing NPZ data.

Reads previously computed extrapolation NPZ files and baseline NPZ files,
reconstructs the per-h result dicts, and calls generate_evaluation_report()
to produce updated markdown reports with all new metrics (|ΔE|, quality
profile, metric warnings, per-point classification).

Does NOT re-run any VQE or MPNN prediction — purely a report format upgrade
from already-computed data.

Usage:
    # Regenerate all topologies
    python scripts/analysis/regenerate_eval_reports.py

    # Specific topology
    python scripts/analysis/regenerate_eval_reports.py --topology chain_1d

    # Include baseline comparison (if baseline NPZ exists)
    python scripts/analysis/regenerate_eval_reports.py --with-baseline

    # Custom output dir
    python scripts/analysis/regenerate_eval_reports.py --output-dir results/extrapolation_evals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EXTRAP_DIR = ROOT / "data" / "large_n_extrapolation"
BASELINE_DIR = EXTRAP_DIR / "_baselines"
DEFAULT_OUTPUT = ROOT / "results" / "extrapolation_evals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate eval reports from existing NPZ data")
    parser.add_argument(
        "--topology",
        "-t",
        type=str,
        default=None,
        help="Regenerate only this topology (default: all found)",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        default=True,
        help="Include random VQE baseline comparison if available (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=1,
        help="HVA depth to filter (default: 1)",
    )
    return parser.parse_args()


def load_npz_as_per_h_results(npz_path: Path, n_qubits: int) -> list[dict]:
    """Reconstruct per_h_results list from an extrapolation NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    h_values = data["h_values"]
    e_key = "e_vqe" if "e_vqe" in data else ("e_pred" if "e_pred" in data else None)
    e_exact = data["e_exact"]
    gaps = data["gaps"] if "gaps" in data else np.ones(len(h_values))
    de_gaps = data["de_gaps"] if "de_gaps" in data else None
    methods = data["method"] if "method" in data else ["mpnn"] * len(h_values)
    theta_opt = data["theta_opt"] if "theta_opt" in data else None

    results = []
    for i in range(len(h_values)):
        e_pred_val = float(data[e_key][i]) if e_key else 0.0
        e_exact_val = float(e_exact[i])
        gap_val = float(gaps[i]) if i < len(gaps) else 1.0
        abs_err = abs(e_pred_val - e_exact_val)
        de_gap_val = float(de_gaps[i]) if de_gaps is not None else (abs_err / max(gap_val, 1e-10))

        result = {
            "h": float(h_values[i]),
            "e_pred": e_pred_val,
            "e_exact": e_exact_val,
            "gap": gap_val,
            "de_gap": de_gap_val,
            "abs_error": abs_err,
            "n_qubits": n_qubits,
            "method": str(methods[i]) if i < len(methods) else "mpnn",
        }

        # Include theta if available (for n_params inference)
        if theta_opt is not None:
            try:
                theta_i = np.asarray(theta_opt[i], dtype=np.float64).flatten()
                if theta_i.size > 0 and np.all(np.isfinite(theta_i)):
                    result["theta"] = theta_i.tolist()
            except (ValueError, TypeError):
                pass

        results.append(result)

    return results


def load_baseline_summary(baseline_path: Path, n_qubits: int) -> dict | None:
    """Load baseline NPZ and compute summary metrics for comparison table."""
    if not baseline_path.exists():
        return None

    try:
        data = np.load(baseline_path)
        h_values = data["h_values"]
        e_vqe = data["e_vqe"]
        e_exact = data["e_exact"]
        gaps = data["gaps"] if "gaps" in data else np.ones(len(h_values))
        n_evals = data["n_evals"] if "n_evals" in data else np.zeros(len(h_values))

        abs_errors = np.abs(e_vqe - e_exact)
        de_gaps = abs_errors / np.maximum(gaps, 1e-10)

        return {
            "mean_de_gap": float(de_gaps.mean()),
            "mean_abs_error": float(abs_errors.mean()),
            "pass_rate_dual": float((de_gaps < 0.05).mean()),
            "total_evals": int(n_evals.sum()),
            "n_points": len(h_values),
        }
    except Exception:
        return None


def infer_checkpoint(topology: str, p_layers: int) -> str:
    """Infer which checkpoint was used from the model zoo manifest."""
    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        entries = _load_manifest()
        multi_n = [
            e
            for e in entries
            if e.topology == topology and e.n_qubits == 0 and e.p_layers == p_layers
        ]
        if multi_n:
            best = max(multi_n, key=lambda e: e.n_training_points)
            return best.checkpoint_file
    except Exception:
        pass
    return "unknown"


def regenerate_topology(
    topology: str,
    p_layers: int,
    output_dir: Path,
    with_baseline: bool = True,
) -> Path | None:
    """Regenerate the eval report for a single topology from NPZ data."""
    from qmbp_simulation.analysis.evaluation_report import generate_evaluation_report
    from qmbp_simulation.analysis.metrics import compute_deploy_summary

    # Find all NPZ files for this topology
    npz_files = sorted(EXTRAP_DIR.glob(f"{topology}_N*_p{p_layers}.npz"))
    if not npz_files:
        print(f"  ⏭️  {topology}: no NPZ files found")
        return None

    # Parse N values and load data
    mpnn_results_by_n: dict[int, dict] = {}
    for npz_path in npz_files:
        try:
            n_str = npz_path.stem.split("_N")[1].split("_")[0]
            n_qubits = int(n_str)
        except (IndexError, ValueError):
            continue

        per_h = load_npz_as_per_h_results(npz_path, n_qubits)
        if not per_h:
            continue

        # Infer n_params from theta length (if available)
        n_params = 0
        for p in per_h:
            if "theta" in p:
                n_params = len(p["theta"])
                break

        # Compute summary via existing utility
        summary = compute_deploy_summary(per_h)

        mpnn_results_by_n[n_qubits] = {
            "per_point": per_h,
            "n_params": n_params,
            "n_qubits": n_qubits,
            **summary,
        }

    if not mpnn_results_by_n:
        print(f"  ⏭️  {topology}: no valid data loaded")
        return None

    # Build comparison dict with baseline (if available)
    comparison: dict | None = None
    if with_baseline:
        comparison = {}
        for n_qubits, mpnn in mpnn_results_by_n.items():
            entry: dict = {
                "N": n_qubits,
                "n_params": mpnn["n_params"],
                "mpnn": {
                    "mean_de_gap": mpnn["mean_de_gap"],
                    "mean_abs_error_per_site": mpnn.get("mean_abs_error_per_site"),
                    "pass_rate_dual": mpnn.get("pass_rate_dual", 0),
                    "n_evals": mpnn.get("n_points", 0),
                },
            }

            # Load baseline
            baseline_path = BASELINE_DIR / f"{topology}_N{n_qubits}_p{p_layers}_random_vqe.npz"
            baseline = load_baseline_summary(baseline_path, n_qubits)
            if baseline:
                entry["random_vqe"] = baseline
                mpnn_evals = mpnn.get("n_points", 1)
                entry["speedup"] = baseline["total_evals"] / max(mpnn_evals, 1)

                # Win rate: compare per-h (use de_gap arrays)
                baseline_data = np.load(baseline_path)
                baseline_de_gaps = np.abs(
                    baseline_data["e_vqe"] - baseline_data["e_exact"]
                ) / np.maximum(
                    baseline_data["gaps"]
                    if "gaps" in baseline_data
                    else np.ones(len(baseline_data["h_values"])),
                    1e-10,
                )
                mpnn_de_gaps = np.array([p["de_gap"] for p in mpnn["per_point"]])

                # Match h-values between MPNN and baseline for fair comparison
                baseline_h = set(round(float(h), 6) for h in baseline_data["h_values"])
                mpnn_h_map = {round(p["h"], 6): p["de_gap"] for p in mpnn["per_point"]}

                wins = 0
                compared = 0
                for j, h in enumerate(baseline_data["h_values"]):
                    h_key = round(float(h), 6)
                    if h_key in mpnn_h_map:
                        compared += 1
                        if mpnn_h_map[h_key] < float(baseline_de_gaps[j]):
                            wins += 1
                entry["mpnn_win_rate"] = wins / max(compared, 1)

            comparison[n_qubits] = entry

    # Determine h_range from data
    all_h = []
    for mpnn in mpnn_results_by_n.values():
        all_h.extend(p["h"] for p in mpnn["per_point"])
    h_range = (min(all_h), max(all_h)) if all_h else (2.5, 5.0)
    n_h_points = max(len(mpnn["per_point"]) for mpnn in mpnn_results_by_n.values())

    # Infer checkpoint
    checkpoint = infer_checkpoint(topology, p_layers)

    # Generate report
    report_path = generate_evaluation_report(
        mpnn_results_by_n=mpnn_results_by_n,
        topology=topology,
        model_name="tfim_bond_resolved",
        checkpoint=checkpoint,
        h_range=h_range,
        n_h_points=n_h_points,
        p_layers=p_layers,
        target_n=sorted(mpnn_results_by_n.keys()),
        comparison=comparison,
        output_dir=str(output_dir),
    )

    n_points_total = sum(len(m["per_point"]) for m in mpnn_results_by_n.values())
    n_vals = sorted(mpnn_results_by_n.keys())
    has_baseline_str = (
        " + baseline"
        if (comparison and any("random_vqe" in v for v in comparison.values()))
        else ""
    )
    print(
        f"  ✅ {topology}: N={n_vals}, {n_points_total} points{has_baseline_str} "
        f"→ {report_path.relative_to(ROOT)}"
    )
    return report_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("REGENERATE EVALUATION REPORTS FROM NPZ DATA")
    print("=" * 70)
    print(f"  Source: {EXTRAP_DIR.relative_to(ROOT)}")
    print(f"  Output: {output_dir.relative_to(ROOT)}")
    print(f"  p_layers: {args.p_layers}")
    print()

    # Discover topologies
    if args.topology:
        topologies = [args.topology]
    else:
        npz_files = sorted(EXTRAP_DIR.glob(f"*_N*_p{args.p_layers}.npz"))
        topologies = sorted(set(f.stem.split("_N")[0] for f in npz_files))

    if not topologies:
        print("  No NPZ files found. Nothing to regenerate.")
        return 0

    print(f"  Topologies: {topologies}")
    print()

    generated = []
    for topo in topologies:
        path = regenerate_topology(
            topo, args.p_layers, output_dir, with_baseline=args.with_baseline
        )
        if path:
            generated.append(path)

    print()
    print(f"  Generated {len(generated)} report(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
