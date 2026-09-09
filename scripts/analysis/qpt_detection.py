#!/usr/bin/env python
"""QPT Detection CLI — thin wrapper over ``qmbp_simulation.analysis.qpt_detection``.

The detection logic now lives in the package (domain code). This script is the
command-line entry point that re-exports the public API for backward
compatibility and adds pretty-printing / JSON saving.

Usage:
    # Using exact ground truth energies
    python scripts/analysis/qpt_detection.py --topology chain_1d --save

    # Using MPNN-predicted energies (tests if GNN captures QPT)
    python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted --save

    # Compare both (key thesis result)
    python scripts/analysis/qpt_detection.py --topology chain_1d --compare --save

    # Restrict h range
    python scripts/analysis/qpt_detection.py --topology chain_1d --h-range 0.5 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root on path (so qmbp_simulation is importable when run directly)
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export the public API from the package (domain-logic home).
from qmbp_simulation.analysis.qpt_detection import (  # noqa: E402
    compute_second_derivative,
    find_critical_field,
    finite_size_scaling,
    get_h_critical,
    load_energy_curves,
    run_qpt_analysis,
)

__all__ = [
    "compute_second_derivative",
    "find_critical_field",
    "finite_size_scaling",
    "get_h_critical",
    "load_energy_curves",
    "run_qpt_analysis",
]

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="QPT Detection via Energy Derivatives — validates that the pipeline captures h_c correctly"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="chain_1d",
        help="Lattice topology (default: chain_1d)",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=1,
        help="HVA circuit depth (default: 1)",
    )
    parser.add_argument(
        "--use-predicted",
        action="store_true",
        help="Use MPNN-predicted energies instead of exact GT",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both exact and predicted, show comparison",
    )
    parser.add_argument(
        "--h-range",
        type=float,
        nargs=2,
        default=None,
        help="Restrict h range for analysis (e.g., 0.5 3.0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    h_range = tuple(args.h_range) if args.h_range else None

    if args.compare:
        # Run both exact and predicted
        print(f"\n{'=' * 60}")
        print(f"QPT Detection: {args.topology} — COMPARISON (GT vs MPNN)")
        print(f"{'=' * 60}")

        result_exact = run_qpt_analysis(args.topology, args.p_layers, False, h_range)
        result_pred = run_qpt_analysis(args.topology, args.p_layers, True, h_range)

        _print_analysis(result_exact, "EXACT (Ground Truth)")
        _print_analysis(result_pred, "MPNN-PREDICTED")

        # Comparison
        hc_exact = result_exact.get("h_c_by_n", {})
        hc_pred = result_pred.get("h_c_by_n", {})
        common_n = sorted(set(hc_exact.keys()) & set(hc_pred.keys()))

        if common_n:
            print(f"\n{'─' * 60}")
            print("COMPARISON: h_c(exact) vs h_c(predicted)")
            print(f"{'─' * 60}")
            print(f"{'N':>4} | {'h_c(exact)':>10} | {'h_c(pred)':>10} | {'Δh_c':>8} | {'|Δ|/h_c':>8}")
            print("-" * 55)
            for n_str in common_n:
                hc_e = hc_exact[n_str]
                hc_p = hc_pred[n_str]
                delta = hc_p - hc_e
                rel = abs(delta) / hc_e if hc_e > 0 else float("inf")
                print(f"{n_str:>4} | {hc_e:>10.4f} | {hc_p:>10.4f} | {delta:>+8.4f} | {rel:>8.2%}")

            mean_rel_error = np.mean([abs(hc_pred[n] - hc_exact[n]) / hc_exact[n] for n in common_n if hc_exact[n] > 0])
            print(f"\n  Mean |Δh_c|/h_c = {mean_rel_error:.2%}")
            captures_qpt = mean_rel_error < 0.05
            print(f"  MPNN captures QPT: {'YES' if captures_qpt else 'NO'} (threshold: <5% relative error)")

        if args.save:
            output = {
                "topology": args.topology,
                "comparison": {
                    "exact": result_exact,
                    "predicted": result_pred,
                    "mean_relative_error": float(mean_rel_error) if common_n else None,
                    "captures_qpt": captures_qpt if common_n else None,
                },
            }
            _save_results(output, args.topology, "comparison")

    else:
        # Single run
        result = run_qpt_analysis(args.topology, args.p_layers, args.use_predicted, h_range)
        source = "MPNN-predicted" if args.use_predicted else "exact (GT)"

        print(f"\n{'=' * 60}")
        print(f"QPT Detection: {args.topology} (source: {source})")
        print(f"{'=' * 60}")

        _print_analysis(result, source)

        if args.save:
            _save_results(result, args.topology, "predicted" if args.use_predicted else "exact")


def _print_analysis(result: dict, label: str) -> None:
    """Pretty-print QPT analysis results."""
    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        return

    print(f"\n── {label} ──")
    print(f"  N values analyzed: {result['n_values_analyzed']}")

    hc = result.get("h_c_by_n", {})
    per_n = result.get("per_n_results", {})

    if hc:
        print(f"\n  {'N':>4} | {'h_c':>6} | {'|d²E/dh²|_max':>14} | {'n_pts':>6} | {'source':>20}")
        print("  " + "-" * 65)
        for n_str in sorted(hc.keys(), key=lambda x: int(x)):
            info = per_n.get(n_str, {})
            print(
                f"  {n_str:>4} | {hc[n_str]:>6.3f} | "
                f"{info.get('peak_magnitude', 0):>14.4f} | "
                f"{info.get('n_points', 0):>6} | "
                f"{info.get('source', ''):>20}"
            )

    fss = result.get("finite_size_scaling")
    if fss:
        print("\n  ── Finite-Size Scaling ──")
        if "error" in fss:
            print(f"  Fit failed: {fss['error']}")
        else:
            print(f"  h_c(∞) = {fss['h_c_inf']:.4f} ± {fss['h_c_inf_err']:.4f}")
            print(f"  Exponent ν = {fss['nu']:.3f} ± {fss['nu_err']:.3f}")
            print(f"  R² = {fss['r_squared']:.4f}")
            # For TFIM chain_1d, exact h_c = 1.0
            if result.get("topology") == "chain_1d":
                error_from_exact = abs(fss["h_c_inf"] - 1.0)
                print(f"  Error from exact (h_c=1.0): {error_from_exact:.4f} ({error_from_exact * 100:.1f}%)")


def _save_results(output: dict, topology: str, suffix: str) -> None:
    """Save results to JSON file."""
    from qmbp_simulation.utils.helpers import json_serialize

    out_dir = _project_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qpt_detection_{topology}_{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
