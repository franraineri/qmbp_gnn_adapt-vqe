#!/usr/bin/env python
"""Amortization Plot: GNN training cost vs cumulative inference savings.

Paso 1.2 of the Efficiency & Dynamics roadmap.

Shows that after ~20-50 queries, the GNN (train once + predict O(1)) becomes
cheaper than running DMRG/ED for each new h-point independently.

Data sources:
  - Zoo manifest: n_training_points per model (DMRG cost = n_points × time_per_point)
  - GT cache: time_s per (topology, N) for DMRG/ED cost estimates
  - MPNN inference: ~1ms per prediction (measured)

Output:
  - JSON: results/analysis/amortization_{topology}.json
  - The data is structured for direct plotting by thesis figure generators.

Usage:
    # Default (chain_1d, uses zoo + GT data)
    python scripts/analysis/amortization_plot.py

    # Specific topology
    python scripts/analysis/amortization_plot.py --topology heavy_hex

    # With custom MPNN training time estimate
    python scripts/analysis/amortization_plot.py --mpnn-training-time 120
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


def estimate_dmrg_cost_per_point(topology: str, n_qubits: int) -> float:
    """Estimate wall-time for a single GT computation at given N.

    Uses time_s from GT cache if available, else analytical estimate.

    Returns
    -------
    float
        Estimated seconds per ground truth point.
    """
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    times = []
    prefix = f"{topology}|{n_qubits}|"

    for key, entry in cache._data.items():
        if not key.startswith(prefix):
            continue
        t = entry.get("time_s", 0)
        if t > 0:
            times.append(t)

    if times:
        return float(np.mean(times))

    # Analytical fallback: ED scales as O(2^N), DMRG as O(N * chi^2)
    if n_qubits <= 14:
        # ED: ~0.01s for N=10, ~1s for N=14, ~30s for N=20
        return 0.01 * (2 ** (n_qubits - 10))
    elif n_qubits <= 22:
        # Sparse eigsh: moderate scaling
        return 0.1 * (n_qubits / 10) ** 3
    else:
        # DMRG: roughly linear in N (chi fixed)
        return 5.0 * (n_qubits / 20) ** 2


def get_zoo_training_info(topology: str) -> list[dict]:
    """Get training metadata from zoo manifest for this topology.

    Returns list of dicts with: n_training_points, checkpoint, pass_rate.
    """
    manifest_path = _project_root / "data" / "model_zoo" / "manifest.json"
    if not manifest_path.exists():
        return []

    manifest = json.load(open(manifest_path))
    entries = []
    for e in manifest:
        if e.get("topology") == topology or e.get("topology") == "multi_topology":
            ntp = e.get("n_training_points", 0)
            if ntp > 0:
                entries.append({
                    "checkpoint": e.get("checkpoint_file", ""),
                    "n_training_points": ntp,
                    "pass_rate": e.get("pass_rate", 0),
                    "topology": e.get("topology", ""),
                })
    return entries


def measure_mpnn_training_time_estimate(n_training_points: int) -> float:
    """Estimate MPNN training time based on n_training_points."""
    # Empirical: ~0.3s per training point (including all epochs)
    return n_training_points * 0.3


def compute_amortization_curves(
    topology: str,
    n_qubits: int = 20,
    mpnn_training_time_override: float | None = None,
) -> dict:
    """Compute amortization curves for DMRG vs GNN.

    Returns data for plotting:
      X axis: number of h-points queried (1, 5, 10, 20, 50, 100, 500, 1000)
      Y axis: cumulative wall-time

    Two curves:
      - DMRG: linear slope = cost_per_point
      - GNN: fixed training cost + n_queries × inference_time
    """
    # Costs
    dmrg_cost_per_point = estimate_dmrg_cost_per_point(topology, n_qubits)

    # GNN costs
    zoo_entries = get_zoo_training_info(topology)
    if zoo_entries:
        best_entry = max(zoo_entries, key=lambda e: e["pass_rate"])
        n_training_points = best_entry["n_training_points"]
    else:
        n_training_points = 50  # Conservative default

    if mpnn_training_time_override is not None:
        gnn_training_cost = mpnn_training_time_override
    else:
        # Training cost = DMRG for training data + MPNN training
        gnn_training_cost = (
            n_training_points * dmrg_cost_per_point  # Generate training data
            + measure_mpnn_training_time_estimate(n_training_points)  # Train MPNN
        )

    gnn_inference_cost = 0.001  # ~1ms per prediction (measured)

    # Query points for the plot
    n_queries = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]

    dmrg_cumulative = [n * dmrg_cost_per_point for n in n_queries]
    gnn_cumulative = [gnn_training_cost + n * gnn_inference_cost for n in n_queries]

    # Find crossover point (where GNN becomes cheaper)
    crossover_n = None
    for n in range(1, 10000):
        if gnn_training_cost + n * gnn_inference_cost < n * dmrg_cost_per_point:
            crossover_n = n
            break

    # Speedup at various query counts
    speedups = {}
    for n in [50, 100, 500, 1000]:
        dmrg_t = n * dmrg_cost_per_point
        gnn_t = gnn_training_cost + n * gnn_inference_cost
        speedups[str(n)] = round(dmrg_t / gnn_t, 1) if gnn_t > 0 else None

    return {
        "topology": topology,
        "n_qubits_reference": n_qubits,
        "dmrg_cost_per_point_s": round(dmrg_cost_per_point, 3),
        "gnn_training_cost_s": round(gnn_training_cost, 1),
        "gnn_inference_cost_s": gnn_inference_cost,
        "n_training_points": n_training_points,
        "crossover_n_queries": crossover_n,
        "plot_data": {
            "n_queries": n_queries,
            "dmrg_cumulative_s": [round(x, 2) for x in dmrg_cumulative],
            "gnn_cumulative_s": [round(x, 2) for x in gnn_cumulative],
        },
        "speedups": speedups,
        "thesis_claim": (
            f"For {topology} N={n_qubits}: DMRG costs {dmrg_cost_per_point:.2f}s/point. "
            f"GNN training costs {gnn_training_cost:.0f}s (one-time), then {gnn_inference_cost*1000:.1f}ms/point. "
            f"Crossover at {crossover_n} queries. "
            f"At 1000 queries: {speedups.get('1000', '?')}× speedup."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Amortization analysis: GNN vs DMRG cost curves"
    )
    parser.add_argument("--topology", type=str, default="chain_1d")
    parser.add_argument(
        "--n-qubits", type=int, nargs="+", default=[10, 16, 20, 30],
        help="Reference N values for cost estimation",
    )
    parser.add_argument(
        "--mpnn-training-time", type=float, default=None,
        help="Override MPNN training time (seconds)",
    )
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    print(f"\n{'='*65}")
    print(f"  AMORTIZATION ANALYSIS: {args.topology}")
    print(f"{'='*65}")

    results_per_n = []
    for n in args.n_qubits:
        result = compute_amortization_curves(
            topology=args.topology,
            n_qubits=n,
            mpnn_training_time_override=args.mpnn_training_time,
        )
        results_per_n.append(result)

        print(f"\n  N={n}:")
        print(f"    DMRG/ED cost:     {result['dmrg_cost_per_point_s']:.3f}s / point")
        print(f"    GNN training:     {result['gnn_training_cost_s']:.0f}s (one-time)")
        print(f"    GNN inference:    {result['gnn_inference_cost_s']*1000:.1f}ms / point")
        print(f"    Crossover:        {result['crossover_n_queries']} queries")
        print(f"    Speedup @1000pts: {result['speedups'].get('1000', '?')}×")

    if args.save:
        from qmbp_simulation.utils.helpers import json_serialize

        out_dir = _project_root / "results" / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"amortization_{args.topology}.json"

        output = {
            "topology": args.topology,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "per_n_results": results_per_n,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=json_serialize)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
