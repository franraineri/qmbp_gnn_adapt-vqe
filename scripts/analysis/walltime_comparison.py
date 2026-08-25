#!/usr/bin/env python
"""Wall-time Comparison: DMRG/ED vs MPNN Inference.

Paso 1.1 of the Efficiency & Dynamics roadmap.

Extracts computation times from the GroundTruthCache and compares against
MPNN inference time (measured live). Produces a summary table showing the
speedup factor per system size N.

Since existing GT cache entries lack time_s (added in this session), this
script can either:
  1. Use existing time_s entries (if available from recent computations).
  2. Re-measure ED/DMRG times by re-computing a sample of GT points with timing.

The MPNN inference time is measured by running forward passes on the loaded model.

Output:
  - Console table: method | N | mean_time | max_time | n_points
  - JSON: results/analysis/walltime_comparison.json
  - CSV (optional): results/analysis/walltime_comparison.csv

Usage:
    # Quick mode: use whatever time_s is in GT cache + measure MPNN
    python scripts/analysis/walltime_comparison.py

    # Backfill mode: re-time a sample of GT computations (more accurate)
    python scripts/analysis/walltime_comparison.py --backfill --sample-per-n 5

    # Specific topology
    python scripts/analysis/walltime_comparison.py --topology heavy_hex --backfill
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GT Timing Extraction / Backfill
# ═══════════════════════════════════════════════════════════════════════════════


def extract_gt_timing(topology: str | None = None) -> dict[int, list[float]]:
    """Extract existing time_s from GT cache entries, grouped by N.

    Returns
    -------
    dict[int, list[float]]
        {N: [time_s_1, time_s_2, ...]} for entries that have time_s > 0.
    """
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    times_by_n: dict[int, list[float]] = defaultdict(list)

    for key, entry in cache._data.items():
        t = entry.get("time_s", 0)
        if t <= 0:
            continue
        parts = key.split("|")
        if len(parts) < 4:
            continue
        topo = parts[0]
        if topology and topo != topology:
            continue
        try:
            n = int(parts[1])
        except ValueError:
            continue
        times_by_n[n].append(t)

    return dict(times_by_n)


def backfill_gt_timing(
    topology: str = "chain_1d",
    sample_per_n: int = 5,
    model: str = "tfim",
) -> dict[int, list[float]]:
    """Re-compute a sample of GT points to measure wall-time.

    Uses get_or_compute which now stores time_s. For entries already cached,
    we force recomputation by temporarily removing from cache.

    Returns
    -------
    dict[int, list[float]]
        {N: [time_s_1, ...]} freshly measured times.
    """
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    times_by_n: dict[int, list[float]] = defaultdict(list)

    # Collect all entries for this topology
    entries_by_n: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for key in list(cache._data.keys()):
        parts = key.split("|")
        if len(parts) < 4:
            continue
        if parts[0] != topology:
            continue
        if parts[2] != model:
            continue
        try:
            n = int(parts[1])
            h = float(parts[3])
        except ValueError:
            continue
        entries_by_n[n].append((key, h))

    for n in sorted(entries_by_n.keys()):
        entries = entries_by_n[n]
        # Sample up to sample_per_n entries
        rng = np.random.default_rng(42)
        indices = rng.choice(len(entries), size=min(sample_per_n, len(entries)), replace=False)

        for idx in indices:
            key, h = entries[idx]
            # Remove from cache to force recomputation with timing
            old_entry = cache._data.pop(key, None)

            t0 = time.perf_counter()
            try:
                cache.get_or_compute(topology, n, model, h, flush=False)
            except Exception as e:
                logger.warning(f"  Backfill failed for {key}: {e}")
                # Restore old entry
                if old_entry:
                    cache._data[key] = old_entry
                continue
            elapsed = time.perf_counter() - t0
            times_by_n[n].append(elapsed)

            # The entry now has time_s stored
            logger.debug(f"  {topology} N={n} h={h:.2f}: {elapsed:.3f}s")

    # Flush once at the end
    cache.flush()
    return dict(times_by_n)


# ═══════════════════════════════════════════════════════════════════════════════
# MPNN Inference Timing
# ═══════════════════════════════════════════════════════════════════════════════


def measure_mpnn_inference(
    topology: str = "chain_1d",
    n_qubits_list: list[int] | None = None,
    n_samples: int = 20,
    p_layers: int = 1,
) -> dict[int, list[float]]:
    """Measure MPNN inference time across different N values.

    Returns
    -------
    dict[int, list[float]]
        {N: [inference_time_1, ...]} in seconds.
    """
    import torch

    from qmbp_simulation.models.hamiltonian import make_lattice
    from qmbp_simulation.predictors.model_zoo import load_best_model_for
    from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

    if n_qubits_list is None:
        n_qubits_list = [4, 6, 8, 10, 12, 14, 16, 20, 30]

    # Load model once
    try:
        model, entry, source = load_best_model_for(
            topology,
            p_layers=p_layers,
            n_target=max(n_qubits_list) if n_qubits_list else 20,
            h_regime="paramagnetic",
        )
    except Exception as e:
        logger.warning(f"  No MPNN available for {topology}: {e}")
        return {}

    model.eval()
    times_by_n: dict[int, list[float]] = defaultdict(list)
    h_values = np.linspace(0.5, 4.0, n_samples)

    for n in n_qubits_list:
        for h in h_values:
            lattice = make_lattice(topology, n, J=1.0, h=h)
            graph = build_unified_bond_resolved_graph(lattice, h_value=h, p_layers=p_layers)

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(graph)
            elapsed = time.perf_counter() - t0
            times_by_n[n].append(elapsed)

    return dict(times_by_n)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Output
# ═══════════════════════════════════════════════════════════════════════════════


def build_comparison_table(
    gt_times: dict[int, list[float]],
    mpnn_times: dict[int, list[float]],
) -> list[dict]:
    """Build comparison records for all N values."""
    all_n = sorted(set(gt_times.keys()) | set(mpnn_times.keys()))
    records = []

    for n in all_n:
        gt = gt_times.get(n, [])
        mpnn = mpnn_times.get(n, [])

        gt_mean = float(np.mean(gt)) if gt else None
        gt_max = float(np.max(gt)) if gt else None
        mpnn_mean = float(np.mean(mpnn)) if mpnn else None

        speedup = gt_mean / mpnn_mean if (gt_mean and mpnn_mean and mpnn_mean > 0) else None

        method = "exact_diag" if n <= 22 else "dmrg"

        records.append(
            {
                "n_qubits": n,
                "method": method,
                "gt_mean_s": gt_mean,
                "gt_max_s": gt_max,
                "gt_n_points": len(gt),
                "mpnn_mean_s": mpnn_mean,
                "mpnn_n_points": len(mpnn),
                "speedup": speedup,
            }
        )

    return records


def print_table(records: list[dict]) -> None:
    """Pretty-print the comparison table."""
    print(f"\n{'=' * 75}")
    print("  WALL-TIME COMPARISON: Classical GT vs MPNN Inference")
    print(f"{'=' * 75}")
    print(
        f"  {'N':>4} | {'Method':<11} | {'GT mean':>9} | {'GT max':>9} | "
        f"{'MPNN mean':>10} | {'Speedup':>8} | {'GT pts':>6}"
    )
    print(f"  {'-' * 4}-+-{'-' * 11}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 6}")

    for r in records:
        gt_m = f"{r['gt_mean_s']:.3f}s" if r["gt_mean_s"] else "—"
        gt_x = f"{r['gt_max_s']:.1f}s" if r["gt_max_s"] else "—"
        mpnn_m = f"{r['mpnn_mean_s'] * 1000:.2f}ms" if r["mpnn_mean_s"] else "—"
        sp = f"{r['speedup']:.0f}×" if r["speedup"] else "—"
        print(
            f"  {r['n_qubits']:>4} | {r['method']:<11} | {gt_m:>9} | {gt_x:>9} | "
            f"{mpnn_m:>10} | {sp:>8} | {r['gt_n_points']:>6}"
        )

    # Summary
    valid = [r for r in records if r["speedup"]]
    if valid:
        mean_speedup = np.mean([r["speedup"] for r in valid])
        max_speedup = max(r["speedup"] for r in valid)
        print(f"\n  Mean speedup: {mean_speedup:.0f}×")
        print(
            f"  Max speedup:  {max_speedup:.0f}× (N={max(valid, key=lambda x: x['speedup'])['n_qubits']})"
        )
        print(
            "\n  Thesis claim: GNN amortizes O(100) DMRG points into O(∞) predictions at ~1ms each."
        )


def save_results(records: list[dict], topology: str) -> Path:
    """Save results to JSON."""
    from qmbp_simulation.utils.helpers import json_serialize

    out_dir = _project_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"walltime_comparison_{topology}.json"

    output = {
        "topology": topology,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
        "summary": {
            "n_values_with_speedup": len([r for r in records if r["speedup"]]),
            "mean_speedup": float(np.mean([r["speedup"] for r in records if r["speedup"]]))
            if any(r["speedup"] for r in records)
            else None,
            "max_speedup": float(max((r["speedup"] for r in records if r["speedup"]), default=0))
            or None,
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Wall-time comparison: DMRG/ED vs MPNN inference")
    parser.add_argument(
        "--topology",
        type=str,
        default="chain_1d",
        help="Topology to analyze (default: chain_1d)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Re-compute a sample of GT points to measure timing",
    )
    parser.add_argument(
        "--sample-per-n",
        type=int,
        default=5,
        help="Points per N to backfill timing (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfim",
        help="Hamiltonian model (default: tfim)",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=1,
        help="HVA depth for MPNN (default: 1)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON (default: True)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    topology = args.topology

    # ── Step 1: Get GT timing ─────────────────────────────────────────────
    print(f"\n  Topology: {topology}")

    gt_times = extract_gt_timing(topology)
    n_existing = sum(len(v) for v in gt_times.values())
    print(f"  GT cache entries with time_s: {n_existing}")

    if args.backfill or n_existing == 0:
        print(f"  Backfilling GT timing (sample_per_n={args.sample_per_n})...")
        backfill_times = backfill_gt_timing(
            topology=topology,
            sample_per_n=args.sample_per_n,
            model=args.model,
        )
        # Merge (backfill takes priority for fresh measurements)
        for n, times in backfill_times.items():
            gt_times[n] = times
        print(f"  Backfilled: {sum(len(v) for v in backfill_times.values())} points")

    # ── Step 2: Measure MPNN inference ────────────────────────────────────
    n_values = sorted(gt_times.keys()) if gt_times else [4, 6, 8, 10, 16, 20]
    print(f"  Measuring MPNN inference for N={n_values}...")

    mpnn_times = measure_mpnn_inference(
        topology=topology,
        n_qubits_list=n_values,
        p_layers=args.p_layers,
    )

    # ── Step 3: Build comparison ──────────────────────────────────────────
    records = build_comparison_table(gt_times, mpnn_times)
    print_table(records)

    # ── Step 4: Save ──────────────────────────────────────────────────────
    if args.save:
        save_results(records, topology)


if __name__ == "__main__":
    main()
