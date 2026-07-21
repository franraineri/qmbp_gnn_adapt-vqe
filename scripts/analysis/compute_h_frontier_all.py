#!/usr/bin/env python3
"""Compute h_frontier matrix across ALL topologies and models.

Scans exp_noiseless/ and exp_scaling/ for VQE results with frontier data.
Produces separate matrices for:
- TFIM by topology (chain_1d, ladder, heavy_hex, square, triangular)
- Multiple models by model name (tfim, heisenberg, heisenberg_transverse, kitaev)

Usage:
    python scripts/analysis/compute_h_frontier_all.py
    python scripts/analysis/compute_h_frontier_all.py --model tfim --topology chain_1d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Compute h_frontier across topologies/models")
    parser.add_argument("--model", type=str, default=None, help="Filter by model")
    parser.add_argument("--topology", type=str, default=None, help="Filter by topology")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def interpolate_frontier(h_values, de_gaps, threshold):
    """Linear interpolation at the pass/fail crossing."""
    for i in range(len(h_values) - 1):
        h_high, h_low = h_values[i], h_values[i + 1]
        dg_high, dg_low = de_gaps[i], de_gaps[i + 1]
        if not (np.isfinite(dg_high) and np.isfinite(dg_low)):
            continue
        if dg_high < threshold <= dg_low:
            denom = dg_low - dg_high
            if denom < 1e-15:
                return float((h_high + h_low) / 2)
            frac = (threshold - dg_high) / denom
            return float(h_high - frac * (h_high - h_low))
    return None


def scan_results(base_dirs: list[Path], model_filter=None, topo_filter=None):
    """Scan result directories and extract frontier data."""
    # Key: (model, topology, N, p) -> list of frontier values
    data: dict[tuple[str, str, int, int], list[float]] = {}

    for base in base_dirs:
        if not base.exists():
            continue
        for f in sorted(base.rglob("run_*.json")):
            try:
                with open(f) as fp:
                    d = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue

            cfg = d.get("config", {}).get("system", {})
            n = cfg.get("n_qubits")
            p = cfg.get("p_layers")
            model = cfg.get("model", "tfim")
            topology = cfg.get("topology")
            if isinstance(topology, list):
                topology = topology[0] if topology else None

            # Also check 'topologies' field (NoiselessPipeline format)
            if not topology:
                topos_list = cfg.get("topologies", [])
                if topos_list:
                    topology = topos_list[0] if isinstance(topos_list, list) else topos_list

            if not all([n, p]):
                continue
            if model_filter and model != model_filter:
                continue
            if topo_filter and topology and topology != topo_filter:
                continue

            # Find section with VQE results
            results = d.get("results", {})
            s2 = None
            for key in ["section_2", "section_3", "section_4"]:
                candidate = results.get(key, {}).get("data", {})
                if candidate and "per_seed" in candidate:
                    s2 = candidate
                    break
                # NoiselessPipeline format: topologies.<topo>.per_point
                if candidate and "topologies" in candidate:
                    for topo_key, topo_data in candidate.get("topologies", {}).items():
                        if "per_point" in topo_data:
                            pts = topo_data["per_point"]
                            h_vals = [r["h"] for r in pts]
                            de_gaps_vals = [r.get("de_gap", 1.0) for r in pts]
                            frontier = interpolate_frontier(h_vals, de_gaps_vals, THRESHOLD)
                            if frontier is not None:
                                actual_topo = topo_key if topo_key != topology else topology
                                fkey = (model, actual_topo, n, p)
                                if fkey not in data:
                                    data[fkey] = []
                                data[fkey].append(frontier)
                    continue
            if not s2:
                continue

            for seed_data in s2["per_seed"]:
                pts = seed_data["results"]
                h_vals = [r["h"] for r in pts]
                de_gaps_vals = [r.get("de_gap", 1.0) for r in pts]
                frontier = interpolate_frontier(h_vals, de_gaps_vals, THRESHOLD)
                if frontier is not None:
                    key = (model, topology, n, p)
                    if key not in data:
                        data[key] = []
                    data[key].append(frontier)

    return data


def print_matrix(data, group_by="topology"):
    """Print h_frontier matrices grouped by topology or model."""
    if not data:
        print("No frontier data found.")
        return

    # Group
    groups = set()
    for key in data:
        if group_by == "topology":
            groups.add((key[0], key[1]))  # (model, topology)
        else:
            groups.add(key[0])  # model

    for group in sorted(groups):
        if group_by == "topology":
            model, topo = group
            subset = {k: v for k, v in data.items() if k[0] == model and k[1] == topo}
            title = f"{model} / {topo}"
        else:
            subset = {k: v for k, v in data.items() if k[0] == group}
            title = f"model={group}"

        if not subset:
            continue

        ns = sorted(set(k[2] for k in subset))
        ps = sorted(set(k[3] for k in subset))

        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        header = f"{'N':>5} " + "  ".join(f"{'p=' + str(p):>7}" for p in ps)
        print(header)
        print("-" * (6 + 9 * len(ps)))

        for n in ns:
            row = f"{n:>5} "
            for p in ps:
                key = (
                    group[0] if group_by == "topology" else group,
                    group[1] if group_by == "topology" else "",
                    n,
                    p,
                )
                # Try both key formats
                vals = data.get(key, [])
                if not vals:
                    # Try without topology for model-grouped
                    for k, v in subset.items():
                        if k[2] == n and k[3] == p:
                            vals = v
                            break
                if vals:
                    median = np.median(vals)
                    row += f"  {median:5.2f} "
                else:
                    row += "      - "
            print(row)


def main():
    args = parse_args()

    scan_dirs = [
        ROOT / "results" / "experiments" / "exp_noiseless",
        ROOT / "results" / "experiments" / "exp_scaling" / "validation",
    ]

    data = scan_results(scan_dirs, model_filter=args.model, topo_filter=args.topology)

    print(f"h_frontier matrices (ΔE/gap = {args.threshold * 100:.0f}% crossing)")
    print(f"Scanned: {sum(1 for d in scan_dirs if d.exists())} directories")
    print(f"Total (model, topo, N, p) combinations: {len(data)}")

    # Print per (model, topology)
    print_matrix(data, group_by="topology")


if __name__ == "__main__":
    main()
