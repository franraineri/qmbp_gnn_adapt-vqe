#!/usr/bin/env python3
"""Compute precise h_frontier values via linear interpolation.

h_frontier is the exact h-value where ΔE/gap crosses the threshold (default 5%),
obtained by linear interpolation between the last passing point and first
failing point in each VQE sweep.

Outputs both a formatted matrix and linear fit coefficients for h_frontier(N, p).

Usage:
    .venv/bin/python scripts/analysis/compute_h_frontier.py
    .venv/bin/python scripts/analysis/compute_h_frontier.py --threshold 0.03
    .venv/bin/python scripts/analysis/compute_h_frontier.py --json
    .venv/bin/python scripts/analysis/compute_h_frontier.py --min-n 20  # exclude N<20 from fits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "results" / "experiments" / "exp_scaling" / "validation" / "tfim" / "chain_1d"
THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Compute precise h_frontier via interpolation")
    parser.add_argument("--dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="ΔE/gap threshold for pass/fail (default: 0.05)",
    )
    parser.add_argument(
        "--min-n", type=int, default=20, help="Minimum N to include in linear fits (default: 20)"
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    return parser.parse_args()


def interpolate_frontier(
    h_values: list[float],
    de_gaps: list[float],
    threshold: float,
) -> float | None:
    """Find h where ΔE/gap crosses threshold via linear interpolation.

    Scans from high-h to low-h (descending sweep order). Returns the
    interpolated h-value at the first crossing point.

    Returns None if:
    - All points pass (frontier is below the tested range)
    - All points fail (frontier is above the tested range)
    - de_gaps are not monotonically crossing (noise/non-monotonic)
    """
    if len(h_values) < 2:
        return None

    for i in range(len(h_values) - 1):
        h_high, h_low = h_values[i], h_values[i + 1]
        dg_high, dg_low = de_gaps[i], de_gaps[i + 1]

        # Guard: skip NaN/Inf values
        if not (np.isfinite(dg_high) and np.isfinite(dg_low)):
            continue

        # Crossing: high passes, low fails
        if dg_high < threshold <= dg_low:
            denominator = dg_low - dg_high
            if denominator < 1e-15:
                # Degenerate case: both values at threshold
                return float((h_high + h_low) / 2)
            frac = (threshold - dg_high) / denominator
            h_cross = h_high - frac * (h_high - h_low)
            return float(h_cross)

    return None


def main():
    args = parse_args()
    threshold = args.threshold
    base = args.dir

    if not base.exists():
        print(f"ERROR: Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    files = sorted(base.glob("run_*.json"))
    if not files:
        print(f"ERROR: No result files in {base}", file=sys.stderr)
        sys.exit(1)

    # Collect frontier values per (N, p) across all runs and seeds
    matrix: dict[tuple[int, int], list[float]] = {}
    n_files_used = 0

    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue

        cfg = d.get("config", {}).get("system", {})
        n = cfg.get("n_qubits")
        p = cfg.get("p_layers")
        if not n or not p:
            continue

        s2 = d.get("results", {}).get("section_2", {}).get("data", {})
        if not s2 or "per_seed" not in s2:
            continue

        n_files_used += 1

        # Compute frontier for each seed independently
        for seed_data in s2["per_seed"]:
            pts = seed_data["results"]
            h_vals = [r["h"] for r in pts]
            de_gaps_vals = [r.get("de_gap", 1.0) for r in pts]
            frontier = interpolate_frontier(h_vals, de_gaps_vals, threshold)
            if frontier is not None:
                key = (n, p)
                if key not in matrix:
                    matrix[key] = []
                matrix[key].append(frontier)

    if not matrix:
        print("ERROR: No frontier crossings found in any file.", file=sys.stderr)
        sys.exit(1)

    # Compute statistics per (N, p)
    frontier_median: dict[tuple[int, int], float] = {}
    frontier_std: dict[tuple[int, int], float] = {}
    frontier_count: dict[tuple[int, int], int] = {}

    for key, values in matrix.items():
        frontier_median[key] = float(np.median(values))
        frontier_std[key] = float(np.std(values)) if len(values) > 1 else 0.0
        frontier_count[key] = len(values)

    # JSON output
    if args.json:
        out = [
            {
                "n": k[0],
                "p": k[1],
                "h_frontier": frontier_median[k],
                "std": frontier_std[k],
                "n_measurements": frontier_count[k],
            }
            for k in sorted(frontier_median.keys())
        ]
        print(json.dumps(out, indent=2))
        return

    # Print matrix
    ns = sorted(set(k[0] for k in frontier_median))
    ps = sorted(set(k[1] for k in frontier_median))

    print(f"h_frontier(N, p) — interpolated ΔE/gap = {threshold * 100:.0f}% crossing")
    print(f"Source: {base.relative_to(ROOT)} ({n_files_used} files)")
    print("Method: linear interpolation between last PASS and first FAIL")
    print("Stats: median ± std across seeds/runs\n")

    header = f"{'N':>5} " + "  ".join(f"{'p=' + str(p):>9}" for p in ps)
    print(header)
    print("-" * (6 + 11 * len(ps)))
    for n in ns:
        row = f"{n:>5} "
        for p in ps:
            val = frontier_median.get((n, p))
            std = frontier_std.get((n, p), 0)
            cnt = frontier_count.get((n, p), 0)
            if val is not None:
                if std > 0.05:
                    row += f" {val:5.2f}±{std:.2f}"
                elif cnt > 1:
                    row += f"  {val:5.2f}({cnt})"
                else:
                    row += f"  {val:5.2f}   "
            else:
                row += "         -"
        print(row)

    # Fit analysis
    print(f"\n\nLINEAR FIT: h_frontier(N) = a + b·N  (N ≥ {args.min_n})")
    print("-" * 60)
    for p in ps:
        pts_p = [(k[0], v) for k, v in frontier_median.items() if k[1] == p and k[0] >= args.min_n]
        if len(pts_p) < 3:
            print(f"  p={p}: insufficient data ({len(pts_p)} points, need ≥3)")
            continue
        ns_arr = np.array([x[0] for x in pts_p])
        hs_arr = np.array([x[1] for x in pts_p])
        coeffs = np.polyfit(ns_arr, hs_arr, 1)
        residuals = hs_arr - np.polyval(coeffs, ns_arr)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((hs_arr - np.mean(hs_arr)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        print(
            f"  p={p}: h_frontier = {coeffs[1]:.3f} + {coeffs[0]:.5f}·N  "
            f"(R²={r2:.4f}, n={len(pts_p)} points)"
        )

    # Physical interpretation
    print("\n\nPHYSICS SUMMARY:")
    print("  h_c (TFIM critical point) = 1.0")
    for p in ps:
        pts_p = [(k[0], v) for k, v in frontier_median.items() if k[1] == p and k[0] >= args.min_n]
        if pts_p:
            mean_h = np.mean([x[1] for x in pts_p])
            print(
                f"  p={p}: mean h_frontier = {mean_h:.2f} (distance from h_c: {mean_h - 1.0:.2f})"
            )


if __name__ == "__main__":
    main()
