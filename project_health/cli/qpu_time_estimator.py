#!/usr/bin/env python3
"""QPU time estimation per circuit using actual transpiled circuit properties.

Reads seed=100 envelope data for each (config, h-value) combination and
estimates QPU execution time based on:
  - n_2q_gates, depth_2q (from circuit_stats)
  - Mitigation method overhead (PEA noise learning, ZNE noise factors)
  - IBM Kingston CLOPS throughput model (depth-aware)

This gives a PER-CIRCUIT estimate, not a per-config average — showing how
circuit structure (which varies with h for AQC) affects hardware time.

Usage:
    # Default (5 configs, h=3.25-4.0, seed=100)
    python analyze_aqc.py

    # Custom h-values
    python analyze_aqc.py --h-values 1.5,2.0,3.25,3.5,3.75,4.0

    # Custom configs
    python analyze_aqc.py --configs C0_raw,C5_full_pea_balanced

    # Save to file
    python analyze_aqc.py --output qpu_estimate.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ─── IBM Kingston throughput model ─────────────────────────────────────────
# Reference: QPUThroughputProfile.ibm_kingston() in preflight.py
# Base CLOPS = 3750 at ref_depth=60, ref_n_qubits=100
# Depth scaling: effective_clops = base_clops × (ref_depth / actual_depth)^0.3

BASE_CLOPS = 3750
REF_DEPTH = 60
DEPTH_EXPONENT = 0.3
SHOTS = 16384

# Mitigation overhead model (calibrated from QPU measurements):
#   - Raw/DD: 2× (compilation + job dispatch)
#   - PEA: 7× (Pauli-Lindblad fitting + amplification + scheduling)
#   plus PEA noise learning: num_randomizations × shots_per_randomization / CLOPS
OVERHEAD_FACTORS = {
    "none": 2.0,  # C0_raw, C1_dd_only
    "pea_light": 7.0,  # C4: 32×128 = 4K learning shots
    "pea_balanced": 7.0,  # C5: 48×192 = 9K learning shots
    "pea_aqc": 7.0,  # C16: same PEA as balanced
}

PEA_LEARNING_SHOTS = {
    "none": 0,
    "pea_light": 32 * 128,  # 4,096
    "pea_balanced": 48 * 192,  # 9,216
    "pea_aqc": 48 * 192,  # 9,216
}

# ZNE noise factors (determines how many circuit variants are run)
ZNE_FACTORS = {
    "none": 1,
    "pea_light": 3,  # [1, 1.5, 3]
    "pea_balanced": 3,  # [1, 1.5, 3]
    "pea_aqc": 3,
}

CONFIG_METHOD = {
    "C0_raw": "none",
    "C1_dd_only": "none",
    "C4_full_pea_light": "pea_light",
    "C5_full_pea_balanced": "pea_balanced",
    "C16_aqc_pea": "pea_aqc",
}


def compute_effective_clops(depth: int) -> float:
    """Depth-aware CLOPS: deeper circuits → lower throughput."""
    if depth <= 0:
        return BASE_CLOPS
    return BASE_CLOPS * (REF_DEPTH / depth) ** DEPTH_EXPONENT


def estimate_circuit_qpu_time(
    depth: int,
    n_2q: int,
    method: str,
    shots: int = SHOTS,
) -> dict[str, float]:
    """Estimate QPU time for one circuit execution.

    Returns breakdown: base_time, zne_time, pea_learning_time, total with overhead.
    """
    eff_clops = compute_effective_clops(depth)
    overhead = OVERHEAD_FACTORS[method]
    n_factors = ZNE_FACTORS[method]
    pea_shots = PEA_LEARNING_SHOTS[method]

    # Base circuit execution time
    base_time_s = shots / eff_clops

    # ZNE: run circuit at multiple noise factors
    zne_time_s = base_time_s * n_factors

    # PEA noise learning (one-time per job, amortized per h-point)
    pea_learning_s = pea_shots / eff_clops if pea_shots > 0 else 0.0

    # Total with IBM overhead
    total_s = (zne_time_s + pea_learning_s) * overhead

    return {
        "effective_clops": eff_clops,
        "base_time_s": base_time_s,
        "zne_time_s": zne_time_s,
        "pea_learning_s": pea_learning_s,
        "overhead_factor": overhead,
        "total_s": total_s,
    }


# ─── Load real circuit data from seed=100 results ─────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="QPU time estimation per circuit")
    parser.add_argument(
        "--configs",
        type=str,
        default="C0_raw,C1_dd_only,C4_full_pea_light,C5_full_pea_balanced,C16_aqc_pea",
        help="CSV of configs",
    )
    parser.add_argument("--h-values", type=str, default="3.25,3.5,3.75,4.0", help="CSV of h-values")
    parser.add_argument("--seed", type=int, default=100, help="Seed")
    parser.add_argument(
        "--mode", type=str, default="fake_backend", choices=["fake_backend", "hardware"]
    )
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    return parser.parse_args()


args = parse_args()
base = Path("results/mitigation_benchmark") / args.mode
configs = [c.strip() for c in args.configs.split(",")]
h_values = [float(h.strip()) for h in args.h_values.split(",")]
seed = args.seed

output_lines: list[str] = []


def out(line: str = "") -> None:
    """Print and collect output."""
    print(line)
    output_lines.append(line)


out("=" * 90)
out("  QPU TIME PER CIRCUIT — Based on Actual Transpiled Properties")
out(f"  Backend: IBM Kingston (Heron R2, CLOPS={BASE_CLOPS}@depth={REF_DEPTH})")
out(f"  Seed: {seed} | Mode: {args.mode} | h-values: {h_values}")
out("=" * 90)

# Collect data per (config, h)
all_rows = []
out(
    f"\n{'Config':<25s} {'h':>5s} {'n_2q':>5s} {'d_2q':>5s} {'depth':>6s} "
    f"{'CLOPS':>6s} {'Base(s)':>7s} {'ZNE(s)':>7s} {'PEA(s)':>7s} "
    f"{'Total(s)':>8s} {'ΔE/gap':>8s}"
)
out("-" * 100)

for cfg in configs:
    method = CONFIG_METHOD.get(cfg, "none")
    for h in h_values:
        h_str = f"h{str(h).replace('.', 'p')}"
        if seed == 42:
            files = sorted(base.glob(f"{cfg}/{h_str}_run_*.json"))
            files = [f for f in files if "_seed" not in f.stem]
        else:
            files = sorted(base.glob(f"{cfg}/{h_str}_*seed{seed}.json"))
        if not files:
            continue
        data = json.loads(files[-1].read_text())
        cs = data.get("circuit_stats", {})
        results = data.get("results", {})
        if results.get("e_raw") is None and results.get("e_mitigated") is None:
            continue

        n_2q = int(cs.get("n_2q_gates", 18))
        d_2q = int(cs.get("depth_2q", 14))
        depth = int(cs.get("depth", 60))
        de_gap = results.get("delta_e_gap")

        est = estimate_circuit_qpu_time(depth, n_2q, method)

        de_str = f"{de_gap:.4f}" if de_gap is not None else "N/A"
        out(
            f"  {cfg:<23s} {h:>5.2f} {n_2q:>5d} {d_2q:>5d} {depth:>6d} "
            f"{est['effective_clops']:>6.0f} {est['base_time_s']:>7.2f} "
            f"{est['zne_time_s']:>7.2f} {est['pea_learning_s']:>7.2f} "
            f"{est['total_s']:>8.1f} {de_str:>8s}"
        )

        all_rows.append(
            {
                "config": cfg,
                "h": h,
                "n_2q": n_2q,
                "d_2q": d_2q,
                "depth": depth,
                "method": method,
                "total_s": est["total_s"],
                "de_gap": de_gap,
            }
        )

# ─── Summary by config (across all h-values) ──────────────────────────────

out("\n" + "=" * 90)
out("  SUMMARY BY CONFIG (sum across all h-values)")
out("=" * 90)

from collections import defaultdict

by_config = defaultdict(list)
for r in all_rows:
    by_config[r["config"]].append(r)

out(
    f"\n{'Config':<25s} {'N runs':>6s} {'Mean n_2q':>9s} {'Mean depth':>10s} "
    f"{'Per-h (s)':>9s} {'Total (s)':>9s} {'Total (min)':>10s} {'Mean ΔE/gap':>11s}"
)
out("-" * 95)

grand_total = 0.0
for cfg in configs:
    rows = by_config.get(cfg, [])
    if not rows:
        continue
    n = len(rows)
    mean_n2q = np.mean([r["n_2q"] for r in rows])
    mean_depth = np.mean([r["depth"] for r in rows])
    total_time = sum(r["total_s"] for r in rows)
    per_h = total_time / n if n > 0 else 0
    de_vals = [r["de_gap"] for r in rows if r["de_gap"] is not None]
    mean_de = np.mean(de_vals) if de_vals else None
    grand_total += total_time

    de_str = f"{mean_de:.4f}" if mean_de is not None else "N/A"
    out(
        f"  {cfg:<23s} {n:>6d} {mean_n2q:>9.0f} {mean_depth:>10.0f} "
        f"{per_h:>9.1f} {total_time:>9.1f} {total_time / 60:>10.1f} {de_str:>11s}"
    )

out(
    f"\n  {'GRAND TOTAL':<23s} {len(all_rows):>6d} {'':>9s} {'':>10s} "
    f"{'':>9s} {grand_total:>9.1f} {grand_total / 60:>10.1f}"
)

# ─── Comparison: depth effect on time ──────────────────────────────────────

out("\n" + "=" * 90)
out("  DEPTH EFFECT ON QPU TIME (C16 AQC: variable depth vs h)")
out("=" * 90)

c16_rows = by_config.get("C16_aqc_pea", [])
if c16_rows:
    out(
        f"\n{'h':>6s} {'n_2q':>5s} {'depth':>6s} {'CLOPS_eff':>9s} {'Time(s)':>8s} "
        f"{'vs C5 time':>10s} {'vs C5 ΔE':>10s}"
    )
    out("-" * 60)

    # Get C5 times for comparison
    c5_by_h = {r["h"]: r for r in by_config.get("C5_full_pea_balanced", [])}

    for r in sorted(c16_rows, key=lambda x: x["h"]):
        eff_clops = compute_effective_clops(r["depth"])
        c5_ref = c5_by_h.get(r["h"])
        ratio_time = r["total_s"] / c5_ref["total_s"] if c5_ref else None
        ratio_de = ""
        if r["de_gap"] is not None and c5_ref and c5_ref["de_gap"] is not None:
            if c5_ref["de_gap"] > 0:
                ratio_de = f"{r['de_gap'] / c5_ref['de_gap']:.2f}×"

        ratio_str = f"{ratio_time:.2f}×" if ratio_time else "N/A"
        print(
            f"  {r['h']:>5.2f} {r['n_2q']:>5d} {r['depth']:>6d} {eff_clops:>9.0f} "
            f"{r['total_s']:>8.1f} {ratio_str:>10s} {ratio_de:>10s}"
        )

# ─── Final recommendation ─────────────────────────────────────────────────

out("\n" + "=" * 90)
out("  EXECUTION PLAN RECOMMENDATION")
out("=" * 90)

# Only paramagnetic h-values for hardware
hw_rows = [r for r in all_rows if r["h"] >= 3.0]
hw_total = sum(r["total_s"] for r in hw_rows)
hw_n = len(hw_rows)

out(f"""
  Hardware target: h ∈ {{3.25, 3.5, 3.75, 4.0}} (paramagnetic regime)
  Configs: C0, C1, C4, C5, C16 (5 configs × 4 h = 20 circuits)

  Estimated QPU time (20 circuits):
    Total:     {hw_total:.0f}s = {hw_total / 60:.1f} min
    Per-job:   {hw_total / hw_n:.1f}s average

  Wall-clock estimates:
    Batch mode (1 queue wait):  {hw_total / 60:.0f} min + ~5 min queue = ~{hw_total / 60 + 5:.0f} min
    Sequential (20 queue waits): {hw_total / 60:.0f} min + ~20 min queue = ~{hw_total / 60 + 20:.0f} min

  Cost at $1.60/min QPU:  ${hw_total / 60 * 1.6:.0f} USD
""")

# ─── Save to file if requested ─────────────────────────────────────────────

if args.output:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines))
    print(f"\n  → Saved to {out_path}", file=sys.stderr)
