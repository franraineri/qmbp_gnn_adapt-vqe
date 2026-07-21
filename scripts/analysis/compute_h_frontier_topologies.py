"""Compute h_frontier for TFIM_longitudinal across all topologies, N=10, P=2..8.

h_frontier = exact h where ΔE/gap crosses 5%, via linear interpolation between
last failing and first passing point. Median across seeds.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe")
RESULTS_DIR = ROOT / "results" / "experiments" / "exp_noiseless" / "tfim_longitudinal"

DE_GAP_THRESHOLD = 0.05

# Collect all results
raw_data = defaultdict(list)  # (topology, p) -> list of {h, de_gap, fidelity}

for json_file in RESULTS_DIR.rglob("run_2026071*.json"):
    try:
        with open(json_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue

    config = data.get("config", {})
    system = config.get("system", {})
    n_qubits = system.get("n_qubits", 0)
    p_layers = system.get("p_layers", 0)
    topo_list = system.get("topologies", [])
    topo = topo_list[0] if topo_list else ""

    if n_qubits != 10:
        continue
    if p_layers < 2 or p_layers > 8:
        continue

    res = data.get("results", {})
    s2 = res.get("section_2", {})
    s2_data = s2.get("data", {})
    topos = s2_data.get("topologies", {})

    # Try the topology name directly
    topo_data = topos.get(topo, {})
    vqe_points = topo_data.get("per_point", [])
    if not vqe_points:
        continue

    for pt in vqe_points:
        h = pt.get("h")
        de_gap = pt.get("de_gap")
        fidelity = pt.get("fidelity")
        if h is not None and de_gap is not None:
            raw_data[(topo, p_layers)].append({"h": h, "de_gap": de_gap, "fidelity": fidelity})

print(f"Loaded data for {len(raw_data)} (topology, p) configurations")
for key in sorted(raw_data.keys()):
    print(f"  {key[0]:12s} p={key[1]}: {len(raw_data[key]):3d} points")
print()


def compute_frontier(points, threshold=0.05):
    """Compute h_frontier via linear interpolation."""
    h_groups = defaultdict(list)
    for pt in points:
        h_groups[round(pt["h"], 4)].append(pt["de_gap"])

    h_vals = sorted(h_groups.keys())
    median_dg = [np.median(h_groups[h]) for h in h_vals]

    h_frontier = None
    for i in range(len(h_vals) - 1):
        if median_dg[i] >= threshold and median_dg[i + 1] < threshold:
            h0, h1 = h_vals[i], h_vals[i + 1]
            dg0, dg1 = median_dg[i], median_dg[i + 1]
            if dg0 != dg1:
                h_frontier = h0 + (threshold - dg0) / (dg1 - dg0) * (h1 - h0)
            else:
                h_frontier = (h0 + h1) / 2
            break

    if h_frontier is None:
        if all(dg < threshold for dg in median_dg):
            h_frontier = h_vals[0]

    n_total = len(points)
    n_pass = sum(1 for pt in points if pt["de_gap"] < threshold)
    pass_rate = n_pass / n_total if n_total > 0 else 0
    fid_pass = [
        pt["fidelity"] for pt in points if pt["de_gap"] < threshold and pt["fidelity"] is not None
    ]
    mean_fidelity_pass = float(np.mean(fid_pass)) if fid_pass else 0

    return {
        "h_frontier": h_frontier,
        "pass_rate": pass_rate,
        "n_pass": n_pass,
        "n_total": n_total,
        "mean_fidelity_pass": mean_fidelity_pass,
        "min_de_gap": float(min(pt["de_gap"] for pt in points)),
        "h_min_tested": h_vals[0],
        "h_max_tested": h_vals[-1],
    }


# Compute frontiers
topos_order = ["chain_1d", "ladder", "heavy_hex", "square", "kagome", "triangular"]

results = {}
for key, points in raw_data.items():
    results[key] = compute_frontier(points)

# ══════════════════════════════════════════════════════════════════
# MATRIX TABLE
# ══════════════════════════════════════════════════════════════════
print("=" * 95)
print("  h_frontier MATRIX — TFIM_longitudinal, N=10, all topologies")
print("  Definition: h where ΔE/gap crosses 5% (linear interpolation, median across seeds)")
print("=" * 95)
print()
print(
    f"{'Topology':<14} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}  {'edges':>5}"
)
print("-" * 95)

# Edge counts at N=10 for reference
from qmbp_simulation import make_lattice

edge_counts = {}
for topo in topos_order:
    lat = make_lattice(topo, 10, J=1.0, h=1.0)
    edge_counts[topo] = len(lat.edges)

for topo in topos_order:
    row = f"{topo:<14}"
    for p in range(2, 9):
        key = (topo, p)
        if key in results and results[key]["h_frontier"] is not None:
            row += f" {results[key]['h_frontier']:>6.3f}"
        elif key in results:
            row += "   FAIL"
        else:
            row += "     —"
    row += f"  {edge_counts[topo]:>5}"
    print(row)

print("-" * 95)
print()

# ══════════════════════════════════════════════════════════════════
# DISTANCE FROM h_c
# ══════════════════════════════════════════════════════════════════
print("Distance from h_c=1.0 (h_frontier - h_c):")
print("-" * 95)
for topo in topos_order:
    row = f"{topo:<14}"
    for p in range(2, 9):
        key = (topo, p)
        if key in results and results[key]["h_frontier"] is not None:
            dist = results[key]["h_frontier"] - 1.0
            row += f" {dist:>+6.3f}"
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# ══════════════════════════════════════════════════════════════════
# PASS RATE TABLE
# ══════════════════════════════════════════════════════════════════
print("Pass rate (fraction of h-points with ΔE/gap < 5%):")
print("-" * 95)
print(
    f"{'Topology':<14} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}"
)
print("-" * 95)
for topo in topos_order:
    row = f"{topo:<14}"
    for p in range(2, 9):
        key = (topo, p)
        if key in results:
            row += f" {results[key]['pass_rate']:>5.0%} "
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# ══════════════════════════════════════════════════════════════════
# MEAN FIDELITY
# ══════════════════════════════════════════════════════════════════
print("Mean fidelity (at passing h-points only):")
print("-" * 95)
print(
    f"{'Topology':<14} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}"
)
print("-" * 95)
for topo in topos_order:
    row = f"{topo:<14}"
    for p in range(2, 9):
        key = (topo, p)
        if key in results and results[key]["mean_fidelity_pass"] > 0:
            row += f" {results[key]['mean_fidelity_pass']:>6.4f}"
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# ══════════════════════════════════════════════════════════════════
# BEST ΔE/gap
# ══════════════════════════════════════════════════════════════════
print("Best ΔE/gap achieved (minimum across all h-points):")
print("-" * 95)
print(
    f"{'Topology':<14} {'p=2':>8} {'p=3':>8} {'p=4':>8} {'p=5':>8} {'p=6':>8} {'p=7':>8} {'p=8':>8}"
)
print("-" * 95)
for topo in topos_order:
    row = f"{topo:<14}"
    for p in range(2, 9):
        key = (topo, p)
        if key in results:
            row += f" {results[key]['min_de_gap']:>7.1e}"
        else:
            row += "      —"
    print(row)
print("-" * 95)
print()

# ══════════════════════════════════════════════════════════════════
# DETAILED PER-H BREAKDOWN
# ══════════════════════════════════════════════════════════════════
print("=" * 95)
print("  DETAILED PER-H BREAKDOWN")
print("=" * 95)

for topo in topos_order:
    has_data = any((topo, p) in results for p in range(2, 9))
    if not has_data:
        continue

    print(f"\n{'─' * 80}")
    print(f"  {topo} ({edge_counts[topo]} edges)")
    print(f"{'─' * 80}")

    for p in range(2, 9):
        key = (topo, p)
        if key not in raw_data or not raw_data[key]:
            continue

        pts = sorted(raw_data[key], key=lambda x: x["h"])
        h_groups = defaultdict(list)
        for pt in pts:
            h_groups[round(pt["h"], 3)].append(pt["de_gap"])

        r = results[key]
        hf = r["h_frontier"]
        hf_str = f"{hf:.3f}" if hf else "FAIL"
        print(
            f"  P={p}: h_front={hf_str:<7} pass={r['pass_rate']:.0%} ({r['n_pass']}/{r['n_total']})"
        )

        h_summary = []
        for h_val in sorted(h_groups.keys()):
            dg_list = h_groups[h_val]
            n_pass = sum(1 for dg in dg_list if dg < DE_GAP_THRESHOLD)
            n_tot = len(dg_list)
            avg_dg = sum(dg_list) / len(dg_list)
            status = "✓" if n_pass == n_tot else ("~" if n_pass > 0 else "✗")
            h_summary.append(f"{h_val:.2f}:{status}({avg_dg:.1e})")

        for i in range(0, len(h_summary), 5):
            chunk = "  ".join(h_summary[i : i + 5])
            print(f"        {chunk}")

print()
