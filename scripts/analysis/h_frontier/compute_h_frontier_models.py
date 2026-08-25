"""Compute precise h_frontier via linear interpolation for model exploration.

h_frontier = exact h where ΔE/gap crosses 5%, interpolated between the last
failing and first passing point. Reports per (model, p).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe")
RESULTS_DIR = ROOT / "results" / "experiments" / "exp_noiseless"

DE_GAP_THRESHOLD = 0.05

# Collect all results
raw_data = defaultdict(list)  # (model, p) -> list of (h, de_gap) from all seeds

for json_file in RESULTS_DIR.rglob("run_2026071*.json"):
    if "chain_1d" not in str(json_file):
        continue
    try:
        with open(json_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue

    config = data.get("config", {})
    system = config.get("system", {})
    n_qubits = system.get("n_qubits", 0)
    p_layers = system.get("p_layers", 0)
    model = system.get("model", "")
    topo_list = system.get("topologies", [])
    topo = topo_list[0] if topo_list else ""

    if n_qubits != 10 or topo != "chain_1d":
        continue
    if p_layers < 2 or p_layers > 8:
        continue

    res = data.get("results", {})
    s2 = res.get("section_2", {})
    s2_data = s2.get("data", {})
    topos = s2_data.get("topologies", {})
    chain_data = topos.get("chain_1d", {})
    vqe_points = chain_data.get("per_point", [])
    if not vqe_points:
        continue

    for pt in vqe_points:
        h = pt.get("h")
        de_gap = pt.get("de_gap")
        fidelity = pt.get("fidelity")
        if h is not None and de_gap is not None:
            raw_data[(model, p_layers)].append({"h": h, "de_gap": de_gap, "fidelity": fidelity})

print(f"Loaded data for {len(raw_data)} (model, p) configurations")
print()


def compute_frontier(points, threshold=0.05):
    """Compute h_frontier via linear interpolation.

    Returns the interpolated h where ΔE/gap crosses threshold,
    between the last failing and first passing point (ascending in h).
    Also returns stats.
    """
    # Group by h, take median de_gap per h
    h_groups = defaultdict(list)
    for pt in points:
        h_groups[round(pt["h"], 4)].append(pt["de_gap"])

    h_vals = sorted(h_groups.keys())
    median_dg = [np.median(h_groups[h]) for h in h_vals]

    # Walk ascending to find crossing
    h_frontier = None
    for i in range(len(h_vals) - 1):
        if median_dg[i] >= threshold and median_dg[i + 1] < threshold:
            # Linear interpolation
            h0, h1 = h_vals[i], h_vals[i + 1]
            dg0, dg1 = median_dg[i], median_dg[i + 1]
            # dg(h) = dg0 + (dg1-dg0)/(h1-h0) * (h - h0) = threshold
            if dg0 != dg1:
                h_frontier = h0 + (threshold - dg0) / (dg1 - dg0) * (h1 - h0)
            else:
                h_frontier = (h0 + h1) / 2
            break

    # If no crossing found, check if all pass or all fail
    if h_frontier is None:
        if all(dg < threshold for dg in median_dg):
            h_frontier = h_vals[0]  # All pass — frontier below sweep
        # else: all fail

    # Stats
    n_total = len(points)
    n_pass = sum(1 for pt in points if pt["de_gap"] < threshold)
    pass_rate = n_pass / n_total if n_total > 0 else 0
    mean_fidelity_pass = (
        np.mean(
            [
                pt["fidelity"]
                for pt in points
                if pt["de_gap"] < threshold and pt["fidelity"] is not None
            ]
        )
        if n_pass > 0
        else 0
    )

    return {
        "h_frontier": h_frontier,
        "pass_rate": pass_rate,
        "n_pass": n_pass,
        "n_total": n_total,
        "mean_fidelity_pass": float(mean_fidelity_pass),
        "min_de_gap": float(min(pt["de_gap"] for pt in points)),
        "h_min_tested": h_vals[0],
        "h_max_tested": h_vals[-1],
    }


# Compute frontiers
models_order = [
    "tfim",
    "tfim_longitudinal",
    "tfim_frustrated",
    "kitaev",
    "heisenberg",
    "heisenberg_transverse",
    "xy",
    "tfim_bond_resolved",
]

results = {}
for key, points in raw_data.items():
    results[key] = compute_frontier(points)

# Print matrix
print("=" * 95)
print("  h_frontier MATRIX — N=10, chain_1d")
print("  Definition: h where ΔE/gap crosses 5% (linear interpolation, median across seeds)")
print("=" * 95)
print()
print(f"{'Model':<22} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}")
print("-" * 95)

for mdl in models_order:
    row = f"{mdl:<22}"
    for p in range(2, 9):
        key = (mdl, p)
        if key in results and results[key]["h_frontier"] is not None:
            row += f" {results[key]['h_frontier']:>6.3f}"
        elif key in results:
            row += "   FAIL"
        else:
            row += "     —"
    print(row)

print("-" * 95)
print()

# Distance from h_c
print("Distance from h_c (h_frontier - h_c):")
print("-" * 95)
H_CRITICAL = {
    "tfim": 1.0,
    "tfim_longitudinal": 1.0,
    "tfim_frustrated": 1.0,
    "kitaev": 2.0,
    "heisenberg": 2.5,
    "heisenberg_transverse": 2.5,
    "xy": 2.0,
    "tfim_bond_resolved": 1.0,
}
for mdl in models_order:
    h_c = H_CRITICAL.get(mdl, 1.0)
    row = f"{mdl:<22}"
    for p in range(2, 9):
        key = (mdl, p)
        if key in results and results[key]["h_frontier"] is not None:
            dist = results[key]["h_frontier"] - h_c
            row += f" {dist:>+6.3f}"
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# Pass rate table
print("Pass rate (fraction of h-points with ΔE/gap < 5%):")
print("-" * 95)
print(f"{'Model':<22} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}")
print("-" * 95)
for mdl in models_order:
    row = f"{mdl:<22}"
    for p in range(2, 9):
        key = (mdl, p)
        if key in results:
            row += f" {results[key]['pass_rate']:>5.0%} "
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# Mean fidelity for passing points
print("Mean fidelity (at passing h-points only):")
print("-" * 95)
print(f"{'Model':<22} {'p=2':>7} {'p=3':>7} {'p=4':>7} {'p=5':>7} {'p=6':>7} {'p=7':>7} {'p=8':>7}")
print("-" * 95)
for mdl in models_order:
    row = f"{mdl:<22}"
    for p in range(2, 9):
        key = (mdl, p)
        if key in results and results[key]["mean_fidelity_pass"] > 0:
            row += f" {results[key]['mean_fidelity_pass']:>6.4f}"
        else:
            row += "     —"
    print(row)
print("-" * 95)
print()

# Best ΔE/gap achieved (min across all h)
print("Best ΔE/gap achieved (minimum across all h-points):")
print("-" * 95)
print(f"{'Model':<22} {'p=2':>8} {'p=3':>8} {'p=4':>8} {'p=5':>8} {'p=6':>8} {'p=7':>8} {'p=8':>8}")
print("-" * 95)
for mdl in models_order:
    row = f"{mdl:<22}"
    for p in range(2, 9):
        key = (mdl, p)
        if key in results:
            row += f" {results[key]['min_de_gap']:>7.1e}"
        else:
            row += "      —"
    print(row)
print("-" * 95)
