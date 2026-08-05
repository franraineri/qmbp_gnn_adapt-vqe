#!/usr/bin/env python3
"""Deep analysis of ladder NPZ training data — per-h quality + theta structure."""
import sys
import numpy as np
from pathlib import Path

ROOT = Path.cwd()
NPZ_DIR = ROOT / "data" / "multi_n_training"

topo_npz = sorted(NPZ_DIR.glob("ladder_N*.npz"))

if not topo_npz:
    print(f"No ladder NPZ files found in {NPZ_DIR}")
    sys.exit(1)

print("=" * 70)
print("LADDER — DETAILED NPZ ANALYSIS (best theta_opt per N)")
print("=" * 70)

for npz_file in topo_npz:
    data = np.load(npz_file, allow_pickle=True)
    h_vals = data["h_values"]
    theta = data["theta_opt"]
    n_pts = len(h_vals)
    n_params = theta.shape[1] if theta.ndim == 2 else 0
    N = int(npz_file.stem.split("_N")[1].split("_")[0])

    e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
    e_vqe = data[e_key] if e_key else np.zeros(n_pts)
    e_exact = data["e_exact"] if "e_exact" in data else np.zeros(n_pts)
    abs_err = np.abs(e_vqe - e_exact)

    if "de_gaps" in data:
        de_gaps = np.array(data["de_gaps"], dtype=float)
    elif "gaps" in data:
        gaps = np.array(data["gaps"], dtype=float)
        de_gaps = abs_err / np.maximum(gaps, 1e-10)
    else:
        de_gaps = abs_err

    methods = data["method"] if "method" in data else np.array(["?"] * n_pts)

    print(f"\n{'-' * 70}")
    print(f"N={N} | {npz_file.name} | {n_pts} pts, {n_params} params")
    print(f"{'-' * 70}")
    print(f"  {'h':>6} {'E_vqe':>12} {'E_exact':>12} {'|dE|':>10} {'dE/gap':>8} {'Pass':>5} Method")
    print(f"  {'---':>6} {'---':>12} {'---':>12} {'---':>10} {'---':>8} {'---':>5} {'---':>10}")

    sort_idx = np.argsort(h_vals)
    for i in sort_idx:
        h = h_vals[i]
        ev = e_vqe[i]
        ex = e_exact[i]
        ae = abs_err[i]
        dg = de_gaps[i]
        m = str(methods[i]) if i < len(methods) else "?"
        status = "OK" if dg < 0.05 else ("~" if dg < 0.10 else "X")
        print(f"  {h:>6.3f} {ev:>12.6f} {ex:>12.6f} {ae:>10.2e} {dg:>8.4f} {status:>5} {m}")

    # theta statistics
    print(f"\n  theta_opt statistics:")
    print(f"    Range: [{theta.min():.4f}, {theta.max():.4f}]")
    print(f"    Mean |theta|: {np.mean(np.abs(theta)):.4f}")
    if n_pts > 1:
        theta_sorted = theta[sort_idx]
        diffs = np.abs(np.diff(theta_sorted, axis=0))
        max_jump = diffs.max()
        mean_jump = diffs.mean()
        print(f"    Max consecutive jump: {max_jump:.4f}")
        print(f"    Mean consecutive jump: {mean_jump:.4f}")
        smoothness = "smooth" if max_jump < 0.5 else ("moderate" if max_jump < 1.0 else "discontinuous")
        print(f"    Smoothness: {smoothness}")

    # Summary
    n_pass5 = int(np.sum(de_gaps < 0.05))
    n_pass10 = int(np.sum(de_gaps < 0.10))
    print(f"\n  Summary: pass@5%={n_pass5}/{n_pts} ({n_pass5/n_pts:.0%}), "
          f"pass@10%={n_pass10}/{n_pts} ({n_pass10/n_pts:.0%})")
    print(f"  Mean |dE|={abs_err.mean():.4f}, Mean dE/gap={de_gaps.mean():.4f}")

    passing_h = h_vals[de_gaps < 0.05]
    if len(passing_h) > 0:
        print(f"  Safe zone: h in [{passing_h.min():.2f}, {passing_h.max():.2f}]")
    failing_h = h_vals[de_gaps >= 0.05]
    if len(failing_h) > 0:
        print(f"  Failing h: {[f'{h:.2f}' for h in sorted(failing_h)]}")


print(f"\n{'=' * 70}")
print("CROSS-N SCALING SUMMARY")
print(f"{'=' * 70}")
print(f"  {'N':>4} {'Pts':>4} {'Params':>6} {'Pass5%':>8} {'h_min':>6} {'h_max':>6} {'Mean_dE_gap':>12}")
for npz_file in topo_npz:
    data = np.load(npz_file, allow_pickle=True)
    h_vals = data["h_values"]
    theta = data["theta_opt"]
    n_pts = len(h_vals)
    n_params = theta.shape[1] if theta.ndim == 2 else 0
    N = int(npz_file.stem.split("_N")[1].split("_")[0])
    e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
    abs_err = np.abs(data[e_key] - data["e_exact"]) if e_key and "e_exact" in data else np.zeros(n_pts)
    if "de_gaps" in data:
        de_gaps = np.array(data["de_gaps"], dtype=float)
    elif "gaps" in data:
        de_gaps = abs_err / np.maximum(np.array(data["gaps"], dtype=float), 1e-10)
    else:
        de_gaps = abs_err
    n_pass = int(np.sum(de_gaps < 0.05))
    print(f"  {N:>4} {n_pts:>4} {n_params:>6} {n_pass}/{n_pts:<5} {h_vals.min():>6.2f} {h_vals.max():>6.2f} {de_gaps.mean():>12.4f}")
