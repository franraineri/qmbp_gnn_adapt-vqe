#!/usr/bin/env python3
"""Detailed per-observable error comparison: noisy vs QESEM mitigated."""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

RESULT_PATH = (
    _ROOT
    / "results/recovered/qesem-tier1/qesem_recovered_d628a502-677a-4610-a78c-3d5266c0cdbf.json"
)

with open(RESULT_PATH) as f:
    data = json.load(f)

h_value = 3.5
lattice = make_lattice("heavy_hex", 10, h=h_value, J=1.0)
H = HamiltonianBuilder().build(lattice)
gt = ClassicalSolver().solve(H, lattice)

meta = data["metadata"]
results_obs = meta["results"][0]

exact_x = list(gt.per_site_mag_x)
exact_zz = list(gt.per_bond_corr_zz)
exact_vals = [gt.ground_energy] + exact_x + exact_zz

obs_labels = (
    ["H_TFIM"]
    + [f"X_{i}" for i in range(10)]
    + [f"ZZ_{i}{i + 1}" for i in range(6)]
    + [f"ZZ_nnn{i}" for i in range(3)]
)

print()
print("=" * 85)
print("  DETAILED ERROR ANALYSIS: NOISY (raw) vs QESEM (mitigated)")
print(f"  Circuit: HVA p=1, N=10, heavy_hex, h={h_value}")
print(f"  Backend: ibm_kingston | RZZ fidelity: {meta['gate_fidelities']['RZZ'] * 100:.2f}%")
print("=" * 85)
print()
header = f"  {'Observable':<10} | {'Exact':>9} | {'Noisy':>9} | {'QESEM':>9} | {'Noisy Err':>9} | {'Mitig Err':>9} | {'Reduction':>9}"
print(header)
sep = f"  {'-' * 10}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9}"
print(sep)

total_noisy_err = 0.0
total_mitig_err = 0.0
x_noisy_errs = []
x_mitig_errs = []
zz_noisy_errs = []
zz_mitig_errs = []

for i, obs_data in enumerate(results_obs):
    label = obs_labels[i] if i < len(obs_labels) else f"obs_{i}"
    exact = exact_vals[i] if i < len(exact_vals) else 0.0

    obs_info = obs_data[1]
    qesem_val = obs_info["qesem"]["value"]
    noisy_val = obs_info["unmitigated"]["value"]

    noisy_err = abs(noisy_val - exact)
    mitig_err = abs(qesem_val - exact)
    reduction = 1.0 - (mitig_err / noisy_err) if noisy_err > 1e-10 else 0.0

    total_noisy_err += noisy_err
    total_mitig_err += mitig_err

    if 1 <= i <= 10:
        x_noisy_errs.append(noisy_err)
        x_mitig_errs.append(mitig_err)
    elif 11 <= i <= 19:
        zz_noisy_errs.append(noisy_err)
        zz_mitig_errs.append(mitig_err)

    print(
        f"  {label:<10} | {exact:9.4f} | {noisy_val:9.4f} | {qesem_val:9.4f} | {noisy_err:9.4f} | {mitig_err:9.4f} | {reduction:8.1%}"
    )

print(sep)
total_reduction = 1.0 - (total_mitig_err / total_noisy_err)
print(
    f"  {'TOTAL':<10} | {'':>9} | {'':>9} | {'':>9} | {total_noisy_err:9.4f} | {total_mitig_err:9.4f} | {total_reduction:8.1%}"
)

print()
print("  == ERROR BREAKDOWN BY OBSERVABLE TYPE ==")
print()
print("  Energy (H_TFIM):")
e_noisy_err = abs(data["noisy_energy"] - gt.ground_energy)
e_mitig_err = abs(data["energy_mitigated"] - gt.ground_energy)
print(f"    Noisy error:    {e_noisy_err:.4f} ({e_noisy_err / gt.gap * 100:.2f}% of gap)")
print(f"    Mitigated error: {e_mitig_err:.4f} ({e_mitig_err / gt.gap * 100:.2f}% of gap)")
print(f"    Reduction:       {(1 - e_mitig_err / e_noisy_err) * 100:.1f}%")
print()
print("  Per-site X_i (magnetization):")
print(f"    Mean noisy error:    {np.mean(x_noisy_errs):.4f}")
print(f"    Mean mitigated error: {np.mean(x_mitig_errs):.4f}")
print(f"    Mean reduction:       {(1 - np.mean(x_mitig_errs) / np.mean(x_noisy_errs)) * 100:.1f}%")
print(f"    Max mitigated error:  {np.max(x_mitig_errs):.4f} (site {np.argmax(x_mitig_errs)})")
print()
print("  Per-bond ZZ_ij (correlators):")
print(f"    Mean noisy error:    {np.mean(zz_noisy_errs):.4f}")
print(f"    Mean mitigated error: {np.mean(zz_mitig_errs):.4f}")
print(
    f"    Mean reduction:       {(1 - np.mean(zz_mitig_errs) / np.mean(zz_noisy_errs)) * 100:.1f}%"
)
print()

# Noise scaling analysis (3-point extrapolation quality)
print("  == QESEM NOISE SCALING QUALITY ==")
print()
energy_obs = results_obs[0][1]
ns = energy_obs.get("noise_scaling", {})
pts = ns.get("results_with_REM", [])
if len(pts) >= 3:
    e_0 = pts[0]["value"]  # extrapolated (scale=0)
    e_1 = pts[2]["value"]  # physical (scale=1)
    e_2 = pts[1]["value"]  # amplified (scale=2)
    std_0 = pts[0]["error_bar"]
    std_1 = pts[2]["error_bar"]
    std_2 = pts[1]["error_bar"]

    # Linear extrapolation check: does (scale=1) - (scale=2) predict (scale=0)?
    slope = e_1 - e_2  # energy change per unit noise scale
    linear_predicted = e_1 - slope  # extrapolate to scale=0
    actual_extrapolated = e_0

    print("  3-point noise scaling:")
    print(f"    E(scale=0) = {e_0:.4f} +/- {std_0:.4f}  [QESEM quasi-prob extrapolation]")
    print(f"    E(scale=1) = {e_1:.4f} +/- {std_1:.4f}  [physical noise level]")
    print(f"    E(scale=2) = {e_2:.4f} +/- {std_2:.4f}  [2x amplified noise]")
    print()
    print("  Extrapolation quality:")
    print(f"    Noise slope (dE/d_scale):  {slope:.4f}")
    print(f"    Linear prediction (naive): {linear_predicted:.4f}")
    print(f"    QESEM result:              {actual_extrapolated:.4f}")
    print(f"    Exact:                     {gt.ground_energy:.4f}")
    print()
    print(f"    Linear pred. error vs exact: {abs(linear_predicted - gt.ground_energy):.4f}")
    print(f"    QESEM error vs exact:        {abs(actual_extrapolated - gt.ground_energy):.4f}")
    print()

    # SNR at each noise level
    signal_1 = abs(e_1 - gt.ground_energy)
    signal_0 = abs(e_0 - gt.ground_energy)
    print("  Signal-to-noise at each scale:")
    print(
        f"    scale=1: signal={abs(gt.ground_energy - e_1):.3f}, std={std_1:.3f}, SNR={abs(gt.ground_energy - e_1) / std_1:.1f}"
    )
    print(
        f"    scale=0: signal={abs(gt.ground_energy - e_0):.3f}, std={std_0:.3f}, SNR={abs(gt.ground_energy - e_0) / std_0:.1f}"
    )
    print()

    # Precision analysis
    print("  Precision budget analysis:")
    print("    Requested precision:   0.01")
    print(f"    Achieved std (energy): {std_0:.4f}")
    print(f"    Ratio (achieved/target): {std_0 / 0.01:.1f}x")
    print(f"    Implication: QESEM needed ~{(std_0 / 0.01) ** 2:.0f}x more shots to converge")
    print(f"    That would require: ~{428 * (std_0 / 0.01) ** 2:.0f}s QPU time (vs 428s budget)")

print()
print("=" * 85)
