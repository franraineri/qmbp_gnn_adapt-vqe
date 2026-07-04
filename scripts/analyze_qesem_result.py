#!/usr/bin/env python3
"""Analyze recovered QESEM result against exact values."""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

# Load QESEM result
result_path = _ROOT / "results/recovered/qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json"
with open(result_path) as f:
    data = json.load(f)

# Compute exact ground state for h=4.0, N=10, heavy_hex
lattice = make_lattice("heavy_hex", 10, h=4.0, J=1.0)
builder = HamiltonianBuilder()
H = builder.build(lattice)
solver = ClassicalSolver()
gt = solver.solve(H, lattice)

# Extract QESEM values
e_qesem = data["energy_mitigated"]
e_std = data["energy_std"]
e_noisy = data["noisy_energy"]
x_values = data["x_values"]
zz_values = data["zz_values"]

# Compute metrics
delta_e = abs(e_qesem - gt.ground_energy)
delta_e_gap = delta_e / gt.gap
noisy_error = abs(e_noisy - gt.ground_energy)
gain = 1.0 - (delta_e / noisy_error) if noisy_error > 1e-10 else 0.0

print()
print("=" * 60)
print("  QESEM RESULT ANALYSIS - h=4.0, N=10, heavy_hex, p=1")
print("=" * 60)
print(f"  E_exact:        {gt.ground_energy:.6f}")
print(f"  E_QESEM:        {e_qesem:.6f} +/- {e_std:.6f}")
print(f"  E_noisy (raw):  {e_noisy:.6f}")
print(f"  Delta_E:        {delta_e:.6f}")
print(f"  Delta_E/gap:    {delta_e_gap:.4f} ({delta_e_gap * 100:.2f}%)")
print(f"  Gap:            {gt.gap:.6f}")
print(f"  Noisy error:    {noisy_error:.6f}")
print(f"  ZNE gain:       {gain:.4f} ({gain * 100:.1f}%)")
print()
verdict = "PASS" if delta_e_gap < 0.05 else "FAIL"
print(f"  VERDICT: {verdict} (threshold: Delta_E/gap < 5%)")
print()

# Per-site observables
print("  -- Per-site Observables --")
print(f"  <X> mean:      {np.mean(x_values):.4f} (ideal ~ 1.0 at h=4)")
print(f"  <X> std:       {np.std(x_values):.4f}")
print(f"  <ZZ> NN mean:  {np.mean(zz_values[:6]):.4f} (bonds 0-5, nearest-neighbor)")
print(f"  <ZZ> NNN mean: {np.mean(zz_values[6:]):.4f} (bonds 6-8, non-NN heavy-hex)")
print()

# Phase classification
mag_x = float(np.mean(np.abs(x_values)))
corr_zz = float(np.mean(np.abs(zz_values[:6])))
phase = "paramagnetic" if mag_x > 0.5 else "ferromagnetic"
print(f"  Phase classification: {phase}")
print(f"    |<X>| mean = {mag_x:.4f}")
print(f"    |<ZZ>| NN mean = {corr_zz:.4f}")
print()

# QPU resources
meta = data.get("metadata", {})
print("  -- QPU Resources --")
print(f"  Total QPU time:    {meta.get('total_qpu_time', '?')}s")
print(f"  Total shots:       {meta.get('total_shots', '?')}")
print(f"  Mitigation shots:  {meta.get('mitigation_shots', '?')}")
gate_fid = meta.get("gate_fidelities", {})
if gate_fid:
    for gate, fid in gate_fid.items():
        print(f"  Gate fidelity {gate}: {fid * 100:.2f}%")
print()

# Comparison with PEA-ZNE simulation benchmark
print("  -- Comparison with PEA-ZNE (FakeTorino simulation) --")
pea_sim_delta = 0.0037  # 0.37% from mitigation benchmark C5
print(f"  PEA-ZNE (sim):  Delta_E/gap ~ {pea_sim_delta * 100:.2f}%")
print(f"  QESEM (real):   Delta_E/gap = {delta_e_gap * 100:.2f}%")
if delta_e_gap < 0.05:
    ratio = delta_e_gap / pea_sim_delta
    print(f"  Ratio: QESEM/PEA_sim = {ratio:.1f}x")
    if ratio < 2:
        print("  -> Excellent: QESEM on real QPU within 2x of ideal simulation")
    elif ratio < 5:
        print("  -> Good: real QPU noise adds modest overhead vs simulation")
    else:
        print("  -> Check: large gap suggests precision budget was insufficient")
print()

# Noise scaling data (QESEM internal)
results_meta = meta.get("results")
if results_meta and isinstance(results_meta, list):
    first_obs = results_meta[0][0] if results_meta[0] else None
    if first_obs and len(first_obs) > 1:
        ns = first_obs[1].get("noise_scaling", {})
        if ns:
            print("  -- QESEM Internal Noise Scaling (energy observable) --")
            for point in ns.get("results_with_REM", []):
                print(
                    f"    scale={point['scale']:.1f}: E={point['value']:.4f} +/- {point['error_bar']:.4f}"
                )
print()
print("=" * 60)
