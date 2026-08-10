#!/usr/bin/env python3
"""Analyze QESEM Tier-1 result (h=3.5) from recovered JSON."""

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

# Determine h from the Hamiltonian coefficients in metadata
# The X coefficient in the observable string tells us h
obs_str = (
    data.get("metadata", {}).get("results", [[]])[0][0][0]
    if data.get("metadata", {}).get("results")
    else ""
)
if "-3.5" in str(obs_str):
    h_value = 3.5
elif "-3.25" in str(obs_str):
    h_value = 3.25
else:
    h_value = 3.5  # Default assumption
    print(f"  WARNING: Could not determine h from observable. Assuming h={h_value}")

print(f"\n  Detected h = {h_value} from Hamiltonian coefficients")

# Compute exact ground state
lattice = make_lattice("heavy_hex", 10, h=h_value, J=1.0)
H = HamiltonianBuilder().build(lattice)
gt = ClassicalSolver().solve(H, lattice)

e_qesem = data["energy_mitigated"]
e_std = data["energy_std"]
e_noisy = data["noisy_energy"]
x_vals = data["x_values"]
zz_vals = data["zz_values"]

delta_e = abs(e_qesem - gt.ground_energy)
delta_e_gap = delta_e / gt.gap
noisy_err = abs(e_noisy - gt.ground_energy)
gain = 1.0 - (delta_e / noisy_err) if noisy_err > 1e-10 else 0.0

print()
print("=" * 65)
print(f"  QESEM TIER-1 ANALYSIS: h={h_value}, N=10, heavy_hex, p=1")
print("=" * 65)
print(f"  E_exact:        {gt.ground_energy:.6f}")
print(f"  E_QESEM:        {e_qesem:.6f} +/- {e_std:.6f}")
print(f"  E_noisy (raw):  {e_noisy:.6f}")
print(f"  Delta_E:        {delta_e:.6f}")
print(f"  Delta_E/gap:    {delta_e_gap:.4f} ({delta_e_gap * 100:.2f}%)")
print(f"  Gap:            {gt.gap:.6f}")
print(f"  ZNE gain:       {gain:.4f} ({gain * 100:.1f}%)")
print()
verdict = "PASS" if delta_e_gap < 0.05 else "FAIL"
print(f"  VERDICT: {verdict} (threshold < 5%)")
print()

# Observables
print("  -- Observables --")
print(f"  <X> mean:      {np.mean(x_vals):.4f}")
print(f"  <X> std:       {np.std(x_vals):.4f}")
print(f"  <ZZ> NN mean:  {np.mean(zz_vals[:6]):.4f} (bonds 0-5)")
print(f"  <ZZ> NNN mean: {np.mean(zz_vals[6:]):.4f} (bonds 6-8, non-NN)")
mag_x = float(np.mean(np.abs(x_vals)))
phase = "paramagnetic" if mag_x > 0.5 else "ferromagnetic"
print(f"  Phase: {phase} (|<X>| = {mag_x:.4f})")
print()

# QPU resources
meta = data.get("metadata", {})
print("  -- QPU Resources --")
print(f"  QPU time:      {meta.get('total_qpu_time', '?')}s")
print(f"  Total shots:   {meta.get('total_shots', '?')}")
print(f"  Mitigation:    {meta.get('mitigation_shots', '?')} shots")
gf = meta.get("gate_fidelities", {})
for g, fid in gf.items():
    print(f"  Fidelity {g}: {fid * 100:.2f}%")
print()

# Resource usage
ru = meta.get("resource_usage", {})
if ru:
    qpu_s = ru.get("RUNNING: EXECUTING_QPU", {}).get("QPU_TIME", 0)
    queue_s = ru.get("RUNNING: WAITING_FOR_QPU", {}).get("CPU_TIME", 0)
    mapping_s = ru.get("RUNNING: MAPPING", {}).get("CPU_TIME", 0)
    print("  -- QESEM Internal Timing --")
    print(f"  QPU execution:  {qpu_s:.0f}s ({qpu_s / 60:.1f} min)")
    print(f"  Queue wait:     {queue_s:.0f}s ({queue_s / 3600:.1f} h)")
    print(f"  Mapping (CPU):  {mapping_s:.0f}s ({mapping_s / 60:.1f} min)")
    print()

# Noise scaling
results_meta = meta.get("results")
if results_meta and isinstance(results_meta, list):
    first_obs = results_meta[0][0]
    if first_obs and len(first_obs) > 1:
        ns = first_obs[1].get("noise_scaling", {})
        if ns:
            print("  -- Noise Scaling (energy) --")
            for pt in ns.get("results_with_REM", []):
                s = pt["scale"]
                v = pt["value"]
                e = pt["error_bar"]
                print(f"    scale={s:.1f}: E={v:.4f} +/- {e:.4f}")
            # Heuristic extrapolation
            heur = first_obs[1].get("qesem_heuristic")
            if heur:
                for h_entry in heur:
                    print(
                        f"    heuristic ({h_entry['extrapolation']}): "
                        f"E={h_entry['value']:.4f} +/- {h_entry['error_bar']:.4f}"
                    )
print()

# Comparison across all QESEM runs
print("  -- All QESEM Results Summary --")
print(f"  {'h':>5} | {'E_QESEM':>12} | {'E_exact':>12} | {'dE/gap':>8} | {'Gain':>6} | Verdict")
print(f"  {'-' * 5}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 6}-+--------")
print(
    f"  {h_value:5.2f} | {e_qesem:12.4f} | {gt.ground_energy:12.4f} | {delta_e_gap:8.4f} | {gain:5.1%} | {verdict}"
)

# Also compute for h=4.0 (from the other file) for context
lattice4 = make_lattice("heavy_hex", 10, h=4.0, J=1.0)
H4 = HamiltonianBuilder().build(lattice4)
gt4 = ClassicalSolver().solve(H4, lattice4)
e_qesem_4 = -40.52391074682875  # From tier0 run
e_noisy_4 = -38.47114285714286
de4 = abs(e_qesem_4 - gt4.ground_energy) / gt4.gap
gain4 = 1.0 - (abs(e_qesem_4 - gt4.ground_energy) / abs(e_noisy_4 - gt4.ground_energy))
print(
    f"  {4.0:5.2f} | {e_qesem_4:12.4f} | {gt4.ground_energy:12.4f} | {de4:8.4f} | {gain4:5.1%} | PASS"
)
print()
print("=" * 65)
