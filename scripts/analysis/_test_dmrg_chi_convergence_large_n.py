"""Critical test: Does DMRG(graph) lose precision at N=16,22 near h_c?

Tests chi=32,64,128,256 vs exact diag at N=16,22 on heavy_hex.
If chi=64 ≠ exact at h=1.0, that's where quantum advantage COULD exist.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.utils.helpers import json_dump

spec = get_model_spec("tfim")
solver = ClassicalSolver()

topology = "heavy_hex"
chi_values = [32, 64, 128, 256]
h_values = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
n_values = [16, 22]

print(f"\nDMRG Chi-Convergence Test: {topology}")
print(f"Question: At N=16,22 near h_c, does DMRG lose precision with limited chi?")
print(f"{'='*90}")

results = []

for N in n_values:
    print(f"\n  N={N} {'─'*70}")
    print(f"  {'h':>5} {'E_exact':>14} "
          + " ".join(f"{'χ='+str(c):>14}" for c in chi_values)
          + f" {'|ΔE|_χ32':>10} {'|ΔE|_χ64':>10}")

    for h in h_values:
        t0 = time.perf_counter()
        lattice = make_lattice(topology, N, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

        # Exact diag
        gt = solver.solve(H, lattice, method="exact")
        e_exact = gt.ground_energy
        gap = gt.gap

        # DMRG at each chi
        energies = {}
        for chi in chi_values:
            gt_chi = solver.solve(H, lattice, method="dmrg", chi_max=chi)
            energies[chi] = gt_chi.ground_energy

        elapsed = time.perf_counter() - t0

        de_32 = abs(energies[32] - e_exact)
        de_64 = abs(energies[64] - e_exact)
        de_128 = abs(energies[128] - e_exact)

        chi_strs = " ".join(f"{energies[c]:>14.8f}" for c in chi_values)
        print(f"  {h:>5.1f} {e_exact:>14.8f} {chi_strs} {de_32:>10.2e} {de_64:>10.2e}  ({elapsed:.0f}s)")

        results.append({
            "N": N, "h": h, "e_exact": e_exact, "gap": gap,
            "energies": {str(c): energies[c] for c in chi_values},
            "de_32": de_32, "de_64": de_64, "de_128": de_128,
            "de_gap_32": de_32/max(gap,1e-10), "de_gap_64": de_64/max(gap,1e-10),
        })

# Summary
print(f"\n{'='*90}")
print("SUMMARY — Is chi=64 sufficient?")
for N in n_values:
    n_rows = [r for r in results if r["N"] == N]
    max_de_64 = max(r["de_64"] for r in n_rows)
    max_dg_64 = max(r["de_gap_64"] for r in n_rows)
    worst_h = max(n_rows, key=lambda r: r["de_64"])["h"]
    print(f"  N={N}: max |ΔE|(χ=64) = {max_de_64:.2e} at h={worst_h} "
          f"(ΔE/gap = {max_dg_64:.6f} = {max_dg_64*100:.4f}%)")

json_dump({"results": results, "chi_values": chi_values},
          ROOT / "results" / "analysis" / "dmrg_chi_convergence_N16_N22.json")
print(f"\nSaved to results/analysis/dmrg_chi_convergence_N16_N22.json")
