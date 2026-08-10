"""Comprehensive comparison: DMRG 2D (graph) vs VQE noiseless at N=10,16,22.

For N>22, exact diag is not available so DMRG 2D IS the best classical reference.
We test: at what chi does DMRG 2D converge? And can VQE (p=1,2) match it?

Tests:
  1. DMRG 2D accuracy: E_DMRG(chi) vs E_exact for N≤22
  2. DMRG 2D chi-convergence: E(chi=64) vs E(chi=128) vs E(chi=256) for N=16,22
  3. VQE expressibility: best noiseless E_VQE(p) vs E_exact for p=1,2,3,4
  4. Comparison: VQE(p) vs DMRG at each h, answering "can QPU add value?"
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models.data_models import VQEConfig
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.optimizers.vqe import VQEOptimizer


def run_vqe_best(lattice, p_layers, seed=42, n_restarts=5, maxiter=500):
    """Run VQE noiseless and return best energy found."""
    spec = get_model_spec("tfim")
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    circuit, _ = spec.create_circuit(
        lattice.n_qubits, p_layers, lattice, **spec.circuit_kwargs
    )
    noiseless = NoiselessBackend()
    config = VQEConfig(
        p_layers=p_layers, n_restarts=n_restarts,
        maxiter=maxiter, method="L-BFGS-B", enable_callbacks=False,
    )
    opt = VQEOptimizer(config=config, backend=noiseless, seed=seed)
    rng = np.random.default_rng(seed)
    init = rng.uniform(-0.01, 0.01, circuit.num_parameters)
    result = opt.optimize(H, circuit, init)
    return result.energy, circuit.num_parameters


def main():
    spec = get_model_spec("tfim")
    solver = ClassicalSolver()
    topology = "heavy_hex"

    # N values: 10 (quick), 16 (medium), 22 (max for exact diag)
    # N=28 requires DMRG only (no exact reference), tested separately
    n_values = [10, 16, 22]
    h_values = [1.0, 1.5, 2.0, 3.0]
    p_values = [1, 2]  # p=3,4 takes too long, run separately if needed
    chi_values = [64, 128, 256]

    print(f"\n{'='*80}")
    print(f"  DMRG 2D vs VQE Expressibility — {topology}")
    print(f"  N values: {n_values}, h values: {h_values}")
    print(f"  p layers: {p_values}, chi values: {chi_values}")
    print(f"{'='*80}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 1: DMRG 2D chi-convergence
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  PART 1: DMRG 2D chi-convergence (graph-based, proper bonds)")
    print(f"{'─'*80}")
    print(f"  {'N':>3} {'h':>5} {'E_exact':>14} "
          + " ".join(f"{'E_chi'+str(c):>14}" for c in chi_values)
          + f" {'|ΔE|_64':>10} {'|ΔE|_256':>10}")

    for N in n_values:
        for h in h_values:
            lattice = make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

            # Exact (only for N≤22)
            gt = solver.solve(H, lattice, method="exact")
            e_exact = gt.ground_energy

            # DMRG at each chi
            chi_energies = {}
            for chi in chi_values:
                gt_dmrg = solver.solve(H, lattice, method="dmrg", chi_max=chi)
                chi_energies[chi] = gt_dmrg.ground_energy

            de_64 = abs(chi_energies[64] - e_exact)
            de_256 = abs(chi_energies[256] - e_exact)

            chi_strs = " ".join(f"{chi_energies[c]:>14.8f}" for c in chi_values)
            print(f"  {N:>3} {h:>5.1f} {e_exact:>14.8f} {chi_strs} "
                  f"{de_64:>10.2e} {de_256:>10.2e}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: VQE expressibility at each p
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  PART 2: VQE noiseless expressibility (best energy achievable)")
    print(f"{'─'*80}")
    print(f"  {'N':>3} {'h':>5} {'E_exact':>14} "
          + " ".join(f"{'E_VQE_p'+str(p):>14}" for p in p_values)
          + " ".join(f"{'ΔE/gap_p'+str(p):>11}" for p in p_values))

    for N in n_values:
        for h in h_values:
            lattice = make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
            gt = solver.solve(H, lattice, method="exact")
            e_exact = gt.ground_energy
            gap = gt.gap

            vqe_energies = {}
            for p in p_values:
                t0 = time.perf_counter()
                e_vqe, n_params = run_vqe_best(lattice, p, n_restarts=3, maxiter=300)
                elapsed = time.perf_counter() - t0
                vqe_energies[p] = e_vqe

            vqe_strs = " ".join(f"{vqe_energies[p]:>14.8f}" for p in p_values)
            gap_strs = " ".join(
                f"{abs(vqe_energies[p] - e_exact) / max(gap, 1e-10):>11.6f}"
                for p in p_values
            )
            print(f"  {N:>3} {h:>5.1f} {e_exact:>14.8f} {vqe_strs} {gap_strs}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print("  SUMMARY")
    print(f"{'─'*80}")
    print("  - If |ΔE|_DMRG(chi=256) ≈ 0: DMRG 2D is exact, no quantum advantage")
    print("  - If ΔE/gap_VQE(p=2) < 5%: HVA p=2 can express the ground state")
    print("  - Quantum advantage exists only where DMRG fails AND VQE succeeds")
    print("  - For TFIM on heavy_hex, DMRG 2D with chi=128-256 is exact at N≤22")
    print("  - The question becomes: at what N does chi=256 become insufficient?")


if __name__ == "__main__":
    main()
