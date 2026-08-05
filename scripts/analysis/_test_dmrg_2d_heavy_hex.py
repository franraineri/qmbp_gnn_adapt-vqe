"""Proof-of-concept: DMRG with proper heavy_hex 2D bonds via CouplingMPOModel.

Compares:
  - E_exact (eigsh)
  - E_DMRG_2D (CouplingMPOModel with all heavy_hex edges)
  - E_DMRG_1D (TFIChain snake — current fallback, misses 2D bonds)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tenpy.algorithms import dmrg as tenpy_dmrg
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.models.model_registry import get_model_spec


class HeavyHexTFIM(CouplingMPOModel):
    """TFIM on arbitrary graph topology via explicit coupling terms.

    H = -J * sum_{(i,j) in edges} Z_i Z_j - h * sum_i X_i

    TeNPy spin-1/2 convention: Z = 2*Sz, X = 2*Sx
    So: ZZ term → -4J * Sz_i * Sz_j, X term → -2h * Sx_i
    """

    def init_terms(self, model_params):
        J_val = model_params.get("J", 1.0)
        h_val = model_params.get("h", 1.0)
        edges = model_params.get("edges", [])

        # On-site: -h * X_i = -2h * Sx_i (use add_onsite_term with MPS index)
        for i in range(self.lat.N_sites):
            self.add_onsite_term(-2.0 * h_val, i, "Sx")

        # Coupling: -J * Z_i Z_j = -4J * Sz_i Sz_j per edge
        for i, j in edges:
            if i < j:
                self.add_coupling_term(-4.0 * J_val, i, j, "Sz", "Sz")
            else:
                self.add_coupling_term(-4.0 * J_val, j, i, "Sz", "Sz")


def run_dmrg_2d(our_lattice, chi_max=128):
    """Run DMRG with proper 2D bond connectivity."""
    N = our_lattice.n_qubits
    h_val = float(our_lattice.h)
    J_val = float(our_lattice.J)
    edges = our_lattice.edges

    site = SpinHalfSite(conserve=None)
    lat = Chain(L=N, site=site, bc_MPS="finite")

    params = {"lattice": lat, "J": J_val, "h": h_val, "edges": edges}
    model = HeavyHexTFIM(params)

    psi = MPS.from_lat_product_state(lat, [["up"]] * N)
    dmrg_params = {
        "mixer": True,
        "max_E_err": 1e-12,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-14},
        "max_sweeps": 80,
    }
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    e0, _ = eng.run()
    return float(e0)


if __name__ == "__main__":
    spec = get_model_spec("tfim")
    solver = ClassicalSolver()

    N = 10
    topology = "heavy_hex"

    print(f"\nDMRG 2D (proper bonds) vs 1D (snake) vs Exact — {topology} N={N}")
    print(f"{'h':>5} {'E_exact':>14} {'E_DMRG_2D':>14} {'E_DMRG_1D':>14} "
          f"{'|ΔE|_2D':>12} {'|ΔE|_1D':>12} {'ΔE/gap_2D':>10} {'ΔE/gap_1D':>10}")
    print("-" * 100)

    for h in [1.0, 1.5, 2.0, 3.0, 4.0]:
        lattice = make_lattice(topology, N, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

        # Exact
        gt = solver.solve(H, lattice, method="exact")
        e_exact = gt.ground_energy
        gap = gt.gap

        # DMRG 2D (proper heavy_hex bonds)
        e_2d = run_dmrg_2d(lattice, chi_max=128)

        # DMRG 1D (current TFIChain fallback)
        gt_1d = solver.solve(H, lattice, method="dmrg")
        e_1d = gt_1d.ground_energy

        de_2d = abs(e_2d - e_exact)
        de_1d = abs(e_1d - e_exact)
        dg_2d = de_2d / max(gap, 1e-10)
        dg_1d = de_1d / max(gap, 1e-10)

        print(f"{h:>5.1f} {e_exact:>14.8f} {e_2d:>14.8f} {e_1d:>14.8f} "
              f"{de_2d:>12.2e} {de_1d:>12.2e} {dg_2d:>10.6f} {dg_1d:>10.6f}")

    print("\nInterpretation:")
    print("  DMRG_2D uses all heavy_hex edges → should be exact (or near-exact)")
    print("  DMRG_1D uses TFIChain (only sequential bonds) → misses 2D connectivity")
    print("  If DMRG_2D ≈ exact → the gap is purely modeling, fixable in our solver")
