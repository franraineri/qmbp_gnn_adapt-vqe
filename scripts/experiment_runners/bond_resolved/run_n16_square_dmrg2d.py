#!/usr/bin/env python3
"""N=16 Square (4x4) Bond-Resolved HVA with DMRG 2D ground truth.

Tests the bond-resolved pipeline at N=16 on a genuine 2D square lattice
using the new DMRG 2D solver (TeNPy SpinModel + Square lattice).

This is a scaling proof: N=16 has 40 bond-resolved parameters (24 edges + 16 sites)
and requires DMRG for ground truth (exceeds exact diag limit of N=15).

Expected runtime: ~5-8 min per h-point (high-dimensional VQE landscape).
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


class N16SquareDMRG2DRunner(ValidationRunner):
    """N=16 Square lattice (4x4) with DMRG 2D + bond-resolved HVA."""

    runner_id = "n16_square_dmrg2d"
    experiment_id = "N16_SQUARE_DMRG2D"
    description = "N=16 square (4x4) bond-resolved HVA with DMRG 2D ground truth"
    hypothesis = (
        "Bond-resolved HVA at N=16 (4x4 square, 40 params) converges "
        "to DMRG ground state with fidelity proxy dE/gap < 5%"
    )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": 16,
                "p_layers": 1,
                "topology": "square",
                "grid_shape": "4x4",
                "model": "tfim_bond_resolved",
                "n_edges": 24,
                "n_params": 40,
            },
            "seed": 42,
        }

    def setup(self):
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.models.model_registry import get_model_spec

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="DMRG 2D Ground Truth (4x4 square)",
                fn=self.section_dmrg_ground_truth,
                hypothesis="DMRG 2D converges for TFIM on 4x4 square at h=5.0,6.0",
            ),
            Section(
                id=2,
                name="Bond-Resolved VQE at N=16",
                fn=self.section_vqe_convergence,
                hypothesis="VQE with 40 params converges to dE/gap < 5% at h>=5.0",
            ),
        ]

    def section_dmrg_ground_truth(self) -> dict:
        """Verify DMRG 2D produces reliable ground truth for N=16 square."""
        N = 16
        TOPOLOGY = "square"
        H_VALS = [6.0, 5.5, 5.0]

        results = []
        for h in H_VALS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)

            t0 = time.time()
            gt = self.solver.solve(H, lattice, method="dmrg")
            elapsed = time.time() - t0

            results.append(
                {
                    "h": h,
                    "ground_energy": float(gt.ground_energy),
                    "gap": float(gt.gap),
                    "mag_x": float(gt.mag_x),
                    "elapsed_s": elapsed,
                }
            )
            logger.info(
                f"  h={h:.1f} | E0={gt.ground_energy:.4f} | "
                f"gap={gt.gap:.4f} | mag_x={gt.mag_x:.4f} | {elapsed:.1f}s"
            )

        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "method": "dmrg_2d",
            "results": results,
            "pass": all(r["gap"] > 0 for r in results),
        }

    def section_vqe_convergence(self) -> dict:
        """Bond-resolved VQE convergence at N=16 (40 parameters)."""
        N, P = 16, 1
        TOPOLOGY = "square"
        H_VALS = [6.0, 5.5, 5.0]
        seed = 42

        opt = self.VQEOptimizer(
            self.VQEConfig(
                n_restarts=5,
                restart_sigma=0.03,  # Smaller: 40-dim needs gentler perturbation
                maxiter=2000,
            ),
            seed=seed,
        )

        lattice_ref = self.make_lattice(TOPOLOGY, N, h=6.0)
        n_edges = len(lattice_ref.edges)
        n_params = n_edges + N

        prev_theta = None
        results = []

        for h in H_VALS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice, method="dmrg")
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            t0 = time.time()
            res = opt.optimize(
                H,
                qc,
                init,
                exact_energy=gt.ground_energy,
                exact_state=None,  # DMRG doesn't return statevector
            )
            elapsed = time.time() - t0
            prev_theta = res.theta_opt.copy()

            de_gap = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
            results.append(
                {
                    "h": h,
                    "energy_error": float(res.energy_error),
                    "delta_e_gap": float(de_gap),
                    "n_params": n_params,
                    "elapsed_s": elapsed,
                }
            )
            logger.info(
                f"  h={h:.1f} | dE/gap={de_gap:.4f} | "
                f"E_err={res.energy_error:.4f} | params={n_params} | {elapsed:.1f}s"
            )

        all_pass = all(r["delta_e_gap"] < 0.05 for r in results)
        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "p_layers": P,
            "n_params": n_params,
            "n_edges": n_edges,
            "grid_shape": "4x4",
            "results": results,
            "pass": all_pass,
        }


if __name__ == "__main__":
    N16SquareDMRG2DRunner.main()
