"""E4: TFIM with Longitudinal Field (2D Phase Diagram).

Hypothesis: The MPNN trained on standard TFIM can predict parameters for
TFIM + longitudinal field (H = -J·ZZ - h·X - g·Z) with g as additional
node feature, demonstrating cross-model generalization.

Method:
    1. Build H = -J·Σ(ZZ) - h·Σ(X) - g·Σ(Z) for g ∈ {0, 0.1, 0.2, 0.3, 0.5}
    2. Run VQE sweep per g-value
    3. Train MPNN with (h, g) as features
    4. Test generalization at held-out (h, g) pairs

Expected outcome: HVA p=2 works for g <= 0.3; MPNN generalizes across g.

References:
    - Dutta et al. (2015) Quantum Phase Transitions in Transverse Field Spin Models
    - Lee et al. (2026) arXiv:2602.19752 — GNN generalizes across Hamiltonians
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


def build_tfim_longitudinal(N: int, J: float, h: float, g: float):
    """Build TFIM + longitudinal field Hamiltonian.

    H = -J * Σ(ZᵢZᵢ₊₁) - h * Σ(Xᵢ) - g * Σ(Zᵢ)
    """
    from qiskit.quantum_info import SparsePauliOp

    terms = []
    coeffs = []

    # ZZ interactions (nearest-neighbor, open boundary)
    for i in range(N - 1):
        pauli = ["I"] * N
        pauli[i] = "Z"
        pauli[i + 1] = "Z"
        terms.append("".join(reversed(pauli)))
        coeffs.append(-J)

    # Transverse field (X)
    for i in range(N):
        pauli = ["I"] * N
        pauli[i] = "X"
        terms.append("".join(reversed(pauli)))
        coeffs.append(-h)

    # Longitudinal field (Z)
    if abs(g) > 1e-10:
        for i in range(N):
            pauli = ["I"] * N
            pauli[i] = "Z"
            terms.append("".join(reversed(pauli)))
            coeffs.append(-g)

    return SparsePauliOp.from_list(list(zip(terms, coeffs, strict=False))).simplify()


def exact_diag_sparse(hamiltonian):
    """Compute ground state energy and gap via exact diagonalization."""
    from scipy.sparse.linalg import eigsh

    H_matrix = hamiltonian.to_matrix()
    if H_matrix.shape[0] <= 64:
        eigenvalues = np.linalg.eigvalsh(
            H_matrix.toarray() if hasattr(H_matrix, "toarray") else H_matrix
        )
        e0 = eigenvalues[0]
        e1 = eigenvalues[1]
    else:
        eigenvalues = eigsh(H_matrix, k=2, which="SA", return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        e0 = eigenvalues[0]
        e1 = eigenvalues[1]

    return float(e0), float(e1 - e0)


class ExperimentE4(BaseExperiment):
    """TFIM + longitudinal field: 2D phase diagram generalization."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="E4",
            category="E",
            description="TFIM + longitudinal field: 2D phase diagram with MPNN",
            hypothesis=(
                "HVA p=2 works for g<=0.3 and MPNN generalizes across g "
                "with (h, g) as input features."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[1.0, 1.25, 1.5, 1.75, 2.0],
                g_longitudinal=0.0,
                model="tfim_longitudinal",
            ),
            vqe=VQEConfig(n_restarts=5, sigma=0.1),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Override: use standard HVA circuit."""
        from qmbp_simulation import HVACircuitBuilder, make_lattice

        self._setup_logging()
        self.hva = HVACircuitBuilder()
        self.logger = logging.getLogger(__name__)

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        self.logger.info(f"E4 setup: N={N}, p={p}, model=TFIM+longitudinal")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE across (h, g) grid."""
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import Statevector, state_fidelity
        from scipy.optimize import minimize

        np.random.seed(seed)
        estimator = StatevectorEstimator()
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        g_values = [0.0, 0.1, 0.2, 0.3, 0.5]

        metrics = []

        for g in g_values:
            prev_theta = np.random.uniform(-0.01, 0.01, n_params)

            for h in sorted(h_values, reverse=True):
                t0 = time.time()
                H = build_tfim_longitudinal(N, J=1.0, h=h, g=g)
                exact_energy, gap = exact_diag_sparse(H)

                def cost_fn(params, _H=H):
                    bound = self.circuit.assign_parameters(params)
                    job = estimator.run([(bound, _H)])
                    return float(job.result()[0].data.evs)

                best_energy = float("inf")
                best_theta = prev_theta.copy()
                total_evals = 0

                for restart in range(self.config.vqe.n_restarts):
                    x0 = (
                        prev_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                        if restart > 0
                        else prev_theta.copy()
                    )
                    x0 = np.clip(x0, -np.pi, np.pi)
                    result = minimize(
                        cost_fn,
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * n_params,
                        options={"maxiter": 500, "ftol": 1e-14},
                    )
                    total_evals += result.nfev
                    if result.fun < best_energy:
                        best_energy = result.fun
                        best_theta = result.x.copy()

                prev_theta = best_theta.copy()

                # Compute fidelity
                H_matrix = H.to_matrix()
                eigenvalues_arr, eigenvectors = np.linalg.eigh(
                    H_matrix.toarray() if hasattr(H_matrix, "toarray") else H_matrix
                )
                gs_vector = eigenvectors[:, 0]
                bound_circuit = self.circuit.assign_parameters(best_theta)
                sv_vqe = Statevector(bound_circuit)
                fid = float(state_fidelity(sv_vqe, Statevector(gs_vector)))

                de_gap = abs(best_energy - exact_energy) / gap if gap > 1e-10 else 0.0
                elapsed = time.time() - t0

                m = ExperimentMetrics(
                    h_value=h,
                    energy=best_energy,
                    exact_energy=exact_energy,
                    energy_error=abs(best_energy - exact_energy),
                    gap=gap,
                    relative_error=de_gap,
                    fidelity=fid,
                    seed=seed,
                    wall_time_s=elapsed,
                    n_evaluations=total_evals,
                    technique_metadata={
                        "g": float(g),
                        "fidelity": float(fid),
                        "de_gap": float(de_gap),
                        "hva_sufficient": fid >= 0.93,
                    },
                )
                metrics.append(m)

        return metrics
