"""
Hardware Deployer — Dual-route deployment: Adapt-VQE (main) + QRC (fallback).

Main route: bind θ_pred to HVA → AdaptVQE (max_iterations=2) → measure local
observables via EstimatorV2 (or StatevectorEstimator for noiseless simulation).

Fallback route: QRC encode → measure → linear readout.

Phase classification uses data-driven ⟨X⟩ = ⟨ZZ⟩ crossover from Phase 1 exact
data (not hardcoded h_c=1.0).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import AdaptVQE, VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.exceptions import AlgorithmError

from .config import DeployResult, GroundTruthResult, LatticeConfig

logger = logging.getLogger(__name__)


class HardwareDeployer:
    """Execute the trained pipeline on quantum hardware (or noiseless sim)."""

    def __init__(self, estimator=None) -> None:
        self._estimator = estimator or StatevectorEstimator()

    # ── Task 8.1 + 8.2: deploy_adapt_vqe() ──────────────────────────

    def deploy_adapt_vqe(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        theta_pred: np.ndarray,
        lattice: LatticeConfig,
        exact: GroundTruthResult,
        *,
        max_iterations: int = 2,
        gradient_threshold: float = 1e-3,
    ) -> DeployResult:
        """Main route: GNN/MPNN inference → bind θ_pred → AdaptVQE → measure.

        Parameters
        ----------
        circuit : QuantumCircuit — parameterized HVA
        hamiltonian : SparsePauliOp — test Hamiltonian
        theta_pred : np.ndarray — predicted parameters from MPNN
        lattice : LatticeConfig
        exact : GroundTruthResult — exact reference for validation
        max_iterations : int — AdaptVQE max iterations (≤ 2)
        gradient_threshold : float — AdaptVQE gradient threshold
        """
        from .hamiltonian_builder import HamiltonianBuilder
        builder = HamiltonianBuilder()
        n = lattice.n_qubits

        # Bind θ_pred to HVA as initial state
        initial_state = circuit.assign_parameters(theta_pred)

        # Build Pauli pool from lattice observables
        ops_x, ops_zz = builder.build_local_observables(lattice)
        pauli_pool = ops_zz + ops_x

        # VQE solver for AdaptVQE
        vqe_solver = VQE(
            estimator=self._estimator,
            ansatz=QuantumCircuit(n),
            optimizer=L_BFGS_B(maxiter=100),
        )

        # Restricted AdaptVQE
        adapt_vqe = AdaptVQE(
            vqe_solver,
            operators=pauli_pool,
            max_iterations=max_iterations,
            gradient_threshold=gradient_threshold,
            eigenvalue_threshold=1e-6,
            initial_state=initial_state,
        )

        # Task 8.2: handle AlgorithmError at iteration 0
        adapt_iterations = 0
        converged_at_init = False

        try:
            adapt_result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)
            e_final = float(adapt_result.eigenvalue.real)
            adapt_iterations = int(adapt_result.num_iterations)
            final_qc = adapt_result.optimal_circuit.assign_parameters(
                adapt_result.optimal_point
            )
            logger.info(
                f"AdaptVQE completed: {adapt_iterations} iterations, "
                f"E={e_final:.6f}"
            )
        except AlgorithmError as e:
            if "convergence threshold" in str(e).lower() or "first iteration" in str(e).lower():
                converged_at_init = True
                adapt_iterations = 0
                final_qc = initial_state
                e_final = float(
                    self._estimator.run(
                        [(initial_state, hamiltonian)]
                    ).result()[0].data.evs
                )
                logger.info(
                    f"AdaptVQE converged at iteration 0 (ideal warm-start). "
                    f"E={e_final:.6f}"
                )
            else:
                raise

        # Measure local observables
        sv_final = Statevector(final_qc)
        mag_x_pred = float(np.mean([
            sv_final.expectation_value(op).real for op in ops_x
        ]))
        corr_zz_pred = float(np.mean([
            sv_final.expectation_value(op).real for op in ops_zz
        ]))

        # Fidelity (noiseless only)
        fidelity = None
        if exact.ground_state is not None:
            fidelity = float(state_fidelity(sv_final, Statevector(exact.ground_state)))

        # Task 8.4 + 8.5: build DeployResult with metrics and phase classification
        return self._build_deploy_result(
            route="adapt_vqe",
            h_test=exact.h_value,
            predicted_energy=e_final,
            exact=exact,
            mag_x_pred=mag_x_pred,
            corr_zz_pred=corr_zz_pred,
            fidelity=fidelity,
            adapt_iterations=adapt_iterations,
        )

    # ── Task 8.3: deploy_qrc() ───────────────────────────────────────

    def deploy_qrc(
        self,
        qrc_pipeline: "QRCPipeline",
        h_test: float,
        exact: GroundTruthResult,
    ) -> DeployResult:
        """Fallback route: QRC encode → measure → linear readout.

        Parameters
        ----------
        qrc_pipeline : QRCPipeline — trained QRC pipeline
        h_test : float — test transverse field value
        exact : GroundTruthResult — exact reference
        """
        mag_x_pred, corr_zz_pred = qrc_pipeline.predict(h_test)

        # QRC doesn't produce a circuit energy directly; estimate from observables
        # E ≈ -J * n_bonds * avg_corr_zz - h * n_sites * avg_mag_x
        n_bonds = len(exact.per_bond_corr_zz)
        n_sites = len(exact.per_site_mag_x)
        # Infer J from exact data: J = -E_zz_contribution / n_bonds / avg_corr_zz
        # For simplicity, use the exact energy decomposition when available
        predicted_energy = -1.0 * n_bonds * corr_zz_pred - h_test * n_sites * mag_x_pred

        return self._build_deploy_result(
            route="qrc",
            h_test=h_test,
            predicted_energy=predicted_energy,
            exact=exact,
            mag_x_pred=mag_x_pred,
            corr_zz_pred=corr_zz_pred,
            fidelity=None,
            adapt_iterations=0,
        )

    # ── Task 8.4: DeployResult construction ──────────────────────────

    def _build_deploy_result(
        self,
        route: str,
        h_test: float,
        predicted_energy: float,
        exact: GroundTruthResult,
        mag_x_pred: float,
        corr_zz_pred: float,
        fidelity: Optional[float],
        adapt_iterations: int,
    ) -> DeployResult:
        """Build DeployResult with all validation metrics and pass/fail checklist."""
        delta_e = abs(predicted_energy - exact.ground_energy)
        gap = exact.gap if exact.gap > 1e-10 else 1e-10
        delta_e_over_gap = delta_e / gap
        mag_x_error = abs(mag_x_pred - exact.mag_x)
        corr_zz_error = abs(corr_zz_pred - exact.corr_zz)

        # Task 8.5: phase classification via observable crossover
        phase_label = self.classify_phase(mag_x_pred, corr_zz_pred)

        # Validation checklist (ordered by priority)
        checklist = {
            "delta_e_over_gap_lt_5pct": delta_e_over_gap < 0.05,
            "mag_x_error_lt_1e-2": mag_x_error < 1e-2,
            "corr_zz_error_lt_1e-2": corr_zz_error < 1e-2,
            "delta_e_lt_1e-2": delta_e < 1e-2,
            "fidelity_gte_995": fidelity is not None and fidelity >= 0.995,
            "adapt_iterations_lte_2": adapt_iterations <= 2,
        }

        return DeployResult(
            route=route,
            h_test=h_test,
            predicted_energy=predicted_energy,
            delta_e=delta_e,
            delta_e_over_gap=delta_e_over_gap,
            mag_x_pred=mag_x_pred,
            corr_zz_pred=corr_zz_pred,
            mag_x_error=mag_x_error,
            corr_zz_error=corr_zz_error,
            fidelity=fidelity,
            adapt_iterations=adapt_iterations,
            phase_label=phase_label,
            metrics_checklist=checklist,
        )

    # ── Task 8.5: phase classification ───────────────────────────────

    @staticmethod
    def classify_phase(mag_x: float, corr_zz: float) -> str:
        """Classify phase using data-driven ⟨X⟩ = ⟨ZZ⟩ crossover.

        When |⟨X⟩| > |⟨ZZ⟩|, the system is in the paramagnetic phase.
        When |⟨ZZ⟩| > |⟨X⟩|, the system is in the ferromagnetic phase.
        This is data-driven (not hardcoded h_c=1.0).
        """
        if abs(mag_x) > abs(corr_zz):
            return "paramagnetic"
        else:
            return "ferromagnetic"

    @staticmethod
    def find_critical_point(
        h_values: np.ndarray,
        mag_x_values: np.ndarray,
        corr_zz_values: np.ndarray,
    ) -> float:
        """Find the data-driven critical point where ⟨X⟩ = ⟨ZZ⟩ crossover occurs.

        Uses linear interpolation between the two h-points where the crossover
        happens.
        """
        diff = np.abs(mag_x_values) - np.abs(corr_zz_values)
        # Find sign change
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                # Linear interpolation
                h_c = h_values[i] + (h_values[i + 1] - h_values[i]) * (
                    abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
                )
                return float(h_c)
        # Fallback: return h where |diff| is smallest
        return float(h_values[np.argmin(np.abs(diff))])
