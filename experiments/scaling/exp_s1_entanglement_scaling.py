"""S1: Entanglement Entropy vs Valid Regime Boundary.

Hypothesis: The entanglement entropy S(h_min, N) is approximately constant
across all N, meaning the HVA p=2 has a fixed entanglement capacity and
h_min is where the ground state exceeds that capacity.

Method:
    1. For each N in [4, 6, 8, 10], compute exact ground state at h_min(N)
    2. Calculate half-chain entanglement entropy S = -Tr(ρ_A log₂ ρ_A)
    3. Also compute S(h) for a sweep to show the full curve
    4. Check if S(h_min) ≈ constant across N
    5. Repeat for p=1 boundaries

Expected outcome:
    S(h_min) ≈ 0.5-0.8 bits (constant within std < 0.1).
    If confirmed, the scaling exponent β is a consequence of how S(h, N)
    scales with N in the TFIM (known analytically).

Thesis value: HIGH — transforms empirical scaling law into causal explanation.

References:
    - A3 experiment: h_min = 1.0 + 0.020·N^1.33 (p=2), 1.0 + 0.212·N^0.60 (p=1)
    - Dutta et al. (2015): TFIM entanglement scaling
    - Wiersema et al. (2020): HVA entanglement structure
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import AnalysisConfig, ExperimentConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)

# Known valid regime boundaries from A3 + p1-scaling experiments
H_MIN_P2: dict[int, float] = {4: 0.95, 6: 1.20, 8: 1.30, 10: 1.40}
H_MIN_P1: dict[int, float] = {6: 1.60, 10: 1.90}


def compute_entanglement_entropy(statevector: np.ndarray, n_qubits: int) -> float:
    """Compute half-chain entanglement entropy S = -Tr(ρ_A log₂ ρ_A).

    Parameters
    ----------
    statevector : np.ndarray
        Full statevector of shape (2^N,).
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    float
        Von Neumann entropy of the reduced density matrix (in bits).
    """
    n_a = n_qubits // 2
    n_b = n_qubits - n_a
    dim_a = 2**n_a
    dim_b = 2**n_b

    # Reshape into bipartite system and compute reduced density matrix
    psi = statevector.reshape(dim_a, dim_b)
    # Schmidt decomposition via SVD
    singular_values = np.linalg.svd(psi, compute_uv=False)

    # Entanglement entropy from Schmidt coefficients
    probs = singular_values**2
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 1e-15]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


class ExperimentS1(BaseExperiment):
    """Entanglement entropy correlation with valid regime boundary."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S1",
            category="S",
            description="Entanglement entropy at h_min — explains scaling law causally",
            hypothesis=(
                "S(h_min, N) is approximately constant across N, meaning "
                "HVA p=2 has a fixed entanglement capacity ~0.5-0.8 bits. "
                "The scaling exponent β=1.33 is a consequence of how "
                "S(h, N) grows with N in the TFIM."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            ),
            analysis=AnalysisConfig(scaling_n_values=[4, 6, 8, 10]),
            seeds=[42],  # Entropy is deterministic — 1 seed sufficient
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def setup(self) -> None:
        """Setup solver for exact diagonalization."""
        from qmbp_simulation import ClassicalSolver, HamiltonianBuilder

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        logger.info(
            "S1 setup: N_values=%s, h_sweep=%s",
            self.config.analysis.scaling_n_values,
            self.config.system.h_values,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Compute entanglement entropy at h_min and across h-sweep for each N."""
        from qmbp_simulation import make_lattice

        N_values = self.config.analysis.scaling_n_values
        h_sweep = self.config.system.h_values
        metrics = []

        s_at_boundary_p2 = {}
        s_at_boundary_p1 = {}

        for N in N_values:
            t0 = time.time()
            s_curve = {}

            # Compute S(h) for the full sweep
            for h in h_sweep:
                lattice = make_lattice("chain_1d", N, J=1.0, h=h)
                H = self.builder.build(lattice)
                result = self.solver.solve(H, lattice)
                if result.ground_state is None:
                    logger.warning("  N=%d, h=%.2f: no statevector (DMRG?), skipping", N, h)
                    continue
                sv = np.asarray(result.ground_state)
                s_curve[h] = compute_entanglement_entropy(sv, N)

            # Compute S at h_min(N) for p=2
            h_min_p2 = H_MIN_P2.get(N)
            s_boundary_p2 = None
            if h_min_p2 is not None:
                lattice = make_lattice("chain_1d", N, J=1.0, h=h_min_p2)
                H = self.builder.build(lattice)
                result = self.solver.solve(H, lattice)
                if result.ground_state is not None:
                    sv = np.asarray(result.ground_state)
                    s_boundary_p2 = compute_entanglement_entropy(sv, N)
                    s_at_boundary_p2[N] = s_boundary_p2

            # Compute S at h_min(N) for p=1 (if available)
            h_min_p1 = H_MIN_P1.get(N)
            s_boundary_p1 = None
            if h_min_p1 is not None:
                lattice = make_lattice("chain_1d", N, J=1.0, h=h_min_p1)
                H = self.builder.build(lattice)
                result = self.solver.solve(H, lattice)
                if result.ground_state is not None:
                    sv = np.asarray(result.ground_state)
                    s_boundary_p1 = compute_entanglement_entropy(sv, N)
                    s_at_boundary_p1[N] = s_boundary_p1

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h_min_p2 if h_min_p2 else 0.0,
                energy=0.0,
                exact_energy=0.0,
                energy_error=0.0,
                gap=1.0,
                relative_error=0.0,
                seed=seed,
                wall_time_s=elapsed,
                converged=True,
                technique_metadata={
                    "N": N,
                    "h_min_p2": h_min_p2,
                    "h_min_p1": h_min_p1,
                    "S_at_boundary_p2": s_boundary_p2,
                    "S_at_boundary_p1": s_boundary_p1,
                    "S_curve": {str(h): s for h, s in s_curve.items()},
                    "max_S": max(s_curve.values()),
                    "S_at_h_1.0": s_curve.get(1.0),
                },
            )
            metrics.append(m)
            logger.info(
                "  N=%d: S(h_min_p2=%.2f)=%.4f, S(h=1.0)=%.4f (%.1fs)",
                N,
                h_min_p2 or 0,
                s_boundary_p2 or 0,
                s_curve.get(1.0, 0),
                elapsed,
            )

        # Summary analysis
        if s_at_boundary_p2:
            values = list(s_at_boundary_p2.values())
            mean_s = np.mean(values)
            std_s = np.std(values)
            logger.info(
                "\n  === RESULT: S(h_min, p=2) = %.4f ± %.4f (N=%s) ===",
                mean_s,
                std_s,
                list(s_at_boundary_p2.keys()),
            )
            logger.info(
                "  Hypothesis %s: std=%.4f %s 0.1",
                "CONFIRMED" if std_s < 0.1 else "REJECTED",
                std_s,
                "<" if std_s < 0.1 else ">=",
            )

        if s_at_boundary_p1:
            values = list(s_at_boundary_p1.values())
            mean_s = np.mean(values)
            std_s = np.std(values)
            logger.info(
                "  === RESULT: S(h_min, p=1) = %.4f ± %.4f (N=%s) ===",
                mean_s,
                std_s,
                list(s_at_boundary_p1.keys()),
            )

        return metrics
