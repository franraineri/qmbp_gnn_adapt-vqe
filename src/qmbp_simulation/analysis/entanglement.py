"""Entanglement analysis for ground state characterization.

Computes Von Neumann entanglement entropy to quantify ground state
complexity and explain HVA expressibility limits. The key insight:
HVA circuits with limited depth can only represent states up to a
certain entanglement entropy threshold.

Usage:
    from qmbp_simulation.analysis import EntanglementAnalyzer, EntanglementResult

    analyzer = EntanglementAnalyzer()
    entropy = analyzer.compute_half_chain_entropy(statevector, n_qubits=6)
    results = analyzer.analyze_sweep(h_values, ground_states, n_qubits=6)
    capacity = analyzer.find_hva_capacity_threshold(results, fidelities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntanglementResult:
    """Result of entanglement analysis for a single ground state.

    Attributes
    ----------
    h : float
        External field value.
    entropy : float
        Half-chain Von Neumann entanglement entropy S = -Tr(ρ_A log₂ ρ_A).
    n_qubits : int
        Total number of qubits in the system.
    partition_size : int
        Size of partition A (N // 2).
    max_entropy : float
        Maximum possible entropy for this partition: log₂(min(dim_A, dim_B)).
    normalized_entropy : float
        Entropy normalized by max_entropy ∈ [0, 1].
    """

    h: float
    entropy: float
    n_qubits: int
    partition_size: int
    max_entropy: float
    normalized_entropy: float


class EntanglementAnalyzer:
    """Computes entanglement entropy for ground state characterization.

    Uses the Schmidt decomposition (SVD) to compute the half-chain
    Von Neumann entanglement entropy efficiently.
    """

    def compute_half_chain_entropy(
        self,
        statevector: np.ndarray,
        n_qubits: int,
    ) -> float:
        """Compute S = -Tr(ρ_A log₂ ρ_A) for partition A = first N//2 qubits.

        Parameters
        ----------
        statevector : np.ndarray
            Normalized state vector with 2^N components.
        n_qubits : int
            Total number of qubits.

        Returns
        -------
        float
            Entanglement entropy in bits (base-2 logarithm).

        Notes
        -----
        Uses SVD of the reshaped state matrix to compute Schmidt coefficients.
        This is numerically stable and O(2^N) in memory.
        """
        n_a = n_qubits // 2
        n_b = n_qubits - n_a
        dim_a = 2**n_a
        dim_b = 2**n_b

        # Reshape statevector into bipartite matrix
        psi_matrix = statevector.reshape(dim_a, dim_b)

        # Schmidt decomposition via SVD
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

        # Schmidt probabilities (eigenvalues of reduced density matrix)
        probs = singular_values**2
        # Filter numerical zeros to avoid log(0)
        probs = probs[probs > 1e-15]

        # Von Neumann entropy: S = -Σ p_i log₂(p_i)
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    def analyze_sweep(
        self,
        h_values: np.ndarray,
        ground_states: list[np.ndarray],
        n_qubits: int,
    ) -> list[EntanglementResult]:
        """Analyze entanglement entropy for a full h-sweep.

        Parameters
        ----------
        h_values : np.ndarray
            External field values.
        ground_states : list[np.ndarray]
            Ground state vectors for each h-value.
        n_qubits : int
            Number of qubits.

        Returns
        -------
        list[EntanglementResult]
            Entanglement results for each h-point.
        """
        n_a = n_qubits // 2
        max_entropy = float(np.log2(min(2**n_a, 2 ** (n_qubits - n_a))))

        results = []
        for h, psi in zip(h_values, ground_states, strict=False):
            entropy = self.compute_half_chain_entropy(psi, n_qubits)
            results.append(
                EntanglementResult(
                    h=float(h),
                    entropy=entropy,
                    n_qubits=n_qubits,
                    partition_size=n_a,
                    max_entropy=max_entropy,
                    normalized_entropy=entropy / max_entropy if max_entropy > 0 else 0.0,
                )
            )

        logger.info(
            "Entanglement sweep: %d points, S_min=%.3f, S_max=%.3f (max possible=%.3f)",
            len(results),
            min(r.entropy for r in results) if results else 0,
            max(r.entropy for r in results) if results else 0,
            max_entropy,
        )
        return results

    def find_hva_capacity_threshold(
        self,
        entanglement_results: list[EntanglementResult],
        fidelities: list[float],
        fidelity_threshold: float = 0.93,
    ) -> float | None:
        """Find the maximum entropy where HVA still achieves fidelity ≥ threshold.

        This identifies the "capacity" of the HVA ansatz: the maximum
        ground state complexity it can represent at the given depth.

        Parameters
        ----------
        entanglement_results : list[EntanglementResult]
            Entanglement results from analyze_sweep.
        fidelities : list[float]
            VQE fidelities at each h-point (same order as entanglement_results).
        fidelity_threshold : float
            Minimum fidelity to consider a point "representable".

        Returns
        -------
        float | None
            Maximum entropy where fidelity ≥ threshold, or None if no point qualifies.
        """
        max_entropy = None
        for ent, fid in zip(entanglement_results, fidelities, strict=False):
            if fid >= fidelity_threshold:
                if max_entropy is None or ent.entropy > max_entropy:
                    max_entropy = ent.entropy

        if max_entropy is not None:
            logger.info(
                "HVA capacity at fidelity≥%.2f: S_max=%.4f bits",
                fidelity_threshold,
                max_entropy,
            )
        else:
            logger.warning(
                "No h-point achieves fidelity≥%.2f — HVA capacity is zero for this model.",
                fidelity_threshold,
            )
        return max_entropy
