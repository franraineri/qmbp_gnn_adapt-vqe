"""Unit tests for EntanglementAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qmbp_simulation.analysis.entanglement import (
    EntanglementAnalyzer,
    EntanglementResult,
)


@pytest.fixture
def analyzer():
    """Fresh EntanglementAnalyzer instance."""
    return EntanglementAnalyzer()


class TestEntanglementAnalyzer:
    """Tests for entanglement entropy computation."""

    def test_product_state_zero_entropy(self, analyzer):
        """|00...0⟩ is a product state with S=0."""
        n_qubits = 4
        psi = np.zeros(2**n_qubits)
        psi[0] = 1.0  # |0000⟩
        entropy = analyzer.compute_half_chain_entropy(psi, n_qubits)
        assert_allclose(entropy, 0.0, atol=1e-10)

    def test_bell_state_one_bit_entropy(self, analyzer):
        """Bell state (|00⟩ + |11⟩)/√2 has S=1.0 bit."""
        n_qubits = 2
        psi = np.zeros(4)
        psi[0] = 1.0 / np.sqrt(2)  # |00⟩
        psi[3] = 1.0 / np.sqrt(2)  # |11⟩
        entropy = analyzer.compute_half_chain_entropy(psi, n_qubits)
        assert_allclose(entropy, 1.0, atol=1e-10)

    def test_ghz_state_one_bit_entropy(self, analyzer):
        """GHZ state (|0000⟩ + |1111⟩)/√2 for N=4 has S=1.0 bit."""
        n_qubits = 4
        psi = np.zeros(2**n_qubits)
        psi[0] = 1.0 / np.sqrt(2)   # |0000⟩
        psi[-1] = 1.0 / np.sqrt(2)  # |1111⟩
        entropy = analyzer.compute_half_chain_entropy(psi, n_qubits)
        assert_allclose(entropy, 1.0, atol=1e-10)

    def test_entropy_bounds(self, analyzer):
        """For random states, 0 ≤ S ≤ N/2."""
        n_qubits = 4
        rng = np.random.default_rng(42)
        # Random normalized state
        psi = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
        psi /= np.linalg.norm(psi)
        entropy = analyzer.compute_half_chain_entropy(psi, n_qubits)
        assert entropy >= 0.0
        max_entropy = n_qubits / 2  # log2(2^(N/2)) = N/2
        assert entropy <= max_entropy + 1e-10

    def test_analyze_sweep_length(self, analyzer):
        """Output length matches input."""
        n_qubits = 4
        h_values = np.array([0.5, 1.0, 1.5, 2.0])
        # Use product states for simplicity
        ground_states = [np.zeros(2**n_qubits) for _ in h_values]
        for gs in ground_states:
            gs[0] = 1.0
        results = analyzer.analyze_sweep(h_values, ground_states, n_qubits)
        assert len(results) == len(h_values)

    def test_analyze_sweep_normalized_entropy(self, analyzer):
        """All normalized_entropy values are in [0, 1]."""
        n_qubits = 4
        h_values = np.array([0.5, 1.0, 1.5])
        rng = np.random.default_rng(123)
        ground_states = []
        for _ in h_values:
            psi = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
            psi /= np.linalg.norm(psi)
            ground_states.append(psi)
        results = analyzer.analyze_sweep(h_values, ground_states, n_qubits)
        for r in results:
            assert 0.0 <= r.normalized_entropy <= 1.0 + 1e-10

    def test_find_hva_capacity_no_qualifying_points(self, analyzer):
        """Returns None when no fidelity passes threshold."""
        results = [
            EntanglementResult(h=1.0, entropy=0.5, n_qubits=4,
                               partition_size=2, max_entropy=2.0, normalized_entropy=0.25),
            EntanglementResult(h=1.5, entropy=0.8, n_qubits=4,
                               partition_size=2, max_entropy=2.0, normalized_entropy=0.40),
        ]
        fidelities = [0.5, 0.6]  # All below default threshold 0.93
        capacity = analyzer.find_hva_capacity_threshold(results, fidelities)
        assert capacity is None

    def test_find_hva_capacity_returns_max_entropy(self, analyzer):
        """Returns max entropy among qualifying points."""
        results = [
            EntanglementResult(h=1.0, entropy=0.3, n_qubits=4,
                               partition_size=2, max_entropy=2.0, normalized_entropy=0.15),
            EntanglementResult(h=1.5, entropy=0.8, n_qubits=4,
                               partition_size=2, max_entropy=2.0, normalized_entropy=0.40),
            EntanglementResult(h=2.0, entropy=0.5, n_qubits=4,
                               partition_size=2, max_entropy=2.0, normalized_entropy=0.25),
        ]
        fidelities = [0.95, 0.90, 0.97]  # indices 0 and 2 pass
        capacity = analyzer.find_hva_capacity_threshold(results, fidelities)
        assert_allclose(capacity, 0.5)  # max of 0.3 and 0.5

    def test_entanglement_result_fields(self, analyzer):
        """Verify all dataclass fields are populated correctly."""
        n_qubits = 4
        h_values = np.array([1.0])
        psi = np.zeros(2**n_qubits)
        psi[0] = 1.0
        results = analyzer.analyze_sweep(h_values, [psi], n_qubits)
        r = results[0]
        assert r.h == 1.0
        assert r.n_qubits == 4
        assert r.partition_size == 2
        assert r.max_entropy == 2.0  # log2(2^2) = 2
        assert 0.0 <= r.normalized_entropy <= 1.0
        assert isinstance(r.entropy, float)
