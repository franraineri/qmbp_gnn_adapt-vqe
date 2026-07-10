"""Unit tests for qmbp_simulation.circuits module."""

from __future__ import annotations

import pytest

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.models import VQEConfig, make_lattice


class TestHVACircuitBuilder:
    """Test HVACircuitBuilder.create() parameter count and constraints."""

    def test_parameter_count_p1(self, small_lattice):
        builder = HVACircuitBuilder()
        qc, theta = builder.create(4, 1, small_lattice)
        # p=1 → 2*1 = 2 parameters
        assert len(theta) == 2
        assert qc.num_parameters == 2

    def test_parameter_count_p2(self, small_lattice):
        builder = HVACircuitBuilder()
        qc, theta = builder.create(4, 2, small_lattice)
        # p=2 → 2*2 = 4 parameters
        assert len(theta) == 4
        assert qc.num_parameters == 4

    def test_p_greater_than_max_raises_value_error(self, small_lattice):
        """p_layers above MAX_P_LAYERS is rejected by VQEConfig (not circuit builder)."""
        from qmbp_simulation.models.constants import MAX_P_LAYERS

        # Circuit builder allows any p (validation moved to VQEConfig)
        builder = HVACircuitBuilder()
        qc, _ = builder.create(4, 3, small_lattice)
        assert qc.num_parameters == 6  # 2 params/layer × 3 layers

        # VQEConfig enforces the upper bound
        with pytest.raises(ValueError, match="p_layers must be ≤"):
            VQEConfig(p_layers=MAX_P_LAYERS + 1)

    def test_qubit_mismatch_raises(self):
        lattice = make_lattice("chain_1d", 4)
        builder = HVACircuitBuilder()
        with pytest.raises(ValueError, match="does not match"):
            builder.create(6, 1, lattice)

    def test_circuit_has_correct_qubit_count(self, small_lattice):
        builder = HVACircuitBuilder()
        qc, _ = builder.create(4, 1, small_lattice)
        assert qc.num_qubits == 4
