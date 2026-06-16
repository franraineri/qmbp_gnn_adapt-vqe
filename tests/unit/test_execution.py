"""Unit tests for qmbp_simulation.execution module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.execution import (
    ExecutionBackend,
    HardwareBackend,
    NoiselessBackend,
)
from qmbp_simulation.models import HamiltonianBuilder


class TestNoiselessBackend:
    """Test NoiselessBackend.evaluate() returns finite float."""

    def test_evaluate_returns_finite_float(self, small_lattice, small_circuit):
        backend = NoiselessBackend()
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, theta = small_circuit
        params = np.zeros(len(theta))
        energy = backend.evaluate(qc, H, params)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_name_property(self):
        backend = NoiselessBackend()
        assert backend.name == "noiseless_statevector"


class TestHardwareBackend:
    """Test HardwareBackend basic interface."""

    def test_hardware_mode_requires_credentials(self):
        """Hardware mode raises ValueError without IBM_KEY."""
        from qmbp_simulation.execution.hardware import HardwareConfig

        config = HardwareConfig(mode="hardware")
        backend = HardwareBackend(config=config)
        with pytest.raises(ValueError, match="IBM_KEY"):
            _ = backend.backend  # Triggers lazy connection

    def test_name_property(self):
        from qmbp_simulation.execution.hardware import HardwareConfig

        config = HardwareConfig(mode="fake_backend", backend_name="ibm_kingston")
        backend = HardwareBackend(config=config)
        assert "ibm_kingston" in backend.name


class TestBackendPolymorphism:
    """Test backend polymorphism — same interface, different implementations."""

    def test_all_backends_are_execution_backend(self):
        assert issubclass(NoiselessBackend, ExecutionBackend)
        assert issubclass(HardwareBackend, ExecutionBackend)

    def test_noiseless_and_mock_share_interface(
        self, noiseless_backend, mock_backend, small_lattice, small_circuit
    ):
        """Both backends implement evaluate() with same signature."""
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, theta = small_circuit
        params = np.zeros(len(theta))

        # Both should return float
        e1 = noiseless_backend.evaluate(qc, H, params)
        e2 = mock_backend.evaluate(qc, H, params)
        assert isinstance(e1, float)
        assert isinstance(e2, float)
