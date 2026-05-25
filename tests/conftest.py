"""Shared fixtures for qmbp_simulation test suite."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import ExecutionBackend, NoiselessBackend
from qmbp_simulation.models import LatticeConfig, make_lattice

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed_rng():
    """Pin random seeds for reproducibility."""
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Lattice and Circuit Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_lattice() -> LatticeConfig:
    """N=4 chain lattice for fast unit tests."""
    return make_lattice("chain_1d", 4, J=1.0, h=1.0)


@pytest.fixture
def small_circuit(small_lattice) -> tuple[QuantumCircuit, object]:
    """N=4, p=1 HVA circuit for fast unit tests."""
    builder = HVACircuitBuilder()
    qc, theta = builder.create(4, 1, small_lattice)
    return qc, theta


# ─────────────────────────────────────────────────────────────────────────────
# Backend Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def noiseless_backend() -> NoiselessBackend:
    """Exact statevector backend for tests."""
    return NoiselessBackend()


class _MockBackend(ExecutionBackend):
    """Backend that returns a constant energy for fast optimizer tests."""

    def __init__(self, constant_energy: float = -5.0):
        self._energy = constant_energy

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        return self._energy

    @property
    def name(self) -> str:
        return "mock_constant"


@pytest.fixture
def mock_backend() -> _MockBackend:
    """Backend returning constant energy=-5.0 for fast optimizer tests."""
    return _MockBackend(constant_energy=-5.0)
