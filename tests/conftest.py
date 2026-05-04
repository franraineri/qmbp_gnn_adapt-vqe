"""Shared fixtures for V6 pipeline tests."""

import numpy as np
import pytest

from src.poc.v6 import (
    HamiltonianBuilder,
    make_lattice,
    ClassicalSolver,
    HVACircuitBuilder,
    LatticeConfig,
    GroundTruthResult,
)


@pytest.fixture(autouse=True)
def seed_rng():
    """Pin random seeds for reproducibility."""
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass


@pytest.fixture
def builder():
    return HamiltonianBuilder()


@pytest.fixture
def solver():
    return ClassicalSolver()


@pytest.fixture
def hva():
    return HVACircuitBuilder()


@pytest.fixture
def chain_6() -> LatticeConfig:
    """Standard 1D chain N=6 lattice for testing."""
    return make_lattice("chain_1d", 6, J=1.0, h=1.0)


@pytest.fixture
def h_values_reduced() -> np.ndarray:
    """Reduced h-grid for fast tests (6 points)."""
    return np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])


@pytest.fixture
def exact_data_reduced(builder, solver, h_values_reduced):
    """Exact diag results for the reduced h-grid."""
    results = []
    for h in h_values_reduced:
        lat = make_lattice("chain_1d", 6, J=1.0, h=h)
        H = builder.build(lat)
        results.append(solver.solve(H, lat))
    return results
