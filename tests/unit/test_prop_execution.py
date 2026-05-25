"""Property-based tests for qmbp_simulation.execution submodule.

Uses Hypothesis to verify universal properties of execution backends
across many random parameter inputs.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend, NoisyBackend
from qmbp_simulation.models import HamiltonianBuilder, make_lattice

# ─────────────────────────────────────────────────────────────────────────────
# Property 6: Execution backend polymorphism
# **Validates: Requirements 5.2, 5.5**
# ─────────────────────────────────────────────────────────────────────────────


# Fixed small lattice (N=4, chain_1d, p=1 → 2 parameters) for speed
_LATTICE = make_lattice("chain_1d", 4, J=1.0, h=1.0)
_BUILDER = HVACircuitBuilder()
_QC, _THETA = _BUILDER.create(4, 1, _LATTICE)
_H = HamiltonianBuilder().build(_LATTICE)


class TestProperty6ExecutionBackendPolymorphism:
    """Property 6: For any valid circuit parameters (random floats in [-π, π]),
    evaluate() returns a finite float for both NoiselessBackend and NoisyBackend.
    """

    @given(
        params=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi),
            min_size=2,
            max_size=2,
        ).map(np.array),
    )
    @settings(max_examples=30, deadline=None)
    def test_noiseless_evaluate_returns_finite_float(self, params):
        """NoiselessBackend.evaluate() returns a finite float for any valid params."""
        backend = NoiselessBackend()
        energy = backend.evaluate(_QC, _H, params)

        assert isinstance(energy, float), f"Expected float, got {type(energy)}"
        assert np.isfinite(energy), f"Expected finite, got {energy}"

    @given(
        params=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi),
            min_size=2,
            max_size=2,
        ).map(np.array),
    )
    @settings(max_examples=30, deadline=None)
    def test_noisy_evaluate_returns_finite_float(self, params):
        """NoisyBackend.evaluate() (Gaussian shot noise mode) returns a finite float."""
        backend = NoisyBackend(shots=8192, noise_model=None)
        energy = backend.evaluate(_QC, _H, params)

        assert isinstance(energy, float), f"Expected float, got {type(energy)}"
        assert np.isfinite(energy), f"Expected finite, got {energy}"
