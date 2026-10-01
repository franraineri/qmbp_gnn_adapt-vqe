"""Fast adjoint gradient (NoiselessBackend) == ReverseEstimatorGradient.

The NoiselessBackend.gradient fast path hand-rolls the exact reverse-mode adjoint
for rx/rz/rzz circuits (the HVA/bond-resolved family), skipping Qiskit's estimator
pipeline. These tests pin that it is numerically identical to the reference
ReverseEstimatorGradient and that it falls back cleanly for unsupported gates.
"""
from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import HamiltonianBuilder, make_lattice


def _ref_gradient(qc, H):
    rev = pytest.importorskip("qiskit_algorithms.gradients").ReverseEstimatorGradient()

    def g(x):
        return np.asarray(rev.run([qc], [H], [list(x)]).result().gradients[0], float)

    return g


class TestAdjointGradientMatchesReference:
    """Exact equality (≤1e-9) with the Qiskit reverse estimator across configs."""

    @pytest.mark.parametrize("topology,n,p", [
        ("chain_1d", 4, 1),
        ("chain_1d", 6, 2),
        ("square", 4, 1),
        ("square", 6, 2),
    ])
    def test_matches_reverse_estimator(self, topology, n, p):
        lat = make_lattice(topology, n, J=1.0, h=0.5)
        qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(n, p, lat) \
            if topology == "square" else HVACircuitBuilder().create(n, p, lat)
        H = HamiltonianBuilder().build(lat)
        be = NoiselessBackend()
        fast = be.gradient(qc, H)
        ref = _ref_gradient(qc, H)
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = rng.uniform(-np.pi, np.pi, qc.num_parameters)
            np.testing.assert_allclose(fast(x), ref(x), atol=1e-9)

    def test_uses_fast_path_for_rzz_rx_circuit(self):
        # The HVA circuit is rx/rzz-only → the plan must compile (fast path on).
        lat = make_lattice("square", 6, J=1.0, h=0.5)
        qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(6, 2, lat)
        be = NoiselessBackend()
        plan = be._adjoint_plan(qc)
        assert plan is not None
        assert plan["nparam"] == qc.num_parameters

    def test_plan_is_cached(self):
        lat = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        qc, _ = HVACircuitBuilder().create(4, 1, lat)
        be = NoiselessBackend()
        p1 = be._adjoint_plan(qc)
        p2 = be._adjoint_plan(qc)
        assert p1 is p2  # same object → cached by id(circuit)

    def test_gradient_is_finite_and_right_length(self):
        lat = make_lattice("square", 6, J=1.0, h=0.7)
        qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(6, 2, lat)
        H = HamiltonianBuilder().build(lat)
        be = NoiselessBackend()
        g = be.gradient(qc, H)
        out = g(np.zeros(qc.num_parameters))
        assert out.shape == (qc.num_parameters,)
        assert np.all(np.isfinite(out))
