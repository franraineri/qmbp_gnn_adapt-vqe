"""Tests for the shared quench-dynamics helpers (analysis.dynamics) and the
frustrated / Kitaev Trotter steps (circuits.trotter).

Covers the primitives extracted from the three dynamics runners so the single
source of truth is pinned: MPS truncation, classical crossover (dense + sparse),
the ⟨Z_iZ_j⟩ correlator, 2q counting, exact time evolution, and the two Trotter
steps (energy conservation under exact unitary evolution).
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from qmbp_simulation.analysis.dynamics import (
    NIGHTHAWK_CZ_ERROR,
    NOISE_FLOOR,
    classical_crossover,
    classical_crossover_sparse,
    correlator_zz,
    count_2q_gates,
    evolve_states,
    fidelity_from_2q,
    gates_to_floor,
    tnoise_from_2q_curve,
    truncate_mps,
)
from qmbp_simulation.circuits.trotter import (
    build_frustrated_trotter_step,
    build_kitaev_trotter_step,
)


def _random_state(n_qubits, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return psi / np.linalg.norm(psi)


class TestTruncateMps:
    def test_large_chi_is_lossless(self):
        n = 4
        psi = _random_state(n)
        # chi >= 2^(n/2) can represent any state exactly.
        trunc = truncate_mps(psi, n, chi_max=16)
        fidelity = abs(np.vdot(psi, trunc)) ** 2
        np.testing.assert_allclose(fidelity, 1.0, atol=1e-10)

    def test_output_is_normalized(self):
        n = 4
        trunc = truncate_mps(_random_state(n), n, chi_max=2)
        np.testing.assert_allclose(np.linalg.norm(trunc), 1.0, atol=1e-10)

    def test_product_state_survives_chi1(self):
        # |+⟩^n is a bond-dimension-1 product state → χ=1 keeps it exact.
        n = 4
        qc = QuantumCircuit(n)
        qc.h(range(n))
        psi = np.asarray(Statevector(qc).data)
        trunc = truncate_mps(psi, n, chi_max=1)
        np.testing.assert_allclose(abs(np.vdot(psi, trunc)) ** 2, 1.0, atol=1e-10)


class TestCorrelatorZz:
    def test_matches_dense_operator(self):
        n = 4
        psi = _random_state(n, seed=3)
        for i, j in [(0, 1), (0, 3), (1, 2)]:
            # Dense ⟨Z_i Z_j⟩ via Pauli string (qiskit little-endian: qubit 0 = rightmost).
            labels = ["I"] * n
            labels[n - 1 - i] = "Z"
            labels[n - 1 - j] = "Z"
            from qiskit.quantum_info import SparsePauliOp

            op = SparsePauliOp("".join(labels))
            dense = float(np.real(Statevector(psi).expectation_value(op)))
            np.testing.assert_allclose(correlator_zz(psi, i, j, n), dense, atol=1e-10)

    def test_self_correlator_is_one(self):
        n = 3
        psi = _random_state(n)
        np.testing.assert_allclose(correlator_zz(psi, 1, 1, n), 1.0, atol=1e-10)


class TestCount2qGates:
    def test_counts_two_qubit_ops(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.rzz(0.3, 0, 1)
        qc.rxx(0.2, 1, 2)
        qc.rx(0.1, 2)
        assert count_2q_gates(qc) == 2


class TestEvolveStates:
    def test_norm_preserved(self):
        from scipy.linalg import expm

        n = 3
        psi0 = _random_state(n)
        H = np.diag(np.arange(2**n, dtype=float))
        U = expm(-1j * H * 0.1)
        states = evolve_states(psi0, 5, propagator=U)
        assert len(states) == 6
        for s in states:
            np.testing.assert_allclose(np.linalg.norm(s), 1.0, atol=1e-10)

    def test_propagator_and_evolver_agree(self):
        from scipy.linalg import expm

        n = 3
        psi0 = _random_state(n, seed=7)
        H = np.diag(np.linspace(-1, 1, 2**n))
        U = expm(-1j * H * 0.1)
        via_prop = evolve_states(psi0, 4, propagator=U)
        via_evolver = evolve_states(psi0, 4, evolver=lambda p: U @ p)
        for a, b in zip(via_prop, via_evolver, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_requires_exactly_one_rule(self):
        psi0 = _random_state(2)
        with pytest.raises(ValueError):
            evolve_states(psi0, 3)  # neither propagator nor evolver


class TestClassicalCrossover:
    def test_monotone_in_chi(self):
        # Larger χ tracks the exact state at least as long → t* non-decreasing.
        # Use a dense random Hermitian H as the evolution generator (entangling).
        n = 6
        rng = np.random.default_rng(5)
        A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        H = A + A.conj().T  # Hermitian
        psi0 = _random_state(n, seed=5)
        out = classical_crossover(psi0, H, dt=0.1, n_steps=8, n_qubits=n,
                                  chi_values=[1, 2, 4, 8])
        # None (never crossed) sorts last via a large sentinel.
        big = 10**6
        seq = [out[c] if out[c] is not None else big for c in (1, 2, 4, 8)]
        assert seq == sorted(seq)

    def test_sparse_matches_dense(self):
        from scipy.linalg import expm
        from scipy.sparse import csr_matrix

        n = 4
        H = np.diag(np.linspace(-2, 2, 2**n)).astype(complex)
        # Make it entangling-ish with an off-diagonal coupling.
        H[0, -1] = H[-1, 0] = 0.5
        psi0 = _random_state(n, seed=9)
        dt, steps, chis = 0.4, 6, [2, 4]
        dense = classical_crossover(psi0, H, dt, steps, n, chis,
                                    propagator=expm(-1j * H * dt))
        sparse = classical_crossover_sparse(psi0, csr_matrix(H), dt, steps, n, chis)
        assert dense == sparse


class TestTrotterSteps:
    @pytest.mark.parametrize("builder_call", [
        lambda n, nn, nnn: build_frustrated_trotter_step(n, nn, nnn, h=1.5, dt=0.05, j2=0.5),
    ])
    def test_frustrated_step_is_unitary(self, builder_call):
        n = 4
        nn = [(0, 1), (1, 2), (2, 3)]
        nnn = [(0, 2), (1, 3)]
        step = builder_call(n, nn, nnn)
        u = Operator(step).data
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2**n), atol=1e-10)

    def test_kitaev_step_is_unitary(self):
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        step = build_kitaev_trotter_step(n, edges, mu=0.8, dt=0.05, delta=0.3)
        u = Operator(step).data
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2**n), atol=1e-10)

    def test_frustrated_order1_vs_order2_differ(self):
        n = 4
        nn = [(0, 1), (1, 2), (2, 3)]
        nnn = [(0, 2)]
        s1 = build_frustrated_trotter_step(n, nn, nnn, h=1.0, dt=0.2, j2=0.5, order=1)
        s2 = build_frustrated_trotter_step(n, nn, nnn, h=1.0, dt=0.2, j2=0.5, order=2)
        assert np.max(np.abs(Operator(s1).data - Operator(s2).data)) > 1e-6

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            build_kitaev_trotter_step(3, [(0, 1)], mu=0.5, dt=0.1, order=3)


class TestNoiseFloor:
    def test_floor_is_inverse_e(self):
        np.testing.assert_allclose(NOISE_FLOOR, np.exp(-1.0), atol=1e-15)


class TestAnalyticNoiseModel:
    """Analytic t_noise model F=exp(−N₂q·ε₂q) with the Nighthawk r2 median CZ error.
    Derivation + sources in internal/documentation/hardware/."""

    def test_default_error_is_median_cz(self):
        np.testing.assert_allclose(NIGHTHAWK_CZ_ERROR, 3e-3, atol=1e-12)

    def test_fidelity_decays_monotonically(self):
        f = [fidelity_from_2q(n) for n in (0, 50, 100, 300, 1000)]
        assert f[0] == 1.0
        assert all(f[i] > f[i + 1] for i in range(len(f) - 1))

    def test_fidelity_matches_closed_form(self):
        np.testing.assert_allclose(fidelity_from_2q(100, 2e-3), np.exp(-0.2), atol=1e-12)

    def test_gates_to_floor_is_inverse_error(self):
        # 1/e floor → N* = 1/ε. Median CZ (3e-3) → ~333.
        assert gates_to_floor(3e-3) == 333
        assert gates_to_floor(1e-3) == 1000

    def test_tnoise_curve_crosses_at_expected_step(self):
        # Cumulative 2q grows 200 per step; at ε=3e-3 the floor (1/e, N*=333) is
        # crossed when cumulative first exceeds 333 → step 2 (0,200,400,...).
        cumulative = [200 * k for k in range(6)]
        out = tnoise_from_2q_curve(cumulative, eps_2q=3e-3)
        assert out["t_noise_step"] == 2
        assert out["source"] == "analytic_exp_decay"
        assert out["eps_2q"] == 3e-3

    def test_tnoise_curve_no_crossing_when_cheap(self):
        # A cheap circuit (well under 333 2q) never crosses the floor.
        out = tnoise_from_2q_curve([0, 20, 40, 60], eps_2q=3e-3)
        assert out["t_noise_step"] is None

    def test_lower_eps_models_mitigation(self):
        # Lowering ε (noise suppression) pushes t_noise later / removes it.
        cumulative = [150 * k for k in range(6)]
        hi = tnoise_from_2q_curve(cumulative, eps_2q=3e-3)["t_noise_step"]
        lo = tnoise_from_2q_curve(cumulative, eps_2q=5e-4)["t_noise_step"]
        assert lo is None or (hi is not None and lo > hi)


class TestRobustJsonPersistence:
    """The dynamics runners persist via write_json_atomic: non-finite entries in
    observable curves (rate=−ln L/N can be +inf) must become JSON null, and the
    write must round-trip through a STRICT parser (allow_nan=False)."""

    def test_non_finite_in_curves_becomes_null(self, tmp_path):
        import json

        from qmbp_simulation.utils.helpers import write_json_atomic

        doc = {
            "schema": "dyn_test_v1",
            "quenches": [
                {"h2": 2.5, "rate": [0.0, float("inf"), 0.1, float("nan")],
                 "loschmidt": [1.0, 0.0, 0.5, 0.9]},
            ],
            "t_noise": float("inf"),
        }
        out = tmp_path / "dyn.json"
        write_json_atomic(out, doc)

        # Strict parse: json.load rejects bare NaN/Infinity tokens by default
        # only when parse_constant raises — assert there are no such tokens.
        text = out.read_text()
        assert "Infinity" not in text
        assert "NaN" not in text

        def _no_constants(_):
            raise AssertionError("non-finite token leaked into JSON")

        loaded = json.loads(text, parse_constant=_no_constants)
        q = loaded["quenches"][0]
        assert q["rate"][1] is None  # inf → null
        assert q["rate"][3] is None  # nan → null
        assert loaded["t_noise"] is None

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        from qmbp_simulation.utils.helpers import write_json_atomic

        out = tmp_path / "sub" / "dyn.json"
        write_json_atomic(out, {"ok": True, "arr": np.array([1.0, 2.0])})
        assert out.exists()
        # No leftover PID-tagged temp files in the directory.
        assert not list(out.parent.glob("*.tmp.*"))
