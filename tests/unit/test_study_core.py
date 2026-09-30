"""Unit tests for the shared frustrated-HVA study core.

Guards the ``src``-side building blocks the vl_vs_hva runners were consolidated
onto (``run_basin_count``, ``run_n18_fair_convergence``, ``run_ansatz_variants``).
Per the repo rule these import only from ``src`` (``qmbp_simulation``), never from
``scripts/``. Deterministic, small-N (N=6 frustrated square), no cloud.

The regression risks covered:
  1. ground_state — exact shapes/consistency of the (lat, qc, H, psi, e0, gap,
     n_nn, n_nnn) contract every runner unpacks.
  2. make_cost_fid — cost is ⟨H⟩ (≥ e0 by variational bound) and fid is a valid
     overlap in [0, 1]; the gradient matches finite differences.
  3. starvation_probe — total_nit is the two-segment sum, late_gain ≥ 0, and the
     final energy is no worse than the halfway energy.
  4. diagnose_starvation — the converged / starved / local_min classification.
  5. optimize_bestof — best-of is selected by ENERGY, warm_theta seeds restart 0,
     and a mismatched warm_theta is ignored.
  6. cx_and_params — non-zero-angle transpile yields a positive 2q count (no
     spurious zero from RZZ identity cancellation).
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.framework.study_core import (
    cx_and_params,
    diagnose_starvation,
    ground_state,
    make_cost_fid,
    optimize_bestof,
    starvation_probe,
)


@pytest.fixture(scope="module")
def gs_n6():
    """Ground state + circuit for N=6 frustrated square p=1 (exact, deterministic)."""
    return ground_state("square", 6, 0.5, 0.5, 1)


class TestGroundState:
    """The (lat, qc, H, psi, e0, gap, n_nn, n_nnn) contract every runner unpacks."""

    def test_shapes_and_consistency(self, gs_n6):
        lat, qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        assert qc.num_parameters == n_nn + n_nnn + 6  # p=1: [nn | nnn | x]
        assert psi.shape == (2**6,)
        assert np.iscomplexobj(psi)
        np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-8)
        assert gap > 0
        assert n_nn == len(lat.edges)

    def test_reproducible_to_solver_tolerance(self):
        # eigsh (ARPACK) is iterative — reproducible to solver tolerance, not bit-identical.
        e0_a = ground_state("square", 6, 0.5, 0.5, 1)[4]
        e0_b = ground_state("square", 6, 0.5, 0.5, 1)[4]
        np.testing.assert_allclose(e0_a, e0_b, atol=1e-9)

    def test_psi_is_ground_eigenvector(self, gs_n6):
        _, _, H, psi, e0, _, _, _ = gs_n6
        Hm = H.to_matrix(sparse=True)
        expectation = float(np.real(np.vdot(psi, Hm @ psi)))
        np.testing.assert_allclose(expectation, e0, atol=1e-6)


class TestCostFid:
    """cost = ⟨H⟩ (variational lower-bounded by e0); fid ∈ [0, 1]; grad ≈ FD."""

    def test_cost_respects_variational_bound(self, gs_n6):
        _, qc, H, psi, e0, _, _, _ = gs_n6
        cost, fid, _, _ = make_cost_fid(qc, H, psi)
        rng = np.random.default_rng(1)
        for _ in range(5):
            x = rng.uniform(-np.pi, np.pi, qc.num_parameters)
            assert cost(x) >= e0 - 1e-6
            assert 0.0 <= fid(x) <= 1.0 + 1e-9

    def test_gradient_matches_finite_difference(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, _, grad, _ = make_cost_fid(qc, H, psi)
        rng = np.random.default_rng(2)
        x = rng.uniform(-1.0, 1.0, qc.num_parameters)
        g = np.asarray(grad(x))
        eps = 1e-6
        for i in (0, len(x) // 2, len(x) - 1):
            dx = np.zeros_like(x)
            dx[i] = eps
            fd = (cost(x + dx) - cost(x - dx)) / (2 * eps)
            np.testing.assert_allclose(g[i], fd, atol=1e-4)


class TestStarvationProbe:
    """Two-segment probe: total_nit sums both segments, energy is monotone down."""

    def test_two_segment_consistency(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, _, grad, _ = make_cost_fid(qc, H, psi)
        # A non-trivial start actually exercises descent (θ=0 is near-stationary
        # for this ansatz and can report nit=0).
        rng = np.random.default_rng(7)
        x0 = rng.uniform(-0.5, 0.5, qc.num_parameters)
        x_fin, e_fin, total_nit, e_half, late_gain, converged = starvation_probe(cost, grad, x0, maxiter=200)
        assert total_nit >= 1
        assert e_fin <= e_half + 1e-9  # second segment never increases E
        np.testing.assert_allclose(late_gain, e_half - e_fin, atol=1e-12)
        assert isinstance(converged, bool)

    def test_maxiter_one_is_safe(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, _, grad, _ = make_cost_fid(qc, H, psi)
        # half = max(1//2, 1) = 1, rest = 0 — must not crash.
        out = starvation_probe(cost, grad, np.zeros(qc.num_parameters), maxiter=1)
        assert len(out) == 6


class TestDiagnoseStarvation:
    """converged / starved / local_min classification and custom labels."""

    def test_converged_wins_regardless_of_gain(self):
        assert diagnose_starvation(True, 5.0) == "converged"

    def test_starved_when_still_dropping(self):
        assert diagnose_starvation(False, 1e-2) == "starved"

    def test_local_min_when_flat(self):
        assert diagnose_starvation(False, 1e-9) == "local_min"

    def test_custom_labels(self):
        assert diagnose_starvation(False, 1e-2, starved_label="S") == "S"
        assert diagnose_starvation(False, 0.0, local_min_label="L") == "L"


class TestOptimizeBestof:
    """Best-of selection by energy; warm_theta seeds restart 0 when it fits."""

    def test_best_is_lowest_energy_and_valid_fidelity(self, gs_n6):
        _, qc, H, psi, e0, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        best_fid, best_e, runs = optimize_bestof(cost, fid, grad, qc.num_parameters, restarts=3, maxiter=200, seed0=123)
        assert len(runs) == 3
        assert best_e == min(r["energy"] for r in runs)
        assert best_e >= e0 - 1e-6
        assert 0.0 <= best_fid <= 1.0 + 1e-9

    def test_restart0_uses_warm_theta_when_length_matches(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        npar = qc.num_parameters
        warm = np.full(npar, 0.15)
        _, _, runs = optimize_bestof(cost, fid, grad, npar, restarts=1, maxiter=1, seed0=1, warm_theta=warm)
        np.testing.assert_allclose(runs[0]["theta_init"], warm, atol=1e-12)

    def test_mismatched_warm_theta_is_ignored(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        npar = qc.num_parameters
        warm = np.full(npar + 3, 0.15)  # wrong length
        _, _, runs = optimize_bestof(cost, fid, grad, npar, restarts=1, maxiter=1, seed0=1, warm_theta=warm)
        np.testing.assert_allclose(runs[0]["theta_init"], np.zeros(npar), atol=1e-12)

    def test_random_restarts_are_seeded(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        npar = qc.num_parameters
        _, e_a, _ = optimize_bestof(cost, fid, grad, npar, restarts=2, maxiter=50, seed0=999)
        _, e_b, _ = optimize_bestof(cost, fid, grad, npar, restarts=2, maxiter=50, seed0=999)
        assert e_a == e_b

    def test_on_restart_fires_per_restart_with_growing_runs(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        seen = []
        optimize_bestof(
            cost,
            fid,
            grad,
            qc.num_parameters,
            restarts=3,
            maxiter=30,
            seed0=7,
            on_restart=lambda runs, best: seen.append((len(runs), best)),
        )
        # Fired once per restart, runs list grows 1,2,3, best always present.
        assert [n for n, _ in seen] == [1, 2, 3]
        assert all(b is not None and "fidelity" in b for _, b in seen)

    def test_on_restart_exception_does_not_break_optimization(self, gs_n6):
        _, qc, H, psi, _, _, _, _ = gs_n6
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)

        def boom(runs, best):
            raise RuntimeError("persistence failed")

        # A failing persister must not crash the optimization.
        best_fid, _, runs = optimize_bestof(
            cost, fid, grad, qc.num_parameters, restarts=2, maxiter=30, seed0=1, on_restart=boom
        )
        assert len(runs) == 2
        assert 0.0 <= best_fid <= 1.0 + 1e-9


class TestCxAndParams:
    """Transpiled 2q count is positive (non-zero angles avoid RZZ cancellation)."""

    def test_positive_2q_and_param_count(self, gs_n6):
        _, qc, _, _, _, _, _, _ = gs_n6
        n_2q, npar = cx_and_params(qc)
        assert n_2q > 0
        assert npar == qc.num_parameters
