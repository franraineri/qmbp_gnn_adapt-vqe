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


class TestOptimizeXSpaceBestof:
    """optimize_xspace_bestof spends exploration only on the θ_x subspace."""

    @staticmethod
    def _quad(target):
        def cost(t):
            import numpy as np
            return float(np.sum((t - target) ** 2))

        def fid(t):
            return 1.0 / (1.0 + cost(t))

        def grad(t):
            return 2.0 * (t - target)

        return cost, fid, grad

    def test_restart0_starts_at_seed(self):
        import numpy as np

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.array([0.5, 0.5, 0.1, 0.2])
        cost, fid, grad = self._quad(0.3)
        _bf, _be, runs = optimize_xspace_bestof(
            cost, fid, grad, seed, [2, 3], restarts=2, maxiter=50, seed0=1)
        assert runs[0]["theta_init"] == seed.tolist()

    def test_zz_anchored_x_perturbed_on_restarts(self):
        import numpy as np

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.array([0.5, 0.5, 0.1, 0.2])
        cost, fid, grad = self._quad(0.3)
        _bf, _be, runs = optimize_xspace_bestof(
            cost, fid, grad, seed, [2, 3], restarts=3, maxiter=50, seed0=7, sigma_x=0.3)
        r1 = np.array(runs[1]["theta_init"])
        assert r1[0] == 0.5 and r1[1] == 0.5  # ZZ untouched
        assert not (r1[2] == 0.1 and r1[3] == 0.2)  # X perturbed

    def test_converges_to_target(self):
        import numpy as np

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.zeros(4)
        cost, fid, grad = self._quad(0.3)
        bf, be, _runs = optimize_xspace_bestof(
            cost, fid, grad, seed, [2, 3], restarts=2, maxiter=200, seed0=1)
        assert be < 1e-6  # full L-BFGS still relaxes all params
        assert bf > 0.99

    def test_out_of_range_indices_raise(self):
        import numpy as np
        import pytest

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.zeros(4)
        cost, fid, grad = self._quad(0.3)
        with pytest.raises(ValueError):
            optimize_xspace_bestof(cost, fid, grad, seed, [9], restarts=1,
                                   maxiter=10, seed0=1)


class TestBuildVariantRow:
    """Canonical per-variant result row (single source of the dict shape)."""

    def _runs(self):
        return [
            {"restart": 0, "energy": -1.0, "fidelity": 0.90, "theta_final": [0.1, 0.2]},
            {"restart": 1, "energy": -1.5, "fidelity": 0.95, "theta_final": [0.3, 0.4]},
        ]

    def test_core_metrics_and_derived_fields(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row("v1", 0.95, -1.5, self._runs(),
                                n_2q=100, n_params=20, e0=-2.0, gap=0.5)
        assert row["variant"] == "v1"
        assert row["best_fidelity"] == 0.95
        assert row["abs_error"] == 0.5          # |-1.5 - (-2.0)|
        assert row["de_gap"] == 1.0             # 0.5 / 0.5
        assert row["fidelity_per_cx"] == 0.95 / 100
        # lowest-energy restart supplies theta
        assert row["best_theta_final"] == [0.3, 0.4]

    def test_gap_zero_yields_none_de_gap(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row("v", 0.5, -1.0, self._runs(),
                                n_2q=10, n_params=5, e0=-1.0, gap=0.0)
        assert row["de_gap"] is None

    def test_bond_selection_bookkeeping(self):
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import build_variant_row

        sel = BondSelection(nn_edges=[(0, 1), (1, 2)], nnn_edges=[(0, 2)],
                            provenance="top_k")
        row = build_variant_row("masked", 0.9, -1.0, self._runs(),
                                n_2q=50, n_params=10, e0=-1.1, gap=0.3,
                                bond_selection=sel)
        assert row["n_nn_bonds"] == 2
        assert row["n_nnn_bonds"] == 1
        assert row["selection_provenance"] == "top_k"

    def test_full_selection_defaults_and_extra(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row("v", 0.9, -1.0, self._runs(), n_2q=50,
                                n_params=10, e0=-1.1, gap=0.3,
                                blocks=["nn", "x"], rx_final=True,
                                seed_kind="analytic", seconds=12.34,
                                custom_field="xyz")
        assert row["selection_provenance"] == "full"  # no bond_selection
        assert row["blocks"] == ["nn", "x"]
        assert row["rx_final"] is True
        assert row["seed_kind"] == "analytic"
        assert row["seconds"] == 12.3
        assert row["custom_field"] == "xyz"  # **extra merged


class TestConvergeCircuit:
    """converge_circuit = make_cost_fid + optimize_bestof in one call."""

    def test_matches_manual_pipeline(self, gs_n6):
        from qmbp_simulation.framework.study_core import (
            converge_circuit,
            make_cost_fid,
            optimize_bestof,
        )

        _lat, qc, H, psi, _e0, _gap, _n_nn, _n_nnn = gs_n6
        # Manual pipeline
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        bf_m, be_m, runs_m = optimize_bestof(
            cost, fid, grad, qc.num_parameters, restarts=1, maxiter=200, seed0=7)
        # Wrapper (same seed/budget → identical numerics)
        bf_w, be_w, runs_w = converge_circuit(
            qc, H, psi, restarts=1, maxiter=200, seed0=7)
        assert bf_w == pytest.approx(bf_m)
        assert be_w == pytest.approx(be_m)
        assert len(runs_w) == len(runs_m)

    def test_shared_backend_is_used(self, gs_n6):
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.framework.study_core import converge_circuit

        _lat, qc, H, psi, _e0, _gap, _n_nn, _n_nnn = gs_n6
        backend = NoiselessBackend()
        bf, be, runs = converge_circuit(
            qc, H, psi, restarts=1, maxiter=100, seed0=1, backend=backend)
        assert 0.0 <= bf <= 1.0
        assert len(runs) == 1


class TestPrepareWarmstart:
    """prepare_warmstart = the one-call runner integration of the combined cascade."""

    def test_returns_seed_and_report(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, micro_descent=6)
        assert r["seed"].size == qc.num_parameters
        assert r["provenance"]  # non-empty
        assert isinstance(r["report"], list) and len(r["report"]) >= 1
        assert "_fid" in r and "_grad" in r  # evaluators returned for reuse

    def test_combined_includes_calibrated_candidate(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, strategy="combined",
                              micro_descent=6)
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" in labels

    def test_regime_strategy_excludes_calibrated(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, strategy="regime",
                              micro_descent=6)
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" not in labels

    def test_init_fidelity_is_valid_overlap(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, micro_descent=6)
        assert 0.0 <= r["init_fidelity"] <= 1.0 + 1e-9

    def test_negative_methods_never_appear(self, gs_n6):
        # ensemble + block_mix are validated-negative → never in the report,
        # and prepare_warmstart exposes no way to turn them on.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, micro_descent=6)
        labels = [x["label"] for x in r["report"]]
        assert not any(l.startswith("ensemble<") for l in labels)
        assert not any(l.startswith("block_mix<") for l in labels)

    def test_profile_recorded(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, model="tfim_frustrated",
                              topology="square", micro_descent=6)
        assert r["_profile"]["calibrated_family"] is True
        assert r["_profile"]["include_ensemble"] is False

    def test_non_ising_model_disables_analytic_seeds(self, gs_n6):
        # Heisenberg/XY/Kitaev: Ising-calibrated seeds OFF, regime still present.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6,
                              p_layers=1, h=0.5, J2=0.5, model="heisenberg",
                              topology="square", micro_descent=6)
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" not in labels
        assert "structural" not in labels
        assert r["_profile"]["calibrated_family"] is False


class TestWarmstartProfile:
    """warmstart_profile — structure-aware technique selection."""

    def test_ising_family_full_stack(self):
        from qmbp_simulation.analysis.warmstart import warmstart_profile

        for m in ("tfim", "tfim_frustrated", "tfim_bond_resolved", None):
            p = warmstart_profile(model=m, topology="square", J2=0.5)
            assert p["include_calibrated"] and p["include_structural"]
            assert p["include_regime"]
            assert not p["include_ensemble"] and not p["include_block_mix"]

    def test_non_ising_disables_calibrated(self):
        from qmbp_simulation.analysis.warmstart import warmstart_profile

        for m in ("heisenberg", "xy", "kitaev", "heisenberg_transverse"):
            p = warmstart_profile(model=m)
            assert not p["include_calibrated"]
            assert not p["include_structural"]
            assert p["include_regime"]            # general seed stays on
            assert p["calibrated_family"] is False

    def test_negatives_always_off(self):
        from qmbp_simulation.analysis.warmstart import warmstart_profile

        for m in ("tfim", "heisenberg", None):
            p = warmstart_profile(model=m)
            assert p["include_ensemble"] is False
            assert p["include_block_mix"] is False
