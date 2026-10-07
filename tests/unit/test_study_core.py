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
        _bf, _be, runs = optimize_xspace_bestof(cost, fid, grad, seed, [2, 3], restarts=2, maxiter=50, seed0=1)
        assert runs[0]["theta_init"] == seed.tolist()

    def test_zz_anchored_x_perturbed_on_restarts(self):
        import numpy as np

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.array([0.5, 0.5, 0.1, 0.2])
        cost, fid, grad = self._quad(0.3)
        _bf, _be, runs = optimize_xspace_bestof(
            cost, fid, grad, seed, [2, 3], restarts=3, maxiter=50, seed0=7, sigma_x=0.3
        )
        r1 = np.array(runs[1]["theta_init"])
        assert r1[0] == 0.5 and r1[1] == 0.5  # ZZ untouched
        assert not (r1[2] == 0.1 and r1[3] == 0.2)  # X perturbed

    def test_converges_to_target(self):
        import numpy as np

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.zeros(4)
        cost, fid, grad = self._quad(0.3)
        bf, be, _runs = optimize_xspace_bestof(cost, fid, grad, seed, [2, 3], restarts=2, maxiter=200, seed0=1)
        assert be < 1e-6  # full L-BFGS still relaxes all params
        assert bf > 0.99

    def test_out_of_range_indices_raise(self):
        import numpy as np
        import pytest

        from qmbp_simulation.framework.study_core import optimize_xspace_bestof

        seed = np.zeros(4)
        cost, fid, grad = self._quad(0.3)
        with pytest.raises(ValueError):
            optimize_xspace_bestof(cost, fid, grad, seed, [9], restarts=1, maxiter=10, seed0=1)


class TestBuildVariantRow:
    """Canonical per-variant result row (single source of the dict shape)."""

    def _runs(self):
        return [
            {"restart": 0, "energy": -1.0, "fidelity": 0.90, "theta_final": [0.1, 0.2]},
            {"restart": 1, "energy": -1.5, "fidelity": 0.95, "theta_final": [0.3, 0.4]},
        ]

    def test_core_metrics_and_derived_fields(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row("v1", 0.95, -1.5, self._runs(), n_2q=100, n_params=20, e0=-2.0, gap=0.5)
        assert row["variant"] == "v1"
        assert row["best_fidelity"] == 0.95
        assert row["abs_error"] == 0.5  # |-1.5 - (-2.0)|
        assert row["de_gap"] == 1.0  # 0.5 / 0.5
        assert row["fidelity_per_cx"] == 0.95 / 100
        # lowest-energy restart supplies theta
        assert row["best_theta_final"] == [0.3, 0.4]

    def test_gap_zero_yields_none_de_gap(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row("v", 0.5, -1.0, self._runs(), n_2q=10, n_params=5, e0=-1.0, gap=0.0)
        assert row["de_gap"] is None

    def test_bond_selection_bookkeeping(self):
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import build_variant_row

        sel = BondSelection(nn_edges=[(0, 1), (1, 2)], nnn_edges=[(0, 2)], provenance="top_k")
        row = build_variant_row(
            "masked", 0.9, -1.0, self._runs(), n_2q=50, n_params=10, e0=-1.1, gap=0.3, bond_selection=sel
        )
        assert row["n_nn_bonds"] == 2
        assert row["n_nnn_bonds"] == 1
        assert row["selection_provenance"] == "top_k"

    def test_full_selection_defaults_and_extra(self):
        from qmbp_simulation.framework.study_core import build_variant_row

        row = build_variant_row(
            "v",
            0.9,
            -1.0,
            self._runs(),
            n_2q=50,
            n_params=10,
            e0=-1.1,
            gap=0.3,
            blocks=["nn", "x"],
            rx_final=True,
            seed_kind="analytic",
            seconds=12.34,
            custom_field="xyz",
        )
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
        bf_m, be_m, runs_m = optimize_bestof(cost, fid, grad, qc.num_parameters, restarts=1, maxiter=200, seed0=7)
        # Wrapper (same seed/budget → identical numerics)
        bf_w, be_w, runs_w = converge_circuit(qc, H, psi, restarts=1, maxiter=200, seed0=7)
        assert bf_w == pytest.approx(bf_m)
        assert be_w == pytest.approx(be_m)
        assert len(runs_w) == len(runs_m)

    def test_shared_backend_is_used(self, gs_n6):
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.framework.study_core import converge_circuit

        _lat, qc, H, psi, _e0, _gap, _n_nn, _n_nnn = gs_n6
        backend = NoiselessBackend()
        bf, be, runs = converge_circuit(qc, H, psi, restarts=1, maxiter=100, seed0=1, backend=backend)
        assert 0.0 <= bf <= 1.0
        assert len(runs) == 1


class TestPrepareWarmstart:
    """prepare_warmstart = the one-call runner integration of the combined cascade."""

    def test_returns_seed_and_report(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6, p_layers=1, h=0.5, J2=0.5, micro_descent=6
        )
        assert r["seed"].size == qc.num_parameters
        assert r["provenance"]  # non-empty
        assert isinstance(r["report"], list) and len(r["report"]) >= 1
        assert "_fid" in r and "_grad" in r  # evaluators returned for reuse

    def test_combined_includes_calibrated_candidate(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            strategy="combined",
            micro_descent=6,
        )
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" in labels

    def test_regime_strategy_excludes_calibrated(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            strategy="regime",
            micro_descent=6,
        )
        labels = [x["label"] for x in r["report"]]
        assert "calibrated" not in labels

    def test_init_fidelity_is_valid_overlap(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6, p_layers=1, h=0.5, J2=0.5, micro_descent=6
        )
        assert 0.0 <= r["init_fidelity"] <= 1.0 + 1e-9

    def test_negative_methods_never_appear(self, gs_n6):
        # ensemble + block_mix are validated-negative → never in the report,
        # and prepare_warmstart exposes no way to turn them on.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6, p_layers=1, h=0.5, J2=0.5, micro_descent=6
        )
        labels = [x["label"] for x in r["report"]]
        assert not any(l.startswith("ensemble<") for l in labels)
        assert not any(l.startswith("block_mix<") for l in labels)

    def test_profile_recorded(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            model="tfim_frustrated",
            topology="square",
            micro_descent=6,
        )
        assert r["_profile"]["calibrated_family"] is True
        assert r["_profile"]["include_ensemble"] is False

    def test_non_ising_model_disables_analytic_seeds(self, gs_n6):
        # Heisenberg/XY/Kitaev: Ising-calibrated seeds OFF, regime still present.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            model="heisenberg",
            topology="square",
            micro_descent=6,
        )
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
            assert p["include_regime"]  # general seed stays on
            assert p["calibrated_family"] is False

    def test_negatives_always_off(self):
        from qmbp_simulation.analysis.warmstart import warmstart_profile

        for m in ("tfim", "heisenberg", None):
            p = warmstart_profile(model=m)
            assert p["include_ensemble"] is False
            assert p["include_block_mix"] is False


class TestPrepareWarmstartTwoPass:
    """prepare_warmstart two-pass selector wiring (M1+M2 activation)."""

    def test_off_by_default(self, gs_n6):
        # Default (two_pass=False): no short-descent, exhaustive behaviour.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6, p_layers=1, h=0.5, J2=0.5, micro_descent=6
        )
        assert r["_two_pass"] is False
        assert r["_short_descent"] is None

    def test_two_pass_sets_short_descent(self, gs_n6):
        # two_pass=True derives a short budget from the full micro_descent.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=6, p_layers=1, h=0.5, J2=0.5, micro_descent=40, two_pass=True
        )
        assert r["_two_pass"] is True
        assert isinstance(r["_short_descent"], int)
        assert 0 < r["_short_descent"] <= 40

    def test_explicit_short_descent_honored(self, gs_n6):
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            micro_descent=40,
            two_pass=True,
            short_descent=12,
        )
        assert r["_short_descent"] == 12

    def test_two_pass_returns_valid_seed(self, gs_n6):
        # The two-pass path still yields a valid, full-length warm-start seed.
        from qmbp_simulation.framework.study_core import prepare_warmstart

        lat, qc, H, psi, _e0, _gap, n_nn, n_nnn = gs_n6
        r = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=6,
            p_layers=1,
            h=0.5,
            J2=0.5,
            micro_descent=12,
            two_pass=True,
            two_pass_top_k=2,
        )
        assert r["seed"].size == qc.num_parameters
        assert r["provenance"].endswith("+descent")
        assert 0.0 <= r["init_fidelity"] <= 1.0 + 1e-9


class TestRemapThetaGrowNnn:
    """_remap_theta_grow_nnn — warm-start carry-over when the nnn block grows."""

    def test_inserts_zeros_for_new_nnn_and_preserves_blocks(self):
        from qmbp_simulation.framework.study_core import _remap_theta_grow_nnn

        n_nn, n_nnn_prev, n_added, n_q = 3, 2, 2, 4
        # layout [nn(3) | nnn(2) | x(4)] = 9 params
        prev = np.arange(1, 10, dtype=float)  # 1..9
        out = _remap_theta_grow_nnn(prev, n_nn, n_nnn_prev, n_added, n_q)
        assert out.shape[0] == n_nn + n_nnn_prev + n_added + n_q  # 11
        np.testing.assert_allclose(out[:3], prev[:3])  # nn kept
        np.testing.assert_allclose(out[3:5], prev[3:5])  # old nnn kept
        np.testing.assert_allclose(out[5:7], np.zeros(2))  # new nnn = 0
        np.testing.assert_allclose(out[-4:], prev[-4:])  # x kept

    def test_zero_added_is_identity(self):
        from qmbp_simulation.framework.study_core import _remap_theta_grow_nnn

        prev = np.arange(1, 10, dtype=float)
        out = _remap_theta_grow_nnn(prev, 3, 2, 0, 4)
        np.testing.assert_allclose(out, prev)


class TestBondEnergyGradients:
    """bond_energy_gradients — ADAPT growth signal at the optimized state."""

    def test_shapes_align_with_candidates(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import bond_energy_gradients
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        # nn-only backbone θ layout: [nn | (0 nnn) | x] = n_nn + 6
        theta = np.zeros(n_nn + 6)
        g_nn, g_nnn = bond_energy_gradients(HVACircuitBuilder(), 6, lat, sel, [], nnn, H, psi, theta)
        assert g_nn.size == 0
        assert g_nnn.shape[0] == len(nnn)
        assert np.all(g_nnn >= 0.0)  # magnitudes

    def test_nonzero_at_optimized_state(self, gs_n6):
        # At a NON-symmetric θ the nnn gradient is generically nonzero (the whole
        # point of evaluating at the optimized state, not θ=0).
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import bond_energy_gradients
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        rng = np.random.default_rng(0)
        theta = rng.uniform(-0.5, 0.5, n_nn + 6)
        _g_nn, g_nnn = bond_energy_gradients(HVACircuitBuilder(), 6, lat, sel, [], nnn, H, psi, theta)
        assert float(np.max(g_nnn)) > 0.0


class TestGrowBondsAdapt:
    """grow_bonds_adapt — the shared ADAPT bond-growth loop."""

    def _args(self, gs_n6):
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        nn_all = list(lat.edges)
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        return lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all

    def test_grows_and_records_steps(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_bonds_adapt

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, reached = grow_bonds_adapt(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=100,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            warm_per_step=True,
        )
        assert len(steps) >= 1
        s0 = steps[0]
        for key in (
            "step",
            "n_bonds",
            "best_fidelity",
            "n_2q_transpiled",
            "fidelity_per_cx",
            "de_gap",
            "best_theta_final",
        ):
            assert key in s0
        assert 0.0 <= s0["best_fidelity"] <= 1.0 + 1e-9
        assert steps[-1].get("stop_reason") is not None
        # nnn count is non-decreasing across steps (growth only adds)
        nnn_counts = [s["n_nnn_bonds"] for s in steps]
        assert nnn_counts == sorted(nnn_counts)

    def test_efficiency_stop_sets_reason(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_bonds_adapt

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, _reached = grow_bonds_adapt(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=100,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            warm_per_step=True,
            efficiency_stop=True,
            efficiency_patience=1,
        )
        assert steps[-1]["stop_reason"] in (
            "efficiency_plateau",
            "full_layer",
            "grad_below_tol",
            "target_fid",
            "max_steps",
        )

    def test_callable_grow_step_adaptive(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_bonds_adapt

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)

        # coarse 3 early, fine 1 later
        def gstep(idx, n_sel, n_tot):
            return 3 if n_sel < 0.5 * n_tot else 1

        steps, _reached = grow_bonds_adapt(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=80,
            seed0=1,
            target_fid=1.1,
            grow_step=gstep,
            warm_per_step=True,
        )
        # first growth jump adds up to 3 nnn (coarse)
        if len(steps) >= 2:
            assert steps[1]["n_nnn_bonds"] - steps[0]["n_nnn_bonds"] <= 3

    def test_on_step_callback_fires(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_bonds_adapt

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        seen = []
        grow_bonds_adapt(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=80,
            seed0=1,
            target_fid=1.1,
            grow_step=2,
            on_step=lambda s: seen.append(s["step"]),
        )
        assert seen == list(range(len(seen)))  # one call per step, in order

    def test_seed_selection_starts_from_given_bonds(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import grow_bonds_adapt

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        seed_sel = BondSelection(nn_edges=list(nn_all), nnn_edges=list(nnn_all[:2]))
        steps, _reached = grow_bonds_adapt(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=80,
            seed0=1,
            target_fid=1.1,
            grow_step=2,
            seed_selection=seed_sel,
        )
        assert steps[0]["n_nnn_bonds"] == 2  # started from the seeded 2 nnn


class TestPoolWarmstartHelpers:
    """Pure θ-remap helpers for the unrestricted-p pool growth."""

    def test_block_param_size(self):
        from qmbp_simulation.framework.study_core import _block_param_size

        assert _block_param_size("nn", 7, 6, 6) == 7
        assert _block_param_size("nnn", 7, 6, 6) == 6
        assert _block_param_size("x", 7, 6, 6) == 6
        assert _block_param_size("z", 7, 6, 6) == 6

    def test_append_blocks_warm_zero_fills_tail(self):
        from qmbp_simulation.framework.study_core import _append_blocks_warm

        prev = np.arange(1, 14, dtype=float)  # 13 = [nn7 | nnn? ...] arbitrary
        # append ["nn","nnn","x"] with n_nn=3 n_nnn=2 n_q=4 → +9 zeros
        out = _append_blocks_warm(prev, ["nn", "nnn", "x"], 3, 2, 4)
        assert out.shape[0] == prev.shape[0] + 9
        np.testing.assert_allclose(out[:13], prev)
        np.testing.assert_allclose(out[13:], np.zeros(9))

    def test_warm_for_added_bonds_inserts_zeros_in_each_nnn_block(self):
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import _warm_for_added_bonds

        # blocks [nn, nnn, x] with sel n_nn=2 n_nnn=1 n_q=3 → θ layout [2 | 1 | 3] = 6
        sel = BondSelection(nn_edges=[(0, 1), (1, 2)], nnn_edges=[(0, 2)])
        theta = np.array([10, 11, 20, 30, 31, 32], float)  # nn=10,11 nnn=20 x=30,31,32
        # add 2 nnn bonds → each nnn block widens by 2 (zeros) → new len 2+ (1+2) +3 = 8
        out = _warm_for_added_bonds(theta, ["nn", "nnn", "x"], sel, 2, 2, 1, 3)
        assert out.shape[0] == 8
        np.testing.assert_allclose(out[:2], [10, 11])  # nn kept
        np.testing.assert_allclose(out[2:3], [20])  # old nnn kept
        np.testing.assert_allclose(out[3:5], [0, 0])  # 2 new nnn = 0
        np.testing.assert_allclose(out[5:], [30, 31, 32])  # x kept

    def test_warm_for_added_bonds_multilayer(self):
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import _warm_for_added_bonds

        # two layers [nn,nnn,x, nn,nnn,x]; each nnn block must widen independently
        sel = BondSelection(nn_edges=[(0, 1)], nnn_edges=[(0, 2)])
        # layout per layer: nn1 nnn1 x3 → 5; two layers → 10
        theta = np.arange(10, dtype=float)
        out = _warm_for_added_bonds(theta, ["nn", "nnn", "x", "nn", "nnn", "x"], sel, 1, 1, 1, 3)
        # each of the 2 nnn blocks gains 1 zero → +2 total
        assert out.shape[0] == 12


class TestGrowAdaptPool:
    """grow_adapt_pool — unrestricted-p growth by Δfid/Δ2q."""

    def _args(self, gs_n6):
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        nn_all = list(lat.edges)
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        return lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all

    def test_grows_both_action_kinds_and_breaks_p1(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=250,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=3,
            efficiency_patience=3,
        )
        kinds = {s["action_kind"] for s in steps}
        assert "seed" in kinds
        assert "repeat_layer" in kinds  # the new unrestricted-p action fired
        # every step's θ length matches its reported n_params
        for s in steps:
            assert len(s["best_theta_final"]) == s["n_params"]
        # n_layers grows beyond 1 (p was raised locally)
        assert max(s["n_layers"] for s in steps) >= 2
        assert steps[-1].get("stop_reason") is not None

    def test_no_repeat_layer_falls_back_to_bonds_only(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, _reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=200,
            seed0=1,
            target_fid=1.1,
            grow_step=2,
            allow_repeat_layer=False,
            max_layers=3,
            efficiency_patience=5,
        )
        # with repeat disabled, only seed + add_bonds actions occur, 1 layer always
        assert all(s["action_kind"] in ("seed", "add_bonds") for s in steps)
        assert all(s["n_layers"] == 1 for s in steps)

    def test_pool_actions_seam_returns_both(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient
        from qmbp_simulation.framework.study_core import _adapt_pool_actions
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        rng = np.random.default_rng(0)
        theta = rng.uniform(-0.3, 0.3, n_nn + 6)  # [nn | 0 nnn | x]
        acts = _adapt_pool_actions(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            ["nn", "nnn", "x"],
            sel,
            theta,
            nnn_all,
            n_nn,
            6,
            grow_step=2,
            grad_tol=1e-6,
            repeat_block=["nn", "nnn", "x"],
            allow_repeat_layer=True,
            max_layers=3,
            rank_by_gradient=rank_by_gradient,
            _BondSelection=BondSelection,
        )
        kinds = {a["kind"] for a in acts}
        assert kinds == {"add_bonds", "repeat_layer"}
        for a in acts:
            assert set(a.keys()) >= {"kind", "blocks", "selection", "warm", "delta_2q_hint"}

    def test_pool_actions_symmetry_fallback_keeps_add_bonds(self, gs_n6):
        # At θ=0 (symmetric point) the nnn gradient vanishes; the fallback must
        # still offer add_bonds instead of starving the loop.
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient
        from qmbp_simulation.framework.study_core import _adapt_pool_actions
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        theta = np.zeros(n_nn + 6)  # exact symmetric point
        acts = _adapt_pool_actions(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            ["nn", "nnn", "x"],
            sel,
            theta,
            nnn_all,
            n_nn,
            6,
            grow_step=2,
            grad_tol=1e-6,
            repeat_block=["nn", "nnn", "x"],
            allow_repeat_layer=False,
            max_layers=3,
            rank_by_gradient=rank_by_gradient,
            _BondSelection=BondSelection,
        )
        assert any(a["kind"] == "add_bonds" for a in acts)


class TestGrowAdaptPoolStopRule:
    """The Δfid-aware stop rule and the repeat-layer symmetry-breaking warm."""

    def _args(self, gs_n6):
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        return (lat, H, psi, e0, gap, n_nn, n_nnn, list(lat.edges), HamiltonianBuilder._generate_nnn_edges(lat))

    def test_strong_dfid_keeps_growing_past_efficiency_plateau(self):
        """A step with large Δfid but falling fid/CX must NOT trigger the plateau
        stop when min_dfid_keep_growing is set — grow in the hard phase."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import ground_state, grow_adapt_pool
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        # h=0.9 is the phase where the old rule stopped at 1 layer.
        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state("square", 6, 0.9, 0.5, 1)
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        steps, _reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=list(lat.edges),
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=250,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=4,
            efficiency_patience=2,
            min_dfid_keep_growing=0.01,
        )
        # With the Δfid-aware rule the loop grows past 1 layer (the old failure).
        assert max(s["n_layers"] for s in steps) >= 2
        # And fidelity ends well above the single-layer plateau.
        assert max(s["best_fidelity"] for s in steps) > 0.90

    def test_min_dfid_zero_restores_efficiency_only_rule(self, gs_n6):
        """min_dfid_keep_growing=0 → only fid/CX governs (pre-fix behavior)."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, _reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=200,
            seed0=1,
            target_fid=1.1,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=4,
            efficiency_patience=1,
            min_dfid_keep_growing=0.0,
        )
        # The run terminates with a recognized reason (not an exception / no hang).
        assert steps[-1].get("stop_reason") in (
            "efficiency_plateau",
            "no_action",
            "diverged",
            "max_steps",
            "target_fid",
        )

    def test_repeat_layer_activates_with_single_restart(self):
        """The perturbed repeat-layer warm activates the new layer even with
        restarts=1 (zero-fill alone left it at the symmetric θ=0 point)."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import ground_state, grow_adapt_pool
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state("square", 6, 0.9, 0.5, 1)
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        steps, _reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=list(lat.edges),
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=250,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=4,
            efficiency_patience=3,
            min_dfid_keep_growing=0.01,
        )
        rep = [s for s in steps if s["action_kind"] == "repeat_layer"]
        assert rep, "repeat_layer never chosen"
        # A repeat_layer step genuinely raised fidelity (layer not left inert).
        assert max(s["best_fidelity"] for s in steps) > 0.90


class TestGrowAdaptPoolAccelerators:
    """Opt-in speed/quality levers: fast_rank, layer_growth_penalty, warm_theta0."""

    def _args(self, gs_n6):
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        return (lat, H, psi, e0, gap, n_nn, n_nnn, list(lat.edges), HamiltonianBuilder._generate_nnn_edges(lat))

    def test_fast_rank_runs_and_converges_single_candidate(self, gs_n6):
        """fast_rank runs end-to-end and converges only the gradient-best action.

        NOTE: fast_rank is NOT guaranteed to match the converge-all fidelity for
        this heterogeneous pool (add_bonds vs repeat_layer) — measured to mis-rank
        at N10 h=0.7. This test only pins that the mode executes and returns a
        valid growth curve; quality parity is asserted only for the operator-list
        loop where the pool is homogeneous.
        """
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        steps, _r = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=200,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=4,
            efficiency_patience=3,
            fast_rank=True,
        )
        assert len(steps) >= 1
        for s in steps:
            assert 0.0 <= s["best_fidelity"] <= 1.0 + 1e-9
            assert len(s["best_theta_final"]) == s["n_params"]
        assert steps[-1].get("stop_reason") is not None

    def test_layer_penalty_does_not_increase_layers(self, gs_n6):
        """A positive layer_growth_penalty never grows MORE layers than penalty=0
        (it only ever tightens the keep-growing bar)."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        common = dict(
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=200,
            seed0=1,
            target_fid=1.1,  # never hit
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=6,
            efficiency_patience=2,
            min_dfid_keep_growing=0.02,
        )
        b = HVACircuitBuilder()
        flat, _ = grow_adapt_pool(b, 6, lat, H, psi, e0, gap, layer_growth_penalty=0.0, **common)
        pen, _ = grow_adapt_pool(b, 6, lat, H, psi, e0, gap, layer_growth_penalty=2.0, **common)
        layers_flat = max(s["n_layers"] for s in flat)
        layers_pen = max(s["n_layers"] for s in pen)
        assert layers_pen <= layers_flat

    def test_warm_theta0_is_used_at_seed(self, gs_n6):
        """A provided warm_theta0 of the right length seeds step 0 (no error, and
        the seed step records a valid fidelity)."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool

        lat, H, psi, e0, gap, n_nn, n_nnn, nn_all, nnn_all = self._args(gs_n6)
        # seed circuit [nn,nnn,x] with nn-only backbone → npar = n_nn + 0 + 6
        warm0 = np.zeros(n_nn + 6)
        steps, _r = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=1,
            maxiter=150,
            seed0=1,
            target_fid=1.1,
            grow_step=2,
            warm_theta0=warm0,
            efficiency_patience=2,
        )
        assert steps[0]["action_kind"] == "seed"
        assert 0.0 <= steps[0]["best_fidelity"] <= 1.0 + 1e-9

    def test_candidate_grad_score_is_finite_for_valid_action(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection, rank_by_gradient
        from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant
        from qmbp_simulation.framework.study_core import (
            _adapt_pool_actions,
            _candidate_grad_score,
            make_cost_fid,
        )
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        rng = np.random.default_rng(0)
        theta = rng.uniform(-0.3, 0.3, n_nn + 6)
        acts = _adapt_pool_actions(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            ["nn", "nnn", "x"],
            sel,
            theta,
            nnn_all,
            n_nn,
            6,
            grow_step=2,
            grad_tol=1e-6,
            repeat_block=["nn", "nnn", "x"],
            allow_repeat_layer=True,
            max_layers=3,
            rank_by_gradient=rank_by_gradient,
            _BondSelection=BondSelection,
        )
        for act in acts:
            score = _candidate_grad_score(
                HVACircuitBuilder(),
                6,
                lat,
                H,
                psi,
                act,
                theta,
                2 * n_nn,
                make_cost_fid,
                build_variant,
                make_masked_variant,
            )
            assert np.isfinite(score)


class TestOptimizerGuidanceHelpers:
    """2c analytic seed, 1a fidelity-gradient ranking, 2b block preconditioner."""

    def test_analytic_seed_moves_new_param_toward_minimum(self):
        """A 1D convex cost in the new param → analytic seed lands near its min."""
        from qmbp_simulation.framework.study_core import analytic_seed_new_params

        # cost = (θ[1] - 0.7)^2 + const; new param is index 1, starts at 0.
        def cost(x):
            return (x[1] - 0.7) ** 2 + 0.1 * x[0] ** 2

        warm = np.array([0.3, 0.0])
        out = analytic_seed_new_params(cost, warm, [1])
        assert out[0] == 0.3  # untouched param preserved
        assert abs(out[1] - 0.7) < 0.05  # new param seeded near the 1D minimum

    def test_analytic_seed_handles_flat_direction(self):
        """A flat/non-convex direction keeps a bounded nudge (no divide blow-up)."""
        from qmbp_simulation.framework.study_core import analytic_seed_new_params

        def cost(x):
            return 0.0  # perfectly flat

        out = analytic_seed_new_params(cost, np.array([0.0, 0.0]), [0, 1])
        assert np.all(np.isfinite(out))
        assert np.all(np.abs(out) <= np.pi)

    def test_fidelity_bond_gradients_shapes_and_nonneg(self, gs_n6):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.framework.study_core import fidelity_bond_gradients
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, _e0, _gap, n_nn, _n_nnn = gs_n6
        nnn = HamiltonianBuilder._generate_nnn_edges(lat)
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        rng = np.random.default_rng(0)
        theta = rng.uniform(-0.3, 0.3, n_nn + 6)
        g_nn, g_nnn = fidelity_bond_gradients(HVACircuitBuilder(), 6, lat, sel, [], nnn, H, psi, theta)
        assert g_nn.size == 0
        assert g_nnn.shape[0] == len(nnn)
        assert np.all(g_nnn >= 0.0)

    def test_block_precondition_scales_normalized_and_clipped(self):
        from qmbp_simulation.framework.study_core import block_precondition_scales

        # blocks [nn, nnn, x], sel n_nn=3 n_nnn=2 n_q=4 → 9 params.
        grad = np.concatenate([np.full(3, 2.0), np.full(2, 0.5), np.full(4, 0.1)])
        D = block_precondition_scales(["nn", "nnn", "x"], 3, 2, 4, grad)
        assert D.shape[0] == 9
        assert np.all(D >= 0.1) and np.all(D <= 10.0)  # clipped to [floor, cap]
        # Stiff block (nn, largest grad) gets the SMALLEST scale; soft block (x) largest.
        assert D[0] < D[-1]

    def test_preconditioned_converge_matches_plain_when_D_is_ones(self, gs_n6):
        """D=ones → converge_circuit_preconditioned is identical to converge_circuit."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant
        from qmbp_simulation.framework.study_core import (
            converge_circuit,
            converge_circuit_preconditioned,
        )

        lat, _qc, H, psi, _e0, _gap, _n_nn, _n_nnn = gs_n6
        sel = BondSelection(nn_edges=list(lat.edges), nnn_edges=[])
        v = make_masked_variant("t", "t", ["nn", "nnn", "x"], sel)
        qc, _ = build_variant(HVACircuitBuilder(), 6, lat, v)
        a = converge_circuit(qc, H, psi, restarts=1, maxiter=100, seed0=3)
        b = converge_circuit_preconditioned(qc, H, psi, restarts=1, maxiter=100, seed0=3, D=np.ones(qc.num_parameters))
        assert a[0] == pytest.approx(b[0], abs=1e-9)  # same best fidelity

    def test_three_levers_run_combined(self, gs_n6):
        """2c+1a+2b together run end-to-end and return a valid growth curve."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.framework.study_core import grow_adapt_pool
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat, _qc, H, psi, e0, gap, n_nn, n_nnn = gs_n6
        nn_all = list(lat.edges)
        nnn_all = HamiltonianBuilder._generate_nnn_edges(lat)
        steps, reached = grow_adapt_pool(
            HVACircuitBuilder(),
            6,
            lat,
            H,
            psi,
            e0,
            gap,
            nn_all=nn_all,
            nnn_all=nnn_all,
            n_nn=n_nn,
            n_nnn=n_nnn,
            restarts=2,
            maxiter=250,
            seed0=1,
            target_fid=0.999,
            grow_step=2,
            allow_repeat_layer=True,
            max_layers=4,
            efficiency_patience=3,
            analytic_seed_new=True,
            select_by_fidelity=True,
            block_precondition=True,
        )
        assert len(steps) >= 1
        for s in steps:
            assert len(s["best_theta_final"]) == s["n_params"]
            assert 0.0 <= s["best_fidelity"] <= 1.0 + 1e-9
        assert max(s["best_fidelity"] for s in steps) > 0.9
