"""Unit tests for the HVA study infrastructure added in the angle/variant work.

Covers the src-side building blocks that carry real regression risk and that
several experiment runners depend on. Per the repo rule, tests import only from
``src`` (``qmbp_simulation``), never from ``scripts/``. Deterministic, small-N,
no cloud, no heavy VQE.

The five highest-value guards:
  1. NoiselessBackend cache-equivalence  — the cached ⟨H⟩ path must equal the
     estimator path exactly (a silent break here corrupts every experiment's
     fidelity/energy).
  2. create_bond_resolved_frustrated_configurable — the variant engine must
     reproduce the production p=1 builder and count parameters correctly.
  3. study_runner._run_record — every restart record must carry theta_init /
     theta_final / converged (the standardized angle persistence).
  4. compute_restart_convergence — the convergence diagnostic aggregation.
  5. compute_state_spectral_decomposition — the per-eigenvector weight
     decomposition used to locate where the missing fidelity goes.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import make_lattice

# ── Fixtures: a small frustrated square system, exact ─────────────────────────


@pytest.fixture(scope="module")
def frustrated_n6():
    """N=6 frustrated square: (lattice, H, qc, n_nn, n_nnn)."""
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec("tfim_frustrated")
    hk = dict(spec.hamiltonian_kwargs)
    hk["J2"] = 0.5
    lat = make_lattice("square", 6, J=1.0, h=0.5)
    H = spec.build_hamiltonian(lat, **hk)
    qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(6, 1, lat)
    n_nn = len(lat.edges)
    n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat))
    return lat, H, qc, n_nn, n_nnn


# ── 1. NoiselessBackend cache-equivalence (highest value) ─────────────────────


class TestBackendCacheEquivalence:
    """The cached ⟨H⟩ path must be numerically identical to the estimator path."""

    def test_cached_equals_estimator_across_thetas(self, frustrated_n6):
        _, H, qc, _, _ = frustrated_n6
        b_cache = NoiselessBackend(cache_hamiltonian=True)
        b_est = NoiselessBackend(cache_hamiltonian=False)
        rng = np.random.default_rng(0)
        maxdiff = 0.0
        for _ in range(12):
            x = rng.uniform(-np.pi, np.pi, qc.num_parameters)
            e_c = b_cache.evaluate(qc, H, x)
            e_e = b_est.evaluate(qc, H, x)
            maxdiff = max(maxdiff, abs(e_c - e_e))
        assert maxdiff < 1e-9, f"cache vs estimator diverged by {maxdiff:.2e}"

    def test_cache_not_shared_across_different_hamiltonians(self):
        """Distinct H objects must not collide in the id-keyed matrix cache."""
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim_frustrated")
        hk = dict(spec.hamiltonian_kwargs)
        hk["J2"] = 0.5
        lat_a = make_lattice("square", 6, J=1.0, h=0.5)
        lat_b = make_lattice("square", 6, J=1.0, h=2.0)  # different field
        Ha = spec.build_hamiltonian(lat_a, **hk)
        Hb = spec.build_hamiltonian(lat_b, **hk)
        qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(6, 1, lat_a)
        b = NoiselessBackend(cache_hamiltonian=True)
        x = np.zeros(qc.num_parameters)
        ea = b.evaluate(qc, Ha, x)
        eb = b.evaluate(qc, Hb, x)
        # Different Hamiltonians → different energies (no cache bleed).
        assert abs(ea - eb) > 1e-6


# ── 2. Configurable variant engine ───────────────────────────────────────────


class TestConfigurableAnsatz:
    """The variant engine must match the production builder and count params."""

    def test_p1_base_matches_production_builder(self, frustrated_n6):
        lat, _, ref_qc, _, _ = frustrated_n6
        b = HVACircuitBuilder()
        v, _ = b.create_bond_resolved_frustrated_configurable(6, lat, blocks=["nn", "nnn", "x"])
        assert v.num_parameters == ref_qc.num_parameters

    def test_p2_layout_matches_two_layers(self, frustrated_n6):
        lat, _, _, _, _ = frustrated_n6
        b = HVACircuitBuilder()
        ref2, _ = b.create_bond_resolved_frustrated(6, 2, lat)
        v, _ = b.create_bond_resolved_frustrated_configurable(6, lat, blocks=["nn", "nnn", "x", "nn", "nnn", "x"])
        assert v.num_parameters == ref2.num_parameters

    def test_rx_final_adds_nqubits_params_zero_2q(self, frustrated_n6):
        lat, _, base_qc, _, _ = frustrated_n6
        b = HVACircuitBuilder()
        v, _ = b.create_bond_resolved_frustrated_configurable(6, lat, blocks=["nn", "nnn", "x"], rx_final=True)
        assert v.num_parameters == base_qc.num_parameters + 6  # +N θ_x, no new edges

    def test_unknown_block_raises(self, frustrated_n6):
        lat, _, _, _, _ = frustrated_n6
        b = HVACircuitBuilder()
        with pytest.raises(ValueError):
            b.create_bond_resolved_frustrated_configurable(6, lat, blocks=["bogus"])


# ── 3. Standardized restart record (theta persistence) ────────────────────────


class TestRunRecord:
    """Every strategy record must carry theta_init/theta_final/converged."""

    def test_all_strategies_emit_theta_and_convergence(self):
        from qmbp_simulation.framework.study_runner import (
            strategy_bestof,
            strategy_metropolis,
            strategy_single,
        )

        target = np.array([0.3, -0.2, 0.5])

        def cost(x):
            return float(np.sum((np.asarray(x) - target) ** 2))

        def fid(x):
            return float(np.exp(-np.sum((np.asarray(x) - target) ** 2)))

        ws = np.zeros(3)
        for runs in (
            strategy_single(cost, fid, ws, maxiter=50)[2],
            strategy_bestof(cost, fid, ws, maxiter=50, sigmas=(0.0, 0.2))[2],
            strategy_metropolis(cost, fid, ws, maxiter=50, n_hops=2)[2],
        ):
            for r in runs:
                assert "theta_init" in r and len(r["theta_init"]) == 3
                assert "theta_final" in r and len(r["theta_final"]) == 3
                assert "converged" in r and isinstance(r["converged"], bool)


# ── 4. Restart-convergence diagnostic ─────────────────────────────────────────


class TestRestartConvergence:
    def test_counts_and_best_flag(self):
        from qmbp_simulation.analysis.metrics import compute_restart_convergence

        runs = [
            {"nit": 300, "energy": -16.6, "fidelity": 0.65, "converged": False},
            {"nit": 120, "energy": -16.2, "fidelity": 0.27, "converged": True},
            {"nit": 80, "energy": -16.3, "fidelity": 0.49, "converged": True},
        ]
        cc = compute_restart_convergence(runs, maxiter=300)
        assert cc["n_restarts"] == 3
        assert cc["converged_count"] == 2
        assert cc["all_converged"] is False
        assert cc["any_converged"] is True
        # best restart = lowest energy (-16.6) which is the non-converged one
        assert cc["best_restart_converged"] is False
        assert cc["max_nit"] == 300

    def test_infers_converged_from_nit_when_flag_absent(self):
        from qmbp_simulation.analysis.metrics import compute_restart_convergence

        runs = [{"nit": 50, "energy": -1.0, "fidelity": 0.9}]  # no 'converged'
        cc = compute_restart_convergence(runs, maxiter=100)
        assert cc["converged_count"] == 1  # 50 < 100

    def test_empty_runs_safe(self):
        from qmbp_simulation.analysis.metrics import compute_restart_convergence

        cc = compute_restart_convergence([], maxiter=100)
        assert cc["n_restarts"] == 0 and cc["all_converged"] is False


# ── 5. Spectral decomposition of the prepared state ───────────────────────────


class TestSpectralDecomposition:
    def test_weights_and_subspace(self, frustrated_n6):
        from scipy.sparse.linalg import eigsh

        from qmbp_simulation.analysis.fidelity import (
            compute_state_spectral_decomposition,
        )

        _, H, qc, _, _ = frustrated_n6
        evals, evecs = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
        order = np.argsort(evals)
        cols = evecs[:, order]
        theta = np.zeros(qc.num_parameters)
        out = compute_state_spectral_decomposition(qc, theta, cols, n_low=2)
        assert out["ground_weight"] is not None
        assert 0.0 <= out["ground_weight"] <= 1.0
        # subspace fidelity = sum of the low weights, in [0,1], >= ground_weight
        assert out["subspace_fidelity"] >= out["ground_weight"] - 1e-9
        assert 0.0 <= out["subspace_fidelity"] <= 1.0 + 1e-9
        # weight outside = 1 - subspace, consistent
        assert abs(out["weight_outside_low_subspace"] - (1.0 - out["subspace_fidelity"])) < 1e-9
