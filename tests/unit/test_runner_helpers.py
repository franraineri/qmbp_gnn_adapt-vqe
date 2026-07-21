"""Tests for ValidationRunner helper methods added in the refactoring session.

Covers:
- _resolve_topology (topology argument resolution)
- evaluate_noiseless_at_h (noiseless energy evaluation shortcut)
- predict_mpnn_at_h (MPNN inference shortcut)
- setup_noisy_estimation (noisy infrastructure setup)
- ZNE constants centralization (models.constants)

Edge cases tested:
- Topology as list vs string
- Missing args attributes (graceful fallback)
- Variational principle compliance
- Different topologies and system sizes
- Without setup_physics() (fallback imports)
- With setup_physics() (reuses cached objects)
- NaN/Inf handling
- Wrong-size theta vectors
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner

# ─────────────────────────────────────────────────────────────────────────────
# Minimal concrete runner for testing
# ─────────────────────────────────────────────────────────────────────────────


class _MinimalRunner(ValidationRunner):
    """Concrete runner for testing base class helpers."""

    runner_id = "test_runner"
    experiment_id = "TEST"
    description = "Test runner for helpers"
    hypothesis = "Helpers work"

    def define_sections(self):
        return [Section(id=1, name="noop", fn=lambda: {}, hypothesis="noop")]


@pytest.fixture
def runner_with_physics():
    """Runner with setup_physics() called (full infrastructure)."""
    args = argparse.Namespace(
        n_qubits=4,
        p_layers=1,
        topology=["chain_1d"],
        model="tfim",
        h_min=1.0,
        h_max=2.0,
        h_points=3,
        seeds=[42],
        maxiter=100,
        n_restarts=1,
        verbose=False,
        section=None,
        dry_run=False,
        skip_preflight=False,
        stop_on_failure=False,
        validate_vqe=True,
        validate_theta=True,
        theta_validation_level=4,
        strict_validation=False,
        resume=None,
        save_artifacts="never",
        no_bidirectional=False,
        force_bidirectional=False,
        preset=None,
        output=None,
        model_params=None,
    )
    runner = _MinimalRunner(args)
    runner.setup_physics()
    return runner


@pytest.fixture
def runner_no_physics():
    """Runner WITHOUT setup_physics() — tests fallback path."""
    args = argparse.Namespace(
        n_qubits=4,
        p_layers=1,
        topology="chain_1d",
        model="tfim",
        verbose=False,
        section=None,
        dry_run=False,
        skip_preflight=False,
        stop_on_failure=False,
        validate_vqe=True,
        validate_theta=True,
        theta_validation_level=4,
        strict_validation=False,
        resume=None,
        save_artifacts="never",
        no_bidirectional=False,
        force_bidirectional=False,
        preset=None,
    )
    runner = _MinimalRunner(args)
    return runner


# ═══════════════════════════════════════════════════════════════════════════════
# Test: _resolve_topology
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveTopology:
    """Test _resolve_topology handles all input formats."""

    def test_explicit_string_override(self, runner_with_physics):
        assert runner_with_physics._resolve_topology("heavy_hex") == "heavy_hex"

    def test_explicit_none_uses_args_list(self, runner_with_physics):
        # args.topology = ["chain_1d"]
        assert runner_with_physics._resolve_topology(None) == "chain_1d"

    def test_args_topology_as_string(self, runner_no_physics):
        # args.topology = "chain_1d" (not a list)
        assert runner_no_physics._resolve_topology(None) == "chain_1d"

    def test_args_topology_missing_defaults_chain(self):
        """If args has no topology attr at all, default to chain_1d."""
        args = argparse.Namespace(
            n_qubits=4,
            verbose=False,
            section=None,
            dry_run=False,
            skip_preflight=False,
            stop_on_failure=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            preset=None,
        )
        runner = _MinimalRunner(args)
        assert runner._resolve_topology(None) == "chain_1d"

    def test_args_topology_multi_element_list(self):
        """With multiple topologies, takes first."""
        args = argparse.Namespace(
            n_qubits=4,
            p_layers=1,
            topology=["heavy_hex", "chain_1d"],
            model="tfim",
            verbose=False,
            section=None,
            dry_run=False,
            skip_preflight=False,
            stop_on_failure=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            preset=None,
        )
        runner = _MinimalRunner(args)
        assert runner._resolve_topology(None) == "heavy_hex"

    def test_args_topology_empty_list_defaults_chain(self):
        """Empty list defaults to chain_1d (no IndexError)."""
        args = argparse.Namespace(
            n_qubits=4,
            p_layers=1,
            topology=[],
            model="tfim",
            verbose=False,
            section=None,
            dry_run=False,
            skip_preflight=False,
            stop_on_failure=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            preset=None,
        )
        runner = _MinimalRunner(args)
        assert runner._resolve_topology(None) == "chain_1d"

    def test_args_topology_empty_string_defaults_chain(self):
        """Empty string defaults to chain_1d."""
        args = argparse.Namespace(
            n_qubits=4,
            p_layers=1,
            topology="",
            model="tfim",
            verbose=False,
            section=None,
            dry_run=False,
            skip_preflight=False,
            stop_on_failure=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            preset=None,
        )
        runner = _MinimalRunner(args)
        assert runner._resolve_topology(None) == "chain_1d"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: evaluate_noiseless_at_h
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateNoiselessAtH:
    """Test evaluate_noiseless_at_h for correctness and edge cases."""

    def test_returns_float(self, runner_with_physics):
        theta = np.array([0.1, -0.05])
        e = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=theta)
        assert isinstance(e, float)

    def test_returns_finite(self, runner_with_physics):
        theta = np.zeros(2)
        e = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=theta)
        assert np.isfinite(e)

    def test_different_h_gives_different_energy(self, runner_with_physics):
        theta = np.array([0.1, -0.05])
        e1 = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=theta)
        e2 = runner_with_physics.evaluate_noiseless_at_h(h=1.5, theta=theta)
        assert e1 != e2

    def test_different_theta_gives_different_energy(self, runner_with_physics):
        e1 = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=np.array([0.1, 0.1]))
        e2 = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=np.array([1.0, 1.0]))
        assert e1 != e2

    def test_variational_principle(self, runner_with_physics):
        """Energy from evaluate must be >= exact ground state energy."""
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers import ClassicalSolver

        h = 2.0
        spec = get_model_spec("tfim")
        lattice = runner_with_physics.make_lattice("chain_1d", 4, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        solver = ClassicalSolver()
        gt = solver.solve(H, lattice)
        e_exact = gt.ground_energy

        # Random theta should give energy above ground state
        rng = np.random.default_rng(123)
        for _ in range(5):
            theta = rng.uniform(-1, 1, 2)
            e = runner_with_physics.evaluate_noiseless_at_h(h=h, theta=theta)
            assert e >= e_exact - 1e-6, f"Variational principle violated: {e} < {e_exact}"

    def test_explicit_topology_override(self, runner_with_physics):
        """Can override topology even if args says chain_1d."""
        theta = np.zeros(2)  # p=1 tfim
        # ladder N=4 has same n_params for p=1 tfim
        e = runner_with_physics.evaluate_noiseless_at_h(
            h=2.0, theta=theta, topology="ladder", n_qubits=4, p_layers=1
        )
        assert np.isfinite(e)

    def test_works_without_setup_physics(self, runner_no_physics):
        """Fallback path: imports fresh when setup_physics wasn't called."""
        theta = np.array([0.1, -0.05])
        e = runner_no_physics.evaluate_noiseless_at_h(
            h=2.0, theta=theta, topology="chain_1d", n_qubits=4, p_layers=1
        )
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_model_override(self, runner_with_physics):
        """Can specify model explicitly."""
        theta = np.array([0.1, -0.05, 0.02])  # tfim_longitudinal has 3 params
        e = runner_with_physics.evaluate_noiseless_at_h(
            h=2.0, theta=theta, model="tfim_longitudinal", n_qubits=4, p_layers=1
        )
        assert np.isfinite(e)

    def test_zero_theta_gives_finite(self, runner_with_physics):
        """θ=0 should give a valid (though not optimal) energy."""
        theta = np.zeros(2)
        e = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=theta)
        assert np.isfinite(e)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: predict_mpnn_at_h
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictMPNNAtH:
    """Test predict_mpnn_at_h for correctness and edge cases."""

    @pytest.fixture
    def dummy_predictor(self):
        """Untrained MPNN predictor for shape/type testing."""
        from qmbp_simulation.predictors import MPNNPredictor

        predictor = MPNNPredictor(node_features=2, output_dim=2, hidden_dim=16)
        predictor.eval()
        return predictor

    def test_returns_numpy_array(self, runner_with_physics, dummy_predictor):
        theta = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=2.0)
        assert isinstance(theta, np.ndarray)

    def test_correct_output_shape(self, runner_with_physics, dummy_predictor):
        """Output should match predictor output_dim."""
        theta = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=2.0)
        assert theta.shape == (2,)

    def test_returns_finite_values(self, runner_with_physics, dummy_predictor):
        theta = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=2.0)
        assert np.all(np.isfinite(theta))

    def test_different_h_different_prediction(self, runner_with_physics, dummy_predictor):
        """Different h-features should produce different outputs."""
        t1 = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=1.0)
        t2 = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=3.0)
        assert not np.allclose(t1, t2, atol=1e-6)

    def test_deterministic_same_h(self, runner_with_physics, dummy_predictor):
        """Same h should always give same prediction (eval mode, no dropout)."""
        t1 = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=2.0)
        t2 = runner_with_physics.predict_mpnn_at_h(dummy_predictor, h=2.0)
        np.testing.assert_array_equal(t1, t2)

    def test_topology_override(self, runner_with_physics, dummy_predictor):
        """Can predict with a different topology than args."""
        theta = runner_with_physics.predict_mpnn_at_h(
            dummy_predictor, h=2.0, topology="ladder", n_qubits=4
        )
        assert theta.shape == (2,)
        assert np.all(np.isfinite(theta))

    def test_works_without_setup_physics(self, runner_no_physics):
        """Fallback: creates HamiltonianBuilder if self.builder is None."""
        from qmbp_simulation.predictors import MPNNPredictor

        predictor = MPNNPredictor(node_features=2, output_dim=2, hidden_dim=16)
        predictor.eval()
        theta = runner_no_physics.predict_mpnn_at_h(
            predictor, h=2.0, topology="chain_1d", n_qubits=4
        )
        assert theta.shape == (2,)
        assert np.all(np.isfinite(theta))

    def test_larger_output_dim(self, runner_with_physics):
        """Works with models that have more parameters (p=2)."""
        from qmbp_simulation.predictors import MPNNPredictor

        # p=2 tfim has 4 params
        predictor = MPNNPredictor(node_features=2, output_dim=4, hidden_dim=16)
        predictor.eval()
        theta = runner_with_physics.predict_mpnn_at_h(predictor, h=2.0)
        assert theta.shape == (4,)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: ZNE Constants Centralization
# ═══════════════════════════════════════════════════════════════════════════════


class TestZNEConstants:
    """Verify ZNE constants are correct and importable."""

    def test_all_constants_importable(self):
        from qmbp_simulation.models.constants import (
            DE_GAP_THRESHOLD,
            ZNE_CES_PERTURBATIVE_THRESHOLD,
            ZNE_DEFAULT_N_CANDIDATE_LAYOUTS,
            ZNE_DEFAULT_NOISE_FACTORS,
            ZNE_DEFAULT_SHOTS,
            ZNE_LINEAR_REGIME_CX_LIMIT,
        )

        assert DE_GAP_THRESHOLD == 0.05
        assert ZNE_DEFAULT_SHOTS == 16384
        assert ZNE_DEFAULT_NOISE_FACTORS == (1, 3, 5)
        assert ZNE_DEFAULT_N_CANDIDATE_LAYOUTS == 20
        assert ZNE_LINEAR_REGIME_CX_LIMIT == 18
        assert ZNE_CES_PERTURBATIVE_THRESHOLD == 0.3

    def test_noise_factors_are_odd_integers(self):
        from qmbp_simulation.models.constants import ZNE_DEFAULT_NOISE_FACTORS

        for nf in ZNE_DEFAULT_NOISE_FACTORS:
            assert nf % 2 == 1, f"Noise factor {nf} is not odd"
            assert nf >= 1, f"Noise factor {nf} < 1"

    def test_shots_is_power_of_two(self):
        from qmbp_simulation.models.constants import ZNE_DEFAULT_SHOTS

        # 16384 = 2^14
        assert ZNE_DEFAULT_SHOTS & (ZNE_DEFAULT_SHOTS - 1) == 0

    def test_de_gap_threshold_positive(self):
        from qmbp_simulation.models.constants import DE_GAP_THRESHOLD

        assert 0 < DE_GAP_THRESHOLD < 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test: setup_noisy_estimation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestSetupNoisyEstimation:
    """Test setup_noisy_estimation (requires qiskit-ibm-runtime)."""

    def test_creates_fake_backend(self, runner_with_physics):
        runner_with_physics.setup_noisy_estimation(4)
        assert hasattr(runner_with_physics, "fake_backend")
        assert runner_with_physics.fake_backend is not None

    def test_creates_noisy_config(self, runner_with_physics):
        runner_with_physics.setup_noisy_estimation(4, shots=8192)
        assert runner_with_physics.noisy_config.shots == 8192

    def test_creates_candidate_layouts(self, runner_with_physics):
        runner_with_physics.setup_noisy_estimation(4)
        assert len(runner_with_physics.candidates) > 0

    def test_binds_utility_functions(self, runner_with_physics):
        runner_with_physics.setup_noisy_estimation(4)
        assert callable(runner_with_physics.noisy_estimate)
        assert callable(runner_with_physics.run_gf_zne)
        assert callable(runner_with_physics.run_pea_zne)
        assert callable(runner_with_physics.run_adaptive_zne)
        assert callable(runner_with_physics.select_low_ces)
        assert callable(runner_with_physics.affine_correct_energy)

    def test_reduces_candidates_for_large_n(self, runner_with_physics):
        """N>=16 should get fewer candidates (faster BFS)."""
        runner_with_physics.setup_noisy_estimation(16)
        # Default for N>=16 is 10 (not 20)
        assert len(runner_with_physics.candidates) <= 10

    def test_custom_seed(self, runner_with_physics):
        runner_with_physics.setup_noisy_estimation(4, seed_simulator=99)
        assert runner_with_physics.noisy_config.seed_simulator == 99


# ═══════════════════════════════════════════════════════════════════════════════
# Test: safe_per_h_loop edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafePerHLoop:
    """Test safe_per_h_loop error isolation and edge cases."""

    def test_all_succeed(self):
        results = ValidationRunner.safe_per_h_loop(
            [1.0, 2.0, 3.0], lambda h: {"h": h, "value": h * 2}, "test"
        )
        assert len(results) == 3
        assert results[0]["value"] == 2.0

    def test_some_fail_gracefully(self):
        """Failed points are skipped, successful ones preserved."""

        def fn(h):
            if h == 2.0:
                raise ValueError("intentional failure")
            return {"h": h}

        results = ValidationRunner.safe_per_h_loop([1.0, 2.0, 3.0], fn, "test")
        assert len(results) == 2
        assert all(r["h"] != 2.0 for r in results)

    def test_none_return_skipped(self):
        """fn returning None signals a skip."""
        results = ValidationRunner.safe_per_h_loop(
            [1.0, 2.0, 3.0], lambda h: None if h == 2.0 else {"h": h}, "test"
        )
        assert len(results) == 2

    def test_all_fail_returns_empty(self):
        """If all points fail, returns empty list (not exception)."""
        results = ValidationRunner.safe_per_h_loop([1.0, 2.0], lambda h: None, "test")
        assert results == []

    def test_empty_h_values(self):
        """Empty input gives empty output."""
        results = ValidationRunner.safe_per_h_loop([], lambda h: {"h": h}, "test")
        assert results == []

    def test_exception_types_isolated(self):
        """Various exception types don't propagate."""

        def fn(h):
            if h == 1.0:
                raise TypeError("type error")
            if h == 2.0:
                raise RuntimeError("runtime")
            return {"h": h}

        results = ValidationRunner.safe_per_h_loop([1.0, 2.0, 3.0], fn, "test")
        assert len(results) == 1
        assert results[0]["h"] == 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Integration (predict → evaluate)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictEvaluateIntegration:
    """End-to-end: predict θ with MPNN, then evaluate energy."""

    def test_predict_then_evaluate_gives_finite_energy(self, runner_with_physics):
        from qmbp_simulation.predictors import MPNNPredictor

        predictor = MPNNPredictor(node_features=2, output_dim=2, hidden_dim=16)
        predictor.eval()

        theta_pred = runner_with_physics.predict_mpnn_at_h(predictor, h=2.0)
        e = runner_with_physics.evaluate_noiseless_at_h(h=2.0, theta=theta_pred)

        assert np.isfinite(e)
        assert isinstance(e, float)

    def test_predict_evaluate_variational_principle(self, runner_with_physics):
        """Predicted θ should still respect variational principle."""
        from qmbp_simulation.predictors import MPNNPredictor

        predictor = MPNNPredictor(node_features=2, output_dim=2, hidden_dim=16)
        predictor.eval()

        h = 2.0
        theta_pred = runner_with_physics.predict_mpnn_at_h(predictor, h=h)
        e_pred = runner_with_physics.evaluate_noiseless_at_h(h=h, theta=theta_pred)

        e_exact, _ = runner_with_physics.exact_ground_state("chain_1d", 4, h)
        assert e_pred >= e_exact - 1e-6
