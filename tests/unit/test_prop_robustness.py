"""Property-based tests for pipeline robustness fixes.

Validates the numerical stability, reproducibility, and error handling
improvements using Hypothesis to explore edge cases systematically.

Properties tested:
- P-R1: Energy evaluation always returns finite float for valid params
- P-R2: Shot noise is stochastic (different on consecutive calls)
- P-R3: Same seed produces identical VQE restart sequences
- P-R4: Same seed produces identical SPSA perturbation sequences
- P-R5: Parameter count mismatch always raises ValueError
- P-R6: SPSA parameters always stay within configured bounds
- P-R7: Warm-start always prefers previous_theta over zeros
- P-R8: Dataset validation catches NaN in any critical array
- P-R9: VQE energy is always ≥ exact ground state energy (variational principle)
- P-R10: MPNN output shape matches output_dim for any valid graph
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend, NoisyBackend
from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
    make_lattice,
)
from qmbp_simulation.optimizers import SPSAOptimizer, VQEOptimizer
from qmbp_simulation.pipeline import load_phase12_dataset, save_phase12_dataset
from qmbp_simulation.predictors import MPNNPredictor

# ─────────────────────────────────────────────────────────────────────────────
# Shared test infrastructure
# ─────────────────────────────────────────────────────────────────────────────

_LATTICE_4 = make_lattice("chain_1d", 4, J=1.0, h=1.5)
_HVA = HVACircuitBuilder()
_QC_4, _ = _HVA.create(4, 1, _LATTICE_4)
_H_4 = HamiltonianBuilder().build(_LATTICE_4)
_N_PARAMS_4 = _QC_4.num_parameters  # 2

# Strategy for valid circuit parameters
valid_params_2 = st.lists(
    st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=2,
).map(np.array)


# ─────────────────────────────────────────────────────────────────────────────
# P-R1: Energy evaluation always returns finite float
# ─────────────────────────────────────────────────────────────────────────────


class TestPR1EnergyAlwaysFinite:
    """For any valid parameter vector in [-π, π]^n, evaluate() returns
    a finite float. This must hold for both noiseless and noisy backends.
    """

    @given(params=valid_params_2)
    @settings(max_examples=50, deadline=None)
    def test_noiseless_always_finite(self, params):
        """NoiselessBackend.evaluate() is finite for all valid params."""
        backend = NoiselessBackend()
        energy = backend.evaluate(_QC_4, _H_4, params)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    @given(params=valid_params_2)
    @settings(max_examples=50, deadline=None)
    def test_noisy_gaussian_always_finite(self, params):
        """NoisyBackend (Gaussian mode) is finite for all valid params."""
        backend = NoisyBackend(shots=1024, seed_simulator=42)
        energy = backend.evaluate(_QC_4, _H_4, params)
        assert isinstance(energy, float)
        assert np.isfinite(energy)


# ─────────────────────────────────────────────────────────────────────────────
# P-R2: Shot noise is stochastic
# ─────────────────────────────────────────────────────────────────────────────


class TestPR2ShotNoiseStochastic:
    """For any fixed parameter vector, consecutive evaluate() calls on the
    same NoisyBackend instance produce different energies (shot noise advances).
    """

    @given(params=valid_params_2)
    @settings(max_examples=30, deadline=None)
    def test_consecutive_calls_differ(self, params):
        """Two consecutive evaluations with same params give different results."""
        backend = NoisyBackend(shots=100, seed_simulator=None)
        e1 = backend.evaluate(_QC_4, _H_4, params)
        e2 = backend.evaluate(_QC_4, _H_4, params)
        # With 100 shots, noise std ≈ 0.1 — extremely unlikely to be equal
        # (probability < 1e-15 for continuous distribution)
        assert e1 != e2, "Consecutive evaluations returned identical energy"

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        params=valid_params_2,
    )
    @settings(max_examples=30, deadline=None)
    def test_same_seed_same_sequence(self, seed, params):
        """Two backends with same seed produce same first N evaluations."""
        backend_a = NoisyBackend(shots=100, seed_simulator=seed)
        backend_b = NoisyBackend(shots=100, seed_simulator=seed)

        seq_a = [backend_a.evaluate(_QC_4, _H_4, params) for _ in range(3)]
        seq_b = [backend_b.evaluate(_QC_4, _H_4, params) for _ in range(3)]

        np.testing.assert_array_equal(seq_a, seq_b)


# ─────────────────────────────────────────────────────────────────────────────
# P-R3: VQE restart reproducibility
# ─────────────────────────────────────────────────────────────────────────────


class TestPR3VQERestartReproducibility:
    """For any seed, two VQEOptimizer instances with the same seed produce
    identical optimization results (same restart perturbations).
    """

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def test_same_seed_identical_sweep(self, seed):
        """Same seed → identical descending sweep results."""
        config = VQEConfig(p_layers=1, n_restarts=2, maxiter=10)
        h_values = np.array([2.0, 1.5])

        # Use a fixed lattice with h=2.0 for the first point so the initial
        # guess from get_initial_guess (which uses global np.random) is
        # deterministic within the same test invocation.
        np.random.seed(seed % 1000)
        opt_a = VQEOptimizer(config, seed=seed)
        results_a = opt_a.descending_sweep(h_values, _QC_4, _LATTICE_4)

        np.random.seed(seed % 1000)
        opt_b = VQEOptimizer(config, seed=seed)
        results_b = opt_b.descending_sweep(h_values, _QC_4, _LATTICE_4)

        for ra, rb in zip(results_a, results_b, strict=True):
            # Energy should match within L-BFGS-B numerical precision
            np.testing.assert_allclose(ra.energy, rb.energy, atol=1e-6)
            np.testing.assert_allclose(ra.theta_opt, rb.theta_opt, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# P-R4: SPSA perturbation reproducibility
# ─────────────────────────────────────────────────────────────────────────────


class TestPR4SPSAReproducibility:
    """For any seed, two SPSAOptimizer instances with the same seed produce
    identical optimization trajectories.
    """

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=20, deadline=None)
    def test_same_seed_identical_result(self, seed):
        """Same seed → identical SPSA result."""
        backend = NoiselessBackend()
        initial = np.array([0.1, 0.2])

        spsa_a = SPSAOptimizer(backend=backend, a=0.1, c=0.05, seed=seed)
        spsa_b = SPSAOptimizer(backend=backend, a=0.1, c=0.05, seed=seed)

        result_a = spsa_a.optimize(_QC_4, _H_4, initial, n_iterations=10)
        result_b = spsa_b.optimize(_QC_4, _H_4, initial, n_iterations=10)

        np.testing.assert_allclose(result_a.energy, result_b.energy, atol=1e-12)
        np.testing.assert_allclose(result_a.theta_opt, result_b.theta_opt, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# P-R5: Parameter count mismatch detection
# ─────────────────────────────────────────────────────────────────────────────


class TestPR5ParameterCountMismatch:
    """For any parameter vector whose length ≠ circuit.num_parameters,
    evaluate() raises ValueError.
    """

    @given(
        n_params=st.integers(min_value=0, max_value=10).filter(lambda n: n != _N_PARAMS_4),
    )
    @settings(max_examples=20, deadline=None)
    def test_wrong_length_raises(self, n_params):
        """Wrong parameter count always raises ValueError."""
        backend = NoiselessBackend()
        params = np.zeros(n_params)

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            backend.evaluate(_QC_4, _H_4, params)

    @given(
        n_params=st.integers(min_value=0, max_value=10).filter(lambda n: n != _N_PARAMS_4),
    )
    @settings(max_examples=20, deadline=None)
    def test_noisy_wrong_length_raises(self, n_params):
        """NoisyBackend also rejects wrong parameter count."""
        backend = NoisyBackend(shots=100, seed_simulator=42)
        params = np.zeros(n_params)

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            backend.evaluate(_QC_4, _H_4, params)


# ─────────────────────────────────────────────────────────────────────────────
# P-R6: SPSA bounds enforcement
# ─────────────────────────────────────────────────────────────────────────────


class TestPR6SPSABoundsEnforcement:
    """For any bounds (lo, hi) where lo < hi, SPSA output parameters are
    always within [lo, hi].
    """

    @given(
        lo=st.floats(min_value=-3.5, max_value=-0.1, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=0.1, max_value=3.5, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=20, deadline=None)
    def test_output_within_bounds(self, lo, hi, seed):
        """SPSA output is always within configured bounds."""
        assume(lo < hi)
        backend = NoiselessBackend()
        initial = np.array([0.0, 0.0])

        spsa = SPSAOptimizer(backend=backend, a=0.5, c=0.1, bounds=(lo, hi), seed=seed)
        result = spsa.optimize(_QC_4, _H_4, initial, n_iterations=20)

        assert np.all(result.theta_opt >= lo - 1e-10), (
            f"Parameter below lower bound: {result.theta_opt.min()} < {lo}"
        )
        assert np.all(result.theta_opt <= hi + 1e-10), (
            f"Parameter above upper bound: {result.theta_opt.max()} > {hi}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P-R7: Warm-start preference
# ─────────────────────────────────────────────────────────────────────────────


class TestPR7WarmStartPreference:
    """For any h_value (including h=0) and any previous_theta, get_initial_guess
    always returns previous_theta when it is provided.
    """

    @given(
        h_value=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        previous_theta=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=4,
        ).map(np.array),
    )
    @settings(max_examples=50, deadline=None)
    def test_previous_theta_always_used(self, h_value, previous_theta):
        """previous_theta is always returned (copy) regardless of h_value."""
        config = VQEConfig(p_layers=1, warm_start_seed_zeros=True)
        n_params = len(previous_theta)

        result = VQEOptimizer.get_initial_guess(
            n_params=n_params,
            h_value=h_value,
            config=config,
            previous_theta=previous_theta,
        )

        np.testing.assert_array_equal(result, previous_theta)
        assert result is not previous_theta  # Must be a copy


# ─────────────────────────────────────────────────────────────────────────────
# P-R8: Dataset NaN detection
# ─────────────────────────────────────────────────────────────────────────────


class TestPR8DatasetNaNDetection:
    """For any dataset where a critical array contains NaN or Inf,
    load_phase12_dataset raises ValueError.
    """

    @given(
        bad_index=st.integers(min_value=0, max_value=2),
        bad_value=st.sampled_from([np.nan, np.inf, -np.inf]),
        array_name=st.sampled_from(["ground_energies", "vqe_energies", "gaps"]),
    )
    @settings(max_examples=30, deadline=None)
    def test_nan_inf_in_critical_arrays_raises(self, bad_index, bad_value, array_name):
        """NaN/Inf in ground_energies, vqe_energies, or gaps raises ValueError."""
        data = {
            "h_values": np.array([2.0, 1.5, 1.0]),
            "J": 1.0,
            "n_qubits": 4,
            "p_layers": 1,
            "ground_energies": np.array([-5.0, -4.5, -4.0]),
            "gaps": np.array([0.5, 0.3, 0.2]),
            "mag_x": np.array([0.8, 0.6, 0.3]),
            "corr_zz": np.array([0.2, 0.4, 0.7]),
            "theta_opt": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "vqe_energies": np.array([-4.9, -4.4, -3.9]),
            "fidelities": np.array([0.99, 0.97, 0.95]),
        }

        # Inject bad value
        data[array_name][bad_index] = bad_value

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "bad.npz"
            save_phase12_dataset(filepath, **data)
            with pytest.raises(ValueError, match="NaN/Inf"):
                load_phase12_dataset(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# P-R9: Variational principle (VQE energy ≥ exact)
# ─────────────────────────────────────────────────────────────────────────────


class TestPR9VariationalPrinciple:
    """For any valid parameters, the circuit energy is always ≥ the exact
    ground state energy (variational principle).
    """

    @given(params=valid_params_2)
    @settings(max_examples=50, deadline=None)
    def test_energy_geq_ground_state(self, params):
        """E(θ) ≥ E_exact for all θ (variational principle)."""
        from qmbp_simulation.solvers import ClassicalSolver

        backend = NoiselessBackend()
        solver = ClassicalSolver()

        exact = solver.solve(_H_4, _LATTICE_4)
        energy = backend.evaluate(_QC_4, _H_4, params)

        # Variational principle: circuit energy ≥ ground state energy
        # Allow tiny numerical tolerance
        assert energy >= exact.ground_energy - 1e-10, (
            f"Variational principle violated: E(θ)={energy:.8f} < E_exact={exact.ground_energy:.8f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P-R10: MPNN output shape consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestPR10MPNNOutputShape:
    """For any valid graph input matching the model's node_features,
    the MPNN output has shape [1, output_dim].
    """

    @given(
        h_value=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
        output_dim=st.sampled_from([2, 4]),
        hidden_dim=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=30, deadline=None)
    def test_output_shape_correct(self, h_value, output_dim, hidden_dim):
        """MPNN output shape is [1, output_dim] for single-graph input."""
        from torch_geometric.data import Data

        lattice = make_lattice("chain_1d", 4, J=1.0, h=h_value)
        builder = HamiltonianBuilder()
        edge_index_np, coord = builder.build_graph_data(lattice)

        h_feat = np.full(4, float(h_value))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model = MPNNPredictor(
            node_features=2, hidden_dim=hidden_dim, n_layers=2, output_dim=output_dim
        )
        model.eval()

        with torch.no_grad():
            output = model(graph)

        assert output.shape == (1, output_dim), (
            f"Expected shape (1, {output_dim}), got {output.shape}"
        )
        # Output must be finite
        assert torch.all(torch.isfinite(output)), "MPNN output contains NaN/Inf"
