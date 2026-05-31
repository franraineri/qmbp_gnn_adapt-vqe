"""Integration tests for model-aware Heisenberg pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qmbp_simulation.models import HamiltonianBuilder, VQEConfig, make_lattice
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.pipeline.runner import PipelineRunner
from qmbp_simulation.solvers import ClassicalSolver

pytestmark = pytest.mark.integration

# All tests use N=4 for speed (<5s each)
N_QUBITS = 4
TOPOLOGY = "chain_1d"
SEED = 42


@pytest.fixture
def chain_lattice():
    """N=4 chain lattice."""
    return make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=1.5)


@pytest.fixture
def heisenberg_spec():
    """Heisenberg model spec."""
    return get_model_spec("heisenberg")


@pytest.fixture
def tfim_spec():
    """TFIM model spec."""
    return get_model_spec("tfim")


class TestHeisenbergPipelineIntegration:
    """Integration tests for model-aware pipeline dispatch."""

    def test_phase1_heisenberg_produces_different_energies_than_tfim(
        self, chain_lattice, heisenberg_spec, tfim_spec
    ):
        """Same lattice, different model → different ground energies."""
        h_values = np.array([2.0, 1.5])
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)

        runner_heis = PipelineRunner(
            chain_lattice, config, model_spec=heisenberg_spec, seed=SEED
        )
        runner_tfim = PipelineRunner(
            chain_lattice, config, model_spec=tfim_spec, seed=SEED
        )

        results_heis = runner_heis.run_phase1(h_values)
        results_tfim = runner_tfim.run_phase1(h_values)

        # Energies must differ — different Hamiltonians
        for r_h, r_t in zip(results_heis, results_tfim):
            assert r_h.ground_energy != pytest.approx(r_t.ground_energy, abs=1e-6)

    def test_phase2_heisenberg_produces_8_params(
        self, chain_lattice, heisenberg_spec
    ):
        """VQE results have 8 parameters (4 per layer × 2 layers)."""
        h_values = np.array([2.0, 1.5])
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)

        runner = PipelineRunner(
            chain_lattice, config, model_spec=heisenberg_spec, seed=SEED
        )
        exact = runner.run_phase1(h_values)
        vqe_results = runner.run_phase2(h_values, exact)

        for r in vqe_results:
            assert len(r.theta_opt) == 8  # 4 params/layer × 2 layers

    def test_phase2_tfim_backward_compat(self, chain_lattice):
        """PipelineRunner without model_spec still works (4 params for p=2)."""
        h_values = np.array([2.0, 1.5])
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)

        # No model_spec → defaults to TFIM behavior
        runner = PipelineRunner(chain_lattice, config, seed=SEED)
        exact = runner.run_phase1(h_values)
        vqe_results = runner.run_phase2(h_values, exact)

        for r in vqe_results:
            assert len(r.theta_opt) == 4  # 2 params/layer × 2 layers

    def test_pipeline_model_spec_dispatch_hamiltonian(
        self, chain_lattice, heisenberg_spec, tfim_spec
    ):
        """Verify Heisenberg Hamiltonian is used (energy differs from TFIM)."""
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)
        h_values = np.array([1.5])

        runner_heis = PipelineRunner(
            chain_lattice, config, model_spec=heisenberg_spec, seed=SEED
        )
        runner_tfim = PipelineRunner(
            chain_lattice, config, model_spec=tfim_spec, seed=SEED
        )

        e_heis = runner_heis.run_phase1(h_values)[0].ground_energy
        e_tfim = runner_tfim.run_phase1(h_values)[0].ground_energy

        # Different Hamiltonians → different ground state energies
        assert abs(e_heis - e_tfim) > 0.1

    def test_pipeline_model_spec_dispatch_circuit(
        self, chain_lattice, heisenberg_spec, tfim_spec
    ):
        """Verify Heisenberg circuit is used (param count differs from TFIM)."""
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)

        runner_heis = PipelineRunner(
            chain_lattice, config, model_spec=heisenberg_spec, seed=SEED
        )
        runner_tfim = PipelineRunner(
            chain_lattice, config, model_spec=tfim_spec, seed=SEED
        )

        n_heis = runner_heis._n_variational_params()
        n_tfim = runner_tfim._n_variational_params()

        assert n_heis == 8  # 4 per layer × 2
        assert n_tfim == 4  # 2 per layer × 2

    def test_n_variational_params_helper(
        self, chain_lattice, heisenberg_spec, tfim_spec
    ):
        """Verify _n_variational_params returns correct values for each model."""
        config_p1 = VQEConfig(p_layers=1, n_restarts=1, maxiter=50)
        config_p2 = VQEConfig(p_layers=2, n_restarts=1, maxiter=50)

        # Heisenberg p=1 → 4, p=2 → 8
        runner_h1 = PipelineRunner(
            chain_lattice, config_p1, model_spec=heisenberg_spec, seed=SEED
        )
        runner_h2 = PipelineRunner(
            chain_lattice, config_p2, model_spec=heisenberg_spec, seed=SEED
        )
        assert runner_h1._n_variational_params() == 4
        assert runner_h2._n_variational_params() == 8

        # TFIM p=1 → 2, p=2 → 4
        runner_t1 = PipelineRunner(
            chain_lattice, config_p1, model_spec=tfim_spec, seed=SEED
        )
        runner_t2 = PipelineRunner(
            chain_lattice, config_p2, model_spec=tfim_spec, seed=SEED
        )
        assert runner_t1._n_variational_params() == 2
        assert runner_t2._n_variational_params() == 4

        # No model_spec → TFIM default
        runner_default = PipelineRunner(chain_lattice, config_p2, seed=SEED)
        assert runner_default._n_variational_params() == 4

    def test_phase3_output_dim_matches_model(
        self, chain_lattice, heisenberg_spec
    ):
        """Verify MPNN output_dim = 8 for Heisenberg (4 params/layer × 2 layers).

        Uses a small dataset to verify the MPNN is configured correctly.
        """
        h_values = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
        config = VQEConfig(p_layers=2, n_restarts=1, maxiter=100)

        runner = PipelineRunner(
            chain_lattice, config, model_spec=heisenberg_spec, seed=SEED
        )
        exact = runner.run_phase1(h_values)
        vqe_results = runner.run_phase2(h_values, exact)

        # Verify all VQE results have 8 parameters
        for r in vqe_results:
            assert len(r.theta_opt) == 8

        # The output dimension for MPNN should match total params
        expected_output_dim = heisenberg_spec.total_params_for_p(config.p_layers)
        assert expected_output_dim == 8
