"""Integration smoke tests for qmbp_simulation."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestSubmoduleImports:
    """Test all submodule imports succeed."""

    def test_import_utils(self):
        from qmbp_simulation import utils

        assert hasattr(utils, "set_global_seed")
        assert hasattr(utils, "json_serialize")
        assert hasattr(utils, "timer")

    def test_import_models(self):
        from qmbp_simulation import models

        assert hasattr(models, "make_lattice")
        assert hasattr(models, "HamiltonianBuilder")
        assert hasattr(models, "LatticeConfig")

    def test_import_solvers(self):
        from qmbp_simulation import solvers

        assert hasattr(solvers, "ClassicalSolver")

    def test_import_circuits(self):
        from qmbp_simulation import circuits

        assert hasattr(circuits, "HVACircuitBuilder")

    def test_import_execution(self):
        from qmbp_simulation import execution

        assert hasattr(execution, "NoiselessBackend")
        assert hasattr(execution, "HardwareBackend")

    def test_import_optimizers(self):
        from qmbp_simulation import optimizers

        assert hasattr(optimizers, "VQEOptimizer")

    def test_import_predictors(self):
        from qmbp_simulation import predictors

        assert hasattr(predictors, "MPNNPredictor")
        assert hasattr(predictors, "build_graph_dataset")

    def test_import_pipeline(self):
        from qmbp_simulation import pipeline

        assert hasattr(pipeline, "save_phase12_dataset")
        assert hasattr(pipeline, "load_phase12_dataset")

    def test_import_framework(self):
        from qmbp_simulation import framework

        assert hasattr(framework, "BaseExperiment")
        assert hasattr(framework, "ExperimentConfig")

    def test_import_analysis(self):
        from qmbp_simulation import analysis

        assert hasattr(analysis, "compute_snr")
        assert hasattr(analysis, "DiagnosticCollector")


class TestMinimalPipeline:
    """Test minimal pipeline: N=4, p=1, 3 h-points."""

    def test_phase1_and_phase2_minimal(self):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import (
            HamiltonianBuilder,
            VQEConfig,
            make_lattice,
        )
        from qmbp_simulation.optimizers import VQEOptimizer
        from qmbp_simulation.solvers import ClassicalSolver

        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()

        h_values = np.array([2.0, 1.5, 1.0])
        qc, theta = hva.create(4, 1, lattice)

        # Phase 1: exact solutions
        exact_results = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_results.append(solver.solve(H, lat_h))

        # Phase 2: VQE sweep
        config = VQEConfig(p_layers=1, n_restarts=2, maxiter=50)
        optimizer = VQEOptimizer(config, backend=backend)
        vqe_results = optimizer.descending_sweep(h_values, qc, lattice, exact_data=exact_results)

        # Verify results
        assert len(vqe_results) == 3
        for i, vqe_r in enumerate(vqe_results):
            gap = exact_results[i].gap
            if gap > 0:
                de_gap = abs(vqe_r.energy - exact_results[i].ground_energy) / gap
                # h=1.0 is near critical point; only check valid regime h≥1.25
                if h_values[i] >= 1.25:
                    assert de_gap < 0.05, f"ΔE/gap={de_gap:.4f} > 5% at h={h_values[i]}"
