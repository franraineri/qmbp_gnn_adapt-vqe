"""End-to-end pipeline integration test: Phase 1 → 2 → 3 → 4."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
    make_lattice,
)
from qmbp_simulation.optimizers import VQEOptimizer
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)
from qmbp_simulation.solvers import ClassicalSolver

pytestmark = pytest.mark.integration


class TestPipelineE2E:
    """End-to-end pipeline test: Phase 1 → 2 → 3 → 4."""

    def test_full_pipeline_n4_p1(self):
        """Run all 4 phases for N=4, p=1 and verify ΔE/gap < 10%."""
        import torch

        torch.manual_seed(42)
        np.random.seed(42)

        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()

        # Use 5 h-points for training, 1 for test
        h_train = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
        h_test = np.array([1.6])
        qc, theta = hva.create(4, 1, lattice)

        # ── Phase 1: Exact solutions ──
        exact_results = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_results.append(solver.solve(H, lat_h))

        # ── Phase 2: VQE sweep ──
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=100)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)
        vqe_results = optimizer.descending_sweep(h_train, qc, lattice, exact_data=exact_results)

        # Collect arrays for Phase 3
        theta_opt = np.array([r.theta_opt for r in vqe_results])
        e_exact = np.array([r.ground_energy for r in exact_results])
        fidelities = np.array([r.fidelity for r in vqe_results])

        # ── Phase 3: MPNN training ──
        dataset = build_graph_dataset(lattice, h_train, theta_opt, e_exact, fidelities)
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=2)
        train_mpnn(model, dataset, n_epochs=500, lr=1e-3, patience=100, seed=42)

        # ── Phase 4: Deploy at test point ──
        from torch_geometric.data import Data

        lat_test = make_lattice("chain_1d", 4, J=1.0, h=float(h_test[0]))
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        # Build graph for prediction
        h_feat = np.full(4, float(h_test[0]))
        coord = lattice.coordination_numbers.astype(float)
        edge_index_np, _ = builder.build_graph_data(lattice)
        x = torch.tensor(np.stack([h_feat, coord], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        test_data = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            theta_pred = model(test_data).squeeze().numpy()

        # Evaluate predicted parameters
        e_pred = backend.evaluate(qc, H_test, theta_pred)
        gap = exact_test.gap
        de_gap = abs(e_pred - exact_test.ground_energy) / gap if gap > 0 else 0

        assert de_gap < 0.10, f"ΔE/gap = {de_gap:.4f} exceeds 10% threshold at h={h_test[0]}"
