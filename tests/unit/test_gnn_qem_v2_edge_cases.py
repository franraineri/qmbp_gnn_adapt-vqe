"""Edge-case tests for GNN-QEM V2 — focuses on inputs that could break the model.

Tests conditions that are likely in production:
- Empty/missing calibration data
- Single-node graphs (N=1)
- Disconnected graphs (no edges)
- Very large N (memory/performance)
- NaN/Inf in inputs
- Mismatched edge_index and gate_errors_2q lengths
- Zero-shot (untrained model) behavior
- Batch of mixed-size graphs
"""

import numpy as np
import pytest
import torch

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfigV2,
    GNNQEMCorrectorV2,
    QEMSampleV2,
    build_qem_graph_v2,
    correct_energy_v2,
    save_qem_samples_v2,
    load_qem_samples_v2,
    save_qem_v2_checkpoint,
    load_qem_v2_checkpoint,
    train_gnn_qem_v2,
)


def _sample(n_qubits=6, **overrides) -> QEMSampleV2:
    """Factory for test samples with sensible defaults."""
    n_edges = max(n_qubits - 1, 0)
    src = list(range(n_edges)) + list(range(1, n_qubits))
    dst = list(range(1, n_qubits)) + list(range(n_edges))
    defaults = dict(
        noisy_energy=-4.5, exact_energy=-5.0, h_value=2.5,
        n_2q_gates=2 * n_edges, ces=0.12, topology="chain_1d",
        n_qubits=n_qubits,
        qubit_t1=[120.0] * n_qubits, qubit_t2=[80.0] * n_qubits,
        readout_errors=[0.015] * n_qubits,
        gate_errors_2q=[0.007] * len(src),
        edge_index=np.array([src, dst], dtype=int) if n_qubits > 1 else np.zeros((2, 0), dtype=int),
        gap=1.5, n_cx_per_qubit=[3.0] * n_qubits,
        qubit_degree=[2] * n_qubits, ces_2q=0.10, ces_readout=0.05,
    )
    defaults.update(overrides)
    return QEMSampleV2(**defaults)


def _model(hidden=32, heads=2, layers=2):
    """Small model for fast tests."""
    config = GNNQEMConfigV2(hidden_dim=hidden, n_heads=heads, n_layers=layers)
    return GNNQEMCorrectorV2(config)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Empty / Missing Calibration Data
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingCalibration:
    """Model should not crash when calibration lists are empty."""

    def test_empty_t1_t2(self):
        s = _sample(qubit_t1=[], qubit_t2=[])
        data = build_qem_graph_v2(s)
        assert data.x.shape == (6, 6)
        # Should use defaults (1.0 for T1, 0.8 for T2)
        assert torch.all(data.x[:, 0] == 1.0)  # T1 default

    def test_empty_gate_errors(self):
        s = _sample(gate_errors_2q=[])
        data = build_qem_graph_v2(s)
        # edge_attr should have default 0.005
        assert data.edge_attr.max().item() == pytest.approx(0.005, abs=1e-6)

    def test_empty_readout(self):
        s = _sample(readout_errors=[])
        data = build_qem_graph_v2(s)
        assert data.x.shape == (6, 6)

    def test_empty_cx_per_qubit(self):
        s = _sample(n_cx_per_qubit=[])
        data = build_qem_graph_v2(s)
        # Should default to 0.5
        assert data.x[:, 4].mean().item() == pytest.approx(0.5, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Extreme Graph Sizes
# ═══════════════════════════════════════════════════════════════════════════

class TestExtremeGraphSizes:
    """Model must handle very small and moderately large graphs."""

    def test_single_qubit_no_edges(self):
        """N=1: no edges, no message passing. Should still produce output."""
        s = _sample(n_qubits=1, n_2q_gates=0, gate_errors_2q=[],
                    qubit_t1=[100.0], qubit_t2=[80.0], readout_errors=[0.01],
                    n_cx_per_qubit=[0.0], qubit_degree=[0],
                    edge_index=np.zeros((2, 0), dtype=int))
        model = _model()
        result = correct_energy_v2(model, s, confidence_threshold=0.0)
        assert np.isfinite(result.corrected_energy)
        assert 0 <= result.confidence <= 1

    def test_two_qubits(self):
        """N=2: minimal graph with one edge."""
        s = _sample(n_qubits=2, gate_errors_2q=[0.01, 0.01],
                    edge_index=np.array([[0, 1], [1, 0]], dtype=int),
                    qubit_t1=[100.0, 110.0], qubit_t2=[80.0, 85.0],
                    readout_errors=[0.01, 0.02], n_cx_per_qubit=[1.0, 1.0],
                    qubit_degree=[1, 1])
        model = _model()
        result = correct_energy_v2(model, s, confidence_threshold=0.0)
        assert np.isfinite(result.corrected_energy)

    def test_large_n_50(self):
        """N=50: verify no OOM or shape issues."""
        s = _sample(n_qubits=50)
        model = _model()
        result = correct_energy_v2(model, s, confidence_threshold=0.0)
        assert np.isfinite(result.corrected_energy)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: NaN / Inf in Inputs
# ═══════════════════════════════════════════════════════════════════════════

class TestNaNInfInputs:
    """Model should handle (or gracefully reject) non-finite inputs."""

    def test_nan_in_noisy_energy(self):
        """NaN noisy energy → correction should still not crash."""
        s = _sample(noisy_energy=float("nan"))
        data = build_qem_graph_v2(s)
        # The graph builds fine (NaN propagates to y target)
        assert not torch.isfinite(data.y).all()  # y = exact - nan = nan

    def test_inf_in_t1(self):
        """Inf T1 → node feature should not propagate crash."""
        s = _sample(qubit_t1=[float("inf")] * 6)
        data = build_qem_graph_v2(s)
        # Inf/100 = Inf in node feature — model may produce NaN
        # This is acceptable: caller should validate inputs
        assert data.x.shape == (6, 6)

    def test_zero_gap(self):
        """Gap=0 is valid (gapless system). Context should be 0."""
        s = _sample(gap=0.0)
        data = build_qem_graph_v2(s)
        assert data.context[0, 5].item() == 0.0  # gap_norm = 0/10 = 0


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Mismatched Lengths
# ═══════════════════════════════════════════════════════════════════════════

class TestMismatchedLengths:
    """Inputs with inconsistent array lengths should not crash."""

    def test_fewer_t1_than_qubits(self):
        """T1 list shorter than n_qubits → padding should kick in."""
        s = _sample(n_qubits=6, qubit_t1=[100.0, 110.0])  # only 2 of 6
        data = build_qem_graph_v2(s)
        assert data.x.shape == (6, 6)
        # First 2 should reflect values, rest should be default (100/100=1.0)
        assert data.x[0, 0].item() == pytest.approx(1.0, abs=0.01)

    def test_more_gate_errors_than_edges(self):
        """Extra gate_errors_2q entries → should not crash (truncated)."""
        s = _sample(gate_errors_2q=[0.01] * 100)  # way more than edges
        data = build_qem_graph_v2(s)
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    def test_fewer_gate_errors_than_edges(self):
        """Fewer gate_errors_2q → remaining edges get default."""
        s = _sample(gate_errors_2q=[0.02])  # only 1 of many edges
        data = build_qem_graph_v2(s)
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Virtual Node with Batching
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchedInference:
    """Multiple graphs of different sizes in one batch."""

    def test_batch_mixed_sizes(self):
        """Batch graphs of N=4, N=6, N=10 together."""
        from torch_geometric.loader import DataLoader

        model = _model()
        model.eval()

        samples = [_sample(n_qubits=n) for n in [4, 6, 10]]
        graphs = [build_qem_graph_v2(s) for s in samples]

        loader = DataLoader(graphs, batch_size=3)
        batch = next(iter(loader))

        with torch.no_grad():
            delta_e, confidence = model(batch)

        assert delta_e.shape == (3, 1)
        assert confidence.shape == (3, 1)
        assert torch.all(torch.isfinite(delta_e))
        assert torch.all((confidence >= 0) & (confidence <= 1))


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Training with Minimal Data
# ═══════════════════════════════════════════════════════════════════════════

class TestMinimalTraining:
    """Training with bare minimum data should not crash."""

    def test_train_with_5_samples(self):
        """Minimum viable training: 5 train + 2 val."""
        model = _model()
        samples = [_sample(h_value=1.5 + i * 0.5) for i in range(7)]
        dataset = [build_qem_graph_v2(s) for s in samples]

        config = GNNQEMConfigV2(hidden_dim=32, n_heads=2, n_layers=2, epochs=10, patience=5)
        result = train_gnn_qem_v2(model, dataset[:5], dataset[5:], config)
        assert result.best_epoch >= 0
        assert np.isfinite(result.val_mae)

    def test_train_rejects_too_few_samples(self):
        """< 5 training samples should raise ValueError."""
        model = _model()
        samples = [_sample(h_value=h) for h in [2.0, 3.0, 4.0]]
        dataset = [build_qem_graph_v2(s) for s in samples]

        config = GNNQEMConfigV2(hidden_dim=32, n_heads=2, n_layers=2, epochs=5, patience=3)
        with pytest.raises(ValueError, match="≥5"):
            train_gnn_qem_v2(model, dataset[:2], dataset[2:], config)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Checkpoint Versioning
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpointVersioning:
    """V2 loader rejects V1 checkpoints and vice versa."""

    def test_v2_loader_rejects_v1_format(self, tmp_path):
        """load_qem_v2_checkpoint should reject a V1-style checkpoint."""
        from qmbp_simulation.predictors.gnn_qem import GNNQEMCorrector, GNNQEMConfig, save_qem_checkpoint
        v1_model = GNNQEMCorrector(GNNQEMConfig())
        v1_path = tmp_path / "v1.pt"
        save_qem_checkpoint(v1_model, v1_path)

        with pytest.raises(ValueError, match="Not a V2"):
            load_qem_v2_checkpoint(v1_path)

    def test_v2_checkpoint_contains_version(self, tmp_path):
        """V2 checkpoint must have version='2.0' marker."""
        model = _model()
        path = tmp_path / "v2.pt"
        save_qem_v2_checkpoint(model, path)

        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["version"] == "2.0"
        assert "config" in raw
        assert "state_dict" in raw


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case: Sample Persistence with Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestSamplePersistence:
    """Serialization edge cases."""

    def test_empty_sample_list(self, tmp_path):
        """Saving/loading empty list should work."""
        path = tmp_path / "empty.json"
        save_qem_samples_v2([], path)
        loaded = load_qem_samples_v2(path)
        assert loaded == []

    def test_sample_with_large_edge_index(self, tmp_path):
        """Large edge_index (N=50) serializes correctly."""
        s = _sample(n_qubits=50)
        path = tmp_path / "large.json"
        save_qem_samples_v2([s], path)
        loaded = load_qem_samples_v2(path)
        assert loaded[0].n_qubits == 50
        assert loaded[0].edge_index.shape[0] == 2

    def test_v1_data_loads_as_v2_with_defaults(self, tmp_path):
        """V1-format JSON (missing V2 fields) loads with defaults."""
        import json
        v1_data = [{
            "noisy_energy": -4.0, "exact_energy": -5.0, "h_value": 2.0,
            "n_2q_gates": 10, "ces": 0.1, "topology": "chain_1d", "n_qubits": 6,
            "qubit_t1": [100.0] * 6, "qubit_t2": [80.0] * 6,
            "readout_errors": [0.01] * 6, "gate_errors_2q": [0.005] * 5,
            "edge_index": [[0,1,2,3,4], [1,2,3,4,5]],
        }]
        path = tmp_path / "v1_format.json"
        with open(path, "w") as f:
            json.dump(v1_data, f)

        loaded = load_qem_samples_v2(path)
        assert len(loaded) == 1
        assert loaded[0].gap == 0.0  # default
        assert loaded[0].ces_2q == 0.1  # falls back to ces
        assert loaded[0].n_cx_per_qubit == []  # default
