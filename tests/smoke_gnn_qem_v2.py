"""Quick smoke test for GNN-QEM V2 architecture."""
import numpy as np
import torch

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMConfigV2,
    GNNQEMCorrector,
    GNNQEMCorrectorV2,
    QEMSampleV2,
    build_qem_graph_v2,
    correct_energy_v2,
    save_qem_samples_v2,
    load_qem_samples_v2,
    save_qem_v2_checkpoint,
    load_qem_v2_checkpoint,
    QEMSample,
)


def _make_sample_v2(n_qubits: int = 6) -> QEMSampleV2:
    """Create a realistic V2 sample for testing."""
    n_edges = n_qubits - 1  # chain
    src = list(range(n_edges)) + list(range(1, n_qubits))
    dst = list(range(1, n_qubits)) + list(range(n_edges))
    return QEMSampleV2(
        noisy_energy=-4.5,
        exact_energy=-5.0,
        h_value=2.5,
        n_2q_gates=2 * n_edges,
        ces=0.12,
        topology="chain_1d",
        n_qubits=n_qubits,
        qubit_t1=[120.0] * n_qubits,
        qubit_t2=[80.0] * n_qubits,
        readout_errors=[0.015] * n_qubits,
        gate_errors_2q=[0.007] * len(src),
        edge_index=np.array([src, dst], dtype=int),
        gap=1.5,
        n_cx_per_qubit=[2.0] + [4.0] * (n_qubits - 2) + [2.0],
        qubit_degree=[1] + [2] * (n_qubits - 2) + [1],
        ces_2q=0.10,
        ces_readout=0.05,
    )


def test_v2_model_instantiation():
    """V2 model instantiates with correct parameter count."""
    config = GNNQEMConfigV2(hidden_dim=64, n_heads=2, n_layers=2)
    model = GNNQEMCorrectorV2(config)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    print(f"✓ Model: {n_params:,} parameters")


def test_v2_graph_construction():
    """V2 graph has correct shapes and features."""
    sample = _make_sample_v2(6)
    data = build_qem_graph_v2(sample, augment=False)
    assert data.x.shape == (6, 6), f"Expected (6,6), got {data.x.shape}"
    assert data.edge_attr.shape[1] == 1
    assert data.context.shape == (1, 7)
    assert data.y.shape == (1, 1)
    print(f"✓ Graph shapes OK: x={data.x.shape}, edge_attr={data.edge_attr.shape}")


def test_v2_forward_pass():
    """V2 model produces valid output shapes."""
    config = GNNQEMConfigV2(hidden_dim=64, n_heads=2, n_layers=2)
    model = GNNQEMCorrectorV2(config)
    sample = _make_sample_v2(6)
    data = build_qem_graph_v2(sample)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        delta_e, confidence = model(data)
    assert delta_e.shape == (1, 1)
    assert 0 <= confidence.item() <= 1
    print(f"✓ Forward: ΔE={delta_e.item():.6f}, conf={confidence.item():.4f}")


def test_v2_augmentation():
    """Calibration augmentation modifies node/edge features."""
    sample = _make_sample_v2(6)
    data_clean = build_qem_graph_v2(sample, augment=False)
    data_aug = build_qem_graph_v2(sample, augment=True, augment_scale=0.3)
    assert not torch.allclose(data_clean.x, data_aug.x)
    print("✓ Augmentation changes features")


def test_v2_correction_api():
    """correct_energy_v2 returns same type as V1."""
    config = GNNQEMConfigV2(hidden_dim=64, n_heads=2, n_layers=2)
    model = GNNQEMCorrectorV2(config)
    sample = _make_sample_v2(6)
    result = correct_energy_v2(model, sample, confidence_threshold=0.0)
    assert result.correction_applied is True
    assert hasattr(result, "corrected_energy")
    assert hasattr(result, "confidence")
    print(f"✓ Correction API: E_corr={result.corrected_energy:.4f}")


def test_v2_checkpoint_roundtrip(tmp_path=None):
    """Save/load V2 checkpoint preserves model weights."""
    from pathlib import Path
    import tempfile

    config = GNNQEMConfigV2(hidden_dim=64, n_heads=2, n_layers=2)
    model = GNNQEMCorrectorV2(config)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_v2.pt"
        save_qem_v2_checkpoint(model, path)
        loaded_model, _, _ = load_qem_v2_checkpoint(path)

    # Verify weights match
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), loaded_model.state_dict().items()
    ):
        assert k1 == k2
        assert torch.allclose(v1, v2)
    print("✓ Checkpoint roundtrip preserves weights")


def test_v2_sample_persistence():
    """Save/load V2 samples roundtrip."""
    from pathlib import Path
    import tempfile

    samples = [_make_sample_v2(6), _make_sample_v2(10)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "samples.json"
        save_qem_samples_v2(samples, path)
        loaded = load_qem_samples_v2(path)

    assert len(loaded) == 2
    assert loaded[0].gap == 1.5
    assert loaded[1].n_qubits == 10
    assert loaded[0].n_cx_per_qubit == [2.0, 4.0, 4.0, 4.0, 4.0, 2.0]
    print("✓ V2 sample persistence roundtrip")


def test_v1_backward_compat():
    """V1 model still instantiates correctly alongside V2."""
    v1 = GNNQEMCorrector(GNNQEMConfig())
    v1_params = sum(p.numel() for p in v1.parameters())
    assert v1_params > 0
    print(f"✓ V1 backward compat: {v1_params:,} params")


def test_v2_variable_n():
    """V2 handles different graph sizes (N=6, N=10, N=20)."""
    config = GNNQEMConfigV2(hidden_dim=64, n_heads=2, n_layers=2)
    model = GNNQEMCorrectorV2(config)
    model.eval()

    for n in [6, 10, 20]:
        sample = _make_sample_v2(n)
        data = build_qem_graph_v2(sample)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            delta_e, conf = model(data)
        assert delta_e.shape == (1, 1)
    print("✓ Variable N (6, 10, 20) all produce valid output")


def test_v2_smoke():
    """Run all V2 smoke tests."""
    test_v2_model_instantiation()
    test_v2_graph_construction()
    test_v2_forward_pass()
    test_v2_augmentation()
    test_v2_correction_api()
    test_v2_checkpoint_roundtrip()
    test_v2_sample_persistence()
    test_v1_backward_compat()
    test_v2_variable_n()
    print("\n🎉 All V2 smoke tests passed!")


if __name__ == "__main__":
    test_v2_smoke()
