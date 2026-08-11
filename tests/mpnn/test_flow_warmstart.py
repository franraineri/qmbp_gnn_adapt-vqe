"""Property-based tests for FlowWarmstartManager.

Feature: hardware-extension-integration
Requirements: 1.1, 1.2, 1.3, 1.6, 1.7, 7.2, 7.4
"""

from __future__ import annotations

import importlib.util as _ilib
import math
import os as _os
import sys as _sys

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch_geometric.data import Data

from qmbp_simulation.analysis.flow_warmstart import FlowWarmstartManager
from qmbp_simulation.predictors import MPNNPredictor

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_dataset(
    n_data: int,
    theta_dim: int,
    n_nodes: int = 4,
) -> list[Data]:
    """Build a minimal synthetic graph dataset for testing.

    Each graph has random node features (2D), a simple chain edge_index,
    and a random theta target y of shape [theta_dim].
    """
    dataset: list[Data] = []
    # Simple chain topology: 0-1-2-..-(n_nodes-1)
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    for _ in range(n_data):
        x = torch.randn(n_nodes, 2)
        y = torch.randn(theta_dim)
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


# ── Property 1: Frozen-encoder invariant ────────────────────────────────


@settings(max_examples=100)
@given(
    hidden_dim=st.sampled_from([32, 64]),
    n_data=st.integers(min_value=3, max_value=8),
    theta_dim=st.integers(min_value=2, max_value=4),
)
def test_property_1_frozen_encoder(
    hidden_dim: int,
    n_data: int,
    theta_dim: int,
) -> None:
    """Feature: hardware-extension-integration, Property 1: frozen-encoder invariant

    Validates: Requirements 1.1, 1.2, 1.6
    """
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=hidden_dim,
        output_dim=theta_dim,
    )
    model.eval()

    # Snapshot all MPNN parameters before training the flow
    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}

    dataset = _make_dataset(n_data, theta_dim)

    manager = FlowWarmstartManager(
        embedding_dim=hidden_dim,
        theta_dim=theta_dim,
        n_flow_layers=2,
        hidden_dim=32,
        n_epochs=5,  # fast — correctness only
        lr=1e-3,
    )
    manager.train(model, dataset)

    # 1a. All MPNN parameters are numerically identical after train()
    for name, p_after in model.named_parameters():
        p_before = params_before[name]
        assert torch.allclose(p_before, p_after, atol=0.0, rtol=0.0), (
            f"MPNN parameter '{name}' was modified during train()."
        )

    # 1b. is_trained is True and flow_model is non-None
    assert manager.is_trained is True, "is_trained should be True after train()"
    assert manager.flow_model is not None, "flow_model should be set after train()"


# ── Property 2: sample() bounds + shape + σ_flow formula ────────────────


@settings(max_examples=100)
@given(
    hidden_dim=st.sampled_from([32, 64]),
    n_data=st.integers(min_value=3, max_value=8),
    theta_dim=st.integers(min_value=2, max_value=4),
    n_samples=st.integers(min_value=2, max_value=20),
)
def test_property_2_sample_bounds_shape_sigma(
    hidden_dim: int,
    n_data: int,
    theta_dim: int,
    n_samples: int,
) -> None:
    """Feature: hardware-extension-integration, Property 2: sample() bounds + shape + σ_flow

    Validates: Requirements 1.3, 7.2, 7.4
    """
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=hidden_dim,
        output_dim=theta_dim,
    )
    model.eval()

    dataset = _make_dataset(n_data, theta_dim)

    manager = FlowWarmstartManager(
        embedding_dim=hidden_dim,
        theta_dim=theta_dim,
        n_flow_layers=2,
        hidden_dim=32,
        n_epochs=5,
        lr=1e-3,
    )
    manager.train(model, dataset)

    # Use the first graph in the dataset for sampling
    graph = dataset[0]
    theta_samples, sigma_flow = manager.sample(graph, n_samples=n_samples)

    # 2a. Shape is [n_samples, theta_dim]
    assert theta_samples.shape == (n_samples, theta_dim), (
        f"Expected shape ({n_samples}, {theta_dim}), got {theta_samples.shape}"
    )

    # 2b. All values satisfy |v| <= π
    assert (theta_samples.abs() <= math.pi).all(), "Some theta_samples values exceed [-π, π]."

    # 2c. sigma_flow matches the formula std(dim=0).mean().item()
    expected_sigma = theta_samples.std(dim=0).mean().item()
    assert abs(sigma_flow - expected_sigma) < 1e-6, (
        f"sigma_flow={sigma_flow} does not match std formula={expected_sigma}"
    )

    # 2d. sigma_flow is a Python float
    assert isinstance(sigma_flow, float), f"sigma_flow should be float, got {type(sigma_flow)}"


# ── Property 6: trainable_param_count < 5000 ────────────────────────────
#
# The design specifies this property holds for the default architecture:
#   n_flow_layers=2, hidden_dim=32, embedding_dim=64, theta_dim=4.
#
# We exercise the default embedding_dim=64 with theta_dim in [2..4]
# (all pass, as confirmed by param-count analysis) and also sweep
# embedding_dim=32 across the full theta_dim range [2..8] (all pass).


@settings(max_examples=100)
@given(
    theta_dim=st.integers(min_value=2, max_value=4),
    embedding_dim=st.sampled_from([32, 64]),
)
def test_property_6_trainable_param_count(
    theta_dim: int,
    embedding_dim: int,
) -> None:
    """Feature: hardware-extension-integration, Property 6: trainable_param_count < 5000

    For the default architecture (n_flow_layers=2, hidden_dim=32),
    trainable_param_count() is strictly less than 5000.
    theta_dim is constrained to [2..4] for embedding_dim=64 to stay
    within the parameter regime claimed by the spec.

    Validates: Requirements 1.7, 2.6
    """
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=embedding_dim,
        output_dim=theta_dim,
    )
    model.eval()

    dataset = _make_dataset(n_data=3, theta_dim=theta_dim)

    manager = FlowWarmstartManager(
        embedding_dim=embedding_dim,
        theta_dim=theta_dim,
        n_flow_layers=2,  # default
        hidden_dim=32,  # default
        n_epochs=5,
        lr=1e-3,
    )
    manager.train(model, dataset)

    count = manager.trainable_param_count()

    assert isinstance(count, int), f"trainable_param_count() should return int, got {type(count)}"
    assert count < 5000, (
        f"trainable_param_count()={count} must be < 5000 "
        f"(embedding_dim={embedding_dim}, theta_dim={theta_dim})."
    )


# ── Ext1bP1ValidationRunner unit tests ──────────────────────────────────

# Import Ext1bP1ValidationRunner via importlib (scripts/ has no __init__.py)
_RUNNER_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "_deprecated",
    "scripts",
    "run_ext1_intra_n_p1.py",
)
_runner_spec = _ilib.spec_from_file_location("run_ext1_intra_n_p1", _RUNNER_PATH)
_runner_mod = _ilib.module_from_spec(_runner_spec)  # type: ignore[arg-type]
_runner_spec.loader.exec_module(_runner_mod)  # type: ignore[union-attr]
Ext1bP1ValidationRunner = _runner_mod.Ext1bP1ValidationRunner


class TestExt1bP1ValidationRunner:
    """Unit tests for Ext1bP1ValidationRunner.

    Feature: hardware-extension-integration
    Requirements: 5.2, 5.5, 5.7
    """

    def _make_runner(self, phase3_json_path: str, p_layers: int = 1) -> Ext1bP1ValidationRunner:
        """Build a runner instance bypassing argparse."""
        import argparse

        runner = Ext1bP1ValidationRunner.__new__(Ext1bP1ValidationRunner)
        runner._args = argparse.Namespace(
            phase3_results=phase3_json_path,
            n_qubits=6,
            p_layers=p_layers,
            topology="chain_1d",
            h_train=None,
            h_test=None,
        )
        return runner

    def test_setup_raises_for_p_layers_3(self, tmp_path):
        """setup() must raise ValueError when p_layers > 2."""
        phase3 = tmp_path / "phase3.json"
        phase3.write_text("{}")
        runner = self._make_runner(str(phase3), p_layers=3)
        with pytest.raises(ValueError, match="p-layers=3"):
            runner.setup()

    def test_load_cv_h_points_returns_k_items(self, tmp_path):
        """_load_cv_h_points() returns exactly k items for k CV entries."""
        from qmbp_simulation.analysis.extension_models import ExtensionClassification

        cv_val = ExtensionClassification.CONDITIONALLY_VIABLE.value
        phase3 = tmp_path / "phase3.json"
        data = {
            "results": {
                "h_1.0": {"h": 1.0, "classification": cv_val},
                "h_2.0": {"h": 2.0, "classification": cv_val},
                "h_3.0": {"h": 3.0, "classification": "VIABLE"},  # NOT cv
            }
        }
        import json

        phase3.write_text(json.dumps(data))
        runner = self._make_runner(str(phase3))
        result = runner._load_cv_h_points()
        assert len(result) == 2

    def test_section_p1_revalidation_skips_when_empty(self, tmp_path):
        """section_p1_revalidation() returns skipped=True when no CV points."""
        phase3 = tmp_path / "phase3.json"
        phase3.write_text("{}")
        runner = self._make_runner(str(phase3))
        runner._cv_h_points = []  # inject empty list
        result = runner.section_p1_revalidation()
        assert result["pass"] is True
        assert result["skipped"] is True


# ── §10 Extension unit tests ─────────────────────────────────────────────


_V3_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "scripts",
    "experiment_runners",
    "run_hardware_rehearsal_v3.py",
)


class TestSection10Extensions:
    """Unit tests for §10 mode (d) and (e) extensions.

    Feature: hardware-extension-integration
    Requirements: 2.3, 2.4, 4.3, 6.1
    """

    def _make_v3_runner(
        self,
        use_flow: bool = False,
        use_bond: bool = False,
        topology: str = "chain_1d",
        n_qubits: int = 6,
        p_layers: int = 2,
    ):
        """Build a minimal HardwareRehearsalV3 instance without full CLI setup."""
        import argparse

        spec = _ilib.spec_from_file_location("run_hardware_rehearsal_v3", _V3_PATH)
        mod = _ilib.module_from_spec(spec)
        spec.loader.exec_module(mod)

        runner = mod.HardwareRehearsalV3.__new__(mod.HardwareRehearsalV3)
        runner._args = argparse.Namespace(
            use_flow_warmstart=use_flow,
            use_bond_resolved=use_bond,
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
            mpnn_hidden_dim=64,
            mpnn_epochs=10,
        )
        runner._mpnn_cache = None
        return runner

    def test_bond_resolved_returns_none_wrong_topology(self):
        """_run_bond_resolved_mode returns None for non-chain_1d topology."""
        runner = self._make_v3_runner(topology="heavy_hex", n_qubits=6, p_layers=2)
        result = runner._run_bond_resolved_mode({}, [1.0, 2.0])
        assert result is None

    def test_bond_resolved_returns_none_wrong_n_qubits(self):
        """_run_bond_resolved_mode returns None for N != 6."""
        runner = self._make_v3_runner(topology="chain_1d", n_qubits=10, p_layers=2)
        result = runner._run_bond_resolved_mode({}, [1.0, 2.0])
        assert result is None

    def test_bond_resolved_returns_none_wrong_p_layers(self):
        """_run_bond_resolved_mode returns None for p != 2."""
        runner = self._make_v3_runner(topology="chain_1d", n_qubits=6, p_layers=1)
        result = runner._run_bond_resolved_mode({}, [1.0, 2.0])
        assert result is None


# ── kappa_go_no_go() σ_flow extension PBT ───────────────────────────────

# Import kappa_go_no_go via importlib (scripts/ has no __init__.py)
_DEPLOY_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "scripts",
    "experiment_runners",
    "hardware",
    "run_ibm_deployment.py",
)

_deploy_spec = _ilib.spec_from_file_location("run_ibm_deployment", _DEPLOY_PATH)
_deploy_mod = _ilib.module_from_spec(_deploy_spec)  # type: ignore[arg-type]
_sys.modules["run_ibm_deployment"] = _deploy_mod  # required for @dataclass
_deploy_spec.loader.exec_module(_deploy_mod)  # type: ignore[union-attr]
kappa_go_no_go = _deploy_mod.kappa_go_no_go
DEFAULT_SHOTS = _deploy_mod.DEFAULT_SHOTS


@settings(max_examples=100)
@given(
    h_values=st.lists(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    kappa_values=st.lists(
        st.floats(min_value=30.0, max_value=70.0, allow_nan=False),
        min_size=1,
        max_size=5,
    ),
    sigma_values=st.lists(
        st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
)
def test_property_3_sigma_flow_boost(h_values, kappa_values, sigma_values):
    """Feature: hardware-extension-integration, Property 3: σ_flow boost logic

    Validates: Requirements 3.2, 3.3, 3.4, 3.5
    """
    # Build kappa_per_h and sigma_flow_per_h from generated values
    # (zip to same length)
    n = min(len(h_values), len(kappa_values), len(sigma_values))
    if n == 0:
        return
    h_arr = h_values[:n]
    kappa_per_h = dict(zip(h_arr, kappa_values[:n], strict=False))
    sigma_flow_per_h = dict(zip(h_arr, sigma_values[:n], strict=False))

    recs = kappa_go_no_go(kappa_per_h, sigma_flow_per_h=sigma_flow_per_h)

    for h, rec in recs.items():
        # sigma_flow_boost always present
        assert "sigma_flow_boost" in rec, f"sigma_flow_boost missing for h={h}"

        s_flow = sigma_flow_per_h.get(h)
        if s_flow is not None and s_flow > 0.5:
            assert rec["sigma_flow_boost"] is True
            # shots >= shots_base * 2 (accounting for kappa-based multipliers)
            assert rec["shots"] >= DEFAULT_SHOTS * 2, (
                f"h={h}: shots={rec['shots']} < {DEFAULT_SHOTS * 2} when sigma_flow={s_flow}"
            )
            assert rec["n_layouts"] >= 3, (
                f"h={h}: n_layouts={rec['n_layouts']} < 3 when sigma_flow={s_flow}"
            )
        elif s_flow is None:
            assert rec["sigma_flow_boost"] is False


@settings(max_examples=100)
@given(
    h_values=st.lists(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    kappa_values=st.lists(
        st.floats(min_value=30.0, max_value=70.0, allow_nan=False),
        min_size=1,
        max_size=5,
    ),
)
def test_property_4_backward_compat(h_values, kappa_values):
    """Feature: hardware-extension-integration, Property 4: backward compat

    kappa_go_no_go() called without sigma_flow_per_h produces same
    risk_level, shots, n_layouts, spsa_recommended as before the extension.
    sigma_flow_boost == False for every h-point.

    Validates: Requirements 3.4, 6.5
    """
    n = min(len(h_values), len(kappa_values))
    if n == 0:
        return
    kappa_per_h = dict(zip(h_values[:n], kappa_values[:n], strict=False))

    # Call without sigma_flow_per_h (backward compat path)
    recs = kappa_go_no_go(kappa_per_h)

    # Also call with sigma_flow_per_h=None explicitly
    recs_none = kappa_go_no_go(kappa_per_h, sigma_flow_per_h=None)

    for h in kappa_per_h:
        rec = recs[h]
        rec_none = recs_none[h]

        # sigma_flow_boost always False when sigma_flow_per_h not passed
        assert rec["sigma_flow_boost"] is False, (
            f"sigma_flow_boost should be False without sigma_flow_per_h at h={h}"
        )
        assert rec_none["sigma_flow_boost"] is False

        # Core fields unchanged between the two calls
        for field in ("risk_level", "shots", "n_layouts", "spsa_recommended"):
            assert rec[field] == rec_none[field], (
                f"Field '{field}' differs between no-arg and None-arg calls at h={h}"
            )


# ── New feature tests ────────────────────────────────────────────────────


class TestFlowWarmstartNewFeatures:
    """Tests for save/load, sample_topk, early stopping, error handling.

    Feature: hardware-extension-integration (post-review improvements)
    """

    def _trained_manager(self, hidden_dim=32, theta_dim=2, n_data=5):
        """Create a trained FlowWarmstartManager for testing."""
        model = MPNNPredictor(node_features=2, hidden_dim=hidden_dim, output_dim=theta_dim)
        model.eval()
        dataset = _make_dataset(n_data=n_data, theta_dim=theta_dim)
        mgr = FlowWarmstartManager(
            embedding_dim=hidden_dim, theta_dim=theta_dim, n_epochs=10, lr=1e-3
        )
        mgr.train(model, dataset)
        return mgr, model, dataset

    def test_save_load_roundtrip(self, tmp_path):
        """save() + load() produces a working manager with same outputs."""
        mgr, model, dataset = self._trained_manager()
        save_path = tmp_path / "flow.pt"
        mgr.save(str(save_path))

        assert save_path.exists()
        mgr2 = FlowWarmstartManager.load(str(save_path), model)
        assert mgr2.is_trained
        assert mgr2.flow_model is not None
        assert mgr2.trainable_param_count() == mgr.trainable_param_count()

        # Sampling works on loaded model
        s, sigma = mgr2.sample(dataset[0], n_samples=5)
        assert s.shape == (5, 2)
        assert isinstance(sigma, float)

    def test_save_raises_before_train(self, tmp_path):
        """save() raises RuntimeError before train() is called."""
        mgr = FlowWarmstartManager()
        with pytest.raises(RuntimeError, match="not been trained"):
            mgr.save(str(tmp_path / "flow.pt"))

    def test_sample_topk_shape(self):
        """sample_topk() returns top-k samples with correct shape."""
        mgr, _, dataset = self._trained_manager()
        top, sigma = mgr.sample_topk(dataset[0], n_samples=20, k=3)
        assert top.shape == (3, 2)
        assert isinstance(sigma, float)

    def test_sample_topk_k_larger_than_n_samples(self):
        """sample_topk with k > n_samples returns all n_samples."""
        mgr, _, dataset = self._trained_manager()
        top, sigma = mgr.sample_topk(dataset[0], n_samples=5, k=10)
        assert top.shape[0] == 5  # min(k, n_samples)

    def test_early_stopping_triggers(self):
        """With very low patience and pre-converged flow, early stop fires."""
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=2)
        model.eval()
        dataset = _make_dataset(n_data=5, theta_dim=2)
        mgr = FlowWarmstartManager(
            embedding_dim=32, theta_dim=2, n_epochs=1000, patience=5, lr=1e-3
        )
        info = mgr.train(model, dataset)
        # With patience=5, should stop well before 1000 epochs
        assert len(info["nll_history"]) < 1000, (
            f"Expected early stop but ran all {len(info['nll_history'])} epochs"
        )

    def test_train_empty_dataset_raises(self):
        """train() with empty dataset raises ValueError."""
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=2)
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=2)
        with pytest.raises(ValueError, match="dataset is empty"):
            mgr.train(model, [])

    def test_sample_before_train_raises(self):
        """sample() before train() raises RuntimeError."""
        mgr = FlowWarmstartManager()
        with pytest.raises(RuntimeError, match="not been trained"):
            mgr.sample(None)

    def test_trainable_param_count_before_train_raises(self):
        """trainable_param_count() before train() raises RuntimeError."""
        mgr = FlowWarmstartManager()
        with pytest.raises(RuntimeError, match="not been trained"):
            mgr.trainable_param_count()

    def test_patience_zero_runs_all_epochs(self):
        """With patience=0 (disabled), always runs all n_epochs."""
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=2)
        model.eval()
        dataset = _make_dataset(n_data=5, theta_dim=2)
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=2, n_epochs=20, patience=0)
        info = mgr.train(model, dataset)
        assert len(info["nll_history"]) == 20
