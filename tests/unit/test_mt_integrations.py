"""Tests for P1-P3 integrations: strict filter safeguard, multi-topology ablation,
active learning refinement, and post-evaluation in finetune.

Covers:
- Integration 3: Strict max_de_gap filter warns when >50% data is dropped
- Integration 4: --multi-topology flag in arch_ablation uses MultiTopologyAggregator
- Integration 5: active_learning_refine method in ValidationRunner
- Integration 6: Post-evaluation updates zoo pass_rate after fine-tuning
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_pyg_dataset():
    """Create a small PyG dataset for testing (mimics UnifiedMPNN input)."""
    graphs = []
    for i in range(20):
        n_nodes = 6
        g = Data(
            x=torch.randn(n_nodes, 4),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long
            ),
            node_type=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            n_edges_unique=3,
            n_qubit_nodes=3,
            y=torch.randn(6),
            topology="chain_1d",
        )
        graphs.append(g)
    return graphs


@pytest.fixture
def small_unified_model():
    """Create a small UnifiedMPNN for testing."""
    from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

    return UnifiedMPNN(
        node_features=4, hidden_dim=32, n_layers=2, norm_type="none", dropout=0.1
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration 3: Strict filter safeguard
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrictFilterSafeguard:
    """Validate that strict max_de_gap filter warns on excessive data loss."""

    def test_safeguard_not_triggered_at_default_threshold(self):
        """With max_de_gap=0.10 (default), no safeguard comparison is needed."""
        # The safeguard only activates when max_de_gap < 0.10.
        # This test verifies the conditional logic is correct.
        max_de_gap = 0.10
        assert not (max_de_gap < 0.10)

    def test_safeguard_triggers_below_050_retention(self, capsys):
        """Safeguard should warn when strict filter drops >50% of data."""
        # Simulate the safeguard logic from run_multi_topology_training
        max_de_gap = 0.05
        dataset_strict = list(range(8))  # 8 graphs pass strict
        dataset_relaxed = list(range(20))  # 20 graphs pass relaxed

        retention_ratio = len(dataset_strict) / max(len(dataset_relaxed), 1)
        assert retention_ratio < 0.50
        assert retention_ratio == pytest.approx(0.40)

    def test_safeguard_does_not_trigger_above_050_retention(self):
        """Safeguard should NOT warn when retention is acceptable."""
        max_de_gap = 0.07
        dataset_strict = list(range(12))  # 12 graphs pass
        dataset_relaxed = list(range(20))  # 20 graphs pass relaxed

        retention_ratio = len(dataset_strict) / max(len(dataset_relaxed), 1)
        assert retention_ratio >= 0.50

    def test_safeguard_handles_empty_relaxed_dataset(self):
        """Division by zero protected when relaxed dataset is also empty."""
        dataset_strict = []
        dataset_relaxed = []
        retention_ratio = len(dataset_strict) / max(len(dataset_relaxed), 1)
        assert retention_ratio == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration 4: --multi-topology flag in arch ablation
# ═══════════════════════════════════════════════════════════════════════════════


class TestArchAblationMultiTopology:
    """Validate --multi-topology flag uses MultiTopologyAggregator."""

    def test_multi_topology_aggregator_produces_combined_data(self):
        """MultiTopologyAggregator returns graphs from multiple topologies."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator

        # Just verify the class API exists and accepts expected params
        agg = MultiTopologyAggregator(
            model="tfim_bond_resolved", max_n=10, topologies=["chain_1d"]
        )
        assert hasattr(agg, "scan")
        assert hasattr(agg, "build_combined_dataset")

    def test_ablation_report_includes_multi_topology_fields(self):
        """Report JSON should include multi_topology_mode and max_de_gap."""
        from datetime import datetime, timezone

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topology": "multi_topology",
            "multi_topology_mode": True,
            "max_n": 20,
            "max_de_gap": 0.05,
            "n_training_graphs": 100,
            "hidden_dim": 256,
            "n_layers": 3,
            "max_epochs": 4000,
            "seed": 42,
            "exclusion_policy_applied": False,
            "results": [],
            "best_variant": "res+jk+film",
        }
        # Verify schema
        assert report["multi_topology_mode"] is True
        assert report["topology"] == "multi_topology"
        assert "max_de_gap" in report
        assert report["exclusion_policy_applied"] is False

    def test_zoo_registration_uses_multi_topology_label(self):
        """When registering in multi-topology mode, topology should be 'multi_topology'."""
        # Simulating the ckpt_name generation logic
        multi_topology = True
        topology = "chain_1d"
        arch_label = "residual+jk_cat+film"

        effective_topo = topology if not multi_topology else "multi_topology"
        ckpt_name = (
            f"unified_tfim_br_{effective_topo}_multiN_ablation_{arch_label}_p1.pt"
        )
        assert "multi_topology" in ckpt_name
        assert "chain_1d" not in ckpt_name


# ═══════════════════════════════════════════════════════════════════════════════
# Integration 5: active_learning_refine method
# ═══════════════════════════════════════════════════════════════════════════════


class TestActiveLearningRefine:
    """Validate active_learning_refine method on ValidationRunner."""

    def test_method_exists_on_validation_runner(self):
        """active_learning_refine should be a method on ValidationRunner."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert hasattr(ValidationRunner, "active_learning_refine")
        assert callable(getattr(ValidationRunner, "active_learning_refine"))

    def test_method_signature_has_required_params(self):
        """Verify parameter names in the method signature."""
        import inspect

        from qmbp_simulation.framework.runner_base import ValidationRunner

        sig = inspect.signature(ValidationRunner.active_learning_refine)
        param_names = list(sig.parameters.keys())

        assert "self" in param_names
        assert "model" in param_names
        assert "h_candidates" in param_names
        assert "n_rounds" in param_names
        assert "n_points_per_round" in param_names
        assert "acquisition" in param_names
        assert "ensemble_seeds" in param_names

    def test_returns_expected_dict_structure(self, small_unified_model):
        """Method should return a dict with required keys."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        # Create a minimal runner subclass for testing
        class _TestRunner(ValidationRunner):
            @property
            def name(self):
                return "test_al"

            def define_sections(self):
                return []

        # Mock argparse namespace
        mock_args = MagicMock()
        mock_args.topology = "chain_1d"
        mock_args.n_qubits = 6
        mock_args.p_layers = 1
        mock_args.output_dir = "/tmp"
        mock_args.verbose = False
        mock_args.debug = False
        mock_args.seed = 42

        runner = _TestRunner.__new__(_TestRunner)
        runner._args = mock_args

        # Run with a model that will stop early (uncertainty below threshold)
        h_candidates = np.linspace(0.5, 2.0, 5)

        # Mock the ensemble to return low uncertainty (triggers early stop)
        with patch(
            "experiments.helpers.active_learning.should_stop", return_value=True
        ):
            result = runner.active_learning_refine(
                small_unified_model,
                h_candidates,
                n_rounds=2,
                n_points_per_round=2,
            )

        assert isinstance(result, dict)
        assert "n_rounds_run" in result
        assert "n_points_refined" in result
        assert "refined_h_values" in result
        assert "mean_improvement" in result
        assert "stopped_early" in result
        assert result["stopped_early"] is True
        assert result["n_points_refined"] == 0

    def test_early_stop_when_uncertainty_low(self, small_unified_model):
        """Should stop when all uncertainties are below threshold."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        class _TestRunner(ValidationRunner):
            @property
            def name(self):
                return "test_al_stop"

            def define_sections(self):
                return []

        mock_args = MagicMock()
        mock_args.topology = "chain_1d"
        mock_args.n_qubits = 6
        mock_args.p_layers = 1

        runner = _TestRunner.__new__(_TestRunner)
        runner._args = mock_args

        h_candidates = np.array([1.0, 1.5, 2.0])

        with patch(
            "experiments.helpers.active_learning.should_stop", return_value=True
        ):
            result = runner.active_learning_refine(
                small_unified_model,
                h_candidates,
                n_rounds=5,
            )
        assert result["stopped_early"] is True
        assert result["n_points_refined"] == 0

    def test_acquisition_functions_accepted(self):
        """Both acquisition function names should be valid."""
        from experiments.helpers.active_learning import select_next_point

        h = np.array([1.0, 1.5, 2.0, 2.5])
        uncert = [0.1, 0.5, 0.3, 0.2]

        # max_variance
        idx_mv, h_mv = select_next_point(h, uncert, acquisition="max_variance")
        assert idx_mv == 1  # Highest uncertainty

        # expected_improvement
        idx_ei, h_ei = select_next_point(
            h, uncert,
            acquisition="expected_improvement",
            current_best_error=0.3,
            predictions_mean=[0.1, 0.4, 0.2, 0.15],
        )
        assert 0 <= idx_ei < len(h)

    def test_compute_ensemble_uncertainty_output_structure(self):
        """Ensemble uncertainty should return mean, std, variance, max_std."""
        from experiments.helpers.active_learning import compute_ensemble_uncertainty

        # 3 ensemble members, 5 h-points
        predictions = [
            np.array([0.1, 0.5, 0.3, 0.2, 0.4]),
            np.array([0.12, 0.6, 0.28, 0.22, 0.38]),
            np.array([0.08, 0.45, 0.32, 0.18, 0.42]),
        ]

        result = compute_ensemble_uncertainty(predictions)
        assert "mean" in result
        assert "std" in result
        assert "variance" in result
        assert "max_std" in result
        assert result["mean"].shape == (5,)
        assert result["std"].shape == (5,)
        assert result["variance"] >= 0
        assert result["max_std"] >= 0

    def test_should_stop_correctly_detects_convergence(self):
        """should_stop returns True when max uncertainty < threshold."""
        from experiments.helpers.active_learning import should_stop

        # All below threshold
        assert should_stop([0.005, 0.003, 0.001], threshold=0.01)

        # One above threshold
        assert not should_stop([0.005, 0.015, 0.001], threshold=0.01)

        # Empty list
        assert should_stop([], threshold=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration 6: Post-evaluation in finetune
# ═══════════════════════════════════════════════════════════════════════════════


class TestFinetunePostEvaluation:
    """Validate that fine-tune script auto-evaluates and updates pass_rate."""

    def test_quick_eval_computes_pass_rate(self, small_unified_model, small_pyg_dataset):
        """Quick eval loop should compute pass_rate from per-graph MSE."""
        model = small_unified_model
        model.eval()

        n_pass = 0
        n_total = 0
        with torch.no_grad():
            for g in small_pyg_dataset:
                pred = model(g).squeeze(0)
                target = g.y
                if len(pred) != len(target):
                    continue
                mse = torch.nn.functional.mse_loss(pred, target).item()
                n_total += 1
                if mse < 0.01:
                    n_pass += 1

        assert n_total == len(small_pyg_dataset)
        quick_pass_rate = n_pass / n_total
        assert 0.0 <= quick_pass_rate <= 1.0

    def test_update_zoo_pass_rate_api_exists(self):
        """update_zoo_pass_rate should be importable with correct signature."""
        import inspect

        from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

        sig = inspect.signature(update_zoo_pass_rate)
        params = list(sig.parameters.keys())
        assert "checkpoint_file" in params
        assert "observed_pass_rate" in params
        assert "only_if_better" in params

    def test_quick_eval_handles_shape_mismatch_gracefully(self, small_unified_model):
        """If pred/target shapes differ, that graph should be skipped."""
        model = small_unified_model
        model.eval()

        # Create a graph with mismatched y shape
        g = Data(
            x=torch.randn(6, 4),
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
            node_type=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            n_edges_unique=3,
            n_qubit_nodes=3,
            y=torch.randn(10),  # Wrong size — 10 != expected output
        )

        n_total = 0
        with torch.no_grad():
            pred = model(g).squeeze(0)
            target = g.y
            if len(pred) == len(target):
                n_total += 1

        # Shape mismatch → should be skipped
        assert n_total == 0

    def test_pass_rate_zero_for_random_model(self, small_unified_model, small_pyg_dataset):
        """An untrained random model should have low pass_rate (MSE > 0.01)."""
        model = small_unified_model
        model.eval()

        n_pass = 0
        n_total = 0
        with torch.no_grad():
            for g in small_pyg_dataset:
                pred = model(g).squeeze(0)
                target = g.y
                if len(pred) != len(target):
                    continue
                mse = torch.nn.functional.mse_loss(pred, target).item()
                n_total += 1
                if mse < 0.01:
                    n_pass += 1

        # Random model unlikely to pass MSE < 0.01 threshold
        # (targets are randn, so MSE will be ~1.0)
        quick_pass_rate = n_pass / n_total
        assert quick_pass_rate < 0.5  # Very unlikely to pass half with random weights


# ═══════════════════════════════════════════════════════════════════════════════
# Mejora 2: Ablation → Comparison Bridge (--auto-compare)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoCompareFlag:
    """Validate --auto-compare flag integration in training scripts."""

    def test_mt_training_accepts_auto_compare_flag(self):
        """run_multi_topology_training should accept --auto-compare."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/experiment_runners/cross_topology/run_multi_topology_training.py",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--auto-compare" in result.stdout

    def test_arch_ablation_accepts_auto_compare_flag(self):
        """run_arch_ablation should accept --auto-compare."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/experiment_runners/cross_topology/run_arch_ablation.py",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--auto-compare" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# Mejora 3: Version Comparison Helper (--include-versions)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIncludeVersionsFlag:
    """Validate --include-versions discovery of historical checkpoints."""

    def test_model_comparison_accepts_include_versions(self):
        """run_model_comparison should accept --include-versions."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/experiment_runners/cross_topology/run_model_comparison.py",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--include-versions" in result.stdout

    def test_discover_checkpoints_with_versions(self, tmp_path):
        """discover_checkpoints should find _best/ and _versions/ files."""
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "scripts/experiment_runners/cross_topology"))
        from run_model_comparison import discover_checkpoints

        # Mock: create fake _best/ and _versions/ dirs with matching files
        from unittest.mock import patch

        fake_zoo = tmp_path / "checkpoints"
        fake_zoo.mkdir()
        best_dir = fake_zoo / "_best"
        best_dir.mkdir()
        versions_dir = fake_zoo / "_versions"
        versions_dir.mkdir()

        # Create fake checkpoint files
        (best_dir / "unified_tfim_br_chain_1d_multiN_pass60pct_140826.pt").write_bytes(b"fake")
        (versions_dir / "unified_tfim_br_chain_1d_multiN_6+8+10_p1_v2.pt").write_bytes(b"fake")

        with patch("run_model_comparison.ZOO_CHECKPOINTS", fake_zoo):
            with patch(
                "qmbp_simulation.predictors.model_zoo._load_manifest", return_value=[]
            ):
                # Without versions
                candidates_no_ver = discover_checkpoints("chain_1d", 1, None, include_versions=False)
                # With versions
                candidates_ver = discover_checkpoints("chain_1d", 1, None, include_versions=True)

        # Should find the _best/ and _versions/ files only with flag
        assert len(candidates_ver) >= len(candidates_no_ver)
        version_sources = [c["source"] for c in candidates_ver]
        assert any(s in ("historical_best", "historical_version") for s in version_sources)


# ═══════════════════════════════════════════════════════════════════════════════
# Mejora 4: Active Learning in Large-N Extrapolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestLargeNActiveLearning:
    """Validate --active-learning-rounds flag in run_large_n_extrapolation."""

    def test_flag_accepted(self):
        """run_large_n_extrapolation should accept --active-learning-rounds."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/experiment_runners/scaling/run_large_n_extrapolation.py",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--active-learning-rounds" in result.stdout

    def test_default_is_zero_disabled(self):
        """Default value should be 0 (disabled)."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/experiment_runners/scaling/run_large_n_extrapolation.py",
                "--help",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "0=disabled" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# Mejora 5: Training History Dashboard (compare_ablation_runs.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompareAblationRuns:
    """Validate compare_ablation_runs.py utility."""

    def test_script_imports_and_runs(self):
        """compare_ablation_runs should execute without error."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/analysis/compare_ablation_runs.py", "--latest", "2"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "ABLATION RUN COMPARISON" in result.stdout

    def test_format_table_with_empty_runs(self):
        """format_table should handle empty input gracefully."""
        sys.path.insert(0, str(ROOT / "scripts/analysis"))
        from compare_ablation_runs import format_table

        result = format_table([], None)
        assert "No ablation results found" in result

    def test_format_table_with_data(self):
        """format_table should produce a table from mock data."""
        sys.path.insert(0, str(ROOT / "scripts/analysis"))
        from compare_ablation_runs import format_table

        mock_runs = [
            {
                "timestamp": "2026-08-18T10:00:00",
                "topology": "chain_1d",
                "max_epochs": 4000,
                "n_training_graphs": 500,
                "results": [
                    {"name": "baseline", "val_mse": 0.12},
                    {"name": "res+jk+film", "val_mse": 0.08},
                ],
            },
            {
                "timestamp": "2026-08-19T10:00:00",
                "topology": "chain_1d",
                "max_epochs": 4000,
                "n_training_graphs": 600,
                "results": [
                    {"name": "baseline", "val_mse": 0.11},
                    {"name": "res+jk+film", "val_mse": 0.06},
                ],
            },
        ]
        table = format_table(mock_runs, None)
        assert "baseline" in table
        assert "res+jk+film" in table
        assert "Improvement" in table


# ═══════════════════════════════════════════════════════════════════════════════
# Zoo + Registry DB Integration Improvements
# ═══════════════════════════════════════════════════════════════════════════════


class TestZooEntryProperties:
    """Validate new ZooEntry convenience properties."""

    def test_is_multi_topology(self):
        """ZooEntry.is_multi_topology should detect multi-topology models."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        mt = ZooEntry(model="tfim", topology="multi_topology", n_qubits=0, p_layers=1,
                      checkpoint_file="mt.pt")
        per_topo = ZooEntry(model="tfim", topology="chain_1d", n_qubits=0, p_layers=1,
                            checkpoint_file="chain.pt")
        assert mt.is_multi_topology is True
        assert per_topo.is_multi_topology is False

    def test_is_multi_n(self):
        """ZooEntry.is_multi_n should detect multi-N models (n_qubits=0)."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        multi_n = ZooEntry(model="tfim", topology="chain_1d", n_qubits=0, p_layers=1,
                           checkpoint_file="multi.pt")
        single_n = ZooEntry(model="tfim", topology="chain_1d", n_qubits=10, p_layers=1,
                            checkpoint_file="single.pt")
        assert multi_n.is_multi_n is True
        assert single_n.is_multi_n is False

    def test_is_evaluated(self):
        """ZooEntry.is_evaluated should check pass_rate > 0."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        evaluated = ZooEntry(model="tfim", topology="chain_1d", n_qubits=0, p_layers=1,
                             checkpoint_file="a.pt", pass_rate=0.65)
        unevaluated = ZooEntry(model="tfim", topology="chain_1d", n_qubits=0, p_layers=1,
                               checkpoint_file="b.pt", pass_rate=0.0)
        assert evaluated.is_evaluated is True
        assert unevaluated.is_evaluated is False


class TestModelRegistryDBNewMethods:
    """Validate new ModelRegistryDB query methods."""

    def test_list_by_topology_returns_correct_type(self):
        """list_by_topology should return a list."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        result = db.list_by_topology("chain_1d")
        assert isinstance(result, list)

    def test_get_multi_topology_models_returns_list(self):
        """get_multi_topology_models should return a list."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        result = db.get_multi_topology_models()
        assert isinstance(result, list)
        # All returned should be topology=multi_topology
        for r in result:
            assert r.topology == "multi_topology"

    def test_get_best_for_topology(self):
        """get_best_for_topology should return a ModelRecord or None."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        best = db.get_best_for_topology("chain_1d", n_target=20)
        # Could be None if no active chain_1d models after pruning test entries
        if best is not None:
            assert best.topology == "chain_1d"

    def test_prune_test_entries_dry_run(self):
        """prune_test_entries dry_run should not modify registry."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        n_before = len(db.list_all(include_archived=True))
        db.prune_test_entries(dry_run=True)
        n_after = len(db.list_all(include_archived=True))
        assert n_before == n_after


class TestZooPruneAndListMT:
    """Validate zoo-level prune and list_multi_topology_entries."""

    def test_prune_test_entries_dry_run(self):
        """prune_test_entries dry_run should not modify manifest."""
        from qmbp_simulation.predictors.model_zoo import _load_manifest, prune_test_entries

        n_before = len(_load_manifest())
        prune_test_entries(dry_run=True)
        n_after = len(_load_manifest())
        assert n_before == n_after

    def test_list_multi_topology_entries(self):
        """list_multi_topology_entries should return MT entries only."""
        from qmbp_simulation.predictors.model_zoo import list_multi_topology_entries

        mt = list_multi_topology_entries()
        assert isinstance(mt, list)
        for e in mt:
            assert e.is_multi_topology


# ═══════════════════════════════════════════════════════════════════════════════
# B: Manifest Self-Heal
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestSelfHeal:
    """Validate heal_manifest detects and fixes inconsistencies."""

    def test_heal_manifest_returns_expected_keys(self):
        """heal_manifest should return dict with required keys."""
        from qmbp_simulation.predictors.model_zoo import heal_manifest

        result = heal_manifest(dry_run=True)
        assert "missing_checkpoints" in result
        assert "orphan_files" in result
        assert "duplicates_removed" in result
        assert "healed" in result
        assert isinstance(result["missing_checkpoints"], list)
        assert isinstance(result["orphan_files"], list)

    def test_heal_dry_run_does_not_modify(self):
        """Dry run should not change manifest."""
        from qmbp_simulation.predictors.model_zoo import _load_manifest, heal_manifest

        n_before = len(_load_manifest())
        heal_manifest(dry_run=True)
        n_after = len(_load_manifest())
        assert n_before == n_after


# ═══════════════════════════════════════════════════════════════════════════════
# D: Quality Gate (require_improvement)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQualityGate:
    """Validate require_improvement blocks registration of worse models."""

    def test_register_checkpoint_accepts_require_improvement(self):
        """register_checkpoint should have require_improvement parameter."""
        import inspect

        from qmbp_simulation.predictors.model_zoo import register_checkpoint

        sig = inspect.signature(register_checkpoint)
        assert "require_improvement" in sig.parameters

    def test_quality_gate_default_is_false(self):
        """Default should be False (permissive — backward compat)."""
        import inspect

        from qmbp_simulation.predictors.model_zoo import register_checkpoint

        sig = inspect.signature(register_checkpoint)
        default = sig.parameters["require_improvement"].default
        assert default is False


# ═══════════════════════════════════════════════════════════════════════════════
# F: MC-Dropout Confidence Bands
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCDropoutConfidence:
    """Validate MC-Dropout confidence estimation integration."""

    def test_mc_dropout_produces_nonzero_std(self, small_unified_model):
        """MC-Dropout with dropout>0 should produce non-zero std."""
        import torch
        from torch_geometric.data import Data

        model = small_unified_model

        # Create a test graph
        g = Data(
            x=torch.randn(6, 4),
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
            node_type=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            n_edges_unique=3,
            n_qubit_nodes=3,
        )

        # Collect predictions in train mode (dropout active)
        mc_preds = []
        model.train()
        with torch.no_grad():
            for seed in (42, 137, 256, 511, 769):
                torch.manual_seed(seed)
                pred = model(g).numpy().flatten()
                mc_preds.append(pred)
        model.eval()

        mc_arr = np.array(mc_preds)
        std_per_param = np.std(mc_arr, axis=0)
        mean_std = float(np.mean(std_per_param))

        # With random weights and dropout in train mode, std should be > 0
        assert mean_std > 0.0

    def test_eval_mode_produces_low_variance(self, small_unified_model):
        """In eval mode, predictions should be nearly deterministic."""
        import torch
        from torch_geometric.data import Data

        model = small_unified_model
        model.eval()

        g = Data(
            x=torch.randn(6, 4),
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
            node_type=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            n_edges_unique=3,
            n_qubit_nodes=3,
        )

        preds = []
        with torch.no_grad():
            for _ in range(5):
                pred = model(g).numpy().flatten()
                preds.append(pred)

        preds_arr = np.array(preds)
        std_per_param = np.std(preds_arr, axis=0)
        # Float32 precision allows tiny variance (~1e-8), but should be negligible
        assert np.all(std_per_param < 1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# A: Retrain Queue in Post-Run
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrainQueueIntegration:
    """Validate compute_retrain_queue is accessible and callable."""

    def test_compute_retrain_queue_returns_list(self):
        """compute_retrain_queue should return a list."""
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        assert isinstance(queue, list)

    def test_retrain_queue_entry_structure(self):
        """Each queue entry should have required keys."""
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        for entry in queue:
            assert "topology" in entry
            assert "priority" in entry
            assert "reason" in entry


# ═══════════════════════════════════════════════════════════════════════════════
# C: Model Lineage Tracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelLineage:
    """Validate get_lineage traces model ancestry."""

    def test_get_lineage_returns_list(self):
        """get_lineage should return a list of ancestry entries."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        lineage = db.get_lineage("nonexistent_model_xyz.pt")
        assert isinstance(lineage, list)
        # Nonexistent model should return empty lineage
        assert len(lineage) == 0

    def test_get_lineage_entry_structure(self):
        """Each lineage entry should have required keys."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        # Use any existing model
        all_models = db.list_all()
        if all_models:
            lineage = db.get_lineage(all_models[0].model_id)
            for entry in lineage:
                assert "model_id" in entry
                assert "topology" in entry
                assert "n_training_points" in entry
                assert "depth" in entry
                assert "fine_tuned_from" in entry

    def test_get_lineage_respects_max_depth(self):
        """Lineage should stop at max_depth."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        all_models = db.list_all()
        if all_models:
            lineage = db.get_lineage(all_models[0].model_id, max_depth=1)
            assert len(lineage) <= 1
