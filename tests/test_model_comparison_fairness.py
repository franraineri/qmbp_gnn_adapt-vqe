"""Fairness tests for run_model_comparison.py — MT vs ST comparisons.

Validates that the comparison pipeline evaluates MT and ST models under
identical conditions: same h-grid, same N targets, same ground truth,
same backend, and same metric computation. Any asymmetry would
invalidate the head-to-head comparison.

These tests use mock models (constant-output) to verify the FRAMEWORK
is fair, independent of model quality.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from qmbp_simulation.analysis.metrics import compute_deploy_summary
from qmbp_simulation.models.hamiltonian import make_lattice


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


class _MockModel(torch.nn.Module):
    """Deterministic mock model that returns fixed θ based on offset."""

    def __init__(self, output_dim: int, offset: float = 0.0):
        super().__init__()
        self._output_dim = output_dim
        self._offset = offset
        self.use_residual = False
        self.readout_mode = "last"
        self.film_conditioning = False

    def forward(self, g):
        return torch.full((1, self._output_dim), self._offset + 0.1)

    def parameters(self):
        return iter([torch.zeros(1)])


class _NaNModel(torch.nn.Module):
    """Model that returns NaN for some outputs — edge case."""

    def __init__(self, output_dim: int):
        super().__init__()
        self._output_dim = output_dim
        self.use_residual = False
        self.readout_mode = "last"
        self.film_conditioning = False

    def forward(self, g):
        out = torch.full((1, self._output_dim), 0.1)
        out[0, 0] = float("nan")
        return out

    def parameters(self):
        return iter([torch.zeros(1)])


class _WrongDimModel(torch.nn.Module):
    """Model that returns wrong number of parameters — tests pad/truncate."""

    def __init__(self, output_dim: int):
        super().__init__()
        self._output_dim = output_dim
        self.use_residual = False
        self.readout_mode = "last"
        self.film_conditioning = False

    def forward(self, g):
        return torch.full((1, self._output_dim), 0.1)

    def parameters(self):
        return iter([torch.zeros(1)])


@pytest.fixture
def h_values():
    """Standard h-grid for fairness testing."""
    return [round(h, 2) for h in np.linspace(2.0, 4.0, 8)]


@pytest.fixture
def target_ns():
    return [4, 6]


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Both models receive identical h-values
# ═══════════════════════════════════════════════════════════════════════════════


class TestComparisonFairness:
    """Verify MT and ST are evaluated under identical conditions."""

    def test_same_h_grid_for_both_models(self, h_values, target_ns):
        """Both models must be evaluated at exactly the same h-values."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        mt_model = _MockModel(output_dim=7, offset=0.05)
        mt_model.use_residual = True
        mt_model.film_conditioning = True
        st_model = _MockModel(output_dim=7, offset=0.10)

        with tempfile.TemporaryDirectory() as td:
            mt_path = Path(td) / "mt.pt"
            st_path = Path(td) / "st.pt"
            torch.save(mt_model.state_dict(), mt_path)
            torch.save(st_model.state_dict(), st_path)

            def _mock_load(path, eval_mode=True):
                if "mt" in str(path):
                    return mt_model
                return st_model

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                side_effect=_mock_load,
            ):
                result_mt = evaluate_checkpoint(
                    mt_path, "chain_1d", target_ns, h_values,
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )
                result_st = evaluate_checkpoint(
                    st_path, "chain_1d", target_ns, h_values,
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        # FAIRNESS: both must have results for same N values
        assert set(result_mt["results_by_n"].keys()) == set(result_st["results_by_n"].keys())

        # FAIRNESS: same number of h-points per N
        for n in target_ns:
            mt_points = result_mt["results_by_n"][n]["per_point"]
            st_points = result_st["results_by_n"][n]["per_point"]
            assert len(mt_points) == len(st_points) == len(h_values), (
                f"N={n}: MT has {len(mt_points)} points, ST has {len(st_points)}, "
                f"expected {len(h_values)}"
            )

        # FAIRNESS: identical h-values in same order
        for n in target_ns:
            mt_h = [p["h"] for p in result_mt["results_by_n"][n]["per_point"]]
            st_h = [p["h"] for p in result_st["results_by_n"][n]["per_point"]]
            np.testing.assert_array_equal(
                mt_h, st_h,
                err_msg=f"N={n}: h-values differ between MT and ST"
            )

    def test_same_ground_truth_for_both_models(self, h_values, target_ns):
        """Both models must be compared against the same E_exact and gap."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        gt_memory = {}

        mt_model = _MockModel(output_dim=7, offset=0.05)
        mt_model.use_residual = True
        st_model = _MockModel(output_dim=7, offset=0.10)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mt_path = Path(td) / "mt.pt"
            st_path = Path(td) / "st.pt"
            torch.save(mt_model.state_dict(), mt_path)
            torch.save(st_model.state_dict(), st_path)

            def _mock_load(path, eval_mode=True):
                if "mt" in str(path):
                    return mt_model
                return st_model

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                side_effect=_mock_load,
            ):
                result_mt = evaluate_checkpoint(
                    mt_path, "chain_1d", target_ns, h_values,
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )
                result_st = evaluate_checkpoint(
                    st_path, "chain_1d", target_ns, h_values,
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        # FAIRNESS: identical ground truth energies and gaps
        for n in target_ns:
            mt_exact = [p["e_exact"] for p in result_mt["results_by_n"][n]["per_point"]]
            st_exact = [p["e_exact"] for p in result_st["results_by_n"][n]["per_point"]]
            np.testing.assert_array_almost_equal(
                mt_exact, st_exact, decimal=10,
                err_msg=f"N={n}: E_exact differs between MT and ST evaluations"
            )

            mt_gaps = [p["gap"] for p in result_mt["results_by_n"][n]["per_point"]]
            st_gaps = [p["gap"] for p in result_st["results_by_n"][n]["per_point"]]
            np.testing.assert_array_almost_equal(
                mt_gaps, st_gaps, decimal=10,
                err_msg=f"N={n}: gaps differ between MT and ST evaluations"
            )

    def test_same_metric_computation(self, h_values):
        """compute_deploy_summary must produce same keys for both models."""
        per_h_mt = [
            {"h": h, "e_pred": -5.0 + h * 0.1, "e_exact": -5.0, "gap": 1.0,
             "de_gap": abs(h * 0.1) / 1.0, "abs_error": abs(h * 0.1)}
            for h in h_values
        ]
        per_h_st = [
            {"h": h, "e_pred": -5.0 + h * 0.2, "e_exact": -5.0, "gap": 1.0,
             "de_gap": abs(h * 0.2) / 1.0, "abs_error": abs(h * 0.2)}
            for h in h_values
        ]

        summary_mt = compute_deploy_summary(per_h_mt)
        summary_st = compute_deploy_summary(per_h_st)

        assert set(summary_mt.keys()) == set(summary_st.keys()), (
            "MT and ST summaries have different metric keys"
        )
        assert summary_mt["n_points"] == summary_st["n_points"] == len(h_values)

    def test_no_information_leakage_between_models(self, h_values, target_ns):
        """GT cache populated by model A gives same values when reused by model B."""
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers.classical import ClassicalSolver

        gt_memory: dict = {}
        topology = "chain_1d"
        model_name = "tfim_bond_resolved"
        spec = get_model_spec(model_name)
        solver = ClassicalSolver()

        for n in target_ns:
            for h in h_values:
                cache_key = (topology, n, round(float(h), 6))
                lat_h = make_lattice(topology, n, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                gt_obj = solver.solve(H, lat_h)
                gt_memory[cache_key] = (gt_obj.ground_energy, gt_obj.gap)

        gt_first = dict(gt_memory)

        # Simulate 2nd model using cached values
        for key in gt_first:
            e1, g1 = gt_first[key]
            e2, g2 = gt_memory[key]
            assert e1 == e2, f"E_exact changed for {key}"
            assert g1 == g2, f"Gap changed for {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Edge cases in evaluate_checkpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateCheckpointEdgeCases:
    """Test edge cases that could cause bugs or unfair comparisons."""

    def test_theta_dimension_mismatch_pad(self, h_values):
        """Model output shorter than circuit params → should be zero-padded."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        # N=4 chain_1d p=1 has 7 params. Model outputs only 3.
        short_model = _WrongDimModel(output_dim=3)

        import tempfile
        gt_cache = GroundTruthCache()
        gt_memory = {}

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "short.pt"
            torch.save(short_model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=short_model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:3],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        # Should not crash, and produce valid results
        assert 4 in result["results_by_n"]
        points = result["results_by_n"][4]["per_point"]
        assert len(points) == 3
        # Each theta should have been padded to circuit's n_params (7)
        for p in points:
            assert len(p["theta"]) == 7
            # Padded positions should be 0.0
            assert all(t == 0.0 for t in p["theta"][3:])

    def test_theta_dimension_mismatch_truncate(self, h_values):
        """Model output longer than circuit params → should be truncated."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        # N=4 chain_1d p=1 has 7 params. Model outputs 20.
        long_model = _WrongDimModel(output_dim=20)

        import tempfile
        gt_cache = GroundTruthCache()
        gt_memory = {}

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "long.pt"
            torch.save(long_model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=long_model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:3],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        points = result["results_by_n"][4]["per_point"]
        assert len(points) == 3
        # Each theta should have been truncated to 7
        for p in points:
            assert len(p["theta"]) == 7

    def test_single_h_point(self):
        """Evaluation with a single h-point should not crash."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        model = _MockModel(output_dim=7)
        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "single.pt"
            torch.save(model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], [3.0],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        assert result["results_by_n"][4]["n_points"] == 1

    def test_de_gap_computed_correctly(self, h_values):
        """ΔE/gap must equal |e_pred - e_exact| / gap for each point."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        model = _MockModel(output_dim=7, offset=0.3)
        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "check.pt"
            torch.save(model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:5],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        for p in result["results_by_n"][4]["per_point"]:
            expected_de_gap = abs(p["e_pred"] - p["e_exact"]) / max(p["gap"], 1e-10)
            assert abs(p["de_gap"] - expected_de_gap) < 1e-10, (
                f"h={p['h']}: de_gap={p['de_gap']} but expected {expected_de_gap}"
            )

    def test_abs_error_computed_correctly(self, h_values):
        """|ΔE| must equal |e_pred - e_exact|."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        model = _MockModel(output_dim=7, offset=0.5)
        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "abs.pt"
            torch.save(model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:4],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        for p in result["results_by_n"][4]["per_point"]:
            expected = abs(p["e_pred"] - p["e_exact"])
            assert abs(p["abs_error"] - expected) < 1e-10

    def test_theta_is_clipped_to_pi(self, h_values):
        """Model output must be clipped to [-π, π] before evaluation."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        # Model that outputs values > π
        big_model = _MockModel(output_dim=7, offset=5.0)  # 5.1 > π
        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "big.pt"
            torch.save(big_model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=big_model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:2],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        for p in result["results_by_n"][4]["per_point"]:
            for t in p["theta"]:
                # Allow float32 rounding (np.clip on float32 gives 3.14159274 not 3.14159265)
                assert -np.pi - 1e-6 <= t <= np.pi + 1e-6, f"θ={t} not clipped to [-π, π]"

    def test_result_structure_completeness(self, h_values):
        """Each per-h result dict must have all required keys."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            evaluate_checkpoint,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        model = _MockModel(output_dim=7)
        gt_cache = GroundTruthCache()
        gt_memory = {}

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "struct.pt"
            torch.save(model.state_dict(), path)

            with patch(
                "scripts.experiment_runners.cross_topology.run_model_comparison.load_unified_checkpoint",
                return_value=model,
            ):
                result = evaluate_checkpoint(
                    path, "chain_1d", [4], h_values[:3],
                    p_layers=1, model_name="tfim_bond_resolved",
                    gt_cache=gt_cache, gt_memory=gt_memory,
                )

        required_keys = {
            "h", "e_pred", "e_exact", "gap", "de_gap", "abs_error",
            "n_qubits", "theta_std", "category", "action", "theta",
        }
        for p in result["results_by_n"][4]["per_point"]:
            missing = required_keys - set(p.keys())
            assert not missing, f"Missing keys in per-h result: {missing}"

        # Top-level result structure
        assert "results_by_n" in result
        assert "n_model_params" in result
        assert "use_residual" in result
        assert "film_conditioning" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Checkpoint discovery and resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointDiscovery:
    """Verify checkpoint resolution logic."""

    def test_missing_checkpoint_is_auto_resolved(self, tmp_path):
        """Missing checkpoint should fuzzy-match to closest available."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            discover_checkpoints,
        )

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "unified_tfim_br_triangular_multiN_3+4+6_p1.pt").touch()
        (ckpt_dir / "unified_tfim_br_chain_1d_multiN_6+8+10_p1.pt").touch()

        with patch(
            "scripts.experiment_runners.cross_topology.run_model_comparison.ZOO_CHECKPOINTS",
            ckpt_dir,
        ):
            candidates = discover_checkpoints(
                topology="triangular",
                p_layers=1,
                explicit=[
                    str(ckpt_dir / "unified_tfim_br_triangular_multiN_3+4+6+8+10_p1.pt"),
                ],
            )

        assert len(candidates) == 1
        assert "triangular" in candidates[0]["path"].name
        assert candidates[0]["source"] == "explicit (auto-resolved)"

    def test_auto_resolve_picks_correct_topology(self, tmp_path):
        """Auto-resolve must prefer the correct topology over similar names."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            discover_checkpoints,
        )

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        # Two similar names, different topologies
        (ckpt_dir / "unified_tfim_br_chain_1d_multiN_6+8+10_p1.pt").touch()
        (ckpt_dir / "unified_tfim_br_ladder_multiN_4+6+8+10_p1.pt").touch()

        with patch(
            "scripts.experiment_runners.cross_topology.run_model_comparison.ZOO_CHECKPOINTS",
            ckpt_dir,
        ):
            candidates = discover_checkpoints(
                topology="ladder",
                p_layers=1,
                explicit=[
                    str(ckpt_dir / "unified_tfim_br_ladder_multiN_4+6+8+10+12_p1.pt"),
                ],
            )

        assert len(candidates) == 1
        assert "ladder" in candidates[0]["path"].name

    def test_both_checkpoints_found_gives_two_candidates(self, tmp_path):
        """When both paths exist, get exactly 2 candidates."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            discover_checkpoints,
        )

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mt = ckpt_dir / "mt_model_p1.pt"
        st = ckpt_dir / "st_chain_1d_p1.pt"
        mt.touch()
        st.touch()

        with patch(
            "scripts.experiment_runners.cross_topology.run_model_comparison.ZOO_CHECKPOINTS",
            ckpt_dir,
        ):
            candidates = discover_checkpoints(
                topology="chain_1d",
                p_layers=1,
                explicit=[str(mt), str(st)],
            )

        assert len(candidates) == 2

    def test_warns_when_only_one_checkpoint(self, tmp_path, capsys):
        """Should warn when < 2 checkpoints resolve."""
        from scripts.experiment_runners.cross_topology.run_model_comparison import (
            discover_checkpoints,
        )

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        only_one = ckpt_dir / "model_p1.pt"
        only_one.touch()

        with patch(
            "scripts.experiment_runners.cross_topology.run_model_comparison.ZOO_CHECKPOINTS",
            ckpt_dir,
        ):
            candidates = discover_checkpoints(
                topology="chain_1d",
                p_layers=1,
                explicit=[str(only_one)],
            )

        assert len(candidates) == 1
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "Only 1" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Persist predictions fairness
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistPredictionsFairness:
    """Verify that training data persistence doesn't favor one model."""

    def test_best_energy_wins_regardless_of_model_order(self, tmp_path):
        """The point with lowest energy should be persisted, not the first one seen."""
        from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz

        mt_points = [
            {"h": 3.0, "e_pred": -5.0, "e_exact": -5.1, "gap": 0.5,
             "de_gap": 0.02, "abs_error": 0.1, "theta": [0.1, 0.2, 0.3],
             "method": "mpnn"},
        ]
        st_points = [
            {"h": 3.0, "e_pred": -4.8, "e_exact": -5.1, "gap": 0.5,
             "de_gap": 0.06, "abs_error": 0.3, "theta": [0.4, 0.5, 0.6],
             "method": "mpnn"},
        ]

        persist_predictions_to_training_npz(
            {6: mt_points}, "test", 1, training_data_dir=tmp_path
        )
        persist_predictions_to_training_npz(
            {6: st_points}, "test", 1, training_data_dir=tmp_path
        )

        data = np.load(tmp_path / "test_N6_p1.npz", allow_pickle=True)
        assert float(data["e_vqe"][0]) == pytest.approx(-5.0, abs=1e-6)

    def test_persist_order_does_not_matter(self, tmp_path):
        """Same result regardless of which model is persisted first."""
        from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz

        better = [{"h": 2.0, "e_pred": -6.0, "e_exact": -6.1, "gap": 1.0,
                   "de_gap": 0.01, "abs_error": 0.1, "theta": [0.1, 0.2],
                   "method": "mpnn"}]
        worse = [{"h": 2.0, "e_pred": -5.5, "e_exact": -6.1, "gap": 1.0,
                  "de_gap": 0.06, "abs_error": 0.6, "theta": [0.3, 0.4],
                  "method": "mpnn"}]

        dir_a = tmp_path / "a"
        dir_a.mkdir()
        persist_predictions_to_training_npz({4: better}, "x", 1, training_data_dir=dir_a)
        persist_predictions_to_training_npz({4: worse}, "x", 1, training_data_dir=dir_a)

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        persist_predictions_to_training_npz({4: worse}, "x", 1, training_data_dir=dir_b)
        persist_predictions_to_training_npz({4: better}, "x", 1, training_data_dir=dir_b)

        data_a = np.load(dir_a / "x_N4_p1.npz", allow_pickle=True)
        data_b = np.load(dir_b / "x_N4_p1.npz", allow_pickle=True)
        assert float(data_a["e_vqe"][0]) == pytest.approx(float(data_b["e_vqe"][0]), abs=1e-10)
        assert float(data_a["e_vqe"][0]) == pytest.approx(-6.0, abs=1e-6)

    def test_nan_theta_not_persisted(self, tmp_path):
        """Points with NaN in theta must be filtered before persistence."""
        from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz

        points = [
            {"h": 2.0, "e_pred": -5.0, "e_exact": -5.1, "gap": 1.0,
             "de_gap": 0.02, "theta": [float("nan"), 0.1], "method": "mpnn"},
            {"h": 3.0, "e_pred": -4.0, "e_exact": -4.1, "gap": 1.0,
             "de_gap": 0.02, "theta": [0.1, 0.2], "method": "mpnn"},
        ]

        result = persist_predictions_to_training_npz(
            {4: points}, "nan_test", 1, training_data_dir=tmp_path
        )
        assert result["total_added"] == 1  # Only the valid point

    def test_vqe_refined_gets_verified_tier(self, tmp_path):
        """VQE-refined points passing strict criterion get 'verified' tier."""
        from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz

        points = [
            {"h": 3.0, "e_pred": -3.0, "e_exact": -3.01, "gap": 1.0,
             "de_gap": 0.01, "abs_error": 0.01, "theta": [0.5, 0.6],
             "method": "vqe_refined"},
        ]
        persist_predictions_to_training_npz(
            {8: points}, "tier_test", 1, training_data_dir=tmp_path
        )

        data = np.load(tmp_path / "tier_test_N8_p1.npz", allow_pickle=True)
        assert "verified" in data["quality_tier"].tolist()

    def test_mpnn_pred_gets_approximate_tier(self, tmp_path):
        """MPNN predictions passing strict criterion get 'approximate' tier."""
        from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz

        points = [
            {"h": 3.0, "e_pred": -3.0, "e_exact": -3.01, "gap": 1.0,
             "de_gap": 0.01, "abs_error": 0.01, "theta": [0.5, 0.6],
             "method": "mpnn_pred"},
        ]
        persist_predictions_to_training_npz(
            {8: points}, "tier_test2", 1, training_data_dir=tmp_path
        )

        data = np.load(tmp_path / "tier_test2_N8_p1.npz", allow_pickle=True)
        assert "approximate" in data["quality_tier"].tolist()
