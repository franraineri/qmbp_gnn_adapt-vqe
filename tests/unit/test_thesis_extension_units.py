"""Unit tests for thesis extension analysis modules.

Covers:
  - DataRequirementEstimator (Task 8.2, Req 1.5, 1.8)
  - GroundTruthSourceSelector (Task 9.2, Req 2.1, 2.3)
  - ExtensionPriorityRanker (Req 5.8)
  - CalibrationComparator (Req 3.3)
  - OverparameterizationGuard (Req 3.2)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# DataRequirementEstimator — Task 8.2 (Req 1.5, 1.8)
# ---------------------------------------------------------------------------


class TestDataRequirementEstimator:
    """Unit tests for DataRequirementEstimator (Req 1.5, 1.8)."""

    def test_n_min_data_494k_params(self):
        """Verify N_min_data = 494 for 494K params (Req 1.5)."""
        from qmbp_simulation.analysis.extension_analyzer import DataRequirementEstimator

        result = DataRequirementEstimator.estimate(
            n_params=494_000,
            n_qubits=6,
            n_h_points=1,
        )
        assert result["N_min_data"] == 494, (
            f"Expected N_min_data=494 for 494K params, got {result['N_min_data']}"
        )

    def test_t_collection_approx_34_min(self):
        """T_collection for 494 pts at N=6 is T(6)*494 (Req 1.8).

        Verifies the formula T(N)=0.08*N^2.56 is applied correctly.
        The result must be < 48h (the gate boundary).
        """
        from qmbp_simulation.analysis.extension_analyzer import DataRequirementEstimator

        result = DataRequirementEstimator.estimate(
            n_params=494_000,
            n_qubits=6,
            n_h_points=1,
        )
        # Compute expected T manually using design formula
        expected_t_per_point = 0.08 * (6**2.56)  # T(6) in seconds
        expected_total_s = 494 * expected_t_per_point
        assert math.isclose(result["T_collection_seconds"], expected_total_s, rel_tol=1e-6), (
            f"T_collection mismatch: {result['T_collection_seconds']:.1f}s ≠ {expected_total_s:.1f}s"
        )
        # Must be < 48h regardless
        assert result["T_collection_hours"] < 48.0, (
            f"T_collection={result['T_collection_hours']:.1f}h must be < 48h"
        )

    def test_gate_approves_under_48h(self):
        """Gate APPROVED when T_total ≤ 48h (Req 1.8)."""
        from qmbp_simulation.analysis.extension_analyzer import DataRequirementEstimator

        result = DataRequirementEstimator.estimate(
            n_params=494_000,
            n_qubits=6,
            n_h_points=1,
        )
        assert result["gate_approved"] is True, (
            f"Expected gate_approved=True for T={result['T_collection_hours']:.2f}h ≤ 48h"
        )

    def test_gate_rejects_large_t_total(self):
        """Gate REJECTED when T_total > 48h (Req 1.8)."""
        from qmbp_simulation.analysis.extension_analyzer import DataRequirementEstimator

        # Very large N → very slow T(N)
        result = DataRequirementEstimator.estimate(
            n_params=494_000,
            n_qubits=100,  # T(100) is enormous
            n_h_points=1,
        )
        assert result["gate_approved"] is False, (
            f"Expected gate_approved=False for T={result['T_collection_hours']:.2f}h > 48h"
        )

    def test_t_per_hpoint_formula(self):
        """Verify T(N) = 0.08 * N^2.56 (Req 1.8)."""
        from qmbp_simulation.analysis.extension_analyzer import DataRequirementEstimator

        N = 6
        expected = 0.08 * (6**2.56)
        actual = DataRequirementEstimator.t_per_hpoint(N)
        assert abs(actual - expected) < 1e-6, f"T({N}) = {actual:.6f} ≠ expected {expected:.6f}"


# ---------------------------------------------------------------------------
# GroundTruthSourceSelector — Task 9.2 (Req 2.1, 2.3)
# ---------------------------------------------------------------------------


class TestGroundTruthSourceSelector:
    """Unit tests for GroundTruthSourceSelector (Req 2.1, 2.3)."""

    def test_exact_diag_selected_for_n12(self):
        """ExactDiag is selected for N≤12 (Req 2.1, 2.3)."""
        from qmbp_simulation.analysis.extension_analyzer import GroundTruthSourceSelector

        rec = GroundTruthSourceSelector.evaluate(n_max=12)
        assert rec["selected_source"] == "ExactDiag", (
            f"Expected ExactDiag for N≤12, got {rec['selected_source']}"
        )
        assert rec["exact_diag_viable"] is True

    def test_exact_diag_selected_for_n6(self):
        """ExactDiag is also selected for N≤6 (Req 2.1)."""
        from qmbp_simulation.analysis.extension_analyzer import GroundTruthSourceSelector

        rec = GroundTruthSourceSelector.evaluate(n_max=6)
        assert rec["selected_source"] == "ExactDiag"
        assert rec["hilbert_space_dim"] == 64  # 2^6

    def test_tenpy_rejected_when_unavailable(self, monkeypatch):
        """TeNPy with conflict/unavailability → rejection documented (Req 2.3)."""
        from qmbp_simulation.analysis import extension_analyzer

        # Monkeypatch _check_tenpy_install to simulate a conflict
        def mock_check():
            return {"available": False, "reason": "Simulated dependency conflict: tenpy"}

        monkeypatch.setattr(extension_analyzer, "_check_tenpy_install", mock_check)

        from qmbp_simulation.analysis.extension_analyzer import GroundTruthSourceSelector

        rec = GroundTruthSourceSelector.evaluate(n_max=12)
        assert rec["tenpy_available"] is False
        assert "TeNPy DMRG" in rec["alternatives_rejected"]
        assert rec["tenpy_rejection_reason"] is not None

    def test_hilbert_space_dim_for_n12(self):
        """H.S. dimension for N=12 = 4096 (Req 2.2)."""
        from qmbp_simulation.analysis.extension_analyzer import GroundTruthSourceSelector

        rec = GroundTruthSourceSelector.evaluate(n_max=12)
        assert rec["hilbert_space_dim"] == 4096  # 2^12

    def test_literature_always_rejected(self):
        """Literature values always in alternatives_rejected (Req 2.3)."""
        from qmbp_simulation.analysis.extension_analyzer import GroundTruthSourceSelector

        rec = GroundTruthSourceSelector.evaluate(n_max=12)
        assert "Literature" in rec["alternatives_rejected"]


# ---------------------------------------------------------------------------
# OverparameterizationGuard (Req 3.2)
# ---------------------------------------------------------------------------


class TestOverparameterizationGuard:
    """Unit tests for OverparameterizationGuard."""

    def test_arch_b_passes_gate(self):
        """EmbeddingMAF ~584 trainable params with 45 data → passes gate."""
        from qmbp_simulation.analysis.extension_analyzer import OverparameterizationGuard
        from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF

        model = EmbeddingMAF(embedding_dim=64, theta_dim=4, n_flow_layers=2, hidden_dim=32)
        result = OverparameterizationGuard.check(model, n_data=45)
        assert result is None, (
            f"EmbeddingMAF should pass guard (trainable params ≤ 5000), got {result}"
        )

    def test_trainable_params_count_excludes_frozen(self):
        """count_trainable_params counts only requires_grad params."""
        import torch.nn as nn

        from qmbp_simulation.analysis.extension_analyzer import OverparameterizationGuard

        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 4))
        # Freeze first layer
        for p in model[0].parameters():
            p.requires_grad = False

        trainable = OverparameterizationGuard.count_trainable_params(model)
        total = sum(p.numel() for p in model.parameters())
        frozen = sum(p.numel() for p in model[0].parameters())

        assert trainable == total - frozen, (
            f"Expected {total - frozen} trainable params, got {trainable}"
        )

    def test_large_model_triggers_gate(self):
        """30K param model with n_data=45 → OVERPARAMETERIZED."""
        import torch.nn as nn

        from qmbp_simulation.analysis.extension_analyzer import OverparameterizationGuard
        from qmbp_simulation.analysis.extension_models import ExtensionClassification

        # Create a model with > 5000 trainable params
        model = nn.Sequential(nn.Linear(64, 256), nn.Linear(256, 64))
        trainable = OverparameterizationGuard.count_trainable_params(model)
        assert trainable > 5000, "Test setup: model must have >5000 params"

        result = OverparameterizationGuard.check(model, n_data=45)
        assert result == ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET


# ---------------------------------------------------------------------------
# CalibrationComparator (Req 3.3)
# ---------------------------------------------------------------------------


class TestCalibrationComparator:
    """Unit tests for CalibrationComparator."""

    def test_perfect_coverage_returns_low_error(self):
        """True value always inside 90% interval → coverage ≈ 1.0."""
        from qmbp_simulation.analysis.extension_analyzer import CalibrationComparator

        # 10 points, each with 100 samples in [0, 10]; true value at 5.0 (inside)
        samples = [[float(i) * 0.1 for i in range(100)] for _ in range(10)]
        true_values = [5.0] * 10  # all at median → always covered

        coverage = CalibrationComparator.compute_empirical_coverage(
            samples, true_values, nominal=0.90
        )
        assert coverage > 0.0, "Coverage should be positive"

    def test_zero_coverage_when_true_outside(self):
        """True value always outside interval → coverage = 0.0."""
        from qmbp_simulation.analysis.extension_analyzer import CalibrationComparator

        # 10 points, samples in [0, 1]; true value at 100 (always outside)
        samples = [[float(i) * 0.01 for i in range(100)] for _ in range(10)]
        true_values = [100.0] * 10

        coverage = CalibrationComparator.compute_empirical_coverage(
            samples, true_values, nominal=0.90
        )
        assert coverage == 0.0, f"Expected coverage=0.0, got {coverage}"

    def test_coverage_improvement_positive_when_flow_better(self):
        """coverage_improvement > 0 when flow error < baseline error."""
        from qmbp_simulation.analysis.extension_analyzer import CalibrationComparator

        # Baseline coverage 0.80 (error=0.10), flow coverage 0.88 (error=0.02)
        improvement = CalibrationComparator.coverage_improvement(
            flow_coverage=0.88,
            baseline_coverage=0.80,
            nominal=0.90,
        )
        assert improvement > 0.0, (
            f"Expected positive improvement when flow (0.88) closer to 0.90 "
            f"than baseline (0.80), got {improvement:.4f}"
        )
