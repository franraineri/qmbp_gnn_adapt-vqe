"""Regression tests for bug fixes in thesis extension pipeline."""

import pytest

from qmbp_simulation.analysis.extension_analyzer import (
    DataRequirementEstimator,
    Ext3Analyzer,
    PrerequisiteChecker,
    ThesisImpactReporter,
)
from qmbp_simulation.analysis.extension_classifiers import ClassificationEngine
from qmbp_simulation.analysis.extension_models import ExtensionClassification

# ---------------------------------------------------------------------------
# C1 — hilbert_space_dimension returns int, not tuple
# ---------------------------------------------------------------------------


def test_hilbert_space_dimension_returns_int():
    dim = ClassificationEngine.hilbert_space_dimension(12)
    assert isinstance(dim, int)
    assert dim == 4096


def test_hilbert_space_dimension_values():
    for n in range(1, 22):
        assert ClassificationEngine.hilbert_space_dimension(n) == 2**n


def test_hilbert_space_dimension_flagged():
    dim, flag = ClassificationEngine.hilbert_space_dimension_flagged(12)
    assert dim == 4096 and flag is False
    dim2, flag2 = ClassificationEngine.hilbert_space_dimension_flagged(20)
    assert dim2 == 2**20 and flag2 is True


# ---------------------------------------------------------------------------
# m2 — classify_intra_n uses math.ceil (not floor division)
# ---------------------------------------------------------------------------


def test_classify_intra_n_ceil_n6():
    # n_total=6: ceil(5*6/6)=5 — standard case
    r = ClassificationEngine.classify_intra_n(0.007, 5, 6)
    assert r == ExtensionClassification.CONDITIONALLY_VIABLE
    r2 = ClassificationEngine.classify_intra_n(0.007, 4, 6)
    assert r2 == ExtensionClassification.REJECTED_INSUFFICIENT_DATA


def test_classify_intra_n_ceil_n7():
    # n_total=7: ceil(5*7/6)=6 — floor division gave 5 (wrong, too lenient)
    r_pass = ClassificationEngine.classify_intra_n(0.007, 6, 7)
    assert r_pass == ExtensionClassification.CONDITIONALLY_VIABLE
    r_fail = ClassificationEngine.classify_intra_n(0.007, 5, 7)
    assert r_fail == ExtensionClassification.REJECTED_INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# M2 — n_h_points is stored as metadata, does NOT multiply T_collection
# ---------------------------------------------------------------------------


def test_data_requirement_n_h_points_stored_as_metadata():
    """n_h_points must appear in the result dict as metadata, not multiply T_collection."""
    est = DataRequirementEstimator.estimate(494000, 6, 6)
    assert "n_h_points" in est, "n_h_points must be stored in result dict"
    assert est["n_h_points"] == 6


def test_data_requirement_t_collection_does_not_scale_with_n_h_points():
    """T_collection must NOT multiply by n_h_points — N_min is total evaluations needed."""
    est1 = DataRequirementEstimator.estimate(494000, 6, 1)
    est6 = DataRequirementEstimator.estimate(494000, 6, 6)
    # T_collection must be identical regardless of n_h_points
    assert abs(est1["T_collection_seconds"] - est6["T_collection_seconds"]) < 0.01, (
        "T_collection_seconds must not vary with n_h_points "
        "(N_min_data already represents total evaluations)"
    )


# ---------------------------------------------------------------------------
# M6 — ThesisImpactReporter.generate_all exists
# ---------------------------------------------------------------------------


def test_thesis_impact_reporter_generate_all_exists():
    assert hasattr(ThesisImpactReporter, "generate_all")
    assert callable(ThesisImpactReporter.generate_all)


# ---------------------------------------------------------------------------
# C3 — Ext3Analyzer uses classify_flow_architecture
# ---------------------------------------------------------------------------


def test_ext3_uses_classify_flow_architecture():
    analyzer = Ext3Analyzer(
        mc_dropout_baseline={"coverage_90": 0.85, "mean_sharpness": 0.10},
        n_data=45,
    )
    result = analyzer.run()
    rm = result.raw_metrics
    assert "flow_classification_basis" in rm, "flow_classification_basis missing"
    assert "flow_calibration_improvement" in rm
    assert "flow_de_gap_assumed" in rm
    # Structural analysis: calibration_improvement=0.0, de_gap=0.03 → CONDITIONALLY_VIABLE
    assert result.classification == ExtensionClassification.CONDITIONALLY_VIABLE


# ---------------------------------------------------------------------------
# M7 + M8 — ext2 prereq failure uses proper thesis narrative and 5.0h estimate
# ---------------------------------------------------------------------------


def test_ext2_prerequisite_failure_narrative_and_time():
    checker = PrerequisiteChecker()
    prereq = checker._run_all_internal()
    ext2r = prereq.get("ext2_result")
    if ext2r is None:
        pytest.skip("Kagomé lattice available — prerequisite did not fail")
    # M7: thesis_narrative must not contain raw exception/TypeError
    assert "TypeError" not in ext2r.thesis_narrative
    assert "geometry" not in ext2r.thesis_narrative or "future work" in ext2r.thesis_narrative
    # M8: estimated_time uses Ext2Analyzer's natural estimate, not 0.0
    assert ext2r.estimated_time_to_result_hours == 5.0


# ---------------------------------------------------------------------------
# M9 — _check_tenpy_install no longer calls subprocess
# ---------------------------------------------------------------------------


def test_tenpy_check_is_fast():
    """Ensure the tenpy check is fast — no pip install --dry-run subprocess call."""
    import subprocess
    from unittest.mock import patch

    from qmbp_simulation.analysis.extension_analyzer import _check_tenpy_install

    # Capture any subprocess.run calls and verify none are pip install
    captured_args = []
    original_run = subprocess.run

    def tracking_run(args, **kwargs):
        captured_args.append(args)
        return original_run(args, **kwargs)

    with patch("subprocess.run", side_effect=tracking_run):
        result = _check_tenpy_install()

    pip_calls = [a for a in captured_args if isinstance(a, list) and "pip" in a]
    assert len(pip_calls) == 0, f"pip subprocess call should not occur: {pip_calls}"
    assert "available" in result
