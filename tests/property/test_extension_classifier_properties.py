"""Property-based tests for ClassificationEngine and ExtensionPriorityRanker.

Feature: thesis-extensions-analysis

Properties covered:
  Property 1: classify_cross_n is correct for all ΔE/gap values (Req 1.3)
  Property 2: N_min_data satisfies params/data ≤ 1000 (Req 1.5)
  Property 3: classify_intra_n is consistent with threshold (Req 1.6)
  Property 4: classify_hardware consistent with CX=18 threshold (Req 1.7, 2.4)
  Property 5: hilbert_space_dimension = 2^N (Req 2.2)
  Property 6: classify_expressibility correct for f ∈ [0,1] (Req 2.6)
  Property 7: classify_flow_architecture respects all sub-criteria (Req 3.2, 3.6, 3.7, 3.8)
  Property 8: RejectionReport always contains all required fields (Req 4.5)
  Property 9: ExtensionPriorityRanker produces valid total order (Req 5.8)
  Property 10: CalibrationComparator returns near-zero error for perfect calibration (Req 3.3)
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qmbp_simulation.analysis.extension_classifiers import (
    EXACT_DIAG_N_CEILING,
    ClassificationEngine,
)
from qmbp_simulation.analysis.extension_models import (
    ExtensionClassification,
)
from qmbp_simulation.analysis.extension_ranker import ExtensionPriorityRanker, ExtensionScore

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

positive_int = st.integers(min_value=1, max_value=10_000_000)
n_sites_st = st.integers(min_value=1, max_value=30)
cx_count_st = st.integers(min_value=0, max_value=200)
fidelity_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# ΔE/gap values in physically plausible range [0, 1]
de_gap_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Flow architecture inputs
calib_improvement_st = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False)
n_params_st = st.integers(min_value=1, max_value=100_000)
n_data_st = st.integers(min_value=1, max_value=500)

# Ranking score component strategies
impact_st = st.floats(min_value=0.0, max_value=3.0, allow_nan=False)
risk_st = st.floats(min_value=0.0, max_value=3.0, allow_nan=False)
time_st = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)


# ===========================================================================
# Property 1: classify_cross_n — Validates: Requirements 1.3
# ===========================================================================


@given(st.lists(de_gap_st, min_size=1, max_size=10))
@settings(max_examples=300)
def test_property_1_classify_cross_n_all_above_threshold(de_gap_list):
    """**Validates: Requirements 1.3**

    Property 1: If ALL values ≥ 0.05 → REJECTED_INSUFFICIENT_DATA.
    """
    values = [max(v, 0.05) for v in de_gap_list]  # force all ≥ 0.05
    result = ClassificationEngine.classify_cross_n(values)
    assert result == ExtensionClassification.REJECTED_INSUFFICIENT_DATA, (
        f"Expected REJECTED_INSUFFICIENT_DATA for all-≥-0.05 list, got {result}"
    )


@given(st.lists(de_gap_st, min_size=1, max_size=9), de_gap_st)
@settings(max_examples=300)
def test_property_1_classify_cross_n_at_least_one_below(others, below):
    """**Validates: Requirements 1.3**

    Property 1: If at least one value < 0.05 → NOT REJECTED_INSUFFICIENT_DATA.
    """
    assume(below < 0.05)
    values = others + [below]
    result = ClassificationEngine.classify_cross_n(values)
    assert result != ExtensionClassification.REJECTED_INSUFFICIENT_DATA, (
        f"Expected non-rejected for list containing {below} < 0.05, got {result}"
    )


# ===========================================================================
# Property 2: N_min_data satisfies params/data ≤ 1000 — Validates: Req 1.5
# ===========================================================================


@given(positive_int)
@settings(max_examples=500)
def test_property_2_n_min_data_ratio_satisfied(n_params):
    """**Validates: Requirements 1.5**

    Property 2: n_params / N_min_data ≤ 1000 for all positive n_params.
    """
    n_min = ClassificationEngine.compute_n_min_data(n_params, ratio_threshold=1000)
    assert n_min >= 1, f"N_min_data must be at least 1, got {n_min}"
    ratio = n_params / n_min
    assert ratio <= 1000.0, f"Ratio {n_params}/{n_min} = {ratio:.2f} exceeds 1000 threshold"


@given(positive_int)
@settings(max_examples=200)
def test_property_2_n_min_data_is_minimal(n_params):
    """**Validates: Requirements 1.5**

    N_min_data should be the MINIMUM value satisfying the ratio constraint,
    i.e., (N_min - 1) should NOT satisfy it (or N_min_data = 1).
    """
    n_min = ClassificationEngine.compute_n_min_data(n_params, ratio_threshold=1000)
    assert n_min == math.ceil(n_params / 1000), (
        f"N_min_data={n_min} should equal ceil({n_params}/1000)={math.ceil(n_params / 1000)}"
    )


# ===========================================================================
# Property 3: classify_intra_n — Validates: Req 1.6
# ===========================================================================


@given(
    st.floats(min_value=0.0, max_value=0.01, allow_nan=False),
    st.integers(min_value=2, max_value=12),
)
@settings(max_examples=300)
def test_property_3_intra_n_conditionally_viable_when_both_conditions_met(de_gap, n_total):
    """**Validates: Requirements 1.6**

    Property 3: ΔE/gap ≤ 0.01 AND n_pass ≥ ceil(5*n_total/6) → CONDITIONALLY_VIABLE.
    Uses math.ceil to match the implementation.
    """
    n_pass = math.ceil(5 * n_total / 6)  # minimum required (matches impl with ceil)
    result = ClassificationEngine.classify_intra_n(de_gap, n_pass, n_total)
    assert result == ExtensionClassification.CONDITIONALLY_VIABLE, (
        f"Expected CONDITIONALLY_VIABLE for de_gap={de_gap:.4f}, "
        f"n_pass={n_pass}/{n_total} (ceil threshold={n_pass}), got {result}"
    )


@given(
    st.floats(min_value=0.011, max_value=1.0, allow_nan=False),
    st.integers(min_value=1, max_value=12),
)
@settings(max_examples=200)
def test_property_3_intra_n_rejected_when_de_gap_too_high(de_gap, n_total):
    """**Validates: Requirements 1.6**

    Property 3: ΔE/gap > 0.01 → NOT CONDITIONALLY_VIABLE (regardless of n_pass).
    """
    n_pass = n_total  # give maximum passes
    result = ClassificationEngine.classify_intra_n(de_gap, n_pass, n_total)
    assert result != ExtensionClassification.CONDITIONALLY_VIABLE, (
        f"Expected rejection for de_gap={de_gap:.4f} > 0.01, got {result}"
    )


@given(
    st.floats(min_value=0.0, max_value=0.01, allow_nan=False),
    st.integers(min_value=6, max_value=12),
)
@settings(max_examples=200)
def test_property_3_intra_n_rejected_when_n_pass_too_low(de_gap, n_total):
    """**Validates: Requirements 1.6**

    Property 3: n_pass < ceil(5*n_total/6) → NOT CONDITIONALLY_VIABLE
    (even if de_gap ≤ 0.01).

    Uses math.ceil to match the implementation (ceil is stricter than // for
    n_total not divisible by 6).
    """
    threshold = math.ceil(5 * n_total / 6)
    assume(threshold > 0)
    n_pass = threshold - 1  # one below threshold
    assume(n_pass >= 0)
    result = ClassificationEngine.classify_intra_n(de_gap, n_pass, n_total)
    assert result != ExtensionClassification.CONDITIONALLY_VIABLE, (
        f"Expected rejection for n_pass={n_pass} < ceil-threshold={threshold} "
        f"(n_total={n_total}), got {result}"
    )


# ===========================================================================
# Property 4: classify_hardware — Validates: Req 1.7, 2.4
# ===========================================================================


@given(st.integers(min_value=19, max_value=500))
@settings(max_examples=200)
def test_property_4_hardware_incompatible_above_threshold(cx_count):
    """**Validates: Requirements 1.7, 2.4**

    Property 4: cx_count > 18 → HARDWARE_INCOMPATIBLE.
    """
    result = ClassificationEngine.classify_hardware(cx_count)
    assert result == ExtensionClassification.HARDWARE_INCOMPATIBLE, (
        f"Expected HARDWARE_INCOMPATIBLE for cx_count={cx_count} > 18, got {result}"
    )


@given(st.integers(min_value=0, max_value=18))
@settings(max_examples=100)
def test_property_4_hardware_viable_at_or_below_threshold(cx_count):
    """**Validates: Requirements 1.7, 2.4**

    Property 4: cx_count ≤ 18 → VIABLE.
    """
    result = ClassificationEngine.classify_hardware(cx_count)
    assert result == ExtensionClassification.VIABLE, (
        f"Expected VIABLE for cx_count={cx_count} ≤ 18, got {result}"
    )


# ===========================================================================
# Property 5: hilbert_space_dimension = 2^N — Validates: Req 2.2
# ===========================================================================


@given(n_sites_st)
@settings(max_examples=200)
def test_property_5_hilbert_dimension_is_power_of_2(n_sites):
    """**Validates: Requirements 2.2**

    Property 5: For any positive N, result is exactly 2**N.
    """
    result = ClassificationEngine.hilbert_space_dimension(n_sites)
    expected = 2**n_sites
    assert result == expected, (
        f"hilbert_space_dimension({n_sites}) = {result} ≠ 2^{n_sites} = {expected}"
    )


def test_property_5_ceiling_warning_emitted_for_large_n(caplog):
    """**Validates: Requirements 2.2**

    Property 5: N > 18 emits a ceiling warning.
    Uses a fixed set of values > EXACT_DIAG_N_CEILING.
    """
    import logging

    for n_sites in [19, 20, 24, 30]:
        with caplog.at_level(
            logging.WARNING, logger="qmbp_simulation.analysis.extension_classifiers"
        ):
            ClassificationEngine.hilbert_space_dimension(n_sites)
        assert any(
            "ceiling" in rec.message.lower() or "exceed" in rec.message.lower()
            for rec in caplog.records
        ), f"Expected ceiling warning for N={n_sites} > {EXACT_DIAG_N_CEILING}"
        caplog.clear()


# ===========================================================================
# Property 6: classify_expressibility — Validates: Req 2.6
# ===========================================================================


@given(st.floats(min_value=0.0, max_value=0.5999, allow_nan=False))
@settings(max_examples=300)
def test_property_6_expressibility_insufficient_below_threshold(fidelity):
    """**Validates: Requirements 2.6**

    Property 6: f < 0.60 → EXPRESSIBILITY_INSUFFICIENT.
    """
    result = ClassificationEngine.classify_expressibility(fidelity)
    assert result == ExtensionClassification.EXPRESSIBILITY_INSUFFICIENT, (
        f"Expected EXPRESSIBILITY_INSUFFICIENT for f={fidelity:.4f} < 0.60, got {result}"
    )


@given(st.floats(min_value=0.60, max_value=1.0, allow_nan=False))
@settings(max_examples=200)
def test_property_6_viable_at_or_above_threshold(fidelity):
    """**Validates: Requirements 2.6**

    Property 6: f ≥ 0.60 → VIABLE.
    """
    result = ClassificationEngine.classify_expressibility(fidelity)
    assert result == ExtensionClassification.VIABLE, (
        f"Expected VIABLE for f={fidelity:.4f} ≥ 0.60, got {result}"
    )


# ===========================================================================
# Property 7: classify_flow_architecture — Validates: Req 3.2, 3.6, 3.7, 3.8
# ===========================================================================


@given(
    calib_improvement_st,
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.integers(min_value=5001, max_value=100_000),
    st.integers(min_value=1, max_value=49),
)
@settings(max_examples=300)
def test_property_7_overparameterized_takes_priority(calib_improvement, de_gap, n_params, n_data):
    """**Validates: Requirements 3.2, 3.6, 3.7, 3.8**

    Property 7: n_params > 5000 AND n_data < 50 → OVERPARAMETERIZED_FOR_DATASET,
    regardless of other inputs.
    """
    result = ClassificationEngine.classify_flow_architecture(
        calibration_improvement=calib_improvement,
        de_gap=de_gap,
        n_params=n_params,
        n_data=n_data,
    )
    assert result == ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET, (
        f"Expected OVERPARAMETERIZED for n_params={n_params}, n_data={n_data}, got {result}"
    )


@given(
    calib_improvement_st,
    st.floats(min_value=0.10, max_value=1.0, allow_nan=False),
    st.integers(min_value=1, max_value=5000),
    st.integers(min_value=1, max_value=500),
)
@settings(max_examples=200)
def test_property_7_degraded_when_de_gap_high(calib_improvement, de_gap, n_params, n_data):
    """**Validates: Requirements 3.2, 3.6, 3.7, 3.8**

    Property 7: de_gap ≥ 0.10 (and not overparameterized) → DEGRADED_VS_BASELINE.
    """
    result = ClassificationEngine.classify_flow_architecture(
        calibration_improvement=calib_improvement,
        de_gap=de_gap,
        n_params=n_params,
        n_data=n_data,
    )
    assert result == ExtensionClassification.DEGRADED_VS_BASELINE, (
        f"Expected DEGRADED_VS_BASELINE for de_gap={de_gap:.4f} ≥ 0.10, got {result}"
    )


@given(
    st.floats(min_value=0.02, max_value=0.5, allow_nan=False),
    st.floats(min_value=0.0, max_value=0.0499, allow_nan=False),
    st.integers(min_value=1, max_value=5000),
    st.integers(min_value=50, max_value=500),
)
@settings(max_examples=200)
def test_property_7_viable_when_improvement_and_low_de_gap(
    calib_improvement, de_gap, n_params, n_data
):
    """**Validates: Requirements 3.2, 3.6, 3.7, 3.8**

    Property 7: calib_improvement ≥ 0.02 AND de_gap < 0.05 (not overparameterized,
    not degraded) → VIABLE.
    """
    result = ClassificationEngine.classify_flow_architecture(
        calibration_improvement=calib_improvement,
        de_gap=de_gap,
        n_params=n_params,
        n_data=n_data,
    )
    assert result == ExtensionClassification.VIABLE, (
        f"Expected VIABLE for calib_impr={calib_improvement:.3f}, de_gap={de_gap:.4f}, got {result}"
    )


@given(
    st.floats(min_value=-0.5, max_value=0.0199, allow_nan=False),
    st.floats(min_value=0.0, max_value=0.0999, allow_nan=False),
    st.integers(min_value=1, max_value=5000),
    st.integers(min_value=50, max_value=500),
)
@settings(max_examples=200)
def test_property_7_conditionally_viable_otherwise(calib_improvement, de_gap, n_params, n_data):
    """**Validates: Requirements 3.2, 3.6, 3.7, 3.8**

    Property 7: All other combinations (no overparameterization, not degraded,
    not clearly viable) → CONDITIONALLY_VIABLE.
    """
    result = ClassificationEngine.classify_flow_architecture(
        calibration_improvement=calib_improvement,
        de_gap=de_gap,
        n_params=n_params,
        n_data=n_data,
    )
    assert result == ExtensionClassification.CONDITIONALLY_VIABLE, (
        f"Expected CONDITIONALLY_VIABLE for calib_impr={calib_improvement:.3f}, "
        f"de_gap={de_gap:.4f}, got {result}"
    )


# ===========================================================================
# Property 8: RejectionReport always contains all required fields — Req 4.5
# ===========================================================================


from qmbp_simulation.analysis.extension_analyzer import RejectionReportGenerator

REJECTION_CLASSES_LIST = [
    ExtensionClassification.REJECTED_INSUFFICIENT_DATA,
    ExtensionClassification.HARDWARE_INCOMPATIBLE,
    ExtensionClassification.EXPRESSIBILITY_INSUFFICIENT,
    ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET,
    ExtensionClassification.DEGRADED_VS_BASELINE,
    ExtensionClassification.HARD_PHYSICS_LIMIT,
    ExtensionClassification.PREREQUISITE_FAILED,
]

criterion_id_st = st.from_regex(r"[1-5]\.[1-9][0-9]?", fullmatch=True)
measured_value_st = st.one_of(
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    st.text(min_size=1, max_size=50),
)
threshold_st = measured_value_st
narrative_st = st.text(min_size=1, max_size=200)
ext_id_st = st.sampled_from(["ext1", "ext2", "ext3"])
classification_st = st.sampled_from(REJECTION_CLASSES_LIST)


@given(
    ext_id_st,
    criterion_id_st,
    narrative_st,
    threshold_st,
    measured_value_st,
    classification_st,
)
@settings(max_examples=200)
def test_property_8_rejection_report_all_fields_nonempty(
    extension_id, criterion_id, narrative, threshold, measured_value, classification
):
    """**Validates: Requirements 4.5**

    Property 8: Every rejection event produces a RejectionReport with
    non-empty criterion_id, non-None classification, non-None measured_value,
    non-None threshold, and non-empty narrative.
    """
    report = RejectionReportGenerator.generate(
        extension_id=extension_id,
        criterion_id=criterion_id,
        criterion_description="Test criterion description",
        classification=classification,
        measured_value=measured_value,
        threshold=threshold,
        narrative=narrative,
    )

    assert report.criterion_id, f"criterion_id must not be empty, got {report.criterion_id!r}"
    assert report.classification is not None, "classification must not be None"
    assert report.measured_value is not None, "measured_value must not be None"
    assert report.threshold is not None, "threshold must not be None"
    assert report.narrative, f"narrative must not be empty, got {report.narrative!r}"
    assert report.extension_id, f"extension_id must not be empty, got {report.extension_id!r}"
    assert report.timestamp, f"timestamp must not be empty, got {report.timestamp!r}"

    # Verify round-trip serialization
    d = report.to_dict()
    assert d["classification"] == classification.value
    assert d["criterion_id"] == criterion_id


# ===========================================================================
# Property 9: ExtensionPriorityRanker produces valid total order — Req 5.8
# ===========================================================================


@st.composite
def three_distinct_scores(draw):
    """Generate 3 ExtensionScore with distinct narrative_impact values."""
    impacts = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=3.0, allow_nan=False),
            min_size=3,
            max_size=3,
            unique=True,
        )
    )
    assume(len(set(impacts)) == 3)
    scores = []
    for i, (eid, impact) in enumerate(zip(["ext1", "ext2", "ext3"], impacts, strict=False)):
        scores.append(
            ExtensionScore(
                extension_id=eid,
                narrative_impact=impact,
                implementation_risk=draw(st.floats(min_value=0.0, max_value=3.0, allow_nan=False)),
                time_to_result=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
            )
        )
    return scores


@given(three_distinct_scores())
@settings(max_examples=300)
def test_property_9_ranking_is_permutation(scores):
    """**Validates: Requirements 5.8**

    Property 9: rank() returns a permutation of all three extension IDs.
    """
    ranker = ExtensionPriorityRanker()
    ranking = ranker.rank(scores)
    assert set(ranking) == {"ext1", "ext2", "ext3"}, (
        f"Ranking must contain exactly ext1/ext2/ext3, got {ranking}"
    )
    assert len(ranking) == 3, f"Ranking must have 3 elements, got {len(ranking)}"


@given(three_distinct_scores())
@settings(max_examples=300)
def test_property_9_ranking_is_ordered_by_narrative_impact(scores):
    """**Validates: Requirements 5.8**

    Property 9: With distinct narrative_impact values, higher impact → earlier rank.
    """
    ranker = ExtensionPriorityRanker()
    ranking = ranker.rank(scores)

    score_map = {s.extension_id: s for s in scores}
    for i in range(len(ranking) - 1):
        a = score_map[ranking[i]]
        b = score_map[ranking[i + 1]]
        assert a.narrative_impact >= b.narrative_impact, (
            f"Transitivity violated: {ranking[i]} (impact={a.narrative_impact:.3f}) "
            f"ranked before {ranking[i + 1]} (impact={b.narrative_impact:.3f})"
        )


@given(
    st.floats(min_value=0.0, max_value=3.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=2.9999, allow_nan=False),
    st.floats(min_value=0.0, max_value=2.9999, allow_nan=False),
)
@settings(max_examples=200)
def test_property_9_tiebreak_by_risk(impact, risk_low, risk_high):
    """**Validates: Requirements 5.8**

    Property 9: Equal narrative_impact → lower implementation_risk (higher score) ranks first.
    (implementation_risk score: 3=low_risk preferred → higher is better)
    """
    assume(risk_high < 3.0)
    assume(risk_low < risk_high)

    scores = [
        ExtensionScore(
            "ext1", narrative_impact=impact, implementation_risk=risk_high, time_to_result=1.0
        ),
        ExtensionScore(
            "ext2", narrative_impact=impact, implementation_risk=risk_low, time_to_result=1.0
        ),
        ExtensionScore("ext3", narrative_impact=0.0, implementation_risk=0.0, time_to_result=1.0),
    ]
    ranker = ExtensionPriorityRanker()
    ranking = ranker.rank(scores)

    # ext1 has higher risk_score (=risk_high) → ranks before ext2 (=risk_low)
    idx1 = ranking.index("ext1")
    idx2 = ranking.index("ext2")
    assert idx1 < idx2, (
        f"ext1 (risk={risk_high:.3f}) should rank before ext2 (risk={risk_low:.3f}) "
        f"when narrative_impact is equal. Got ranking: {ranking}"
    )


# ===========================================================================
# Property 10: CalibrationComparator returns near-zero error for perfect
#              calibration — Validates: Req 3.3
# ===========================================================================

from qmbp_simulation.analysis.extension_analyzer import CalibrationComparator


def test_property_10_calibration_error_near_zero_for_perfect_model():
    """**Validates: Requirements 3.3**

    Property 10: A perfectly calibrated model produces |coverage_90 - 0.90| < 0.05.

    Construction: For each test point we draw n_samples from N(0,1) and use an
    INDEPENDENT draw from N(0,1) as the true value. By construction the true value
    is exchangeable with the samples → empirical 90% coverage ≈ 90%.

    Using a fixed large n_points=200 and n_samples=500 for statistical stability.
    Multiple seeds are tested to ensure no seed-specific artifact.
    """
    import random

    for seed in [42, 7, 137]:
        rng = random.Random(seed)
        n_points = 200
        n_samples = 500

        samples_list = []
        true_values = []
        for _ in range(n_points):
            draws = [rng.gauss(0.0, 1.0) for _ in range(n_samples + 1)]
            samples_list.append(draws[:n_samples])
            true_values.append(draws[-1])

        empirical_coverage = CalibrationComparator.compute_empirical_coverage(
            samples=samples_list,
            true_values=true_values,
            nominal=0.90,
        )

        cal_error = CalibrationComparator.calibration_error(empirical_coverage, nominal=0.90)

        assert cal_error < 0.05, (
            f"Expected calibration error < 0.05 for seed={seed}, "
            f"got {cal_error:.4f} (empirical coverage = {empirical_coverage:.4f})"
        )
