"""Property-based tests for MitigationBenchmarkAnalyzer.

# Feature: mitigation-benchmark, Properties 12, 13, 21
# **Validates: Requirements 8.5, 13.1, 13.3, 13.4, 18.2**
#
# Property 12: Derived metrics formulas correctness
#   improvement_vs_raw, precision_per_shot, net_benefit match formulas.
#
# Property 13: Pareto frontier non-dominance
#   No config on the frontier is dominated by any other config.
#
# Property 21: Hypothesis verdicts are valid enum values
#   verdict ∈ {"CONFIRMED", "REFUTED", "INCONCLUSIVE"} for all hypotheses.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add project root to path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from project_health.analysis.mitigation_benchmark_analyzer import (
    HYPOTHESIS_MAP,
    MitigationBenchmarkAnalyzer,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Strategy for energy values: finite, non-extreme floats
energy_st = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for positive floats (shots, fidelity)
positive_float_st = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)

# Strategy for fidelity in [0, 1]
fidelity_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for delta_e_gap in [0, 1] (typical range)
delta_e_gap_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for total shots (positive integer)
shots_st = st.integers(min_value=1, max_value=10_000_000)

# Strategy for overhead factor (positive float >= 1.0)
overhead_st = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

# Strategy for generating a random config id
config_id_st = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
    min_size=2,
    max_size=20,
).map(lambda s: f"C{s}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def make_synthetic_entry(
    config_id: str,
    h_value: float,
    e_raw: float,
    e_mitigated: float,
    e_exact: float,
    delta_e_gap: float,
    shots: int = 16384,
    fidelity_estimate: float = 0.85,
) -> dict:
    """Create a synthetic result envelope for testing."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": h_value,
            "execution_mode": "fake_backend",
        },
        "circuit_stats": {
            "fidelity_estimate": fidelity_estimate,
        },
        "results": {
            "e_raw": e_raw,
            "e_mitigated": e_mitigated,
            "e_exact": e_exact,
            "delta_e_gap": delta_e_gap,
        },
        "shots": shots,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Property 12: Derived metrics formulas correctness
# **Validates: Requirements 13.1, 13.3, 13.4**
# ═══════════════════════════════════════════════════════════════════════════════


class TestDerivedMetricsFormulas:
    """Property 12: Derived metrics formulas correctness.

    **Validates: Requirements 13.1, 13.3, 13.4**

    For any valid inputs:
    - improvement_vs_raw = (e_raw - e_mitigated) / (e_raw - e_exact) when denom != 0
    - precision_per_shot = (1 - delta_e_gap) / total_shots
    - net_benefit = fidelity_estimate × (1 - delta_e_gap)
    """

    @given(
        e_raw=energy_st,
        e_mitigated=energy_st,
        e_exact=energy_st,
    )
    @settings(max_examples=200, deadline=None)
    def test_improvement_vs_raw_formula(self, e_raw: float, e_mitigated: float, e_exact: float):
        """improvement_vs_raw matches formula when denominator != 0."""
        denom = e_raw - e_exact
        assume(abs(denom) > 1e-12)

        # Build analyzer with synthetic entry
        analyzer = MitigationBenchmarkAnalyzer()
        entry = make_synthetic_entry(
            config_id="C0_raw",
            h_value=3.5,
            e_raw=e_raw,
            e_mitigated=e_raw,
            e_exact=e_exact,
            delta_e_gap=0.05,
            shots=16384,
        )
        # C0_raw is baseline — add a second config with mitigated energy
        entry_mit = make_synthetic_entry(
            config_id="C5_test",
            h_value=3.5,
            e_raw=e_raw,
            e_mitigated=e_mitigated,
            e_exact=e_exact,
            delta_e_gap=0.02,
            shots=16384,
        )
        analyzer.entries = [entry, entry_mit]
        metrics = analyzer.compute_derived_metrics()

        # Verify the formula for C5_test
        c5_metrics = metrics.get("C5_test")
        assert c5_metrics is not None

        if c5_metrics["improvement_vs_raw"] is not None:
            expected = (e_raw - e_mitigated) / denom
            assert abs(c5_metrics["improvement_vs_raw"] - expected) < 1e-10, (
                f"improvement_vs_raw={c5_metrics['improvement_vs_raw']}, expected={expected}"
            )

    @given(
        delta_e_gap=delta_e_gap_st,
        total_shots=shots_st,
    )
    @settings(max_examples=200, deadline=None)
    def test_precision_per_shot_formula(self, delta_e_gap: float, total_shots: int):
        """precision_per_shot = (1 - delta_e_gap) / total_shots."""
        analyzer = MitigationBenchmarkAnalyzer()
        entry = make_synthetic_entry(
            config_id="C0_raw",
            h_value=3.5,
            e_raw=-5.0,
            e_mitigated=-5.0,
            e_exact=-6.0,
            delta_e_gap=delta_e_gap,
            shots=total_shots,
        )
        analyzer.entries = [entry]
        metrics = analyzer.compute_derived_metrics()

        c0_metrics = metrics.get("C0_raw")
        assert c0_metrics is not None
        assert c0_metrics["precision_per_shot"] is not None

        expected = (1.0 - delta_e_gap) / total_shots
        assert abs(c0_metrics["precision_per_shot"] - expected) < 1e-10, (
            f"precision_per_shot={c0_metrics['precision_per_shot']}, expected={expected}"
        )

    @given(
        fidelity=fidelity_st,
        delta_e_gap=delta_e_gap_st,
    )
    @settings(max_examples=200, deadline=None)
    def test_net_benefit_formula(self, fidelity: float, delta_e_gap: float):
        """net_benefit = fidelity_estimate × (1 - delta_e_gap)."""
        analyzer = MitigationBenchmarkAnalyzer()
        entry = make_synthetic_entry(
            config_id="C0_raw",
            h_value=3.5,
            e_raw=-5.0,
            e_mitigated=-5.0,
            e_exact=-6.0,
            delta_e_gap=delta_e_gap,
            shots=16384,
            fidelity_estimate=fidelity,
        )
        analyzer.entries = [entry]
        metrics = analyzer.compute_derived_metrics()

        c0_metrics = metrics.get("C0_raw")
        assert c0_metrics is not None
        assert c0_metrics["net_benefit"] is not None

        expected = fidelity * (1.0 - delta_e_gap)
        assert abs(c0_metrics["net_benefit"] - expected) < 1e-10, (
            f"net_benefit={c0_metrics['net_benefit']}, expected={expected}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 13: Pareto frontier non-dominance
# **Validates: Requirements 8.5**
# ═══════════════════════════════════════════════════════════════════════════════


class TestParetoFrontierNonDominance:
    """Property 13: Pareto frontier non-dominance.

    **Validates: Requirements 8.5**

    For any set of configs with (overhead_factor, mean_delta_e_gap) pairs,
    no config on the Pareto frontier is dominated by any other config.
    A config is dominated if another has BOTH ≤ overhead AND ≤ gap with at
    least one strictly less.
    """

    @given(
        data=st.lists(
            st.tuples(
                config_id_st,
                overhead_st,  # overhead_factor
                delta_e_gap_st,  # mean_delta_e_gap
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_no_frontier_config_is_dominated(self, data):
        """No config on the Pareto frontier is dominated by any other."""
        # Ensure unique config IDs
        seen_ids = set()
        unique_data = []
        for cid, overhead, gap in data:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_data.append((cid, overhead, gap))
        assume(len(unique_data) >= 2)

        # Build analyzer with synthetic derived metrics (bypass entries)
        analyzer = MitigationBenchmarkAnalyzer()
        analyzer.derived_metrics = {}
        for cid, overhead, gap in unique_data:
            analyzer.derived_metrics[cid] = {
                "config_id": cid,
                "mean_delta_e_gap": gap,
                "overhead_factor": overhead,
                "improvement_vs_raw": None,
                "precision_per_shot": None,
                "net_benefit": None,
                "n_entries": 1,
            }

        frontier = analyzer.compute_pareto_frontier()

        # Verify non-dominance: for each frontier config, no other config
        # has BOTH lower-or-equal overhead AND lower-or-equal gap with at
        # least one strictly less
        for f_cid in frontier:
            f_oh = analyzer.derived_metrics[f_cid]["overhead_factor"]
            f_gap = analyzer.derived_metrics[f_cid]["mean_delta_e_gap"]

            for other_cid, other_metrics in analyzer.derived_metrics.items():
                if other_cid == f_cid:
                    continue
                o_oh = other_metrics["overhead_factor"]
                o_gap = other_metrics["mean_delta_e_gap"]

                # Check that other does NOT dominate frontier config
                if o_oh <= f_oh and o_gap <= f_gap:
                    assert not (o_oh < f_oh or o_gap < f_gap), (
                        f"Frontier config {f_cid} (oh={f_oh}, gap={f_gap}) "
                        f"is dominated by {other_cid} (oh={o_oh}, gap={o_gap})"
                    )

    @given(
        data=st.lists(
            st.tuples(
                config_id_st,
                overhead_st,
                delta_e_gap_st,
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_frontier_contains_all_non_dominated(self, data):
        """Every non-dominated config appears on the frontier."""
        seen_ids = set()
        unique_data = []
        for cid, overhead, gap in data:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_data.append((cid, overhead, gap))
        assume(len(unique_data) >= 2)

        analyzer = MitigationBenchmarkAnalyzer()
        analyzer.derived_metrics = {}
        for cid, overhead, gap in unique_data:
            analyzer.derived_metrics[cid] = {
                "config_id": cid,
                "mean_delta_e_gap": gap,
                "overhead_factor": overhead,
                "improvement_vs_raw": None,
                "precision_per_shot": None,
                "net_benefit": None,
                "n_entries": 1,
            }

        frontier = analyzer.compute_pareto_frontier()
        frontier_set = set(frontier)

        # Find all non-dominated configs manually
        for cid, overhead, gap in unique_data:
            dominated = False
            for other_cid, o_oh, o_gap in unique_data:
                if other_cid == cid:
                    continue
                if o_oh <= overhead and o_gap <= gap:
                    if o_oh < overhead or o_gap < gap:
                        dominated = True
                        break
            if not dominated:
                assert cid in frontier_set, (
                    f"Non-dominated config {cid} (oh={overhead}, gap={gap}) "
                    f"not found in frontier: {frontier}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 21: Hypothesis verdicts are valid enum values
# **Validates: Requirements 18.2**
# ═══════════════════════════════════════════════════════════════════════════════

VALID_VERDICTS = {"CONFIRMED", "REFUTED", "INCONCLUSIVE"}

# Strategy: pick pairs of config IDs from HYPOTHESIS_MAP
hypothesis_ids_st = st.sampled_from(list(HYPOTHESIS_MAP.keys()))


class TestHypothesisVerdictsValidEnum:
    """Property 21: Hypothesis verdicts are valid enum values.

    **Validates: Requirements 18.2**

    For any hypothesis evaluation computed by compute_hypothesis_verdicts(),
    the verdict field is exactly one of: "CONFIRMED", "REFUTED", "INCONCLUSIVE".
    """

    @given(
        entries_data=st.lists(
            st.tuples(
                st.sampled_from(
                    sorted(
                        set(
                            cid for h_def in HYPOTHESIS_MAP.values() for cid in h_def["config_pair"]
                        )
                    )
                ),
                st.floats(min_value=3.0, max_value=4.0, allow_nan=False, allow_infinity=False),
                delta_e_gap_st,
            ),
            min_size=2,
            max_size=30,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_all_verdicts_are_valid_enum(self, entries_data):
        """All hypothesis verdicts are in the valid set."""
        analyzer = MitigationBenchmarkAnalyzer()

        # Build synthetic entries for the config IDs referenced in HYPOTHESIS_MAP
        entries = []
        for config_id, h_value, delta_e_gap in entries_data:
            entry = make_synthetic_entry(
                config_id=config_id,
                h_value=h_value,
                e_raw=-5.0 + delta_e_gap,
                e_mitigated=-5.0,
                e_exact=-6.0,
                delta_e_gap=delta_e_gap,
                shots=16384,
            )
            entries.append(entry)

        analyzer.entries = entries
        verdicts = analyzer.compute_hypothesis_verdicts()

        # Every verdict must be valid
        for v in verdicts:
            assert "verdict" in v, f"Missing 'verdict' key in {v}"
            assert v["verdict"] in VALID_VERDICTS, (
                f"Hypothesis {v.get('hypothesis_id')}: "
                f"verdict '{v['verdict']}' not in {VALID_VERDICTS}"
            )

    def test_all_hypotheses_evaluated_with_full_data(self):
        """When data exists for all configs, all 19 hypotheses are evaluated."""
        analyzer = MitigationBenchmarkAnalyzer()

        # Generate entries for ALL config IDs referenced in HYPOTHESIS_MAP
        all_config_ids = sorted(
            set(cid for h_def in HYPOTHESIS_MAP.values() for cid in h_def["config_pair"])
        )
        entries = []
        for config_id in all_config_ids:
            for h_val in [3.25, 3.5, 3.75, 4.0]:
                entry = make_synthetic_entry(
                    config_id=config_id,
                    h_value=h_val,
                    e_raw=-4.5,
                    e_mitigated=-5.0,
                    e_exact=-6.0,
                    delta_e_gap=0.05,
                    shots=16384,
                )
                entries.append(entry)

        analyzer.entries = entries
        verdicts = analyzer.compute_hypothesis_verdicts()

        # All 19 hypotheses should be evaluated
        assert len(verdicts) == len(HYPOTHESIS_MAP), (
            f"Expected {len(HYPOTHESIS_MAP)} verdicts, got {len(verdicts)}"
        )

        # Every verdict is valid
        for v in verdicts:
            assert v["verdict"] in VALID_VERDICTS, (
                f"Invalid verdict '{v['verdict']}' for {v['hypothesis_id']}"
            )

    @given(h_id=hypothesis_ids_st)
    @settings(max_examples=50, deadline=None)
    def test_individual_hypothesis_verdict_is_valid(self, h_id: str):
        """Each individual hypothesis produces a valid verdict enum."""
        h_def = HYPOTHESIS_MAP[h_id]
        config_pair = h_def["config_pair"]

        analyzer = MitigationBenchmarkAnalyzer()
        entries = []
        for config_id in config_pair:
            for h_val in [3.25, 3.5, 3.75, 4.0]:
                entry = make_synthetic_entry(
                    config_id=config_id,
                    h_value=h_val,
                    e_raw=-4.5,
                    e_mitigated=-5.2,
                    e_exact=-6.0,
                    delta_e_gap=0.03,
                    shots=16384,
                )
                entries.append(entry)

        analyzer.entries = entries
        verdicts = analyzer.compute_hypothesis_verdicts()

        # Find the verdict for our hypothesis
        matching = [v for v in verdicts if v["hypothesis_id"] == h_id]
        assert len(matching) == 1, f"Expected exactly 1 verdict for {h_id}, got {len(matching)}"
        assert matching[0]["verdict"] in VALID_VERDICTS, (
            f"{h_id}: verdict '{matching[0]['verdict']}' not in {VALID_VERDICTS}"
        )
