"""Unit tests for comparative analysis module."""

from __future__ import annotations

from numpy.testing import assert_allclose

from qmbp_simulation.analysis.comparative import (
    classify_outcome,
    classify_result,
    compute_cx_budget,
    compute_staggered_magnetization,
    filter_by_threshold,
    find_h_min,
    find_minimum_viable_threshold,
)


class TestComparativeAnalysis:
    """Tests for comparative analysis helpers."""

    def test_compute_cx_budget_heisenberg(self):
        """Heisenberg CX budget = 3 × n_edges."""
        n_edges = 5
        assert compute_cx_budget(n_edges, "heisenberg") == 3 * n_edges

    def test_compute_cx_budget_tfim(self):
        """TFIM CX budget = 1 × n_edges."""
        n_edges = 5
        assert compute_cx_budget(n_edges, "tfim") == n_edges

    def test_compute_cx_budget_ratio(self):
        """Heisenberg/TFIM CX ratio = 3."""
        n_edges = 7
        heis = compute_cx_budget(n_edges, "heisenberg")
        tfim = compute_cx_budget(n_edges, "tfim")
        assert heis / tfim == 3

    def test_classify_result_negative(self):
        """max_fid < 0.60 → 'fundamental_expressibility_limitation'."""
        assert classify_result(0.55) == "fundamental_expressibility_limitation"

    def test_classify_result_partial(self):
        """0.60 ≤ max_fid < 0.93 → 'partial_expressibility'."""
        assert classify_result(0.75) == "partial_expressibility"

    def test_classify_result_viable(self):
        """max_fid ≥ 0.93 → 'viable_regime'."""
        assert classify_result(0.95) == "viable_regime"

    def test_classify_outcome_full_success(self):
        """ΔE/gap < 0.05 + valid regime → 'full_success'."""
        result = classify_outcome(delta_e_over_gap=0.03, has_valid_regime=True)
        assert result == "full_success"

    def test_classify_outcome_partial_success(self):
        """ΔE/gap ≥ 0.05 + valid regime → 'partial_success'."""
        result = classify_outcome(delta_e_over_gap=0.10, has_valid_regime=True)
        assert result == "partial_success"

    def test_classify_outcome_failure(self):
        """No valid regime → 'failure'."""
        result = classify_outcome(delta_e_over_gap=0.01, has_valid_regime=False)
        assert result == "failure"

    def test_staggered_magnetization_neel(self):
        """Néel state [+1, -1, +1, -1] → M_s = 1.0."""
        z_exp = [1.0, -1.0, 1.0, -1.0]
        m_s = compute_staggered_magnetization(z_exp)
        assert_allclose(m_s, 1.0, atol=1e-10)

    def test_staggered_magnetization_uniform(self):
        """Uniform state [+1, +1, +1, +1] → M_s = 0.0."""
        z_exp = [1.0, 1.0, 1.0, 1.0]
        m_s = compute_staggered_magnetization(z_exp)
        assert_allclose(m_s, 0.0, atol=1e-10)

    def test_staggered_magnetization_antiferro(self):
        """Anti-Néel [-1, 1, -1, 1] → M_s = -1.0."""
        z_exp = [-1.0, 1.0, -1.0, 1.0]
        m_s = compute_staggered_magnetization(z_exp)
        assert_allclose(m_s, -1.0, atol=1e-10)

    def test_filter_by_threshold(self):
        """Returns correct indices above threshold."""
        fidelities = [0.80, 0.95, 0.70, 0.99, 0.50]
        indices = filter_by_threshold(fidelities, 0.90)
        assert indices == [1, 3]

    def test_filter_by_threshold_empty(self):
        """Returns [] when nothing passes."""
        fidelities = [0.50, 0.60, 0.70]
        indices = filter_by_threshold(fidelities, 0.90)
        assert indices == []

    def test_find_minimum_viable_threshold_standard(self):
        """Finds highest threshold with ≥5 qualifying points."""
        # 6 points above 0.80, only 3 above 0.93
        fidelities = [0.95, 0.94, 0.93, 0.85, 0.82, 0.81, 0.60, 0.50]
        threshold = find_minimum_viable_threshold(fidelities, min_points=5)
        assert threshold == 0.80

    def test_find_minimum_viable_threshold_none(self):
        """Returns None when no threshold yields enough points."""
        fidelities = [0.55, 0.50, 0.45]
        threshold = find_minimum_viable_threshold(fidelities, min_points=5)
        assert threshold is None

    def test_find_h_min_found(self):
        """Returns correct h_min when criterion is met."""
        h_values = [2.0, 1.5, 1.0, 0.5]
        fidelities_by_seed = {
            42: [0.99, 0.95, 0.80, 0.50],
            43: [0.98, 0.94, 0.85, 0.55],
            44: [0.97, 0.93, 0.70, 0.40],
        }
        h_min = find_h_min(fidelities_by_seed, h_values, threshold=0.93, min_seeds=2)
        # At h=2.0 (idx 0): all 3 pass → first qualifying
        assert h_min == 2.0

    def test_find_h_min_not_found(self):
        """Returns None when no h qualifies."""
        h_values = [2.0, 1.5, 1.0]
        fidelities_by_seed = {
            42: [0.80, 0.70, 0.50],
            43: [0.75, 0.65, 0.45],
        }
        h_min = find_h_min(fidelities_by_seed, h_values, threshold=0.93, min_seeds=2)
        assert h_min is None
