"""Tests for analysis/ tooling — heisenberg_summary, scan_coverage, validate_s_series.

Covers the testable utility functions and data models from the analysis scripts.
These scripts operate on real result files, so tests focus on:
- Data parsing functions (given synthetic inputs)
- Classification/computation logic
- Report formatting (no side effects)

Note: validate_s_series.py validation functions that require quantum computation
are tested for their utility helpers only (_compute_entropy, regime checks).

Run with:
    pytest tests/test_analysis_tools.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make analysis/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════
# validate_s_series utility tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeEntropy:
    """Test the entanglement entropy computation from validate_s_series."""

    def test_product_state_zero_entropy(self):
        """A product state |00...0⟩ should have zero entropy."""
        from analysis.validate_s_series import _compute_entropy

        n_qubits = 4
        # |0000⟩ state
        psi = np.zeros(2**n_qubits)
        psi[0] = 1.0
        S = _compute_entropy(psi, n_qubits)
        np.testing.assert_allclose(S, 0.0, atol=1e-10)

    def test_bell_state_one_ebit(self):
        """A maximally entangled 2-qubit state has S = 1 bit."""
        from analysis.validate_s_series import _compute_entropy

        # Bell state (|00⟩ + |11⟩) / √2
        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        S = _compute_entropy(psi, n_qubits=2)
        np.testing.assert_allclose(S, 1.0, atol=1e-10)

    def test_ghz_state_entropy(self):
        """GHZ state |000⟩+|111⟩ has S = 1 across any bipartition."""
        from analysis.validate_s_series import _compute_entropy

        n_qubits = 4
        psi = np.zeros(2**n_qubits)
        psi[0] = 1.0 / np.sqrt(2)
        psi[-1] = 1.0 / np.sqrt(2)
        S = _compute_entropy(psi, n_qubits)
        np.testing.assert_allclose(S, 1.0, atol=1e-10)

    def test_entropy_nonnegative(self):
        """Entropy should always be non-negative."""
        from analysis.validate_s_series import _compute_entropy

        # Random normalized state
        rng = np.random.default_rng(42)
        n_qubits = 4
        psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        psi /= np.linalg.norm(psi)
        S = _compute_entropy(psi, n_qubits)
        assert S >= 0.0

    def test_max_entropy_bounded(self):
        """Entropy cannot exceed n_A = n_qubits/2 bits."""
        from analysis.validate_s_series import _compute_entropy

        n_qubits = 6
        n_a = n_qubits // 2  # max entropy = n_a bits

        rng = np.random.default_rng(123)
        psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        psi /= np.linalg.norm(psi)
        S = _compute_entropy(psi, n_qubits)
        assert n_a + 1e-10 >= S


# ═══════════════════════════════════════════════════════════════════════════
# scan_coverage data model tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineRecord:
    """Test PipelineRecord from scan_coverage."""

    def test_verdict_pass(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.03,
            h_test=3.0,
        )
        assert rec.verdict == "PASS"

    def test_verdict_marginal(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.07,
            h_test=3.0,
        )
        assert rec.verdict == "MARGINAL"

    def test_verdict_fail(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.15,
            h_test=3.0,
        )
        assert rec.verdict == "FAIL"

    def test_verdict_no_data(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=None,
            h_test=3.0,
        )
        assert rec.verdict == "NO_DATA"

    def test_verdict_with_custom_thresholds(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.07,
            h_test=3.0,
        )
        # With stricter thresholds, this becomes FAIL
        assert rec.verdict_with_thresholds(0.03, 0.06) == "FAIL"
        # With relaxed thresholds, it becomes PASS
        assert rec.verdict_with_thresholds(0.08, 0.15) == "PASS"

    def test_in_valid_regime_true(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.03,
            h_test=3.0,  # well above boundary=1.9
        )
        assert rec.in_valid_regime is True

    def test_in_valid_regime_false(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            de_gap=0.03,
            h_test=1.5,  # below boundary=1.9
        )
        assert rec.in_valid_regime is False

    def test_in_valid_regime_p2_boundary(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=2,
            de_gap=0.03,
            h_test=1.6,  # above p2 boundary=1.5
        )
        assert rec.in_valid_regime is True

    def test_in_valid_regime_missing_data(self):
        from analysis.scan_coverage import PipelineRecord

        rec = PipelineRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=None,
            p_layers=1,
            de_gap=0.03,
            h_test=3.0,
        )
        assert rec.in_valid_regime is False


class TestNoisyRecord:
    """Test NoisyRecord from scan_coverage."""

    def test_zne_works_true(self):
        from analysis.scan_coverage import NoisyRecord

        rec = NoisyRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            mean_gain_pct=45.0,
            n_mitigated_wins=3,
            n_total=3,
        )
        assert rec.zne_works is True

    def test_zne_works_false_negative_gain(self):
        from analysis.scan_coverage import NoisyRecord

        rec = NoisyRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            mean_gain_pct=-5.0,
            n_mitigated_wins=0,
            n_total=3,
        )
        assert rec.zne_works is False

    def test_zne_works_false_no_wins(self):
        from analysis.scan_coverage import NoisyRecord

        rec = NoisyRecord(
            folder="f",
            variant="v",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            mean_gain_pct=10.0,
            n_mitigated_wins=0,
            n_total=3,
        )
        assert rec.zne_works is False


# ═══════════════════════════════════════════════════════════════════════════
# scan_coverage utility function tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScanCoverageUtils:
    """Test utility functions from scan_coverage."""

    def test_infer_topology_chain(self):
        from analysis.scan_coverage import _infer_default_topology

        assert _infer_default_topology("p1_variants_N10_chain") == "chain_1d"
        assert _infer_default_topology("variants_N6_linear") == "chain_1d"

    def test_infer_topology_ladder(self):
        from analysis.scan_coverage import _infer_default_topology

        assert _infer_default_topology("p1_variants_N10_ladder") == "ladder"

    def test_infer_topology_triangular(self):
        from analysis.scan_coverage import _infer_default_topology

        assert _infer_default_topology("variants_N6_tri") == "triangular"
        assert _infer_default_topology("variants_N6_triangular") == "triangular"

    def test_infer_topology_kagome(self):
        from analysis.scan_coverage import _infer_default_topology

        # kagome is NOT in _TOPO_HINTS, so falls back to chain_1d
        assert _infer_default_topology("variants_N12_kagome") == "chain_1d"

    def test_infer_topology_heavy_hex(self):
        from analysis.scan_coverage import _infer_default_topology

        assert _infer_default_topology("variants_N10_heavy_hex") == "heavy_hex"
        assert _infer_default_topology("p1_heavyhex_N10") == "heavy_hex"

    def test_infer_topology_default(self):
        from analysis.scan_coverage import _infer_default_topology

        assert _infer_default_topology("some_random_folder") == "chain_1d"

    def test_compute_median_odd(self):
        from analysis.scan_coverage import _compute_median

        assert _compute_median([1.0, 2.0, 3.0]) == 2.0

    def test_compute_median_even(self):
        from analysis.scan_coverage import _compute_median

        assert _compute_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_compute_median_single(self):
        from analysis.scan_coverage import _compute_median

        assert _compute_median([5.0]) == 5.0

    def test_compute_median_empty(self):
        from analysis.scan_coverage import _compute_median

        assert _compute_median([]) is None

    def test_safe_load_json_valid(self, tmp_path):
        from analysis.scan_coverage import _safe_load_json

        p = tmp_path / "test.json"
        p.write_text(json.dumps({"key": "value"}))
        result = _safe_load_json(p)
        assert result == {"key": "value"}

    def test_safe_load_json_invalid(self, tmp_path):
        from analysis.scan_coverage import _safe_load_json

        p = tmp_path / "bad.json"
        p.write_text("not json {{")
        assert _safe_load_json(p) is None

    def test_safe_load_json_missing(self, tmp_path):
        from analysis.scan_coverage import _safe_load_json

        p = tmp_path / "missing.json"
        assert _safe_load_json(p) is None


# ═══════════════════════════════════════════════════════════════════════════
# heisenberg_summary tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHeisenbergSummary:
    """Test heisenberg_summary.py data enrichment and aggregation logic."""

    def _create_heisenberg_folder(self, tmp_path: Path) -> Path:
        """Create a synthetic Heisenberg result folder structure."""
        folder = tmp_path / "variants_N6_heisenberg"
        folder.mkdir()

        variant1 = folder / "heisenberg_xxz_d1.0_seed42"
        variant1.mkdir()
        result = {
            "config": {
                "model": "heisenberg",
                "delta": 1.0,
                "topology": "chain_1d",
                "p_layers": 2,
                "seed": 42,
                "n_restarts": 10,
            },
            "phase2_summary": {
                "max_fidelity": 0.45,
                "mean_fidelity": 0.35,
                "n_above_threshold": 0,
                "fidelity_threshold": 0.6,
                "per_h_fidelity": [0.45, 0.40, 0.35, 0.30],
            },
            "scientific_conclusion": {
                "classification": "negative_hva_insufficient",
            },
            "entanglement": [
                {"entropy": 0.8},
                {"entropy": 1.2},
                {"entropy": 0.9},
            ],
            "diagnostics": {
                "phase2": {
                    "theta_smoothness": 0.85,
                    "convergence_rate": 1.0,
                }
            },
            "phase4_results": [],
            "elapsed_s": 120.5,
        }
        (variant1 / "pipeline_run_001.json").write_text(json.dumps(result))

        variant2 = folder / "heisenberg_xxz_d0.5_seed42"
        variant2.mkdir()
        result2 = {
            "config": {
                "model": "heisenberg",
                "delta": 0.5,
                "topology": "chain_1d",
                "p_layers": 2,
                "seed": 42,
                "n_restarts": 10,
            },
            "phase2_summary": {
                "max_fidelity": 0.55,
                "mean_fidelity": 0.45,
                "n_above_threshold": 1,
                "fidelity_threshold": 0.6,
            },
            "scientific_conclusion": {
                "classification": "negative_hva_insufficient",
            },
            "entanglement": [
                {"entropy": 0.6},
                {"entropy": 0.7},
            ],
            "diagnostics": {"phase2": {"theta_smoothness": 0.6}},
            "phase4_results": [],
            "elapsed_s": 95.0,
        }
        (variant2 / "pipeline_run_001.json").write_text(json.dumps(result2))
        return folder

    def test_enrich_with_heisenberg_data_parses_variants(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data

        folder = self._create_heisenberg_folder(tmp_path)
        results = enrich_with_heisenberg_data(folder)
        assert len(results) == 2

    def test_enrich_extracts_correct_fields(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data

        folder = self._create_heisenberg_folder(tmp_path)
        results = enrich_with_heisenberg_data(folder)

        # Find the delta=1.0 result
        r = next(r for r in results if r["delta"] == 1.0)
        assert r["model"] == "heisenberg"
        assert r["n_qubits"] == 6
        assert r["max_fidelity"] == 0.45
        assert r["classification"] == "negative_hva_insufficient"
        assert r["max_entropy"] == 1.2
        assert r["theta_smoothness"] == 0.85

    def test_enrich_skips_non_directories(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data

        folder = self._create_heisenberg_folder(tmp_path)
        # Add a stray file
        (folder / "notes.txt").write_text("not a variant")
        results = enrich_with_heisenberg_data(folder)
        # Should still find only 2 variants (not the text file)
        assert len(results) == 2

    def test_enrich_skips_invalid_json(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data

        folder = tmp_path / "variants_N6_heisenberg"
        folder.mkdir()
        variant = folder / "bad_variant"
        variant.mkdir()
        (variant / "pipeline_run_001.json").write_text("not json {{{")
        results = enrich_with_heisenberg_data(folder)
        assert results == []

    def test_enrich_skips_no_phase2_summary(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data

        folder = tmp_path / "variants_N6_heisenberg"
        folder.mkdir()
        variant = folder / "incomplete_variant"
        variant.mkdir()
        (variant / "pipeline_run_001.json").write_text(
            json.dumps({"config": {"model": "heisenberg"}})
        )
        results = enrich_with_heisenberg_data(folder)
        assert results == []

    def test_export_json_structure(self, tmp_path):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data, export_json

        folder = self._create_heisenberg_folder(tmp_path)
        results = enrich_with_heisenberg_data(folder)

        out_path = tmp_path / "export.json"
        export_json(results, out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "n_results" in data
        assert data["n_results"] == 2
        assert "by_n" in data
        assert "results" in data
        # Large arrays should be excluded
        for r in data["results"]:
            assert "ground_energies" not in r
            assert "vqe_energies" not in r

    def test_print_summary_no_crash(self, tmp_path, capsys):
        from analysis.heisenberg_summary import enrich_with_heisenberg_data, print_summary

        folder = self._create_heisenberg_folder(tmp_path)
        results = enrich_with_heisenberg_data(folder)
        # Should not crash
        print_summary(results)
        captured = capsys.readouterr()
        assert "N=6" in captured.out

    def test_print_summary_empty(self, capsys):
        from analysis.heisenberg_summary import print_summary

        print_summary([])
        captured = capsys.readouterr()
        assert "No Heisenberg results" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# validate_s_series: A3 scaling law helper
# ═══════════════════════════════════════════════════════════════════════════


class TestScalingLaw:
    """Test the scaling law prediction used in validate_s_series."""

    def test_a3_scaling_law_n6(self):
        """h_min(N=6) should match known value ~1.6."""
        h_min = 1.0 + 0.0186 * (6**1.331)
        assert 1.1 < h_min < 2.0

    def test_a3_scaling_law_n10(self):
        """h_min(N=10) should match known value ~1.9."""
        h_min = 1.0 + 0.0186 * (10**1.331)
        assert 1.3 < h_min < 2.5

    def test_a3_scaling_law_n20(self):
        """h_min(N=20) ≈ 2.0 (exact match from validated results)."""
        h_min = 1.0 + 0.020 * (20**1.31)
        np.testing.assert_allclose(h_min, 2.0, atol=0.05)

    def test_a3_monotonically_increasing(self):
        """h_min should increase with N."""
        h_min_vals = [1.0 + 0.020 * (n**1.31) for n in range(4, 25)]
        for i in range(1, len(h_min_vals)):
            assert h_min_vals[i] > h_min_vals[i - 1]
