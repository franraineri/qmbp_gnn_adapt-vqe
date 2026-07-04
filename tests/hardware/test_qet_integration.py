#!/usr/bin/env python3
"""Test QET (Quasi-probabilistic Error Tuning) integration.

Tests:
1. WLS extrapolation from synthetic noise-scale data
2. Parsing of noise_scaling from real QESEM metadata
3. QESEMResult dataclass with new QET fields
4. Edge cases (insufficient points, zero stds)
5. Parsing of QESEM heuristic from metadata
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation.execution.hardware.qesem import (
    QESEMResult,
    _parse_qesem_heuristic,
    _parse_qet_noise_scaling_results,
    extrapolate_qet_wls,
)


class TestExtrapolateQetWls:
    """Tests for WLS extrapolation from QET noise-scale data."""

    def test_linear_decay_model(self):
        """Linear noise model: E(s) = E_ideal + slope*s."""
        E_ideal = -40.57
        slope = 1.0
        scales = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        noise_data = {}
        for s in scales:
            true_val = E_ideal + slope * s
            std = 0.05 + 0.03 * s
            noise_data[s] = (true_val, std)

        e_extrap, std_extrap = extrapolate_qet_wls(noise_data, extrapolation_order=1)
        assert abs(e_extrap - E_ideal) < 1e-10
        assert std_extrap > 0

    def test_two_point_linear(self):
        """Minimum 2 points for linear extrapolation."""
        noise_data = {1.0: (-39.5, 0.06), 2.0: (-38.5, 0.10)}
        e, std = extrapolate_qet_wls(noise_data, extrapolation_order=1)
        assert abs(e - (-40.5)) < 1e-10

    def test_insufficient_points_raises(self):
        """Single point cannot extrapolate linearly."""
        with pytest.raises(ValueError, match="Need at least 2 points"):
            extrapolate_qet_wls({1.0: (-39.5, 0.06)}, extrapolation_order=1)

    def test_quadratic_extrapolation(self):
        """Order-2 extrapolation with 3+ points."""
        noise_data = {}
        for s in [0.5, 1.0, 1.5, 2.0]:
            val = -40.0 + 0.5 * s + 0.2 * s**2
            noise_data[s] = (val, 0.05)

        e, std = extrapolate_qet_wls(noise_data, extrapolation_order=2)
        assert abs(e - (-40.0)) < 1e-8

    def test_insufficient_points_for_quadratic(self):
        """Quadratic needs at least 3 points."""
        with pytest.raises(ValueError, match="Need at least 3 points"):
            extrapolate_qet_wls(
                {1.0: (-39.5, 0.06), 2.0: (-38.5, 0.10)},
                extrapolation_order=2,
            )

    def test_weighting_prefers_low_std_points(self):
        """Points with smaller std should have more influence."""
        data_weighted = {
            1.0: (-39.5, 0.01),
            2.0: (-38.0, 1.0),
        }
        e_w, _ = extrapolate_qet_wls(data_weighted, extrapolation_order=1)
        assert e_w < -39.5

    def test_zero_std_handling(self):
        """Zero std should not cause division by zero."""
        noise_data = {1.0: (-39.5, 0.0), 2.0: (-38.5, 0.0)}
        e, std = extrapolate_qet_wls(noise_data, extrapolation_order=1)
        assert np.isfinite(e)
        assert np.isfinite(std)


class TestParseQetNoiseScalingResults:
    """Tests for parsing noise_scaling from QESEM metadata."""

    def test_parse_real_metadata(self):
        """Parse noise_scaling from actual recovered QESEM job."""
        result_path = (
            _ROOT
            / "results/recovered"
            / "qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json"
        )
        if not result_path.exists():
            pytest.skip("Recovered QESEM result not available")

        with open(result_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        n_obs = 20

        scale_results = _parse_qet_noise_scaling_results(metadata, n_obs)
        assert len(scale_results) == n_obs

        energy_scales = scale_results[0]
        assert len(energy_scales) == 3
        assert 0.0 in energy_scales
        assert 1.0 in energy_scales
        assert 2.0 in energy_scales

        e_0, std_0 = energy_scales[0.0]
        assert -42 < e_0 < -38
        assert std_0 > 0

    def test_empty_metadata(self):
        """Empty metadata returns empty list."""
        assert _parse_qet_noise_scaling_results({}, 20) == []
        assert _parse_qet_noise_scaling_results({"results": None}, 20) == []
        assert _parse_qet_noise_scaling_results({"results": []}, 20) == []

    def test_malformed_metadata(self):
        """Malformed metadata doesn't crash."""
        assert _parse_qet_noise_scaling_results({"results": "bad"}, 20) == []
        assert _parse_qet_noise_scaling_results({"results": [None]}, 20) == []


class TestParseQesemHeuristic:
    """Tests for parsing QESEM heuristic from metadata."""

    def test_parse_real_heuristic(self):
        """Parse heuristic from actual QESEM job metadata."""
        result_path = (
            _ROOT
            / "results/recovered"
            / "qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json"
        )
        if not result_path.exists():
            pytest.skip("Recovered QESEM result not available")

        with open(result_path) as f:
            data = json.load(f)

        heur_e, heur_std = _parse_qesem_heuristic(data["metadata"])
        assert heur_e is not None
        assert heur_std is not None
        assert -42 < heur_e < -39
        assert heur_std > 0

    def test_missing_heuristic(self):
        """Missing heuristic returns (None, None)."""
        assert _parse_qesem_heuristic({}) == (None, None)
        assert _parse_qesem_heuristic({"results": []}) == (None, None)

    def test_null_heuristic(self):
        """When qesem_heuristic is null."""
        metadata = {"results": [[["obs_repr", {"qesem_heuristic": None}]]]}
        assert _parse_qesem_heuristic(metadata) == (None, None)


class TestQESEMResultDataclass:
    """Tests for QESEMResult with new QET fields."""

    def test_default_values(self):
        """New fields have sensible defaults."""
        r = QESEMResult(
            energy_mitigated=-40.5,
            energy_std=0.05,
            x_values=[0.98],
            zz_values=[0.12],
            x_stds=[0.01],
            zz_stds=[0.005],
            noisy_energy=-38.5,
            noisy_x_values=[0.93],
            noisy_zz_values=[0.10],
            job_id="test",
        )
        assert r.noise_scale_results is None
        assert r.extrapolation_method == "qesem_standard"
        assert r.qesem_heuristic_energy is None
        assert r.qesem_heuristic_std is None

    def test_qet_fields(self):
        """QET fields can be set."""
        scale_data = [{0.5: (-40.2, 0.04), 1.0: (-39.5, 0.06)}]
        r = QESEMResult(
            energy_mitigated=-40.9,
            energy_std=0.07,
            x_values=[0.99],
            zz_values=[0.12],
            x_stds=[0.01],
            zz_stds=[0.005],
            noisy_energy=-38.5,
            noisy_x_values=[0.93],
            noisy_zz_values=[0.10],
            job_id="qet-test",
            noise_scale_results=scale_data,
            extrapolation_method="qet_user_wls",
            qesem_heuristic_energy=-41.1,
            qesem_heuristic_std=0.5,
        )
        assert r.extrapolation_method == "qet_user_wls"
        assert r.noise_scale_results == scale_data
        assert r.qesem_heuristic_energy == -41.1


class TestHardwareConfigQET:
    """Tests for HardwareConfig with qesem_noise_scales field."""

    def test_default_none(self):
        """Default is None (standard QESEM flow)."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        c = HardwareConfig()
        assert c.qesem_noise_scales is None

    def test_explicit_scales(self):
        """Can set explicit noise scales."""
        from qmbp_simulation.execution.hardware.config import HardwareConfig

        scales = {0.5: 0.02, 1.0: 0.01, 1.5: 0.03}
        c = HardwareConfig(qesem_noise_scales=scales)
        assert c.qesem_noise_scales == scales
        assert c.qesem_noise_scales[0.5] == 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
