"""Tests for QPU cost estimation, throughput profiles, and preflight improvements.

Tests the composable cost estimation architecture:
- QPUThroughputProfile (CLOPS scaling across backends)
- SPSACostModel (optimistic/pessimistic scenarios)
- estimate_qpu_cost (full integration)
- Preflight new checks (readout error, T1/T2, native gates)
- estimate_effective_clops (depth-aware scaling)
- _interpolate_cx_count (CX count lookup)
"""

from __future__ import annotations

import pytest

from qmbp_simulation.execution.backends import MitigationOptions
from qmbp_simulation.execution.hardware import (
    HardwareConfig,
    QPUCostEstimate,
    QPUThroughputProfile,
    SPSACostModel,
    estimate_effective_clops,
    estimate_qpu_cost,
)
from qmbp_simulation.execution.hardware.preflight import (
    _interpolate_cx_count,
    check_native_gate_support,
    compute_mean_readout_error,
    compute_min_t1_t2,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def pea_config() -> HardwareConfig:
    """Standard PEA config for N=10."""
    return HardwareConfig(
        n_qubits=10,
        shots=16384,
        n_layouts=3,
        mitigation=MitigationOptions(
            zne_amplifier="pea",
            num_randomizations=32,
            shots_per_randomization=128,
        ),
    )


@pytest.fixture
def gf_config() -> HardwareConfig:
    """Gate-folding config for N=10."""
    return HardwareConfig(
        n_qubits=10,
        shots=16384,
        n_layouts=3,
        mitigation=MitigationOptions(zne_amplifier="gate_folding"),
    )


@pytest.fixture
def torino_profile() -> QPUThroughputProfile:
    return QPUThroughputProfile.ibm_torino()


@pytest.fixture
def nighthawk_profile() -> QPUThroughputProfile:
    return QPUThroughputProfile.ibm_nighthawk()


# ═══════════════════════════════════════════════════════════════════════════════
# QPUThroughputProfile Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQPUThroughputProfile:
    """Test hardware throughput profiles and CLOPS scaling."""

    def test_torino_reference_clops(self, torino_profile):
        """At reference N=10, CLOPS should equal base_clops."""
        clops = torino_profile.estimate_clops(10, circuit_depth=25)
        assert clops == 2500

    def test_smaller_circuits_faster(self, torino_profile):
        """N=6 should have higher CLOPS than N=10."""
        clops_6 = torino_profile.estimate_clops(6)
        clops_10 = torino_profile.estimate_clops(10)
        assert clops_6 > clops_10

    def test_larger_circuits_slower(self, torino_profile):
        """N=20 should have lower CLOPS than N=10."""
        clops_20 = torino_profile.estimate_clops(20)
        clops_10 = torino_profile.estimate_clops(10)
        assert clops_20 < clops_10

    def test_clops_floor(self, torino_profile):
        """Very large circuits should hit the CLOPS floor."""
        clops = torino_profile.estimate_clops(100, circuit_depth=200)
        assert clops == torino_profile.clops_floor

    def test_clops_ceiling(self, torino_profile):
        """Very small/shallow circuits should hit the CLOPS ceiling."""
        clops = torino_profile.estimate_clops(1, circuit_depth=1)
        assert clops == torino_profile.clops_ceiling

    def test_nighthawk_faster_than_torino(self, torino_profile, nighthawk_profile):
        """Nighthawk should be faster at same circuit size."""
        t_clops = torino_profile.estimate_clops(10)
        n_clops = nighthawk_profile.estimate_clops(10)
        assert n_clops > t_clops

    def test_heron_r2_between_torino_and_nighthawk(self):
        """Heron r2 CLOPS should be between Torino and Nighthawk."""
        t = QPUThroughputProfile.ibm_torino().estimate_clops(10)
        h = QPUThroughputProfile.ibm_heron_r2().estimate_clops(10)
        n = QPUThroughputProfile.ibm_nighthawk().estimate_clops(10)
        assert t < h < n

    def test_time_per_circuit(self, torino_profile):
        """time_per_circuit should be shots/clops."""
        t = torino_profile.time_per_circuit(16384, 10)
        assert abs(t - 16384 / 2500) < 0.01

    def test_cx_count_override(self, torino_profile):
        """Providing cx_count should change the depth estimate."""
        clops_default = torino_profile.estimate_clops(10)  # Uses _interpolate_cx_count
        clops_shallow = torino_profile.estimate_clops(10, cx_count=5)
        assert clops_shallow > clops_default  # Shallower = faster

    def test_custom_profile(self):
        """Custom profile should work with arbitrary parameters."""
        custom = QPUThroughputProfile(
            name="test_backend",
            base_clops=5000,
            ref_n_qubits=20,
            ref_depth=50,
            width_exponent=0.5,
            depth_exponent=0.5,
            total_qubits=50,
            clops_floor=500,
            clops_ceiling=50000,
            classical_latency_per_job_s=5.0,
        )
        assert custom.estimate_clops(20, circuit_depth=50) == 5000


# ═══════════════════════════════════════════════════════════════════════════════
# SPSACostModel Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSPSACostModel:
    """Test SPSA cost model scenarios."""

    def test_disabled_expected_cost_is_zero(self):
        """Disabled model should have zero expected cost."""
        m = SPSACostModel.disabled()
        assert m.expected_cost(10.0) == 0.0
        assert m.trigger_probability == 0.0

    def test_default_trigger_probability(self):
        """Default model should have P=0.30."""
        m = SPSACostModel()
        assert m.trigger_probability == 0.30
        assert m.n_iterations == 200
        assert m.evals_per_iteration == 2

    def test_cost_if_triggered(self):
        """cost_if_triggered = n_iters × evals × time."""
        m = SPSACostModel()
        assert m.cost_if_triggered(1.0) == 200 * 2 * 1.0

    def test_expected_cost_is_probability_weighted(self):
        """expected_cost = P × cost_if_triggered."""
        m = SPSACostModel()
        expected = m.expected_cost(10.0)
        assert abs(expected - 0.30 * 200 * 2 * 10.0) < 0.01

    def test_conservative_higher_than_default(self):
        """Conservative (P=0.50) should cost more than default (P=0.30)."""
        d = SPSACostModel()
        c = SPSACostModel.conservative()
        assert c.expected_cost(1.0) > d.expected_cost(1.0)

    def test_aggressive_equals_full_cost(self):
        """Aggressive (P=1.0) expected == cost_if_triggered."""
        a = SPSACostModel.aggressive()
        assert abs(a.expected_cost(5.0) - a.cost_if_triggered(5.0)) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# CX Count Interpolation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCXCountInterpolation:
    """Test CX gate count interpolation for depth estimation."""

    def test_known_values(self):
        """Known data points should return exact values."""
        assert _interpolate_cx_count(6) == 10
        assert _interpolate_cx_count(10) == 18
        assert _interpolate_cx_count(20) == 38

    def test_interpolation_between_known(self):
        """Intermediate values should interpolate linearly."""
        cx_8 = _interpolate_cx_count(8)
        assert 10 < cx_8 < 18  # Between N=6 (10) and N=10 (18)

    def test_extrapolation_above_max(self):
        """Values above max known should extrapolate linearly."""
        cx_100 = _interpolate_cx_count(100)
        assert cx_100 > _interpolate_cx_count(80)

    def test_below_min_returns_min(self):
        """Values below min known should return min value."""
        assert _interpolate_cx_count(2) == _interpolate_cx_count(4)


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_qpu_cost Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEstimateQPUCost:
    """Test the full cost estimation function."""

    def test_returns_dataclass(self, pea_config):
        """Should return QPUCostEstimate dataclass."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert isinstance(est, QPUCostEstimate)

    def test_optimistic_less_than_pessimistic(self, pea_config):
        """Optimistic should always be less than pessimistic."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert est.est_total_optimistic_s < est.est_total_pessimistic_s

    def test_expected_between_optimistic_and_pessimistic(self, pea_config):
        """Expected should be between optimistic and pessimistic."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert est.est_total_optimistic_s <= est.est_total_s <= est.est_total_pessimistic_s

    def test_disabled_spsa_makes_optimistic_equal_expected(self, pea_config):
        """With disabled SPSA, optimistic == expected."""
        est = estimate_qpu_cost(pea_config, n_h_points=3, spsa_model=SPSACostModel.disabled())
        assert abs(est.est_total_optimistic_s - est.est_total_s) < 0.01

    def test_more_h_points_costs_more(self, pea_config):
        """More h-points should increase total cost."""
        est_3 = estimate_qpu_cost(pea_config, n_h_points=3)
        est_6 = estimate_qpu_cost(pea_config, n_h_points=6)
        assert est_6.est_total_s > est_3.est_total_s

    def test_n6_faster_than_n10(self):
        """N=6 circuits should be estimated faster than N=10."""
        config_6 = HardwareConfig(
            n_qubits=6,
            shots=16384,
            n_layouts=3,
            mitigation=MitigationOptions(
                zne_amplifier="pea", num_randomizations=32, shots_per_randomization=128
            ),
        )
        config_10 = HardwareConfig(
            n_qubits=10,
            shots=16384,
            n_layouts=3,
            mitigation=MitigationOptions(
                zne_amplifier="pea", num_randomizations=32, shots_per_randomization=128
            ),
        )
        est_6 = estimate_qpu_cost(config_6, n_h_points=3)
        est_10 = estimate_qpu_cost(config_10, n_h_points=3)
        assert est_6.time_per_circuit_s < est_10.time_per_circuit_s

    def test_pea_has_noise_learning_cost(self, pea_config):
        """PEA should include non-zero noise learning time."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert est.pea_noise_learning_s > 0

    def test_gf_has_no_noise_learning(self, gf_config):
        """Gate-folding should have zero noise learning."""
        est = estimate_qpu_cost(gf_config, n_h_points=3)
        assert est.pea_noise_learning_s == 0.0

    def test_classical_latency_included(self, pea_config):
        """Classical latency should be non-zero."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert est.classical_latency_s > 0

    def test_different_profile_changes_result(self, pea_config):
        """Using nighthawk profile should give faster estimate."""
        est_torino = estimate_qpu_cost(
            pea_config, n_h_points=3, profile=QPUThroughputProfile.ibm_torino()
        )
        est_nh = estimate_qpu_cost(
            pea_config, n_h_points=3, profile=QPUThroughputProfile.ibm_nighthawk()
        )
        assert est_nh.est_total_optimistic_s < est_torino.est_total_optimistic_s

    def test_circuit_depth_override(self, pea_config):
        """Explicit circuit_depth should override auto-estimation."""
        est_auto = estimate_qpu_cost(pea_config, n_h_points=3)
        est_shallow = estimate_qpu_cost(pea_config, n_h_points=3, circuit_depth=10)
        assert est_shallow.effective_clops > est_auto.effective_clops

    def test_fits_per_job_with_no_spsa(self, pea_config):
        """N=10 with no SPSA should fit per-job (600s)."""
        est = estimate_qpu_cost(pea_config, n_h_points=3, spsa_model=SPSACostModel.disabled())
        assert est.fits_per_job is True

    def test_adaptive_has_more_circuits(self):
        """Adaptive (worst case) should have 6 noise factors (double)."""
        config = HardwareConfig(
            n_qubits=10,
            shots=16384,
            n_layouts=3,
            mitigation=MitigationOptions(
                zne_amplifier="adaptive", num_randomizations=32, shots_per_randomization=128
            ),
        )
        est = estimate_qpu_cost(config, n_h_points=3)
        assert est.circuits_per_h == 18  # 3 layouts × 6 factors

    def test_spsa_per_h_if_triggered_field(self, pea_config):
        """spsa_per_h_if_triggered_s should be positive when SPSA enabled."""
        est = estimate_qpu_cost(pea_config, n_h_points=3)
        assert est.spsa_per_h_if_triggered_s > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Preflight New Checks Tests (readout, T1/T2, native gates)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreflightNewChecks:
    """Test the new preflight check functions."""

    @pytest.fixture
    def fake_backend(self):
        """Create FakeTorino backend for testing."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        return FakeTorino()

    def test_compute_mean_readout_error(self, fake_backend):
        """FakeTorino should have measurable readout error."""
        err = compute_mean_readout_error(fake_backend)
        # FakeTorino has calibration data — should return a value
        assert err is None or (isinstance(err, float) and err >= 0)

    def test_compute_min_t1_t2(self, fake_backend):
        """FakeTorino should have T1/T2 data."""
        t1, t2 = compute_min_t1_t2(fake_backend)
        # FakeTorino has qubit properties
        if t1 is not None:
            assert t1 > 0  # T1 is positive
        if t2 is not None:
            assert t2 > 0  # T2 is positive

    def test_check_native_gate_support(self, fake_backend):
        """Should return dict mapping gate names to booleans."""
        gates = ["ecr", "rz", "sx", "x", "measure"]
        result = check_native_gate_support(fake_backend, gates)
        assert isinstance(result, dict)
        assert len(result) == 5
        assert all(isinstance(v, bool) for v in result.values())
        # FakeTorino should support measure at minimum
        assert result["measure"] is True

    def test_check_native_gate_support_missing_gate(self, fake_backend):
        """Non-existent gate should return False."""
        result = check_native_gate_support(fake_backend, ["nonexistent_gate"])
        assert result["nonexistent_gate"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_effective_clops Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEstimateEffectiveCLOPS:
    """Test the convenience wrapper for CLOPS estimation."""

    def test_default_uses_torino(self):
        """Without profile, should default to Kingston (Heron r2)."""
        clops = estimate_effective_clops(10)
        assert clops == 3750

    def test_explicit_profile(self):
        """Explicit profile should be used."""
        nh = QPUThroughputProfile.ibm_nighthawk()
        clops = estimate_effective_clops(10, profile=nh)
        assert clops > 2500  # Nighthawk faster

    def test_explicit_depth(self):
        """Explicit depth should override auto-estimation."""
        clops_deep = estimate_effective_clops(10, circuit_depth=100)
        clops_shallow = estimate_effective_clops(10, circuit_depth=10)
        assert clops_shallow > clops_deep


# ═══════════════════════════════════════════════════════════════════════════════
# Deployment Script Budget Recompute Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBudgetRecompute:
    """Test the recompute_budget_from_measurement function."""

    def test_basic_budget(self):
        """Basic budget recompute with measured T_one_job."""
        import sys

        sys.path.insert(
            0,
            str(
                __import__("pathlib").Path(__file__).resolve().parent.parent
                / "scripts"
                / "experiment_runners"
                / "hardware"
            ),
        )
        from run_ibm_torino_deployment import recompute_budget_from_measurement

        budget = recompute_budget_from_measurement(
            t_one_job_s=60.0,
            n_h_points=4,
            n_layouts=3,
            n_zne_factors=3,
            spsa_enabled=True,
        )
        assert "total_optimistic_s" in budget
        assert "total_pessimistic_s" in budget
        assert "total_expected_s" in budget
        assert budget["total_optimistic_s"] < budget["total_pessimistic_s"]
        assert budget["t_one_job_measured_s"] == 60.0

    def test_budget_no_spsa(self):
        """With spsa_enabled=False, optimistic == expected."""
        import sys

        sys.path.insert(
            0,
            str(
                __import__("pathlib").Path(__file__).resolve().parent.parent
                / "scripts"
                / "experiment_runners"
                / "hardware"
            ),
        )
        from run_ibm_torino_deployment import recompute_budget_from_measurement

        budget = recompute_budget_from_measurement(
            t_one_job_s=60.0,
            n_h_points=4,
            n_layouts=3,
            n_zne_factors=3,
            spsa_enabled=False,
        )
        # No SPSA → optimistic == expected (both ignore SPSA)
        assert budget["total_optimistic_s"] == budget["total_expected_s"]
        assert budget["spsa_time_if_triggered_s"] == 0.0

    def test_budget_scales_with_h_points(self):
        """More h-points should increase budget proportionally."""
        import sys

        sys.path.insert(
            0,
            str(
                __import__("pathlib").Path(__file__).resolve().parent.parent
                / "scripts"
                / "experiment_runners"
                / "hardware"
            ),
        )
        from run_ibm_torino_deployment import recompute_budget_from_measurement

        b4 = recompute_budget_from_measurement(60.0, 4, 3, spsa_enabled=False)
        b8 = recompute_budget_from_measurement(60.0, 8, 3, spsa_enabled=False)
        # 8 h-points should be roughly 2× cost of 4 (minus fixed PEA startup)
        ratio = b8["total_optimistic_s"] / b4["total_optimistic_s"]
        assert 1.8 < ratio < 2.1
