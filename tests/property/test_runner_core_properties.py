"""Property-based tests for runner core: routing, persistence, and cache.

# Feature: mitigation-benchmark, Properties 4, 5, 6, 7, 19, 22, 23
# **Validates: Requirements 3.1-3.8, 4.1-4.8, 10.1, 10.3, 15.2, 19.1, 19.3**
#
# Property 4: Execution routing correctness
#   Each zne_method dispatches to the correct function.
#
# Property 5: ResultEnvelope completeness
#   All required sections present (including hardware_calibration null/dict).
#
# Property 6: Result path determinism
#   Same inputs produce the same path pattern.
#
# Property 7: Idempotency — skip existing results
#   No overwrite when file already exists.
#
# Property 19: ClassicalSolver cache consistency
#   Shared h_value → bit-exact e_exact/gap.
#
# Property 22: Affine correction on raw configs
#   affine_enabled=True + zne_method=None → e_mitigated = affine(e_raw).
#
# Property 23: Derived circuit_stats formula correctness
#   Formulas match spec for all envelopes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark_configs import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
    BenchmarkConfig,
)
from experiment_runners.hardware.run_mitigation_benchmark import (
    _build_envelope,
    _build_result_path,
    _classical_cache,
    _get_exact_energy,
    apply_affine_on_raw,
    compute_derived_circuit_stats,
    route_execution,
    run_single_config,
)

from qmbp_simulation.execution import affine_correct_energy

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Strategy for selecting a valid config from the registry
valid_config_st = st.sampled_from(list(BENCHMARK_CONFIGS.values()))

# Strategy for h-values in the benchmark range
h_value_st = st.sampled_from([3.0, 3.25, 3.5, 3.75, 4.0])

# Strategy for execution mode
mode_st = st.sampled_from(["fake_backend", "hardware"])

# Strategy for seeds
seed_st = st.integers(min_value=1, max_value=10000)

# Strategy for positive floats (for circuit stats)
pos_int_st = st.integers(min_value=1, max_value=5000)
pos_float_st = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy for n_2q_logical (must be >= 0)
n_2q_logical_st = st.integers(min_value=0, max_value=500)


# ═══════════════════════════════════════════════════════════════════════════════
# Property 4: Execution routing correctness
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
# ═══════════════════════════════════════════════════════════════════════════════


# Mapping from zne_method to the internal function that should be called
_EXPECTED_DISPATCH = {
    None: "_execute_raw",
    "gf": "_execute_gate_folding",
    "pea": "_execute_pea",
    "mitiq_zne": "_execute_mitiq_zne",
    "mitiq_cdr": "_execute_mitiq_cdr",
    "mitiq_ddd_zne": "_execute_mitiq_ddd",
}


class TestExecutionRoutingCorrectness:
    """Property 4: Execution routing correctness.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    For each config in BENCHMARK_CONFIGS, verify route_execution dispatches
    to the correct internal function based on zne_method.
    """

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_route_dispatches_to_correct_function(self, config: BenchmarkConfig):
        """route_execution dispatches based on config.zne_method."""
        expected_fn = _EXPECTED_DISPATCH[config.zne_method]
        module_path = "experiment_runners.hardware.run_mitigation_benchmark"

        mock_circuit = MagicMock()
        mock_observable = MagicMock()
        mock_backend = MagicMock()

        fake_result = {
            "e_mitigated": -5.0,
            "e_raw": -4.0,
            "zne_r2": 0.99,
            "shots": 16384,
        }

        with patch(f"{module_path}.{expected_fn}", return_value=fake_result) as mock_fn:
            # Also patch GNN-QEM to avoid side effects for C10
            with patch(
                f"{module_path}._apply_gnn_qem_postprocessing",
                return_value=fake_result,
            ):
                result = route_execution(
                    config=config,
                    transpiled_circuit=mock_circuit,
                    H_mapped=mock_observable,
                    backend=mock_backend,
                    shots=16384,
                    h_value=3.5,
                    e_exact=-10.0,
                    gap=0.5,
                )

            mock_fn.assert_called_once()

    def test_unknown_zne_method_raises_value_error(self):
        """route_execution raises ValueError for unknown zne_method."""
        # Create a mock config with invalid zne_method
        mock_config = MagicMock(spec=BenchmarkConfig)
        mock_config.zne_method = "unknown_method"
        mock_config.gnn_qem_enabled = False

        with pytest.raises(ValueError, match="Unknown zne_method"):
            route_execution(
                config=mock_config,
                transpiled_circuit=MagicMock(),
                H_mapped=MagicMock(),
                backend=MagicMock(),
                shots=16384,
                h_value=3.5,
                e_exact=-10.0,
                gap=0.5,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 5: ResultEnvelope completeness
# **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**
# ═══════════════════════════════════════════════════════════════════════════════

# Required top-level sections in ResultEnvelope
_REQUIRED_SECTIONS = {
    "benchmark_metadata",
    "circuit_stats",
    "timing",
    "results",
    "shots",
    "mitigation_config",
    "hardware_calibration",
}

# Required keys within benchmark_metadata
_REQUIRED_METADATA_KEYS = {
    "config_id",
    "execution_mode",
    "h_value",
    "timestamp",
    "benchmark_version",
    "seed",
}

# Required keys within results
_REQUIRED_RESULTS_KEYS = {
    "e_mitigated",
    "e_raw",
    "e_exact",
    "delta_e_gap",
    "phase_label",
    "correct_label",
}


class TestResultEnvelopeCompleteness:
    """Property 5: ResultEnvelope completeness.

    **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

    For any successful execution, the envelope contains all required sections
    with all mandatory fields populated. hardware_calibration is None for
    fake_backend and dict for hardware.
    """

    @given(config=valid_config_st, h_value=h_value_st, seed=seed_st)
    @settings(max_examples=50, deadline=None)
    def test_envelope_has_all_required_sections(
        self, config: BenchmarkConfig, h_value: float, seed: int
    ):
        """_build_envelope produces a dict with all required top-level sections."""
        circuit_stats = {
            "depth_transpiled": 50,
            "depth_logical": 20,
            "n_2q_gates": 34,
            "n_1q_gates": 40,
            "depth_2q": 10,
            "max_idle_stretch": 5,
        }
        error_budget = {"fidelity_estimate": 0.95}
        execution_result = {
            "e_mitigated": -9.5,
            "e_raw": -8.0,
            "e_exact": -10.0,
            "delta_e_gap": 0.05,
            "improvement_vs_raw": 0.75,
            "zne_r2": 0.99,
            "phase_label": "ordered",
            "correct_label": True,
            "per_site_magnetization_std": 0.02,
            "energy_within_physical_bounds": True,
            "shots": 16384,
            "qpu_seconds": None,
            "noise_learning_time_s": None,
        }

        envelope = _build_envelope(
            config=config,
            h_value=h_value,
            mode="fake_backend",
            seed=seed,
            circuit_stats=circuit_stats,
            error_budget=error_budget,
            execution_result=execution_result,
            wall_time_s=1.5,
            hardware_calibration=None,
        )

        for section in _REQUIRED_SECTIONS:
            assert section in envelope, (
                f"Missing required section '{section}' in envelope for {config.config_id}"
            )

    @given(config=valid_config_st, h_value=h_value_st, seed=seed_st)
    @settings(max_examples=50, deadline=None)
    def test_envelope_metadata_has_required_keys(
        self, config: BenchmarkConfig, h_value: float, seed: int
    ):
        """benchmark_metadata contains all required keys."""
        circuit_stats = {"depth_transpiled": 50, "max_idle_stretch": 5}
        error_budget = {"fidelity_estimate": 0.95}
        execution_result = {
            "e_mitigated": -9.5,
            "e_raw": -8.0,
            "shots": 16384,
        }

        envelope = _build_envelope(
            config=config,
            h_value=h_value,
            mode="fake_backend",
            seed=seed,
            circuit_stats=circuit_stats,
            error_budget=error_budget,
            execution_result=execution_result,
            wall_time_s=1.0,
            hardware_calibration=None,
        )

        metadata = envelope["benchmark_metadata"]
        for key in _REQUIRED_METADATA_KEYS:
            assert key in metadata, f"Missing '{key}' in benchmark_metadata for {config.config_id}"
        assert metadata["config_id"] == config.config_id
        assert metadata["h_value"] == h_value
        assert metadata["seed"] == seed

    @given(config=valid_config_st)
    @settings(max_examples=30, deadline=None)
    def test_hardware_calibration_null_for_fake_backend(self, config: BenchmarkConfig):
        """hardware_calibration is None when mode is fake_backend."""
        circuit_stats = {"depth_transpiled": 50, "max_idle_stretch": 5}
        error_budget = {"fidelity_estimate": 0.95}
        execution_result = {"e_mitigated": -9.5, "e_raw": -8.0, "shots": 16384}

        envelope = _build_envelope(
            config=config,
            h_value=3.5,
            mode="fake_backend",
            seed=42,
            circuit_stats=circuit_stats,
            error_budget=error_budget,
            execution_result=execution_result,
            wall_time_s=1.0,
            hardware_calibration=None,
        )

        assert envelope["hardware_calibration"] is None, (
            f"hardware_calibration should be None for fake_backend, "
            f"got {envelope['hardware_calibration']}"
        )

    @given(config=valid_config_st)
    @settings(max_examples=30, deadline=None)
    def test_hardware_calibration_dict_for_hardware(self, config: BenchmarkConfig):
        """hardware_calibration is a dict with required keys for hardware mode."""
        circuit_stats = {"depth_transpiled": 50, "max_idle_stretch": 5}
        error_budget = {"fidelity_estimate": 0.95}
        execution_result = {"e_mitigated": -9.5, "e_raw": -8.0, "shots": 16384}

        hw_cal = {
            "t1_mean_layout": 150.0,
            "t2_mean_layout": 100.0,
            "cx_error_mean_layout": 0.005,
            "readout_error_mean": 0.01,
            "calibration_age_hours": 2.5,
            "job_execution_time_s": 30.0,
        }

        envelope = _build_envelope(
            config=config,
            h_value=3.5,
            mode="hardware",
            seed=42,
            circuit_stats=circuit_stats,
            error_budget=error_budget,
            execution_result=execution_result,
            wall_time_s=35.0,
            hardware_calibration=hw_cal,
        )

        assert envelope["hardware_calibration"] is not None
        assert isinstance(envelope["hardware_calibration"], dict)
        required_hw_keys = {
            "t1_mean_layout",
            "t2_mean_layout",
            "cx_error_mean_layout",
            "readout_error_mean",
            "calibration_age_hours",
            "job_execution_time_s",
        }
        for key in required_hw_keys:
            assert key in envelope["hardware_calibration"], (
                f"Missing '{key}' in hardware_calibration"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 6: Result path determinism
# **Validates: Requirements 4.1, 10.1, 10.3**
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultPathDeterminism:
    """Property 6: Result path determinism.

    **Validates: Requirements 4.1, 10.1, 10.3**

    For any (config_id, h_value, mode, seed), _build_result_path produces a
    path containing the config_id and h-value pattern, and same inputs
    produce paths with the same directory structure.
    """

    @given(config=valid_config_st, h_value=h_value_st, mode=mode_st, seed=seed_st)
    @settings(max_examples=100, deadline=None)
    def test_path_contains_config_id(
        self, config: BenchmarkConfig, h_value: float, mode: str, seed: int
    ):
        """Result path contains the config_id as a directory component."""
        path = _build_result_path(config.config_id, h_value, mode, seed)
        assert config.config_id in str(path), (
            f"Path {path} does not contain config_id '{config.config_id}'"
        )

    @given(config=valid_config_st, h_value=h_value_st, mode=mode_st, seed=seed_st)
    @settings(max_examples=100, deadline=None)
    def test_path_contains_mode(
        self, config: BenchmarkConfig, h_value: float, mode: str, seed: int
    ):
        """Result path contains the execution mode as a directory component."""
        path = _build_result_path(config.config_id, h_value, mode, seed)
        assert mode in str(path), f"Path {path} does not contain mode '{mode}'"

    @given(config=valid_config_st, h_value=h_value_st, mode=mode_st, seed=seed_st)
    @settings(max_examples=100, deadline=None)
    def test_path_contains_h_value_pattern(
        self, config: BenchmarkConfig, h_value: float, mode: str, seed: int
    ):
        """Result path contains h-value formatted as 'h{val}' with dot→p."""
        path = _build_result_path(config.config_id, h_value, mode, seed)
        h_str = f"h{str(h_value).replace('.', 'p')}"
        assert h_str in str(path), f"Path {path} does not contain h-value pattern '{h_str}'"

    @given(config=valid_config_st, h_value=h_value_st, mode=mode_st, seed=seed_st)
    @settings(max_examples=50, deadline=None)
    def test_path_is_json_file(self, config: BenchmarkConfig, h_value: float, mode: str, seed: int):
        """Result path ends with .json extension."""
        path = _build_result_path(config.config_id, h_value, mode, seed)
        assert str(path).endswith(".json"), f"Path {path} does not end with .json"

    @given(
        config=valid_config_st,
        h_value=h_value_st,
        mode=mode_st,
        seed=st.integers(min_value=1, max_value=41),
    )
    @settings(max_examples=50, deadline=None)
    def test_non_default_seed_in_filename(
        self, config: BenchmarkConfig, h_value: float, mode: str, seed: int
    ):
        """When seed != 42, filename includes seed suffix."""
        assume(seed != 42)
        path = _build_result_path(config.config_id, h_value, mode, seed)
        assert f"seed{seed}" in path.name, (
            f"Path filename '{path.name}' does not contain 'seed{seed}' for non-default seed={seed}"
        )

    def test_default_seed_not_in_filename(self):
        """When seed == 42 (default), filename does NOT include seed suffix."""
        path = _build_result_path("C0_raw", 3.5, "fake_backend", 42)
        assert "seed" not in path.name, (
            f"Path filename '{path.name}' should not contain 'seed' for default seed=42"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 7: Idempotency — skip existing results
# **Validates: Requirements 4.8**
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdempotencySkipExisting:
    """Property 7: Idempotency — skip existing results.

    **Validates: Requirements 4.8**

    When a result file already exists on disk, run_single_config returns
    empty dict without executing or overwriting.
    """

    @given(config=valid_config_st, h_value=h_value_st)
    @settings(max_examples=20, deadline=None)
    def test_skip_when_file_exists(self, config: BenchmarkConfig, h_value: float):
        """run_single_config returns {} when result file already exists."""
        import tempfile

        mode = "fake_backend"
        seed = 42

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_path = tmp_path / f"{config.config_id}_h{h_value}.json"
            fake_path.write_text('{"existing": true}')

            module_path = "experiment_runners.hardware.run_mitigation_benchmark"
            with patch(f"{module_path}._build_result_path", return_value=fake_path):
                result = run_single_config(config, h_value, mode, 16384, seed)

            assert result == {}, f"Expected empty dict for existing file, got {result}"

            # Verify the file was NOT overwritten
            content = fake_path.read_text()
            assert content == '{"existing": true}', "File was overwritten — idempotency violated"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 19: ClassicalSolver cache consistency
# **Validates: Requirements 19.1, 19.3**
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassicalSolverCacheConsistency:
    """Property 19: ClassicalSolver cache consistency.

    **Validates: Requirements 19.1, 19.3**

    Calling _get_exact_energy twice with the same h_value returns
    bit-exact identical values (same object from cache).
    """

    @given(h_value=h_value_st)
    @settings(max_examples=5, deadline=None)
    def test_cache_returns_bit_exact_values(self, h_value: float):
        """Two calls with same h_value produce identical (e_exact, gap)."""
        # Clear cache to test fresh
        _classical_cache.clear()

        first = _get_exact_energy(h_value)
        second = _get_exact_energy(h_value)

        # Must be bit-exact identical (same cached tuple)
        assert first[0] == second[0], (
            f"e_exact not bit-exact: {first[0]} vs {second[0]} for h={h_value}"
        )
        assert first[1] == second[1], (
            f"gap not bit-exact: {first[1]} vs {second[1]} for h={h_value}"
        )

    @given(h_value=h_value_st)
    @settings(max_examples=5, deadline=None)
    def test_cache_hit_returns_same_object(self, h_value: float):
        """Cached result is the exact same tuple object (identity check)."""
        _classical_cache.clear()

        first = _get_exact_energy(h_value)
        second = _get_exact_energy(h_value)

        # Same tuple object → proves cache is being used
        assert first is second, f"Cache not returning same object for h={h_value}"

    @given(h_value=h_value_st)
    @settings(max_examples=5, deadline=None)
    def test_cached_values_are_physically_valid(self, h_value: float):
        """Cached values have correct physical properties."""
        _classical_cache.clear()

        e_exact, gap = _get_exact_energy(h_value)

        # Energy should be negative for TFIM (ground state is below zero)
        assert e_exact < 0, f"Ground energy should be negative for TFIM, got {e_exact}"
        # Gap must be non-negative
        assert gap >= 0, f"Spectral gap must be >= 0, got {gap}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 22: Affine correction on raw configs
# **Validates: Requirements 3.8**
# ═══════════════════════════════════════════════════════════════════════════════


class TestAffineCorrectionOnRaw:
    """Property 22: Affine correction on raw configs.

    **Validates: Requirements 3.8**

    For configs with affine_enabled=True and zne_method=None, apply_affine_on_raw
    sets e_mitigated = affine_correct_energy(e_raw, e_ground, e_upper).
    For configs with zne_method != None, no change is made.
    """

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_affine_applied_when_affine_enabled_and_no_zne(self, config: BenchmarkConfig):
        """When affine_enabled=True and zne_method=None, e_mitigated is set."""
        assume(config.affine_enabled and config.zne_method is None)

        e_raw = -7.5
        e_exact = -10.0
        e_upper = 10.0
        execution_result = {"e_raw": e_raw, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        # e_mitigated should now be set to the affine-corrected value
        assert result["e_mitigated"] is not None, (
            f"e_mitigated should be set for {config.config_id} with "
            f"affine_enabled=True and zne_method=None"
        )

        # Verify it matches affine_correct_energy
        expected = affine_correct_energy(e_raw, e_exact, e_upper)
        assert result["e_mitigated"] == expected.corrected_energy, (
            f"e_mitigated={result['e_mitigated']} != "
            f"affine_correct_energy={expected.corrected_energy}"
        )

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_no_change_when_zne_method_set(self, config: BenchmarkConfig):
        """When zne_method is not None, apply_affine_on_raw makes no change."""
        assume(config.zne_method is not None)

        e_raw = -7.5
        e_mitigated_original = -9.0
        e_exact = -10.0
        e_upper = 10.0
        execution_result = {"e_raw": e_raw, "e_mitigated": e_mitigated_original}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        assert result["e_mitigated"] == e_mitigated_original, (
            f"e_mitigated changed from {e_mitigated_original} to "
            f"{result['e_mitigated']} but zne_method={config.zne_method} "
            f"(should not apply affine when ZNE is active)"
        )

    @given(config=valid_config_st)
    @settings(max_examples=50, deadline=None)
    def test_no_change_when_affine_disabled(self, config: BenchmarkConfig):
        """When affine_enabled=False, apply_affine_on_raw makes no change."""
        assume(not config.affine_enabled)

        e_raw = -7.5
        e_exact = -10.0
        e_upper = 10.0
        execution_result = {"e_raw": e_raw, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

        assert result["e_mitigated"] is None, (
            f"e_mitigated should remain None when affine_enabled=False, got {result['e_mitigated']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 23: Derived circuit_stats formula correctness
# **Validates: Requirements 4.4**
# ═══════════════════════════════════════════════════════════════════════════════


# Strategy for generating circuit stats dicts
@st.composite
def circuit_stats_st(draw):
    """Generate a valid circuit stats dict for compute_derived_circuit_stats."""
    depth_transpiled = draw(st.integers(min_value=1, max_value=5000))
    max_idle_stretch = draw(st.integers(min_value=0, max_value=500))
    n_2q_gates = draw(st.integers(min_value=1, max_value=500))
    depth_logical = draw(st.integers(min_value=1, max_value=2000))

    return {
        "depth_transpiled": depth_transpiled,
        "max_idle_stretch": max_idle_stretch,
        "n_2q_gates": n_2q_gates,
        "depth_logical": depth_logical,
    }


class TestDerivedCircuitStatsFormulas:
    """Property 23: Derived circuit_stats formula correctness.

    **Validates: Requirements 4.4**

    For any ResultEnvelope, the three derived formulas are:
    - circuit_depth_with_dd_estimate = depth_transpiled + max_idle_stretch
    - routing_overhead_pct = (n_2q_gates - n_2q_logical) / n_2q_logical × 100
      (0.0 if n_2q_logical=0)
    - transpiled_vs_logical_ratio = depth_transpiled / depth_logical
      (0.0 if depth_logical=0)
    """

    @given(stats=circuit_stats_st(), n_2q_logical=st.integers(min_value=1, max_value=500))
    @settings(max_examples=200, deadline=None)
    def test_circuit_depth_with_dd_estimate(self, stats: dict, n_2q_logical: int):
        """circuit_depth_with_dd_estimate = depth_transpiled + max_idle_stretch."""
        result = compute_derived_circuit_stats(stats, n_2q_logical)
        expected = stats["depth_transpiled"] + stats["max_idle_stretch"]
        assert result["circuit_depth_with_dd_estimate"] == expected, (
            f"Expected {expected}, got {result['circuit_depth_with_dd_estimate']}"
        )

    @given(stats=circuit_stats_st(), n_2q_logical=st.integers(min_value=1, max_value=500))
    @settings(max_examples=200, deadline=None)
    def test_routing_overhead_pct(self, stats: dict, n_2q_logical: int):
        """routing_overhead_pct = (n_2q_gates - n_2q_logical) / n_2q_logical × 100."""
        result = compute_derived_circuit_stats(stats, n_2q_logical)
        expected = (stats["n_2q_gates"] - n_2q_logical) / n_2q_logical * 100
        assert abs(result["routing_overhead_pct"] - expected) < 1e-10, (
            f"Expected {expected}, got {result['routing_overhead_pct']}"
        )

    @given(stats=circuit_stats_st())
    @settings(max_examples=100, deadline=None)
    def test_routing_overhead_zero_when_n_2q_logical_zero(self, stats: dict):
        """routing_overhead_pct = 0.0 when n_2q_logical=0."""
        result = compute_derived_circuit_stats(stats, n_2q_logical=0)
        assert result["routing_overhead_pct"] == 0.0, (
            f"Expected 0.0 for n_2q_logical=0, got {result['routing_overhead_pct']}"
        )

    @given(stats=circuit_stats_st(), n_2q_logical=st.integers(min_value=1, max_value=500))
    @settings(max_examples=200, deadline=None)
    def test_transpiled_vs_logical_ratio(self, stats: dict, n_2q_logical: int):
        """transpiled_vs_logical_ratio = depth_transpiled / depth_logical."""
        result = compute_derived_circuit_stats(stats, n_2q_logical)
        expected = stats["depth_transpiled"] / stats["depth_logical"]
        assert abs(result["transpiled_vs_logical_ratio"] - expected) < 1e-10, (
            f"Expected {expected}, got {result['transpiled_vs_logical_ratio']}"
        )

    @given(n_2q_logical=st.integers(min_value=0, max_value=500))
    @settings(max_examples=50, deadline=None)
    def test_transpiled_vs_logical_ratio_zero_when_depth_logical_zero(self, n_2q_logical: int):
        """transpiled_vs_logical_ratio = 0.0 when depth_logical=0."""
        stats = {
            "depth_transpiled": 100,
            "max_idle_stretch": 10,
            "n_2q_gates": 34,
            "depth_logical": 0,
        }
        result = compute_derived_circuit_stats(stats, n_2q_logical)
        assert result["transpiled_vs_logical_ratio"] == 0.0, (
            f"Expected 0.0 for depth_logical=0, got {result['transpiled_vs_logical_ratio']}"
        )

    @given(stats=circuit_stats_st(), n_2q_logical=st.integers(min_value=0, max_value=500))
    @settings(max_examples=100, deadline=None)
    def test_original_stats_not_mutated(self, stats: dict, n_2q_logical: int):
        """compute_derived_circuit_stats does not mutate the input dict."""
        original_keys = set(stats.keys())
        original_values = {k: v for k, v in stats.items()}

        compute_derived_circuit_stats(stats, n_2q_logical)

        assert set(stats.keys()) == original_keys, "Input dict keys were mutated"
        for k in original_keys:
            assert stats[k] == original_values[k], f"Input dict value for '{k}' was mutated"
