"""Quick sanity check: import the persistence module and verify functions."""

import sys

sys.path.insert(0, "/Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe")

from scripts.experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
)
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    BENCHMARK_VERSION,
    RESULTS_BASE,
    _build_envelope,
    _build_result_path,
    _save_result,
)

print(f"BENCHMARK_VERSION = {BENCHMARK_VERSION!r}")
print(f"RESULTS_BASE = {RESULTS_BASE}")
print(f"_build_result_path callable: {callable(_build_result_path)}")
print(f"_build_envelope callable: {callable(_build_envelope)}")
print(f"_save_result callable: {callable(_save_result)}")

# Test _build_result_path
path = _build_result_path("C0_raw", 3.25, "fake_backend", 42)
print(f"\nResult path (seed=42): {path}")
assert "h3p25" in str(path)
assert "fake_backend" in str(path)
assert "C0_raw" in str(path)
assert "_seed" not in str(path), "seed=42 should NOT have _seed suffix"

path_seed = _build_result_path("C5_full_pea_balanced", 3.75, "hardware", 99)
print(f"Result path (seed=99): {path_seed}")
assert "h3p75" in str(path_seed)
assert "_seed99" in str(path_seed), "seed!=42 should have _seed suffix"
assert "hardware" in str(path_seed)

# Test _build_envelope with mock data
config = BENCHMARK_CONFIGS["C5_full_pea_balanced"]
envelope = _build_envelope(
    config=config,
    h_value=3.5,
    mode="fake_backend",
    seed=42,
    circuit_stats={
        "depth_logical": 10,
        "depth_transpiled": 45,
        "n_2q_gates": 34,
        "circuit_depth_with_dd_estimate": 52,
        "routing_overhead_pct": 0.0,
        "transpiled_vs_logical_ratio": 4.5,
    },
    error_budget={"fidelity_estimate": 0.87},
    execution_result={
        "e_mitigated": -12.5,
        "e_raw": -11.0,
        "e_exact": -12.8,
        "delta_e_gap": 0.023,
        "improvement_vs_raw": 0.83,
        "zne_r2": 0.998,
        "phase_label": "paramagnetic",
        "correct_label": True,
        "per_site_magnetization_std": 0.012,
        "energy_within_physical_bounds": True,
        "qpu_seconds": None,
        "noise_learning_time_s": 1.2,
        "shots": 16384,
    },
    wall_time_s=45.3,
    hardware_calibration=None,
)

# Verify all required sections
assert "benchmark_metadata" in envelope
assert "circuit_stats" in envelope
assert "timing" in envelope
assert "results" in envelope
assert "shots" in envelope
assert "mitigation_config" in envelope
assert "hardware_calibration" in envelope

# Verify benchmark_metadata
meta = envelope["benchmark_metadata"]
assert meta["config_id"] == "C5_full_pea_balanced"
assert meta["execution_mode"] == "fake_backend"
assert meta["h_value"] == 3.5
assert meta["benchmark_version"] == "1.0"
assert meta["seed"] == 42

# Verify circuit_stats includes derived metrics
cs = envelope["circuit_stats"]
assert cs["circuit_depth_with_dd_estimate"] == 52
assert cs["routing_overhead_pct"] == 0.0
assert cs["transpiled_vs_logical_ratio"] == 4.5
assert cs["optimization_level"] == 2
assert cs["fidelity_estimate"] == 0.87

# Verify results section
results = envelope["results"]
assert results["per_site_magnetization_std"] == 0.012
assert results["energy_within_physical_bounds"] is True
assert results["e_mitigated"] == -12.5

# Verify hardware_calibration is null for fake_backend
assert envelope["hardware_calibration"] is None
# Verify aqc_metrics absent (not provided)
assert "aqc_metrics" not in envelope

# Test with aqc_metrics
envelope_aqc = _build_envelope(
    config=BENCHMARK_CONFIGS["C16_aqc_pea"],
    h_value=3.0,
    mode="fake_backend",
    seed=42,
    circuit_stats={"depth_logical": 10, "depth_transpiled": 30},
    error_budget={"fidelity_estimate": 0.92},
    execution_result={"e_mitigated": -12.0, "shots": 16384},
    wall_time_s=60.0,
    aqc_metrics={"aqc_fidelity": 0.999, "aqc_n_2q_compressed": 18},
)
assert "aqc_metrics" in envelope_aqc
assert envelope_aqc["aqc_metrics"]["aqc_fidelity"] == 0.999

# Test with hardware_calibration populated
hw_calib = {
    "t1_mean_layout": 150.0,
    "t2_mean_layout": 100.0,
    "cx_error_mean_layout": 0.005,
    "readout_error_mean": 0.012,
    "calibration_age_hours": 2.5,
    "job_execution_time_s": 120.0,
}
envelope_hw = _build_envelope(
    config=config,
    h_value=3.5,
    mode="hardware",
    seed=42,
    circuit_stats={"depth_logical": 10, "depth_transpiled": 45},
    error_budget={"fidelity_estimate": 0.85},
    execution_result={"e_mitigated": -12.3, "shots": 16384},
    wall_time_s=180.0,
    hardware_calibration=hw_calib,
)
assert envelope_hw["hardware_calibration"] is not None
assert envelope_hw["hardware_calibration"]["t1_mean_layout"] == 150.0

print("\n✅ All sanity checks passed!")
