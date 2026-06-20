"""Lightweight test for scaling digest tooling — NO heavy imports (no pytorch, qiskit, etc).

Tests only project_health.digest and project_health.analysis.scaling_analyzer
which depend only on stdlib + numpy.
"""

import json
import sys
import tempfile
from pathlib import Path

# Only need project_health on path (no src/ needed for these modules)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ═══════════════════════════════════════════════════════════════════════
# Test 1: ScalingResult model
# ═══════════════════════════════════════════════════════════════════════
from project_health.digest.models import (
    ModeComparisonResult,
    N120SweepResult,
    ScalingResult,
)

sr = ScalingResult(
    source_file="test.json",
    folder="scaling",
    n_qubits=50,
    p_layers=1,
    topology="chain_1d",
    strategy="aer_mps",
    chi_max=64,
    precision=0.005,
    seed=42,
    h_values=[5.0, 4.5],
    n_pass=2,
    n_total=2,
    all_passed=True,
    mean_de_gap=0.003,
    max_de_gap=0.004,
    phase1_time_s=10,
    phase2_time_s=50,
    total_time_s=60,
    per_h_de_gap=[0.003, 0.004],
    per_h_passed=[True, True],
)
assert sr.n_qubits == 50
assert sr.all_passed is True
print("✅ Test 1: ScalingResult model works")

# ═══════════════════════════════════════════════════════════════════════
# Test 2: ModeComparisonResult model
# ═══════════════════════════════════════════════════════════════════════
mc = ModeComparisonResult(
    source_file="mc.json",
    results=[{"N": 40, "h": 5.5}],
    all_det_pass=True,
    all_sto_pass=True,
    mean_speedup=138.7,
    mean_energy_diff=0.17,
    modes_consistent=False,
)
assert mc.mean_speedup == 138.7
assert mc.modes_consistent is False
print("✅ Test 2: ModeComparisonResult model works")

# ═══════════════════════════════════════════════════════════════════════
# Test 3: N120SweepResult model
# ═══════════════════════════════════════════════════════════════════════
n120 = N120SweepResult(
    source_file="n120.json",
    n_qubits=120,
    h_min_safe=12.09,
    h_values=[12.59, 13.09, 13.59],
    seeds=[42, 43, 44],
    total_time_s=442,
    n_total=15,
    n_pass=15,
    pass_rate=1.0,
    mean_de_gap=0.00019,
    max_de_gap=0.0003,
    std_de_gap=0.00004,
    bootstrap_ci_95=[0.00017, 0.00021],
    scaling_law_validated=True,
)
assert n120.scaling_law_validated is True
assert n120.n_pass == 15
print("✅ Test 3: N120SweepResult model works")

# ═══════════════════════════════════════════════════════════════════════
# Test 4: format_scaling_text
# ═══════════════════════════════════════════════════════════════════════
from project_health.digest.formatters import format_scaling_text

output = format_scaling_text([sr], mode_comparison=mc, n120_sweep=n120, verbose=True)
assert "MPS Scaling Validation" in output
assert "N=120" in output
assert "Mode Comparison" in output
assert "138.7" in output
assert "Thesis Summary" in output
print(f"✅ Test 4: format_scaling_text produces {len(output)} chars")

# ═══════════════════════════════════════════════════════════════════════
# Test 5: format_scaling_text with empty data
# ═══════════════════════════════════════════════════════════════════════
output_empty = format_scaling_text([], mode_comparison=None, n120_sweep=None)
assert "No MPS scaling" in output_empty
print("✅ Test 5: format_scaling_text handles empty input")

# ═══════════════════════════════════════════════════════════════════════
# Test 6: Scanner parse on real data (file-based, no heavy deps)
# ═══════════════════════════════════════════════════════════════════════
from project_health.digest.scanner import ResultScanner

# Create a temp dir with synthetic scaling JSON
with tempfile.TemporaryDirectory() as tmpdir:
    scaling_dir = Path(tmpdir) / "scaling"
    scaling_dir.mkdir()

    # Write a valid scaling_N50 result
    scaling_data = {
        "experiment": "mps_scaling_validation",
        "metadata": {
            "n": 50,
            "topology": "chain_1d",
            "strategy": "aer_mps",
            "chi_max": 64,
            "precision": 0.005,
            "seeds": [42, 43],
            "h_values": [5.0, 4.5],
            "p_layers": 1,
        },
        "timing": {"phase1_dmrg_s": 10, "phase2_vqe_s": 80, "total_s": 90},
        "summary": {"n_pass": 4, "n_total": 4, "all_passed": True},
        "vqe_results": [
            {
                "seed": 42,
                "results": [
                    {
                        "h": 5.0,
                        "vqe_energy": -250.0,
                        "dmrg_energy": -250.01,
                        "gap": 8.0,
                        "de_gap": 0.001,
                        "time_s": 20,
                        "passed": True,
                    },
                    {
                        "h": 4.5,
                        "vqe_energy": -225.0,
                        "dmrg_energy": -225.02,
                        "gap": 7.0,
                        "de_gap": 0.003,
                        "time_s": 20,
                        "passed": True,
                    },
                ],
            },
            {
                "seed": 43,
                "results": [
                    {
                        "h": 5.0,
                        "vqe_energy": -250.0,
                        "dmrg_energy": -250.01,
                        "gap": 8.0,
                        "de_gap": 0.001,
                        "time_s": 20,
                        "passed": True,
                    },
                    {
                        "h": 4.5,
                        "vqe_energy": -225.0,
                        "dmrg_energy": -225.02,
                        "gap": 7.0,
                        "de_gap": 0.003,
                        "time_s": 20,
                        "passed": True,
                    },
                ],
            },
        ],
    }
    (scaling_dir / "scaling_N50_aer_mps_20260610_120000.json").write_text(
        json.dumps(scaling_data, indent=2)
    )

    # Write N=120 sweep
    n120_data = {
        "experiment": "N120_full_sweep",
        "n_qubits": 120,
        "h_min_safe": 12.09,
        "h_values": [12.59, 13.09],
        "seeds": [42, 43],
        "total_time_s": 200,
        "summary": {
            "n_total": 4,
            "n_pass": 4,
            "pass_rate": 1.0,
            "mean_de_gap": 0.0002,
            "max_de_gap": 0.0003,
            "std_de_gap": 0.00005,
            "bootstrap_ci_95_mean_de_gap": [0.00018, 0.00022],
        },
        "scaling_law": {
            "formula": "h_min = 1.5 + 0.020 * N^1.31",
            "h_min_predicted": 12.09,
            "validated": True,
        },
    }
    (scaling_dir / "scaling_N120_full_sweep.json").write_text(json.dumps(n120_data, indent=2))

    # Write mode comparison
    mc_data = {
        "experiment": "mps_mode_comparison",
        "results": [
            {
                "N": 50,
                "h": 6.5,
                "deterministic": {"de_gap": 0.001},
                "stochastic": {"de_gap": 0.01},
                "comparison": {"speedup_factor": 160},
            }
        ],
        "summary": {
            "all_det_pass": True,
            "all_sto_pass": True,
            "mean_speedup": 160.0,
            "mean_energy_diff": 0.05,
            "modes_consistent": True,
        },
    }
    (scaling_dir / "mps_mode_comparison.json").write_text(json.dumps(mc_data, indent=2))

    # Scan
    scanner = ResultScanner(results_root=Path(tmpdir))
    results = scanner.scan_scaling()
    assert len(results) == 1
    assert results[0].n_qubits == 50
    assert results[0].n_pass == 4
    print("✅ Test 6a: scan_scaling parses synthetic data")

    n120_parsed = scanner.scan_n120_sweep()
    assert n120_parsed is not None
    assert n120_parsed.n_qubits == 120
    assert n120_parsed.scaling_law_validated is True
    print("✅ Test 6b: scan_n120_sweep parses synthetic data")

    mc_parsed = scanner.scan_mode_comparison()
    assert mc_parsed is not None
    assert mc_parsed.mean_speedup == 160.0
    assert mc_parsed.modes_consistent is True
    print("✅ Test 6c: scan_mode_comparison parses synthetic data")

# ═══════════════════════════════════════════════════════════════════════
# Test 7: scaling_analyzer validate_scaling_law (corrected formula)
# ═══════════════════════════════════════════════════════════════════════
from project_health.analysis.scaling_analyzer import (
    ScalingPointResult,
    ScalingRunSummary,
    build_cross_n_comparison,
    detect_anomalies,
    validate_scaling_law,
)

run = ScalingRunSummary(
    n_qubits=40,
    topology="chain_1d",
    strategy="aer_mps",
    chi_max=64,
    precision=0.005,
    seed=42,
    p_layers=1,
    h_values=[4.0, 4.5, 5.0],
    phase1_time_s=10,
    phase2_time_s=100,
    total_time_s=110,
    n_pass=3,
    n_total=3,
    all_passed=True,
    mean_de_gap=0.003,
    max_de_gap=0.005,
    min_de_gap=0.001,
    per_h_results=[
        ScalingPointResult(
            h=4.0,
            vqe_energy=-200.0,
            dmrg_energy=-200.01,
            gap=7.0,
            de_gap=0.003,
            time_s=30,
            passed=True,
        ),
        ScalingPointResult(
            h=4.5,
            vqe_energy=-225.0,
            dmrg_energy=-225.01,
            gap=7.5,
            de_gap=0.004,
            time_s=30,
            passed=True,
        ),
        ScalingPointResult(
            h=5.0,
            vqe_energy=-250.0,
            dmrg_energy=-250.01,
            gap=8.0,
            de_gap=0.005,
            time_s=30,
            passed=True,
        ),
    ],
)
v = validate_scaling_law(run)
# Corrected formula: h_min = 1.5 + 0.020 * 40^1.31 ≈ 4.01
assert v.n_qubits == 40
assert v.actual_h_min == 4.0
assert abs(v.predicted_h_min - 4.01) < 0.05, f"predicted={v.predicted_h_min}"
assert v.prediction_error < 0.5
assert v.within_tolerance is True
print("✅ Test 7: validate_scaling_law uses corrected formula (1.5 + 0.020·N^1.31)")

# ═══════════════════════════════════════════════════════════════════════
# Test 8: detect_anomalies
# ═══════════════════════════════════════════════════════════════════════
anomalies = detect_anomalies([run])
assert len(anomalies) == 0  # Normal run, no anomalies
print("✅ Test 8: detect_anomalies reports no issues for healthy run")

# Test with anomalous run
bad_run = ScalingRunSummary(
    n_qubits=40,
    topology="chain_1d",
    strategy="aer_mps",
    chi_max=64,
    precision=0.005,
    seed=42,
    p_layers=1,
    h_values=[4.0],
    phase1_time_s=500,
    phase2_time_s=8000,
    total_time_s=8500,
    n_pass=0,
    n_total=1,
    all_passed=False,
    mean_de_gap=1.5,
    max_de_gap=1.5,
    min_de_gap=1.5,
    per_h_results=[],
)
bad_anomalies = detect_anomalies([bad_run])
assert len(bad_anomalies) >= 2  # timing + all failed + max_de_gap
print(f"✅ Test 8b: detect_anomalies catches {len(bad_anomalies)} issues")

# ═══════════════════════════════════════════════════════════════════════
# Test 9: build_cross_n_comparison
# ═══════════════════════════════════════════════════════════════════════
run80 = ScalingRunSummary(
    n_qubits=80,
    topology="chain_1d",
    strategy="aer_mps",
    chi_max=64,
    precision=0.005,
    seed=42,
    p_layers=1,
    h_values=[8.0, 8.5],
    phase1_time_s=20,
    phase2_time_s=60,
    total_time_s=80,
    n_pass=2,
    n_total=2,
    all_passed=True,
    mean_de_gap=0.001,
    max_de_gap=0.001,
    min_de_gap=0.001,
    per_h_results=[],
)
comparison = build_cross_n_comparison([run, run80])
assert comparison is not None
assert comparison.n_values == [40, 80]
assert comparison.scaling_law_valid is True
print("✅ Test 9: build_cross_n_comparison aggregates correctly")

# ═══════════════════════════════════════════════════════════════════════
# Test 10: CLI --kind scaling (dry-run import test)
# ═══════════════════════════════════════════════════════════════════════
from project_health.digest.__main__ import parse_args

# Just verify the parser accepts --kind scaling
sys.argv = ["digest", "--kind", "scaling", "--results-dir", "/nonexistent"]
args = parse_args()
assert args.kind == "scaling"
print("✅ Test 10: CLI parser accepts --kind scaling")

print("\n" + "=" * 50)
print("ALL 10 TESTS PASSED ✅")
print("=" * 50)
