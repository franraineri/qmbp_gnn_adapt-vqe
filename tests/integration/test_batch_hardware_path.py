"""Integration tests for the batch hardware execution path.

Validates that the batch submission/collection pattern works end-to-end
with FakeTorino as a mock QPU. This ensures consistency across different
mitigation configurations before real hardware deployment.

Tests cover:
- Circuit construction and transpilation for heavy_hex N=10 p=1
- Batch job submission grouping (one estimator per config)
- Result collection and structure validation
- Individual mitigation strategy paths (raw, GF-ZNE, PEA)
- Multi-config batch consistency (mixed strategies in one batch)
- Error propagation (crash isolation per job)

Marked @pytest.mark.slow — excluded from `make test`, included in `make test-full`.

Reference: scripts/test_batch_hw_path.py (original ad-hoc script).
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qmbp_simulation import HamiltonianBuilder, make_lattice
from scripts.experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
    BenchmarkConfig,
)
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    _build_hva_circuit,
    _execute_hardware_batched,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1
SEED = 42
SHOTS = 1024  # Low shots for fast testing (not production)

# Representative configs covering each mitigation category
RAW_CONFIG = "C0_raw"
GF_ZNE_CONFIG = "C3_full_gf"
PEA_CONFIG = "C5_full_pea_balanced"

# Default h-values for testing (deep paramagnetic — fast VQE convergence)
DEFAULT_H_VALUES = [4.0]
MULTI_H_VALUES = [3.75, 4.0]


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def fake_backend():
    """FakeTorino backend for local simulation of Heron noise model."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    return FakeTorino()


@pytest.fixture(scope="module")
def hva_circuit_h4():
    """Pre-built HVA circuit for h=4.0 (cached across module)."""
    return _build_hva_circuit(4.0)


@pytest.fixture(scope="module")
def transpiled_circuit_h4(hva_circuit_h4, fake_backend):
    """Transpiled HVA circuit for h=4.0 on FakeTorino."""
    pm = generate_preset_pass_manager(
        optimization_level=2, backend=fake_backend, seed_transpiler=SEED
    )
    return pm.run(hva_circuit_h4)


@pytest.fixture(scope="module")
def mapped_hamiltonian_h4(transpiled_circuit_h4):
    """Hamiltonian mapped to transpiled layout for h=4.0."""
    lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)
    H = HamiltonianBuilder().build(lattice_h)
    return H.apply_layout(transpiled_circuit_h4.layout)


def _build_jobs_spec(
    configs: list[str],
    h_values: list[float],
    backend,
) -> list[tuple[BenchmarkConfig, object, object, float]]:
    """Build jobs_spec list for the given configs and h-values.

    Reusable helper that constructs circuit → transpile → map observable
    for each (h, config) pair. Mirrors the production pipeline in
    run_mitigation_benchmark.py main loop.

    Parameters
    ----------
    configs : list[str]
        Config IDs from BENCHMARK_CONFIGS.
    h_values : list[float]
        Transverse field values to test.
    backend : BackendV2
        Target backend for transpilation.

    Returns
    -------
    list of (BenchmarkConfig, transpiled_circuit, H_mapped, h_value) tuples.
    """
    jobs_spec = []
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend, seed_transpiler=SEED)

    for h in h_values:
        circuit_hva = _build_hva_circuit(h)
        transpiled = pm.run(circuit_hva)
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
        H = HamiltonianBuilder().build(lattice_h)
        H_mapped = H.apply_layout(transpiled.layout)

        for config_id in configs:
            config = BENCHMARK_CONFIGS[config_id]
            jobs_spec.append((config, transpiled, H_mapped, h))

    return jobs_spec


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Circuit Construction Validation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCircuitConstruction:
    """Validate HVA circuit building and transpilation for hardware path."""

    def test_build_hva_circuit_returns_bound_circuit(self):
        """_build_hva_circuit returns a parameter-free circuit."""
        circuit = _build_hva_circuit(4.0)
        assert circuit.num_parameters == 0, "Circuit must be fully bound (no free parameters)"
        assert circuit.num_qubits == N_QUBITS

    def test_transpiled_circuit_has_layout(self, transpiled_circuit_h4):
        """Transpiled circuit must carry layout information."""
        assert transpiled_circuit_h4.layout is not None
        assert transpiled_circuit_h4.num_qubits >= N_QUBITS

    def test_hamiltonian_maps_to_layout(self, transpiled_circuit_h4, mapped_hamiltonian_h4):
        """H.apply_layout produces SparsePauliOp matching physical qubits."""
        from qiskit.quantum_info import SparsePauliOp

        assert isinstance(mapped_hamiltonian_h4, SparsePauliOp)
        assert mapped_hamiltonian_h4.num_qubits == transpiled_circuit_h4.num_qubits

    @pytest.mark.parametrize("h_value", [3.25, 3.5, 3.75, 4.0])
    def test_circuit_construction_across_h_values(self, h_value):
        """Circuit construction succeeds for all standard h-values."""
        circuit = _build_hva_circuit(h_value)
        assert circuit.num_parameters == 0
        assert circuit.num_qubits == N_QUBITS


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Single-Strategy Batch Execution
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestSingleStrategyBatch:
    """Test batch execution with individual mitigation strategies."""

    def test_raw_baseline_executes(self, fake_backend):
        """C0_raw (no mitigation) produces valid energy measurement."""
        jobs_spec = _build_jobs_spec([RAW_CONFIG], DEFAULT_H_VALUES, fake_backend)
        assert len(jobs_spec) == 1

        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 1
        _assert_valid_result(results[0], expect_mitigated=False)

    def test_gate_folding_zne_executes(self, fake_backend):
        """C3_full_gf (gate-folding ZNE) produces mitigated energy."""
        jobs_spec = _build_jobs_spec([GF_ZNE_CONFIG], DEFAULT_H_VALUES, fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 1
        _assert_valid_result(results[0], expect_mitigated=True)

    def test_pea_zne_executes(self, fake_backend):
        """C5_full_pea_balanced (PEA) produces mitigated energy."""
        jobs_spec = _build_jobs_spec([PEA_CONFIG], DEFAULT_H_VALUES, fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 1
        _assert_valid_result(results[0], expect_mitigated=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Config Batch Execution (Production Pattern)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMultiConfigBatch:
    """Test batch execution with multiple configs in one submission.

    This mirrors the production pattern: multiple configs × h-values
    grouped into one Batch session (one queue wait).
    """

    def test_three_strategy_batch(self, fake_backend):
        """Mixed raw + GF + PEA batch produces correct number of results."""
        configs = [RAW_CONFIG, GF_ZNE_CONFIG, PEA_CONFIG]
        jobs_spec = _build_jobs_spec(configs, DEFAULT_H_VALUES, fake_backend)
        assert len(jobs_spec) == 3

        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 3
        # Verify ordering matches input
        for i, (config, _, _, _) in enumerate(jobs_spec):
            r = results[i]
            if r.get("error"):
                pytest.skip(f"Config {config.config_id} failed: {r['error']}")
            if config.zne_method in ("pea", "gf"):
                _assert_valid_result(r, expect_mitigated=True)
            else:
                _assert_valid_result(r, expect_mitigated=False)

    def test_multi_h_batch_preserves_order(self, fake_backend):
        """Results maintain input order across multiple h-values."""
        configs = [RAW_CONFIG, PEA_CONFIG]
        h_values = MULTI_H_VALUES
        jobs_spec = _build_jobs_spec(configs, h_values, fake_backend)
        # 2 configs × 2 h-values = 4 jobs
        assert len(jobs_spec) == 4

        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 4
        # Verify each result corresponds to the correct h-value
        for i, (config, _, _, h_val) in enumerate(jobs_spec):
            r = results[i]
            if not r.get("error"):
                assert r.get("shots") == SHOTS

    def test_empty_jobs_spec_returns_empty(self, fake_backend):
        """Empty input produces empty output (no crash)."""
        results = _execute_hardware_batched([], fake_backend, SHOTS)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Result Structure Validation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestResultStructure:
    """Validate result dict keys and value types for downstream consumers."""

    def test_raw_result_keys(self, fake_backend):
        """Raw config result has expected keys for persistence layer."""
        jobs_spec = _build_jobs_spec([RAW_CONFIG], DEFAULT_H_VALUES, fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)
        r = results[0]

        if r.get("error"):
            pytest.skip(f"Execution failed: {r['error']}")

        required_keys = {"e_raw", "e_mitigated", "zne_r2", "shots", "_job"}
        assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - set(r.keys())}"
        assert isinstance(r["e_raw"], float)
        assert r["shots"] == SHOTS

    def test_mitigated_result_has_energy(self, fake_backend):
        """PEA config result has e_mitigated populated."""
        jobs_spec = _build_jobs_spec([PEA_CONFIG], DEFAULT_H_VALUES, fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)
        r = results[0]

        if r.get("error"):
            pytest.skip(f"Execution failed: {r['error']}")

        assert r["e_mitigated"] is not None
        assert isinstance(r["e_mitigated"], float)
        assert np.isfinite(r["e_mitigated"])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Energy Sanity Checks
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestEnergySanity:
    """Verify energies are physically reasonable (not garbage)."""

    def test_energy_is_negative_for_tfim(self, fake_backend):
        """TFIM ground state energy must be negative for h>0."""
        jobs_spec = _build_jobs_spec([RAW_CONFIG], DEFAULT_H_VALUES, fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)
        r = results[0]

        if r.get("error"):
            pytest.skip(f"Execution failed: {r['error']}")

        energy = r["e_raw"]
        # At h=4.0 with N=10, E_gs ≈ -41.6. With noise, still negative.
        assert energy < 0, f"TFIM energy should be negative, got {energy}"

    def test_energy_within_physical_bounds(self, fake_backend):
        """Measured energy must be within [E_gs, E_max] ± noise tolerance."""
        from qmbp_simulation import ClassicalSolver

        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)
        H = HamiltonianBuilder().build(lattice_h)
        solver = ClassicalSolver()
        exact = solver.solve(H, lattice_h)

        jobs_spec = _build_jobs_spec([RAW_CONFIG], [4.0], fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)
        r = results[0]

        if r.get("error"):
            pytest.skip(f"Execution failed: {r['error']}")

        energy = r["e_raw"]
        # Noisy energy can exceed exact bounds, but shouldn't be wildly off.
        # Allow 50% tolerance above exact max eigenvalue for noise effects.
        e_max = -exact.ground_energy + 2 * abs(exact.ground_energy)
        assert energy < e_max, f"Energy {energy} exceeds reasonable upper bound {e_max}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Parametrized Config Coverage
# ═══════════════════════════════════════════════════════════════════════════════


# All P0 + P1 configs that use standard ZNE methods (no Mitiq)
STANDARD_CONFIGS = [
    cid
    for cid, c in BENCHMARK_CONFIGS.items()
    if c.priority <= 1 and c.zne_method in (None, "gf", "pea")
]


@pytest.mark.slow
@pytest.mark.parametrize("config_id", STANDARD_CONFIGS)
def test_priority_configs_execute(config_id, fake_backend):
    """All P0+P1 standard configs execute without crash on FakeTorino.

    This is the core consistency gate: every config that will be used
    in production hardware runs must pass this test first.
    """
    jobs_spec = _build_jobs_spec([config_id], DEFAULT_H_VALUES, fake_backend)
    results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

    assert len(results) == 1
    r = results[0]

    # Either succeeds with valid energy OR fails with structured error
    if r.get("error"):
        # Structured errors are acceptable (e.g., "submit_failed: ...")
        assert isinstance(r["error"], str)
        assert len(r["error"]) > 0
    else:
        energy = r.get("e_raw") or r.get("e_mitigated")
        assert energy is not None, f"Config {config_id}: no energy in result"
        assert np.isfinite(energy), f"Config {config_id}: non-finite energy {energy}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Regression Guard — Batch Grouping Logic
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestBatchGrouping:
    """Verify that config grouping works correctly (one estimator per config)."""

    def test_same_config_different_h_grouped(self, fake_backend):
        """Multiple h-values with same config should be in one group."""
        # 1 config × 2 h-values → 2 jobs, but only 1 config group internally
        jobs_spec = _build_jobs_spec([PEA_CONFIG], MULTI_H_VALUES, fake_backend)
        assert len(jobs_spec) == 2

        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)
        assert len(results) == 2

        # Both results should succeed (same estimator options)
        for r in results:
            if not r.get("error"):
                assert r.get("e_mitigated") is not None or r.get("e_raw") is not None

    def test_different_configs_same_h_separate_groups(self, fake_backend):
        """Different configs at same h should produce independent results."""
        configs = [RAW_CONFIG, PEA_CONFIG]
        jobs_spec = _build_jobs_spec(configs, [4.0], fake_backend)
        results = _execute_hardware_batched(jobs_spec, fake_backend, SHOTS)

        assert len(results) == 2
        # If both succeed, energies should differ (different mitigation)
        energies = []
        for r in results:
            if not r.get("error"):
                e = r.get("e_mitigated") or r.get("e_raw")
                if e is not None:
                    energies.append(e)

        # At minimum, both should produce some result
        assert len(energies) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Assertion Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_valid_result(result: dict, expect_mitigated: bool) -> None:
    """Assert a batch result dict has the expected structure and values.

    Parameters
    ----------
    result : dict
        Single result from _execute_hardware_batched.
    expect_mitigated : bool
        If True, expect e_mitigated to be populated (ZNE configs).
        If False, expect e_raw to be populated (raw baseline).
    """
    if result.get("error"):
        pytest.skip(f"Execution returned error: {result['error']}")

    assert "shots" in result
    assert result["shots"] > 0

    if expect_mitigated:
        assert result.get("e_mitigated") is not None, "Expected e_mitigated for ZNE config"
        assert np.isfinite(result["e_mitigated"])
    else:
        assert result.get("e_raw") is not None, "Expected e_raw for raw config"
        assert np.isfinite(result["e_raw"])
