"""End-to-end integration tests for mapomatic layout optimizer.

Validates all 8 integration points of the VF2 layout optimizer
across the hardware pipeline:
1. Public API imports
2. HardwareConfig mapomatic fields
3. submission.py dispatcher (VF2 vs BFS)
4. Full pipeline quality comparison (VF2 vs BFS on FakeTorino)
5. Structured log events (method field)
6. project_health analyzer
7. Sanity check (hardware_readiness)
8. BenchmarkSuite component

Requires: qiskit-ibm-runtime (FakeTorino) + mapomatic.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from qmbp_simulation.execution.hardware.config import HardwareConfig
from qmbp_simulation.execution.hardware.layout_optimizer import (
    MAPOMATIC_AVAILABLE,
)

try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    FAKE_TORINO_AVAILABLE = True
except ImportError:
    FAKE_TORINO_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FAKE_TORINO_AVAILABLE,
        reason="qiskit-ibm-runtime fake provider not available",
    ),
    pytest.mark.skipif(not MAPOMATIC_AVAILABLE, reason="mapomatic not installed"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def backend():
    """FakeTorino backend (133 qubits, heavy-hex)."""
    return FakeTorino()


@pytest.fixture(scope="module")
def hva_circuit_n10():
    """HVA-like N=10 chain p=1 circuit (bound)."""
    qc = QuantumCircuit(10)
    for i in range(9):
        qc.cz(i, i + 1)
    for i in range(10):
        qc.rx(0.5, i)
    return qc


@pytest.fixture(scope="module")
def logger():
    """StructuredLogger for capturing events."""
    from qmbp_simulation.framework.logging import StructuredLogger

    return StructuredLogger("test_mapomatic_e2e")


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Public API imports
# ═══════════════════════════════════════════════════════════════════════════


class TestMapomaticPublicAPI:
    """Verify all public API symbols are importable."""

    def test_all_exports_accessible(self):
        """All layout optimizer exports are importable from hardware package."""
        from qmbp_simulation.execution.hardware import (
            MAPOMATIC_AVAILABLE,
            LayoutOptimizationResult,
            build_filtered_coupling_map,
            compute_layout_fidelity_cost,
            find_vf2_layouts,
            rank_backends,
            select_optimal_layouts,
        )

        assert MAPOMATIC_AVAILABLE is True
        assert callable(build_filtered_coupling_map)
        assert callable(find_vf2_layouts)
        assert callable(compute_layout_fidelity_cost)
        assert callable(select_optimal_layouts)
        assert callable(rank_backends)
        assert LayoutOptimizationResult is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: HardwareConfig fields
# ═══════════════════════════════════════════════════════════════════════════


class TestHardwareConfigMapomatic:
    """Verify HardwareConfig contains all mapomatic fields with correct defaults."""

    def test_all_fields_present(self):
        config = HardwareConfig()
        assert hasattr(config, "use_mapomatic")
        assert hasattr(config, "layout_max_2q_error")
        assert hasattr(config, "layout_min_t1_us")
        assert hasattr(config, "layout_call_limit")
        assert hasattr(config, "layout_exclude_qubits")
        assert hasattr(config, "layout_strategy")

    def test_defaults_are_correct(self):
        config = HardwareConfig()
        assert config.use_mapomatic is True
        assert config.layout_max_2q_error == 0.01
        assert config.layout_min_t1_us == 50.0
        assert config.layout_call_limit == 100_000
        assert config.layout_exclude_qubits == []
        assert config.layout_strategy == "lowest_cost"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: submission.py dispatcher
# ═══════════════════════════════════════════════════════════════════════════


class TestSubmissionDispatcher:
    """Verify select_layouts_for_hardware dispatches VF2 vs BFS correctly."""

    def test_vf2_path_when_enabled(self, backend, hva_circuit_n10, logger):
        """VF2 path is used when use_mapomatic=True."""
        config = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        result = select_layouts_for_hardware(hva_circuit_n10, backend, config, logger)

        assert len(result.layouts) >= 3  # May escalate via P2-A dynamic layout
        assert len(result.transpiled_circuits) >= 3
        assert len(result.ces_values) >= 3
        assert all(ces > 0 for ces in result.ces_values)

    def test_bfs_path_when_disabled(self, backend, hva_circuit_n10, logger):
        """BFS path is used when use_mapomatic=False."""
        config = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=False,
            max_ces=1.0,
        )
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        result = select_layouts_for_hardware(hva_circuit_n10, backend, config, logger)

        assert len(result.layouts) >= 3  # May escalate via P2-A
        assert len(result.transpiled_circuits) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Quality comparison (VF2 vs BFS)
# ═══════════════════════════════════════════════════════════════════════════


class TestVF2QualityAdvantage:
    """Verify VF2 produces measurably better layouts than BFS."""

    def test_vf2_has_lower_ces_than_bfs(self, backend, hva_circuit_n10, logger):
        """VF2 layouts have lower CES (less accumulated error) than BFS."""
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        config_vf2 = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        config_bfs = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=False,
            max_ces=1.0,
        )

        result_vf2 = select_layouts_for_hardware(hva_circuit_n10, backend, config_vf2, logger)
        result_bfs = select_layouts_for_hardware(hva_circuit_n10, backend, config_bfs, logger)

        vf2_mean_ces = float(np.mean(result_vf2.ces_values))
        bfs_mean_ces = float(np.mean(result_bfs.ces_values))

        # VF2 should be at least 2× better (typically 5-6×)
        assert vf2_mean_ces < bfs_mean_ces
        improvement = bfs_mean_ces / vf2_mean_ces
        assert improvement > 2.0, f"VF2 only {improvement:.1f}× better, expected >2×"

    def test_vf2_layouts_are_valid_size(self, backend, hva_circuit_n10, logger):
        """VF2 layouts have exactly N=10 physical qubits each."""
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        config = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        result = select_layouts_for_hardware(hva_circuit_n10, backend, config, logger)

        for layout in result.layouts:
            assert len(layout) == 10, f"Layout should have 10 qubits, got {len(layout)}"
            assert all(isinstance(q, int) for q in layout)


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Structured log events
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredLogEvents:
    """Verify layout selection events are properly logged."""

    def test_vf2_method_logged(self, backend, hva_circuit_n10):
        """VF2 selection logs method='mapomatic_vf2'."""
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware
        from qmbp_simulation.framework.logging import StructuredLogger

        fresh_logger = StructuredLogger("log_test")
        config = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        select_layouts_for_hardware(hva_circuit_n10, backend, config, fresh_logger)

        method_events = [e for e in fresh_logger.events if e.event_type == "layout_method"]
        assert len(method_events) >= 1
        assert method_events[0].data["method"] == "mapomatic_vf2"

    def test_selection_event_has_ces_values(self, backend, hva_circuit_n10):
        """Layout selection event includes CES values and count."""
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware
        from qmbp_simulation.framework.logging import StructuredLogger

        fresh_logger = StructuredLogger("ces_test")
        config = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        select_layouts_for_hardware(hva_circuit_n10, backend, config, fresh_logger)

        selection_events = [e for e in fresh_logger.events if e.event_type == "layout_selection"]
        assert len(selection_events) >= 1
        data = selection_events[0].data
        assert "ces_values" in data
        assert "n_selected" in data
        assert data["n_selected"] == 3
        assert len(data["ces_values"]) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: project_health analyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestLayoutOptimizerAnalyzer:
    """Verify the project_health analyzer runs without errors."""

    def test_analyze_returns_report(self):
        """Analyzer produces a valid report."""
        from project_health.analysis.hardware.layout_optimizer_analyzer import analyze

        report = analyze(verbose=False)
        assert report.mapomatic_available is True
        assert report.mapomatic_version != ""
        assert isinstance(report.records, list)
        assert isinstance(report.recommendations, list)

    def test_report_to_dict_serializable(self):
        """Report is JSON-serializable."""
        import json

        from project_health.analysis.hardware.layout_optimizer_analyzer import analyze

        report = analyze(verbose=False)
        d = report.to_dict()
        # Should not raise
        json.dumps(d, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Sanity check
# ═══════════════════════════════════════════════════════════════════════════


class TestSanityCheckIntegration:
    """Verify the sanity check for layout optimizer passes."""

    def test_all_checks_pass(self):
        """All hardware_readiness sanity checks should pass."""
        from project_health.analysis.validation.sanity_check import check_layout_optimizer_integration

        checks = check_layout_optimizer_integration(verbose=False)
        assert len(checks) == 3
        for check in checks:
            assert check.passed, f"Check '{check.name}' failed: {check.message}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Benchmark component
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkComponent:
    """Verify the layout benchmark works and produces quality metrics."""

    def test_bench_layout_runs(self):
        """Layout benchmark executes and returns valid result."""
        from qmbp_simulation.framework.benchmarking import _bench_layout

        result = _bench_layout(6, 2)
        assert result.component == "layout"
        assert result.n_qubits == 6
        assert result.elapsed_s > 0
        assert result.details["mapomatic_available"] is True
        assert result.details["method"] == "vf2"

    def test_bench_layout_includes_quality_metrics(self):
        """Benchmark includes CES and SWAP metrics for comparison."""
        from qmbp_simulation.framework.benchmarking import _bench_layout

        result = _bench_layout(6, 2)
        details = result.details

        # VF2 metrics present
        assert "vf2_best_ces" in details
        assert "vf2_n_2q_gates" in details
        assert details["vf2_best_ces"] > 0

        # BFS metrics present
        assert "bfs_best_ces" in details
        assert "bfs_n_2q_gates" in details

        # Comparative metrics
        assert "ces_improvement_ratio" in details
        assert details["ces_improvement_ratio"] >= 1.0
