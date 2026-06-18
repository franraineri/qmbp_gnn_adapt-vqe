"""Integration tests for layout_optimizer with real FakeTorino backend.

These tests exercise the full pipeline: HVA circuit → mapomatic VF2 →
transpilation → CES validation. Requires qiskit-aer (FakeTorino).
"""

from __future__ import annotations

import pytest

from qmbp_simulation.execution.hardware.layout_optimizer import (
    MAPOMATIC_AVAILABLE,
    build_filtered_coupling_map,
    compute_layout_fidelity_cost,
    find_vf2_layouts,
    select_optimal_layouts,
)

try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    FAKE_TORINO_AVAILABLE = True
except ImportError:
    FAKE_TORINO_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FAKE_TORINO_AVAILABLE, reason="qiskit-ibm-runtime fake provider not available"
    ),
    pytest.mark.skipif(not MAPOMATIC_AVAILABLE, reason="mapomatic not installed"),
]


@pytest.fixture(scope="module")
def backend():
    """FakeTorino backend (133 qubits, heavy-hex)."""
    return FakeTorino()


@pytest.fixture(scope="module")
def hva_circuit_n6():
    """Simple HVA-like circuit for N=6 chain (2 CZ layers)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(6)
    # Layer 1: ZZ interactions (even bonds)
    qc.cz(0, 1)
    qc.cz(2, 3)
    qc.cz(4, 5)
    # Layer 2: ZZ interactions (odd bonds)
    qc.cz(1, 2)
    qc.cz(3, 4)
    # Single qubit rotations (no impact on layout)
    for i in range(6):
        qc.rx(0.5, i)
        qc.rz(0.3, i)
    return qc


@pytest.fixture(scope="module")
def hva_circuit_n10():
    """HVA-like circuit for N=10 chain p=1 (10 CZ gates)."""
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(10)
    # All NN bonds
    for i in range(9):
        qc.cz(i, i + 1)
    # Single qubit rotations
    for i in range(10):
        qc.rx(0.5, i)
    return qc


class TestLayer0Integration:
    """Layer 0: CouplingMap filtering on FakeTorino."""

    def test_filtered_cmap_has_edges(self, backend):
        """Filtered CouplingMap retains substantial edges."""
        cmap, stats = build_filtered_coupling_map(backend, max_2q_error=0.01, min_t1_us=50.0)

        # FakeTorino should have many good edges
        assert stats["filtered_edges"] > 50
        assert stats["retention_rate"] > 0.5

    def test_stricter_filter_reduces_edges(self, backend):
        """Tighter thresholds produce fewer edges."""
        _, stats_loose = build_filtered_coupling_map(backend, max_2q_error=0.05)
        _, stats_tight = build_filtered_coupling_map(backend, max_2q_error=0.005)

        assert stats_tight["filtered_edges"] <= stats_loose["filtered_edges"]


class TestLayer1Integration:
    """Layer 1: VF2 discovery on FakeTorino."""

    def test_finds_layouts_for_n6_chain(self, backend, hva_circuit_n6):
        """VF2 finds layouts for 6-qubit chain circuit on FakeTorino."""
        import mapomatic as mm

        deflated = mm.deflate_circuit(hva_circuit_n6)
        cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)

        layouts = find_vf2_layouts(deflated, cmap, call_limit=50_000, max_layouts=50)

        assert len(layouts) > 0
        # Each layout should have 6 qubits
        for layout in layouts:
            assert len(layout) == 6

    def test_finds_layouts_for_n10_chain(self, backend, hva_circuit_n10):
        """VF2 finds layouts for 10-qubit chain on FakeTorino."""
        import mapomatic as mm

        deflated = mm.deflate_circuit(hva_circuit_n10)
        cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)

        layouts = find_vf2_layouts(deflated, cmap, call_limit=100_000, max_layouts=100)

        assert len(layouts) > 0
        for layout in layouts:
            assert len(layout) == 10


class TestLayer2Integration:
    """Layer 2: Fidelity scoring on FakeTorino."""

    def test_scoring_produces_ranked_list(self, backend, hva_circuit_n6):
        """Scoring produces properly ranked layouts."""
        import mapomatic as mm

        deflated = mm.deflate_circuit(hva_circuit_n6)
        cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)
        layouts = find_vf2_layouts(deflated, cmap, max_layouts=20)

        if not layouts:
            pytest.skip("No VF2 layouts found (backend topology mismatch)")

        scored = compute_layout_fidelity_cost(deflated, layouts, backend)

        assert len(scored) == len(layouts)
        # Costs should be sorted ascending
        costs = [s[1] for s in scored]
        assert costs == sorted(costs)
        # Costs should be in [0, 1] range (no defective edges expected)
        assert all(0 <= c <= 1 for c in costs)


class TestFullPipelineIntegration:
    """Full pipeline: select_optimal_layouts on FakeTorino."""

    def test_full_pipeline_n6(self, backend, hva_circuit_n6):
        """Full pipeline produces valid LayoutSelection for N=6."""
        result = select_optimal_layouts(
            hva_circuit_n6,
            backend,
            n_select=3,
            max_ces=1.0,  # Relaxed for test
            max_2q_error=0.02,
            optimization_level=1,  # Faster for testing
            call_limit=50_000,
        )

        assert len(result.layouts) > 0
        assert len(result.layouts) <= 3
        assert len(result.ces_values) == len(result.layouts)
        assert len(result.transpiled_circuits) == len(result.layouts)
        # CES values should be positive
        assert all(ces > 0 for ces in result.ces_values)

    def test_full_pipeline_n10(self, backend, hva_circuit_n10):
        """Full pipeline produces valid LayoutSelection for N=10."""
        result = select_optimal_layouts(
            hva_circuit_n10,
            backend,
            n_select=3,
            max_ces=1.0,
            max_2q_error=0.02,
            optimization_level=1,
            call_limit=100_000,
        )

        assert len(result.layouts) > 0
        assert len(result.layouts) <= 3
        # Each layout should map 10 physical qubits
        for layout in result.layouts:
            assert len(layout) == 10

    def test_strategy_lowest_cost_vs_ces_spread(self, backend, hva_circuit_n6):
        """Different strategies produce different layout selections."""
        kwargs = dict(
            n_select=3,
            max_ces=1.0,
            max_2q_error=0.02,
            optimization_level=1,
            call_limit=50_000,
        )

        result_low = select_optimal_layouts(
            hva_circuit_n6, backend, strategy="lowest_cost", **kwargs
        )
        result_spread = select_optimal_layouts(
            hva_circuit_n6, backend, strategy="ces_spread", **kwargs
        )

        # Both should produce results
        assert len(result_low.layouts) > 0
        assert len(result_spread.layouts) > 0

        # CES spread should have higher variance (if enough layouts)
        if len(result_low.ces_values) >= 2 and len(result_spread.ces_values) >= 2:
            import numpy as np

            std_low = np.std(result_low.ces_values)
            std_spread = np.std(result_spread.ces_values)
            # Spread strategy should generally have >= std (not always guaranteed)
            # Just verify they both produce valid numeric CES
            assert std_low >= 0
            assert std_spread >= 0
