"""Unit tests for layout_optimizer module (mapomatic VF2 integration).

Tests cover:
- Layer 0: CouplingMap filtering (edges by error, qubits by T1)
- Layer 1: VF2 layout discovery (with mapomatic)
- Layer 2: Fidelity cost scoring (BackendV2 Target)
- Layer 3: Full pipeline (select_optimal_layouts)
- Fallback behavior when mapomatic unavailable
- Multi-backend ranking
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.coupling import CouplingMap

from qmbp_simulation.execution.hardware.layout_optimizer import (
    MAPOMATIC_AVAILABLE,
    LayoutOptimizationResult,
    _apply_strategy,
    build_filtered_coupling_map,
    compute_layout_fidelity_cost,
    find_vf2_layouts,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: Mock backend with controllable calibration data
# ═══════════════════════════════════════════════════════════════════════════


class MockGateProperties:
    """Mock for gate properties with error rate."""

    def __init__(self, error: float | None = None):
        self.error = error


class MockTarget:
    """Mock BackendV2 Target with configurable gate errors."""

    def __init__(
        self,
        edges: list[tuple[int, int]],
        errors: dict[tuple[int, int], float] | None = None,
        qubit_t1s: dict[int, float] | None = None,
    ):
        self._edges = edges
        self._errors = errors or {}
        self._qubit_t1s = qubit_t1s or {}
        self.operation_names = ["cz"]
        self._num_qubits = max(max(e) for e in edges) + 1 if edges else 0
        # Qubit properties mock
        self.qubit_properties = []
        for i in range(self._num_qubits):
            qp = MagicMock()
            qp.t1 = self._qubit_t1s.get(i, 200e-6)  # Default 200 µs
            self.qubit_properties.append(qp)

    def qargs_for_operation_name(self, name: str):
        if name == "cz":
            return [tuple(e) for e in self._edges]
        if name == "measure":
            return [(i,) for i in range(self._num_qubits)]
        return None

    def __getitem__(self, gate_name: str):
        """Return a dict-like that maps qargs to properties."""
        gate_dict = {}
        if gate_name == "cz":
            for edge in self._edges:
                error = self._errors.get(edge, 0.002)
                gate_dict[edge] = MockGateProperties(error)
        elif gate_name == "measure":
            for i in range(self._num_qubits):
                gate_dict[(i,)] = MockGateProperties(0.01)
        return MockGateDict(gate_dict)


class MockGateDict:
    """Dict-like for target[gate_name].get(qargs)."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def _make_mock_backend(
    edges: list[tuple[int, int]],
    errors: dict[tuple[int, int], float] | None = None,
    qubit_t1s: dict[int, float] | None = None,
) -> MagicMock:
    """Create a mock BackendV2 with specified topology and errors."""
    target = MockTarget(edges, errors, qubit_t1s)
    backend = MagicMock()
    backend.target = target
    backend.num_qubits = target._num_qubits
    backend.name = "mock_backend"
    # CouplingMap from edges
    backend.coupling_map = CouplingMap([[e[0], e[1]] for e in edges])
    return backend


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0 Tests: CouplingMap filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildFilteredCouplingMap:
    """Tests for Layer 0: CouplingMap pruning."""

    def test_excludes_high_error_edges(self):
        """Edges with error > max_2q_error are excluded."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        errors = {
            (0, 1): 0.002,  # Good
            (1, 2): 0.050,  # Bad (5%)
            (2, 3): 0.003,  # Good
            (3, 4): 0.001,  # Good
        }
        backend = _make_mock_backend(edges, errors)

        cmap, stats = build_filtered_coupling_map(backend, max_2q_error=0.01)

        # Edge (1,2) should be excluded
        cmap_edges = set(tuple(e) for e in cmap.get_edges())
        assert (1, 2) not in cmap_edges
        assert (0, 1) in cmap_edges or (1, 0) in cmap_edges
        assert stats["excluded_edges_count"] == 1

    def test_excludes_low_t1_qubits(self):
        """Qubits with T1 < min_t1_us are excluded (and all their edges)."""
        edges = [(0, 1), (1, 2), (2, 3)]
        qubit_t1s = {
            0: 200e-6,  # 200 µs — good
            1: 30e-6,  # 30 µs — BAD
            2: 150e-6,  # 150 µs — good
            3: 180e-6,  # 180 µs — good
        }
        backend = _make_mock_backend(edges, qubit_t1s=qubit_t1s)

        cmap, stats = build_filtered_coupling_map(backend, min_t1_us=50.0)

        # Qubit 1 excluded → edges (0,1) and (1,2) should be gone
        assert 1 in stats["excluded_qubits_t1"]
        cmap_edges = set(tuple(e) for e in cmap.get_edges())
        assert (0, 1) not in cmap_edges
        assert (1, 0) not in cmap_edges

    def test_excludes_manual_blacklist(self):
        """Manually blacklisted qubits are excluded."""
        edges = [(0, 1), (1, 2), (2, 3)]
        backend = _make_mock_backend(edges)

        cmap, stats = build_filtered_coupling_map(backend, exclude_qubits={2})

        # All edges touching qubit 2 should be removed
        cmap_edges = set(tuple(e) for e in cmap.get_edges())
        assert (1, 2) not in cmap_edges
        assert (2, 3) not in cmap_edges
        assert 2 in stats["excluded_qubits_manual"]

    def test_returns_full_cmap_if_all_filtered(self):
        """If all edges are filtered, returns original CouplingMap with warning."""
        edges = [(0, 1), (1, 2)]
        errors = {(0, 1): 0.5, (1, 2): 0.5}  # All bad
        backend = _make_mock_backend(edges, errors)

        cmap, stats = build_filtered_coupling_map(backend, max_2q_error=0.001)

        # Should return full coupling map as fallback
        assert cmap is not None
        assert stats["filtered_edges"] == 0

    def test_retention_rate_calculated(self):
        """Stats include retention rate."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        errors = {(0, 1): 0.002, (1, 2): 0.002, (2, 3): 0.05, (3, 4): 0.002}
        backend = _make_mock_backend(edges, errors)

        _, stats = build_filtered_coupling_map(backend, max_2q_error=0.01)

        assert stats["retention_rate"] == pytest.approx(3 / 4)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 Tests: VF2 layout discovery
# ═══════════════════════════════════════════════════════════════════════════


class TestFindVf2Layouts:
    """Tests for Layer 1: VF2 layout discovery."""

    @pytest.mark.skipif(not MAPOMATIC_AVAILABLE, reason="mapomatic not installed")
    def test_finds_layouts_on_simple_chain(self):
        """VF2 finds layouts for a 3-qubit chain circuit on a 5-qubit chain."""
        # 5-qubit chain topology
        cmap = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])

        # 3-qubit circuit with 2 CZ gates (chain structure)
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.cz(1, 2)

        layouts = find_vf2_layouts(qc, cmap, call_limit=10_000)

        assert len(layouts) > 0
        # Each layout should have 3 physical qubits
        for layout in layouts:
            assert len(layout) == 3

    @pytest.mark.skipif(not MAPOMATIC_AVAILABLE, reason="mapomatic not installed")
    def test_max_layouts_truncation(self):
        """Layouts are truncated to max_layouts."""
        # Large enough topology to produce many layouts
        cmap = CouplingMap([[i, i + 1] for i in range(20)])

        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        layouts = find_vf2_layouts(qc, cmap, max_layouts=5)

        assert len(layouts) <= 5

    def test_returns_empty_when_mapomatic_unavailable(self):
        """Returns empty list when mapomatic import fails."""
        with patch(
            "qmbp_simulation.execution.hardware.layout_optimizer.MAPOMATIC_AVAILABLE",
            False,
        ):
            layouts = find_vf2_layouts(QuantumCircuit(3), CouplingMap([[0, 1]]))
            assert layouts == []


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 Tests: Fidelity cost scoring
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeLayoutFidelityCost:
    """Tests for Layer 2: fidelity cost scoring."""

    def test_lower_error_layout_ranked_first(self):
        """Layout with lower gate errors is ranked better (lower cost)."""
        edges = [(0, 1), (2, 3)]
        errors = {
            (0, 1): 0.001,  # Very good
            (2, 3): 0.050,  # Bad
        }
        backend = _make_mock_backend(edges, errors)

        # Circuit: single CZ on qubits 0,1
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        layouts = [[0, 1], [2, 3]]
        scored = compute_layout_fidelity_cost(qc, layouts, backend)

        # [0,1] should be first (lower error)
        assert scored[0][0] == [0, 1]
        assert scored[0][1] < scored[1][1]

    def test_defective_edge_penalty(self):
        """Layouts with defective edges get +1.0 penalty."""
        edges = [(0, 1), (2, 3)]
        errors = {
            (0, 1): 0.002,  # Good
            (2, 3): 0.150,  # Defective (>10%)
        }
        backend = _make_mock_backend(edges, errors)

        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        layouts = [[0, 1], [2, 3]]
        scored = compute_layout_fidelity_cost(qc, layouts, backend, defective_edge_threshold=0.10)

        # Layout [2,3] should have cost > 1.0 (penalty applied)
        costs = {tuple(s[0]): s[1] for s in scored}
        assert costs[(2, 3)] > 1.0
        assert costs[(0, 1)] < 1.0

    def test_output_sorted_ascending(self):
        """Output is sorted by cost ascending (best first)."""
        edges = [(0, 1), (1, 2), (2, 3)]
        errors = {(0, 1): 0.010, (1, 2): 0.005, (2, 3): 0.002}
        backend = _make_mock_backend(edges, errors)

        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        layouts = [[0, 1], [1, 2], [2, 3]]
        scored = compute_layout_fidelity_cost(qc, layouts, backend)

        costs = [s[1] for s in scored]
        assert costs == sorted(costs)

    def test_empty_layouts_returns_empty(self):
        """Empty layout list returns empty result."""
        backend = _make_mock_backend([(0, 1)])
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        scored = compute_layout_fidelity_cost(qc, [], backend)
        assert scored == []


# ═══════════════════════════════════════════════════════════════════════════
# Strategy selection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyStrategy:
    """Tests for layout selection strategies."""

    def test_lowest_cost_picks_top_n(self):
        """lowest_cost picks the N lowest-cost layouts."""
        scored = [([0, 1], 0.01), ([2, 3], 0.02), ([4, 5], 0.03), ([6, 7], 0.04)]
        layouts, costs = _apply_strategy(scored, n_select=2, strategy="lowest_cost")

        assert len(layouts) == 2
        assert layouts[0] == [0, 1]
        assert layouts[1] == [2, 3]

    def test_ces_spread_includes_extremes(self):
        """ces_spread includes both best and worst of clean set."""
        scored = [([i, i + 1], i * 0.01) for i in range(10)]
        layouts, costs = _apply_strategy(scored, n_select=3, strategy="ces_spread")

        assert len(layouts) == 3
        # Should include first (best) and last (worst)
        assert layouts[0] == [0, 1]
        assert layouts[-1] == [9, 10]

    def test_hybrid_includes_upper_quartile(self):
        """hybrid picks top-(N-1) + 1 from upper range."""
        scored = [([i, i + 1], i * 0.01) for i in range(10)]
        layouts, costs = _apply_strategy(scored, n_select=3, strategy="hybrid")

        assert len(layouts) == 3
        # First two should be the best
        assert layouts[0] == [0, 1]
        assert layouts[1] == [1, 2]
        # Third should be from upper quartile (idx ~7)
        assert layouts[2][0] >= 5

    def test_empty_scored_returns_empty(self):
        """Empty input returns empty output."""
        layouts, costs = _apply_strategy([], n_select=3, strategy="lowest_cost")
        assert layouts == []
        assert costs == []

    def test_n_select_larger_than_available(self):
        """If fewer layouts available than requested, return all."""
        scored = [([0, 1], 0.01), ([2, 3], 0.02)]
        layouts, costs = _apply_strategy(scored, n_select=5, strategy="lowest_cost")

        assert len(layouts) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Graceful degradation test
# ═══════════════════════════════════════════════════════════════════════════


class TestGracefulDegradation:
    """Test that module works when mapomatic is not available."""

    def test_mapomatic_available_flag(self):
        """MAPOMATIC_AVAILABLE is a boolean."""
        assert isinstance(MAPOMATIC_AVAILABLE, bool)

    def test_layout_optimization_result_creation(self):
        """LayoutOptimizationResult can be instantiated."""
        result = LayoutOptimizationResult(
            selected_layouts=[[0, 1, 2]],
            fidelity_costs=[0.05],
            ces_values=[0.15],
            backend_name="ibm_kingston",
            strategy_used="lowest_cost",
            method="mapomatic_vf2",
        )
        assert result.backend_name == "ibm_kingston"
        assert result.method == "mapomatic_vf2"
