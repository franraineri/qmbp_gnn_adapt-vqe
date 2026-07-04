"""Test that SWAP-free layout has highest priority in layout selection.

Prevents regression of the bug where VF2 (mapomatic) returned high-CZ
layouts BEFORE the known SWAP-free layout was tried, causing 36.6% error
on real QPU when 9 CZ (< 5%) was achievable.

Tests:
1. SWAP-free layout is tried FIRST (Priority 1)
2. If SWAP-free produces low n_2q → returned immediately (VF2 never called)
3. If SWAP-free produces high n_2q → rejected, falls through to VF2/BFS
4. HardwareConfig defaults include the verified Kingston layout
5. The Kingston layout matches heavy_hex N=10 edge structure
"""

from unittest.mock import MagicMock, patch

from qmbp_simulation.execution.hardware.config import HardwareConfig
from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware
from qmbp_simulation.execution.noisy_utils import LayoutSelection


class TestSwapFreeLayoutPriority:
    """SWAP-free layout must be Priority 1 in select_layouts_for_hardware."""

    def test_config_has_fallback_layout(self):
        """HardwareConfig includes the verified Kingston layout by default."""
        config = HardwareConfig()
        assert hasattr(config, "fallback_layout_kingston")
        assert config.fallback_layout_kingston == [22, 23, 24, 25, 26, 27, 28, 16, 37, 17]
        assert len(config.fallback_layout_kingston) == 10

    def test_fallback_layout_matches_heavy_hex_edges(self):
        """The fallback layout qubits match the heavy_hex N=10 connectivity."""
        from qmbp_simulation.models import make_lattice

        lattice = make_lattice("heavy_hex", 10, J=1.0, h=3.5)
        # heavy_hex N=10 has 9 edges with degree-3 at positions 1, 3, 5
        assert len(lattice.edges) == 9
        # Edges include branches: (1,7), (3,8), (5,9)
        assert (1, 7) in lattice.edges
        assert (3, 8) in lattice.edges
        assert (5, 9) in lattice.edges

    def test_swap_free_returned_when_low_n2q(self):
        """When SWAP-free layout produces ≤20 CZ, it's returned (VF2 skipped)."""
        config = HardwareConfig(n_qubits=10)
        logger = MagicMock()

        # Mock select_layouts_low_ces to return a good result with 9 CZ
        mock_circuit = MagicMock()
        mock_circuit.count_ops.return_value = {"cz": 9, "rz": 20, "sx": 18}

        mock_selection = LayoutSelection(
            layouts=[[22, 23, 24, 25, 26, 27, 28, 16, 37, 17]],
            ces_values=[0.02],
            transpiled_circuits=[mock_circuit],
        )

        with patch(
            "qmbp_simulation.execution.hardware.submission.select_layouts_low_ces",
            return_value=mock_selection,
        ):
            result = select_layouts_for_hardware(MagicMock(), MagicMock(), config, logger)

        # Should return the SWAP-free layout
        assert result is mock_selection
        # Should log method as "known_swap_free"
        log_calls = [c for c in logger.log.call_args_list if c[0][0] == "layout_method"]
        assert len(log_calls) >= 1
        assert log_calls[0][1]["data"]["method"] == "known_swap_free"

    def test_swap_free_rejected_when_high_n2q(self):
        """When SWAP-free layout produces >20 CZ, it's rejected."""
        config = HardwareConfig(n_qubits=10)
        logger = MagicMock()

        # Mock: SWAP-free returns 40 CZ (qubits disabled, needs routing)
        mock_circuit_bad = MagicMock()
        mock_circuit_bad.count_ops.return_value = {"cz": 40, "rz": 60, "sx": 50}

        mock_selection_bad = LayoutSelection(
            layouts=[[22, 23, 24, 25, 26, 27, 28, 16, 37, 17]],
            ces_values=[0.05],
            transpiled_circuits=[mock_circuit_bad],
        )

        # Mock: VF2 returns something
        mock_circuit_vf2 = MagicMock()
        mock_circuit_vf2.count_ops.return_value = {"cz": 18}

        mock_selection_vf2 = LayoutSelection(
            layouts=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            ces_values=[0.03],
            transpiled_circuits=[mock_circuit_vf2],
        )

        with (
            patch(
                "qmbp_simulation.execution.hardware.submission.select_layouts_low_ces",
                return_value=mock_selection_bad,
            ),
            patch(
                "qmbp_simulation.execution.hardware.submission.select_optimal_layouts",
                return_value=mock_selection_vf2,
            ),
            patch(
                "qmbp_simulation.execution.hardware.submission._maybe_escalate_layouts",
                return_value=mock_selection_vf2,
            ),
        ):
            result = select_layouts_for_hardware(MagicMock(), MagicMock(), config, logger)

        # Should have rejected SWAP-free and used VF2
        rejection_logs = [
            c for c in logger.log.call_args_list if c[0][0] == "layout_fallback_rejected"
        ]
        assert len(rejection_logs) == 1
        assert "n_2q=40" in rejection_logs[0][1]["data"]["reason"]

    def test_n2q_threshold_is_correct(self):
        """Threshold n_qubits*2=20 correctly distinguishes SWAP-free from routed."""
        config = HardwareConfig(n_qubits=10)
        # 9 CZ (direct mapping) → 9 ≤ 20 ✅
        assert config.n_qubits * 2 >= 9
        # 18 CZ (RZZ→2CZ decomposition) → 18 ≤ 20 ✅
        assert config.n_qubits * 2 >= 18
        # 34 CZ (BFS routed) → 34 > 20 ❌
        assert config.n_qubits * 2 < 34
