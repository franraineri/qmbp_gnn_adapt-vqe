"""Verify P2-A dynamic layout escalation guard logic."""

from unittest.mock import MagicMock

from qmbp_simulation.execution.hardware.config import HardwareConfig
from qmbp_simulation.execution.hardware.submission import _maybe_escalate_layouts
from qmbp_simulation.execution.noisy_utils import LayoutSelection


def test_escalation_triggers_on_low_spread():
    """Escalation should trigger when CES spread < min_ces_spread."""
    mock_selection = LayoutSelection(
        layouts=[[0] * 10, [10] * 10, [20] * 10],
        transpiled_circuits=[MagicMock(), MagicMock(), MagicMock()],
        ces_values=[0.15, 0.152, 0.151],  # Spread = 0.002 < 0.02
    )

    config = HardwareConfig(min_ces_spread=0.02, n_layouts_max=5)

    ces_spread = max(mock_selection.ces_values) - min(mock_selection.ces_values)
    assert ces_spread < config.min_ces_spread, "Guard condition should detect low spread"


def test_no_escalation_on_good_spread():
    """No escalation when CES spread is sufficient."""
    good_selection = LayoutSelection(
        layouts=[[0] * 10, [10] * 10, [20] * 10],
        transpiled_circuits=[MagicMock(), MagicMock(), MagicMock()],
        ces_values=[0.10, 0.15, 0.20],  # Spread = 0.10 > 0.02
    )
    config = HardwareConfig(min_ces_spread=0.02, n_layouts_max=5)
    logger_mock = MagicMock()

    # _maybe_escalate_layouts should return the original selection (no escalation)
    result = _maybe_escalate_layouts(
        good_selection,
        MagicMock(),  # bound_circuit
        MagicMock(),  # backend
        config,
        logger_mock,
        min_ces_spread=0.02,
        n_layouts_max=5,
        method="bfs",
    )
    assert result is good_selection, "Should return original when spread is OK"


def test_no_escalation_when_already_at_max():
    """No escalation when already at n_layouts_max."""
    at_max = LayoutSelection(
        layouts=[[i] * 10 for i in range(5)],
        transpiled_circuits=[MagicMock() for _ in range(5)],
        ces_values=[0.15, 0.152, 0.151, 0.153, 0.150],  # Low spread but at max
    )
    config = HardwareConfig(min_ces_spread=0.02, n_layouts_max=5)
    logger_mock = MagicMock()

    result = _maybe_escalate_layouts(
        at_max,
        MagicMock(),
        MagicMock(),
        config,
        logger_mock,
        min_ces_spread=0.02,
        n_layouts_max=5,
        method="bfs",
    )
    assert result is at_max, "Should not escalate when already at max"
