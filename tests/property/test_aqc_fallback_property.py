"""Property-based tests for AQC fidelity fallback behavior.

# Feature: mitigation-benchmark, Property 14
# **Validates: Requirements 9.2**
#
# Property 14: AQC fidelity fallback behavior
#   - fidelity < 0.998 triggers fallback with aqc_fallback_triggered=True
#   - fidelity >= 0.998 uses the compressed circuit with aqc_fallback_triggered=False
#   - ImportError (qiskit-addon-aqc-tensor not installed) always triggers fallback
#
# We test this by mocking:
#   1. AQCCircuitCompressor to return controlled fidelity values
#   2. The import mechanism to simulate missing qiskit-addon-aqc-tensor
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add scripts to path for import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from experiment_runners.hardware.run_mitigation_benchmark import (
    _build_aqc_circuit,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_AQC_FIDELITY_THRESHOLD = 0.998

# Expected keys in the aqc_metrics dict
AQC_METRICS_KEYS = frozenset(
    {
        "aqc_fidelity",
        "aqc_n_2q_compressed",
        "aqc_2q_reduction_pct",
        "aqc_compression_time_s",
        "aqc_fallback_triggered",
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

# Fidelity values spanning the threshold (realistic range 0.5–1.0)
fidelity_st = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)

# h_values for TFIM (valid regime for heavy_hex N=10)
h_value_st = st.floats(min_value=3.0, max_value=4.5, allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _build_mock_compression_result(fidelity: float) -> MagicMock:
    """Build a mock AQC compression result with controlled fidelity."""
    result = MagicMock()
    result.fidelity = fidelity
    result.n_2q_compressed = 18
    result.n_2q_reduction_pct = 47.0
    # compressed_circuit needs to look like a QuantumCircuit
    mock_circuit = MagicMock()
    mock_circuit.num_qubits = 10
    result.compressed_circuit = mock_circuit
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Property 14: AQC fidelity fallback behavior
# **Validates: Requirements 9.2**
# ═══════════════════════════════════════════════════════════════════════════════


class TestAQCFidelityFallback:
    """Property 14: AQC fidelity fallback behavior.

    **Validates: Requirements 9.2**

    For any AQC config (C16-C18) where compression fidelity < 0.998:
      - aqc_fallback_triggered is True
      - execution proceeds with the p=1 direct circuit (a valid QuantumCircuit)

    For any AQC config where compression fidelity >= 0.998:
      - aqc_fallback_triggered is False
      - the compressed circuit is used
    """

    # -------------------------------------------------------------------
    # Sub-property: fidelity < 0.998 triggers fallback
    # -------------------------------------------------------------------

    @given(fidelity=fidelity_st, h_value=h_value_st)
    @settings(max_examples=50, deadline=None)
    def test_fallback_triggered_below_threshold(self, fidelity: float, h_value: float):
        """When AQC fidelity < 0.998, fallback is triggered."""
        assume(fidelity < _AQC_FIDELITY_THRESHOLD)

        mock_result = _build_mock_compression_result(fidelity)

        with (
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCircuitCompressor",
                create=True,
            ) as MockCompressorCls,
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCompressionConfig",
                create=True,
            ),
        ):
            # Make the import succeed (simulate qiskit-addon-aqc-tensor present)
            # Patch at the import location inside _build_aqc_circuit
            with patch.dict(
                "sys.modules",
                {
                    "qmbp_simulation.circuits.aqc_compression": MagicMock(
                        AQCCircuitCompressor=MockCompressorCls,
                        AQCCompressionConfig=MagicMock(),
                    )
                },
            ):
                instance = MockCompressorCls.return_value
                instance.compress_circuit.return_value = mock_result

                circuit, aqc_metrics = _build_aqc_circuit(h_value)

        assert aqc_metrics is not None
        assert aqc_metrics["aqc_fallback_triggered"] is True
        assert aqc_metrics["aqc_fidelity"] == fidelity
        assert set(aqc_metrics.keys()) == AQC_METRICS_KEYS

    # -------------------------------------------------------------------
    # Sub-property: fidelity >= 0.998 does NOT trigger fallback
    # -------------------------------------------------------------------

    @given(
        fidelity=st.floats(min_value=0.998, max_value=1.0, allow_nan=False, allow_infinity=False),
        h_value=h_value_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_no_fallback_above_threshold(self, fidelity: float, h_value: float):
        """When AQC fidelity >= 0.998, compressed circuit is used (no fallback)."""

        mock_result = _build_mock_compression_result(fidelity)

        with (
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCircuitCompressor",
                create=True,
            ) as MockCompressorCls,
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCompressionConfig",
                create=True,
            ),
        ):
            with patch.dict(
                "sys.modules",
                {
                    "qmbp_simulation.circuits.aqc_compression": MagicMock(
                        AQCCircuitCompressor=MockCompressorCls,
                        AQCCompressionConfig=MagicMock(),
                    )
                },
            ):
                instance = MockCompressorCls.return_value
                instance.compress_circuit.return_value = mock_result

                circuit, aqc_metrics = _build_aqc_circuit(h_value)

        assert aqc_metrics is not None
        assert aqc_metrics["aqc_fallback_triggered"] is False
        assert aqc_metrics["aqc_fidelity"] == fidelity
        assert set(aqc_metrics.keys()) == AQC_METRICS_KEYS
        # The circuit returned should be the compressed one
        assert circuit is mock_result.compressed_circuit

    # -------------------------------------------------------------------
    # Sub-property: ImportError always triggers fallback
    # -------------------------------------------------------------------

    @given(h_value=h_value_st)
    @settings(max_examples=50, deadline=None)
    def test_import_error_triggers_fallback(self, h_value: float):
        """When qiskit-addon-aqc-tensor not installed, fallback always triggers."""
        # Ensure the aqc_compression module import fails
        with patch.dict(
            "sys.modules",
            {"qmbp_simulation.circuits.aqc_compression": None},
        ):
            circuit, aqc_metrics = _build_aqc_circuit(h_value)

        assert aqc_metrics is not None
        assert aqc_metrics["aqc_fallback_triggered"] is True
        assert aqc_metrics["aqc_fidelity"] == 0.0
        assert aqc_metrics["aqc_n_2q_compressed"] == 0
        assert aqc_metrics["aqc_2q_reduction_pct"] == 0.0
        assert aqc_metrics["aqc_compression_time_s"] == 0.0
        assert set(aqc_metrics.keys()) == AQC_METRICS_KEYS

    # -------------------------------------------------------------------
    # Sub-property: fallback always returns a valid QuantumCircuit
    # -------------------------------------------------------------------

    @given(fidelity=fidelity_st, h_value=h_value_st)
    @settings(max_examples=50, deadline=None)
    def test_fallback_returns_valid_circuit(self, fidelity: float, h_value: float):
        """Regardless of fidelity, _build_aqc_circuit always returns a circuit."""
        assume(fidelity < _AQC_FIDELITY_THRESHOLD)

        mock_result = _build_mock_compression_result(fidelity)

        with (
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCircuitCompressor",
                create=True,
            ) as MockCompressorCls,
            patch(
                "experiment_runners.hardware.run_mitigation_benchmark.AQCCompressionConfig",
                create=True,
            ),
        ):
            with patch.dict(
                "sys.modules",
                {
                    "qmbp_simulation.circuits.aqc_compression": MagicMock(
                        AQCCircuitCompressor=MockCompressorCls,
                        AQCCompressionConfig=MagicMock(),
                    )
                },
            ):
                instance = MockCompressorCls.return_value
                instance.compress_circuit.return_value = mock_result

                circuit, aqc_metrics = _build_aqc_circuit(h_value)

        # When fallback triggers, we get a real QuantumCircuit from _build_hva_circuit
        from qiskit.circuit import QuantumCircuit

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 10  # N_QUBITS = 10
