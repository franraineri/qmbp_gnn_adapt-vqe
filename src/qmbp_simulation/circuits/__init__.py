"""Circuits submodule — HVA circuit construction and AQC compression."""

from qmbp_simulation.circuits.hva import HVACircuitBuilder

__all__ = [
    "HVACircuitBuilder",
    "AQCCircuitCompressor",
    "AQCCompressionConfig",
    "AQCCompressionResult",
    "CompressionValidation",
    "AQCCompressionCache",
]


def __getattr__(name: str):
    """Lazy imports for optional AQC-Tensor module (avoids heavy deps at package load)."""
    _aqc_names = {
        "AQCCircuitCompressor",
        "AQCCompressionConfig",
        "AQCCompressionResult",
        "CompressionValidation",
        "AQCCompressionCache",
    }
    if name in _aqc_names:
        from qmbp_simulation.circuits.aqc_compression import (
            AQCCircuitCompressor,
            AQCCompressionCache,
            AQCCompressionConfig,
            AQCCompressionResult,
            CompressionValidation,
        )

        _map = {
            "AQCCircuitCompressor": AQCCircuitCompressor,
            "AQCCompressionConfig": AQCCompressionConfig,
            "AQCCompressionResult": AQCCompressionResult,
            "CompressionValidation": CompressionValidation,
            "AQCCompressionCache": AQCCompressionCache,
        }
        return _map[name]
    raise AttributeError(f"module 'qmbp_simulation.circuits' has no attribute {name!r}")
