"""
Pipeline Runner — DEPRECATED STUB.

The full implementation has been moved to ``_deprecated/pipeline/runner.py``.
This stub re-exports all public symbols for backward compatibility.
New code should use :class:`~qmbp_simulation.pipeline.accelerated.AcceleratedVQE`
or :class:`~qmbp_simulation.framework.runner_base.ValidationRunner`.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

# ── Locate the real implementation in _deprecated/ ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/qmbp_simulation/pipeline → root
_DEPRECATED_MODULE = _PROJECT_ROOT / "_deprecated" / "pipeline" / "runner.py"

if not _DEPRECATED_MODULE.exists():
    raise ImportError(
        f"Deprecated pipeline/runner.py not found at {_DEPRECATED_MODULE}. "
        "If you removed _deprecated/, also remove this stub."
    )

# ── Load module from _deprecated path ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "qmbp_simulation.pipeline._runner_deprecated", _DEPRECATED_MODULE
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# ── Re-export public API (warnings are inside the functions/class) ────────
PipelineRunner = _mod.PipelineRunner
run_accelerated = _mod.run_accelerated
run_exact_diag_sweep = _mod.run_exact_diag_sweep

__all__ = ["PipelineRunner", "run_accelerated", "run_exact_diag_sweep"]
