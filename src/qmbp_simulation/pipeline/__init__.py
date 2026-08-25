"""Pipeline submodule — orchestration, dataset I/O, and QRC fallback.

.. deprecated::
    ``PipelineRunner``, ``run_accelerated``, and ``run_exact_diag_sweep`` are
    legacy. New code should use ``AcceleratedVQE`` or inherit from
    ``ValidationRunner``. These exports are kept for backward compatibility.
"""

from qmbp_simulation.pipeline.accelerated import AcceleratedConfig, AcceleratedResult, AcceleratedVQE
from qmbp_simulation.pipeline.dataset_io import load_phase12_dataset, save_phase12_dataset
from qmbp_simulation.pipeline.runner import PipelineRunner, run_accelerated, run_exact_diag_sweep

__all__ = [
    # Active
    "AcceleratedVQE",
    "AcceleratedConfig",
    "AcceleratedResult",
    "load_phase12_dataset",
    "save_phase12_dataset",
    # Legacy (deprecated — kept for backward compat)
    "PipelineRunner",
    "run_exact_diag_sweep",
    "run_accelerated",
]
