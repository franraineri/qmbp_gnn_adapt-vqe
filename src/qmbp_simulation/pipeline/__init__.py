"""Pipeline submodule — orchestration and dataset I/O.

New code should use ``AcceleratedVQE`` or inherit from ``ValidationRunner``.
"""

from qmbp_simulation.pipeline.accelerated import AcceleratedConfig, AcceleratedResult, AcceleratedVQE
from qmbp_simulation.pipeline.dataset_io import load_phase12_dataset, save_phase12_dataset

__all__ = [
    "AcceleratedVQE",
    "AcceleratedConfig",
    "AcceleratedResult",
    "load_phase12_dataset",
    "save_phase12_dataset",
]
