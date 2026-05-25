"""Individual experiment implementations.

Each module defines a single experiment class that inherits from BaseExperiment.
Experiments are registered in EXPERIMENT_REGISTRY for CLI discovery.

NOTE: Only experiments with implemented scripts are registered here.
See documentation/STATUS-V8.md for the full planned list.
"""

# Registry: experiment_id -> (module_path, class_name)
# Only includes experiments with actual implementations.
EXPERIMENT_REGISTRY: dict[str, tuple[str, str]] = {
    "A3": ("scripts.experiments_v8.experiments.exp_a3_scaling_law", "ExperimentA3"),
    "B1": ("scripts.experiments_v8.experiments.exp_b1_analytical", "ExperimentB1"),
    "B2": ("scripts.experiments_v8.experiments.exp_b2_freezing", "ExperimentB2"),
    "B4": ("scripts.experiments_v8.experiments.exp_b4_hessian", "ExperimentB4"),
    "C1": ("scripts.experiments_v8.experiments.exp_c1_physics_loss", "ExperimentC1"),
    "C3": ("scripts.experiments_v8.experiments.exp_c3_sign", "ExperimentC3"),
    "D1": ("scripts.experiments_v8.experiments.exp_d1_weight_space", "ExperimentD1"),
    "E4": ("scripts.experiments_v8.experiments.exp_e4_longitudinal", "ExperimentE4"),
    "F1": ("scripts.experiments_v8.experiments.exp_f1_dypp", "ExperimentF1"),
    "F3": ("scripts.experiments_v8.experiments.exp_f3_fluctuation", "ExperimentF3"),
}

# Experiments planned but not yet implemented (for reference only)
_PLANNED_NOT_IMPLEMENTED = {
    "A1": "Orthogonal projection DMRG (infrastructure, not in final plan)",
    "A2": "TCI landscape mapping (needs xfac library)",
    "B3": "Light Cone Cancellation (high effort, separate sprint)",
    "D3": "Tensor completion landscape (needs tensorly)",
    "E1": "N=30 full pipeline (depends on C3 + B1 validation)",
    "E3": "Active learning h-grid (technique exists, script pending)",
}


def get_experiment_class(experiment_id: str):
    """Dynamically import and return the experiment class."""
    import importlib

    exp_id = experiment_id.upper()
    if exp_id not in EXPERIMENT_REGISTRY:
        # Check if it's a planned but unimplemented experiment
        if exp_id in _PLANNED_NOT_IMPLEMENTED:
            raise ValueError(
                f"Experiment {exp_id} is planned but not yet implemented: "
                f"{_PLANNED_NOT_IMPLEMENTED[exp_id]}. "
                f"Available: {sorted(EXPERIMENT_REGISTRY.keys())}"
            )
        raise ValueError(
            f"Unknown experiment: {exp_id}. Available: {sorted(EXPERIMENT_REGISTRY.keys())}"
        )

    module_path, class_name = EXPERIMENT_REGISTRY[exp_id]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
