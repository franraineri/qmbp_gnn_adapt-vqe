"""Reusable technique modules for experiments.

These are building blocks imported by experiment scripts:
    - dypp: Dynamic Parameter Prediction
    - sign_equivariant: Z2 symmetry handling
    - parameter_freezing: TITAN-style parameter freezing
    - analytical_init: Perturbation theory initialization
    - physics_loss: Physics-informed MPNN loss
    - hessian_restart: Hessian-guided adaptive restarts
    - active_learning: Ensemble-based active learning
    - graph_utils: Topology-aware graph construction for MPNN
    - scaling_utils: Power-law fitting, transpilation metrics, chi-convergence
"""

from experiments.helpers.active_learning import (
    compute_ensemble_uncertainty,
    max_variance_acquisition,
    select_next_point,
    should_stop,
)
from experiments.helpers.analytical_init import (
    analytical_init_p1,
    analytical_init_p2,
    validate_analytical_init,
)
from experiments.helpers.dypp import (
    dypp_linear,
    dypp_predict,
    dypp_quadratic,
    evaluate_dypp_quality,
)
from experiments.helpers.graph_utils import (
    build_experiment_dataset,
    compare_theta_arrays,
    evaluate_bond_resolved_variant,
    predict_theta,
    predict_theta_batch,
    train_bond_resolved_variant,
)
from experiments.helpers.hessian_restart import (
    hessian_guided_vqe,
    standard_multistart_vqe,
)
from experiments.helpers.parameter_freezing import (
    analyze_parameter_activity,
    frozen_vqe,
)
from experiments.helpers.physics_loss import (
    PhysicsInformedLoss,
    evaluate_energy_batch,
    select_eval_subset,
)
from experiments.helpers.scaling_utils import (
    analyze_chi_convergence,
    compute_transpilation_metrics,
    evaluate_at_multiple_chi,
    fit_power_law,
)
from experiments.helpers.sign_equivariant import (
    SignInvariantLoss,
    canonicalize_dataset,
    canonicalize_sign,
    detect_sign_inconsistency,
)
