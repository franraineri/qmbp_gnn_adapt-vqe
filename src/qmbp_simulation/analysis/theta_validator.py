"""Theta Validation Module — Post-prediction quality assurance for MPNN outputs.

Provides a modular, layered validation pipeline for θ_pred vectors produced
by the MPNN predictor. Each validation level is independently configurable
and produces structured diagnostics that integrate with DiagnosticCollector.

Validation Levels (ordered by computational cost):
  L1: Bound check — θ ∈ [μ-kσ, μ+kσ] of training distribution
  L2: NaN/Inf guard — numerical sanity
  L3: Interpolation consistency — distance to nearest training neighbors
  L4: State fidelity — |⟨ψ(θ_pred)|ψ_exact⟩|²
  L5: Variational gradient norm — ||∇E(θ_pred)|| proximity to minimum
  L6: MC Dropout uncertainty — σ(θ_pred) over K stochastic passes
  L7: Parameter sensitivity — ∂E/∂θᵢ identifies fragile angles

Usage:
    from qmbp_simulation.analysis import ThetaValidator, ThetaValidationReport

    validator = ThetaValidator.from_training_data(theta_opt_array)
    report = validator.validate(theta_pred, level=4, circuit=qc, hamiltonian=H,
                                exact_state=psi_exact)
    report.passes()  # True if all enabled checks pass
    report.to_dict()  # JSON-serializable diagnostics

References:
    - Gal & Ghahramani (2016): MC Dropout as Bayesian approximation.
    - Fontana et al. (2024, arXiv:2402.18953): VQE landscape analysis.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_BOUND_SIGMAS = 3.0  # k for μ±kσ bound check
DEFAULT_INTERPOLATION_THRESHOLD = 2.0  # max allowed distance / training spacing
DEFAULT_GRADIENT_THRESHOLD = 0.1  # ||∇E|| below this → near minimum
DEFAULT_MC_DROPOUT_PASSES = 20  # K stochastic forward passes
DEFAULT_SENSITIVITY_EPSILON = 0.01  # finite-difference step for ∂E/∂θᵢ
FIDELITY_WARNING_THRESHOLD = 0.90  # below this, θ_pred is suspicious
FIDELITY_FAILURE_THRESHOLD = 0.70  # below this, θ_pred is unreliable


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class BoundCheckResult:
    """Result of L1 bound checking."""

    passed: bool
    n_out_of_bounds: int
    out_of_bounds_indices: list[int]
    theta_min_training: float
    theta_max_training: float
    bounds_used: tuple[float, float]  # (lower, upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_out_of_bounds": self.n_out_of_bounds,
            "out_of_bounds_indices": self.out_of_bounds_indices,
            "bounds_used": list(self.bounds_used),
        }


@dataclass
class NumericalSanityResult:
    """Result of L2 NaN/Inf guard."""

    passed: bool
    has_nan: bool
    has_inf: bool
    nan_indices: list[int]
    inf_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "nan_indices": self.nan_indices,
            "inf_indices": self.inf_indices,
        }


@dataclass
class InterpolationResult:
    """Result of L3 interpolation consistency check."""

    passed: bool
    distance_to_nearest: float
    mean_training_spacing: float
    ratio: float  # distance / spacing
    nearest_h_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "distance_to_nearest": self.distance_to_nearest,
            "mean_training_spacing": self.mean_training_spacing,
            "ratio": self.ratio,
            "nearest_h_value": self.nearest_h_value,
        }


@dataclass
class FidelityResult:
    """Result of L4 state fidelity computation."""

    passed: bool
    fidelity: float
    warning: bool  # True if below warning threshold but above failure

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "fidelity": self.fidelity,
            "warning": self.warning,
        }


@dataclass
class GradientNormResult:
    """Result of L5 variational gradient norm check."""

    passed: bool
    gradient_norm: float
    per_param_gradients: list[float]
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gradient_norm": self.gradient_norm,
            "per_param_gradients": self.per_param_gradients,
            "threshold": self.threshold,
        }


@dataclass
class MCDropoutResult:
    """Result of L6 MC Dropout uncertainty estimation."""

    passed: bool
    mean_std: float  # mean of per-parameter std
    per_param_std: list[float]
    coefficient_of_variation: float  # mean_std / mean(|θ_pred|)
    n_passes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mean_std": self.mean_std,
            "per_param_std": self.per_param_std,
            "coefficient_of_variation": self.coefficient_of_variation,
            "n_passes": self.n_passes,
        }


@dataclass
class SensitivityResult:
    """Result of L7 parameter sensitivity analysis."""

    passed: bool
    max_sensitivity: float
    per_param_sensitivity: list[float]
    fragile_indices: list[int]  # indices where |∂E/∂θᵢ| > threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_sensitivity": self.max_sensitivity,
            "per_param_sensitivity": self.per_param_sensitivity,
            "fragile_indices": self.fragile_indices,
        }


@dataclass
class ThetaValidationReport:
    """Aggregate report from all validation levels.

    Contains results from each level that was executed, plus an overall
    pass/fail verdict and confidence score.
    """

    level_executed: int  # highest level run (1-7)
    bound_check: BoundCheckResult | None = None
    numerical_sanity: NumericalSanityResult | None = None
    interpolation: InterpolationResult | None = None
    fidelity: FidelityResult | None = None
    gradient_norm: GradientNormResult | None = None
    mc_dropout: MCDropoutResult | None = None
    sensitivity: SensitivityResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def passes(self) -> bool:
        """Overall pass: all executed levels passed."""
        checks = [
            self.bound_check,
            self.numerical_sanity,
            self.interpolation,
            self.fidelity,
            self.gradient_norm,
            self.mc_dropout,
            self.sensitivity,
        ]
        return all(c.passed for c in checks if c is not None)

    @property
    def confidence_score(self) -> float:
        """Composite confidence in [0, 1] based on executed checks.

        Weights: L1=0.05, L2=0.05, L3=0.15, L4=0.35, L5=0.15, L6=0.15, L7=0.10
        """
        weights = [0.05, 0.05, 0.15, 0.35, 0.15, 0.15, 0.10]
        checks = [
            self.bound_check,
            self.numerical_sanity,
            self.interpolation,
            self.fidelity,
            self.gradient_norm,
            self.mc_dropout,
            self.sensitivity,
        ]
        total_weight = 0.0
        score = 0.0
        for check, w in zip(checks, weights, strict=False):
            if check is not None:
                total_weight += w
                if check.passed:
                    score += w
                # Partial credit for fidelity
                elif isinstance(check, FidelityResult) and check.fidelity > 0.5:
                    score += w * check.fidelity

        return score / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary for DiagnosticCollector integration."""
        result: dict[str, Any] = {
            "level_executed": self.level_executed,
            "overall_pass": self.passes(),
            "confidence_score": round(self.confidence_score, 4),
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self.bound_check is not None:
            result["L1_bound_check"] = self.bound_check.to_dict()
        if self.numerical_sanity is not None:
            result["L2_numerical_sanity"] = self.numerical_sanity.to_dict()
        if self.interpolation is not None:
            result["L3_interpolation"] = self.interpolation.to_dict()
        if self.fidelity is not None:
            result["L4_fidelity"] = self.fidelity.to_dict()
        if self.gradient_norm is not None:
            result["L5_gradient_norm"] = self.gradient_norm.to_dict()
        if self.mc_dropout is not None:
            result["L6_mc_dropout"] = self.mc_dropout.to_dict()
        if self.sensitivity is not None:
            result["L7_sensitivity"] = self.sensitivity.to_dict()
        return result


# ── Main Validator Class ─────────────────────────────────────────────────────


class ThetaValidator:
    """Modular validator for MPNN-predicted variational angles.

    Constructed from training data statistics, then validates arbitrary
    θ_pred vectors against learned bounds, distribution, and (optionally)
    quantum circuit evaluation.

    Parameters
    ----------
    theta_mean : np.ndarray [n_params]
        Per-parameter mean from training θ_opt.
    theta_std : np.ndarray [n_params]
        Per-parameter std from training θ_opt.
    theta_min : np.ndarray [n_params]
        Per-parameter minimum from training θ_opt.
    theta_max : np.ndarray [n_params]
        Per-parameter maximum from training θ_opt.
    training_thetas : np.ndarray [n_points, n_params] | None
        Full training θ_opt array for interpolation checks.
    training_h_values : np.ndarray [n_points] | None
        Corresponding h-values for interpolation checks.
    bound_sigmas : float
        Number of standard deviations for bound check (default 3.0).
    """

    def __init__(
        self,
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
        theta_min: np.ndarray,
        theta_max: np.ndarray,
        training_thetas: np.ndarray | None = None,
        training_h_values: np.ndarray | None = None,
        bound_sigmas: float = DEFAULT_BOUND_SIGMAS,
    ) -> None:
        self.theta_mean = np.asarray(theta_mean, dtype=np.float64)
        self.theta_std = np.asarray(theta_std, dtype=np.float64)
        self.theta_min = np.asarray(theta_min, dtype=np.float64)
        self.theta_max = np.asarray(theta_max, dtype=np.float64)
        self.training_thetas = training_thetas
        self.training_h_values = training_h_values
        self.bound_sigmas = bound_sigmas
        self.n_params = len(theta_mean)

    @classmethod
    def from_training_data(
        cls,
        theta_opt: np.ndarray,
        h_values: np.ndarray | None = None,
        bound_sigmas: float = DEFAULT_BOUND_SIGMAS,
    ) -> ThetaValidator:
        """Construct validator from the Phase 2 training θ_opt array.

        Parameters
        ----------
        theta_opt : np.ndarray [n_points, n_params]
            Optimized VQE parameters (rows = h-points).
        h_values : np.ndarray [n_points] | None
            Corresponding h-values (for interpolation checks).
        bound_sigmas : float
            Number of sigmas for bound check.

        Returns
        -------
        ThetaValidator
            Configured validator instance.
        """
        theta_opt = np.asarray(theta_opt, dtype=np.float64)
        if theta_opt.ndim == 1:
            theta_opt = theta_opt.reshape(1, -1)

        return cls(
            theta_mean=np.mean(theta_opt, axis=0),
            theta_std=np.std(theta_opt, axis=0),
            theta_min=np.min(theta_opt, axis=0),
            theta_max=np.max(theta_opt, axis=0),
            training_thetas=theta_opt,
            training_h_values=np.asarray(h_values) if h_values is not None else None,
            bound_sigmas=bound_sigmas,
        )

    # ── L1: Bound Check ──────────────────────────────────────────────────

    def check_bounds(self, theta_pred: np.ndarray) -> BoundCheckResult:
        """L1: Verify θ_pred is within μ ± k·σ of training distribution.

        Also uses hard bounds: [min(training) - margin, max(training) + margin]
        where margin = k·σ. A parameter outside BOTH statistical and hard bounds
        is flagged.

        For degenerate distributions (std≈0), uses a minimum absolute margin
        of 0.1 to avoid false positives from numerical noise.
        """
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()

        # Statistical bounds: μ ± k·σ (with floor on σ to avoid zero-width)
        # Floor at 0.1/k ensures minimum ±0.1 absolute margin even for constant data
        min_sigma = 0.1 / self.bound_sigmas
        sigma_safe = np.maximum(self.theta_std, min_sigma)
        lower_stat = self.theta_mean - self.bound_sigmas * sigma_safe
        upper_stat = self.theta_mean + self.bound_sigmas * sigma_safe

        # Hard bounds: training range ± margin
        margin = self.bound_sigmas * sigma_safe
        lower_hard = self.theta_min - margin
        upper_hard = self.theta_max + margin

        # Combined: use the more permissive of the two
        lower = np.minimum(lower_stat, lower_hard)
        upper = np.maximum(upper_stat, upper_hard)

        out_of_bounds = (theta_pred < lower) | (theta_pred > upper)
        oob_indices = list(np.where(out_of_bounds)[0])

        passed = len(oob_indices) == 0
        if not passed:
            logger.warning(
                f"L1 BOUND CHECK FAIL: {len(oob_indices)}/{self.n_params} params "
                f"out of bounds. Indices: {oob_indices[:5]}{'...' if len(oob_indices) > 5 else ''}"
            )

        return BoundCheckResult(
            passed=passed,
            n_out_of_bounds=len(oob_indices),
            out_of_bounds_indices=oob_indices,
            theta_min_training=float(np.min(self.theta_min)),
            theta_max_training=float(np.max(self.theta_max)),
            bounds_used=(float(np.min(lower)), float(np.max(upper))),
        )

    # ── L2: Numerical Sanity ─────────────────────────────────────────────

    def check_numerical_sanity(self, theta_pred: np.ndarray) -> NumericalSanityResult:
        """L2: Check for NaN and Inf values in θ_pred."""
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()

        nan_mask = np.isnan(theta_pred)
        inf_mask = np.isinf(theta_pred)

        nan_indices = list(np.where(nan_mask)[0])
        inf_indices = list(np.where(inf_mask)[0])

        has_nan = len(nan_indices) > 0
        has_inf = len(inf_indices) > 0
        passed = not has_nan and not has_inf

        if not passed:
            logger.error(
                f"L2 NUMERICAL SANITY FAIL: NaN={has_nan} ({len(nan_indices)}), "
                f"Inf={has_inf} ({len(inf_indices)})"
            )

        return NumericalSanityResult(
            passed=passed,
            has_nan=has_nan,
            has_inf=has_inf,
            nan_indices=nan_indices,
            inf_indices=inf_indices,
        )

    # ── L3: Interpolation Consistency ────────────────────────────────────

    def check_interpolation(
        self,
        theta_pred: np.ndarray,
        h_test: float | None = None,
        threshold: float = DEFAULT_INTERPOLATION_THRESHOLD,
    ) -> InterpolationResult:
        """L3: Check θ_pred consistency with nearest training neighbors.

        Computes the L2 distance from θ_pred to the nearest training θ_opt,
        normalized by the mean spacing between consecutive training points.
        A ratio > threshold indicates the prediction is an outlier.
        """
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()

        if self.training_thetas is None or len(self.training_thetas) < 2:
            # Cannot perform interpolation check without training data
            return InterpolationResult(
                passed=True,
                distance_to_nearest=0.0,
                mean_training_spacing=0.0,
                ratio=0.0,
                nearest_h_value=None,
            )

        # Compute distances to all training points
        distances = np.linalg.norm(self.training_thetas - theta_pred, axis=1)
        nearest_idx = int(np.argmin(distances))
        distance_to_nearest = float(distances[nearest_idx])

        # Mean spacing between consecutive training points
        consecutive_dists = np.linalg.norm(np.diff(self.training_thetas, axis=0), axis=1)
        mean_spacing = float(np.mean(consecutive_dists))

        # Avoid division by zero
        mean_spacing_safe = max(mean_spacing, 1e-10)
        ratio = distance_to_nearest / mean_spacing_safe

        nearest_h = None
        if self.training_h_values is not None and nearest_idx < len(self.training_h_values):
            nearest_h = float(self.training_h_values[nearest_idx])

        passed = ratio <= threshold
        if not passed:
            logger.warning(
                f"L3 INTERPOLATION FAIL: distance/spacing ratio={ratio:.2f} "
                f"(threshold={threshold}). θ_pred is far from training manifold."
            )

        return InterpolationResult(
            passed=passed,
            distance_to_nearest=distance_to_nearest,
            mean_training_spacing=mean_spacing,
            ratio=ratio,
            nearest_h_value=nearest_h,
        )

    # ── L4: State Fidelity ───────────────────────────────────────────────

    def check_fidelity(
        self,
        theta_pred: np.ndarray,
        circuit: Any,
        exact_state: np.ndarray,
        warning_threshold: float = FIDELITY_WARNING_THRESHOLD,
        failure_threshold: float = FIDELITY_FAILURE_THRESHOLD,
    ) -> FidelityResult:
        """L4: Compute |⟨ψ(θ_pred)|ψ_exact⟩|².

        Parameters
        ----------
        theta_pred : np.ndarray
            Predicted variational parameters.
        circuit : QuantumCircuit
            Parametrized HVA circuit.
        exact_state : np.ndarray
            Exact ground state vector.
        warning_threshold : float
            Fidelity below this triggers a warning (default 0.90).
        failure_threshold : float
            Fidelity below this is a hard failure (default 0.70).

        Returns
        -------
        FidelityResult
        """
        from qiskit.quantum_info import Statevector, state_fidelity

        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()
        bound = circuit.assign_parameters(theta_pred)
        sv_pred = Statevector(bound)
        sv_exact = Statevector(exact_state)

        fid = float(state_fidelity(sv_pred, sv_exact))
        warning = fid < warning_threshold and fid >= failure_threshold
        passed = fid >= failure_threshold

        if not passed:
            logger.error(
                f"L4 FIDELITY FAIL: F={fid:.4f} < {failure_threshold} "
                f"— θ_pred produces a state far from the ground state."
            )
        elif warning:
            logger.warning(
                f"L4 FIDELITY WARNING: F={fid:.4f} < {warning_threshold} "
                f"— θ_pred state overlap is below ideal."
            )

        return FidelityResult(
            passed=passed,
            fidelity=fid,
            warning=warning,
        )

    # ── L5: Variational Gradient Norm ────────────────────────────────────

    def check_gradient_norm(
        self,
        theta_pred: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        threshold: float = DEFAULT_GRADIENT_THRESHOLD,
        epsilon: float = DEFAULT_SENSITIVITY_EPSILON,
    ) -> GradientNormResult:
        """L5: Verify ||∇E(θ_pred)|| is small (near a local minimum).

        Uses central finite differences: ∂E/∂θᵢ ≈ (E(θ+εeᵢ) - E(θ-εeᵢ)) / 2ε

        Parameters
        ----------
        theta_pred : np.ndarray
            Predicted parameters.
        energy_fn : callable
            E(θ) → float energy evaluation function.
        threshold : float
            Maximum acceptable gradient norm (default 0.1).
        epsilon : float
            Finite-difference step size (default 0.01).

        Returns
        -------
        GradientNormResult
        """
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()
        n = len(theta_pred)
        gradients = np.zeros(n)

        for i in range(n):
            theta_plus = theta_pred.copy()
            theta_minus = theta_pred.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon
            gradients[i] = (energy_fn(theta_plus) - energy_fn(theta_minus)) / (2 * epsilon)

        grad_norm = float(np.linalg.norm(gradients))
        passed = grad_norm <= threshold

        if not passed:
            logger.warning(
                f"L5 GRADIENT NORM FAIL: ||∇E||={grad_norm:.4f} > {threshold} "
                f"— θ_pred is NOT near a local minimum."
            )

        return GradientNormResult(
            passed=passed,
            gradient_norm=grad_norm,
            per_param_gradients=[float(g) for g in gradients],
            threshold=threshold,
        )

    # ── L6: MC Dropout Uncertainty ───────────────────────────────────────

    def check_mc_dropout(
        self,
        model: Any,
        graph_data: Any,
        n_passes: int = DEFAULT_MC_DROPOUT_PASSES,
        cv_threshold: float = 0.10,
    ) -> MCDropoutResult:
        """L6: Estimate prediction uncertainty via MC Dropout.

        Activates dropout at inference time and runs K forward passes to
        estimate the posterior predictive variance. High variance indicates
        model uncertainty about the prediction.

        Parameters
        ----------
        model : MPNNPredictor
            Trained MPNN model (must have Dropout layers).
        graph_data : Data
            Input graph for the test h-value.
        n_passes : int
            Number of stochastic forward passes (default 20).
        cv_threshold : float
            Maximum coefficient of variation (std/|mean|) for passing.

        Returns
        -------
        MCDropoutResult
        """
        import torch

        # Enable dropout for uncertainty estimation
        model.train()  # Activates dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_passes):
                pred = model(graph_data).numpy().flatten()
                predictions.append(pred)

        # Restore eval mode
        model.eval()

        predictions_arr = np.array(predictions)  # [K, n_params]
        per_param_std = np.std(predictions_arr, axis=0)
        mean_pred = np.mean(predictions_arr, axis=0)

        mean_std = float(np.mean(per_param_std))

        # Coefficient of variation: normalized uncertainty
        mean_abs = float(np.mean(np.abs(mean_pred)))
        cv = mean_std / max(mean_abs, 1e-10)

        passed = cv <= cv_threshold

        if not passed:
            logger.warning(
                f"L6 MC DROPOUT FAIL: CV={cv:.4f} > {cv_threshold} "
                f"— high model uncertainty. Mean σ={mean_std:.4f}"
            )

        return MCDropoutResult(
            passed=passed,
            mean_std=mean_std,
            per_param_std=[float(s) for s in per_param_std],
            coefficient_of_variation=cv,
            n_passes=n_passes,
        )

    # ── L7: Parameter Sensitivity ────────────────────────────────────────

    def check_sensitivity(
        self,
        theta_pred: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        epsilon: float = DEFAULT_SENSITIVITY_EPSILON,
        sensitivity_threshold: float = 1.0,
    ) -> SensitivityResult:
        """L7: Identify fragile parameters where small changes cause large E shifts.

        Computes |∂E/∂θᵢ| for each parameter and flags those above threshold.
        Unlike L5 (which checks the norm), this identifies WHICH specific
        parameters are sensitive — useful for hardware error budgeting.

        Parameters
        ----------
        theta_pred : np.ndarray
            Predicted parameters.
        energy_fn : callable
            E(θ) → float.
        epsilon : float
            Finite-difference step (default 0.01).
        sensitivity_threshold : float
            Maximum |∂E/∂θᵢ| before flagging as fragile.

        Returns
        -------
        SensitivityResult
        """
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()
        n = len(theta_pred)
        sensitivities = np.zeros(n)

        for i in range(n):
            theta_plus = theta_pred.copy()
            theta_minus = theta_pred.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon
            sensitivities[i] = abs((energy_fn(theta_plus) - energy_fn(theta_minus)) / (2 * epsilon))

        max_sens = float(np.max(sensitivities))
        fragile = list(np.where(sensitivities > sensitivity_threshold)[0])
        passed = len(fragile) == 0

        if not passed:
            logger.warning(
                f"L7 SENSITIVITY FAIL: {len(fragile)} fragile params "
                f"(max |∂E/∂θ|={max_sens:.4f}, threshold={sensitivity_threshold}). "
                f"Indices: {fragile[:5]}{'...' if len(fragile) > 5 else ''}"
            )

        return SensitivityResult(
            passed=passed,
            max_sensitivity=max_sens,
            per_param_sensitivity=[float(s) for s in sensitivities],
            fragile_indices=fragile,
        )

    # ── Orchestrator ─────────────────────────────────────────────────────

    def validate(
        self,
        theta_pred: np.ndarray,
        level: int = 4,
        *,
        h_test: float | None = None,
        circuit: Any = None,
        exact_state: np.ndarray | None = None,
        energy_fn: Callable[[np.ndarray], float] | None = None,
        model: Any = None,
        graph_data: Any = None,
        interpolation_threshold: float = DEFAULT_INTERPOLATION_THRESHOLD,
        gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
        mc_passes: int = DEFAULT_MC_DROPOUT_PASSES,
        mc_cv_threshold: float = 0.10,
        sensitivity_threshold: float = 1.0,
        fidelity_warning: float = FIDELITY_WARNING_THRESHOLD,
        fidelity_failure: float = FIDELITY_FAILURE_THRESHOLD,
    ) -> ThetaValidationReport:
        """Run validation up to the specified level.

        Parameters
        ----------
        theta_pred : np.ndarray
            Predicted variational parameters from MPNN.
        level : int
            Maximum validation level to execute (1-7). Higher levels
            require more inputs and are more expensive.
        h_test : float | None
            Test h-value (used for interpolation context).
        circuit : QuantumCircuit | None
            Required for L4+.
        exact_state : np.ndarray | None
            Required for L4.
        energy_fn : callable | None
            E(θ) → float. Required for L5 and L7.
        model : MPNNPredictor | None
            Required for L6.
        graph_data : Data | None
            Required for L6.
        interpolation_threshold : float
            L3 threshold (default 2.0).
        gradient_threshold : float
            L5 threshold (default 0.1).
        mc_passes : int
            L6 number of MC passes (default 20).
        mc_cv_threshold : float
            L6 coefficient of variation threshold.
        sensitivity_threshold : float
            L7 per-param sensitivity threshold.
        fidelity_warning : float
            L4 warning threshold.
        fidelity_failure : float
            L4 failure threshold.

        Returns
        -------
        ThetaValidationReport
            Aggregate report with all executed levels.
        """
        theta_pred = np.asarray(theta_pred, dtype=np.float64).flatten()
        report = ThetaValidationReport(level_executed=level)

        # L1: Bound check (always)
        if level >= 1:
            report.bound_check = self.check_bounds(theta_pred)
            if not report.bound_check.passed:
                report.warnings.append(
                    f"L1: {report.bound_check.n_out_of_bounds} params out of bounds"
                )

        # L2: NaN/Inf (always)
        if level >= 2:
            report.numerical_sanity = self.check_numerical_sanity(theta_pred)
            if not report.numerical_sanity.passed:
                report.errors.append("L2: NaN/Inf detected in θ_pred — CRITICAL")
                # Abort further checks if numerically invalid
                return report

        # L3: Interpolation consistency
        if level >= 3:
            report.interpolation = self.check_interpolation(
                theta_pred, h_test=h_test, threshold=interpolation_threshold
            )
            if not report.interpolation.passed:
                report.warnings.append(
                    f"L3: distance/spacing ratio={report.interpolation.ratio:.2f}"
                )

        # L4: State fidelity (requires circuit + exact_state)
        if level >= 4:
            if circuit is not None and exact_state is not None:
                try:
                    report.fidelity = self.check_fidelity(
                        theta_pred,
                        circuit,
                        exact_state,
                        warning_threshold=fidelity_warning,
                        failure_threshold=fidelity_failure,
                    )
                    if report.fidelity.warning:
                        report.warnings.append(
                            f"L4: fidelity={report.fidelity.fidelity:.4f} < {fidelity_warning}"
                        )
                    if not report.fidelity.passed:
                        report.errors.append(
                            f"L4: fidelity={report.fidelity.fidelity:.4f} < {fidelity_failure}"
                        )
                except Exception as e:
                    logger.error(f"L4 fidelity check failed: {e}")
                    report.errors.append(f"L4: exception — {e}")
            else:
                logger.debug("L4 skipped: circuit or exact_state not provided")

        # L5: Gradient norm (requires energy_fn)
        if level >= 5:
            if energy_fn is not None:
                report.gradient_norm = self.check_gradient_norm(
                    theta_pred, energy_fn, threshold=gradient_threshold
                )
                if not report.gradient_norm.passed:
                    report.warnings.append(f"L5: ||∇E||={report.gradient_norm.gradient_norm:.4f}")
            else:
                logger.debug("L5 skipped: energy_fn not provided")

        # L6: MC Dropout (requires model + graph_data)
        if level >= 6:
            if model is not None and graph_data is not None:
                report.mc_dropout = self.check_mc_dropout(
                    model, graph_data, n_passes=mc_passes, cv_threshold=mc_cv_threshold
                )
                if not report.mc_dropout.passed:
                    report.warnings.append(
                        f"L6: CV={report.mc_dropout.coefficient_of_variation:.4f}"
                    )
            else:
                logger.debug("L6 skipped: model or graph_data not provided")

        # L7: Sensitivity (requires energy_fn)
        if level >= 7:
            if energy_fn is not None:
                report.sensitivity = self.check_sensitivity(
                    theta_pred,
                    energy_fn,
                    sensitivity_threshold=sensitivity_threshold,
                )
                if not report.sensitivity.passed:
                    report.warnings.append(
                        f"L7: {len(report.sensitivity.fragile_indices)} fragile params"
                    )
            else:
                logger.debug("L7 skipped: energy_fn not provided")

        # Log summary
        status = "PASS" if report.passes() else "FAIL"
        logger.info(
            f"θ_pred validation L1-L{level}: {status} (confidence={report.confidence_score:.2f})"
        )

        return report
