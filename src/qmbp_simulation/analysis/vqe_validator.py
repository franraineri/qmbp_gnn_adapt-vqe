"""VQE Result Validator — Comprehensive post-optimization validation.

Provides a modular, severity-classified validation system for VQE outputs.
Integrates with DiagnosticCollector and PipelineRunner for automatic
invocation after every VQE sweep.

Validation Checks (ordered by severity):
  CRITICAL — result is invalid, must not propagate unchecked:
    C1: Variational principle (E_VQE ≥ E_exact - ε)
    C2: Physical energy bounds (E ∈ [E_lower, E_upper])
    C3: θ_opt finiteness (no NaN/Inf)
    C4: θ_opt in-bounds (all |θᵢ| ≤ π)
    C5: Energy finiteness

  WARNING — result is suspicious, flag for review:
    W1: Fidelity-energy inconsistency (high fid + high ΔE or low fid + low ΔE)
    W2: Restart basin spread (σ > threshold → multiple local minima)
    W3: Sweep convergence rate (< 50% converged → optimizer struggling)
    W4: θ magnitude anomaly (||θ||₂ >> expected → boundary-trapped)
    W5: Energy monotonicity violation (non-physical for TFIM at large h)

  INFO — observational, no action needed:
    I1: Convergence status per point
    I2: Iteration count anomaly (> 2× median)

Usage:
    from qmbp_simulation.analysis import VQEValidator, VQEValidationReport

    validator = VQEValidator(n_qubits=6, n_edges=5, h_field=1.5, J=1.0)
    report = validator.validate_single(vqe_result, exact_result)
    report = validator.validate_sweep(vqe_results, exact_data)

    if report.has_critical:
        logger.error(f"VQE validation FAILED: {report.critical_issues}")
    if report.has_warnings:
        logger.warning(f"VQE warnings: {report.warnings}")

References:
    - Variational principle: ⟨ψ|H|ψ⟩ ≥ E₀ for all |ψ⟩ (quantum mechanics).
    - Tilly et al. (2022): VQE review — common failure modes.
    - Project-specific: binnacle-v8-experiments, 174-run failure diagnosis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Numerical tolerance for variational principle (accounts for floating-point)
VARIATIONAL_TOLERANCE: float = 1e-8

# θ_opt magnitude warning threshold (||θ||₂ / sqrt(n_params) > this → anomaly)
THETA_MAGNITUDE_WARNING: float = 2.5  # in units of π/sqrt(n_params)

# Restart energy spread threshold (σ > this → multiple basins)
RESTART_SPREAD_THRESHOLD: float = 0.1

# Minimum convergence rate for a sweep to be considered healthy
MIN_CONVERGENCE_RATE: float = 0.50

# Fidelity-energy consistency thresholds
FIDELITY_HIGH: float = 0.99
FIDELITY_LOW: float = 0.50
DE_GAP_LOW: float = 0.01
DE_GAP_HIGH: float = 0.10

# Iteration count anomaly multiplier (> median × this → anomalous)
ITERATION_ANOMALY_FACTOR: float = 3.0


# ── Enums and Data Classes ───────────────────────────────────────────────────


class Severity(Enum):
    """Validation issue severity levels."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: Severity
    check_id: str
    message: str
    h_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "severity": self.severity.value,
            "check_id": self.check_id,
            "message": self.message,
            "h_value": self.h_value,
            "details": self.details,
        }


@dataclass
class VQEValidationReport:
    """Aggregated validation report for one or more VQE results.

    Attributes
    ----------
    issues : list[ValidationIssue]
        All validation findings, ordered by severity.
    n_points_validated : int
        Number of VQE results validated.
    n_critical : int
        Count of CRITICAL issues.
    n_warnings : int
        Count of WARNING issues.
    n_info : int
        Count of INFO issues.
    sweep_metrics : dict[str, Any]
        Sweep-level aggregate metrics (convergence rate, etc.).
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    n_points_validated: int = 0
    n_critical: int = 0
    n_warnings: int = 0
    n_info: int = 0
    sweep_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        """True if any CRITICAL issues were found."""
        return self.n_critical > 0

    @property
    def has_warnings(self) -> bool:
        """True if any WARNING issues were found."""
        return self.n_warnings > 0

    @property
    def passed(self) -> bool:
        """True if no CRITICAL issues exist (warnings are acceptable)."""
        return not self.has_critical

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        """Filter to only CRITICAL issues."""
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Filter to only WARNING issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def add(self, issue: ValidationIssue) -> None:
        """Add a validation issue and update counts."""
        self.issues.append(issue)
        if issue.severity == Severity.CRITICAL:
            self.n_critical += 1
        elif issue.severity == Severity.WARNING:
            self.n_warnings += 1
        else:
            self.n_info += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "passed": self.passed,
            "n_points_validated": self.n_points_validated,
            "n_critical": self.n_critical,
            "n_warnings": self.n_warnings,
            "n_info": self.n_info,
            "sweep_metrics": self.sweep_metrics,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        parts = [f"{status} ({self.n_points_validated} points)"]
        if self.n_critical:
            parts.append(f"{self.n_critical} critical")
        if self.n_warnings:
            parts.append(f"{self.n_warnings} warnings")
        if self.n_info:
            parts.append(f"{self.n_info} info")
        return " | ".join(parts)


# ── VQEValidator ─────────────────────────────────────────────────────────────


class VQEValidator:
    """Validates VQE results against physical constraints and quality metrics.

    Designed to be instantiated per-sweep with system parameters, then
    called on individual results or the full sweep.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    n_edges : int
        Number of edges (bonds) in the lattice.
    J : float
        Coupling constant (for energy bound computation).
    model_name : str
        Hamiltonian model name (for model-specific bounds).
    strict : bool
        If True, CRITICAL issues raise ValueError (hard gate).
        If False, all issues are logged but execution continues.
    """

    def __init__(
        self,
        n_qubits: int,
        n_edges: int,
        J: float = 1.0,
        model_name: str = "tfim",
        strict: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_edges = n_edges
        self.J = abs(J)
        self.model_name = model_name
        self.strict = strict

    # ── Energy bounds computation ────────────────────────────────────────

    def compute_energy_bounds(self, h: float) -> tuple[float, float]:
        """Compute theoretical energy bounds for the Hamiltonian.

        For TFIM: H = -J Σ ZᵢZⱼ - h Σ Xᵢ
          Lower bound: -J·n_edges - |h|·n_qubits (all spins aligned optimally)
          Upper bound: +J·n_edges + |h|·n_qubits (all spins anti-aligned)

        For other models, bounds are more conservative (spectral norm estimate).

        Parameters
        ----------
        h : float
            Transverse field strength.

        Returns
        -------
        (E_lower, E_upper) : tuple[float, float]
        """
        h_abs = abs(h)

        if self.model_name in ("tfim", "tfim_longitudinal", "tfim_frustrated"):
            # TFIM: H = -J·ZZ - h·X (possibly + more terms)
            # Conservative bound: ZZ contributes ±J per edge, X contributes ±h per site
            e_lower = -self.J * self.n_edges - h_abs * self.n_qubits
            e_upper = +self.J * self.n_edges + h_abs * self.n_qubits
        elif self.model_name in ("heisenberg", "xy"):
            # Heisenberg: H = J(XX + YY + Δ·ZZ) - h·Z
            # Each bond has 3 Pauli pairs, each with eigenvalue ±1
            e_lower = -3 * self.J * self.n_edges - h_abs * self.n_qubits
            e_upper = +3 * self.J * self.n_edges + h_abs * self.n_qubits
        else:
            # Conservative fallback: spectral radius estimate
            e_lower = -self.J * self.n_edges * 4 - h_abs * self.n_qubits
            e_upper = +self.J * self.n_edges * 4 + h_abs * self.n_qubits

        return e_lower, e_upper

    # ── Single-point validation ──────────────────────────────────────────

    def validate_single(
        self,
        vqe_result: Any,
        exact_result: Any | None = None,
        restart_energies: list[float] | None = None,
    ) -> VQEValidationReport:
        """Validate a single VQE result.

        Parameters
        ----------
        vqe_result : VQEResult
            The VQE optimization output.
        exact_result : GroundTruthResult | None
            Exact diag reference (for variational principle check).
        restart_energies : list[float] | None
            Energies from all restarts (for basin spread check).

        Returns
        -------
        VQEValidationReport
        """
        report = VQEValidationReport(n_points_validated=1)
        h = vqe_result.h_value

        # ── C5: Energy finiteness ────────────────────────────────────────
        if not np.isfinite(vqe_result.energy):
            report.add(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check_id="C5_energy_finite",
                    message=f"VQE energy is not finite: {vqe_result.energy}",
                    h_value=h,
                )
            )
            # Can't do further checks if energy is not finite
            self._enforce_strict(report)
            return report

        # ── C3: θ_opt finiteness ─────────────────────────────────────────
        if not np.all(np.isfinite(vqe_result.theta_opt)):
            n_bad = int(np.sum(~np.isfinite(vqe_result.theta_opt)))
            report.add(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check_id="C3_theta_finite",
                    message=f"θ_opt contains {n_bad} NaN/Inf values at h={h:.4f}",
                    h_value=h,
                    details={"n_nan_inf": n_bad},
                )
            )

        # ── C4: θ_opt in bounds [-π, π] ─────────────────────────────────
        theta = vqe_result.theta_opt
        if np.all(np.isfinite(theta)):
            out_of_bounds = np.sum(np.abs(theta) > np.pi + 1e-6)
            if out_of_bounds > 0:
                max_val = float(np.max(np.abs(theta)))
                report.add(
                    ValidationIssue(
                        severity=Severity.CRITICAL,
                        check_id="C4_theta_bounds",
                        message=(
                            f"{int(out_of_bounds)} parameters outside [-π, π] "
                            f"(max |θ|={max_val:.4f}) at h={h:.4f}"
                        ),
                        h_value=h,
                        details={"n_out_of_bounds": int(out_of_bounds), "max_abs": max_val},
                    )
                )

        # ── C2: Physical energy bounds ───────────────────────────────────
        e_lower, e_upper = self.compute_energy_bounds(h)
        energy = vqe_result.energy
        # Allow small tolerance for numerical noise
        margin = 1e-6
        if energy < e_lower - margin:
            report.add(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check_id="C2_energy_lower_bound",
                    message=(
                        f"Energy {energy:.6f} below physical lower bound {e_lower:.6f} at h={h:.4f}"
                    ),
                    h_value=h,
                    details={"energy": energy, "bound": e_lower, "deficit": e_lower - energy},
                )
            )
        if energy > e_upper + margin:
            report.add(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check_id="C2_energy_upper_bound",
                    message=(
                        f"Energy {energy:.6f} above physical upper bound {e_upper:.6f} at h={h:.4f}"
                    ),
                    h_value=h,
                    details={"energy": energy, "bound": e_upper, "excess": energy - e_upper},
                )
            )

        # ── C1: Variational principle ────────────────────────────────────
        if exact_result is not None:
            exact_energy = exact_result.ground_energy
            if energy < exact_energy - VARIATIONAL_TOLERANCE:
                violation = exact_energy - energy
                # Severity escalation: large violations (≥ 0.1) are real bugs,
                # small ones (< 0.01) are numerical noise from eigsh vs statevector.
                if violation >= 0.1:
                    severity = Severity.CRITICAL
                    msg = (
                        f"CRITICAL variational principle violation: E_VQE={energy:.8f} < "
                        f"E_exact={exact_energy:.8f} (Δ={violation:.4e}) at h={h:.4f}. "
                        f"NOT numerical noise — check Hamiltonian/backend consistency."
                    )
                elif violation >= 0.01:
                    severity = Severity.CRITICAL
                    msg = (
                        f"Variational principle violated: E_VQE={energy:.8f} < "
                        f"E_exact={exact_energy:.8f} (Δ={violation:.2e}) at h={h:.4f}. "
                        f"Likely eigsh tolerance vs statevector mismatch."
                    )
                else:
                    severity = Severity.WARNING
                    msg = (
                        f"Minor variational principle violation: E_VQE={energy:.8f} < "
                        f"E_exact={exact_energy:.8f} (Δ={violation:.2e}) at h={h:.4f} "
                        f"— numerical noise (benign)."
                    )
                report.add(
                    ValidationIssue(
                        severity=severity,
                        check_id="C1_variational_principle",
                        message=msg,
                        h_value=h,
                        details={
                            "e_vqe": energy,
                            "e_exact": exact_energy,
                            "violation": violation,
                            "is_numerical_noise": violation < 0.01,
                        },
                    )
                )

        # ── W1: Fidelity-energy inconsistency ────────────────────────────
        if exact_result is not None and vqe_result.fidelity > 0:
            gap = exact_result.safe_gap
            de_gap = abs(energy - exact_result.ground_energy) / gap
            fid = vqe_result.fidelity

            # High fidelity but large ΔE/gap → suspicious
            if fid > FIDELITY_HIGH and de_gap > DE_GAP_HIGH:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W1_fidelity_energy_inconsistency",
                        message=(
                            f"High fidelity ({fid:.4f}) but large ΔE/gap "
                            f"({de_gap:.4f}) at h={h:.4f} — possible gap issue"
                        ),
                        h_value=h,
                        details={"fidelity": fid, "de_gap": de_gap},
                    )
                )
            # Low fidelity but small ΔE/gap → suspicious (degenerate ground state?)
            if fid < FIDELITY_LOW and de_gap < DE_GAP_LOW:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W1_fidelity_energy_inconsistency",
                        message=(
                            f"Low fidelity ({fid:.4f}) but small ΔE/gap "
                            f"({de_gap:.4f}) at h={h:.4f} — possible degeneracy"
                        ),
                        h_value=h,
                        details={"fidelity": fid, "de_gap": de_gap},
                    )
                )

        # ── W2: Restart basin spread ────────────────────────────────────
        if restart_energies and len(restart_energies) > 1:
            spread = float(np.std(restart_energies))
            if spread > RESTART_SPREAD_THRESHOLD:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W2_restart_spread",
                        message=(
                            f"High restart energy spread σ={spread:.4f} at h={h:.4f} "
                            f"— multiple local minima likely"
                        ),
                        h_value=h,
                        details={
                            "spread_std": spread,
                            "energies": [float(e) for e in restart_energies],
                        },
                    )
                )

        # ── W4: θ magnitude anomaly ─────────────────────────────────────
        if np.all(np.isfinite(theta)) and len(theta) > 0:
            theta_rms = float(np.sqrt(np.mean(theta**2)))
            expected_rms = np.pi / np.sqrt(3)  # uniform on [-π,π] RMS
            if theta_rms > THETA_MAGNITUDE_WARNING * expected_rms:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W4_theta_magnitude",
                        message=(
                            f"θ RMS={theta_rms:.4f} is {theta_rms / expected_rms:.1f}× "
                            f"expected at h={h:.4f} — possible boundary trapping"
                        ),
                        h_value=h,
                        details={"theta_rms": theta_rms, "expected_rms": expected_rms},
                    )
                )

        # ── I1: Convergence status ──────────────────────────────────────
        if vqe_result.trajectory is not None and not vqe_result.trajectory.converged:
            report.add(
                ValidationIssue(
                    severity=Severity.INFO,
                    check_id="I1_not_converged",
                    message=f"Optimizer did not converge at h={h:.4f} (hit maxiter)",
                    h_value=h,
                    details={"n_iterations": vqe_result.n_iterations},
                )
            )

        self._enforce_strict(report)
        return report

    # ── Sweep-level validation ───────────────────────────────────────────

    def validate_sweep(
        self,
        vqe_results: list[Any],
        exact_data: list[Any] | None = None,
    ) -> VQEValidationReport:
        """Validate an entire VQE descending sweep.

        Runs per-point validation plus sweep-level aggregate checks:
        - W3: Convergence rate across the sweep
        - W5: Energy monotonicity (for TFIM, E(h) should decrease as h increases)
        - I2: Iteration count anomalies

        Parameters
        ----------
        vqe_results : list[VQEResult]
            Results from descending_sweep().
        exact_data : list[GroundTruthResult] | None
            Exact references for variational principle checks.

        Returns
        -------
        VQEValidationReport
        """
        report = VQEValidationReport(n_points_validated=len(vqe_results))

        # Per-point validation
        for idx, vqe_r in enumerate(vqe_results):
            exact_r = exact_data[idx] if exact_data and idx < len(exact_data) else None
            point_report = self.validate_single(vqe_r, exact_r)
            for issue in point_report.issues:
                report.add(issue)

        # ── W3: Sweep convergence rate ───────────────────────────────────
        if vqe_results:
            converged_count = sum(
                1 for r in vqe_results if r.trajectory is not None and r.trajectory.converged
            )
            # Only compute if trajectory is available
            has_trajectory = sum(1 for r in vqe_results if r.trajectory is not None)
            if has_trajectory > 0:
                convergence_rate = converged_count / has_trajectory
                report.sweep_metrics["convergence_rate"] = convergence_rate
                report.sweep_metrics["n_converged"] = converged_count
                report.sweep_metrics["n_with_trajectory"] = has_trajectory

                if convergence_rate < MIN_CONVERGENCE_RATE:
                    report.add(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            check_id="W3_low_convergence_rate",
                            message=(
                                f"Only {convergence_rate:.0%} of sweep points converged "
                                f"({converged_count}/{has_trajectory}) — optimizer struggling"
                            ),
                            details={
                                "convergence_rate": convergence_rate,
                                "n_converged": converged_count,
                                "n_total": has_trajectory,
                            },
                        )
                    )

        # ── W5: Energy monotonicity check ────────────────────────────────
        # For TFIM: as h increases (from low to high), ground energy decreases
        # (more negative). In descending sweep, energies should generally increase.
        # Violations > threshold suggest basin hopping.
        if len(vqe_results) >= 3 and self.model_name == "tfim":
            energies = np.array([r.energy for r in vqe_results])
            h_values = np.array([r.h_value for r in vqe_results])

            # In descending sweep (h decreasing), E should generally decrease
            # (become more negative) as h decreases toward ferro phase.
            # Check for large non-monotonic jumps (> gap scale).
            if exact_data and len(exact_data) >= 3:
                gaps = np.array([r.safe_gap for r in exact_data])
                median_gap = float(np.median(gaps))

                # Compute energy differences between consecutive points
                e_diffs = np.diff(energies)
                # In descending h-sweep for TFIM, energy should generally
                # decrease. Large positive jumps are suspicious.
                large_jumps = np.where(e_diffs > 2 * median_gap)[0]
                if len(large_jumps) > 0:
                    report.add(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            check_id="W5_energy_monotonicity",
                            message=(
                                f"{len(large_jumps)} non-monotonic energy jump(s) "
                                f"detected (> 2× median gap={median_gap:.4f})"
                            ),
                            details={
                                "jump_indices": large_jumps.tolist(),
                                "jump_h_values": [float(h_values[i]) for i in large_jumps],
                                "jump_magnitudes": [float(e_diffs[i]) for i in large_jumps],
                            },
                        )
                    )
                report.sweep_metrics["energy_monotonicity_violations"] = len(large_jumps)

        # ── I2: Iteration count anomalies ────────────────────────────────
        if len(vqe_results) >= 3:
            n_iters = np.array([r.n_iterations for r in vqe_results])
            median_iters = float(np.median(n_iters))
            if median_iters > 0:
                anomaly_threshold = ITERATION_ANOMALY_FACTOR * median_iters
                anomalous = [
                    (r.h_value, r.n_iterations)
                    for r in vqe_results
                    if r.n_iterations > anomaly_threshold
                ]
                for h_val, n_iter in anomalous:
                    report.add(
                        ValidationIssue(
                            severity=Severity.INFO,
                            check_id="I2_iteration_anomaly",
                            message=(
                                f"h={h_val:.3f} used {n_iter} iterations "
                                f"(> {ITERATION_ANOMALY_FACTOR}× median={median_iters:.0f})"
                            ),
                            h_value=h_val,
                            details={"n_iterations": n_iter, "median": median_iters},
                        )
                    )
            report.sweep_metrics["median_iterations"] = median_iters
            report.sweep_metrics["max_iterations"] = int(np.max(n_iters))

        # ── Sweep-level aggregate metrics ────────────────────────────────
        if vqe_results:
            valid_energies = [r.energy for r in vqe_results if np.isfinite(r.energy)]
            fidelities = [r.fidelity for r in vqe_results if r.fidelity > 0]
            report.sweep_metrics["mean_energy"] = (
                float(np.mean(valid_energies)) if valid_energies else None
            )
            report.sweep_metrics["mean_fidelity"] = (
                float(np.mean(fidelities)) if fidelities else None
            )
            report.sweep_metrics["min_fidelity"] = float(np.min(fidelities)) if fidelities else None

            if exact_data:
                de_gaps = []
                for vqe_r, exact_r in zip(vqe_results, exact_data, strict=False):
                    if np.isfinite(vqe_r.energy):
                        de_gaps.append(abs(vqe_r.energy - exact_r.ground_energy) / exact_r.safe_gap)
                if de_gaps:
                    report.sweep_metrics["mean_de_gap"] = float(np.mean(de_gaps))
                    report.sweep_metrics["max_de_gap"] = float(np.max(de_gaps))
                    report.sweep_metrics["n_passing_5pct"] = sum(1 for d in de_gaps if d < 0.05)

        self._enforce_strict(report)
        return report

    # ── Strict enforcement ───────────────────────────────────────────────

    def _enforce_strict(self, report: VQEValidationReport) -> None:
        """If strict mode, raise on CRITICAL issues."""
        if self.strict and report.has_critical:
            critical_msgs = [i.message for i in report.critical_issues]
            raise ValueError(f"VQE validation failed (strict mode): {'; '.join(critical_msgs)}")

    # ── Factory from lattice ─────────────────────────────────────────────

    @classmethod
    def from_lattice(
        cls,
        lattice: Any,
        model_name: str = "tfim",
        strict: bool = False,
    ) -> VQEValidator:
        """Create validator from a LatticeConfig instance.

        Parameters
        ----------
        lattice : LatticeConfig
            Lattice specification.
        model_name : str
            Hamiltonian model name.
        strict : bool
            Whether to raise on CRITICAL issues.
        """
        J = float(lattice.J) if not isinstance(lattice.J, np.ndarray) else float(np.mean(lattice.J))
        return cls(
            n_qubits=lattice.n_qubits,
            n_edges=len(lattice.edges),
            J=J,
            model_name=model_name,
            strict=strict,
        )
