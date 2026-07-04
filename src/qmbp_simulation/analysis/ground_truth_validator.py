"""Ground Truth Validator — Phase 1 post-computation validation.

Validates exact diagonalization and DMRG results for physical consistency
before they propagate to downstream phases (VQE, MPNN, Deploy).

Reuses the same severity model as VQEValidator:
  CRITICAL — result is invalid, must not propagate
  WARNING — result is suspicious, flag for review
  INFO — observational

Validation Checks:
  C1: Gap positivity (gap > 0 for all points)
  C2: Energy bounds (E ∈ [E_lower, E_upper])
  C3: Observable bounds (|⟨X⟩| ≤ 1, |⟨ZZ⟩| ≤ 1 per site/bond)
  C4: Energy finiteness (no NaN/Inf)
  W1: Gap floor detection (all gaps = 2π/N → DMRG excitation failed)
  W2: Energy monotonicity (non-monotonic for TFIM at h >> h_c)
  W3: Symmetry breaking (⟨X⟩ ≈ 0 in paramagnetic phase)
  W4: Gap too small for safe ΔE/gap computation
  I1: Method used (exact vs DMRG) per point

Usage:
    from qmbp_simulation.analysis import GroundTruthValidator

    validator = GroundTruthValidator(n_qubits=20, n_edges=19, model="tfim")
    report = validator.validate(ground_truth_results)

    if report.has_critical:
        logger.error(f"Phase 1 FAILED: {report.critical_issues}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qmbp_simulation.analysis.vqe_validator import (
    Severity,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Minimum acceptable gap for ΔE/gap to be meaningful
MIN_USEFUL_GAP: float = 1e-4

# Threshold for detecting paramagnetic symmetry breaking
PM_PHASE_H_THRESHOLD: float = 2.0  # h > this → expect ⟨X⟩ ≈ 1
PM_PHASE_MX_MIN: float = 0.1  # ⟨X⟩ below this in PM phase is suspicious


# ── Report ───────────────────────────────────────────────────────────────────


@dataclass
class GroundTruthValidationReport:
    """Validation report for Phase 1 (exact diag / DMRG) results."""

    issues: list[ValidationIssue] = field(default_factory=list)
    n_points_validated: int = 0
    n_critical: int = 0
    n_warnings: int = 0
    n_info: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return self.n_critical > 0

    @property
    def has_warnings(self) -> bool:
        return self.n_warnings > 0

    @property
    def passed(self) -> bool:
        return not self.has_critical

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == Severity.CRITICAL:
            self.n_critical += 1
        elif issue.severity == Severity.WARNING:
            self.n_warnings += 1
        else:
            self.n_info += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_points_validated": self.n_points_validated,
            "n_critical": self.n_critical,
            "n_warnings": self.n_warnings,
            "n_info": self.n_info,
            "details": self.details,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        parts = [f"{status} ({self.n_points_validated} points)"]
        if self.n_critical:
            parts.append(f"{self.n_critical} critical")
        if self.n_warnings:
            parts.append(f"{self.n_warnings} warnings")
        return " | ".join(parts)


# ── Validator ────────────────────────────────────────────────────────────────


class GroundTruthValidator:
    """Validates Phase 1 ground truth results for physical consistency.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_edges : int
        Number of lattice edges (bonds).
    J : float
        Coupling constant.
    model : str
        Hamiltonian model name.
    """

    def __init__(
        self,
        n_qubits: int,
        n_edges: int,
        J: float = 1.0,
        model: str = "tfim",
    ) -> None:
        self.n_qubits = n_qubits
        self.n_edges = n_edges
        self.J = abs(J)
        self.model = model

    @classmethod
    def from_lattice(cls, lattice, model: str = "tfim") -> GroundTruthValidator:
        """Create from LatticeConfig."""
        J = float(lattice.J) if np.isscalar(lattice.J) else float(np.mean(lattice.J))
        return cls(
            n_qubits=lattice.n_qubits,
            n_edges=len(lattice.edges),
            J=J,
            model=model,
        )

    def _energy_bounds(self, h: float) -> tuple[float, float]:
        """Spectral energy bounds for the Hamiltonian."""
        h_abs = abs(h)
        if self.model in ("tfim", "tfim_longitudinal", "tfim_frustrated"):
            e_lower = -self.J * self.n_edges - h_abs * self.n_qubits
            e_upper = +self.J * self.n_edges + h_abs * self.n_qubits
        elif self.model in ("heisenberg", "xy"):
            e_lower = -3 * self.J * self.n_edges - h_abs * self.n_qubits
            e_upper = +3 * self.J * self.n_edges + h_abs * self.n_qubits
        else:
            e_lower = -4 * self.J * self.n_edges - h_abs * self.n_qubits
            e_upper = +4 * self.J * self.n_edges + h_abs * self.n_qubits
        return e_lower, e_upper

    def validate(self, results: list) -> GroundTruthValidationReport:
        """Validate a list of GroundTruthResult objects.

        Parameters
        ----------
        results : list[GroundTruthResult]
            Phase 1 results (any order, typically descending h).

        Returns
        -------
        GroundTruthValidationReport
        """
        report = GroundTruthValidationReport(n_points_validated=len(results))

        if not results:
            report.add(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    check_id="C0_empty",
                    message="No ground truth results to validate",
                )
            )
            return report

        n = self.n_qubits
        finite_floor = 2 * np.pi / n if n > 0 else 0.0

        for r in results:
            h = r.h_value

            # C4: Energy finiteness
            if not np.isfinite(r.ground_energy):
                report.add(
                    ValidationIssue(
                        severity=Severity.CRITICAL,
                        check_id="C4_energy_finite",
                        message=f"E={r.ground_energy} not finite at h={h:.4f}",
                        h_value=h,
                    )
                )
                continue

            # C1: Gap positivity
            if r.gap <= 0:
                report.add(
                    ValidationIssue(
                        severity=Severity.CRITICAL,
                        check_id="C1_gap_positive",
                        message=f"gap={r.gap:.6f} ≤ 0 at h={h:.4f}",
                        h_value=h,
                    )
                )

            # C2: Energy bounds
            e_lo, e_hi = self._energy_bounds(h)
            if r.ground_energy < e_lo - 1e-6:
                report.add(
                    ValidationIssue(
                        severity=Severity.CRITICAL,
                        check_id="C2_energy_lower",
                        message=(f"E={r.ground_energy:.6f} < lower bound {e_lo:.6f} at h={h:.4f}"),
                        h_value=h,
                        details={"energy": r.ground_energy, "bound": e_lo},
                    )
                )
            if r.ground_energy > e_hi + 1e-6:
                report.add(
                    ValidationIssue(
                        severity=Severity.CRITICAL,
                        check_id="C2_energy_upper",
                        message=(f"E={r.ground_energy:.6f} > upper bound {e_hi:.6f} at h={h:.4f}"),
                        h_value=h,
                        details={"energy": r.ground_energy, "bound": e_hi},
                    )
                )

            # C3: Observable bounds
            if r.per_site_mag_x is not None:
                mx_max = float(np.max(np.abs(r.per_site_mag_x)))
                if mx_max > 1.0 + 1e-6:
                    report.add(
                        ValidationIssue(
                            severity=Severity.CRITICAL,
                            check_id="C3_obs_x_bound",
                            message=f"|⟨X⟩|={mx_max:.4f} > 1 at h={h:.4f}",
                            h_value=h,
                        )
                    )
            if r.per_bond_corr_zz is not None:
                zz_max = float(np.max(np.abs(r.per_bond_corr_zz)))
                if zz_max > 1.0 + 1e-6:
                    report.add(
                        ValidationIssue(
                            severity=Severity.CRITICAL,
                            check_id="C3_obs_zz_bound",
                            message=f"|⟨ZZ⟩|={zz_max:.4f} > 1 at h={h:.4f}",
                            h_value=h,
                        )
                    )

            # W4: Gap too small for meaningful ΔE/gap
            if 0 < r.gap < MIN_USEFUL_GAP:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W4_gap_tiny",
                        message=(
                            f"gap={r.gap:.2e} < {MIN_USEFUL_GAP} at h={h:.4f} — "
                            f"ΔE/gap will be numerically unstable"
                        ),
                        h_value=h,
                    )
                )

        # W1: Gap floor detection (all gaps = 2π/N)
        gaps = [r.gap for r in results]
        n_at_floor = sum(1 for g in gaps if abs(g - finite_floor) < 1e-4)
        if n_at_floor == len(results) and len(results) > 1:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check_id="W1_gap_floor_all",
                    message=(
                        f"All {len(results)} gaps = 2π/N = {finite_floor:.4f} "
                        f"— DMRG excited-state likely failed for all points"
                    ),
                    details={"finite_size_floor": finite_floor, "n_at_floor": n_at_floor},
                )
            )
        elif n_at_floor > len(results) * 0.5:
            report.add(
                ValidationIssue(
                    severity=Severity.WARNING,
                    check_id="W1_gap_floor_many",
                    message=(
                        f"{n_at_floor}/{len(results)} gaps at finite-size floor "
                        f"2π/N = {finite_floor:.4f}"
                    ),
                    details={"finite_size_floor": finite_floor, "n_at_floor": n_at_floor},
                )
            )

        # W2: Energy monotonicity (TFIM: E decreases as |h| increases)
        if self.model in ("tfim", "tfim_longitudinal") and len(results) >= 3:
            sorted_by_h = sorted(results, key=lambda r: r.h_value)
            energies = [r.ground_energy for r in sorted_by_h]
            h_vals = [r.h_value for r in sorted_by_h]
            # E should decrease (more negative) as h increases
            for i in range(len(energies) - 1):
                if energies[i + 1] > energies[i] + 1e-6:
                    report.add(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            check_id="W2_energy_monotonicity",
                            message=(
                                f"E({h_vals[i + 1]:.3f})={energies[i + 1]:.4f} > "
                                f"E({h_vals[i]:.3f})={energies[i]:.4f} "
                                f"(non-monotonic for TFIM)"
                            ),
                            h_value=h_vals[i + 1],
                        )
                    )
                    break  # Only report first violation

        # W3: Symmetry breaking detection
        pm_points = [r for r in results if r.h_value > PM_PHASE_H_THRESHOLD]
        if pm_points:
            low_mx = [r for r in pm_points if r.mag_x < PM_PHASE_MX_MIN]
            if low_mx:
                report.add(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        check_id="W3_symmetry_breaking",
                        message=(
                            f"⟨X⟩ < {PM_PHASE_MX_MIN} at {len(low_mx)} points "
                            f"with h > {PM_PHASE_H_THRESHOLD} (expect ⟨X⟩ ≈ 1 in PM phase) "
                            f"— likely DMRG Z₂ symmetry breaking"
                        ),
                        details={
                            "affected_h": [r.h_value for r in low_mx[:5]],
                            "mag_x_values": [r.mag_x for r in low_mx[:5]],
                        },
                    )
                )

        # I1: Method info
        n_with_gs = sum(1 for r in results if r.ground_state is not None)
        n_without_gs = len(results) - n_with_gs
        if n_without_gs > 0 and n_with_gs > 0:
            report.add(
                ValidationIssue(
                    severity=Severity.INFO,
                    check_id="I1_mixed_methods",
                    message=(
                        f"{n_with_gs} points have ground_state (exact diag), "
                        f"{n_without_gs} do not (DMRG)"
                    ),
                )
            )

        # Aggregate details
        report.details = {
            "gap_min": min(gaps) if gaps else None,
            "gap_max": max(gaps) if gaps else None,
            "n_at_gap_floor": n_at_floor,
            "e_range": [
                min(r.ground_energy for r in results),
                max(r.ground_energy for r in results),
            ],
            "method_exact": n_with_gs,
            "method_dmrg": n_without_gs,
        }

        return report
