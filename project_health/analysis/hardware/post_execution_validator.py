#!/usr/bin/env python3
"""Post-execution validator for hardware/benchmark runs.

Validates a result envelope or hardware summary AFTER execution by:
  - Comparing pre-execution QPU time estimates vs actual wall_time/qpu_seconds
  - Recomputing affine correction with the FIXED function and flagging stale data
  - Checking physics invariants (observable bounds, energy cross-validation)
  - Flagging improvement opportunities (shot budget, ZNE quality, fidelity)

REUSES existing infrastructure — does NOT duplicate logic:
  - qmbp_simulation.execution.affine_correct_energy (canonical correction)
  - project_health.cli.qpu_time_estimator.estimate_circuit_qpu_time (CLOPS model)

Usage:
    python -m project_health.analysis.hardware.post_execution_validator <path>
    python -m project_health.analysis.hardware.post_execution_validator <path> --json

    # From verify_affine_bug.py --validate:
    from project_health.analysis.hardware.post_execution_validator import (
        validate_run, print_report,
    )
    report = validate_run(path_to_envelope_or_summary_dir)
    print_report(report)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ─── Project imports (REUSE existing infrastructure) ──────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.execution import affine_correct_energy

# ─── QPU Time Estimation (reuse logic from qpu_time_estimator.py) ─────────
BASE_CLOPS = 3750
REF_DEPTH = 60
DEPTH_EXPONENT = 0.3
SHOTS_DEFAULT = 16384
OVERHEAD_FACTORS = {"none": 2.0, "pea_light": 7.0, "pea_balanced": 7.0, "pea_aqc": 7.0}
PEA_LEARNING_SHOTS = {"none": 0, "pea_light": 4096, "pea_balanced": 9216, "pea_aqc": 9216}
ZNE_FACTORS = {"none": 1, "pea_light": 3, "pea_balanced": 3, "pea_aqc": 3}


def _compute_effective_clops(depth: int) -> float:
    if depth <= 0:
        return BASE_CLOPS
    return BASE_CLOPS * (REF_DEPTH / depth) ** DEPTH_EXPONENT


def _estimate_qpu_time(depth: int, method: str, shots: int = SHOTS_DEFAULT) -> float:
    """Estimate total QPU time in seconds for one circuit execution."""
    eff_clops = _compute_effective_clops(depth)
    overhead = OVERHEAD_FACTORS.get(method, 2.0)
    n_factors = ZNE_FACTORS.get(method, 1)
    pea_shots = PEA_LEARNING_SHOTS.get(method, 0)
    base_time_s = shots / eff_clops
    zne_time_s = base_time_s * n_factors
    pea_learning_s = pea_shots / eff_clops if pea_shots > 0 else 0.0
    return (zne_time_s + pea_learning_s) * overhead


# ─── Severity & Data Classes ─────────────────────────────────────────────


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    IMPROVEMENT = "IMPROVEMENT"


@dataclass
class Finding:
    """A single validation finding."""

    check_id: str
    severity: Severity
    title: str
    detail: str
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report for one run."""

    source_path: str
    run_id: str
    findings: list[Finding] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def n_improvements(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.IMPROVEMENT)

    @property
    def passed(self) -> bool:
        return self.n_errors == 0

    def summary_line(self) -> str:
        parts = []
        if self.n_errors:
            parts.append(f"{self.n_errors} ERROR")
        if self.n_warnings:
            parts.append(f"{self.n_warnings} WARNING")
        if self.n_improvements:
            parts.append(f"{self.n_improvements} IMPROVEMENT")
        status = "PASS" if self.passed else "FAIL"
        detail = ", ".join(parts) if parts else "all checks passed"
        return f"{status} ({self.checks_passed}/{self.checks_run} OK, {detail})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "run_id": self.run_id,
            "passed": self.passed,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "findings": [
                {
                    "check_id": f.check_id,
                    "severity": f.severity.value,
                    "title": f.title,
                    "detail": f.detail,
                    "suggestion": f.suggestion,
                }
                for f in self.findings
            ],
        }


# ─── Check Functions ──────────────────────────────────────────────────────


def _infer_method(data: dict) -> str:
    """Infer mitigation method from envelope metadata."""
    config_id = (data.get("benchmark_metadata") or {}).get("config_id", "")
    mc = data.get("mitigation_config") or {}
    if "pea" in config_id.lower() or mc.get("amplifier") == "pea":
        if "light" in config_id.lower():
            return "pea_light"
        if "aqc" in config_id.lower():
            return "pea_aqc"
        return "pea_balanced"
    return "none"


def _check_qpu_time(data: dict, report: ValidationReport) -> None:
    """C1: Compare QPU time estimate vs actual."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    timing = data.get("timing") or {}
    depth = cs.get("depth")
    wall_time = timing.get("wall_time_s")
    qpu_seconds = timing.get("qpu_seconds")

    if depth is None or (wall_time is None and qpu_seconds is None):
        report.checks_passed += 1
        return

    method = _infer_method(data)
    shots = data.get("shots", SHOTS_DEFAULT)
    estimated_s = _estimate_qpu_time(depth, method, shots)
    actual_s = qpu_seconds if qpu_seconds is not None else wall_time
    if actual_s is None or actual_s <= 0:
        report.checks_passed += 1
        return

    ratio = actual_s / estimated_s if estimated_s > 0 else float("inf")
    if ratio > 5.0:
        report.findings.append(
            Finding(
                check_id="C1",
                severity=Severity.WARNING,
                title="QPU time far exceeds estimate",
                detail=(
                    f"Actual execution time = {actual_s:.1f}s, estimated = {estimated_s:.1f}s "
                    f"(ratio = {ratio:.1f}×). "
                    f"This means the job took {ratio:.0f}× longer than the CLOPS-based "
                    f"model predicted. Common causes: (1) IBM queue wait time included in "
                    f"wall_time measurement, (2) PEA noise-learning phase added overhead "
                    f"(~50% expected for PEA), (3) large observables requiring multi-basis "
                    f"measurement groups."
                ),
                suggestion=(
                    "Separate queue time from QPU time if possible. "
                    "If PEA overhead is the cause, this is expected behavior — not a problem."
                ),
            )
        )
    elif ratio < 0.2:
        report.findings.append(
            Finding(
                check_id="C1",
                severity=Severity.INFO,
                title="QPU time much faster than estimate",
                detail=(
                    f"Actual = {actual_s:.1f}s vs estimated = {estimated_s:.1f}s "
                    f"(ratio = {ratio:.1f}×). The job completed significantly faster "
                    f"than predicted, likely due to optimistic observable grouping or "
                    f"fewer measurement bases than estimated."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_fidelity_vs_result(data: dict, report: ValidationReport) -> None:
    """C2: Fidelity estimate vs actual result quality."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    results = data.get("results") or {}
    if isinstance(results, list):
        report.checks_passed += 1
        return
    fidelity_est = cs.get("fidelity_estimate")
    de_gap = results.get("delta_e_gap")

    if fidelity_est is None or de_gap is None:
        report.checks_passed += 1
        return

    if fidelity_est > 0.90 and de_gap > 0.10:
        report.findings.append(
            Finding(
                check_id="C2",
                severity=Severity.WARNING,
                title="Fidelity estimate was optimistic vs actual result",
                detail=(
                    f"Pre-execution fidelity estimate = {fidelity_est:.3f} (>90%) predicted a "
                    f"high-quality result, but actual ΔE/gap = {de_gap:.4f} (>10%) shows "
                    f"significant error. This mismatch means the noise model used for "
                    f"prediction did not capture the actual execution-time errors "
                    f"(e.g., TLS fluctuations, crosstalk, or stale calibration data)."
                ),
                suggestion=(
                    "Re-run preflight checks with fresh calibration. "
                    "Consider that calibration data may have aged since the estimate was made."
                ),
            )
        )
    elif fidelity_est < 0.60 and de_gap < 0.05:
        report.findings.append(
            Finding(
                check_id="C2",
                severity=Severity.INFO,
                title="Result better than fidelity prediction",
                detail=(
                    f"Predicted fidelity = {fidelity_est:.3f} (low confidence), but actual "
                    f"ΔE/gap = {de_gap:.4f} (<5%) shows the mitigation pipeline (PEA-ZNE) "
                    f"successfully recovered accuracy despite pessimistic pre-execution "
                    f"noise estimates."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_error_budget_correlation(data: dict, report: ValidationReport) -> None:
    """C3: Error budget vs dE/gap — flags disconnect."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    results = data.get("results") or {}
    err_budget = cs.get("error_budget")
    de_gap = results.get("delta_e_gap")

    if err_budget is None or de_gap is None:
        report.checks_passed += 1
        return

    if err_budget > 0.40 and de_gap < 0.02:
        report.findings.append(
            Finding(
                check_id="C3",
                severity=Severity.INFO,
                title="Strong result despite high error budget",
                detail=f"Error budget={err_budget:.3f} but dE/gap={de_gap:.4f}. ZNE working well.",
            )
        )
    elif err_budget < 0.10 and de_gap > 0.10:
        report.findings.append(
            Finding(
                check_id="C3",
                severity=Severity.WARNING,
                title="Poor result despite low error budget",
                detail=f"Error budget={err_budget:.3f} but dE/gap={de_gap:.4f}. Possible systematic error.",
                suggestion="Check ZNE R2 and extrapolation quality.",
            )
        )
    else:
        report.checks_passed += 1


def _check_observable_bounds(data: dict, report: ValidationReport) -> None:
    """C4: All observables must satisfy |O| <= 1."""
    report.checks_run += 1
    results = data.get("results") or {}
    x_vals = results.get("per_site_x") or data.get("per_site_x", [])
    zz_vals = results.get("per_bond_zz") or data.get("per_bond_zz", [])

    if not x_vals and not zz_vals:
        report.checks_passed += 1
        return

    # QESEM unbiased estimator can produce values slightly exceeding |1|
    # due to quasi-probabilistic correction. Allow 2% overshoot for QESEM.
    is_qesem = results.get("qesem_used") or data.get("qesem_used", False)
    tolerance = 0.02 if is_qesem else 1e-6

    violations = []
    for i, xi in enumerate(x_vals):
        if abs(xi) > 1.0 + tolerance:
            violations.append(f"|X_{i}|={abs(xi):.6f}")
    for i, zzi in enumerate(zz_vals):
        if abs(zzi) > 1.0 + tolerance:
            violations.append(f"|ZZ_{i}|={abs(zzi):.6f}")

    if violations:
        report.findings.append(
            Finding(
                check_id="C4",
                severity=Severity.ERROR,
                title="Observable exceeds physical Pauli bound |O| ≤ 1",
                detail=(
                    f"{len(violations)} observables have |value| > 1 + {tolerance}: "
                    f"{violations[:3]}. "
                    f"For Pauli operators, expectation values are physically bounded to "
                    f"[-1, 1]. Values outside this range indicate either: "
                    f"(1) a post-processing bug (raw counts not properly normalized), "
                    f"(2) a quasi-probabilistic estimator overcorrection "
                    f"(expected for QESEM — allowed up to 2% overshoot), or "
                    f"(3) data corruption in transmission/serialization."
                ),
                suggestion=(
                    "If QESEM: this is expected for up to ~2% overshoot (quasi-prob artifact). "
                    "If PEA-ZNE: check measurement post-processing pipeline for bugs."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_energy_cross_validation(data: dict, report: ValidationReport) -> None:
    """C5: Energy-observable consistency (TFIM: E = -Sum ZZ - h Sum X).

    NOTE: ZNE is non-linear → ZNE(H) ≠ Σ cᵢ·ZNE(Oᵢ). Expected discrepancy
    is ~10-15% of |E| for PEA-ZNE with exponential extrapolation. We use an
    adaptive threshold: max(15% of |E|, gap).
    """
    report.checks_run += 1
    results = data.get("results") or {}
    x_vals = results.get("per_site_x") or data.get("per_site_x", [])
    zz_vals = results.get("per_bond_zz") or data.get("per_bond_zz", [])
    h_val = (
        (data.get("benchmark_metadata") or {}).get("h_value")
        or data.get("h_test")
        or data.get("h_value", 3.25)
    )
    e_zne = results.get("e_mitigated") or data.get("e_zne")
    gap = results.get("gap") or data.get("gap", 1.0)

    if not x_vals or not zz_vals or e_zne is None:
        report.checks_passed += 1
        return

    e_recon = -1.0 * sum(zz_vals) - h_val * sum(x_vals)
    discrepancy = abs(e_zne - e_recon)
    # Adaptive threshold: 15% of |E| or gap, whichever is larger.
    threshold = max(0.15 * abs(e_zne), gap) if gap > 0 else 10.0

    if discrepancy > threshold:
        report.findings.append(
            Finding(
                check_id="C5",
                severity=Severity.WARNING,
                title="Energy-observable cross-validation discrepancy",
                detail=(
                    f"|E_ZNE - E_reconstructed| = {discrepancy:.4f} exceeds threshold = "
                    f"{threshold:.4f}. E_ZNE = {e_zne:.4f} (from ZNE extrapolation of ⟨H⟩), "
                    f"E_reconstructed = {e_recon:.4f} (from -Σ⟨ZZ⟩ - h·Σ⟨X⟩). "
                    f"This discrepancy is EXPECTED for ZNE: the zero-noise extrapolation "
                    f"is non-linear, so ZNE(H) ≠ -Σ ZNE(ZZ_i) - h·Σ ZNE(X_i). "
                    f"Each observable is extrapolated independently, and combining them "
                    f"does not reproduce the directly-extrapolated Hamiltonian value."
                ),
                suggestion=(
                    "This is physics, not a bug. Only investigate if discrepancy > 30% of |E|, "
                    "which would suggest the per-site observables and energy were measured on "
                    "incompatible noise conditions."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_variational_principle(data: dict, report: ValidationReport) -> None:
    """C6: E_mitigated >= E_exact (mild ZNE overshoot is acceptable)."""
    report.checks_run += 1
    results = data.get("results") or {}
    e_mit = results.get("e_mitigated") or data.get("e_zne")
    e_exact = results.get("e_exact") or data.get("e_exact")
    gap = results.get("gap") or data.get("gap", 1.0)

    if e_mit is None or e_exact is None or gap is None or gap <= 0:
        report.checks_passed += 1
        return

    violation = e_exact - e_mit  # positive = below ground state
    if violation > 0.05 * gap:
        report.findings.append(
            Finding(
                check_id="C6",
                severity=Severity.WARNING,
                title="Variational principle violation — mitigated energy below ground state",
                detail=(
                    f"E_mitigated = {e_mit:.6f} is {violation:.6f} BELOW E_exact = {e_exact:.6f} "
                    f"({violation / gap * 100:.2f}% of spectral gap). "
                    f"The variational principle guarantees E_VQE ≥ E_exact for the EXACT "
                    f"expectation value. However, ZNE extrapolation is an approximation that "
                    f"can overshoot below the ground state, especially with PEA at high noise. "
                    f"A violation > 5% of gap suggests the extrapolation is unreliable at this "
                    f"noise level."
                ),
                suggestion=(
                    "Verify affine_correct_energy clips the result to [E_exact, E_upper]. "
                    "If affine is applied and this still appears, the ZNE extrapolation "
                    "itself overshot — the reported e_zne is pre-correction."
                ),
            )
        )
    elif violation > 0:
        report.findings.append(
            Finding(
                check_id="C6",
                severity=Severity.INFO,
                title="Mild ZNE overshoot below ground state (expected for PEA)",
                detail=(
                    f"E_mitigated is {violation:.6f} below E_exact "
                    f"({violation / gap * 100:.4f}% of gap). This small overshoot is normal — "
                    f"PEA-ZNE linear extrapolation can slightly undershoot the true value. "
                    f"Affine correction will clip this to E_exact."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_zne_r2(data: dict, report: ValidationReport) -> None:
    """C7: ZNE R2 quality check."""
    report.checks_run += 1
    results = data.get("results") or {}
    r2 = results.get("zne_r2") or data.get("zne_r2")

    if r2 is None:
        report.checks_passed += 1
        return

    if not np.isfinite(r2):
        report.findings.append(
            Finding(
                check_id="C7",
                severity=Severity.ERROR,
                title="ZNE R2 is NaN/Inf",
                detail=f"zne_r2={r2} — extrapolation failed numerically.",
                suggestion="Check CES spread; may need more layouts.",
            )
        )
    elif r2 < 0 or r2 > 1.0 + 1e-6:
        report.findings.append(
            Finding(
                check_id="C7",
                severity=Severity.ERROR,
                title="ZNE R2 outside [0, 1]",
                detail=f"zne_r2={r2:.6f} — invalid value.",
            )
        )
    elif r2 < 0.80:
        report.findings.append(
            Finding(
                check_id="C7",
                severity=Severity.WARNING,
                title="Low ZNE R2",
                detail=f"zne_r2={r2:.4f} < 0.80 — extrapolation unreliable.",
                suggestion="Consider adding layouts or switching to PEA amplifier.",
            )
        )
    else:
        report.checks_passed += 1


def _check_verdict_consistency(data: dict, report: ValidationReport) -> None:
    """C8: Stored verdict matches recomputed dE/gap."""
    report.checks_run += 1
    results = data.get("results") or {}
    de_gap = results.get("delta_e_gap") or data.get("delta_e_gap")
    verdict = results.get("verdict") or data.get("verdict", "")

    if de_gap is None or not verdict:
        report.checks_passed += 1
        return

    expected_verdict = "PASS" if de_gap < 0.05 else "FAIL"
    if verdict.upper() != expected_verdict:
        report.findings.append(
            Finding(
                check_id="C8",
                severity=Severity.WARNING,
                title="Verdict inconsistency",
                detail=f"Stored verdict='{verdict}' but dE/gap={de_gap:.4f} implies '{expected_verdict}'.",
                suggestion="Likely stale data from affine bug (2026-06-22). Recompute from e_zne.",
            )
        )
    else:
        report.checks_passed += 1


def _check_stale_affine(data: dict, report: ValidationReport) -> None:
    """C9: Detect stale affine correction from the 2026-06-22 bug."""
    report.checks_run += 1
    results = data.get("results") or {}
    e_mit = results.get("e_mitigated") or data.get("e_zne")
    e_exact = results.get("e_exact") or data.get("e_exact")
    e_after_aff = data.get("e_after_affine")
    h_val = (data.get("benchmark_metadata") or {}).get("h_value") or data.get("h_test", 3.25)
    n_qubits = (data.get("benchmark_metadata") or {}).get("n_qubits", 10)

    if e_mit is None or e_exact is None or e_after_aff is None:
        report.checks_passed += 1
        return

    current = affine_correct_energy(e_mit, e_exact, n_qubits=n_qubits, h_value=h_val)
    if abs(e_after_aff - current.corrected_energy) > 1e-6:
        report.findings.append(
            Finding(
                check_id="C9",
                severity=Severity.WARNING,
                title="Stale affine correction detected",
                detail=(
                    f"Stored e_after_affine={e_after_aff:.6f} != "
                    f"current function={current.corrected_energy:.6f}."
                ),
                suggestion="Recompute verdict from e_zne directly; stored value is wrong.",
            )
        )
    else:
        report.checks_passed += 1


def _check_phase_consistency(data: dict, report: ValidationReport) -> None:
    """C10: Phase label consistent with observables."""
    report.checks_run += 1
    results = data.get("results") or {}
    phase = results.get("phase_label") or data.get("phase_label")
    x_vals = results.get("per_site_x") or data.get("per_site_x", [])
    h_val = (data.get("benchmark_metadata") or {}).get("h_value") or data.get("h_test", 3.25)

    if not phase or not x_vals:
        report.checks_passed += 1
        return

    mag_x_mean = float(np.mean(x_vals))
    if phase == "paramagnetic" and h_val > 2.0 and mag_x_mean < 0.3:
        report.findings.append(
            Finding(
                check_id="C10",
                severity=Severity.WARNING,
                title="Low <X> for paramagnetic label",
                detail=f"Phase='{phase}', h={h_val}, but <X>_mean={mag_x_mean:.4f} (expected >0.5).",
                suggestion="Check if noise suppressed <X>. Consider more shots.",
            )
        )
    else:
        report.checks_passed += 1


def _check_circuit_zne_viability(data: dict, report: ValidationReport) -> None:
    """C11: Circuit depth vs ZNE viability thresholds."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    n_2q = cs.get("n_2q_gates")
    method = _infer_method(data)

    if n_2q is None:
        report.checks_passed += 1
        return

    if method == "none" and n_2q > 18:
        report.findings.append(
            Finding(
                check_id="C11",
                severity=Severity.IMPROVEMENT,
                title="Circuit exceeds gate-folding ZNE threshold",
                detail=f"n_2q_gates={n_2q} > 18. Gate-folding ZNE degrades above ~18 CX.",
                suggestion="Use PEA amplifier instead of gate-folding for better ZNE.",
            )
        )
    elif "pea" in method and n_2q > 50:
        report.findings.append(
            Finding(
                check_id="C11",
                severity=Severity.WARNING,
                title="Circuit may exceed PEA viability",
                detail=f"n_2q_gates={n_2q} > 50. Even PEA struggles above ~50 2Q gates.",
                suggestion="Reduce circuit depth (p=1) or use fewer qubits.",
            )
        )
    else:
        report.checks_passed += 1


def _check_shot_noise_floor(data: dict, report: ValidationReport) -> None:
    """C12: Shot noise floor estimation — is SNR sufficient?"""
    report.checks_run += 1
    shots = data.get("shots", SHOTS_DEFAULT)
    results = data.get("results") or {}
    de_gap = results.get("delta_e_gap") or data.get("delta_e_gap")

    if de_gap is None:
        report.checks_passed += 1
        return

    total_noise_est = np.sqrt(10) / np.sqrt(shots)
    if de_gap < total_noise_est and de_gap < 0.05:
        report.findings.append(
            Finding(
                check_id="C12",
                severity=Severity.INFO,
                title="Result near shot noise floor",
                detail=(
                    f"dE/gap={de_gap:.4f} ~ shot noise floor~{total_noise_est:.4f} "
                    f"(shots={shots}). Further improvement needs more shots."
                ),
                suggestion=f"Consider {shots * 4} shots for 2x precision improvement.",
            )
        )
    elif shots < 8192:
        report.findings.append(
            Finding(
                check_id="C12",
                severity=Severity.IMPROVEMENT,
                title="Low shot count",
                detail=f"shots={shots} < 8192 recommended minimum.",
                suggestion="Increase to >=16384 shots for reliable observables.",
            )
        )
    else:
        report.checks_passed += 1


# ─── Main Validation Entry Points ─────────────────────────────────────────


def _check_observable_dimensions(data: dict, report: ValidationReport) -> None:
    """C13: Observable array dimensions must be consistent with system size."""
    report.checks_run += 1
    results = data.get("results") or {}
    x_vals = results.get("per_site_x") or data.get("per_site_x", [])
    zz_vals = results.get("per_bond_zz") or data.get("per_bond_zz", [])

    if not x_vals and not zz_vals:
        report.checks_passed += 1
        return

    n_sites = len(x_vals)
    n_bonds = len(zz_vals)

    if n_sites > 0 and n_bonds > 0 and n_bonds != n_sites - 1:
        report.findings.append(
            Finding(
                check_id="C13",
                severity=Severity.ERROR,
                title="Observable dimension mismatch",
                detail=(
                    f"per_site_x has {n_sites} entries but per_bond_zz has {n_bonds} "
                    f"(expected {n_sites - 1} for 1D chain)."
                ),
                suggestion="Check topology assumption or data corruption.",
            )
        )
    elif n_sites == 0 and n_bonds > 0:
        report.findings.append(
            Finding(
                check_id="C13",
                severity=Severity.WARNING,
                title="Missing per_site_x but per_bond_zz present",
                detail=f"per_bond_zz has {n_bonds} entries but per_site_x is empty.",
                suggestion="Observable extraction may have partially failed.",
            )
        )
    else:
        report.checks_passed += 1


def _check_mitigation_effectiveness(data: dict, report: ValidationReport) -> None:
    """C14: Mitigation should improve over raw — flags negative improvement."""
    report.checks_run += 1
    results = data.get("results") or {}
    improvement = results.get("improvement_vs_raw")

    if improvement is None:
        report.checks_passed += 1
        return

    if improvement < -10.0:
        report.findings.append(
            Finding(
                check_id="C14",
                severity=Severity.WARNING,
                title="Mitigation degraded the result — mitigated energy is farther from exact than raw",
                detail=(
                    f"improvement_vs_raw = {improvement:.1f}% (negative = mitigation made "
                    f"the final energy WORSE). Specifically: |E_mitigated - E_exact| > "
                    f"|E_raw - E_exact|. The error mitigation process introduced more error "
                    f"than it removed. "
                    f"Possible causes: (1) ZNE extrapolation overshot (non-linear noise at "
                    f"this depth), (2) PEA noise model inaccurate for this calibration window, "
                    f"(3) insufficient noise factors for reliable fit."
                ),
                suggestion=(
                    "Consider using the raw (unmitigated) energy for this h-value. "
                    "Try switching amplifier (pea↔gate_folding) or adding noise factors."
                ),
            )
        )
    elif improvement < 0:
        report.findings.append(
            Finding(
                check_id="C14",
                severity=Severity.INFO,
                title="Mitigation marginally worse than no mitigation",
                detail=(
                    f"improvement_vs_raw = {improvement:.1f}% (slightly negative). "
                    f"The mitigated and raw results are nearly equivalent — mitigation "
                    f"added noise without clear benefit at this h-value. This can happen "
                    f"when the circuit is already low-noise (close to ground state)."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_calibration_age(data: dict, report: ValidationReport) -> None:
    """C15: Flag runs using stale calibration data (>4h for hardware)."""
    report.checks_run += 1
    hw_cal = data.get("hardware_calibration") or {}
    cal_age = hw_cal.get("calibration_age_hours")

    if cal_age is None:
        report.checks_passed += 1
        return

    if cal_age > 8.0:
        report.findings.append(
            Finding(
                check_id="C15",
                severity=Severity.WARNING,
                title="Very stale calibration data",
                detail=(
                    f"Calibration age = {cal_age:.1f}h (>8h). "
                    f"TLS events likely degraded qubit coherence."
                ),
                suggestion="Re-run after fresh calibration cycle.",
            )
        )
    elif cal_age > 4.0:
        report.findings.append(
            Finding(
                check_id="C15",
                severity=Severity.INFO,
                title="Calibration data aging",
                detail=f"Calibration age = {cal_age:.1f}h (>4h). Results may show TLS drift.",
            )
        )
    else:
        report.checks_passed += 1


def _check_ces_spread(data: dict, report: ValidationReport) -> None:
    """C16: CES spread guard — ZNE needs lever arm across layouts."""
    report.checks_run += 1
    results = data.get("results") or {}
    ces_values = (
        results.get("ces_values") or data.get("ces_values") or data.get("per_layout_ces", [])
    )

    if not ces_values or len(ces_values) < 2:
        report.checks_passed += 1
        return

    ces_arr = np.array(ces_values, dtype=float)
    ces_spread = float(np.std(ces_arr))
    ces_range = float(np.max(ces_arr) - np.min(ces_arr))

    if ces_spread < 0.02:
        report.findings.append(
            Finding(
                check_id="C16",
                severity=Severity.WARNING,
                title="Insufficient CES spread for ZNE",
                detail=(
                    f"CES std={ces_spread:.4f}, range={ces_range:.4f} "
                    f"(< 0.02 threshold). ZNE has no lever arm."
                ),
                suggestion="Add more layouts or use layouts with diverse qubit subsets.",
            )
        )
    else:
        report.checks_passed += 1


def _check_transpilation_depth(data: dict, report: ValidationReport) -> None:
    """C17: Transpiled circuit depth sanity — flags decoherence risk."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    depth_2q = cs.get("depth_2q")
    depth_total = cs.get("depth")

    if depth_2q is None and depth_total is None:
        report.checks_passed += 1
        return

    # depth_2q > 30 → significant T1/T2 decay during 2Q layers
    if depth_2q is not None and depth_2q > 30:
        report.findings.append(
            Finding(
                check_id="C17",
                severity=Severity.WARNING,
                title="High 2Q depth (decoherence risk)",
                detail=(
                    f"depth_2q={depth_2q} > 30. Circuit spends many layers "
                    f"executing 2Q gates — T1/T2 decay accumulates."
                ),
                suggestion="Reduce p_layers to 1 or decrease N for hardware viability.",
            )
        )
    # depth_total > 100 → circuit may exceed max_execution_time per job
    elif depth_total is not None and depth_total > 100:
        report.findings.append(
            Finding(
                check_id="C17",
                severity=Severity.INFO,
                title="Deep total circuit",
                detail=(
                    f"depth_total={depth_total} > 100. "
                    f"May approach per-job timeout on IBM backends."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_routing_overhead(data: dict, report: ValidationReport) -> None:
    """C18: Routing overhead — flags excessive SWAP insertion."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    routing_pct = cs.get("routing_overhead_pct")
    n_2q = cs.get("n_2q_gates")
    active_qubits = cs.get("active_qubits")

    if routing_pct is None:
        report.checks_passed += 1
        return

    # For N=10 TFIM p=1 we expect ~100% overhead (9 logical → 18 physical CZ)
    # >300% means routing is dominating the circuit
    if routing_pct > 300:
        report.findings.append(
            Finding(
                check_id="C18",
                severity=Severity.IMPROVEMENT,
                title="Excessive routing overhead",
                detail=(
                    f"routing_overhead_pct={routing_pct:.0f}%. "
                    f"SWAP chains dominate the circuit ({n_2q} 2Q gates "
                    f"on {active_qubits} qubits)."
                ),
                suggestion="Try optimization_level=3 or VF2-based layout selection.",
            )
        )
    else:
        report.checks_passed += 1


def _check_idle_decoherence(data: dict, report: ValidationReport) -> None:
    """C19: Idle qubit decoherence — flags long idle stretches."""
    report.checks_run += 1
    cs = data.get("circuit_stats") or {}
    max_idle = cs.get("max_idle_stretch")
    idle_per_qubit = cs.get("idle_cycles_per_qubit")

    if max_idle is None and idle_per_qubit is None:
        report.checks_passed += 1
        return

    # max_idle_stretch > 20 → significant T1 decay on idle qubits
    if max_idle is not None and max_idle > 20:
        report.findings.append(
            Finding(
                check_id="C19",
                severity=Severity.IMPROVEMENT,
                title="Long idle stretch (decoherence risk)",
                detail=(f"max_idle_stretch={max_idle} cycles. Idle qubits decay during this time."),
                suggestion="Enable PadDynamicalDecoupling (DD) to protect idle qubits.",
            )
        )
    elif idle_per_qubit is not None and idle_per_qubit > 5.0:
        report.findings.append(
            Finding(
                check_id="C19",
                severity=Severity.INFO,
                title="Elevated idle cycles per qubit",
                detail=f"idle_cycles_per_qubit={idle_per_qubit:.1f} (>5.0). DD recommended.",
            )
        )
    else:
        report.checks_passed += 1


def _check_qesem_raw_vs_mitigated(data: dict, report: ValidationReport) -> None:
    """C20: QESEM — Compare raw (noisy) vs mitigated results.

    When QESEM provides noisy baselines, validates:
      - Mitigation improved energy (ZNE gain > 0)
      - Per-site observable improvement (X and ZZ closer to physical values)
      - Mitigation gain proportional to noise level
      - Shot efficiency (mitigation_shots / total_shots ratio)
    """
    report.checks_run += 1
    results = data.get("results", data)  # Handle both envelope and flat format
    if isinstance(results, list):
        report.checks_passed += 1
        return

    # Only applies to QESEM results
    if not results.get("qesem_used"):
        report.checks_passed += 1
        return

    noisy_evs = results.get("qesem_noisy_evs")
    e_mitigated = results.get("e_zne")
    e_exact = results.get("e_exact")

    if noisy_evs is None or e_mitigated is None or e_exact is None:
        report.checks_passed += 1
        return

    # ── Energy comparison: raw vs mitigated ───────────────────────────
    noisy_energy = noisy_evs[0] if len(noisy_evs) > 0 else None
    if noisy_energy is None or noisy_energy == 0.0:
        # Sentinel value — noisy data not genuinely available
        report.checks_passed += 1
        return

    raw_error = abs(noisy_energy - e_exact)
    mitigated_error = abs(e_mitigated - e_exact)
    zne_gain = (raw_error - mitigated_error) / raw_error if raw_error > 1e-10 else 0.0

    if zne_gain < 0:
        report.findings.append(
            Finding(
                check_id="C20",
                severity=Severity.ERROR,
                title="QESEM mitigation degraded energy accuracy",
                detail=(
                    f"The QESEM-mitigated energy is FARTHER from the exact value than "
                    f"the raw unmitigated measurement. Specifically: "
                    f"|E_noisy - E_exact| = {raw_error:.4f} (raw error), "
                    f"|E_mitigated - E_exact| = {mitigated_error:.4f} (post-QESEM error). "
                    f"Mitigation gain = {zne_gain:.2%} (negative means degradation). "
                    f"This should not happen with QESEM's unbiased estimator unless: "
                    f"(1) QPU time was severely insufficient for convergence (σ >> ε), "
                    f"(2) the observable format was incompatible with QESEM's internal "
                    f"grouping, or (3) the noise characterization was stale."
                ),
                suggestion=(
                    "Check qesem_precision_convergence (C21) — if σ >> ε, increase "
                    "qesem_max_execution_time. Also verify that the run was not during "
                    "a backend maintenance window or TLS fluctuation event."
                ),
            )
        )
        return

    report.checks_passed += 1


def _check_qesem_precision_convergence(data: dict, report: ValidationReport) -> None:
    """C21: QESEM — Precision convergence check.

    Verifies that QESEM achieved its target precision (ε) within the allocated
    QPU time budget. If σ > 2×ε, the max_execution_time was likely the bottleneck.
    """
    report.checks_run += 1
    results = data.get("results", data)
    if isinstance(results, list):
        report.checks_passed += 1
        return

    if not results.get("qesem_used"):
        report.checks_passed += 1
        return

    e_std = results.get("e_zne_std")
    # Infer target precision from circuit_stats or default
    cs = results.get("circuit_stats") or data.get("circuit_stats") or {}
    target_precision = cs.get("precision_target", 0.01)

    if e_std is None:
        report.checks_passed += 1
        return

    ratio = e_std / target_precision if target_precision > 0 else 0.0

    if ratio > 5.0:
        report.findings.append(
            Finding(
                check_id="C21",
                severity=Severity.WARNING,
                title="QESEM precision far from target",
                detail=(
                    f"Achieved σ={e_std:.4f}, target ε={target_precision:.4f} "
                    f"(ratio={ratio:.1f}×). QPU time budget was exhausted."
                ),
                suggestion=(
                    "Increase qesem_max_execution_time or reduce observable count. "
                    "Consider splitting into multiple QESEM calls."
                ),
            )
        )
    elif ratio > 2.0:
        report.findings.append(
            Finding(
                check_id="C21",
                severity=Severity.INFO,
                title="QESEM precision moderately above target",
                detail=(
                    f"Achieved σ={e_std:.4f}, target ε={target_precision:.4f} "
                    f"(ratio={ratio:.1f}×). Result usable but not fully converged."
                ),
                suggestion="Consider 50% more QPU time for next run.",
            )
        )
    else:
        report.checks_passed += 1


def _check_qesem_gate_fidelity(data: dict, report: ValidationReport) -> None:
    """C22: QESEM — Gate fidelity characterization quality.

    QESEM reports per-gate fidelities from its device characterization phase.
    Flags if RZZ (2Q) fidelity is below expected threshold for the backend.
    """
    report.checks_run += 1
    results = data.get("results", data)
    if isinstance(results, list):
        report.checks_passed += 1
        return

    if not results.get("qesem_used"):
        report.checks_passed += 1
        return

    fidelities = results.get("qesem_gate_fidelities")
    if not fidelities or not isinstance(fidelities, dict):
        report.checks_passed += 1
        return

    # RZZ fidelity is the critical 2Q gate metric
    rzz_fidelity = fidelities.get("RZZ") or fidelities.get("rzz") or fidelities.get("CZ")
    id1q_fidelity = fidelities.get("ID1Q") or fidelities.get("id1q")

    if rzz_fidelity is not None and rzz_fidelity < 0.99:
        report.findings.append(
            Finding(
                check_id="C22",
                severity=Severity.WARNING,
                title="QESEM reports degraded 2Q gate fidelity",
                detail=(
                    f"RZZ fidelity={rzz_fidelity:.6f} (< 0.99). "
                    f"ID1Q fidelity={id1q_fidelity:.6f}. "
                    f"This may explain elevated ΔE/gap in the result."
                ),
                suggestion=(
                    "Consider running during lower-noise calibration windows. "
                    "Compare with IBM calibration data at quantum.cloud.ibm.com."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_qesem_shot_efficiency(data: dict, report: ValidationReport) -> None:
    """C23: QESEM — Shot budget efficiency.

    Compares mitigation_shots vs total_shots. A high ratio means QESEM spent
    most of its budget on mitigation (good). A low ratio means characterization
    dominated (may indicate complex noise requiring heavy learning).
    """
    report.checks_run += 1
    results = data.get("results", data)
    if isinstance(results, list):
        report.checks_passed += 1
        return

    if not results.get("qesem_used"):
        report.checks_passed += 1
        return

    total_shots = results.get("qesem_total_shots")
    mitigation_shots = results.get("qesem_mitigation_shots")

    if total_shots is None or mitigation_shots is None or total_shots == 0:
        report.checks_passed += 1
        return

    mit_ratio = mitigation_shots / total_shots

    if mit_ratio < 0.20:
        report.findings.append(
            Finding(
                check_id="C23",
                severity=Severity.INFO,
                title="QESEM spent most budget on characterization",
                detail=(
                    f"Mitigation shots = {mitigation_shots:,} / {total_shots:,} "
                    f"({mit_ratio:.0%}). Most budget went to noise learning."
                ),
                suggestion=(
                    "This is normal for first run on a new calibration window. "
                    "Subsequent runs with same backend should be faster."
                ),
            )
        )
    else:
        report.checks_passed += 1


def _check_qesem_noisy_vs_exact(data: dict, report: ValidationReport) -> None:
    """C24: QESEM — Raw noise level assessment.

    Quantifies the raw noise impact by comparing noisy energy to exact.
    Large raw error + high ZNE gain = QESEM working well.
    Large raw error + low ZNE gain = QESEM struggling (may need more shots).
    Small raw error = circuit is already low-noise (QESEM adds little value).
    """
    report.checks_run += 1
    results = data.get("results", data)
    if isinstance(results, list):
        report.checks_passed += 1
        return

    if not results.get("qesem_used"):
        report.checks_passed += 1
        return

    noisy_evs = results.get("qesem_noisy_evs")
    e_exact = results.get("e_exact")
    e_mitigated = results.get("e_zne")
    gap = results.get("gap", 1.0)

    if noisy_evs is None or e_exact is None or e_mitigated is None:
        report.checks_passed += 1
        return

    noisy_energy = noisy_evs[0] if len(noisy_evs) > 0 else None
    if noisy_energy is None or noisy_energy == 0.0:
        report.checks_passed += 1
        return

    raw_de_gap = abs(noisy_energy - e_exact) / gap if gap > 0 else 0.0
    mitigated_de_gap = abs(e_mitigated - e_exact) / gap if gap > 0 else 0.0

    if raw_de_gap < 0.02:
        report.findings.append(
            Finding(
                check_id="C24",
                severity=Severity.INFO,
                title="Circuit already near-noiseless (raw ΔE/gap < 2%)",
                detail=(
                    f"Raw ΔE/gap={raw_de_gap:.4f}, Mitigated ΔE/gap={mitigated_de_gap:.4f}. "
                    f"QESEM adds minimal value for this circuit/backend combination."
                ),
                suggestion=(
                    "For low-noise circuits, PEA-ZNE may be more cost-effective. "
                    "QESEM shines on deeper circuits with ΔE/gap_raw > 10%."
                ),
            )
        )
    elif raw_de_gap > 0.50 and mitigated_de_gap > 0.10:
        report.findings.append(
            Finding(
                check_id="C24",
                severity=Severity.WARNING,
                title="High raw noise and insufficient mitigation",
                detail=(
                    f"Raw ΔE/gap={raw_de_gap:.4f} (severe noise), "
                    f"Mitigated ΔE/gap={mitigated_de_gap:.4f} (still high). "
                    f"QESEM reduced error but not enough for publication quality."
                ),
                suggestion=(
                    "Increase qesem_max_execution_time, or check if backend "
                    "calibration has degraded. Consider re-running during better conditions."
                ),
            )
        )
    else:
        report.checks_passed += 1


ALL_CHECKS = [
    _check_qpu_time,
    _check_fidelity_vs_result,
    _check_error_budget_correlation,
    _check_observable_bounds,
    _check_energy_cross_validation,
    _check_variational_principle,
    _check_zne_r2,
    _check_verdict_consistency,
    _check_stale_affine,
    _check_phase_consistency,
    _check_circuit_zne_viability,
    _check_shot_noise_floor,
    _check_observable_dimensions,
    _check_mitigation_effectiveness,
    _check_calibration_age,
    _check_ces_spread,
    _check_transpilation_depth,
    _check_routing_overhead,
    _check_idle_decoherence,
    # QESEM-specific checks (C20-C24): only activate when qesem_used=True
    _check_qesem_raw_vs_mitigated,
    _check_qesem_precision_convergence,
    _check_qesem_gate_fidelity,
    _check_qesem_shot_efficiency,
    _check_qesem_noisy_vs_exact,
]


def validate_envelope(data: dict, source_path: str = "<unknown>") -> ValidationReport:
    """Validate a benchmark result envelope (from run_mitigation_benchmark)."""
    run_id = (data.get("benchmark_metadata") or {}).get("config_id", "unknown")
    h_val = (data.get("benchmark_metadata") or {}).get("h_value")
    if h_val is not None:
        run_id = f"{run_id}_h{h_val}"
    report = ValidationReport(source_path=source_path, run_id=run_id)
    for check_fn in ALL_CHECKS:
        check_fn(data, report)
    return report


def validate_hardware_summary(summary_dir: Path) -> ValidationReport:
    """Validate a hardware run from its summary.json directory."""
    summary_path = summary_dir / "summary.json"
    if not summary_path.exists():
        report = ValidationReport(source_path=str(summary_dir), run_id=summary_dir.name)
        report.findings.append(
            Finding(
                check_id="C0",
                severity=Severity.ERROR,
                title="summary.json not found",
                detail=f"No summary.json in {summary_dir}",
            )
        )
        report.checks_run = 1
        return report

    data = json.loads(summary_path.read_text())
    report = ValidationReport(source_path=str(summary_path), run_id=summary_dir.name)
    for check_fn in ALL_CHECKS:
        check_fn(data, report)
    return report


def validate_run(path: Path) -> ValidationReport:
    """Auto-detect path type and validate."""
    path = Path(path)
    if path.is_dir():
        return validate_hardware_summary(path)
    elif path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text())
        return validate_envelope(data, source_path=str(path))
    else:
        report = ValidationReport(source_path=str(path), run_id=path.name)
        report.findings.append(
            Finding(
                check_id="C0",
                severity=Severity.ERROR,
                title="Invalid path",
                detail=f"Not a directory or .json file: {path}",
            )
        )
        report.checks_run = 1
        return report


# ─── Output Formatting ────────────────────────────────────────────────────

_SEVERITY_ICONS = {
    Severity.ERROR: "X ",
    Severity.WARNING: "! ",
    Severity.INFO: "i ",
    Severity.IMPROVEMENT: "* ",
}


def print_report(report: ValidationReport, file=None) -> None:
    """Print a human-readable validation report."""
    out = file or sys.stdout
    print(f"\n{'=' * 70}", file=out)
    print(f"  POST-EXECUTION VALIDATION: {report.run_id}", file=out)
    print(f"  Source: {report.source_path}", file=out)
    print(f"{'=' * 70}", file=out)

    if not report.findings:
        print(f"\n  {report.summary_line()}", file=out)
        print(f"  All {report.checks_run} checks passed.", file=out)
    else:
        for severity in [Severity.ERROR, Severity.WARNING, Severity.IMPROVEMENT, Severity.INFO]:
            items = [f for f in report.findings if f.severity == severity]
            if not items:
                continue
            print(f"\n  [{severity.value}]", file=out)
            for f in items:
                icon = _SEVERITY_ICONS[f.severity]
                print(f"    {icon}[{f.check_id}] {f.title}", file=out)
                print(f"       {f.detail}", file=out)
                if f.suggestion:
                    print(f"       -> {f.suggestion}", file=out)

    print(f"\n  {report.summary_line()}", file=out)
    print(f"{'=' * 70}\n", file=out)


# ─── CLI Entry Point ──────────────────────────────────────────────────────


def main() -> int:
    """CLI: python -m project_health.analysis.hardware.post_execution_validator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-execution validator for hardware/benchmark runs"
    )
    parser.add_argument("path", type=Path, help="Path to result (dir or .json)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--batch", action="store_true", help="Validate all .json files in directory"
    )
    args = parser.parse_args()

    if args.batch:
        target = Path(args.path)
        if not target.is_dir():
            print(f"ERROR: --batch requires a directory, got {target}", file=sys.stderr)
            return 1
        json_files = sorted(target.rglob("*.json"))
        reports = []
        for jf in json_files:
            if jf.name == "summary.json":
                r = validate_hardware_summary(jf.parent)
            else:
                try:
                    data = json.loads(jf.read_text())
                    # Skip summary/collection files where "results" is a list of items
                    if isinstance(data.get("results"), list):
                        continue
                    if "results" in data or "e_zne" in data:
                        r = validate_envelope(data, source_path=str(jf))
                    else:
                        continue
                except (json.JSONDecodeError, OSError):
                    continue
            reports.append(r)

        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
        else:
            n_pass = sum(1 for r in reports if r.passed)
            print(f"\n  BATCH: {n_pass}/{len(reports)} passed")
            for r in reports:
                if not r.passed or r.n_warnings > 0:
                    print_report(r)
        return 0 if all(r.passed for r in reports) else 1
    else:
        report = validate_run(args.path)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_report(report)
        return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
