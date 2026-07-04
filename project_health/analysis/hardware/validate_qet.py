#!/usr/bin/env python3
"""QET (Quasi-probabilistic Error Tuning) Post-Execution Validator.

Validates QESEM/QET results by cross-comparing multiple estimation methods
on the same noise-scaling data. Designed to run after any QESEM execution
(standard or QET mode) to assess extrapolation quality and identify issues.

Usage:
    # Validate a specific recovered QESEM result
    .venv/bin/python project_health/analysis/hardware/validate_qet.py \\
        results/recovered/qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json

    # Validate all recovered QESEM results
    .venv/bin/python project_health/analysis/hardware/validate_qet.py --all

    # Validate with custom ground truth
    .venv/bin/python project_health/analysis/hardware/validate_qet.py \\
        results/recovered/qesem_recovered_82aa33cc.json --e-exact -40.5657

Checks performed:
    1. Noise-scale data availability and completeness
    2. WLS linear extrapolation quality vs QESEM result
    3. QESEM heuristic (exponential) quality assessment
    4. Scale monotonicity (energy should degrade with noise)
    5. Complementary pair detection and correlation check
    6. Per-observable consistency across methods
    7. Precision convergence (achieved σ vs target ε)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation.execution.hardware.qesem import (
    _parse_qesem_heuristic,
    _parse_qet_noise_scaling_results,
    extrapolate_qet_wls,
)

# ── Ground truth values (TFIM N=10 OBC) ──────────────────────────────────
GROUND_TRUTH: dict[float, dict[str, float]] = {
    4.0: {"e_exact": -40.565690435512735, "gap": 5.921971082752528},
    3.5: {"e_exact": -35.524253452300285, "gap": 4.762916470256498},
    3.25: {"e_exact": -32.997148093199104, "gap": 4.175000000000000},
    3.0: {"e_exact": -30.466164734096000, "gap": 3.580000000000000},
}

DE_GAP_THRESHOLD = 0.05  # 5%


@dataclass
class QETValidationIssue:
    """Single validation issue found during QET analysis."""

    severity: str  # "error", "warning", "info"
    check: str  # Which check produced this
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class QETValidationReport:
    """Complete validation report for one QESEM/QET result."""

    job_id: str
    h_value: float | None
    e_exact: float | None
    gap: float | None
    issues: list[QETValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if no error-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] job={self.job_id[:12]}... h={self.h_value} "
            f"| {len(self.issues)} issues ({self.n_warnings} warnings)"
        )


def validate_qet_result(
    data: dict,
    e_exact: float | None = None,
    gap: float | None = None,
) -> QETValidationReport:
    """Run all QET validation checks on a recovered QESEM result.

    Parameters
    ----------
    data : dict
        Loaded JSON from a recovered QESEM result file.
    e_exact : float | None
        Exact ground-state energy. Auto-detected from h-value if None.
    gap : float | None
        Spectral gap. Auto-detected from h-value if None.

    Returns
    -------
    QETValidationReport
    """
    job_id = data.get("job_id", "unknown")
    metadata = data.get("metadata", {})

    # Detect h-value from energy or explicit field
    h_value = data.get("h_value")
    if h_value is None:
        e_mit = data.get("energy_mitigated")
        if e_mit is not None:
            if abs(e_mit) > 38:
                h_value = 4.0
            elif abs(e_mit) > 33:
                h_value = 3.5

    # Resolve ground truth
    if e_exact is None and h_value in GROUND_TRUTH:
        e_exact = GROUND_TRUTH[h_value]["e_exact"]
        gap = GROUND_TRUTH[h_value]["gap"]

    report = QETValidationReport(job_id=job_id, h_value=h_value, e_exact=e_exact, gap=gap)

    # ── Check 1: Noise-scale data availability ───────────────────────────
    n_obs = len(data.get("evs", []))
    if n_obs == 0:
        # Try pub_results format
        if "pub_results" in data:
            n_obs = len(data["pub_results"][0].get("evs", []))

    scale_results = _parse_qet_noise_scaling_results(metadata, max(n_obs, 20))

    if not scale_results:
        report.issues.append(
            QETValidationIssue(
                severity="warning",
                check="noise_scale_availability",
                message="No noise_scaling data found in metadata. Cannot perform QET validation.",
            )
        )
        report.metrics["n_scale_observables"] = 0
        return report

    energy_scales = scale_results[0] if scale_results else {}
    report.metrics["n_scale_observables"] = len(scale_results)
    report.metrics["energy_scales_available"] = sorted(energy_scales.keys())
    report.metrics["n_energy_scale_points"] = len(energy_scales)

    if len(energy_scales) < 2:
        report.issues.append(
            QETValidationIssue(
                severity="warning",
                check="noise_scale_availability",
                message=f"Only {len(energy_scales)} energy scale point(s). Need ≥2 for extrapolation.",
            )
        )

    # ── Check 2: Scale monotonicity ──────────────────────────────────────
    # Energy should generally degrade (become less negative) with higher noise
    if len(energy_scales) >= 2:
        sorted_scales = sorted(energy_scales.keys())
        values_by_scale = [energy_scales[s][0] for s in sorted_scales]
        # For TFIM (negative energy), increasing noise → less negative → values increase
        monotonic_violations = 0
        for i in range(len(values_by_scale) - 1):
            if values_by_scale[i + 1] < values_by_scale[i] - 0.5:
                monotonic_violations += 1

        report.metrics["scale_monotonicity_violations"] = monotonic_violations
        if monotonic_violations > 0:
            report.issues.append(
                QETValidationIssue(
                    severity="warning",
                    check="scale_monotonicity",
                    message=(
                        f"{monotonic_violations} monotonicity violation(s) in energy vs noise scale. "
                        f"Physics expectation: as noise increases (higher scale), the measured "
                        f"energy should move away from the exact value (become less negative "
                        f"for TFIM). A violation means a higher-noise measurement accidentally "
                        f"gave a better result — likely a statistical fluctuation, not a "
                        f"systematic issue."
                    ),
                    data={"scales": sorted_scales, "values": values_by_scale},
                )
            )

    # ── Check 3: WLS extrapolation quality ───────────────────────────────
    scales_gt0 = {s: v for s, v in energy_scales.items() if s > 0}
    if len(scales_gt0) >= 2:
        e_wls, std_wls = extrapolate_qet_wls(scales_gt0, extrapolation_order=1)
        report.metrics["wls_linear_energy"] = e_wls
        report.metrics["wls_linear_std"] = std_wls

        if e_exact is not None and gap is not None:
            wls_de_gap = abs(e_wls - e_exact) / gap
            report.metrics["wls_linear_de_gap"] = wls_de_gap

            if wls_de_gap > 0.20:
                report.issues.append(
                    QETValidationIssue(
                        severity="info",
                        check="wls_extrapolation",
                        message=(
                            f"WLS linear extrapolation from {len(scales_gt0)} noise-scale points "
                            f"(scales {sorted(scales_gt0.keys())}) to zero-noise limit gives "
                            f"ΔE/gap={wls_de_gap:.4f} ({wls_de_gap * 100:.1f}%). "
                            f"This exceeds 20% — the linear fit from only 2 points is unreliable. "
                            f"To improve: use QET explicit mode with 5+ scales "
                            f"(e.g., {{0.3: 0.02, 0.5: 0.02, 0.7: 0.02, 1.3: 0.03, 2.0: 0.03}}) "
                            f"to provide enough data for a robust extrapolation."
                        ),
                        data={
                            "e_wls": e_wls,
                            "de_gap": wls_de_gap,
                            "scales_used": sorted(scales_gt0.keys()),
                        },
                    )
                )

    # ── Check 4: QESEM standard result quality ───────────────────────────
    e_qesem = data.get("energy_mitigated")
    e_std = data.get("energy_std")
    if e_qesem is not None and e_exact is not None and gap is not None:
        qesem_de_gap = abs(e_qesem - e_exact) / gap
        report.metrics["qesem_de_gap"] = qesem_de_gap
        report.metrics["qesem_energy"] = e_qesem
        report.metrics["qesem_std"] = e_std

        if qesem_de_gap >= DE_GAP_THRESHOLD:
            report.issues.append(
                QETValidationIssue(
                    severity="error",
                    check="qesem_accuracy",
                    message=(
                        f"QESEM mitigated energy error exceeds acceptance threshold: "
                        f"|E_mitigated - E_exact|/gap = {qesem_de_gap:.4f} ({qesem_de_gap * 100:.2f}%), "
                        f"threshold = {DE_GAP_THRESHOLD * 100:.0f}%. "
                        f"The circuit+mitigation did not achieve chemical-grade accuracy. "
                        f"Possible causes: insufficient QPU time (σ too large), "
                        f"circuit parameters far from optimal, or device noise too high "
                        f"for the characterization model."
                    ),
                    data={"e_qesem": e_qesem, "e_exact": e_exact, "de_gap": qesem_de_gap},
                )
            )

    # ── Check 5: QESEM heuristic comparison ──────────────────────────────
    heur_e, heur_std = _parse_qesem_heuristic(metadata)
    if heur_e is not None:
        report.metrics["heuristic_energy"] = heur_e
        report.metrics["heuristic_std"] = heur_std
        if e_exact is not None and gap is not None:
            heur_de_gap = abs(heur_e - e_exact) / gap
            report.metrics["heuristic_de_gap"] = heur_de_gap

    # ── Check 6: Complementary pair detection ────────────────────────────
    if len(energy_scales) >= 2:
        complementary_pairs = []
        scale_list = sorted(energy_scales.keys())
        for i, s1 in enumerate(scale_list):
            for s2 in scale_list[i + 1 :]:
                if abs(s1 + s2 - 2.0) < 0.01:
                    complementary_pairs.append((s1, s2))
        report.metrics["complementary_pairs"] = complementary_pairs
        report.metrics["n_independent_points"] = len(energy_scales) - len(complementary_pairs)

        if complementary_pairs:
            report.issues.append(
                QETValidationIssue(
                    severity="info",
                    check="complementary_pairs",
                    message=(
                        f"Found {len(complementary_pairs)} complementary pair(s): {complementary_pairs}. "
                        f"Complementary pairs (s₁ + s₂ = 2.0) share the same circuit data — "
                        f"they are NOT statistically independent measurements. "
                        f"For ZNE error bars, only count explicitly-requested scales as "
                        f"independent data points. "
                        f"Effective independent points for fit: {len(energy_scales) - len(complementary_pairs)}."
                    ),
                )
            )

    # ── Check 7: Precision convergence ───────────────────────────────────
    if e_std is not None:
        requested_precision = metadata.get("precision_target") or 0.01
        precision_ratio = e_std / requested_precision if requested_precision > 0 else 0
        report.metrics["precision_ratio"] = precision_ratio
        report.metrics["achieved_std"] = e_std
        report.metrics["requested_precision"] = requested_precision

        if precision_ratio > 2.0:
            report.issues.append(
                QETValidationIssue(
                    severity="warning",
                    check="precision_convergence",
                    message=(
                        f"QESEM did not converge to target precision: achieved σ={e_std:.4f} "
                        f"but target was ε={requested_precision:.4f} (ratio: {precision_ratio:.1f}×). "
                        f"This means the statistical uncertainty on the mitigated energy is "
                        f"{precision_ratio:.1f}× larger than requested. "
                        f"Root cause: the QPU time cap (max_execution_time) was reached before "
                        f"QESEM could accumulate enough shots for convergence. "
                        f"Fix: increase qesem_max_execution_time (current budget produced σ ∝ 1/√T)."
                    ),
                    data={
                        "ratio": precision_ratio,
                        "achieved": e_std,
                        "target": requested_precision,
                    },
                )
            )

    # ── Check 8: Per-observable cross-consistency ────────────────────────
    x_values = data.get("x_values", [])
    if x_values:
        mag_x_mean = float(np.mean(x_values))
        mag_x_std = float(np.std(x_values))
        report.metrics["mag_x_mean"] = mag_x_mean
        report.metrics["mag_x_std"] = mag_x_std
        report.metrics["mag_x_range"] = [float(min(x_values)), float(max(x_values))]

        # For h >> h_c, all X_i should be near 1.0
        if h_value and h_value > 2.0:
            if mag_x_mean < 0.8:
                report.issues.append(
                    QETValidationIssue(
                        severity="warning",
                        check="observable_consistency",
                        message=(
                            f"Mean per-site magnetization ⟨X⟩={mag_x_mean:.4f} is unexpectedly "
                            f"low for h={h_value} (deep paramagnetic phase, where all spins "
                            f"should align with the transverse field → ⟨X⟩ ≈ 1.0). "
                            f"A low value suggests either: (1) severe residual noise after "
                            f"mitigation, (2) wrong VQE parameters submitted to hardware, or "
                            f"(3) a circuit construction error."
                        ),
                    )
                )

    # ── Check 9: Gate fidelity assessment ────────────────────────────────
    gate_fid = metadata.get("gate_fidelities", {})
    if gate_fid:
        report.metrics["gate_fidelities"] = gate_fid
        rzz_fid = gate_fid.get("RZZ") or gate_fid.get("CZ")
        if rzz_fid is not None and rzz_fid < 0.99:
            report.issues.append(
                QETValidationIssue(
                    severity="warning",
                    check="gate_fidelity",
                    message=(
                        f"Two-qubit gate fidelity = {rzz_fid:.4f} ({rzz_fid * 100:.2f}%), "
                        f"below the 99% threshold. Lower gate fidelity means QESEM needs "
                        f"more shots to achieve the same precision (mitigation overhead "
                        f"scales as ~1/fidelity²). Results may still be valid but will "
                        f"require proportionally more QPU time to converge."
                    ),
                    data={"fidelity": rzz_fid},
                )
            )

    # ── Check 10: ZNE gain assessment ────────────────────────────────────
    noisy_energy = data.get("noisy_energy")
    if (
        noisy_energy is not None
        and noisy_energy != 0.0
        and e_qesem is not None
        and e_exact is not None
    ):
        raw_error = abs(noisy_energy - e_exact)
        mit_error = abs(e_qesem - e_exact)
        if raw_error > 1e-10:
            zne_gain = 1.0 - (mit_error / raw_error)
            report.metrics["zne_gain"] = zne_gain
            report.metrics["noisy_error"] = raw_error
            report.metrics["mitigated_error"] = mit_error

            if zne_gain < 0:
                report.issues.append(
                    QETValidationIssue(
                        severity="error",
                        check="zne_gain",
                        message=(
                            f"Negative mitigation gain ({zne_gain:.2%}): the QESEM-mitigated "
                            f"energy (|E_mit - E_exact| = {mit_error:.4f}) is farther from "
                            f"the exact value than the raw unmitigated result "
                            f"(|E_noisy - E_exact| = {raw_error:.4f}). "
                            f"The mitigation process degraded accuracy instead of improving it. "
                            f"Possible causes: insufficient QPU time for convergence, "
                            f"statistical overcorrection, or characterization error."
                        ),
                    )
                )

    return report


def print_report(report: QETValidationReport) -> None:
    """Print a formatted validation report to stdout."""
    status = "✅ PASS" if report.passed else "❌ FAIL"
    print(f"\n{'═' * 70}")
    print(f"  QET VALIDATION REPORT — {status}")
    print(f"{'═' * 70}")
    print(f"  Job ID: {report.job_id}")
    print(f"  h-value: {report.h_value}")
    if report.e_exact:
        print(f"  E_exact: {report.e_exact:.6f}")
    if report.gap:
        print(f"  Gap: {report.gap:.4f}")
    print()

    # Key metrics
    m = report.metrics
    print("  ── Key Metrics ──")
    if "qesem_energy" in m:
        de = m.get("qesem_de_gap", 0)
        print(f"  QESEM energy:      {m['qesem_energy']:.4f} ± {m.get('qesem_std', 0):.4f}")
        print(f"  ΔE/gap (QESEM):    {de:.4f} ({de * 100:.2f}%)")
    if "wls_linear_energy" in m:
        de = m.get("wls_linear_de_gap", 0)
        print(
            f"  WLS energy:        {m['wls_linear_energy']:.4f} ± {m.get('wls_linear_std', 0):.4f}"
        )
        print(f"  ΔE/gap (WLS):      {de:.4f} ({de * 100:.2f}%)")
    if "heuristic_energy" in m:
        de = m.get("heuristic_de_gap", 0)
        print(f"  Heuristic energy:  {m['heuristic_energy']:.4f} ± {m.get('heuristic_std', 0):.4f}")
        print(f"  ΔE/gap (heuristic):{de:.4f} ({de * 100:.2f}%)")
    if "zne_gain" in m:
        print(f"  ZNE gain:          {m['zne_gain']:.2%}")
    if "precision_ratio" in m:
        print(f"  Precision ratio:   {m['precision_ratio']:.1f}× (σ/ε)")
    if "mag_x_mean" in m:
        print(f"  ⟨X⟩ mean:          {m['mag_x_mean']:.4f} ± {m.get('mag_x_std', 0):.4f}")
    if "gate_fidelities" in m:
        print(f"  Gate fidelities:   {m['gate_fidelities']}")
    if "energy_scales_available" in m:
        print(f"  Noise scales:      {m['energy_scales_available']}")
    if "n_independent_points" in m:
        print(f"  Independent pts:   {m['n_independent_points']}")

    # Issues
    if report.issues:
        print()
        print("  ── Issues ──")
        for issue in report.issues:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
            print(f"  {icon} [{issue.check}] {issue.message}")
    else:
        print()
        print("  ── No issues found ──")

    print(f"\n{'═' * 70}\n")


def validate_file(path: Path, e_exact: float | None = None) -> QETValidationReport:
    """Load a QESEM result file and run validation."""
    with open(path) as f:
        data = json.load(f)
    gap = None
    if e_exact is not None:
        # Caller provided e_exact but not gap — try to auto-resolve
        for h, gt in GROUND_TRUTH.items():
            if abs(gt["e_exact"] - e_exact) < 0.01:
                gap = gt["gap"]
                break
    return validate_qet_result(data, e_exact=e_exact, gap=gap)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QET/QESEM Post-Execution Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="QESEM result JSON file(s) to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all recovered QESEM results in results/recovered/",
    )
    parser.add_argument(
        "--e-exact",
        type=float,
        default=None,
        help="Override exact energy (auto-detected from h-value if omitted)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    files: list[Path] = []

    if args.all:
        recovered_dir = _ROOT / "results/recovered"
        files = sorted(recovered_dir.glob("**/qesem_recovered_*.json"))
        if not files:
            print("  No recovered QESEM results found.")
            sys.exit(0)
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        parser.print_help()
        sys.exit(0)

    reports: list[QETValidationReport] = []
    for path in files:
        if not path.exists():
            print(f"  ⚠️ File not found: {path}")
            continue
        report = validate_file(path, e_exact=args.e_exact)
        reports.append(report)

        if args.json:
            pass  # Print at end
        else:
            print_report(report)

    # Summary
    n_pass = sum(1 for r in reports if r.passed)
    n_total = len(reports)

    if args.json:
        from qmbp_simulation.utils.helpers import json_serialize

        output = {
            "n_files": n_total,
            "n_passed": n_pass,
            "reports": [
                {
                    "job_id": r.job_id,
                    "h_value": r.h_value,
                    "passed": r.passed,
                    "n_issues": len(r.issues),
                    "metrics": r.metrics,
                    "issues": [
                        {"severity": i.severity, "check": i.check, "message": i.message}
                        for i in r.issues
                    ],
                }
                for r in reports
            ],
        }
        print(json.dumps(output, indent=2, default=json_serialize))
    else:
        print(f"{'═' * 70}")
        print(f"  SUMMARY: {n_pass}/{n_total} passed")
        print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
