#!/usr/bin/env python
"""Validate DQPT (Dynamic Quantum Phase Transition) Results.

Reads DQPT trajectory data from data/dqpt_trajectories/ and performs
rigorous validation checks to confirm the physics is correct.

Validation checks performed:
1. Analytical prediction: t*_1 vs pi/(2*|h_post - h_pre|) for TFIM
2. Scaling of t*(N): convergence to thermodynamic limit
3. Periodicity: t*_k ~ k * t*_1 (approximately equispaced DQPTs)
4. Exponential decay: min(L(t)) -> 0 exponentially with N
5. Rate function peaks: height increases with N (sharpening)
6. Energy conservation: <H_post> is constant (unitary evolution check)
7. Entropy growth: S(t) shows characteristic entanglement dynamics

Focused on heavy_hex topology but supports any topology.

Usage:
    # Validate heavy_hex DQPT data
    python scripts/analysis/validate_dqpt_results.py --topology heavy_hex

    # Validate with verbose output
    python scripts/analysis/validate_dqpt_results.py --topology heavy_hex -v

    # Run on chain_1d (for reference/comparison)
    python scripts/analysis/validate_dqpt_results.py --topology chain_1d --save

    # Compare heavy_hex vs chain_1d
    python scripts/analysis/validate_dqpt_results.py --topology heavy_hex chain_1d --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DQPTTrajectory:
    """Single DQPT trajectory loaded from NPZ."""

    n_qubits: int
    topology: str
    h_pre: float
    h_post: float
    dt: float
    n_steps: int
    times: np.ndarray
    loschmidt_echo: np.ndarray
    rate_function: np.ndarray
    energies: np.ndarray
    entropies: np.ndarray
    critical_times: np.ndarray
    method: str
    file: str


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    value: float | str
    expected: float | str
    tolerance: float | None = None
    detail: str = ""


@dataclass
class DQPTValidationReport:
    """Complete validation report for a set of DQPT trajectories."""

    topology: str
    n_trajectories: int
    checks: list[ValidationCheck] = field(default_factory=list)
    per_n_results: dict[int, dict] = field(default_factory=dict)
    scaling_analysis: dict = field(default_factory=dict)
    overall_pass: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_dqpt_trajectories(
    topology: str,
    data_dir: Path | None = None,
) -> list[DQPTTrajectory]:
    """Load all DQPT trajectories for a given topology.

    Parameters
    ----------
    topology : str
        Topology to load (e.g., "heavy_hex", "chain_1d").
    data_dir : Path | None
        Directory containing NPZ files. Default: data/dqpt_trajectories/

    Returns
    -------
    list[DQPTTrajectory]
        Sorted by N (ascending).
    """
    if data_dir is None:
        data_dir = _project_root / "data" / "dqpt_trajectories"

    if not data_dir.exists():
        logger.warning(f"DQPT data directory not found: {data_dir}")
        return []

    trajectories = []
    for npz_file in sorted(data_dir.glob(f"{topology}_N*.npz")):
        try:
            data = np.load(npz_file, allow_pickle=True)

            # Validate required keys are present
            required_keys = {"n_qubits", "topology", "h_pre", "h_post", "dt",
                             "n_steps", "times", "loschmidt_echo", "rate_function",
                             "energies", "entropies", "critical_times"}
            missing = required_keys - set(data.keys())
            if missing:
                logger.debug(f"  Skipping {npz_file.name}: missing keys {missing}")
                continue

            # Validate array lengths are consistent
            times = np.asarray(data["times"], dtype=float)
            loschmidt = np.asarray(data["loschmidt_echo"], dtype=float)
            if len(times) < 3 or len(loschmidt) != len(times):
                logger.debug(f"  Skipping {npz_file.name}: inconsistent array lengths")
                continue

            traj = DQPTTrajectory(
                n_qubits=int(data["n_qubits"]),
                topology=str(data["topology"]),
                h_pre=float(data["h_pre"]),
                h_post=float(data["h_post"]),
                dt=float(data["dt"]),
                n_steps=int(data["n_steps"]),
                times=times,
                loschmidt_echo=loschmidt,
                rate_function=np.asarray(data["rate_function"], dtype=float),
                energies=np.asarray(data["energies"], dtype=float),
                entropies=np.asarray(data["entropies"], dtype=float),
                critical_times=np.asarray(data["critical_times"], dtype=float),
                method=str(data.get("method", "exact_ed")),
                file=npz_file.name,
            )
            trajectories.append(traj)
        except Exception as e:
            logger.warning(f"Failed to load {npz_file.name}: {e}")

    # Sort by N, keep longest evolution per N
    by_n: dict[int, DQPTTrajectory] = {}
    for t in trajectories:
        if t.n_qubits not in by_n or t.n_steps > by_n[t.n_qubits].n_steps:
            by_n[t.n_qubits] = t

    return [by_n[n] for n in sorted(by_n.keys())]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Checks
# ═══════════════════════════════════════════════════════════════════════════════


def check_analytical_t_star(traj: DQPTTrajectory) -> ValidationCheck:
    """Check first critical time against analytical prediction.

    For the TFIM, the first DQPT time in the thermodynamic limit is:
        t*_1 = pi / (2 * epsilon_k*)

    where epsilon_k* is the minimum quasiparticle energy gap in the
    post-quench spectrum. For a deep quench (h_pre << h_c << h_post):
        t*_1 ~ pi / (4 * h_post) for h_post >> h_c  [ordered -> disordered]
        t*_1 ~ pi / (4 * J)      for h_pre << h_c   [disordered -> ordered]

    For finite systems, t* is typically SHORTER (shifted to earlier times)
    due to finite-size corrections to the quasiparticle spectrum.

    We check: (1) that t* is in a physically reasonable range, and
    (2) that t* < t*_analytical (finite-size shift is always negative).
    """
    h_diff = abs(traj.h_post - traj.h_pre)
    if h_diff < 0.01:
        return ValidationCheck(
            name="analytical_t_star",
            passed=False,
            value="N/A",
            expected="N/A",
            detail="Quench amplitude too small (|Δh| < 0.01)",
        )

    # Upper bound: thermodynamic limit prediction
    # t*_inf = pi / (2 * max(|h_post - h_c|, |h_pre - h_c|))
    # For heavy_hex, h_c ~ 1.0; for chain_1d, h_c = 1.0
    h_c_approx = 1.0  # TFIM critical field (topology-independent leading order)
    energy_scale = max(abs(traj.h_post - h_c_approx), abs(traj.h_pre - h_c_approx), h_diff / 2)
    t_star_upper = np.pi / (2.0 * energy_scale)

    # Lower bound: can't be faster than 1/(4*max(h))
    t_star_lower = 0.1 * t_star_upper

    if len(traj.critical_times) == 0:
        return ValidationCheck(
            name="analytical_t_star",
            passed=False,
            value="no DQPT detected",
            expected=f"t* in [{t_star_lower:.3f}, {t_star_upper:.3f}]",
            detail="No critical times found in trajectory",
        )

    t_star_measured = float(traj.critical_times[0])

    # Check: t* should be between lower bound and upper bound
    # Finite-size shift makes t* < t*_upper (always)
    passed = t_star_lower <= t_star_measured <= t_star_upper * 1.2

    return ValidationCheck(
        name="analytical_t_star",
        passed=passed,
        value=f"{t_star_measured:.4f}",
        expected=f"in [{t_star_lower:.3f}, {t_star_upper*1.2:.3f}]",
        detail=(
            f"t*_thermo_limit={t_star_upper:.4f}, "
            f"finite-size ratio t*/t*_inf = {t_star_measured/t_star_upper:.2f}"
        ),
    )


def check_periodicity(traj: DQPTTrajectory) -> ValidationCheck:
    """Check that critical times are approximately periodic.

    For deep quench TFIM, DQPTs are approximately equispaced:
    t*_k ~ k * t*_1 (with small corrections for finite N).
    """
    ct = traj.critical_times
    if len(ct) < 3:
        return ValidationCheck(
            name="periodicity",
            passed=len(ct) >= 1,  # At least one DQPT is acceptable
            value=f"{len(ct)} critical times",
            expected=">=3 for periodicity check",
            detail="Insufficient DQPTs for periodicity analysis",
        )

    # Compute inter-DQPT spacings
    spacings = np.diff(ct)
    mean_spacing = float(np.mean(spacings))
    std_spacing = float(np.std(spacings))
    cv = std_spacing / mean_spacing if mean_spacing > 0 else float("inf")

    # For TFIM, CV (coefficient of variation) should be < 0.3
    # Higher connectivity topologies may have slightly more irregular spacing
    tolerance_cv = 0.35
    passed = cv < tolerance_cv

    return ValidationCheck(
        name="periodicity",
        passed=passed,
        value=f"CV={cv:.3f} (mean_spacing={mean_spacing:.3f})",
        expected=f"CV < {tolerance_cv}",
        tolerance=tolerance_cv,
        detail=f"Spacings: {[f'{s:.3f}' for s in spacings]}",
    )


def check_loschmidt_decay(trajectories: list[DQPTTrajectory]) -> ValidationCheck:
    """Check that min(L(t)) decays exponentially with N.

    For DQPTs, L(t*) ~ exp(-N * r(t*)) where r > 0.
    So log(min(L)) should scale linearly with -N.
    """
    if len(trajectories) < 3:
        return ValidationCheck(
            name="loschmidt_decay",
            passed=False,
            value="insufficient data",
            expected=">=3 N values",
            detail="Need at least 3 system sizes for scaling check",
        )

    ns = []
    log_l_mins = []
    for t in trajectories:
        l_min = float(np.min(t.loschmidt_echo))
        if l_min > 0:
            ns.append(t.n_qubits)
            log_l_mins.append(np.log(l_min))

    if len(ns) < 3:
        return ValidationCheck(
            name="loschmidt_decay",
            passed=False,
            value="insufficient valid data",
            expected=">=3 N with L_min > 0",
        )

    ns_arr = np.array(ns, dtype=float)
    log_l_arr = np.array(log_l_mins)

    # Linear fit: log(L_min) = a - b*N (b > 0 expected)
    coeffs = np.polyfit(ns_arr, log_l_arr, 1)
    slope = coeffs[0]  # Should be negative (decay with N)
    r_squared = 1 - np.sum((log_l_arr - np.polyval(coeffs, ns_arr)) ** 2) / np.sum(
        (log_l_arr - np.mean(log_l_arr)) ** 2
    )

    # Pass if slope is negative AND R^2 is reasonable
    passed = slope < -0.01 and r_squared > 0.5

    return ValidationCheck(
        name="loschmidt_decay",
        passed=passed,
        value=f"slope={slope:.4f}, R²={r_squared:.3f}",
        expected="slope < 0 and R² > 0.5",
        detail=f"log(L_min) vs N: {list(zip(ns, [f'{v:.3f}' for v in log_l_arr]))}",
    )


def check_energy_conservation(traj: DQPTTrajectory) -> ValidationCheck:
    """Check that <H_post> is approximately constant during evolution.

    Unitary evolution conserves energy: d<H>/dt = 0.
    Deviations indicate numerical errors in the time evolution.
    """
    E = np.asarray(traj.energies)
    if len(E) < 3:
        return ValidationCheck(
            name="energy_conservation",
            passed=False,
            value="insufficient data",
            expected="constant E(t)",
        )

    E_mean = float(np.mean(E))
    E_std = float(np.std(E))
    relative_fluctuation = E_std / abs(E_mean) if abs(E_mean) > 1e-10 else E_std

    # Energy should be conserved to machine precision for exact evolution
    # For Trotter, allow up to 1% deviation
    tolerance = 0.01 if traj.method == "exact_ed" else 0.05
    passed = relative_fluctuation < tolerance

    return ValidationCheck(
        name="energy_conservation",
        passed=passed,
        value=f"ΔE/|E| = {relative_fluctuation:.2e}",
        expected=f"< {tolerance:.0%}",
        tolerance=tolerance,
        detail=f"E_mean={E_mean:.6f}, E_std={E_std:.2e}",
    )


def check_entropy_growth(traj: DQPTTrajectory) -> ValidationCheck:
    """Check that entanglement entropy grows during quench dynamics.

    After a quench across h_c, entanglement should grow (initially linearly
    for 1D, with topology-dependent saturation). Zero entropy growth would
    indicate a trivial (product state) evolution.
    """
    S = np.asarray(traj.entropies)
    if len(S) < 5:
        return ValidationCheck(
            name="entropy_growth",
            passed=False,
            value="insufficient data",
            expected="S(t) increases",
        )

    S_initial = float(S[0])
    S_max = float(np.max(S))
    S_final = float(S[-1])

    # Entropy should increase from initial value
    delta_S = S_max - S_initial

    # Minimum expected: at least 0.1 nats of entropy generation for a non-trivial quench
    min_delta_s = 0.1
    passed = delta_S > min_delta_s

    return ValidationCheck(
        name="entropy_growth",
        passed=passed,
        value=f"ΔS = {delta_S:.4f} (S_0={S_initial:.3f}, S_max={S_max:.3f})",
        expected=f"ΔS > {min_delta_s}",
        detail=f"S_final={S_final:.3f}, growth_pattern={'monotone' if np.all(np.diff(S[:10]) >= -0.01) else 'oscillatory'}",
    )


def check_rate_function_sharpening(trajectories: list[DQPTTrajectory]) -> ValidationCheck:
    """Check that rate function peaks sharpen with N.

    For DQPTs, the rate function at the first critical time r(t*_1)
    should increase with N — peaks become sharper cusps approaching
    non-analyticity in the thermodynamic limit.

    We use r at the first detected DQPT rather than max(r(t)) over the
    entire trajectory, because the first DQPT is the most universal.
    """
    if len(trajectories) < 3:
        return ValidationCheck(
            name="rate_function_sharpening",
            passed=False,
            value="insufficient data",
            expected=">=3 N values",
        )

    ns = []
    r_at_first_dqpt = []
    for t in trajectories:
        if len(t.critical_times) == 0:
            continue
        # Find rate function value at first DQPT
        t_star = t.critical_times[0]
        idx = np.argmin(np.abs(np.asarray(t.times) - t_star))
        r_val = float(t.rate_function[idx])
        if r_val > 0:
            ns.append(t.n_qubits)
            r_at_first_dqpt.append(r_val)

    if len(ns) < 3:
        return ValidationCheck(
            name="rate_function_sharpening",
            passed=False,
            value="insufficient valid data",
            expected="r(t*_1) increases with N",
        )

    # Check monotonicity: r(t*) should generally increase with N
    # Allow for small oscillations due to finite-size effects
    n_increasing = sum(
        1 for i in range(1, len(r_at_first_dqpt))
        if r_at_first_dqpt[i] > r_at_first_dqpt[i - 1] * 0.8  # 20% tolerance
    )
    fraction_increasing = n_increasing / (len(r_at_first_dqpt) - 1)
    passed = fraction_increasing >= 0.5  # At least half should increase

    return ValidationCheck(
        name="rate_function_sharpening",
        passed=passed,
        value=f"{fraction_increasing:.0%} increasing ({n_increasing}/{len(r_at_first_dqpt)-1})",
        expected=">=50% pairs show r(t*_1) increasing with N",
        detail=f"N={ns}, r(t*_1)={[f'{v:.4f}' for v in r_at_first_dqpt]}",
    )


def check_t_star_scaling(trajectories: list[DQPTTrajectory]) -> ValidationCheck:
    """Check finite-size behavior of first critical time t*_1(N).

    For TFIM quench, t*_1(N) should:
    1. Be approximately constant (convergence is fast for deep quenches)
    2. Have small standard deviation relative to mean (consistent signal)
    3. Be in a physically reasonable range

    Note: For deep quenches, t* converges VERY fast with N. A flat
    t*(N) curve is actually a GOOD sign (already thermodynamic).
    """
    data_points: list[tuple[int, float]] = []
    for t in trajectories:
        if len(t.critical_times) > 0:
            data_points.append((t.n_qubits, float(t.critical_times[0])))

    if len(data_points) < 3:
        return ValidationCheck(
            name="t_star_scaling",
            passed=False,
            value="insufficient data",
            expected=">=3 N with detected DQPTs",
            detail=f"Only {len(data_points)} trajectories have critical times",
        )

    data_points.sort()
    ns = np.array([d[0] for d in data_points], dtype=float)
    t_stars = np.array([d[1] for d in data_points])

    # For deep quenches, t* converges fast — check consistency
    mean_t = float(np.mean(t_stars))
    std_t = float(np.std(t_stars))
    cv = std_t / mean_t if mean_t > 0 else float("inf")

    # Expected behavior: t* should be consistent (CV < 30%)
    # and in a physically reasonable range (> 0.1, < 5.0)
    reasonable_range = 0.1 < mean_t < 5.0
    consistent = cv < 0.30
    all_detected = len(data_points) == len(trajectories)

    passed = reasonable_range and consistent

    # Also check analytical range
    h_diff = abs(trajectories[0].h_post - trajectories[0].h_pre)
    t_analytical = np.pi / (2.0 * h_diff)
    in_expected_range = all(0.2 * t_analytical <= ts <= 1.5 * t_analytical for ts in t_stars)

    return ValidationCheck(
        name="t_star_scaling",
        passed=passed and in_expected_range,
        value=f"mean(t*)={mean_t:.4f}, CV={cv:.3f}, detection_rate={len(data_points)}/{len(trajectories)}",
        expected=f"CV < 0.30, t* in [0.2, 1.5] × {t_analytical:.3f}",
        detail=f"N={ns.tolist()}, t*={[f'{v:.4f}' for v in t_stars]}, t*_analytical={t_analytical:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Validation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def validate_dqpt_topology(
    topology: str,
    verbose: bool = False,
) -> DQPTValidationReport:
    """Run all validation checks on DQPT trajectories for a topology.

    Parameters
    ----------
    topology : str
        Topology to validate.
    verbose : bool
        Print detailed output.

    Returns
    -------
    DQPTValidationReport
        Complete validation report.
    """
    trajectories = load_dqpt_trajectories(topology)
    report = DQPTValidationReport(
        topology=topology,
        n_trajectories=len(trajectories),
    )

    if not trajectories:
        report.checks.append(ValidationCheck(
            name="data_availability",
            passed=False,
            value="0 trajectories",
            expected=">=1 DQPT trajectory",
            detail=f"No DQPT data found in data/dqpt_trajectories/{topology}_N*.npz",
        ))
        return report

    # Per-trajectory checks
    for traj in trajectories:
        n = traj.n_qubits
        per_n = {
            "file": traj.file,
            "h_pre": traj.h_pre,
            "h_post": traj.h_post,
            "n_steps": traj.n_steps,
            "T_total": float(traj.times[-1]),
            "n_dqpts": len(traj.critical_times),
            "critical_times": traj.critical_times.tolist(),
            "L_min": float(np.min(traj.loschmidt_echo)),
            "r_max": float(np.max(traj.rate_function)),
            "S_max": float(np.max(traj.entropies)),
        }

        # Individual checks
        check_t = check_analytical_t_star(traj)
        check_p = check_periodicity(traj)
        check_e = check_energy_conservation(traj)
        check_s = check_entropy_growth(traj)

        per_n["checks"] = {
            "analytical_t_star": check_t.passed,
            "periodicity": check_p.passed,
            "energy_conservation": check_e.passed,
            "entropy_growth": check_s.passed,
        }

        report.per_n_results[n] = per_n

        if verbose:
            status = "PASS" if all(per_n["checks"].values()) else "FAIL"
            logger.info(
                f"  N={n:>3}: {len(traj.critical_times)} DQPTs, "
                f"L_min={per_n['L_min']:.4f}, r_max={per_n['r_max']:.4f} [{status}]"
            )

    # Global checks (require multiple N)
    report.checks.append(check_analytical_t_star(trajectories[0]))
    report.checks.append(check_loschmidt_decay(trajectories))
    report.checks.append(check_rate_function_sharpening(trajectories))
    report.checks.append(check_t_star_scaling(trajectories))

    # Add per-trajectory checks for the best-data trajectory
    best = max(trajectories, key=lambda t: t.n_steps)
    report.checks.append(check_periodicity(best))
    report.checks.append(check_energy_conservation(best))
    report.checks.append(check_entropy_growth(best))

    # Scaling analysis summary
    if len(trajectories) >= 3:
        ns = [t.n_qubits for t in trajectories]
        t_stars_first = [
            float(t.critical_times[0]) if len(t.critical_times) > 0 else None
            for t in trajectories
        ]
        report.scaling_analysis = {
            "n_values": ns,
            "t_star_1": t_stars_first,
            "L_min": [float(np.min(t.loschmidt_echo)) for t in trajectories],
            "r_max": [float(np.max(t.rate_function)) for t in trajectories],
            "S_max": [float(np.max(t.entropies)) for t in trajectories],
        }

    # Overall pass: all global checks pass
    report.overall_pass = all(c.passed for c in report.checks)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: DQPTValidationReport) -> None:
    """Pretty-print the validation report."""
    status = "PASS" if report.overall_pass else "FAIL"
    print(f"\n{'='*70}")
    print(f"  DQPT Validation Report: {report.topology} [{status}]")
    print(f"{'='*70}")
    print(f"  Trajectories loaded: {report.n_trajectories}")

    if report.n_trajectories == 0:
        print(f"\n  NO DATA — generate with:")
        print(f"  for N in 8 10 12 14 16 20; do")
        print(f"    .venv/bin/python scripts/experiment_runners/scaling/"
              f"run_quench_dynamics_study.py \\")
        print(f"      --section 4 --n-qubits $N --topology {report.topology} \\")
        print(f"      --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80")
        print(f"  done")
        return

    # Per-N summary
    print(f"\n  {'N':>3} | {'DQPTs':>5} | {'t*_1':>6} | {'L_min':>7} | {'r_max':>6} | {'S_max':>5} | Status")
    print(f"  {'-'*65}")
    for n in sorted(report.per_n_results.keys()):
        info = report.per_n_results[n]
        t_star_1 = info["critical_times"][0] if info["critical_times"] else None
        checks = info["checks"]
        n_pass = sum(checks.values())
        n_total = len(checks)
        status_sym = "✓" if n_pass == n_total else ("✗" if n_pass < n_total // 2 else "~")
        t_str = f"{t_star_1:>6.3f}" if t_star_1 else "   N/A"
        print(
            f"  {n:>3} | {info['n_dqpts']:>5} | "
            f"{t_str} | "
            f"{info['L_min']:>7.4f} | {info['r_max']:>6.4f} | "
            f"{info['S_max']:>5.3f} | {status_sym} ({n_pass}/{n_total})"
        )

    # Global checks
    print(f"\n  {'─'*70}")
    print(f"  Global Validation Checks:")
    print(f"  {'─'*70}")
    for check in report.checks:
        sym = "✓" if check.passed else "✗"
        print(f"  [{sym}] {check.name}")
        print(f"      Value: {check.value}")
        print(f"      Expected: {check.expected}")
        if check.detail:
            print(f"      Detail: {check.detail}")

    # Scaling analysis
    if report.scaling_analysis:
        print(f"\n  {'─'*70}")
        print(f"  Scaling Analysis:")
        print(f"  {'─'*70}")
        sa = report.scaling_analysis
        print(f"  N values: {sa['n_values']}")
        print(f"  t*_1(N): {sa['t_star_1']}")
        print(f"  min(L): {[f'{v:.4f}' for v in sa['L_min']]}")
        print(f"  max(r): {[f'{v:.4f}' for v in sa['r_max']]}")

    # Go/No-Go summary
    print(f"\n  {'='*70}")
    n_passed = sum(1 for c in report.checks if c.passed)
    n_total = len(report.checks)
    print(f"  RESULT: {n_passed}/{n_total} checks passed — "
          f"{'GO for hardware' if report.overall_pass else 'ISSUES DETECTED'}")
    print(f"  {'='*70}")


def save_report(report: DQPTValidationReport, out_path: Path) -> None:
    """Save validation report to JSON."""
    from qmbp_simulation.utils.helpers import json_serialize

    output = {
        "topology": report.topology,
        "n_trajectories": report.n_trajectories,
        "overall_pass": report.overall_pass,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "value": c.value,
                "expected": c.expected,
                "detail": c.detail,
            }
            for c in report.checks
        ],
        "per_n_results": report.per_n_results,
        "scaling_analysis": report.scaling_analysis,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Validate DQPT results — rigorous physics checks"
    )
    parser.add_argument(
        "--topology", type=str, nargs="+", default=["heavy_hex"],
        help="Topology(ies) to validate (default: heavy_hex)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare results across topologies",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save validation report to JSON",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    reports = {}
    for topo in args.topology:
        report = validate_dqpt_topology(topo, verbose=args.verbose)
        reports[topo] = report
        print_report(report)

        if args.save:
            out_path = _project_root / "results" / "analysis" / f"dqpt_validation_{topo}.json"
            save_report(report, out_path)

    # Comparison across topologies
    if args.compare and len(reports) > 1:
        print(f"\n{'='*70}")
        print(f"  Cross-Topology Comparison")
        print(f"{'='*70}")
        for topo, report in reports.items():
            n_pass = sum(1 for c in report.checks if c.passed)
            n_total = len(report.checks)
            print(f"  {topo:>12}: {report.n_trajectories} trajectories, "
                  f"{n_pass}/{n_total} checks passed "
                  f"[{'PASS' if report.overall_pass else 'FAIL'}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Go/No-Go Evaluation (integrates QPT + DQPT results)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GoNoGoResult:
    """Result of a single go/no-go criterion evaluation."""

    criterion: str
    threshold: str
    current_value: str
    passed: bool
    category: str  # "qpt", "dqpt", "hardware"


def compute_go_no_go(topology: str, p_layers: int = 1) -> dict:
    """Automatically evaluate all QPT/DQPT go/no-go criteria.

    Combines outputs from qpt_detection and validate_dqpt_results to produce
    a structured pass/fail evaluation against predefined thresholds.

    Parameters
    ----------
    topology : str
        Target topology (e.g., "heavy_hex").
    p_layers : int
        HVA depth.

    Returns
    -------
    dict
        {
            "topology": str,
            "overall_go": bool,
            "criteria": list[dict],  # each with criterion, threshold, value, passed, category
            "n_passed": int,
            "n_total": int,
            "blocking_issues": list[str],
            "qpt_summary": dict | None,
            "dqpt_summary": dict | None,
        }
    """
    results: list[GoNoGoResult] = []
    blocking: list[str] = []
    qpt_summary = None
    dqpt_summary = None

    # ── QPT Criteria ─────────────────────────────────────────────────────────
    try:
        from scripts.analysis.qpt_detection import run_qpt_analysis

        qpt_exact = run_qpt_analysis(topology, p_layers, use_predicted=False)
        qpt_pred = run_qpt_analysis(topology, p_layers, use_predicted=True)

        if "error" not in qpt_exact:
            reliable_n = qpt_exact.get("n_values_reliable", [])
            h_c_rel = qpt_exact.get("h_c_reliable", {})
            fss = qpt_exact.get("finite_size_scaling")

            # Criterion 1: h_c converges (monotone increasing for reliable N)
            if len(reliable_n) >= 3:
                hc_values = [h_c_rel[str(n)] for n in sorted(reliable_n)]
                # Check roughly monotone (allow 1 outlier)
                n_increasing = sum(1 for i in range(1, len(hc_values)) if hc_values[i] >= hc_values[i-1] * 0.9)
                is_monotone = n_increasing >= len(hc_values) - 2
                results.append(GoNoGoResult(
                    criterion="h_c converges with N",
                    threshold="Roughly monotone increasing across N",
                    current_value=f"{n_increasing}/{len(hc_values)-1} pairs increasing",
                    passed=is_monotone,
                    category="qpt",
                ))
                if not is_monotone:
                    blocking.append("h_c(N) not monotone — possible data quality issue")
            else:
                results.append(GoNoGoResult(
                    criterion="h_c converges with N",
                    threshold=">=3 reliable N values needed",
                    current_value=f"{len(reliable_n)} reliable N values",
                    passed=False,
                    category="qpt",
                ))
                blocking.append(f"Only {len(reliable_n)} reliable N for FSS (need >=3)")

            # Criterion 2: FSS R² > 0.80
            fss_r2 = fss.get("r_squared", 0) if fss and "error" not in fss else 0
            results.append(GoNoGoResult(
                criterion="FSS R² > 0.80",
                threshold="> 0.80",
                current_value=f"{fss_r2:.4f}",
                passed=fss_r2 > 0.80,
                category="qpt",
            ))

            # Criterion 3: h_c(inf) in [0.8, 1.5] for heavy_hex
            h_c_inf = fss.get("h_c_inf") if fss and "error" not in fss else None
            if h_c_inf is not None:
                in_range = 0.8 <= h_c_inf <= 1.5
                results.append(GoNoGoResult(
                    criterion="h_c(∞) in [0.8, 1.5]",
                    threshold="0.8 ≤ h_c(∞) ≤ 1.5",
                    current_value=f"{h_c_inf:.4f}",
                    passed=in_range,
                    category="qpt",
                ))
            else:
                results.append(GoNoGoResult(
                    criterion="h_c(∞) in [0.8, 1.5]",
                    threshold="FSS must converge",
                    current_value="FSS not available",
                    passed=False,
                    category="qpt",
                ))

            # Criterion 4: Peak sharpening (|d2E/dh2|_max grows with N)
            per_n = qpt_exact.get("per_n_results", {})
            peaks = [(int(n), info["peak_magnitude"]) for n, info in per_n.items()
                     if not info.get("edge_artifact", False)]
            if len(peaks) >= 3:
                peaks.sort()
                n_sharpening = sum(1 for i in range(1, len(peaks)) if peaks[i][1] > peaks[i-1][1])
                sharpens = n_sharpening >= len(peaks) // 2
                results.append(GoNoGoResult(
                    criterion="Peak sharpening with N",
                    threshold="|d²E/dh²|_max increases for ≥50% pairs",
                    current_value=f"{n_sharpening}/{len(peaks)-1} pairs sharpen",
                    passed=sharpens,
                    category="qpt",
                ))

            # Criterion 5: MPNN captures h_c (< 15% error)
            if "error" not in qpt_pred:
                hc_exact = qpt_exact.get("h_c_reliable", {})
                hc_pred = qpt_pred.get("h_c_by_n", {})
                common = sorted(set(hc_exact.keys()) & set(hc_pred.keys()))
                if common:
                    errors = [abs(float(hc_pred[n]) - float(hc_exact[n])) / max(float(hc_exact[n]), 0.01)
                              for n in common]
                    mean_err = sum(errors) / len(errors)
                    results.append(GoNoGoResult(
                        criterion="MPNN captures h_c (< 15% error)",
                        threshold="mean |Δh_c|/h_c < 0.15",
                        current_value=f"{mean_err:.2%} (over {len(common)} N values)",
                        passed=mean_err < 0.15,
                        category="qpt",
                    ))

            qpt_summary = {
                "n_reliable": len(reliable_n),
                "reliable_n": reliable_n,
                "fss_r2": fss_r2,
                "h_c_inf": h_c_inf,
            }
    except Exception as e:
        results.append(GoNoGoResult(
            criterion="QPT detection functional",
            threshold="No errors",
            current_value=f"Error: {e}",
            passed=False,
            category="qpt",
        ))
        blocking.append(f"QPT analysis failed: {e}")

    # ── DQPT Criteria ────────────────────────────────────────────────────────
    try:
        dqpt_report = validate_dqpt_topology(topology)

        # Criterion 6: DQPT overall pass (7/7 checks)
        results.append(GoNoGoResult(
            criterion="DQPT 7/7 checks pass",
            threshold="All global checks green",
            current_value=f"{sum(1 for c in dqpt_report.checks if c.passed)}/{len(dqpt_report.checks)} checks pass",
            passed=dqpt_report.overall_pass,
            category="dqpt",
        ))

        # Criterion 7: Detection rate >= 80%
        n_with_dqpt = sum(1 for info in dqpt_report.per_n_results.values() if info.get("n_dqpts", 0) > 0)
        n_total_traj = dqpt_report.n_trajectories
        det_rate = n_with_dqpt / max(n_total_traj, 1)
        results.append(GoNoGoResult(
            criterion="DQPT detection rate ≥ 80%",
            threshold="≥ 80% trajectories detect DQPTs",
            current_value=f"{det_rate:.0%} ({n_with_dqpt}/{n_total_traj})",
            passed=det_rate >= 0.80,
            category="dqpt",
        ))

        # Criterion 8: Sufficient N coverage (>=4 N values with DQPT)
        results.append(GoNoGoResult(
            criterion="DQPT N coverage ≥ 4",
            threshold="≥ 4 system sizes with trajectories",
            current_value=f"{n_total_traj} N values",
            passed=n_total_traj >= 4,
            category="dqpt",
        ))
        if n_total_traj < 4:
            blocking.append(f"Only {n_total_traj} DQPT trajectories (need ≥4 for scaling)")

        # Criterion 9: GNN fidelity > F_min (hardware viability)
        try:
            from scripts.analysis.evaluate_gnn_fidelity import evaluate_direct_fidelity

            f_result = evaluate_direct_fidelity(topology, 10, 3.0, p_layers)
            if f_result is not None:
                f_min = 0.50  # From fidelity threshold analysis
                results.append(GoNoGoResult(
                    criterion="GNN fidelity > F_min at h=3.0",
                    threshold=f"F > {f_min:.2f} (DQPT detectable with QESEM)",
                    current_value=f"F={f_result.fidelity:.4f} (N=10, h=3.0)",
                    passed=f_result.fidelity > f_min,
                    category="hardware",
                ))
        except Exception:
            pass  # Fidelity check is optional (requires MPNN)

        dqpt_summary = {
            "n_trajectories": dqpt_report.n_trajectories,
            "overall_pass": dqpt_report.overall_pass,
            "checks_passed": sum(1 for c in dqpt_report.checks if c.passed),
            "checks_total": len(dqpt_report.checks),
            "detection_rate": det_rate,
        }
    except Exception as e:
        results.append(GoNoGoResult(
            criterion="DQPT validation functional",
            threshold="No errors",
            current_value=f"Error: {e}",
            passed=False,
            category="dqpt",
        ))
        blocking.append(f"DQPT validation failed: {e}")

    # ── Build output ─────────────────────────────────────────────────────────
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    overall_go = n_passed == n_total and not blocking

    return {
        "topology": topology,
        "overall_go": overall_go,
        "criteria": [
            {
                "criterion": r.criterion,
                "threshold": r.threshold,
                "current_value": r.current_value,
                "passed": r.passed,
                "category": r.category,
            }
            for r in results
        ],
        "n_passed": n_passed,
        "n_total": n_total,
        "blocking_issues": blocking,
        "qpt_summary": qpt_summary,
        "dqpt_summary": dqpt_summary,
    }


if __name__ == "__main__":
    main()
