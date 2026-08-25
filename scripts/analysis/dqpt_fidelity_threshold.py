#!/usr/bin/env python
"""DQPT Fidelity Threshold Analysis.

Determines the minimum state preparation fidelity F_min required for
DQPTs to remain detectable after quench dynamics. This is the key
go/no-go criterion for hardware: if F(GNN) > F_min, the QPU experiment
will produce observable DQPTs.

Method:
    For each fidelity F in [1.0, 0.95, ..., 0.30]:
    1. Construct |ψ_approx> = sqrt(F)|ψ_0> + sqrt(1-F)|ψ_perp>
       where |ψ_perp> is orthogonal to |ψ_0> (first excited state).
    2. Evolve under H(h_post) for n_steps.
    3. Compute Loschmidt echo L(t) = |<ψ_approx|ψ(t)>|^2
    4. Measure: min(L(t)), t*_shift, r_peak amplitude.
    5. Determine F_min = lowest F where r_peak > detectability threshold.

The detectability threshold is 0.02 (QESEM precision from arXiv:2608.05202).

Usage:
    python scripts/analysis/dqpt_fidelity_threshold.py \
        --topology heavy_hex --n-qubits 10 \
        --fidelities 1.0 0.95 0.90 0.85 0.80 0.70 0.50 0.30 \
        --h-pre 3.0 --h-post 0.5 --dt 0.05 --steps 60 --save

    python scripts/analysis/dqpt_fidelity_threshold.py \
        --topology chain_1d --n-qubits 10 --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

# QESEM detectability threshold (from IBM+Qedma benchmark arXiv:2608.05202)
QESEM_PRECISION = 0.02


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FidelityScanResult:
    """Result of DQPT analysis at a single fidelity value."""

    fidelity: float
    n_dqpts: int
    critical_times: list[float]
    L_min: float
    r_peak: float
    t_star_1: float | None  # First critical time (None if no DQPT)
    t_star_shift: float | None  # |t*(F) - t*(F=1)| / t*(F=1)
    detectable: bool  # r_peak > QESEM_PRECISION


@dataclass
class FidelityThresholdReport:
    """Complete fidelity threshold analysis report."""

    topology: str
    n_qubits: int
    h_pre: float
    h_post: float
    dt: float
    n_steps: int
    results: list[FidelityScanResult] = field(default_factory=list)
    f_min: float | None = None  # Minimum fidelity for DQPT detection
    t_star_reference: float | None = None  # t* at F=1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Core Computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_ground_and_excited(
    topology: str, n_qubits: int, h: float, model: str = "tfim"
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute ground state and first excited state vectors.

    Also persists E_0 and gap to GroundTruthCache for reuse by other modules
    (QPT detection, extrapolation evaluation, etc.).

    Returns
    -------
    tuple
        (psi_0, psi_1, E_0, E_1) — ground state, first excited state,
        ground energy, first excited energy.
    """
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
    H_op = builder.build(lattice)

    if n_qubits <= 14:
        # Dense diagonalization for small systems
        H_dense = np.asarray(H_op.to_matrix())
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
        psi_0 = eigenvectors[:, 0]
        psi_1 = eigenvectors[:, 1]
        E_0, E_1 = eigenvalues[0], eigenvalues[1]
    else:
        # Sparse: get 2 lowest eigenstates
        H_sparse = H_op.to_matrix(sparse=True)
        eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which="SA")
        # eigsh returns in ascending order for SA
        idx = np.argsort(eigenvalues)
        psi_0 = eigenvectors[:, idx[0]]
        psi_1 = eigenvectors[:, idx[1]]
        E_0, E_1 = eigenvalues[idx[0]], eigenvalues[idx[1]]

    psi_0 /= np.linalg.norm(psi_0)
    psi_1 /= np.linalg.norm(psi_1)

    # Persist to GroundTruthCache (crash-safe, immediate flush)
    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        existing = gt_cache.get(topology, n_qubits, model, h)
        if existing is None:
            gt_cache.put(
                topology=topology,
                n_qubits=n_qubits,
                model=model,
                h=h,
                energy=float(E_0),
                gap=float(E_1 - E_0),
                method="eigsh_k2" if n_qubits > 14 else "eigh_dense",
            )
            gt_cache.flush()
    except Exception:
        pass  # GT persistence is best-effort

    return psi_0, psi_1, float(E_0), float(E_1)


def construct_approximate_state(
    psi_0: np.ndarray, psi_perp: np.ndarray, fidelity: float
) -> np.ndarray:
    """Construct |ψ_approx> = sqrt(F)|ψ_0> + sqrt(1-F)|ψ_perp>.

    Parameters
    ----------
    psi_0 : np.ndarray
        Target ground state.
    psi_perp : np.ndarray
        Orthogonal state (first excited state).
    fidelity : float
        Target fidelity F = |<ψ_0|ψ_approx>|^2.

    Returns
    -------
    np.ndarray
        Normalized approximate state with fidelity F to psi_0.
    """
    if fidelity >= 1.0:
        return psi_0.copy()
    if fidelity <= 0.0:
        return psi_perp.copy()

    psi_approx = np.sqrt(fidelity) * psi_0 + np.sqrt(1.0 - fidelity) * psi_perp
    psi_approx /= np.linalg.norm(psi_approx)
    return psi_approx


def evolve_and_measure_dqpt(
    psi_init: np.ndarray,
    topology: str,
    n_qubits: int,
    h_post: float,
    dt: float,
    n_steps: int,
    *,
    H_post_op=None,
) -> dict:
    """Evolve initial state under H(h_post) and measure DQPT observables.

    Parameters
    ----------
    psi_init : np.ndarray
        Initial state vector.
    topology : str
        Lattice topology.
    n_qubits : int
        System size.
    h_post : float
        Post-quench field.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    H_post_op : SparsePauliOp | None
        Pre-built Hamiltonian operator (avoids rebuilding if caller already has it).

    Returns
    -------
    dict
        {times, loschmidt_echo, rate_function, critical_times, L_min, r_peak}
    """
    from qmbp_simulation.analysis.observables import (
        detect_dqpt_critical_times,
        loschmidt_echo,
        rate_function,
    )

    if H_post_op is None:
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice

        builder = HamiltonianBuilder()
        lattice_post = make_lattice(topology, n_qubits, J=1.0, h=h_post)
        H_post_op = builder.build(lattice_post)

    _DENSE_LIMIT = 14

    if n_qubits <= _DENSE_LIMIT:
        H_post = np.asarray(H_post_op.to_matrix())
        U_dt = expm(-1j * H_post * dt)
        use_sparse = False
    else:
        H_post_sparse = H_post_op.to_matrix(sparse=True)
        use_sparse = True

    psi_t = psi_init.copy().astype(complex)
    times = [0.0]
    loschmidt_values = [1.0]
    rate_values = [0.0]

    for step in range(1, n_steps + 1):
        if use_sparse:
            psi_t = expm_multiply(-1j * H_post_sparse * dt, psi_t)
        else:
            psi_t = U_dt @ psi_t
        psi_t /= np.linalg.norm(psi_t)

        t = step * dt
        times.append(t)

        L_t = loschmidt_echo(psi_init, psi_t)
        loschmidt_values.append(L_t)
        rate_values.append(rate_function(L_t, n_qubits))

    # Detect DQPTs
    critical_times = detect_dqpt_critical_times(times, loschmidt_values, threshold=0.1)

    # Peak rate function value
    r_arr = np.array(rate_values)
    r_peak = float(np.max(r_arr))

    return {
        "times": times,
        "loschmidt_echo": loschmidt_values,
        "rate_function": rate_values,
        "critical_times": critical_times,
        "L_min": float(min(loschmidt_values)),
        "r_peak": r_peak,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_fidelity_threshold_scan(
    topology: str,
    n_qubits: int,
    fidelities: list[float],
    h_pre: float = 3.0,
    h_post: float = 0.5,
    dt: float = 0.05,
    n_steps: int = 60,
    *,
    detectability_threshold: float = QESEM_PRECISION,
) -> FidelityThresholdReport:
    """Run complete fidelity threshold scan.

    Parameters
    ----------
    topology : str
        Lattice topology.
    n_qubits : int
        System size (ED-accessible: ≤22).
    fidelities : list[float]
        Fidelity values to scan (should include 1.0 for reference).
    h_pre : float
        Pre-quench field (ground state preparation point).
    h_post : float
        Post-quench field (evolution Hamiltonian).
    dt : float
        Time step.
    n_steps : int
        Number of evolution steps.
    detectability_threshold : float
        Minimum r_peak for DQPT to be considered detectable.

    Returns
    -------
    FidelityThresholdReport
        Complete analysis with F_min determination.
    """
    report = FidelityThresholdReport(
        topology=topology,
        n_qubits=n_qubits,
        h_pre=h_pre,
        h_post=h_post,
        dt=dt,
        n_steps=n_steps,
    )

    # Get ground state + excited state at h_pre
    logger.info(f"  Computing ground + excited states (N={n_qubits}, h={h_pre})...")
    psi_0, psi_1, E_0, E_1 = compute_ground_and_excited(topology, n_qubits, h_pre)
    gap = E_1 - E_0
    logger.info(f"  E_0={E_0:.6f}, gap={gap:.6f}")

    # Pre-build post-quench Hamiltonian ONCE (reused across all F scans)
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    lattice_post = make_lattice(topology, n_qubits, J=1.0, h=h_post)
    H_post_op = builder.build(lattice_post)

    # Also persist GT for h_post (free since we built the H)
    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache
        from qmbp_simulation.solvers.classical import ClassicalSolver

        gt_cache = GroundTruthCache()
        if gt_cache.get(topology, n_qubits, "tfim", h_post) is None:
            solver = ClassicalSolver()
            gt_post = solver.solve(H_post_op, lattice_post)
            gt_cache.put(
                topology=topology, n_qubits=n_qubits, model="tfim", h=h_post,
                energy=gt_post.ground_energy, gap=gt_post.gap, method="fidelity_threshold_scan",
            )
            gt_cache.flush()
    except Exception:
        pass

    # Reference: F=1.0 (exact ground state)
    t_star_ref = None

    # ── Checkpoint resume (crash-safe: skip already-computed fidelities) ─────
    checkpoint_dir = _project_root / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cp_path = checkpoint_dir / f"fidelity_threshold_{topology}_N{n_qubits}.json"
    completed_fidelities: dict[str, dict] = {}

    if cp_path.exists():
        try:
            with open(cp_path) as f:
                cp_data = json.load(f)
            completed_fidelities = cp_data.get("results_by_f", {})
            t_star_ref = cp_data.get("t_star_ref")
            logger.info(f"  Resumed checkpoint: {len(completed_fidelities)} fidelities already done")
        except Exception:
            completed_fidelities = {}

    # Scan fidelities
    fidelities_sorted = sorted(fidelities, reverse=True)
    for F in fidelities_sorted:
        f_key = f"{F:.4f}"

        # Skip if already completed (from checkpoint)
        if f_key in completed_fidelities:
            cached = completed_fidelities[f_key]
            scan_result = FidelityScanResult(
                fidelity=F,
                n_dqpts=cached["n_dqpts"],
                critical_times=cached["critical_times"],
                L_min=cached["L_min"],
                r_peak=cached["r_peak"],
                t_star_1=cached["t_star_1"],
                t_star_shift=cached["t_star_shift"],
                detectable=cached["detectable"],
            )
            report.results.append(scan_result)
            if t_star_ref is None and (F == 1.0 or F >= 0.99):
                t_star_ref = cached["t_star_1"]
                report.t_star_reference = t_star_ref
            logger.info(f"  F={F:.2f}: (from checkpoint) r_peak={cached['r_peak']:.4f}")
            continue

        psi_approx = construct_approximate_state(psi_0, psi_1, F)

        # Verify fidelity
        actual_F = float(np.abs(np.vdot(psi_0, psi_approx)) ** 2)
        assert abs(actual_F - F) < 1e-10, f"Fidelity mismatch: {actual_F} != {F}"

        # Evolve and measure (H_post_op built once, passed in)
        result = evolve_and_measure_dqpt(
            psi_approx, topology, n_qubits, h_post, dt, n_steps,
            H_post_op=H_post_op,
        )

        # t* shift relative to F=1.0
        t_star_1 = result["critical_times"][0] if result["critical_times"] else None
        if F == 1.0 or (F >= 0.99 and t_star_ref is None):
            t_star_ref = t_star_1
            report.t_star_reference = t_star_ref

        t_star_shift = None
        if t_star_1 is not None and t_star_ref is not None and t_star_ref > 0:
            t_star_shift = abs(t_star_1 - t_star_ref) / t_star_ref

        scan_result = FidelityScanResult(
            fidelity=F,
            n_dqpts=len(result["critical_times"]),
            critical_times=result["critical_times"],
            L_min=result["L_min"],
            r_peak=result["r_peak"],
            t_star_1=t_star_1,
            t_star_shift=t_star_shift,
            detectable=result["r_peak"] > detectability_threshold,
        )
        report.results.append(scan_result)

        # ── Immediate checkpoint save (crash-safe) ───────────────────────
        completed_fidelities[f_key] = {
            "fidelity": F,
            "n_dqpts": scan_result.n_dqpts,
            "critical_times": scan_result.critical_times,
            "L_min": scan_result.L_min,
            "r_peak": scan_result.r_peak,
            "t_star_1": scan_result.t_star_1,
            "t_star_shift": scan_result.t_star_shift,
            "detectable": scan_result.detectable,
        }
        try:
            with open(cp_path, "w") as f:
                json.dump({"results_by_f": completed_fidelities, "t_star_ref": t_star_ref}, f)
        except Exception:
            pass  # Checkpoint save is best-effort

        logger.info(
            f"  F={F:.2f}: DQPTs={scan_result.n_dqpts}, "
            f"r_peak={scan_result.r_peak:.4f}, "
            f"L_min={scan_result.L_min:.4f}, "
            f"detectable={'✓' if scan_result.detectable else '✗'}"
        )

    # Determine F_min: lowest F where DQPTs are still detectable
    detectable_results = [r for r in report.results if r.detectable]
    if detectable_results:
        report.f_min = min(r.fidelity for r in detectable_results)
    else:
        report.f_min = None  # No fidelity produces detectable DQPTs (bad quench params)

    # Clean up checkpoint on successful completion
    try:
        if cp_path.exists():
            cp_path.unlink()
    except Exception:
        pass

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: FidelityThresholdReport) -> None:
    """Pretty-print the fidelity threshold report."""
    print(f"\n{'='*70}")
    print(f"  DQPT Fidelity Threshold: {report.topology} N={report.n_qubits}")
    print(f"  Quench: h={report.h_pre} → {report.h_post}, T={report.n_steps * report.dt:.1f}")
    print(f"{'='*70}")

    print(f"\n  {'F':>5} | {'DQPTs':>5} | {'r_peak':>7} | {'L_min':>7} | {'t*_1':>6} | {'shift':>6} | Det?")
    print(f"  {'-'*60}")
    for r in report.results:
        t_str = f"{r.t_star_1:.3f}" if r.t_star_1 else "  N/A"
        shift_str = f"{r.t_star_shift:.2%}" if r.t_star_shift is not None else "  ref"
        det_str = "✓" if r.detectable else "✗"
        print(
            f"  {r.fidelity:>5.2f} | {r.n_dqpts:>5} | "
            f"{r.r_peak:>7.4f} | {r.L_min:>7.4f} | "
            f"{t_str} | {shift_str} | {det_str}"
        )

    print(f"\n  {'─'*60}")
    if report.f_min is not None:
        print(f"  F_min = {report.f_min:.2f} (lowest F with r_peak > {QESEM_PRECISION})")
        print(f"  → GNN fidelity must exceed {report.f_min:.2f} for hardware DQPT detection")
    else:
        print(f"  ⚠️ No detectable DQPTs at any fidelity — check quench parameters")
    print(f"  t*_reference (F=1.0) = {report.t_star_reference}")


def save_report(report: FidelityThresholdReport, out_path: Path) -> None:
    """Save report to JSON."""
    from qmbp_simulation.utils.helpers import json_serialize

    output = {
        "topology": report.topology,
        "n_qubits": report.n_qubits,
        "h_pre": report.h_pre,
        "h_post": report.h_post,
        "dt": report.dt,
        "n_steps": report.n_steps,
        "f_min": report.f_min,
        "t_star_reference": report.t_star_reference,
        "detectability_threshold": QESEM_PRECISION,
        "results": [
            {
                "fidelity": r.fidelity,
                "n_dqpts": r.n_dqpts,
                "critical_times": r.critical_times,
                "L_min": r.L_min,
                "r_peak": r.r_peak,
                "t_star_1": r.t_star_1,
                "t_star_shift": r.t_star_shift,
                "detectable": r.detectable,
            }
            for r in report.results
        ],
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
        description="DQPT Fidelity Threshold — determine minimum F for hardware"
    )
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument(
        "--fidelities", type=float, nargs="+",
        default=[1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30],
    )
    parser.add_argument("--h-pre", type=float, default=3.0)
    parser.add_argument("--h-post", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    report = run_fidelity_threshold_scan(
        topology=args.topology,
        n_qubits=args.n_qubits,
        fidelities=args.fidelities,
        h_pre=args.h_pre,
        h_post=args.h_post,
        dt=args.dt,
        n_steps=args.steps,
    )

    print_report(report)

    if args.save:
        out_path = (
            _project_root / "results" / "analysis"
            / f"dqpt_fidelity_threshold_{args.topology}_N{args.n_qubits}.json"
        )
        save_report(report, out_path)


if __name__ == "__main__":
    main()
