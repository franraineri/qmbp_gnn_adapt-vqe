#!/usr/bin/env python3
"""Frustrated TFIM Full Execution Suite.

Runs VQE experiments across multiple configurations to validate the
frustrated TFIM extension under realistic pipeline conditions.

Sections:
1. Expressibility sweep: fidelity vs J₂ at fixed h (3 seeds)
2. Descending h-sweep at J₂=0.3 (warm-start viability)
3. Cross-topology: chain + ladder at J₂=0.3
4. Phase diagram slice: varying J₂ and h simultaneously
5. Comparison with standard TFIM at same parameters (g=0 baseline)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("frustrated_tfim")
logger.setLevel(logging.INFO)

from datetime import UTC

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector, state_fidelity

from qmbp_simulation import HamiltonianBuilder, make_lattice
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend

builder = HamiltonianBuilder()
hva = HVACircuitBuilder()
backend = NoiselessBackend()
estimator = StatevectorEstimator()


def _exact_gs(H):
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    evals, evecs = np.linalg.eigh(mat)
    return evals[0], evals[1] - evals[0], evecs[:, 0]


def _vqe(circuit, H, prev_theta, rng, n_restarts=5, maxiter=500):
    best_e = float("inf")
    best_t = prev_theta.copy()
    total_evals = 0
    for r in range(n_restarts):
        x0 = prev_theta + rng.normal(0, 0.1, len(prev_theta)) if r > 0 else prev_theta.copy()
        x0 = np.clip(x0, -np.pi, np.pi)
        res = minimize(
            lambda p, _H=H, _c=circuit: backend.evaluate(_c, _H, p),
            x0,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * len(prev_theta),
            options={"maxiter": maxiter, "ftol": 1e-14},
        )
        total_evals += res.nfev
        if res.fun < best_e:
            best_e = res.fun
            best_t = res.x.copy()
    return best_e, best_t, total_evals


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Expressibility vs J₂ (N=6, p=2, h=1.5, 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════


def section_1():
    logger.info("=" * 65)
    logger.info("SECTION 1: Expressibility vs J₂ (N=6, p=2, h=1.5)")
    logger.info("=" * 65)

    N, p, h = 6, 2, 1.5
    j2_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    seeds = [42, 43, 44]

    lattice = make_lattice("chain_1d", N, J=1.0, h=h)
    circuit, _ = hva.create_frustrated_tfim(N, p, lattice)
    n_params = circuit.num_parameters

    logger.info(f"  Config: N={N}, p={p}, h={h}, params={n_params}")
    logger.info(f"  Seeds: {seeds}, J₂ range: {j2_values}")
    logger.info("")
    logger.info(
        f"  {'J₂':>4} | {'Mean Fid':>8} | {'Std':>6} | {'Min Fid':>7} | {'Mean ΔE/gap':>11}"
    )
    logger.info(f"  {'-' * 4}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 11}")

    for j2 in j2_values:
        H = builder.build_frustrated_tfim(lattice, J2=j2)
        e_exact, gap, gs = _exact_gs(H)
        fids = []
        de_gaps = []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            energy, theta_opt, _ = _vqe(circuit, H, x0, rng, n_restarts=10, maxiter=500)
            sv = Statevector(circuit.assign_parameters(theta_opt))
            fid = float(state_fidelity(sv, Statevector(gs)))
            fids.append(fid)
            de_gaps.append(abs(energy - e_exact) / max(gap, 1e-10))

        logger.info(
            f"  {j2:>4.1f} | {np.mean(fids):>8.4f} | {np.std(fids):>6.4f} | "
            f"{np.min(fids):>7.4f} | {np.mean(de_gaps):>11.4f}"
        )

    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Descending h-sweep at J₂=0.3 (warm-start viability, 3 seeds)
# ═══════════════════════════════════════════════════════════════════════════════


def section_2():
    logger.info("=" * 65)
    logger.info("SECTION 2: Descending h-sweep (J₂=0.3, warm-start)")
    logger.info("=" * 65)

    N, p, J2 = 6, 2, 0.3
    h_sweep = [2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75]
    seeds = [42, 43, 44]

    lattice_ref = make_lattice("chain_1d", N, J=1.0, h=2.0)
    circuit, _ = hva.create_frustrated_tfim(N, p, lattice_ref)
    n_params = circuit.num_parameters

    logger.info(f"  Config: N={N}, p={p}, J₂={J2}, params={n_params}")
    logger.info(f"  h-sweep: {h_sweep} (descending)")
    logger.info("")
    logger.info(f"  {'h':>5} | {'Fidelity':>8} | {'ΔE/gap':>11} | {'θ-smooth':>9} | {'Pass':>4}")
    logger.info(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 11}-+-{'-' * 9}-+-{'-' * 4}")

    all_smoothness = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        for h in h_sweep:
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build_frustrated_tfim(lattice, J2=J2)
            e_exact, gap, gs = _exact_gs(H)

            energy, best_theta, _ = _vqe(circuit, H, prev_theta, rng, n_restarts=5)
            theta_change = float(np.max(np.abs(best_theta - prev_theta)))
            all_smoothness.append(theta_change)
            prev_theta = best_theta.copy()

            sv = Statevector(circuit.assign_parameters(best_theta))
            fid = float(state_fidelity(sv, Statevector(gs)))
            de_gap = abs(energy - e_exact) / max(gap, 1e-10)
            passed = "✓" if de_gap < 0.05 else "✗"

            if seed == 42:  # Print one seed for readability (all 3 are computed)
                logger.info(
                    f"  {h:>5.2f} | {fid:>8.4f} | {de_gap:>11.4f} | {theta_change:>9.4f} | {passed}"
                )

    max_smooth = max(all_smoothness) if all_smoothness else 0
    logger.info(f"\n  Max θ-smoothness across all seeds: {max_smooth:.4f}")
    logger.info(f"  Warm-start viable: {'✅ YES' if max_smooth < 1.0 else '❌ NO'}")
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Cross-topology (chain + ladder, J₂=0.3, N=6)
# ═══════════════════════════════════════════════════════════════════════════════


def section_3():
    logger.info("=" * 65)
    logger.info("SECTION 3: Cross-topology (chain_1d + ladder) at J₂=0.3")
    logger.info("=" * 65)

    N, p, J2 = 6, 2, 0.3
    h_values = [2.0, 1.5, 1.0]
    topologies = ["chain_1d", "ladder"]
    seeds = [42, 43, 44]

    logger.info(f"  Config: N={N}, p={p}, J₂={J2}")
    logger.info(f"  h-values: {h_values}, topologies: {topologies}")
    logger.info("")
    logger.info(
        f"  {'Topology':<10} | {'Mean Fid':>8} | {'Min Fid':>7} | {'Mean ΔE/gap':>11} | {'Points':>6}"
    )
    logger.info(f"  {'-' * 10}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 11}-+-{'-' * 6}")

    for topology in topologies:
        fids_all = []
        de_gaps_all = []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            lattice = make_lattice(topology, N, J=1.0, h=2.0)
            circuit, _ = hva.create_frustrated_tfim(N, p, lattice)
            n_params = circuit.num_parameters
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            for h in sorted(h_values, reverse=True):
                lat_h = make_lattice(topology, N, J=1.0, h=h)
                H = builder.build_frustrated_tfim(lat_h, J2=J2)
                e_exact, gap, gs = _exact_gs(H)

                energy, best_theta, _ = _vqe(circuit, H, prev_theta, rng, n_restarts=5)
                prev_theta = best_theta.copy()

                sv = Statevector(circuit.assign_parameters(best_theta))
                fid = float(state_fidelity(sv, Statevector(gs)))
                fids_all.append(fid)
                de_gaps_all.append(abs(energy - e_exact) / max(gap, 1e-10))

        logger.info(
            f"  {topology:<10} | {np.mean(fids_all):>8.4f} | {np.min(fids_all):>7.4f} | "
            f"{np.mean(de_gaps_all):>11.4f} | {len(fids_all):>6}"
        )

    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: 2D Phase Diagram Slice (varying h AND J₂)
# ═══════════════════════════════════════════════════════════════════════════════


def section_4():
    logger.info("=" * 65)
    logger.info("SECTION 4: 2D Phase Diagram (h × J₂ grid)")
    logger.info("=" * 65)

    N, p = 6, 2
    h_values = [2.0, 1.5, 1.0]
    j2_values = [0.0, 0.3, 0.5, 0.7]
    seed = 42

    lattice_ref = make_lattice("chain_1d", N, J=1.0, h=2.0)
    circuit, _ = hva.create_frustrated_tfim(N, p, lattice_ref)
    n_params = circuit.num_parameters
    rng = np.random.default_rng(seed)

    logger.info(
        f"  Grid: {len(h_values)} h × {len(j2_values)} J₂ = {len(h_values) * len(j2_values)} points"
    )
    logger.info("")
    logger.info(
        f"  {'h':>5} | {'J₂':>4} | {'Fidelity':>8} | {'ΔE/gap':>7} | {'Gap':>6} | {'Pass':>4}"
    )
    logger.info(f"  {'-' * 5}-+-{'-' * 4}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 4}")

    n_pass = 0
    n_total = 0

    for j2 in j2_values:
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        for h in sorted(h_values, reverse=True):
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build_frustrated_tfim(lattice, J2=j2)
            e_exact, gap, gs = _exact_gs(H)

            energy, best_theta, _ = _vqe(circuit, H, prev_theta, rng, n_restarts=5)
            prev_theta = best_theta.copy()

            sv = Statevector(circuit.assign_parameters(best_theta))
            fid = float(state_fidelity(sv, Statevector(gs)))
            de_gap = abs(energy - e_exact) / max(gap, 1e-10)
            passed = "✓" if de_gap < 0.05 else "✗"
            if de_gap < 0.05:
                n_pass += 1
            n_total += 1

            logger.info(
                f"  {h:>5.2f} | {j2:>4.1f} | {fid:>8.4f} | {de_gap:>7.4f} | {gap:>6.3f} | {passed}"
            )

    logger.info(f"\n  Pass rate: {n_pass}/{n_total} ({100 * n_pass / n_total:.0f}%)")
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Direct Comparison with Standard TFIM
# ═══════════════════════════════════════════════════════════════════════════════


def section_5():
    logger.info("=" * 65)
    logger.info("SECTION 5: Frustrated vs Standard TFIM (same pipeline)")
    logger.info("=" * 65)

    N, p = 6, 2
    h_values = [2.0, 1.5, 1.25, 1.0]
    seed = 42
    rng = np.random.default_rng(seed)

    lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)

    # Standard TFIM (2 params/layer)
    circuit_std, _ = hva.create(N, p, lattice)
    # Frustrated TFIM J2=0.3 (3 params/layer)
    circuit_frust, _ = hva.create_frustrated_tfim(N, p, lattice)

    logger.info(f"  Standard TFIM: {circuit_std.num_parameters} params")
    logger.info(f"  Frustrated J₂=0.3: {circuit_frust.num_parameters} params")
    logger.info("")
    logger.info(
        f"  {'h':>5} | {'Std Fid':>7} | {'Std ΔE/gap':>10} | {'Frust Fid':>9} | {'Frust ΔE/gap':>12}"
    )
    logger.info(f"  {'-' * 5}-+-{'-' * 7}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 12}")

    prev_std = rng.uniform(-0.01, 0.01, circuit_std.num_parameters)
    rng2 = np.random.default_rng(seed)
    prev_frust = rng2.uniform(-0.01, 0.01, circuit_frust.num_parameters)

    for h in sorted(h_values, reverse=True):
        lat_h = make_lattice("chain_1d", N, J=1.0, h=h)

        # Standard TFIM
        H_std = builder.build(lat_h)
        e_std_exact, gap_std, gs_std = _exact_gs(H_std)
        energy_std, prev_std, _ = _vqe(circuit_std, H_std, prev_std, rng, n_restarts=5)
        sv_std = Statevector(circuit_std.assign_parameters(prev_std))
        fid_std = float(state_fidelity(sv_std, Statevector(gs_std)))
        de_std = abs(energy_std - e_std_exact) / max(gap_std, 1e-10)

        # Frustrated TFIM J2=0.3
        H_frust = builder.build_frustrated_tfim(lat_h, J2=0.3)
        e_frust_exact, gap_frust, gs_frust = _exact_gs(H_frust)
        energy_frust, prev_frust, _ = _vqe(circuit_frust, H_frust, prev_frust, rng2, n_restarts=5)
        sv_frust = Statevector(circuit_frust.assign_parameters(prev_frust))
        fid_frust = float(state_fidelity(sv_frust, Statevector(gs_frust)))
        de_frust = abs(energy_frust - e_frust_exact) / max(gap_frust, 1e-10)

        logger.info(
            f"  {h:>5.2f} | {fid_std:>7.4f} | {de_std:>10.4f} | {fid_frust:>9.4f} | {de_frust:>12.4f}"
        )

    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Scaling N=4,6,10 at J₂=0.3 (valid regime boundary)
# ═══════════════════════════════════════════════════════════════════════════════


def section_6():
    logger.info("=" * 65)
    logger.info("SECTION 6: Scaling N=4,6,10 (J₂=0.3, p=2, boundary)")
    logger.info("=" * 65)

    J2, p = 0.3, 2
    N_values = [4, 6, 10]
    h_grid = np.arange(0.5, 2.55, 0.25)
    seed = 42

    logger.info(f"  Config: J₂={J2}, p={p}, h_grid step=0.25")
    logger.info("")
    logger.info(f"  {'N':>3} | {'h_min (5%)':>10} | {'Mean Fid (h≥1.5)':>17} | {'Pass Rate':>9}")
    logger.info(f"  {'-' * 3}-+-{'-' * 10}-+-{'-' * 17}-+-{'-' * 9}")

    results_scaling = []

    for N in N_values:
        rng = np.random.default_rng(seed)
        lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
        circuit, _ = hva.create_frustrated_tfim(N, p, lattice)
        n_params = circuit.num_parameters
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        h_min = None
        fids_high_h = []
        n_pass = 0
        n_total = 0

        for h in np.sort(h_grid)[::-1]:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=float(h))
            H = builder.build_frustrated_tfim(lat_h, J2=J2)
            e_exact, gap, gs = _exact_gs(H)

            energy, best_theta, _ = _vqe(circuit, H, prev_theta, rng, n_restarts=3)
            prev_theta = best_theta.copy()

            sv = Statevector(circuit.assign_parameters(best_theta))
            fid = float(state_fidelity(sv, Statevector(gs)))
            de_gap = abs(energy - e_exact) / max(gap, 1e-10)

            if de_gap < 0.05:
                n_pass += 1
            else:
                h_min = float(h)
            n_total += 1

            if h >= 1.5:
                fids_high_h.append(fid)

        mean_fid_high = np.mean(fids_high_h) if fids_high_h else 0
        pass_rate = n_pass / n_total if n_total > 0 else 0
        h_min_str = f"{h_min:.2f}" if h_min else "< 0.50"

        logger.info(f"  {N:>3} | {h_min_str:>10} | {mean_fid_high:>17.4f} | {pass_rate:>8.0%}")
        results_scaling.append(
            {"N": N, "h_min": h_min, "mean_fid": mean_fid_high, "pass_rate": pass_rate}
        )

    logger.info("")
    return results_scaling


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Seed Robustness (3 seeds × full sweep, check variance)
# ═══════════════════════════════════════════════════════════════════════════════


def section_7():
    logger.info("=" * 65)
    logger.info("SECTION 7: Seed Robustness (variance across seeds)")
    logger.info("=" * 65)

    N, p, J2 = 6, 2, 0.3
    h_values = [2.0, 1.75, 1.5, 1.25]
    seeds = [42, 43, 44]

    lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
    circuit, _ = hva.create_frustrated_tfim(N, p, lattice)
    n_params = circuit.num_parameters

    logger.info(f"  Config: N={N}, p={p}, J₂={J2}")
    logger.info(f"  h-values: {h_values}, seeds: {seeds}")
    logger.info("")
    logger.info(
        f"  {'h':>5} | {'Fid (42)':>8} | {'Fid (43)':>8} | {'Fid (44)':>8} | {'Std':>6} | {'Robust':>6}"
    )
    logger.info(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}")

    all_robust = True

    for h in sorted(h_values, reverse=True):
        fids = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build_frustrated_tfim(lat_h, J2=J2)
            _, _, gs = _exact_gs(H)

            _, best_theta, _ = _vqe(circuit, H, prev_theta, rng, n_restarts=5)
            sv = Statevector(circuit.assign_parameters(best_theta))
            fid = float(state_fidelity(sv, Statevector(gs)))
            fids.append(fid)

        std = np.std(fids)
        robust = "✅" if std < 0.01 else "⚠️"
        if std >= 0.01:
            all_robust = False

        logger.info(
            f"  {h:>5.2f} | {fids[0]:>8.4f} | {fids[1]:>8.4f} | {fids[2]:>8.4f} | "
            f"{std:>6.4f} | {robust}"
        )

    logger.info(f"\n  Seed-independent (std<0.01 all h): {'✅ YES' if all_robust else '⚠️ NO'}")
    logger.info("")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT SAVING
# ═══════════════════════════════════════════════════════════════════════════════


def save_results(all_results: dict, elapsed: float):
    """Save all execution results to JSON for reproducibility and digest integration."""
    import json
    from datetime import datetime

    output_dir = Path("results/experiments/exp_frustrated_tfim")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"run_{timestamp}.json"

    payload = {
        "config": {
            "experiment_id": "E4c",
            "category": "E",
            "model": "tfim_frustrated",
            "description": "Frustrated TFIM (J1-J2) validation suite",
            "hypothesis": "HVA p=2 with NNN RZZ achieves fid≥0.90 for J₂≤0.5",
            "system": {
                "n_qubits": 6,
                "p_layers": 2,
                "topology": "chain_1d",
                "model": "tfim_frustrated",
            },
            "seeds": [42, 43, 44],
        },
        "results": all_results,
        "elapsed_s": elapsed,
        "timestamp": timestamp,
        "environment": {
            "python": sys.version.split()[0],
        },
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"  Results saved to: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    t_start = time.time()

    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  FRUSTRATED TFIM (J1-J2) — FULL EXECUTION SUITE              ║")
    logger.info("║  H = -J₁·ZZ_nn + J₂·ZZ_nnn - h·X                            ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")
    logger.info("")

    all_results = {}

    section_1()
    all_results["section_1"] = "expressibility_vs_j2"

    section_2()
    all_results["section_2"] = "descending_h_sweep"

    section_3()
    all_results["section_3"] = "cross_topology"

    section_4()
    all_results["section_4"] = "phase_diagram_2d"

    section_5()
    all_results["section_5"] = "comparison_standard_tfim"

    scaling = section_6()
    all_results["section_6_scaling"] = scaling

    section_7()
    all_results["section_7"] = "seed_robustness"

    elapsed = time.time() - t_start

    logger.info("")
    logger.info("=" * 65)
    logger.info(f"TOTAL TIME: {elapsed:.1f}s")
    output = save_results(all_results, elapsed)
    logger.info("=" * 65)
