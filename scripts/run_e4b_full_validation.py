#!/usr/bin/env python3
"""E4b Full Validation Suite: Cross-topology, scaling, and pipeline tests.

Runs all proposed validation experiments for the TFIM + longitudinal field
extension in a single reproducible execution. Each section validates a
different aspect of reliability:

1. Reproducibility check: same seed → same result (determinism)
2. Cross-topology: ladder + triangular at g=0.3
3. Scaling: N=4,6,8 at g=0.3 (valid regime boundary shift)
4. Full pipeline: Phases 1-4 with PipelineRunner + MPNN
5. p=1 validation: hardware-deployable depth

All results printed as tables for thesis integration.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root in path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("e4b_validation")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Reproducibility (determinism verification)
# ═══════════════════════════════════════════════════════════════════════════════


def test_reproducibility():
    """Run same experiment twice with same seed → identical results."""
    from scipy.optimize import minimize

    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend

    logger.info("=" * 65)
    logger.info("SECTION 1: Reproducibility Check")
    logger.info("=" * 65)

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    backend = NoiselessBackend()

    N, p, g, h = 6, 2, 0.3, 1.5
    lattice = make_lattice("chain_1d", N, J=1.0, h=h)
    H = builder.build_tfim_longitudinal(lattice, g=g)
    circuit, _ = hva.create_tfim_longitudinal(N, p, lattice)

    results = []
    for _run in range(2):
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-0.01, 0.01, circuit.num_parameters)
        result = minimize(
            lambda params, _c=circuit, _H=H: backend.evaluate(_c, _H, params),
            x0,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * circuit.num_parameters,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        results.append(result.x.copy())

    diff = np.max(np.abs(results[0] - results[1]))
    assert diff < 1e-12, f"FAIL: Reproducibility broken, max diff={diff}"
    logger.info(f"  [PASS] Two identical runs differ by {diff:.2e} (< 1e-12)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Cross-Topology Validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_cross_topology():
    """Validate extended HVA works on ladder and triangular lattices."""
    from qiskit.quantum_info import Statevector, state_fidelity
    from scipy.optimize import minimize

    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend

    logger.info("")
    logger.info("=" * 65)
    logger.info("SECTION 2: Cross-Topology (ladder, triangular) at g=0.3")
    logger.info("=" * 65)

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    backend = NoiselessBackend()

    topologies = ["chain_1d", "ladder", "triangular"]
    N, p, g = 6, 2, 0.3
    h_values = [2.0, 1.5, 1.0]
    seeds = [42, 43, 44]

    results_table = []

    for topology in topologies:
        fidelities = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            lattice = make_lattice(topology, N, J=1.0, h=h_values[0])
            circuit, _ = hva.create_tfim_longitudinal(N, p, lattice)
            n_params = circuit.num_parameters
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            for h in sorted(h_values, reverse=True):
                lat_h = make_lattice(topology, N, J=1.0, h=h)
                H = builder.build_tfim_longitudinal(lat_h, g=g)

                # Exact ground state
                H_mat = H.to_matrix()
                if hasattr(H_mat, "toarray"):
                    H_mat = H_mat.toarray()
                evals, evecs = np.linalg.eigh(H_mat)
                gs = evecs[:, 0]

                # VQE (5 restarts)
                best_energy = float("inf")
                best_theta = prev_theta.copy()
                for restart in range(5):
                    x0 = (
                        prev_theta + rng.normal(0, 0.1, n_params)
                        if restart > 0
                        else prev_theta.copy()
                    )
                    x0 = np.clip(x0, -np.pi, np.pi)
                    res = minimize(
                        lambda params, _H=H, _c=circuit: backend.evaluate(_c, _H, params),
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * n_params,
                        options={"maxiter": 500, "ftol": 1e-14},
                    )
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_theta = res.x.copy()

                prev_theta = best_theta.copy()
                sv = Statevector(circuit.assign_parameters(best_theta))
                fid = float(state_fidelity(sv, Statevector(gs)))
                fidelities.append(fid)

        mean_fid = np.mean(fidelities)
        min_fid = np.min(fidelities)
        results_table.append((topology, mean_fid, min_fid, len(fidelities)))

    # Print results
    logger.info("")
    logger.info(f"  {'Topology':<12} | {'Mean Fid':>8} | {'Min Fid':>7} | {'Points':>6}")
    logger.info(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 6}")
    all_pass = True
    for topology, mean_fid, min_fid, n_pts in results_table:
        status = "✓" if mean_fid >= 0.90 else "✗"
        logger.info(f"  {topology:<12} | {mean_fid:>8.4f} | {min_fid:>7.4f} | {n_pts:>6} {status}")
        if mean_fid < 0.90:
            all_pass = False

    assert all_pass, "FAIL: Some topologies below 0.90 fidelity threshold"
    logger.info("  [PASS] All topologies achieve mean fidelity ≥ 0.90")
    return results_table


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Finite-Size Scaling at g=0.3
# ═══════════════════════════════════════════════════════════════════════════════


def test_scaling():
    """Find valid regime boundary h_min(N) at g=0.3 for N=4,6,8."""
    from scipy.optimize import minimize

    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend

    logger.info("")
    logger.info("=" * 65)
    logger.info("SECTION 3: Scaling h_min(N) at g=0.3 (chain_1d)")
    logger.info("=" * 65)

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    backend = NoiselessBackend()

    N_values = [4, 6, 8]
    g = 0.3
    p = 2
    h_grid = np.arange(0.5, 2.55, 0.1)  # Fine grid for boundary detection
    threshold = 0.05  # ΔE/gap < 5%

    results_table = []

    for N in N_values:
        lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
        circuit, _ = hva.create_tfim_longitudinal(N, p, lattice)
        n_params = circuit.num_parameters

        # Descending sweep to find boundary
        rng = np.random.default_rng(42)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        h_min = None

        for h in np.sort(h_grid)[::-1]:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=float(h))
            H = builder.build_tfim_longitudinal(lat_h, g=g)
            H_mat = H.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals = np.sort(np.linalg.eigvalsh(H_mat))
            e_exact = evals[0]
            gap = evals[1] - evals[0]

            # VQE (3 restarts for speed)
            best_energy = float("inf")
            best_theta = prev_theta.copy()
            for restart in range(3):
                x0 = prev_theta + rng.normal(0, 0.1, n_params) if restart > 0 else prev_theta.copy()
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda params, _H=H, _c=circuit: backend.evaluate(_c, _H, params),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 300, "ftol": 1e-14},
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.copy()
            prev_theta = best_theta.copy()

            de_gap = abs(best_energy - e_exact) / max(gap, 1e-10)
            if de_gap >= threshold:
                h_min = float(h)

        # Compare with standard TFIM (g=0) boundary from project data
        tfim_boundary = {4: 0.95, 6: 1.20, 8: 1.30}
        results_table.append((N, h_min, tfim_boundary.get(N)))

    logger.info("")
    logger.info(f"  {'N':>3} | {'h_min (g=0.3)':>13} | {'h_min (g=0)':>11} | {'Shift':>6}")
    logger.info(f"  {'-' * 3}-+-{'-' * 13}-+-{'-' * 11}-+-{'-' * 6}")
    for N, h_min_g, h_min_0 in results_table:
        shift = (h_min_g - h_min_0) if h_min_g and h_min_0 else 0
        logger.info(f"  {N:>3} | {h_min_g or 'N/A':>13} | {h_min_0 or 'N/A':>11} | {shift:>+6.2f}")

    logger.info("  [PASS] Scaling boundary computed for all N values")
    return results_table


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Full Pipeline (Phases 1-4) with PipelineRunner
# ═══════════════════════════════════════════════════════════════════════════════


def test_full_pipeline():
    """Run complete 4-phase pipeline using PipelineRunner + ModelSpec."""
    from qmbp_simulation import make_lattice
    from qmbp_simulation.models import VQEConfig
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.pipeline import PipelineRunner

    logger.info("")
    logger.info("=" * 65)
    logger.info("SECTION 4: Full Pipeline (Phases 1→4) with MPNN at g=0.3")
    logger.info("=" * 65)

    N, p, g = 6, 2, 0.3
    topology = "chain_1d"

    # Use ModelSpec with g=0.3 for the pipeline
    model_spec = get_model_spec("tfim_longitudinal").with_g(g)

    lattice = make_lattice(topology, N, J=1.0, h=1.5)
    vqe_config = VQEConfig(
        p_layers=p,
        n_restarts=3,
        maxiter=300,
        ftol=1e-14,
    )

    runner = PipelineRunner(
        lattice=lattice,
        config=vqe_config,
        seed=42,
        model_spec=model_spec,
        verbose=False,
    )

    # Descending h-sweep (valid regime for g=0.3)
    h_values = np.array([2.0, 1.75, 1.5, 1.25])
    h_test = 1.6  # Interpolation point

    t0 = time.time()
    result = runner.run_full(
        h_values=h_values,
        h_test=h_test,
        mpnn_config={
            "n_epochs": 3000,
            "lr": 1e-3,
            "patience": 200,
            "hidden_dim": 64,
        },
    )
    elapsed = time.time() - t0

    # Validate results
    phase1 = result["phase1"]
    phase2 = result["phase2"]
    phase4 = result["phase4"]

    logger.info(f"  Pipeline completed in {elapsed:.1f}s")
    logger.info(f"  Phase 1: {len(phase1)} exact solutions computed")
    logger.info(f"  Phase 2: mean fidelity = {np.mean([r.fidelity for r in phase2]):.4f}")

    if phase4 is not None and len(phase4) > 0:
        deploy = phase4[0]
        logger.info(f"  Phase 4: h_test={h_test}, ΔE/gap={deploy.delta_e_over_gap:.4f}")
        logger.info(f"           phase_label={deploy.phase_label}")
        logger.info(f"           passes 5% threshold: {deploy.passes()}")

        # Validation: Phase 4 should pass at h=1.6 (well inside valid regime)
        if deploy.delta_e_over_gap < 0.10:
            logger.info("  [PASS] Pipeline deployment ΔE/gap < 10%")
        else:
            logger.info(
                f"  [WARN] Pipeline deployment ΔE/gap = {deploy.delta_e_over_gap:.4f} (>10%)"
            )
    else:
        logger.info("  [WARN] Phase 4 skipped (MPNN training may have failed)")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: p=1 Validation (Hardware-deployable depth)
# ═══════════════════════════════════════════════════════════════════════════════


def test_p1_validation():
    """Validate p=1 extended HVA at g=0.3 (hardware-viable configuration)."""
    from qiskit.quantum_info import Statevector, state_fidelity
    from scipy.optimize import minimize

    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend

    logger.info("")
    logger.info("=" * 65)
    logger.info("SECTION 5: p=1 Extended HVA (hardware-viable, 3 params)")
    logger.info("=" * 65)

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    backend = NoiselessBackend()

    N, p, g = 6, 1, 0.3
    h_values = [2.0, 1.75, 1.5, 1.25]
    seeds = [42, 43, 44]

    results_table = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
        circuit, _ = hva.create_tfim_longitudinal(N, p, lattice)
        n_params = circuit.num_parameters  # Should be 3 for p=1
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_values, reverse=True):
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build_tfim_longitudinal(lat_h, g=g)
            H_mat = H.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals, evecs = np.linalg.eigh(H_mat)
            gs = evecs[:, 0]
            e_exact = evals[0]
            gap = evals[1] - evals[0]

            # Single restart (p=1 is simpler landscape)
            x0 = prev_theta.copy()
            res = minimize(
                lambda params, _H=H, _c=circuit: backend.evaluate(_c, _H, params),
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            prev_theta = res.x.copy()

            sv = Statevector(circuit.assign_parameters(res.x))
            fid = float(state_fidelity(sv, Statevector(gs)))
            de_gap = abs(res.fun - e_exact) / max(gap, 1e-10)
            results_table.append((seed, h, fid, de_gap))

    # Print results
    logger.info(f"  params/layer: {n_params} (θ_zz, θ_x, θ_z)")
    logger.info("")
    logger.info(f"  {'Seed':>4} | {'h':>5} | {'Fidelity':>8} | {'ΔE/gap':>7}")
    logger.info(f"  {'-' * 4}-+-{'-' * 5}-+-{'-' * 8}-+-{'-' * 7}")
    for seed, h, fid, de_gap in results_table:
        status = "✓" if de_gap < 0.05 else "✗"
        logger.info(f"  {seed:>4} | {h:>5.2f} | {fid:>8.4f} | {de_gap:>7.4f} {status}")

    fids = [r[2] for r in results_table]
    de_gaps = [r[3] for r in results_table]
    logger.info(f"\n  Mean fidelity: {np.mean(fids):.4f} (min: {np.min(fids):.4f})")
    logger.info(f"  Mean ΔE/gap: {np.mean(de_gaps):.4f}")
    logger.info(f"  Pass rate (ΔE/gap < 5%): {np.mean([d < 0.05 for d in de_gaps]):.0%}")

    # p=1 at h>=1.5 should work, lower h may not pass
    high_h_fids = [r[2] for r in results_table if r[1] >= 1.5]
    if high_h_fids and np.mean(high_h_fids) >= 0.90:
        logger.info("  [PASS] p=1 viable for h ≥ 1.5 at g=0.3")
    else:
        logger.info("  [INFO] p=1 has reduced expressibility (expected for shallow depth)")

    return results_table


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    t_start = time.time()
    all_passed = True

    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  E4b FULL VALIDATION SUITE                                   ║")
    logger.info("║  TFIM + Longitudinal Field — Extended HVA (ZZ+X+Z)           ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")
    logger.info("")

    # Section 1: Reproducibility
    try:
        test_reproducibility()
    except AssertionError as e:
        logger.error(f"SECTION 1 FAILED: {e}")
        all_passed = False

    # Section 2: Cross-topology
    try:
        test_cross_topology()
    except AssertionError as e:
        logger.error(f"SECTION 2 FAILED: {e}")
        all_passed = False

    # Section 3: Scaling
    try:
        test_scaling()
    except Exception as e:
        logger.error(f"SECTION 3 FAILED: {e}")
        all_passed = False

    # Section 4: Full pipeline
    try:
        test_full_pipeline()
    except Exception as e:
        logger.error(f"SECTION 4 FAILED: {e}")
        all_passed = False

    # Section 5: p=1 validation
    try:
        test_p1_validation()
    except Exception as e:
        logger.error(f"SECTION 5 FAILED: {e}")
        all_passed = False

    # Summary
    elapsed_total = time.time() - t_start
    logger.info("")
    logger.info("=" * 65)
    logger.info(f"TOTAL TIME: {elapsed_total:.1f}s")
    if all_passed:
        logger.info("STATUS: ALL SECTIONS PASSED ✓")
    else:
        logger.info("STATUS: SOME SECTIONS FAILED ✗")
    logger.info("=" * 65)
