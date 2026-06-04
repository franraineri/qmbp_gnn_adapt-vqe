#!/usr/bin/env python3
"""Smoke test for the qmbp_simulation package.

Imports all public submodules, runs a minimal pipeline (N=4, p=1, 3 h-points),
and verifies ΔE/gap < 5%. Should complete in under 30 seconds.

Usage:
    python tests/smoke_test.py

Exit codes:
    0 — All checks passed
    1 — Import failure or pipeline error
    2 — ΔE/gap threshold exceeded
"""

from __future__ import annotations

import sys
import time


def check_imports() -> list[str]:
    """Import all public submodules and report failures."""
    failures: list[str] = []
    submodules = [
        "qmbp_simulation",
        "qmbp_simulation.utils",
        "qmbp_simulation.models",
        "qmbp_simulation.solvers",
        "qmbp_simulation.circuits",
        "qmbp_simulation.execution",
        "qmbp_simulation.optimizers",
        "qmbp_simulation.predictors",
        "qmbp_simulation.pipeline",
        "qmbp_simulation.framework",
        "qmbp_simulation.analysis",
    ]
    for mod_name in submodules:
        try:
            __import__(mod_name)
        except Exception as e:
            failures.append(f"{mod_name}: {e}")
    return failures


def run_minimal_pipeline() -> list[dict]:
    """Run Phase 1 + Phase 2 on N=4, p=1, 3 h-points.

    Returns list of dicts with h, delta_e_over_gap for each point.
    """
    import numpy as np

    from qmbp_simulation import (
        ClassicalSolver,
        HamiltonianBuilder,
        HVACircuitBuilder,
        VQEOptimizer,
        make_lattice,
    )
    from qmbp_simulation.models import VQEConfig

    N = 4
    p = 1
    J = 1.0
    h_values = np.array([2.0, 1.75, 1.5])  # Descending order, valid regime for p=1

    # Build lattice and tools
    base_lattice = make_lattice("chain_1d", N, J=J, h=2.0)
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()

    # Phase 1: Exact diagonalization
    print("  Phase 1: Exact diagonalization...")
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=float(h))
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    # Phase 2: VQE descending sweep
    print("  Phase 2: VQE optimization (p=1, 1 restart)...")
    config = VQEConfig(
        p_layers=p,
        n_restarts=1,
        maxiter=200,
        enable_callbacks=False,
    )
    circuit, _ = hva.create(N, p, base_lattice)
    optimizer = VQEOptimizer(config=config)
    vqe_results = optimizer.descending_sweep(
        h_values=h_values,
        circuit=circuit,
        lattice=base_lattice,
        exact_data=exact_data,
    )

    # Compute ΔE/gap for each point
    results = []
    for vqe_r, exact_r in zip(vqe_results, exact_data, strict=False):
        gap = max(exact_r.gap, 1e-10)
        delta_e = abs(vqe_r.energy - exact_r.ground_energy)
        de_gap = delta_e / gap
        results.append(
            {
                "h": exact_r.h_value,
                "delta_e": delta_e,
                "gap": exact_r.gap,
                "delta_e_over_gap": de_gap,
                "energy": vqe_r.energy,
                "exact_energy": exact_r.ground_energy,
            }
        )

    return results


def main() -> int:
    t0 = time.time()
    threshold = 0.05  # 5%

    print("=" * 60)
    print("SMOKE TEST: qmbp_simulation package")
    print("=" * 60)

    # Step 1: Import checks
    print("\n[1/2] Checking imports...")
    failures = check_imports()
    if failures:
        print("\n  IMPORT FAILURES:")
        for f in failures:
            print(f"    - {f}")
        print(f"\nSMOKE TEST FAILED: {len(failures)} import(s) broken")
        return 1
    print("  All 11 submodules imported successfully.")

    # Step 2: Minimal pipeline
    print("\n[2/2] Running minimal pipeline (N=4, p=1, h=[2.0, 1.75, 1.5])...")
    try:
        results = run_minimal_pipeline()
    except Exception as e:
        print(f"\n  PIPELINE ERROR: {e}")
        print("\nSMOKE TEST FAILED: Pipeline execution error")
        return 1

    # Step 3: Verify results
    print("\n  Results:")
    all_pass = True
    for r in results:
        status = "PASS" if r["delta_e_over_gap"] < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"    h={r['h']:.1f}: "
            f"ΔE/gap={r['delta_e_over_gap']:.4f} "
            f"(ΔE={r['delta_e']:.2e}, gap={r['gap']:.4f}) "
            f"[{status}]"
        )

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    if not all_pass:
        print(f"\nSMOKE TEST FAILED: ΔE/gap >= {threshold:.0%} at one or more points")
        return 2

    if elapsed > 30:
        print(f"\nWARNING: Smoke test took {elapsed:.1f}s (target: <30s)")

    print(f"\nSMOKE TEST PASSED ({elapsed:.1f}s)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
