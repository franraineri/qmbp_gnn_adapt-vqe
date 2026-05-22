#!/usr/bin/env python
"""B4-lite: Quick Hessian analysis at VQE minima."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from scripts.experiments_v8.core.landscape import compute_hessian
from src.poc.v6 import ClassicalSolver, HamiltonianBuilder, HVACircuitBuilder, make_lattice

builder, solver, hva = HamiltonianBuilder(), ClassicalSolver(), HVACircuitBuilder()
estimator = StatevectorEstimator()
N, p = 6, 2
qc, _ = hva.create(N, p, make_lattice("chain_1d", N, J=1.0, h=1.0))
n_params = qc.num_parameters

print("B4-LITE: Hessian at VQE Minima (N=6, p=2)")
print(f"{'h':<6}{'DE/gap':<9}{'Type':<12}{'Eigenvalues':<36}{'Cond':<8}")
print("-" * 68)

np.random.seed(42)
prev = np.random.uniform(-0.01, 0.01, n_params)

for h in [2.0, 1.5, 1.25, 1.0]:
    lat = make_lattice("chain_1d", N, J=1.0, h=h)
    H = builder.build(lat)
    exact = solver.solve(H, lat)

    def cost(params, _H=H):
        return float(estimator.run([(qc.assign_parameters(params), _H)]).result()[0].data.evs)

    best_e, best_t = float("inf"), prev.copy()
    for r in range(5):
        x0 = prev + np.random.normal(0, 0.1, n_params) if r > 0 else prev
        x0 = np.clip(x0, -np.pi, np.pi)
        res = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": 300, "ftol": 1e-14},
        )
        if res.fun < best_e:
            best_e, best_t = res.fun, res.x.copy()

    gap = exact.gap if exact.gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
    de = abs(best_e - exact.ground_energy) / gap

    H_mat = compute_hessian(cost, best_t, epsilon=5e-3)
    eigs = np.linalg.eigvalsh(H_mat)
    n_neg = int(np.sum(eigs < -0.01))
    tp = "minimum" if n_neg == 0 else f"saddle({n_neg})"
    pos = eigs[eigs > 0.01]
    cond = pos[-1] / pos[0] if len(pos) >= 2 else 1.0
    eig_s = ",".join(f"{e:.1f}" for e in eigs)
    print(f"{h:<6.2f}{de:<9.4f}{tp:<12}[{eig_s}]{cond:>8.1f}")
    prev = best_t.copy()

print("\nInterpretation:")
print("  minimum = genuine local minimum (all eigenvalues > 0)")
print("  saddle(N) = N negative eigenvalues (VQE stuck at saddle)")
print("  High condition number = ill-conditioned (flat directions)")
