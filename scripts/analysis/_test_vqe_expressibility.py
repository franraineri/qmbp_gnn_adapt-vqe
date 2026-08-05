"""Test: Can HVA p=1 VQE (noiseless) reach the TRUE 2D ground state on heavy_hex?

This is the critical test:
- If VQE noiseless achieves ΔE/gap < 5% → HVA CAN express the ground state
- If VQE noiseless has large ΔE/gap → HVA p=1 is EXPRESSIBILITY-limited

If HVA is expressibility-limited, then neither QPU nor classical can help — 
you need deeper circuits (p>1) or a different ansatz.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models.data_models import VQEConfig
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.optimizers.vqe import VQEOptimizer

spec = get_model_spec("tfim")
solver = ClassicalSolver()
noiseless = NoiselessBackend()

N = 10
topology = "heavy_hex"
p = 1

print(f"\nHVA p={p} Expressibility Test: {topology} N={N}")
print(f"Can VQE (noiseless, unlimited optimizer) reach the exact ground state?")
print(f"{'h':>5} {'E_exact':>14} {'E_VQE_best':>14} {'|ΔE|':>10} {'ΔE/gap':>10} {'%':>7}")
print("-" * 65)

lattice_ref = make_lattice(topology, N, J=1.0, h=4.0)
circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)

rng = np.random.default_rng(42)
prev_theta = rng.uniform(-0.01, 0.01, circuit.num_parameters)

# Descending sweep for warm-start
for h in [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:
    lattice = make_lattice(topology, N, J=1.0, h=h)
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    gt = solver.solve(H, lattice, method="exact")
    e_exact = gt.ground_energy
    gap = gt.gap

    config = VQEConfig(p_layers=p, n_restarts=5, maxiter=500, method="L-BFGS-B")
    opt = VQEOptimizer(config=config, backend=noiseless, seed=42)
    result = opt.optimize(H, circuit, prev_theta.copy(), exact_energy=e_exact)
    prev_theta = result.theta_opt.copy()

    de = abs(result.energy - e_exact)
    de_gap = de / max(gap, 1e-10)
    print(f"{h:>5.1f} {e_exact:>14.8f} {result.energy:>14.8f} {de:>10.2e} {de_gap:>10.6f} {de_gap*100:>6.2f}%")

print("\nInterpretation:")
print("  ΔE/gap < 5% → HVA p=1 CAN express this ground state")
print("  ΔE/gap > 5% → EXPRESSIBILITY LIMIT (neither QPU nor classical helps)")
print("  ΔE/gap > 50% → HVA p=1 fundamentally cannot reach this state")
