"""Profile each step of the noisy pipeline at N=14 to find the freeze."""
import sys
import time

sys.path.insert(0, "src")

import numpy as np

print("1: FakeTorino...", flush=True)
t0 = time.time()
from qiskit_ibm_runtime.fake_provider import FakeTorino
fb = FakeTorino()
print(f"   OK {time.time()-t0:.1f}s ({fb.num_qubits} qubits)", flush=True)

print("2: Circuit N=14...", flush=True)
t0 = time.time()
from qmbp_simulation import make_lattice, HamiltonianBuilder
from qmbp_simulation.circuits import HVACircuitBuilder
lattice = make_lattice("chain_1d", 14, J=1.0, h=3.0)
H = HamiltonianBuilder().build(lattice)
circuit, _ = HVACircuitBuilder().create(14, 1, lattice)
bound = circuit.assign_parameters(np.array([0.1, 0.2]))
print(f"   OK {time.time()-t0:.1f}s depth={bound.depth()}", flush=True)

print("3: find_layouts...", flush=True)
t0 = time.time()
from qmbp_simulation.execution.noisy_utils import build_adjacency, find_layouts_bfs
adj = build_adjacency(fb)
cands = find_layouts_bfs(adj, 14, n_candidates=10)
print(f"   OK {time.time()-t0:.1f}s found={len(cands)}", flush=True)

print("4: Transpile 1 layout...", flush=True)
t0 = time.time()
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=2, backend=fb, initial_layout=cands[0])
tr = pm.run(bound)
n2q = sum(1 for i in tr.data if i.operation.num_qubits == 2)
print(f"   OK {time.time()-t0:.1f}s 2Q={n2q} depth={tr.depth()}", flush=True)

print("5: noisy_estimate (BackendEstimatorV2)...", flush=True)
t0 = time.time()
from qmbp_simulation.execution.noisy_utils import noisy_estimate, NoisyEstimatorConfig
cfg = NoisyEstimatorConfig(shots=4096, seed_simulator=42)
H_m = H.apply_layout(tr.layout)
e = noisy_estimate(tr, H_m, fb, cfg)
print(f"   OK {time.time()-t0:.1f}s E={e:.4f}", flush=True)

print("6: run_pea_zne (3 factors)...", flush=True)
t0 = time.time()
from qmbp_simulation.execution.noisy_utils import run_pea_zne
pea = run_pea_zne(tr, H_m, fb, cfg, noise_factors=(1, 3, 5))
print(f"   OK {time.time()-t0:.1f}s E={pea.extrapolated_value:.4f} R2={pea.r_squared:.3f}", flush=True)

print("\nALL DONE N=14", flush=True)
