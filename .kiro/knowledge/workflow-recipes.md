# Workflow Recipes — Code Reference

## Hamiltonian Builder (TFIM)

```python
from qiskit.quantum_info import SparsePauliOp

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits - 1):
        terms.append(("ZZ", [i, i + 1], -J))
    for i in range(n_qubits):
        terms.append(("X", [i], -h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)
```

## SparsePauliOp Construction Patterns

```python
from qiskit.quantum_info import SparsePauliOp

# Sparse list (preferred for parameterized Hamiltonians)
H = SparsePauliOp.from_sparse_list([
    ("ZZ", [i, i+1], -J) for i in range(n-1)
] + [
    ("X", [i], -h) for i in range(n)
], num_qubits=n)

# Local observables for phase characterization
mag_x = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n) for i in range(n)]
corr_zz = [SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=n) for i in range(n-1)]

# Arithmetic
H_total = H_interaction + H_field
H_total.simplify()
H_total.to_matrix()  # dense matrix for exact diag
```

## Phase 1: Exact Diagonalization Sweep

```python
import numpy as np
from qiskit.quantum_info import SparsePauliOp

def sweep_ground_truth(n_qubits, J, h_values):
    results = []
    for h in h_values:
        H = build_tfim_hamiltonian(n_qubits, J, h)
        evals, evecs = np.linalg.eigh(H.to_matrix())
        psi_gs = evecs[:, 0]
        results.append({
            "h": h, "J": J,
            "ground_energy": evals[0],
            "ground_state": psi_gs,
            "gap": evals[1] - evals[0],
            "mag_x": [psi_gs.conj() @ SparsePauliOp.from_sparse_list(
                [("X", [i], 1.0)], num_qubits=n_qubits
            ).to_matrix() @ psi_gs for i in range(n_qubits)],
        })
    return results
```

## Phase 2: HVA Construction (TFIM)

```python
from qiskit.circuit import QuantumCircuit, Parameter

def build_tfim_hva(n_qubits, p_layers):
    qc = QuantumCircuit(n_qubits)
    # MANDATORY: |+⟩^N initial state (paramagnetic ground state at h→∞)
    qc.h(range(n_qubits))
    params = []
    for layer in range(p_layers):
        t_zz = Parameter(f"t_zz_{layer}")
        params.append(t_zz)
        for i in range(n_qubits - 1):
            qc.rzz(2 * t_zz, i, i + 1)
        t_x = Parameter(f"t_x_{layer}")
        params.append(t_x)
        for i in range(n_qubits):
            qc.rx(2 * t_x, i)
    return qc, params
```

## Phase 2: Warm-Start VQE Sweep

```python
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

def warm_start_sweep(hva_circuit, n_qubits, J, h_values, p_layers):
    estimator = StatevectorEstimator()
    # Small perturbation to escape the symmetry saddle point at θ=0
    theta = np.random.uniform(-0.01, 0.01, 2 * p_layers)
    results = []
    # Sweep DESCENDING: h=2→0 (|+⟩^N is exact at h→∞, warm-start carries downward)
    for h in reversed(h_values):
        H = build_tfim_hamiltonian(n_qubits, J, h)
        def cost(params):
            bound = hva_circuit.assign_parameters(params)
            return estimator.run([(bound, H)]).result()[0].data.evs
        opt = minimize(cost, theta, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12})
        theta = opt.x
        results.append({"h": h, "theta_opt": opt.x.copy(), "energy": opt.fun})
    results.reverse()  # reorder by ascending h for persistence
    return results
```

## Primitives V2 Execution Patterns

```python
# LOCAL simulation (Phase 1-2)
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
bound_qc = hva_circuit.assign_parameters(theta)
job = estimator.run([(bound_qc, hamiltonian)])
energy = job.result()[0].data.evs

# HARDWARE execution (Phase 4)
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=n, operational=True)
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(hva_circuit)
isa_obs = hamiltonian.apply_layout(isa_qc.layout)
estimator = EstimatorV2(backend)
job = estimator.run([(isa_qc, isa_obs)])
energy = job.result()[0].data.evs
```

## Sampling

```python
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
qc_measured = qc.copy()
qc_measured.measure_all()
job = sampler.run([qc_measured], shots=4096)
counts = job.result()[0].data.meas.get_counts()
```

## Parameter Binding & Statevector Utilities

```python
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, state_fidelity

theta = ParameterVector("θ", length=n_params)
bound_qc = circuit.assign_parameters(dict(zip(theta, values)))

sv = Statevector(bound_circuit)
fidelity = state_fidelity(sv, target_state)
expectation = sv.expectation_value(observable)
```

## Algorithms (standalone package)

```python
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
```

## Phase 4: AdaptVQE with Warm-Start

```python
from qiskit_algorithms import AdaptVQE, VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.exceptions import AlgorithmError

# Pauli pool: non-commuting terms of the Hamiltonian
pauli_pool = [
    SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=n)
    for i in range(n - 1)
] + [
    SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n)
    for i in range(n)
]

# Bound HVA circuit as initial state
initial_state = hva_circuit.assign_parameters(theta_pred)

vqe_solver = VQE(
    estimator=StatevectorEstimator(),
    ansatz=QuantumCircuit(n),  # placeholder, AdaptVQE overwrites
    optimizer=L_BFGS_B(maxiter=100),
)

adapt_vqe = AdaptVQE(
    vqe_solver,
    operators=pauli_pool,
    max_iterations=2,
    gradient_threshold=1e-3,
    initial_state=initial_state,
)

try:
    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
except AlgorithmError as e:
    if "first iteration" in str(e):
        # SUCCESS: warm-start already optimal, 0 layers added
        energy = StatevectorEstimator().run(
            [(initial_state, hamiltonian)]
        ).result()[0].data.evs
    else:
        raise
```

## Phase 4: Hardware with GNN Warm-Start

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def deploy_gnn_warmstart(gnn_model, hamiltonian_graph, hva_circuit, hamiltonian, backend):
    theta_pred = gnn_model(hamiltonian_graph).detach().numpy()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_qc = pm.run(hva_circuit)
    isa_obs = hamiltonian.apply_layout(isa_qc.layout)
    estimator = EstimatorV2(backend)
    bound = isa_qc.assign_parameters(theta_pred)
    job = estimator.run([(bound, isa_obs)])
    return job.result()[0].data.evs
```
