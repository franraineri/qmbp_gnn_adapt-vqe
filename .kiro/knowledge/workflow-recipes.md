# Workflow Recipes — V6 Code Templates

> These are code templates for common operations. For rules and constraints, see SKILL.md.
> For V6, prefer using the modular imports over inline code.

## V6 Module Quick Reference

```python
# All core imports
from src.poc.v6 import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer, HardwareDeployer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult, DeployResult,
    save_phase12_dataset, load_phase12_dataset,
)
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.qrc_pipeline import QRCPipeline
from src.poc.v6.pipeline_utils import assert_observable_locality
```

## Phase 1: Ground Truth (V6 pattern)

```python
builder = HamiltonianBuilder()
solver = ClassicalSolver()

exact_data = []
for h in h_values:
    lat_h = make_lattice("chain_1d", N, J=J, h=h)
    H = builder.build(lat_h)
    exact_data.append(solver.solve(H, lat_h))
```

## Phase 2: VQE Sweep (V6 pattern)

```python
hva = HVACircuitBuilder()
base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
qc, theta = hva.create(N, p_layers, base_lattice)

config = VQEConfig(n_restarts=3, maxiter=1000, ftol=1e-14, enable_callbacks=True)
optimizer = VQEOptimizer(config)
vqe_results = optimizer.descending_sweep(h_values, qc, base_lattice, exact_data)
```

## Phase 3: MPNN Training (V6 pattern)

```python
import torch

dataset = build_graph_dataset(
    base_lattice, h_values,
    np.array([r.theta_opt for r in vqe_results]),
    np.array([d.ground_energy for d in exact_data]),
    fidelities=np.array([r.fidelity for r in vqe_results]),
    fidelity_threshold=0.93,
)

model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=2*p_layers)
result = train_mpnn(model, dataset, n_epochs=4000, lr=1e-3, patience=150)
```

## Phase 4: Deployment (V6 pattern)

```python
# MPNN prediction for unseen h
model.eval()
edge_idx, coord = builder.build_graph_data(base_lattice)
x_test = torch.tensor(
    np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
    dtype=torch.float32,
)
from torch_geometric.data import Data
test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
with torch.no_grad():
    theta_pred = model(test_graph).numpy().flatten()

# Adapt-VQE route
deployer = HardwareDeployer()
result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)

# QRC fallback route
qrc = QRCPipeline(seed=42)
qrc.build_reservoir(N, p_layers, base_lattice)
qrc.train_readout(h_values, mag_x_array, corr_zz_array)
qrc_result = deployer.deploy_qrc(qrc, h_test, exact_test)
```

## Dataset Save/Load with Integrity

```python
save_phase12_dataset(
    "phase1_phase2_tfim_N6_p2_v6.npz",
    h_values=h_values, J=J, n_qubits=N, p_layers=p_layers,
    ground_energies=..., gaps=..., mag_x=..., corr_zz=...,
    theta_opt=..., vqe_energies=..., fidelities=...,
)

# Loading validates cost_function="energy" (prevents V5.x failure)
data = load_phase12_dataset("phase1_phase2_tfim_N6_p2_v6.npz")
```

## Primitives V2 Patterns

```python
# LOCAL simulation
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
bound_qc = circuit.assign_parameters(theta)
energy = float(estimator.run([(bound_qc, hamiltonian)]).result()[0].data.evs)

# HARDWARE execution
from qiskit_ibm_runtime import EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(circuit)
isa_obs = hamiltonian.apply_layout(isa_qc.layout)
estimator = EstimatorV2(backend)
energy = float(estimator.run([(isa_qc, isa_obs)]).result()[0].data.evs)
```
