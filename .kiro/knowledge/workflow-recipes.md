# Workflow Recipes — V6 Code Templates

> These are code templates for common operations. For rules and constraints, see SKILL.md.
> For V6, prefer using the modular imports over inline code.

## V6 Module Quick Reference

```python
# All core imports (V6.0)
from src.poc.v6 import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult, DeployResult,
    save_phase12_dataset, load_phase12_dataset,
)
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.mpnn_predictor import save_mpnn_checkpoint, load_mpnn_checkpoint
from src.poc.v6.qrc_pipeline import QRCPipeline
from src.poc.v6.pipeline_utils import assert_observable_locality

# V6.1 extensions (hardware + analysis)
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
from src.poc.v6.analysis_utils import WeightGradientAnalyzer
from src.poc.v6.config_v61 import (
    DeployResultV61, LayoutResult, GradientAnalysisResult, MPNNCheckpoint,
    SHOT_BUDGET_SMALL, SHOT_BUDGET_MEDIUM, SHOT_BUDGET_LARGE,
)

# V6.0 deployer (simulation-only, lighter dependencies)
from src.poc.v6.hardware_deployer import HardwareDeployer
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

# Adapt-VQE route (V6.0 — simulation only)
from src.poc.v6.hardware_deployer import HardwareDeployer
deployer = HardwareDeployer()
result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)

# V6.1 Deployer — full hardware path (simulation or real)
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
deployer_v61 = HardwareDeployerV61(mode="simulation")  # or mode="hardware"
result_v61 = deployer_v61.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
# Returns DeployResultV61 with ZNE data, uncertainty, per-site observables

# QRC fallback route
from src.poc.v6.qrc_pipeline import QRCPipeline
qrc = QRCPipeline(seed=42)
qrc.build_reservoir(N, p_layers, base_lattice)
qrc.train_readout(h_values, mag_x_array, corr_zz_array)
qrc_result = deployer.deploy_qrc(qrc, h_test, exact_test)
```

## Weight Gradient Analysis (V6.1 — zero QPU cost)

```python
from src.poc.v6.analysis_utils import WeightGradientAnalyzer

analyzer = WeightGradientAnalyzer(model)  # trained MPNN
grad_result = analyzer.analyze(dataset)

# grad_result.total_gradient_norms — array of ‖∂L/∂W‖₂ per h-value
# grad_result.per_layer_gradient_norms — dict: {"ginconv_0": [...], "head": [...]}
# grad_result.peak_h_values — h-values where peaks detected
# grad_result.critical_region_detected — True if peaks in h ∈ [0.8, 1.4]
```

## MPNN Checkpoint Save/Load (V6.1)

```python
from src.poc.v6.mpnn_predictor import save_mpnn_checkpoint, load_mpnn_checkpoint

# Save with metadata
save_mpnn_checkpoint(model, "model.pt", {"epochs": 6000, "mse": 0.003})

# Load reconstructs correct architecture from metadata
loaded_model = load_mpnn_checkpoint("model.pt")
```

## Per-Parameter Heads (V6.1 — optional)

```python
model = MPNNPredictor(
    node_features=2, hidden_dim=64, n_layers=3, output_dim=2*p,
    per_parameter_heads=True,  # separate heads for θ_zz and θ_x
)
result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3)
# result["zz_head_loss_history"] and result["x_head_loss_history"] available
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

# HARDWARE execution (Phase 4 — IBM Torino/Heron)
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=os.environ["IBM_KEY"],
                               instance=os.environ["IBM_INSTANCE_CRN"])
backend = service.backend("ibm_torino")

# Transpile
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(circuit.assign_parameters(theta))
isa_obs = [obs.apply_layout(isa_qc.layout) for obs in observables]

# Configure error mitigation (resilience level 2 = TREX + ZNE + twirling)
estimator = EstimatorV2(mode=backend)
estimator.options.dynamical_decoupling.enable = True
estimator.options.dynamical_decoupling.sequence_type = "XpXm"
estimator.options.twirling.enable_gates = True
estimator.options.twirling.num_randomizations = 32
estimator.options.twirling.shots_per_randomization = 256  # 8192 total
estimator.options.resilience.measure_mitigation = True  # TREX
estimator.options.resilience.zne_mitigation = True
estimator.options.resilience.zne.noise_factors = [1, 2, 3]
estimator.options.resilience.zne.extrapolator = "exponential"

# Execute
job = estimator.run([(isa_qc, isa_obs)])
result = job.result()
energies = [r.data.evs for r in result]
```

## Phase 4: Observable Grouping (Shot Efficiency)

```python
# All ⟨X_i⟩ commute — single measurement basis
x_obs = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=N) for i in range(N)]

# All ⟨Z_iZ_{i+1}⟩ commute — single measurement basis
zz_obs = [SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=N) for i in range(N-1)]

# Submit as grouped observables (Qiskit groups commuting automatically)
all_obs = x_obs + zz_obs  # 2 measurement bases total, not N+N-1 circuits
job = estimator.run([(isa_qc, all_obs)])
```

## Experiment Logging with Auto-Registry

```bash
# Standard experiment run with binnacle logging
python scripts/run_notebooks.py --binnacle --label "N6 baseline v6.1"

# N=10 scaling experiment (auto-routes to binnacle-N10.md)
python scripts/run_notebooks.py --binnacle --label "N10 hidden128" --timeout 900

# Dry-run to verify pre-flight without executing
python scripts/run_notebooks.py --dry-run

# Prune old results (keep last 10 runs)
python scripts/run_notebooks.py --keep-last 10 --phase all --binnacle --label "cleanup run"
```

The auto-registry produces:
- `scripts/notebook_results/run_summary_<timestamp>_<hash>.json` — structured JSON with full metrics, environment, and cell outputs
- Binnacle markdown entry (if `--binnacle`) — appended to `documentation/binnacles/` with auto-observations and run-to-run comparison

### JSON Summary Structure

```json
{
  "timestamp": "2026-05-08T...",
  "run_id": "a1b2c3d4",
  "label": "N6 baseline v6.1",
  "phases": "all",
  "environment": {"git_commit": "f52160d", "git_branch": "version_6.1", ...},
  "results": [
    {
      "notebook": "poc_v6_phases1_2.ipynb",
      "success": true,
      "elapsed_seconds": 45.2,
      "peak_memory_mb": 312.5,
      "slowest_cell_seconds": 28.1,
      "metrics": {
        "dataset_avg_fidelity": 99.2,
        "dataset_fid_ge_93pct": 25,
        "dataset_n_points": 27
      }
    }
  ],
  "total_elapsed": 180.5,
  "all_passed": true
}
```
