---
name: quantum-hva-thesis
description: Core rules and constraints for the Hybrid GNN-HVA Framework for Topological Phase Characterization thesis. Use when writing quantum circuits, building Hamiltonians, running VQE, or making architectural decisions.
---

# Quantum Computing & Condensed Matter — Core Rules

Expert in quantum computing, variational quantum algorithms, condensed matter physics, and hybrid classical-quantum architectures for the Master's Thesis: **Hybrid GNN-HVA Framework for Topological Phase Characterization**.

## Governing Principle (Mele et al., Nature Physics 2026)

1. **Depth truncation**: Non-unital noise truncates circuits to O(log n). ALL HVA circuits MUST be shallow: p ≤ 2 layers.
2. **Local observables only**: Characterize phases via ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩, local energy density. NEVER global state fidelity on hardware.
3. **Stable gradients**: Shallow circuits + local costs = no barren plateaus. GNN warm-start exploits this.

## Architectural Constraints (Non-Negotiable)

- **Ansatz**: ONLY HVA. NEVER HEA.
- **Depth**: p ≤ 2 layers.
- **Initial state**: |+⟩^N (`qc.h(range(n))`). MANDATORY.
- **Warm-start**: θ_opt(Hᵢ) seeds Hᵢ₊₁. Sweep DESCENDING h=2→0. Init with `np.random.uniform(-0.01, 0.01)`, never zeros.
- **AdaptVQE**: max_iterations ≤ 2.
- **Observables**: `SparsePauliOp`, local quantities only on hardware.
- **Fallbacks**: 2D → quasi-1D spin ladders. Noise → SPT phases.

## Qiskit 2.x Rules

| Do ✅ | Don't ❌ |
|-------|---------|
| `SparsePauliOp.from_sparse_list(...)` | `PauliSumOp`, `opflow` |
| `qiskit.primitives.StatevectorEstimator` | `qiskit.execute()`, `Aer.get_backend()` |
| `qiskit_ibm_runtime.EstimatorV2` | Primitives V1, `backend.run()` |
| `from qiskit_algorithms import ...` | `from qiskit.algorithms import ...` |
| `circuit.assign_parameters(theta)` | Manual parameter substitution |
| `generate_preset_pass_manager(optimization_level=2)` | `transpile()` |
| `result[0].data.evs` | Legacy result formats |

Forbidden: `qiskit.opflow`, `qiskit.algorithms` (old path), `PauliSumOp`, `WeightedPauliOperator`, `qiskit.execute()`, `Aer.get_backend()`, `backend.run()`, `transpile()`, Primitives V1.

## Pipeline

| Phase | Goal | Output |
|-------|------|--------|
| 1 | Classical ground truth (Exact Diag / DMRG) | (h,J) → ψ, local observables |
| 2 | HVA θ_opt via warm-start VQE | θ_opt dataset; diagnostic metrics (timing, θ smoothness) when verbose |
| 3 | MPNN predictor (graph → θ_pred) | Trained model; per-h MSE, generalization gap when verbose |
| 4 | Hardware deployment (EstimatorV2) | Mitigated VQE results; SNR, energy decomposition, CES correlation when verbose |
| 4b | Noisy simulation (FakeTorino + BackendEstimatorV2) | Local ZNE validation; 3-mode comparison (noiseless/noisy-raw/ZNE-mitigated) |

## Validation (pass/fail order)

1. ΔE/gap < 5% (primary)
2. ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ errors < 1e-2
3. Fidelity ≥ 99.5% (noiseless only, never hardware)
4. ADAPT iterations ≤ 2

## Literature Validation (Phase 3 Architecture)

Our GNN/MLP warm-start approach is validated by three independent 2024-2026 papers:

- **NN-VQE** (Miao et al., PRApplied 2024): MLP h→θ for parameterized spin Hamiltonians. 20 training points, dropout regularization, active learning. Directly validates our PoC MLP design.
- **Qracle** (Zhang et al., 2025): GNN-based VQE parameter initializer. Unified Hamiltonian+ansatz graph encoding. Up to 64% fewer optimization steps. Validates our GNN scaling path.
- **Flow-VQE** (Zou et al., npj QI 2026): Generative normalizing flows for warm-start. Up to 50x acceleration. Alternative approach — we chose deterministic mapping (simpler, sufficient for smooth TFIM landscape).

Key takeaway: GNN-based initialization works best on physically structured Hamiltonians (spin systems), poorly on random circuits. Our spin-system focus is optimal.

## Extended Literature Validation (2025-2026)

### Architecture Validation
- **GNN > CNN by 36%** for circuit property prediction (Meng et al., 2025, arXiv:2504.00464). Validates GINConv choice.
- **GINConv theoretical foundation** (Xu et al., ICLR 2019): GIN is as powerful as Weisfeiler-Lehman test — maximally expressive among MPNNs.
- **GNN for Ising magnetization** (Slavin, 2025, arXiv:2507.17509): GNN predicts magnetic properties from lattice graph. Directly validates our graph→physical-property paradigm.
- **HVA > HEA confirmed** (Tripathi et al., 2026, arXiv:2604.20961): Benchmarks on TFIM 1D/2D/3D up to 27 spins. HVA consistently outperforms EfficientSU2.

### Physics Limits Validation
- **h=1.25 ceiling is physics** (Tripathi et al., 2026): HVA p=2 struggles with entanglement entropy at criticality — independent confirmation of our 2-3/6 ceiling.
- **N/2 layers needed for thermodynamic limit** (Sumeet et al., 2025, arXiv:2310.07600): For N=6, need p=3 — our p=2 constraint means we cannot reach thermodynamic-limit accuracy at criticality.
- **Hardware noise broadens critical crossover** (Sharma, 2026, arXiv:2601.17515): IQM Garnet experiments show noise smears the phase transition. Expected behavior for our Phase 4.

### Error Mitigation Validation
- **Inhomogeneous ZNE** (Uvarov et al., 2024, arXiv:2307.11156): Linear energy-CES extrapolation using different qubit mappings. Applicable to IBM Torino.
- **Learned DD on IBM** (Pokharel et al., 2025, arXiv:2403.02294): Genetic algorithm DD sequences on 100 qubits. Scalable, no retraining needed.
- **NN-enhanced ZNE** (Sun et al., 2025, arXiv:2501.01646): MLP extrapolation constrains errors to O(10⁻²). Better than polynomial fit.
- **Experimental VQE+ZNE on Ising** (Ma et al., 2025, arXiv:2504.06554): 4-spin Ising on superconducting processor with analog ZNE.

### Hardware Deployment Expectations
- **Shot noise dominates at 4096 shots** (~1.6e-2 per observable) — exceeds our ⟨X⟩ signal (~8e-3 at N=10). Use ≥8192 shots.
- **Phase classification works despite noise** — Sharma (2026) shows ground-state energies are reliably captured even when observables are noisy.
- **TN advantage boundary at N≈20 for 2D** (Martin et al., 2026, arXiv:2602.04676) — our N=6-10 results demonstrate pipeline methodology, not quantum advantage.

## Improvement Opportunities (IBM Qiskit Compatible)

### Phase 4 Enhancements (implement before hardware runs)
1. **Inhomogeneous ZNE**: multiple `generate_preset_pass_manager()` calls with different `initial_layout` → different CES → linear extrapolation
2. **Learned DD**: `PadDynamicalDecoupling` pass with optimized sequences (Qiskit native)
3. **Shot budget**: increase to 8192 minimum (Qiskit `EstimatorV2` `precision` parameter)
4. **Observable grouping**: group commuting Paulis to reduce circuit executions (Qiskit `ObservablesArray`)

### Phase 3 Enhancements (optional)
5. **MPNN weight analysis**: detect phase transitions from trained weight structure (zero QPU cost)
6. **Active learning**: identify high-uncertainty h-regions, run targeted VQE, retrain
7. **Noise-aware training**: train MPNN on `AerSimulator` noisy VQE data for hardware-optimized predictions

## Current PoC (V6.0)

- 1D TFIM, N=6, HVA p=2, |+⟩^N
- **Modular architecture**: 14 Python modules under `src/poc/v6/` + `experimental/` subpackage
- Phases 1-2: `src/poc/v6/poc_v6_phases1_2.ipynb` / Phases 3-4: `src/poc/v6/poc_v6_phases3_4.ipynb`
- Non-uniform h-grid: Δh=0.05 near critical region h∈[0.8,1.4], Δh=0.1 elsewhere (27 points)
- **MPNN predictor** (PyTorch Geometric GINConv + global_mean_pool) — replaces V4 MLP
- Fidelity filter ≥ 0.93, dropout=0.1
- **QRC fallback route**: fixed HVA reservoir + Rx(h) encoding + linear regression readout
- Dataset metadata: `cost_function="energy"`, `version="v6.0"` (prevents V5.x phase coupling failure)
- **Known limit**: HVA p=2 + |+⟩^N cannot express ferromagnetic ground state (h<1.0). Validated for h≥1.0.
- **Pipeline Core** (`pipeline_core.py`): single-source-of-truth for Phase 1→4 execution pattern. Scripts delegate to `run_full_pipeline()` or individual `run_phaseN()` functions.
- **Experimental subpackage** (`experimental/`): deprecated approaches (GATPredictor, augmentation) kept for benchmark reproducibility. NEVER use in new code.

### V6 Module Imports

```python
from src.poc.v6 import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer, LatticeConfig, VQEConfig,
    GroundTruthResult, VQEResult, DeployResult,
    save_phase12_dataset, load_phase12_dataset,
)
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.qrc_pipeline import QRCPipeline
from src.poc.v6.pipeline_core import PipelineCoreConfig, run_full_pipeline  # preferred for scripts
```

## Pipeline Observability

The `DiagnosticCollector` (in `src/poc/v6/diagnostics.py`) instruments the pipeline when `--verbose` or `--debug` is passed to `run_v61_parametric.py`.

### Usage Pattern
```python
from src.poc.v6.diagnostics import DiagnosticCollector, configure_pipeline_logging

logger = configure_pipeline_logging(verbose=True)
collector = DiagnosticCollector(verbose=True, save_dir=Path("scripts/notebook_results"))

# Record after each phase; call save_checkpoint("phaseN") after each
collector.record_vqe_point(h, n_iters, restart_energies, theta_opt, elapsed_s)
collector.record_mpnn_per_h_error(h_values, per_h_mse)
collector.record_deployment(h_test, result, per_layout_data)

# Final output
result["diagnostics"] = collector.to_dict()
collector.cleanup_checkpoints()
```

### Key Metrics
- **SNR**: `|⟨O⟩| * √shots` — measurement reliability (use ≥8192 shots for SNR > 1)
- **θ smoothness**: `max_i ||θ(h_i) - θ(h_{i-1})||_∞` — MPNN learnability predictor
- **Energy decomposition**: separates `error_from_circuit` (physics limit) from `error_from_mpnn` (ML error)
- **Classification confidence**: `|⟨X⟩ - ⟨ZZ⟩| * √shots` — phase label reliability

### CLI Flags
- `--verbose` / `-v`: INFO logging + DiagnosticCollector + VQE callbacks + checkpoints
- `--debug`: DEBUG logging + all verbose features (per-iteration detail)
- Neither: WARNING only, no diagnostics, byte-identical output to baseline

### Checkpoint Pattern
- Files: `checkpoint_<run_id>_<phase>.json` in output directory
- Written after each phase for crash recovery
- Deleted on successful completion

## IBM Connection Pattern

```python
import os
from qiskit_ibm_catalog import QiskitFunctionsCatalog
from qiskit_ibm_runtime import QiskitRuntimeService

ibm_token = os.environ.get("IBM_KEY")
ibm_instance = os.environ.get("IBM_INSTANCE_CRN")
backend_name = "ibm_torino"

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token, instance=ibm_instance)
catalog = QiskitFunctionsCatalog(instance=ibm_instance, token=ibm_token)
```


## Noisy Simulation Workflow

### 3-Mode Comparison Methodology

The noisy simulation workflow validates ZNE effectiveness locally before real QPU deployment by comparing three execution modes at each h-value:

1. **Noiseless** (`mode="simulation"`): StatevectorEstimator — exact baseline (ceiling)
2. **Noisy raw** (`mode="noisy_simulation"`, `n_layouts=1`): FakeTorino noise, no ZNE — shows noise impact
3. **ZNE mitigated** (`mode="noisy_simulation"`, `n_layouts=3`): FakeTorino noise + inhomogeneous ZNE — shows mitigation gain

### Backend & Estimator Pattern

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.primitives import BackendEstimatorV2

# FakeTorino provides real Torino calibration data (133 qubits, heavy-hex)
# BackendEstimatorV2 executes circuits through the noise model locally
# No DD/twirling/TREX — isolates ZNE contribution
deployer = HardwareDeployerV61(mode="noisy_simulation", n_layouts=3, seed=42)
```

### Success Criteria

- `n_mitigated_wins >= 4`: ZNE-mitigated ΔE/gap < noisy-raw ΔE/gap for at least 4 of 6 h-values
- `n_good_r_squared >= 3`: ZNE linear fit R² > 0.8 for at least 3 of 6 h-values
- Both must hold for `success_criteria_met = True`

### Running the Sweep

```bash
# Standard sweep (N=6, 6 h-values × 3 modes, ~5 min)
python scripts/run_v61_noisy.py

# Thesis-grade sweep (N=10, ~10 min)
python scripts/run_v61_noisy.py --n10
```

Results saved as timestamped JSON in `scripts/notebook_results/`.
