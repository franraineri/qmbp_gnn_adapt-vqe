---
inclusion: fileMatch
fileMatchPattern: "src/qmbp_simulation/**/*.py"
---

# Code Style — qmbp_simulation Package

## Package Structure

```
src/qmbp_simulation/
├── __init__.py              ← Package-level re-exports
├── utils/                   ← Seed, JSON, timing (no internal deps)
│   └── helpers.py
├── models/                  ← LatticeConfig, Hamiltonians, data models
│   ├── data_models.py       ← Shared dataclasses
│   ├── constants.py         ← Physics constants
│   └── hamiltonian.py       ← HamiltonianBuilder, make_lattice
├── solvers/                 ← ExactDiag, DMRG
│   └── classical.py
├── circuits/                ← HVA builder
│   └── hva.py
├── execution/               ← Backend ABC + implementations
│   └── backends.py
├── optimizers/              ← VQE, SPSA
│   ├── vqe.py
│   └── spsa.py
├── predictors/              ← MPNN model, training, checkpoints
│   └── mpnn.py
├── pipeline/                ← Orchestration, dataset I/O
│   ├── dataset_io.py
│   ├── runner.py           ← PipelineRunner + run_exact_diag_sweep helper
│   └── qrc.py
├── framework/               ← Experiment engine, CLI, benchmarking, result I/O
│   ├── base.py             ← BaseExperiment lifecycle
│   ├── config.py           ← ExperimentConfig, SystemConfig, VQEConfig, MPNNConfig
│   ├── criteria.py         ← EXPERIMENT_CRITERIA, compute_verdict (single source)
│   ├── metrics.py          ← ExperimentMetrics, WarmColdComparison
│   ├── logging.py          ← StructuredLogger + ProgressReporter
│   ├── cli.py             ← Shared CLI argument groups and validation
│   ├── result_io.py       ← Standardized result saving/loading
│   ├── result_store.py    ← Result querying, comparison, CATEGORY_MAP
│   ├── preflight.py       ← Pre-run validation (variants + experiments)
│   ├── benchmarking.py    ← BenchmarkSuite, BenchmarkResult
│   ├── variant_runner.py  ← PipelineVariant, RunResult, VariantRunner, run_variant_script
│   └── runner_base.py     ← ValidationRunner, ExperimentRunner, VariantPipelineRunner
└── analysis/                ← Gradient analysis, diagnostics, comparison
    ├── gradient.py
    ├── diagnostics.py
    ├── metrics.py
    ├── landscape.py
    └── data_models.py
```

## Lattice Construction
- Always use `make_lattice()` to create LatticeConfig instances.
- Never use `copy.copy()` or `copy.deepcopy()` on dataclasses to change `h`.
- To vary h across a sweep, construct a new LatticeConfig per point via `make_lattice()`.

## Imports

### Core (from package top-level)
```python
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult,
    save_phase12_dataset, load_phase12_dataset,
    PipelineRunner,
)
```

### Execution backends
```python
from qmbp_simulation.execution import (
    ExecutionBackend, NoiselessBackend, NoisyBackend,
    HardwareBackend, MitigationOptions,
)
```

### Noisy simulation utilities
```python
from qmbp_simulation.execution import (
    NoisyEstimatorConfig, build_adjacency, find_layouts_bfs,
    compute_circuit_ces, select_layouts_by_circuit_ces,
    select_layouts_low_ces,
    noisy_estimate, linear_zne,
    # Gate-folding ZNE
    fold_gates, run_gate_folding_zne, run_gate_folding_zne_deployment,
    GateFoldingZNEResult, GateFoldingDeploymentResult,
    # PEA (Probabilistic Error Amplification)
    run_pea_zne, run_pea_zne_deployment,
    PEAResult, PEADeploymentResult,
)
```

### MPNN / Predictors (not in top-level to avoid heavy torch imports)
```python
from qmbp_simulation.predictors import (
    MPNNPredictor, build_graph_dataset, train_mpnn,
    save_mpnn_checkpoint, load_mpnn_checkpoint,
)
```

### Framework (experiment engine + CLI + result I/O + benchmarking)
```python
from qmbp_simulation.framework import (
    # Experiment engine
    BaseExperiment, ExperimentConfig, ExperimentMetrics,
    WarmColdComparison, StructuredLogger, ProgressReporter,
    # CLI argument groups
    create_base_parser, add_system_args, add_sweep_args,
    add_vqe_args, add_mpnn_args, add_output_args, add_noisy_args,
    add_result_filter_args, add_format_args, add_variant_runner_args,
    validate_descending_sweep, validate_system_size,
    configure_logging, build_mpnn_config_dict, resolve_output_dir,
    # Result I/O
    save_experiment_result, save_pipeline_result, save_benchmark_result,
    build_result_envelope, load_result, generate_timestamp,
    # Result store
    ResultStore, CATEGORY_MAP,
    # Experiment criteria (single source of truth)
    EXPERIMENT_CRITERIA, REJECTION_IS_FINDING, compute_verdict,
    # Benchmarking
    BenchmarkSuite, BenchmarkResult,
    # Variant runner (for topology variant scripts)
    PipelineVariant, RunResult, VariantRunner, run_variant_script,
    # Runner bases (for all scripts/run_*.py)
    ExperimentRunner, ValidationRunner, VariantPipelineRunner,
    Section, SectionResult, resolve_project_root,
)
```

### Analysis and diagnostics
```python
from qmbp_simulation.analysis import (
    WeightGradientAnalyzer, DiagnosticCollector,
    compute_snr, compute_theta_smoothness,
    compute_classification_confidence, compute_energy_decomposition,
    compute_hessian, landscape_fluctuation,
)
```

### Pipeline
```python
from qmbp_simulation.pipeline import PipelineRunner, run_exact_diag_sweep
```

### NEVER use these patterns
```python
# ❌ WRONG — legacy sys.path hacks
import sys; sys.path.insert(0, ...)
from src.poc.v6 import ...

# ❌ WRONG — importing from archive
from archive.src_poc_v6_BAK import ...

# ❌ WRONG — importing from experiments inside the package
from experiments.helpers import ...  # Only valid in other experiments/scripts

# ❌ WRONG — duplicating noisy utilities in scripts
def build_adjacency(backend): ...  # Use from qmbp_simulation.execution

# ❌ WRONG — duplicating JSON serialization logic
def _json_default(obj):  # Use json_serialize from utils
    if isinstance(obj, np.integer): ...

# ❌ WRONG — duplicating experiment criteria
EXPERIMENT_CRITERIA = {...}  # Import from qmbp_simulation.framework.criteria
```

## JSON Serialization (single canonical implementation)

**All JSON serialization MUST use `json_serialize` from `qmbp_simulation.utils.helpers`.**

```python
from qmbp_simulation.utils.helpers import json_serialize, json_dump

# For writing files:
json_dump(data, path)  # Preferred — handles mkdir + indent

# For json.dump with custom objects:
json.dump(data, f, indent=2, default=json_serialize)

# For pre-serializing before embedding in dicts:
serialized = json_serialize(my_numpy_object)
```

**Never create local `_json_default` functions.** The canonical `json_serialize` handles:
`np.bool_`, `np.integer`, `np.floating`, `np.ndarray`, `Path`, `datetime`, dataclasses, `NaN/Inf → None`.

## Experiment Criteria (single source of truth)

**All experiment success criteria MUST live in `framework/criteria.py`.**

```python
from qmbp_simulation.framework.criteria import (
    EXPERIMENT_CRITERIA,      # dict[str, dict[str, Any]]
    REJECTION_IS_FINDING,     # set[str]
    compute_verdict,          # (exp_id, summary) → (verdict, desc)
)
```

- Never duplicate criteria dicts in scripts or digest modules.
- `project_health/digest/models.py` re-exports from criteria.py for backward compat.
- To add a new experiment: add its entry to `EXPERIMENT_CRITERIA` in `criteria.py`.

## Module Dependency Order

Modules follow a strict DAG (no circular imports possible):

```
utils → models → solvers, circuits → execution → optimizers
                  models → predictors
         solvers, optimizers, predictors, analysis → pipeline
                  pipeline, analysis → framework
         models, predictors → analysis
```

- A module may only import from modules **above** it in this order.
- No module in `src/qmbp_simulation/` imports from `experiments/` or `scripts/`.

## Qiskit Patterns
- SparsePauliOp via `from_sparse_list()` only.
- Energy evaluation: `StatevectorEstimator().run([(bound_qc, H)]).result()[0].data.evs`
- Never use `H.to_matrix()` for validation on large systems — use Pauli-level checks.

## Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` (in `models/constants.py`)
- Parameters: match the physics notation where possible (h, J, theta, psi)
- Modules: `snake_case.py`
- Subpackages: `snake_case/`

## Error Handling
- Raise `ValueError` for constraint violations (p>2, invalid topology, ascending sweep).
- Use `logging.warning()` for recoverable issues (DMRG fallback, gap computation failure).
- Use `assert` only for invariant checks (QRC no-training invariant).
- `HardwareBackend` raises `NotImplementedError` until IBM Runtime is configured.

## Adding a New Hamiltonian (Extension Pattern)

When adding a new spin model to the framework:

### 1. Hamiltonian (`models/hamiltonian.py`)
```python
def build_<model>(self, lattice: LatticeConfig, **kwargs) -> SparsePauliOp:
    """Build H = ... (describe terms)."""
    # Use SparsePauliOp.from_sparse_list only
    # Call self.validate(H, n) at the end
```

### 2. Circuit (`circuits/hva.py`)
```python
def create_<model>(self, n_qubits, p_layers, lattice, **kwargs):
    """HVA matching the Hamiltonian structure."""
    # Enforce MAX_P_LAYERS
    # Return (qc, theta) with params_per_layer × p_layers parameters
```

### 3. Registry (`models/model_registry.py`)
```python
register_model(ModelSpec(
    name="<model>",
    params_per_layer=<N>,
    build_hamiltonian=builder.build_<model>,
    build_observables=builder.build_<model>_observables,  # or build_local_observables
    create_circuit=_create_<model>,  # lazy import
    hamiltonian_kwargs={<defaults>},
    ...
))
```

### 4. Usage in Experiments (via ModelSpec)
```python
from qmbp_simulation.models.model_registry import get_model_spec

spec = get_model_spec("<model>")
# Vary parameters:
spec_custom = spec.with_params(param1=val1, param2=val2)
# Build:
H = spec_custom.build_hamiltonian(lattice, **spec_custom.hamiltonian_kwargs)
qc, theta = spec_custom.create_circuit(N, p, lattice, **spec_custom.circuit_kwargs)
```

### 5. Digest Integration (automatic)
Results with `"model": "<model>"` in their JSON are automatically detected by the digest system.
Filter with `--model <model>` and group with `--group-by model`.

### Available Models
| Name | H | Params/layer | Status |
|------|---|:---:|--------|
| `tfim` | −J·ZZ − h·X | 2 | ✅ Production |
| `tfim_longitudinal` | −J·ZZ − h·X − g·Z | 3 | ✅ Validated (E4b) |
| `tfim_frustrated` | −J₁·ZZ_nn + J₂·ZZ_nnn − h·X | 3 | ✅ Noiseless only (27 CZ@N=6) |
| `heisenberg` | J(XX+YY+Δ·ZZ) − h·Z | 4 | ⚠️ p≤2 insufficient |
| `xy` | J(XX+YY) − h·Z | 4 | ⚠️ Same limits as Heisenberg |
