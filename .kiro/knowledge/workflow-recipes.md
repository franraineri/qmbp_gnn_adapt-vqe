# Workflow Recipes — qmbp_simulation Code Templates

> These are code templates for common operations using the current package.
> For rules and constraints, see `.kiro/skills/quantum/SKILL.md`.
> For project status, see `.kiro/steering/project-status.md`.

## Quick Reference: Core Imports

```python
# Core imports (from package top-level)
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult,
    save_phase12_dataset, load_phase12_dataset,
    PipelineRunner,
)

# Execution backends
from qmbp_simulation.execution import (
    ExecutionBackend, NoiselessBackend, NoisyBackend, HardwareBackend,
)

# Noisy simulation utilities
from qmbp_simulation.execution import (
    NoisyEstimatorConfig, build_adjacency, find_layouts_bfs,
    compute_circuit_ces, select_layouts_by_circuit_ces,
    noisy_estimate, linear_zne,
)

# MPNN / Predictors (not in top-level to avoid heavy torch imports)
from qmbp_simulation.predictors import (
    MPNNPredictor, build_graph_dataset, train_mpnn,
    save_mpnn_checkpoint, load_mpnn_checkpoint,
)

# Framework (experiment engine + CLI + result I/O)
from qmbp_simulation.framework import (
    BaseExperiment, ExperimentConfig, ExperimentMetrics,
    StructuredLogger, ProgressReporter,
    ResultStore, CATEGORY_MAP,
    BenchmarkSuite, BenchmarkResult,
    create_base_parser, add_system_args, add_sweep_args,
    add_vqe_args, add_mpnn_args, add_output_args,
    validate_descending_sweep, configure_logging,
    save_experiment_result, save_pipeline_result,
    build_result_envelope, load_result,
)

# Analysis and diagnostics
from qmbp_simulation.analysis import (
    WeightGradientAnalyzer, DiagnosticCollector,
    compute_snr, compute_theta_smoothness,
    compute_hessian, landscape_fluctuation,
)

# Pipeline helpers
from qmbp_simulation.pipeline import run_exact_diag_sweep
```

## Phase 1: Ground Truth

```python
from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
from qmbp_simulation.pipeline import run_exact_diag_sweep
import numpy as np

# Option A: Using the helper function
h_values = np.linspace(2.0, 0.5, 31)
exact_data = run_exact_diag_sweep(h_values, n_qubits=6)

# Option B: Manual loop (when you need custom lattice params)
builder = HamiltonianBuilder()
solver = ClassicalSolver()
exact_data = []
for h in h_values:
    lat_h = make_lattice("chain_1d", N, J=J, h=h)
    H = builder.build(lat_h)
    exact_data.append(solver.solve(H, lat_h))
```

## Phase 2: VQE Sweep

```python
from qmbp_simulation import HVACircuitBuilder, VQEOptimizer, make_lattice
from qmbp_simulation.models import VQEConfig

hva = HVACircuitBuilder()
base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
qc, theta = hva.create(N, p_layers, base_lattice)

config = VQEConfig(n_restarts=5, maxiter=1000, ftol=1e-14, enable_callbacks=True)
optimizer = VQEOptimizer(config)
vqe_results = optimizer.descending_sweep(h_values, qc, base_lattice, exact_data)
```

## Phase 3: MPNN Training

```python
import torch
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

dataset = build_graph_dataset(
    base_lattice, h_values,
    np.array([r.theta_opt for r in vqe_results]),
    np.array([d.ground_energy for d in exact_data]),
    fidelities=np.array([r.fidelity for r in vqe_results]),
    fidelity_threshold=0.93,
)

model = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=2*p_layers)
result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500)
```

## Phase 4: Deployment (MPNN Prediction)

```python
import torch
from torch_geometric.data import Data

model.eval()
edge_idx, coord = builder.build_graph_data(base_lattice)
x_test = torch.tensor(
    np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
    dtype=torch.float32,
)
test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
with torch.no_grad():
    theta_pred = model(test_graph).numpy().flatten()
```

## Full Pipeline (Recommended)

```python
from qmbp_simulation import PipelineRunner, make_lattice
from qmbp_simulation.models import VQEConfig

lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=2.0)
config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000)

runner = PipelineRunner(lattice=lattice, config=config, verbose=True)
results = runner.run_full(
    h_values=np.linspace(2.0, 0.5, 31),
    h_test=[1.5],
    mpnn_config={"hidden_dim": 128, "n_epochs": 6000, "patience": 500},
)

# Results contain: phase1, phase2, phase3, phase4, diagnostics
for deploy in results["phase4"]:
    print(f"h={deploy.h_test}: ΔE/gap={deploy.delta_e_over_gap:.4f}")
```

## Creating a New Script (CLI Pattern)

```python
#!/usr/bin/env python3
"""My new script description."""
from qmbp_simulation.framework import (
    create_base_parser, add_system_args, add_sweep_args, add_output_args,
    validate_descending_sweep, configure_logging, resolve_output_dir,
    save_experiment_result, build_result_envelope, ProgressReporter,
)

def main():
    parser = create_base_parser("My Script", epilog="Examples: ...")
    add_system_args(parser)
    add_sweep_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, debug=args.debug)
    h_values = validate_descending_sweep(args.h_values)
    output_dir = resolve_output_dir(args.output_dir)

    reporter = ProgressReporter("My Script")
    with reporter.phase(1, "Computing ground truth") as p:
        # ... do work ...
        p.detail("17 points computed")
    reporter.summary({"mean_de_gap": 0.014})

    # Save results
    result = build_result_envelope(
        config={"n_qubits": args.n_qubits},
        results=my_data,
        elapsed_s=reporter.total_elapsed_s,
    )
    save_experiment_result(result, experiment_id="X1")

if __name__ == "__main__":
    main()
```

## Creating a New Experiment

```python
from qmbp_simulation.framework import BaseExperiment, ExperimentConfig, ExperimentMetrics

class ExperimentX1(BaseExperiment):
    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="X1",
            name="My New Experiment",
            description="Testing hypothesis X",
            hypothesis="X should improve Y by Z%",
            system=SystemConfig(n_qubits=6, p_layers=2),
            seeds=[42, 43, 44],
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        # Your experiment logic here
        ...
```

Register in `experiments/<category>/__init__.py` and run with:
```bash
python scripts/run_experiment.py --exp X1 --verbose
```

## Benchmarking

```python
from qmbp_simulation.framework import BenchmarkSuite

# Programmatic usage
suite = BenchmarkSuite(n_qubits=[4, 6, 8, 10], n_repeats=5)
results = suite.run(components=["solver", "vqe", "mpnn"])
suite.print_summary(results)

# Or via CLI
# python scripts/benchmark.py --components solver vqe --n-qubits 4 6 8
```

## Result Comparison

```python
from qmbp_simulation.framework import ResultStore

store = ResultStore()
available = store.list_experiments()
comparisons = store.compare_experiments(available)

# Category-based filtering
optimization_exps = store.resolve_category("optimization")
scaling_exps = store.resolve_category("A")  # By letter prefix

# Noisy analysis
noisy_results = store.load_noisy_results()
correlations = store.analyze_noisy_correlations(noisy_results)
grouped = store.analyze_noisy_by_group(noisy_results, "n_layouts")
```

## Progress Reporting

```python
from qmbp_simulation.framework import ProgressReporter

reporter = ProgressReporter("Pipeline N=10")
with reporter.phase(1, "Exact diagonalization") as p:
    exact_data = run_exact_diag_sweep(h_values, n_qubits=10)
    p.detail(f"{len(exact_data)} points, gap_min={min(r.gap for r in exact_data):.4f}")

with reporter.phase(2, "VQE optimization") as p:
    vqe_results = optimizer.descending_sweep(...)
    p.detail(f"mean fidelity = {np.mean([r.fidelity for r in vqe_results]):.4f}")

reporter.summary({"total_points": len(exact_data), "elapsed_s": reporter.total_elapsed_s})
```

## Noisy Simulation (ZNE)

```python
from qmbp_simulation.execution import (
    NoisyEstimatorConfig, build_adjacency, find_layouts_bfs,
    compute_circuit_ces, select_layouts_by_circuit_ces,
    noisy_estimate, linear_zne,
)
from qiskit_ibm_runtime.fake_provider import FakeTorino

backend = FakeTorino()
config = NoisyEstimatorConfig(shots=16384, seed_simulator=42)

# Layout selection
adj = build_adjacency(backend)
candidates = find_layouts_bfs(adj, n_qubits=6, n_candidates=30)
layouts = select_layouts_by_circuit_ces(circuit, backend, candidates, n_select=3)

# Noisy estimation per layout
energies = []
for layout in layouts:
    e = noisy_estimate(circuit, hamiltonian, theta, backend, layout, config)
    energies.append(e)

# ZNE extrapolation
zne_result = linear_zne(ces_values, energies)
# zne_result.extrapolated_energy, zne_result.r_squared, zne_result.gain
```

## Primitives V2 Patterns

```python
# LOCAL simulation (noiseless)
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
bound_qc = circuit.assign_parameters(theta)
energy = float(estimator.run([(bound_qc, hamiltonian)]).result()[0].data.evs)

# HARDWARE execution (IBM Torino)
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.backend("ibm_torino")

pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(circuit.assign_parameters(theta))
isa_obs = [obs.apply_layout(isa_qc.layout) for obs in observables]

estimator = EstimatorV2(mode=backend)
estimator.options.dynamical_decoupling.enable = True
estimator.options.dynamical_decoupling.sequence_type = "XpXm"
estimator.options.twirling.enable_gates = True
estimator.options.resilience.zne_mitigation = True
estimator.options.resilience.zne.noise_factors = [1, 2, 3]

job = estimator.run([(isa_qc, isa_obs)])
result = job.result()
```

## Dataset Save/Load

```python
from qmbp_simulation import save_phase12_dataset, load_phase12_dataset

save_phase12_dataset(
    "phase12_N6_p2.npz",
    h_values=h_values, J=J, n_qubits=N, p_layers=p,
    ground_energies=..., gaps=..., mag_x=..., corr_zz=...,
    theta_opt=..., vqe_energies=..., fidelities=...,
)

data = load_phase12_dataset("phase12_N6_p2.npz")
```


## Decision Guide: Which Tool to Use

| I need to... | Use this | Example |
|---|---|---|
| Validate a script before running | `scripts/preflight.py` | `--from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py` |
| Run a registered experiment | `scripts/run_experiment.py --exp <ID>` | `--exp B4 --verbose` |
| Run the full 4-phase pipeline | `scripts/run_pipeline.py` | `--n-qubits 6 --p 2` |
| Compare experiment results | `scripts/compare.py --all` | `--category optimization` |
| Benchmark performance | `scripts/benchmark.py` | `--components solver vqe` |
| Validate package works | `tests/smoke_test.py` | (no args needed) |
| Compute ground truth only | `run_exact_diag_sweep()` | See pipeline module |
| Run pipeline programmatically | `PipelineRunner.run_full()` | See Example 1 below |
| Save results consistently | `save_experiment_result()` | See result_io module |
| Parse CLI args (new script) | `create_base_parser()` + `add_*_args()` | See cli module |
| Report progress to console | `ProgressReporter` | See logging module |
| Detect performance regressions | `BenchmarkSuite` | See benchmarking module |
| Query/compare past results | `ResultStore` | See result_store module |

## When Writing a New Script

Always follow this pattern:

1. **Import from `qmbp_simulation.framework.cli`** for argument parsing
2. **Use `validate_descending_sweep()`** for h-values (enforces descending order)
3. **Use `configure_logging()`** for consistent log levels
4. **Use `ProgressReporter`** for console output (not raw `print()`)
5. **Use `build_result_envelope()` + `save_experiment_result()`** for saving
6. **Never use `sys.path.insert()`** — the package is installed
7. **Run preflight before first execution** — `python scripts/preflight.py --from-script <your_script>`

## When Writing a New Experiment

1. Inherit from `BaseExperiment`
2. Implement `default_config()` → returns `ExperimentConfig`
3. Implement `run_single(seed)` → returns `list[ExperimentMetrics]`
4. Register in `experiments/<category>/__init__.py`
5. Add to `EXPERIMENT_REGISTRY` in `scripts/run_experiment.py`
6. Run with `python scripts/run_experiment.py --exp <ID>`

## When Analyzing Results

1. Use `ResultStore()` to discover and load results
2. Use `store.compare_experiments(ids)` for verdict-based comparison
3. Use `store.resolve_category("optimization")` to filter by category
4. Use `store.load_noisy_results()` + `analyze_noisy_correlations()` for ZNE analysis
5. Use `store.format_experiment_table()` for human-readable output

## Framework Module Cheat Sheet

### `framework/cli.py`
```python
from qmbp_simulation.framework import (
    create_base_parser,       # → argparse.ArgumentParser
    add_system_args,          # Adds: --n-qubits, --topology, --J, --periodic, --p
    add_sweep_args,           # Adds: --h-values, --h-test
    add_vqe_args,             # Adds: --n-restarts, --maxiter, --sigma
    add_mpnn_args,            # Adds: --hidden-dim, --n-layers, --n-epochs, --lr, --patience
    add_output_args,          # Adds: --output-dir, --verbose, --debug
    validate_descending_sweep,  # list[float] | None → np.ndarray (descending)
    validate_system_size,     # (n_qubits, p_layers) → list[str] warnings
    configure_logging,        # (verbose, debug) → sets logging level
    build_mpnn_config_dict,   # args → dict for PipelineRunner
    resolve_output_dir,       # str → Path (creates dir)
)
```

### `framework/result_io.py`
```python
from qmbp_simulation.framework import (
    generate_timestamp,        # → "20260526_143000"
    build_result_envelope,     # (config, results, summary, elapsed_s) → dict
    save_experiment_result,    # (data, exp_id) → Path (results/experiments/exp_{id}/run_{ts}.json)
    save_pipeline_result,      # (data, output_dir) → Path (pipeline_run_{ts}.json)
    save_benchmark_result,     # (data, output_path) → Path
    load_result,               # (path) → dict
)
```

### `framework/result_store.py`
```python
from qmbp_simulation.framework import ResultStore, CATEGORY_MAP

store = ResultStore()                              # Default: results/experiments/
store.list_experiments()                           # → ["A3", "B4", "D1", ...]
store.list_categories()                            # → {"optimization": ["B", "C3", "G4"], ...}
store.resolve_category("optimization")             # → ["B1", "B2", "B4", "C3", "G4"]
store.load_latest("B4")                            # → dict (most recent run)
store.load_all_runs("B4")                          # → list[dict]
store.compare_experiments(["B4", "D1", "F3"])      # → list[dict] with verdicts
store.format_experiment_table(comparisons)          # → str (aligned table)
store.load_noisy_results()                         # → list[dict]
store.analyze_noisy_correlations(results)           # → dict with R², gain stats
store.analyze_noisy_by_group(results, "n_layouts") # → dict[val, stats]
```

### `framework/benchmarking.py`
```python
from qmbp_simulation.framework import BenchmarkSuite, BenchmarkResult

suite = BenchmarkSuite(n_qubits=[4, 6, 8], n_repeats=3, verbose=True)
results = suite.run(components=["solver", "vqe", "circuit", "mpnn"])
suite.print_summary(results)
data = suite.to_dict(results)  # → JSON-serializable dict
```

### `framework/logging.py`
```python
from qmbp_simulation.framework import ProgressReporter, StructuredLogger

# ProgressReporter — console output for multi-phase tasks
reporter = ProgressReporter("Title")
with reporter.phase(1, "Description") as p:
    p.detail("message")
reporter.checkpoint("label", "value")
reporter.summary({"key": value})
reporter.total_elapsed_s  # float

# StructuredLogger — machine-parseable event log
slog = StructuredLogger("A3")
slog.log("vqe_start", seed=42, h_value=1.5)
slog.start_timer("vqe_point")
slog.stop_timer("vqe_point", event_type="vqe_complete", data={"energy": -5.2})
slog.save(Path("results/experiments/exp_a3/log.json"))
```

### `framework/preflight.py`
```python
from qmbp_simulation.framework import (
    PreflightChecker,          # Main checker class
    PreflightReport,           # Aggregated results
    VariantSpec,               # Minimal variant specification
    P1_VALID_REGIME,           # dict[(topo, N), threshold] for p=1
    P2_VALID_REGIME,           # dict[(topo, N), threshold] for p=2
    get_valid_regime,          # (p) → regime dict
    get_regime_threshold,      # (topo, N, p) → float threshold
    specs_from_pipeline_variants,  # list[PipelineVariant] → list[VariantSpec]
    specs_from_json,           # Path → list[VariantSpec]
    specs_from_variant_runner, # (build_fn, build_fn, build_fn, N) → list[VariantSpec]
)

# Programmatic usage
specs = [VariantSpec(id="V1", topology="chain_1d", n_qubits=10, p=1,
                     h_values=[4.0, 3.5, 3.0], h_test=[2.75],
                     output_dir="results/v1")]
checker = PreflightChecker(specs, project_root=Path("."))
report = checker.run_all(verbose=True)
# report.has_errors → bool
# report.errors → list[Issue]
# report.warnings → list[Issue]

# From variant runner builders
specs = specs_from_variant_runner(
    build_noiseless_variants, build_noisy_variants,
    build_extended_variants, n_qubits=10,
)
checker = PreflightChecker(specs, strict=True)  # warnings → errors
report = checker.run_all()
```

**CLI usage:**
```bash
# Validate before running
python scripts/preflight.py --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

# Strict mode (CI — warnings become errors)
python scripts/preflight.py --from-script my_script.py --strict

# Quiet mode (summary only)
python scripts/preflight.py --from-script my_script.py --quiet

# From JSON
python scripts/preflight.py --from-json variants.json

# Via Makefile
make preflight SCRIPT=scripts/experiment_runners/run_p1_pipeline_variants_r2.py
```

**Checks performed (9 total):**
1. `script_exists` — Pipeline script referenced by variants exists
2. `minimum_config` — h_values, h_test, topology, n_qubits are defined
3. `h_test_unseen` — h_test NOT in training set (data leakage)
4. `h_test_valid_regime` — h_test within valid regime for topology/N/p
5. `h_values_valid_regime` — Training points within valid regime
6. `interpolation` — h_test within training range (not extrapolation)
7. `descending_sweep` — h_values in descending order (warm-start)
8. `duplicate_ids` — No duplicate variant IDs
9. `output_fresh` — Output directories don't already have results

**Hook integration:** The `preflight-before-run` hook automatically triggers
preflight validation before any shell command that executes a variant runner
script. If errors are found, execution is blocked.

### `pipeline/runner.py`
```python
from qmbp_simulation.pipeline import PipelineRunner, run_exact_diag_sweep

# Standalone Phase 1
exact_data = run_exact_diag_sweep(h_values, n_qubits=6, topology="chain_1d", J=1.0)

# Full pipeline
runner = PipelineRunner(lattice=lattice, config=vqe_config, verbose=True)
results = runner.run_full(h_values=h_values, h_test=[1.5], mpnn_config={...})
# results keys: "phase1", "phase2", "phase3", "phase4", "diagnostics"
```


## Simulation Modes

### Noiseless (default — all thesis results)
```python
from qmbp_simulation.execution import NoiselessBackend
backend = NoiselessBackend()  # StatevectorEstimator, exact
# This is the default for PipelineRunner and VQEOptimizer
```

### Noisy (Gaussian shot noise approximation)
```python
from qmbp_simulation.execution import NoisyBackend
backend = NoisyBackend(shots=8192, seed_simulator=42)
# Energy = exact + N(0, 1/√shots) — fast, no qiskit-aer needed
```

### Noisy (Full noise model via FakeTorino)
```python
from qmbp_simulation.execution import NoisyBackend
from qiskit_ibm_runtime.fake_provider import FakeTorino

fake = FakeTorino()
backend = NoisyBackend(shots=16384, noise_model=fake.noise_model, seed_simulator=42)
# Full AerSimulator with device noise — requires qiskit-aer
```

### ZNE Deployment (inhomogeneous zero-noise extrapolation)
```python
from qmbp_simulation.execution import (
    NoisyEstimatorConfig, build_adjacency, find_layouts_bfs,
    select_layouts_by_circuit_ces, run_zne_deployment,
)
from qiskit_ibm_runtime.fake_provider import FakeTorino

backend = FakeTorino()
config = NoisyEstimatorConfig(shots=16384, seed_simulator=42)

# 1. Find layouts
adj = build_adjacency(backend)
candidates = find_layouts_bfs(adj, n_qubits=N, n_candidates=30)

# 2. Select by circuit CES
layout_sel = select_layouts_by_circuit_ces(bound_circuit, backend, candidates, n_select=3)

# 3. Run ZNE
result = run_zne_deployment(bound_circuit, H, backend, layout_sel, config, N)
# result.energy_zne.extrapolated, result.energy_zne.r_squared
```

### Passing Backend to VQEOptimizer or PipelineRunner
```python
# VQEOptimizer with noisy backend
from qmbp_simulation import VQEOptimizer
from qmbp_simulation.execution import NoisyBackend

optimizer = VQEOptimizer(config=vqe_config, backend=NoisyBackend(shots=8192))
vqe_results = optimizer.descending_sweep(h_values, circuit, lattice, exact_data)

# PipelineRunner with noisy backend
from qmbp_simulation import PipelineRunner
runner = PipelineRunner(lattice=lattice, config=vqe_config, backend=NoisyBackend(shots=8192))
```

## Available Techniques (experiments/helpers/)

| Technique | Module | Key function | When to use |
|---|---|---|---|
| Parameter freezing | `parameter_freezing.py` | `frozen_vqe()`, `analyze_parameter_activity()` | h≥1.5, reduce cost 75% |
| Hessian check | `hessian_restart.py` | `hessian_guided_vqe()` | Verify minimum is genuine |
| Analytical init | `analytical_init.py` | `analytical_init_p2()` | h>>1 (fast but wrong basin) |
| Physics loss | `physics_loss.py` | `PhysicsInformedLoss` | MPNN training regularization |
| DyPP | `dypp.py` | `dypp_linear()`, `dypp_quadratic()` | Extrapolate θ (rejected: 8-13% only) |
| Sign equivariance | `sign_equivariant.py` | `canonicalize_sign()` | N=20 Z₂ symmetry |
| Active learning | `active_learning.py` | `select_next_point()` | Ensemble-based point selection |

### Importing techniques
```python
from experiments.helpers import (
    frozen_vqe, analyze_parameter_activity,
    hessian_guided_vqe, standard_multistart_vqe,
    analytical_init_p1, analytical_init_p2,
    PhysicsInformedLoss, evaluate_energy_batch,
    dypp_linear, dypp_quadratic, dypp_predict,
    canonicalize_sign, detect_sign_inconsistency,
    compute_ensemble_uncertainty, select_next_point,
)
```

## Choosing the Right Approach

| Goal | Backend | Technique | Command |
|---|---|---|---|
| Standard pipeline validation | `NoiselessBackend` | 5 restarts | `run_pipeline.py --n-qubits 6` |
| N=10 pipeline | `NoiselessBackend` | hidden=128, patience=500 | `run_pipeline.py --n-qubits 10 --hidden-dim 128` |
| Check for saddle points | `NoiselessBackend` | Hessian | `run_experiment.py --exp B4` |
| Reduce cost at h≥1.5 | `NoiselessBackend` | Freezing | `run_experiment.py --exp B2` |
| Detect phase transition | `NoiselessBackend` | Weight gradient | `run_experiment.py --exp D1` |
| Test ZNE at N=6 | `NoisyBackend` (FakeTorino) | ZNE + layouts | `run_zne_deployment()` |
| Landscape analysis | `NoiselessBackend` | Random sampling | `run_experiment.py --exp F3` |
| Scaling law | `NoiselessBackend` | Multi-N | `run_experiment.py --exp A3` |
| Hardware deployment | `HardwareBackend` | DD+twirling+TREX+ZNE | Pending IBM integration |
