# Hybrid GNN-HVA Framework for Topological Phase Characterization

## Overview

Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project
accelerates Variational Quantum Eigensolvers (VQE) for quantum phase characterization
using a predictive hybrid architecture: a classical Graph Neural Network (GNN) trained
on Tensor Network data provides "Intelligent Warm-Start" initialization for a shallow,
physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

**Key constraint** (Mele et al., Nature Physics 2026): Non-unital noise truncates
circuits to O(log n). All HVA circuits are shallow (p ≤ 2 layers), enforced by
pre-commit hooks.

## Quick Start

```bash
# Clone and setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"

# Verify installation
python -c "from qmbp_simulation import HamiltonianBuilder, make_lattice; print('OK')"

# Run smoke test (N=4, p=1, <30s)
python tests/smoke_test.py

# Run full test suite
make test

# Run lint + tests + smoke in one command
make check-full

# Project health report
make health

# Generate thesis figures (PDF, 300dpi, no titles)
make figures-thesis
# → documentation/thesis_figures/ (21 PDF vector files)

# Generate analysis figures (PNG, with titles)
make figures

# Coverage report
make coverage
```

## Package Structure

```
project-root/
├── src/
│   └── qmbp_simulation/           # Installable package (Zone 1: Framework)
│       ├── __init__.py             # Package-level re-exports
│       ├── utils/                  # Seed, JSON, timing (no internal deps)
│       ├── models/                 # LatticeConfig, Hamiltonians, data models
│       ├── solvers/                # ExactDiag, DMRG
│       ├── circuits/               # HVA builder
│       ├── execution/              # Backend ABC + noiseless/noisy/hardware
│       ├── optimizers/             # VQE, SPSA
│       ├── predictors/             # MPNN model, training, checkpoints
│       ├── pipeline/               # Orchestration, dataset I/O
│       ├── framework/              # Experiment engine, CLI, benchmarking, result I/O
│       └── analysis/               # Gradient analysis, diagnostics, landscape
├── experiments/                    # Experiment scripts (Zone 2: Consumers)
│   ├── optimization/               # VQE technique experiments (B1, B2, B4, C3, G4)
│   ├── scaling/                    # Finite-size scaling (A3, G3)
│   ├── landscape/                  # Hessian, fluctuation (F1, F3)
│   ├── predictor/                  # MPNN enhancements (C1, D1, E3, G1, G2, G5)
│   ├── hardware/                   # Hardware deployment
│   ├── generalization/             # Model-agnostic tests (E4)
│   └── helpers/                    # DyPP, sign canon, freezing, etc.
├── scripts/                        # CLI entry points (Zone 2: Consumers)
│   ├── experiment_runners/         # Pipeline & variant runners
│   │   ├── experiment_run_helpers/
│   │   │   ├── run_experiment.py   # Run experiments by ID
│   │   │   └── run_pipeline.py     # Full 4-phase pipeline
│   │   ├── run_thesis_variants-*.py # Topology-specific variant runners
│   │   └── run_p1_pipeline_variants*.py # p=1 multi-topology variants
│   ├── compare.py                  # Shim → project_health/compare.py
│   ├── preflight.py                # Pre-flight validation
│   ├── smoke_test.py               # Quick validation (<30s)
│   └── benchmark.py                # Performance benchmarking
├── project_health/                 # Phase 4 tooling (unified analysis)
│   ├── __init__.py                 # Public API
│   ├── __main__.py                 # CLI (python -m project_health)
│   ├── engine.py                   # Core orchestration
│   ├── coverage.py                 # Coverage gaps + analytics
│   ├── models.py                   # Typed data models
│   ├── reporter.py                 # Output formatters
│   ├── state.py                    # Delta tracking persistence
│   ├── figures.py                  # Matplotlib figure generation
│   ├── compare.py                  # Cross-experiment comparison CLI
│   ├── analysis/                   # Analysis scripts (canonical location)
│   │   ├── diagnose.py             # Automated failure root cause analysis
│   │   ├── scan_coverage.py        # Coverage scanner + gap analysis
│   │   ├── verify_claims.py        # Thesis claim verification
│   │   ├── verify_results.py       # Pipeline result verification against specs
│   │   ├── validate_s_series.py    # S-series experiment validation
│   │   ├── heisenberg_summary.py   # Heisenberg XXZ cross-N comparison
│   │   ├── sanity_check.py         # 26 automated checks (physics + data integrity)
│   │   ├── scaling_analyzer.py     # MPS scaling law validation (N=40-120)
│   │   ├── scaling_extensions_analyzer.py  # E5: bond-dim, HE, NLCE analysis
│   │   └── statistical_tests.py    # Shared statistical test utilities
│   └── digest/                     # Result digest & scanning
│       ├── scanner.py              # ResultScanner (parse all results)
│       ├── formatters.py           # Output formatting
│       └── models.py               # Result data models
├── tests/                          # pytest suite
├── results/                        # Experiment outputs (gitignored)
├── analysis/                       # Cross-experiment analysis & thesis figures
│   ├── FINDINGS_INDEX.md           # Master index (36 findings with confidence)
│   ├── 10_key_findings_corrected.md # Corrected findings post-verification
│   ├── scripts/                    # Backward-compat shims (delegate to project_health/)
│   ├── figures/                    # Thesis-quality PNG figures
│   └── raw_data/                   # Parsed JSON for analysis scripts
├── documentation/                  # Thesis docs, binnacles, bibliography
└── pyproject.toml                  # Package config, Ruff, pytest
```

## Scripts & Tools Reference

All scripts live in `scripts/` and use the framework via `from qmbp_simulation import ...`.

### `scripts/experiment_runners/experiment_run_helpers/run_experiment.py` — Experiment Runner

Unified CLI for running any registered experiment by ID. Experiments inherit from
`BaseExperiment` and follow the lifecycle: `setup() → run() → analyze() → report() → save()`.

```bash
python scripts/experiment_runners/experiment_run_helpers/run_experiment.py --list
python scripts/experiment_runners/experiment_run_helpers/run_experiment.py --exp B4
python scripts/experiment_runners/experiment_run_helpers/run_experiment.py --exp B4 D1 F1
python scripts/experiment_runners/experiment_run_helpers/run_experiment.py --exp A3 --seeds 42 43 --verbose
python scripts/experiment_runners/experiment_run_helpers/run_experiment.py --exp B1 --n-qubits 10 --p 1
```

### `scripts/experiment_runners/experiment_run_helpers/run_pipeline.py` — Full Pipeline

Executes the complete 4-phase pipeline (exact diag → VQE → MPNN → deployment).
Uses the framework's `PipelineRunner`, CLI argument groups, and result I/O.

```bash
python scripts/experiment_runners/experiment_run_helpers/run_pipeline.py --n-qubits 6 --p 2
python scripts/experiment_runners/experiment_run_helpers/run_pipeline.py --n-qubits 10 --h-values 2.0 1.75 1.5 1.25
python scripts/experiment_runners/experiment_run_helpers/run_pipeline.py --n-qubits 6 --output-dir results/my_run --verbose
python scripts/experiment_runners/experiment_run_helpers/run_pipeline.py --n-qubits 6 --skip-phase3 --skip-phase4
```

### `scripts/compare.py` — Result Comparison

> **Canonical location**: `project_health/compare.py`
> The `scripts/compare.py` shim delegates to the new location for backward compatibility.

Evaluates experiments against their own success criteria (not a blanket baseline).
Verdicts: `confirmed` (hypothesis holds), `rejected` (disproved = valid finding),
`failed` (unexpected). Uses `ResultStore` from the framework.

```bash
python scripts/compare.py --all                      # Compare all experiments
python scripts/compare.py --exp G1 G5 B4             # Compare specific ones
python scripts/compare.py --category optimization    # By category name
python scripts/compare.py --noisy                    # Analyze ZNE results
python scripts/compare.py --noisy --group-by n_layouts
python scripts/compare.py --all --json output.json   # Save JSON output
```

### `scripts/benchmark.py` — Performance Benchmarking

Benchmarks core pipeline components (solver, circuit, VQE, MPNN) at various system
sizes. Uses `BenchmarkSuite` from the framework for programmatic access.

```bash
python scripts/benchmark.py                          # Run all benchmarks
python scripts/benchmark.py --components solver vqe  # Specific components
python scripts/benchmark.py --n-qubits 4 6 8 10     # Custom sizes
python scripts/benchmark.py --output bench.json      # Save results
```

### `scripts/digest/` — Result Digest & Analysis

> **Canonical location**: `project_health/digest/`

Extracts key knowledge from all experiment results by kind (noiseless, noisy, experiment).
Supports filtering, grouping, statistical analysis, outlier detection, and side-by-side
comparison. Lightweight (no torch import). See [`project_health/digest/README.md`](project_health/digest/README.md)
for full documentation.

```bash
python -m project_health.digest --kind noiseless --group-by topology
python -m project_health.digest --kind noisy --group-by n_qubits
python -m project_health.digest --kind experiment --sort verdict --verbose
python -m project_health.digest --stats --topology ladder
python -m project_health.digest --outliers
python -m project_health.digest --compare variants_N10_ladder variants_N10_triangular
```

### `tests/smoke_test.py` — Quick Validation

Imports all submodules and runs a minimal pipeline (N=4, p=1, 3 h-points).
Verifies ΔE/gap < 5%. Should complete in under 30 seconds.

```bash
python tests/smoke_test.py
```

### `scripts/preflight.py` — Pre-flight Validation

Validates variant runner configurations before execution. Always run before executing
a variant runner script for the first time. A Kiro hook enforces this automatically.

```bash
python scripts/preflight.py --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py
python scripts/preflight.py --from-script my_script.py --strict
python scripts/preflight.py --from-json variants.json
# Or via Makefile:
make preflight SCRIPT=scripts/experiment_runners/run_p1_pipeline_variants_r2.py
```

Checks: h_test not in training set, h_test within valid regime, descending order,
interpolation (not extrapolation), no duplicate IDs, fresh output directories.

### `scripts/experiment_runners/bond_resolved/run_scaling_extensions.py` — Scaling Extensions (E5)

Multi-section validation runner for N=120 bond-dimension test, Hamiltonian engineering
comparison, and NLCE (Numerical Linked-Cluster Expansion) thermodynamic limit.

```bash
# Dry run (list sections)
python scripts/experiment_runners/bond_resolved/run_scaling_extensions.py --dry-run

# Individual sections
python scripts/experiment_runners/bond_resolved/run_scaling_extensions.py --section 1   # Bond dim N=120
python scripts/experiment_runners/bond_resolved/run_scaling_extensions.py --section 3   # HE comparison
python scripts/experiment_runners/bond_resolved/run_scaling_extensions.py --section 4 5 # NLCE (TFIM + frustrated)

# Full suite
python scripts/experiment_runners/bond_resolved/run_scaling_extensions.py

# Analysis (post-execution)
make extensions                                                          # Quick report
make cross-topology                                                      # Cross-topology results
python -m project_health.analysis.scaling_extensions_analyzer --verbose --cross-check --thesis-tables
```

### NLCE Module (`qmbp_simulation.analysis.nlce`)

Modular Numerical Linked-Cluster Expansion framework for 1D spin systems.
Decomposes thermodynamic-limit properties as sums over finite clusters with
Euler subtraction. Pluggable cluster solver (exact diag, DMRG, or VQE).

```python
from qmbp_simulation.analysis import NLCERunner, NLCEConfig, tfim_analytical_energy_per_site

config = NLCEConfig(l_max=10, model="tfim")
runner = NLCERunner(config)
result = runner.compute(h=2.0)
print(f"E/N = {result.energy_per_site:.8f}, converged={result.converged}")

# Frustrated TFIM (novel result — no analytical formula)
config_f = NLCEConfig(l_max=8, model="tfim_frustrated", J2=0.5)
runner_f = NLCERunner(config_f)
results = runner_f.compute_sweep([1.5, 2.0, 3.0, 4.0])
```

## Framework Modules (for programmatic use)

The `src/qmbp_simulation/framework/` subpackage provides reusable infrastructure:

| Module | Purpose | Key exports |
|--------|---------|-------------|
| `cli.py` | Shared CLI argument groups | `create_base_parser`, `add_system_args`, `add_sweep_args`, `add_vqe_args`, `add_mpnn_args`, `add_output_args`, `validate_descending_sweep`, `configure_logging` |
| `result_io.py` | Standardized result saving | `save_experiment_result`, `save_pipeline_result`, `save_benchmark_result`, `build_result_envelope`, `load_result` |
| `result_store.py` | Result querying & comparison | `ResultStore`, `CATEGORY_MAP` |
| `benchmarking.py` | Performance regression suite | `BenchmarkSuite`, `BenchmarkResult` |
| `logging.py` | Structured events + progress | `StructuredLogger`, `ProgressReporter` |
| `config.py` | Typed experiment configs | `ExperimentConfig`, `SystemConfig`, `VQEConfig`, `MPNNConfig` |
| `base.py` | Experiment lifecycle | `BaseExperiment` |
| `metrics.py` | Result dataclasses | `ExperimentMetrics`, `WarmColdComparison` |

### Usage Examples

Below are concrete, copy-paste-ready examples for every common task.

---

#### Example 1: Run the Full Pipeline End-to-End (Programmatic)

```python
"""Run the complete 4-phase pipeline for N=6, p=2 and check deployment."""
import numpy as np
from qmbp_simulation import PipelineRunner, make_lattice
from qmbp_simulation.models import VQEConfig

# 1. Configure
lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=2.0)
vqe_config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000)

# 2. Run
runner = PipelineRunner(lattice=lattice, config=vqe_config, verbose=True)
results = runner.run_full(
    h_values=np.linspace(2.0, 1.25, 16),  # Must be descending
    h_test=[1.5],                          # Unseen deployment point
    mpnn_config={"hidden_dim": 128, "n_epochs": 6000, "patience": 500},
)

# 3. Check results
for deploy in results["phase4"]:
    status = "PASS" if deploy.delta_e_over_gap < 0.05 else "FAIL"
    print(f"h={deploy.h_test}: ΔE/gap={deploy.delta_e_over_gap:.4f} [{status}]")
    print(f"  Phase: {deploy.phase_label}, Energy: {deploy.predicted_energy:.6f}")
    print(f"  Observables: <X>={deploy.mag_x_pred:.4f}, <ZZ>={deploy.corr_zz_pred:.4f}")

# 4. Access diagnostics (always-on)
diag = results["diagnostics"]
print(f"Phase 1 elapsed: {diag.get('phase1', {}).get('elapsed_s', 0):.1f}s")
```

---

#### Example 2: Run Only Phase 1 (Exact Diagonalization)

```python
"""Compute ground truth energies without VQE or MPNN."""
import numpy as np
from qmbp_simulation.pipeline import run_exact_diag_sweep

h_values = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
exact_data = run_exact_diag_sweep(h_values, n_qubits=10, topology="chain_1d", J=1.0)

for r in exact_data:
    print(f"h={r.h_value:.2f}: E₀={r.ground_energy:.6f}, gap={r.gap:.4f}, "
          f"<X>={r.mag_x:.4f}, <ZZ>={r.corr_zz:.4f}")
```

---

#### Example 3: Create a New CLI Script Using Framework Tools

```python
#!/usr/bin/env python3
"""Example: custom analysis script using framework CLI and result I/O."""
from qmbp_simulation.framework import (
    create_base_parser,
    add_system_args,
    add_sweep_args,
    add_output_args,
    validate_descending_sweep,
    configure_logging,
    resolve_output_dir,
    build_result_envelope,
    save_experiment_result,
    ProgressReporter,
)
from qmbp_simulation.pipeline import run_exact_diag_sweep
import numpy as np


def main():
    # 1. Parse arguments (reusable groups — no boilerplate)
    parser = create_base_parser(
        "Gap Analysis",
        epilog="Example: %(prog)s --n-qubits 10 --h-values 2.0 1.5 1.0",
    )
    add_system_args(parser)
    add_sweep_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    # 2. Configure
    configure_logging(verbose=args.verbose, debug=args.debug)
    h_values = validate_descending_sweep(args.h_values)
    output_dir = resolve_output_dir(args.output_dir)

    # 3. Execute with progress reporting
    reporter = ProgressReporter(f"Gap Analysis N={args.n_qubits}")

    with reporter.phase(1, "Computing ground truth") as p:
        exact_data = run_exact_diag_sweep(
            h_values, n_qubits=args.n_qubits, topology=args.topology, J=args.J
        )
        p.detail(f"{len(exact_data)} points computed")

    with reporter.phase(2, "Analyzing gaps") as p:
        gaps = np.array([r.gap for r in exact_data])
        gap_min_idx = np.argmin(gaps)
        p.detail(f"Min gap = {gaps[gap_min_idx]:.6f} at h = {h_values[gap_min_idx]:.2f}")

    # 4. Save results (standardized envelope)
    result = build_result_envelope(
        config={"n_qubits": args.n_qubits, "topology": args.topology, "J": args.J},
        results={"h_values": h_values.tolist(), "gaps": gaps.tolist()},
        summary={"gap_min": float(gaps.min()), "h_at_gap_min": float(h_values[gap_min_idx])},
        elapsed_s=reporter.total_elapsed_s,
    )
    path = save_experiment_result(result, experiment_id="gap_analysis")
    reporter.summary({"gap_min": float(gaps.min()), "saved_to": str(path)})


if __name__ == "__main__":
    main()
```

---

#### Example 4: Create a New Experiment (BaseExperiment Pattern)

```python
"""Example: experiment that tests VQE convergence at different restart counts."""
from qmbp_simulation.framework import BaseExperiment, ExperimentConfig, ExperimentMetrics
from qmbp_simulation.framework.config import SystemConfig, VQEConfig
import numpy as np


class ExperimentX1(BaseExperiment):
    """Test how n_restarts affects VQE convergence quality."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="X1",
            category="optimization",
            description="VQE restart count vs convergence quality",
            hypothesis="5 restarts sufficient for ΔE/gap < 5% at N=6",
            system=SystemConfig(n_qubits=6, p_layers=2, h_values=[2.0, 1.75, 1.5, 1.25]),
            vqe=VQEConfig(optimizer="L-BFGS-B", n_restarts=5, maxiter=1000),
            seeds=[42, 43, 44],
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run one seed — test 1, 3, 5, 7 restarts."""
        from qmbp_simulation import (
            HamiltonianBuilder, make_lattice, ClassicalSolver,
            HVACircuitBuilder, VQEOptimizer,
        )
        from qmbp_simulation.models import VQEConfig as ModelVQEConfig

        results = []
        for n_restarts in [1, 3, 5, 7]:
            # Setup
            lattice = make_lattice("chain_1d", self.config.system.n_qubits, J=1.0, h=2.0)
            builder = HamiltonianBuilder()
            solver = ClassicalSolver()
            hva = HVACircuitBuilder()
            circuit, _ = hva.create(self.config.system.n_qubits, 2, lattice)

            # Phase 1
            h_values = np.array(self.config.system.h_values)
            exact_data = []
            for h in h_values:
                lat_h = make_lattice("chain_1d", self.config.system.n_qubits, J=1.0, h=h)
                exact_data.append(solver.solve(builder.build(lat_h), lat_h))

            # Phase 2
            config = ModelVQEConfig(p_layers=2, n_restarts=n_restarts, maxiter=1000)
            optimizer = VQEOptimizer(config=config)
            vqe_results = optimizer.descending_sweep(h_values, circuit, lattice, exact_data)

            # Metrics
            de_gaps = [
                abs(v.energy - e.ground_energy) / max(e.gap, 1e-10)
                for v, e in zip(vqe_results, exact_data)
            ]
            results.append(ExperimentMetrics(
                seed=seed,
                h_value=0.0,  # Aggregate
                delta_e_over_gap=float(np.mean(de_gaps)),
                pass_threshold=0.05,
                extra={"n_restarts": n_restarts, "per_h_de_gap": de_gaps},
            ))
        return results
```

Register in `experiments/optimization/__init__.py` and run:
```bash
python scripts/run_experiment.py --exp X1 --verbose
```

---

#### Example 5: Compare Experiment Results Programmatically

```python
"""Load and compare experiment results without the CLI."""
from qmbp_simulation.framework import ResultStore, CATEGORY_MAP

store = ResultStore()

# List what's available
available = store.list_experiments()
print(f"Available experiments: {available}")

# Compare all
comparisons = store.compare_experiments(available)
print(store.format_experiment_table(comparisons))

# Filter by category
optimization_exps = store.resolve_category("optimization", available)
scaling_exps = store.resolve_category("scaling", available)
predictor_exps = store.resolve_category("predictor", available)

# Load a specific experiment's latest result
result = store.load_latest("B4")
if result:
    summary = result.get("analysis", {}).get("summary", {})
    print(f"B4: pass_rate={summary.get('pass_rate', 0):.0%}")

# Analyze noisy/ZNE results
noisy = store.load_noisy_results()
if noisy:
    stats = store.analyze_noisy_correlations(noisy)
    print(f"ZNE: mean R²={stats['mean_r2']:.4f}, helps {stats['pct_helps']:.0f}%")
    grouped = store.analyze_noisy_by_group(noisy, "n_layouts")
    print(store.format_noisy_table(grouped, "n_layouts"))
```

---

#### Example 6: Benchmark Performance and Detect Regressions

```python
"""Run benchmarks programmatically and compare against previous results."""
from qmbp_simulation.framework import BenchmarkSuite
from qmbp_simulation.framework.result_io import save_benchmark_result, load_result
from pathlib import Path

# Run current benchmarks
suite = BenchmarkSuite(n_qubits=[4, 6, 8, 10], n_repeats=5, verbose=True)
results = suite.run(components=["solver", "vqe", "circuit", "mpnn"])
suite.print_summary(results)

# Save for future comparison
data = suite.to_dict(results)
save_benchmark_result(data, output_path=Path("results/benchmarks/latest.json"))

# Compare against previous run (regression detection)
try:
    previous = load_result(Path("results/benchmarks/baseline.json"))
    for curr in results:
        for prev in previous["results"]:
            if prev["component"] == curr.component and prev["n_qubits"] == curr.n_qubits:
                ratio = curr.elapsed_s / prev["elapsed_s"]
                status = "OK" if ratio < 1.5 else "REGRESSION"
                print(f"  {curr.component} N={curr.n_qubits}: "
                      f"{ratio:.2f}x vs baseline [{status}]")
except FileNotFoundError:
    print("No baseline found — saving current as baseline")
    save_benchmark_result(data, output_path=Path("results/benchmarks/baseline.json"))
```

---

#### Example 7: VQE Optimization with Custom Configuration

```python
"""Run VQE with specific optimizer settings and analyze convergence."""
import numpy as np
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
)
from qmbp_simulation.models import VQEConfig

N, p, J = 6, 2, 1.0
h_values = np.linspace(2.0, 1.25, 16)  # Descending

# Setup
lattice = make_lattice("chain_1d", N, J=J, h=2.0)
builder = HamiltonianBuilder()
solver = ClassicalSolver()
hva = HVACircuitBuilder()
circuit, _ = hva.create(N, p, lattice)

# Phase 1
exact_data = []
for h in h_values:
    lat_h = make_lattice("chain_1d", N, J=J, h=h)
    exact_data.append(solver.solve(builder.build(lat_h), lat_h))

# Phase 2 with custom config
config = VQEConfig(
    p_layers=p,
    n_restarts=5,
    maxiter=1000,
    ftol=1e-14,
    enable_callbacks=True,  # Track convergence trajectory
)
optimizer = VQEOptimizer(config=config)
vqe_results = optimizer.descending_sweep(h_values, circuit, lattice, exact_data)

# Analyze
for vqe_r, exact_r in zip(vqe_results, exact_data):
    de_gap = abs(vqe_r.energy - exact_r.ground_energy) / max(exact_r.gap, 1e-10)
    print(f"h={exact_r.h_value:.2f}: ΔE/gap={de_gap:.4f}, "
          f"fid={vqe_r.fidelity:.4f}, iters={vqe_r.n_iterations}")
```

---

#### Example 8: MPNN Training and Prediction

```python
"""Train an MPNN predictor and use it for parameter prediction."""
import numpy as np
import torch
from torch_geometric.data import Data
from qmbp_simulation import HamiltonianBuilder, make_lattice
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

# Assume vqe_results and exact_data from Phase 1+2 are available
N, p = 6, 2
lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)

# Build training dataset (fidelity-filtered)
dataset = build_graph_dataset(
    lattice=lattice,
    h_values=h_values,
    theta_opt=np.array([r.theta_opt for r in vqe_results]),
    e_exact=np.array([r.ground_energy for r in exact_data]),
    fidelities=np.array([r.fidelity for r in vqe_results]),
    fidelity_threshold=0.93,  # Only high-quality training data
)

# Train
model = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=2*p)
train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500, seed=42)
print(f"Final MSE: {train_result['best_loss']:.6f}")

# Predict at unseen h-value
builder = HamiltonianBuilder()
h_test = 1.5
edge_idx, coord = builder.build_graph_data(make_lattice("chain_1d", N, J=1.0, h=h_test))
x_test = torch.tensor(
    np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
    dtype=torch.float32,
)
test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

model.eval()
with torch.no_grad():
    theta_pred = model(test_graph).numpy().flatten()
print(f"Predicted θ at h={h_test}: {theta_pred}")
```

---

#### Example 9: Save and Load Results (Standardized Format)

```python
"""Demonstrate the standardized result saving/loading pattern."""
from pathlib import Path
from qmbp_simulation.framework import (
    build_result_envelope,
    save_experiment_result,
    save_pipeline_result,
    load_result,
    generate_timestamp,
)

# Build a result envelope (standard structure for all results)
result = build_result_envelope(
    config={
        "n_qubits": 10,
        "p_layers": 2,
        "h_values": [2.0, 1.75, 1.5, 1.25],
        "n_restarts": 5,
    },
    results={
        "per_h": [
            {"h": 2.0, "de_gap": 0.005, "fidelity": 0.998},
            {"h": 1.75, "de_gap": 0.008, "fidelity": 0.995},
            {"h": 1.5, "de_gap": 0.014, "fidelity": 0.990},
            {"h": 1.25, "de_gap": 0.030, "fidelity": 0.970},
        ],
    },
    summary={"mean_de_gap": 0.014, "pass_rate": 1.0, "total_time_s": 42.5},
    elapsed_s=42.5,
)

# Save as experiment result → results/experiments/exp_b4/run_20260526_143000.json
path = save_experiment_result(result, experiment_id="B4")
print(f"Saved to: {path}")

# Save as pipeline result → results/pipeline/pipeline_run_20260526_143000.json
path = save_pipeline_result(result, output_dir=Path("results/pipeline"))
print(f"Saved to: {path}")

# Load back
loaded = load_result(path)
print(f"Loaded: {loaded['summary']['mean_de_gap']}")
```

---

#### Example 10: Progress Reporting for Long-Running Tasks

```python
"""Use ProgressReporter for consistent console output in multi-phase work."""
from qmbp_simulation.framework import ProgressReporter
import time

reporter = ProgressReporter("Custom Analysis N=10")

with reporter.phase(1, "Loading data") as p:
    time.sleep(0.5)  # Simulate work
    p.detail("Loaded 17 h-points from checkpoint")

with reporter.phase(2, "Computing Hessians") as p:
    time.sleep(1.0)  # Simulate work
    p.detail("4/4 points have positive-definite Hessian")
    p.detail("No saddle points detected")

with reporter.phase(3, "Generating report") as p:
    time.sleep(0.2)
    p.detail("Saved to results/experiments/exp_b4/")

reporter.summary({
    "saddle_points": 0,
    "condition_number_max": 12.5,
    "all_genuine_minima": True,
})

# Output:
# ============================================================
#   Custom Analysis N=10
# ============================================================
#
#   Phase 1: Loading data...
#     Loaded 17 h-points from checkpoint
#     Done in 0.5s
#
#   Phase 2: Computing Hessians...
#     4/4 points have positive-definite Hessian
#     No saddle points detected
#     Done in 1.0s
#
#   Phase 3: Generating report...
#     Saved to results/experiments/exp_b4/
#     Done in 0.2s
#
# ============================================================
#   Complete in 1.7s
# ============================================================
#     Phase 1: Loading data (0.5s)
#     Phase 2: Computing Hessians (1.0s)
#     Phase 3: Generating report (0.2s)
#
#     saddle_points: 0
#     condition_number_max: 12.5000
#     all_genuine_minima: True
```

## Simulation Modes & Techniques

This section documents how to run noiseless vs noisy simulations, how to apply
different optimization techniques, and how to combine them. This is the reference
for choosing the right approach for a given research question.

### Execution Backends

The framework provides three backends via the `ExecutionBackend` ABC. All optimizers
and the `PipelineRunner` accept any backend — same code runs noiseless, noisy, or hardware.

| Backend | Use case | Speed | Accuracy |
|---------|----------|-------|----------|
| `NoiselessBackend` | Development, validation, thesis results | Fast | Exact |
| `NoisyBackend` | Shot noise studies, noise-aware training | Medium | Approximate |
| `HardwareBackend` | IBM Torino deployment | Slow | Real device |

#### Noiseless Simulation (default)

```python
"""Standard noiseless simulation — exact statevector, no shot noise."""
from qmbp_simulation import PipelineRunner, make_lattice
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import VQEConfig
import numpy as np

lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=2.0)
config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000)

# NoiselessBackend is the default — no need to pass explicitly
runner = PipelineRunner(lattice=lattice, config=config)
results = runner.run_full(h_values=np.linspace(2.0, 1.25, 16), h_test=[1.5])

# Or explicitly:
backend = NoiselessBackend()
runner = PipelineRunner(lattice=lattice, config=config, backend=backend)
```

#### Noisy Simulation (Gaussian shot noise)

```python
"""Noisy simulation with configurable shot noise (no hardware noise model)."""
from qmbp_simulation import PipelineRunner, make_lattice
from qmbp_simulation.execution import NoisyBackend
from qmbp_simulation.models import VQEConfig
import numpy as np

lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=2.0)
config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000)

# Gaussian shot noise: exact energy + N(0, 1/√shots)
backend = NoisyBackend(shots=8192, seed_simulator=42)
runner = PipelineRunner(lattice=lattice, config=config, backend=backend)
results = runner.run_full(h_values=np.linspace(2.0, 1.25, 16), h_test=[1.5])
```

#### Noisy Simulation (Full noise model via FakeTorino)

```python
"""Full noise model simulation using FakeTorino (requires qiskit-aer)."""
from qmbp_simulation.execution import NoisyBackend
from qiskit_ibm_runtime.fake_provider import FakeTorino

fake_backend = FakeTorino()
noise_model = fake_backend.noise_model  # Extract noise model

backend = NoisyBackend(
    shots=16384,
    noise_model=noise_model,
    seed_simulator=42,
)

# Use with VQEOptimizer directly (for single-point evaluation)
from qmbp_simulation import VQEOptimizer, HVACircuitBuilder, make_lattice, HamiltonianBuilder
from qmbp_simulation.models import VQEConfig

lattice = make_lattice("chain_1d", n_qubits=6, J=1.0, h=1.5)
H = HamiltonianBuilder().build(lattice)
circuit, _ = HVACircuitBuilder().create(6, 2, lattice)

# Evaluate energy at specific parameters
energy = backend.evaluate(circuit, H, params=np.array([0.5, 0.8, 0.3, 0.6]))
print(f"Noisy energy: {energy:.6f}")
```

#### ZNE (Zero-Noise Extrapolation) with Layout Selection

```python
"""Full ZNE workflow: layout selection → multi-layout estimation → extrapolation."""
from qmbp_simulation.execution import (
    NoisyEstimatorConfig,
    build_adjacency,
    find_layouts_bfs,
    select_layouts_by_circuit_ces,
    run_zne_deployment,
)
from qmbp_simulation import HVACircuitBuilder, HamiltonianBuilder, make_lattice
from qiskit_ibm_runtime.fake_provider import FakeTorino
import numpy as np

# Setup
N, p = 6, 2
fake_backend = FakeTorino()
lattice = make_lattice("chain_1d", N, J=1.0, h=1.5)
H = HamiltonianBuilder().build(lattice)
circuit, _ = HVACircuitBuilder().create(N, p, lattice)
bound_circuit = circuit.assign_parameters(np.array([0.5, 0.8, 0.3, 0.6]))

# 1. Find candidate layouts on the hardware topology
adj = build_adjacency(fake_backend)
candidates = find_layouts_bfs(adj, n_qubits=N, n_candidates=30, seed=42)

# 2. Select layouts by circuit CES (post-transpilation error metric)
layout_selection = select_layouts_by_circuit_ces(
    bound_circuit, fake_backend, candidates, n_select=3
)
print(f"Selected {len(layout_selection.layouts)} layouts")
print(f"CES values: {layout_selection.ces_values}")

# 2b. Alternative: select LOW-CES layouts for p=1 (perturbative regime)
#     Use this for p=1 hardware deployment where ZNE works best with low CES.
#     Validated: 8/9 seeds positive across chain_1d, ladder, triangular at N=10.
from qmbp_simulation.execution import select_layouts_low_ces
layout_selection_p1 = select_layouts_low_ces(
    bound_circuit, fake_backend, candidates, n_select=3, max_ces=0.5
)

# 3. Run ZNE deployment (measures across layouts, extrapolates to CES=0)
config = NoisyEstimatorConfig(shots=16384, seed_simulator=42)
zne_result = run_zne_deployment(
    bound_circuit=bound_circuit,
    hamiltonian=H,
    backend=fake_backend,
    layout_selection=layout_selection,
    config=config,
    n_qubits=N,
    per_site=False,  # Set True for per-site <X_i> ZNE
)

print(f"ZNE energy: {zne_result.energy_zne.extrapolated:.6f}")
print(f"R²: {zne_result.energy_zne.r_squared:.4f}")
print(f"Gain vs worst layout: {zne_result.energy_zne.gain:.4f}")
```

---

### Optimization Techniques

The `experiments/helpers/` module provides reusable techniques. Each can be
applied independently or combined within an experiment.

#### Utility: Topology-Aware Graph Construction

All experiments that build MPNN datasets or predict parameters should use
the shared `graph_utils` module instead of hardcoding chain_1d edges:

```python
"""Build dataset and predict using any topology (chain_1d, ladder, triangular, kagome)."""
from experiments.helpers.graph_utils import build_experiment_dataset, predict_theta

# Build training dataset (topology-aware)
dataset = build_experiment_dataset(self, h_values, theta_array)

# Predict at a single h-value
theta_pred = predict_theta(self, model, h_test)

# Batch prediction (efficient single forward pass)
from experiments.helpers.graph_utils import predict_theta_batch
thetas = predict_theta_batch(self, model, [1.5, 1.75, 2.0])
```

#### Technique: Parameter Freezing (TITAN-style)

Freeze insensitive parameters at large h to reduce optimization cost.

```python
"""Freeze θ_zz2, θ_x2 at h≥1.5 — validated: 0% accuracy loss, 75% cost reduction."""
from experiments.helpers import frozen_vqe, analyze_parameter_activity
from qmbp_simulation import VQEOptimizer, HVACircuitBuilder, make_lattice, HamiltonianBuilder
from qmbp_simulation.models import VQEConfig
import numpy as np

N, p = 6, 2
lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
circuit, _ = HVACircuitBuilder().create(N, p, lattice)
H = HamiltonianBuilder().build(lattice)

# Analyze which parameters are active at this h-value
# Returns dict with per-parameter sensitivity
activity = analyze_parameter_activity(circuit, H, theta_init=np.zeros(4))

# Run VQE with frozen parameters (indices 2,3 = θ_zz2, θ_x2)
result = frozen_vqe(
    circuit=circuit,
    hamiltonian=H,
    freeze_indices=[2, 3],       # Freeze last 2 params
    freeze_values=[0.15, 0.25],  # Fixed values from warm-start
    n_restarts=1,                # Only optimize 2 active params
    maxiter=500,
)
print(f"Energy with freeze: {result['energy']:.6f}")
print(f"Active params optimized: {result['active_params']}")
```

#### Technique: Hessian-Guided Restarts

Check if VQE converged to a saddle point; escape if needed.

```python
"""Hessian check at VQE minimum — escape saddle points via eigenvector perturbation."""
from experiments.helpers import hessian_guided_vqe, standard_multistart_vqe
from qmbp_simulation import HVACircuitBuilder, make_lattice, HamiltonianBuilder
import numpy as np

N, p = 6, 2
lattice = make_lattice("chain_1d", N, J=1.0, h=1.5)
circuit, _ = HVACircuitBuilder().create(N, p, lattice)
H = HamiltonianBuilder().build(lattice)

# Standard multi-start (baseline)
baseline = standard_multistart_vqe(circuit, H, n_restarts=5, maxiter=500)

# Hessian-guided: 1 restart + Hessian check + escape if saddle
hessian_result = hessian_guided_vqe(
    circuit=circuit,
    hamiltonian=H,
    n_restarts=1,
    maxiter=500,
    hessian_epsilon=5e-3,
    escape_threshold=-1e-6,  # Negative eigenvalue → saddle point
)
print(f"Baseline energy: {baseline['energy']:.6f} ({baseline['n_restarts']} restarts)")
print(f"Hessian energy:  {hessian_result['energy']:.6f} (1 restart + check)")
print(f"Is genuine minimum: {hessian_result['is_minimum']}")
print(f"Condition number: {hessian_result['condition_number']:.1f}")
```

#### Technique: Analytical Initialization

Use perturbation theory for initial guess (fast convergence, but may find wrong basin).

```python
"""Analytical init from perturbation theory — use for h >> 1 only."""
from experiments.helpers import analytical_init_p2, validate_analytical_init
import numpy as np

# Get analytical initial guess for p=2 at given h
h = 3.0
J = 1.0
theta_init = analytical_init_p2(h=h, J=J)
print(f"Analytical init at h={h}: {theta_init}")
# Output: [0.167, 0.654, 0.050, 0.196] (θ_zz1, θ_x1, θ_zz2, θ_x2)

# Validate: compare analytical vs random init
validation = validate_analytical_init(
    h_values=np.array([2.0, 3.0, 4.0]),
    n_qubits=6, p_layers=2, J=1.0,
    n_restarts_baseline=5,
)
for v in validation:
    print(f"h={v['h']}: analytical ΔE/gap={v['analytical_de_gap']:.4f}, "
          f"random ΔE/gap={v['random_de_gap']:.4f}, "
          f"iter_savings={v['iter_savings']:.0%}")
```

#### Technique: Physics-Informed MPNN Loss

Add energy-based regularization to MPNN training.

```python
"""Physics loss: penalize MPNN predictions that give high energy."""
from experiments.helpers import PhysicsInformedLoss, evaluate_energy_batch
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn
from qmbp_simulation import HVACircuitBuilder, HamiltonianBuilder, make_lattice
import numpy as np

# Standard MPNN training (baseline)
model_baseline = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=4)
train_mpnn(model_baseline, dataset, n_epochs=6000, lr=1e-3, patience=500)

# Physics-informed training
model_physics = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=4)
physics_loss = PhysicsInformedLoss(
    circuit=circuit,
    lattice=lattice,
    builder=HamiltonianBuilder(),
    weight=0.1,              # λ for physics term
    start_epoch=1000,        # Only activate after MSE converges
    eval_every=100,          # Evaluate energy every 100 epochs (expensive)
)
# Note: PhysicsInformedLoss is used as a callback during training
# See experiments/predictor/exp_c1_physics_loss.py for full integration
```

#### Technique: DyPP (Dynamic Parameter Prediction)

Extrapolate parameters from previous h-points (rejected: only 8-13% savings).

```python
"""DyPP: predict θ(h) from θ(h+Δh) and θ(h+2Δh) via polynomial extrapolation."""
from experiments.helpers import dypp_predict, dypp_linear, dypp_quadratic
import numpy as np

# Given optimized parameters at previous h-points
theta_history = {
    2.0: np.array([0.25, 0.78, 0.12, 0.35]),
    1.75: np.array([0.32, 0.82, 0.15, 0.38]),
    1.5: np.array([0.41, 0.88, 0.19, 0.42]),
}

# Predict θ at next h-point using linear extrapolation
theta_pred = dypp_linear(
    h_target=1.25,
    h_prev=[1.75, 1.5],
    theta_prev=[theta_history[1.75], theta_history[1.5]],
)
print(f"DyPP prediction at h=1.25: {theta_pred}")

# Or quadratic (needs 3 points)
theta_pred_quad = dypp_quadratic(
    h_target=1.25,
    h_prev=[2.0, 1.75, 1.5],
    theta_prev=[theta_history[2.0], theta_history[1.75], theta_history[1.5]],
)
```

#### Technique: Weight-Space Phase Detection (D1)

Detect phase transitions from MPNN weight gradients (zero QPU cost).

```python
"""Detect h_c from MPNN training loss gradient — no quantum hardware needed."""
from qmbp_simulation.analysis import WeightGradientAnalyzer
from qmbp_simulation.predictors import MPNNPredictor

# After training an MPNN model on Phase 2 data:
analyzer = WeightGradientAnalyzer(model)  # trained MPNNPredictor
grad_result = analyzer.analyze(dataset)

# grad_result contains:
print(f"Gradient norms per h: {grad_result.total_gradient_norms}")
print(f"Peak h-values: {grad_result.peak_h_values}")
print(f"Critical region detected: {grad_result.critical_region_detected}")
# Peak near h≈0.7-1.0 indicates phase transition (h_c ≈ 1.0 for TFIM)
```

---

### Choosing the Right Approach

| Research question | Backend | Technique | Script/Command |
|---|---|---|---|
| Validate pipeline at N=6 | Noiseless | Standard (5 restarts) | `run_pipeline.py --n-qubits 6` |
| Test at N=10 | Noiseless | Standard (5 restarts, h=128) | `run_pipeline.py --n-qubits 10 --hidden-dim 128 --patience 500` |
| Test at N=10 p=1 | Noiseless | 1 restart, h≥1.9 (chain) | `run_p1_pipeline_variants_r2.py` |
| Test at N=20 p=2 | Noiseless (MPS) | 7 restarts, no freeze | `run_experiment.py --exp G3` |
| Test at N=20 p=1 | Noiseless (MPS) | 5 restarts, h≥2.25 | S5 experiment |
| Heavy-hex (IBM Torino native) | Noiseless | 1 restart (p=1), h≥3.25 | `run_thesis_variants-heavy_hex.py` |
| Check for saddle points | Noiseless | Hessian check | `run_experiment.py --exp B4` |
| Reduce VQE cost at h≥1.5 | Noiseless | Parameter freezing | `run_experiment.py --exp B2` |
| Detect phase transition | Noiseless | Weight gradient (D1) | `run_experiment.py --exp D1` |
| Test ZNE at N=6 p=2 | Noisy (FakeTorino) | ZNE + layout selection | Use `run_zne_deployment()` |
| Test ZNE at N=10 p=1 | Noisy (FakeTorino) | ZNE + 3 layouts (gain=+49%) | p=1 ZNE variants |
| Validate noise resilience | Noisy (Gaussian) | SPSA optimizer | Use `NoisyBackend(shots=8192)` |
| Hardware deployment | Hardware | p=1 heavy-hex + PEA-ZNE | `run_ibm_torino_deployment.py` |
| Landscape analysis | Noiseless | Random sampling | `run_experiment.py --exp F3` |
| Scaling law | Noiseless | Multi-N sweep | `run_experiment.py --exp A3` |
| Data efficiency | Noiseless | Reduced training set | `run_experiment.py --exp G1` |
| Heisenberg model test | Noiseless | Model-agnostic pipeline | `run_thesis_variants-heisenberg.py` |

---

### Combining Techniques in a Custom Script

```python
"""Example: VQE with freezing + Hessian check + noisy evaluation."""
import numpy as np
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
)
from qmbp_simulation.execution import NoisyBackend
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.framework import ProgressReporter, build_result_envelope, save_experiment_result
from experiments.helpers import frozen_vqe, hessian_guided_vqe

N, p = 6, 2
h_values = np.array([2.0, 1.75, 1.5, 1.25])

reporter = ProgressReporter("Combined Technique Test")

with reporter.phase(1, "Noiseless VQE with freezing at h≥1.5") as phase:
    lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    circuit, _ = hva.create(N, p, lattice)

    results_noiseless = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lat_h)
        exact = solver.solve(H, lat_h)

        if h >= 1.5:
            # Use freezing at large h (75% cost reduction)
            r = frozen_vqe(circuit, H, freeze_indices=[2, 3],
                          freeze_values=[0.15, 0.25], n_restarts=1, maxiter=500)
        else:
            # Full optimization at small h
            r = hessian_guided_vqe(circuit, H, n_restarts=3, maxiter=500)

        de_gap = abs(r['energy'] - exact.ground_energy) / max(exact.gap, 1e-10)
        results_noiseless.append({"h": h, "de_gap": de_gap, "technique": r.get("technique", "full")})
        phase.detail(f"h={h:.2f}: ΔE/gap={de_gap:.4f}")

with reporter.phase(2, "Noisy evaluation of best parameters") as phase:
    noisy_backend = NoisyBackend(shots=8192, seed_simulator=42)
    for r in results_noiseless:
        # Re-evaluate with shot noise to estimate hardware performance
        lat_h = make_lattice("chain_1d", N, J=1.0, h=r['h'])
        H = builder.build(lat_h)
        noisy_energy = noisy_backend.evaluate(circuit, H, r.get('params', np.zeros(4)))
        phase.detail(f"h={r['h']:.2f}: noisy energy = {noisy_energy:.6f}")

reporter.summary({"mean_de_gap": np.mean([r['de_gap'] for r in results_noiseless])})

# Save
result = build_result_envelope(
    config={"N": N, "p": p, "techniques": ["freezing", "hessian"]},
    results=results_noiseless,
    elapsed_s=reporter.total_elapsed_s,
)
save_experiment_result(result, experiment_id="combined_test")
```

---

## Import Examples

```python
# Core imports (from package top-level)
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult,
    save_phase12_dataset, load_phase12_dataset,
    PipelineRunner,
)

# Submodule imports
from qmbp_simulation.execution import NoiselessBackend, NoisyBackend
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn
from qmbp_simulation.framework import BaseExperiment, ExperimentConfig, ExperimentMetrics
from qmbp_simulation.analysis import WeightGradientAnalyzer, DiagnosticCollector
from qmbp_simulation.pipeline import run_exact_diag_sweep
```

## The 4-Phase Pipeline

1. **Phase 1 — Classical Ground Truth**: Exact diag (N<15) or DMRG/TeNPy (N≤40)
2. **Phase 2 — HVA VQE**: Descending h-sweep with warm-start, L-BFGS-B, p≤2
3. **Phase 3 — MPNN Predictor**: GINConv + global pooling, fidelity-filtered data
4. **Phase 4 — Deployment**: MPNN warm-start → hardware VQE with error mitigation

The `PipelineRunner` includes always-on diagnostics via `DiagnosticCollector` —
every run captures timing, convergence, θ-smoothness, per-h MSE, and energy
decomposition metrics automatically.

## Running Tests

```bash
make test              # Fast tests (~12s)
make test-full         # All tests including slow (~60s)
make lint              # Ruff linter
make check-full        # lint + test + smoke-test (~15s)
```

## Key Results

| System | ΔE/gap | Status |
|--------|:------:|--------|
| N=6 p=2, h≥1.25 (chain_1d) | < 5% | ✅ Thesis-ready |
| N=10 p=2, h≥1.5 (chain/ladder/triangular/heavy-hex) | < 5% | ✅ Thesis-ready |
| N=10 p=1, h≥1.9 (chain), h≥3.25 (ladder/heavy-hex) | < 5% | ✅ Thesis-ready |
| N=20 p=1, h≥2.25 (chain) | 2.48% | ✅ Validated (MPNN) |
| N=20 p=2, h≥2.0 (chain) | 1.75% | ✅ Validated (MPS) |
| Heavy-hex p=1 N=10 (IBM Torino native) | 0.56% | ✅ Hardware-ready (module validated) |
| ZNE p=1 N=10 (heavy-hex, 3 layouts) | +62.7% gain | ✅ Confirmed |
| Heisenberg XXZ (all Δ, all topologies) | fidelity ≈ 0% | ❌ HVA p≤2 cannot express |

## Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Quantum circuits | Qiskit | 1.4.x |
| Hardware runtime | qiskit-ibm-runtime | ≥0.20 |
| Noisy simulation | qiskit-aer (MPS) | ≥0.14 |
| ML predictor | PyTorch + PyTorch Geometric | 2.x + 2.x |
| Tensor networks | TeNPy | 1.1.x |
| Linting | Ruff | ≥0.11 |
| Testing | pytest + Hypothesis | ≥8.0 + ≥6.98 |
| Git hooks | pre-commit | ≥3.6 |

## Constraints (enforced by pre-commit)

- HVA only, never HEA. p ≤ 2 layers.
- Primitives V2 only (no deprecated Qiskit APIs)
- Fidelity threshold ≥ 0.93 in training data (TFIM), ≥ 0.60 (Heisenberg)
- Heisenberg HVA p≤2 CANNOT work — do not attempt (30 runs + N=10/16 scaling confirm)
- ZNE threshold: ~18 CX gates. Use p=1 for N≥10 hardware deployment.
- No secrets in commits (gitleaks)
- Conventional commits (commitizen)

## Project Status (2026-06-09)

| Metric | Value |
|--------|-------|
| Pipeline runs (total) | 430+ |
| Noiseless runs | 329 |
| Noisy/ZNE runs | 93 |
| MPS scaling runs (N=40-80) | 8 |
| Formal experiments | 49 |
| Confirmed | 33 ✅ |
| Rejected (valid negative) | 8 ⚠️ |
| Useful-outcome rate | 84% |
| Topologies validated | 5 |
| Max system size | N=80 (MPS), N=100 (cross-N zero-shot) |
| Thesis findings corroborated | 15/22 (68%) strong |
| Compute time (total) | 17.6 hours |

**Next**: IBM Torino hardware deployment (QPU credentials needed) + thesis compilation.

For detailed status see [`documentation/ESTADO_PROYECTO.md`](documentation/ESTADO_PROYECTO.md).

## Development

```bash
pip install -e ".[dev,test]"
ruff check src/ experiments/ scripts/ tests/
ruff format src/ experiments/ scripts/ tests/
pre-commit install
pre-commit run --all-files
```

---

*Franco Raineri — Universidad de Buenos Aires, 2026*
