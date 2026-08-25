---
name: pipeline-operations
description: Technical operations guide for all experiment runners, model training, evaluation, maintenance, and data flow in the GNN-HVA project. Use when running experiments, training models, evaluating results, debugging pipelines, or performing maintenance.
---

# Pipeline Operations — Technical Guide

Operational reference for running experiments, training MPNN models, evaluating predictions, and maintaining data integrity in the GNN-HVA project.

## Abstraction-First Principle

When creating or modifying ANY runner, script, or experiment:

1. **Use ValidationRunner methods first** — they encapsulate caching, error handling, backend selection, and persistence in one call. Never reimplement what the base class provides.
2. **Use model_zoo high-level functions** — `load_best_for_cross_n()`, `register_checkpoint_with_training_metrics()`, `compute_model_readiness()`. Never interact with manifest.json directly.
3. **Use unified_mpnn functions** — `train_unified_mpnn()`, `fine_tune_unified_mpnn()`, `load_unified_checkpoint()`. Never instantiate UnifiedMPNN manually unless doing architecture research.
4. **Use compute_deploy_summary()** — Never manually aggregate per_h_results into pass rates.
5. **Use MultiNAggregator.scan() / MultiTopologyAggregator.scan()** — Never manually glob NPZ files or parse their structure.

The base class and library handle: caching, persistence, anti-regression, quality tiers, exclusion filtering, checkpoint resume, and post-run maintenance. Reimplementing any of these is a bug.

## Runner Architecture

All experiment runners extend `ValidationRunner` (from `framework.runner_base`). They share:
- CLI arg parsing via `framework.cli`
- Section-based execution (each method tagged with `@Section`)
- Auto-maintenance after `run()`: dashboard regen, exclusion detection, ResultIndex refresh
- Zoo pass_rate auto-update from section results

## Training Runners

### AcceleratedCrossNRunner (Per-Topology)

**Path**: `scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py`
**Purpose**: Train per-topology MPNN (VQE data gen + MPNN training in iterative loop)
**Output**: `unified_tfim_br_{topo}_multiN_{n_values}_p{p}.pt`

```bash
# From scratch
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --force-retrain --multi-n-train

# Iterative improve from existing checkpoint
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology ladder --target-n 20 --iterative-improve --max-iterations 3 \
    --checkpoint data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt
```

Key flags: `--topology`, `--target-n`, `--force-retrain`, `--iterative-improve`, `--max-iterations`, `--checkpoint`

### Multi-Topology Training (MT Universal)

**Path**: `scripts/experiment_runners/cross_topology/run_multi_topology_training.py`
**Purpose**: Train single model on ALL topologies simultaneously
**Output**: `unified_tfim_br_MT_{arch_label}_p1.pt`

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
    --use-residual --film --curriculum --epochs 5000
```

Key flags: `--use-residual`, `--film`, `--curriculum`, `--epochs`, `--hidden-dim`

### Fine-tune from MT

**Path**: `scripts/experiment_runners/cross_topology/run_finetune_from_mt.py`
**Purpose**: Specialize MT model for one topology (transfer learning)
**Output**: `unified_tfim_br_{topo}_fromMT_{n_values}_p{p}.pt`

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
    --topology ladder --epochs 1000
```

## Evaluation Runners

### Large-N Extrapolation

**Path**: `scripts/experiment_runners/scaling/run_large_n_extrapolation.py`
**Purpose**: Test MPNN predictions at N=20-200 vs DMRG ground truth
**Output**: NPZ (extrapolation data) + JSON (run envelope) + markdown eval report

```bash
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d --target-n 30 40 60 --skip-random-baseline
```

### Model Comparison (MT vs ST)

**Path**: `scripts/experiment_runners/cross_topology/run_model_comparison.py`
**Purpose**: Side-by-side evaluation of multiple checkpoints on same data
**Output**: JSON + per-model markdown + zoo pass_rate_by_n update

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
    --topology ladder --target-n 20 26 --auto-detect --promote-best
```

`--auto-detect` finds best ST + MT models from zoo. `--promote-best` updates zoo manifest.

### Architecture Ablation

**Path**: `scripts/experiment_runners/cross_topology/run_arch_ablation.py`
**Purpose**: Compare architecture variants (residual, FiLM, JK, etc.) with same data
**Output**: JSON ablation report

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py \
    --topology chain_1d --epochs 2000 --register-best
```

### Evaluate Zoo Models

**Path**: `scripts/analysis/evaluate_zoo_models.py`
**Purpose**: Batch-evaluate ALL zoo models on NPZ training data
**Output**: `results/model_evaluation_report.md` + zoo pass_rate updates

```bash
.venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo --energy-eval
```

## Model Selection Logic

`load_best_mpnn_for_cross_n()` in runner_base follows this priority:

1. **Zoo per-topology model** (`topology=topo, n_qubits=0`) — readiness ≥ 0.50
2. **Zoo multi-topology model** (`topology="multi_topology"`) — readiness ≥ 0.30
3. **Train new** from NPZ data via `MultiNAggregator` + `train_unified_mpnn`

Readiness computed by `compute_model_readiness()` from `predictors.model_zoo`.

## Data Stores & Flow

| Store | Path | Written By | Read By |
|-------|------|-----------|---------|
| VQE training NPZ | `data/multi_n_training/{topo}_N{n}_p{p}.npz` | AcceleratedCrossNRunner | MultiNAggregator.scan() |
| Extrapolation NPZ | `data/large_n_extrapolation/{topo}_N{n}_p{p}.npz` | LargeNExtrapolationRunner | MultiNAggregator (tier 2) |
| Zoo manifest | `data/model_zoo/manifest.json` | register_checkpoint() | load_best_for_cross_n() |
| Zoo checkpoints | `data/model_zoo/checkpoints/*.pt` | training runners | load_unified_checkpoint() |
| Zoo versions | `data/model_zoo/checkpoints/_versions/` | auto on overwrite | manual recovery |
| Ground truth | `data/ground_truth_cache.json` | GroundTruthCache.flush() | exact_ground_state() |
| Eval cache | `data/eval_cache.json` | CachedBackend | CachedBackend.evaluate() |
| Dashboard | `data/model_quality_dashboard.json` | post_experiment_sync() | load_best_for_cross_n() |
| Exclusions | `data/training_exclusions.json` | auto_detect_exclusions() | MultiNAggregator (N-filter) |
| Model registry | `data/model_zoo/model_registry.json` | register_checkpoint_with_training_metrics() | provenance queries |

## UnifiedMPNN Architecture

```python
UnifiedMPNN(
    node_features=4,          # UNIFIED_NODE_FEATURES
    hidden_dim=256,           # production default
    n_layers=3,               # GINConv layers
    norm_type="none",         # MANDATORY for cross-N generalization
    dropout=0.1,
    type_embedding_dim=16,    # learned qubit/gate/rx embedding
    gate_readout=True,        # predict θ_zz from gate nodes directly
    use_residual=True,        # skip connections
    readout_mode="last",      # "last" | "jk_cat" | "jk_max"
    film_conditioning=True,   # FiLM modulation by h
)
```

`norm_type="none"` is mandatory — batch/layer norm breaks cross-N generalization.

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Per-topology | `unified_tfim_br_{topo}_multiN_{n_values}_p{p}.pt` | `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt` |
| Multi-topology | `unified_tfim_br_MT_{arch}_p1.pt` | `unified_tfim_br_MT_residual+film_p1.pt` |
| Fine-tuned | `unified_tfim_br_{topo}_fromMT_{n_values}_p{p}.pt` | `unified_tfim_br_ladder_fromMT_4+6+8+10+12_p1.pt` |
| Eval reports | `eval_{topo}[_MT]_{timestamp}.md` | `eval_ladder_MT_20260819_050021.md` |

## Quality Gates

| Gate | Threshold | Function |
|------|-----------|----------|
| Per-point pass | ΔE/gap < 5% AND \|ΔE\| < 0.10 | `is_point_failure()` |
| Training data viability | ≥5 points, ≥1 N-value | `validate_training_dataset()` |
| Zoo model readiness (ST) | score ≥ 0.50 | `load_best_for_cross_n()` |
| MT fallback readiness | score ≥ 0.30 | `load_best_mpnn_for_cross_n()` |
| Exclusion filter | contaminated_training, gap_masking | N-level filter in runner_base |
| NPZ anti-regression | lower energy wins | `upsert_theta_npz()` |

## Key Library Functions

| Need | Import |
|------|--------|
| Load best model | `from qmbp_simulation.predictors.model_zoo import load_best_for_cross_n` |
| Model readiness score | `from qmbp_simulation.predictors.model_zoo import compute_model_readiness` |
| Register trained model | `from qmbp_simulation.predictors.model_zoo import register_checkpoint_with_training_metrics` |
| Update zoo pass_rate | `from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate` |
| Update pass_rate_by_n | `from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate_by_n` |
| Backfill from history | `from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons` |
| Deploy summary stats | `from qmbp_simulation.analysis.metrics import compute_deploy_summary` |
| Validate training data | `from qmbp_simulation.analysis.metrics import validate_training_dataset` |
| Load exclusions | `from qmbp_simulation.analysis.metrics import load_training_exclusions` |
| Post-experiment sync | `from qmbp_simulation.analysis.metrics import post_experiment_sync` |
| GT↔NPZ coherence | `from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence` |
| MT vs ST comparison | `from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison` |
| Eval report generation | `from qmbp_simulation.analysis.evaluation_report import generate_evaluation_report` |
| Aggregate training data | `from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator` |
| MT aggregate | `from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator` |
| Train model | `from qmbp_simulation.predictors.unified_mpnn import train_unified_mpnn` |
| Fine-tune model | `from qmbp_simulation.predictors.unified_mpnn import fine_tune_unified_mpnn` |
| Load checkpoint | `from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint` |
| Build graph input | `from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph` |

## Maintenance Commands

```bash
# ── DATA INTEGRITY ──
# Full consistency check (zoo ↔ dashboard ↔ GT ↔ registry)
.venv/bin/python scripts/maintenance/query_model_registry.py consistency

# Fix stale e_exact in NPZ from GT cache
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence; validate_gt_npz_coherence(fix=True)"

# Full post-experiment sync (all stores in correct order)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import post_experiment_sync; post_experiment_sync(verbose=True)"

# ── ZOO MANAGEMENT ──
# Full audit + fix
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --coherence --sync-exclusions

# Quick coherence check
.venv/bin/python scripts/maintenance/check_zoo_coherence.py

# Retrain queue
.venv/bin/python scripts/maintenance/check_zoo_coherence.py --retrain-queue

# ── COMPARISON & REPORTING ──
# MT vs ST global comparison
.venv/bin/python scripts/maintenance/query_model_registry.py compare -v

# Filtered by topology
.venv/bin/python scripts/maintenance/query_model_registry.py compare -t chain_1d -v

# Best model for deployment
.venv/bin/python scripts/maintenance/query_model_registry.py best -t chain_1d -n 20

# Failure diagnostics
.venv/bin/python scripts/maintenance/query_model_registry.py diagnose "unified*chain*"

# ── DOCUMENTATION ──
# Regenerate eval report + coverage
.venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py
.venv/bin/python scripts/maintenance/update_project_status.py
```

## Auto-Maintenance (built into runner_base.run())

After every runner execution, automatically:
1. `_log_data_quality_feedback()` → best pass_rate → `auto_update_zoo_pass_rate()`
2. Detached subprocess: `post_experiment_sync()` (dashboard + exclusions + ResultIndex + eval report + coverage)
3. Hook `zoo-coherence-check` (on agentStop): `check_zoo_coherence.py`

## Runner Base Patterns (MANDATORY — Always prefer these over manual implementations)

Every `ValidationRunner` subclass inherits these methods. They are the **primary API** for all experiment logic. Using lower-level primitives (manual backend instantiation, manual cache access, manual dict construction) when these exist is an anti-pattern.

### Ground Truth (2-level cache: in-memory + disk-persistent GroundTruthCache)
```python
e_exact, gap = self.exact_ground_state(topology, n_qubits, h, model="tfim")
# NEVER: GroundTruthCache().get(...) or ClassicalSolver().solve(...) directly
```

### Backend Selection (auto N-threshold, topology-aware)
```python
backend = self.select_backend(n_qubits)  # Statevector ≤22, MPS >22
backend = self.select_backend(n_qubits, for_vqe_loop=True)  # stricter threshold
# NEVER: if n > 22: MPSBackend(...) else: NoiselessBackend(...)
```

### Cached Evaluation (context manager — auto-flushes on exit)
```python
with self.get_cached_backend(topology=topo, n_qubits=N, model="tfim", p_layers=p) as eval_backend:
    eval_backend.set_h(h)
    energy = eval_backend.evaluate(circuit, H, theta)
# NEVER: EvalCache().make_key(...) or manual flush
```

### Per-H Result Dict (standardized keys, correct types, ΔE/gap auto-computed)
```python
result = self.build_per_h_result(h, e_pred, e_exact, gap, fidelity=fid, method="warm")
# NEVER: {"h_test": h, "de_gap": ...} — inconsistent keys will break compute_deploy_summary
```

### Deploy Summary (pass rates, means, classification — all in one call)
```python
from qmbp_simulation.analysis.metrics import compute_deploy_summary
summary = compute_deploy_summary(per_h_results)
# Returns: n_points, pass_rate_5pct, pass_rate_10pct, mean_de_gap, mean_fidelity, etc.
# NEVER: manual n_pass / len(results) counting
```

### MPNN Loading (hierarchical: zoo ST → zoo MT → train new)
```python
mpnn = self.load_mpnn_from_zoo()
mpnn = self.load_mpnn_from_zoo(model="tfim", n_qubits=10, allow_cross_n=True)
# NEVER: load_unified_checkpoint("path/to/specific/file.pt") in a runner section
# The zoo handles versioning, readiness gating, and fallback selection
```

### MPNN Training (full pipeline: data aggregation + training + registration)
```python
# For per-topology training inside a runner:
from qmbp_simulation.predictors.unified_mpnn import train_unified_mpnn
from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

aggregator = MultiNAggregator(topology=topo, model="tfim_bond_resolved", p_layers=1)
dataset = aggregator.scan()  # Handles exclusions, quality tiers, deduplication
model = train_unified_mpnn(dataset, epochs=3000, use_residual=True, film=True)
# Then register:
from qmbp_simulation.predictors.model_zoo import register_checkpoint_with_training_metrics
register_checkpoint_with_training_metrics(model, entry, metrics)
# NEVER: manual torch.save() or glob("data/multi_n_training/*.npz")
```

### VQE Checkpoint Resume (typed contract with param validation)
```python
checkpoint = self.load_vqe_checkpoint(topology, n_params=circuit.num_parameters)
if checkpoint is not None:
    results, prev_theta = checkpoint
# NEVER: self.load_checkpoint(f"vqe_{topology}") with manual key access
```

### H-Grid Generation
```python
self._h_values = self.generate_h_grid()           # non-uniform (dense near h_c)
self._h_values = self.generate_h_grid(uniform=True)  # uniform
# NEVER: np.linspace(0.1, 2.0, 20) — loses critical-region density
```

### Fidelity (safe, N-guarded)
```python
fidelity = self.safe_compute_fidelity(circuit, theta, topology, n_qubits, h, model="tfim")
# Handles N-check (skips for N>22), solver call, and errors automatically
```

## Workflow Decision Tree

**"I want to improve model for topology X":**
1. Check current state: `query_model_registry.py compare -t X -v`
2. Check data availability: `query_model_registry.py list --topology X`
3. If NPZ data sufficient → retrain: `run_accelerated_cross_n.py --topology X --force-retrain`
4. If need more N-values → generate data: `run_accelerated_cross_n.py --topology X --target-n N`
5. After training → evaluate: `run_model_comparison.py --topology X --auto-detect`

**"I want to test extrapolation to large N":**
1. `run_large_n_extrapolation.py --topology X --target-n 30 40 60`
2. Check results: `cat results/model_evaluation_report.md`

**"MT model isn't performing well on topology X":**
1. Fine-tune: `run_finetune_from_mt.py --topology X --epochs 1000`
2. Compare: `run_model_comparison.py --topology X --auto-detect --promote-best`

**"Data seems inconsistent":**
1. `query_model_registry.py consistency`
2. `validate_gt_npz_coherence(fix=True)`
3. `audit_and_fix_model_zoo.py --fix --coherence`

## Important Constraints

- **Abstraction hierarchy**: ValidationRunner methods > model_zoo functions > unified_mpnn functions > raw pytorch/numpy. Always use the highest-level abstraction available.
- **norm_type="none"** is mandatory in UnifiedMPNN for cross-N generalization
- **Dual criterion** for pass: ΔE/gap < 5% AND |ΔE| < 0.10 (both must hold)
- **NPZ anti-regression**: only overwrite if new energy is lower
- **Exclusion policy**: N-values with `contaminated_training` or `gap_masking` are auto-filtered from training
- **Post-run sync** happens automatically — don't call manually unless debugging
- **New runner checklist**: subclass ValidationRunner → use `setup_physics()` → use `self.exact_ground_state()` → use `self.build_per_h_result()` → use `compute_deploy_summary()` → register model via zoo API. If you find yourself writing manual H construction, manual backend creation, or manual result dicts, you're doing it wrong.
- All runners use `.venv/bin/python` (project virtualenv)
