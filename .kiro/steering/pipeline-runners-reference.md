---
inclusion: manual
---

# Pipeline Runners & Model Training Reference

Technical reference for AI agents. Describes all runners, their interconnections, data flow, and model persistence patterns.

## Runner Inventory

### Training Runners

| Runner | Path | Purpose | Model Output |
|--------|------|---------|--------------|
| `AcceleratedCrossNRunner` | `scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py` | Per-topology training (VQE + MPNN iterative) | `unified_tfim_br_{topo}_multiN_{n_values}_p{p}.pt` |
| `run_multi_topology_training.py` | `scripts/experiment_runners/cross_topology/run_multi_topology_training.py` | Universal MT model from all topologies | `unified_tfim_br_MT_{arch_label}_p1.pt` |
| `run_finetune_from_mt.py` | `scripts/experiment_runners/cross_topology/run_finetune_from_mt.py` | Specialize MT model for one topology | `unified_tfim_br_{topo}_fromMT_{n_values}_p{p}.pt` |

### Evaluation Runners

| Runner | Path | Purpose | Output |
|--------|------|---------|--------|
| `LargeNExtrapolationRunner` | `scripts/experiment_runners/scaling/run_large_n_extrapolation.py` | Predict θ at N=20-200, evaluate vs DMRG | NPZ + JSON + markdown eval report |
| `run_model_comparison.py` | `scripts/experiment_runners/cross_topology/run_model_comparison.py` | Side-by-side comparison of multiple checkpoints | JSON + per-model markdown |
| `run_arch_ablation.py` | `scripts/experiment_runners/cross_topology/run_arch_ablation.py` | Compare architecture variants (same data) | JSON ablation report |
| `evaluate_zoo_models.py` | `scripts/analysis/evaluate_zoo_models.py` | Evaluate all zoo models on NPZ training data | Markdown + zoo pass_rate update |

### Maintenance/Analysis

| Script | Path | Purpose |
|--------|------|---------|
| `audit_and_fix_model_zoo.py` | `scripts/maintenance/audit_and_fix_model_zoo.py` | Zoo integrity + coherence + exclusion sync |
| `check_zoo_coherence.py` | `scripts/maintenance/check_zoo_coherence.py` | Quick zoo↔dashboard coherence check (hook) |
| `update_cross_n_coverage.py` | `scripts/maintenance/update_cross_n_coverage.py` | Regenerate coverage documentation |
| `update_project_status.py` | `scripts/maintenance/update_project_status.py` | Refresh project-status.md from ResultIndex |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ VQE Training Data                                                            │
│ data/multi_n_training/{topo}_N{n}_p{p}.npz                                  │
│   Written by: AcceleratedCrossNRunner (section_train, iterative_improve)     │
│   Read by: MultiNAggregator.scan(), MultiTopologyAggregator.scan()           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Extrapolation Data                                                           │
│ data/large_n_extrapolation/{topo}_N{n}_p{p}.npz                             │
│   Written by: LargeNExtrapolationRunner._persist_extrapolation_npz()         │
│   Read by: MultiNAggregator.scan() (source 2: approximate tier)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Model Zoo                                                                    │
│ data/model_zoo/manifest.json          — model metadata (ZooEntry list)       │
│ data/model_zoo/checkpoints/*.pt       — model weights                        │
│ data/model_zoo/checkpoints/_versions/ — versioned backups (auto)             │
│ data/model_zoo/checkpoints/_best/     — best pass_rate per config (auto)     │
│ data/model_zoo/checkpoints/_archive/  — orphaned/superseded models           │
│ data/model_zoo/model_registry.json    — ModelRegistryDB (provenance)         │
│   Written by: register_checkpoint(), register_checkpoint_with_training_metrics() │
│   Read by: load_best_for_cross_n(), load_pretrained(), list_pretrained()     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Ground Truth Cache                                                           │
│ data/ground_truth_cache.json                                                 │
│   Written by: GroundTruthCache.flush() (after DMRG/eigsh)                   │
│   Read by: exact_ground_state() in ValidationRunner (2-level: memory+disk)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Eval Cache                                                                   │
│ data/eval_cache.json                                                         │
│   Written by: CachedBackend (transparent, per θ-hash)                       │
│   Read by: CachedBackend.evaluate() (avoids recomputing same circuit eval)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Dashboard                                                                    │
│ data/model_quality_dashboard.json                                            │
│   Written by: generate_model_quality_dashboard() (auto after every run)     │
│   Read by: load_best_for_cross_n (quality gate), audit scripts              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Exclusion Registry                                                           │
│ data/training_exclusions.json                                                │
│   Written by: auto_detect_exclusions() (auto after every run)               │
│   Read by: MultiNAggregator.scan(), runner_base N-level filter              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Selection Logic (load_best_mpnn_for_cross_n)

Priority chain in `runner_base.py`:
1. **Zoo per-topology model** (`topology=topo, n_qubits=0`) — quality-gated via `compute_model_readiness`
2. **Zoo multi-topology model** (`topology="multi_topology"`) — readiness ≥ 0.30
3. **Train new** from NPZ data via `MultiNAggregator` + `train_unified_mpnn`

## Architecture (UnifiedMPNN)

```python
UnifiedMPNN(
    node_features=4,          # UNIFIED_NODE_FEATURES
    hidden_dim=256,           # production default
    n_layers=3,               # GINConv layers
    norm_type="none",         # MANDATORY for cross-N
    dropout=0.1,
    type_embedding_dim=16,    # learned qubit/gate/rx embedding
    gate_readout=True,        # predict θ_zz from gate nodes directly
    use_residual=True,        # P1: skip connections
    readout_mode="last",      # P2: "last" | "jk_cat" | "jk_max"
    film_conditioning=True,   # P4: FiLM modulation by h
)
```

Save/load handles all variants via checkpoint metadata + state_dict key inference.

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Per-topology | `unified_tfim_br_{topo}_multiN_{n_values}_p{p}.pt` | `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt` |
| Multi-topology | `unified_tfim_br_MT_{arch}_p1.pt` | `unified_tfim_br_MT_residual+film_p1.pt` |
| Fine-tuned from MT | `unified_tfim_br_{topo}_fromMT_{n_values}_p{p}.pt` | `unified_tfim_br_ladder_fromMT_4+6+8+10+12_p1.pt` |
| Eval reports (MT) | `eval_{topo}_MT_{timestamp}.md` | `eval_ladder_MT_20260819_050021.md` |
| Eval reports (ST) | `eval_{topo}_{timestamp}.md` | `eval_chain_1d_20260818_164925.md` |
| Versions | `{name}_v{n}.pt` in `_versions/` | `unified_tfim_br_MT_residual+film_p1_v2.pt` |

## Key Library Functions

| Function | Module | Used By |
|----------|--------|---------|
| `load_best_for_cross_n()` | `predictors.model_zoo` | LargeNExtrap, AcceleratedCrossN |
| `compute_model_readiness()` | `predictors.model_zoo` | Selection policy (per-topo + MT) |
| `register_checkpoint_with_training_metrics()` | `predictors.model_zoo` | All training runners |
| `update_zoo_pass_rate()` | `predictors.model_zoo` | evaluate_zoo, model_comparison, auto from runner |
| `compute_deploy_summary()` | `analysis.metrics` | All evaluation sections |
| `validate_training_dataset()` | `analysis.metrics` | Pre-training validation |
| `load_training_exclusions()` | `analysis.metrics` | N-level exclusion filter |
| `generate_evaluation_report()` | `analysis.evaluation_report` | LargeNExtrap, model_comparison |
| `MultiNAggregator.scan()` | `predictors.multi_n_aggregator` | Training data aggregation |
| `MultiTopologyAggregator.scan()` | `predictors.multi_n_aggregator` | MT training data |
| `train_unified_mpnn()` | `predictors.unified_mpnn` | All training paths |
| `fine_tune_unified_mpnn()` | `predictors.unified_mpnn` | Fine-tune, curriculum phase B |
| `load_unified_checkpoint()` | `predictors.unified_mpnn` | All model loading |
| `save_unified_checkpoint()` | `predictors.unified_mpnn` | All model saving |
| `build_unified_bond_resolved_graph()` | `predictors.unified_graph` | All inference paths |

## Command Patterns

```bash
# ── TRAINING ──

# MT model (best techniques)
.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
    --use-residual --film --curriculum --epochs 5000

# Per-topology iterative improve
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology ladder --target-n 20 --iterative-improve --max-iterations 3 \
    --checkpoint data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt

# Fine-tune MT for specific topology
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
    --topology ladder --epochs 1000

# Per-topology from scratch
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --force-retrain --multi-n-train

# ── EVALUATION ──

# Evaluate all zoo models
.venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo --energy-eval

# Extrapolation test
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d --target-n 30 40 60 --skip-random-baseline

# Model comparison (MT vs ST)
.venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
    --topology ladder --target-n 20 26 --auto-detect --promote-best

# Architecture ablation
.venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py \
    --topology chain_1d --epochs 2000 --register-best

# ── MAINTENANCE ──

# Full audit + fix
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --coherence --sync-exclusions

# Quick coherence check
.venv/bin/python scripts/maintenance/check_zoo_coherence.py

# Dashboard regeneration
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py
```

## Auto-Maintenance (built into runner_base.run())

After every runner execution, automatically:
1. `_log_data_quality_feedback()` → extracts best pass_rate from sections → `auto_update_zoo_pass_rate()`
2. Detached subprocess: `generate_model_quality_dashboard()` + `auto_detect_exclusions()` + `ResultIndex.refresh_status()`
3. Hook `zoo-coherence-check` (on agentStop): runs `check_zoo_coherence.py`

## Quality Gates

| Gate | Threshold | Where Applied |
|------|-----------|---------------|
| Dual criterion (per-point) | ΔE/gap < 5% AND \|ΔE\| < 0.10 | `is_point_failure()` |
| Training data viability | min 5 points, min 1 N-value | `validate_training_dataset()` |
| Zoo model readiness | score ≥ 0.50 | `load_best_for_cross_n()` |
| MT fallback readiness | score ≥ 0.30 | `load_best_mpnn_for_cross_n()` |
| Exclusion (hard modes) | contaminated_training, gap_masking | N-level filter in runner_base |
| NPZ anti-regression | lower energy wins | `upsert_theta_npz()` |
| Zoo overwrite | version to `_versions/`, best to `_best/` | `register_checkpoint(overwrite=True)` |
