---
inclusion: fileMatch
fileMatchPattern: "**/predictors/mpnn*,**/predictors/__init__*,**/train_mpnn*,**/build_graph*,**/gnn_architecture*"
---

# MPNN Predictor Context (invoke with #context-mpnn-predictor)

> Pre-digested context for MPNN architecture, training, debugging, and capacity decisions.

## Architecture (GINConv + global_mean_pool)

```python
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

model = MPNNPredictor(
    node_features=2,       # [h_i, position_encoding]
    hidden_dim=128,        # 64 for N=6, 128 for N≥10
    n_layers=3,            # GINConv message-passing layers
    output_dim=2*p_layers, # θ_zz, θ_x per HVA layer
    dropout=0.1,           # After first hidden layer
    norm_type="batch",     # "batch"|"layer"|"none" — use "none" for cross-N
)
```

### Why GINConv (not GAT, not GCN)
- GIN is provably as powerful as Weisfeiler-Lehman test (Xu et al. ICLR 2019).
- For uniform lattices (all edges equivalent), attention adds nothing.
- GATConv considered only for non-uniform couplings or mixed topologies.
- GNN > CNN by 36% for circuit property prediction (Meng et al. 2025).

### Capacity Scaling Rule

| N | hidden_dim | Rationale |
|---|-----------|-----------|
| 6 | 64 | 128 overfits on 17 training points |
| 10 | 128 | 64 underfits — more graph structure |
| 20+ | 128 | MPS-based, same capacity sufficient |
| 40-80 | 128 | Cross-N validated with norm_type="none" |

Rule of thumb: `hidden_dim ≈ 10-13 × N_nodes`. But 128 works for all N≥10.

## Graph Representation

```python
# Node features: [h_value, position_encoding]
# Edge index: connectivity from lattice topology
# Edge features: [J_ij] coupling strengths (uniform=1.0 for TFIM)
# Target: θ_opt ∈ ℝ^(2p) for TFIM HVA

dataset = build_graph_dataset(
    base_lattice, h_values,
    theta_opt_array,          # shape (n_points, 2*p)
    ground_energies,          # for physics callback
    fidelities=fid_array,     # for fidelity filter
    fidelity_threshold=0.93,  # CRITICAL: reject bad training points
)
```

## Training Recipe

```python
result = train_mpnn(
    model, dataset,
    n_epochs=6000,
    lr=1e-3,
    patience=300,      # 500 for N=10
    train_ratio=0.8,
    energy_callback=True,  # Validates θ_pred physically every 100 epochs
)
# result.best_model, result.train_losses, result.val_losses, result.best_epoch
```

### Key Training Decisions (validated)

| Decision | Evidence |
|----------|----------|
| Fidelity filter ≥ 0.93 | Points below → θ_opt doesn't represent GS, poisons model |
| 6000 epochs | Sufficient for convergence with patience-based early stop |
| lr=1e-3 | Stable with Adam; lr=1e-2 oscillates on small datasets |
| Dropout=0.1 | Proven to improve generalization on ≤30 training points (NN-VQE) |
| Physics callback | Catches "good MSE but bad energy" early |
| Train on valid regime ONLY | Invalid regime data (h < h_min) poisons predictions |

## norm_type Parameter (CRITICAL for cross-N)

```python
# Fixed-N training (default):
model = MPNNPredictor(..., norm_type="batch")

# Cross-N zero-shot (MUST use):
model = MPNNPredictor(..., norm_type="none")

# Alternative (untested for cross-N):
model = MPNNPredictor(..., norm_type="layer")
```

**Why**: BatchNorm on chain_1d captures graph-SIZE artifact (all nodes identical post-GINConv → zero intra-graph variance → running_stats encode N, not features). With BN: 18.5% error. Without: 0.13%.

## Checkpoint Management

```python
from qmbp_simulation.predictors import save_mpnn_checkpoint, load_mpnn_checkpoint

save_mpnn_checkpoint(model, optimizer, epoch, loss, path)
model, optimizer, epoch, loss = load_mpnn_checkpoint(path, model_class=MPNNPredictor)
```

## Quality Diagnostics

| Metric | Good | Suspect | Failure |
|--------|------|---------|---------|
| `generalization_gap` | < 1e-4 | 1e-4 to 0.01 | > 0.01 (overfit) |
| `theta_zz_mse` | < 1e-4 | 1e-4 to 1e-3 | > 1e-3 |
| `theta_x_mse` | < 1e-4 | 1e-4 to 1e-3 | > 1e-3 |
| Energy callback ΔE/gap | < 0.05 | 0.05-0.10 | > 0.10 |

## Known Failure Modes

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| gen_gap > 0.01 | Overfit (too many epochs, too few points) | Reduce epochs, increase patience, add data |
| θ_x underpredicted 25-40% | BatchNorm + cross-N + chain_1d | Use norm_type="none" |
| Good MSE but bad ΔE/gap | Optimizing wrong metric | Enable energy_callback |
| Training loss NaN | lr too high or data issue | Reduce lr to 1e-4, check fidelity filter |
| θ_opt inconsistent across seeds | Z₂ sign ambiguity | Warm-start resolves (C3), or use 3+ restarts |

## DO NOT

- Train on points with fidelity < 0.93 (poisons model with non-GS angles).
- Use hidden_dim=128 for N=6 (overfits on 17 points — use 64).
- Use norm_type="batch" for cross-N on chain_1d (25-40% error).
- Use noise-aware training (V7 5B: 6× worse — shot noise corrupts targets).
- Expect MPNN to extrapolate below h_min_safe (interpolation only).
- Use physics-informed loss at N=10 (C1@N=10: -12.3% — only helps with full h-range).
- Skip the fidelity filter for N≥15 where DMRG gives ground_state=None (manually restrict h-grid instead).

## Source Files

- #[[file:src/qmbp_simulation/predictors/mpnn.py]]
- #[[file:src/qmbp_simulation/predictors/__init__.py]]
- #[[file:.kiro/knowledge/gnn-architecture.md]]
- #[[file:documentation/binnacles/binnacle-cross-n-zero-shot.md]]
- #[[file:documentation/binnacles/binnacle-mpnn-eval-suite.md]]

## MPNN Evaluation Suite (ValidationRunner helpers, 2026-06-15)

9 reusable evaluation methods available on any `ValidationRunner` subclass:

### Basic suite (sections 10-14)
```python
# S10: Compare warm-start speedup vs random and prev-h
result = runner.benchmark_mpnn_warmstart(topology, n, h_train, h_test, ...)
# → speedup_vs_random, speedup_vs_prev_h, init_de_gap, per_h

# S11: LOO cross-validation (generalization estimate)
result = runner.mpnn_leave_one_out_cv(topology, n, h_train, ...)
# → pass_rate, per_fold de_gap, full_model_train_mse

# S12: Landscape decomposition (circuit vs ML error)
result = runner.mpnn_landscape_quality(topology, n, h_train, h_test, ...)
# → error_circuit, error_mpnn, error_total, mean_curvature, theta_deviation

# S13: Interpolation vs extrapolation boundary
result = runner.mpnn_interpolation_extrapolation(topology, n, h_train, h_interp, h_extrap, ...)
# → interp/extrap pass_rate, degradation_factor
```

### Extended suite (sections 15-19)
```python
# S15: Speedup scaling with N (p_layers_per_n for hardware-realistic comparison)
result = runner.mpnn_scaling_with_system_size(topology, [4,6,10], h_train, h_test,
    p_layers_per_n={4:2, 6:2, 10:1}, ...)  # p=1 for N≥10 (ZNE limit)
# → per_n speedup, scaling_trend, speedup_slope_per_N

# S16: Sample efficiency curve
result = runner.mpnn_learning_curve(topology, n, h_pool, h_test, ...)
# → per_size de_gap, critical_size (min k for 80% pass rate)

# S17: Zero-shot cross-topology transfer
result = runner.mpnn_topology_transfer(source_topo, target_topo, n, h_train, h_test, ...)
# → transfer_ratio, zero_shot/in_dist pass_rate

# S18: Multi-seed LOO stability
result = runner.mpnn_data_efficiency_vs_loo(topology, n, h_pool, n_seeds=3, ...)
# → mean/std pass_rate, cv, robust flag

# S19: κ(h) as hardware risk proxy
result = runner.mpnn_curvature_noise_correlation(topology, n, h_grid, noise_levels=[0.01,0.1], ...)
# → per_h kappa + noise_sensitivity, Pearson r per sigma
```

### Run via V3 runner
```bash
# All 10 MPNN sections for production config
python scripts/experiment_runners/run_hardware_rehearsal_v3.py \
  --skip-hardware-sections \
  --n-qubits 10 --topology heavy_hex --p-layers 1 \
  --h-train 4.5 4.25 4.0 3.75 3.5 3.25 3.0 --h-test 4.0 3.25 \
  --scaling-sizes 4 6 10 --scaling-p-layers 2 2 1 \
  --mpnn-epochs 3000 --vqe-restarts 1

# Analyze
python -m project_health.analysis.mpnn_eval_analyzer --thesis-table
```

### Validated results (heavy_hex N=10 p=1, 2026-06-15)
- S10: speedup=2.45x, init ΔE/gap=**0.39%** (hardware-ready without any VQE)
- S11: LOO 100% (7 folds), mean ΔE/gap=**0.38%**
- S17: chain→ladder transfer FAILS (ratio=200x — GNN is NOT cross-topology for params)
- S19: |r|=0.52 (heavy_hex κ range [111-174] — use V2 go/no-go, not κ-based)
