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
