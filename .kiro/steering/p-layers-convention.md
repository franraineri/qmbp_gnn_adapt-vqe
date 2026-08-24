inclusion: always

# p_layers Data Separation Convention

## Overview

All data artifacts, caches, and model zoo entries are distinguished by `p_layers`.
This document defines the canonical naming and organization rules.

## File Naming Conventions

| Artifact | Path Pattern | Example |
|----------|-------------|---------|
| Training NPZ | `data/multi_n_training/{topo}_N{n}_p{p}.npz` | `chain_1d_N10_p2.npz` |
| Extrapolation NPZ | `data/large_n_extrapolation/{topo}_N{n}_p{p}.npz` | `ladder_N30_p2.npz` |
| Zoo checkpoint | `data/model_zoo/checkpoints/..._p{p}.pt` | `unified_tfim_br_chain_1d_multiN_6+8+10_p2.pt` |
| Eval report | `results/extrapolation_evals/{topo}_p{p}/eval_{topo}_{ts}.md` | `chain_1d_p2/eval_chain_1d_20260820.md` |

## Metadata Fields

| Store | Field | How p is tracked |
|-------|-------|------------------|
| Zoo manifest | `p_layers: int` | Per-entry field |
| Result envelope (JSON) | `config.system.p_layers` | In experiment config |
| ResultIndex | `p_layers` | Queryable: `idx.query(p_layers=2)` |
| Dashboard | `configs[].p_layers` | Per-config entry |
| Eval cache key | `model\|topo\|N\|p_layers\|J\|h\|theta_hash` | Part of cache key |
| Ground Truth cache | `topo\|N\|model\|h` | p-independent (correct) |

## API Usage

```python
# MultiNAggregator — pass p_layers explicitly
agg = MultiNAggregator(topology="chain_1d", p_layers=2)
agg.scan()  # finds *_p2.npz files

# MultiTopologyAggregator
mt = MultiTopologyAggregator(p_layers=2)

# CachedBackend — pass p_layers for correct cache key
backend = self.get_cached_backend(topology=topo, n_qubits=N, model="tfim", p_layers=2)

# Zoo query — filters by p_layers
from qmbp_simulation.predictors.model_zoo import load_best_model_for
model, entry, source = load_best_model_for("chain_1d", p_layers=2)

# ResultIndex — query by p_layers
idx = ResultIndex()
runs = idx.query(model="tfim_bond_resolved", topology="chain_1d", p_layers=2)

# Warm-start from lower p
from qmbp_simulation.utils.helpers import tile_theta_for_higher_p, load_theta_from_npz
theta_p1 = load_theta_from_npz("chain_1d", 10, p_layers=1)
theta_p2_init = tile_theta_for_higher_p(theta_p1[h], p_target=2, expected_n_params=38)

# Transfer learning
from qmbp_simulation.predictors.unified_mpnn import transfer_model_to_higher_p
model, metrics = transfer_model_to_higher_p(
    source_checkpoint="data/model_zoo/checkpoints/..._p1.pt",
    p_target=2,
    topology="chain_1d",
)
```

## Runner CLI

```bash
# Generate p=2 training data
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d \
    --p-layers 2 \
    --train-n 6 8 10 \
    --multi-n-train

# Evaluate p=2 model
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d \
    --p-layers 2 \
    --target-n 20 30
```

## Ground Truth Cache

The GT cache is **intentionally p-independent**. Ground truth energy E₀(h) and
spectral gap Δ(h) are properties of the Hamiltonian, not the variational circuit.
The same GT entry is reused for p=1 and p=2 evaluations.

## Anti-patterns

- ❌ Hardcoding `_p1.npz` in glob patterns → use `self.p_layers`
- ❌ Assuming n_params implies p_layers → use explicit p_layers field
- ❌ Mixing p=1 and p=2 data in the same training dataset
- ❌ Using a p=1 model to predict p=2 theta without transfer/retrain
