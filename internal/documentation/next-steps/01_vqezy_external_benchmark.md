# Integration Plan 01: VQEzy External Benchmark

**Paper:** Zhang et al. (2025b) — VQEzy: An Open-Source Dataset for Parameter Initialization in VQEs  
**arXiv:** 2509.17322  
**Code:** ✅ `https://github.com/chizhang24/VQEzy`  
**Priority:** HIGH (1 day effort, high publishability impact)

> **🤖 AI Agent Instruction:** Before implementing this integration, clone and study
> the VQEzy repository at https://github.com/chizhang24/VQEzy. Examine:
> 1. Dataset format (JSON/NPZ structure, field names, units)
> 2. How instances are organized by domain (condensed_matter/, quantum_chem/, combo_opt/)
> 3. What ansätze are used per task (filter for HVA-compatible ones)
> 4. The `data/` directory structure and loading utilities they provide
> 5. Copy their data-loading utilities directly into our `vqezy_loader.py` rather than reimplementing

## What It Does

VQEzy provides 12,110 pre-computed VQE instances across 3 domains (condensed matter,
quantum chemistry, combinatorial optimization) and 7 representative tasks with full
optimization trajectories and optimal θ*.

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ Spin Hamiltonian instances are directly compatible |
| Requires new dependencies? | ❌ Just numpy/JSON data loading |
| Reuses existing modules? | ✅ `MPNNPredictor`, `build_graph_dataset`, `train_mpnn` |
| Model/topology restrictions? | Filter for: TFIM-class Hamiltonians, chain_1d/ladder |
| N range? | VQEzy covers N=4-20 (overlaps our range perfectly) |

## How To Integrate

### What It Proves

Our MPNN (trained on our Phase 2 data) generalizes zero-shot to VQEzy instances.
Alternatively: train on VQEzy data, deploy on our test points. Both validate
cross-dataset transfer.

### Conditions Where It Makes Sense

- **Models:** TFIM, TFIM variants (subset of VQEzy's "condensed matter" domain)
- **Topologies:** chain_1d, ladder (VQEzy's spin chain tasks)
- **p:** 1-4 (match VQEzy's ansatz depths)
- **N:** 4-16 (overlap zone)

### Integration Architecture

```
src/qmbp_simulation/
└── predictors/
    └── external_benchmarks/
        ├── __init__.py
        ├── vqezy_loader.py         # Load & filter VQEzy dataset
        └── benchmark_evaluator.py  # Evaluate our MPNN on external data
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `predictors.mpnn.MPNNPredictor` | Trained model (zero-shot eval) |
| `predictors.mpnn.build_graph_dataset` | Convert VQEzy instances to PyG Data |
| `models.make_lattice` | Reconstruct lattice from VQEzy edge lists |
| `execution.NoiselessBackend` | Evaluate ΔE/gap of predictions |
| `framework.runner_base.ValidationRunner` | Structured benchmark script |

### Implementation Steps

1. **Download VQEzy dataset** from GitHub (JSON/NPZ files)
2. **Write `vqezy_loader.py`** that:
   - Reads VQEzy format (instance spec + trajectory + θ_opt)
   - Filters for spin-chain Hamiltonians compatible with our TFIM models
   - Converts to `(LatticeConfig, h_values, theta_opt, e_exact)` tuples
3. **Write `benchmark_evaluator.py`** that:
   - Loads our trained MPNN checkpoint
   - Converts VQEzy instances via `build_graph_dataset()`
   - Evaluates ΔE/gap zero-shot (no retraining)
   - Reports per-instance and aggregate metrics
4. **Create script** `scripts/analysis/benchmark_vqezy.py` using `ValidationRunner`

### Expected Output

```json
{
  "n_instances_evaluated": 150,
  "n_pass_5pct": 120,
  "pass_rate": 0.80,
  "mean_de_gap": 0.032,
  "median_de_gap": 0.018,
  "comparison": "Qracle: 64% fewer iters → our pipeline: 80% zero-shot pass"
}
```

### Success Criterion

- PassRate ≥ 50% zero-shot on VQEzy spin instances → publishable finding
- PassRate ≥ 70% → strong generalization claim for Paper A

### Risks

- VQEzy may use different ansätze (HEA, UCCSD) — need to filter for HVA-compatible
- Different θ conventions (period, sign) — apply `canonicalize_theta()` first
- Lattice edge ordering may differ — verify via Hamiltonian matrix comparison
