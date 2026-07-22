# Integration Plan 06: Noise-Aware MPNN Training (Karim2025-style)

**Paper:** Karim et al. (2025) — Fast and Noise-aware ML Variational Quantum Eigensolver Optimiser  
**arXiv:** 2503.20210  
**Code:** ❌ No public repository  
**Priority:** HIGH (1 week, critical for hardware deployment — Phase 4)

## What It Does

Instead of training the MPNN on noiseless VQE data (Phase 2) and then deploying on
noisy hardware, Karim2025 trains directly on VQE data collected UNDER NOISE. The
predicted θ* are already adapted to the noisy landscape, so they perform better on
hardware without needing post-hoc error mitigation.

Key insight: The optimal θ under noise ≠ optimal θ without noise. The noise shifts
the energy landscape, and θ_opt(noisy) compensates for this shift. An MPNN trained
on noisy data learns this compensation implicitly.

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ Same MPNN, different training data source |
| Requires new dependencies? | ❌ Qiskit AerSimulator (already available) |
| Reuses existing modules? | ✅ Full pipeline (just swap backend in Phase 2) |
| Addresses a real problem? | ✅ Current MPNN overpredicts on hardware (trained noiseless) |
| Data available? | ⚠️ Need to run Phase 2 with NoisyBackend (FakeTorino) |
| Publishable? | ✅ Direct comparison: noiseless-trained vs noise-aware MPNN on hardware |

## How To Integrate

### What It Proves

That MPNN predictions trained on noisy VQE data achieve lower ΔE/gap on hardware
than predictions from a noiseless-trained MPNN, eliminating or reducing the need
for ZNE/PEA error mitigation at the prediction level.

### Conditions Where It Makes Sense

- **Models:** `tfim`, `tfim_longitudinal` (hardware-viable CX budget)
- **Topologies:** chain_1d, heavy_hex (native to IBM hardware)
- **N:** 6-10 (FakeTorino simulation feasible; real hardware available)
- **p:** 1-2 (deeper circuits have too much noise)
- **Backend:** FakeTorino (local), then ibm_torino (real)

### When NOT to Use

- Noiseless benchmarking (defeats the purpose)
- `tfim_frustrated`, `heisenberg` (too many CX gates for hardware)
- N > 14 on simulator (FakeTorino simulation time explodes)
- When ZNE already brings ΔE/gap < 5% (noise-aware adds complexity for no gain)

### Integration Architecture

```
src/qmbp_simulation/
└── predictors/
    ├── mpnn.py                    # ✅ EXISTS: MPNNPredictor, train_mpnn
    └── noise_aware_training.py    # NEW: Noisy Phase 2 data collection + training
```

OR more simply, no new module needed — just a modified pipeline run:

```
scripts/experiment_runners/
└── noisy/
    ├── run_noise_aware_pipeline.py  # NEW: Phase 2 with NoisyBackend
    └── compare_noiseless_vs_noisy.py # NEW: Head-to-head comparison
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `predictors.mpnn.MPNNPredictor` | SAME architecture (no changes!) |
| `predictors.mpnn.build_graph_dataset` | SAME dataset builder |
| `predictors.mpnn.train_mpnn` | SAME training loop |
| `execution.NoisyBackend` | FakeTorino backend for Phase 2 |
| `execution.noisy_utils.noisy_estimate` | Shot-based energy evaluation |
| `optimizers.vqe.VQEOptimizer` | SAME optimizer (swap backend only) |
| `pipeline.runner.PipelineRunner` | Full pipeline with `backend=NoisyBackend()` |

### Implementation Steps

The beauty of this approach: **NO new architecture needed**. The entire change is:

```python
# Current (noiseless):
runner = PipelineRunner(lattice, config, backend=NoiselessBackend())

# Noise-aware:
runner = PipelineRunner(lattice, config, backend=NoisyBackend(shots=8192))
```

1. **Run Phase 2 with `NoisyBackend`** (FakeTorino, 8192 shots):
   - VQEOptimizer auto-switches to COBYLA (shot noise → no gradients)
   - θ_opt(noisy) will differ from θ_opt(noiseless) by noise-induced shift
   - Need more restarts (noise makes landscape rougher): n_restarts=10-15
   - Wall-clock: ~10× slower than noiseless (shot-based evaluation)

2. **Train MPNN on noisy θ_opt** (same `train_mpnn()` call):
   - Input: same graph (h, coord)
   - Target: θ_opt(noisy) instead of θ_opt(noiseless)
   - Same architecture, same hyperparams

3. **Deploy comparison script** (`scripts/experiment_runners/noisy/compare_noiseless_vs_noisy.py`):
   - Load noiseless-trained MPNN (A) and noise-trained MPNN (B)
   - Deploy both on FakeTorino at test h-points
   - Compare ΔE/gap: A (noiseless MPNN + ZNE) vs B (noise-aware MPNN, no ZNE)
   - Also test: B + ZNE (should be best)

4. **Integration with existing noisy pipeline**:
   - Extend `run_noiseless_pipeline.py` with `--noisy-training` flag
   - Or create separate `run_noise_aware_pipeline.py` (cleaner)

### Expected Output

```json
{
  "model": "tfim_longitudinal",
  "topology": "heavy_hex",
  "N": 10,
  "p": 1,
  "shots": 8192,
  "comparison": {
    "noiseless_mpnn_raw": {"mean_de_gap": 0.18, "pass_rate": 0.40},
    "noiseless_mpnn_zne": {"mean_de_gap": 0.06, "pass_rate": 0.80},
    "noise_aware_mpnn_raw": {"mean_de_gap": 0.08, "pass_rate": 0.70},
    "noise_aware_mpnn_zne": {"mean_de_gap": 0.03, "pass_rate": 0.95}
  },
  "conclusion": "Noise-aware training + ZNE achieves best performance"
}
```

### Success Criterion

- Noise-aware MPNN (no ZNE) outperforms noiseless MPNN (no ZNE) by ≥ 30% → validates approach
- Noise-aware + ZNE achieves ≥ 90% pass rate → optimal combined strategy
- Training data collection feasible in < 1 hour on FakeTorino (N=10, 35 h-points)

### Risks

- FakeTorino VQE with COBYLA may not converge well (noisy + gradient-free)
  → Need more restarts (15-20) and possibly higher maxiter (2000)
- θ_opt(noisy) may have higher variance across seeds → need more seeds (5-10)
- Noise model may not match real hardware perfectly → validate on ibm_torino
- Phase 2 wall-clock with shots is 10-50× slower → practical for N≤10 only
- The "noise-aware" θ is specific to a noise model → doesn't transfer across backends
  (retrain for each QPU or noise level)

### Relationship to Existing Infrastructure

Our codebase already has all the pieces:
- `NoisyBackend` with FakeTorino noise model ✅
- `VQEOptimizer` with COBYLA auto-switch for shot-based ✅  
- `PipelineRunner` accepts any `ExecutionBackend` ✅
- ZNE infrastructure (`run_gate_folding_zne`, `run_pea_zne`) ✅

The only "new" thing is: running Phase 2 with NoisyBackend and saving those θ
as MPNN training targets. This is literally a one-line change to the pipeline
configuration. The comparison script is the real deliverable.
