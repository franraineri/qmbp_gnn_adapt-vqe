# Integration Plan 03: Flow-VQE Conditional Multi-Shot Predictor

**Paper:** Zou et al. (2026) — Generative flow-based warm start of VQE  
**arXiv:** 2507.01726 (npj Quantum Information)  
**Code:** ✅ `https://github.com/olsson-group/Flow-VQE`  
**Priority:** HIGH (1 week, directly improves pass rate near h_min frontier)

> **🤖 AI Agent Instruction:** Before implementing, clone and study the Flow-VQE
> repository at https://github.com/olsson-group/Flow-VQE. Key files to examine:
> 1. The normalizing flow architecture (layers, conditioning mechanism)
> 2. The preference-based training loop (not standard NLL — different from our EmbeddingMAF)
> 3. How they condition the flow on molecular/Hamiltonian properties
> 4. Their sampling + selection strategy (how many samples, selection criterion)
> 5. Compare their flow architecture with our existing `analysis/normalizing_flow.py`
>    (EmbeddingMAF, MAFLayer) — identify what we can reuse vs what needs adaptation
> 6. Their approach uses preference-based training; ours uses NLL. Decide which fits better.

## What It Does

Instead of predicting a single deterministic θ* (like our MPNN), Flow-VQE uses
conditional normalizing flows to model the full distribution P(θ|H). At inference,
it samples K candidates from the learned distribution and selects the best one via
a single energy evaluation each. This handles multimodal landscapes where the MPNN
picks the wrong mode.

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ We already have `EmbeddingMAF` + `FlowWarmstartManager` |
| Requires new dependencies? | ❌ PyTorch only (already have MAF layers!) |
| Reuses existing modules? | ✅ Directly extends `analysis/normalizing_flow.py` |
| Addresses a real problem? | ✅ Near h_min, MPNN sometimes picks wrong local minimum |
| Publishable? | ✅ "100% pass rate with flow multi-shot" vs "95% with deterministic" |

## How To Integrate

### What It Proves

That a generative (multi-shot) predictor eliminates residual failures near the
expressibility boundary h_min, achieving 100% pass rate where deterministic MPNN
achieves 95-98%.

### Conditions Where It Makes Sense

- **Models:** `tfim`, `tfim_longitudinal`, `tfim_bond_resolved`
- **Topologies:** ALL (especially triangular, square where landscape is harder)
- **N:** 10-20 (where multimodality becomes relevant)
- **p:** 2-4 (higher p = more local minima = more value from multi-shot)
- **h range:** Near h_min frontier (h ∈ [1.0, 2.0] for p=2)

### When NOT to Use

- Deep paramagnetic (h >> h_c): landscape is convex, MPNN suffices
- p=1 chain_1d: landscape is trivial, one mode only
- Time-critical deployment: K energy evaluations vs 1 for MPNN

### Integration Architecture

We ALREADY have the building blocks in `analysis/normalizing_flow.py` and
`analysis/flow_warmstart.py`. The integration extends `FlowWarmstartManager`:

```
src/qmbp_simulation/
└── analysis/
    ├── normalizing_flow.py      # ✅ EXISTS: EmbeddingMAF, MAFLayer
    ├── flow_warmstart.py        # ✅ EXISTS: FlowWarmstartManager
    └── flow_multishot.py        # NEW: Multi-shot sampling + selection
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `analysis.normalizing_flow.EmbeddingMAF` | Core flow architecture (already 584 params) |
| `analysis.normalizing_flow.MAFLayer` | Masked autoregressive layers |
| `analysis.flow_warmstart.FlowWarmstartManager` | Training + inference manager |
| `analysis.flow_warmstart._extract_embedding` | Frozen MPNN → embedding extraction |
| `predictors.mpnn.MPNNPredictor` | Frozen encoder for conditioning |
| `execution.NoiselessBackend` | K-shot energy evaluation |
| `pipeline.runner.PipelineRunner` | Extend Phase 3 with optional flow |

### Implementation Steps

1. **Create `analysis/flow_multishot.py`** (~80 lines):
   ```python
   class FlowMultiShotPredictor:
       """Sample K θ-candidates from trained flow, select best by energy."""

       def __init__(self, flow_manager: FlowWarmstartManager, K: int = 5):
           self.flow = flow_manager
           self.K = K

       def predict_best(self, graph: Data, hamiltonian, circuit, backend) -> np.ndarray:
           """Sample K, evaluate each, return lowest-energy θ."""
           samples = self.flow.sample(graph, n_samples=self.K)  # [K, 2p]
           energies = [backend.evaluate(circuit, hamiltonian, s) for s in samples]
           return samples[np.argmin(energies)]
   ```

2. **Extend `FlowWarmstartManager.sample()`** to return multiple samples (currently
   returns 1). This is a 5-line change: repeat the inverse flow K times with
   different z ~ N(0,I) draws.

3. **Add `--flow-multishot K` flag** to `run_noiseless_pipeline.py`:
   - After Phase 3 MPNN training, train flow on residuals
   - During deploy, use `FlowMultiShotPredictor` instead of raw MPNN

4. **Benchmark script** `scripts/analysis/benchmark_flow_multishot.py`:
   - Compare: MPNN-only vs Flow-1-shot vs Flow-5-shot vs Flow-10-shot
   - Measure: pass rate, mean ΔE/gap, wall-clock time
   - Focus on h-points near h_min where MPNN fails

### Expected Output

```json
{
  "method": "flow_multishot_K5",
  "model": "tfim_longitudinal",
  "topology": "chain_1d",
  "N": 10,
  "p": 2,
  "pass_rate_mpnn_only": 0.95,
  "pass_rate_flow_K1": 0.95,
  "pass_rate_flow_K5": 1.00,
  "pass_rate_flow_K10": 1.00,
  "mean_de_gap_mpnn": 0.028,
  "mean_de_gap_flow_K5": 0.015,
  "extra_time_factor": 5.0,
  "n_test_points": 20
}
```

### Success Criterion

- Flow K=5 achieves 100% pass rate where MPNN alone is 95% → publication-ready
- Extra cost (5× energy evals) is acceptable (still 29-90× faster than full VQE)
- Flow training adds < 30s to pipeline (584 params, 500 epochs)

### Risks

- For TFIM chain_1d, landscape is so smooth that MPNN already gets 100% →
  flow adds nothing. Must test on harder cases (triangular, near h_min).
- Flow may overfit on small datasets (17-35 points) → use validation split
- σ_flow uncertainty from existing FlowWarmstartManager may already provide
  similar benefit via the existing go/no-go gate — check before building new code
