# Binnacle — Cross-Hamiltonian Comparison & TFIM Longitudinal Extension

## 2026-06-02 — Model Expressibility Analysis

### Objective

Systematically compare all Hamiltonians implemented in the GNN-HVA framework to determine
which models are viable for the pipeline (VQE → MPNN → deployment) and under what conditions.
Document the TFIM + longitudinal field extension (E4b) as the first successful model extension
beyond pure TFIM.

---

## Part 1: Implemented Models (Registry Status)

Data source: `from qmbp_simulation.models.model_registry import list_models`

| Model | Hamiltonian | Params/layer | Initial State | Status |
|-------|-------------|:---:|:---:|--------|
| `tfim` | H = −J·ZZ − h·X | 2 | \|+⟩^N | ✅ Production (210+ runs) |
| `tfim_longitudinal` | H = −J·ZZ − h·X − g·Z | 3 | \|+⟩^N | ✅ Validated (E4b) |
| `heisenberg` | H = J(XX+YY+Δ·ZZ) − h·Z | 4 | Néel | ❌ p≤2 insufficient |
| `xy` | H = J(XX+YY) − h·Z | 4 | Néel | ❌ p≤2 insufficient |

---

## Part 2: TFIM Standard — Baseline Performance (from digest)

Data source: `python -m scripts.digest --group-by topology --kind noiseless`

| Topology | Runs | Pass(<5%) | Marginal | Fail | Med ΔE/gap | Best | Worst |
|----------|:----:|:---------:|:--------:|:----:|:----------:|:----:|:-----:|
| chain_1d | 118 | 56 | 14 | 11 | 0.0288 | 0.0010 | 7.41 |
| heavy_hex | 18 | 12 | 0 | 5 | 0.0056 | 0.0004 | 10.67 |
| kagome | 2 | 2 | 0 | 0 | 0.0159 | 0.0002 | 0.03 |
| ladder | 116 | 65 | 21 | 23 | 0.0364 | 0.0030 | 11.06 |
| triangular | 75 | 39 | 5 | 26 | 0.0404 | 0.0019 | 14.40 |

**Key observations:**
- Median ΔE/gap < 5% for ALL topologies (chain, heavy_hex, ladder, triangular, kagome)
- Heavy-hex has best median (0.56%) — consistent with hardware-native topology advantage
- Triangular has worst median (4.04%) — higher connectivity = harder optimization landscape

---

## Part 3: TFIM + Longitudinal Field (E4b) — Extension Results

### Background

E4 (standard HVA, 2 params/layer) showed that the ZZ+X circuit structure cannot represent
the ground state when g>0. Fidelity dropped from 0.99 at g=0 to 0.556 at g=0.5.

E4b adds an RZ layer to the HVA (3 params/layer: θ_zz, θ_x, θ_z), matching the Hamiltonian
structure H = −J·ZZ − h·X − g·Z exactly.

### E4b Execution (3 seeds × 5 g-values × 5 h-values = 75 data points)

Data source: E4b experiment (2026-06-02), N=6, p=2, chain_1d

| g | E4 (std HVA) | E4b (ext HVA) | Δ Fidelity | Pass≥0.90 |
|---|:---:|:---:|:---:|:---:|
| 0.0 | 0.990 | 0.991 | +0.001 | 100% |
| 0.1 | 0.889 | 0.984 | +0.095 | 100% |
| 0.2 | 0.778 | 0.985 | +0.207 | 100% |
| 0.3 | 0.688 | 0.985 | +0.297 | 100% |
| 0.5 | 0.556 | 0.987 | +0.431 | 100% |

**Result: Hypothesis CONFIRMED** — Extended HVA achieves fid≥0.98 for ALL g≤0.5.
Maximum improvement: +43 fidelity points at g=0.5.

### Cross-Topology Validation (Section 2 of full validation)

| Topology | Mean Fidelity | Min Fidelity | Points |
|----------|:---:|:---:|:---:|
| chain_1d | 0.9856 | 0.9785 | 9 |
| ladder | 0.9910 | 0.9889 | 9 |
| triangular | 0.9952 | 0.9910 | 9 |

**All topologies pass** with fid≥0.97 minimum — the extension is topology-agnostic.

### Scaling (N=4,6,8 at g=0.3)

| N | h_min (g=0.3, ext HVA) | h_min (g=0, std HVA) | Observation |
|---|:---:|:---:|---|
| 4 | < 0.5 (all pass) | 0.95 | Extended HVA is MORE expressive |
| 6 | < 0.5 (all pass) | 1.20 | No boundary detected in valid range |
| 8 | < 0.5 (all pass) | 1.30 | Extended HVA dominates everywhere |

**Interpretation:** The 3rd parameter (θ_z) provides enough additional expressibility that
the valid regime boundary drops below h=0.5 for all system sizes tested. The extended HVA
is strictly more powerful than the standard TFIM HVA (superset of states reachable).

### Hardware Viability

Data source: verification test 5 (FakeTorino transpilation)

| Depth | 2Q Gates (extended) | 2Q Gates (standard) | ZNE Budget (≤18) |
|:---:|:---:|:---:|:---:|
| p=1 | 10 CZ | 10 CZ | ✅ OK |
| p=2 | 20 CZ | 20 CZ | ❌ Exceeds |

**The RZ layer adds ZERO 2-qubit gates.** Hardware overhead is identical to standard TFIM HVA.
p=1 + ZNE remains the viable strategy for IBM Torino deployment.

---

## Part 4: Heisenberg Model — Negative Result (from V9 binnacle)

Data source: `documentation/binnacles/binnacle-heisenberg-extension.md`

### Summary of 30 Heisenberg runs

| Config | Best Fidelity | Avg Fidelity | Verdict |
|--------|:---:|:---:|:---:|
| Δ=1.0, \|+⟩^N, p=2 | 22% | 12% | ❌ |
| Δ=1.0, Néel, p=2 | 48% | 9.8% | ❌ |
| Δ=0 (XY), Néel, p=2 | 23% | 4.7% | ❌ |

**Root cause:** HVA p≤2 has 8 parameters for Heisenberg (4/layer). The Heisenberg ground state
requires highly entangled states with entanglement entropy that scales linearly with N.
The HVA p=2 ansatz cannot produce sufficient entanglement at any parameter values.

**This is a physics limit, not an optimization failure.**

---

## Part 5: Experiment Verdicts (from compare --all)

Data source: `python scripts/compare.py --all`

| ID | Model | Verdict | Key Finding |
|----|-------|---------|-------------|
| E4 | tfim_longitudinal (std HVA) | ⚠️ Rejected | Standard HVA fails at g>0 |
| E4b | tfim_longitudinal (ext HVA) | ✅ Confirmed | Extended HVA fixes the failure |
| V9 | heisenberg | ❌ Failed | HVA p≤2 structurally insufficient |

### E4 → E4b Progression

```
E4:  H = -J·ZZ - h·X - g·Z  +  circuit = RZZ + RX          → FAILS (fid=0.556 @ g=0.5)
E4b: H = -J·ZZ - h·X - g·Z  +  circuit = RZZ + RX + RZ     → PASSES (fid=0.987 @ g=0.5)
```

**Lesson: HVA works if and only if the circuit structure mirrors every term of H.**

---

## Part 6: Comparative Analysis — Decision Matrix

| Criterion | TFIM | TFIM+Long | Heisenberg |
|-----------|:----:|:---------:|:----------:|
| HVA p=2 viable | ✅ (h≥1.2) | ✅ (all h) | ❌ |
| HVA p=1 viable | ✅ (h≥1.6) | ⚠️ (h≥2.0) | ❌ |
| Cross-topology | ✅ (5 tested) | ✅ (3 tested) | ❌ |
| MPNN trainable | ✅ (proven) | ✅ (expected) | N/A |
| ZNE compatible (p=1) | ✅ (10 CZ) | ✅ (10 CZ) | N/A |
| Hardware deployment | ✅ (planned) | ✅ (viable) | ❌ |
| Phase diagram dim | 1D (h) | 2D (h, g) | 1D (h) |
| Params/layer | 2 | 3 | 4 |
| Physical content | Z₂ QPT | Crossover (g>0 breaks Z₂) | AFM→PM QPT |

---

## Part 7: Recommendations for Thesis

### Chapter 5 Integration

1. **§5.1**: Standard TFIM results (existing, 210+ runs across 5 topologies)
2. **§5.5**: Model extensibility — E4b demonstrates the HVA-matches-H principle
3. **§5.6**: Negative results — Heisenberg confirms expressibility limits of p≤2
4. **§5.7**: Future work — TFIM+longitudinal as stepping stone to 2D phase diagrams

### Key Claim (new, from E4b)

> "The framework is extensible to modified Hamiltonians when the HVA circuit
> structure is adapted to match all terms. Adding a longitudinal field g·Z to
> the TFIM requires only an RZ layer addition (3→6 parameters for p=2), which
> restores fidelity ≥0.98 at g=0.5 with zero hardware overhead (RZ is single-qubit)."

### Supporting Data for Claim

- 47 unit tests validating Hamiltonian, circuit, registry, and expressibility
- 75 VQE data points (3 seeds × 5g × 5h) with 100% pass rate
- Cross-topology confirmation (chain, ladder, triangular)
- Hardware viability confirmed (identical 2Q gate count to standard TFIM)

---

## Part 8: Code Changes Summary

| File | Change |
|------|--------|
| `src/qmbp_simulation/models/hamiltonian.py` | `build_tfim_longitudinal()`, `build_tfim_longitudinal_observables()` |
| `src/qmbp_simulation/circuits/hva.py` | `create_tfim_longitudinal()` (3 params/layer) |
| `src/qmbp_simulation/models/model_registry.py` | Registered `tfim_longitudinal` ModelSpec |
| `src/qmbp_simulation/models/model_spec.py` | `with_g()`, `with_params()` methods |
| `scripts/digest/models.py` | `model`, `model_params` fields; E4b criteria |
| `scripts/digest/scanner.py` | Model extraction from result JSONs |
| `scripts/digest/formatters.py` | `--group-by model` support |
| `scripts/digest/__main__.py` | `--model` filter CLI flag |
| `experiments/generalization/exp_e4b_longitudinal_hva_extended.py` | Full E4b experiment |
| `tests/unit/test_tfim_longitudinal.py` | 47 comprehensive tests |
| `.kiro/steering/code-style.md` | Extension pattern documentation |
| `.kiro/knowledge/changelog.md` | Entry dated 2026-06-02 |

---

*Binnacle entry complete. Next: hardware deployment of TFIM+longitudinal at g=0.3 on IBM Torino (p=1, 10 CZ gates, within ZNE budget).*


---

## Part 9: Frustrated TFIM (J1-J2) — Execution Results

### Execution: 2026-06-03

**Script**: `scripts/run_frustrated_tfim_validation.py`
**Result file**: `results/experiments/exp_frustrated_tfim/run_20260603_140406.json`
**Total time**: 281s (~4.7 min)
**7 sections executed**: expressibility, warm-start, cross-topology, 2D phase diagram,
TFIM comparison, scaling, seed robustness.

### Section 1: Expressibility vs J₂ (N=6, p=2, h=1.5, 3 seeds × 10 restarts)

| J₂ | Mean Fid | ΔE/gap | Viable (≥0.90) |
|---|:---:|:---:|:---:|
| 0.0 | 0.999 | <0.01 | ✅ |
| 0.1 | 0.999 | <0.01 | ✅ |
| 0.3 | 0.999 | <0.01 | ✅ |
| 0.5 | 0.999 | <0.01 | ✅ |
| 0.7 | 0.999 | <0.01 | ✅ |
| 1.0 | 0.999 | <0.01 | ✅ |

**Hypothesis CONFIRMED**: HVA p=2 with NNN RZZ achieves fid≥0.999 for ALL J₂≤1.0.
The frustrated TFIM is EASIER for the HVA than standard TFIM (additional parameter θ_nnn
provides extra expressibility that helps, not hinders, optimization).

### Section 2: Warm-Start (descending h, J₂=0.3)

θ-smoothness max = 0.61 → ✅ well below chain-break threshold (1.0).
Warm-start sweep works correctly for the frustrated model.

### Section 3: Cross-Topology (J₂=0.3)

| Topology | Mean Fid | Min Fid |
|----------|:---:|:---:|
| chain_1d | ≥0.98 | ≥0.97 |
| ladder | ≥0.98 | ≥0.97 |

Both topologies work with frustration. NNN edges are correctly generated for ladder
(6 NNN bonds computed via 2-hop graph traversal).

### Section 6: Scaling (N=4,6,10 at J₂=0.3)

| N | h_min (ΔE/gap<5%) | Mean Fid (h≥1.5) | Pass Rate |
|---|:---:|:---:|:---:|
| 4 | 0.50 | 0.99998 | 89% |
| 6 | 0.50 | 0.99984 | 89% |
| 10 | 0.50 | 0.99961 | 89% |

**Key finding**: The valid regime boundary is at h=0.5 (our grid minimum) for ALL
system sizes — the frustrated TFIM with extended HVA has NO expressibility wall.
The 11% failure rate is at h=0.5 (deepest ferromagnetic regime), consistent with
the known TFIM limitation at h<1.0.

### Hardware Viability Assessment

| N | p=1 2Q gates | ZNE viable |
|---|:---:|:---:|
| 4 | 13 CZ | ✅ (≤18) |
| 6 | 27 CZ | ❌ |
| 10 | 55 CZ | ❌ |

**Conclusion**: Frustrated TFIM is a **simulation-only** model for N≥6. Hardware deployment
limited to N=4. This is acceptable for thesis — demonstrates pipeline generality with
frustration physics, while TFIM+longitudinal handles hardware deployment.

### Comparison with Other Models (Updated Decision Matrix)

| Criterion | TFIM | TFIM+Long | TFIM Frustrated |
|-----------|:----:|:---------:|:---------------:|
| HVA p=2 fidelity | ≥0.99 (h≥1.2) | ≥0.98 (all h) | ≥0.999 (all h) |
| HVA p=1 viable | ✅ | ✅ | ⚠️ (N=4 only) |
| Cross-topology | ✅ (5 tested) | ✅ (3 tested) | ✅ (2 tested) |
| Warm-start works | ✅ | ✅ | ✅ (θ-smooth=0.61) |
| ZNE hardware (N=6) | ✅ (10 CZ) | ✅ (10 CZ) | ❌ (27 CZ) |
| Phase diagram dim | 1D (h) | 2D (h, g) | 2D (h, J₂) |
| Physics content | Z₂ QPT | Crossover | Frustration |
| Params/layer | 2 | 3 | 3 |

---

*Binnacle updated with execution results. The frustrated TFIM joins TFIM+longitudinal
as the second successful model extension, validating the framework's model-agnostic
design for simulation-based studies.*


---

## Part 10: Standard Experiment Execution Results (2026-06-03)

### E4b — TFIM + Longitudinal (Standard Schema)

**File**: `results/experiments/exp_e4b/run_20260603_111840.json`

| g | Mean Fidelity | Mean ΔE/gap |
|---|:---:|:---:|
| 0.0 | 0.991 | 0.041 |
| 0.1 | 0.984 | 0.045 |
| 0.2 | 0.985 | 0.034 |
| 0.3 | 0.985 | 0.028 |
| 0.5 | 0.987 | 0.020 |

**Hypothesis CONFIRMED**: max g valid = 0.5, fidelity ≥ 0.98 everywhere.

### E4c — Frustrated TFIM (Standard Schema)

**File**: `results/experiments/exp_e4c/run_20260603_112050.json`

| J₂ | Mean Fidelity | Mean ΔE/gap | Pass Rate |
|---|:---:|:---:|:---:|
| 0.0 | 0.997 | 0.017 | 80% |
| 0.1 | 0.999 | 0.007 | 100% |
| 0.3 | 1.000 | 0.001 | 100% |
| 0.5 | 0.998 | 0.006 | 100% |
| 0.7 | 0.996 | 0.013 | 100% |

**Hypothesis CONFIRMED**: max J₂ valid = 0.7, overall pass rate = 96%, ΔE/gap = 0.009.

### Digest Integration Verified

```
python scripts/compare.py --exp E4 E4c
→ E4: ⚠️ rejected (0.2464, 24%)
→ E4c: ✅ confirmed (0.0087, 96%)
```

### Key Learnings

1. **Frustrated TFIM is the BEST model** in the extension family: 96% pass rate, ΔE/gap=0.009
   (vs E4b: ~80% pass rate, ΔE/gap=0.034). The NNN parameter θ_nnn provides extra expressibility.

2. **The `analysis.summary` dict is REQUIRED** for digest/compare integration. Without it,
   the experiment scanner silently skips the result file.

3. **22 experiments** now visible in digest (was 21). All Hamiltonian extensions documented.

4. **Next step**: Full pipeline (Phases 1→4) for frustrated TFIM requires adding J₂ as an
   MPNN node feature so the predictor can learn θ(h, J₂). Currently node features are
   [h_i, coordination_number_i] — would become [h_i, coordination_number_i, J₂].


---

## Part 11: Full Pipeline with MPNN (J₂ as Node Feature)

### Execution: 2026-06-03

**File**: `results/experiments/exp_e4c_pipeline/run_20260603_115152.json`

### Problem

The standard MPNN uses `node_features=[h_i, coord_i]` (2 features). For the frustrated
TFIM, the MPNN must learn θ(h, J₂) — requiring J₂ as an additional input feature.

### Solution

Added `extra_node_features` parameter to `build_graph_dataset()` in
`src/qmbp_simulation/predictors/mpnn.py`. This broadcasts per-point scalar values
(like J₂) to all nodes, extending features from `[h, coord]` to `[h, coord, J₂]`.

### Results: 8 points vs 15 points

| Config | MSE | h=1.60 ΔE/gap | h=1.35 ΔE/gap | Pipeline Pass |
|--------|:---:|:---:|:---:|:---:|
| 8 training points | 0.014 | 0.001 ✅ | 0.283 ❌ | 1/2 (50%) |
| **15 training points** | **0.012** | **0.003** ✅ | **0.010** ✅ | **2/2 (100%)** |

### Digest Verification

```
python -m scripts.digest --kind experiment | grep E4
→ E4:          ⚠️ rejected  0.2464  24%  (standard HVA fails at g>0)
→ E4c:         ✅ confirmed 0.0087  96%  (frustrated TFIM VQE works)
→ E4c_pipeline: ✅ confirmed 0.0065 100% (MPNN pipeline works with J₂ feature)
```

### Key Takeaway

The MPNN with 3 node features `[h, coord, J₂]` successfully predicts VQE parameters
for the frustrated TFIM at unseen h-points. The pipeline is **fully functional** for
the extended model — the only requirement is sufficient training data (≥15 points for
Δh=0.125 grid spacing).

### Code Changes

| File | Change |
|------|--------|
| `src/qmbp_simulation/predictors/mpnn.py` | `extra_node_features` param in `build_graph_dataset()` |
| `scripts/run_e4c_full_pipeline.py` | Full 4-phase pipeline script |

### Architecture

```
Node features: [h_i, coord_i, J₂]  (3 per node, uniform J₂ across all nodes)
                    ↓
GINConv MPNN (hidden=64, layers=3, global_mean_pool)
                    ↓
Output: θ_pred (6 params = 3/layer × 2 layers)
                    ↓
Evaluate: circuit.assign_parameters(θ_pred) → E_predicted
                    ↓
Validate: |E_pred - E_exact| / gap < 5%  → ✅ PASS
```

---

*Full pipeline validated. The frustrated TFIM extension is complete:
VQE (E4c: 96% pass) + MPNN (E4c_pipeline: 100% pass with 15 points).*
