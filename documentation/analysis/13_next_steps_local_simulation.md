# Next Steps — Local Simulation Opportunities

**Date**: 2026-06-03 (updated with deep ROI analysis)
**Context**: All thesis-critical simulation work complete (430+ runs, 49 experiments,
84% useful-outcome rate). Hardware deployment on IBM Torino is the primary remaining task.
This document identifies what local simulation work could still add value.

---

## 0. Deep ROI Analysis — What Actually Moves the Thesis Forward?

### Thesis Claims That Need Support (from tesis-v2.1.tex)

The thesis makes 5 key claims:
1. HVA p≤2 resolves TFIM physics → **FULLY SUPPORTED** (329 noiseless runs)
2. MPNN eliminates quantum optimization → **FULLY SUPPORTED** (0 ADAPT iterations)
3. Pipeline scales to N=10 cross-topology → **FULLY SUPPORTED** (Tables 5.1-5.4)
4. ZNE reduces hardware errors → **PARTIALLY SUPPORTED** (simulated, not on real QPU)
5. Weight gradients detect phases → **FULLY SUPPORTED** (D1 + T1c, 100% agreement)

The ONLY unsupported claim is **real hardware execution**. Everything else has
conclusive simulation data. The thesis Chapter 4 explicitly promises "despliegue en
hardware IBM con mitigación de errores" — this is the gap.

### Honest Assessment: What Would NEW Simulations Accomplish?

| Proposed Simulation | Thesis Claim Supported | Actually Needed? |
|---------------------|:----------------------:|:----:|
| Gate-folding ZNE validation | Claim #4 (hardware readiness) | **YES** — blocking for QPU |
| MPNN J₂-grid density (T1a extension) | None directly | No — nice-to-have |
| Kagome expansion | Claim #3 (topology-agnostic) | Marginally — already 5 topologies |
| Honeycomb topology | Claim #3 (topology-agnostic) | No — diminishing returns |
| Cluster state | New claim (topological) | No — adds scope, doesn't close gaps |
| Active learning | Claim #2 improvement | No — already 0 iterations needed |

**Conclusion**: Only ONE simulation has genuine thesis-blocking value: gate-folding ZNE
validation on FakeTorino. Everything else is incremental or additive scope.

---

## 1. BLOCKING: Gate-Folding ZNE Validation

### Why This Is The Only Critical Simulation

The hardware rehearsal (HW_REHEARSAL) discovered that CES-based inhomogeneous ZNE
**does not work** for heavy-hex N=10 p=1 because all layouts have uniform CES≈0.15.
IBM's gate-folding ZNE (noise factors [1,3,5]) is the proposed replacement, but it
has NOT been validated locally. If it also fails, we need a fundamentally different
error mitigation strategy before spending QPU credits.

### What To Validate

```python
# IBM Runtime ZNE options (gate folding)
from qiskit_ibm_runtime import EstimatorV2, Options

options = Options()
options.resilience.zne_mitigation = True
options.resilience.zne.noise_factors = [1, 3, 5]
options.resilience.zne.extrapolator = "linear"
options.resilience.zne.amplifier = "gate_folding"  # default
```

Locally, simulate this by:
1. Transpile circuit to FakeTorino with optimization_level=2
2. For noise_factor=k: fold each 2Q gate k times (G → G·G†·G repeated)
3. Execute each folded circuit on `BackendEstimatorV2(FakeTorino)`
4. Linear extrapolation of E(noise_factor) → E(0)

### Success Criteria
- R² > 0.95 for linear fit (E vs noise factor)
- ZNE gain > +30% (recover at least 30% of noisy→noiseless gap)
- Consistent across 3 h_test values [4.0, 3.5, 3.25]

### Effort: 2-3 hours (implement gate folding + run 9 points)
### Impact: **CRITICAL** — determines whether hardware deployment proceeds

---

## 2. HIGH-VALUE Simulations (Strengthen Existing Claims)

### 2.1 Unsupervised Phase Detection from θ Parameters (arXiv:2506.06678)

**Why this matters more than I initially assessed:**

A recent paper (June 2025) shows that VQE circuit parameters θ(h) themselves
encode phase transition information and can be extracted unsupervisedly via a
Variational Autoencoder with attention mechanism. This is **exactly what our D1
weight-gradient analysis does**, but from a different angle.

**Proposed simulation**: Apply PCA or simple clustering to our existing θ_opt(h) data
(already collected in Phase 2 for 329 runs) and check if the phase transition
emerges as a discontinuity in the θ-manifold without any labels.

**Why high-value**:
- Zero additional VQE computation needed (data already exists)
- If it works, it's a **second independent phase detection method** (D1 uses MPNN
  weights, this uses raw VQE parameters)
- Directly citable as independent validation of arXiv:2506.06678
- Could be a subsection in Chapter 5 (Discussion) at near-zero cost

**Effort**: 1-2 hours (PCA + clustering on existing Phase 2 data)
**Impact**: Adds a novel analysis result without any new simulation

### 2.2 Noise-Robust Phase Signatures (arXiv:2402.18953)

Another relevant 2024 paper shows that VQE parameter derivatives (∂θ/∂h)
provide noise-robust phase transition signatures even when fidelity is poor.
We already HAVE this data (θ_smoothness), and our D1 analysis essentially does this.

**Proposed validation**: Plot |∂θ/∂h| from our Phase 2 data and confirm it peaks
near h_c. Compare against the D1 gradient peak. If they coincide, it strengthens
the D1 finding with a literature-backed theoretical justification.

**Effort**: 30 minutes (derivative of existing data)
**Impact**: Literature validation of D1 with zero new simulation

---

## 3. REVISED Priority-Ordered Action Plan

### Tier 1: BLOCKING (must do before hardware)

| # | Action | Effort | Value |
|---|--------|:---:|---|
| 1 | **Gate-folding ZNE on FakeTorino** (heavy-hex p=1 N=10) | 3h | Determines hardware viability |

### Tier 2: HIGH VALUE at ZERO compute cost (analysis of existing data)

| # | Action | Effort | Value |
|---|--------|:---:|---|
| 2 | PCA/clustering of θ_opt(h) for unsupervised phase detection | 1h | Novel analysis, no new VQE |
| 3 | ∂θ/∂h derivative plot vs D1 peaks comparison | 30min | Literature validation of D1 |
| 4 | Generate thesis-quality figures (project_health.figures --theme thesis) | 10min | Chapter 5 figures |

### Tier 3: MEDIUM VALUE (marginal thesis improvement)

| # | Action | Effort | Value |
|---|--------|:---:|---|
| 5 | MPNN J₂-density study (8 J₂ values, extends T1a) | 30min | Stronger 2D claim |
| 6 | TFIM+longitudinal full pipeline at g=0.3 (E2E proof) | 1h | Extensibility section |

### Tier 4: LOW VALUE (only if hardware deployment is delayed)

| # | Action | Effort | Value |
|---|--------|:---:|---|
| 7 | Kagome 3-seed expansion (N=10) | 1h | 6th topology (marginal) |
| 8 | Honeycomb topology (z=3) | 2h | Expected easiest (marginal) |
| 9 | Cluster state expressibility (topological attempt) | 1h | Likely fails, adds scope |

---

## 4. What NOT To Simulate (Over-Engineering Traps)

| Tempting Idea | Why It's A Trap |
|---------------|-----------------|
| "More topologies" (honeycomb, square, etc.) | 5 topologies already proves the claim. Adding a 6th or 7th has zero marginal thesis value. The thesis already says "topology-agnostic" based on 5 distinct topologies. |
| "Active learning loop" | MPNN error is 95% of total error, but ΔE/gap is already < 5% in-regime. Reducing MPNN error from 3.5% to 2% doesn't change any thesis conclusion. |
| "3D topologies" | CX budget incompatible (z=6 → 36+ CX). Known dead end. |
| "New Hamiltonians" (cluster, boundary, etc.) | Thesis scope is TFIM + extensions. Adding more models adds words to write without strengthening core claims. TFIM+longitudinal and frustrated are already done. |
| "MPNN architecture improvements" | The pipeline works. Improving MPNN from 3.5% to 2% median doesn't change Chapter 4 tables or any claim. |
| "More seeds" | G5 already proves seed-independence (std=0.0003). |

---

## 5. The Real Critical Path

```
Current state ──→ Gate-folding ZNE validation (2-3h)
                       │
                       ├── SUCCESS → IBM Torino hardware (Chapter 4)
                       │                  └── Thesis complete
                       │
                       └── FAILURE → Need alternative mitigation
                                         ├── PEC (expensive in shots)
                                         ├── Layout averaging (no extrapolation)
                                         └── Document as "simulation-validated
                                             pipeline, hardware deployment
                                             remains future work"
```

The honest assessment is: if gate-folding ZNE works on FakeTorino, everything
else is optional polish. If it fails, the thesis can still be written as a
simulation-validated framework with clear hardware deployment path.

---

## 6. Recommended Execution Order (Total: ~5 hours)

1. **Gate-folding ZNE** (3h) — the one thing that matters
2. **θ_opt PCA analysis** (1h) — free novel result from existing data
3. **∂θ/∂h vs D1 comparison** (30min) — literature validation
4. **Thesis figures** (10min) — `python -m project_health.figures --theme thesis`
5. If time remains: MPNN J₂-density (30min)

Everything beyond this is procrastination disguised as productivity.

---

## 7. Hamiltonians for Hardware (Final Answer)

| Priority | Model | Config | CX | Why |
|:---:|---|---|:---:|---|
| 1 | **TFIM standard** | p=1, N=10, heavy-hex, h_test=3.25 | 18 | Most validated, seed-independent (std=0.0003) |
| 2 | **TFIM+longitudinal** | p=1, N=10, heavy-hex, g=0.3, h_test=3.25 | 18 | Zero extra CX, demonstrates extensibility |

No other model is hardware-viable under the framework constraints (Heisenberg/XY: p≤2 insufficient;
frustrated TFIM: 27+ CZ exceeds ZNE budget; Kitaev: 20 CZ + fid=16%).

---

## 8. What NOT To Simulate (Documented Dead Ends)

| Candidate | Reason to Skip | Evidence |
|-----------|----------------|----------|
| Heisenberg/XXZ at any Δ | fid=0% with p≤2, linear N-scaling of failure | V9: 30 runs |
| Kitaev chain | 20 CZ@N=6, fid=16%, 3 simultaneous barriers | Verification script |
| 3D topologies | CX ≫ 18, meaningless ZNE | CX scaling analysis |
| TFIM frustrated on hardware | 27 CZ@N=6, far exceeds budget | Transpilation count |
| More VQE restarts | Already at landscape optimum (B4, F3) | G5: std=0.0003 |
| Cross-topology transfer | Zero-shot fails (ΔE/gap 3-10×) | S2: confirmed negative |
| SPSA refinement on warm-start | Hurts by -146% | V7: 4B |
| Schwinger lattice gauge | Not a spin model, outside thesis scope | Architecture mismatch |
| More topologies (6th, 7th) | 5 already proves topology-agnostic claim | Diminishing returns |
| MPNN architecture improvements | ΔE/gap already < 5%, improvement doesn't change conclusions | 95% MPNN error is fine |
| Active learning | Reduces error from 3.5% → 2% — no thesis claim changes | Over-engineering |

---

## 9. Literature-Backed Novel Analysis Opportunities (Zero Compute Cost)

### 9.1 Unsupervised Phase Detection from θ_opt(h) — arXiv:2506.06678

Chen et al. (2025) demonstrate that VQE circuit parameters contain hidden correlations
that encode phase transition information extractable via unsupervised learning (VAE +
attention). Our existing Phase 2 data (329 runs × 15-27 h-values × 2-6 θ params) is
**exactly this dataset**. Apply PCA to θ_opt(h) trajectories → check if the principal
component transition coincides with h_c.

### 9.2 Noise-Robust Phase Signatures — arXiv:2402.18953

Fontana et al. (2024) show that parameter sensitivity |∂C/∂θ| and parameter
discontinuities provide noise-robust phase transition indicators in VQE, even when
ground-state overlap is poor. Our θ_smoothness metric IS this signal. Reframing it
with their theoretical backing adds literature support to D1.

### 9.3 Phase Diagram Sketching from Low-Depth VQE — arXiv:2301.09369

Kattemölle & van Wezel (2023) prove that VQE with poor fidelity still correctly
identifies phase transition LOCATIONS. This directly supports our thesis claim that
classification works even when ΔE/gap > 5% (as seen in HW_REHEARSAL: classification
was 100% correct despite ΔE/gap ~100%).

---

*Document complete. Primary next action: gate-folding ZNE validation (action #1).*
*Everything else is polish or procrastination.*


---

## 10. Post-Hamiltonian-Extension Next Steps (2026-06-03)

**Context**: E4b (TFIM+longitudinal) and E4c (TFIM frustrated) are fully validated.
Full MPNN pipeline working with `extra_node_features`. 23 experiments in digest
(13 confirmed, 5 rejected, 5 failed). See `binnacle-hamiltonian-comparison.md`.

### 10.1 2D MPNN Predictor (train across J₂ grid) — HIGH VALUE

**What**: Train a single MPNN on a (h × J₂) grid, then predict θ at unseen (h, J₂) pairs.
Currently E4c_pipeline trains at fixed J₂=0.3. A 2D predictor would generalize across J₂.

**Novel contribution**: No existing paper shows a GNN predicting VQE parameters across
a 2D Hamiltonian parameter space. This is a genuine thesis-level finding.

**Implementation**: Sweep J₂ ∈ {0.1, 0.3, 0.5} × h_grid (15 pts each) = 45 training points.
Test at unseen (h, J₂) combinations. `extra_node_features` already supports this.

**Effort**: ~1 day. **Impact**: Novel thesis contribution (§5.5 or §5.7).

### 10.2 Noisy Simulation for TFIM+Longitudinal (ZNE transfer) — MEDIUM VALUE

**What**: Run ZNE (inhomogeneous, 3 layouts) on TFIM+longitudinal at g=0.3, p=1, N=6.
Same CX budget as standard TFIM (10 CZ), so ZNE should work identically.

**What it proves**: Error mitigation transfers to extended models without modification.
This is the "hardware readiness" argument for the model extension.

**Effort**: ~2 hours. Uses existing `NoisyBackend`. **Impact**: Confirms §5.5 hardware claim.

### 10.3 Weight-Space Phase Detection for Frustrated TFIM (D1 analog) — MEDIUM VALUE

**What**: Apply `WeightGradientAnalyzer` to E4c's trained MPNN. Does ||dW/dh|| peak
at the crossover point? Does frustration shift/broaden the peak?

**What it proves**: Zero-QPU phase detection generalizes to new physics.

**Effort**: ~3 hours. **Impact**: Extends D1 finding to new model (§5.1 extension).

### 10.4 Finite-Size Scaling for Extended Models — LOW-MEDIUM VALUE

**What**: Measure h_min(N) for TFIM+longitudinal and TFIM frustrated at N=4,6,8,10.
Compare scaling exponent β across models.

**What it proves**: How the valid-regime boundary scales with system size differs by model.

**Effort**: ~4 hours compute. **Impact**: Table in §5.5.

### 10.5 Cross-Model MPNN Transfer — EXPLORATORY

**What**: Train MPNN on standard TFIM data → evaluate on TFIM+longitudinal (g→0 limit).
Does knowledge transfer between related Hamiltonians?

**Effort**: ~3 hours. **Impact**: If positive → strong §5.7 claim. If negative → validates
that model-specific training is necessary (consistent with S2 cross-topology failure).

### Priority Order

1. **10.1** (2D MPNN) — novel, directly thesis-impactful
2. **10.2** (noisy ZNE transfer) — confirms hardware claim
3. **10.3** (D1 for frustrated) — extends existing finding
4. **10.4** (scaling) — quantitative but not blocking
5. **10.5** (transfer) — exploratory, may fail

**Decision**: Options 10.1 and 10.2 are the only ones with genuine thesis-advancing value.
The rest are incremental polish.
