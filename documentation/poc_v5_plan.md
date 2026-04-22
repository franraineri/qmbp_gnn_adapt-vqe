# PoC V5 Improvement Plan

## Strategic Goal
Each improvement targets a technique needed for the real implementation (larger N, 2D lattices, IBM hardware). The PoC is a learning vehicle, not an end in itself.

---

## Diagnosis: Where the Pipeline Fails

The cascading error chain:
1. **Phase 2** — HVA p=2 with |+⟩^N can only reach fid≈0.988 at h=1.25 (ΔE≈4e-02). This is the **expressibility ceiling**, not an optimization failure.
2. **Phase 3** — MLP faithfully learns the imperfect θ_opt targets. MSE converges to ~1e-04 but energy error stays at ~1.4e-01 for the worst training points. The MLP is not the bottleneck — the training data quality is.
3. **Phase 4** — ADAPT-VQE with 2 iterations can't recover from a mediocre warm-start. Gradient is still 0.45 at termination — it needs more iterations but is capped.

**Root cause:** The test point h=1.25 sits in the zone where Phase 2 targets are mediocre. The pipeline works well where Phase 2 works well (h≥1.5).

---

## Plan: 3 Improvements (ordered by impact and learning value)

### Improvement A: Energy-Aware Training Loss (Phase 3)
**What:** Replace pure MSE loss on θ with a physics-informed loss that includes the actual quantum energy.
**How:** Every K epochs, compute E(θ_pred) via StatevectorEstimator for a batch of training points. Add an energy penalty term: `loss = MSE(θ_pred, θ_opt) + λ * mean(|E(θ_pred) - E_exact|²)`.
**Why for real implementation:** On hardware, θ_opt from Phase 2 will be noisy. A loss that directly penalizes energy error makes the predictor robust to imperfect training labels — essential when Phase 1 uses DMRG (approximate) instead of exact diagonalization.
**Expected impact:** The MLP will learn to predict θ values that minimize energy, not just match the VQE's (possibly suboptimal) θ_opt. This breaks the cascading error chain.
**Complexity:** Medium. Requires differentiating through the quantum circuit evaluation (or using finite-difference gradients).

### Improvement B: Local-Observable VQE Cost Function (Phase 2)
**What:** Replace the global energy cost function in Phase 2 VQE with a weighted sum of local observable errors: `cost = Σ_i |⟨X_i⟩_ansatz - ⟨X_i⟩_exact|² + Σ_i |⟨Z_iZ_{i+1}⟩_ansatz - ⟨Z_iZ_{i+1}⟩_exact|²`.
**How:** Use the exact observables from Phase 1 as targets. Optimize θ to match local observables rather than global energy.
**Why for real implementation:** Per Mele et al., local cost functions don't suffer barren plateaus under noise. Training the VQE on local observables is exactly what we'll do on hardware. This also directly optimizes the metrics we care about (priority #2 in the validation table).
**Expected impact:** The VQE may find θ values that have worse global energy but better local observables — which is what matters for phase characterization. This could push the observable errors below 1e-2 even at h=1.25.
**Complexity:** Medium. Requires computing N + (N-1) expectation values per cost function call instead of 1.

### Improvement C: Bidirectional Sweep with Best-of-Two Selection (Phase 2)
**What:** Run the VQE sweep in both directions (h=2→0 AND h=0→2) and keep the best θ_opt per h-point.
**How:** For the ascending sweep, initialize from the ferromagnetic ground state |↑↑...↑⟩ (all-zero state, no Hadamards) with a modified HVA. Compare fidelities and keep the winner.
**Why for real implementation:** Different initial states access different regions of the optimization landscape. The descending sweep (from |+⟩^N) works well in the paramagnetic regime but fails in the ferromagnetic regime. The ascending sweep from |0⟩^N should work well for small h. This technique generalizes to any system where the ground state character changes across a phase transition.
**Expected impact:** Dramatically improves Phase 2 quality for h < 1.0, expanding the valid training regime. More training data → better MLP → better deployment.
**Complexity:** High. Requires a second HVA circuit with different initial state, and a merging strategy.

---

## Implementation Order

```
V5.0: Improvement A (energy-aware loss)
      → Learn: physics-informed ML training, hybrid quantum-classical loss
      → Validates: whether better training breaks the cascading error chain

V5.1: Improvement B (local-observable VQE cost)
      → Learn: local cost functions for NISQ, Mele et al. in practice
      → Validates: whether optimizing for observables beats optimizing for energy

V5.2: Improvement C (bidirectional sweep)
      → Learn: multi-initial-state strategies, phase-aware optimization
      → Validates: whether expanding the training regime improves generalization
```

Each version is independently testable. Run the full pipeline after each improvement and compare the 6-metric checklist.

---

## Success Criteria for V5

| Metric | Current V4 (h=1.25) | Target V5 | Notes |
|---|---|---|---|
| ΔE/gap | 4.28% ✅ | < 3% | Maintain |
| ⟨X⟩ error | 2.66e-02 ❌ | < 1e-2 | Primary target |
| ⟨ZZ⟩ error | 4.75e-02 ❌ | < 1e-2 | Primary target |
| ΔE | 3.82e-02 ❌ | < 2e-2 | Aspirational |
| Fidelity | 0.988 ❌ | > 0.995 | Noiseless only |
| ADAPT iters | 2 ✅ | ≤ 2 | Maintain |
| **Checklist** | **2/6** | **≥ 4/6** | |

## What This Teaches for the Real Implementation

| PoC Technique | Real Implementation Equivalent |
|---|---|
| Energy-aware MLP loss | GNN trained with DMRG energies as regularizer |
| Local-observable VQE cost | Hardware VQE cost function (noise-resilient) |
| Bidirectional sweep | Phase-aware initialization strategy for GNN |
| Multi-start optimization | Ensemble methods for robust θ_opt generation |
| Fidelity-filtered training | Quality-gated dataset curation from tensor networks |
