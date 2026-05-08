# Literature Synthesis — Critical Insights for the GNN-HVA Pipeline

> Synthesized knowledge from 50+ papers in the bibliography. This document captures
> key findings, validates our approach, identifies weaknesses, and suggests improvements.

---

## 1. Validation of Our Core Architecture

### HVA > HEA for TFIM (Confirmed by Tripathi et al. 2026)

Tripathi et al. benchmarked HVA vs EfficientSU2 (HEA) on TFIM in 1D, 2D, and 3D up to 27 spins:
- HVA consistently achieves lower energy variance and better entanglement entropy reproduction
- HVA with symmetry breaking captures degenerate ground states that plain HVA misses
- **Critical insight for us**: At the critical point, HVA p=2 struggles with entanglement entropy accuracy even at N=6. This is NOT a pipeline deficiency — it's a fundamental expressibility limit confirmed independently.
- **Action**: Our h=1.25 ceiling (2-3/6 checklist) is validated as a physics limit, not a bug.

### GNN Superiority for Circuit Property Prediction (Meng et al. 2025)

Meng et al. compared GNN vs CNN for predicting quantum circuit output expectation values:
- GNN outperforms CNN by average 36.2% on direct comparison tasks
- GNN naturally captures circuit topology (gate connectivity) that CNNs must learn implicitly
- Node features encoding noise information improves noisy predictions
- **Validation**: Our choice of GINConv MPNN over MLP/CNN is well-supported.

### GNN for Spin System Magnetization (Slavin 2025)

Slavin demonstrated GNN predicting magnetization in quasi-1D Ising systems:
- Lattice geometry encoded as graph → GNN → magnetization prediction
- Captures plateaus, critical transitions, geometric frustration effects
- Trained on Monte Carlo data (analogous to our exact diag training)
- **Direct validation**: Our "graph → physical property" paradigm works for Ising systems specifically.

### Warm-Start Effectiveness (Puig et al. 2025, Martin et al. 2026)

- Puig et al. proved warm-starts provide larger loss variances (stronger gradients) vs random init
- Martin et al. showed TN warm-starts access "enhanced gradient zones" that don't shrink exponentially
- **Our advantage**: Descending sweep h=2→0 is a particularly effective warm-start because the landscape changes smoothly with h. This is validated by both theoretical and numerical evidence.

The GNN-HVA pipeline is the right architecture for this thesis. The literature confirms every major design decision. The improvements available are all within the current architecture (better error mitigation, more shots, weight analysis) rather than requiring a redesign.

The only scenario where you'd want to change architecture is if you were starting a new project targeting:

Random/unstructured Hamiltonians → GQE would be better
Chemistry problems → UCCSD + ADAPT-VQE would be necessary
Fault-tolerant hardware → QPE would replace VQE entirely
---

## 2. Critical Weaknesses & Honest Assessment

### The h=1.0 Critical Region Problem

Multiple sources confirm our observation:
- Sumeet et al. (2025): Need N/2 HVA layers per cluster for thermodynamic limit convergence. For N=6, that's p=3 — we're constrained to p=2.
- Tripathi et al. (2026): Entanglement entropy at criticality requires deeper circuits than p=2 can provide.
- **Honest assessment**: Our pipeline CANNOT accurately characterize the critical point with p≤2. This is a fundamental trade-off of the Mele et al. constraint. We sacrifice critical-point precision for noise resilience.
- **Mitigation**: Focus thesis narrative on phase CLASSIFICATION (which side of the transition), not critical exponent extraction.

### Hardware Noise Broadens Critical Crossover (Sharma 2026)

Sharma's IQM Garnet experiments show:
- Ground-state energies are reliably captured by shallow VQE across the full parameter space
- BUT magnetic order parameters and correlation-sensitive observables show significant noise broadening
- The critical crossover region becomes "smeared" on hardware
- **Impact on us**: Our Phase 4 hardware results will show correct phase classification away from h_c, but the transition region will be blurred. This is expected and should be presented as a feature (noise resilience of the classification), not a failure.

### MPNN Generalization Limits

- Qracle (Zhang et al. 2025): GNN works best on physically structured Hamiltonians, poorly on random circuits
- Bincoletto et al. (2025): Transferability works for "similar" systems but degrades for qualitatively different structures
- **Risk for us**: Our MPNN trained on 1D chains may not transfer well to ladders/2D without retraining or fine-tuning. The graph structure changes qualitatively (degree distribution, cycles).
- **Mitigation**: Plan for topology-specific fine-tuning when scaling to ladders. The MPNN architecture (GINConv + global_mean_pool) is topology-agnostic by design, but the learned weights are topology-specific.

### Shot Noise Dominance on Hardware

- Sharma (2026): With practical shot budgets (4096), shot noise dominates over gate errors for local observables
- Ma et al. (2025): Even with ZNE, statistical uncertainty from finite shots limits precision
- **Impact**: Our simulated ⟨X⟩ errors of ~8e-3 will be overwhelmed by shot noise (~1.6e-2) on hardware. Need either more shots (8192+) or measurement optimization (grouping commuting observables).

---

## 3. Methodological Improvements to Consider

### Near-Term Improvements (Low Effort, High Impact)

**A. Neural-Network-Enhanced ZNE (Sun et al. 2025)**
- Instead of linear/polynomial extrapolation to zero noise, train a small NN on the noise-energy relationship
- Constrains errors to O(10⁻²)–O(10⁻¹) vs O(10⁻¹) for standard ZNE
- **Implementation**: After Phase 4 hardware runs at noise levels 1x, 2x, 3x, fit a 2-layer MLP instead of linear regression. Minimal code change.

**B. Learned Dynamical Decoupling (Pokharel et al. 2025)**
- Genetic algorithm finds device-specific DD sequences that outperform canonical (XY4, CPMG)
- Scales to 100 qubits, generalizes from small sub-circuits
- **Implementation**: Use Qiskit's DD pass manager but with optimized sequences from GADD. Free improvement (no extra shots).

**C. Inhomogeneous ZNE (Uvarov et al. 2024)**
- Exploit the fact that different qubits/gates have different error rates on IBM hardware
- Map abstract qubits to physical qubits in different configurations to achieve different total error sums
- Linear energy-CES extrapolation to zero noise
- **Implementation**: Multiple transpilations with different qubit mappings → natural noise scaling without gate folding.

**D. U-VQNHE Post-Processing (Kim et al. 2026)**
- Learnable diagonal post-processing of measurement outcomes with variational safety guarantees
- Tested specifically on TFIM — directly applicable
- **Implementation**: After hardware measurements, apply learned reweighting to improve energy estimates. Classical post-processing only.

### Medium-Term Improvements (Medium Effort)

**E. Attention Mechanism for Parameter Correlations (Li et al. 2026)**
- VQE parameters have hidden correlations that encode phase information
- Attention mechanism captures these correlations better than simple regression
- Could detect phase transitions from parameter structure alone (unsupervised)
- **Implementation**: Add self-attention layer after GINConv message passing, before the MLP head. Moderate architecture change.

**F. Active Learning for Training Data (Miao et al. 2024)**
- Actively select h-values near phase transitions for VQE training
- Can halve dataset size while maintaining accuracy
- **Implementation**: After initial MPNN training, identify h-regions with highest prediction uncertainty, run additional VQE points there, retrain. Iterative process.

**G. Noise-Aware Training (Karim et al. 2025)**
- Train MPNN on data from noisy simulations (not just noiseless)
- Model learns to predict parameters that are optimal UNDER noise
- Shows resilience to coherent errors when deployed on real hardware
- **Implementation**: Run Phase 2 VQE with a noise model (AerSimulator with IBM noise), use those θ_opt for MPNN training. Significant compute cost but better hardware results.

### Long-Term / Future Work

**H. Generative Circuit Design (Nakaji et al. 2025 — GQE)**
- Replace "predict parameters for fixed circuit" with "generate optimal circuit structure"
- Transformer generates gate sequences conditioned on target Hamiltonian
- Paradigm shift: no VQE optimization needed at all
- **Assessment**: Too radical for current thesis scope, but excellent future direction.

**I. 2D Tensor Network Pre-optimization (Martin et al. 2026)**
- For 2D scaling (Kagome, triangular), DMRG becomes inefficient
- Differentiable 2D tensor networks can pre-optimize VQE circuits directly
- Identifies the "quantum advantage regime" where QPU beats TN
- **Assessment**: Critical for the N=36 Kagome target. Should be Phase 1 replacement for 2D systems.

**J. Quantum Reservoir with Time Crystal Dynamics (Yin et al. 2025)**
- Our QRC uses fixed random HVA parameters — suboptimal reservoir
- Discrete time crystal dynamics create richer feature spaces
- Topological noise robustness demonstrated experimentally
- **Assessment**: Could significantly improve our QRC fallback route. Medium implementation effort.

---

## 4. Theoretical Insights That Strengthen Our Narrative

### The Three-Way Synergy is Unique (Mele et al. 2026 + Cerezo et al. 2021)

No other known approach simultaneously achieves:
1. Noise resilience (shallow circuits survive decoherence)
2. Trainability (local costs avoid barren plateaus)
3. Physical expressibility (HVA captures relevant physics)

This is our strongest thesis argument. The GNN warm-start adds a fourth dimension:
4. Efficiency (near-zero quantum optimization cost)

### Classical Simulability Boundary (Martin et al. 2026)

Martin et al. identify specific regimes where:
- TN can simulate the VQE circuit classically (no quantum advantage)
- QPU provides genuine advantage over TN simulation

For TFIM: the advantage regime begins at N≈20 for 2D systems. Our N=6-10 1D results are classically simulable — the thesis value is in demonstrating the PIPELINE, not quantum advantage per se. Quantum advantage comes at the scaling targets (N=36 Kagome).

### NLCE + VQE for Thermodynamic Limit (Sumeet et al. 2025)

Numerical Linked-Cluster Expansion combined with VQE can extrapolate to infinite system size:
- Each cluster solved with VQE (our pipeline handles this)
- NLCE combines cluster results to approximate thermodynamic limit
- **Insight**: Our pipeline could be embedded as the "cluster solver" in an NLCE framework, giving access to thermodynamic-limit physics from small-N VQE runs.

### Phase Transitions in Weight Space (Hernandes et al. 2025)

Neural network weights trained across a phase diagram exhibit structural changes at phase boundaries:
- **Provocative idea**: Our MPNN's trained weights might already encode phase transition information
- Could detect transitions by analyzing how MPNN weights change as we sweep h
- No additional quantum measurements needed — purely classical analysis of the trained model

---

## 5. Critiques of Our Approach & Responses

### Critique 1: "p=2 is too shallow for critical physics"
**Response**: Correct, and intentional. We trade critical-point precision for noise resilience (Mele et al. constraint). Phase CLASSIFICATION works; critical exponent extraction does not. This is the right trade-off for NISQ hardware.

### Critique 2: "GNN is overkill for uniform 1D TFIM"
**Response**: True for the PoC (where MLP suffices). The GNN architecture is designed for SCALING — non-uniform couplings, ladders, 2D lattices. The PoC validates the pipeline; the GNN validates the scaling path.

### Critique 3: "27 training points is too few for reliable ML"
**Response**: Validated by Miao et al. (2024) who showed 20 points suffice with dropout regularization. Our fidelity filter (≥0.93) ensures data quality over quantity. Active learning could further optimize point selection.

### Critique 4: "Noiseless simulation doesn't prove hardware viability"
**Response**: Correct. That's why Phase 4 exists. The noiseless results establish the ceiling; hardware results show the floor. The gap quantifies noise impact. Sharma (2026) independently confirms this narrative structure works.

### Critique 5: "QRC is a fallback, not a contribution"
**Response**: QRC demonstrates that even WITHOUT optimization (gradient-free), the pipeline produces useful results. This proves the shallow-circuit + local-observable framework works independently of the optimization strategy. It's a robustness argument, not a primary contribution.

### Critique 6: "No comparison with other ML approaches (CNN, transformer, etc.)"
**Response**: Meng et al. (2025) provides this comparison — GNN outperforms CNN by 36%. We should cite this as external validation rather than reproducing the comparison ourselves.

---

## 6. Key Numbers to Remember

| Fact | Source | Implication |
|------|--------|-------------|
| GNN > CNN by 36% for circuit prediction | Meng 2025 | Validates GINConv choice |
| N/2 HVA layers needed for thermodynamic limit | Sumeet 2025 | p=2 insufficient for N>4 at criticality |
| Warm-start: 10-50× speedup | Puig 2025, our data | Core pipeline advantage |
| Flow-VQE: 50× acceleration | Zou 2026 | Alternative warm-start (generative) |
| Qracle: 64% fewer optimization steps | Zhang 2025 | GNN scaling validated |
| ZNE + NN: errors O(10⁻²) | Sun 2025 | Achievable hardware precision |
| DD on 100 qubits: stable without retraining | Pokharel 2025 | Scalable error suppression |
| Shot noise at 4096: ~1.6e-2 per observable | Sharma 2026 | Hardware precision floor |
| TN advantage boundary: N≈20 for 2D | Martin 2026 | Where QPU becomes necessary |
| Critical crossover broadened by noise | Sharma 2026 | Expected Phase 4 behavior |

---

## 7. Updated Mental Model of the Pipeline

```
                    CLASSICAL DOMAIN                          QUANTUM DOMAIN
                    ──────────────────                        ──────────────
Phase 1: Exact Diag/DMRG → ground truth (h, E₀, ψ₀, ⟨O⟩)
                    │
Phase 2: VQE sweep (warm-start h=2→0) → θ_opt(h) dataset
                    │                                              │
Phase 3: MPNN training (graph → θ_pred)                           │
                    │                                              │
                    ├── θ_pred(h_test) ─────────────────────────→ Phase 4: QPU
                    │                                              │
                    │   [NEW: NN-enhanced ZNE post-processing] ←──┤
                    │   [NEW: Learned DD sequences]                 │
                    │   [NEW: Inhomogeneous qubit mapping]          │
                    │                                              │
                    └── Phase classification: ⟨X⟩, ⟨ZZ⟩ → phase label
```

The pipeline's key insight remains: minimize quantum resource usage by maximizing classical pre-computation. The MPNN is the bridge — it compresses the entire VQE landscape into a single forward pass.

---

## 8. Priority Actions Based on Literature

### Immediate (before thesis submission)
1. Cite Tripathi 2026 to validate h=1.25 ceiling as physics limit
2. Cite Sharma 2026 to set expectations for hardware noise broadening
3. Implement inhomogeneous ZNE (Uvarov) for Phase 4 — low effort, high narrative value
4. Add Xu 2019 and Gilmer 2017 citations to justify GINConv architecture

### Short-term (thesis improvements)
5. Try NN-enhanced ZNE extrapolation (Sun 2025) on hardware data
6. Implement learned DD (Pokharel 2025) via Qiskit DD pass manager
7. Analyze MPNN weight structure across h-sweep (Hernandes 2025 insight)

### Future work section
8. NLCE + VQE for thermodynamic limit extrapolation
9. 2D TN pre-optimization for Kagome scaling
10. Generative circuit design (GQE) as next-generation approach
11. Noise-aware MPNN training for hardware-optimized predictions

### The next high-value actions are:

Ladder topology validation (tests GNN generalization)
Phase 4 hardware deployment with the improved error mitigation stack
MPNN weight analysis for unsupervised phase detection (novel contribution)


### Where the literature suggests we COULD improve (without changing architecture)
Improvement	| Effort |	Impact	| When
Inhomogeneous ZNE for Phase 4	Low	High — better hardware results	Before hardware runs
8192+ shots	Minimal	High — shot noise below signal	Before hardware runs
Learned DD sequences (GADD)	Low	Medium — free error suppression	Before hardware runs
MPNN weight analysis for phase detection	Low	Medium — novel result, zero QPU cost	Anytime
NLCE + VQE for thermodynamic limit	Medium	High — extends to infinite N	Future work
GATConv for ladders/2D (non-uniform edges)	Low	Medium — may help when edges differ	When testing ladders


### The one legitimate concern: is VQE itself the right quantum algorithm?
This is the most interesting question from the literature. Three alternatives exist:

1. Generative Quantum Eigensolver (GQE/SpinGQE) — generates circuits rather than optimizing parameters. Eliminates VQE entirely. But: only validated on 4 qubits, requires transformer training, and loses the physics-informed HVA structure. Verdict: too immature for a thesis, excellent future work.

2. Quantum Subspace Expansion (QSE) — uses VQE as initial state, then expands classically. Weaving et al. (2025) achieved 0.01% error on Kagome this way. Verdict: complementary to our approach, not a replacement. Could be added as Phase 4.5.

3. DMRG-biased contextual subspaces — reduces qubit count before VQE. Verdict: useful for scaling to Kagome, but requires significant new infrastructure. Future work.


---

## 9. V6.1 Implementation Lessons

Key learnings from implementing the V6.1 hardware deployment pipeline.

### 9.1 EstimatorV2 Observable Behavior

A single multi-term `SparsePauliOp` submitted to EstimatorV2 returns a **SCALAR** (the weighted sum of all terms' expectation values). A **LIST** of individual `SparsePauliOp` objects returns an **ARRAY** (one value per observable). This applies to both `StatevectorEstimator` and IBM Runtime `EstimatorV2`. For per-site/per-bond measurements (needed for phase classification), always submit observables as a list of single-term operators — never as a grouped multi-term operator.

### 9.2 Inhomogeneous ZNE Implementation

No existing library implements Uvarov et al. 2024's inhomogeneous ZNE approach. Mitiq uses gate folding (a fundamentally different paradigm — it amplifies noise uniformly by repeating gate sequences). Our implementation uses `generate_preset_pass_manager(initial_layout=[...])` to transpile the same logical circuit onto different physical qubit subsets, then `compute_ces(transpiled)` to measure the actual circuit CES for each layout, then linear regression to extrapolate observables to CES=0. The topology CES (used during layout *selection*) is a fast heuristic; the circuit CES (used for the ZNE *extrapolation axis*) is the true value from the transpiled circuit.

### 9.3 NNConv Aggregation

Use `aggr="add"` (not `"mean"`) for NNConv layers, for consistency with GINConv's WL-test equivalence (Xu et al. 2019, "How Powerful are Graph Neural Networks?"). Mean aggregation loses node degree information — two nodes with different numbers of neighbors but the same average neighbor features become indistinguishable. Sum aggregation preserves this structural information, which matters for distinguishing lattice sites with different coordination numbers (e.g., edge vs bulk sites in ladders).

### 9.4 Calibration Freshness on Modern Backends

Newer IBM backends using the Target API may not expose calibration timestamps via `backend.properties()`. The `properties()` method returns `None` on these backends. Rather than blocking inhomogeneous ZNE (which requires calibration error rates for layout selection), we default to assuming fresh calibration when the timestamp is unavailable. The error rates themselves are still accessible via `backend.target[op_name].get((q0, q1)).error`.

### 9.5 No Reusable Libraries Found

For our specific V6.1 components — inhomogeneous ZNE, layout selection on heavy-hex topology, and weight gradient analysis — no existing libraries or paper repositories provide reusable code. Mitiq handles gate-folding ZNE only. Qiskit Runtime handles DD/twirling/TREX natively via EstimatorV2 options (no custom implementation needed for those). The NN extrapolator (Sun et al. 2025) uses a standard `sklearn.MLPRegressor` — no specialized library required.
