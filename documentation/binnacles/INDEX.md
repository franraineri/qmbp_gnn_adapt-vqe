# Índice Detallado — `documentation/binnacles`

**Archivos**: 28 documentos markdown
**Total**: 80,034 palabras, 11,406 líneas

---

## 1. [binnacle-N10.md](binnacle-N10.md)
**Binnacle — N=10 Scaling Experiments**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-04 | 1436 | 13,158 | 80.8 KB | 252 | 5 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> > Experiments testing the V6 pipeline at N=10 qubits (1D TFIM chain).

**Métricas clave**: ΔE/gap > 10% · ΔE/gap = 4.69% · ΔE/gap = 4.44% · ΔE/gap = 203% · ΔE/gap=11.75% · ΔE/gap=2.74%

**Hallazgos / Aseveraciones** (8):

- Experiments testing the V6 pipeline at N=10 qubits (1D TFIM chain). All experiments use the best configuration from the N=6 hyperparameter sweep: VQE 5 restarts, maxiter=1000, MPNN…
- *After (V6.1)**: ΔE/gap = 4.44% ✅, checklist 4/4
- The weak-rung ladder is a partial success — it demonstrates the NNConv/edge-feature path works and dramatically improves over uniform-J ladder. However, HVA p=2 remains insufficien…
- Dense grid provides marginal improvement for deployment but is valuable for gradient analysis (more data points for peak detection). For thesis gradient analysis figure, use dense …
- *Success criteria: ❌ FAIL**
- *MPNN capacity must scale with system size.** At N=6 (6 nodes, 5 edges), h=64 is optimal and h=128 overfits. At N=10 (10 nodes, 9 edges), h=128 is optimal and h=64 underfits. The r…
- *The HVA expressibility ceiling degrades with N.** At N=6, fidelity reaches 0.995 at h=1.4. At N=10, fidelity only reaches 0.992 at h=1.5. The same p=2 circuit has to control more …
- *The valid operating regime shifts outward with N.** N=6 works at h≥1.4. N=10 only works at h≥1.5. The critical region (h≈1.0) becomes progressively harder because finite-size effe…

**Contenido** (33 secciones):

- **2026-05-04 — V6.0 Benchmark — N=10 chain, best config, h=1.5** (L9)
  - Configuration
  - Per-Run Results (Adapt-VQE at h=1.5)
  - Aggregate Statistics
  - Analysis: N=10 vs N=6
  - Key Findings
  - Next Steps for N=10
- **2026-05-05 — N=10 Hyperparameter Sweep (7 experiments, 14 executions)** (L62)
  - Methodology
  - Results
  - Key Findings
  - Recommended Configuration for N=10
  - Expected Checklist by Test Point (N=10)
  - N=10 vs N=6 Comparison (best config for each)
- **Key Lessons Learned — N=10** (L136)
- **Conceptual Note: Does Data Augmentation Make Sense?** (L153)
- **2026-05-08 14:45 — Parametric V6.1 Run (5 configs, N=10)** (L181)
- **2026-05-08 14:48 — Parametric V6.1 Run (1 configs, N=10)** (L198)
- **2026-05-08 14:49 — Parametric V6.1 Run (1 configs, N=10)** (L211)
- **2026-05-08 14:50 — Parametric V6.1 Run (1 configs, N=10)** (L224)
- **2026-05-08 — V6.1 Parametric Exploration: Analysis & Learnings** (L237)
  - Context
  - Consolidated Results
  - Key Findings
  - Updated Recommended Configuration for N=10
  - Expected Checklist by Test Point (N=10, V6.1, seed=43, patience=500)
  - V6.1 Feature Validation Summary
  - … (+1 más)
- **2026-05-08 15:59 — Parametric V6.1 Run (1 configs, N=10)** (L345)
- **2026-05-08 16:00 — Parametric V6.1 Run (1 configs, N=10)** (L358)
- **2026-05-08 — V6.1 Advanced Exploration: Ladder + Dense Grid** (L371)
  - Context
  - Results
  - Analysis: Ladder with Non-Uniform J (NNConv + Edge Features)
  - Analysis: Dense H-Grid (40 points)
  - Comparison: Previous Binnacle "Denser Grid" Finding
  - Updated V6.1 Feature Validation
  - … (+1 más)
- **2026-05-08 16:23 — Thesis Table 4.3 (N=10, 3 seeds × 2 h_test)** (L459)
- **2026-05-08 — Thesis Results Consolidation: Final Learnings** (L480)
  - What We Learned from 15 Definitive Runs (Tables 4.2 + 4.3)
  - Status: Thesis-Ready
- **2026-05-14 13:39 — Parametric V6.1 Run (1 configs, N=10)** (L554)
- **2026-05-14 13:42 — Parametric V6.1 Run (1 configs, N=10)** (L567)
- **2026-05-14 13:45 — Parametric V6.1 Run (1 configs, N=10)** (L580)
- **2026-05-14 13:50 — Parametric V6.1 Run (1 configs, N=10)** (L593)
- **2026-05-14 13:52 — Parametric V6.1 Run (1 configs, N=10)** (L606)
- **2026-05-14 14:18 — Parametric V6.1 Run (1 configs, N=10)** (L619)
- **2026-05-14 — V6.1 Diagnostics Validation + N=10 Noisy Sweep** (L632)
  - Context
  - Parametric Runs Summary (6 executions)
  - Key Diagnostic Metrics (from n10_patience500, seed=43)
  - Noisy Simulation Sweep (N=10, 6 h-values × 3 modes)
  - Analysis: Why ZNE Fails at N=10 but Works at N=6
  - Confirmed Findings (Reproducible)
  - … (+1 más)
- **2026-05-14 — N=6 Noisy Sweep with Diagnostics (Reference Comparison)** (L712)
- **2026-05-14 — N=12 Attempt: Computational Limits** (L729)
- **2026-05-14 — Retrospective: What We've Learned & Experiment Design Principles** (L743)
  - The Story Arc (V3 → V6.1, 40+ experiments)
  - What Experiments Taught Us vs. What Was Redundant
  - Principles for Future Experiments
  - What Experiments Are Actually Worth Running Next
  - What NOT to Run
- **2026-05-14 — Literature Analysis: Was the ZNE Failure Predictable?** (L811)
  - Short Answer — Yes, Three Independent Papers Predicted This
  - Paper 1: Tsubouchi, Sagawa & Takagi (2023) — Fundamental Cost Bound
  - Paper 2: Uvarov et al. (2024) — The Linearity Assumption
  - Paper 3: Rabinovich et al. (2025) — CLP-ZNE as the Solution
  - Paper 4 (Supporting): Wang et al. (2021) — Noise-Induced Barren Plateaus
  - Summary: What the Literature Told Us (That We Didn't Listen To)
  - … (+1 más)
- **2026-05-14 — Defined Next Steps Before Real QPU** (L874)
  - Priority 1: Fix ZNE at N=10 (Simulation — No QPU Needed)
  - Priority 2: Validate N=6 on Real QPU (Needs Credentials)
  - Priority 3: Gradient Analysis Figure (No Execution Needed)
  - What NOT to Do Before QPU
  - Implementation Order
  - New Bibliography Entries Needed
- **2026-05-14 — Experiment A Results: ZNE Layout Scaling (Partial)** (L946)
  - Hypothesis
  - Results (n_layouts=3 and 7 completed; 10 cancelled due to excessive CPU/time)
  - CES Distribution (7 layouts)
  - Conclusion: **Failure is fundamental, not statistical.**
  - Performance Note
  - What This Means for Next Steps
  - … (+1 más)
- **2026-05-15 — Experiment B: DD Pre-Mitigation (BLOCKED)** (L1004)
  - Hypothesis
  - Result: DD could NOT be applied on FakeTorino
  - Comparison (effectively no-DD vs no-DD due to fallback)
  - What This Means
  - Updated Conclusion for ZNE at N=10
  - Do NOT Repeat
- **2026-05-15 — Experiments A' + Gate Folding: CANCELLED (Resource Limits)** (L1051)
  - What Was Attempted
  - Result: Cancelled after 44 minutes at 500% CPU
  - Why This Happened
  - Final Assessment: Local Noisy Simulation at N=10 is Impractical
  - Definitive Conclusion: N=10 Noisy Simulation Experiments Are Complete
  - What Remains Valuable (No More Local Noisy Experiments)
  - … (+1 más)
- **2026-05-15 — Experiment E (Extended): Gradient Analysis Figures for Thesis** (L1099)
  - What Was Generated
  - Key Results
  - Thesis Interpretation
  - Figure Captions (for thesis)
  - Runtime
- **2026-05-15 — Deep Retrospective: What We Learned From This Session** (L1142)
  - The Complete Experiment Arc (2026-05-14 to 2026-05-15)
  - What Worked
  - What Didn't Work
  - The One Key Insight
- **2026-05-15 — Hardware Execution Plan: How to Make It Valuable** (L1227)
  - Philosophy: Every QPU Second Must Teach Something
  - Execution Strategy: Three Tiers
  - EstimatorV2 Configuration for Hardware
  - What NOT to Do on Hardware
  - Data to Record for Every Hardware Job
  - Success Criteria (Thesis-Grade)
  - … (+1 más)
- **2026-05-18 — Cross-Analysis of Last Week's Results (May 14-15 Data)** (L1340)
  - Context
  - Finding 1: Per-Site Observable Inhomogeneity Under Noise
  - Finding 2: ZNE Gain Uniformity Across h-Values
  - Finding 3: MPNN Prediction is Perfect — All Error is Circuit-Limited
  - Finding 4: DD Experiment Timing Anomaly Explained
  - Finding 5: CES Outlier Pattern at N=10
  - … (+2 más)

---

## 2. [binnacle-N6.md](binnacle-N6.md)
**PoC Development Binnacle**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-04-22 | 910 | 7,076 | 42.2 KB | 215 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> PoC V3 executed successfully but showed bottlenecks in Phase 2 (VQE optimization) and Phase 3 (MLP prediction accuracy). V4 applies targeted improvements.

**Métricas clave**: ΔE/gap = 4.37% · fidelity=0.0001 · R²=0.97 · ΔE/gap < 4% · ΔE/gap > 5% · ΔE/gap=5.19%

**Hallazgos / Aseveraciones** (8):

- ΔE/gap at h=1.4: 1.60% ✅
- `hybrid_cost()`: weighted energy + local-observable matching (α=0.5, ×100 scaling on obs term)
- `observable_cost()`: per-site comparison against site-averaged exact values
- Bidirectional sweep: descending (h=2→0) + ascending (h=0→2), both using `hybrid_cost` + `multi_start_vqe`
- Best-of-two selection per h-point
- `compute_energy_loss()`: energy-aware loss term (λ=0.1, every 50 epochs, batch=4)
- Used `.detach()` on model output before quantum eval
- MLP: 1→32→32→4, 4000 epochs, patience=150

**Contenido** (20 secciones):

- **2026-04-22 — PoC V3 → V4 Optimization** (L3)
  - Context
  - V3 Bottleneck Analysis
  - V4 Changes Applied
  - V4 Results
  - Key Takeaways
- **2026-04-22 — Python 3.12 Migration + V4 Re-run** (L72)
  - Environment Update
  - V4 Re-run Results (Python 3.12)
  - Analysis
- **2026-04-22 — PoC V5.0 and V5.1 Results** (L116)
  - V5.0 Changes (from V4)
  - V5.0 Results
  - V5.0 Failure Analysis
  - V5.1 Changes (bug fixes from V5.0)
  - V5.1 Results
  - V5.1 Failure Analysis
  - … (+1 más)
- **2026-04-22 — PoC V5.1 (fixed) Results** (L206)
  - Changes Applied
  - Results
  - Root Cause
  - Lesson Learned
  - V4 Remains Best End-to-End (2/6)
  - Next Step
- **2026-05-04 — PoC V6.0 Architecture Upgrade** (L252)
  - Context
  - V6 Changes (from V4)
  - V6 End-to-End Validation (N=6 TFIM, reduced training)
  - Key Takeaways
- **2026-05-04 — V6.0 Benchmark (3 runs, h_test=1.25)** (L315)
  - Configuration
  - Per-Run Results (Adapt-VQE at h=1.25)
  - Aggregate Statistics
  - Key Observations
- **2026-05-04 — V6.0 Hyperparameter Sweep (7 experiments)** (L350)
  - Methodology
  - Results Summary
  - Analysis
  - Next Steps
- **2026-05-04 — Key Lessons Learned (V3→V6 Summary)** (L394)
  - What Works
  - What Doesn't Work
  - Structural Limits
  - Current Best Configuration
- **2026-05-04 — V6.0 Hyperparameter Sweep Round 2 (7 experiments, 14 executions)** (L421)
  - Experiment Design
  - Results
  - Key Findings
  - Recommended Configuration (Final)
  - Expected Checklist by Test Point
- **2026-05-05 — V6.0 New Parameters Exploration (N=6, h-grid density + restart_sigma)** (L480)
  - Motivation
  - Results
  - Analysis
  - Conclusion
- **2026-05-05 — V6.0 Advanced Techniques (N=6, augmentation + GATConv)** (L513)
  - Motivation
  - Results
  - Analysis
  - Updated Recommendation
- **2026-05-05 — V6.0 Benchmark — routing test N6** (L552)
  - Configuration
  - Per-Run Results (Adapt-VQE at h=1.5)
  - Aggregate Statistics
  - Key Observations
- **Key Lessons Learned — N=6 (Final Summary)** (L585)
- **2026-05-08 14:26 — Notebook Execution — v61-run-1** (L605)
  - Environment
  - poc_v6_phases1_2.ipynb — ✅ PASS
  - poc_v6_phases3_4.ipynb — ✅ PASS
  - Observations (auto-generated)
  - Comparison with Previous Run
- **2026-05-08 14:27 — Notebook Execution — v61-run-2** (L667)
  - Environment
  - poc_v6_phases1_2.ipynb — ✅ PASS
  - poc_v6_phases3_4.ipynb — ✅ PASS
  - Observations (auto-generated)
  - Comparison with Previous Run
- **2026-05-08 14:28 — Notebook Execution — v61-run-3** (L729)
  - Environment
  - poc_v6_phases1_2.ipynb — ✅ PASS
  - poc_v6_phases3_4.ipynb — ✅ PASS
  - Observations (auto-generated)
  - Comparison with Previous Run
- **2026-05-08 14:35 — Parametric V6.1 Run (5 configs)** (L791)
- **2026-05-08 16:23 — Thesis Table 4.2 (N=6, 3 seeds × 3 h_test)** (L807)
- **2026-05-08 — Thesis Consolidation: N=6 Final Assessment** (L832)
  - Summary
  - Key Insight: V6.1 4-Metric Checklist vs V6.0 6-Metric
  - N=6 is Complete
- **2026-05-18 — Cross-Analysis: N=6 Noisy Sweep Contrast with N=10** (L857)
  - Context
  - Per-Site Observable Homogeneity at N=6
  - CES Values: Why ZNE Works at N=6
  - ZNE Gain Comparison (Quantitative)
  - Thesis Implication
  - No New Experiments Needed

---

## 3. [binnacle-bond-resolved-hva.md](binnacle-bond-resolved-hva.md)
**Binnacle — Bond-Resolved HVA: Scaling Toward Quantum Advantage**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-05 | 420 | 2,650 | 17.0 KB | 58 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `scaling`

> > Local (per-bond, per-site) parametrization of HVA circuits to increase

**Métricas clave**: ΔE/gap < 0.23% · R²=0.997 · gain=+30.8% · ΔE/gap = 0.16 · R² > 0.99 · ΔE/gap < 5%

**Hallazgos / Aseveraciones** (8):

- Local (per-bond, per-site) parametrization of HVA circuits to increase expressibility without increasing depth. Key step toward classical intractability of the variational paramete…
- Bond-resolved VQE converges identically to global HVA on chain_1d
- Fidelity > 0.998 across entire valid regime. ΔE/gap < 0.4% at all
- θ_zz shows very little spatial variation (std < 0.002), but θ_x shows
- Bond-resolved HVA on 2D square lattice converges excellently
- N=12 (3×4) with 29 bond-resolved parameters converges to fid > 0.997
- *+49.7% improvement on heavy-hex** (IBM Torino native topology) — a free lunch
- *GNN predicts 19-dim θ_opt** with ΔE/gap < 0.23% — proves GNN is viable at high dim

**Contenido** (11 secciones):

- **Executive Summary** (L13)
- **Thesis Statements (for Chapter 5/6)** (L29)
- **Motivation** (L62)
  - The Simulability Problem
  - Bond-Resolved as Scaling Lever
  - Literature Support
- **Hypothesis** (L97)
- **Design** (L110)
  - Circuit Architecture
  - Parameter Count by Topology (p=1)
  - MPNN Architecture Adaptation
  - Experiment Plan
- **Implementation Plan** (L170)
  - Files to Create/Modify
  - What Does NOT Change
- **Results** (L189)
  - Section 1: N=6 Chain Convergence ✅ PASS (23.5s)
  - Section 2: N=10 Heavy-Hex Convergence ✅ PASS (59.8s)
  - Section 3: Global vs Bond-Resolved ✅ PASS (190.0s)
  - Section 4: Parameter Spatial Structure ✅ PASS (35.9s)
- **Scaling Results (2D Square Lattice) — 2026-06-05** (L253)
  - Section S1: Square N=9 (3×3) ✅ PASS (35.9s)
  - Section S2: Square N=12 (3×4) ✅ PASS (360.3s)
  - Section S3: MPNN Training ❌ FAIL (technical bug)
  - Section S4: Noisy Simulation ❌ FAIL (API mismatch)
- **Interpretation & Key Findings** (L305)
  - Finding 1: Bond-resolved is a "free lunch" on heavy-hex
  - Finding 2: Improvement scales with topology non-uniformity
  - Finding 3: The GNN becomes essential
  - Finding 4: Toward quantum advantage
- **Next Steps** (L342)
  - Completed ✅
  - Findings from N=16 (valid negative result)
  - No further simulation steps needed
  - Remaining (hardware-only)
- **E3 Bond-Resolved Scaling: N=40 (2026-06-08)** (L378)
  - Section 0: Sanity Check ✅
  - Section 1: VQE Convergence ❌ (near-miss)
  - Analysis
  - Next Steps

---

## 4. [binnacle-comparative-analysis.md](binnacle-comparative-analysis.md)
**Binnacle — Comparative Analysis Suite**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-21 | 159 | 1,219 | 7.2 KB | 37 | 0 |

**Tags**: `hardware`, `mpnn`, `thesis`, `pea`, `scaling`, `noise`

> Before scaling the framework to larger systems or hardware, we need to understand exactly what each component contributes and where the limits are. This session

**Métricas clave**: ΔE/gap > 400% · ΔE/gap < 3%

**Contenido** (9 secciones):

- **2026-05-21 — Systematic Pipeline Characterization** (L3)
  - Context
- **Comparison 1: Warm-Start Gain vs h** (L13)
- **Comparison 2: Error Decomposition** (L32)
- **Comparison 3: Ablation Study** (L55)
- **Comparison 4: Training Efficiency** (L75)
- **Comparison 5: Scaling Law** (L92)
- **Analysis A: MPNN Jacobian (Phase Transition Detection)** (L112)
- **Analysis B: Zero-Shot Phase Classification** (L130)
- **Summary of All Findings** (L149)

---

## 5. [binnacle-cross-n-zero-shot.md](binnacle-cross-n-zero-shot.md)
**Binnacle — Cross-N Zero-Shot GNN Generalization**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 200 | 1,288 | 8.6 KB | 44 | 1 |

**Tags**: `mpnn`, `thesis`, `scaling`, `topology`, `dmrg`

> Date: 2026-06-08

**Métricas clave**: ΔE/gap = 423% · ΔE/gap = 160%

**Hallazgos / Aseveraciones** (1):

- Removing BatchNorm is the primary fix (18.5%→0.2%). N/100 provides

**Contenido** (8 secciones):

- **Hypothesis** (L7)
- **Background** (L14)
- **Discovery: BatchNorm is the Failure Mode** (L20)
  - Observation
  - Full Validation Matrix (completed 2026-06-08)
  - Multi-Seed Robustness (target N=60)
  - Ablation E: norm_type=none WITHOUT N-feature (2 features only)
  - Root Cause Analysis
  - Why theta_x Is Specifically Affected
- **Fix** (L80)
- **Results (Validated)** (L100)
  - v3 GNN (norm_type=none, 3 features):
- **Limitations and Pending Validation** (L116)
  - What is confirmed (high confidence):
  - What is NOT yet confirmed (pending bond-resolved experiment):
  - Applicability Scope
- **Files** (L140)
- **Cross-Topology Transfer: First Attempt (2026-06-08/09)** (L157)
  - What Was Tested
  - Root Cause Analysis
  - Fix Implemented (2026-06-09)
  - Next Steps

---

## 6. [binnacle-d1-weight-space-phase-detection.md](binnacle-d1-weight-space-phase-detection.md)
**Binnacle: D1 — Zero-QPU Phase Detection from MPNN Weight Space**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-22 | 235 | 1,519 | 9.1 KB | 27 | 1 |

**Tags**: `mpnn`, `tesis`, `tfim`, `pea`, `scaling`

> > Date: 2026-05-22

**Hallazgos / Aseveraciones** (8):

- Date: 2026-05-22 Runs: N=6 (run_20260522_125609), N=10 (run_20260522_132546) Seeds: 42, 43, 44 Status: Hypothesis CONFIRMED — novel thesis contribution
- Peak at h ≈ 0.66-0.70 (seeds 43, 44) — within 0.3 of h_c=1.0
- Peak at h ≈ 2.46 (seed 42) — meaningless, MPNN didn't learn structure
- θ_opt(h) changes rapidly (the ground state structure changes)
- The MPNN must encode this rapid change in its weights
- The prediction gradient ||dθ_pred/dh|| reflects this rapid change
- In the paramagnetic phase (h >> 1): θ_opt varies smoothly
- In the ferromagnetic phase (h << 1): θ_opt is nearly constant

**Contenido** (10 secciones):

- **1. Hypothesis** (L10)
- **2. Method** (L19)
- **3. Results** (L28)
  - Per-Seed Peak Locations
  - Aggregate Results
  - Key Observation: Seed Quality Matters
- **4. Physical Interpretation** (L66)
  - Why does ||dθ_pred/dh|| peak near h_c?
  - Why does MPNN-B peak at the training boundary?
  - Connection to Hernandes et al. (2025)
- **5. Scaling Analysis** (L100)
- **6. Limitations and Caveats** (L119)
- **7. Thesis Value** (L139)
- **8. What Could Be Validated Further** (L160)
  - High-Value Extensions (recommended)
  - Lower-Priority Extensions
- **9. Recommended Next Experiment — RESOLVED** (L187)
- **10. Summary** (L211)
  - Key Discovery

---

## 7. [binnacle-e3-bond-resolved-scaling.md](binnacle-e3-bond-resolved-scaling.md)
**Binnacle — E3: Bond-Resolved HVA at N=40**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 146 | 885 | 5.5 KB | 32 | 0 |

**Tags**: `mpnn`, `thesis`, `scaling`, `topology`

> > Bond-resolved HVA (79 parameters, p=1) scaled to N=40 using MPS-VQE.

**Métricas clave**: ΔE/gap < 10% · ΔE/gap < 5%

**Hallazgos / Aseveraciones** (8):

- Bond-resolved HVA (79 parameters, p=1) scaled to N=40 using MPS-VQE. Tests GNN necessity in high-dimensional variational parameter spaces. Date: 2026-06-08 Status: ✅ Section 0+1 co…
- At 79 variational parameters (bond-resolved HVA, N=40), cold-start VQE with COBYLA requires >1000 evaluations (133 min on MPS) and still fails to reach the 5% ΔE/gap threshold. In …
- H1: VQE converges to ΔE/gap < 5% at easy h-points (deep paramagnetic) — ❌ REJECTED (cold-start)
- H3: The cold-start failure proves GNN warm-start is NECESSARY — ✅ CONFIRMED
- Bond-resolved circuit with uniform parameters produces IDENTICAL
- [x] Section 0: Sanity ✅ (19.4s)
- [x] Section 1: Cold-start convergence ❌ → THESIS RESULT (6.59% COBYLA, 13.92% SPSA)
- [x] Warm-start validation: 0.75% with uniform init (no optimizer needed) ✅

**Contenido** (9 secciones):

- **Hypothesis** (L12)
- **Configuration** (L25)
- **Section 0: Sanity Check — ✅ PASS (19.4s)** (L40)
- **Section 1: VQE Convergence — ❌ FAIL (cold-start, 11066s)** (L58)
  - θ_opt Analysis (COBYLA converged state)
- **Key Finding: GNN Warm-Start is NECESSARY at 79 Parameters** (L83)
- **Thesis Value (Chapter 6)** (L98)
- **Comparison with Previous Results** (L119)
- **Next Steps** (L130)
- **Files** (L140)

---

## 8. [binnacle-e4b-hardware-readiness.md](binnacle-e4b-hardware-readiness.md)
**Binnacle — E4b Hardware Readiness Validation**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-03 | 380 | 2,607 | 15.8 KB | 70 | 0 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `tfim`, `scaling`

> > Fecha: 2026-06-03

**Métricas clave**: gain: +91.8% · gain: +88.8% · ΔE/gap=14% · ΔE/gap<5% · gain=+62.7% · R²=0.997 · gain=+84.7%

**Hallazgos / Aseveraciones** (8):

- Fecha: 2026-06-03 Script: `scripts/run_e4b_hardware_readiness.py` Objetivo: Validar que tfim_longitudinal (g>0) se comporta comparable al TFIM estándar en las dimensiones críticas …
- La extensión TFIM + longitudinal (H = −J·ZZ − h·X − g·Z) es completamente compatible con el pipeline GNN-HVA en simulación (p=2, g≤0.5, fid≥0.98) con cero overhead de gates 2Q. A p…
- chain_1d es seed-independent (std=0.004) ✅
- ZNE fails at N=10 p=2 (6/7 negative) ✅ — consistente con nuestro finding
- 7 grid points sufficient ✅ — nuestra Section 8 usa exactamente 7
- TFIM mean gain: +91.8%
- Longitudinal mean gain: +88.8%
- *Diferencia: 2.9% < 5%** → ZNE funciona equivalentemente

**Contenido** (9 secciones):

- **Contexto** (L11)
- **Resultados** (L28)
  - Section 6: ZNE Noisy Simulation — ✅ H6 CONFIRMED
  - Section 7: θ-Smoothness — ✅ H7 CONFIRMED
  - Section 8: MPNN Generalization Gap — ✅ H8 CONFIRMED
  - Section 9: g-Sensitivity Sweep — ✅ H9 CONFIRMED (con caveat importante)
  - Section 10: Phase Classification — ✅ H10 CONFIRMED
- **Análisis: ¿Se Necesitan Más Experimentos?** (L170)
  - Cobertura actual del tfim_longitudinal
  - Gaps Identificados
- **Decisión Final: NO SE NECESITAN MÁS EXPERIMENTOS** (L231)
- **Comparación con Digest (herramientas existentes)** (L239)
  - E4 (standard HVA) vs E4b (extended HVA) — desde el digest
  - Noisy runs — ZNE comparison from digest
- **Resumen para la Tesis (Chapter 5)** (L269)
  - Claim principal
  - Datos de soporte (5 hipótesis, todas confirmadas)
  - Limitación documentada
- **Archivos de Resultados** (L299)
- **Validación Cruzada con Herramientas del Proyecto** (L314)
  - compare.py --all
  - scripts.digest --kind noisy (ZNE baselines)
  - scripts.digest --kind noiseless --stats
  - analysis/verify_claims.py
  - analysis/scan_coverage.py
- **Conclusión del Análisis Cruzado** (L368)

---

## 9. [binnacle-e5-scaling-extensions.md](binnacle-e5-scaling-extensions.md)
**Binnacle — E5: Scaling Extensions (HE + NLCE)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 162 | 1,017 | 6.4 KB | 37 | 0 |

**Tags**: `tesis`, `tfim`, `scaling`, `dmrg`

> > Fecha: 2026-06-08

**Métricas clave**: ΔE/gap=100% · ΔE/gap<5%

**Hallazgos / Aseveraciones** (1):

- Fecha: 2026-06-08 Experiment ID: E5_SCALING_EXT Estado: PARCIALMENTE COMPLETADO (3/5 secciones pasaron) Runner: `scripts/experiment_runners/bond_resolved/run_scaling_extensions.py`…

**Contenido** (6 secciones):

- **Hipótesis** (L12)
- **Resumen de Resultados** (L20)
- **Findings Principales** (L32)
  - Finding 1: HE reduce dimensionalidad 7.6× (Sección 3)
  - Finding 2: L-BFGS-B impracticable para bond-resolved (bug fix)
  - Finding 3: NLCE TFIM converge al límite termodinámico (Sección 4)
  - Finding 4: NLCE Frustrated TFIM — resultado novel (Sección 5)
  - Finding 5: Secciones 1-2 fallaron por límite DMRG_QUBIT_LIMIT=100
- **Problemas Técnicos Encontrados** (L117)
- **Thesis Tables Generadas** (L133)
  - Table 5.25: MPS Exactness (parcial — N=120 pendiente)
  - Table 5.26: Hamiltonian Engineering vs GNN
- **Archivos Clave** (L154)

---

## 10. [binnacle-gate-folding-zne.md](binnacle-gate-folding-zne.md)
**Binnacle — Gate-Folding ZNE Implementation & Validation**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-04 | 838 | 4,769 | 32.1 KB | 131 | 17 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> > Implementation of digital gate-folding ZNE as an alternative to CES-based

**Métricas clave**: R²=0.04 · R²=0.997 · gain=0% · gain=+12% · gain=16% · R²>0.99

**Hallazgos / Aseveraciones** (8):

- Implementation of digital gate-folding ZNE as an alternative to CES-based inhomogeneous ZNE. Motivated by the hardware rehearsal finding that CES-ZNE fails on heavy_hex due to unif…
- With enough candidates (≥20), CES-ZNE can find spread
- The +10–15% gain IS the correct result for this regime
- *Noise factors**: [1, 3, 5] (standard gate-folding)
- *Extrapolator**: linear (E(nf) = a·nf + b → E(0) = b)
- *Backend**: FakeTorino (local noise simulation)
- *Comparison baseline**: CES-ZNE with `select_layouts_by_circuit_ces`, 3 layouts
- Mean GF-ZNE gain: +9–15% (always positive)

**Contenido** (20 secciones):

- **Motivation** (L12)
- **Implementation** (L27)
  - Core Functions (in `src/qmbp_simulation/execution/noisy_utils.py`)
  - Key Design Decisions
  - Exports
- **Experiment Results** (L62)
  - Configuration
  - Cross-Topology Summary
  - Per-h-Point Detail
- **Key Findings** (L111)
  - 1. GF-ZNE provides consistent positive gain across ALL topologies
  - 2. Both methods have excellent R² (>0.99)
  - 3. CES-ZNE failure mode: extrapolation in wrong direction
  - 4. heavy_hex N=10: GF-ZNE works but doesn't dominate
  - 5. Ladder topology: CES-ZNE nearly useless
- **Comparison with Hardware Rehearsal Findings** (L153)
- **Recommendation for Hardware Deployment** (L169)
  - IBM Runtime Configuration (for real hardware)
- **Supplementary Analysis** (L200)
  - Circuit Depth Impact
  - Gain Reconciliation with Previous CES-ZNE Results
  - Statistical Summary (all 12 h-points, 3 topologies)
  - Conclusion
- **Files Modified/Created** (L259)
- **Result Files** (L269)
- **Reproduction** (L279)
- **Addendum: PEA-ZNE Comparison (2026-06-04, run 2)** (L300)
  - 3-Way Results (chain_1d N=6 p=1)
  - Summary
  - Why PEA Outperforms Gate-Folding
  - Trade-offs
  - Updated Recommendation for Hardware Deployment
  - Result File
- **PEA-ZNE Comprehensive Validation (2026-06-04, run 3)** (L373)
  - Multi-Seed Results (chain_1d N=6 p=1, seeds=[42,43,44], 4 h-points)
  - Overall Statistics (12 evaluations: 3 seeds × 4 h-points)
  - Verdict
  - Why PEA Achieves ~95% Gain (vs GF-ZNE ~11%)
  - Result File
  - project_health Verification
- **PEA-ZNE Hardware Readiness (2026-06-04, heavy_hex N=10 p=1)** (L441)
  - Experiment: `PEA_HW_READY`
  - Results
  - Key Metrics
  - Critical Finding: GF-ZNE Fails on heavy_hex N=10 p=1
  - Implications for Real Hardware
  - Result Files
  - … (+1 más)
- **PEA Full Pipeline Validation (2026-06-04, MPNN + PEA-ZNE + Classify)** (L524)
  - Experiment: `PEA_PIPELINE`
  - Results
  - PEA-ZNE with MPNN-Predicted Parameters
  - Critical Finding: PEA Works with Imperfect Parameters
  - Phase Classification Note
  - Result File
- **Final Experiment Summary (all 5 ZNE experiments)** (L585)
- **Final Consolidated Analysis (2026-06-04)** (L599)
  - Coverage Complete (via `compare.py --zne`)
  - Analysis Tool Added
  - Definitive Ranking
- **Cross-Topology Validation (2026-06-04, ZNE_CROSS_TOPO)** (L633)
  - Motivation
  - Configuration
  - Results
  - Statistical Summary (18 evaluations, 3 topologies)
  - Key Findings
  - Updated Definitive Ranking (all 6 experiments, 60+ h-points)
  - … (+3 más)
- **Limitations & Caveats (for thesis Chapter 5)** (L732)
  - 1. PEA Gain in Simulation vs Hardware
  - 2. When ZNE Provides No Benefit
  - 3. PEA Requires qiskit-aer for Local Validation
  - 4. Phase Classification Does Not Need ZNE
  - 5. Comparison is Internal Only
- **Thesis Contribution Summary** (L782)
- **Addendum: QESEM Validation & Adaptive Strategy Refactor (2026-06-05)** (L804)
  - Literature Confirmation
  - Adaptive Strategy Change
  - New Complementary Techniques (same commit)

---

## 11. [binnacle-gnn-qem-validation.md](binnacle-gnn-qem-validation.md)
**Binnacle — GNN-QEM: Graph Neural Network for Quantum Error Mitigation**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-05 | 552 | 3,215 | 22.6 KB | 84 | 6 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `pea`, `topology`

> > GNN-based energy correction model that learns noise propagation patterns

**Métricas clave**: R²=0.9996 · R²=0.36

**Hallazgos / Aseveraciones** (2):

- GNN-based energy correction model that learns noise propagation patterns from hardware topology. Post-ZNE correction: E_corrected = E_noisy + ΔE_GNN. Date: 2026-06-05 Status: ✅ Mod…
- A GINConv GNN processing the hardware coupling map with qubit calibration features can predict energy correction magnitudes zero-shot on unseen topologies. In the standard setting …

**Contenido** (14 secciones):

- **Executive Summary** (L12)
- **Thesis Statements** (L32)
- **Motivation** (L81)
- **Architecture** (L99)
- **Implementation** (L128)
  - Core Components
  - Key Design Decisions
- **Experiment: Quick Validation (chain_1d N=6)** (L167)
  - Configuration
  - Results
  - Per-Sample Results
- **Caveats and Limitations** (L219)
- **Next Steps** (L238)
- **References** (L259)
- **Files Modified/Created** (L269)
- **Ablation Study: What Does the GNN Actually Learn? (2026-06-06)** (L287)
  - Motivation
  - Results Summary
  - Key Findings
  - Interpretation
  - Publishable Claim (Revised)
  - Result Files
- **VQE-Realistic Training + Circuit Selection Mode (2026-06-06)** (L358)
  - Context
  - Data
  - Results
  - Key Findings
  - Practical Impact
  - Updated Role Summary
  - … (+1 más)
- **Addendum: Pipeline Integration (2026-06-06)** (L431)
  - Integration into HardwareBackend
  - API
  - Modified Files
  - Design Decisions
- **Addendum: Post-ZNE Residual Validation (2026-06-06)** (L486)
  - Critical Finding
  - Interpretation
  - Conclusion for Deployment
  - Impact on Thesis
  - Updated HardwareBackend Logic

---

## 12. [binnacle-hamed-v7.md](binnacle-hamed-v7.md)
**Binnacle — Hamed's Feedback Experiments (V7)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-18 | 1039 | 7,693 | 46.2 KB | 242 | 0 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `scaling`

> Hamed Mohammadbagherpoor provided feedback on the GNN-HVA pipeline during a meeting.

**Métricas clave**: fidelity > 0.97 · fidelity=0.997 · ΔE/gap=1.36% · fidelity=0.992 · ΔE/gap=2.79% · ΔE/gap < 5%

**Hallazgos / Aseveraciones** (8):

- *Immediate:** Use SPSA as the optimizer for Phase 4 hardware deployment
- *Thesis:** Include Nevergrad comparison in optimization section (validates L-BFGS-B)
- *Thesis:** Include QRC comparison in warm-start section (validates MPNN scalability)
- *Future work:** Re-run noise-aware training with FakeTorino full noise model
- *Future work:** Run MPS experiment for N=20 scaling demonstration
- Rapin & Teytaud (2018) — Nevergrad library
- Kutvonen et al. (2020) — QRC optimization (Nature Sci Rep)
- arXiv:2510.00171 — QRC with Jaynes-Cummings model

**Contenido** (18 secciones):

- **2026-05-18 — Meeting Feedback & Experimental Validation** (L3)
  - Context
  - Pre-Experiment Analysis
- **Experiment 1: Nevergrad vs L-BFGS-B** (L32)
  - Hypothesis
  - Configuration
  - Results
  - Conclusion
  - Learning
- **Experiment 2: QRC → MLP Warm-Start** (L67)
  - Hypothesis
  - Configuration
  - Results
  - Conclusion
  - Learning
- **Experiment 3: MPS Simulation (Script Ready)** (L109)
  - Hypothesis
  - Status
  - Expected Outcome
- **Experiment 4: SPSA vs COBYLA vs L-BFGS-B Under Shot Noise** (L127)
  - Hypothesis
  - Configuration
  - Results
  - Conclusion
  - Learning
- **Experiment 5: Noise-Aware MPNN Training** (L164)
  - Hypothesis
  - Configuration
  - Results
  - Conclusion
  - Learning
- **Summary of Findings** (L206)
  - Action Items
  - New Bibliography Entries Added
- **2026-05-18 — Phase A Full Execution (Rigorous Protocol)** (L235)
  - Context
  - Experiment 4A: SPSA Hyperparameter Grid Search (DEFINITIVE)
  - Experiment 4B: SPSA Warm-Start Refinement (DEFINITIVE)
  - Experiment 1A: Nevergrad Fair Budget Comparison (DEFINITIVE)
  - Experiment 3A: MPS Accuracy Validation (ISSUE FOUND)
  - Phase A Summary
  - … (+1 más)
- **Cross-Comparison: V7 Phase A vs V6.1 Established Results** (L374)
  - Context
  - Optimizer Comparison (V7 1A vs V6.1 Phase 2)
  - SPSA Warm-Start (V7 4B) vs V6.1 MPNN Prediction
  - MPS Validation (V7 3A) vs V6.1 Statevector
  - V7 Findings That Inform V6.1 Pipeline
  - What V7 Adds Beyond V6.1
  - … (+1 más)
- **2026-05-18 — Phase B Partial Execution (3C + 2B)** (L444)
  - Experiment 3C: MPS VQE at N=20
  - Experiment 2B: QRC vs MPNN at N=10
  - Phase B Decisions
- **2026-05-18 — Phase C Execution + Deep Analysis** (L499)
  - Experiment 5A: Noisy VQE Data Generation
  - Experiment 5B: Noise-Aware vs Noiseless MPNN Training (DEFINITIVE)
  - Experiment 4C: SPSA vs COBYLA Under FakeTorino Noise (DEFINITIVE)
  - Experiment 3C (re-run): MPS VQE at N=20 with L-BFGS-B
  - Deep Analysis: What All Results Mean Together
  - Decisions for Remaining Experiments
  - … (+1 más)
- **2026-05-18 — Final Experiments (4E, 5E) + Closure** (L619)
  - Experiment 4E: SPSA + ZNE Integration
  - Experiment 5E: Iterative Refinement (3 Rounds)
  - Experiment 3C (final): MPS VQE at N=20 with Multi-Restart L-BFGS-B
- **V7 Experiment Suite — Final Summary** (L664)
  - Completed Experiments (12 of 22 planned)
  - Skipped Experiments (10 — justified by results)
  - Top-Level Conclusions for Thesis
- **2026-05-18 — Transfer Learning N=6→N=10 (DEFINITIVE NEGATIVE)** (L717)
- **2026-05-19 — Full V6.1 Pipeline at N=20 (First Execution)** (L740)
  - Per-Phase Results
  - Analysis
  - Comparison: V7 3C (direct VQE) vs Full Pipeline
  - Root Cause & Fix Path
  - Decision
- **2026-05-19 — Full V6.1 Pipeline at N=20 (Second Run, Energy Filter)** (L812)
  - Results
  - Comparison: Run 1 vs Run 2
  - Analysis
  - Key Learning
  - Conclusion for N=20 Pipeline
  - Decision
- **2026-05-19 — N=20 Pipeline SUCCESS (ΔE/gap = 1.75%) ✅** (L889)
  - Results
  - What Fixed It (comparison across 3 runs)
  - Key Insight
  - Thesis Claim (VALIDATED)
  - Optimal N=20 Configuration (DEFINITIVE)
- **2026-05-19 — Deep Analysis: N=20 Journey (3 Runs) & Universal Lessons** (L950)
  - The Story in Numbers
  - The Three Mistakes and Their Fixes
  - Universal Principles (Apply to ALL Future Scaling)
  - DO and DON'T List
  - Impact on Thesis

---

## 13. [binnacle-hamiltonian-candidates.md](binnacle-hamiltonian-candidates.md)
**Binnacle — Evaluación de Hamiltonianos Candidatos para el Framework**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-02 | 531 | 3,693 | 22.7 KB | 81 | 2 |

**Tags**: `zne`, `hardware`, `tesis`, `heisenberg`, `tfim`, `noise`

> Determinar qué Hamiltonianos adicionales son viables para el framework GNN-HVA bajo las

**Métricas clave**: R² > 0.99 · R²>0.99 · R²<0.9

**Hallazgos / Aseveraciones** (7):

- *Gate budget 2×:** RXX + RYY por bond = 4 CX, vs RZZ = 2 CX para TFIM
- *Estado inicial inadecuado:** |+⟩^N es el ground state de H_X, no del Kitaev
- *Expresividad insuficiente:** 3 params no capturan las correlaciones de pairing
- *El Kitaev NO es equivalente a TFIM:** Aunque ambos son free-fermion, el Kitaev
- Un ansatz diferente (no HVA standard)
- Estado inicial adaptado (BCS-like state)
- p≥3 capas o ansatz con más parámetros por capa

**Contenido** (13 secciones):

- **2026-06-02 — Investigación de Literatura para Nuevos Modelos** (L3)
  - Objetivo
  - Metodología
- **Criterios de Viabilidad (TODOS deben cumplirse)** (L21)
- **Candidato 1: TFIM + Campo Longitudinal (IMPLEMENTADO ✅)** (L34)
  - Soporte bibliográfico
- **Candidato 2: TFIM Frustrado (J₁-J₂ o NNN)** (L58)
  - Análisis de riesgo
  - Decisión: NO IMPLEMENTAR ahora
- **Candidato 3: Cadena de Kitaev (Topología)** (L88)
  - Soporte bibliográfico FUERTE
  - Análisis de viabilidad técnica
  - Estimación CX gates
  - Decisión: INVESTIGAR MÁS antes de implementar
- **Candidato 4: XXZ con Anisotropía Variable (Δ como parámetro)** (L153)
  - Decisión: DESCARTADO
- **Candidato 5: TFIM con Boundary Fields (condiciones de contorno)** (L179)
  - Soporte bibliográfico
  - Análisis
  - Decisión: NO PRIORIZAR (guardar para trabajo futuro si hay tiempo)
- **Candidato 6: Modelo de Schwinger Lattice (QED 1+1)** (L214)
  - Decisión: DESCARTADO (fuera del scope de la tesis: spin models)
- **Resumen de Evaluación** (L239)
- **Recomendación Final** (L252)
  - Para la tesis (implementar)
  - Para trabajo futuro (documentar pero no implementar)
  - Próximo paso concreto
- **Referencias Clave Nuevas (para agregar a bibliografía)** (L277)
- **ADDENDUM: Kitaev Chain Verification Results (2026-06-02)** (L291)
  - Resultado: 🔴 NO VIABLE
  - Root Cause Analysis
  - Conclusión Final
- **ADDENDUM 2: Análisis Detallado de Barreras del Framework para Kitaev (2026-06-03)** (L345)
  - Objetivo
  - Barrera 1: Estado Inicial |+⟩^N — Overlap nulo con ground state de Kitaev
  - Barrera 2: Budget CX — 20 CZ a N=6 p=1 (excede ZNE ≤ 18)
  - Barrera 3: Parámetros globales por capa — Insuficiente para capturar pairing
  - Barrera 4 (emergente): La combinación de problemas es peor que la suma
  - Tabla resumen: Reglas vs Cambios necesarios
  - … (+1 más)

---

## 14. [binnacle-hamiltonian-comparison.md](binnacle-hamiltonian-comparison.md)
**Binnacle — Cross-Hamiltonian Comparison & TFIM Longitudinal Extension**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-02 | 419 | 2,549 | 15.5 KB | 104 | 4 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `heisenberg`, `tfim`

> Systematically compare all Hamiltonians implemented in the GNN-HVA framework to determine

**Métricas clave**: ΔE/gap < 5% · fidelity ≥0.98 · ΔE/gap<5% · fidelity ≥ 0.98 · ΔE/gap = 0.009 · ΔE/gap=0.009

**Hallazgos / Aseveraciones** (8):

- The framework is extensible to modified Hamiltonians when the HVA circuit structure is adapted to match all terms. Adding a longitudinal field g·Z to the TFIM requires only an RZ l…
- Frustrated TFIM is a simulation-only model for N≥6. Hardware deployment
- *§5.1**: Standard TFIM results (existing, 430+ runs across 5 topologies)
- *§5.5**: Model extensibility — E4b demonstrates the HVA-matches-H principle
- *§5.6**: Negative results — Heisenberg confirms expressibility limits of p≤2
- *§5.7**: Future work — TFIM+longitudinal as stepping stone to 2D phase diagrams
- 47 unit tests validating Hamiltonian, circuit, registry, and expressibility
- 75 VQE data points (3 seeds × 5g × 5h) with 100% pass rate

**Contenido** (12 secciones):

- **2026-06-02 — Model Expressibility Analysis** (L3)
  - Objective
- **Part 1: Implemented Models (Registry Status)** (L14)
- **Part 2: TFIM Standard — Baseline Performance (from digest)** (L27)
- **Part 3: TFIM + Longitudinal Field (E4b) — Extension Results** (L46)
  - Background
  - E4b Execution (3 seeds × 5 g-values × 5 h-values = 75 data points)
  - Cross-Topology Validation (Section 2 of full validation)
  - Scaling (N=4,6,8 at g=0.3)
  - Hardware Viability
- **Part 4: Heisenberg Model — Negative Result (from V9 binnacle)** (L107)
  - Summary of 30 Heisenberg runs
- **Part 5: Experiment Verdicts (from compare --all)** (L127)
  - E4 → E4b Progression
- **Part 6: Comparative Analysis — Decision Matrix** (L148)
- **Part 7: Recommendations for Thesis** (L164)
  - Chapter 5 Integration
  - Key Claim (new, from E4b)
  - Supporting Data for Claim
- **Part 8: Code Changes Summary** (L189)
- **Part 9: Frustrated TFIM (J1-J2) — Execution Results** (L213)
  - Execution: 2026-06-03
  - Section 1: Expressibility vs J₂ (N=6, p=2, h=1.5, 3 seeds × 10 restarts)
  - Section 2: Warm-Start (descending h, J₂=0.3)
  - Section 3: Cross-Topology (J₂=0.3)
  - Section 6: Scaling (N=4,6,10 at J₂=0.3)
  - Hardware Viability Assessment
  - … (+1 más)
- **Part 10: Standard Experiment Execution Results (2026-06-03)** (L300)
  - E4b — TFIM + Longitudinal (Standard Schema)
  - E4c — Frustrated TFIM (Standard Schema)
  - Digest Integration Verified
  - Key Learnings
- **Part 11: Full Pipeline with MPNN (J₂ as Node Feature)** (L355)
  - Execution: 2026-06-03
  - Problem
  - Solution
  - Results: 8 points vs 15 points
  - Digest Verification
  - Key Takeaway
  - … (+2 más)

---

## 15. [binnacle-heisenberg-extension.md](binnacle-heisenberg-extension.md)
**Binnacle — Heisenberg Model Extension & Baseline Comparison**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-21 | 525 | 3,796 | 23.2 KB | 120 | 5 |

**Tags**: `hardware`, `mpnn`, `thesis`, `heisenberg`, `tfim`, `scaling`

> Validate that the GNN-HVA framework is model-agnostic by extending to the Heisenberg XXZ model, and quantify the value of MPNN warm-start via random baseline co

**Métricas clave**: ΔE/gap < 5% · fidelity = 0.05

**Hallazgos / Aseveraciones** (8):

- ALL anisotropy values produce zero fidelity on chain_1d. The failure is independent of Δ
- Perfectly seed-independent (std=0). The θ_smoothness varies because VQE finds different degenerate local minima at different h-points, but all have zero overlap with the ground sta…
- More restarts do NOT help. The landscape has a single accessible basin (E≈-3) that is far from the ground state (E≈-19). This is not a local minimum problem — it's a fundamental ex…
- Even at h=4.0 (deep paramagnetic limit where S≈0), fidelity remains zero. The problem is NOT entanglement — it's that the HVA circuit structure (XX+YY+ZZ+Z rotations with Néel init…
- Counterintuitively, more complex topologies (more edges) give slightly HIGHER fidelity. This suggests the additional connectivity provides more variational freedom, partially compe…
- The pipeline is correct. TFIM achieves 99.99% fidelity at the same h-values where Heisenberg achieves 0%. The failure is model-specific
- Fidelity is uniformly zero across all Δ values. The failure is not anisotropy-dependent
- The XY model on ladder with seed=44 achieves 31.4% fidelity at h=2.0. This is the ONLY configuration across all 30 runs that shows meaningful (>5%) fidelity. The seed-dependence (0…

**Contenido** (9 secciones):

- **2026-05-21 — Model-Agnostic Validation & Random Baseline** (L3)
  - Objective
- **Part 1: Random Baseline Comparison (Implemented)** (L11)
  - What was added
  - Integration test result (N=6, TFIM, h=1.5, fake θ)
  - CLI
  - Files modified
- **Part 2: Heisenberg XXZ Extension (Finding)** (L46)
  - Experiments executed
  - Finding
  - Physical explanation
  - Thesis implication
  - Files created
- **Validation** (L92)
- **2026-06-01 — Comprehensive Heisenberg XXZ Variant Experiments (30 runs)** (L100)
  - Objective
  - Method
  - Execution Summary
  - Key Numerical Results
  - Root Cause Analysis
  - Diagnosis Distribution (from `diagnose.py`)
  - … (+2 más)
- **2026-06-01 — N=10 Scaling Verification (3 key variants)** (L267)
  - Objective
  - Results
  - Key Observations
  - Conclusion
  - Tool Used
- **2026-06-01 — N=16 Scaling Verification (3 key variants)** (L308)
  - Objective
  - Execution
  - Results — Cross-N Scaling Table
  - Key Findings
  - TFIM N=16 Clarification
  - Conclusion
  - … (+1 más)
- **2026-06-01 — Sanity Checks (Circuit + VQE Verification)** (L373)
  - Test 2: Circuit Structure ✅
  - Test 3: VQE Optimization ✅
  - Bonus: Exact Energy Verification
  - Scientific Validity Conclusion
  - Tool
- **2026-06-01 — Test 4: Depth Scaling Validation (p=1→6)** (L446)
  - Objective
  - Results
  - Scientific Conclusions
  - Tool

---

## 16. [binnacle-mps-scaling.md](binnacle-mps-scaling.md)
**Binnacle — MPS Scaling to N>30**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-07 | 376 | 2,348 | 13.9 KB | 79 | 0 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> > First demonstration of the GNN-HVA pipeline operating beyond statevector

**Métricas clave**: ΔE/gap < 1% · ΔE/gap < 0.1% · ΔE/gap < 5% · ΔE/gap = 324%

**Hallazgos / Aseveraciones** (8):

- First demonstration of the GNN-HVA pipeline operating beyond statevector limits (N>22) using MPS-based VQE evaluation. Validates Phase 1 (DMRG) and Phase 2 (VQE) at N=40 and N=50 f…
- P1: TeNPy exact vs statevector cross-validation (diff<1e-10) ✅
- P2: Aer MPS within 10×precision statistical bound ✅
- P3: Seed reproducibility (identical results) ✅
- P4: VQEConfig rejects invalid method names ✅
- P5: COBYLA dispatch works without TypeError ✅
- P6: Default VQEConfig backward compatibility ✅
- P7: Dynamic chi_max formula correctness ✅

**Contenido** (16 secciones):

- **Motivation** (L12)
- **Implementation Summary** (L25)
  - New infrastructure (2026-06-07)
  - Tests (17/17 pass)
- **N=40 Results — PASS ✅ (2026-06-07)** (L53)
  - Phase 1: DMRG Ground Truth (15.4s total)
  - Phase 2: MPS-VQE (COBYLA, 3 restarts, maxiter=500)
  - Scaling Law Validation
  - Key Findings
- **N=50 Results — PASS ✅ (2026-06-07)** (L115)
  - Phase 1: DMRG Ground Truth (19.6s total)
  - Phase 2: MPS-VQE (COBYLA, 3 restarts, maxiter=500)
  - Scaling Law
- **Cross-N Summary (N=40 + N=50 + N=80)** (L150)
- **N=80 Results — PASS ✅ (2026-06-07)** (L167)
  - Phase 1: DMRG (69.4s total, χ_actual=9-11)
  - Phase 2: MPS-VQE (39.4s total — only 5-14s per h-point!)
- **Thesis Value** (L189)
- **Next Steps** (L199)
- **Cross-N Transfer Finding (SCALE-4, 2026-06-08)** (L212)
- **Multi-Seed Finding (2026-06-08)** (L230)
- **Files** (L247)
- **C2: Timing Scaling Law (2026-06-08)** (L258)
- **B2: Noisy Analytical Rehearsal (2026-06-08)** (L282)
- **Phase 3 Results Summary (2026-06-08)** (L301)
  - Run 1: Multi-seed training (27 points)
  - Run 2: Single-seed extrapolation (9 points, h-test at boundary)
- **Experiments Running (2026-06-08)** (L330)
- **A1: Zero-Shot Cross-N GNN — NEGATIVE RESULT ❌ (2026-06-08)** (L338)

---

## 17. [binnacle-noisy-variants.md](binnacle-noisy-variants.md)
**Binnacle — Noisy Simulation Variants (2026-05-25)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 145 | 1,010 | 6.1 KB | 35 | 0 |

**Tags**: `zne`, `hardware`, `thesis`, `scaling`, `topology`

> > Three novel experiments to address ZNE failure at N=10.

**Métricas clave**: R²=0.95 · R² = 0.956 · R²<0.05 · R²>0.94 · ΔE/gap = 1.4 · R²>0.5

**Hallazgos / Aseveraciones** (1):

- Three novel experiments to address ZNE failure at N=10. Final runs with FakeTorino (qiskit_ibm_runtime fixed) + explicit layout selection

**Contenido** (6 secciones):

- **Executive Summary** (L8)
- **Variant 1: p=1 Noisy Sweep at N=10 (CONFIRMED ✅)** (L19)
  - Hypothesis
  - Results (FakeTorino, explicit layout selection)
  - Key Findings
  - Comparison
  - Thesis Value: HIGH
- **Variant 2: Non-Linear Extrapolation (COMPLETE ✅)** (L59)
  - Hypothesis
  - Results (from archived N=10 data, no new simulations)
  - V2 Extended: Validation on N=6
  - Decision Rule
- **Variant 3: Per-Observable ZNE at N=10 (CONFIRMED ✅)** (L92)
  - Hypothesis
  - Results (FakeTorino, explicit layout selection)
  - Per-Site R² Map (h=1.5)
  - Key Findings
  - Thesis Value: HIGH
- **Infrastructure Resolution** (L126)
- **Validated Decisions** (L135)

---

## 18. [binnacle-p1-scaling.md](binnacle-p1-scaling.md)
**Binnacle — HVA p=1 Scaling Experiments**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-18 | 385 | 3,045 | 17.4 KB | 77 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `scaling`

> > Study of the depth-expressibility tradeoff: reducing HVA from p=2 to p=1

**Métricas clave**: ΔE/gap < 5% · ΔE/gap = 6.0% · ΔE/gap = 76%

**Hallazgos / Aseveraciones** (3):

- Study of the depth-expressibility tradeoff: reducing HVA from p=2 to p=1 to enable scaling to larger system sizes (N=20, N=30) on hardware. Core hypothesis: p=1 (2 parameters) prov…
- *VQE at N=20 p=1**: ✅ Works. Boundary confirmed at h ≥ 2.25 (2/3 seeds perfect, 1 fixable)
- p=1 at N=20 is the sweet spot for hardware deployment. It has the same CX budget as the already-validated p=2 N=10 configuration, while demonstrating scaling to a larger system

**Contenido** (9 secciones):

- **Motivation** (L12)
  - Circuit Depth Analysis
- **2026-05-21 — Sub-experiment 6A: p=1 vs p=2 Accuracy (N=6, N=10)** (L39)
  - Hypothesis
  - Configuration
  - Results: N=6
  - Results: N=10
  - Valid Regime Boundaries (all 3 seeds must pass)
  - Key Observations
- **2026-05-21 — Sub-experiment 6D: Boundary Detection + Circuit Metrics (COMPLETE)** (L92)
  - Configuration
  - Results
  - N=20 Boundary Failure Analysis
- **2026-05-21 — Sub-experiment 6B: p=1 Full Pipeline at N=20 (COMPLETE)** (L127)
  - Hypothesis
  - Configuration
  - Phase 2 Results: VQE at N=20, p=1 (all seeds)
  - Phase 3 Results: MPNN Training
  - Phase 4 Results: Deployment
  - Analysis
  - … (+1 más)
- **Consolidated Analysis** (L217)
  - Valid Regime Scaling Law
  - Hardware Viability Argument
  - MPNN Implications
  - Full Pipeline Status at N=20 p=1
  - Thesis Narrative
  - New Physics Insights
- **Technical Notes & Caveats** (L286)
  - DMRG Gap Issue at N≥15
  - Execution Times (2026-05-21)
  - Why p=1 Results Are Seed-Independent (REVISED)
  - Limitations
- **Decisions Made** (L352)
- **Next Steps** (L366)
- **Files** (L378)

---

## 19. [binnacle-s-series-results.md](binnacle-s-series-results.md)
**Binnacle — S-Series Experiment Results**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-01 | 235 | 1,667 | 10.2 KB | 59 | 0 |

**Tags**: `hardware`, `mpnn`, `tesis`, `pea`, `scaling`, `topology`

> > Fecha: 2026-06-01

**Métricas clave**: R²=0.999 · ΔE/gap < 5% · ΔE/gap = 2.48% · ΔE/gap=2.48%

**Hallazgos / Aseveraciones** (6):

- Fecha: 2026-06-01 Experimentos: S1, S2, S3, S4, S5, S6 Foco: Mejoras de escalabilidad sin acceso a hardware Tiempo total: ~2.8 horas (S3: 92 min, S5: 173 min, S4: 7 min, S6: 1.5 mi…
- Seed 45: 38.2% ❌ (MPNN diverge)
- Seed 46: 2.73% ✅
- Seed 47: 7.46% ❌ (marginal)
- Seed 48: 77.9% ❌ (MPNN diverge)
- Seed 49: 67.3% ❌ (MPNN diverge)

**Contenido** (10 secciones):

- **Resumen Ejecutivo** (L10)
- **S1: Entropía de Entrelazamiento vs Ley de Escalado ✅ (con caveat)** (L25)
- **S2: Cross-Topology Transfer ❌ (Negativo Útil)** (L60)
- **S3: Landscape Analysis N=20 ✅** (L83)
- **S4: Data Efficiency N=10 ✅ (corregido tras validación)** (L110)
- **S5: Pipeline Completo N=20 p=1 con MPNN ✅** (L145)
- **S6: MC-Dropout UQ ✅** (L176)
- **Decisiones Actualizadas** (L200)
- **Contribuciones a la Tesis (por sección)** (L214)
- **Archivos de Resultados** (L226)

---

## 20. [binnacle-s8-negative-result.md](binnacle-s8-negative-result.md)
**EXP-S8/S8b: Finite-Size Scaling of h_c via Weight Gradients — NEGATIVE RESULT**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-01 | 141 | 887 | 5.6 KB | 15 | 2 |

**Tags**: `mpnn`, `thesis`, `tfim`, `pea`, `scaling`, `topology`

> Date: 2026-06-01

**Hallazgos / Aseveraciones** (8):

- MLP: gradient norms ~80× normal (h_peak at boundary h=2.34)
- MPNN: MSE > 1e-2 (vs normal ~1e-4), gradient norms ~500-1200×
- VQE data quality degrades sharply below h≈1.0 (outside valid regime)
- The θ_opt values at h<1.0 are noisy/inconsistent across the h-sweep
- The MPNN's loss landscape has the steepest gradient where data is most irregular
- This boundary effect overwhelms any physical signal from the phase transition
- *Training data**: They use exact ground states (NQS), we use VQE-optimized θ
- *Metric**: They measure weight-space *distance* between models trained at

**Contenido** (8 secciones):

- **Hypothesis** (L9)
- **Variants Tested** (L18)
- **Configuration (both variants)** (L25)
- **Results** (L35)
  - S8 (MLP proxy)
  - S8b (MPNN GINConv)
  - Seed 44 Pathology
- **Root Cause Analysis** (L75)
  - Why S8 (MLP) fails
  - Why S8b (MPNN) fails
  - Why Hernandes et al. (2025) works but we don't
- **Conclusions** (L106)
- **Thesis Contribution** (L123)
- **Files** (L136)

---

## 21. [binnacle-scaling-improvements.md](binnacle-scaling-improvements.md)
**Binnacle — Mejoras de Escalabilidad (Análisis Pre-Ejecución)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-01 | 214 | 1,212 | 7.8 KB | 14 | 2 |

**Tags**: `hardware`, `mpnn`, `tesis`, `tfim`, `scaling`, `topology`

> > Fecha: 2026-06-01

**Métricas clave**: ΔE/gap=1.58% · ΔE/gap < 2% · ΔE/gap < 5% · ΔE/gap < 3% · ΔE/gap < 10%

**Hallazgos / Aseveraciones** (5):

- Fecha: 2026-06-01 Objetivo: Identificar mejoras ejecutables sin hardware real para fortalecer la narrativa de escalabilidad de la tesis. Estado: PLANIFICACIÓN (ningún experimento e…
- *N=20 p=1 con MPNN real** — solo se probó con interpolación lineal (C3)
- *N=16 nunca completó Phase 3** — fidelity filter rechaza datos
- *No hay explicación teórica** de la ley de escalado h_min(N)
- *Cross-topology transfer** nunca se probó (train-on-one, deploy-on-another)

**Contenido** (11 secciones):

- **Resumen Ejecutivo** (L10)
- **Mejora 1: Pipeline completo N=20 p=1 con MPNN** (L22)
- **Mejora 2: Fix N=16 p=1 (régimen correcto)** (L46)
- **Mejora 3: Entropía de entrelazamiento vs h_min (explicación teórica)** (L72)
- **Mejora 4: Cross-topology transfer learning** (L95)
- **Mejora 5: Landscape analysis a N=20 (F3 + B4 extendido)** (L119)
- **Mejora 6: Data efficiency a N=10 (extensión G1)** (L138)
- **Dependencias y Orden de Ejecución** (L156)
- **Criterios de Éxito** (L169)
- **Notas** (L182)
- **Archivos Creados** (L191)
  - Ejecución

---

## 22. [binnacle-session-20260521.md](binnacle-session-20260521.md)
**Binnacle — Session 2026-05-21**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 195 | 1,023 | 7.8 KB | 39 | 4 |

**Tags**: `hardware`, `mpnn`, `thesis`, `heisenberg`, `tfim`, `pea`

> Added automatic warm-start vs cold-start comparison to every Phase 4 deployment.

**Métricas clave**: gain=83% · ΔE/gap=2.6% · gain=98.9% · gain=82.4% · ΔE/gap=0.057 · ΔE/gap=8.98

**Contenido** (11 secciones):

- **Full Session Summary** (L3)
  - Duration: ~3 hours
  - Scope: Pipeline enhancement, model extension, comparative analysis, quantum utility proofs
- **1. Random Baseline Comparison (IMPLEMENTED)** (L10)
  - What
  - Files Modified
  - Design Decisions
  - Validation
- **2. Heisenberg Model Extension (IMPLEMENTED + FINDING)** (L37)
  - What
  - Files Created/Modified
  - Experiments Executed
  - Finding
  - Thesis Value
- **3. Comparative Analysis Suite (6 COMPARISONS + 2 ANALYSES)** (L64)
  - File Created
  - Results
  - Key Findings
- **4. Quantum Utility Proofs (3 PROOFS EXECUTED)** (L89)
  - File Created
  - Results
  - Proof 1 Detail: Classical Cost Explosion
  - Proof 2 Detail: Warm-Start Under Noise
  - Proof 3 Detail: Cross-Size Prediction
- **5. Documentation Updates** (L127)
- **6. Result Files Generated** (L142)
- **7. Test Suite Status** (L163)
- **8. Scripts Created This Session** (L169)
- **9. Thesis Arguments Strengthened** (L179)
- **10. Next Steps (Recommended)** (L190)

---

## 23. [binnacle-theta-pca-unsupervised-detection.md](binnacle-theta-pca-unsupervised-detection.md)
**Binnacle: Unsupervised Phase Detection from θ_opt(h) — Tasks 2 & 3**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-04 | 284 | 2,104 | 13.8 KB | 41 | 1 |

**Tags**: `mpnn`, `thesis`, `tfim`, `pea`, `topology`, `noise`

> > Date: 2026-06-04

**Hallazgos / Aseveraciones** (8):

- Date: 2026-06-04 Scripts: `scripts/analysis/extract_theta_trajectories.py`, `theta_pca_phase_detection.py`, `theta_derivative_analysis.py` Sanity check: `python -m project_health.a…
- We demonstrate that Principal Component Analysis of the HVA variational parameters θ_opt(h) provides unsupervised phase detection at zero additional quantum computational cost. For…
- The raw VQE parameter derivative |∂θ_opt/∂h| independently corroborates the D1 weight-gradient phase detection. For chain_1d (N=6, p=2), |∂θ/∂h| peaks at h=1.25 (Δh=0.25 from h_c),…
- The HVA p=2 parameter space for the 1D TFIM is effectively one-dimensional: PC1 explains 99.96% of total θ_opt variance across the entire h-sweep. This implies that the four HVA pa…
- The VQE parameter derivative |∂θ_opt/∂h| independently corroborates the D1 weight-gradient phase detection. For chain_1d (N=6, p=2), |∂θ/∂h| peaks at h=1.25 (Δh=0.25 from h_c=1.0),…
- PCA detection: 1/2 topologies pass (chain_1d: ✅, ladder: ✗)
- Normalize θ_opt(h) to unit variance (StandardScaler)
- K-means (k=2) → cluster boundary location

**Contenido** (14 secciones):

- **Executive Summary** (L10)
- **Thesis Statements** (L35)
  - Statement 1 — Unsupervised Phase Detection (§5.1, Novel Contribution)
  - Statement 2 — Independent Corroboration of D1 (§5.1, Supporting Evidence)
  - Statement 3 — Effective Dimensionality of HVA (§4.2, Ansatz Analysis)
- **1. Hypothesis (Task 2)** (L71)
- **2. Hypothesis (Task 3)** (L78)
- **3. Data Extraction (Task 2.1)** (L85)
- **4. PCA Results (Task 2.2)** (L105)
  - Method
  - Per-Topology Results
  - Key Finding
  - Formal Outcome
- **5. Derivative vs D1 Comparison (Task 3)** (L157)
  - Method
  - Results
  - Note on Pearson Correlation
  - Thesis Paragraph (auto-generated)
- **6. Connection to D1 Binnacle** (L201)
- **7. Outputs** (L218)
- **8. Reproducibility** (L231)
- **9. Thesis Contribution** (L245)
- **10. Lessons Learned** (L255)
- **11. What This Does NOT Prove** (L270)
- **12. Potential Extensions (not executed — for future work if needed)** (L279)

---

## 24. [binnacle-v8-experiments-initial.md](binnacle-v8-experiments-initial.md)
**Binnacle — V8 Noiseless Simulation Experiments**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-22 | 404 | 2,645 | 16.6 KB | 59 | 3 |

**Tags**: `hardware`, `mpnn`, `thesis`, `tfim`, `scaling`, `noise`

> > New experiments beyond V7, focused on landscape analysis, scaling laws,

**Métricas clave**: ΔE/gap=10 · ΔE/gap<2% · ΔE/gap < 0.05 · R² = 0.9998 · R² = 0.9848 · R²=0.9998

**Hallazgos / Aseveraciones** (1):

- New experiments beyond V7, focused on landscape analysis, scaling laws, and methodological validation. All noiseless, all local. Date: 2026-05-22 Framework: `scripts/experiments_v8…

**Contenido** (10 secciones):

- **Session 2026-05-22 — First V8 Execution (4 experiments)** (L12)
  - Context
- **Experiment F3: Landscape Fluctuation Analysis** (L31)
  - Hypothesis
  - Configuration
  - Results
  - Conclusions
  - Thesis Value
  - Learning
- **Experiment B1: Analytical Initial Guess Validation** (L94)
  - Hypothesis
  - Configuration
  - Results
  - Conclusions
  - Thesis Value
  - Learning
- **Experiment A3: Finite-Size Scaling Law** (L156)
  - Hypothesis
  - Configuration
  - Results
  - Scaling Law Fits
  - Predictions
  - p=1 vs p=2 Comparison
  - … (+3 más)
- **Experiment B4-lite: Hessian Analysis at VQE Minima** (L262)
  - Hypothesis
  - Configuration
  - Results
  - Conclusions
  - Thesis Value
  - Learning
- **Cross-Experiment Synthesis** (L325)
  - The Complete Picture of HVA p=2 Limitations
  - Implications for Hardware Deployment
- **Files Generated** (L350)
- **Decisions Made** (L365)
- **Next Steps (from this session)** (L378)
- **Environment & Reproducibility** (L388)

---

## 25. [binnacle-v8-experiments-round1.md](binnacle-v8-experiments-round1.md)
**Binnacle: V8 Experiments — Round 1 Results**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-22 | 335 | 2,197 | 13.5 KB | 28 | 0 |

**Tags**: `mpnn`, `thesis`, `tfim`, `pea`, `scaling`

> > Date: 2026-05-22

**Métricas clave**: ΔE/gap = 0.0158 · ΔE/gap < 2% · ΔE/gap=1.58% · ΔE/gap=0.437

**Hallazgos / Aseveraciones** (1):

- Date: 2026-05-22 Experiments: F1, B4, D1, B2, E4 All at N=6, p=2, chain_1d, 3 seeds (42, 43, 44)

**Contenido** (13 secciones):

- **Summary Table** (L9)
- **Experiment F1: DyPP Extrapolation** (L21)
  - Result: Hypothesis NOT confirmed
- **Experiment B4: Hessian-Guided Restarts** (L49)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment D1: Weight-Space Phase Detection** (L80)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment B2: TITAN Parameter Freezing** (L119)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment E4: TFIM + Longitudinal Field** (L147)
  - Result: Hypothesis PARTIALLY REJECTED
- **Cross-Experiment Insights** (L187)
  - 1. HVA p=2 Landscape is Benign (B4 + B2 combined)
  - 2. MPNN Weight Space Encodes Phase Information (D1)
  - 3. DyPP is Not Worth It (F1)
  - 4. HVA is Model-Specific (E4)
- **Updated Project Status Implications** (L210)
- **Next Steps** (L222)
- **Experiment C3: Sign Canonicalization (Run 2 — 3 restarts, 100 maxiter)** (L231)
  - Result: Hypothesis SUPERSEDED — problem doesn't exist ✅
- **E3 and C1: Not Yet Implemented** (L264)
- **Experiment C3: Run 3 — Reproducibility Check (3 restarts, 100 maxiter)** (L280)
  - Result: Partial — 2/3 seeds pass (vs 3/3 in run 2)
- **Experiment C1: Physics-Informed MPNN Loss** (L301)
  - Result: Modest improvement (3.9%), hypothesis NOT confirmed at 10-30% level

---

## 26. [binnacle-v8-experiments-round2.md](binnacle-v8-experiments-round2.md)
**Binnacle: V8 Experiments — Round 2 Results**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-22 | 242 | 1,521 | 9.1 KB | 56 | 1 |

**Tags**: `mpnn`, `thesis`, `tfim`, `pea`, `scaling`

> > Date: 2026-05-22 (afternoon session)

**Métricas clave**: ΔE/gap<3.4%

**Hallazgos / Aseveraciones** (1):

- Date: 2026-05-22 (afternoon session) Experiments: B4@N=10, F3@p=1, D1-regularized, C1@N=10 Focus: Scaling validation and robustness improvements

**Contenido** (8 secciones):

- **Summary Table** (L9)
- **Experiment B4 at N=10: Hessian Landscape Verification** (L20)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment F3 at p=1: Landscape Comparison** (L50)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment D1-Regularized: Robust Phase Detection** (L86)
  - Result: Hypothesis CONFIRMED ✅
- **Experiment C1 at N=10: Physics Loss Scaling** (L136)
  - Result: Hypothesis NOT CONFIRMED ❌
- **Cross-Experiment Synthesis (Round 2)** (L181)
  - Landscape Properties are Universal (B4@N=10 + F3@p=1)
  - Regularization is Key for D1 (D1-reg)
  - Physics Loss: Context-Dependent (C1@N=10)
- **Updated Validated Decisions** (L218)
- **Files Generated** (L232)

---

## 27. [binnacle-v8-round2-pipeline-characterization.md](binnacle-v8-round2-pipeline-characterization.md)
**Binnacle: V8 Round 2 — Pipeline Characterization (G-Series)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-25 | 222 | 1,555 | 8.8 KB | 39 | 0 |

**Tags**: `mpnn`, `thesis`, `pea`, `scaling`, `noise`

> > Date: 2026-05-25

**Métricas clave**: ΔE/gap < 2.1% · ΔE/gap < 5% · ΔE/gap = 1.26

**Hallazgos / Aseveraciones** (1):

- Date: 2026-05-25 Experiments: G1, G2, G3, G4, G5 Focus: Data efficiency, uncertainty, scaling, adaptive VQE, robustness All at N=6 p=2 (except G3: N=20 p=2)

**Contenido** (9 secciones):

- **Summary Table** (L10)
- **G5: Cross-Seed Generalization ✅** (L22)
- **G4: Condition Number vs Restarts ❌** (L44)
- **G1: Data Efficiency Curve ✅** (L78)
- **G2: Ensemble Uncertainty Calibration ❌** (L113)
- **G3: N=20 p=2 Optimized Pipeline ❌** (L139)
- **Cross-Experiment Insights** (L178)
  - 1. The pipeline is remarkably robust at N=6 (G5)
  - 2. Data efficiency is much better than expected (G1)
  - 3. N=6 findings don't transfer to N=20 (G3 vs B4/B2)
  - 4. Simple uncertainty quantification doesn't work (G2)
  - 5. Condition number is not useful for adaptive VQE (G4)
- **Updated Optimal Configurations** (L201)
- **Thesis Contributions from Round 2** (L214)

---

## 28. [binnacle-zne-robustness.md](binnacle-zne-robustness.md)
**Binnacle — ZNE Robustness Validation at N=10 (2026-05-25)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 276 | 1,686 | 10.4 KB | 48 | 3 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `pea`, `topology`

> > Rigorous statistical validation of inhomogeneous ZNE with corrected pipeline.

**Métricas clave**: R² > 0.8 · R²=0.93 · R² > 0.95 · gain = +19% · R² < 0.85 · gain = -63%

**Hallazgos / Aseveraciones** (8):

- Rigorous statistical validation of inhomogeneous ZNE with corrected pipeline. 72 evaluations: 3 pipeline seeds × 3 layout seeds × 2 n_layouts × 4 h-values. Fixes applied: seed_simu…
- Generated via `scripts/compare.py --noisy` using the unified `ResultStore` framework. The `analyze_robustness.py` script has been absorbed into the framework (`ResultStore.analyze_…
- Pipeline seed has minimal effect. R² is stable across seeds
- Layout seed is THE dominant factor. Some layout sets
- h-value has minimal effect on R² (all ~0.93)
- The best achievable layout has CES ≈ 0.3 (not zero)
- The worst layout has CES ≈ 32-48 (deep non-perturbative regime)
- The linear fit is dominated by the high-CES outlier

**Contenido** (9 secciones):

- **Executive Summary** (L9)
- **Root Cause Analysis: Why ZNE Overshoots** (L23)
  - The Problem
  - Evidence
  - Why n=3 > n=5
  - Why layout_seed=42 always fails
- **Statistical Breakdown** (L73)
  - By Pipeline Seed (MPNN/VQE randomness)
  - By Layout Seed (qubit selection randomness)
  - By h-value
  - Correlation Analysis
- **Corrected Pipeline (vs earlier runs)** (L121)
- **Recommendations for Thesis** (L132)
  - 1. Report ZNE as "conditionally effective"
  - 2. Use "best-of-layouts" as the baseline, not "first layout"
  - 3. Consider CES-capped extrapolation
  - 4. The real value of inhomogeneous ZNE
  - 5. For hardware deployment
- **Validated Decisions (Updated)** (L172)
- **Comparison with Original Results (2026-05-14)** (L186)
- **Files** (L202)
- **Cross-Experiment Analysis (2026-05-26)** (L211)
  - New Finding: ZNE Overshoot Mechanism Clarified
  - Actionable Recommendation
  - Cross-Experiment Context
  - Updated Decision Table
  - Tool Used

---
