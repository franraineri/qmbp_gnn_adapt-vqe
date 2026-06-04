# Validated Decisions — GNN-HVA Framework

**Purpose**: Reference document for all experimentally validated decisions.
Do NOT duplicate this content in other files — reference this document instead.

---

## V7 Validated Decisions (2026-05-18)

| Decision | Evidence | Experiment |
|----------|----------|------------|
| L-BFGS-B optimal (noiseless) | Wins by 31-95% over Nevergrad | 1A |
| SPSA optimal (hardware): a=0.1, c=0.05, A=10 | Grid search 36 configs × 10 seeds | 4A |
| SPSA refinement HURTS warm-start | -146% at h=2.0 | 4B |
| SPSA 3× better than COBYLA under noise | Direct comparison | 4C |
| MPNN = QRC at N=10 (<1% difference) | Predictor NOT the bottleneck | 2B |
| MPS exact for 1D HVA (chi=64 sufficient) | |MPS-SV|=1e-14 | 3A/3B |
| MPS VQE at N=20 passes at h=2.0 | ΔE/gap≈1% | 3C |
| Noise-aware training fails under shot noise | 6× worse | 5B |
| Iterative refinement modest (9% gain) | Saturates in 2 rounds | 5E |

**Binnacle**: `documentation/binnacles/binnacle-v7-experiments.md`

---

## V8 Validated Decisions (2026-05-22)

### VQE & Optimization

| Decision | Evidence | Experiment |
|----------|----------|------------|
| p=1: 1 restart sufficient at all N | Single basin, simpler landscape | B4, F3@p=1 |
| p=2: 5 restarts conservative sweet spot | No saddle points but restart paradox at 3 | B4 |
| N=20 p=2: use 7 restarts, no freeze | 1 restart + freeze FAILS (ΔE/gap=1.26) | G3 |
| Landscape N-independent (condition numbers) | N=10 matches N=6 within 10% | B4@N=10 |
| Parameter freezing: θ_zz2, θ_x2 at h≥1.5 | 0% accuracy loss, 75% cost reduction | B2 |
| Sign canonicalization NOT needed | Descending warm-start breaks Z₂ naturally | C3 |
| DyPP rejected | Only 8-13% savings (hypothesis was 30-50%) | F1 |
| Analytical init converges to wrong basin | 97% fewer iterations but worse ΔE/gap | B1 |

### Predictor & Training

| Decision | Evidence | Experiment |
|----------|----------|------------|
| 9 points sufficient at N=6 (47% reduction) | Seeds 43/44 pass with 5 points | G1 |
| k=7-9 for N=10 cross-seed robustness | k=5 works for 50% of seeds only | S4 |
| Pipeline is seed-independent | std=0.004, all seeds pass | G5 |
| κ does NOT predict restart needs | r=-0.29. Use h-value as proxy | G4 |
| Naive ensemble UQ not calibrated | r=0.195 | G2 |
| MC-Dropout UQ calibrated (r=0.82) | 4.2× improvement over G2 | S6 |
| Physics-informed loss: +3.9% at N=6 | Safe, no regression. Modest. | C1 |
| Physics-informed loss: HURTS at N=10 | -12.3%. Only helps with full h-range | C1@N=10 |

### Phase Detection

| Decision | Evidence | Experiment |
|----------|----------|------------|
| Weight gradient peaks detect h_c | MPNN-A peak at h≈0.7 (near h_c=1.0) | D1 |
| Dropout=0.1 makes detection robust | std=0.13 vs 0.90. 5 seeds consistent | D1-reg |
| Overfitting shifts peak (needs regularization) | loss=0 → peak at h≈0.69 | D1-dense |

### Landscape

| Decision | Evidence | Experiment |
|----------|----------|------------|
| No barren plateaus | Fluctuation >1.0 everywhere | F3 |
| p=1 landscape simpler | Fluctuation 1.38 vs 1.99 | F3@p=1 |
| fraction_near_gs: training-free boundary predictor | 0% at h<1.0, 5%+ at h>1.5 | F3 |
| N=20 has 2-3 local minima | κ=73 (lower than N=6's 1399) | S3 |

### Generalization & Scaling

| Decision | Evidence | Experiment |
|----------|----------|------------|
| HVA p=2 is TFIM-specific | Fails at g>0 (fidelity→0.89 at g=0.1) | E4 |
| N=6 findings do NOT transfer to N=20 | Landscape is N-dependent (multiple basins) | G3, S3 |
| No zero-shot cross-topology transfer | ΔE/gap 3-10× when transferring | S2 |
| N=20 p=1 pipeline works (2.48%) | 9/9 pass. Interpolation beats MPNN for p=1 | S5 |

**Binnacles**: `binnacle-v8-experiments-*.md`, `binnacle-v8-round2-*.md`, `binnacle-s-series-results.md`

---

## V9 Validated Decisions (2026-06-01) — Heisenberg XXZ

| Decision | Evidence | Experiment |
|----------|----------|------------|
| HVA p≤2 CANNOT express Heisenberg ground states | Max fidelity ≈ 0% across ALL Δ, topologies, seeds | 30 runs |
| Failure is symmetry-sector trapping | VQE converges to E≈-3 vs E_exact≈-19 | Sanity check |
| Failure scales linearly with N | E_gap ≈ 3.8×N (gets worse, not better) | N=6/10/16 |
| Néel initial state insufficient | Cannot reach paramagnetic GS even at h=4.0 | Entanglement analysis |
| Entanglement explains expressibility | S>0 at all h (Heisenberg) vs S≈0 (TFIM) | S(h) computation |
| Model-agnostic pipeline architecture is sound | Same code, different ModelSpec → correct dispatch | Framework test |
| No noisy experiments needed for Heisenberg | CX=30 far exceeds ZNE threshold (18) | CX count |
| Depth scaling: p=5 reaches 47.7% fidelity | Saturates below 50% even at p=6 | Depth sweep |
| XY model: zero fidelity at all depths up to p=6 | More fundamental incompatibility than Heisenberg | Depth sweep |

**Binnacle**: `documentation/binnacles/binnacle-heisenberg-extension.md`
**Results**: `results/thesis/variants_N{6,10,16}_heisenberg/`

---

## p=1 Scaling Validated Decisions (2026-05-21 → 2026-06-01)

| Decision | Evidence |
|----------|----------|
| p=1 valid regime (corrected): chain N=10 h≥1.9, ladder N=10 h≥3.0 (safe: 3.25), triangular N=10 h≥3.5 | R2 verification (9/9 pass at safe boundaries) |
| p=1 ZNE works at N=10 (gain=+49%) | 9 runs, 3 topologies × 3 seeds |
| Heavy-hex p=1 ZNE: +62.7% gain (R²=0.998) | 3 runs, all seeds positive |
| p=1 more consistent than p=2 | std=0.002 vs 0.47 (COMP-4) |
| CX reduction: exactly 50% at all N | p=1 N=20 = 38 CX ≈ p=2 N=10 = 36 CX |
| θ_x constant (±3π/8) for all h | Only θ_zz varies → effectively 1D mapping |
| Seed 43 problematic for ladder | Consistent chain breaks at N=6, 10, 16 |
| Seed 44 problematic for triangular | Chain breaks at N=6 |
| N=16 p=1: Phase 3 does not complete | Fidelity filter rejects data (valid regime too narrow) |
| 16k shots sufficient (32k identical) | Noise is layout-dominated, not shot-dominated |
| 3 layouts sufficient (5 gives +3% marginal) | Not worth 67% more QPU time |
| 1 restart sufficient for p=1 | ΔE/gap=0.006 (heavy-hex) |
| p=2 unrescuable with more layouts | 5 layouts still fails (gain=-27%) |

**Binnacle**: `documentation/binnacles/binnacle-p1-scaling.md`

---

## Negative Results (valid scientific contributions)

| Finding | Implication | Experiment |
|---------|-------------|------------|
| HVA is TFIM-specific | Does not generalize to other spin models | E4, V9 |
| DyPP is redundant | Warm-start already near-optimal for 4 params | F1 |
| Naive ensemble UQ not calibrated | Need MC-Dropout or bootstrap | G2 |
| N=6 findings don't scale to N=20 | Landscape is N-dependent | G3 |
| κ doesn't predict difficulty | h-value is better proxy | G4 |
| No cross-topology transfer | Each topology needs own training data | S2 |
| Condition number (G4) | r=-0.29, not predictive | G4 |

---

## Tier 1 Validated Decisions (2026-06-03)

| Decision | Evidence | Experiment |
|----------|----------|------------|
| MPNN 2D predictor works within training J₂ range | ΔE/gap < 5% for seen J₂, 0% at unseen J₂ (5-value grid) | T1a |
| J₂ interpolation fails with 5 training values | Pass rate = 0% at J₂=0.15, 0.35, 0.45 — grid too sparse | T1a |
| ZNE gain equivalent for TFIM+longitudinal vs standard | |gain_diff| < 5% (0 extra CX from g·Z term) | T1b |
| TFIM+longitudinal hardware-viable at g≤0.1 | fid≥0.93 maintained, p=1 sufficient | T1b |
| D1 weight gradient generalizes to frustrated TFIM | Gradient peaks track crossover for ALL J₂ tested | T1c |
| D1 detection is model-agnostic (within TFIM family) | 100% agreement between standard TFIM and frustrated TFIM D1 peaks | T1c |

**Session results**: `documentation/analysis/12_tier1_session_results.md`

---

## ZNE Strategy Validated Decisions (2026-06-04)

| Decision | Evidence | Experiment |
|----------|----------|------------|
| CES-ZNE fails on heavy_hex | R²=0.04, gain=0% (uniform CES≈0.15, no spread) | HW_REHEARSAL |
| Gate-folding ZNE validated | +12% gain, R²>0.99, wins 9/12 vs CES | GF_ZNE_CMP |
| PEA-ZNE universally superior | +94.4% gain, 18/18 wins vs GF (t=46.32, p<10⁻¹⁹) | ZNE_CROSS_TOPO |
| PEA-ZNE cross-topology robust | Validated on chain_1d, heavy_hex, ladder (3 seeds each) | ZNE_CROSS_TOPO |
| GF-ZNE fails on heavy_hex (shallow circuits) | R²=0.47 — extrapolation meaningless at depth≤3 | PEA_HW_READY |
| PEA works with imperfect MPNN θ | +81% gain even with prediction error | PEA_PIPELINE |
| Amplifier strategy: PEA primary, GF fallback | PEA 4.6× better, ~50% extra QPU overhead justified | All ZNE exps |

**Binnacle**: `documentation/binnacles/binnacle-gate-folding-zne.md`

---

## Unsupervised Phase Detection Validated Decisions (2026-06-04)

| Decision | Evidence | Experiment |
|----------|----------|------------|
| PCA of θ_opt(h) detects h_c for chain_1d | Peak at h=1.25 (Δh=0.25 from h_c=1.0) | Task 2 |
| PC1 explains >99% variance (p=2) | θ-space is effectively 1D for HVA p=2 | Task 2 |
| K-means (k=2) does NOT reliably detect h_c | Boundary at h≈1.58 (too high) | Task 2 |
| |∂θ/∂h| corroborates D1 peak | Agreement Δh=0.18 with D1 valid-regime peak (h=1.07) | Task 3 |
| Detection requires h-grid covering h_c | Ladder (h∈[2,4]) fails — data limitation, not method | Task 2 |
| Method is zero-cost (uses existing VQE data) | No additional QPU or VQE runs needed | Tasks 2+3 |

**Binnacle**: `documentation/binnacles/binnacle-theta-pca-unsupervised-detection.md`
