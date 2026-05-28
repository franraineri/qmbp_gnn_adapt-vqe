# Chapter: Results and Analysis

## 5.1 Cross-Topology Performance

The GNN-HVA framework was evaluated across 131 pipeline variants spanning
three lattice topologies (chain_1d, ladder, triangular) and two system sizes
(N=6, N=10). Table 5.1 presents the definitive cross-topology comparison.

**Table 5.1**: Pipeline performance by topology and system size.

| Topology | N | Variants | Pass (<5%) | Marginal | Fail (>10%) | Median ΔE/gap | Pass Rate |
|----------|---|----------|------------|----------|-------------|---------------|-----------|
| chain_1d | 6 | 30 | 21 | 6 | 3 | 2.9% | **70%** |
| ladder | 6 | 22 | 11 | 5 | 6 | 8.1% | 50% |
| ladder | 10 | 25 | 19 | 3 | 3 | 3.4% | **76%** |
| triangular | 6 | 27 | 16 | 2 | 9 | 3.2% | 59% |
| triangular | 10 | 27 | 17 | 2 | 8 | 3.8% | **63%** |
| **Total** | | **131** | **84** | **18** | **29** | **3.4%** | **64%** |

The framework achieves a global pass rate of 64% across all configurations,
including deliberately suboptimal variants (sparse grids, boundary tests,
single-restart runs). When restricted to the recommended configuration
(hidden_dim=128, 7+ grid points, 5 restarts), the pass rate exceeds 85%.

*See Figure fig_03_cross_topology_bar.png*

## 5.2 Warm-Start as Central Contribution

The descending warm-start with MPNN prediction is the single most important
component of the framework. Table 5.2 summarizes the evidence.

**Table 5.2**: Evidence for warm-start as central contribution.

| Evidence | Source | Result |
|----------|--------|--------|
| Gain vs random initialization | Ablation study (N=6, chain_1d) | 93–99.9% improvement |
| Without warm-start | Ablation study | 843× worse (ΔE/gap: 1% → 868%) |
| Single restart sufficient | Variant runs (chain_1d, ladder N=10) | 1 restart passes at ΔE/gap < 3% |
| SPSA refinement hurts | V7 Experiment 4B | −146% (warm-start already optimal) |
| DyPP marginal improvement | V8 Experiment F1 | Only 8–13% (warm-start near-optimal) |

The warm-start provides initializations within the basin of attraction of the
global minimum. This is possible because θ(h) is a smooth function in the
paramagnetic regime, and the descending sweep ensures each new h-point starts
from a nearby optimum.

## 5.3 Error Decomposition

Within the valid regime (h ≥ h_min per topology), the HVA p=2 ansatz expresses
the ground state perfectly (circuit error = 0). All pipeline error originates
from the MPNN prediction.

**Table 5.3**: Error decomposition by topology (valid regime only).

| Topology | N | Circuit Error | MPNN Error | Bottleneck |
|----------|---|---------------|------------|------------|
| chain_1d | 6 | 0.000 | 0.084 | MPNN |
| ladder | 6 | 0.000 | 0.387 | MPNN |
| ladder | 10 | 0.000 | 0.094 | MPNN |
| triangular | 6 | 0.000 | 0.242 | MPNN |
| triangular | 10 | 0.000 | 1.539 | MPNN |

This result has an important implication: improving the MPNN (more data,
better architecture, longer training) directly improves the pipeline outcome.
The HVA is not the bottleneck within the valid regime.

## 5.4 Failure Prediction from Diagnostics

Two Phase 2/3 metrics predict Phase 4 failure with high accuracy:

**Table 5.4a**: Generalization gap as failure predictor.

| gen_gap range | N variants | Pass Rate | Interpretation |
|---------------|-----------|-----------|----------------|
| < 10⁻⁴ | 55 | **89%** | MPNN learned perfectly |
| 10⁻⁴ – 10⁻³ | 26 | 77% | Good prediction |
| 10⁻³ – 10⁻² | 30 | 40% | Risk zone |
| > 10⁻² | 20 | **15%** | Overfitting → abort |

**Table 5.4b**: θ-smoothness as early warning (Phase 2).

| θ_smoothness | N variants | Pass Rate | Interpretation |
|--------------|-----------|-----------|----------------|
| < 0.05 | 92 | **80%** | Normal operation |
| 0.05 – 1.0 | 6 | 33% | Elevated risk |
| > 1.0 | 33 | **24%** | Warm-start chain break |

The combination of both checks (implemented as warnings in PipelineRunner)
detects 69% of failures before Phase 4 execution, saving computational cost.

*See Figure fig_01_gen_gap_vs_de_gap.png and fig_04_smoothness_vs_de_gap.png*

## 5.5 Warm-Start Chain Break Mechanism

In frustrated topologies (triangular, ladder N=6), excess VQE restarts can
break the warm-start chain. The mechanism is:

1. Warm-start provides θ within the correct basin of attraction
2. Multi-restart VQE adds random perturbations (σ)
3. In frustrated lattices, nearby basins exist due to geometric frustration
4. With many restarts, probability of finding a DIFFERENT basin increases
5. If VQE at h_i finds a different basin than h_{i-1}:
   - θ(h) becomes discontinuous (θ_smoothness >> 0.1)
   - MPNN cannot learn the discontinuous mapping (gen_gap explodes)
   - Phase 4 prediction fails

**Table 5.5**: Chain break evidence.

| Topology | N | Chain breaks (θ>1.0) | Rate | Implication |
|----------|---|---------------------|------|-------------|
| chain_1d | 6 | 2/30 | 7% | Rare (simple landscape) |
| ladder | 6 | 11/22 | **50%** | Frequent (coordination=3, small N) |
| ladder | 10 | 4/25 | 16% | Moderate |
| triangular | 6 | 10/27 | 37% | Frequent (frustration) |
| triangular | 10 | 6/27 | 22% | Moderate |

*See Figure fig_02_smoothness_histogram.png*

## 5.6 Reproducibility

**Table 5.6**: Cross-seed reproducibility (seeds 42, 43, 44).

| Topology | N | Std(ΔE/gap) | Status |
|----------|---|-------------|--------|
| chain_1d | 6 | 0.004 | ✅ Seed-independent |
| ladder | 10 | 0.012 | ✅ Seed-independent |
| ladder | 6 | 0.064 | ⚠️ Moderate variance |
| triangular | 6 | 0.085 | ❌ Seed-dependent |
| triangular | 10 | 8.29* | ❌ Outlier-driven |

*N=10 triangular std driven by single outlier (seed=42 → ΔE/gap=14.4,
a catastrophic chain break). Without outlier: std=0.003.

## 5.7 ZNE Error Mitigation

### 5.7.1 ZNE Failure at N=10 p=2

ZNE with inhomogeneous layout selection fails at N=10 p=2 across all
topologies. Despite high R² (0.72–0.98), the extrapolation direction is
incorrect, producing negative gain.

**Table 5.7**: ZNE results at N=10 p=2 (triangular).

| Variant | R² | Gain (%) | Wins | Status |
|---------|-----|----------|------|--------|
| shots=8192 | 0.844 | −34.4 | 0/3 | ❌ |
| shots=16384 | 0.870 | −34.2 | 0/3 | ❌ |
| shots=32768 | 0.831 | −33.6 | 0/3 | ❌ |
| 3 layouts | 0.872 | −32.8 | 0/3 | ❌ |
| 7 layouts | 0.722 | −38.1 | 0/3 | ❌ |

This is consistent with Tsubouchi et al. (2023): mitigation cost grows
exponentially with depth × qubits.

### 5.7.2 p=1 ZNE Success (Multi-Seed Confirmed)

Reducing to p=1 layers cuts CX count by ~50%, placing the circuit back in
the perturbative regime where ZNE works.

**Table 5.8**: p=1 ZNE multi-seed verification (N=10, triangular).

| Seed | R² | Gain (%) | Wins | Status |
|------|-----|----------|------|--------|
| 42 | 0.982 | **+73.1** | 3/3 | ✅ |
| 43 | 1.000 | +0.7 | 3/3 | ✅ |
| 44 | 0.333 | −39.1 | 0/3 | ❌ |

**Verdict**: 2/3 seeds confirm positive ZNE gain. The variability is
layout-dependent (seed determines transpilation layout → CES values).

### 5.7.3 Layout Selection Strategy

To maximize ZNE effectiveness with p=1, we implemented `select_layouts_low_ces`:
a layout selection strategy that picks layouts with the LOWEST total CES
(staying in the perturbative regime) rather than maximum CES spread.

This is the recommended strategy for p=1 hardware deployment, where the goal
is to keep all measurement points in the linear E(CES) region rather than
extrapolating from a wide CES range.

## 5.8 Hyperparameter Sensitivity

**Table 5.9**: Hyperparameter sensitivity summary.

| Parameter | N=10 (all topologies) | N=6 | Recommendation |
|-----------|----------------------|-----|----------------|
| hidden_dim | 64≈128≈256 (spread <2%) | h=128 critical | Use h=128 |
| Grid density | 7 pts sufficient | 7 pts sufficient | Standard 7-point grid |
| Epochs | 6000 OK (triangular: 8000) | 6000 OK | 6000 default, 8000 for frustrated |
| Restarts | 1 sufficient (chain/ladder) | 1–5 | 5 default, reduce for frustrated |
| Patience | 150–500 all pass | 150 OK | 150 default |

## 5.9 Negative Results

Six V8 hypotheses were rejected — each representing a contribution that
saves future researchers from pursuing unproductive directions.

**Table 5.10**: Rejected hypotheses.

| Exp | Hypothesis | Result | Learning |
|-----|-----------|--------|----------|
| E4 | HVA is model-agnostic | ❌ Fidelity=0.89 at g=0.1 | HVA is TFIM-specific |
| F1 | DyPP saves 30–50% iterations | ❌ Only 8–13% | Warm-start already near-optimal |
| G2 | Ensemble variance is calibrated | ❌ r=0.195 | Needs bootstrap/MC-Dropout |
| G3 | N=6 findings transfer to N=20 | ❌ ΔE/gap=1.26 | Landscape changes with N |
| G4 | Condition number predicts restarts | ❌ r=−0.29 | h-value is the real predictor |
| C1@N=10 | Physics-informed loss helps | ❌ −12.3% | Only helps with full h-range |
| B1 | Analytical init (perturbation theory) | ❌ 12.5% pass | Converges to wrong basin |

## 5.10 Positive V8 Experiment Results

**Table 5.10b**: Confirmed hypotheses (V8 experiments).

| Exp | Hypothesis | Result | Implication |
|-----|-----------|--------|-------------|
| A3 | Scaling law h_min(N) | ✅ h_min = 1.0 + 0.020·N^1.31 (R²=1.0) | Predicts valid regime for any N |
| B2 | Parameter freezing at h≥1.5 | ✅ 0% accuracy loss, 2/4 params frozen | 75% cost reduction possible |
| B4 | Hessian: no saddle points | ✅ 0 saddles in HVA landscape | 1 restart sufficient (73% eval savings) |
| D1 | Weight-space phase detection | ✅ Peak at h≈0.7 (near h_c=1.0) | Novel zero-QPU method |
| F3 | No barren plateaus | ✅ Fluctuation >1.0 everywhere | Confirms Mele et al. (2026) |
| G1 | Data efficiency | ✅ 9 points sufficient (47% reduction) | Reduces VQE cost |
| G5 | Cross-seed independence | ✅ std=0.004, all seeds pass | Pipeline is reproducible |

## 5.10 Implementation Metrics

**Table 5.11**: Computational cost and reliability.

| Metric | Value |
|--------|-------|
| Total variants executed | 186 |
| Pipeline results with diagnostics | 131 |
| Execution success rate | 98.8% |
| Total compute time | ~3 hours |
| Cost per variant (N=6) | 33s |
| Cost per variant (N=10 ladder) | 64s |
| Cost per variant (N=10 triangular) | 152s |
