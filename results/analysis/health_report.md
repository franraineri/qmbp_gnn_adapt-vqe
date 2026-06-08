# Project Health Report

**Generated:** 2026-06-06T00:03:13+00:00
**Results dir:** `results`

## Scan Overview

| Metric | Count |
|--------|------:|
| Noiseless pipeline runs | 329 |
| Noisy/ZNE runs | 93 |
| Experiments | 43 |
| **Total result files** | **465** |

## Experiment Verdicts

- ✅ Confirmed: **28** | ⚠️ Rejected: **8** | ❌ Failed: **7**
- Useful-outcome rate: **84%**

| ID | Verdict | Pass% | Criteria |
|:---|:--------|------:|:---------|
| A3 | ✅ confirmed | 100% | Scaling law R²>0.99 |
| A3_N20 | ✅ confirmed | 100% | Scaling at N=20 |
| B1 | ❌ failed | 12% | ΔE/gap < 5% |
| B2 | ✅ confirmed | 67% | Freeze works at h≥1.5 |
| B4 | ✅ confirmed | 75% | No saddle points (physics-limited pts excluded) |
| C1 | ❌ failed | 100% | Physics loss < 5% |
| C3 | ❌ failed | 67% | N=20 VQE < 5% |
| D1 | ✅ confirmed | 1% | Gradient peak detected near h_c |
| E3 | ❌ failed | 100% | ΔE/gap < 5% |
| E4 | ⚠️ rejected | 24% | HVA fails at g>0 |
| E4b | ✅ confirmed | 100% | Extended HVA fid≥0.90 at g≤0.5 |
| E4b_hardware_readiness | ✅ confirmed | — | 5/5 hypotheses confirmed |
| E4c | ✅ confirmed | 96% | Frustrated TFIM fid≥0.90 at J₂≤0.5 |
| E4c_pipeline | ✅ confirmed | 100% | ΔE/gap < 5% |
| F1 | ⚠️ rejected | 64% | DyPP > 30% |
| F3 | ✅ confirmed | 0% | Fluctuation > 1.0 everywhere |
| G1 | ✅ confirmed | 86% | ≤9 pts sufficient |
| G2 | ⚠️ rejected | 52% | Ensemble r > 0.7 |
| G3 | ⚠️ rejected | 11% | N=20 < 5% |
| G4 | ⚠️ rejected | 73% | κ predicts restarts |
| G5 | ✅ confirmed | 92% | Seed-independent (std<0.01) |
| GF_ZNE_CMP | ✅ confirmed | 100% | GF-ZNE R²>0.9 and gain>0% (consistent noise reduction) |
| HW_REHEARSAL | ⚠️ rejected | 0% | Full pipeline on FakeTorino (ZNE + classification) |
| HW_REHEARSAL_V2 | ✅ confirmed | 78% | HardwareBackend(fake_backend) with PEA/GF/adaptive ZNE |
| MPS_HW | ✅ confirmed | 100% | MPS chi-proxy matches hardware |
| PEA_HW_READY | ✅ confirmed | 100% | PEA-ZNE gain>GF-ZNE on heavy_hex N=10 (hardware target) |
| PEA_PIPELINE | ✅ confirmed | 80% | PEA-ZNE gain>50% with MPNN predictions (full pipeline) |
| PEA_ZNE_VAL | ✅ confirmed | 100% | PEA-ZNE R²>0.9 and gain>GF-ZNE (multi-seed validation) |
| S1 | ✅ confirmed | 100% | Entanglement scaling detected |
| S2 | ❌ failed | 11% | Cross-topology transfer works |
| S3 | ✅ confirmed | 44% | ΔE/gap < 5% |
| S4 | ✅ confirmed | 100% | N=10 data efficiency |
| S5 | ✅ confirmed | 100% | N=20 p=1 pipeline |
| S6 | ❌ failed | 7% | MC-Dropout UQ calibrated |
| S8 | ⚠️ rejected | 0% | D1 finite-size scaling |
| S8b | ⚠️ rejected | 0% | MPNN finite-size scaling |
| T1a | ✅ confirmed | 50% | 2D MPNN interpolates in J₂ dimension |
| T1a_dense | ✅ confirmed | 75% | Dense J₂ grid (8 values) enables cross-J₂ interpolation >50% |
| T1b | ✅ confirmed | 75% | ZNE transfers to longitudinal model (R²>0.95, gain>30%) |
| T1c | ✅ confirmed | 100% | D1 weight gradient generalizes to frustrated TFIM |
| TRANSPILER_EXPLORATION | ❌ failed | 100% | ΔE/gap < 5% |
| ZNE_3WAY | ✅ confirmed | 100% | PEA-ZNE gain ≥ GF-ZNE gain (targeted noise amplification) |
| ZNE_CROSS_TOPO | ✅ confirmed | 100% | PEA>GF across all topologies (paired t-test p<0.05, R²>0.9) |

## Noiseless Quality

- Pass rate (ΔE/gap < 5%): **62%**
- Median ΔE/gap: **0.0354**

| Topology | Runs | Pass% | Median | Best | Worst |
|:---------|-----:|------:|-------:|-----:|------:|
| chain_1d | 81 | 69% | 0.0288 | 0.0010 | 7.4079 |
| heavy_hex | 17 | 71% | 0.0056 | 0.0004 | 10.6710 |
| kagome | 2 | 100% | 0.0159 | 0.0002 | 0.0316 |
| ladder | 109 | 60% | 0.0364 | 0.0030 | 11.0644 |
| triangular | 70 | 56% | 0.0404 | 0.0019 | 14.4009 |

## VQE Convergence Quality

- Mean convergence rate: **99.58%**
- Mean θ-smoothness: **1.0469** (max: 6.1393)
- ⚠️ Chain break warnings: **96** (θ > 1.0)

## MPNN Training Quality

- Mean generalization gap: **0.004918**
- Max generalization gap: **0.078951**
- ⚠️ Overfit warnings: **41** (gen_gap > 0.01)

## Result Distribution

| Dimension | Breakdown |
|:----------|:----------|
| Model | heisenberg: 30, tfim: 295, xy: 4 |
| Topology | chain_1d: 153, heavy_hex: 25, kagome: 2, ladder: 144, triangular: 98 |
| N-qubits | N=6: 193, N=10: 211, N=16: 17, N=24: 1 |
| p-layers | p=1: 57, p=2: 272 |

## Timing

- Total compute: **14.0 hours** across 455 runs
- Per-run: mean=110.5s, median=26.9s, max=10367.4s

## Actionable Items

1. 🟡 **[MEDIUM]** ZNE validation missing for 2 config(s)
   - p=1 noiseless exists but no ZNE: chain_1d N=6, ladder N=6
2. 🟡 **[MEDIUM]** VQE chain breaks: 96/329 runs (29%) have θ-smoothness > 1.0
   - Chain breaks indicate VQE angle discontinuities across the h-sweep. Max θ-smoothness: 6.14. Root cause: descending sweep lost state continuity. Consider narrower h-grid or additional VQE restarts for affected topologies.
3. 🟡 **[MEDIUM]** VQE convergence issue: worst case only 75%
   - Mean convergence: 99.58%. Runs with < 80% convergence may need more restarts or higher max_iterations.
4. 🟡 **[MEDIUM]** MPNN overfitting: 41/279 runs (15%) have gen_gap > 0.01
   - Max generalization gap: 0.0790. Overfitting correlates with small training sets or insufficient regularization. Consider increasing patience, reducing learning rate, or using more h-points in sweep.
5. 🟡 **[MEDIUM]** p=1 under-represented: only 57/329 (17%) of noiseless runs
   - p=1 is the recommended hardware strategy. Thesis generalization claims need balanced p=1 coverage across topologies.
6. 🟡 **[MEDIUM]** 7 experiment(s) partially passed — may improve with tuning
   - Near-threshold: B1(12%), C1(100%), C3(67%), E3(100%), S2(11%), S6(7%), TRANSPILER_EXPLORATION(100%). These have partial success and may cross threshold with optimized configs (more restarts, adjusted h-grid, larger N).
7. ⚪ **[LOW]** 4 result(s) tested outside valid regime
   - These are expected to fail — not actionable unless they passed unexpectedly.
8. ⚪ **[LOW]** MPNN dominates error: 95% of total ΔE comes from MPNN prediction
   - Mean circuit error: 0.0222, Mean MPNN error: 0.4119. Circuit expressibility is not the bottleneck — focus improvement on MPNN architecture/training.
