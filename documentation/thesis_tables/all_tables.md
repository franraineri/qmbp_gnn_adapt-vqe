# Thesis Tables (Auto-Generated)

## T1 — Global Pipeline Performance Summary

*Aggregated pipeline performance across all topologies and system sizes. Pass criterion: ΔE/gap < 5%.*

| Topology | N | Runs | Pass Rate | Median ΔE/gap | Mean ΔE/gap | Best ΔE/gap | Mean θ-smooth | Mean Gen.Gap | Mean Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| chain_1d | 6 | 37 | 70% | 0.0222 | 0.0919 | 0.0028 | 0.569 | 2.65e-03 | 41 |
| chain_1d | 10 | 44 | 68% | 0.0290 | 0.2120 | 0.0010 | 0.464 | 5.28e-04 | 26 |
| heavy_hex | 10 | 17 | 71% | 0.0056 | 1.6293 | 0.0004 | 1.365 | 1.77e-03 | 24 |
| kagome | 6 | 1 | 100% | 0.0316 | 0.0316 | 0.0316 | 3.177 | 2.39e-03 | 19 |
| kagome | 10 | 1 | 100% | 0.0002 | 0.0002 | 0.0002 | 0.015 | 1.94e-04 | 25 |
| ladder | 6 | 43 | 56% | 0.0322 | 0.2524 | 0.0030 | 1.274 | 1.19e-02 | 191 |
| ladder | 10 | 66 | 62% | 0.0368 | 0.3623 | 0.0120 | 0.896 | 1.50e-03 | 32 |
| triangular | 6 | 39 | 54% | 0.0479 | 0.3333 | 0.0019 | 1.326 | 1.22e-02 | 18 |
| triangular | 10 | 31 | 58% | 0.0389 | 1.0610 | 0.0141 | 0.959 | 4.27e-03 | 27 |
| **ALL** | — | 279 | 62% | 0.0354 | 0.4341 | 0.0002 | — | — | — |

**Notes**: Total runs: 279. Topologies: 5.


---

## T2 — ZNE Strategy Comparison

*Comparison of ZNE amplification strategies across all experiments. PEA = Probabilistic Error Amplification, GF = Gate Folding, CES = Circuit Error Scaling.*

| Strategy | Runs | Mean Gain (%) | Std Gain | Mean R² | Win Rate | Best Gain (%) | Topologies |
|---|---|---|---|---|---|---|---|
| UNKNOWN | 93 | +28.5 | 48.6 | 0.968 | 199/349 | +87.4 | chain_1d, heavy_hex, ladder, triangular |

**Notes**: Win rate = fraction of h-points where ZNE improves over noisy raw.


---

## T3 — Scaling Law Validation

*Pipeline performance across system sizes N=6 to N=80. Scaling law: $h_{min} = 1.0 + 0.020 \cdot N^{1.31} + 0.50$.*

| N | Topology | Backend | Runs | Pass Rate | Mean ΔE/gap (%) | Max ΔE/gap (%) | h_min Predicted | h_min Used | Total Time |
|---|---|---|---|---|---|---|---|---|---|
| 6 | chain_1d | statevector | 37 | 70% | 9.19 | 162.20 | 1.71 | 1.15 | — |
| 10 | chain_1d | statevector | 44 | 68% | 21.20 | 740.79 | 1.91 | 1.15 | — |
| 40 | chain_1d | aer_mps | 1 | 5/5 | 0.49 | 0.60 | 4.01 | 4.01 | 1571s |
| 40 | chain_1d | aer_mps | 1 | 27/27 | 0.68 | 2.36 | 4.01 | 3.00 | 12230s |
| 50 | chain_1d | aer_mps | 1 | 5/5 | 0.36 | 0.49 | 4.86 | 4.86 | 1803s |
| 50 | chain_1d | aer_mps | 1 | 5/5 | 0.36 | 0.49 | 4.86 | 4.86 | 2947s |
| 50 | chain_1d | aer_mps | 1 | 5/5 | 0.36 | 0.49 | 4.86 | 4.86 | 3032s |
| 50 | chain_1d | aer_mps | 1 | 5/5 | 0.36 | 0.49 | 4.86 | 4.86 | 5829s |
| 50 | chain_1d | aer_mps | 1 | 5/5 | 0.36 | 0.49 | 4.86 | 4.86 | 4919s |
| 80 | chain_1d | aer_mps | 1 | 5/5 | 0.08 | 0.10 | 7.72 | 7.72 | 109s |

**Notes**: Pass criterion: ΔE/gap < 5%. MPS backend used for N≥40.


---

## T4 — GNN-QEM Error Correction Results

*Summary of GNN-based quantum error mitigation across all evaluation modes.*

| Experiment | Mode | Metric | Value | N Points | Verdict |
|---|---|---|---|---|---|
| In-Distribution | Correction | Error Reduction | 0.0% | 0 | ⚠️ |
| Cross-Topology | Zero-Shot Transfer | Improvement Rate | 0% | 0 | ⚠️ |
| Ablation (no E_noisy) | Predictive | GNN vs MLP Accuracy | GNN=0% / MLP=0% | 0 | ✅ |
| Post-ZNE Composability | Correction | Regression Rate | 0/0 | 0 | ❌ (Expected) |
| Circuit Selection | Predictive (no E_noisy) | Spearman ρ | 0.000 | 0 | ⚠️ |

**Notes**: GINConv(3L, h=64), 30K params. Trained on chain_1d + ladder noise data.


---

## T5 — Experiment Verdicts Summary

*Classification of all formal experiments by category and outcome. Rejected results represent valid negative findings (contribute to knowledge).*

| Category | Total | Confirmed | Rejected (valid) | Failed | Success % |
|---|---|---|---|---|---|
| A | 2 | 2 | 0 | 0 | 100% |
| B | 5 | 4 | 0 | 1 | 80% |
| C | 2 | 0 | 0 | 2 | 0% |
| D | 1 | 1 | 0 | 0 | 100% |
| E | 8 | 4 | 1 | 3 | 62% |
| F | 2 | 1 | 1 | 0 | 100% |
| G | 5 | 2 | 3 | 0 | 100% |
| Gf | 1 | 1 | 0 | 0 | 100% |
| H | 2 | 1 | 1 | 0 | 100% |
| M | 1 | 1 | 0 | 0 | 100% |
| N | 1 | 1 | 0 | 0 | 100% |
| S | 8 | 4 | 2 | 2 | 75% |
| T | 5 | 4 | 0 | 1 | 80% |
| Zne | 6 | 6 | 0 | 0 | 100% |
| **TOTAL** | 49 | 32 | 8 | 9 | 82% |

**Notes**: Total experiments: 49. Useful rate (confirmed+rejected): 82%


---

## T6 — Cross-Topology GNN Transfer Performance

*GNN generalization across unseen topologies and system sizes.*

| Experiment Type | Source | Target | Mean ΔE/gap | Pass Rate | N Points | Verdict |
|---|---|---|---|---|---|---|
| Cross N Validation | triangular, heavy_hex | triangular, heavy_hex | 5.0186 | 0/0 | 0 | FAIL |
| Cross N Validation | triangular, heavy_hex | triangular, heavy_hex | 5.0186 | 0/0 | 0 | FAIL |
| Cross Topology Transfer | triangular, heavy_hex | heavy_hex, triangular | 7.1940 | 0/0 | 0 | FAIL |
| Orchestrator Summary | — | — | 0.0000 | 0/0 | 0 | PARTIAL (errors) |

**Notes**: norm_type='none' used for all cross-topology experiments (BatchNorm harmful).


---

## T7 — Failure Mode Distribution

*Root cause classification of 105 failed pipeline runs (ΔE/gap ≥ 5%).*

| Failure Mode | Count | Percentage | Detection Phase | Preventable |
|---|---|---|---|---|
| CHAIN_BREAK | 47 | 45% | Phase 2 (θ_smooth > 1.0) | Yes — pre-run regime check |
| OTHER | 43 | 41% | Phase 4 | No — inherent limit |
| MPNN_OVERFIT | 15 | 14% | Phase 3 (gen_gap > 0.01) | Yes — early stopping |

**Notes**: 69% of failures are preventable through pre-run regime checking.


---

## T8 — Hyperparameter Sensitivity Analysis (N=10)

*Sensitivity of pipeline performance to key hyperparameters at N=10.*

| Parameter | Values Tested | Median ΔE/gap Range | Relative Spread | Sensitivity |
|---|---|---|---|---|
| hidden_dim | 64, 128, 256 | 0.0335 – 0.0372 | 10% | LOW |
| n_restarts | 1, 3, 5, 7 | 0.0168 – 0.0750 | 141% | HIGH |
| topology | chain_1d, heavy_hex, ladder, triangular | 0.0056 – 0.0389 | 120% | HIGH |
| seed | 42, 43, 44 | 0.0334 – 0.0421 | 23% | LOW |

**Notes**: LOW = <30% relative spread, MODERATE = 30-70%, HIGH = >70%.


---

## T9 — MPS Backend Performance

*Performance of the Matrix Product State backend for large system sizes (N > 30).*

| N | χ_max | Strategy | Mean ΔE/gap (%) | Max ΔE/gap (%) | Phase 1 (s) | Phase 2 (s) | Total (s) | Status |
|---|---|---|---|---|---|---|---|---|
| 40 | 64 | aer_mps | 0.49 | 0.60 | 15 | 1555 | 1571 | ✅ ALL PASS |
| 40 | 64 | aer_mps | 0.68 | 2.36 | 33 | 12198 | 12230 | ✅ ALL PASS |
| 50 | 64 | aer_mps | 0.36 | 0.49 | 20 | 1784 | 1803 | ✅ ALL PASS |
| 50 | 64 | aer_mps | 0.36 | 0.49 | 21 | 2926 | 2947 | ✅ ALL PASS |
| 50 | 64 | aer_mps | 0.36 | 0.49 | 41 | 2991 | 3032 | ✅ ALL PASS |
| 50 | 64 | aer_mps | 0.36 | 0.49 | 18 | 5811 | 5829 | ✅ ALL PASS |
| 50 | 64 | aer_mps | 0.36 | 0.49 | 132 | 4788 | 4919 | ✅ ALL PASS |
| 80 | 64 | aer_mps | 0.08 | 0.10 | 69 | 39 | 109 | ✅ ALL PASS |

**Notes**: χ=64 validated exact for HVA p≤2 on 1D TFIM. COBYLA optimizer used (L-BFGS-B fails with shots).


---

## T10 — Phase-by-Phase Timing Breakdown

*Average time spent in each pipeline phase by system size.*

| N | Runs | Phase 1 (s) | Phase 2 (s) | Phase 3 (s) | Total (s) | Phase 2 % |
|---|---|---|---|---|---|---|
| 6 | 150 | 0.0 | — | 66.1 | 78.5 | — |
| 10 | 161 | 6.0 | — | 14.3 | 28.8 | — |
| 16 | 17 | 7.4 | — | — | 85.8 | — |
| 24 | 1 | 10.3 | — | — | 1491.2 | — |

**Notes**: Phase 2 (VQE) dominates at all system sizes. Phase 1 negligible for N≤10.

