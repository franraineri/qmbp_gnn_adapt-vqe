```
================================================================================
  PIPELINE EXECUTION SUMMARY
  Generated: 2026-08-18T20:44:27Z
================================================================================

┌─ ARCHITECTURE ABLATION ─────────────────────────────────────────────┐
│ Topology: multi_topology | Epochs: 500 | Graphs: 2199 | Filter: max_de_gap=0.05
│ Best: residual
│ Variant                 val_MSE        MSE  Epochs            Stop
│ baseline               2.35e-01   2.03e-01     500       completed
│ residual               1.87e-01   1.37e-01     500       completed ★
│ jk_cat                 2.49e-01   1.83e-01     401 overfitting_det
│ film                   2.32e-01   2.06e-01     451 overfitting_det
│ res+jk+film            2.29e-01   1.83e-01     500       completed
└────────────────────────────────────────────────────────────────────┘

┌─ TRAINING CONVERGENCE ─────────────────────────────────────────────┐
│ ladder_iter1_p1_20260818_173827.npz         1201ep MSE=1.15e-01 val=1.36e-01 (↓-8%)
│ ladder_iter2_p1_20260818_180536.npz          351ep MSE=1.15e-01 val=1.36e-01 (↓-3%)
│ mt_training_20260818_173611.npz              401ep MSE=1.70e-01 val=2.22e-01 (↓96%)
│ mt_training_20260818_181232.npz              351ep MSE=1.29e-01 val=1.83e-01 (↓24%)
│ triangular_iter1_p1_20260818_165533.npz      251ep MSE=1.37e-01 val=1.52e-01 (↓7%)
└────────────────────────────────────────────────────────────────────┘

┌─ MULTI-TOPOLOGY vs SINGLE-TOPOLOGY (head-to-head) ─────────────────┐
│ Topology      MT ΔE/gap  Single ΔE/gap   Winner     Source
│ ──────────── ────────── ────────────── ──────── ──────────
│ chain_1d         0.0585         0.2199 🟢 MT    comparison
│ heavy_hex           N/A            N/A 🔴 single   zoo_only
│ ladder           0.6287         0.7810 🟢 MT    comparison
│ square              N/A            N/A 🟢 MT      zoo_only
│ triangular          N/A            N/A 🟢 MT      zoo_only
│
│ Score: MT wins 4 | Single wins 1 | Ties 0
│ MT ΔE/gap advantage: +0.1569 (positive = MT has lower error)
│ Verdict: MT model generalizes BETTER across topologies
└────────────────────────────────────────────────────────────────────┘

┌─ ZOO MODEL STATUS ────────────────────────────────────────────────┐
│ chain_1d        2 models | 2 eval'd | best=100% |   725 pts
│ heavy_hex       2 models | 2 eval'd | best=100% |   453 pts
│ ladder          1 models | 1 eval'd | best= 18% |   470 pts
│ multi_topology  1 models | 1 eval'd | best= 84% |  2239 pts
│ square          1 models | 1 eval'd | best= 33% |   560 pts
│ triangular      1 models | 1 eval'd | best= 25% |   231 pts
└────────────────────────────────────────────────────────────────────┘

┌─ LARGE-N EXTRAPOLATION ────────────────────────────────────────────┐
│ Topology        N  Pts  Pass%   ΔE/gap  Grade
│ chain_1d       16    6  100%   0.0206      B
│ chain_1d       20    6  100%   0.0090      B
│ chain_1d       30   20   70%   0.0358      B
│ chain_1d       40   16   62%   0.0480      C
│ chain_1d       60   16   31%   0.0862      D
│ chain_1d      100   12   25%   0.1111      D
│ chain_1d      150    3    0%   0.7841      F
│ chain_1d      200    3    0%   1.0472      F
│ heavy_hex      20   25   36%   0.1946      F
│ heavy_hex      30   14   36%   1.2121      F
│ heavy_hex      40    6   83%   0.0390      C
│ ladder         20   16   38%   0.2666      F
│ ladder         26    6   33%   0.6711      F
│ ladder         30   14   43%   0.5130      F
│ ladder         40    6   17%   1.5579      F
│ square         16   18   33%   0.0904      D
│ square         20   18    6%   0.9344      F
│ square         30   12    0%   2.5784      F
│ triangular     12   10   20%   1.7850      F
│ triangular     16   10    0%  28.7030      F
│ triangular     24   10    0%  23.4736      F
└────────────────────────────────────────────────────────────────────┘

┌─ MODEL COMPARISONS (latest per topology) ─────────────────────────┐
│ chain_1d     | 3 models → Winner: orphan (unified_tfim_br_multitopo_chain_1d+heavy) (arch=residual+film)
│ heavy_hex    | 3 models → Winner: None (arch=None)
│ ladder       | 9 models → Winner: None (arch=None)
│ square       | 8 models → Winner: None (arch=None)
└────────────────────────────────────────────────────────────────────┘

================================================================================
```
