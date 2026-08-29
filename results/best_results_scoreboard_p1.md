# Best Results Scoreboard — p=1

**Updated**: 2026-08-29 00:08 UTC
**p_layers**: 1
**Reference h-value**: 2.50 (hardest region near h_critical; actual h used noted per entry)
**Reports scanned**: 313
**Criterion**: Best ΔE/gap achieved at h≈2.5 per (topology × N)

> This report shows the **best single-point result ever achieved** at h≈2.5 for each (topology, N) combination at **p=1**, in the **extrapolation regime** (N values tested with MPNN zero-shot prediction).
> Each p has its own scoreboard file — p=1 and p=2 are different ansätze and never compete against each other.
> It does NOT average over h — it tracks the hardest operating point near h_critical.
> For in-distribution quality (training N), see `model_evaluation_report.md`.
> Grade thresholds: A (|ΔE|<0.05), B (<0.10), C (<0.30), D (<1.00), F (≥1.00).

---

## Summary: Best Grade per Topology

| Topology | Max N evaluated | Best grade | Best |ΔE| (any N) | Best model type | N trained up to |
|---|---|---|---|---|---|
| chain_1d | 80 | B | 0.0549 | ST | 80 |
| heavy_hex | 60 | A | 0.0328 | ST | 60 |
| ladder | 40 | B | 0.0955 | ST | 40 |
| square | 30 | B | 0.0704 | ST | 30 |
| triangular | 24 | B | 0.0990 | MT | 16 |

---

## chain_1d

**h used**: varies (2.420 – 2.500)

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0549 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 | `eval_chain_1d_20260821_034724.md` |
| 6 | 0.0733 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 8 | 0.0716 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 0.0928 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151217.md` |
| 12 | 0.0924 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 14 | 0.6441 | N/A | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 | `eval_chain_1d_20260821_034724.md` |
| 16 | 0.3032 | 0.9664 | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-28 | `eval_chain_1d_20260828_232203.md` |
| 20 | 0.0848 | N/A | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 21 | 0.3437 | N/A | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 26 | 0.5790 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_151454.md` |
| 30 | 0.1320 | N/A | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 31 | 0.6772 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 40 | 0.1793 | N/A | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 60 | 0.2739 | N/A | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 80 | 1.6095 | N/A | F | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-20 | `eval_chain_1d_20260820_163151.md` |


## heavy_hex

**h used**: varies (2.500 – 2.630)

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0328 | N/A | A | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-21 | `eval_heavy_hex_20260821_022124.md` |
| 6 | 0.0929 | N/A | B | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 | `eval_heavy_hex_MT_20260821_022124.md` |
| 8 | 0.0898 | N/A | B | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 | `eval_heavy_hex_20260824_183127.md` |
| 10 | 0.0441 | N/A | A | ST | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 | `eval_heavy_hex_20260820_141344.md` |
| 12 | 0.2262 | N/A | C | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 | `eval_heavy_hex_MT_20260821_022124.md` |
| 14 | 0.0928 | N/A | B | ST | data/model_zoo/checkpoints/unified_tf... | 2026-08-23 | `eval_heavy_hex_20260823_031107.md` |
| 16 | 0.0501 | N/A | B | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 | `eval_heavy_hex_20260824_160401.md` |
| 18 | 1.5039 | N/A | F | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 | `eval_heavy_hex_20260824_160900.md` |
| 20 | 0.0705 | N/A | B | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 21 | 1.1831 | N/A | F | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_031400.md` |
| 22 | 1.0425 | N/A | F | MT | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 | `eval_heavy_hex_MT_20260824_005323.md` |
| 24 | 0.1037 | N/A | C | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_180207.md` |
| 26 | 0.0942 | N/A | B | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_180207.md` |
| 29 | 0.7108 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_095240.md` ([json](results/model_comparison/compare_heavy_hex_20260819_095240.json)) |
| 30 | 0.1321 | N/A | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 32 | 1.9998 | N/A | F | MT | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 | `eval_heavy_hex_MT_20260824_005323.md` |
| 40 | 0.1416 | N/A | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 50 | 0.1741 | N/A | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 | `eval_heavy_hex_20260824_045946.md` |
| 60 | 0.2739 | N/A | C | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-24 | `eval_heavy_hex_20260824_055819.md` |


## ladder

**h used**: 2.500

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.1004 | N/A | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_091547.md` ([json](results/model_comparison/compare_ladder_20260819_091547.json)) |
| 6 | 0.0955 | N/A | B | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 8 | 0.1499 | N/A | C | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 10 | 0.2105 | N/A | C | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_194327.md` |
| 12 | 0.2677 | N/A | C | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_044849.md` ([json](results/model_comparison/compare_ladder_20260819_044850.json)) |
| 14 | 0.3363 | N/A | D | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 16 | 0.3642 | N/A | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_194327.md` |
| 18 | 0.4937 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 20 | 0.4652 | N/A | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 22 | 0.6113 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 24 | 0.6715 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 26 | 0.6306 | N/A | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 30 | 0.7350 | N/A | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 40 | 0.9671 | N/A | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |


## square

**h used**: varies (2.500 – 2.570)

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0704 | N/A | B | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 6 | 0.1166 | N/A | C | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 8 | 0.1708 | N/A | C | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 10 | 0.2487 | N/A | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_153202.md` ([json](results/experiments/exp_model_comparison/tfim_bond_resolved/square/run_20260819_152900.json)) |
| 12 | 0.3840 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 14 | 0.4655 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 16 | 0.5203 | N/A | D | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |
| 18 | 0.6918 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 20 | 0.7209 | N/A | D | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |
| 21 | 0.8662 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 30 | 1.2997 | N/A | F | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |


## triangular

**h used**: 2.500

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 3 | 0.0990 | N/A | B | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092425.md` ([json](results/model_comparison/compare_triangular_20260819_092425.json)) |
| 4 | 0.1691 | N/A | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 6 | 0.3334 | N/A | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 8 | 1.0689 | N/A | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 1.1853 | N/A | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_145755.md` |
| 11 | 1.8466 | N/A | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 12 | 2.4575 | N/A | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 13 | 3.0778 | N/A | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 14 | 3.7469 | N/A | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-21 | `eval_triangular_20260821_043600.md` |
| 16 | 5.2136 | N/A | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_160246.md` |
| 24 | 17.5899 | N/A | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-18 | `eval_triangular_20260818_130854.md` |


---

*Auto-generated by `scripts/analysis/generate_best_results_scoreboard.py`*
*Data sources: `results/extrapolation_evals/` + `results/model_comparison/`*
---

## Cross-Validation (vs ModelRegistryDB)

| Issue | Detail |
|---|---|
| ⚠️ | chain_1d p=1 N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | chain_1d p=1 N=80: scoreboard |ΔE|@h=2.5 = 1.609 >> registry mean ΔE/gap = 0.144 — h=2.5 is anomalously hard for this config |
| ⚠️ | heavy_hex p=1 N=4: scoreboard grade=A but zoo pass_rate_by_n[4]=0% — possible stale zoo data |
| ⚠️ | heavy_hex p=1 N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | ladder p=1 N=20: scoreboard |ΔE|@h=2.5 = 0.465 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=26: scoreboard |ΔE|@h=2.5 = 0.631 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=30: scoreboard |ΔE|@h=2.5 = 0.735 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=40: scoreboard |ΔE|@h=2.5 = 0.967 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | square p=1 N=30: scoreboard |ΔE|@h=2.5 = 1.300 >> registry mean ΔE/gap = 1.847 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular p=1 N=16: scoreboard |ΔE|@h=2.5 = 5.214 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular p=1 N=24: scoreboard |ΔE|@h=2.5 = 17.590 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |


---

## chain_1d near h_critical (h≈1.0)

Best-ever single-point result for **chain_1d** at h≈1.0 (±0.15), the critical region where the gap is smallest and the ansatz is most stressed. Same ranking as the main scoreboard (lowest |ΔE| per N).

**h used**: 1.130

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 10 | 0.3450 | 0.8785 | D | ST | unified_tfim_bond_resolved_chain_1d_n... | 2026-08-28 | `eval_chain_1d_20260828_224901.md` |
| 20 | 0.9428 | N/A | D | ST | unifMPNN__chain_1d_p1_h_0p5_1p5 | 2026-08-28 | `eval_chain_1d_20260828_224043.md` |


---

## chain_1d near h_critical (h≈1.5)

Best-ever single-point result for **chain_1d** at h≈1.5 (±0.15), the critical region where the gap is smallest and the ansatz is most stressed. Same ranking as the main scoreboard (lowest |ΔE| per N).

**h used**: varies (1.500 – 1.610)

| N | |ΔE| | Fidelity | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|:--------:|:-----:|:-----:|-----------|------|--------|
| 10 | 0.1313 | 0.9733 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-28 | `eval_chain_1d_20260828_223158.md` |
| 16 | 0.6070 | N/A | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151217.md` |
| 20 | 0.5635 | N/A | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151217.md` |
| 26 | 1.0598 | N/A | F | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151454.md` |
| 30 | 1.6112 | N/A | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_151454.md` |
