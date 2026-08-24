# Best Results Scoreboard

**Updated**: 2026-08-24 14:47 UTC
**Reference h-value**: 2.50 (hardest region near h_critical; actual h used noted per entry)
**Reports scanned**: 299
**Criterion**: Best ΔE/gap achieved at h≈2.5 per (topology × N)

> This report shows the **best single-point result ever achieved** at h≈2.5 for each
> (topology, N) combination in the **extrapolation regime** (N values tested with MPNN zero-shot prediction).
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

| N | |ΔE| | ΔE/gap | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|-------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0549 | 0.0158 | 3.4808 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 | `eval_chain_1d_20260821_034724.md` |
| 6 | 0.0733 | 0.0224 | 3.2687 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 8 | 0.0716 | 0.0226 | 3.1711 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 0.0928 | 0.0313 | 2.9602 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151217.md` |
| 12 | 0.0924 | 0.0299 | 3.0867 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 14 | 0.6441 | 0.2100 | 3.0662 | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 | `eval_chain_1d_20260821_034724.md` |
| 16 | 0.3151 | 0.1032 | 3.0522 | D | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-20 | `eval_chain_1d_MT_20260820_141417.md` |
| 20 | 0.0848 | 0.0282 | 3.0099 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 21 | 0.3437 | 0.1142 | 3.0090 | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 26 | 0.5790 | 0.1926 | 3.0058 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_151454.md` |
| 30 | 0.1320 | 0.0439 | 3.0044 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 31 | 0.6772 | 0.2254 | 3.0041 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 40 | 0.1793 | 0.0597 | 3.0025 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 60 | 0.2739 | 0.0913 | 3.0011 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 80 | 1.6095 | 0.5364 | 3.0006 | F | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-20 | `eval_chain_1d_20260820_163151.md` |


## heavy_hex

**h used**: varies (2.500 – 2.600)

| N | |ΔE| | ΔE/gap | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|-------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0328 | 0.0097 | 3.3907 | A | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-21 | `eval_heavy_hex_20260821_022124.md` |
| 6 | 0.0929 | 0.0284 | 3.2687 | B | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 | `eval_heavy_hex_MT_20260821_022124.md` |
| 8 | 0.1124 | 0.0372 | 3.0223 | C | ST | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 | `eval_heavy_hex_20260824_042828.md` |
| 10 | 0.0441 | 0.0150 | 2.9452 | A | ST | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 | `eval_heavy_hex_20260820_141344.md` |
| 12 | 0.2262 | 0.0733 | 3.0867 | C | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 | `eval_heavy_hex_MT_20260821_022124.md` |
| 14 | 0.0928 | 0.0324 | 2.8645 | B | ST | data/model_zoo/checkpoints/unified_tf... | 2026-08-23 | `eval_heavy_hex_20260823_031107.md` |
| 16 | 0.0580 | 0.0204 | 2.8412 | B | ST | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 | `eval_heavy_hex_20260820_141430.md` |
| 20 | 0.0705 | 0.0583 | 1.2099 | B | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 21 | 1.1831 | 0.9864 | 1.1994 | F | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_031400.md` |
| 22 | 1.0425 | 0.8760 | 1.1900 | F | MT | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 | `eval_heavy_hex_MT_20260824_005323.md` |
| 24 | 0.1037 | 0.0884 | 1.1735 | C | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_180207.md` |
| 26 | 0.0942 | 0.0813 | 1.1597 | B | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 | `eval_heavy_hex_20260823_180207.md` |
| 29 | 0.7108 | 0.5295 | 1.3424 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_095240.md` ([json](results/model_comparison/compare_heavy_hex_20260819_095240.json)) |
| 30 | 0.1321 | 0.1161 | 1.1377 | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 32 | 1.9998 | 1.7715 | 1.1289 | F | MT | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 | `eval_heavy_hex_MT_20260824_005323.md` |
| 40 | 0.1416 | 0.1284 | 1.1025 | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 50 | 0.1741 | 0.1609 | 1.0816 | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 | `eval_heavy_hex_20260824_045946.md` |
| 60 | 0.2739 | 0.2565 | 1.0678 | C | ST | data/model_zoo/checkpoints/unified_mu... | 2026-08-24 | `eval_heavy_hex_20260824_055819.md` |


## ladder

**h used**: 2.500

| N | |ΔE| | ΔE/gap | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|-------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.1004 | 0.0330 | 3.0377 | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_091547.md` ([json](results/model_comparison/compare_ladder_20260819_091547.json)) |
| 6 | 0.0955 | 0.0376 | 2.5418 | B | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 8 | 0.1499 | 0.0662 | 2.2654 | C | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 10 | 0.2105 | 0.1005 | 2.0937 | C | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_194327.md` |
| 12 | 0.2677 | 0.1353 | 1.9791 | C | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_044849.md` ([json](results/model_comparison/compare_ladder_20260819_044850.json)) |
| 14 | 0.3363 | 0.1772 | 1.8984 | D | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_030805.md` |
| 16 | 0.3642 | 0.1980 | 1.8395 | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 | `eval_ladder_20260821_194327.md` |
| 18 | 0.4937 | 0.2751 | 1.7950 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 20 | 0.4652 | 1.4809 | 0.3142 | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 22 | 0.6113 | 2.1405 | 0.2856 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 24 | 0.6715 | 2.5648 | 0.2618 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 26 | 0.6306 | 2.6094 | 0.2417 | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 30 | 0.7350 | 3.5093 | 0.2094 | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 40 | 0.9671 | 6.1568 | 0.1571 | D | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |


## square

**h used**: varies (2.500 – 2.570)

| N | |ΔE| | ΔE/gap | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|-------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0704 | 0.0232 | 3.0377 | B | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 6 | 0.1166 | 0.0459 | 2.5418 | C | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 8 | 0.1708 | 0.0771 | 2.2139 | C | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 10 | 0.2487 | 0.1141 | 2.1796 | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_153202.md` ([json](results/experiments/exp_model_comparison/tfim_bond_resolved/square/run_20260819_152900.json)) |
| 12 | 0.3840 | 0.2301 | 1.6687 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 14 | 0.4655 | 0.2996 | 1.5538 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 16 | 0.5203 | 0.3941 | 1.3203 | D | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |
| 18 | 0.6918 | 0.5443 | 1.2709 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 20 | 0.7209 | 2.2948 | 0.3142 | D | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |
| 21 | 0.8662 | 2.8952 | 0.2992 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 30 | 1.2997 | 6.2054 | 0.2094 | F | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |


## triangular

**h used**: 2.500

| N | |ΔE| | ΔE/gap | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-----:|-------:|----:|:-----:|:-----:|-----------|------|--------|
| 3 | 0.0990 | 0.0318 | 3.1139 | B | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092425.md` ([json](results/model_comparison/compare_triangular_20260819_092425.json)) |
| 4 | 0.1691 | 0.0815 | 2.0755 | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 6 | 0.3334 | 0.2246 | 1.4844 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 8 | 1.0689 | 1.5770 | 0.6778 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 1.1853 | 2.1972 | 0.5395 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_145755.md` |
| 11 | 1.8466 | 5.4112 | 0.3413 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 12 | 2.4575 | 11.5427 | 0.2129 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 13 | 3.0778 | 23.3216 | 0.1320 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 14 | 3.7469 | 46.1325 | 0.0812 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-21 | `eval_triangular_20260821_043600.md` |
| 16 | 5.2136 | 170.9028 | 0.0305 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_160246.md` |
| 24 | 17.5899 | 67.1886 | 0.2618 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-18 | `eval_triangular_20260818_130854.md` |


---

*Auto-generated by `scripts/analysis/generate_best_results_scoreboard.py`*
*Data sources: `results/extrapolation_evals/` + `results/model_comparison/`*
---

## Cross-Validation (vs ModelRegistryDB)

| Issue | Detail |
|---|---|
| ⚠️ | chain_1d N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | chain_1d N=80: scoreboard |ΔE|@h=2.5 = 1.609 >> registry mean ΔE/gap = 0.144 — h=2.5 is anomalously hard for this config |
| ⚠️ | heavy_hex N=4: scoreboard grade=A but zoo pass_rate_by_n[4]=0% — possible stale zoo data |
| ⚠️ | heavy_hex N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | ladder N=20: scoreboard |ΔE|@h=2.5 = 0.465 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=26: scoreboard |ΔE|@h=2.5 = 0.631 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=30: scoreboard |ΔE|@h=2.5 = 0.735 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=40: scoreboard |ΔE|@h=2.5 = 0.967 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | square N=30: scoreboard |ΔE|@h=2.5 = 1.300 >> registry mean ΔE/gap = 1.847 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular N=16: scoreboard |ΔE|@h=2.5 = 5.214 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular N=24: scoreboard |ΔE|@h=2.5 = 17.590 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
