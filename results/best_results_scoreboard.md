# Best Results Scoreboard

**Updated**: 2026-08-20 19:09 UTC
**Reference h-value**: 2.50 (hardest region near h_critical; actual h used noted per entry)
**Reports scanned**: 168
**Criterion**: Best ΔE/gap achieved at h≈2.5 per (topology × N)

> This report shows the **best single-point result ever achieved** at h≈2.5 for each
> (topology, N) combination in the **extrapolation regime** (N values tested with MPNN zero-shot prediction).
> It does NOT average over h — it tracks the hardest operating point near h_critical.
> For in-distribution quality (training N), see `model_evaluation_report.md`.
> Grade thresholds: A (<3%), B (<5%), C (<10%), D (<50%), F (≥50%).

---

## Summary: Best Grade per Topology

| Topology | Max N evaluated | Best grade | Best ΔE/gap (any N) | Best model type | N trained up to |
|---|---|---|---|---|---|
| chain_1d | 80 | A | 0.0224 | ST | 80 |
| heavy_hex | 40 | A | 0.0150 | ST | 40 |
| ladder | 40 | B | 0.0330 | MT | 40 |
| square | 30 | A | 0.0232 | ST | 30 |
| triangular | 24 | B | 0.0318 | MT | 11 |

---

## chain_1d

**h used**: varies (2.420 – 2.500)

| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-------:|-----:|------:|----:|:-----:|:-----:|-----------|------|--------|
| 6 | 0.0224 | 0.0733 | 1.22e-02 | 3.2687 | A | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 8 | 0.0226 | 0.0716 | 8.95e-03 | 3.1711 | A | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 0.0313 | 0.0928 | 9.28e-03 | 2.9602 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_151217.md` |
| 12 | 0.0299 | 0.0924 | 7.70e-03 | 3.0867 | A | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_092053.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 16 | 0.1032 | 0.3151 | 1.97e-02 | 3.0522 | D | MT | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-20 | `eval_chain_1d_MT_20260820_141417.md` |
| 20 | 0.0282 | 0.0848 | 4.24e-03 | 3.0099 | A | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 21 | 0.1142 | 0.3437 | 1.64e-02 | 3.0090 | D | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 | `eval_chain_1d_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 26 | 0.1926 | 0.5790 | 2.23e-02 | 3.0058 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_151454.md` |
| 30 | 0.0439 | 0.1320 | 4.40e-03 | 3.0044 | B | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 31 | 0.2254 | 0.6772 | 2.18e-02 | 3.0041 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_chain_1d_MT_20260819_091537.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 40 | 0.0597 | 0.1793 | 4.48e-03 | 3.0025 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 60 | 0.0913 | 0.2739 | 4.57e-03 | 3.0011 | C | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 | `eval_chain_1d_20260817_164925.md` |
| 80 | 0.5364 | 1.6095 | 2.01e-02 | 3.0006 | F | ST | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-20 | `eval_chain_1d_20260820_163151.md` |


## heavy_hex

**h used**: varies (2.500 – 2.600)

| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-------:|-----:|------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0443 | 0.1590 | 3.98e-02 | 3.5852 | B | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-19 | `eval_heavy_hex_20260819_091346.md` ([json](results/model_comparison/compare_heavy_hex_20260819_091346.json)) |
| 6 | 0.0347 | 0.1203 | 2.01e-02 | 3.4652 | B | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_091346.md` ([json](results/model_comparison/compare_heavy_hex_20260819_091346.json)) |
| 10 | 0.0150 | 0.0441 | 4.41e-03 | 2.9452 | A | ST | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 | `eval_heavy_hex_20260820_141344.md` |
| 12 | 0.0812 | 0.2667 | 2.22e-02 | 3.2851 | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_091346.md` ([json](results/model_comparison/compare_heavy_hex_20260819_091346.json)) |
| 16 | 0.0204 | 0.0580 | 3.63e-03 | 2.8412 | A | ST | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 | `eval_heavy_hex_20260820_141430.md` |
| 20 | 0.0583 | 0.0705 | 3.53e-03 | 1.2099 | C | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 24 | 0.3588 | 0.4927 | 2.05e-02 | 1.3733 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_095240.md` ([json](results/model_comparison/compare_heavy_hex_20260819_095240.json)) |
| 26 | 0.6179 | 0.7166 | 2.76e-02 | 1.1597 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_154620.md` |
| 29 | 0.5295 | 0.7108 | 2.45e-02 | 1.3424 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_heavy_hex_MT_20260819_095240.md` ([json](results/model_comparison/compare_heavy_hex_20260819_095240.json)) |
| 30 | 0.1161 | 0.1321 | 4.40e-03 | 1.1377 | D | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |
| 40 | 0.1284 | 0.1416 | 3.54e-03 | 1.1025 | D | ST | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 | `eval_heavy_hex_20260817_164925.md` |


## ladder

**h used**: varies (2.500 – 2.570)

| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-------:|-----:|------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0330 | 0.1004 | 2.51e-02 | 3.0377 | B | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_091547.md` ([json](results/model_comparison/compare_ladder_20260819_091547.json)) |
| 6 | 0.0583 | 0.1482 | 2.47e-02 | 2.5418 | C | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_091546.md` ([json](results/model_comparison/compare_chain_1d_20260819_091537.json)) |
| 8 | 0.0686 | 0.1554 | 1.94e-02 | 2.2654 | C | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_044849.md` ([json](results/model_comparison/compare_ladder_20260819_044850.json)) |
| 10 | 0.1010 | 0.2114 | 2.11e-02 | 2.0937 | D | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_044849.md` ([json](results/model_comparison/compare_ladder_20260819_044850.json)) |
| 12 | 0.1353 | 0.2677 | 2.23e-02 | 1.9791 | D | ST | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 | `eval_ladder_20260819_044849.md` ([json](results/model_comparison/compare_ladder_20260819_044850.json)) |
| 16 | 0.2028 | 0.4028 | 2.52e-02 | 1.9855 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_153043.md` ([json](results/experiments/exp_model_comparison/tfim_bond_resolved/ladder/run_20260819_152655.json)) |
| 18 | 0.2751 | 0.4937 | 2.74e-02 | 1.7950 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 20 | 1.4809 | 0.4652 | 2.33e-02 | 0.3142 | F | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 22 | 2.1405 | 0.6113 | 2.78e-02 | 0.2856 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 24 | 2.5648 | 0.6715 | 2.80e-02 | 0.2618 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_ladder_MT_20260819_095421.md` ([json](results/model_comparison/compare_ladder_20260819_095421.json)) |
| 26 | 2.6094 | 0.6306 | 2.43e-02 | 0.2417 | F | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 30 | 3.5093 | 0.7350 | 2.45e-02 | 0.2094 | F | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |
| 40 | 6.1568 | 0.9671 | 2.42e-02 | 0.1571 | F | ST | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 | `eval_ladder_20260817_164857.md` |


## square

**h used**: varies (2.500 – 2.570)

| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-------:|-----:|------:|----:|:-----:|:-----:|-----------|------|--------|
| 4 | 0.0232 | 0.0704 | 1.76e-02 | 3.0377 | A | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 6 | 0.0459 | 0.1166 | 1.94e-02 | 2.5418 | B | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_092102.md` ([json](results/model_comparison/compare_square_20260819_092102.json)) |
| 8 | 0.0771 | 0.1708 | 2.13e-02 | 2.2139 | C | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 | `eval_square_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 10 | 0.1141 | 0.2487 | 2.49e-02 | 2.1796 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_153202.md` ([json](results/experiments/exp_model_comparison/tfim_bond_resolved/square/run_20260819_152900.json)) |
| 12 | 0.2301 | 0.3840 | 3.20e-02 | 1.6687 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_050014.md` ([json](results/model_comparison/compare_square_20260819_050014.json)) |
| 14 | 0.2996 | 0.4655 | 3.33e-02 | 1.5538 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 16 | 0.3858 | 0.5660 | 3.54e-02 | 1.4672 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_153202.md` ([json](results/experiments/exp_model_comparison/tfim_bond_resolved/square/run_20260819_152900.json)) |
| 18 | 0.5443 | 0.6918 | 3.84e-02 | 1.2709 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 20 | 2.2948 | 0.7209 | 3.60e-02 | 0.3142 | F | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |
| 21 | 2.8952 | 0.8662 | 4.12e-02 | 0.2992 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_square_MT_20260819_092417.md` ([json](results/model_comparison/compare_square_20260819_092417.json)) |
| 30 | 6.2054 | 1.2997 | 4.33e-02 | 0.2094 | F | ST | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 | `eval_square_20260817_164925.md` |


## triangular

**h used**: 2.500

| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | Model | Checkpoint | Date | Source |
|--:|-------:|-----:|------:|----:|:-----:|:-----:|-----------|------|--------|
| 3 | 0.0318 | 0.0990 | 3.30e-02 | 3.1139 | B | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092425.md` ([json](results/model_comparison/compare_triangular_20260819_092425.json)) |
| 4 | 0.0815 | 0.1691 | 4.23e-02 | 2.0755 | C | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 6 | 0.2246 | 0.3334 | 5.56e-02 | 1.4844 | D | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_050021.md` ([json](results/model_comparison/compare_triangular_20260819_050021.json)) |
| 8 | 1.5770 | 1.0689 | 1.34e-01 | 0.6778 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 10 | 2.1972 | 1.1853 | 1.19e-01 | 0.5395 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_145755.md` |
| 11 | 5.4112 | 1.8466 | 1.68e-01 | 0.3413 | F | MT | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 | `eval_triangular_MT_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 12 | 11.5427 | 2.4575 | 2.05e-01 | 0.2129 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 13 | 23.3216 | 3.0778 | 2.37e-01 | 0.1320 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_092529.md` ([json](results/model_comparison/compare_chain_1d_20260819_092053.json)) |
| 16 | 170.9028 | 5.2136 | 3.26e-01 | 0.0305 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 | `eval_triangular_20260819_160246.md` |
| 24 | 67.1886 | 17.5899 | 7.33e-01 | 0.2618 | F | ST | unified_tfim_br_triangular_multiN_3+4... | 2026-08-18 | `eval_triangular_20260818_130854.md` |


---

*Auto-generated by `scripts/analysis/generate_best_results_scoreboard.py`*
*Data sources: `results/extrapolation_evals/` + `results/model_comparison/`*
---

## Cross-Validation (vs ModelRegistryDB)

| Issue | Detail |
|---|---|
| ⚠️ | chain_1d N=20: scoreboard grade=A but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | chain_1d N=30: scoreboard grade=B but zoo pass_rate_by_n[30]=0% — possible stale zoo data |
| ⚠️ | chain_1d N=80: scoreboard ΔE/gap@h=2.5 = 0.536 >> registry mean ΔE/gap = 0.162 — h=2.5 is anomalously hard for this config |
| ⚠️ | heavy_hex N=4: scoreboard grade=B but zoo pass_rate_by_n[4]=0% — possible stale zoo data |
| ⚠️ | ladder N=20: scoreboard ΔE/gap@h=2.5 = 1.481 >> registry mean ΔE/gap = 0.309 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=26: scoreboard ΔE/gap@h=2.5 = 2.609 >> registry mean ΔE/gap = 0.309 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=30: scoreboard ΔE/gap@h=2.5 = 3.509 >> registry mean ΔE/gap = 0.309 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder N=40: scoreboard ΔE/gap@h=2.5 = 6.157 >> registry mean ΔE/gap = 0.309 — h=2.5 is anomalously hard for this config |
| ⚠️ | square N=20: scoreboard ΔE/gap@h=2.5 = 2.295 >> registry mean ΔE/gap = 0.441 — h=2.5 is anomalously hard for this config |
| ⚠️ | square N=30: scoreboard ΔE/gap@h=2.5 = 6.205 >> registry mean ΔE/gap = 0.441 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular N=16: scoreboard ΔE/gap@h=2.5 = 170.903 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular N=24: scoreboard ΔE/gap@h=2.5 = 67.189 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
