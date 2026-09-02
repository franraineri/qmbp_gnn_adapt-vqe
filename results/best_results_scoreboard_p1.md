# Best Results Scoreboard — p=1

**Updated**: 2026-09-02 20:23 UTC
**p_layers**: 1
**Reference h-value**: 2.50 (hardest region near h_critical; actual h used noted per entry)
**Reports scanned**: 341
**Criterion**: Best ΔE/gap achieved at h≈2.5 per (topology × N)

> This report shows the **best single-point result ever achieved** at h≈2.5 for each (topology, N) combination at **p=1**, in the **extrapolation regime** (N values tested with MPNN zero-shot prediction).
> Each p has its own scoreboard file — p=1 and p=2 are different ansätze and never compete against each other.
> It does NOT average over h — it tracks the hardest operating point near h_critical.
> For in-distribution quality (training N), see `model_evaluation_report.md`.
> Grade thresholds: A (|ΔE|<0.05), B (<0.10), C (<0.30), D (<1.00), F (≥1.00).

---

## Summary: Best Grade per Topology

| Topology | Max N evaluated | Best grade | Mean |ΔE| (any N) | N trained up to |
|---|---|---|---|---|
| chain_1d | 80 | B | 0.1882 | 80 |
| heavy_hex | 60 | A | 0.1242 | 60 |
| ladder | 40 | B | 0.2451 | 40 |
| square | 30 | B | 0.1588 | 30 |
| triangular | 24 | B | 0.1164 | 16 |

---

## chain_1d

**h used**: varies (2.420 – 2.500)

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 4 | 0.1882 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 |
| 6 | 0.2556 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 8 | 0.3724 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 10 | 0.2782 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 12 | 0.4283 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 14 | 1.0839 | N/A | D | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-21 |
| 16 | 0.6622 | 0.9664 | D | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-28 |
| 20 | 0.4397 | N/A | B | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 |
| 21 | 0.4115 | N/A | D | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 26 | 0.6689 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 30 | 0.4355 | N/A | C | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 |
| 31 | 0.8269 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 40 | 0.3327 | N/A | C | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 |
| 60 | 0.5058 | N/A | C | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-17 |
| 80 | 1.6095 | N/A | F | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-20 |


## heavy_hex

**h used**: varies (2.500 – 2.630)

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 4 | 0.1242 | N/A | A | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-21 |
| 6 | 0.4185 | N/A | B | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 |
| 8 | 0.3984 | N/A | B | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 |
| 10 | 0.2650 | N/A | A | unified_tfim_br_heavy_hex_fromMT_4+6+... | 2026-08-20 |
| 12 | 0.8065 | N/A | C | unified_tfim_br_MT_residual+film_p1.pt | 2026-08-21 |
| 14 | 0.3229 | N/A | B | data/model_zoo/checkpoints/unified_tf... | 2026-08-23 |
| 16 | 4.4818 | N/A | B | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 |
| 18 | 1.5585 | N/A | F | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 |
| 20 | 7.1400 | N/A | B | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 |
| 21 | 1.2991 | N/A | F | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 |
| 22 | 1.0425 | N/A | F | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 |
| 24 | 0.3619 | N/A | C | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 |
| 26 | 0.3714 | N/A | B | data/model_zoo/checkpoints/unified_mu... | 2026-08-23 |
| 29 | 1.4806 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 30 | 11.9061 | N/A | C | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 |
| 32 | 1.9998 | N/A | F | data/model_zoo/checkpoints/unified_tf... | 2026-08-24 |
| 40 | 2.1640 | N/A | C | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-17 |
| 50 | 4.3070 | N/A | C | unified_tfim_br_heavy_hex_multiN_4+6+... | 2026-08-24 |
| 60 | 7.2024 | N/A | C | data/model_zoo/checkpoints/unified_mu... | 2026-08-24 |


## ladder

**h used**: 2.500

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 4 | 0.3460 | N/A | C | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 6 | 0.2451 | N/A | B | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 |
| 8 | 0.3219 | N/A | C | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 |
| 10 | 0.3560 | N/A | C | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 |
| 12 | 0.4844 | N/A | C | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-19 |
| 14 | 0.5984 | N/A | D | unified_tfim_br_ladder_fromMT_4+6+8+1... | 2026-08-21 |
| 16 | 0.5207 | N/A | D | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-21 |
| 18 | 1.0305 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 20 | 0.6031 | N/A | D | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 |
| 22 | 1.2797 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 24 | 1.4056 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 26 | 0.8338 | N/A | D | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 |
| 30 | 0.9997 | N/A | D | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 |
| 40 | 1.3292 | N/A | D | unified_tfim_br_ladder_multiN_4+6+8+1... | 2026-08-17 |


## square

**h used**: varies (2.500 – 2.570)

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 4 | 0.1588 | N/A | B | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 |
| 6 | 0.1874 | N/A | C | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 |
| 8 | 0.2418 | N/A | C | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-19 |
| 10 | 0.5461 | N/A | C | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 12 | 1.0388 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 14 | 1.0361 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 16 | 0.9049 | N/A | D | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 |
| 18 | 1.7009 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 20 | 1.2679 | N/A | D | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 |
| 21 | 1.9818 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 30 | 2.5911 | N/A | F | unified_tfim_br_square_multiN_4+6+8+1... | 2026-08-17 |


## triangular

**h used**: 2.500

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 3 | 0.1164 | N/A | B | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 4 | 0.3469 | N/A | C | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 6 | 0.5589 | N/A | D | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 8 | 1.3773 | N/A | F | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 10 | 1.4567 | N/A | F | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 11 | 2.6501 | N/A | F | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
| 12 | 3.2522 | N/A | F | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 |
| 13 | 4.4694 | N/A | F | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 |
| 14 | 3.8323 | N/A | F | unified_tfim_br_triangular_multiN_3+4... | 2026-08-21 |
| 16 | 6.8781 | N/A | F | unified_tfim_br_triangular_multiN_3+4... | 2026-08-19 |
| 24 | 17.5899 | N/A | F | unified_tfim_br_triangular_multiN_3+4... | 2026-08-18 |


---

*Auto-generated by `scripts/analysis/generate_best_results_scoreboard.py`*
*Data sources: `results/extrapolation_evals/` + `results/model_comparison/`*
---

## Cross-Validation (vs ModelRegistryDB)

| Issue | Detail |
|---|---|
| ⚠️ | chain_1d p=1 N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | heavy_hex p=1 N=4: scoreboard grade=A but zoo pass_rate_by_n[4]=0% — possible stale zoo data |
| ⚠️ | heavy_hex p=1 N=16: scoreboard grade=B but zoo pass_rate_by_n[16]=0% — possible stale zoo data |
| ⚠️ | heavy_hex p=1 N=20: scoreboard grade=B but zoo pass_rate_by_n[20]=0% — possible stale zoo data |
| ⚠️ | ladder p=1 N=20: scoreboard |ΔE|@h=2.5 = 0.465 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=26: scoreboard |ΔE|@h=2.5 = 0.631 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=30: scoreboard |ΔE|@h=2.5 = 0.735 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | ladder p=1 N=40: scoreboard |ΔE|@h=2.5 = 0.967 >> registry mean ΔE/gap = 0.172 — h=2.5 is anomalously hard for this config |
| ⚠️ | square p=1 N=30: scoreboard |ΔE|@h=2.5 = 1.300 >> registry mean ΔE/gap = 1.828 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular p=1 N=16: scoreboard |ΔE|@h=2.5 = 5.214 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |
| ⚠️ | triangular p=1 N=24: scoreboard |ΔE|@h=2.5 = 17.590 >> registry mean ΔE/gap = 7.155 — h=2.5 is anomalously hard for this config |


---

## chain_1d near h_critical (h≈1.0)

Best-ever single-point result for **chain_1d** at h≈1.0 (±0.15), the critical region where the gap is smallest and the ansatz is most stressed. Same ranking as the main scoreboard (lowest |ΔE| per N).

**h used**: 1.130

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 10 | 1.8609 | 0.8785 | D | unified_tfim_bond_resolved_chain_1d_n... | 2026-08-28 |
| 20 | 5.4470 | N/A | D | unifMPNN__chain_1d_p1_h_0p5_1p5 | 2026-08-28 |


---

## chain_1d near h_critical (h≈1.5)

Best-ever single-point result for **chain_1d** at h≈1.5 (±0.15), the critical region where the gap is smallest and the ansatz is most stressed. Same ranking as the main scoreboard (lowest |ΔE| per N).

**h used**: varies (1.500 – 1.610)

| N | mean |ΔE| | Fidelity | Grade | Checkpoint | Date |
|--:|--------:|:--------:|:-----:|-----------|------|
| 10 | 1.2886 | 0.9733 | C | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-28 |
| 16 | 0.8654 | N/A | D | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 20 | 4.2847 | N/A | D | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 26 | 1.2528 | N/A | F | unified_tfim_br_chain_1d_multiN_6+8+1... | 2026-08-19 |
| 30 | 1.7332 | N/A | F | unified_tfim_br_multitopo_chain_1d+he... | 2026-08-19 |
