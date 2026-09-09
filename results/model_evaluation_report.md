# Zoo Model Evaluation Report

**Generated**: 2026-09-09 14:19 UTC
**Elapsed**: 0.4s
**Models evaluated**: 20

---

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

*No evaluation data available.*

## triangular — `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

*No evaluation data available.*

## ladder — `unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26+40_p1.pt`

*No evaluation data available.*

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt`

*No evaluation data available.*

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v5.pt`

*No evaluation data available.*

## chain_1d — `unifMPNN__chain_1d_p1_h_0p5_1p5.pt`

*No evaluation data available.*

## multi_topology — `unifMPNN__MT_p1_res_film_base.pt`

*No evaluation data available.*

## chain_1d — `unified_tfim_br_chain_1d_multiN_4_p2.pt`

*No evaluation data available.*

## chain_1d — `unifMPNN__chain_1d_p1_signinv_v2.pt`

*No evaluation data available.*

## chain_1d — `unifMPNN__chain_1d_p1_signinv_fid_v1.pt`

*No evaluation data available.*

## chain_1d — `unifMPNN__chain_1d_p1_signinv_fid_dot3_v1.pt`

*No evaluation data available.*

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt`

*No evaluation data available.*

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1_v4.pt`

*No evaluation data available.*

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt`

*No evaluation data available.*

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18+20+21+26+30+40_p1.pt`

*No evaluation data available.*

## heavy_hex — `unified_multiN_heavyhex_p1.pt`

*No evaluation data available.*

## ladder — `unified_tfim_br_ladder_fromMT_4+6+8+10+12+14+20+26+30_p1.pt`

*No evaluation data available.*

## ladder — `unified_tfim_br_ladder_multiN_4+6+8+10+12+14+20+26+30_p1_v4.pt`

*No evaluation data available.*

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4.pt`

*No evaluation data available.*

## heavy_hex — `unifMPNN__heavy_hex_p1_signinv_fid_v1.pt`

*No evaluation data available.*

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | — | — | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | — | — | F (failing) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26 | — | — | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | — | — | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | — | — | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_h_0p5_1p5.pt | — | — | F (failing) |
| multi_topology | unifMPNN__MT_p1_res_film_base.pt | — | — | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_4_p2.pt | — | — | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_signinv_v2.pt | — | — | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_signinv_fid_v1.pt | — | — | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_signinv_fid_dot3_v1.pt | — | — | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | — | — | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | — | — | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p | — | — | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18 | — | — | F (failing) |
| heavy_hex | unified_multiN_heavyhex_p1.pt | — | — | F (failing) |
| ladder | unified_tfim_br_ladder_fromMT_4+6+8+10+12+14+20+26 | — | — | F (failing) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+14+20+26 | — | — | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4 | — | — | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_signinv_fid_v1.pt | — | — | F (failing) |

---

# MT vs ST Head-to-Head Comparison

**Generated**: 2026-09-09 14:19 UTC
**Score**: MT **7** — ST **1** — Ties **7**
**MT avg quality_score**: 0.091 | **ST avg quality_score**: 0.079

## Per-Topology Summary

| Topology | MT score | ST score | Winner | Δ | MT wins | ST wins |
|----------|:--------:|:--------:|:------:|:-:|:-------:|:-------:|
| chain_1d | 0.202 | 0.263 | 🔴 ST | -0.061 | 2 | 1 |
| heavy_hex | 0.126 | 0.053 | 🟢 MT | +0.073 | 3 | 0 |
| ladder | 0.074 | 0.055 | ⚪ tie | +0.019 | 0 | 0 |
| square | 0.059 | 0.000 | 🟢 MT | +0.059 | 2 | 0 |
| triangular | 0.008 | 0.017 | ⚪ tie | -0.009 | 0 | 0 |

## Per-N Breakdown

| Topology | N | MT score | MT ΔE/gap | MT grade | ST score | ST ΔE/gap | ST grade | Winner |
|----------|:-:|:--------:|:---------:|:--------:|:--------:|:---------:|:--------:|:------:|
| chain_1d | 10 | 0.271 | 9.0% | D | 0.686 | 3.8% | B | ❌ ST |
| chain_1d | 16 | 0.179 | 14.9% | F | 0.068 | 23.5% | F | ✅ MT |
| chain_1d | 20 | 0.158 | 18.7% | F | 0.034 | 33.2% | F | ✅ MT |
| heavy_hex | 10 | 0.164 | 13.3% | F | 0.077 | 18.7% | F | ✅ MT |
| heavy_hex | 16 | 0.122 | 20.9% | F | 0.055 | 29.3% | F | ✅ MT |
| heavy_hex | 20 | 0.093 | 83.5% | F | 0.026 | 161.6% | F | ✅ MT |
| ladder | 10 | 0.091 | 37.9% | F | 0.068 | 48.0% | F | — tie |
| ladder | 16 | 0.068 | 181.1% | F | 0.050 | 230.1% | F | — tie |
| ladder | 20 | 0.064 | 162.5% | F | 0.047 | 203.9% | F | — tie |
| square | 10 | 0.085 | 41.1% | F | 0.000 | — | — | ✅ MT |
| square | 16 | 0.033 | — | F | 0.000 | — | — | ✅ MT |
| triangular | 8 | 0.020 | 42.2% | F | 0.029 | 40.9% | F | — tie |
| triangular | 11 | 0.006 | 129.7% | F | 0.018 | 116.0% | F | — tie |
| triangular | 12 | 0.003 | 281.2% | F | 0.012 | 234.7% | F | — tie |
| triangular | 13 | 0.002 | 573.8% | F | 0.009 | 459.7% | F | — tie |

---
*Auto-generated from model_comparison/ (15 comparisons)*
*Decision metric: quality_score (continuous 0-1, sigmoid-based on mean ΔE/gap + P90 )*

*Generated by `scripts/analysis/evaluate_zoo_models.py`*