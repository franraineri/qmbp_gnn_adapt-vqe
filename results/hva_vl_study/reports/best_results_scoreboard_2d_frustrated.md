# Best-results scoreboard — square tfim_frustrated (J2=0.5)

Best run per configuration `(method, N, h, variant, loader)`. Auto-generated — do not edit by hand. Entries update only when a new run's fidelity improves on the stored one.

- Schema: `state_prep_scoreboard_v1` — 122 entries.

| N | h | method | variant | loader/χ | F | \|ΔE\| | 2q | total | depth | converged | updated |
|---|------|--------|---------|----------|------:|-------:|---:|------:|------:|-----------|---------|
| 6 | 0.50 | HVA | p2_base | — | 0.9957 | 0.0140 | 52 | — | — | 1/2 | 2026-09-29 |
| 6 | 0.50 | HVA | nnn p=1 (fair-conv best) | — | 0.9088 | 0.1998 | 26 | 87 | 29 | 6/6 seeds | 2026-09-29 |
| 6 | 0.50 | HVA | p1_base | — | 0.6967 | 0.3047 | 26 | — | — | 2/2 | 2026-09-29 |
| 8 | 0.30 | HVA | p2_full_ref (nnn=11) | — | 0.9892 | 0.0162 | 84 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 8 | 0.30 | HVA | p2_topk_regime (nnn=6) | — | 0.9881 | 0.0182 | 64 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 8 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9984 | 0.0068 | 146 | — | — | masked | 2026-10-02 |
| 8 | 0.50 | HVA | p3_half_nn_rx_topk6of11 (nnn=6) | — | 0.9973 | 0.0097 | 116 | — | — | masked | 2026-10-02 |
| 8 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9913 | 0.0261 | 104 | — | — | masked | 2026-10-02 |
| 8 | 0.50 | HVA | p2_half_nn_rx_topk6of11 (nnn=6) | — | 0.9898 | 0.0300 | 84 | — | — | masked | 2026-10-02 |
| 8 | 0.50 | HVA | p2_topk_regime (nnn=6) | — | 0.9638 | 0.0852 | 64 | — | — | second_order+descent | 2026-10-01 |
| 8 | 0.50 | HVA | p2_full_ref (nnn=11) | — | 0.9628 | 0.0738 | 84 | — | — | calibrated+descent | 2026-10-01 |
| 8 | 0.50 | HVA | p1_half_nn_rx_full | — | 0.9459 | 0.1043 | 62 | — | — | masked | 2026-10-02 |
| 8 | 0.50 | HVA | p1_half_nn_rx_topk6of11 (nnn=6) | — | 0.9445 | 0.1083 | 52 | — | — | masked | 2026-10-02 |
| 8 | 1.30 | HVA | p2_full_ref (nnn=11) | — | 0.9986 | 0.0114 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 8 | 1.30 | HVA | p2_topk_regime (nnn=6) | — | 0.9914 | 0.0424 | 64 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 9 | 0.50 | HVA | nnn p=2 (ceiling) | — | 0.9574 | 0.1281 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-09-29 |
| 9 | 0.50 | HVA | nnn p=2 | — | 0.9523 | 0.1318 | 104 | 273 | 73 | n/a (random best-of) | 2026-09-29 |
| 9 | 0.50 | HVA | nn p=2 | — | 0.7696 | 0.2975 | 48 | 189 | 49 | n/a (random best-of) | 2026-09-29 |
| 9 | 0.50 | HVA | nnn p=1 | — | 0.7358 | 0.5378 | 52 | 150 | 41 | n/a (random best-of) | 2026-09-29 |
| 9 | 0.50 | HVA | nn p=1 | — | 0.7253 | 0.6316 | 24 | 108 | 32 | n/a (random best-of) | 2026-09-29 |
| 9 | 0.50 | VL | L8/F50 | vector | 0.9431 | 0.2750 | 120 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | L8/F20 | vector | 0.9200 | 0.4033 | 120 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | L4/F50 | vector | 0.9085 | 0.4210 | 60 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | L4/F20 | vector | 0.8844 | 0.5229 | 60 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | L2/F50 | vector | 0.8359 | 0.6290 | 30 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | L2/F20 | vector | 0.8093 | 0.7165 | 30 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.50 | VL | default L2/F20 | vector | 0.8086 | 0.7165 | 30 | 207 | 53 | n/a (fixed circuit) | 2026-09-29 |
| 9 | 0.70 | HVA | p2_full_ref (nnn=14) | — | 0.9856 | 0.0647 | 104 | — | — | second_order+descent | 2026-10-01 |
| 9 | 0.70 | HVA | p2_topk_regime (nnn=7) | — | 0.9759 | 0.1081 | 76 | — | — | second_order+descent | 2026-10-01 |
| 9 | 1.00 | HVA | nnn p=2 (ceiling) | — | 0.9964 | 0.0263 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-09-29 |
| 9 | 1.00 | HVA | nnn p=2 | — | 0.9954 | 0.0302 | 104 | 273 | 73 | n/a (random best-of) | 2026-09-29 |
| 9 | 1.00 | HVA | nn p=2 | — | 0.9393 | 0.1830 | 48 | 189 | 49 | n/a (random best-of) | 2026-09-29 |
| 9 | 1.00 | HVA | nnn p=1 | — | 0.9332 | 0.2515 | 52 | 150 | 41 | n/a (random best-of) | 2026-09-29 |
| 9 | 1.00 | HVA | nn p=1 | — | 0.9144 | 0.3060 | 24 | 108 | 32 | n/a (random best-of) | 2026-09-29 |
| 9 | 1.00 | VL | L8/F50 | vector | 0.9754 | 0.1981 | 120 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | L4/F50 | vector | 0.9665 | 0.2432 | 60 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | L8/F20 | vector | 0.9656 | 0.2599 | 120 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | L4/F20 | vector | 0.9547 | 0.3093 | 60 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | L2/F50 | vector | 0.9082 | 0.4741 | 30 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | default L2/F20 | vector | 0.8999 | 0.5399 | 30 | 215 | 53 | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.00 | VL | L2/F20 | vector | 0.8999 | 0.5399 | 30 | — | — | n/a (fixed circuit) | 2026-09-29 |
| 9 | 1.80 | HVA | p2_full_ref (nnn=14) | — | 0.9974 | 0.0216 | 104 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 9 | 1.80 | HVA | p2_topk_regime (nnn=7) | — | 0.9904 | 0.0643 | 76 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 9 | 2.00 | HVA | nnn p=2 (ceiling) | — | 0.9997 | 0.0043 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-09-29 |
| 9 | 2.00 | VL | default L2/F20 | vector | 0.9613 | 0.3887 | 30 | 216 | 53 | n/a (fixed circuit) | 2026-09-29 |
| 9 | 3.00 | HVA | nnn p=2 (ceiling) | — | 1.0000 | 0.0010 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-09-29 |
| 9 | 3.00 | VL | default L2/F20 | vector | 0.9870 | 0.1721 | 30 | 216 | 53 | n/a (fixed circuit) | 2026-09-29 |
| 10 | 0.30 | HVA | p2_full_ref (nnn=15) | — | 0.9996 | 0.0014 | 112 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 10 | 0.30 | HVA | p2_topk_regime (nnn=5) | — | 0.9995 | 0.0020 | 72 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 10 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9984 | 0.0045 | 194 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p3_half_nn_rx_topk8of15 (nnn=8) | — | 0.9977 | 0.0057 | 152 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p3_base | — | 0.9963 | 0.0084 | 168 | — | — | 2/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9957 | 0.0107 | 138 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_half_nn_rx_topk8of15 (nnn=8) | — | 0.9945 | 0.0137 | 110 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_base | — | 0.9917 | 0.0189 | 112 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p2_full_ref (nnn=15) | — | 0.9898 | 0.0202 | 112 | — | — | second_order+descent | 2026-10-01 |
| 10 | 0.50 | HVA | p2_full_ref | — | 0.9893 | 0.0221 | 112 | — | — | 1/2 | 2026-10-01 |
| 10 | 0.50 | HVA | p1_half_nn_rx | — | 0.9891 | 0.0234 | 82 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p1_half_nn_rx_full | — | 0.9891 | 0.0234 | 82 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk11of15 (nnn=11) | — | 0.9890 | 0.0235 | 74 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_base_full | — | 0.9890 | 0.0218 | 112 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk8of15 (nnn=8) | — | 0.9889 | 0.0238 | 68 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_topk_regime (nnn=8) | — | 0.9889 | 0.0226 | 84 | — | — | second_order+descent | 2026-10-01 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk5of15 (nnn=5) | — | 0.9880 | 0.0260 | 62 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p1_half_nn | — | 0.9873 | 0.0267 | 82 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p2_topk_f0.7 (nnn=10) | — | 0.9865 | — | 92 | — | — | masked | 2026-10-01 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk5of15_nn10of13 (nnn=5) | — | 0.9862 | 0.0307 | 50 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_topk_f0.5 (nnn=8) | — | 0.9861 | — | 84 | — | — | masked | 2026-10-01 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk3of15_nn10of13 (nnn=3) | — | 0.9843 | 0.0326 | 46 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_base_topk4of15_nn10of13 (nnn=4) | — | 0.9829 | 0.0344 | 56 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p2_base_topk3of15_nn10of13 (nnn=3) | — | 0.9826 | 0.0349 | 52 | — | — | masked | 2026-10-02 |
| 10 | 0.50 | HVA | p1_half_nnn | — | 0.9740 | 0.0447 | 86 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p2_topk_f0.3 (nnn=4) | — | 0.9670 | — | 68 | — | — | masked | 2026-10-01 |
| 10 | 0.50 | HVA | p1_rx_extra | — | 0.4784 | 1.2048 | 56 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p1_rz_extra | — | 0.4191 | 1.2327 | 56 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p1_base | — | 0.4089 | 1.3023 | 56 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p1_interleaved | — | 0.3450 | 1.3632 | 56 | — | — | 4/4 | 2026-09-29 |
| 10 | 0.50 | HVA | p1_pruned (nnn=15) | — | 0.2672 | 1.4632 | 56 | — | — | 2/2 | 2026-10-01 |
| 10 | 0.50 | HVA | p1_topk_nnn (nnn=8) | — | 0.0060 | 2.1724 | 42 | — | — | 2/2 | 2026-10-01 |
| 10 | 0.70 | HVA | p2_full_ref (nnn=15) | — | 0.9400 | 0.1902 | 112 | — | — | second_order+descent | 2026-10-01 |
| 10 | 0.70 | HVA | p2_topk_regime (nnn=8) | — | 0.8948 | 0.2018 | 84 | — | — | second_order+descent | 2026-10-01 |
| 10 | 1.00 | HVA | p2_full_ref | — | 0.9621 | 0.1274 | 112 | — | — | 2/2 | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step6 (25 bonds) | — | 0.8723 | 0.4223 | 50 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step5 (23 bonds) | — | 0.8654 | 0.4192 | 46 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step3 (19 bonds) | — | 0.8615 | 0.4541 | 38 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step2 (17 bonds) | — | 0.8421 | 0.4622 | 34 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | p1_topk_nnn (nnn=8) | — | 0.8356 | 0.4734 | 42 | — | — | 2/2 | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step4 (21 bonds) | — | 0.8348 | 0.5984 | 42 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step1 (15 bonds) | — | 0.8059 | 0.6832 | 30 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step0 (13 bonds) | — | 0.7715 | 0.7945 | 26 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | adapt step7 (27 bonds) | — | 0.5956 | 2.4739 | 54 | — | — | masked | 2026-10-01 |
| 10 | 1.00 | HVA | p1_pruned (nnn=15) | — | 0.5956 | 2.4739 | 56 | — | — | 2/2 | 2026-10-01 |
| 10 | 1.00 | HVA | p1_full_ref (nnn=15) | — | 0.5956 | 2.4739 | 56 | — | — | 2/2 | 2026-10-01 |
| 10 | 1.00 | VL | default L2/F20 | vector | 0.8966 | 0.6180 | 34 | 237 | 53 | n/a (fixed circuit) | 2026-09-29 |
| 10 | 1.30 | HVA | p2_full_ref (nnn=15) | — | 0.9971 | 0.0208 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 10 | 1.30 | HVA | p2_topk_regime (nnn=8) | — | 0.9931 | 0.0394 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 10 | 1.80 | HVA | p2_full_ref (nnn=15) | — | 0.9985 | 0.0136 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 10 | 1.80 | HVA | p2_topk_regime (nnn=8) | — | 0.9957 | 0.0333 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 12 | 0.30 | HVA | p2_full_ref (nnn=22) | — | 0.9992 | 0.0024 | 156 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 12 | 0.30 | HVA | p2_topk_regime (nnn=7) | — | 0.9987 | 0.0039 | 96 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-01 |
| 12 | 0.50 | HVA | p2_full_ref (nnn=22) | — | 0.9813 | 0.0314 | 156 | — | — | second_order+descent | 2026-10-01 |
| 12 | 0.50 | HVA | p2_topk_regime (nnn=11) | — | 0.9756 | 0.0409 | 112 | — | — | second_order+descent | 2026-10-01 |
| 12 | 1.30 | HVA | p2_full_ref (nnn=22) | — | 0.9758 | 0.1185 | 156 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 12 | 1.30 | HVA | p2_topk_regime (nnn=11) | — | 0.9654 | 0.1517 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-01 |
| 14 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9501 | — | 228 | — | — | masked | 2026-10-02 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk20of27 (nnn=20) | — | 0.9491 | 0.0666 | 200 | — | — | masked | 2026-10-02 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk14of27 (nnn=14) | — | 0.9482 | 0.0696 | 176 | — | — | masked | 2026-10-02 |
| 14 | 0.50 | HVA | p2_full_ref (nnn=27) | — | 0.9333 | 0.0859 | 188 | — | — | transfer<crossN12@0.981>+descent | 2026-10-01 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.15_nn11_nnn8 (nnn=8) | — | 0.8968 | 0.1333 | 98 | — | — | masked | 2026-10-02 |
| 14 | 0.50 | HVA | p2_topk_regime (nnn=14) | — | 0.4022 | 0.3239 | 136 | — | — | second_order+descent | 2026-10-01 |
| 18 | 0.50 | HVA | p2_half_nn_rx | — | 0.9268 | 0.0846 | 318 | — | — | 0/1 | 2026-09-30 |
| 18 | 0.50 | HVA | nnn p=3 (fair-conv best) | — | 0.9241 | 0.0819 | 396 | 918 | 147 | 2/6 seeds | 2026-09-29 |
| 18 | 0.50 | HVA | p3_base | — | 0.9132 | 0.0937 | 396 | — | — | 0/1 | 2026-10-01 |
| 18 | 0.50 | HVA | p2_half_nn | — | 0.9124 | 0.0919 | 318 | — | — | 0/1 | 2026-09-30 |
| 18 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9102 | 0.1056 | 318 | — | — | masked | 2026-10-01 |
| 18 | 0.50 | HVA | p2_topk_regime (nnn=20) | — | 0.6703 | 0.2543 | 188 | — | — | second_order | 2026-10-01 |
| 18 | 0.50 | HVA | p2_full_ref (nnn=39) | — | 0.6648 | 0.2658 | 264 | — | — | transfer<p3_base@0.913>+descent | 2026-10-01 |
| 18 | 0.50 | HVA | p2_base | — | 0.6552 | 0.3403 | 264 | — | — | 4/4 | 2026-09-29 |
| 18 | 0.50 | HVA | p1_half_nn | — | 0.2851 | 0.9491 | 186 | — | — | 4/4 | 2026-09-29 |
| 18 | 0.50 | VL | L8/F50 | mps χ=64 | 0.8625 | 0.5382 | 264 | 1768 | 181 | n/a (fixed circuit) | 2026-09-29 |
| 18 | 1.00 | HVA | nnn p=2 (fair-conv best) | — | 0.9314 | 0.2608 | 264 | 630 | 106 | 6/6 seeds | 2026-09-29 |
| 18 | 1.00 | VL | L8/F50 | mps χ=64 | 0.7924 | 1.3939 | 264 | 1767 | 181 | n/a (fixed circuit) | 2026-09-29 |

