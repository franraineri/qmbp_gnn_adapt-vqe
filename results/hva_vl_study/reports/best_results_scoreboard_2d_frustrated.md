# Best-results scoreboard — square tfim_frustrated (J2=0.5)

Best run per configuration `(method, N, h, variant, loader)`. Auto-generated — do not edit by hand. Entries update only when a new run's fidelity improves on the stored one.

- Schema: `state_prep_scoreboard_v1` — 248 entries.

| N | h | method | variant | loader/χ | F | \|ΔE\| | 2q | total | depth | converged | updated |
|---|------|--------|---------|----------|------:|-------:|---:|------:|------:|-----------|---------|
| 6 | 0.50 | HVA | p2_half_nn_rx_topk3of6 (nnn=3) | — | 0.9995 | — | 54 | — | — | transfer<full>+descent | 2026-10-15 |
| 6 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9993 | 0.0032 | 66 | — | — | masked | 2026-10-15 |
| 6 | 0.50 | HVA | p2_half_nn_rx_topk2of6 (nnn=2) | — | 0.9992 | — | 50 | — | — | transfer<full>+descent | 2026-10-15 |
| 6 | 0.50 | HVA | p2_half_nn_rx_prune0.1_nn7_nnn2 (nnn=2) | — | 0.9991 | 0.0035 | 50 | — | — | masked | 2026-10-15 |
| 6 | 0.50 | HVA | p2_base | — | 0.9957 | 0.0140 | 52 | — | — | 0/2 | 2026-10-15 |
| 6 | 0.50 | HVA | nnn p=1 (fair-conv best) | — | 0.9088 | 0.1998 | 26 | 87 | 29 | 6/6 seeds | 2026-10-15 |
| 8 | 0.30 | HVA | p2_full_ref (nnn=11) | — | 0.9892 | 0.0162 | 84 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 8 | 0.30 | HVA | p2_topk_regime (nnn=6) | — | 0.9881 | 0.0182 | 64 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 8 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9984 | 0.0068 | 146 | — | — | masked | 2026-10-15 |
| 8 | 0.50 | HVA | p3_half_nn_rx_topk6of11 (nnn=6) | — | 0.9973 | 0.0097 | 116 | — | — | masked | 2026-10-15 |
| 8 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9913 | 0.0261 | 104 | — | — | masked | 2026-10-15 |
| 8 | 0.50 | HVA | p2_half_nn_rx_topk6of11 (nnn=6) | — | 0.9898 | 0.0300 | 84 | — | — | masked | 2026-10-15 |
| 8 | 0.50 | HVA | p2_topk_regime (nnn=6) | — | 0.9638 | 0.0852 | 64 | — | — | second_order+descent | 2026-10-15 |
| 8 | 0.50 | HVA | p2_full_ref (nnn=11) | — | 0.9628 | 0.0738 | 84 | — | — | calibrated+descent | 2026-10-15 |
| 8 | 0.50 | HVA | p1_half_nn_rx_full | — | 0.9459 | 0.1043 | 62 | — | — | masked | 2026-10-15 |
| 8 | 0.50 | HVA | p1_half_nn_rx_topk6of11 (nnn=6) | — | 0.9445 | 0.1083 | 52 | — | — | masked | 2026-10-15 |
| 8 | 1.30 | HVA | p2_full_ref (nnn=11) | — | 0.9986 | 0.0114 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 8 | 1.30 | HVA | p2_topk_regime (nnn=6) | — | 0.9914 | 0.0424 | 64 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 9 | 0.50 | HVA | nnn p=2 (ceiling) | — | 0.9574 | 0.1281 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-10-15 |
| 9 | 0.50 | HVA | nnn p=2 | — | 0.9523 | 0.1318 | 104 | 273 | 73 | n/a (random best-of) | 2026-10-15 |
| 9 | 0.50 | HVA | nn p=2 | — | 0.7696 | 0.2975 | 48 | 189 | 49 | n/a (random best-of) | 2026-10-15 |
| 9 | 0.50 | HVA | nnn p=1 | — | 0.7358 | 0.5378 | 52 | 150 | 41 | n/a (random best-of) | 2026-10-15 |
| 9 | 0.50 | HVA | nn p=1 | — | 0.7253 | 0.6316 | 24 | 108 | 32 | n/a (random best-of) | 2026-10-15 |
| 9 | 0.50 | VL | L8/F50 | vector | 0.9431 | 0.2750 | 120 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | L8/F20 | vector | 0.9200 | 0.4033 | 120 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | L4/F50 | vector | 0.9085 | 0.4210 | 60 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | L4/F20 | vector | 0.8844 | 0.5229 | 60 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | L2/F50 | vector | 0.8359 | 0.6290 | 30 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | L2/F20 | vector | 0.8093 | 0.7165 | 30 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.50 | VL | default L2/F20 | vector | 0.8086 | 0.7165 | 30 | 207 | 53 | n/a (fixed circuit) | 2026-10-15 |
| 9 | 0.70 | HVA | p2_full_ref (nnn=14) | — | 0.9856 | 0.0647 | 104 | — | — | second_order+descent | 2026-10-15 |
| 9 | 0.70 | HVA | p2_topk_regime (nnn=7) | — | 0.9759 | 0.1081 | 76 | — | — | second_order+descent | 2026-10-15 |
| 9 | 1.00 | HVA | nnn p=2 (ceiling) | — | 0.9964 | 0.0263 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-10-15 |
| 9 | 1.00 | HVA | nnn p=2 | — | 0.9954 | 0.0302 | 104 | 273 | 73 | n/a (random best-of) | 2026-10-15 |
| 9 | 1.00 | HVA | nn p=2 | — | 0.9393 | 0.1830 | 48 | 189 | 49 | n/a (random best-of) | 2026-10-15 |
| 9 | 1.00 | HVA | nnn p=1 | — | 0.9332 | 0.2515 | 52 | 150 | 41 | n/a (random best-of) | 2026-10-15 |
| 9 | 1.00 | HVA | nn p=1 | — | 0.9144 | 0.3060 | 24 | 108 | 32 | n/a (random best-of) | 2026-10-15 |
| 9 | 1.00 | VL | L8/F50 | vector | 0.9754 | 0.1981 | 120 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | L4/F50 | vector | 0.9665 | 0.2432 | 60 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | L8/F20 | vector | 0.9656 | 0.2599 | 120 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | L4/F20 | vector | 0.9547 | 0.3093 | 60 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | L2/F50 | vector | 0.9082 | 0.4741 | 30 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | default L2/F20 | vector | 0.8999 | 0.5399 | 30 | 215 | 53 | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.00 | VL | L2/F20 | vector | 0.8999 | 0.5399 | 30 | — | — | n/a (fixed circuit) | 2026-10-15 |
| 9 | 1.80 | HVA | p2_full_ref (nnn=14) | — | 0.9974 | 0.0216 | 104 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 9 | 1.80 | HVA | p2_topk_regime (nnn=7) | — | 0.9904 | 0.0643 | 76 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 9 | 2.00 | HVA | nnn p=2 (ceiling) | — | 0.9997 | 0.0043 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-10-15 |
| 9 | 2.00 | VL | default L2/F20 | vector | 0.9613 | 0.3887 | 30 | 216 | 53 | n/a (fixed circuit) | 2026-10-15 |
| 9 | 3.00 | HVA | nnn p=2 (ceiling) | — | 1.0000 | 0.0010 | 104 | 273 | 73 | ceiling (best-of by energy) | 2026-10-15 |
| 9 | 3.00 | VL | default L2/F20 | vector | 0.9870 | 0.1721 | 30 | 216 | 53 | n/a (fixed circuit) | 2026-10-15 |
| 10 | 0.30 | HVA | p2_full_ref (nnn=15) | — | 0.9996 | 0.0014 | 112 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 10 | 0.30 | HVA | p2_topk_regime (nnn=5) | — | 0.9995 | 0.0020 | 72 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 10 | 0.30 | HVA | ADAPT(square)-maxfid | — | 0.9831 | — | 68 | — | — | adapt | 2026-10-15 |
| 10 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9985 | 0.0039 | 194 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p3_half_nn_rx_topk8of15 (nnn=8) | — | 0.9977 | — | 152 | — | — | transfer<full>+descent | 2026-10-15 |
| 10 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9957 | 0.0107 | 138 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_half_nn_rx_topk8of15 (nnn=8) | — | 0.9945 | 0.0137 | 110 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_full_ref | — | 0.9893 | 0.0221 | 112 | — | — | 1/2 | 2026-10-15 |
| 10 | 0.50 | HVA | p2_full_ref (nnn=15) | — | 0.9892 | — | 112 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p1_half_nn_rx | — | 0.9891 | 0.0234 | 82 | — | — | 4/4 | 2026-10-15 |
| 10 | 0.50 | HVA | p2_base_full | — | 0.9890 | 0.0218 | 112 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p1_half_nn_rx_topk8of15 (nnn=8) | — | 0.9889 | 0.0238 | 68 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p1_half_nn_rx_full | — | 0.9889 | 0.0238 | 82 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p3_half_nn_rx_prune0.2_th_nn9_nnn3 (nnn=3) | — | 0.9868 | 0.0290 | 90 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_topk_f0.7 (nnn=10) | — | 0.9865 | — | 92 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_topk_f0.5 (nnn=8) | — | 0.9861 | — | 84 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_base_topk4of15_nn10of13 (nnn=4) | — | 0.9829 | 0.0344 | 56 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_base_topk3of15_nn10of13 (nnn=3) | — | 0.9826 | 0.0349 | 52 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p2_topk_regime (nnn=8) | — | 0.9793 | 0.0446 | 84 | — | — | transfer<donor<N10h0.70>@0.940>+descent | 2026-10-15 |
| 10 | 0.50 | HVA | ADAPT(square)-maxfid | — | 0.9736 | — | 60 | — | — | adapt | 2026-10-15 |
| 10 | 0.50 | HVA | p2_topk_f0.3 (nnn=4) | — | 0.9670 | — | 68 | — | — | masked | 2026-10-15 |
| 10 | 0.50 | HVA | p1_half_nn_rx_prune0.3_nn7_nnn4 (nnn=4) | — | 0.9598 | — | 36 | — | — | analytic+descent | 2026-10-15 |
| 10 | 0.50 | HVA | p1_pruned (nnn=15) | — | 0.2672 | 1.4632 | 56 | — | — | 2/2 | 2026-10-15 |
| 10 | 0.50 | HVA | p1_topk_nnn (nnn=8) | — | 0.0060 | 2.1724 | 42 | — | — | 2/2 | 2026-10-15 |
| 10 | 0.70 | HVA | ADAPT(square)-maxfid | — | 0.9717 | — | 90 | — | — | adapt | 2026-10-15 |
| 10 | 0.70 | HVA | p2_full_ref (nnn=15) | — | 0.9400 | 0.1902 | 112 | — | — | second_order+descent | 2026-10-15 |
| 10 | 0.70 | HVA | p2_topk_regime (nnn=8) | — | 0.8948 | 0.2018 | 84 | — | — | second_order+descent | 2026-10-15 |
| 10 | 0.90 | HVA | ADAPT(square)-maxfid | — | 0.9566 | — | 68 | — | — | adapt | 2026-10-15 |
| 10 | 1.00 | HVA | p2_full_ref | — | 0.9621 | 0.1274 | 112 | — | — | 2/2 | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step6 (25 bonds) | — | 0.8723 | 0.4223 | 50 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step5 (23 bonds) | — | 0.8654 | 0.4192 | 46 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step3 (19 bonds) | — | 0.8615 | 0.4541 | 38 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step2 (17 bonds) | — | 0.8421 | 0.4622 | 34 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | p1_topk_nnn (nnn=8) | — | 0.8356 | 0.4734 | 42 | — | — | 2/2 | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step4 (21 bonds) | — | 0.8348 | 0.5984 | 42 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step1 (15 bonds) | — | 0.8059 | 0.6832 | 30 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step0 (13 bonds) | — | 0.7715 | 0.7945 | 26 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | adapt step7 (27 bonds) | — | 0.5956 | 2.4739 | 54 | — | — | masked | 2026-10-15 |
| 10 | 1.00 | HVA | p1_pruned (nnn=15) | — | 0.5956 | 2.4739 | 56 | — | — | 2/2 | 2026-10-15 |
| 10 | 1.00 | HVA | p1_full_ref (nnn=15) | — | 0.5956 | 2.4739 | 56 | — | — | 2/2 | 2026-10-15 |
| 10 | 1.00 | VL | default L2/F20 | vector | 0.8966 | 0.6180 | 34 | 237 | 53 | n/a (fixed circuit) | 2026-10-15 |
| 10 | 1.30 | HVA | p2_full_ref (nnn=15) | — | 0.9971 | 0.0208 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 10 | 1.30 | HVA | p2_topk_regime (nnn=8) | — | 0.9931 | 0.0394 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 10 | 1.80 | HVA | p2_full_ref (nnn=15) | — | 0.9985 | 0.0136 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 10 | 1.80 | HVA | p2_topk_regime (nnn=8) | — | 0.9957 | 0.0333 | 84 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 12 | 0.30 | HVA | p2_full_ref (nnn=22) | — | 0.9992 | 0.0024 | 156 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 12 | 0.30 | HVA | p2_topk_regime (nnn=7) | — | 0.9987 | 0.0039 | 96 | — | — | flat_renorm(zz_coef=0.11)+descent | 2026-10-15 |
| 12 | 0.50 | HVA | p2_full_ref (nnn=22) | — | 0.9813 | 0.0314 | 156 | — | — | second_order+descent | 2026-10-15 |
| 12 | 0.50 | HVA | p2_topk_regime (nnn=11) | — | 0.9756 | 0.0409 | 112 | — | — | second_order+descent | 2026-10-15 |
| 12 | 1.30 | HVA | p2_full_ref (nnn=22) | — | 0.9758 | 0.1185 | 156 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 12 | 1.30 | HVA | p2_topk_regime (nnn=11) | — | 0.9654 | 0.1517 | 112 | — | — | so_nn_shrink(0.4)+descent | 2026-10-15 |
| 12 | 3.00 | HVA | prune0.3 (nnn=17) | — | 0.9952 | — | 51 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.30 | HVA | ? (nnn=23) | — | 0.1824 | 1.2981 | 86 | — | — | masked | 2026-10-15 |
| 14 | 0.30 | HVA | ? (nnn=24) | — | 0.1824 | 1.2981 | 88 | — | — | masked | 2026-10-15 |
| 14 | 0.30 | HVA | ? (nnn=22) | — | 0.0816 | 1.3921 | 84 | — | — | masked | 2026-10-15 |
| 14 | 0.30 | HVA | ? (nnn=21) | — | 0.0768 | 1.6395 | 82 | — | — | masked | 2026-10-15 |
| 14 | 0.30 | HVA | ? (nnn=20) | — | 0.0748 | 1.6422 | 80 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | ADAPT(square)-maxfid | — | 0.9619 | — | 200 | — | — | adapt | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.9500 | 0.0653 | 228 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | ADAPT(square)-pareto | — | 0.9397 | — | 160 | — | — | adapt | 2026-10-15 |
| 14 | 0.50 | HVA | p2_base_prune0.1_nn19_nnn25 (nnn=25) | — | 0.9352 | 0.0881 | 176 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_full_ref (nnn=27) | — | 0.9333 | 0.0859 | 188 | — | — | transfer<crossN12@0.981>+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.1_nn12_nnn9 (nnn=9) | — | 0.9235 | 0.1116 | 108 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | prune0.3 (nnn=20) | — | 0.9127 | — | 60 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.15_nn11_nnn8 | — | 0.9043 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_base_prune0.2_nn19_nnn23 (nnn=23) | — | 0.9026 | 0.1744 | 168 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.15_nn11_nnn8 (nnn=8) | — | 0.8967 | — | 98 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.2_nn9_nnn7 (nnn=7) | — | 0.7505 | 0.2308 | 82 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.2_nn9_nnn7 | — | 0.7202 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.3_nn8_nnn6 | — | 0.7080 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.3_nn8_nnn6 (nnn=6) | — | 0.7079 | — | 72 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_prune0.05_nn17_nnn14 (nnn=14) | — | 0.6270 | 0.2656 | 158 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk4of27_nn12of20 (nnn=4) | — | 0.6159 | — | 88 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk4of27_nn12of20 | — | 0.6158 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk5of27_nn12of20 (nnn=5) | — | 0.6148 | — | 92 | — | — | analytic+descent | 2026-10-15 |
| 14 | 0.50 | HVA | p2_half_nn_rx_topk5of27_nn12of20 | — | 0.6138 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_base_prune0.05_nn20_nnn27 (nnn=27) | — | 0.5520 | 0.2522 | 188 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | prune0.3 | — | 0.5103 | — | — | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_base_full | — | 0.4460 | 0.2953 | 188 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | p2_topk_regime (nnn=14) | — | 0.4022 | 0.3239 | 136 | — | — | second_order+descent | 2026-10-15 |
| 14 | 0.50 | HVA | ? (nnn=22) | — | 0.2767 | 1.6624 | 84 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | ? (nnn=21) | — | 0.2743 | 1.6650 | 82 | — | — | masked | 2026-10-15 |
| 14 | 0.50 | HVA | ? (nnn=20) | — | 0.2714 | 1.6680 | 80 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_half_nn_rx_full | — | 0.9476 | 0.1217 | 228 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | ADAPT(square)-maxfid | — | 0.9443 | — | 220 | — | — | adapt | 2026-10-15 |
| 14 | 0.70 | HVA | p2_half_nn_rx_prune0.05_nn20_nnn17 (nnn=17) | — | 0.9442 | 0.1340 | 188 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_half_nn_rx_prune0.1_nn19_nnn12 (nnn=12) | — | 0.8903 | 0.1841 | 162 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_base_full | — | 0.8333 | 0.2349 | 188 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_base_prune0.05_nn19_nnn16 (nnn=16) | — | 0.8129 | 0.2559 | 140 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_base_prune0.1_nn16_nnn7 (nnn=7) | — | 0.7550 | 0.3002 | 92 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_half_nn_rx_prune0.2_nn13_nnn7 (nnn=7) | — | 0.6856 | 0.4918 | 106 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | p2_base_prune0.2_nn9_nnn5 (nnn=5) | — | 0.5793 | 0.9115 | 56 | — | — | masked | 2026-10-15 |
| 14 | 0.70 | HVA | ADAPT(square)-pareto | — | 0.4463 | — | 40 | — | — | adapt | 2026-10-15 |
| 16 | 3.00 | HVA | prune0.3 (nnn=24) | — | 0.9941 | — | 72 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.30 | HVA | ADAPT(square)-maxfid | — | 0.9001 | — | 264 | — | — | adapt | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9516 | 0.0590 | 450 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_topk29of39 (nnn=29) | — | 0.9515 | 0.0630 | 390 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_topk20of39 (nnn=20) | — | 0.9359 | 0.0850 | 336 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_nn_dom2_rx_full | — | 0.9252 | 0.0833 | 372 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | nnn p=3 (fair-conv best) | — | 0.9241 | 0.0819 | 396 | 918 | 147 | 2/6 seeds | 2026-10-15 |
| 18 | 0.50 | HVA | p3_base | — | 0.9132 | 0.0937 | 396 | — | — | 0/1 | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk29of39 (nnn=29) | — | 0.9083 | — | 278 | — | — | transfer<full>+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_prune0.1_nn20_nnn19 (nnn=19) | — | 0.9059 | — | 274 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_prune0.05_nn24_nnn26 (nnn=26) | — | 0.9011 | — | 348 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk20of39 (nnn=20) | — | 0.9002 | — | 242 | — | — | transfer<full>+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_full | — | 0.8968 | 0.1222 | 318 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_topk10of39 (nnn=10) | — | 0.8903 | 0.1444 | 276 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk13of39 (nnn=13) | — | 0.8866 | — | 214 | — | — | transfer<full>+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_nn_dom2_rx_topk20of39 (nnn=20) | — | 0.8717 | 0.1133 | 296 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk20of39_nn21of27 (nnn=20) | — | 0.8551 | — | 206 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk13of39_nn16of27 (nnn=13) | — | 0.8483 | — | 148 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_prune0.15_nn14_nnn15 (nnn=15) | — | 0.8435 | — | 202 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_nn_dom_rx_topk20of39 (nnn=20) | — | 0.8387 | 0.1250 | 256 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_nn_dom_rx_full | — | 0.8385 | 0.1238 | 294 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_prune0.1_nn19_nnn12 (nnn=12) | — | 0.8286 | 0.1966 | 162 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk8of39_nn16of27 (nnn=8) | — | 0.8202 | — | 128 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_prune0.05_nn25_nnn18 (nnn=18) | — | 0.7870 | 0.1953 | 222 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_topk8of39_nn16of27 (nnn=8) | — | 0.7841 | — | 176 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_prune0.2_nn13_nnn12 (nnn=12) | — | 0.7820 | — | 176 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_nn_dom_rx_full | — | 0.7785 | 0.1630 | 240 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_nn_dom_rx_topk20of39 (nnn=20) | — | 0.7773 | 0.1628 | 202 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_prune0.05_nn23_nnn16 (nnn=16) | — | 0.7682 | — | 202 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_prune0.3_nn12_nnn12 (nnn=12) | — | 0.6987 | — | 168 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_prune0.2_nn13_nnn9 (nnn=9) | — | 0.6973 | 0.2637 | 114 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_topk_regime (nnn=20) | — | 0.6703 | 0.2543 | 188 | — | — | second_order | 2026-10-15 |
| 18 | 0.50 | HVA | p2_full_ref (nnn=39) | — | 0.6648 | 0.2658 | 264 | — | — | transfer<p3_base@0.913>+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_base_full | — | 0.6389 | 0.3676 | 264 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p3_half_nn_rx_topk5of39_nn16of27 (nnn=5) | — | 0.5635 | — | 158 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_base_prune0.05_nn18_nnn15 (nnn=15) | — | 0.5435 | 0.3857 | 132 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk8of39_nn21of27 (nnn=8) | — | 0.4021 | — | 158 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_base_prune0.2_nn7_nnn9 (nnn=9) | — | 0.3680 | 0.5352 | 64 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | HVA | p2_half_nn_rx_topk13of39_nn21of27 (nnn=13) | — | 0.3229 | — | 178 | — | — | analytic+descent | 2026-10-15 |
| 18 | 0.50 | HVA | p2_base_prune0.1_nn14_nnn10 (nnn=10) | — | 0.1258 | 0.6087 | 96 | — | — | masked | 2026-10-15 |
| 18 | 0.50 | VL | L8/F50 | mps χ=64 | 0.8625 | 0.5382 | 264 | 1768 | 181 | n/a (fixed circuit) | 2026-10-15 |
| 18 | 0.70 | HVA | p2_half_nn_rx_full | — | 0.8932 | 0.2142 | 318 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_half_nn_rx_prune0.05_nn27_nnn25 (nnn=25) | — | 0.8832 | 0.2287 | 262 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_half_nn_rx_prune0.1_nn25_nnn14 (nnn=14) | — | 0.7463 | 0.3689 | 206 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_base_prune0.05_nn27_nnn39 (nnn=39) | — | 0.7276 | 0.5718 | 264 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_base_full | — | 0.7276 | 0.5718 | 264 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_base_prune0.1_nn26_nnn38 (nnn=38) | — | 0.6989 | 0.6243 | 256 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_base_prune0.2_nn26_nnn38 (nnn=38) | — | 0.6989 | 0.6243 | 256 | — | — | masked | 2026-10-15 |
| 18 | 0.70 | HVA | p2_half_nn_rx_prune0.2_nn17_nnn9 (nnn=9) | — | 0.5393 | 0.7290 | 138 | — | — | masked | 2026-10-15 |
| 18 | 1.00 | HVA | nnn p=2 (fair-conv best) | — | 0.9314 | 0.2608 | 264 | 630 | 106 | 6/6 seeds | 2026-10-15 |
| 18 | 1.00 | VL | L8/F50 | mps χ=64 | 0.7924 | 1.3939 | 264 | 1767 | 181 | n/a (fixed circuit) | 2026-10-15 |
| 18 | 3.00 | HVA | prune0.3 (nnn=31) | — | 0.9912 | — | 83 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_full | — | 0.9939 | 0.0202 | 524 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_nn_dom2_rx_full | — | 0.9884 | 0.0388 | 432 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk14of46_fg (nnn=14) | — | 0.9856 | — | 332 | — | — | transfer<full>+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_nn_dom2_rx_topk14of46_fg (nnn=14) | — | 0.9844 | 0.0466 | 304 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk10of46_fg (nnn=10) | — | 0.9837 | 0.0490 | 308 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk9of46_fg (nnn=9) | — | 0.9815 | — | 302 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk12of46_fg (nnn=12) | — | 0.9800 | — | 320 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk7of46_fg (nnn=7) | — | 0.9800 | 0.0566 | 290 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk14of46_th (nnn=14) | — | 0.9800 | 0.0586 | 332 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | diag_axial50 (nnn=35) | — | 0.9790 | — | 326 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk12of46_th (nnn=12) | — | 0.9765 | 0.0581 | 320 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk11of46_fg (nnn=11) | — | 0.9759 | 0.0571 | 314 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | diag_axial50_top20 (nnn=20) | — | 0.9749 | — | 266 | — | — | spec | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk9of46_th (nnn=9) | — | 0.9742 | 0.0594 | 302 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk5of46_fg (nnn=5) | — | 0.9689 | 0.0705 | 278 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk6of46_fg (nnn=6) | — | 0.9664 | 0.0760 | 284 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | full (nnn=46) | — | 0.9647 | — | 370 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | nnn2_half | — | 0.9647 | 0.0923 | 370 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_prune0.25_th_nn13_nnn12 (nnn=12) | — | 0.9491 | — | 176 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | diag_axial50_top8 (nnn=8) | — | 0.9459 | — | 218 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | diag_axial50_top10 (nnn=10) | — | 0.9444 | — | 226 | — | — | spec | 2026-10-15 |
| 20 | 0.50 | HVA | diag_axial50_top12 (nnn=12) | — | 0.9438 | — | 234 | — | — | spec | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_prune0.2_th_nn13_nnn13 (nnn=13) | — | 0.9424 | — | 182 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_prune0.1_nn27_nnn14 (nnn=14) | — | 0.9377 | — | 300 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_prune0.3_th_nn11_nnn10 (nnn=10) | — | 0.8988 | — | 148 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_prune0.2_nn20_nnn2 (nnn=2) | — | 0.8917 | — | 172 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | nnn2 | — | 0.4825 | 1.2671 | 308 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | diag_only (nnn=24) | — | 0.4813 | — | 282 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | ADAPT(square)-maxfid | — | 0.4538 | — | 222 | — | — | adapt | 2026-10-15 |
| 20 | 0.50 | HVA | ADAPT(square)-pareto | — | 0.4515 | — | 198 | — | — | adapt | 2026-10-15 |
| 20 | 0.50 | HVA | nnn1_half | — | 0.2470 | 1.9800 | 216 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | p3_half_nn_rx_topk12of46_fg_nn20of31 (nnn=12) | — | 0.1995 | — | 232 | — | — | analytic+descent | 2026-10-15 |
| 20 | 0.50 | HVA | nn2 | — | 0.0989 | 3.1418 | 124 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | nnn1 | — | 0.0658 | 3.5301 | 154 | — | — | masked | 2026-10-15 |
| 20 | 0.50 | HVA | nn1 | — | 0.0222 | 3.9051 | 62 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_full | — | 0.9652 | 0.1316 | 524 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_topk14of46_th (nnn=14) | — | 0.9472 | 0.2573 | 332 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_topk14of46_fg (nnn=14) | — | 0.9188 | 0.3623 | 332 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_topk9of46_th (nnn=9) | — | 0.8984 | 0.3959 | 302 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | ADAPT(square)-maxfid | — | 0.8831 | — | 264 | — | — | adapt | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_topk9of46_fg (nnn=9) | — | 0.8812 | 0.3855 | 302 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | full (nnn=46) | — | 0.8658 | — | 370 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | nnn2_half | — | 0.8658 | 0.3679 | 370 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | diag_axial50 (nnn=35) | — | 0.8170 | — | 326 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_prune0.1_nn27_nnn18 (nnn=18) | — | 0.7986 | 0.4680 | 324 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | nnn2 | — | 0.7837 | 0.5433 | 308 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | p3_half_nn_rx_prune0.2_nn20_nnn2 (nnn=2) | — | 0.7370 | 0.8761 | 172 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | diag_only (nnn=24) | — | 0.7264 | — | 282 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | nnn1_half | — | 0.7137 | 0.7711 | 216 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | nnn1 | — | 0.6155 | 1.1141 | 154 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | ADAPT(square)-pareto | — | 0.6105 | — | 124 | — | — | adapt | 2026-10-15 |
| 20 | 1.00 | HVA | nn2 | — | 0.5999 | 1.0842 | 124 | — | — | masked | 2026-10-15 |
| 20 | 1.00 | HVA | nn1 | — | 0.5308 | 1.3847 | 62 | — | — | masked | 2026-10-15 |
