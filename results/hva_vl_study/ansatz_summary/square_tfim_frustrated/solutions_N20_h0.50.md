# Soluciones — square tfim_frustrated | N=20 h=0.5

*Auto-generado: 2026-10-15 13:35 UTC. Regenerar con `generate_ansatz_summary.py --n 20 --h 0.5`.*

**Diagonalización exacta:** E0 = -20.14732 | gap físico = 0.4074 | régimen: degenerado (doblete) (manifold dim 2).

**32 soluciones** agregadas de specs + sweeps + ADAPT. Ordenadas por fidelidad/2q (mejor eficiencia primero). ★ = frontera de Pareto (no dominada en fid vs 2q).

| # | Ansatz | Método | Fidelidad | métrica | 2q | nnn bonds | fid/2q | ΔE/gap | Pareto | fuente |
|---|--------|--------|:---------:|:-------:|:--:|:---------:|:------:|:------:|:------:|--------|
| 1 | p3_half_nn_rx_prune0.3_th_nn11_nnn | generation(spec) | 0.8988 | vec | 148 | 10 | 6.073e-03 | – | ★ | p3_half_nn_rx_prune0.3_th_nn11 |
| 2 | p3_half_nn_rx_prune0.25_th_nn13_nn | generation(spec) | 0.9491 | vec | 176 | 12 | 5.392e-03 | – | ★ | p3_half_nn_rx_prune0.25_th_nn1 |
| 3 | p3_half_nn_rx_prune0.2_nn20_nnn2 | generation(spec) | 0.8917 | vec | 172 | 2 | 5.185e-03 | – |  | p3_half_nn_rx_prune0.2_nn20_nn |
| 4 | p3_half_nn_rx_prune0.2_th_nn13_nnn | generation(spec) | 0.9424 | vec | 182 | 13 | 5.178e-03 | – |  | p3_half_nn_rx_prune0.2_th_nn13 |
| 5 | diag_axial50_top8 | generation(sweep) | 0.9459 | sub | 218 | 8 | 4.339e-03 | 0.27 |  | n20_hybrid_prune_square_N20_h0 |
| 6 | diag_axial50_top10 | generation(spec) | 0.9444 | vec | 226 | 10 | 4.179e-03 | – |  | diag_axial50_top10_square_N20_ |
| 7 | diag_axial50_top12 | generation(spec) | 0.9438 | vec | 234 | 12 | 4.033e-03 | – |  | diag_axial50_top12_square_N20_ |
| 8 | diag_axial50_top20 | generation(spec) | 0.9749 | vec | 266 | 20 | 3.665e-03 | – | ★ | diag_axial50_top20_square_N20_ |
| 9 | p3_half_nn_rx_topk5of46_fg | generation(sweep) | 0.9689 | sub | 278 | 5 | 3.485e-03 | – |  | variant_topk_p3_half_nn_rx_fg_ |
| 10 | p3_half_nn_rx_topk6of46_fg | generation(sweep) | 0.9664 | sub | 284 | 6 | 3.403e-03 | – |  | variant_topk_p3_half_nn_rx_fg_ |
| 11 | p3_half_nn_rx_topk7of46_fg | generation(sweep) | 0.9800 | sub | 290 | 7 | 3.379e-03 | – | ★ | variant_topk_p3_half_nn_rx_fg_ |
| 12 | p3_half_nn_rx_topk9of46_fg | generation(spec) | 0.9815 | vec | 302 | 9 | 3.250e-03 | – | ★ | p3_half_nn_rx_topk9of46_fg_squ |
| 13 | p3_half_nn_rx_topk9of46_th | generation(sweep) | 0.9742 | sub | 302 | 9 | 3.226e-03 | – |  | variant_topk_p3_half_nn_rx_th_ |
| 14 | p3_half_nn_rx_topk10of46_fg | generation(sweep) | 0.9837 | sub | 308 | 10 | 3.194e-03 | – | ★ | variant_topk_p3_half_nn_rx_fg_ |
| 15 | p3_half_nn_rx_prune0.1_nn27_nnn14 | generation(spec) | 0.9377 | vec | 300 | 14 | 3.126e-03 | – |  | p3_half_nn_rx_prune0.1_nn27_nn |
| 16 | p3_half_nn_rx_topk11of46_fg | generation(spec) | 0.9759 | vec | 314 | 11 | 3.108e-03 | – |  | p3_half_nn_rx_topk11of46_fg_sq |
| 17 | p3_half_nn_rx_topk12of46_fg | generation(spec) | 0.9800 | vec | 320 | 12 | 3.063e-03 | – |  | p3_half_nn_rx_topk12of46_fg_sq |
| 18 | p3_half_nn_rx_topk12of46_th | generation(sweep) | 0.9765 | sub | 320 | 12 | 3.052e-03 | – |  | variant_topk_p3_half_nn_rx_th_ |
| 19 | diag_axial50 | generation(sweep) | 0.9790 | sub | 326 | 35 | 3.003e-03 | 0.14 |  | n20_geometric_square_N20_h0.50 |
| 20 | p3_half_nn_rx_topk14of46_fg | generation(spec) | 0.9856 | vec | 332 | 14 | 2.969e-03 | – | ★ | p3_half_nn_rx_topk14of46_fg_sq |
| 21 | p3_half_nn_rx_topk14of46_th | generation(sweep) | 0.9800 | sub | 332 | 14 | 2.952e-03 | – |  | variant_topk_p3_half_nn_rx_th_ |
| 22 | nnn2_half | generation(sweep) | 0.9647 | sub | 370 | 46 | 2.607e-03 | 0.23 |  | n20_extended_ansatze_square_N2 |
| 23 | ADAPT(h05)-pareto | adapt_vqe | 0.4515 | vec | 198 | 33 | 2.280e-03 | – |  | adapt_poolA_square_N20_short_n |
| 24 | ADAPT(h05)-maxfid | adapt_vqe | 0.4538 | vec | 222 | 37 | 2.044e-03 | – |  | adapt_poolA_square_N20_short_n |
| 25 | p3_half_nn_rx_full | generation(sweep) | 0.9939 | sub | 524 | 46 | 1.897e-03 | – | ★ | variant_topk_p3_half_nn_rx_th_ |
| 26 | diag_only | generation(sweep) | 0.4813 | sub | 282 | 24 | 1.707e-03 | 3.11 |  | n20_geometric_square_N20_h0.50 |
| 27 | nnn2 | generation(sweep) | 0.4825 | sub | 308 | – | 1.567e-03 | 3.11 |  | n20_extended_ansatze_square_N2 |
| 28 | nnn1_half | generation(sweep) | 0.2470 | sub | 216 | – | 1.144e-03 | 4.86 |  | n20_small_ansatze_square_N20_h |
| 29 | p3_half_nn_rx_topk12of46_fg_nn20of | generation(spec) | 0.1995 | vec | 232 | 12 | 8.599e-04 | – |  | p3_half_nn_rx_topk12of46_fg_nn |
| 30 | nn2 | generation(sweep) | 0.0989 | sub | 124 | – | 7.978e-04 | 7.71 | ★ | n20_small_ansatze_square_N20_h |
| 31 | nnn1 | generation(sweep) | 0.0658 | sub | 154 | – | 4.276e-04 | 8.66 |  | n20_small_ansatze_square_N20_h |
| 32 | nn1 | generation(sweep) | 0.0222 | sub | 62 | – | 3.585e-04 | 9.58 | ★ | n20_small_ansatze_square_N20_h |

## Notas

- **métrica**: `vec` = fidelidad al vector de estado fundamental; `sub` = fidelidad al subespacio del doblete degenerado (la correcta cuando el gap→0). No comparar `vec` vs `sub` directamente.
- **fid/2q**: fidelidad por compuerta de 2 qubits — la métrica de eficiencia del estudio.
- **nnn bonds**: cantidad de bonds next-nearest seleccionados (portadores de la frustración J2).
- **Pareto (★)**: solución no dominada — ninguna otra tiene ≥ fidelidad con ≤ 2q.
