# Accelerated Cross-N Coverage Analysis

**Fecha**: 2026-08-24 (auto-updated by update_cross_n_coverage.py)
**Modelo**: TFIM bond-resolved, p=1
**Método**: AcceleratedVQE + UnifiedMPNN cross-N transfer
**Fuentes**: `model_quality_dashboard.json`, NPZ training data, GT cache

> **Nota**: Las secciones marcadas con `<!-- AUTO-GENERATED -->` se actualizan
> automáticamente. Las secciones de análisis físico y recomendaciones son manuales.

---

<!-- AUTO-GENERATED-BEGIN:executive_summary -->
## Resumen Ejecutivo (auto-generated)

| Topología | N values | Total pts | h-range | Best pass@5% | Zoo (multi-N) | n_max_viable |
|-----------|---------|-----------|---------|--------------|---------------|--------------|
| chain_1d | 4,6,8,10,12,14,15,16,20,26,30,40,60 | 772 | [1.5, 5.5] | 100% | — | 60 |
| heavy_hex | 4,6,8,10,12,14,16,18,20,21,22,24,26,30,40 | 833 | [0.3, 5.5] | 100% | — | 40 |
| ladder | 4,6,8,10,12,14,16,20,26,30,40 | 624 | [1.4, 5.5] | 94% | — | 40 |
| square | 4,6,8,10,12,14,16,20 | 507 | [1.2, 5.5] | 88% | — | 16 |
| triangular | 3,4,6,8,10,12 | 407 | [0.5, 5.5] | 100% | — | 6 |
<!-- AUTO-GENERATED-END:executive_summary -->

---

## Training Data Health

<!-- AUTO-GENERATED-BEGIN:health -->
| Metric | Value | Status |
|--------|-------|--------|
| Total NPZ files | 53 | |
| Total training points | 3143 | |
| **Quality: Useful** | 37 configs | ✅ |
| **Quality: Insufficient** | 13 configs | ⚠️ |
| **Quality: Not Useful** | 3 configs | ❌ |
| NaN in θ | 0 configs | ✅ |
| Zoo integrity | True | ✅ |
| Zoo missing | 0 | ✅ |
| Zoo orphan checkpoints | 0 | ✅ |
| GT coverage gaps | 1913 uncovered h-points | ⚠️ |
| Stale zoo models | 26 | ⚠️ |
| Need retrain | 0 | ✅ |
| High θ discontinuity (>0.5) | 29 configs | ⚠️ |
| Gap masking detected | 27 configs | ⚠️ |
<!-- AUTO-GENERATED-END:health -->

### Quality Tier Distribution (NPZ-level)

<!-- AUTO-GENERATED-BEGIN:quality_tiers -->
| File | Total | Verified ✅ | Approximate ⚠️ | Unverified ❓ |
|------|-------|-------------|----------------|---------------|
| chain_1d_N10_p1.npz | 45 | 40 (89%) | 5 (11%) | 0 (0%) |
| chain_1d_N12_p1.npz | 37 | 27 (73%) | 10 (27%) | 0 (0%) |
| chain_1d_N15_p1.npz | 27 | 16 (59%) | 11 (41%) | 0 (0%) |
| chain_1d_N20_p1.npz | 5 | 5 (100%) | 0 (0%) | 0 (0%) |
| chain_1d_N6_p1.npz | 40 | 39 (98%) | 1 (2%) | 0 (0%) |
| chain_1d_N8_p1.npz | 14 | 14 (100%) | 0 (0%) | 0 (0%) |
| heavy_hex_N10_p1.npz | 99 | 72 (73%) | 25 (25%) | 2 (2%) |
| heavy_hex_N12_p1.npz | 20 | 15 (75%) | 4 (20%) | 1 (5%) |
| heavy_hex_N16_p1.npz | 41 | 39 (95%) | 2 (5%) | 0 (0%) |
| heavy_hex_N4_p1.npz | 15 | 10 (67%) | 5 (33%) | 0 (0%) |
| heavy_hex_N6_p1.npz | 42 | 30 (71%) | 12 (29%) | 0 (0%) |
| ladder_N10_p1.npz | 89 | 9 (10%) | 67 (75%) | 13 (15%) |
| ladder_N12_p1.npz | 39 | 16 (41%) | 23 (59%) | 0 (0%) |
| ladder_N14_p1.npz | 58 | 0 (0%) | 58 (100%) | 0 (0%) |
| ladder_N16_p1.npz | 24 | 2 (8%) | 22 (92%) | 0 (0%) |
| ladder_N20_p1.npz | 29 | 0 (0%) | 18 (62%) | 11 (38%) |
| ladder_N24_p1.npz | 3 | 0 (0%) | 0 (0%) | 3 (100%) |
| ladder_N26_p1.npz | 3 | 0 (0%) | 0 (0%) | 3 (100%) |
| ladder_N4_p1.npz | 13 | 3 (23%) | 5 (38%) | 5 (38%) |
| ladder_N6_p1.npz | 57 | 38 (67%) | 18 (32%) | 1 (2%) |
| ladder_N8_p1.npz | 57 | 27 (47%) | 27 (47%) | 3 (5%) |
| square_N10_p1.npz | 25 | 17 (68%) | 8 (32%) | 0 (0%) |
| square_N12_p1.npz | 25 | 12 (48%) | 10 (40%) | 3 (12%) |
| square_N14_p1.npz | 24 | 0 (0%) | 24 (100%) | 0 (0%) |
| square_N16_p1.npz | 32 | 3 (9%) | 17 (53%) | 12 (38%) |
| square_N20_p1.npz | 7 | 0 (0%) | 7 (100%) | 0 (0%) |
| square_N4_p1.npz | 41 | 35 (85%) | 4 (10%) | 2 (5%) |
| square_N6_p1.npz | 64 | 47 (73%) | 8 (12%) | 9 (14%) |
| square_N8_p1.npz | 43 | 39 (91%) | 4 (9%) | 0 (0%) |
| triangular_N10_p1.npz | 27 | 6 (22%) | 19 (70%) | 2 (7%) |
| triangular_N12_p1.npz | 20 | 4 (20%) | 13 (65%) | 3 (15%) |
| triangular_N14_p1.npz | 5 | 0 (0%) | 0 (0%) | 5 (100%) |
| triangular_N16_p1.npz | 7 | 0 (0%) | 0 (0%) | 7 (100%) |
| triangular_N3_p1.npz | 37 | 22 (59%) | 0 (0%) | 15 (41%) |
| triangular_N4_p1.npz | 36 | 26 (72%) | 3 (8%) | 7 (19%) |
| triangular_N6_p1.npz | 97 | 45 (46%) | 36 (37%) | 16 (16%) |
| triangular_N8_p1.npz | 14 | 4 (29%) | 10 (71%) | 0 (0%) |
| triangular_N9_p1.npz | 32 | 0 (0%) | 17 (53%) | 15 (47%) |
| **TOTAL** | **1293** | **662** (51%) | **493** (38%) | **138** (11%) |
<!-- AUTO-GENERATED-END:quality_tiers -->

**Quality Tier Warnings:**

⚠️ ladder_N14_p1.npz: 58 approximate pts but 0 verified. Consider running --refine-all to convert best predictions to verified.
⚠️ ladder_N20_p1.npz: 18 approximate pts but 0 verified. Consider running --refine-all to convert best predictions to verified.
⚠️ ladder_N24_p1.npz: 100% unverified — run VQE refinement to generate quality data
⚠️ ladder_N26_p1.npz: 100% unverified — run VQE refinement to generate quality data
⚠️ square_N14_p1.npz: 24 approximate pts but 0 verified. Consider running --refine-all to convert best predictions to verified.

*(and 4 more)*

---

## Gap Masking Analysis

Configs where `pass@5% - pass@dual_criterion > 10%` — large gap inflates ΔE/gap metric:

<!-- AUTO-GENERATED-BEGIN:gap_masking -->
| Topology | N | Pass@5% | Pass@dual | Gap masked |
|----------|---|---------|-----------|------------|
| heavy_hex | 26 | 100% | 0% | 100% |
| ladder | 40 | 50% | 0% | 50% |
| chain_1d | 30 | 75% | 29% | 46% |
| ladder | 12 | 94% | 48% | 46% |
| ladder | 16 | 66% | 25% | 42% |
| ladder | 10 | 77% | 37% | 40% |
| chain_1d | 26 | 100% | 67% | 33% |
| ladder | 20 | 54% | 23% | 31% |
| chain_1d | 40 | 71% | 42% | 29% |
| chain_1d | 16 | 100% | 73% | 27% |
| square | 16 | 42% | 15% | 27% |
| triangular | 12 | 33% | 7% | 26% |
| ladder | 8 | 79% | 53% | 26% |
| ladder | 26 | 38% | 12% | 25% |
| chain_1d | 60 | 45% | 23% | 23% |
| ladder | 14 | 48% | 27% | 21% |
| square | 12 | 68% | 47% | 21% |
| triangular | 8 | 30% | 9% | 21% |
| ladder | 30 | 40% | 20% | 20% |
| chain_1d | 14 | 100% | 80% | 20% |
| square | 14 | 73% | 54% | 19% |
| triangular | 10 | 35% | 17% | 19% |
| ladder | 6 | 87% | 72% | 14% |
| square | 10 | 76% | 63% | 13% |
| square | 20 | 12% | 0% | 12% |
| triangular | 6 | 70% | 60% | 10% |
| chain_1d | 20 | 89% | 78% | 10% |
<!-- AUTO-GENERATED-END:gap_masking -->

---

## Detalle por Topología

<!-- AUTO-GENERATED-BEGIN:topo_chain_1d -->
### Chain 1D

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 6 | [2.50, 5.00] | 100% | 100% | 2.50 | 0.05  | div=0.33 STALE |
| 6 | 90 | [1.50, 5.50] | 99% | 99% | 2.50 | 1.57 ⚠️ | div=0.32 STALE |
| 8 | 111 | [1.50, 5.50] | 96% | 96% | 2.50 | 1.64 ⚠️ | div=0.30 STALE |
| 10 | 124 | [1.50, 5.50] | 93% | 85% | 2.50 | 1.58 ⚠️ | (7% masked) div=0.26 STALE |
| 12 | 69 | [1.50, 5.50] | 91% | 91% | 2.50 | 1.45 ⚠️ | div=0.25 STALE |
| 14 | 5 | [3.00, 5.00] | 100% | 80% | 3.00 | 0.00  | ⚠️ GAP MASK +20% div=0.33 STALE |
| 15 | 173 | [1.50, 5.50] | 90% | 86% | 3.00 | 1.96 ⚠️ | div=0.24 STALE |
| 16 | 26 | [3.00, 5.50] | 100% | 73% | 3.00 | 0.29  | ⚠️ GAP MASK +27% div=0.33 STALE |
| 20 | 88 | [1.50, 5.50] | 89% | 78% | 3.00 | 2.70 ⚠️ | (10% masked) div=0.22 STALE |
| 26 | 6 | [3.75, 5.00] | 100% | 67% | 3.75 | 0.04  | ⚠️ GAP MASK +33% div=0.33 STALE |
| 30 | 28 | [2.50, 5.50] | 75% | 29% | 3.75 | 0.29  | ⚠️ GAP MASK +46% div=0.08 |
| 40 | 24 | [2.50, 5.50] | 71% | 42% | 3.75 | 0.30  | ⚠️ GAP MASK +29% div=0.04 |
| 60 | 22 | [2.50, 5.50] | 45% | 23% | 3.75 | 0.25  | ⚠️ GAP MASK +23% div=0.21 |
<!-- AUTO-GENERATED-END:topo_chain_1d -->

<!-- AUTO-GENERATED-BEGIN:topo_heavy_hex -->
### Heavy Hex

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 84 | [0.58, 5.50] | 94% | 94% | 0.96 | 0.08  | — |
| 6 | 42 | [1.90, 4.50] | 100% | 100% | 1.90 | 0.05  | — |
| 8 | 98 | [0.30, 5.00] | 55% | 55% | 1.90 | 1.70 ⚠️ | — |
| 10 | 164 | [1.40, 5.50] | 93% | 90% | 1.90 | 1.62 ⚠️ | — |
| 12 | 77 | [1.40, 4.50] | 87% | 87% | 1.93 | 1.73 ⚠️ | — |
| 14 | 59 | [0.30, 5.00] | 42% | 36% | 1.99 | 1.57 ⚠️ | (7% masked) |
| 16 | 149 | [1.40, 5.50] | 88% | 83% | 2.02 | 3.16 ⚠️ | — |
| 18 | 36 | [2.50, 5.00] | 100% | 100% | 2.50 | 1.57 ⚠️ | — |
| 20 | 38 | [2.00, 5.00] | 55% | 50% | 2.77 | 1.78 ⚠️ | (5% masked) |
| 21 | 8 | [4.08, 5.00] | 100% | 100% | 4.08 | 1.57 ⚠️ | — |
| 22 | 20 | [2.50, 5.00] | 95% | 95% | 4.08 | 1.57 ⚠️ | — |
| 24 | 20 | [2.50, 5.00] | 90% | 90% | 4.08 | 1.60 ⚠️ | — |
| 26 | 5 | [4.00, 5.00] | 100% | 0% | 4.08 | 0.05  | ⚠️ GAP MASK +100% |
| 30 | 27 | [2.00, 5.00] | 52% | 52% | 4.08 | 2.72 ⚠️ | — |
| 40 | 6 | [2.50, 4.50] | 83% | 83% | 4.08 | 1.63 ⚠️ | — |
<!-- AUTO-GENERATED-END:topo_heavy_hex -->

<!-- AUTO-GENERATED-BEGIN:topo_ladder -->
### Ladder

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 45 | [1.36, 5.00] | 76% | 73% | 1.85 | 0.14  | div=0.39 STALE |
| 6 | 69 | [2.00, 4.80] | 87% | 72% | 2.33 | 0.62 ⚠️ | (14% masked) div=0.51 STALE |
| 8 | 98 | [1.80, 5.00] | 79% | 53% | 2.62 | 0.79 ⚠️ | ⚠️ GAP MASK +26% div=0.42 STALE |
| 10 | 156 | [2.00, 5.50] | 77% | 37% | 2.81 | 0.79 ⚠️ | ⚠️ GAP MASK +40% div=0.41 STALE |
| 12 | 54 | [2.50, 5.50] | 94% | 48% | 2.98 | 0.40  | ⚠️ GAP MASK +46% div=0.58 STALE |
| 14 | 33 | [2.00, 4.80] | 48% | 27% | 3.10 | 0.79 ⚠️ | ⚠️ GAP MASK +21% div=0.12 |
| 16 | 101 | [1.80, 5.50] | 66% | 25% | 3.21 | 0.79 ⚠️ | ⚠️ GAP MASK +42% div=0.30 STALE |
| 20 | 35 | [2.50, 5.35] | 54% | 23% | 3.94 | 0.17  | ⚠️ GAP MASK +31% div=0.18 STALE |
| 26 | 16 | [2.50, 5.00] | 38% | 12% | 4.16 | 0.05  | ⚠️ GAP MASK +25% div=0.01 |
| 30 | 15 | [2.50, 5.50] | 40% | 20% | 4.37 | 0.03  | ⚠️ GAP MASK +20% div=0.04 |
| 40 | 2 | [4.50, 5.00] | 50% | 0% | 4.68 | 0.02  | ⚠️ GAP MASK +50% div=0.14 |
<!-- AUTO-GENERATED-END:topo_ladder -->

<!-- AUTO-GENERATED-BEGIN:topo_square -->
### Square

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 41 | [1.40, 4.50] | 85% | 85% | 1.84 | 0.17  | div=0.52 STALE |
| 6 | 121 | [1.40, 4.80] | 88% | 85% | 2.33 | 0.79 ⚠️ | div=0.54 STALE |
| 8 | 119 | [2.00, 5.00] | 87% | 81% | 2.64 | 1.57 ⚠️ | (7% masked) div=0.54 STALE |
| 10 | 94 | [2.00, 5.50] | 76% | 63% | 2.89 | 1.57 ⚠️ | (13% masked) div=0.42 STALE |
| 12 | 72 | [1.80, 5.50] | 68% | 47% | 3.16 | 1.87 ⚠️ | ⚠️ GAP MASK +21% div=0.35 STALE |
| 14 | 26 | [2.50, 5.50] | 73% | 54% | 3.29 | 1.57 ⚠️ | ⚠️ GAP MASK +19% div=0.40 STALE |
| 16 | 26 | [2.50, 5.00] | 42% | 15% | 3.60 | 0.14  | ⚠️ GAP MASK +27% div=0.09 |
| 20 | 8 | [1.18, 5.00] | 12% | 0% | 4.77 | 0.21  | (12% masked) div=0.21 |
<!-- AUTO-GENERATED-END:topo_square -->

<!-- AUTO-GENERATED-BEGIN:topo_triangular -->
### Triangular

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 3 | 37 | [0.50, 4.50] | 100% | 100% | 0.50 | 0.04  | div=0.75 STALE |
| 4 | 36 | [1.24, 4.50] | 72% | 72% | 1.98 | 0.04  | div=0.47 STALE |
| 6 | 145 | [2.13, 5.50] | 70% | 60% | 3.15 | 0.05  | (10% masked) div=0.45 STALE |
| 8 | 77 | [1.80, 4.80] | 30% | 9% | 3.98 | 0.01  | ⚠️ GAP MASK +21% div=0.05 |
| 10 | 54 | [2.00, 5.00] | 35% | 17% | 3.98 | 1.57 ⚠️ | ⚠️ GAP MASK +19% div=0.10 |
| 12 | 58 | [2.50, 5.50] | 33% | 7% | 4.44 | 0.01  | ⚠️ GAP MASK +26% div=0.08 |
<!-- AUTO-GENERATED-END:topo_triangular -->

---

## h_frontier per Topology

h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):

<!-- AUTO-GENERATED-BEGIN:h_frontier -->
| Topología | N=3 | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=15 | N=16 | N=18 | N=20 | N=21 | N=22 | N=24 | N=26 | N=30 | N=40 | N=60 |
|-----------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| chain_1d | — | 2.50 | 2.50 | 2.50 | 2.50 | 2.50 | 3.00 | 3.00 | 3.00 | — | 3.00 | — | — | — | 3.75 | 3.75 | 3.75 | 3.75 |
| heavy_hex | — | 0.96 | 1.90 | 1.90 | 1.90 | 1.93 | 1.99 | — | 2.02 | 2.50 | 2.77 | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 | 4.08 | — |
| ladder | — | 1.85 | 2.33 | 2.62 | 2.81 | 2.98 | 3.10 | — | 3.21 | — | 3.94 | — | — | — | 4.16 | 4.37 | 4.68 | — |
| square | — | 1.84 | 2.33 | 2.64 | 2.89 | 3.16 | 3.29 | — | 3.60 | — | 4.77 | — | — | — | — | — | — | — |
| triangular | 0.50 | 1.98 | 3.15 | 3.98 | 3.98 | 4.44 | — | — | — | — | — | — | — | — | — | — | — | — |
<!-- AUTO-GENERATED-END:h_frontier -->

---

## Cross-N Transfer Summary

<!-- AUTO-GENERATED-BEGIN:cross_n_transfer -->
| Topology | n_max_viable | Best pass@5% | Best cross-N source |
|----------|-------------|-------------|---------------------|
| chain_1d | 60 | 100% | no data |
| heavy_hex | 40 | 100% | no data |
| ladder | 40 | 94% | no data |
| square | 16 | 88% | train_n=10 (@10%=0%) |
| triangular | 6 | 100% | no data |
<!-- AUTO-GENERATED-END:cross_n_transfer -->

---

<!-- AUTO-GENERATED-BEGIN:large_n_extrapolation -->
## Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Model trained on N≤20,
evaluated at N=30-100 via MPS backend. Speedup = VQE_evals / MPNN_evals.

### chain_1d

**Model**: `data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 16 | [2.5, 5.0] | 10 | 0.1050 | 3.19e-02 | 6/10 | 2/10 | 3488× |
| 20 | [2.5, 5.0] | 7 | 0.0678 | 1.81e-02 | 6/7 | 6/7 | 4600× |
| 30 | [2.5, 5.5] | 28 | 0.0351 | 7.10e-03 | 21/28 | 8/28 | 20580× |
| 40 | [2.5, 5.5] | 24 | 0.0405 | 5.90e-03 | 17/24 | 10/24 | 11440× |
| 60 | [2.5, 5.5] | 22 | 0.0725 | 7.45e-03 | 10/22 | 5/22 | 13920× |
| 80 | [2.5, 5.0] | 8 | 0.1658 | 8.85e-03 | 0/8 | 0/8 | — |
| 100 | [2.5, 5.5] | 19 | 0.1369 | 7.98e-03 | 5/19 | 2/19 | 22880× |
| 150 | [4.0, 5.0] | 3 | 0.7841 | 3.58e-02 | 0/3 | 0/3 | — |
| 200 | [4.0, 5.0] | 3 | 1.0472 | 3.59e-02 | 0/3 | 0/3 | — |

### heavy_hex

**Model**: `data/model_zoo/checkpoints/unified_multiN_heavyhex_p1_v2.pt`

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [2.5, 5.0] | 14 | 0.0106 | 6.28e-03 | 14/14 | 11/14 | 2490× |
| 14 | [2.5, 5.0] | 14 | 0.0201 | 7.96e-03 | 13/14 | 11/14 | 3325× |
| 16 | [2.5, 5.0] | 6 | 0.1059 | 3.74e-02 | 0/6 | 0/6 | 3595× |
| 20 | [2.0, 5.0] | 60 | 0.2071 | 1.06e-02 | 37/60 | 28/60 | 16860× |
| 21 | [2.5, 5.0] | 14 | 0.3410 | 4.44e-02 | 0/14 | 0/14 | 4882× |
| 22 | [2.5, 5.0] | 10 | 0.2006 | 1.80e-02 | 6/10 | 6/10 | — |
| 24 | [2.5, 5.0] | 14 | 0.0897 | 8.22e-03 | 9/14 | 9/14 | 10352× |
| 26 | [2.5, 5.0] | 14 | 0.1525 | 1.37e-02 | 9/14 | 9/14 | 10868× |
| 30 | [2.0, 5.0] | 45 | 0.3664 | 1.08e-02 | 32/45 | 30/45 | 40208× |
| 32 | [2.5, 5.0] | 10 | 0.3840 | 2.02e-02 | 6/10 | 6/10 | — |
| 40 | [2.5, 5.0] | 33 | 0.2009 | 1.45e-02 | 13/33 | 13/33 | 9333× |
| 50 | [2.5, 5.0] | 6 | 2.5883 | 1.23e-01 | 0/6 | 0/6 | 11100× |
| 60 | [2.5, 5.0] | 6 | 3.3160 | 1.32e-01 | 0/6 | 0/6 | 13480× |

### ladder

**Model**: `data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1_v2.pt`

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 16 | [2.5, 5.0] | 6 | 0.0784 | 1.40e-02 | 4/6 | 0/6 | 10033× |
| 20 | [2.5, 5.5] | 24 | 0.2592 | 7.24e-03 | 8/24 | 6/24 | 13708× |
| 26 | [2.5, 5.0] | 14 | 0.5173 | 8.95e-03 | 5/14 | 2/14 | 7211× |
| 30 | [2.5, 5.5] | 14 | 0.5130 | 7.44e-03 | 6/14 | 3/14 | 8165× |
| 40 | [2.5, 5.0] | 6 | 1.5579 | 9.88e-03 | 1/6 | 0/6 | 11154× |

### square

**Model**: `data/model_zoo/checkpoints/unified_tfim_br_square_fromMT_4+6+8+10+12+14+16_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 16 | [2.5, 5.0] | 26 | 0.0813 | 1.65e-02 | 11/26 | 4/26 | 10824× |
| 20 | [2.5, 5.0] | 26 | 0.7914 | 1.75e-02 | 3/26 | 1/26 | 17433× |
| 30 | [2.5, 5.0] | 13 | 2.4057 | 2.97e-02 | 0/13 | 0/13 | 8773× |

### triangular

**Model**: `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 12 | [2.5, 5.0] | 10 | 1.7850 | 7.33e-02 | 2/10 | 0/10 | 3588× |
| 16 | [2.5, 5.0] | 10 | 28.7030 | 1.63e-01 | 0/10 | 0/10 | 6197× |
| 24 | [2.5, 5.0] | 10 | 23.4736 | 2.56e-01 | 0/10 | 0/10 | 9520× |

### Extensive Scaling Summary

| Topology | N range | |ΔE|/N (mean) | Variation | Scaling |
|----------|---------|--------------|-----------|---------|
| chain_1d | 16–200 | 1.77e-02 | 6.1× | ⚠️ degrading |
| heavy_hex | 10–60 | 3.44e-02 | 21.1× | ⚠️ degrading |
| ladder | 16–40 | 9.50e-03 | 1.9× | ✅ extensive |
| square | 16–30 | 2.12e-02 | 1.8× | ✅ extensive |
| triangular | 12–24 | 1.64e-01 | 3.5× | ⚠️ degrading |

### MPNN vs Random VQE vs Ground Truth

Comparison at same h-points. MPNN: 1 forward pass (0 QPU). VQE: L-BFGS-B with random init.

| Topology | N | MPNN ΔE/gap | VQE ΔE/gap | MPNN |ΔE|/N | MPNN wins? | Speedup | VQE evals |
|----------|---|-------------|------------|------|-------|---------|-----------|
| chain_1d | 10 | 0.0122 | 2.2902 | 8.53e-03 | ✅ | 2367× | 14,200 |
| chain_1d | 16 | 0.0206 | 2.3192 | 8.95e-03 | ✅ | 3488× | 20,928 |
| chain_1d | 20 | 0.0090 | 2.0355 | 1.79e-03 | ✅ | 4600× | 27,600 |
| chain_1d | 30 | 0.0130 | 2.3093 | 2.69e-03 | ✅ | 6960× | 20,880 |
| chain_1d | 40 | 0.0191 | 7.6963 | 1.89e-03 | ✅ | 8747× | 52,480 |
| chain_1d | 60 | 0.0292 | 7.0227 | 1.92e-03 | ✅ | 13480× | 80,880 |
| chain_1d | 100 | 0.1017 | 14.0382 | 6.85e-03 | ✅ | 22880× | 114,400 |
| heavy_hex | 10 | 0.0048 | 0.0016 | 3.28e-03 | ❌ | 2427× | 14,560 |
| heavy_hex | 14 | 0.0215 | 1.4948 | 9.70e-03 | ✅ | 3325× | 26,600 |
| heavy_hex | 16 | 0.0162 | 2.3631 | 6.73e-03 | ✅ | 3541× | 21,248 |
| heavy_hex | 20 | 0.0173 | 4.1020 | 1.65e-03 | ✅ | 4453× | 26,720 |
| heavy_hex | 21 | 0.2853 | 3.5297 | 4.73e-02 | ✅ | 4882× | 39,060 |
| heavy_hex | 24 | 0.0235 | 4.1442 | 1.82e-03 | ✅ | 10352× | 62,112 |
| heavy_hex | 26 | 0.0215 | 4.0195 | 1.52e-03 | ✅ | 10868× | 65,208 |
| heavy_hex | 30 | 0.0147 | 1.2455 | 2.11e-03 | ✅ | 39660× | 237,960 |
| heavy_hex | 40 | 0.0338 | 7.1373 | 1.51e-03 | ✅ | 9333× | 56,000 |
| heavy_hex | 50 | 2.5883 | 12.1727 | 1.23e-01 | ✅ | 11100× | 66,600 |
| heavy_hex | 60 | 3.3160 | 13.3788 | 1.32e-01 | ✅ | 13480× | 80,880 |
| ladder | 10 | 0.0130 | 0.0120 | 6.37e-03 | ❌ | 2832× | 16,992 |
| ladder | 16 | 0.0208 | 2.7202 | 6.29e-03 | ✅ | 4290× | 25,740 |
| ladder | 20 | 0.0236 | 3.1949 | 5.29e-03 | ✅ | 13708× | 54,831 |
| ladder | 26 | 0.5843 | 0.5997 | 9.36e-03 | ✅ | 7142× | 71,424 |
| ladder | 30 | 0.8963 | 17.4153 | 9.92e-03 | ✅ | 8165× | 48,988 |
| ladder | 40 | 1.5579 | 41.6156 | 9.88e-03 | ✅ | 11154× | 66,924 |
| square | 10 | 0.0509 | 0.0089 | 2.69e-02 | ❌ | 2608× | 15,648 |
| square | 16 | 0.0640 | 3.4592 | 1.28e-02 | ✅ | 4688× | 28,126 |
| square | 20 | 0.2791 | 0.1431 | 2.19e-02 | ❌ | 17433× | 69,732 |
| square | 30 | 2.1018 | 29.8002 | 1.81e-02 | ✅ | 8773× | 52,640 |
| triangular | 10 | 0.0234 | 0.0214 | 9.96e-03 | ❌ | 3766× | 15,066 |
| triangular | 12 | 0.0501 | 0.0467 | 1.48e-02 | ❌ | 3588× | 14,352 |
| triangular | 16 | 0.4003 | 0.4003 | 3.55e-02 | ✅ | 6197× | 37,180 |
| triangular | 24 | 4.0032 | 4.0033 | 4.37e-02 | ✅ | 9520× | 57,120 |

**MPNN win rate**: 26/32 (81%)

**Speedup range**: 2367× – 40208×

<!-- AUTO-GENERATED-END:large_n_extrapolation -->

---

## Quality Tier Distribution

Data quality breakdown by tier (verified=VQE-converged, approximate=MPNN-predicted, unverified=legacy):

<!-- AUTO-GENERATED-BEGIN:tier_breakdown -->
| Topology | Total pts | Verified | Approximate | Unverified |
|----------|-----------|----------|-------------|------------|
| chain_1d | 772 | 560 (72%) | 144 (18%) | 68 (8%) |
| heavy_hex | 833 | 689 (82%) | 82 (9%) | 62 (7%) |
| ladder | 624 | 261 (41%) | 259 (41%) | 104 (16%) |
| square | 507 | 364 (71%) | 96 (18%) | 47 (9%) |
| triangular | 407 | 235 (57%) | 96 (23%) | 76 (18%) |
<!-- AUTO-GENERATED-END:tier_breakdown -->

---

<!-- AUTO-GENERATED-BEGIN:training_plan -->
## Training Plan (auto-generated)

**Total configs**: 53 | ✅ Useful: 37 | ⚠️ Insufficient: 13 | ❌ Not useful: 3

### ❌ DELETE — Not useful for MPNN training

These NPZ files teach the MPNN wrong mappings. Remove or regenerate:

| File | Topology | N | Reason |
|------|----------|---|--------|
| `heavy_hex_N26_p1.npz` | heavy_hex | 26 | 0% dual pass (gap masking: 100% of points appear to pass but |
| `ladder_N40_p1.npz` | ladder | 40 | 0% dual pass (gap masking: 50% of points appear to pass but  |
| `square_N20_p1.npz` | square | 20 | 0% dual pass (gap masking: 12% of points appear to pass but  |

```bash
rm data/multi_n_training/heavy_hex_N26_p1.npz
rm data/multi_n_training/ladder_N40_p1.npz
rm data/multi_n_training/square_N20_p1.npz
```

### ⚠️ IMPROVE — Insufficient signal (need more good points)

Run iterative-improve to densify these configs above the frontier:

| File | Topology | N | Pts | Dual pass | h_frontier | Action |
|------|----------|---|-----|-----------|------------|--------|
| `chain_1d_N14_p1.npz` | chain_1d | 14 | 5 | 80% | 3.00 | iterative-improve h≥3.2 |
| `chain_1d_N26_p1.npz` | chain_1d | 26 | 6 | 67% | 3.75 | iterative-improve h≥4.0 |
| `chain_1d_N30_p1.npz` | chain_1d | 30 | 28 | 29% | 3.75 | iterative-improve h≥4.0 |
| `chain_1d_N60_p1.npz` | chain_1d | 60 | 22 | 23% | 3.75 | iterative-improve h≥4.0 |
| `ladder_N14_p1.npz` | ladder | 14 | 33 | 27% | 3.10 | iterative-improve h≥3.3 |
| `ladder_N16_p1.npz` | ladder | 16 | 101 | 25% | 3.21 | iterative-improve h≥3.4 |
| `ladder_N20_p1.npz` | ladder | 20 | 35 | 23% | 3.94 | iterative-improve h≥4.1 |
| `ladder_N26_p1.npz` | ladder | 26 | 16 | 12% | 4.16 | iterative-improve h≥4.4 |
| `ladder_N30_p1.npz` | ladder | 30 | 15 | 20% | 4.37 | iterative-improve h≥4.6 |
| `square_N16_p1.npz` | square | 16 | 26 | 15% | 3.60 | iterative-improve h≥3.8 |
| `triangular_N8_p1.npz` | triangular | 8 | 77 | 9% | 3.98 | iterative-improve h≥4.2 |
| `triangular_N10_p1.npz` | triangular | 10 | 54 | 17% | 3.98 | iterative-improve h≥4.2 |
| `triangular_N12_p1.npz` | triangular | 12 | 58 | 7% | 4.44 | iterative-improve h≥4.6 |

### ✅ EXPAND — Useful configs (add more h-points for better generalization)

| Topology | N | Pts | Dual pass | h_frontier | Priority |
|----------|---|-----|-----------|------------|----------|
| triangular | 3 | 37 | 100% | 0.50 | LOW (already dense) |
| chain_1d | 4 | 6 | 100% | 2.50 | HIGH (few pts, good quality) |
| heavy_hex | 4 | 84 | 94% | 0.96 | LOW (already dense) |
| ladder | 4 | 45 | 73% | 1.85 | LOW (already dense) |
| square | 4 | 41 | 85% | 1.84 | LOW (already dense) |
| triangular | 4 | 36 | 72% | 1.98 | LOW (already dense) |
| chain_1d | 6 | 90 | 99% | 2.50 | LOW (already dense) |
| heavy_hex | 6 | 42 | 100% | 1.90 | LOW (already dense) |
| ladder | 6 | 69 | 72% | 2.33 | LOW (already dense) |
| square | 6 | 121 | 85% | 2.33 | LOW (already dense) |
| triangular | 6 | 145 | 60% | 3.15 | LOW (already dense) |
| chain_1d | 8 | 111 | 96% | 2.50 | LOW (already dense) |
| heavy_hex | 8 | 98 | 55% | 1.90 | LOW (already dense) |
| ladder | 8 | 98 | 53% | 2.62 | LOW (already dense) |
| square | 8 | 119 | 81% | 2.64 | LOW (already dense) |
| chain_1d | 10 | 124 | 85% | 2.50 | LOW (already dense) |
| heavy_hex | 10 | 164 | 90% | 1.90 | LOW (already dense) |
| ladder | 10 | 156 | 37% | 2.81 | LOW (already dense) |
| square | 10 | 94 | 63% | 2.89 | LOW (already dense) |
| chain_1d | 12 | 69 | 91% | 2.50 | LOW (already dense) |
| heavy_hex | 12 | 77 | 87% | 1.93 | LOW (already dense) |
| ladder | 12 | 54 | 48% | 2.98 | LOW (already dense) |
| square | 12 | 72 | 47% | 3.16 | LOW (already dense) |
| heavy_hex | 14 | 59 | 36% | 1.99 | LOW (already dense) |
| square | 14 | 26 | 54% | 3.29 | MEDIUM (expand range) |
| chain_1d | 15 | 173 | 86% | 3.00 | LOW (already dense) |
| chain_1d | 16 | 26 | 73% | 3.00 | MEDIUM (expand range) |
| heavy_hex | 16 | 149 | 83% | 2.02 | LOW (already dense) |
| heavy_hex | 18 | 36 | 100% | 2.50 | LOW (already dense) |
| chain_1d | 20 | 88 | 78% | 3.00 | LOW (already dense) |
| heavy_hex | 20 | 38 | 50% | 2.77 | LOW (already dense) |
| heavy_hex | 21 | 8 | 100% | 4.08 | HIGH (few pts, good quality) |
| heavy_hex | 22 | 20 | 95% | 4.08 | MEDIUM (expand range) |
| heavy_hex | 24 | 20 | 90% | 4.08 | MEDIUM (expand range) |
| heavy_hex | 30 | 27 | 52% | 4.08 | MEDIUM (expand range) |
| chain_1d | 40 | 24 | 42% | 3.75 | MEDIUM (expand range) |
| heavy_hex | 40 | 6 | 83% | 4.08 | HIGH (few pts, good quality) |

<!-- AUTO-GENERATED-END:training_plan -->

---

## Model Zoo Health

Cross-integration: model_zoo entries + NPZ quality tier scores.

<!-- AUTO-GENERATED-BEGIN:zoo_health -->
| Model | Topology | Pts | Pass% | Q.Score | Verified | Recommendation |
|-------|----------|-----|-------|---------|----------|----------------|
| `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+` | chain_1d | 166 | 48% | 0.95 | 141 | ⚠️ INVESTIGATE: good data but low pass_rate |
| `unified_tfim_br_ladder_multiN_4+6+8+10+12+16_` | ladder | 113 | 0% | 0.76 | 95 | ⚠️ INVESTIGATE: good data but low pass_rate |
| `unified_tfim_br_square_multiN_4+6+8+10+12+14+` | square | 212 | 0% | 0.86 | 153 | ⚠️ INVESTIGATE: good data but low pass_rate |
| `unified_tfim_br_triangular_multiN_3+4+6+8+10+` | triangular | 172 | 0% | 0.77 | 107 | ⚠️ INVESTIGATE: good data but low pass_rate |
| `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16` | heavy_hex | 219 | 88% | 0.93 | 166 | ✅ OK |
<!-- AUTO-GENERATED-END:zoo_health -->
