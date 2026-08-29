# Accelerated Cross-N Coverage Analysis

**Fecha**: 2026-08-29 (auto-updated by update_cross_n_coverage.py)
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
| chain_1d | 4,6,8,10,12,14,15,16,20,26,30,40,60 | 1050 | [0.5, 5.5] | 100% | — | 60 |
| heavy_hex | 4,6,8,10,12,14,16,18,20,21,22,24,26,30,40 | 1224 | [0.2, 5.5] | 100% | — | 40 |
| ladder | 4,6,8,10,12,14,16,20,26,30,40 | 624 | [1.4, 5.5] | 94% | — | 40 |
| square | 4,6,8,10,12,14,16,20 | 507 | [1.2, 5.5] | 88% | — | 16 |
| triangular | 3,4,6,8,10,12 | 407 | [0.5, 5.5] | 100% | — | 6 |
<!-- AUTO-GENERATED-END:executive_summary -->

---

## Training Data Health

<!-- AUTO-GENERATED-BEGIN:health -->
| Metric | Value | Status |
|--------|-------|--------|
| Total NPZ files | 62 | |
| Total training points | 3812 | |
| **Quality: Useful** | 45 configs | ✅ |
| **Quality: Insufficient** | 14 configs | ⚠️ |
| **Quality: Not Useful** | 3 configs | ❌ |
| NaN in θ | 0 configs | ✅ |
| Zoo integrity | True | ✅ |
| Zoo missing | 0 | ✅ |
| Zoo orphan checkpoints | 0 | ✅ |
| GT coverage gaps | 3794 uncovered h-points | ⚠️ |
| Stale zoo models | 23 | ⚠️ |
| Need retrain | 0 | ✅ |
| High θ discontinuity (>0.5) | 39 configs | ⚠️ |
| Gap masking detected | 26 configs | ⚠️ |
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

| N | Puntos | h-range | h≤h_c | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|-------|-----------|------------|---------|-------------|
| 4 | 57 | [0.50, 5.00] | ✓ | 49% | 1.28 | 1.57 ⚠️ | div=0.21 |
| 4 | 18 | [0.50, 2.30] | ✓ | 100% | 1.28 | 2.16 ⚠️ | div=0.00 |
| 6 | 111 | [0.50, 5.50] | ✓ | 84% | 1.55 | 1.57 ⚠️ | div=0.13 |
| 8 | 151 | [0.50, 5.50] | ✓ | 75% | 1.72 | 1.64 ⚠️ | div=0.04 |
| 10 | 166 | [0.50, 5.50] | ✓ | 66% | 1.86 | 1.58 ⚠️ | (5% masked) div=0.00 |
| 12 | 109 | [0.50, 5.50] | ✓ | 58% | 1.93 | 1.45 ⚠️ | div=0.13 |
| 14 | 28 | [0.50, 5.00] | ✓ | 14% | 3.25 | 0.15  | div=0.53 |
| 15 | 173 | [1.50, 5.50] | — | 86% | 3.25 | 1.96 ⚠️ | div=0.20 STALE |
| 16 | 49 | [0.50, 5.50] | ✓ | 39% | 3.67 | 0.29  | (14% masked) div=0.18 |
| 20 | 108 | [0.50, 5.50] | ✓ | 64% | 3.67 | 2.70 ⚠️ | (8% masked) div=0.02 |
| 26 | 6 | [3.75, 5.00] | — | 67% | 4.12 | 0.04  | ⚠️ GAP MASK +33% div=0.29 STALE |
| 30 | 28 | [2.50, 5.50] | — | 29% | 4.12 | 0.29  | ⚠️ GAP MASK +46% div=0.04 |
| 40 | 24 | [2.50, 5.50] | — | 42% | 4.12 | 0.30  | ⚠️ GAP MASK +29% div=0.00 |
| 60 | 22 | [2.50, 5.50] | — | 23% | 4.12 | 0.25  | ⚠️ GAP MASK +23% div=0.25 |
<!-- AUTO-GENERATED-END:topo_chain_1d -->

<!-- AUTO-GENERATED-BEGIN:topo_heavy_hex -->
### Heavy Hex

| N | Puntos | h-range | h≤h_c | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|-------|-----------|------------|---------|-------------|
| 4 | 84 | [0.58, 5.50] | ✓ | 94% | 0.96 | 0.08  | div=0.19 STALE |
| 4 | 55 | [0.25, 4.00] | ✓ | 95% | 0.96 | 3.15 ⚠️ | div=0.10 |
| 6 | 42 | [1.90, 4.50] | — | 100% | 1.90 | 0.05  | div=0.25 STALE |
| 6 | 45 | [0.25, 4.00] | ✓ | 67% | 1.90 | 4.18 ⚠️ | div=0.16 |
| 8 | 98 | [0.30, 5.00] | ✓ | 55% | 1.90 | 1.70 ⚠️ | div=0.20 |
| 8 | 45 | [0.25, 4.00] | ✓ | 67% | 1.90 | 3.70 ⚠️ | div=0.18 |
| 10 | 164 | [1.40, 5.50] | — | 90% | 1.90 | 1.62 ⚠️ | div=0.17 STALE |
| 10 | 62 | [0.25, 4.00] | ✓ | 61% | 1.90 | 3.83 ⚠️ | div=0.19 |
| 12 | 77 | [1.40, 4.50] | — | 88% | 1.90 | 1.73 ⚠️ | div=0.13 |
| 12 | 20 | [1.50, 4.00] | — | 100% | 1.90 | 1.28 ⚠️ | div=0.15 |
| 14 | 59 | [0.30, 5.00] | ✓ | 36% | 1.99 | 1.57 ⚠️ | (7% masked) div=0.33 |
| 14 | 35 | [0.85, 4.00] | ✓ | 77% | 1.99 | 6.15 ⚠️ | div=0.05 |
| 16 | 169 | [0.30, 5.50] | ✓ | 72% | 2.02 | 3.19 ⚠️ | (5% masked) div=0.03 |
| 16 | 20 | [1.50, 4.00] | — | 90% | 2.02 | 6.28 ⚠️ | div=0.05 |
| 18 | 71 | [0.30, 5.00] | ✓ | 51% | 2.37 | 6.16 ⚠️ | div=0.24 |
| 20 | 72 | [0.30, 5.00] | ✓ | 26% | 2.77 | 4.03 ⚠️ | div=0.46 |
| 20 | 20 | [1.50, 4.00] | — | 30% | 2.77 | 3.44 ⚠️ | ⚠️ GAP MASK +15% div=0.40 |
| 21 | 8 | [4.08, 5.00] | — | 100% | 4.08 | 1.57 ⚠️ | div=0.25 STALE |
| 22 | 20 | [2.50, 5.00] | — | 95% | 4.08 | 1.57 ⚠️ | div=0.20 STALE |
| 24 | 20 | [2.50, 5.00] | — | 90% | 4.08 | 1.60 ⚠️ | div=0.15 |
| 26 | 5 | [4.00, 5.00] | — | 0% | N/A | 0.05  | ⚠️ GAP MASK +100% div=0.25 STALE |
| 30 | 27 | [2.00, 5.00] | — | 52% | 4.08 | 2.72 ⚠️ | div=0.23 |
| 40 | 6 | [2.50, 4.50] | — | 83% | 4.08 | 1.63 ⚠️ | div=0.08 |
<!-- AUTO-GENERATED-END:topo_heavy_hex -->

<!-- AUTO-GENERATED-BEGIN:topo_ladder -->
### Ladder

| N | Puntos | h-range | h≤h_c | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|-------|-----------|------------|---------|-------------|
| 4 | 45 | [1.36, 5.00] | — | 73% | 1.85 | 0.14  | div=0.30 STALE |
| 6 | 69 | [2.00, 4.80] | — | 72% | 2.41 | 0.62 ⚠️ | (14% masked) div=0.42 STALE |
| 8 | 98 | [1.80, 5.00] | — | 53% | 2.73 | 0.79 ⚠️ | ⚠️ GAP MASK +26% div=0.33 STALE |
| 10 | 156 | [2.00, 5.50] | — | 37% | 2.92 | 0.79 ⚠️ | ⚠️ GAP MASK +40% div=0.31 STALE |
| 12 | 54 | [2.50, 5.50] | — | 48% | 3.38 | 0.40  | ⚠️ GAP MASK +46% div=0.49 STALE |
| 14 | 33 | [2.00, 4.80] | — | 27% | 3.56 | 0.79 ⚠️ | ⚠️ GAP MASK +21% div=0.03 |
| 16 | 101 | [1.80, 5.50] | — | 25% | 3.80 | 0.79 ⚠️ | ⚠️ GAP MASK +42% div=0.21 STALE |
| 20 | 35 | [2.50, 5.35] | — | 23% | 4.04 | 0.17  | ⚠️ GAP MASK +31% div=0.09 |
| 26 | 16 | [2.50, 5.00] | — | 12% | 4.61 | 0.05  | ⚠️ GAP MASK +25% div=0.08 |
| 30 | 15 | [2.50, 5.50] | — | 20% | 4.77 | 0.03  | ⚠️ GAP MASK +20% div=0.05 |
| 40 | 2 | [4.50, 5.00] | — | 0% | N/A | 0.02  | ⚠️ GAP MASK +50% div=0.05 |
<!-- AUTO-GENERATED-END:topo_ladder -->

<!-- AUTO-GENERATED-BEGIN:topo_square -->
### Square

| N | Puntos | h-range | h≤h_c | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|-------|-----------|------------|---------|-------------|
| 4 | 41 | [1.40, 4.50] | — | 85% | 1.84 | 0.17  | div=0.52 STALE |
| 6 | 121 | [1.40, 4.80] | — | 85% | 2.44 | 0.79 ⚠️ | div=0.54 STALE |
| 8 | 119 | [2.00, 5.00] | — | 81% | 2.82 | 1.57 ⚠️ | (7% masked) div=0.54 STALE |
| 10 | 94 | [2.00, 5.50] | — | 63% | 2.92 | 1.57 ⚠️ | (13% masked) div=0.42 STALE |
| 12 | 72 | [1.80, 5.50] | — | 47% | 3.32 | 1.87 ⚠️ | ⚠️ GAP MASK +21% div=0.35 STALE |
| 14 | 26 | [2.50, 5.50] | — | 54% | 3.85 | 1.57 ⚠️ | ⚠️ GAP MASK +19% div=0.40 STALE |
| 16 | 26 | [2.50, 5.00] | — | 15% | 4.17 | 0.14  | ⚠️ GAP MASK +27% div=0.09 |
| 20 | 8 | [1.18, 5.00] | — | 0% | N/A | 0.21  | (12% masked) div=0.21 |
<!-- AUTO-GENERATED-END:topo_square -->

<!-- AUTO-GENERATED-BEGIN:topo_triangular -->
### Triangular

| N | Puntos | h-range | h≤h_c | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|-------|-----------|------------|---------|-------------|
| 3 | 37 | [0.50, 4.50] | ✓ | 100% | 0.50 | 0.04  | div=0.75 STALE |
| 4 | 36 | [1.24, 4.50] | — | 72% | 1.98 | 0.04  | div=0.47 STALE |
| 6 | 145 | [2.13, 5.50] | — | 60% | 3.43 | 0.05  | (10% masked) div=0.45 STALE |
| 8 | 77 | [1.80, 4.80] | — | 9% | 4.55 | 0.01  | ⚠️ GAP MASK +21% div=0.05 |
| 10 | 54 | [2.00, 5.00] | — | 17% | 4.55 | 1.57 ⚠️ | ⚠️ GAP MASK +19% div=0.10 |
| 12 | 58 | [2.50, 5.50] | — | 7% | 5.19 | 0.01  | ⚠️ GAP MASK +26% div=0.08 |
<!-- AUTO-GENERATED-END:topo_triangular -->

---

## h_frontier per Topology

h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):

<!-- AUTO-GENERATED-BEGIN:h_frontier -->
| Topología | N=3 | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=15 | N=16 | N=18 | N=20 | N=21 | N=22 | N=24 | N=26 | N=30 | N=40 | N=60 |
|-----------|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| chain_1d | — | 1.28 | 1.55 | 1.72 | 1.86 | 1.93 | 3.25 | 3.25 | 3.67 | — | 3.67 | — | — | — | 4.12 | 4.12 | 4.12 | 4.12 |
| heavy_hex | — | 0.96 | 1.90 | 1.90 | 1.90 | 1.90 | 1.99 | — | 2.02 | 2.37 | 2.77 | 4.08 | 4.08 | 4.08 | — | 4.08 | 4.08 | — |
| ladder | — | 1.85 | 2.41 | 2.73 | 2.92 | 3.38 | 3.56 | — | 3.80 | — | 4.04 | — | — | — | 4.61 | 4.77 | — | — |
| square | — | 1.84 | 2.44 | 2.82 | 2.92 | 3.32 | 3.85 | — | 4.17 | — | — | — | — | — | — | — | — | — |
| triangular | 0.50 | 1.98 | 3.43 | 4.55 | 4.55 | 5.19 | — | — | — | — | — | — | — | — | — | — | — | — |
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

**Model**: `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1`

| N | h-range | Pts | ΔE/gap | |ΔE| | Pass@dual | Speedup |
|---|---------|-----|--------|------|-----------|---------|
| 10 | [0.5, 3.0] | 41 | 74.0910 | 0.694 | 4/41 | 2367× |
| 12 | [2.5, 3.0] | 2 | 0.0504 | 0.170 | 0/2 | — |
| 16 | [2.5, 5.0] | 10 | 0.0575 | 0.335 | 2/10 | 3488× |
| 20 | [0.5, 5.0] | 36 | 2.1030 | 1.186 | 6/36 | 4600× |
| 30 | [2.5, 5.5] | 28 | 0.0351 | 0.213 | 8/28 | 20580× |
| 40 | [2.5, 5.5] | 24 | 0.0405 | 0.236 | 10/24 | 11440× |
| 60 | [2.5, 5.5] | 22 | 0.0725 | 0.447 | 5/22 | 13920× |
| 80 | [2.5, 5.0] | 8 | 0.1658 | 0.708 | 0/8 | — |
| 100 | [2.5, 5.5] | 19 | 0.1369 | 0.798 | 2/19 | 22880× |
| 150 | [4.0, 5.0] | 3 | 0.7841 | 5.376 | 0/3 | — |
| 200 | [4.0, 5.0] | 3 | 1.0472 | 7.180 | 0/3 | — |

### heavy_hex

**Model**: `unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE| | Pass@dual | Speedup |
|---|---------|-----|--------|------|-----------|---------|
| 8 | [2.5, 5.0] | 47 | 0.0302 | 0.187 | 25/47 | 1942× |
| 10 | [2.5, 5.0] | 57 | 0.0143 | 0.079 | 47/57 | 2490× |
| 12 | [2.5, 5.0] | 25 | 0.1251 | 0.608 | 0/25 | — |
| 14 | [2.5, 5.0] | 54 | 1.9993 | 8.150 | 12/54 | 3325× |
| 16 | [2.5, 5.0] | 47 | 0.0480 | 0.225 | 6/47 | 3595× |
| 18 | [2.5, 5.0] | 43 | 0.2323 | 1.131 | 3/43 | — |
| 20 | [2.0, 5.0] | 96 | 0.1771 | 0.292 | 31/96 | 16860× |
| 21 | [2.5, 5.0] | 14 | 0.3410 | 0.932 | 0/14 | 4882× |
| 22 | [2.5, 5.0] | 31 | 0.2026 | 0.560 | 6/31 | — |
| 24 | [2.5, 5.0] | 47 | 0.2864 | 0.803 | 11/47 | 10352× |
| 26 | [2.5, 5.0] | 27 | 0.2507 | 0.689 | 11/27 | 10868× |
| 30 | [2.0, 5.0] | 51 | 0.3369 | 0.338 | 33/51 | 40208× |
| 32 | [2.5, 5.0] | 10 | 0.3840 | 0.648 | 6/10 | — |
| 40 | [2.5, 5.0] | 33 | 0.2009 | 0.581 | 13/33 | 9333× |
| 50 | [2.5, 5.0] | 6 | 0.2856 | 0.919 | 2/6 | 11100× |
| 60 | [2.5, 5.0] | 6 | 0.4976 | 1.753 | 1/6 | 13480× |

### ladder

**Model**: `data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1_v2.pt`

| N | h-range | Pts | ΔE/gap | |ΔE| | Pass@dual | Speedup |
|---|---------|-----|--------|------|-----------|---------|
| 16 | [2.5, 5.0] | 6 | 0.0784 | 0.224 | 0/6 | 10033× |
| 20 | [2.5, 5.5] | 24 | 0.2592 | 0.145 | 6/24 | 13708× |
| 26 | [2.5, 5.0] | 14 | 0.5173 | 0.233 | 2/14 | 7211× |
| 30 | [2.5, 5.5] | 14 | 0.5130 | 0.223 | 3/14 | 8165× |
| 40 | [2.5, 5.0] | 6 | 1.5579 | 0.395 | 0/6 | 11154× |

### square

**Model**: `data/model_zoo/checkpoints/unified_tfim_br_square_fromMT_4+6+8+10+12+14+16_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE| | Pass@dual | Speedup |
|---|---------|-----|--------|------|-----------|---------|
| 16 | [2.5, 5.0] | 26 | 0.0813 | 0.264 | 4/26 | 10824× |
| 20 | [2.5, 5.0] | 26 | 0.7914 | 0.351 | 1/26 | 17433× |
| 30 | [2.5, 5.0] | 13 | 2.4057 | 0.890 | 0/13 | 8773× |

### triangular

**Model**: `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | h-range | Pts | ΔE/gap | |ΔE| | Pass@dual | Speedup |
|---|---------|-----|--------|------|-----------|---------|
| 12 | [2.5, 5.0] | 10 | 1.7850 | 0.879 | 0/10 | 3588× |
| 16 | [2.5, 5.0] | 10 | 28.7030 | 2.603 | 0/10 | 6197× |
| 24 | [2.5, 5.0] | 10 | 23.4736 | 6.145 | 0/10 | 9520× |

### Extensive Scaling Summary

| Topology | N range | |ΔE| (mean) | Variation | Scaling |
|----------|---------|-------------|-----------|---------|
| chain_1d | 10–200 | 1.576 | 42.3× | ⚠️ degrading |
| heavy_hex | 8–60 | 1.118 | 103.5× | ⚠️ degrading |
| ladder | 16–40 | 0.244 | 2.7× | ✅ extensive |
| square | 16–30 | 0.502 | 3.4× | ⚠️ degrading |
| triangular | 12–24 | 3.209 | 7.0× | ⚠️ degrading |

### MPNN vs Random VQE vs Ground Truth

Comparison at same h-points. MPNN: 1 forward pass (0 QPU). VQE: L-BFGS-B with random init.

| Topology | N | MPNN ΔE/gap | VQE ΔE/gap | MPNN |ΔE| | MPNN wins? | Speedup | VQE evals |
|----------|---|-------------|------------|---------|-------|---------|-----------|
| chain_1d | 10 | 0.0122 | 2.2902 | 0.085 | ✅ | 2367× | 14,200 |
| chain_1d | 16 | 0.0206 | 2.3192 | 0.143 | ✅ | 3488× | 20,928 |
| chain_1d | 20 | 0.0090 | 2.0355 | 0.036 | ✅ | 4600× | 27,600 |
| chain_1d | 30 | 0.0130 | 2.3093 | 0.081 | ✅ | 6960× | 20,880 |
| chain_1d | 40 | 0.0191 | 7.6963 | 0.075 | ✅ | 8747× | 52,480 |
| chain_1d | 60 | 0.0292 | 7.0227 | 0.115 | ✅ | 13480× | 80,880 |
| chain_1d | 100 | 0.1017 | 14.0382 | 0.685 | ✅ | 22880× | 114,400 |
| heavy_hex | 8 | 0.0312 | 0.0026 | 0.191 | ❌ | 1942× | 38,848 |
| heavy_hex | 10 | 0.0048 | 0.0016 | 0.033 | ❌ | 2427× | 14,560 |
| heavy_hex | 14 | 0.0215 | 1.4948 | 0.136 | ✅ | 3325× | 26,600 |
| heavy_hex | 16 | 0.0162 | 2.3631 | 0.108 | ✅ | 3541× | 21,248 |
| heavy_hex | 20 | 0.0173 | 4.1020 | 0.033 | ✅ | 4453× | 26,720 |
| heavy_hex | 21 | 0.2853 | 3.5297 | 0.993 | ✅ | 4882× | 39,060 |
| heavy_hex | 24 | 0.0235 | 4.1442 | 0.044 | ✅ | 10352× | 62,112 |
| heavy_hex | 26 | 0.0215 | 4.0195 | 0.040 | ✅ | 10868× | 65,208 |
| heavy_hex | 30 | 0.0147 | 1.2455 | 0.063 | ✅ | 39660× | 237,960 |
| heavy_hex | 40 | 0.0338 | 7.1373 | 0.060 | ✅ | 9333× | 56,000 |
| heavy_hex | 50 | 2.5883 | 12.1727 | 6.135 | ✅ | 11100× | 66,600 |
| heavy_hex | 60 | 3.3160 | 13.3788 | 7.948 | ✅ | 13480× | 80,880 |
| ladder | 10 | 0.0130 | 0.0120 | 0.064 | ❌ | 2832× | 16,992 |
| ladder | 16 | 0.0208 | 2.7202 | 0.101 | ✅ | 4290× | 25,740 |
| ladder | 20 | 0.0236 | 3.1949 | 0.106 | ✅ | 13708× | 54,831 |
| ladder | 26 | 0.5843 | 0.5997 | 0.243 | ✅ | 7142× | 71,424 |
| ladder | 30 | 0.8963 | 17.4153 | 0.298 | ✅ | 8165× | 48,988 |
| ladder | 40 | 1.5579 | 41.6156 | 0.395 | ✅ | 11154× | 66,924 |
| square | 10 | 0.0509 | 0.0089 | 0.269 | ❌ | 2608× | 15,648 |
| square | 16 | 0.0640 | 3.4592 | 0.205 | ✅ | 4688× | 28,126 |
| square | 20 | 0.2791 | 0.1431 | 0.438 | ❌ | 17433× | 69,732 |
| square | 30 | 2.1018 | 29.8002 | 0.543 | ✅ | 8773× | 52,640 |
| triangular | 10 | 0.0234 | 0.0214 | 0.100 | ❌ | 3766× | 15,066 |
| triangular | 12 | 0.0501 | 0.0467 | 0.177 | ❌ | 3588× | 14,352 |
| triangular | 16 | 0.4003 | 0.4003 | 0.568 | ✅ | 6197× | 37,180 |
| triangular | 24 | 4.0032 | 4.0033 | 1.048 | ✅ | 9520× | 57,120 |

**MPNN win rate**: 26/33 (78%)

**Speedup range**: 1942× – 40208×

<!-- AUTO-GENERATED-END:large_n_extrapolation -->

---

## Quality Tier Distribution

Data quality breakdown by tier (verified=VQE-converged, approximate=MPNN-predicted, unverified=legacy):

<!-- AUTO-GENERATED-BEGIN:tier_breakdown -->
| Topology | Total pts | Verified | Approximate | Unverified |
|----------|-----------|----------|-------------|------------|
| chain_1d | 1050 | 839 (79%) | 144 (13%) | 67 (6%) |
| heavy_hex | 1224 | 1080 (88%) | 82 (6%) | 62 (5%) |
| ladder | 624 | 261 (41%) | 259 (41%) | 104 (16%) |
| square | 507 | 364 (71%) | 96 (18%) | 47 (9%) |
| triangular | 407 | 235 (57%) | 96 (23%) | 76 (18%) |
<!-- AUTO-GENERATED-END:tier_breakdown -->

---

<!-- AUTO-GENERATED-BEGIN:training_plan -->
## Training Plan (auto-generated)

**Total configs**: 62 | ✅ Useful: 45 | ⚠️ Insufficient: 14 | ❌ Not useful: 3

### ⚠️ IMPROVE — Insufficient signal (need more good points)

Run iterative-improve to densify these configs above the frontier:

| File | Topology | N | Pts | Dual pass | h_frontier | Action |
|------|----------|---|-----|-----------|------------|--------|
| `chain_1d_N14_p1.npz` | chain_1d | 14 | 28 | 14% | 3.25 | iterative-improve h≥3.5 |
| `chain_1d_N26_p1.npz` | chain_1d | 26 | 6 | 67% | 4.12 | iterative-improve h≥4.3 |
| `chain_1d_N30_p1.npz` | chain_1d | 30 | 28 | 29% | 4.12 | iterative-improve h≥4.3 |
| `chain_1d_N60_p1.npz` | chain_1d | 60 | 22 | 23% | 4.12 | iterative-improve h≥4.3 |
| `heavy_hex_N20_p1.npz` | heavy_hex | 20 | 72 | 26% | 2.77 | iterative-improve h≥3.0 |
| `ladder_N14_p1.npz` | ladder | 14 | 33 | 27% | 3.56 | iterative-improve h≥3.8 |
| `ladder_N16_p1.npz` | ladder | 16 | 101 | 25% | 3.80 | iterative-improve h≥4.0 |
| `ladder_N20_p1.npz` | ladder | 20 | 35 | 23% | 4.04 | iterative-improve h≥4.2 |
| `ladder_N26_p1.npz` | ladder | 26 | 16 | 12% | 4.61 | iterative-improve h≥4.8 |
| `ladder_N30_p1.npz` | ladder | 30 | 15 | 20% | 4.77 | iterative-improve h≥5.0 |
| `square_N16_p1.npz` | square | 16 | 26 | 15% | 4.17 | iterative-improve h≥4.4 |
| `triangular_N8_p1.npz` | triangular | 8 | 77 | 9% | 4.55 | iterative-improve h≥4.7 |
| `triangular_N10_p1.npz` | triangular | 10 | 54 | 17% | 4.55 | iterative-improve h≥4.7 |
| `triangular_N12_p1.npz` | triangular | 12 | 58 | 7% | 5.19 | iterative-improve h≥5.4 |

### ✅ EXPAND — Useful configs (add more h-points for better generalization)

| Topology | N | Pts | Dual pass | h_frontier | Priority |
|----------|---|-----|-----------|------------|----------|
| triangular | 3 | 37 | 100% | 0.50 | LOW (already dense) |
| chain_1d | 4 | 57 | 49% | 1.28 | LOW (already dense) |
| chain_1d | 4 | 18 | 100% | 1.28 | HIGH (few pts, good quality) |
| heavy_hex | 4 | 84 | 94% | 0.96 | LOW (already dense) |
| heavy_hex | 4 | 55 | 95% | 0.96 | LOW (already dense) |
| ladder | 4 | 45 | 73% | 1.85 | LOW (already dense) |
| square | 4 | 41 | 85% | 1.84 | LOW (already dense) |
| triangular | 4 | 36 | 72% | 1.98 | LOW (already dense) |
| chain_1d | 6 | 111 | 84% | 1.55 | LOW (already dense) |
| heavy_hex | 6 | 42 | 100% | 1.90 | LOW (already dense) |
| heavy_hex | 6 | 45 | 67% | 1.90 | LOW (already dense) |
| ladder | 6 | 69 | 72% | 2.41 | LOW (already dense) |
| square | 6 | 121 | 85% | 2.44 | LOW (already dense) |
| triangular | 6 | 145 | 60% | 3.43 | LOW (already dense) |
| chain_1d | 8 | 151 | 75% | 1.72 | LOW (already dense) |
| heavy_hex | 8 | 98 | 55% | 1.90 | LOW (already dense) |
| heavy_hex | 8 | 45 | 67% | 1.90 | LOW (already dense) |
| ladder | 8 | 98 | 53% | 2.73 | LOW (already dense) |
| square | 8 | 119 | 81% | 2.82 | LOW (already dense) |
| chain_1d | 10 | 166 | 66% | 1.86 | LOW (already dense) |
| heavy_hex | 10 | 164 | 90% | 1.90 | LOW (already dense) |
| heavy_hex | 10 | 62 | 61% | 1.90 | LOW (already dense) |
| ladder | 10 | 156 | 37% | 2.92 | LOW (already dense) |
| square | 10 | 94 | 63% | 2.92 | LOW (already dense) |
| chain_1d | 12 | 109 | 58% | 1.93 | LOW (already dense) |
| heavy_hex | 12 | 77 | 88% | 1.90 | LOW (already dense) |
| heavy_hex | 12 | 20 | 100% | 1.90 | MEDIUM (expand range) |
| ladder | 12 | 54 | 48% | 3.38 | LOW (already dense) |
| square | 12 | 72 | 47% | 3.32 | LOW (already dense) |
| heavy_hex | 14 | 59 | 36% | 1.99 | LOW (already dense) |
| heavy_hex | 14 | 35 | 77% | 1.99 | LOW (already dense) |
| square | 14 | 26 | 54% | 3.85 | MEDIUM (expand range) |
| chain_1d | 15 | 173 | 86% | 3.25 | LOW (already dense) |
| chain_1d | 16 | 49 | 39% | 3.67 | LOW (already dense) |
| heavy_hex | 16 | 169 | 72% | 2.02 | LOW (already dense) |
| heavy_hex | 16 | 20 | 90% | 2.02 | MEDIUM (expand range) |
| heavy_hex | 18 | 71 | 51% | 2.37 | LOW (already dense) |
| chain_1d | 20 | 108 | 64% | 3.67 | LOW (already dense) |
| heavy_hex | 20 | 20 | 30% | 2.77 | MEDIUM (expand range) |
| heavy_hex | 21 | 8 | 100% | 4.08 | HIGH (few pts, good quality) |
| heavy_hex | 22 | 20 | 95% | 4.08 | MEDIUM (expand range) |
| heavy_hex | 24 | 20 | 90% | 4.08 | MEDIUM (expand range) |
| heavy_hex | 30 | 27 | 52% | 4.08 | MEDIUM (expand range) |
| chain_1d | 40 | 24 | 42% | 4.12 | MEDIUM (expand range) |
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
