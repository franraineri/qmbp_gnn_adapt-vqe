# Accelerated Cross-N Coverage Analysis

**Fecha**: 2026-08-12 (auto-updated by update_cross_n_coverage.py)
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
| chain_1d | 6,8,10,12,15,20 | 333 | [1.5, 5.5] | 100% | — | 20 |
| heavy_hex | 4,6,10,12,16 | 363 | [0.6, 4.5] | 91% | — | 16 |
| ladder | 4,6,8,10,12,16 | 291 | [1.4, 5.5] | 100% | — | 16 |
| square | 4,6,8,10,12,14,16 | 315 | [1.4, 5.5] | 100% | — | 16 |
| triangular | 3,4,6,8,10,12 | 244 | [0.5, 5.5] | 86% | — | 12 |
<!-- AUTO-GENERATED-END:executive_summary -->

---

## Training Data Health

<!-- AUTO-GENERATED-BEGIN:health -->
| Metric | Value | Status |
|--------|-------|--------|
| Total NPZ files | 30 | |
| Total training points | 1546 | |
| **Quality: Useful** | 23 configs | ✅ |
| **Quality: Insufficient** | 7 configs | ⚠️ |
| **Quality: Not Useful** | 0 configs | ✅ |
| NaN in θ | 0 configs | ✅ |
| Zoo integrity | True | ✅ |
| Zoo missing | 0 | ✅ |
| Zoo orphan checkpoints | 8 | ⚠️ cleanup needed |
| GT coverage gaps | 350 uncovered h-points | ⚠️ |
| Stale zoo models | 6 | ⚠️ |
| Need retrain | 10 | 🔄 |
| High θ discontinuity (>0.5) | 12 configs | ⚠️ |
| Gap masking detected | 16 configs | ⚠️ |
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
| ladder | 16 | 100% | 8% | 92% |
| square | 14 | 100% | 38% | 62% |
| ladder | 12 | 100% | 41% | 59% |
| triangular | 8 | 86% | 29% | 57% |
| ladder | 10 | 67% | 12% | 55% |
| triangular | 12 | 60% | 20% | 40% |
| ladder | 8 | 86% | 47% | 39% |
| square | 16 | 44% | 9% | 34% |
| triangular | 10 | 48% | 15% | 32% |
| square | 12 | 76% | 48% | 28% |
| ladder | 6 | 84% | 67% | 18% |
| heavy_hex | 6 | 88% | 71% | 17% |
| square | 10 | 66% | 53% | 13% |
| triangular | 6 | 69% | 58% | 11% |
| chain_1d | 10 | 100% | 89% | 11% |
| chain_1d | 15 | 90% | 79% | 11% |
<!-- AUTO-GENERATED-END:gap_masking -->

---

## Detalle por Topología

<!-- AUTO-GENERATED-BEGIN:topo_chain_1d -->
### Chain 1D

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 6 | 40 | [1.50, 5.50] | 98% | 98% | 1.56 | 0.79 ⚠️ | div=0.03 |
| 8 | 69 | [1.50, 5.50] | 96% | 96% | 1.72 | 0.15  | div=0.04 |
| 10 | 45 | [2.00, 5.50] | 100% | 89% | 2.00 | 1.57 ⚠️ | (11% masked) div=0.00 |
| 12 | 69 | [1.50, 5.50] | 91% | 91% | 1.94 | 1.45 ⚠️ | div=0.09 |
| 15 | 81 | [1.50, 5.50] | 90% | 79% | 2.03 | 0.47  | (11% masked) div=0.10 |
| 20 | 29 | [1.50, 5.50] | 83% | 79% | 2.21 | 0.12  | div=0.17 |
<!-- AUTO-GENERATED-END:topo_chain_1d -->

<!-- AUTO-GENERATED-BEGIN:topo_heavy_hex -->
### Heavy Hex

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 15 | [0.58, 2.50] | 67% | 67% | 0.96 | 0.08  | div=0.33 |
| 6 | 42 | [1.90, 4.50] | 88% | 71% | 2.65 | 0.05  | ⚠️ GAP MASK +17% div=0.12 |
| 10 | 131 | [1.40, 4.50] | 91% | 91% | 1.76 | 1.62 ⚠️ | div=0.09 |
| 12 | 77 | [1.40, 4.50] | 82% | 81% | 1.93 | 1.73 ⚠️ | div=0.18 |
| 16 | 98 | [1.40, 4.50] | 88% | 87% | 2.02 | 1.24 ⚠️ | div=0.12 |
<!-- AUTO-GENERATED-END:topo_heavy_hex -->

<!-- AUTO-GENERATED-BEGIN:topo_ladder -->
### Ladder

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 13 | [1.36, 2.08] | 23% | 23% | 1.85 | 0.11  | — |
| 6 | 57 | [2.00, 4.80] | 84% | 67% | 2.33 | 0.40  | ⚠️ GAP MASK +18% |
| 8 | 57 | [2.08, 4.80] | 86% | 47% | 2.62 | 0.66 ⚠️ | ⚠️ GAP MASK +39% |
| 10 | 101 | [2.00, 4.80] | 67% | 12% | 2.87 | 0.79 ⚠️ | ⚠️ GAP MASK +55% |
| 12 | 39 | [3.06, 5.50] | 100% | 41% | 3.06 | 0.16  | ⚠️ GAP MASK +59% |
| 16 | 24 | [3.28, 5.50] | 100% | 8% | 3.28 | 0.20  | ⚠️ GAP MASK +92% |
<!-- AUTO-GENERATED-END:topo_ladder -->

<!-- AUTO-GENERATED-BEGIN:topo_square -->
### Square

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 41 | [1.40, 4.50] | 85% | 85% | 1.84 | 0.17  | div=0.37 STALE |
| 6 | 64 | [1.40, 4.50] | 77% | 73% | 2.33 | 0.79 ⚠️ | div=0.29 STALE |
| 8 | 91 | [2.00, 5.00] | 84% | 77% | 2.64 | 1.57 ⚠️ | (7% masked) div=0.36 STALE |
| 10 | 38 | [2.00, 4.50] | 66% | 53% | 2.84 | 0.01  | (13% masked) div=0.18 STALE |
| 12 | 25 | [2.40, 4.50] | 76% | 48% | 3.16 | 1.64 ⚠️ | ⚠️ GAP MASK +28% div=0.28 STALE |
| 14 | 24 | [3.38, 5.50] | 100% | 38% | 3.38 | 0.00  | ⚠️ GAP MASK +62% div=0.52 STALE |
| 16 | 32 | [2.70, 4.37] | 44% | 9% | 3.50 | 0.00  | ⚠️ GAP MASK +34% div=0.04 |
<!-- AUTO-GENERATED-END:topo_square -->

<!-- AUTO-GENERATED-BEGIN:topo_triangular -->
### Triangular

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 3 | 37 | [0.50, 4.50] | 59% | 59% | N/A | 0.04  | — |
| 4 | 36 | [1.24, 4.50] | 72% | 72% | 1.98 | 0.04  | — |
| 6 | 97 | [2.13, 5.50] | 69% | 58% | 3.15 | 0.02  | (11% masked) |
| 8 | 14 | [3.86, 4.80] | 86% | 29% | 3.98 | 0.00  | ⚠️ GAP MASK +57% |
| 10 | 40 | [2.00, 4.80] | 48% | 15% | 3.94 | 1.57 ⚠️ | ⚠️ GAP MASK +32% |
| 12 | 20 | [3.81, 5.50] | 60% | 20% | 4.44 | 0.00  | ⚠️ GAP MASK +40% |
<!-- AUTO-GENERATED-END:topo_triangular -->

---

## h_frontier per Topology

h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):

<!-- AUTO-GENERATED-BEGIN:h_frontier -->
| Topología | N=4 | N=6 | N=8 | N=10 | N=12 | N=14 | N=15 | N=16 | N=20 |
|-----------|--- | --- | --- | --- | --- | --- | --- | --- | ---|
| chain_1d | — | 1.56 | 1.72 | 2.00 | 1.94 | — | 2.03 | — | 2.21 |
| heavy_hex | 0.96 | 2.65 | — | 1.76 | 1.93 | — | — | 2.02 | — |
| ladder | 1.85 | 2.33 | 2.62 | 2.87 | 3.06 | — | — | 3.28 | — |
| square | 1.84 | 2.33 | 2.64 | 2.84 | 3.16 | 3.38 | — | 3.50 | — |
| triangular | 1.98 | 3.15 | 3.98 | 3.94 | 4.44 | — | — | — | — |
<!-- AUTO-GENERATED-END:h_frontier -->

---

## Cross-N Transfer Summary

<!-- AUTO-GENERATED-BEGIN:cross_n_transfer -->
| Topology | n_max_viable | Best pass@5% | Best cross-N source |
|----------|-------------|-------------|---------------------|
| chain_1d | 20 | 100% | train_n=10 (@10%=100%) |
| heavy_hex | 16 | 91% | train_n=10 (@10%=93%) |
| ladder | 16 | 100% | train_n=16 (@10%=100%) |
| square | 16 | 100% | no data |
| triangular | 12 | 86% | no data |
<!-- AUTO-GENERATED-END:cross_n_transfer -->

---

<!-- AUTO-GENERATED-BEGIN:large_n_extrapolation -->
## Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Model trained on N≤20,
evaluated at N=30-100 via MPS backend. Speedup = VQE_evals / MPNN_evals.

### chain_1d

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.0] | 8 | 0.0117 | 8.11e-03 | 8/8 | 6/8 | 2367× |
| 16 | [3.5, 5.0] | 6 | 0.0206 | 8.95e-03 | 6/6 | 2/6 | 3488× |
| 20 | [3.5, 5.0] | 6 | 0.0263 | 9.10e-03 | 6/6 | 2/6 | 4480× |
| 30 | [2.5, 5.5] | 18 | 0.0482 | 9.40e-03 | 13/18 | 1/18 | 20580× |
| 40 | [2.5, 5.5] | 12 | 0.0617 | 8.55e-03 | 8/12 | 0/12 | 8600× |
| 60 | [2.5, 5.5] | 12 | 0.1114 | 1.10e-02 | 2/12 | 0/12 | 13920× |
| 100 | [3.5, 5.5] | 12 | 0.1347 | 9.87e-03 | 0/12 | 0/12 | 22880× |

### heavy_hex

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.0] | 8 | 0.0046 | 3.08e-03 | 8/8 | 8/8 | 2427× |
| 16 | [3.0, 5.0] | 15 | 0.0173 | 6.31e-03 | 14/15 | 9/15 | 3595× |
| 20 | [2.5, 5.0] | 23 | 0.4054 | 1.33e-02 | 0/23 | 0/23 | 4480× |
| 30 | [2.5, 4.5] | 11 | 1.9012 | 5.00e-02 | 0/11 | 0/11 | 6920× |

### ladder

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.0] | 6 | 0.0130 | 6.37e-03 | 6/6 | 5/6 | 2832× |
| 16 | [3.5, 5.5] | 12 | 0.0196 | 6.11e-03 | 11/12 | 7/12 | 10033× |
| 20 | [3.5, 5.5] | 14 | 0.2161 | 5.44e-03 | 6/14 | 3/14 | 13708× |
| 30 | [2.5, 5.5] | 7 | 1.3649 | 1.31e-02 | 2/7 | 1/7 | — |

### square

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.0] | 6 | 0.0509 | 2.69e-02 | 4/6 | 0/6 | 2608× |
| 16 | [3.0, 5.0] | 15 | 0.0784 | 2.09e-02 | 4/15 | 0/15 | 10824× |
| 20 | [3.0, 5.0] | 15 | 1.0205 | 2.19e-02 | 1/15 | 0/15 | 17433× |
| 30 | [3.5, 4.5] | 3 | 1.4906 | 1.04e-02 | 0/3 | 0/3 | — |

### triangular

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.5] | 14 | 0.0360 | 1.28e-02 | 11/14 | 6/14 | 3766× |
| 12 | [3.5, 5.5] | 14 | 0.0825 | 1.93e-02 | 7/14 | 1/14 | 3588× |

### Extensive Scaling Summary

| Topology | N range | |ΔE|/N (mean) | Variation | Scaling |
|----------|---------|--------------|-----------|---------|
| chain_1d | 10–100 | 9.29e-03 | 1.4× | ✅ extensive |
| heavy_hex | 10–30 | 1.82e-02 | 16.2× | ⚠️ degrading |
| ladder | 10–30 | 7.76e-03 | 2.4× | ✅ extensive |
| square | 10–30 | 2.00e-02 | 2.6× | ✅ extensive |
| triangular | 10–12 | 1.60e-02 | 1.5× | ✅ extensive |

### MPNN vs Random VQE vs Ground Truth

Comparison at same h-points. MPNN: 1 forward pass (0 QPU). VQE: L-BFGS-B with random init.

| Topology | N | MPNN ΔE/gap | VQE ΔE/gap | MPNN |ΔE|/N | MPNN wins? | Speedup | VQE evals |
|----------|---|-------------|------------|------|-------|---------|-----------|
| chain_1d | 10 | 0.0122 | 2.2902 | 8.53e-03 | ✅ | 2367× | 14,200 |
| chain_1d | 16 | 0.0206 | 2.3192 | 8.95e-03 | ✅ | 3488× | 20,928 |
| chain_1d | 20 | 0.0263 | 2.3289 | 9.10e-03 | ✅ | 4480× | 26,880 |
| chain_1d | 30 | 0.0294 | 2.3794 | 6.01e-03 | ✅ | 13340× | 80,040 |
| chain_1d | 40 | 0.0377 | 7.6010 | 6.45e-03 | ✅ | 8600× | 34,400 |
| chain_1d | 60 | 0.0572 | 5.8887 | 6.52e-03 | ✅ | 13920× | 55,680 |
| chain_1d | 100 | 0.1017 | 14.0382 | 6.85e-03 | ✅ | 22880× | 114,400 |
| heavy_hex | 10 | 0.0048 | 0.0016 | 3.28e-03 | ❌ | 2427× | 14,560 |
| heavy_hex | 16 | 0.0162 | 2.3631 | 6.73e-03 | ✅ | 3541× | 21,248 |
| heavy_hex | 20 | 0.2166 | 8.7539 | 1.38e-02 | ✅ | 4424× | 44,240 |
| heavy_hex | 30 | 1.7908 | 16.7806 | 6.60e-02 | ✅ | 6920× | 41,520 |
| ladder | 10 | 0.0130 | 0.0120 | 6.37e-03 | ❌ | 2832× | 16,992 |
| ladder | 16 | 0.0208 | 2.7202 | 6.29e-03 | ✅ | 4290× | 25,740 |
| ladder | 20 | 0.0236 | 3.1949 | 5.29e-03 | ✅ | 13708× | 54,831 |
| square | 10 | 0.0509 | 0.0089 | 2.69e-02 | ❌ | 2608× | 15,648 |
| square | 16 | 0.0640 | 3.4592 | 1.28e-02 | ✅ | 4688× | 28,126 |
| square | 20 | 0.2791 | 0.1431 | 2.19e-02 | ❌ | 17433× | 69,732 |
| triangular | 10 | 0.0234 | 0.0214 | 9.96e-03 | ❌ | 3766× | 15,066 |
| triangular | 12 | 0.0501 | 0.0467 | 1.48e-02 | ❌ | 3588× | 14,352 |

**MPNN win rate**: 13/19 (68%)

**Speedup range**: 2367× – 22880×

<!-- AUTO-GENERATED-END:large_n_extrapolation -->

---

## Quality Tier Distribution

Data quality breakdown by tier (verified=VQE-converged, approximate=MPNN-predicted, unverified=legacy):

<!-- AUTO-GENERATED-BEGIN:tier_breakdown -->
| Topology | Total pts | Verified | Approximate | Unverified |
|----------|-----------|----------|-------------|------------|
| chain_1d | 333 | 270 (81%) | 63 (18%) | 0 (0%) |
| heavy_hex | 363 | 254 (69%) | 106 (29%) | 3 (0%) |
| ladder | 291 | 108 (37%) | 162 (55%) | 21 (7%) |
| square | 315 | 210 (66%) | 79 (25%) | 26 (8%) |
| triangular | 244 | 121 (49%) | 81 (33%) | 42 (17%) |
<!-- AUTO-GENERATED-END:tier_breakdown -->

---

<!-- AUTO-GENERATED-BEGIN:training_plan -->
## Training Plan (auto-generated)

**Total configs**: 30 | ✅ Useful: 23 | ⚠️ Insufficient: 7 | ❌ Not useful: 0

### ⚠️ IMPROVE — Insufficient signal (need more good points)

Run iterative-improve to densify these configs above the frontier:

| File | Topology | N | Pts | Dual pass | h_frontier | Action |
|------|----------|---|-----|-----------|------------|--------|
| `ladder_N4_p1.npz` | ladder | 4 | 13 | 23% | 1.85 | iterative-improve h≥2.0 |
| `ladder_N10_p1.npz` | ladder | 10 | 101 | 12% | 2.87 | iterative-improve h≥3.1 |
| `ladder_N16_p1.npz` | ladder | 16 | 24 | 8% | 3.28 | iterative-improve h≥3.5 |
| `square_N16_p1.npz` | square | 16 | 32 | 9% | 3.50 | iterative-improve h≥3.7 |
| `triangular_N8_p1.npz` | triangular | 8 | 14 | 29% | 3.98 | iterative-improve h≥4.2 |
| `triangular_N10_p1.npz` | triangular | 10 | 40 | 15% | 3.94 | iterative-improve h≥4.1 |
| `triangular_N12_p1.npz` | triangular | 12 | 20 | 20% | 4.44 | iterative-improve h≥4.6 |

### ✅ EXPAND — Useful configs (add more h-points for better generalization)

| Topology | N | Pts | Dual pass | h_frontier | Priority |
|----------|---|-----|-----------|------------|----------|
| triangular | 3 | 37 | 59% | N/A | LOW (already dense) |
| heavy_hex | 4 | 15 | 67% | 0.96 | MEDIUM (expand range) |
| square | 4 | 41 | 85% | 1.84 | LOW (already dense) |
| triangular | 4 | 36 | 72% | 1.98 | LOW (already dense) |
| chain_1d | 6 | 40 | 98% | 1.56 | LOW (already dense) |
| heavy_hex | 6 | 42 | 71% | 2.65 | LOW (already dense) |
| ladder | 6 | 57 | 67% | 2.33 | LOW (already dense) |
| square | 6 | 64 | 73% | 2.33 | LOW (already dense) |
| triangular | 6 | 97 | 58% | 3.15 | LOW (already dense) |
| chain_1d | 8 | 69 | 96% | 1.72 | LOW (already dense) |
| ladder | 8 | 57 | 47% | 2.62 | LOW (already dense) |
| square | 8 | 91 | 77% | 2.64 | LOW (already dense) |
| chain_1d | 10 | 45 | 89% | 2.00 | LOW (already dense) |
| heavy_hex | 10 | 131 | 91% | 1.76 | LOW (already dense) |
| square | 10 | 38 | 53% | 2.84 | LOW (already dense) |
| chain_1d | 12 | 69 | 91% | 1.94 | LOW (already dense) |
| heavy_hex | 12 | 77 | 81% | 1.93 | LOW (already dense) |
| ladder | 12 | 39 | 41% | 3.06 | LOW (already dense) |
| square | 12 | 25 | 48% | 3.16 | MEDIUM (expand range) |
| square | 14 | 24 | 38% | 3.38 | MEDIUM (expand range) |
| chain_1d | 15 | 81 | 79% | 2.03 | LOW (already dense) |
| heavy_hex | 16 | 98 | 87% | 2.02 | LOW (already dense) |
| chain_1d | 20 | 29 | 79% | 2.21 | MEDIUM (expand range) |

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
