# Accelerated Cross-N Coverage Analysis

**Fecha**: 2026-08-11 (auto-updated by update_cross_n_coverage.py)
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
| chain_1d | 6,8,10,12,15,20 | 153 | [1.5, 5.5] | 100% | — | 20 |
| heavy_hex | 4,6,10,12,16 | 214 | [0.6, 4.5] | 98% | — | 16 |
| ladder | 4,6,8,10,12,14,16,20,24,26 | 403 | [1.4, 5.5] | 86% | — | 20 |
| square | 4,6,8,10,12,14,16,20 | 427 | [1.4, 5.5] | 100% | — | 20 |
| triangular | 3,4,6,8,9,10,12,14,16 | 275 | [0.5, 5.5] | 86% | — | 14 |
<!-- AUTO-GENERATED-END:executive_summary -->

---

## Training Data Health

<!-- AUTO-GENERATED-BEGIN:health -->
| Metric | Value | Status |
|--------|-------|--------|
| Total NPZ files | 38 | |
| Total training points | 1472 | |
| **Quality: Useful** | 25 configs | ✅ |
| **Quality: Insufficient** | 5 configs | ⚠️ |
| **Quality: Not Useful** | 8 configs | ❌ |
| NaN in θ | 0 configs | ✅ |
| Zoo integrity | True | ✅ |
| Zoo missing | 0 | ✅ |
| Zoo orphan checkpoints | 3 | ⚠️ cleanup needed |
| GT coverage gaps | 450 uncovered h-points | ⚠️ |
| Stale zoo models | 6 | ⚠️ |
| Need retrain | 17 | 🔄 |
| High θ discontinuity (>0.5) | 15 configs | ⚠️ |
| Gap masking detected | 19 configs | ⚠️ |
<!-- AUTO-GENERATED-END:health -->

---

## Gap Masking Analysis

Configs where `pass@5% - pass@dual_criterion > 10%` — large gap inflates ΔE/gap metric:

<!-- AUTO-GENERATED-BEGIN:gap_masking -->
| Topology | N | Pass@5% | Pass@dual | Gap masked |
|----------|---|---------|-----------|------------|
| square | 20 | 100% | 0% | 100% |
| ladder | 14 | 66% | 0% | 66% |
| square | 14 | 100% | 38% | 62% |
| ladder | 10 | 71% | 10% | 61% |
| triangular | 8 | 86% | 29% | 57% |
| ladder | 12 | 82% | 33% | 49% |
| triangular | 10 | 70% | 22% | 48% |
| ladder | 16 | 82% | 42% | 40% |
| triangular | 12 | 60% | 20% | 40% |
| ladder | 8 | 86% | 47% | 39% |
| square | 12 | 88% | 50% | 38% |
| triangular | 14 | 32% | 0% | 32% |
| square | 16 | 63% | 32% | 31% |
| ladder | 6 | 84% | 67% | 18% |
| square | 10 | 88% | 71% | 17% |
| triangular | 9 | 16% | 0% | 16% |
| heavy_hex | 6 | 85% | 70% | 15% |
| ladder | 20 | 14% | 0% | 14% |
| triangular | 6 | 64% | 52% | 12% |
<!-- AUTO-GENERATED-END:gap_masking -->

---

## Detalle por Topología

<!-- AUTO-GENERATED-BEGIN:topo_chain_1d -->
### Chain 1D

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 6 | 40 | [1.50, 5.50] | 98% | 98% | 1.56 | 0.79 ⚠️ | div=0.03 |
| 8 | 14 | [2.00, 3.50] | 100% | 100% | 2.00 | 0.03  | div=0.00 |
| 10 | 40 | [1.50, 5.50] | 90% | 90% | 1.84 | 0.09  | div=0.10 |
| 12 | 14 | [2.00, 3.50] | 100% | 100% | 2.00 | 0.05  | div=0.00 |
| 15 | 40 | [1.50, 5.50] | 85% | 85% | 2.03 | 0.47  | div=0.15 |
| 20 | 5 | [3.04, 3.50] | 100% | 100% | 3.04 | 0.06  | div=0.00 |
<!-- AUTO-GENERATED-END:topo_chain_1d -->

<!-- AUTO-GENERATED-BEGIN:topo_heavy_hex -->
### Heavy Hex

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 15 | [0.58, 2.50] | 67% | 67% | 0.96 | 0.08  | div=0.17 |
| 6 | 47 | [1.40, 4.50] | 85% | 70% | 1.56 | 1.60 ⚠️ | (15% masked) div=0.02 |
| 10 | 65 | [1.40, 4.50] | 89% | 89% | 1.76 | 0.23  | div=0.06 |
| 12 | 27 | [2.12, 4.50] | 81% | 78% | 2.54 | 0.90 ⚠️ | div=0.02 |
| 16 | 60 | [2.00, 4.50] | 98% | 92% | 2.05 | 1.24 ⚠️ | (7% masked) div=0.15 |
<!-- AUTO-GENERATED-END:topo_heavy_hex -->

<!-- AUTO-GENERATED-BEGIN:topo_ladder -->
### Ladder

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 13 | [1.36, 2.08] | 23% | 23% | 1.85 | 0.11  | div=0.77 |
| 6 | 57 | [2.00, 4.80] | 84% | 67% | 2.33 | 1.25 ⚠️ | ⚠️ GAP MASK +18% div=0.16 |
| 8 | 57 | [2.08, 4.80] | 86% | 47% | 2.62 | 0.66 ⚠️ | ⚠️ GAP MASK +39% div=0.14 |
| 10 | 89 | [2.00, 4.80] | 71% | 10% | 2.87 | 0.79 ⚠️ | ⚠️ GAP MASK +61% div=0.29 |
| 12 | 49 | [2.58, 5.50] | 82% | 33% | 2.97 | 0.79 ⚠️ | ⚠️ GAP MASK +49% div=0.18 |
| 14 | 58 | [2.80, 5.32] | 66% | 0% | 3.10 | 2.36 ⚠️ | ⚠️ GAP MASK +66% div=0.34 |
| 16 | 45 | [3.00, 5.50] | 82% | 42% | 3.21 | 1.58 ⚠️ | ⚠️ GAP MASK +40% div=0.18 |
| 20 | 29 | [2.50, 3.50] | 14% | 0% | 3.39 | 0.47  | (14% masked) div=0.86 |
| 24 | 3 | [3.34, 3.50] | 0% | 0% | N/A | 0.00  | div=1.00 |
| 26 | 3 | [3.00, 3.50] | 0% | 0% | N/A | 0.01  | div=1.00 |
<!-- AUTO-GENERATED-END:topo_ladder -->

<!-- AUTO-GENERATED-BEGIN:topo_square -->
### Square

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 41 | [1.40, 4.50] | 85% | 85% | 1.84 | 0.17  | div=0.15 |
| 6 | 91 | [1.40, 4.80] | 84% | 78% | 2.33 | 0.79 ⚠️ | (5% masked) div=0.16 |
| 8 | 74 | [2.14, 4.80] | 91% | 81% | 2.64 | 1.58 ⚠️ | (9% masked) div=0.09 |
| 10 | 72 | [2.50, 4.80] | 88% | 71% | 2.84 | 1.57 ⚠️ | ⚠️ GAP MASK +17% div=0.12 |
| 12 | 50 | [2.40, 5.50] | 88% | 50% | 3.16 | 1.66 ⚠️ | ⚠️ GAP MASK +38% div=0.12 |
| 14 | 24 | [3.38, 5.50] | 100% | 38% | 3.38 | 0.00  | ⚠️ GAP MASK +62% div=0.00 |
| 16 | 68 | [2.70, 5.50] | 63% | 32% | 3.51 | 0.79 ⚠️ | ⚠️ GAP MASK +31% div=0.37 |
| 20 | 7 | [3.97, 4.50] | 100% | 0% | 3.97 | 0.08  | ⚠️ GAP MASK +100% div=0.00 |
<!-- AUTO-GENERATED-END:topo_square -->

<!-- AUTO-GENERATED-BEGIN:topo_triangular -->
### Triangular

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 3 | 37 | [0.50, 4.50] | 59% | 59% | N/A | 0.04  | div=0.33 STALE |
| 4 | 36 | [1.24, 4.50] | 72% | 72% | 1.98 | 0.04  | div=0.45 STALE |
| 6 | 77 | [2.13, 4.80] | 64% | 52% | 3.15 | 0.02  | (12% masked) div=0.37 STALE |
| 8 | 14 | [3.86, 4.80] | 86% | 29% | 3.98 | 0.00  | ⚠️ GAP MASK +57% div=0.59 STALE |
| 9 | 32 | [2.12, 4.50] | 16% | 0% | 4.30 | 0.02  | ⚠️ GAP MASK +16% div=0.11 |
| 10 | 27 | [3.20, 4.80] | 70% | 22% | 3.94 | 0.01  | ⚠️ GAP MASK +48% div=0.43 STALE |
| 12 | 20 | [3.81, 5.50] | 60% | 20% | 4.44 | 0.00  | ⚠️ GAP MASK +40% div=0.33 STALE |
| 14 | 25 | [2.12, 5.50] | 32% | 0% | 4.80 | 0.04  | ⚠️ GAP MASK +32% div=0.05 |
| 16 | 7 | [2.88, 3.50] | 0% | 0% | N/A | 0.00  | div=0.27 |
<!-- AUTO-GENERATED-END:topo_triangular -->

---

## h_frontier per Topology

h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):

<!-- AUTO-GENERATED-BEGIN:h_frontier -->
| Topología | N=4 | N=6 | N=8 | N=9 | N=10 | N=12 | N=14 | N=15 | N=16 | N=20 |
|-----------|--- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| chain_1d | — | 1.56 | 2.00 | — | 1.84 | 2.00 | — | 2.03 | — | 3.04 |
| heavy_hex | 0.96 | 1.56 | — | — | 1.76 | 2.54 | — | — | 2.05 | — |
| ladder | 1.85 | 2.33 | 2.62 | — | 2.87 | 2.97 | 3.10 | — | 3.21 | 3.39 |
| square | 1.84 | 2.33 | 2.64 | — | 2.84 | 3.16 | 3.38 | — | 3.51 | 3.97 |
| triangular | 1.98 | 3.15 | 3.98 | 4.30 | 3.94 | 4.44 | 4.80 | — | — | — |
<!-- AUTO-GENERATED-END:h_frontier -->

---

## Cross-N Transfer Summary

<!-- AUTO-GENERATED-BEGIN:cross_n_transfer -->
| Topology | n_max_viable | Best pass@5% | Best cross-N source |
|----------|-------------|-------------|---------------------|
| chain_1d | 20 | 100% | train_n=10 (@10%=100%) |
| heavy_hex | 16 | 98% | no data |
| ladder | 20 | 86% | no data |
| square | 20 | 100% | no data |
| triangular | 14 | 86% | no data |
<!-- AUTO-GENERATED-END:cross_n_transfer -->

---

<!-- AUTO-GENERATED-BEGIN:large_n_extrapolation -->
## Large-N Extrapolation (Zero-Shot)

MPNN predictions at N >> training data. Model trained on N≤20,
evaluated at N=30-100 via MPS backend. Speedup = VQE_evals / MPNN_evals.

### chain_1d

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [4.0, 4.5] | 2 | 0.0103 | 6.84e-03 | 2/2 | 2/2 | — |
| 30 | [2.5, 5.5] | 18 | 0.0482 | 9.40e-03 | 13/18 | 1/18 | 20580× |
| 40 | [2.5, 5.0] | 7 | 0.0737 | 8.54e-03 | 4/7 | 0/7 | 8600× |
| 60 | [2.5, 5.5] | 12 | 0.1114 | 1.10e-02 | 2/12 | 0/12 | 13920× |
| 100 | [3.5, 5.5] | 9 | 0.1470 | 1.10e-02 | 0/9 | 0/9 | 22667× |

### heavy_hex

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 4.5] | 3 | 0.0035 | 2.13e-03 | 3/3 | 3/3 | — |
| 16 | [3.0, 5.0] | 11 | 0.0183 | 6.49e-03 | 10/11 | 7/11 | 3595× |
| 20 | [3.0, 5.0] | 10 | 0.8009 | 1.26e-02 | 0/10 | 0/10 | 4480× |
| 30 | [3.0, 4.5] | 4 | 5.2682 | 3.68e-02 | 0/4 | 0/4 | — |

### ladder

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 16 | [3.5, 5.0] | 4 | 0.0254 | 7.27e-03 | 3/4 | 2/4 | — |
| 20 | [3.5, 5.0] | 4 | 0.0669 | 5.29e-03 | 3/4 | 1/4 | — |
| 30 | [2.5, 5.0] | 3 | 3.0236 | 2.11e-02 | 0/3 | 0/3 | — |

### square

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 16 | [3.0, 4.5] | 4 | 0.0627 | 1.21e-02 | 2/4 | 0/4 | — |
| 20 | [3.0, 4.5] | 4 | 0.3838 | 1.30e-02 | 1/4 | 0/4 | — |
| 30 | [3.5, 4.5] | 3 | 1.4906 | 1.04e-02 | 0/3 | 0/3 | — |

### triangular

| N | h-range | Pts | ΔE/gap | |ΔE|/N | Pass@5% | Pass@dual | Speedup |
|---|---------|-----|--------|--------|---------|-----------|---------|
| 10 | [3.5, 5.0] | 4 | 0.0525 | 1.58e-02 | 3/4 | 1/4 | 3146× |
| 12 | [3.5, 5.0] | 4 | 0.1263 | 2.42e-02 | 2/4 | 0/4 | 3374× |

### Extensive Scaling Summary

| Topology | N range | |ΔE|/N (mean) | Variation | Scaling |
|----------|---------|--------------|-----------|---------|
| chain_1d | 10–100 | 9.37e-03 | 1.6× | ✅ extensive |
| heavy_hex | 10–30 | 1.45e-02 | 17.3× | ⚠️ degrading |
| ladder | 16–30 | 1.12e-02 | 4.0× | ⚠️ degrading |
| square | 16–30 | 1.18e-02 | 1.2× | ✅ extensive |
| triangular | 10–12 | 2.00e-02 | 1.5× | ✅ extensive |

### MPNN vs Random VQE vs Ground Truth

Comparison at same h-points. MPNN: 1 forward pass (0 QPU). VQE: L-BFGS-B with random init.

| Topology | N | MPNN ΔE/gap | VQE ΔE/gap | MPNN |ΔE|/N | MPNN wins? | Speedup | VQE evals |
|----------|---|-------------|------------|------|-------|---------|-----------|
| chain_1d | 30 | 0.0294 | 2.3794 | 6.01e-03 | ✅ | 13340× | 80,040 |
| chain_1d | 40 | 0.0377 | 7.6010 | 6.45e-03 | ✅ | 8600× | 34,400 |
| chain_1d | 60 | 0.0572 | 5.8887 | 6.52e-03 | ✅ | 13920× | 55,680 |
| chain_1d | 100 | 0.1072 | 13.8617 | 7.75e-03 | ✅ | 22667× | 68,000 |
| heavy_hex | 16 | 0.0191 | 2.4163 | 6.11e-03 | ✅ | 3595× | 21,568 |
| heavy_hex | 20 | 0.7027 | 40.5765 | 1.10e-02 | ✅ | 4480× | 26,880 |
| triangular | 10 | 0.0525 | 1.0663 | 1.58e-02 | ✅ | 3146× | 12,586 |
| triangular | 12 | 0.1263 | 0.1180 | 2.42e-02 | ❌ | 3374× | 13,494 |

**MPNN win rate**: 7/8 (87%)

**Speedup range**: 3146× – 22667×

<!-- AUTO-GENERATED-END:large_n_extrapolation -->

---

<!-- AUTO-GENERATED-BEGIN:training_plan -->
## Training Plan (auto-generated)

**Total configs**: 38 | ✅ Useful: 25 | ⚠️ Insufficient: 5 | ❌ Not useful: 8

### ❌ DELETE — Not useful for MPNN training

These NPZ files teach the MPNN wrong mappings. Remove or regenerate:

| File | Topology | N | Reason |
|------|----------|---|--------|
| `ladder_N14_p1.npz` | ladder | 14 | 0% dual pass (gap masking: 66% of points appear to pass but  |
| `ladder_N20_p1.npz` | ladder | 20 | 0% dual pass (gap masking: 14% of points appear to pass but  |
| `ladder_N24_p1.npz` | ladder | 24 | 0% dual pass rate — no learnable signal. Possible causes: p= |
| `ladder_N26_p1.npz` | ladder | 26 | 0% dual pass rate — no learnable signal. Possible causes: p= |
| `square_N20_p1.npz` | square | 20 | 0% dual pass (gap masking: 100% of points appear to pass but |
| `triangular_N9_p1.npz` | triangular | 9 | 0% dual pass (gap masking: 16% of points appear to pass but  |
| `triangular_N14_p1.npz` | triangular | 14 | 0% dual pass (gap masking: 32% of points appear to pass but  |
| `triangular_N16_p1.npz` | triangular | 16 | 0% dual pass rate — no learnable signal. Possible causes: p= |

```bash
rm data/multi_n_training/ladder_N14_p1.npz
rm data/multi_n_training/ladder_N20_p1.npz
rm data/multi_n_training/ladder_N24_p1.npz
rm data/multi_n_training/ladder_N26_p1.npz
rm data/multi_n_training/square_N20_p1.npz
rm data/multi_n_training/triangular_N9_p1.npz
rm data/multi_n_training/triangular_N14_p1.npz
rm data/multi_n_training/triangular_N16_p1.npz
```

### ⚠️ IMPROVE — Insufficient signal (need more good points)

Run iterative-improve to densify these configs above the frontier:

| File | Topology | N | Pts | Dual pass | h_frontier | Action |
|------|----------|---|-----|-----------|------------|--------|
| `ladder_N4_p1.npz` | ladder | 4 | 13 | 23% | 1.85 | iterative-improve h≥2.0 |
| `ladder_N10_p1.npz` | ladder | 10 | 89 | 10% | 2.87 | iterative-improve h≥3.1 |
| `triangular_N8_p1.npz` | triangular | 8 | 14 | 29% | 3.98 | iterative-improve h≥4.2 |
| `triangular_N10_p1.npz` | triangular | 10 | 27 | 22% | 3.94 | iterative-improve h≥4.1 |
| `triangular_N12_p1.npz` | triangular | 12 | 20 | 20% | 4.44 | iterative-improve h≥4.6 |

### ✅ EXPAND — Useful configs (add more h-points for better generalization)

| Topology | N | Pts | Dual pass | h_frontier | Priority |
|----------|---|-----|-----------|------------|----------|
| triangular | 3 | 37 | 59% | N/A | LOW (already dense) |
| heavy_hex | 4 | 15 | 67% | 0.96 | MEDIUM (expand range) |
| square | 4 | 41 | 85% | 1.84 | LOW (already dense) |
| triangular | 4 | 36 | 72% | 1.98 | LOW (already dense) |
| chain_1d | 6 | 40 | 98% | 1.56 | LOW (already dense) |
| heavy_hex | 6 | 47 | 70% | 1.56 | LOW (already dense) |
| ladder | 6 | 57 | 67% | 2.33 | LOW (already dense) |
| square | 6 | 91 | 78% | 2.33 | LOW (already dense) |
| triangular | 6 | 77 | 52% | 3.15 | LOW (already dense) |
| chain_1d | 8 | 14 | 100% | 2.00 | HIGH (few pts, good quality) |
| ladder | 8 | 57 | 47% | 2.62 | LOW (already dense) |
| square | 8 | 74 | 81% | 2.64 | LOW (already dense) |
| chain_1d | 10 | 40 | 90% | 1.84 | LOW (already dense) |
| heavy_hex | 10 | 65 | 89% | 1.76 | LOW (already dense) |
| square | 10 | 72 | 71% | 2.84 | LOW (already dense) |
| chain_1d | 12 | 14 | 100% | 2.00 | HIGH (few pts, good quality) |
| heavy_hex | 12 | 27 | 78% | 2.54 | MEDIUM (expand range) |
| ladder | 12 | 49 | 33% | 2.97 | LOW (already dense) |
| square | 12 | 50 | 50% | 3.16 | LOW (already dense) |
| square | 14 | 24 | 38% | 3.38 | MEDIUM (expand range) |
| chain_1d | 15 | 40 | 85% | 2.03 | LOW (already dense) |
| heavy_hex | 16 | 60 | 92% | 2.05 | LOW (already dense) |
| ladder | 16 | 45 | 42% | 3.21 | LOW (already dense) |
| square | 16 | 68 | 32% | 3.51 | LOW (already dense) |
| chain_1d | 20 | 5 | 100% | 3.04 | HIGH (few pts, good quality) |

<!-- AUTO-GENERATED-END:training_plan -->
