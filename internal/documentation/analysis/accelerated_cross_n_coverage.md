# Accelerated Cross-N Coverage Analysis

**Fecha**: 2026-08-09 (auto-updated by update_cross_n_coverage.py)
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
| chain_1d | 8,12,20 | 33 | [2.0, 3.5] | 100% | — | 20 |
| heavy_hex | 4,6,10,12,16 | 148 | [0.6, 4.5] | 97% | — | 16 |
| ladder | 4,6,8,10,12,14,16,20,24,26 | 168 | [1.4, 4.5] | 100% | — | 20 |
| square | 4,6,8,10,12,16,20 | 224 | [1.4, 4.5] | 100% | — | 20 |
| triangular | 3,4,6,9,10,14,16 | 180 | [0.5, 4.5] | 72% | — | 10 |
<!-- AUTO-GENERATED-END:executive_summary -->

---

## Training Data Health

<!-- AUTO-GENERATED-BEGIN:health -->
| Metric | Value | Status |
|--------|-------|--------|
| Total NPZ files | 32 | |
| Total training points | 753 | |
| NaN in θ | 0 configs | ✅ |
| Zoo integrity | True | ✅ |
| Zoo missing | 0 | ✅ |
| Zoo orphan checkpoints | 20 | ⚠️ cleanup needed |
| GT coverage gaps | 174 uncovered h-points | ⚠️ |
| Stale zoo models | 4 | ⚠️ |
| Need retrain | 14 | 🔄 |
| High θ discontinuity (>0.5) | 9 configs | ⚠️ |
| Gap masking detected | 15 configs | ⚠️ |
<!-- AUTO-GENERATED-END:health -->

---

## Gap Masking Analysis

Configs where `pass@5% - pass@dual_criterion > 10%` — large gap inflates ΔE/gap metric:

<!-- AUTO-GENERATED-BEGIN:gap_masking -->
| Topology | N | Pass@5% | Pass@dual | Gap masked |
|----------|---|---------|-----------|------------|
| ladder | 16 | 100% | 0% | 100% |
| square | 20 | 100% | 0% | 100% |
| ladder | 14 | 80% | 0% | 80% |
| ladder | 12 | 100% | 55% | 45% |
| triangular | 10 | 50% | 8% | 42% |
| square | 16 | 58% | 19% | 39% |
| ladder | 8 | 71% | 43% | 29% |
| square | 12 | 76% | 48% | 28% |
| heavy_hex | 6 | 86% | 65% | 22% |
| ladder | 10 | 40% | 21% | 19% |
| square | 10 | 80% | 64% | 16% |
| triangular | 9 | 16% | 0% | 16% |
| ladder | 20 | 12% | 0% | 12% |
| triangular | 6 | 57% | 45% | 12% |
| ladder | 6 | 68% | 57% | 11% |
<!-- AUTO-GENERATED-END:gap_masking -->

---

## Detalle por Topología

<!-- AUTO-GENERATED-BEGIN:topo_chain_1d -->
### Chain 1D

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 8 | 14 | [2.00, 3.50] | 100% | 100% | 2.00 | 0.03  | div=0.00 |
| 12 | 14 | [2.00, 3.50] | 100% | 100% | 2.00 | 0.05  | div=0.00 |
| 20 | 5 | [3.04, 3.50] | 100% | 100% | 3.04 | 0.06  | div=0.00 |
<!-- AUTO-GENERATED-END:topo_chain_1d -->

<!-- AUTO-GENERATED-BEGIN:topo_heavy_hex -->
### Heavy Hex

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 15 | [0.58, 2.50] | 67% | 67% | 0.96 | 0.08  | div=0.33 |
| 6 | 37 | [2.00, 4.50] | 86% | 65% | 2.65 | 1.60 ⚠️ | ⚠️ GAP MASK +22% div=0.14 |
| 10 | 37 | [1.40, 4.50] | 86% | 86% | 1.76 | 0.06  | div=0.14 |
| 12 | 27 | [2.12, 4.50] | 81% | 78% | 2.54 | 0.90 ⚠️ | div=0.19 |
| 16 | 32 | [2.00, 3.50] | 97% | 94% | 2.02 | 1.24 ⚠️ | div=0.03 |
<!-- AUTO-GENERATED-END:topo_heavy_hex -->

<!-- AUTO-GENERATED-BEGIN:topo_ladder -->
### Ladder

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 13 | [1.36, 2.08] | 23% | 23% | 1.85 | 0.11  | div=0.77 |
| 6 | 28 | [2.00, 3.50] | 68% | 57% | 2.33 | 1.56 ⚠️ | (11% masked) div=0.32 |
| 8 | 28 | [2.08, 3.50] | 71% | 43% | 2.62 | 0.66 ⚠️ | ⚠️ GAP MASK +29% div=0.29 |
| 10 | 43 | [2.00, 3.50] | 40% | 21% | 2.87 | 0.79 ⚠️ | ⚠️ GAP MASK +19% div=0.60 |
| 12 | 11 | [3.06, 4.50] | 100% | 55% | 3.06 | 0.12  | ⚠️ GAP MASK +45% div=0.00 |
| 14 | 10 | [3.00, 3.50] | 80% | 0% | 3.10 | 0.07  | ⚠️ GAP MASK +80% div=0.20 |
| 16 | 5 | [3.28, 3.50] | 100% | 0% | 3.28 | 0.37  | ⚠️ GAP MASK +100% div=0.00 |
| 20 | 24 | [2.50, 3.50] | 12% | 0% | 3.39 | 0.47  | (12% masked) div=0.88 |
| 24 | 3 | [3.34, 3.50] | 0% | 0% | N/A | 0.00  | div=1.00 |
| 26 | 3 | [3.00, 3.50] | 0% | 0% | N/A | 0.01  | div=1.00 |
<!-- AUTO-GENERATED-END:topo_ladder -->

<!-- AUTO-GENERATED-BEGIN:topo_square -->
### Square

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 4 | 41 | [1.40, 4.50] | 85% | 85% | 1.84 | 0.17  | div=0.15 |
| 6 | 64 | [1.40, 4.50] | 77% | 73% | 2.33 | 0.79 ⚠️ | div=0.23 |
| 8 | 26 | [2.14, 4.50] | 77% | 69% | 2.64 | 0.02  | (8% masked) div=0.23 |
| 10 | 25 | [2.50, 4.50] | 80% | 64% | 2.84 | 0.18  | ⚠️ GAP MASK +16% div=0.20 |
| 12 | 25 | [2.40, 4.50] | 76% | 48% | 3.16 | 1.64 ⚠️ | ⚠️ GAP MASK +28% div=0.24 |
| 16 | 36 | [2.70, 4.50] | 58% | 19% | 3.50 | 0.79 ⚠️ | ⚠️ GAP MASK +39% div=0.42 |
| 20 | 7 | [3.97, 4.50] | 100% | 0% | 3.97 | 0.08  | ⚠️ GAP MASK +100% div=0.00 |
<!-- AUTO-GENERATED-END:topo_square -->

<!-- AUTO-GENERATED-BEGIN:topo_triangular -->
### Triangular

| N | Puntos | h-range | Pass@5% | Pass@dual | h_frontier | θ smooth | Observación |
|---|--------|---------|---------|-----------|------------|---------|-------------|
| 3 | 37 | [0.50, 4.50] | 59% | 59% | N/A | 0.04  | div=0.47 STALE |
| 4 | 36 | [1.24, 4.50] | 72% | 72% | 1.98 | 0.04  | div=0.60 STALE |
| 6 | 51 | [2.13, 4.50] | 57% | 45% | 3.15 | 0.02  | (12% masked) div=0.44 STALE |
| 9 | 32 | [2.12, 4.50] | 16% | 0% | 4.30 | 0.02  | ⚠️ GAP MASK +16% div=0.03 |
| 10 | 12 | [3.20, 4.50] | 50% | 8% | 3.94 | 0.01  | ⚠️ GAP MASK +42% div=0.38 STALE |
| 14 | 5 | [2.12, 2.50] | 0% | 0% | N/A | 0.01  | div=0.12 |
| 16 | 7 | [2.88, 3.50] | 0% | 0% | N/A | 0.00  | div=0.12 |
<!-- AUTO-GENERATED-END:topo_triangular -->

---

## h_frontier per Topology

h_frontier = h below which ΔE/gap ≥ 5% (pipeline fails):

<!-- AUTO-GENERATED-BEGIN:h_frontier -->
| Topología | N=4 | N=6 | N=8 | N=9 | N=10 | N=12 | N=14 | N=16 | N=20 |
|-----------|--- | --- | --- | --- | --- | --- | --- | --- | ---|
| chain_1d | — | — | 2.00 | — | — | 2.00 | — | — | 3.04 |
| heavy_hex | 0.96 | 2.65 | — | — | 1.76 | 2.54 | — | 2.02 | — |
| ladder | 1.85 | 2.33 | 2.62 | — | 2.87 | 3.06 | 3.10 | 3.28 | 3.39 |
| square | 1.84 | 2.33 | 2.64 | — | 2.84 | 3.16 | — | 3.50 | 3.97 |
| triangular | 1.98 | 3.15 | — | 4.30 | 3.94 | — | — | — | — |
<!-- AUTO-GENERATED-END:h_frontier -->

---

## Cross-N Transfer Summary

<!-- AUTO-GENERATED-BEGIN:cross_n_transfer -->
| Topology | n_max_viable | Best pass@5% | Best cross-N source |
|----------|-------------|-------------|---------------------|
| chain_1d | 20 | 100% | train_n=10 (@10%=100%) |
| heavy_hex | 16 | 97% | no data |
| ladder | 20 | 100% | no data |
| square | 20 | 100% | no data |
| triangular | 10 | 72% | no data |
<!-- AUTO-GENERATED-END:cross_n_transfer -->

---

<!-- AUTO-GENERATED-BEGIN:training_plan -->
## Training Plan (auto-generated)

**Total configs**: 32 | ✅ Useful: 19 | ⚠️ Insufficient: 4 | ❌ Not useful: 9

### ❌ DELETE — Not useful for MPNN training

These NPZ files teach the MPNN wrong mappings. Remove or regenerate:

| File | Topology | N | Reason |
|------|----------|---|--------|
| `ladder_N14_p1.npz` | ladder | 14 | 0% dual pass (gap masking: 80% of points appear to pass but  |
| `ladder_N16_p1.npz` | ladder | 16 | 0% dual pass (gap masking: 100% of points appear to pass but |
| `ladder_N20_p1.npz` | ladder | 20 | 0% dual pass (gap masking: 12% of points appear to pass but  |
| `ladder_N24_p1.npz` | ladder | 24 | 0% dual pass rate — no learnable signal. Possible causes: p= |
| `ladder_N26_p1.npz` | ladder | 26 | 0% dual pass rate — no learnable signal. Possible causes: p= |
| `square_N20_p1.npz` | square | 20 | 0% dual pass (gap masking: 100% of points appear to pass but |
| `triangular_N9_p1.npz` | triangular | 9 | 0% dual pass (gap masking: 16% of points appear to pass but  |
| `triangular_N14_p1.npz` | triangular | 14 | 0% dual pass rate — no learnable signal. Possible causes: p= |
| `triangular_N16_p1.npz` | triangular | 16 | 0% dual pass rate — no learnable signal. Possible causes: p= |

```bash
rm data/multi_n_training/ladder_N14_p1.npz
rm data/multi_n_training/ladder_N16_p1.npz
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
| `ladder_N10_p1.npz` | ladder | 10 | 43 | 21% | 2.87 | iterative-improve h≥3.1 |
| `square_N16_p1.npz` | square | 16 | 36 | 19% | 3.50 | iterative-improve h≥3.7 |
| `triangular_N10_p1.npz` | triangular | 10 | 12 | 8% | 3.94 | iterative-improve h≥4.1 |

### ✅ EXPAND — Useful configs (add more h-points for better generalization)

| Topology | N | Pts | Dual pass | h_frontier | Priority |
|----------|---|-----|-----------|------------|----------|
| triangular | 3 | 37 | 59% | N/A | LOW (already dense) |
| heavy_hex | 4 | 15 | 67% | 0.96 | MEDIUM (expand range) |
| square | 4 | 41 | 85% | 1.84 | LOW (already dense) |
| triangular | 4 | 36 | 72% | 1.98 | LOW (already dense) |
| heavy_hex | 6 | 37 | 65% | 2.65 | LOW (already dense) |
| ladder | 6 | 28 | 57% | 2.33 | MEDIUM (expand range) |
| square | 6 | 64 | 73% | 2.33 | LOW (already dense) |
| triangular | 6 | 51 | 45% | 3.15 | LOW (already dense) |
| chain_1d | 8 | 14 | 100% | 2.00 | HIGH (few pts, good quality) |
| ladder | 8 | 28 | 43% | 2.62 | MEDIUM (expand range) |
| square | 8 | 26 | 69% | 2.64 | MEDIUM (expand range) |
| heavy_hex | 10 | 37 | 86% | 1.76 | LOW (already dense) |
| square | 10 | 25 | 64% | 2.84 | MEDIUM (expand range) |
| chain_1d | 12 | 14 | 100% | 2.00 | HIGH (few pts, good quality) |
| heavy_hex | 12 | 27 | 78% | 2.54 | MEDIUM (expand range) |
| ladder | 12 | 11 | 55% | 3.06 | MEDIUM (expand range) |
| square | 12 | 25 | 48% | 3.16 | MEDIUM (expand range) |
| heavy_hex | 16 | 32 | 94% | 2.02 | LOW (already dense) |
| chain_1d | 20 | 5 | 100% | 3.04 | HIGH (few pts, good quality) |

<!-- AUTO-GENERATED-END:training_plan -->
