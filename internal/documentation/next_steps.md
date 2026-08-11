# Próximos Pasos — Pipeline GNN-HVA

**Fecha**: 2026-08-11 (actualizado 02:30)
**Estado actual**: 38 NPZ, ~1135 puntos post-cleaning (theta outliers removed)
**Zoo**: 5 checkpoints (all multi-N) | **Extrapolation**: chain_1d N=30, ladder N=20
**Topologías**: chain_1d (n_max=20✅), heavy_hex (16✅), ladder (10⚠️ physics limit), square (16✅), triangular (14⚠️)

### Hallazgos recientes (sesión 2026-08-11)

- **Theta cleaning**: 15 NPZ con discontinuidad → 6 (sequential canonicalize + MAD filter, 158 outliers removed)
- **Zoo consolidado**: 16→5 entries (11 single-N archivados, 3 orphans archivados)
- **Retrain post-clean**: chain_1d 85%→75%@5% (130pts), square 88%→65%@5% (189pts) — both retrained on theta-cleaned data
- **Ladder physics limit CONFIRMED**: 0% dual pass rate is inherent (HVA p=1 + z=3 coordination → intractable landscape). NOT a pipeline failure.
- **Regression detection fixed**: `detect_regressions()` now uses latest-vs-previous (eliminates historical-peak false positives)
- **Coverage matrix fixed**: uses latest run per config (not inflated by historical best)

---

## A. Acciones Inmediatas (prioridad por impacto)

### A1. ~~Reentrenar ladder~~ → CERRADO (physics limit)

**Decisión**: ladder multi-N (0% dual pass) es un **límite de expresividad del ansatz**.
HVA p=1 con coordination z=3 → demasiados parámetros por capa → VQE landscape intractable.
Gap masking inflaba pass_rate de 66-100% a 0% bajo dual criterion.

**Acción**: documentar como finding en thesis (Chapter 4.5, Gap Masking Discovery).
No invertir más compute en ladder N>10.

### A2. ~~Reentrenar modelos zoo desactualizados~~ → COMPLETADO

- chain_1d: retrained (130pts, 75% @5%, F=0.983)
- square: retrained (189pts, 65% @5%, F=0.982)
- heavy_hex: no requiere retrain (score 0.94, data no fue theta-cleaned)
- triangular: retrained anteriormente (27%, physics limit similar a ladder)
- ladder: no retrained (physics limit, 0% dual)

**Zoo incoherences**: de 19 → ~4 (solo h_frontier anomalies).

### A3. Resolver h_frontier anomalías (4 detectadas)

- chain_1d N=12: frontier < N=10 (drop=0.32) — h-range inconsistente [2.0, 3.5] vs [1.5, 5.5]
- heavy_hex N=10: frontier < N=6 (drop=0.89) — necesita más puntos en h=[1.5, 2.5]
- heavy_hex N=16: frontier < N=12 (drop=0.49) — posible gap masking
- triangular N=10: frontier < N=9 (drop=0.37) — datos insuficientes

```bash
# Expand h-range for chain_1d N=12
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --target-n 12 --iterative-improve --max-iterations 3 \
    --h-min 1.5 --h-max 5.5 --h-points 25

# Add low-h data for heavy_hex N=10
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology heavy_hex --target-n 10 --iterative-improve --max-iterations 3 \
    --h-min 1.5 --h-max 4.5 --h-points 25
```

**Impacto**: alinea h_frontier con el patrón esperado (frontier decrece monotónicamente con N).

### A4. Reentrenar triangular (27% → target >50%)

Triangular tiene 140 pts pero solo 27% pass rate en el zoo. A diferencia de ladder,
triangular N≤10 sí tiene pass@dual >0% (22% en N=10, 52% en N=6). Hay señal.

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology triangular --target-n 6 8 10 --iterative-improve \
    --max-iterations 3 --force-retrain --refine-all
```

**Expectativa**: 27% → 40-50% (limitado por physics en N≥9).

---

## B. Escalamiento (esta semana)

### B1. Extrapolación large-N para topologías faltantes

Solo chain_1d tiene datos N>20 confiables (dual criterion). Ladder N=20 es gap-masked.

```bash
# heavy_hex: genuinamente viable (94% dual@N=16)
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology heavy_hex --target-n 20 30 --h-min 2.5 --h-max 4.5

# square: viable con precaución (dual drops at large N)
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology square --target-n 20 --h-min 3.5 --h-max 5.5
```

**Predicción actualizada**:
- heavy_hex N=20: **high confidence** (n_max=16 genuine, 94% dual)
- square N=20: **medium confidence** (n_max=10-12 under dual criterion)
- triangular N=20: **low confidence** (n_max=4 under dual criterion)

### B2. Extender chain_1d a N=100

Ya hay datos para N=30,40,60. Falta N=100 (DMRG viable, ~15 min por punto).

```bash
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d --target-n 100 --h-min 3.5 --h-max 5.0 --h-points 4
```

**Objetivo**: demostrar |ΔE|/N ≈ constante (extensive scaling claim para thesis).

### B3. Validar gap masking en large-N

Los datos N>20 muestran gap masking severo (ladder N=20: 14%→0% under dual).
Verificar con dual criterion para todas las extrapolaciones.

```bash
.venv/bin/python scripts/analysis/compute_h_frontier_all.py --by-tier
```

---

## C. Calidad de Datos (próxima semana)

### C1. ~~Upgrade legacy NPZ restantes~~ → PARTIALLY DONE

Theta-cleaning session canonicalizó 15 archivos. Quedan 6 con smoothness inherente
(π/4 crossover natural). Los quality_tier ya existen en todos los NPZ (added by
`upgrade_npz_quality_tiers.py` run previo).

**Pendiente**: confirmar que el 14% "unverified" bajó tras theta-cleaning.

### C2. Implementar data versioning por topología

Cada NPZ podría tener un header con metadata: pipeline_version, h_grid_type (uniform/dense),
date_generated, n_restarts_used. Esto permite filtrar por calidad de generación.

**Dónde**: extender `upsert_theta_npz` con campo `metadata` dict (JSON-safe).
**Prioridad**: BAJA (nice-to-have para reproducibility, no bloquea thesis).

### C3. Cross-validate MPNN con leave-one-N-out

Entrenar en N={6,8,10,12} y evaluar en N=14,16 (sin haberlo visto).
Comparar vs el modelo entrenado en todos los N.

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_cross_n_validation.py \
    --topology chain_1d --leave-out 14 16
```

---

## D. Hardware Deployment (cuando A3 + B1 estén listos)

### D1. Pre-flight validation

Antes de enviar a IBM Heron, verificar:
1. Circuitos transpilados → profundidad 2Q < 50
2. Layout optimization → CES < 0.02
3. Shot budget → ≥8192 shots (Sharma 2026: 4096 insuficiente)

```bash
.venv/bin/python scripts/hardware/preflight_hw.py --topology chain_1d --n-qubits 10
```

### D2. Rehearsal run (simulador con noise model)

```bash
.venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py \
    --topology chain_1d --n-qubits 10 --h-points 5
```

### D3. IBM Heron deployment

Objetivo: validar que MPNN warm-start produce correct phase classification en hardware.
Success criteria: ΔE/gap < 15% (hardware-relaxed) y correct phase en 80%+ de h-points.

**Topología candidata**: chain_1d N=10 (75% pass@5% noiseless, 10 CZ gates, shallow).

---

## E. Thesis Writing (paralelo)

### E1. Figuras de escalamiento

- Fig: |ΔE|/N vs N (extensive scaling plot) — chain_1d N=6→100
- Fig: h_frontier vs N por topología (con dual criterion bounds)
- Fig: quality tier pie chart (verification coverage)
- Fig: gap masking before/after dual criterion (ladder N=16: 100%→0%)

```bash
.venv/bin/python project_health/analysis/thesis/thesis_figures.py
```

### E2. Tabla comparativa con literatura

| Pipeline | Qracle | NN-VQE | Flow-VQE | Nuestro |
|----------|--------|--------|----------|---------|
| QPU calls | 64% menos | ~100 | ~50 | 0-2 |
| Topologies | 1 | 1 | 1 | 6 |
| Max N tested | 12 | 8 | 10 | 100 |
| Metric | ΔE/gap | ΔE/gap | ΔE/gap | ΔE/gap + |ΔE| (dual) |

### E3. Gap masking como contribución metodológica (READY TO WRITE)

Documentar el hallazgo: ladder/square reportan pass_rate inflado sin dual criterion.
19 configs afectadas. Solo heavy_hex genuinamente pasa a N=16 (94% under dual).
**Este es un finding publicable independientemente de la thesis.**

---

## F. Mejoras Técnicas (cuando haya tiempo)

### F1. GNN-QEM V2 (error correction post-hardware)

GNN que predice el error de hardware y lo corrige. Ya hay datos de training
en `data/gnn_qem/`. Requiere datos de hardware real para validar.

### F2. AQC-Tensor compression

Reducir profundidad de circuito vía AQC (ya implementado en `circuits/aqc_compression.py`).
Podría permitir p=2 en hardware donde hoy solo p=1 cabe.

### F3. Normalizing Flow warm-start

Alternativa a MPNN: usar MAF para muestrear θ desde distribución aprendida.
Ya existe `analysis/normalizing_flow.py`. Comparar vs MPNN en términos de
diversidad de soluciones y escape de mínimos locales.

### F4. Multi-model support (Heisenberg, XY) → OUT OF SCOPE

Los modelos non-TFIM (heisenberg, xy) muestran 50% max pass rate.
Requieren p>2 para ser expresivos → fuera de scope de NISQ thesis.
Documentar como "frontier beyond TFIM" en Discussion (Chapter 5.6).

---

## Métricas de Éxito (actualizado 2026-08-11)

| Milestone | Criterio | Estado |
|-----------|----------|--------|
| Training data quality | >70% verified | ~60% ⚠️ (chain_1d 96%✅, heavy_hex 94%✅, square/triangular lower) |
| Zoo coherence | <5 incoherencias | **4** ⚠️ (h_frontier anomalies only) |
| Zoo model health | all scores >0.70 | 3/5 ⚠️ (ladder=physics limit, triangular=borderline) |
| Theta smoothness | <15 configs >0.5 | **6** ✅ (was 15, now 6 inherent) |
| Extrapolation N=30 | 3+ topologías | 1/5 ⚠️ (chain_1d✅, falta heavy_hex/square) |
| Extensive scaling proof | \|ΔE\|/N ≈ const | chain_1d: 7.4e-3 @ N=30 ✅ |
| Hardware validation | 1 topology deployed | Pendiente |
| Thesis figures | All generated | Pendiente |
| Gap masking documented | dual criterion | ✅ (19 configs, publishable finding) |
| Zoo cleaned | No orphans/stale | ✅ (14 archived, 5 active) |

---

## Prioridad de Ejecución (ordered by thesis impact)

1. **A3** h_frontier anomalies (fixes inconsistencies in thesis tables) — ~1h compute
2. **B1** heavy_hex extrapolation to N=20-30 (high confidence, thesis Chapter 4.3) — ~30min
3. **B2** chain_1d N=100 (extensive scaling proof, thesis Chapter 4.6) — ~1h
4. **E3** Gap masking writeup (thesis Chapter 4.5, ready to write NOW) — 0 compute
5. **E1** Thesis figures generation — ~5min
6. **D1-D2** Hardware preflight + rehearsal (thesis Chapter 4.7) — ~1h
7. **A4** Triangular improvement (nice-to-have, not blocking thesis) — ~2h

---

## Comandos Quick Reference

```bash
# Estado actual
.venv/bin/python scripts/maintenance/quick_health_check.py
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py --dry-run

# Reentrenar + iterative improve
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology TOPO --target-n N --iterative-improve --force-retrain --max-iterations 3

# Extrapolación large-N
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology TOPO --target-n 30 40

# Dashboard refresh (slow: eval_cache scan)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard; generate_model_quality_dashboard()"

# Project status refresh (fast)
.venv/bin/python scripts/maintenance/update_project_status.py

# Scaling report
.venv/bin/python scripts/maintenance/generate_scaling_report.py
```
