# Próximos Pasos — Pipeline GNN-HVA

**Fecha**: 2026-08-10 (actualizado 16:30)  
**Estado actual**: 37 NPZ, 1293 puntos (52% verified, 34% approximate, 14% unverified)  
**Zoo**: 17 checkpoints (5 multi-N) | **Extrapolation**: chain_1d N=30, ladder N=20  
**Topologías**: chain_1d (n_max=20✅), heavy_hex (16✅), ladder (20⚠️), square (20⚠️), triangular (14⚠️)

### Hallazgos recientes (sesión 2026-08-10)

- **Quality-aware loading** integrado en `runner_base.load_best_mpnn_for_cross_n()`
- **Zoo health report**: chain_1d (0.96✅), heavy_hex (0.94✅), ladder (0.73⚠️), square (0.86⚠️), triangular (0.78⚠️)
- **Gap detection**: ladder 23% verified (HIGH), triangular 44% verified (MEDIUM)
- **Extrapolation validado**: chain_1d N=30 ΔE/gap=3.72%, heavy_hex N=10 ΔE/gap=0.35%
- **NPZ fix**: `_persist_extrapolation_npz` corregido para object arrays

---

## A. Acciones Inmediatas (prioridad por impacto)

### A1. Reentrenar ladder y triangular (quality_score < 0.80)

Estas topologías tienen pass_rate=0% (ladder) y 43% (triangular) en el zoo.
La data tiene solo 23% y 44% verificada respectivamente.

```bash
# 1. Refinar puntos approximate → verified (genera VQE ground truth)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology ladder --train-n 6 --target-n 10 12 --mode iterative \
    --max-iterations 4 --refine-all --maxiter 500 --n-restarts 5

# 2. Reentrenar modelo multi-N con datos filtrados
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology ladder --train-n 6 --target-n 14 16 --mode multi-n --force-retrain

# 3. Mismo proceso para triangular
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology triangular --train-n 6 --target-n 10 12 --mode iterative \
    --refine-all --max-iterations 3
```

**Impacto**: elevar ladder de 0% → >50% pass, triangular de 43% → >60%.

### A2. Reentrenar modelos zoo desactualizados (19 incoherencias)

Los datos NPZ mejoraron pero los modelos no fueron reentrenados.

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --train-n 6 --target-n 10 12 15 20 --mode multi-n --force-retrain
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology heavy_hex --train-n 6 --target-n 10 12 16 --mode multi-n --force-retrain
```

**Beneficio**: zoo_pass_rate se alineará con NPZ pass_rate.

### A3. Resolver h_frontier anomalías (3 detectadas)

- chain_1d N=12: frontier < N=10 (drop=0.32)
- heavy_hex N=16: frontier < N=12 (drop=0.49)
- triangular N=10: frontier < N=9 (drop=0.37)

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --train-n 6 --target-n 12 --mode iterative \
    --max-iterations 5 --maxiter 500 --n-restarts 5
```

**Causa probable**: datos insuficientes o h-range inconsistente en esos N.

---

## B. Escalamiento (esta semana)

### B1. Extrapolación large-N para topologías faltantes

Solo chain_1d y ladder tienen datos N>20. Faltan: heavy_hex, square, triangular.

```bash
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology heavy_hex --target-n 20 30 --h-min 2.5 --h-max 5.0
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology square --target-n 20 30 --h-min 3.0 --h-max 5.0
```

**Predicción** (de `compute_extrapolation_viability`):
- heavy_hex N=30: viable (n_max=16, factor 1.9×)
- square N=30: viable (n_max=20, factor 1.5×)
- triangular N=20: borderline (n_max=14, factor 1.4×)

### B2. Extender chain_1d a N=100

Ya hay datos para N=30,40,60. Falta N=100 (DMRG viable, ~15 min por punto).

```bash
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d --target-n 100 --h-min 3.5 --h-max 5.0 --h-points 4
```

**Objetivo**: demostrar |ΔE|/N ≈ constante (extensive scaling claim para thesis).

### B3. Validar gap masking en large-N

Los datos N>20 probablemente muestren gap masking severo. Verificar con dual criterion.

```bash
.venv/bin/python scripts/analysis/compute_h_frontier_all.py --by-tier
```

---

## C. Calidad de Datos (próxima semana)

### C1. Upgrade legacy NPZ restantes

14% de puntos no tienen quality_tier (legacy format).

```bash
.venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py --backup
```

### C2. Implementar data versioning por topología

Cada NPZ podría tener un header con metadata: pipeline_version, h_grid_type (uniform/dense),
date_generated, n_restarts_used. Esto permite filtrar por calidad de generación.

**Dónde**: extender `upsert_theta_npz` con campo `metadata` dict (JSON-safe).

### C3. Cross-validate MPNN con leave-one-N-out

Entrenar en N={6,8,10,12} y evaluar en N=14,16 (sin haberlo visto).
Comparar vs el modelo entrenado en todos los N.

```bash
.venv/bin/python scripts/experiment_runners/cross_topology/run_cross_n_validation.py \
    --topology chain_1d --leave-out 14 16
```

---

## D. Hardware Deployment (cuando B1-B3 estén listos)

### D1. Pre-flight validation

Antes de enviar a IBM Heron, verificar:
1. Circuitos transpilados → profundidad 2Q < 50
2. Layout optimization → CES < 0.02
3. Shot budget → 4096 shots × N_circuits

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

---

## E. Thesis Writing (paralelo)

### E1. Figuras de escalamiento

- Fig: |ΔE|/N vs N (extensive scaling plot)
- Fig: h_frontier vs N por topología
- Fig: quality tier pie chart (verification coverage)

```bash
.venv/bin/python project_health/analysis/thesis/thesis_figures.py
```

### E2. Tabla comparativa con literatura

| Pipeline | Qracle | NN-VQE | Flow-VQE | Nuestro |
|----------|--------|--------|----------|---------|
| QPU calls | 64% menos | ~100 | ~50 | 0-2 |
| Topologies | 1 | 1 | 1 | 6 |
| Max N tested | 12 | 8 | 10 | 100 |

### E3. Gap masking como contribución metodológica

Documentar el hallazgo: ladder/square reportan pass_rate inflado sin dual criterion.
20 configs afectadas. Solo heavy_hex genuinamente pasa a N=16.

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

### F4. Multi-model support (Heisenberg, XY)

Los modelos non-TFIM (heisenberg, xy) muestran 50% max pass rate.
Requieren p>2 para ser expresivos → probablemente fuera de scope de NISQ thesis.
Documentar como "frontier beyond TFIM" en Discussion.

---

## Métricas de Éxito

| Milestone | Criterio | Estado |
|-----------|----------|--------|
| Training data quality | >70% verified | 52% ⚠️ (chain_1d 96%, heavy_hex 94%, ladder 23%❌) |
| Zoo coherence | <5 incoherencias | 19 ❌ (necesita --force-retrain) |
| Zoo model health | all scores >0.70 | 3/5 ⚠️ (ladder, square, triangular below) |
| Extrapolation N=30 | 3+ topologías | 2/5 ⚠️ (chain_1d✅, ladder✅, falta heavy_hex/square/triangular) |
| Extensive scaling proof | |ΔE|/N ≈ const | chain_1d: 7.4e-3 @ N=30 ✅ |
| Hardware validation | 1 topology deployed | Pendiente |
| Thesis figures | All generated | Pendiente |
| Gap masking documented | dual criterion | ✅ (20 configs afectados) |

---

## Comandos Quick Reference

```bash
# Estado actual
.venv/bin/python scripts/maintenance/quick_health_check.py
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py --dry-run

# Reentrenar + iterative improve
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology TOPO --train-n 6 --target-n N --mode iterative --force-retrain

# Extrapolación large-N
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology TOPO --target-n 30 40

# Dashboard refresh
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard; generate_model_quality_dashboard()"

# Scaling report
.venv/bin/python scripts/maintenance/generate_scaling_report.py
```
