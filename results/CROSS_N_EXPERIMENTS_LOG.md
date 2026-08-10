# Cross-N Accelerated Experiments — Log de Resultados

**Fecha:** 2026-07-29
**Runner:** `scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py`
**Modelo:** TFIM bond-resolved, p=1
**Método:** UnifiedMPNN (GINConv, norm_type=none, hidden=256, 3 layers)

---

## CHAIN_1D — Resultados

### Training Quality (same-N evaluation)

| N | h-range | Points | Pass @5% | Pass @10% | Mean ΔE/gap | Mean Fidelity | Anchors | Maxiter | Restarts |
|---|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6 | [2.2, 3.5] | 12 | **83%** | 100% | 0.029 | 0.99 | 6 | 1000 | 7 |
| 8 | [2.2, 3.5] | 12 | **67%** | 92% | 0.047 | 0.98 | 6 | 1000 | 7 |
| 10 | [2.0, 3.5] | 14 | **100%** | 100% | 0.011 | — | 14 | 1400 | 10 |
| 10 (v2) | [2.0, 3.5] | 12 | 50% | 75% | 0.069 | 0.98 | 6 | 1000 | 7 |

**Observación:** La calidad del training depende fuertemente de `n_anchors` y `maxiter`. Con 14 anchors y 1400 maxiter → 100% pass. Con 6 anchors y 1000 maxiter → 50%.

### Cross-N Predictions (modelo entrenado en N=10 con 14 anchors)

| N_target | h-range | Points | Pass @5% | Pass @10% | Mean ΔE/gap | Fidelity | Active Learning |
|----------|---------|--------|:---:|:---:|:---:|:---:|:---:|
| 15 | [2.0, 3.5] | 12 | **83%** | **100%** | 0.028 | 0.93 | — |
| 15 | [1.0, 4.5] | 12 | 67% | 75% | 0.621 | 0.93 | 0 refined |
| 20 | [2.0, 3.5] | 14 | **64%** | **93%** | 0.046 | — | — |
| 22 | [2.0, 3.5] | 12 | 58% | 83% | 0.055 | — | — |

### Cross-N per-h breakdown (N=10 → N=20, best run)

| h | ΔE/gap | |ΔE| | @5% | @10% |
|:---:|:---:|:---:|:---:|:---:|
| 3.50 | 0.016 | 8.2e-2 | ✓ | ✓ |
| 3.38 | 0.018 | 8.4e-2 | ✓ | ✓ |
| 3.27 | 0.019 | 8.9e-2 | ✓ | ✓ |
| 3.15 | 0.022 | 9.5e-2 | ✓ | ✓ |
| 3.04 | 0.025 | 1.0e-1 | ✓ | ✓ |
| 2.92 | 0.028 | 1.1e-1 | ✓ | ✓ |
| 2.81 | 0.032 | 1.2e-1 | ✓ | ✓ |
| 2.69 | 0.037 | 1.3e-1 | ✓ | ✓ |
| 2.58 | 0.044 | 1.4e-1 | ✓ | ✓ |
| 2.46 | 0.052 | 1.5e-1 | ~ | ✓ |
| 2.35 | 0.063 | 1.7e-1 | ~ | ✓ |
| 2.23 | 0.078 | 1.9e-1 | ~ | ✓ |
| 2.12 | 0.096 | 2.2e-1 | ~ | ✓ |
| 2.00 | 0.119 | 2.4e-1 | ✗ | ✗ |

**Conclusión chain_1d:** Cross-N funciona excelente para h > 2.5 (100% pass). Degrada gradualmente hacia h=2.0.

---

## LADDER — Resultados

### Training Quality (same-N evaluation)

| N | h-range | Points | Pass @5% | Pass @10% | Mean ΔE/gap | Mean Fidelity | Anchors | Maxiter | Restarts |
|---|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6 | [2.2, 3.5] | 12 | **83%** | 100% | 0.025 | 0.99 | 6 | 1000 | 7 |
| 8 | [2.0, 3.5] | 14 | 57% | 79% | 0.065 | 0.98 | 14 | 1600 | 12 |
| 10 | [2.0, 3.5] | 14 | 43% | 64% | 0.100 | 0.97 | 14 | 1600 | 12 |
| 14 | [2.0, 3.5] | 14 | **29%** | 50% | 0.179 | 0.95 | 14 | 1600 | 12 |

**Problema:** Ladder con N ≥ 10 tiene training quality muy baja (43% pass). El VQE no converge bien con COBYLA 1600 iters para 23+ params.

### Cross-N Predictions (ladder)

| Source | N_target | Pass @5% | Pass @10% | Mean ΔE/gap | Fidelity | Notas |
|--------|----------|:---:|:---:|:---:|:---:|---|
| N=10 (fresh) | 14, 16, 20 | **0%** | **0%** | 1.6-2.7 | 0.52-0.68 | Colapso total |
| N=10 (fresh) | 20 | 8% | 33% | 0.33 | 0.92 | Muy pobre |
| N=10 + AL | 20 | 8% | 33% | 0.38 | 0.97 | AL no ayudó mucho |
| N=6 (zoo) | 20 | 0% | 0% | 4.27 | — | Inútil |
| Multi-N (8+14) | 20 | — | — | — | — | Pendiente evaluación |

**Conclusión ladder:** Cross-N NO funciona para ladder con el training actual. La calidad de los datos de entrenamiento es insuficiente (el VQE no converge).

---

## Comparación Chain_1d vs Ladder

| Aspecto | Chain_1d | Ladder |
|---------|:---:|:---:|
| Training @5% (N=10) | **100%** | 43% |
| Cross-N @10% (N=20) | **93%** | 33% |
| Params por punto (N=10) | 19 | 23 |
| Coordinación | 2 | 3 |
| VQE difficulty | Baja | **Alta** |
| Cross-N viable | ✅ h > 2.5 | ❌ No con training actual |

---

## Modelos en Zoo

| Checkpoint | Topology | N_train | Pass Rate | Notas |
|-----------|----------|:---:|:---:|---|
| `unified_tfim_br_chain_n10_p1_*.pt` | chain_1d | 10 | 100% | 14 anchors, high quality |
| `unified_tfim_bond_resolved_ladder_n6_p1_*.pt` | ladder | 6 | 83% | 6 anchors |
| `unified_tfim_br_ladder_multiN_8+14_p1.pt` | ladder | 8+14 | 0% | Datos de baja calidad |

## NPZ Training Data

| File | Topology | N | Points | Calidad |
|------|----------|---|--------|---------|
| `chain_1d_N10_p1.npz` | chain_1d | 10 | 14 | ✅ Alta (100% pass) |
| `ladder_N8_p1.npz` | ladder | 8 | 14 | ⚠️ Media (57% pass) |
| `ladder_N14_p1.npz` | ladder | 14 | 14 | ❌ Baja (29% pass) |

---

## Lecciones Aprendidas

1. **La calidad del training es determinante.** Chain_1d con 100% training → 93% cross-N. Ladder con 43% training → 0% cross-N.

2. **Más anchors + más maxiter = mejor training.** 14 anchors con 1400 maxiter vs 6 anchors con 1000 maxiter: diferencia entre 100% y 50% pass.

3. **Ladder necesita un approach diferente:**
   - Más iteraciones (3000+) o L-BFGS-B en vez de COBYLA
   - h-range más restrictivo (h > 2.5 para N=10 ladder)
   - Posiblemente p=2 para tener más expressibility

4. **Active learning no ayuda si el warm-start (θ_pred) está en un valle incorrecto.** Para ladder N=20, el θ_pred del GNN estaba tan lejos del correcto que 200 iters de VQE no eran suficientes.

5. **h < 2.0 es zona de falla universal** — tanto para chain_1d como ladder con cross-N.

---

## Próximos Pasos

- [ ] Mejorar training ladder: más maxiter, L-BFGS-B, o ascending pass bidireccional
- [ ] Generar ladder N=10 con training de alta calidad (como chain_1d)
- [ ] Multi-N ladder con datos de calidad → re-evaluar cross-N
- [ ] Implementar `--iterative-improve` para loop cerrado automático
- [ ] Poblar GroundTruthCache con datos históricos (522 runs)


---

## Plan de Ejecución — Actualizado (2026-07-30)

**Estado:** Las mejoras de código están implementadas. Lo que sigue es ejecución.

### Cambios implementados (resumen)

| Componente | Cambio | Impacto |
|---|---|---|
| `AcceleratedConfig` | +`force_method`, +`bidirectional_anchors` | Control explícito del optimizador |
| `_run_anchor_vqe` | Bidirectional ascending merge con `select_suspicious_points()` | Recupera puntos atrapados en mínimos locales |
| `_compute_ground_truth` | Integrado con `GroundTruthCache` | Evita re-cómputo de ED/DMRG entre sesiones |
| `section_train` NPZ save | Guarda `e_vqe`, `gaps`, `de_gaps` (antes faltaban) | MultiNAggregator filtra correctamente |
| `_upsert_npz` | Preserva `gaps`, recomputa `de_gaps` al guardar | NPZ siempre tiene datos filtrable |
| `MultiNAggregator.scan()` | Computa `de_gaps` on-the-fly si falta del NPZ | Defensa contra NPZ incompletos |
| Defaults del runner | h_min=2.0, anchors=14, maxiter=1500, restarts=10, force_method=COBYLA | Training de alta calidad por defecto |

### Paso 1: Generar training de alta calidad para LADDER

```bash
# Correr con bidirectional + COBYLA forzado + parámetros agresivos
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --train-n 10 --target-n 10 --topology ladder \
  --h-min 2.2 --h-max 3.5 --h-points 14 --n-anchors 14 \
  --maxiter 3000 --n-restarts 15 \
  --force-method COBYLA --bidirectional-anchors
```

**Expectativa:** pass_rate ≥ 80% (vs 43% anterior). Si no alcanza:

```bash
# Fallback: L-BFGS-B (más caro pero usa curvatura)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --train-n 10 --target-n 10 --topology ladder \
  --h-min 2.2 --h-max 3.5 --h-points 14 --n-anchors 14 \
  --maxiter 500 --n-restarts 10 \
  --force-method L-BFGS-B --bidirectional-anchors
```

**Criterio de éxito:** NPZ `ladder_N10_p1.npz` con ≥80% de puntos con de_gap < 0.05.
Verificar con: `.venv/bin/python scripts/maintenance/inspect_data_stores.py`

**Lo que ocurre automáticamente:**
- GT cache se llena con los 14 h-points de N=10 ladder (si no estaban)
- EvalCache acumula las evaluaciones del VQE (miles de entries)
- NPZ se guarda con campos completos (e_vqe, gaps, de_gaps)
- Modelo se exporta al zoo si pass > 80%

### Paso 2: Validar transferibilidad θ para Ladder

Solo después de tener training de calidad. Análisis rápido (~2 min):

```bash
.venv/bin/python scripts/maintenance/inspect_data_stores.py
# Verificar que ladder_N10_p1.npz tiene |ΔE| bajos y de_gaps < 0.05
```

Si ladder N=10 pasa ≥ 80%, correr cross-N inmediato:

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --from-zoo --target-n 14 16 20 --topology ladder \
  --h-min 2.2 --h-max 3.5 --h-points 12
```

**Criterio:** Si N=14 pasa ≥ 50% @10% → cross-N es viable para ladder.
Si N=14 pasa < 25% → la topología no es transferible con single-N training.

### Paso 3: Multi-N Training para chain_1d (paralelo, independiente)

Generar datos faltantes N=8 y N=12:

```bash
# N=8 (statevector, ~2 min)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --train-n 8 --target-n 8 --topology chain_1d \
  --h-min 2.0 --h-max 3.5 --h-points 14 --n-anchors 14 \
  --maxiter 1500 --n-restarts 10 --force-method COBYLA

# N=12 (statevector, ~4 min)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --train-n 12 --target-n 12 --topology chain_1d \
  --h-min 2.0 --h-max 3.5 --h-points 14 --n-anchors 14 \
  --maxiter 1500 --n-restarts 10 --force-method COBYLA
```

Luego entrenar multi-N:

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --multi-n-train --force-retrain --topology chain_1d \
  --target-n 15 20 22 --h-min 2.0 --h-max 3.5 --h-points 12
```

### Paso 4: Iterative Improve para chain_1d N=20

Una vez que el modelo multi-N exista en el zoo:

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --iterative-improve --target-n 20 --topology chain_1d \
  --h-min 2.0 --h-max 3.5 --h-points 14 --max-iterations 3
```

**Lo que ocurre:**
1. GT cache: 100% hit (chain_1d N=20 ya tiene 16 entries)
2. Predice con modelo multi-N → evalúa → identifica failures
3. VQE refine en failures → upsert θ_opt al NPZ con gaps+de_gaps
4. Retrain con MultiNAggregator (filtra max_de_gap=0.10, ahora on-the-fly)
5. Re-export al zoo → siguiente iteración mejora

### Paso 5: Escalar a N=30 (solo si N=20 pasa ≥ 80%)

```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
  --iterative-improve --target-n 30 --topology chain_1d \
  --h-min 2.5 --h-max 3.5 --h-points 8 --max-iterations 2
```

Restricciones para N=30: MPS backend (auto-seleccionado), maxiter=50 en refine,
solo 3 puntos refinados por iteración. h-range restringido a zona segura.

### Orden de ejecución y dependencias

```
Paso 1 (ladder training)  ──┐
                             ├── Paso 2 (validar cross-N ladder)
Paso 3 (chain_1d multi-N) ──┤
                             ├── Paso 4 (iterative improve N=20)
                             │
                             └── Paso 5 (N=30, solo si Paso 4 OK)
```

Pasos 1 y 3 son **independientes**. Ejecutar en paralelo (en 2 terminales).
Pasos 4 y 5 dependen de Paso 3. Paso 2 depende de Paso 1.


---

## Resultados Sesión 2026-07-31

### Ladder N=10 Training (zona segura h=[2.9, 3.5])

**Run:** `run_20260730_233833.json` | Config: COBYLA 3000iter, 15 restarts, bidirectional
**Training pass: 100% (10/10 @5%)** | mean ΔE/gap=0.028 | F_mean=0.990

| h | ΔE/gap | |ΔE| | Fidelity |
|:---:|:---:|:---:|:---:|
| 3.50 | 0.019 | 0.078 | 0.994 |
| 3.43 | 0.020 | 0.079 | 0.994 |
| 3.37 | 0.021 | 0.082 | 0.993 |
| 3.30 | 0.023 | 0.085 | 0.992 |
| 3.23 | 0.025 | 0.090 | 0.991 |
| 3.17 | 0.028 | 0.096 | 0.990 |
| 3.10 | 0.031 | 0.103 | 0.989 |
| 3.03 | 0.035 | 0.111 | 0.988 |
| 2.97 | 0.040 | 0.121 | 0.986 |
| 2.90 | 0.046 | 0.133 | 0.984 |

Modelo exportado al zoo: `unified_tfim_bond_resolved_ladder_n10_p1_20260731T023820.pt`

### Ladder Cross-N (modelo de N=10, zona h=[2.9, 3.5])

**Run:** `run_20260730_234920.json`

| N_target | @5% | @10% | mean ΔE/gap | mean |ΔE| | F_mean | Antes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 14 | 6/10 (60%) | 10/10 (100%) | 0.046 | 0.15 | 0.985 | 0% |
| 16 | 5/10 (50%) | 10/10 (100%) | 0.055 | 0.17 | 0.982 | 0% |
| 20 | 2/10 (20%) | 8/10 (80%) | 0.073 | 0.22 | 0.976 | 0% |

### Chain_1d N=8 Training

**Run:** `run_20260730_225029.json` | Config: COBYLA 1500iter, 10 restarts
**Training pass: 100% (14/14 @5%)** | mean ΔE/gap=0.009 | F_mean=0.997

### Observaciones clave

1. **Ladder cross-N funciona** cuando se restringe al h-range válido (h≥2.9 para p=1)
2. **Error absoluto |ΔE| crece con N**: N=14→0.15, N=16→0.17, N=20→0.22 (no con ΔE/gap que lo oculta)
3. **Fidelity degrada suavemente**: 0.985→0.982→0.976 (correlaciona con |ΔE|)
4. **El 100% @10% para N=14,16** confirma transferibilidad en la safe zone
5. **N=20 @80% sin active learning ni iterative-improve** — margen de mejora claro

### Bug encontrado y corregido

- `CachedBackend.set_h()` no se llamaba en `_run_anchor_vqe` → cache key con h=0 → evaluaciones incorrectas entre h-points
- 20,220 entries corruptas eliminadas del EvalCache
- `EvalCache._save()` ahora usa atomic write (tmp+rename) para prevenir corrupción

### Datos persistidos

- NPZ: `data/multi_n_training/ladder_N10_p1.npz` (10 pts, h=[2.9,3.5], 100% quality)
- NPZ: `data/multi_n_training/chain_1d_N8_p1.npz` (14 pts, h=[2.0,3.5], 100% quality)
- GT cache: +19 entries (ladder N=14,16,20 nuevos h-points)
- Zoo: modelo ladder N=10 con pass_rate=100%


---

## Resultados Sesión 2026-07-31 (continuación)

### Estado final de los data stores

- GT cache: **310 entries**, chain_1d + ladder, N=[4-24]
- EvalCache: **50,000 entries** (lleno, funcional, atomic write)
- NPZ training: **7 archivos** (ver tabla abajo)
- Zoo: **10 modelos** registrados, 0 orphans

### NPZ Training Data — Estado Final

| NPZ | Points | Params | h-range | |ΔE| range |
|-----|:---:|:---:|:---:|:---:|
| chain_1d N=8 | 14 | 15 | [2.0, 3.5] | [0.01, 0.05] |
| chain_1d N=12 | 14 | 23 | [2.0, 3.5] | [0.02, 0.09] |
| chain_1d N=20 | 5 | 39 | [3.0, 3.5] | [0.04, 0.06] |
| ladder N=6 | 12 | 13 | [2.5, 3.5] | [0.03, 0.09] |
| ladder N=8 | 12 | 18 | [2.7, 3.5] | [0.05, 0.11] |
| ladder N=10 | 10 | 23 | [2.9, 3.5] | [0.07, 0.13] |
| ladder N=14 | 10 | 33 | [3.0, 3.5] | [0.11, 0.17] |

### Mejores resultados Cross-N (por topología)

#### chain_1d (modelo single-N=10, 14 puntos de training)

| N_target | @5% | @10% | mean |ΔE| | F_mean | h-range evaluado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 15 | 92% | 100% | — | — | [2.0, 3.5] |
| 20 | 83% | 100% | 0.13 | — | [2.0, 3.5] |
| 22 | 58% | 83% | — | — | [2.0, 3.5] |

#### ladder (modelo multi-N=6+8+10+14, safe zone)

| N_target | @5% | @10% | mean |ΔE| | F_mean | h-range evaluado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 14 (training) | 80% | 100% | 0.04 | — | [3.0, 3.5] |
| 14 (cross-N) | 60% | 100% | — | 0.985 | [2.9, 3.5] |
| 16 (cross-N) | 50% | 100% | 0.17 | 0.982 | [2.9, 3.5] |
| 20 (cross-N, +AL) | 20% | 80% | 0.23 | 0.976 | [2.9, 3.5] |

### Runs ejecutados esta sesión

| Run | Topology | Config | Resultado |
|-----|----------|--------|-----------|
| `run_20260730_225029` | chain_1d | N=8 training | 100% pass, 14pts |
| `run_20260730_231305` | ladder | N=10 training h=[2.2,3.5] | 50% pass (safe zone=100%) |
| `run_20260730_233833` | ladder | N=10 training h=[2.9,3.5] | **100% pass**, zoo export |
| `run_20260730_234920` | ladder | Cross-N N=14,16,20 | 100%/100%/80% @10% |
| `run_20260731_110958` | ladder | N=14 training h=[3.0,3.5] | **80% pass** |
| `run_20260731_111916` | ladder | Multi-N train + cross-N N=20 | 60% @10% |
| `run_20260731_112645` | ladder | Cross-N N=20 + AL (5 refined) | **80% @10%** |
| `run_20260731_114513` | chain_1d | Multi-N + iterative N=20 | 50% pass (converged) |

### Mejoras de código implementadas

1. `AcceleratedConfig` — +force_method, +bidirectional_anchors
2. `_run_anchor_vqe` — set_h() fix, bidirectional ascending merge
3. `_compute_ground_truth` — integrado con GroundTruthCache
4. `_upsert_npz` — preserva gaps, recomputa de_gaps
5. `MultiNAggregator.scan()` — de_gaps on-the-fly si falta
6. `EvalCache._save()` — atomic write (tmp+rename)
7. `analysis/metrics.py` — is_point_failure(), identify_failures() (criterio dual)
8. `framework/cli.py` — add_iterative_improve_args() compartido
9. `runner_base.py` — run_quality_check(), _build_physics_config()
10. `run_noiseless_pipeline` — adoptó is_point_failure + identify_failures
11. Defaults del runner: h_min=2.0, anchors=14, maxiter=1500, restarts=10

### Bugs corregidos

1. **set_h en CachedBackend** — evaluaciones con cache key h=0 → valores incorrectos entre puntos
2. **20,220 entries corruptas** limpiadas del EvalCache
3. **EvalCache corrupción** — atomic write previene truncamiento en crash
4. **MultiNAggregator** — de_gap=0 para NPZ sin campo → ahora conservativo (de_gap=1.0)

### Limitaciones identificadas

- **Ladder p=1**: h_frontier crece con N (~0.05/qubit). Safe zone: N=10→h≥2.9, N=14→h≥3.0
- **force_method="L-BFGS-B"**: el COBYLA_AUTO_SWITCH_THRESHOLD lo overridea para n_params>8
- **Iterative improve para N>22**: requiere MPS (lento) y no tiene fidelity
- **chain_1d multi-N**: 50% pass_rate para N=20 (no mejora sobre single-N=10 que ya daba 93%)
