---
inclusion: fileMatch
fileMatchPattern: "**/accelerated*,**/model_zoo*,**/quality_predictor*,**/run_accelerated*,**/bond_resolved/*,**/multi_n_*,**/eval_cache*,**/ground_truth_cache*,**/PLAN_ITERATIVE*,**/PLAN_MULTI_N*"
---

# Accelerated Pipeline — Guía Operativa

## Runner Principal

```
scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py
```

## Modos de Operación

| Flag | Qué hace |
|------|----------|
| `--train-n X --target-n X` | Genera VQE data para N=X, guarda NPZ, entrena MPNN, evalúa |
| `--from-zoo --target-n Y` | Carga modelo del zoo, predice en N=Y sin entrenar |
| `--multi-n-train --target-n Y` | Combina TODOS los NPZ disponibles, entrena multi-N, predice en Y |
| `--multi-n-train` (sin --force-retrain) | Si ya hay modelo multi-N en zoo → lo reutiliza sin re-entrenar |
| `--force-retrain` | Fuerza re-entrenamiento aunque exista modelo |
| `--iterative-improve` | **SIEMPRE USAR.** Loop: predict → refine → upsert NPZ → retrain → repeat |
| `--max-iterations N` | Máximo de iteraciones para `--iterative-improve` (default: 3) |
| `--improvement-threshold X` | Stop si mejora < X entre iteraciones (default: 0.01) |
| `--budget-only` | Solo estima costo (dry-run de `--iterative-improve`) |
| `--force-method COBYLA` | Forzar COBYLA explícitamente (recomendado para bond-resolved) |
| `--bidirectional-anchors` | Ascending merge selectivo en anchor VQE (mejora convergencia) |
| `--no-eval-cache` | Fuerza re-evaluación de circuitos (ignora cache) |

## REGLA FUNDAMENTAL: Siempre --iterative-improve

**NUNCA** usar `--active-rounds` solo ni `--from-zoo` sin `--iterative-improve`.
Sin iterative-improve, los θ_opt refinados se pierden en el JSON y no se
reutilizan para retraining. El iterative-improve cierra el ciclo:
predict → refine → **guardar en NPZ** → retrain → mejor modelo.

La única excepción: `--train-n X --target-n X` (same-N training) donde el
objetivo es GENERAR training data, no predecir cross-N.

## Caches (automáticos, sin flags)

| Cache | Qué guarda | Path |
|-------|-----------|------|
| Ground Truth | Energías DMRG/ExactDiag | `data/ground_truth_cache.json` |
| Eval Cache | Energías de circuit evaluation | `data/eval_cache.json` |
| Training Data | θ_opt por N (NPZ) | `data/multi_n_training/{topo}_N{X}_p1.npz` |
| Model Zoo | Checkpoints MPNN con SHA256 | `data/model_zoo/checkpoints/` |

Al re-correr el mismo comando, ground truth y eval cache evitan recomputar.

## Restricciones de Topología

- **ladder**: requiere N par
- **chain_1d**: cualquier N ≥ 4
- **No mezclar** datos de topologías distintas en un modelo

## Restricciones de N

| N | Backend eval | Fidelity | Active Learning VQE |
|---|---|---|---|
| ≤ 22 | Statevector (exacto) | ✅ Disponible | ✅ 200 iters |
| 23-30 | MPSBackend(χ=64) | ❌ No disponible | ⚠️ 50 iters (lento) |
| > 30 | MPSBackend(χ=64) | ❌ | ❌ Impracticable |

## Flujo Típico (nueva topología)

```bash
# 1. Generar datos para 2-3 tamaños de N
.venv/bin/python run_accelerated_cross_n.py --train-n 8 --target-n 8 --topology ladder --h-min 2.0 --h-max 3.5 --h-points 14 --n-anchors 14 --maxiter 1600 --n-restarts 12
.venv/bin/python run_accelerated_cross_n.py --train-n 14 --target-n 14 --topology ladder --h-min 2.0 --h-max 3.5 --h-points 14 --n-anchors 14 --maxiter 1600 --n-restarts 12

# 2. Entrenar multi-N y predecir
.venv/bin/python run_accelerated_cross_n.py --multi-n-train --target-n 20 --topology ladder --h-min 2.0 --h-max 3.5 --h-points 12

# 3. Re-correr (reutiliza modelo + caches)
.venv/bin/python run_accelerated_cross_n.py --multi-n-train --target-n 20 --topology ladder --h-min 2.0 --h-max 3.5 --h-points 12
```

## Modelo Multi-N

- `--multi-n-train` busca archivos en `data/multi_n_training/{topology}_N*_p1.npz`
- Combina automáticamente todos los N disponibles
- Entrena UnifiedMPNN con `norm_type="none"` (obligatorio para cross-N)
- Exporta al zoo con nombre `unified_tfim_br_{topo}_multiN_{Ns}_p{p}.pt`
- Segunda ejecución: detecta modelo existente y lo carga sin re-entrenar

## Active Learning 

- Identifica puntos con ΔE/gap > 5% o fidelity < 90%
- VQE warm-start desde θ_pred del GNN (converge rápido)
- Incluye cold-start comparison (random init) para cuantificar beneficio
- Guarda θ_opt refinados (pueden alimentar re-training)
- Para N > 22: maxiter=50 con MPSBackend (más lento pero viable)

## Iterative Improvement (`--iterative-improve`)

Loop automático: predict → identify failures → VQE refine → upsert NPZ → retrain → repeat.

```bash
# Típico: N=20 con 12 h-points, máx 3 iteraciones
.venv/bin/python scripts/.../run_accelerated_cross_n.py \
  --iterative-improve --target-n 20 --topology chain_1d \
  --h-min 2.0 --h-max 3.5 --h-points 12 --max-iterations 3

# Solo estimar costo (no ejecuta)
.venv/bin/python scripts/.../run_accelerated_cross_n.py \
  --iterative-improve --budget-only --target-n 20 --topology chain_1d
```

### Comportamiento del loop

1. Carga modelo del zoo (multi-N o single-N)
2. Predice θ para todos los h-points
3. Evalúa ΔE/gap (cache hits para θ repetidos)
4. Filtra failures vs ansatz-limited (h < `get_regime_threshold()`)
5. Anti-regression: compara θ_pred vs θ_prev del NPZ → usa el mejor como init
6. VQE warm-start solo en failures no-ansatz-limited
7. Upsert θ_opt refinados en NPZ acumulativo (menor energía gana)
8. Retrain multi-N con todos los NPZ disponibles
9. Exporta modelo al zoo → siguiente iteración usa modelo mejorado

### Criterios de parada

- `pass_rate ≥ 90%` → objetivo alcanzado
- `improvement < threshold` → modelo convergió
- Failures todos en `h < h_min_valid` → límite del ansatz
- `max_iterations` alcanzado
- Cache hit rate > 95% → modelo predice mismos θ (convergió)

### Cache reuse en el loop

- **GT**: cache hit 100% desde iter 1 (compute una vez, reusar siempre)
- **Eval**: iter 2+ tiene hits masivos (θ no cambió para puntos que pasaron)
- **NPZ**: upsert acumulativo — siempre tiene el MEJOR θ por h-point
- **Crash recovery**: idempotente — re-run retoma automáticamente vía caches

## Parámetros Recomendados por Topología

### chain_1d (coordinación z=2)
| Parámetro | Valor |
|-----------|:---:|
| h-min | 2.0 |
| h-max | 3.5 |
| h-points | 14 |
| n-anchors | 14 |
| maxiter | 1500 |
| n-restarts | 10 |
| force-method | COBYLA |

### ladder (coordinación z=3)
| Parámetro | Valor |
|-----------|:---:|
| h-min | **2.9** (safe zone para p=1, N≥10) |
| h-max | 3.5 |
| h-points | 10 |
| n-anchors | 10 |
| maxiter | **3000** (23+ params necesita más budget) |
| n-restarts | **15** |
| force-method | COBYLA |
| bidirectional-anchors | **sí** (recupera puntos borderline) |

### h_min safe zone (p=1, empírico)
| Topología | N=6 | N=8 | N=10 | N=14 | N=20 |
|---|:---:|:---:|:---:|:---:|:---:|
| chain_1d | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| ladder | 2.5 | 2.7 | 2.9 | 3.0 | ~3.1 |

El frontier de ladder crece con N (~0.05/qubit) porque la coordinación z=3
requiere más profundidad de circuito para correlacionar vecinos.

## Análisis de Resultados

```bash
.venv/bin/python -m project_health.analysis.accelerated_cross_n_analyzer --verbose
```

## Datos que se Reutilizan Automáticamente

1. **NPZ de training** → MultiNAggregator los encuentra y combina
2. **Ground truth DMRG** → cache hit en evaluaciones repetidas
3. **Circuit evaluations** → eval cache evita re-simular mismo (N, h, θ)
4. **Modelo del zoo** → `--multi-n-train` lo carga sin re-entrenar
5. **Quality predictions** → ResultIndex histórico alimenta el QualityPredictor

## Anti-patterns

- **NUNCA** usar `--active-rounds` sin `--iterative-improve` (datos se pierden)
- **NUNCA** usar `--from-zoo` solo para evaluar (sin iterative-improve no guarda θ)
- NO usar h < h_min_safe para la topología (límite del ansatz, no del modelo)
- NO usar datos de training con pass@5% < 80% para multi-N (contaminan el modelo)
- NO mezclar topologías en el mismo training
- NO usar N impar con topology=ladder
- NO confiar solo en ΔE/gap — reportar siempre |ΔE| y fidelity (gap puede sesgar)
- NO olvidar `--force-method COBYLA` para bond-resolved (auto-switch causa bugs con cache)
- NO usar `--force-retrain` sin razón (desperdicia el modelo existente)

## Métricas: Siempre Reportar las 3

1. **ΔE/gap** — pass/fail criterion (<5%)
2. **|ΔE|** — error físico absoluto (no se oculta tras gap grande)
3. **Fidelity** — calidad del estado (cuando N ≤ 22)

Cuidado: para h > 3.0 el gap es ~4-5, así que ΔE/gap < 5% equivale a |ΔE| < 0.20-0.25.
Eso es un error real significativo aunque ΔE/gap parezca bajo.

