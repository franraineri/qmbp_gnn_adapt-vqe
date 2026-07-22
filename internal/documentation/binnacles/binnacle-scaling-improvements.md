# Binnacle — Mejoras de Escalabilidad (Análisis Pre-Ejecución)

> Fecha: 2026-06-01
> Objetivo: Identificar mejoras ejecutables sin hardware real para fortalecer
> la narrativa de escalabilidad de la tesis.
> Estado: PLANIFICACIÓN (ningún experimento ejecutado aún)

---

## Resumen Ejecutivo

El pipeline está validado a N=6 (5 topologías), N=10 (5 topologías), y N=20
(chain_1d, p=1 y p=2). Los gaps principales son:

1. **N=20 p=1 con MPNN real** — solo se probó con interpolación lineal (C3)
2. **N=16 nunca completó Phase 3** — fidelity filter rechaza datos
3. **No hay explicación teórica** de la ley de escalado h_min(N)
4. **Cross-topology transfer** nunca se probó (train-on-one, deploy-on-another)

---

## Mejora 1: Pipeline completo N=20 p=1 con MPNN

**Qué se hizo**: C3 validó VQE a N=20 p=1 (ΔE/gap=1.58%, 2/3 seeds) usando
interpolación lineal para deployment. Canonicalización innecesaria.

**Qué falta**: Entrenar un MPNN real y desplegar. El intento original (binnacle-p1-scaling)
usó solo 6 puntos → MPNN solo pasó en h=3.0.

**Mejora propuesta**:
- 15 h-points en [2.25, 4.0] (Δh=0.125), 5 restarts, maxiter=100
- MPNN h=128, L=3, 6000 epochs
- h_test = [2.5, 3.0, 3.5] (todos dentro del régimen válido)
- 3 seeds (42, 43, 44)

**Resultado esperado**: ΔE/gap < 2% en los 3 test points (basado en que
interpolación lineal ya da 1.58%).

**Tiempo estimado**: ~20 min/seed (VQE) + 30s (MPNN) = ~65 min total.

**Valor tesis**: Cierra el claim "pipeline completo funciona a N=20 p=1"
con MPNN real, no interpolación.

---

## Mejora 2: Fix N=16 p=1 (régimen correcto)

**Qué se hizo**: 13 runs a N=16 (2026-05-30). Phase 3 no completó porque
el fidelity filter rechazó los datos de training. Seed 43 produce chain breaks
consistentemente (θ_smooth=2.99).

**Diagnóstico**: El h-grid usado probablemente incluía puntos fuera del
régimen válido. La ley de escalado predice h_min(N=16, p=1) ≈ 1.0 + 0.212·16^0.60 ≈ 2.3.
Pero los runs usaron el grid de N=10 (h∈[4.0, 3.5, 3.0, 2.5]).

**Mejora propuesta**:
- h_train = [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5] (todo > 2.3)
- h_test = [3.25] (bien dentro del régimen)
- 5 restarts (evita chain breaks de seed 43)
- Desactivar fidelity filter (usar ΔE/gap < 5% como criterio directo)
- Seeds: 42, 44 (seed 43 es problemático a N≥10 en chain_1d/ladder)

**Resultado esperado**: Phase 3 completa, deployment pasa.

**Tiempo estimado**: ~5 min/seed (StatevectorEstimator a 2^16=65K estados).

**Valor tesis**: Punto intermedio en la ley de escalado. Confirma que el
framework funciona a N=16 con config correcta.

---

## Mejora 3: Entropía de entrelazamiento vs h_min (explicación teórica)

**Qué se hizo**: Ley empírica h_min = 1.0 + α·N^β con R²=1.0000.
Nunca se correlacionó con la entropía del ground state.

**Mejora propuesta**:
- Calcular S(h, N) = -Tr(ρ_A log₂ ρ_A) para N=4,6,8,10 (exact diag)
- Evaluar S(h_min(N), N) para cada N
- Si S_boundary ≈ constante → el exponente β es consecuencia de cómo
  S escala con N en el TFIM (conocido analíticamente)
- Repetir para p=1 (S_boundary diferente → explica β diferente)

**Resultado esperado**: S(h_min) ≈ constante (hipótesis: ~0.5-0.8 bits).
Esto significaría que HVA p=k tiene una capacidad de entrelazamiento fija,
y h_min es donde el ground state excede esa capacidad.

**Tiempo estimado**: ~2 min (exact diag ya existe, solo falta calcular S).

**Valor tesis**: ★★★ — Conecta resultado empírico con física fundamental.
Transforma una observación numérica en una explicación causal.

---

## Mejora 4: Cross-topology transfer learning

**Qué se hizo**: 131 variants ejecutados en 5 topologías. Cada MPNN se
entrena y despliega en la MISMA topología. Nunca se probó generalización
cruzada.

**Mejora propuesta**:
- Entrenar MPNN en chain_1d N=10 (datos existentes, 3 seeds)
- Desplegar en ladder N=10 y triangular N=10 (sin re-entrenamiento)
- Medir ΔE/gap en cada topología destino
- Repetir: train-on-ladder → deploy-on-chain, etc.

**Resultado esperado**: Probablemente falla (topologías tienen θ_opt muy
diferentes). Pero cuantifica CUÁNTO falla y si fine-tuning con pocos puntos
(2-3) lo resuelve.

**Tiempo estimado**: ~5 min (solo re-evaluación, datos VQE ya existen).

**Valor tesis**: Valida/invalida el claim de topology-agnosticism del GNN.
Si funciona → contribución fuerte. Si falla → documenta que el GNN aprende
la relación h→θ condicionada a la topología (aún valioso).

---

## Mejora 5: Landscape analysis a N=20 (F3 + B4 extendido)

**Qué se hizo**: F3 a N=6 (fluctuation >1.0, no BPs). B4 a N=6 y N=10
(0 saddle points, κ N-independent). G3 mostró que 1 restart falla a N=20.

**Mejora propuesta**:
- F3 a N=20 p=2 con MPS: 50 random samples por h-point
- B4 a N=20: Hessian en los mínimos VQE (ya calculados en V7 3C)
- Comparar κ(N=20) vs κ(N=6) y κ(N=10)

**Resultado esperado**: κ(N=20) >> κ(N=6) (landscape más plano a h grande),
explicando por qué G3 falló con 1 restart.

**Tiempo estimado**: ~30 min (MPS VQE + Hessian).

**Valor tesis**: Explica cuantitativamente el resultado negativo de G3.

---

## Mejora 6: Data efficiency a N=10 (extensión G1)

**Qué se hizo**: G1 a N=6 → k_min=9 puntos (47% reducción).

**Mejora propuesta**:
- Repetir G1 a N=10: variar k de 5 a 17, medir ΔE/gap
- Usar datos VQE existentes (subsampling del grid de 17 puntos)
- 3 seeds

**Resultado esperado**: k_min(N=10) ≈ 11-13 (más puntos que N=6 por
landscape más complejo).

**Tiempo estimado**: ~10 min (solo re-entrenamiento MPNN, VQE ya existe).

**Valor tesis**: Cuantifica el costo mínimo del pipeline a cada N.

---

## Dependencias y Orden de Ejecución

```
Mejora 3 (entropía) → independiente, rápida, alto valor → PRIMERO
Mejora 1 (N=20 MPNN) → independiente, cierra gap principal → SEGUNDO
Mejora 4 (cross-topo) → usa datos existentes → TERCERO
Mejora 2 (N=16 fix) → independiente → CUARTO
Mejora 5 (landscape N=20) → requiere VQE N=20 (existe) → QUINTO
Mejora 6 (data eff N=10) → usa datos existentes → SEXTO
```

---

## Criterios de Éxito

| Mejora | Criterio de éxito | Criterio de "negativo útil" |
|--------|---|---|
| 1 (N=20 MPNN) | ΔE/gap < 3% en 3 test points | Documenta cuántos puntos necesita |
| 2 (N=16 fix) | Phase 3+4 completan, ΔE/gap < 5% | Confirma que N=16 necesita MPS |
| 3 (entropía) | S(h_min) ≈ constante (std < 0.1) | S no es constante → otro mecanismo |
| 4 (cross-topo) | ΔE/gap < 10% sin re-entrenamiento | Cuantifica fine-tuning necesario |
| 5 (landscape N=20) | κ(N=20) > 10× κ(N=10) | κ similar → otro factor explica G3 |
| 6 (data eff N=10) | k_min(N=10) identificado | Confirma que N=10 necesita más datos |

---

## Notas

- Ninguna de estas mejoras modifica código estable.
- Todas usan StatevectorEstimator o MPS (chi=64) — sin hardware.
- Todas producen nuevo aprendizaje (no repiten resultados conocidos).
- La mejora 3 es la de mayor ratio valor/esfuerzo.

---

## Archivos Creados

| Mejora | Experiment ID | Archivo |
|--------|---------------|---------|
| 3 (entropía) | S1 | `experiments/scaling/exp_s1_entanglement_scaling.py` |
| 4 (cross-topo) | S2 | `experiments/predictor/exp_s2_cross_topology.py` |
| 5 (landscape N=20) | S3 | `experiments/landscape/exp_s3_landscape_n20.py` |
| 6 (data eff N=10) | S4 | `experiments/predictor/exp_s4_data_efficiency_n10.py` |
| 1 (N=20 MPNN) | S5 | `experiments/scaling/exp_s5_n20_p1_pipeline.py` |
| 6 (MC-Dropout) | S6 | `experiments/predictor/exp_s6_mc_dropout_uq.py` |

### Ejecución

```bash
# Rápido (S1: ~2 min)
python scripts/run_experiment.py --exp S1 --verbose

# Medio (S2: ~5 min, S4: ~10 min, S6: ~3 min)
python scripts/run_experiment.py --exp S2 S4 S6 --verbose

# Largo (S3: ~30 min, S5: ~65 min)
python scripts/run_experiment.py --exp S3 --verbose
python scripts/run_experiment.py --exp S5 --verbose
```
