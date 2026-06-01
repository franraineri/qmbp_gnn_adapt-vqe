# Experimentos Posibles Sin Acceso a Hardware Real

> Análisis basado en el estado actual del proyecto (V8 completo, p=1 ZNE confirmado,
> 131+ variants ejecutados, 60+ experimentos benchmark).
>
> Fecha: 2026-06-01

---

## Contexto

La prioridad activa es el deployment en IBM Torino, pero sin acceso a hardware real,
estas líneas de trabajo producen **nuevo aprendizaje** usando solo simulación local
(StatevectorEstimator, MPS con chi=64).

---

## Priorización

| # | Experimento | Nuevo aprendizaje | Esfuerzo | Valor tesis |
|---|---|---|---|---|
| 1 | Extensión Heisenberg XXZ | Alto (nuevo modelo) | 2 semanas | ★★★ |
| 2 | Fix pipeline p=1 N=20 | Medio (confirma fix) | 2-3 días | ★★★ |
| 3 | Ley de escalado + entropía | Alto (explicación teórica) | 3-4 días | ★★★ |
| 4 | Cross-topology transfer | Alto (claim central) | 1-2 días | ★★☆ |
| 5 | Landscape N=20 (F3+B4) | Alto (explica G3 failure) | 2-3 días | ★★☆ |
| 6 | MC-Dropout UQ | Medio (fix de G2) | 1 día | ★★☆ |
| 7 | Data efficiency N=10/20 | Medio (extensión G1) | 1-2 días | ★☆☆ |
| 8 | D1 phase detection N=20 | Medio (extensión D1) | 1-2 días | ★☆☆ |
| 9 | N=16 p=1 MPS | Bajo (punto intermedio) | 1 día | ★☆☆ |
| 10 | Bootstrap UQ | Bajo (alternativa a #6) | 2 días | ★☆☆ |

---

## 1. Extensión Heisenberg XXZ (spec existente)

**Hipótesis**: Caracterizar los límites de expresividad del HVA p≤2 para modelos
más allá del TFIM.

- Spec completo en `.kiro/specs/heisenberg-xxz-extension/`
- Solo requiere N=6, StatevectorEstimator
- E4 ya mostró fidelity=0.89 con g=0.1; Heisenberg puro da 22% fidelity max
- **Resultado esperado**: Negativo riguroso — cuantifica *por qué* HVA p=2 es
  TFIM-específico (entropía de entrelazamiento vs capacidad del ansatz)
- **Valor**: Cierra el claim de model-specificity con evidencia cuantitativa

---

## 2. Fix Pipeline p=1 N=20

**Hipótesis**: Con más puntos de entrenamiento + 5 restarts, el MPNN despliega
correctamente a N=20 p=1 en todo el régimen válido (h≥2.25).

### Lo que YA se hizo (resumen cronológico)

**Fase 1 — binnacle-p1-scaling (2026-05-21):**
- VQE a N=20 p=1 con 5 restarts, maxiter=500, StatevectorEstimator
- Seeds 42/43 perfectos (h≥2.25 pasa), seed 44 falla en h≥3.0 (bad init)
- MPNN entrenado con solo 6 puntos válidos → deployment solo pasa en h=3.0
- h=2.5 da 10% ΔE/gap, h=2.0 da 76% → MPNN no generaliza con 6 puntos
- Diagnóstico: simetría Z₂ + training set insuficiente

**Fase 2 — Experimento C3 (2026-05-22, V8 Round 1):**
- 3 runs con config mejorada: 3 restarts, 100 maxiter, MPS chi=64
- **Run 2**: 3/3 seeds pasan (ΔE/gap = 0.0158 ± 0.0093) ✅
- **Run 3**: 2/3 seeds pasan (seed 44 cae en local minimum ΔE/gap=0.437)
- **Hallazgo clave**: Canonicalización tiene 0% efecto — el "problema Z₂"
  era un artefacto de VQE insuficiente (1 restart, 50 maxiter)
- **Conclusión**: Con warm-start descendente + 3 restarts, todos los seeds
  convergen al mismo mínimo con signos consistentes

**Fase 3 — Decisiones validadas (project-status):**
- Config óptima: 3 restarts + 100 maxiter + MPS chi=64 → ΔE/gap=1.58% (2/3 seeds)
- 5 restarts necesarios para 100% reliability (no testeado aún)
- Local minimum determinístico en ΔE/gap=0.437 existe → requiere ≥5 restarts
- Sign canonicalization: **NO necesaria** (confirmado en 3 runs, 0% efecto)

### Lo que FALTA (gap identificado)

El experimento C3 usó **interpolación lineal** para deployment (no MPNN real).
El pipeline completo con MPNN entrenado a N=20 p=1 **nunca se ejecutó con la
config corregida**. Los variant runners (`run_p1_pipeline_variants.py` y `_r2.py`)
solo cubren N=10.

**Tareas pendientes concretas:**
1. ~~Canonicalización Z₂~~ → **RESUELTO** (no necesaria con 3+ restarts)
2. ~~Inicialización analítica~~ → **NO NECESARIA** (warm-start resuelve)
3. **Ejecutar pipeline completo N=20 p=1 con MPNN** (no solo interpolación):
   - h_train: 15-20 puntos en [2.25, 4.0] (vs 6 del intento original)
   - VQE: 5 restarts, maxiter=100, MPS chi=64
   - MPNN: h=128, L=3, 6000 epochs
   - h_test: [2.5, 3.0, 3.5] (dentro del régimen válido)
4. **Crear variant runner** para N=20 p=1 (no existe)
5. **Validar con 3 seeds** que MPNN deployment pasa en todo h≥2.5

**Backend**: StatevectorEstimator (2^20 = 1M estados) o MPS chi=64 (~15 min/seed)

### Riesgo residual

El local minimum en ΔE/gap=0.437 es determinístico para seed 44. Con 5 restarts
debería escaparse, pero no está confirmado. Si persiste, la alternativa es usar
solo seeds 42/43 para training (ambos convergen perfectamente).

---

## 3. Ley de Escalado + Entropía de Entrelazamiento

**Hipótesis**: La ley h_min(N) tiene explicación analítica basada en entropía
de entrelazamiento del ground state.

**Datos existentes**:
- h_min = 1.0 + 0.020·N^1.33 (p=2, R²=1.0000)
- h_min = 1.0 + 0.212·N^0.60 (p=1)

**Pregunta**: ¿S(h_min, N) = constante para todo N? Si sí, el exponente β es
consecuencia de cómo S escala con N en el TFIM.

**Ejecución**:
- Calcular S(h, N) para N=4,6,8,10 (exact diag) y N=20 (DMRG)
- Evaluar S en h_min para cada N
- Si S_max(HVA p=k) es constante → explicación cerrada

**Valor**: Conecta resultado empírico con física fundamental (capítulo teórico)

---

## 4. Cross-Topology Transfer del MPNN

**Hipótesis**: Un MPNN entrenado en chain_1d puede predecir θ para
ladder/triangular sin re-entrenamiento.

- Ya hay datos de 5 topologías × múltiples seeds (131 variants)
- Nunca se probó train-on-one → deploy-on-another
- **Si funciona**: El GNN aprende la *física*, no la geometría
- **Si falla**: Cuantifica cuánto fine-tuning necesita cada topología
- Ejecución: re-entrenamiento MPNN + evaluación cruzada (minutos)

**Valor**: Valida/invalida el claim central de topology-agnosticism

---

## 5. Landscape Analysis a N=20 (extensión F3 + B4)

**Hipótesis**: El landscape a N=20 tiene estructura cualitativamente diferente
a N=6/N=10 (explicando por qué G3 falló).

**Contexto**:
- F3 (fluctuation): solo ejecutado a N=6
- B4 (Hessian): ejecutado a N=6 y N=10 (0 saddle points, κ N-independent)
- G3 mostró que "N=6 findings don't transfer to N=20"

**Ejecución**:
- F3 a N=20 con MPS: ¿fluctuation sigue >1.0? ¿fraction_near_gs?
- B4 a N=20: ¿aparecen saddle points? ¿κ crece?
- Comparar landscape p=1 vs p=2 a N=20

**Valor**: Explica *por qué* 1 restart falla a N=20 (G3 negative result)

---

## 6. MC-Dropout para UQ Calibrada (fix de G2)

**Hipótesis**: MC-Dropout (dropout activo en inferencia, 50 forward passes)
produce incertidumbre calibrada (r > 0.7 con ΔE/gap real).

**Contexto**: G2 mostró ensemble naive no funciona (r=0.195)

**Implementación**: ~50 líneas — activar `model.train()` en inferencia,
hacer 50 forward passes, calcular varianza.

**Valor**: UQ sin costo adicional de VQE. Si r>0.7, es publicable.

---

## 7. Data Efficiency a N=10 y N=20 (extensión G1)

**Hipótesis**: La curva k_min(N) escala predeciblemente.

- G1 a N=6: k_min=9 (47% reducción)
- ¿Cuántos puntos necesita N=10? ¿Y N=20?
- Directamente aplicable a reducir costo de VQE en hardware

---

## 8. Weight-Space Phase Detection a N=20 (extensión D1)

**Hipótesis**: Los gradientes del MPNN detectan h_c a N=20, más cerca del
valor termodinámico (h_c=1.0).

- D1 funciona a N=6 (peak h≈0.7) y N=10
- A N=20, finite-size effects son menores → peak debería acercarse a 1.0
- Requiere pipeline N=20 con MPNN entrenado

---

## 9. N=16 p=1 con MPS

**Hipótesis**: Pipeline completo funciona a N=16 p=1 con MPS (chi=64).

- project-status dice "N=16 p=1: Phase 3 does not complete" pero esto fue
  probablemente con régimen incorrecto
- Con h≥2.5 (estimado de la ley de escalado) podría funcionar
- Valida punto intermedio de la ley A3

---

## 10. Bootstrap UQ (alternativa a MC-Dropout)

**Hipótesis**: 5 MPNNs entrenados en subsets bootstrap producen varianza calibrada.

- Más costoso que MC-Dropout (5× entrenamiento)
- Más robusto teóricamente (captura incertidumbre epistémica real)
- Ejecutar solo si MC-Dropout (#6) falla

---

## Criterios de Ejecución

Antes de ejecutar cualquier experimento de esta lista:
1. **Hipótesis clara** — ¿qué aprenderemos que no sabemos?
2. **No duplicar** — verificar binnacles y poc-results.md
3. **Nuevo aprendizaje** — "confirmar lo conocido" no es aprendizaje después de 3 seeds
4. **Documentar** — resultado en binnacle con config, métricas, análisis

---

## Dependencias

```
#1 (Heisenberg) → independiente, spec listo
#2 (Fix N=20) → independiente, datos existentes
#3 (Entropía) → requiere exact diag sweep (existente) + cálculo S
#4 (Cross-topo) → requiere datos de #2 o existentes
#5 (Landscape N=20) → requiere MPS VQE a N=20 (existente)
#6 (MC-Dropout) → requiere MPNN entrenado (existente)
#7 (Data eff.) → independiente
#8 (D1 N=20) → requiere #2 completado
#9 (N=16 p=1) → independiente
#10 (Bootstrap) → ejecutar solo si #6 falla
```
