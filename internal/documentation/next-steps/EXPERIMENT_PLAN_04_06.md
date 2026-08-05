# Plan Experimental: Integración #04 (Unified Graph) + #06 (Noise-Aware MPNN)

**Fecha inicio:** 2026-07-27
**Modelo:** TFIM, J=1.0
**Configuración base:** N=10, p=1, chain_1d, bond-resolved HVA (19 params)

---

## Camino A — Entrenar con J=1.0 (configuración estándar)

### Paso A.1 — Recolección de datos noiseless (baseline VQE)

**Qué hacemos:** VQE descending sweep con NoiselessBackend (StatevectorEstimator).
Produce θ_opt(noiseless) para 20 h-points en [1.3, 3.0].

**Comando:**
```bash
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --n-qubits 10 --p-layers 1 --topology chain_1d \
    --h-min 1.3 --h-max 3.0 --h-points 20 \
    --section 1 --skip-variants ham_noisy unified_noisy
```

**Métricas a registrar:**

| Métrica | Valor |
|---------|-------|
| N h-points | |
| N params (bond-resolved) | |
| Tiempo total VQE noiseless (s) | |
| Mean fidelity | |
| Mean ΔE/gap (noiseless) | |
| Max ΔE/gap (noiseless) | |
| θ smoothness (max‖Δθ‖∞) | |
| N puntos con ΔE/gap < 1% | |

---

### Paso A.2 — Recolección de datos noisy (COBYLA + shot noise)

**Qué hacemos:** VQE descending sweep con NoisyBackend(shots=8192).
COBYLA gradient-free optimizer, 15 restarts, maxiter=2000.
Produce θ_opt(noisy) para los mismos 20 h-points.

**Comando:** (ejecutado por Section 1 del runner)

**Métricas a registrar:**

| Métrica | Valor |
|---------|-------|
| Shots | |
| N restarts | |
| Maxiter | |
| Tiempo total VQE noisy (s) | |
| Noisy convergence rate (ΔE/gap < 20%) | |
| Mean ΔE/gap evaluando θ_noisy en noiseless backend | |
| Max ΔE/gap evaluando θ_noisy en noiseless backend | |
| Mean ‖θ_noisy - θ_noiseless‖₂ | |
| Std ‖θ_noisy - θ_noiseless‖₂ | |
| Correlación Pearson(θ_noisy, θ_noiseless) por parámetro | |

**Observaciones sobre convergencia:**
```
(anotar aquí si COBYLA converge bien o no, cuántos puntos fallan)
```

---

### Paso A.3 — Entrenamiento de las 4 variantes

**Qué hacemos:** Entrenar BondResolvedMPNN (hidden=256, layers=3) en las 4 combinaciones.
Mismo dataset de h-points, misma arquitectura, solo cambian: (1) tipo de grafo, (2) target θ.

| Variante | Grafo | θ target | node_features |
|----------|-------|----------|:---:|
| A | Hamiltonian-only | noiseless | 3 |
| B | Unified (Ham+Circuit) | noiseless | 4 |
| C | Hamiltonian-only | noisy | 3 |
| D | Unified (Ham+Circuit) | noisy | 4 |

**Comando:**
```bash
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --section 2
```

**Métricas por variante:**

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Final MSE | | | | |
| MSE @ epoch 500 | | | | |
| MSE @ epoch 2000 | | | | |
| N epochs hasta convergencia | | | | |
| Training time (s) | | | | |
| Per-h MSE max | | | | |
| Per-h MSE mean | | | | |

**Comparaciones entre variantes (training):**

| Comparación | Valor |
|-------------|-------|
| MSE improvement B vs A (%) | |
| MSE improvement C vs A (%) | |
| MSE improvement D vs A (%) | |
| ¿D < min(B, C)? (composición aditiva) | |

---

### Paso A.4 — Deploy en NoisyBackend (evaluación cuantitativa)

**Qué hacemos:** Predecir θ en puntos intermedios (midpoints entre training h)
y evaluar energía en NoisyBackend(shots=8192). Comparar ΔE/gap por variante.

**Puntos test:** ~19 midpoints entre los 20 training h-values.

**Métricas por variante (noisy deployment):**

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Mean ΔE/gap (noisy) | | | | |
| Median ΔE/gap (noisy) | | | | |
| Max ΔE/gap (noisy) | | | | |
| Std ΔE/gap (noisy) | | | | |
| N puntos ΔE/gap < 5% | | | | |
| N puntos ΔE/gap < 10% | | | | |
| Pass rate @ 5% | | | | |

**Métricas por variante (noiseless deployment — ceiling):**

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Mean ΔE/gap (noiseless) | | | | |
| Pass rate @ 5% (noiseless) | | | | |

**Degradación noise/noiseless por variante:**

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Ratio mean_noisy / mean_noiseless | | | | |
| ΔE/gap increase (noisy - noiseless) | | | | |

---

### Paso A.5 — Análisis estadístico pareado

**Qué hacemos:** Paired t-test sobre las 19 ΔE/gap values de cada par de variantes.
Computar Cohen's d (tamaño del efecto) e improvement rate (% puntos donde gana).

**Comparaciones pareadas:**

| Comparación | Mean diff | Std diff | t-stat | p-value | Cohen's d | % wins |
|-------------|-----------|----------|--------|---------|-----------|--------|
| C vs A (#06 effect) | | | | | | |
| B vs A (#04 effect) | | | | | | |
| D vs A (combined) | | | | | | |
| D vs B (noise benefit over unified) | | | | | | |
| D vs C (graph benefit over noise) | | | | | | |

**Interpretación de Cohen's d:**
- |d| < 0.2: efecto negligible
- 0.2 ≤ |d| < 0.5: efecto pequeño
- 0.5 ≤ |d| < 0.8: efecto mediano
- |d| ≥ 0.8: efecto grande

---

### Paso A.6 — Análisis per-h (¿dónde ayuda cada técnica?)

**Qué hacemos:** Graficar ΔE/gap(h) para las 4 variantes superpuestas.
Identificar en qué regiones de h cada técnica aporta.

**Preguntas cuantitativas:**

| Pregunta | Valor |
|----------|-------|
| ¿En qué rango de h el grafo unificado ayuda más? (B < A) | |
| ¿En qué rango de h el noise-aware ayuda más? (C < A) | |
| ¿Los beneficios se componen (D < min(B,C))? | |
| ¿Hay h-points donde la técnica empeora? | |
| Correlación gap vs mejora (¿ayuda más cerca de h_c?) | |

---

### Paso A.7 — Métricas de grafo (structural analysis)

**Qué hacemos:** Registrar metadata del grafo unificado para entender
si la complejidad estructural se justifica.

| Métrica | chain_1d N=10 p=1 |
|---------|:---:|
| Total nodes (unified) | |
| Total edges (unified) | |
| Node expansion ratio | |
| Edge expansion ratio | |
| Graph density | |
| Training time ratio (unified/ham-only) | |
| Overfitting risk (train_mse / val_mse) | |

---

## Resumen de datos cuantitativos que colectamos

No hay criterios pass/fail binarios. Colectamos:

1. **Distribuciones de ΔE/gap** por variante (19 puntos cada una)
2. **Curvas de entrenamiento** MSE(epoch) por variante
3. **Diferencias pareadas** ΔE/gap_A - ΔE/gap_X por punto
4. **Tamaños del efecto** Cohen's d para cada comparación
5. **Perfiles per-h** que muestran dónde actúa cada técnica
6. **Ratio noise/noiseless** que cuantifica resiliencia al ruido
7. **Métricas de grafo** que justifican la complejidad adicional

Estos datos se interpretan post-hoc, no con umbrales fijos.

---

## Camino B — Extensiones (depende de resultados del Camino A)

El Camino B se activa según lo que observemos en A. Consiste en:

### B.1 — Si el grafo unificado ayuda (Cohen's d ≥ 0.3 en B vs A):
- Repetir en **topologías no-simétricas** (square N=16, ladder N=10)
  donde los gate nodes no son equivalentes entre sí.
- Hipótesis: el beneficio será mayor porque la heterogeneidad
  del circuito aporta información no-redundante.
- Métricas: mismas que A.4/A.5 pero cross-topology.

### B.2 — Si el noise-aware ayuda (Cohen's d ≥ 0.3 en C vs A):
- Repetir con **multiple seeds** (5-10) para cuantificar varianza.
- Probar con **FakeTorino noise model** (AerSimulator) en lugar
  de Gaussian shot noise — más realista.
- Métricas: varianza inter-seed de θ_noisy, y si el MPNN aprende
  un "consensus" θ que es más robusto.

### B.3 — Si la combinación compone (D significativamente < min(B,C)):
- **Paper-ready ablation** con 3 topologías × 5 seeds × 4 variantes.
- Tabla de resultados para publicación.
- Comparar con ZNE: ¿noise-aware MPNN elimina la necesidad de ZNE?

### B.4 — Si nada ayuda (Cohen's d < 0.2 en todas las comparaciones):
- El hallazgo negativo también es publicable: "Para TFIM 1D con HVA
  bond-resolved, el paisaje energético es suficientemente suave para
  que la representación Hamiltonian-only sea óptima."
- Investigar: ¿el grafo unificado ayuda en **modelos con más estructura**
  (Heisenberg, Kitaev) donde el circuito no es uniforme?

---

**Nota:** Este documento se actualiza in-situ con los resultados de cada paso.
Los JSONs crudos se guardan en `results/experiments/exp_unified_noise_combined/`.


---

## Extracted Results (auto-generated)

**Source:** `exp_unified_noise_combined`

### Paso A.1 + A.2 — Data Collection

| Métrica | Valor |
|---------|-------|
| N h-points | 20 |
| N params | 19 |
| Noiseless VQE time (s) | 184.5 |
| Noisy VQE time (s) | 42.2 |
| Noisy convergence rate | 85% |
| Noisy mean ΔE/gap | 0.1034 |
| Noisy max ΔE/gap | 0.3843 |

### Paso A.3 — Training Metrics

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Final MSE | 2.21e-04 | 8.51e-05 | 4.72e-03 | 6.03e-04 |
| Training time (s) | 186.1 | 149.3 | 170.9 | 245.6 |

**MSE improvement B vs A:** 61.4%

**Graph metrics:** 29 nodes, 2.9× expansion, density=0.1133

### Paso A.4 — Noisy Deployment

| Métrica | A | B | C | D |
|---------|---|---|---|---|
| Mean ΔE/gap (noisy) | 0.0604 | 0.0634 | 0.0862 | 0.0924 |
| Max ΔE/gap (noisy) | 0.2507 | 0.2664 | 0.3210 | 0.3340 |
| Pass rate @ 5% | 63% | 58% | 47% | 37% |
| Mean ΔE/gap (noiseless) | 0.0619 | 0.0618 | 0.0862 | 0.0919 |
| Pass rate (noiseless) | 63% | 63% | 42% | 37% |


---

## Resultados del Camino A — Ejecución 2026-07-27

**Config:** N=10, p=1, chain_1d, bond-resolved HVA (19 params), J=1.0, TFIM
**Run:** `results/experiments/exp_unified_noise_combined/run_20260727_215957.json`
**Tiempo total:** 1052s (~17.5 min)

### A.1 + A.2 — Data Collection

| Métrica | Valor |
|---------|-------|
| N h-points | 20 |
| N params (bond-resolved) | 19 |
| Noiseless VQE time | 184.5s |
| Noisy VQE time (COBYLA, 15 restarts, 8192 shots) | 42.2s |
| Noisy convergence rate (ΔE/gap < 20%) | 85% |
| Noisy mean ΔE/gap | 0.1034 |
| Noisy max ΔE/gap | 0.3843 |

### A.3 — Training Metrics (3000 epochs, val_fraction=0.2)

| Métrica | A (ham+noiseless) | B (unified+noiseless) | C (ham+noisy) | D (unified+noisy) |
|---------|:-:|:-:|:-:|:-:|
| Final MSE | 2.21e-04 | **8.51e-05** | 4.72e-03 | 6.03e-04 |
| Training time (s) | 186 | 149 | 171 | 246 |

**MSE improvement B vs A:** 61.4% (unified graph learns θ better)
**Graph metrics:** 29 nodes (2.9× expansion), 92 edges, density=0.1133

### A.4 — Noisy Deployment (NoisyBackend, 19 midpoint test h)

| Métrica | A | B | C | D |
|---------|:-:|:-:|:-:|:-:|
| Mean ΔE/gap (noisy) | **0.0604** | 0.0634 | 0.0862 | 0.0924 |
| Max ΔE/gap (noisy) | 0.2507 | 0.2664 | 0.3210 | 0.3340 |
| Pass rate @ 5% | **63%** | 58% | 47% | 37% |
| Mean ΔE/gap (noiseless) | 0.0619 | **0.0618** | 0.0862 | 0.0919 |
| Pass rate (noiseless) | 63% | 63% | 42% | 37% |

### A.5 — Análisis Estadístico Pareado (N=19 test points)

| Comparación | Mean diff | Cohen's d | p-value | After wins |
|-------------|:---------:|:---------:|:-------:|:----------:|
| C vs A (#06: noise-aware) | -0.0258 (PEOR) | **-1.47** | 1.00 | 0/19 (0%) |
| B vs A (#04: unified graph) | -0.0030 (neutro) | -0.30 | 0.90 | 8/19 (42%) |
| D vs A (combined) | -0.0320 (PEOR) | **-1.32** | 1.00 | 0/19 (0%) |
| D vs B (noise over unified) | -0.0290 (PEOR) | **-1.56** | 1.00 | 0/19 (0%) |

### Interpretación

1. **#06 (Noise-Aware Training) es contraproducente** en este setup.
   - Cohen's d = -1.47: efecto grande en la dirección equivocada.
   - Los targets θ_noisy (COBYLA + Gaussian shot noise) son inherentemente peores
     que θ_noiseless (L-BFGS-B exacto). El MPNN aprende targets ruidosos y
     propaga ese error al deploy.
   - C pierde en 19/19 test points vs A. No hay ni un h-point donde ayude.

2. **#04 (Unified Graph) es neutro en deployment** a pesar de 61% mejor MSE en training.
   - El model fit mejora dramáticamente (8.5e-5 vs 2.2e-4) pero la mejora NO
     se traduce en mejor energía en deploy (ΔE/gap 0.063 vs 0.060).
   - Hipótesis: para chain_1d, la estructura del circuito es uniforme (todos los
     ZZ gates son equivalentes por simetría traslacional) → los gate nodes no
     aportan información nueva que el Hamiltonian graph no tenga.
   - El grafo 3× más grande permite memorizar mejor sin generalizar mejor.

3. **La combinación D es la peor** porque hereda el problema de targets noisy
   con un modelo más complejo que los sobreajusta.

4. **Baseline A (Hamiltonian-only + noiseless θ) es óptimo** para TFIM 1D chain.

### Diagnóstico: ¿Por qué noise-aware no funciona aquí?

El paper de Karim et al. (2025) entrena con ruido REAL de hardware (T1/T2 decoherencia,
crosstalk, readout errors). Nosotros usamos **Gaussian shot noise** (approximación):
```
E_noisy = E_exact + N(0, 1/√shots)
```
Esto NO cambia el landscape — solo agrega ruido i.i.d. al cost function. COBYLA
converge a un mínimo desplazado aleatoriamente, no a un mínimo adaptado al ruido.

Para que #06 funcione necesitaríamos:
- `NoisyBackend(noise_model=FakeTorino)` con AerSimulator (ruido correlacionado real)
- O directamente datos de hardware (IBM Runtime)

### Decisión sobre Camino B

| Rama | ¿Proceder? | Razón |
|------|:---:|---|
| B.1 (unified en square/ladder) | **Sí, vale la pena** | En topologías no-simétricas los gate nodes NO son equivalentes. El resultado neutro en chain_1d no descarta beneficio en 2D. |
| B.2 (noise-aware con FakeTorino) | **Sí, pero como extensión futura** | Requiere qiskit-aer + FakeTorino setup. El Gaussian approx no captura el fenómeno relevante. |
| B.3 (paper-ready ablation) | **No** | Sin mejora demostrada, no hay ablation publicable. |
| B.4 (hallazgo negativo) | **Sí — documentar** | "Para TFIM 1D con HVA bond-resolved, noise-aware training con shot noise es contraproducente" es un finding válido que ahorra tiempo a otros. |

**Próximo paso concreto:** Ejecutar B.1 (unified graph en square N=16) donde
los gate nodes tienen heterogeneidad real (corner vs bulk qubits tienen
distinto número de ZZ gates conectados).
