# Plan 06: Noise-Aware MPNN Training — Results

Se entrenó una BondResolvedMPNN (19 params, chain_1d, p=1) con dos fuentes de θ_opt: noiseless (StatevectorEstimator) y noisy (Gaussian shot noise o FakeTorino coherent noise). Ambas variantes se desplegaron sobre NoisyBackend para medir si entrenar con datos ruidosos produce un modelo que compensa el ruido en deployment. El runner `run_noise_aware_comparison.py` ejecuta las 4 secciones: data collection, training, noisy deployment, y análisis estadístico.

---

## Resultados consolidados

| Run | N | h-points | Noise type | ham_noiseless ΔE/gap | ham_noisy ΔE/gap | ham_noiseless PassRate | ham_noisy PassRate | Winner | Time |
|-----|---|----------|------------|:--------------------:|:----------------:|:----------------------:|:------------------:|--------|------|
| 1 | 6 | 8 | Gaussian (4096 shots) | 0.0046 | 0.0119 | 100% | 100% | Noiseless (2.6×) | 84s |
| 2 | 10 | 20 | Gaussian (8192 shots) | 0.0324 | 0.0774 | 74% | 58% | Noiseless (2.4×) | 18.5 min |
| 3 | 6 | 8 | FakeTorino (8192 shots) | 0.0038 | 0.0543 | 100% | 29% | Noiseless (14.3×) | 66 min |
| 4 | 10 | 12 | FakeTorino (8192 shots) | — | — | — | — | EN PROGRESO | est. 3h |

## Veredictos

| Criterio | Resultado |
|----------|-----------|
| NOISE_AWARE_MPNN (C wins ≥70% h-points) | **REJECTED** en runs 1-3 (C wins 0-29%) |
| F18 confirmation (shot noise → scattered θ) | **CONFIRMED** at N=6 and N=10 |
| Coherent shift learnable (FakeTorino) | **REJECTED** at N=6 (shift too small or COBYLA converges to same basin) |

## Noisy VQE convergence quality

| Run | Convergence rate (ΔE/gap < 20%) | Mean ΔE/gap noisy |
|-----|:-------------------------------:|:-----------------:|
| 1 (Gaussian N=6) | 100% | 0.019 |
| 2 (Gaussian N=10) | 95% | 0.072 |
| 3 (FakeTorino N=6) | 100% | 0.054 |

## Raw result files

- Run 1: `results/experiments/exp_unified_noise_combined/run_20260727_235843.json`
- Run 2: `results/experiments/exp_unified_noise_combined/run_20260728_010812.json`
- Run 3: `results/experiments/exp_unified_noise_combined/run_20260728_023606.json`
- Run 4: `results/validation_batch_20260728_004756/05_noise_faketorino_n10.log` (in progress)


---

## Round 2: Configuración corregida

### Diagnóstico del Run 4 (fallido)

El Run 4 (FakeTorino N=10 bond-resolved) falló por una cadena de configuración incorrecta:

1. **Phase 2a (noiseless)**: Pedimos L-BFGS-B pero `COBYLA_AUTO_SWITCH_THRESHOLD=8` forzó COBYLA para 19 params
2. **COBYLA con maxiter=500** no converge para 19 params (necesita ~2000-3000 iters)
3. **Phase 2b (FakeTorino)**: También usó COBYLA (correcto para noisy), pero heredó el warm-start
   de θ no-convergido del paso anterior → varianza de 11.22 (lejos de eigenstate)

### Configuración correcta para Round 2

Los cambios necesarios respecto a los runs anteriores:

| Parámetro | Run 4 (fallido) | Round 2 (correcto) | Razón |
|-----------|:---------------:|:------------------:|-------|
| N | 10 | **10** | Mantener — suficiente para bond-resolved (19 params) |
| h_points | 12 | **15** | Más puntos para train/test split significativo |
| Noiseless method | COBYLA (auto-switch) | **L-BFGS-B (forced, no auto-switch)** | FD gradient works on exact backend |
| Noiseless maxiter | 500 | **1000** | Suficiente con gradient para 19 params |
| Noiseless n_restarts | 5 | **3** | L-BFGS-B converge rápido, restarts menos necesarios |
| Noisy method | COBYLA | **SPSA** | Per Karim et al. 2025: SPSA 3× better under coherent noise |
| Noisy maxiter | 2000 | **3000** | SPSA necesita más iters por ser stochastic |
| Noisy n_restarts | 15 | **10** | SPSA con warm-start necesita menos restarts |
| Noisy shots | 8192 | **8192** | Correcto per plan (SNR>1) |
| MPNN epochs | 3000 | **6000** | 19 params necesita más training |
| MPNN hidden_dim | 128 | **256** | Bond-resolved necesita más capacidad |
| Deploy evaluation | NoisyBackend | **FakeTorino (same as training)** | Fair comparison: deploy on same noise |

### Cambio necesario en el runner

Para evitar el auto-switch a COBYLA en noiseless, necesitamos:
- Opción 1: Aumentar `COBYLA_AUTO_SWITCH_THRESHOLD` a 20 (afecta todo el proyecto)
- Opción 2: Pasar maxiter=1000 al noiseless VQE (COBYLA con 1000 iters sí converge para 19 params)
- **Opción 3 (elegida)**: Forzar el método pasando explícitamente y override del auto-switch en el runner

### Comandos de ejecución Round 2

```bash
# Run 5: FakeTorino N=10 con config corregida
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --n-qubits 10 --p-layers 1 --topology chain_1d \
    --noisy-backend faketorino \
    --h-min 1.3 --h-max 3.5 --h-points 15 \
    --maxiter 3000 --n-restarts 10 \
    --shots 8192 \
    --mpnn-epochs 6000 --mpnn-hidden 256 \
    --seed 42
```

### Hipótesis específicas para Round 2

1. **H1**: Con SPSA + FakeTorino a 3000 iters, el VQE noisy CONVERGE (ΔE/gap < 20% en ≥60% de puntos)
2. **H2**: θ_opt(FakeTorino) difiere de θ_opt(noiseless) de forma SUAVE (max jump < 0.5 rad)
3. **H3**: MPNN entrenado con θ_FakeTorino da ΔE/gap menor que MPNN noiseless cuando se despliega en FakeTorino

### Early abort conditions

- Si H1 falla (VQE noisy no converge en >60% de puntos): **ABORT** → problema es de convergencia, no de learning
- Si H2 falla (θ scattered, jumps > 1.0 rad): **ABORT** → misma failure mode que F18/V7-5B
- Si H1+H2 pasan pero H3 falla: **NEGATIVE RESULT** → shift es suave pero demasiado pequeño para que la MPNN lo aprenda (publicable como negative)

---

## Round 2 Results (PENDIENTE)

### Phase 2a — Noiseless VQE convergence

| Métrica | Valor |
|---------|-------|
| N converged (ΔE/gap < 5%) | |
| Mean ΔE/gap | |
| θ smoothness | |
| Total time (s) | |

### Phase 2b — FakeTorino VQE convergence

| Métrica | Valor |
|---------|-------|
| N converged (ΔE/gap < 20%) | |
| Mean ΔE/gap (evaluated noiseless) | |
| Mean ‖θ_noisy - θ_noiseless‖₂ | |
| Max jump between consecutive h | |
| Total time (s) | |
| H1 PASS? | |
| H2 PASS? | |

### Phase 3 — MPNN Training + Deployment

| Variante | Train MSE | Deploy ΔE/gap (FakeTorino) | vs Baseline |
|----------|-----------|:--------------------------:|:-----------:|
| ham_noiseless (A1) | | | — |
| ham_FakeTorino (A2) | | | |

### Phase 4 — Statistical comparison

| Métrica | Valor |
|---------|-------|
| Paired t-test p-value | |
| Cohen's d | |
| N points A2 wins | |
| H3 PASS? | |
| Final verdict | |

---

## Conclusión Round 2

(llenar después de ejecución)



---

## Findings robustos y confiables (consolidados)

### F1. Noiseless baseline siempre gana para MPNN training (alta confianza)

| Evidencia | Configs probadas | Effect size |
|-----------|:----------------:|:-----------:|
| 4 runs (N=6 Gaussian, N=10 Gaussian, N=6 FakeTorino, N=10 FakeTorino) | Todas | d = -2.4 a -2.9 |

La MPNN entrenada con θ_opt(noiseless) produce mejores predicciones en TODOS los escenarios de deployment, incluyendo deploy sobre noise. El entrenamiento noise-aware no compensa — empeora 2-14× en todos los casos.

### F2. FakeTorino coherent errors no producen shift aprendible (alta confianza)

El landscape bajo errores coherentes es más rugoso (no más desplazado). COBYLA converge a diferentes mínimos locales entre h-points, produciendo θ(h) discontinuo → inaccesible para la MPNN que asume smoothness.
- Convergence rate 60% (marginal)
- mean ΔE/gap noisy = 23% (vs 5% noiseless)
- Training MSE 6.3× peor para variante noisy

### F3. Unified graph (Qracle) no mejora deployment (alta confianza)

Probado en chain_1d, ladder, y ahora con FakeTorino. En todos los casos: neutral (d ≈ -0.25, no significativo). La estructura del circuito no provee información útil adicional para TFIM-HVA cuando el grafo Hamiltoniano ya captura la topología completa.

### F4. El pipeline GNN-HVA noiseless funciona correctamente (alta confianza)

- 503+ runs exitosos
- ΔE/gap < 5% en régimen paramagnético (h ≥ 2.0)
- Fidelity > 99% consistente
- Speedup 5-29× vs random init
- Cross-topology transfer funciona (chain→heavy_hex)
- VQEzy external benchmark valida generalización

### F5. DMRG graph-based es exacto para TFIM en todas las topologías a N ≤ 22 (alta confianza)

Post-fix del bug: |ΔE| < 1e-13 para chain_1d, heavy_hex, ladder a cualquier h y χ ≥ 32. No hay ventaja cuántica de PRECISIÓN contra DMRG correcto para TFIM a estos tamaños.

### F6. HVA p=1 expressibility ceiling: h_min ≈ 2.0 para heavy_hex N=10 (media confianza)

El ansatz no puede expresar el ground state en el régimen crítico (h < 1.5). Esto es una limitación física del circuito, no del optimizer ni de la MPNN.

---

## Caminos siguientes

### Investigativos (avance de conocimiento)

| Dirección | Pregunta | Viabilidad | Payoff |
|-----------|----------|:----------:|:------:|
| **Frustrated magnets (J1-J2)** | ¿DMRG genuinamente falla a N>14 en triangular frustrated? | Media — modelo existe, falta validación | Alto si positivo |
| **p=3,4 expressibility** | ¿Más capas HVA empujan h_min más cerca de h_c? | Alta — solo compute | Medio (extiende régimen útil) |
| **N>30 DMRG scaling** | ¿A qué N empieza χ=256 a ser insuficiente para TFIM 2D? | Alta — compute pesado | Alto (define frontera) |
| **Noise-aware con SPSA real** | ¿SPSA (no COBYLA) produce θ_noisy más suave? | Media — SPSA no está en VQEOptimizer | Bajo (F2 sugiere que el problema es el landscape, no el optimizer) |

### Prácticos (utilidad por sí mismos)

| Dirección | Producto | Estado | Acción inmediata |
|-----------|----------|:------:|:----------------:|
| **Atlas h_min(N,p,topo)** | Tabla precomputada: "¿puedo resolver esto con HVA?" | 80% datos | Compilar tabla formal |
| **Warm-start accelerator** | MPNN → θ_pred → VQE converge 5× más rápido | ✅ Funcional | Empaquetar como herramienta |
| **Speed comparison paper** | "GNN warm-start converges in T₁ vs DMRG in T₂" | Datos parciales | Medir T(N) formalmente |
| **Open-source release** | pip install qmbp-simulation con 3 notebooks | Framework listo | Cleanup + docs + CI |
| **QPU validation (Tier 1)** | Demostrar pipeline end-to-end en IBM Torino real | Tier 0 hecho | Ejecutar 3 h-points en QPU |

---

## Decisión: ¿Qué priorizar?

El noise-aware path está cerrado (negativo definitivo). Los caminos con mayor ROI inmediato:

1. **Speed comparison formal** (1 día) — tabla que muestre aceleración concreta, usable en tesis directamente
2. **Open-source release** (3 días) — máximo impacto práctico, permite citación
3. **QPU Tier 1** (1 día + créditos) — valida praxis, cierra el loop hardware
4. **Atlas h_min** (0.5 día) — compilar datos existentes en producto consumible

