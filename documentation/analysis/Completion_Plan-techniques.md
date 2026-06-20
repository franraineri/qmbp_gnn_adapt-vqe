---

## 10. Completion Plan — Partially Implemented Techniques

> **Fecha**: 2026-06-19
> **Scope**: Plan para completar las 2 técnicas parcialmente implementadas (#3 y #7).
> **Prioridad**: Post-Kingston deployment. Ninguna bloquea el hardware run actual.

---

### 10.1 Técnica #3: Adaptive Mitigation (GSC-QEMit) — de 60% a 100%

#### Qué ya existe (60%)

| Componente | Estado | Ubicación |
|-----------|:------:|-----------|
| `take_calibration_snapshot()` | ✅ Done | `noisy_utils.py` L2512 |
| `check_calibration_drift()` | ✅ Done | `noisy_utils.py` L2582 |
| `CalibrationSnapshot` dataclass | ✅ Done | T1, T2, gate_errors_2q, readout_errors |
| `DriftReport` dataclass | ✅ Done | t1_drift_pct, recommendation (proceed/abort) |
| `run_adaptive_zne()` PEA→GF fallback | ✅ Done | R² threshold triggers automatic switch |
| Abort on T1 drift >20% | ✅ Done | In `run_ibm_deployment.py` + `run_mitigation_benchmark.py` |

#### Qué falta (40%)

| Componente | Impacto | Esfuerzo |
|-----------|:-------:|:--------:|
| **F3a: Predictive drift model** | MEDIO | 2 días |
| **F3b: Multi-armed bandit strategy selector** | BAJO | 3 días |
| **F3c: Mid-run strategy switching** | MEDIO | 2 días |

#### Plan detallado

##### F3a: Predictive Drift Model (2 días)

**Objetivo**: Predecir T1/T2 drift 15-30 min en el futuro basándose en el historial
de snapshots, para decidir pro-activamente si hay que cambiar de estrategia antes
de que drift cruce el umbral.

**Implementación**:
```
src/qmbp_simulation/execution/
└── drift_forecaster.py   ← NUEVO
```

**Tareas**:
1. Crear `DriftForecaster` class que acumula `CalibrationSnapshot` history
2. Implementar linear extrapolation de T1(t) y gate_error(t) con ventana deslizante
3. Método `forecast_drift(horizon_minutes=30) → DriftForecast` con:
   - `predicted_t1_drift_pct`: drift esperado en el horizonte
   - `time_to_threshold_min`: minutos hasta cruzar 20%
   - `recommendation`: "safe" | "hurry" | "abort_soon"
4. Integrar en `HardwareBackend.run_deployment()`: tomar snapshot cada N h-points,
   llamar `forecast_drift()`, log warning si `time_to_threshold < run_time_remaining`
5. Test: simular T1 decay lineal, verificar que forecast predice correctamente

**Complejidad real**: El modelo es trivial (regresión lineal sobre 3-5 snapshots).
El valor está en la decisión automatizada, no en la sofisticación del forecast.

**Prerrequisitos**: Hardware run data (múltiples snapshots en una sesión). Puede
desarrollarse con datos sintéticos y validarse en primera ejecución en Kingston.

##### F3b: Multi-Armed Bandit Strategy Selector (3 días)

**Objetivo**: Dada una serie de h-points a ejecutar, seleccionar automáticamente
la estrategia de mitigación (PEA/GF/PNA/raw) para cada punto basándose en el
reward observado en puntos anteriores.

**Implementación**:
```
src/qmbp_simulation/execution/
└── strategy_bandit.py   ← NUEVO
```

**Tareas**:
1. Definir `MitigationArm` enum: PEA_BALANCED, PEA_HEAVY, GF_ZNE, RAW
2. Implementar `StrategyBandit` (Upper Confidence Bound):
   - `select_arm(h_value, kappa) → MitigationArm`
   - `update(arm, reward)` donde reward = 1 − ΔE/gap (normalizado)
   - Context: κ (curvature) + h_regime + circuit_depth
3. Warm-start con priors del Mitigation Benchmark V2:
   - PEA_BALANCED: μ=0.996 (0.37% ΔE/gap en sim)
   - GF_ZNE: μ=0.71 (28.7% ΔE/gap en sim)
   - RAW: μ=0.58 (41.9% ΔE/gap en sim)
4. Integrar como opción en deployment: `--strategy adaptive_bandit`
5. Test: mock rewards, verificar que bandit converge a PEA en <5 pulls

**Complejidad real**: UCB es ~50 líneas. El valor está en los priors calibrados
(que ya tenemos del benchmark) y la integración con el deployment loop.

**Cuándo es útil**: Solo si hay budget QPU para >20 h-points en una sesión y
se quiere explorar si alguna estrategia alternativa es mejor en hardware real.
Para el deployment actual (7 configs × 4 h-points), es overkill.

##### F3c: Mid-Run Strategy Switching (2 días)

**Objetivo**: Si durante una ejecución multi-h el drift sube o el R² de ZNE baja,
cambiar automáticamente de PEA → GF (o viceversa) sin abortar el run.

**Implementación**: Modificar `HardwareBackend.run_deployment()` loop.

**Tareas**:
1. Después de cada h-point completado:
   - Evaluar `DriftForecaster.forecast_drift()`
   - Si drift predicho >15%: log warning + switch a GF (más robusto a drift)
   - Si drift predicho >20%: abort (ya implementado)
2. Después de cada ZNE fit:
   - Si R² < 0.85 y strategy=PEA: switch a GF para puntos restantes
   - Si R² > 0.98 y strategy=GF: no cambiar (GF es suficiente)
3. Registrar `strategy_switches` en `HardwareRunResult` metadata
4. Test: mock R² sequence [0.99, 0.99, 0.82, 0.78], verificar switch en punto 3

**Complejidad real**: `run_adaptive_zne()` ya hace fallback PEA→GF per-point.
Esto lo extiende a nivel de sesión (persistir la decisión para puntos futuros).

#### Timeline total: ~7 días (post-Kingston)

```
F3a (drift forecaster):       2 días → DriftForecaster class + tests
F3b (bandit selector):         3 días → StrategyBandit + priors + integration
F3c (mid-run switching):       2 días → Loop modification + metadata
```

**Cuándo ejecutar**: Solo después de tener datos reales de drift en Kingston.
Sin datos de drift real, no hay forma de validar si el forecaster agrega valor
(FakeTorino no tiene drift — el noise model es estático).

---

### 10.2 Técnica #7 (FW-C): Entanglement Entropy como Predictor de Viabilidad — de 70% a 100%

#### Qué ya existe (70%)

| Componente | Estado | Ubicación |
|-----------|:------:|-----------|
| `compute_entanglement_entropy()` | ✅ Done | `experiments/scaling/exp_s1_entanglement_scaling.py` |
| Schmidt decomposition via SVD | ✅ Done | Correcto, handles zeros |
| S(h) sweep para N=4,6,8,10 | ✅ Done | `ExperimentS1.run_single()` |
| S(h_min) at boundary p=2 | ✅ Done | `H_MIN_P2` dict con boundaries conocidos |
| S(h_min) at boundary p=1 | ✅ Done | `H_MIN_P1` dict (N=6,10) |
| Hypothesis: S(h_min)≈const across N | ✅ Coded | Prints confirmed/rejected |
| Experiment registered (S1) | ✅ Done | Runnable via `python scripts/run_experiment.py --exp S1` |

#### Qué falta (30%)

| Componente | Impacto | Esfuerzo |
|-----------|:-------:|:--------:|
| **FC-1: Ejecutar S1 y capturar resultado JSON** | ALTO | 30 min |
| **FC-2: Correlación S(h_min) vs fidelity VQE** | ALTO | 2h |
| **FC-3: Cross-model validation (Heisenberg, frustrated)** | ALTO | 3h |
| **FC-4: Regla predictiva formal + thesis figure** | MEDIO | 2h |

#### Plan detallado

##### FC-1: Ejecutar Experimento S1 (30 min)

```bash
python scripts/run_experiment.py --exp S1 --verbose
```

Esto ya funciona. Producirá `results/experiments/exp_S1/run_<timestamp>.json` con:
- S(h) curves para N=4,6,8,10
- S(h_min_p2) para cada N
- Análisis de constancia (mean ± std)

**Resultado esperado** (del steering/project-status): S(h_min) ≈ 0.5-0.8 bits,
std < 0.1 across N → hypothesis CONFIRMED.

##### FC-2: Correlación S vs Fidelity VQE (2h)

**Objetivo**: Para cada (modelo, N, h), correlacionar la entropía de entanglement
del ground state exacto con la fidelidad VQE alcanzada por HVA p≤2.

**Implementación**: Script de análisis (no nuevo módulo — usa datos existentes).

```
scripts/analysis/
└── entanglement_viability_correlation.py   ← NUEVO
```

**Tareas**:
1. Cargar resultados VQE existentes (V7/V8/V9 JSON en `results/`)
2. Para cada punto (N, h, topology): extraer fidelidad final del VQE
3. Calcular S(L/2) del ground state exacto (reusar `compute_entanglement_entropy`)
4. Scatter plot: S(L/2) vs fidelity, color-coded por modelo
5. Fit: fidelity = f(S) — esperamos sigmoid decreciente
6. Determinar umbral: S_max tal que fid > 0.93 (nuestro threshold TFIM)
7. Output: `results/analysis/entanglement_viability.json` + PNG figure

**Datos ya disponibles** (no requiere ejecución nueva):
- TFIM: V7/V8 binnacles tienen fidelidades para N=6,10 en múltiples h
- Heisenberg: V9 tiene fid_max=48% a p=6 (datos de 30 runs)
- Los ground states exactos se recalculan en <1s (N≤10)

##### FC-3: Cross-Model Validation (3h)

**Objetivo**: Extender la correlación a todos los modelos del registry para
producir una regla universal.

**Tareas**:
1. Calcular S(L/2) para ground states de:
   - `tfim` (h ∈ [0.5, 4.0], N=6)
   - `tfim_longitudinal` (g=0.5, h ∈ [1, 4], N=6)
   - `tfim_frustrated` (J₂=0.5, h ∈ [1, 4], N=6)
   - `heisenberg` (h=3, N=6) — esperamos S>>1
2. Para cada uno, obtener fidelidad VQE de datos existentes (E4b, T1c, V9)
3. Agregar al scatter plot de FC-2
4. Verificar que el umbral S_max es consistente entre modelos

**Resultado esperado** (del status tracker):
- TFIM (h=2): S≈0.5, fid≥0.99 ✅
- TFIM+Long (g=0.3): S≈0.5, fid≥0.98 ✅
- Heisenberg (h=3): S≈2.2, fid=0.48 ❌
- El umbral S_max ≈ 1.0 bit debería separar perfectamente viable/no-viable

##### FC-4: Regla Predictiva + Thesis Figure (2h)

**Objetivo**: Formalizar la regla y producir una figura publicable.

**Entregables**:
1. Regla: "HVA p≤2 es viable si y solo si S(L/2) del ground state ≤ S_max ≈ 1.0 bit"
2. Thesis figure (PDF vectorial): scatter con threshold line + labels de modelos
3. Agregar a `documentation/analysis/09_thesis_tables.md` como Table 5.X
4. JSON con datos: `results/analysis/entanglement_viability_rule.json`

**Comando final**:
```bash
python scripts/analysis/entanglement_viability_correlation.py --thesis-figure
make figures-thesis  # regenera todos los PDFs
```

#### Timeline total: ~1 día (8h)

```
FC-1 (ejecutar S1):              30 min → JSON con S(h) curves
FC-2 (correlación S vs fid):     2h → scatter + fit + threshold
FC-3 (cross-model):              3h → universal rule validation
FC-4 (regla + figure):           2h → thesis-ready output
```

**Cuándo ejecutar**: Inmediatamente — no requiere hardware, no tiene dependencias,
y los datos ya existen. Alto valor para la tesis (transforma una observación
empírica en un criterio predictivo con sustento físico). Puede hacerse en
paralelo con cualquier otra actividad.

---

### 10.3 Priorización

| Técnica | ROI tesis | Dependencia hardware | Prioridad |
|---------|:---------:|:--------------------:|:---------:|
| **FW-C Entanglement Predictor** | ★★★★★ | Ninguna (datos existentes) | **AHORA** |
| F3a Drift Forecaster | ★★☆☆☆ | Datos de drift real (Kingston) | Post-hardware |
| F3c Mid-run Switching | ★★☆☆☆ | Idem | Post-hardware |
| F3b Bandit Selector | ★☆☆☆☆ | Idem + >20 h-points | Último |

**Recomendación**: Ejecutar FW-C completo esta semana (1 día, zero risk). Los
componentes F3* solo tienen sentido después de la primera sesión en Kingston —
sin datos de drift real, son ejercicios teóricos.
