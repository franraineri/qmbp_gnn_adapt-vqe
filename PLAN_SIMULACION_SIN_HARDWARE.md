# Plan de Ejecución: Simulación Sin Acceso a Hardware Real

> Fecha: 2026-06-01
> Estado: Revisado y validado contra resultados existentes
> Prerequisito: V8 completo, 131+ variants, 60+ experimentos benchmark ejecutados

---

## Verificación de No-Duplicación

Antes de proponer cada experimento, se verificó contra:
- `results/experiments/` (15 directorios de resultados: A3, B1, B2, B4, C1, C3, D1, E3, E4, F1, F3, G1-G5)
- `documentation/binnacles/` (12 binnacles con resultados definitivos)
- `results/thesis/` (131+ variants ejecutados)
- `documentation/bibliography/bibliography_curated.md` (44 papers, 20 secciones)
- `documentation/bibliography/alternative_bibliography.md` (15+ papers alternativos)
- `documentation/bibliography/bibliography.md` (bibliografía completa, 28 secciones)
- `.kiro/knowledge/literature-synthesis.md` (síntesis de 50+ papers)
- Búsqueda textual en todo el repositorio para cada tema propuesto
- Búsqueda web (arXiv, 2024-2026) para cada hipótesis propuesta

**Resultado**: Ninguno de los experimentos propuestos tiene resultados previos en el proyecto.
Se identificaron papers relacionados en la literatura (ver sección "Prior Art" por experimento).

---

## Criterios de Inclusión (Rigor Científico)

Cada experimento incluido cumple:
1. **Hipótesis falsificable** — resultado positivo O negativo produce aprendizaje
2. **No duplica** — verificado que no existe en binnacles ni results/
3. **Infraestructura lista** — código base existe o requiere <100 líneas nuevas
4. **Reproducible** — 3 seeds mínimo, configuración completa documentada
5. **Tiempo acotado** — ejecutable en hardware local (MacOS, StatevectorEstimator o MPS chi=64)

---

## Experimentos Aprobados (Tier 1 — Alto Valor)

### EXP-S1: Heisenberg XXZ Regime Discovery + Comparative Analysis

**Hipótesis**: HVA p=2 tiene un régimen válido (fidelity ≥ 0.93) en el límite
paramagnético (h >> J) del modelo Heisenberg XXZ, o bien la entropía de
entrelazamiento del ground state excede la capacidad del ansatz en todo h.

**Justificación científica**:
- E4 mostró fidelity=0.89 con g=0.1 (campo longitudinal), pero NO probó Heisenberg puro
- Dato previo: "max 22% fidelity" mencionado en specs, pero sin sweep sistemático
- La entropía de entrelazamiento proporciona explicación cuantitativa del límite

**Estado del código**:
- `experiments/generalization/exp_regime_discovery.py` — COMPLETO, nunca ejecutado
- `experiments/generalization/exp_comparative_analysis.py` — COMPLETO, nunca ejecutado
- `src/qmbp_simulation/analysis/entanglement.py` — EntanglementAnalyzer COMPLETO
- `src/qmbp_simulation/analysis/comparative.py` — RegimeDiscoveryResult, ComparativeMetrics COMPLETOS
- `src/qmbp_simulation/models/model_spec.py` — ModelSpec + ModelRegistry COMPLETOS

**Verificación de no-duplicación**:
- `results/experiments/` — NO existe exp_rd1/ ni exp_ca1/
- Búsqueda "regime_discovery|RD1|CA1" en results/ — 0 matches
- E4 probó campo longitudinal (g>0), NO Heisenberg (XX+YY+ZZ)

**Configuración exacta**:
```
Experimento RD1:
  N=6, p=2, chain_1d, J=1.0
  Δ ∈ {0.0 (XY), 0.5, 1.0 (isotropic)}
  h sweep: 4.0 → 0.0, step 0.25 (17 puntos)
  VQE: 10 restarts, σ=0.5, maxiter=1500, L-BFGS-B
  Seeds: [42, 43, 44]
  Backend: StatevectorEstimator (noiseless)
  Métricas: fidelity, ΔE/gap, entanglement entropy S(h)
  Thresholds evaluados: [0.93, 0.80, 0.70, 0.60]

Experimento CA1:
  Misma config + baseline TFIM (5 restarts, σ=0.1, maxiter=1000)
  Comparación: CX budget, valid regime width, avg fidelity, S_max
```

**Resultado esperado**: Negativo riguroso (max fidelity < 0.60 para Δ=1.0).
Cuantifica S_max(HVA p=2) vs S(ground state Heisenberg) → explica el gap.

**Ejecución**:
```bash
python scripts/run_experiment.py --exp RD1 --verbose
python scripts/run_experiment.py --exp CA1 --verbose
```

**Tiempo estimado**: ~15 min (RD1: 3 deltas × 3 seeds × 17 h-points × 10 restarts)

**Riesgos**:
- Si fidelity > 0.93 en algún punto → resultado inesperado (positivo parcial)
- Mitigación: documentar como "viable regime" y proceder con Phase 3

**Prior Art (literatura)**:
- Maiti (arXiv:2604.11688, Apr 2026): "Frustration-Induced Expressibility Limitations in VQAs" —
  estudia limitaciones de expresividad en sistemas frustrados, pero usa HEA y modelos frustrados
  (J₁-J₂), no HVA + Heisenberg XXZ con sweep de anisotropía Δ.
- Kochkov et al. (arXiv:2110.06390, 2021): GNN para ground states de Heisenberg, pero como
  variational manifold (NQS), no como predictor de parámetros VQE.
- arXiv:2512.23009: VQE Heisenberg en IQM hardware con symmetry-preserving ansatz, pero
  no usa HVA ni analiza entropía como explicación del límite.
- **Diferenciación**: Nuestro experimento es el primero en (a) usar HVA p≤2 específicamente
  para Heisenberg XXZ, (b) cuantificar el límite via S_max(HVA) vs S(ground state), y
  (c) evaluar múltiples valores de Δ con el mismo framework.

---

### EXP-S2: Entropía de Entrelazamiento y Ley de Escalado

**Hipótesis**: Existe un valor constante S* tal que S(h_min, N) ≈ S* para todo N,
lo que explicaría el exponente β=1.33 de la ley h_min(N) como consecuencia de
cómo la entropía escala con N en el TFIM.

**Justificación científica**:
- A3 estableció h_min = 1.0 + 0.020·N^1.33 (R²=1.0000) empíricamente
- No existe explicación teórica del exponente β=1.33
- Si S(h_min) = constante → β es derivable de la teoría conforme del TFIM
- Conecta resultado numérico con física fundamental (capítulo teórico de tesis)

**Estado del código**:
- `EntanglementAnalyzer.compute_half_chain_entropy()` — implementado, nunca usado
- `ClassicalSolver` — exact diag para N≤10, DMRG para N=20
- Datos de h_min ya conocidos: N=4→0.95, N=6→1.20, N=8→1.30, N=10→1.40, N=20→2.00

**Verificación de no-duplicación**:
- Búsqueda "entanglement.*entropy.*result|S(h_min)" — 0 matches
- EntanglementAnalyzer nunca invocado en ningún experimento ejecutado
- No existe script de análisis de entropía en analysis/ ni experiments/

**Configuración exacta**:
```
Fase 1 — Sweep de entropía:
  N ∈ {4, 6, 8, 10}: exact diag (ClassicalSolver)
  N = 20: DMRG (chi=64, TeNPy)
  h sweep: 0.5 → 4.0, step 0.1 (36 puntos por N)
  Métrica: S = -Tr(ρ_A log₂ ρ_A), partición A = N/2 primeros qubits
  Backend: StatevectorEstimator (N≤10), MPS (N=20)

Fase 2 — Evaluación en h_min:
  Extraer S(h_min, N) para cada N
  Test: ¿S(h_min) ≈ constante? (varianza < 10%)
  Si constante → derivar β analíticamente de S(h,N) del TFIM

Fase 3 — Capacidad HVA:
  Para cada N, encontrar max S donde fidelity ≥ 0.93
  Comparar con S(h_min) → ¿coinciden?
```

**Resultado esperado**: S(h_min) ≈ 0.5-0.8 bits (constante ± 15%).
Si se confirma, el exponente β=1.33 se explica por S(h,N) ~ (h_c/h)^α · log(N).

**Ejecución**: Script nuevo (~80 líneas), usa infraestructura existente.

**Tiempo estimado**: ~5 min (exact diag rápido para N≤10, DMRG ~2 min para N=20)

**Riesgos**:
- S(h_min) podría NO ser constante → resultado negativo (β no tiene explicación simple)
- Mitigación: reportar como "empirical exponent without analytical derivation"
- DMRG a N=20 podría dar gap=0 → usar gap analítico (ya implementado como fallback)

**Prior Art (literatura)**:
- Tripathi et al. (arXiv:2604.20961, 2026): Estudia entanglement entropy vs expressibility
  para TFIM con HVA hasta 27 spins. Confirma que HVA p=2 falla en la región crítica por
  entanglement. Pero NO deriva una ley de escalado h_min(N) ni conecta S con el exponente β.
- arXiv:2602.17662 (Feb 2026): "Entanglement and Ansatz Expressivity for TFIM using VQE" —
  estudia la relación entanglement-expressibility pero para un solo N, no como scaling law.
- arXiv:2501.17533: "Entanglement-informed Construction of Variational Quantum Circuits" —
  usa entanglement para diseñar ansätze, no para explicar leyes de escalado empíricas.
- **Diferenciación**: Nuestra pregunta específica "¿S(h_min, N) = constante?" es original.
  Si se confirma, proporciona una derivación analítica del exponente β=1.33 desde la
  teoría conforme del TFIM — esto no existe en la literatura.

---

### EXP-S3: Cross-Topology Transfer del MPNN

**Hipótesis**: Un MPNN entrenado exclusivamente con datos de chain_1d puede
predecir θ_opt para ladder y triangular con ΔE/gap < 10% (transfer parcial)
o < 5% (transfer completo), sin re-entrenamiento.

**Justificación científica**:
- El GNN (GINConv) procesa la estructura del grafo → debería aprender la *física*
- 131 variants ejecutados en 5 topologías, pero NUNCA se probó train→deploy cruzado
- Si funciona: valida el claim central de "topology-agnostic" del framework
- Si falla: cuantifica cuánto fine-tuning necesita cada topología (también publicable)

**Estado del código**:
- `MPNNPredictor` — soporta cualquier topología via graph_dataset
- `build_graph_dataset()` — construye Data objects con edge_index por topología
- Datos VQE existentes en `results/thesis/` para chain, ladder, triangular (N=6, N=10)
- NO existe script de evaluación cruzada

**Verificación de no-duplicación**:
- Búsqueda "cross.topology.*transfer|train.*on.*one.*deploy" — 0 matches
- Búsqueda "topology.*agnostic.*test" — 0 matches
- Ningún binnacle documenta este experimento

**Configuración exacta**:
```
Protocolo (N=10, p=2):
  1. Cargar datos VQE de chain_1d (results/thesis/variants_N10_multi/)
  2. Entrenar MPNN: h=128, L=3, 6000 epochs, lr=1e-3, seed=42
  3. Evaluar en:
     - chain_1d (in-distribution, baseline)
     - ladder (out-of-distribution)
     - triangular (out-of-distribution)
     - heavy_hex (out-of-distribution)
  4. Métrica: ΔE/gap en h_test para cada topología
  5. Repetir con seeds [42, 43, 44]

Variante inversa:
  - Entrenar en ladder → evaluar en chain_1d
  - Entrenar en triangular → evaluar en chain_1d
  (Para verificar simetría del transfer)
```

**Resultado esperado**: Transfer parcial (ΔE/gap 5-15%) para topologías similares
(chain→ladder), fallo para topologías muy diferentes (chain→triangular).

**Ejecución**: Script nuevo (~120 líneas), usa datos existentes + MPNN training.

**Tiempo estimado**: ~10 min (6000 epochs × 4 evaluaciones × 3 seeds)

**Riesgos**:
- Los datos de diferentes topologías tienen h_values distintos → normalizar
- El MPNN podría memorizar la topología específica → resultado negativo claro
- Mitigación: si falla completamente, probar fine-tuning con 3 puntos (few-shot)

**Prior Art (literatura)**:
- arXiv:1908.09883 (2019): Transfer learning para NQS (RBM) entre tamaños de sistema,
  pero para neural-network quantum states, no para GNN predictores de parámetros VQE.
- Bincoletto et al. (arXiv:2511.03726, 2025): "Transferable ML for circuit parameters" —
  valida transferibilidad entre tamaños de sistema, pero no entre topologías.
- Lee et al. (arXiv:2602.19752, 2026): Graph autoencoder generaliza entre Hamiltonians,
  pero no reporta cross-topology transfer explícito (train-on-one, deploy-on-another).
- Huang et al. (Science, 2022): Prueba teórica de que ML puede generalizar dentro de una
  fase, pero no aborda transferencia entre geometrías de lattice.
- **Diferenciación**: No existe trabajo previo que pruebe train-on-topology-A →
  deploy-on-topology-B para GNN predictores de VQE. Es la primera prueba directa
  del claim "topology-agnostic" que hacen los papers de GNN-VQE.

---

### EXP-S4: Landscape Analysis a N=20 (F3 + B4 Extension)

**Hipótesis**: El landscape de VQE a N=20 p=2 tiene estructura cualitativamente
diferente a N=6/N=10: (a) aparecen saddle points, (b) fraction_near_gs colapsa,
(c) condition number crece significativamente. Esto explica por qué G3 falló.

**Justificación científica**:
- G3 mostró que "1 restart + freeze FAILS at N=20" (ΔE/gap=1.26)
- B4 confirmó 0 saddle points a N=6 y N=10 (κ N-independent)
- Pero G3 contradice esto → el landscape DEBE cambiar entre N=10 y N=20
- Sin esta explicación, el resultado negativo de G3 queda sin diagnóstico

**Estado del código**:
- `landscape_fluctuation()` en `analysis/landscape.py` — implementado
- `compute_hessian()` en `analysis/landscape.py` — implementado
- MPS backend validado para N=20 (chi=64, exacto para 1D HVA)
- F3 ejecutado solo a N=6; B4 ejecutado a N=6 y N=10

**Verificación de no-duplicación**:
- Búsqueda "landscape.*N=20|F3.*N=20|B4.*N=20" — 0 matches
- Búsqueda "fluctuation.*N.?=.?20" — 0 matches
- No existe resultado de landscape a N=20 en ningún binnacle

**Configuración exacta**:
```
F3 a N=20 (fluctuation):
  N=20, p=2, chain_1d, J=1.0
  h_values: [1.5, 1.75, 2.0, 2.5, 3.0]
  n_samples: 100 por (h, seed)
  Parameter sampling: Uniform [-π, π]^4
  Seeds: [42, 43, 44]
  Backend: MPS chi=64 (AerSimulator)
  Métricas: fluctuation = Var(E)/E_mean², fraction_near_gs

B4 a N=20 (Hessian):
  N=20, p=2, chain_1d, J=1.0
  h_values: [2.0, 1.75, 1.5] (solo régimen válido)
  VQE: 5 restarts, maxiter=500, MPS chi=64
  Hessian: central finite differences, ε=5e-3
  Seeds: [42, 43, 44]
  Métricas: eigenvalues, condition number, min type (saddle/minimum)

Comparación p=1 vs p=2 a N=20:
  Misma config pero p=1 (2 params)
  Verificar si p=1 landscape es más benigno (como F3@p=1 a N=6)
```

**Resultado esperado**:
- fraction_near_gs colapsa a ~0 (vs 0.077 a N=6)
- κ crece 10-100× respecto a N=10
- Posibles saddle points (explicaría G3)
- p=1 landscape más benigno (consistente con p=1 success a N=20)

**Ejecución**: Script nuevo (~100 líneas), usa landscape.py + MPS backend.

**Tiempo estimado**: ~30-45 min (MPS VQE a N=20 es lento: ~50s/punto)

**Riesgos**:
- MPS sampling (100 puntos × 5 h-values × 3 seeds = 1500 evaluaciones) podría
  tomar >1h → reducir a 50 samples si necesario
- Hessian a N=20 requiere 4×2+1=9 evaluaciones por punto → ~7 min/h-point
- Mitigación: ejecutar F3 primero (más rápido), B4 solo si F3 muestra cambio

**Prior Art (literatura)**:
- Wiersema et al. (PRX Quantum, 2020): Estudia landscape de HVA pero solo para N≤12
  y no reporta scaling de condition number ni fraction_near_gs con N.
- arXiv:2302.08529: Prueba teórica de que HVA evita barren plateaus, pero no
  caracteriza cómo cambia el landscape con N (saddle points, κ).
- G3 (nuestro, 2026-05-25): Mostró que 1 restart falla a N=20 pero NO explicó por qué.
- B4 (nuestro, N=6 y N=10): Confirmó 0 saddle points y κ N-independent hasta N=10.
- **Diferenciación**: El gap N=10→N=20 nunca se ha caracterizado. Si aparecen saddle
  points o κ crece, es la primera evidencia de un cambio cualitativo del landscape
  de HVA con el tamaño del sistema.

---

## Experimentos Aprobados (Tier 2 — Valor Moderado)

### EXP-S5: MC-Dropout para UQ Calibrada

**Hipótesis**: MC-Dropout (dropout activo en inferencia, T=50 forward passes)
produce incertidumbre calibrada con correlación r > 0.7 entre varianza predicha
y ΔE/gap real, corrigiendo el fallo de G2 (r=0.195 con ensemble naive).

**Justificación científica**:
- G2 demostró que ensemble naive (mismo dato, diferente init) NO funciona (r=0.195)
- MC-Dropout captura incertidumbre epistémica real (Gal & Ghahramani, 2016)
- Si r>0.7 → método UQ publicable sin costo adicional de VQE
- Si r<0.5 → resultado negativo que cierra la línea de UQ para esta arquitectura

**Estado del código**:
- `MPNNPredictor` — NO tiene dropout actualmente (verificado: 0 matches en mpnn.py)
- Requiere: añadir `nn.Dropout(p=0.1)` entre capas GINConv (modificación mínima)
- G2 resultados disponibles como baseline (r=0.195)

**Verificación de no-duplicación**:
- Búsqueda "mc.dropout|MC.Dropout|mc_dropout" — 0 matches en todo el repo
- G2 usó ensemble (5 MPNNs, mismo dato, diferente init) — método diferente
- Dropout NO existe en el MPNN actual

**Configuración exacta**:
```
Modificación al MPNN:
  Añadir nn.Dropout(p=0.1) después de cada GINConv + BatchNorm + ReLU
  (3 capas → 3 dropouts)

Protocolo MC-Dropout:
  1. Entrenar MPNN normalmente (con dropout, 6000 epochs)
  2. En inferencia: model.train() (mantiene dropout activo)
  3. T=50 forward passes por cada h_test
  4. Varianza = Var(θ_pred) sobre T passes
  5. Correlacionar varianza con ΔE/gap real (Pearson r)

Config:
  N=6, p=2, chain_1d
  h_train: 17 puntos [1.0, 2.0]
  h_test: [1.0, 1.1, 1.2, ..., 2.0] (11 puntos, incluye train)
  Seeds: [42, 43, 44]
  MPNN: h=64, L=3, 6000 epochs, dropout=0.1
  T (forward passes): 50
  Métrica: Pearson r(Var(θ), ΔE/gap)
```

**Resultado esperado**: r ∈ [0.5, 0.8] (calibración parcial).
MC-Dropout debería capturar que puntos cerca de h_c tienen mayor incertidumbre.

**Ejecución**: Modificar mpnn.py (+6 líneas), script nuevo (~60 líneas).

**Tiempo estimado**: ~5 min (training 6000 epochs + 50×11 forward passes)

**Riesgos**:
- Dropout p=0.1 podría degradar accuracy del MPNN → verificar MSE no empeora >10%
- Si r<0.3 → MC-Dropout tampoco funciona para esta arquitectura (resultado negativo)
- Requiere modificar código en `src/qmbp_simulation/predictors/mpnn.py` (módulo estable)
- Mitigación: hacer la modificación backward-compatible (dropout=0.0 por defecto)

**Nota sobre código estable**: Este experimento requiere modificar `mpnn.py` (listado
como estable). La modificación es mínima y backward-compatible: añadir parámetro
`dropout_rate: float = 0.0` al constructor. Con dropout_rate=0.0, el comportamiento
es idéntico al actual. Solicitar aprobación explícita antes de ejecutar.

**Prior Art (literatura)**:
- Gal & Ghahramani (ICML, 2016): Fundamento teórico de MC-Dropout como aproximación
  bayesiana. Método bien establecido en ML clásico.
- arXiv:2604.10896 (Apr 2026): Compara quantum shot-based UQ contra MC-Dropout y
  Deep Ensembles. Muestra que quantum UQ logra cobertura dentro del 1-3% del target.
  **Implicación**: MC-Dropout es un baseline razonable pero puede ser superado.
- arXiv:1910.03127: "Evaluating Scalable UQ Methods for DNN-Based Molecular Property
  Prediction" — ensembles y bootstrapping superan consistentemente a MC-Dropout.
  **Riesgo elevado**: La literatura sugiere que MC-Dropout podría dar r<0.5.
- Miao et al. (PRA, 2024): Usa dropout en NN-VQE pero como regularización, no para UQ.
- **Diferenciación**: MC-Dropout para UQ de predicciones GNN-VQE no ha sido probado.
  Pero la literatura sugiere que el resultado podría ser negativo (r<0.5). Esto
  sigue siendo publicable como resultado negativo que cierra la línea.
- **Ajuste de expectativa**: Reducir umbral de éxito de r>0.7 a r>0.5 dado el prior
  de la literatura. Si r<0.3, considerar conformal prediction como alternativa.

---

### EXP-S6: Pipeline Completo p=1 N=20 con MPNN

**Hipótesis**: Con 15-20 puntos de entrenamiento en h∈[2.25, 4.0] y 5 restarts,
el MPNN despliega correctamente a N=20 p=1 con ΔE/gap < 5% en todo h_test ∈ [2.5, 3.5].

**Justificación científica**:
- VQE a N=20 p=1 VALIDADO (C3: ΔE/gap=1.58%, seeds 42/43 perfectos)
- MPNN deployment NUNCA ejecutado con config corregida (solo interpolación lineal en C3)
- El intento original (binnacle-p1-scaling) usó solo 6 puntos → MPNN falló
- Con 15-20 puntos + sign consistency (confirmada por C3), debería funcionar

**Estado del código**:
- VQE N=20 p=1 validado (MPS chi=64, 3 restarts, maxiter=100)
- MPNN training funcional para p=1 (output_dim=2)
- NO existe variant runner para N=20 p=1 (solo N=10)
- C3 confirmó: sign canonicalization NO necesaria con 3+ restarts

**Verificación de no-duplicación**:
- binnacle-p1-scaling: MPNN con 6 puntos → solo h=3.0 pasa (insuficiente)
- C3: usó interpolación lineal, NO MPNN real
- Búsqueda en results/thesis/ — no existe directorio p1_N20_pipeline
- El pipeline end-to-end con MPNN a N=20 p=1 NUNCA se completó

**Configuración exacta**:
```
Phase 1 (Exact Diag):
  N=20, p=1, chain_1d, J=1.0
  h_values: [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25] (8 puntos, descendente)
  + puntos intermedios: [3.875, 3.625, 3.375, 3.125, 2.875, 2.625, 2.375] (7 más)
  Total: 15 puntos en [2.25, 4.0]
  Backend: MPS chi=64 (DMRG para ground truth)

Phase 2 (VQE):
  5 restarts, maxiter=100, σ=0.3, L-BFGS-B
  Descending warm-start
  Seeds: [42, 43, 44]
  Fidelity filter: ≥ 0.93

Phase 3 (MPNN Training):
  h=128, L=3, 6000 epochs, lr=1e-3, patience=500
  output_dim=2 (θ_zz, θ_x)
  Training data: puntos con fidelity ≥ 0.93

Phase 4 (Deployment):
  h_test: [2.5, 3.0, 3.5] (interpolación dentro del régimen válido)
  Métrica: ΔE/gap < 5%
  Baseline: cold-start comparison
```

**Resultado esperado**: ΔE/gap < 3% en todos los h_test (basado en que el mapping
h→θ_zz es smooth y monotónico, y θ_x ≈ constante ±1.178).

**Ejecución**: Crear variant runner para N=20 p=1 (~80 líneas).

**Tiempo estimado**: ~20-30 min (MPS VQE: ~50s/punto × 15 puntos × 5 restarts × 3 seeds)

**Riesgos**:
- Seed 44 tiene local minimum conocido (ΔE/gap=0.437) con 3 restarts
- Con 5 restarts debería escapar, pero no confirmado
- Mitigación: si seed 44 falla, reportar 2/3 seeds pass (consistente con C3)
- MPS VQE es lento → podría tomar >30 min

---

## Experimentos Descartados (con justificación)

### ~~Data Efficiency N=10/20 (extensión G1)~~
**Descartado**: G1 ya estableció k_min=9 a N=6. Extender a N=10/20 es incremental
y no produce nuevo aprendizaje cualitativo. El resultado predecible (k_min crece
linealmente con N) no justifica el costo computacional.

### ~~D1 Phase Detection a N=20~~
**Descartado**: Requiere EXP-S6 completado primero (necesita MPNN entrenado a N=20).
Además, el resultado esperado (peak se acerca a h_c=1.0) es predecible por
finite-size scaling y no aporta nuevo insight metodológico.

### ~~N=16 p=1 con MPS~~
**Descartado**: Es un punto intermedio de la ley A3 que ya tiene R²=1.0000.
Validar un punto más no cambia la conclusión. El resultado de N=16 heavy-hex
(Phase 3 falla por fidelity filter) ya documenta las limitaciones a este tamaño.

### ~~Bootstrap UQ~~
**Descartado**: Más costoso que MC-Dropout (5× entrenamiento) y teóricamente
equivalente. Solo ejecutar si EXP-S5 (MC-Dropout) falla con r<0.3.

---

## Experimentos Adicionales (Nuevas Propuestas)

### EXP-S7: Barren Plateau Scaling Formal (F3 multi-N)

**Hipótesis**: La fluctuation del landscape Var(E)/E² escala como O(1) con N
(no exponencialmente), confirmando cuantitativamente la ausencia de barren
plateaus predicha por Mele et al. (2026) para HVA.

**Justificación científica**:
- F3 solo midió fluctuation a N=6 (resultado: [1.27, 5.26], todo >1.0)
- Mele et al. (2026) predice que HVA NO tiene barren plateaus
- Pero la predicción es para profundidad O(log n) — nuestro p=2 es constante
- Medir fluctuation vs N (4,6,8,10) confirmaría/refutaría cuantitativamente
- Si fluctuation decrece exponencialmente con N → barren plateau emergente

**Verificación de no-duplicación**:
- Búsqueda "barren.*plateau.*scaling|fluctuation.*vs.*N" — 0 matches
- F3 solo ejecutado a N=6 (una sola medición, no scaling)

**Configuración exacta**:
```
N ∈ {4, 6, 8, 10}, p=2, chain_1d, J=1.0
h_values: [1.0, 1.5, 2.0] (3 puntos representativos)
n_samples: 200 por (N, h, seed)
Seeds: [42, 43, 44]
Backend: StatevectorEstimator
Métrica: fluctuation(N) para cada h fijo
Análisis: fit fluctuation ~ a·exp(-b·N) vs fluctuation ~ c (constante)
```

**Resultado esperado**: fluctuation ≈ O(1) para todo N≤10 (sin decay exponencial).
Confirma Mele et al. para HVA p=2 en TFIM.

**Tiempo estimado**: ~3 min (N≤10 es rápido con StatevectorEstimator)

**Prior Art (literatura)**:
- Mele et al. (Nature Physics, 2026): Prueba teórica de ausencia de barren plateaus
  en circuitos shallow bajo ruido. Pero la prueba es para profundidad O(log n) y
  nuestro HVA tiene profundidad constante (p=2). La confirmación empírica es necesaria.
- arXiv:2302.08529 (2023): "HVA without barren plateaus" — prueba teórica para HVA
  específicamente, con condiciones sobre los parámetros. Pero no proporciona datos
  numéricos de fluctuation vs N.
- Cerezo et al. (Nature Comms, 2021): Prueba que costos locales evitan BPs en circuitos
  shallow. Teórico, sin datos empíricos para HVA en TFIM.
- F3 (nuestro, N=6): Confirmó fluctuation>1.0 en un solo punto (N=6). No es scaling.
- **Diferenciación**: Primera medición empírica de fluctuation(N) para HVA p=2 en TFIM
  con múltiples tamaños. Complementa las pruebas teóricas con datos numéricos.

---

### EXP-S8: Finite-Size Scaling de h_c via D1

**Hipótesis**: El peak de ||dW/dh|| del MPNN converge a h_c=1.0 cuando N→∞,
siguiendo la ley de finite-size scaling h_peak(N) = h_c + a·N^(-1/ν) con ν=1.

**Justificación científica**:
- D1 detecta peak en h≈0.7 (N=6) y h≈0.6 (N=10, con dropout=0.1)
- Con N=4,6,8,10 → 4 puntos para fit de finite-size scaling
- Si ν extraído ≈ 1.0 → el MPNN "ve" la transición de fase correctamente
- Conecta machine learning con física de transiciones de fase (publicable)

**Verificación de no-duplicación**:
- Búsqueda "finite.size.*h_c|D1.*scaling" — 0 matches
- D1 ejecutado a N=6 y N=10 individualmente, pero NUNCA como scaling study

**Configuración exacta**:
```
N ∈ {4, 6, 8, 10}, p=2, chain_1d, J=1.0
MPNN: h=128, L=3, 6000 epochs, dropout=0.1 (regularizado, como D1-reg)
h_train: full range [0.5, 2.5] (25 puntos, step 0.1)
Seeds: [42, 43, 44, 45, 46] (5 seeds para estadística robusta)
Métrica: h_peak = argmax ||dW/dh|| para cada (N, seed)
Análisis: fit h_peak(N) = 1.0 + a·N^(-1/ν)
```

**Resultado esperado**: ν ≈ 1.0 ± 0.2 (consistente con TFIM 1D, clase Ising).

**Tiempo estimado**: ~15 min (4 tamaños × 5 seeds × 6000 epochs)

**Riesgos**:
- N=4 podría ser demasiado pequeño para finite-size scaling
- Varianza entre seeds (D1 mostró seed sensitivity sin regularización)
- Mitigación: usar dropout=0.1 (D1-reg confirmó std=0.13 vs 0.90)

**Prior Art (literatura)**:
- Hernandes et al. (arXiv:2503.17140, 2025): Detecta transiciones de fase desde el
  weight space de NQS. Validado en TFIM y J₁-J₂ Heisenberg. Pero NO hace finite-size
  scaling del peak ni extrae ν.
- D1 (nuestro, N=6 y N=10): Detectó peaks en h≈0.7 (N=6) y h≈0.6 (N=10). Pero
  nunca se hizo el fit h_peak(N) = h_c + a·N^(-1/ν) para extraer ν.
- arXiv:2501.03981 (2025): Usa NN para extraer exponentes críticos via finite-size
  scaling, pero con datos de IPR (inverse participation ratio), no weight gradients.
- **Diferenciación**: Primera extracción de ν desde weight-space gradients de un MPNN
  entrenado para VQE. Conecta ML interpretability con física de transiciones de fase
  de forma cuantitativa (no solo cualitativa como Hernandes et al.).

---

## Orden de Ejecución Recomendado

```
Semana 1 (5 días):
  Día 1: EXP-S1 (Heisenberg RD1 + CA1) — código listo, solo ejecutar
  Día 2: EXP-S3 (Cross-topology transfer) — datos existentes, script nuevo
  Día 3: EXP-S7 (Barren plateau scaling) — rápido, 3 min ejecución
         EXP-S8 (D1 finite-size scaling) — 15 min ejecución
  Día 4: EXP-S2 (Entropía + scaling law) — script nuevo, ~5 min ejecución
  Día 5: Buffer / análisis de resultados / documentación

Semana 2 (5 días):
  Día 6: EXP-S5 (MC-Dropout) — requiere modificar mpnn.py (pedir aprobación)
  Día 7: EXP-S4 (Landscape N=20) — 30-45 min ejecución
  Día 8: EXP-S6 (Pipeline p=1 N=20) — 20-30 min ejecución
  Día 9: Buffer / re-runs si necesario
  Día 10: Consolidación final, actualizar binnacles y project-status
```

---

## Dependencias entre Experimentos

```
EXP-S1 (Heisenberg) ──────────── independiente
EXP-S2 (Entropía) ────────────── independiente (usa exact diag existente)
EXP-S3 (Cross-topology) ──────── independiente (usa datos existentes)
EXP-S4 (Landscape N=20) ──────── independiente (usa MPS existente)
EXP-S5 (MC-Dropout) ──────────── independiente (requiere aprobación para mpnn.py)
EXP-S6 (Pipeline p=1 N=20) ───── independiente (VQE validado en C3)
EXP-S7 (Barren plateau) ──────── independiente
EXP-S8 (D1 scaling) ──────────── independiente (D1 ya ejecutado a N=6, N=10)
```

Ningún experimento depende de otro. Pueden ejecutarse en cualquier orden.
La secuencia propuesta optimiza: rápidos primero, lentos después, buffer al final.

---

## Criterios de Éxito por Experimento

| ID | Éxito (positivo) | Éxito (negativo) | Fallo metodológico |
|----|-------------------|-------------------|--------------------|
| S1 | Régimen válido encontrado (fid≥0.93) | S_max(HVA) < S(GS) cuantificado | VQE no converge en 1500 iter |
| S2 | S(h_min) = constante (var<10%) | S(h_min) varía >30% con N | DMRG falla a N=20 |
| S3 | Transfer ΔE/gap < 10% | Transfer falla (>50%) + cuantificado | Datos incompatibles entre topos |
| S4 | Saddle points a N=20 (explica G3) | Landscape benigno (G3 por otra causa) | MPS timeout |
| S5 | r > 0.7 (UQ calibrada) | r < 0.3 (MC-Dropout no funciona) | Dropout degrada accuracy >20% |
| S6 | ΔE/gap < 5% (3/3 seeds) | 2/3 seeds pass (seed 44 issue) | MPNN no converge |
| S7 | Fluctuation O(1) con N | Fluctuation decae exp con N | — |
| S8 | ν ≈ 1.0 ± 0.3 | ν >> 1 o no converge | N=4 demasiado pequeño |

**Nota**: Tanto resultados positivos como negativos son publicables si están
bien caracterizados. Un "fallo metodológico" indica un problema de ejecución
(no de ciencia) que requiere debugging.

---

## Contribuciones a la Tesis por Experimento

| ID | Sección tesis | Tipo de contribución |
|----|---------------|---------------------|
| S1 | §5.5 Generalization | Resultado negativo riguroso: HVA es TFIM-específico |
| S2 | §2.4 Theory | Explicación analítica del exponente β=1.33 |
| S3 | §5.2 Topology | Validación/refutación del claim topology-agnostic |
| S4 | §4.3 Scaling | Explicación del fallo G3 (landscape changes with N) |
| S5 | §3.4 UQ | Método UQ calibrado (o cierre de la línea) |
| S6 | §4.6 N=20 | Pipeline end-to-end a N=20 (milestone de escalabilidad) |
| S7 | §2.3 Trainability | Confirmación cuantitativa de ausencia de barren plateaus |
| S8 | §5.1 Phase Detection | Conexión ML ↔ física de transiciones de fase |

---

## Resumen Ejecutivo

**8 experimentos** propuestos, todos verificados contra resultados existentes:
- **4 Tier 1** (alto valor): S1, S2, S3, S4
- **2 Tier 2** (valor moderado): S5, S6
- **2 adicionales** (nuevas propuestas): S7, S8

**Tiempo total estimado**: ~2-3 horas de ejecución + 2 semanas de desarrollo/análisis.

**Infraestructura requerida**: Solo simulación local (StatevectorEstimator + MPS chi=64).
No requiere hardware cuántico, cloud computing, ni GPUs.

**Resultado mínimo viable**: Si solo se ejecutan S1 + S2 + S3, se obtienen 3 contribuciones
nuevas a la tesis (model-specificity, scaling theory, topology transfer) en ~1 semana.
