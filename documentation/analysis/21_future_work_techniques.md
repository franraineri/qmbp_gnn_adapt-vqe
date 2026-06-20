# Future Work: Advanced Techniques for GNN-HVA Scaling

> Techniques identified as promising extensions beyond the scope of this thesis.
> Included for Chapter 7 (Future Work) discussion with literature references.
>
> **Date**: 2026-06-08
> **Last status update**: 2026-06-19

---

## Implementation Status Tracker

| # | Technique | Status | Evidence |
|:-:|-----------|:------:|----------|
| 1 | NIL (Neighbor-Informed Learning) | 🔴 Not started | No code found. Concept only. |
| 2 | ML-QEM | ✅ **DONE** | `src/qmbp_simulation/predictors/gnn_qem.py` — GNN-QEM fully implemented: `correct_energy()`, `GNNQEMCorrector`, `QEMSample`, cross-topology validated (+72.3% zero-shot), integrated in deployment script and mitigation benchmark (C10_kitchen_sink). |
| 3 | Telemetry-Driven Adaptive Mitigation (GSC-QEMit) | 🟡 **Partial (60%)** | ✅ `take_calibration_snapshot()` — implemented in `noisy_utils.py`. ✅ `check_calibration_drift()` — implemented with T1 drift >20% abort. ✅ `run_adaptive_zne()` — PEA→GF automatic fallback based on R². ❌ Predictive drift model — not implemented. ❌ Multi-armed bandit strategy selection — not implemented. ❌ Mid-run strategy switching — not implemented. |
| 4 | SC-ADAPT-VQE | 🔴 Not started | No code. Not needed (HVA p≤2 already translationally invariant by design). |
| 5 | Utility-Scale Hamiltonian Engineering | 🔴 Not started | Concept only. Bond-resolved HVA is structurally equivalent but not the same approach. |
| 6 | TITAN Parameter Freezing | ✅ **DONE** | `experiments/helpers/parameter_freezing.py` — `analyze_parameter_activity()`, `frozen_vqe()`. `experiments/optimization/exp_b2_freezing.py` — B2 experiment. `config.py` has `freeze_params` + `freeze_after_h`. Used in `exp_g3_n20_optimized.py` (θ_zz2, θ_x2 frozen at h≥1.5). |
| 7 | Probing p≤2 Expressibility (FW-A/B/C) | 🟡 **Partial (FW-C: 70%)** | ✅ FW-C: `experiments/scaling/exp_s1_entanglement_scaling.py` has `compute_entanglement_entropy()` and correlates S(L/2) with viability. ❌ FW-A: depth scaling p=8,10 not executed beyond V9 data. ❌ FW-B: Symmetry-preserving Heisenberg ansatz not implemented. |
| 8 | GNN for Quantum Chip Design | 🔴 Not started | Literature reference only. Different stack level (chip fab vs algorithm). |
| 9 | Qiskit Addons (PNA/SLC/OBP/Cutting) | 🔴 Not started | Evaluated 2026-06-19. None integrated. See Section 9 for detailed assessment. |

### Summary

- **Fully implemented (2/9)**: ML-QEM (#2), TITAN Freezing (#6)
- **Partially implemented (2/9)**: Adaptive Mitigation (#3, 60%), Entanglement Predictor (#7, 70%)
- **Not started (5/9)**: NIL (#1), SC-ADAPT (#4), Hamiltonian Engineering (#5), Chip Design (#8), Qiskit Addons (#9)

---

## 1. Neighbor-Informed Learning (NIL) for Error Mitigation

**What it is**: A unified QEM framework that generalizes ZNE and PEC by using
"neighbor circuits" (small perturbations of the target circuit) as training data
to predict the ideal observable value. Instead of extrapolating along a single
noise axis (ZNE), NIL learns from a neighborhood of circuits in parameter space.

**Why it matters for GNN-HVA**: The GNN already generates θ_opt predictions that
form a natural "neighborhood" around the deployed circuit. NIL could use GNN-
predicted circuits at nearby h-values as the neighbor set, eliminating the need
for explicit noise amplification (no PEA overhead).

**Integration path**: Replace `run_adaptive_zne()` with NIL using GNN predictions
at h ± δh as neighbor circuits. Zero additional QPU cost if GNN predictions are
available for multiple h-values.

**Reference**: Wei et al., "Scalable Quantum Error Mitigation with Neighbor-Informed
Learning," arXiv:2512.12578 (2024).

---

## 2. ML-QEM: Machine Learning for Practical QEM

**What it is**: Train a classical ML model on (noisy_observable, true_observable)
pairs from calibration circuits, then apply to correct new circuits. Demonstrated
at 100 qubits on IBM hardware with 100× cost reduction vs standard mitigation.

**Relationship to current work**: Our GNN-QEM module is conceptually the same
approach but specialized for energy correction with graph structure. The generic
ML-QEM additionally handles arbitrary observables and uses transfer learning
across circuit families.

**Integration path**: The GNN-QEM `correct_energy()` function already implements
this pattern. Future extension: train on multiple observable types (not just
energy) and enable cross-topology transfer via shared latent space.

**Reference**: Czarnik et al., "Machine Learning for Practical Quantum Error
Mitigation," arXiv:2309.17368 (2023). Experiments on IBM 100-qubit hardware.

---

## 3. Telemetry-Driven Adaptive Error Mitigation (GSC-QEMit)

**What it is**: A hierarchical framework that monitors QPU telemetry (T1, T2, gate
errors) in real-time and adaptively selects the mitigation strategy during execution.
Uses a forecast model to predict noise evolution and a multi-armed bandit to optimize
the mitigation/overhead trade-off.

**Current coverage in our pipeline**:
- ✅ `take_calibration_snapshot()` — telemetry capture
- ✅ `check_calibration_drift()` — drift detection (abort if T1 drift > 20%)
- ✅ `run_adaptive_zne()` — automatic PEA → GF fallback based on R²
- ❌ Predictive drift model (not implemented)
- ❌ Multi-armed bandit for strategy selection (not implemented)
- ❌ Mid-run strategy switching (current: decision at run start only)

**When it would help**: Long hardware runs (>1h) where noise characteristics
shift during execution. For our typical runs (~30 min VQE sweep), the static
adaptive_zne is sufficient.

**Reference**: "A Telemetry-Driven Hierarchical Forecast-and-Bandit Framework
for Adaptive Quantum Error Mitigation," arXiv:2604.24551 (2024).

---

## 4. SC-ADAPT-VQE: Scalable Circuits for Translationally Invariant Systems

**What it is**: An algorithm that determines HVA-like circuit structure CLASSICALLY
(on small systems) and then tiles the result to arbitrary system sizes. Demonstrated
on the Schwinger model vacuum at 100 qubits on IBM Eagle.

**Relevance**: Our HVA p=1 for 1D TFIM is already translationally invariant —
the global HVA circuit IS a "scalable circuit" by definition. SC-ADAPT-VQE would
matter if we needed p>2 or non-trivial ansatz structure near criticality.

**Integration path**: Low priority. Our p≤2 constraint (Mele et al.) already ensures
circuits are scalable. SC-ADAPT-VQE is more relevant for non-HVA ansätze.

**Reference**: Farrell et al., "Scalable Circuits for Preparing Ground States on
Digital Quantum Computers: The Schwinger Model Vacuum on 100 Qubits,"
arXiv:2308.04481 (2024). PRX Quantum 5(2), 020315.

---

## 5. Utility-Scale Hamiltonian Engineering (103 qubits Kagome)

**What it is**: Split VQE into local (per-site, classically optimizable) and global
(entanglement, quantum) components. Allows single-layer ansatz at 100+ qubits by
pre-computing the local part analytically.

**Our version**: Bond-resolved HVA with θ_x (local) vs θ_zz (global) is
structurally equivalent. The Kagome paper additionally uses "Hamiltonian engineering"
to modify the physical Hamiltonian to simplify the required ansatz — making defect
triangles couple more strongly to mimic the dynamics.

**Key difference**: They CHANGE the Hamiltonian to fit the hardware. We CHANGE the
parametrization to fit the GNN. Both achieve utility-scale from shallow circuits.

**Reference**: "Utility-Scale Quantum Computation of Ground-State Energy in a 100+
Site Planar Kagome Antiferromagnet via Hamiltonian Engineering,"
arXiv:2507.06361 (2025). IBM Heron r1/r2 processors.

---

## 6. Parameter Freezing (TITAN)

**What it is**: During VQE optimization, identify parameters that converge early
and freeze them — reducing the effective dimension of the landscape. Uses
trajectory analysis to detect convergence.

**Relevance to bond-resolved**: At N=40 with 79 params, many θ_zz bonds in chain_1d
converge to nearly identical values (translational symmetry). TITAN would detect this
and freeze them, reducing to ~2-5 effective parameters. This would make cold-start
VQE viable even at 79 nominal params.

**Key insight for thesis**: "TITAN-style freezing would recover the quasi-2D structure
that makes chain_1d easy. On heavy_hex (non-uniform), fewer parameters freeze →
GNN remains necessary."

**Reference**: "A Trajectory-Informed Technique for Adaptive Parameter Freezing
in Large-Scale VQE," arXiv:2509.15193 (2025).

---

## 7. Probing the p≤2 Expressibility Boundary for XX+YY Models

> **Origen**: Análisis de sesión 2026-06-15 — extensibilidad de modelos y límites del HVA.
> **Contexto completo**: `documentation/binnacles/binnacle-hamiltonian-candidates.md` (Addendum 2).

### El límite conocido

Los modelos con interacciones XX+YY (Heisenberg, Kitaev, XY) requieren entanglement
que escala linealmente con N. HVA p≤2 no puede producir este entanglement — confirmado
en V9 (Heisenberg: fid_max=48% a p=6, ΔE_gap ≈ 3.8N) y en el análisis de Kitaev
(fid_max=16% a N=4 p=1). Esta es una limitación física, no de implementación.

### Qué estudiar y cómo (si tiene sentido ejecutarlo)

**Objetivo**: Confirmar (o refutar) que el límite viene del entanglement y no de
otros factores subsanables como el estado inicial o la cantidad de parámetros.

#### Experimento FW-A: Depth Scaling en punto de alta simetría

**Hipótesis**: Si el límite es de entanglement, la fidelidad debe saturar antes
de alcanzar p=N/2 (límite teórico para reproduir el estado exacto con HVA tipo brick-wall).

**Protocolo**:
- Modelo: Heisenberg XXZ (Δ=1), N=6, estado Néel, h=3 (régimen estudiado)
- Barrer p ∈ {1, 2, 3, 4, 5, 6, 8, 10} (ya tenemos p=2..6 de V9)
- Medir: fidelidad, entanglement entropy S(L/2), gap de energía ΔE/gap
- Resultado esperado: S(L/2) crece logarítmicamente con p, satura antes de fid≥0.90

**Por qué tiene sentido**: V9 ya tiene p=2..6. Solo hay que extender a p=8, p=10.
El experimento toma ~30min. Si la fidelidad satura en p=5-6 (como sugiere V9),
eso confirma que el problema no se resuelve con más profundidad dentro del HVA.

**Condición para ejecutarlo**: Solo si el capítulo 5/6 necesita un gráfico de
"depth scaling saturation" para sostener la claim de límite físico. Los datos de V9 (p≤6)
ya son suficientes para el argumento textual.

#### Experimento FW-B: Symmetry-Preserving Ansatz para Heisenberg

**Hipótesis**: Un ansatz que preserve el sector S_z=0 por construcción evita la
trampa del estado Néel (gradient=0 fuera del sector correcto).

**Protocolo**:
- Implementar `create_heisenberg_symmetric()` con gates que conserven S_z
- Referencia: Sharma et al. (arXiv:2512.23009) — validado en IQM Garnet
- Comparar fidelidad con HVA estándar a p=2 y p=4
- N=6, h=2 (régimen donde HVA falla más claramente)

**Por qué tiene sentido**: Esfuerzo ~1 semana. Si la hipótesis es correcta
(S_z-preserving llega a fid≥0.90 con p=3-4), entonces el Heisenberg pasa de
"no viable" a "viable con ansatz especializado". Eso es una nueva contribución.

**Condición para ejecutarlo**: El proyecto ya tiene toda la infraestructura.
Solo requiere `create_heisenberg_symmetric()` en `circuits/hva.py` y registrar
en el registry como `heisenberg_symmetric`. El registro es el lugar correcto:
no viola el Code Map "Stable" vs "Active Development" (circuits/ está en Stable,
pero el model registry acepta nuevas entradas sin modificar los builders existentes).

**ATENCIÓN**: No ejecutar antes del hardware deployment. Es trabajo post-tesis.

#### Experimento FW-C: Entanglement Entropy como predictor de viabilidad HVA

**Hipótesis**: La entropía de entrelazamiento del ground state exacto (S_exact)
predice si el HVA p≤2 puede expresarlo. Criterio propuesto: si S(L/2) ≤ log(2)
(un bit de entanglement), HVA p=1 es suficiente. Si S(L/2) ≤ 2·log(2), p=2 suficiente.

**Protocolo**:
- Para cada modelo del registry: calcular S(L/2) del ground state exacto (Phase 1, N=6)
- Correlacionar con fidelidad VQE real de Phase 2
- TFIM: S(L/2) ≈ 0.5 (h=2) → 1.0 (h=1.0). Heisenberg: S(L/2) ≈ 2.2 (antiferro)
- Si la correlación R²>0.9: el criterio es predictivo y se puede publicar

**Por qué tiene sentido**: Costo ~0 QPU (solo análisis de datos ya calculados en
Phase 1). ClassicalSolver ya devuelve el ground state exacto — solo añadir
`compute_entanglement_entropy(psi, cut=N//2)` en el análisis. Resultado:
una **regla predictiva de viabilidad de modelos** sin necesidad de ejecutar VQE.

**Prioridad**: ALTA. Bajo costo, alto valor científico, y los datos ya existen
(V9 y el binnacle de Heisenberg tienen todo lo necesario).

### Estado actual de la evidencia

| Modelo | S(L/2) estimada | HVA p=2 fid | Veredicto |
|--------|:-:|:---:|:---:|
| TFIM (h=2) | ~0.5 | ≥0.99 | ✅ |
| TFIM+Long (h=2, g=0.3) | ~0.5 | ≥0.98 | ✅ |
| TFIM frustrated (h=2, J₂=0.3) | ~0.5 | ≥0.999 | ✅ |
| Heisenberg (h=3) | ~2.2 | 0.48% (p=6) | ❌ |
| Kitaev (μ=1.5) | ~1.0-1.5 | 0.16 (p=1) | ❌ |

La correlación ya es visible. El FW-C solo requiere calcular S explícitamente
y ajustar el umbral.

### Conexión con literatura

- **Mele et al. (Nature Physics 2026)**: El límite de profundidad O(log N) en
  presencia de ruido no-unital es la restricción dura que hace que HVA p=2 sea
  el límite práctico. No es negociable para hardware NISQ.
- **Sumeet et al. (arXiv:2310.07600)**: Demuestran que se necesitan N/2 capas para
  alcanzar el límite termodinámico exacto — para N=6 eso es p=3, que ya excede el
  presupuesto de ZNE.
- **Tripathi et al. (arXiv:2604.20961)**: Confirman que HVA p=2 lucha con la
  entropía de entrelazamiento en la criticidad — validación independiente de nuestro resultado.

### Refs

- Sharma et al., arXiv:2512.23009 (symmetry-preserving para Heisenberg)
- Javanmard et al., arXiv:2401.02355 (MPS-inspired ansatz para Kagome)
- Mele et al., Nature Physics 2026 (límite de profundidad — regla p≤2)
- Sumeet et al., arXiv:2310.07600 (N/2 layers para límite termodinámico)
- Binnacle detallado: `documentation/binnacles/binnacle-hamiltonian-candidates.md` (Addendum 2)
- Resultados Heisenberg: `documentation/binnacles/binnacle-heisenberg-extension.md`
- Análisis de sesión: `documentation/analysis/15_heisenberg_future_work.md`

---

## 8. GNN for Quantum Chip Parameter Design

**What it is**: Use GNN to design parameters of superconducting quantum chips
(junction frequencies, coupling strengths). Achieves 51% fewer errors than
state-of-the-art on 870-qubit chips, 200× faster.

**Parallel to our work**: Same insight (graph structure encodes spatial relationships)
applied to a different level of the stack. Our GNN maps graph→circuit_params;
their GNN maps graph→chip_params.

**Thesis connection**: "Graph neural networks are proving essential across the full
quantum computing stack — from chip design (870 qubits) to error mitigation
(GNN-QEM) to variational parameter prediction (this work)."

**Reference**: "Scalable Parameter Design for Superconducting Quantum Circuits
with Graph Neural Networks," arXiv:2411.16354 (2024).

---

## 9. Qiskit Addons — Error Mitigation Techniques for Future Scaling

> **Fecha**: 2026-06-19
> **Contexto**: Evaluación completa del ecosistema Qiskit addons contra nuestro
> pipeline GNN-HVA. Técnicas que NO aplican hoy pero podrían ser útiles en
> escenarios de escalado o certificación rigurosa.

### Resumen de decisiones

| Addon | Relevancia actual | Relevancia futura | Acción |
|-------|:-----------------:|:-----------------:|--------|
| qiskit-addon-opt-mapper | ❌ 0% | ❌ 0% | Descartado (dominio incompatible) |
| qiskit-addon-mpf | ❌ 0% | ❌ 0% | Descartado (no hay Trotter error) |
| qiskit-addon-obp | ❌ 0% | ⚠️ 15% | Solo si p≥3 (circuitos profundos) |
| **qiskit-addon-cutting** | ❌ 0% | ⚠️ 60% | Solo si **N>100 en QPU** real |
| **qiskit-addon-pna** | ⚠️ 40% | ✅ 70% | Verificación independiente de PEA |
| **qiskit-addon-slc** | ⚠️ 30% | ✅ 60% | Error bounds formales / certificación |

---

### 9.1 Propagated Noise Absorption (PNA) — `qiskit-addon-pna`

**Docs**: https://qiskit.github.io/qiskit-addon-pna/
**Paper**: Tutorial usa ibm_kingston + kicked Ising 30q (nuestro mismo backend/modelo)
**Versión**: v0.2.0

#### Qué hace

Mitiga errores de gates 2Q absorbiendo el inverso del noise model aprendido
(Pauli-Lindblad) directamente en el observable medido. El resultado es un
observable modificado Õ que, al medirse en el circuito ruidoso, cancela el ruido
sin amplificarlo.

```
Pipeline: NoiseLearnerV3 → Pauli propagation (anti-noise) → Observable Õ → Medir
```

#### Diferencia fundamental con PEA-ZNE

| Aspecto | PEA-ZNE (actual) | PNA |
|---------|:----------------:|:---:|
| Modifica el circuito | Sí (amplifica noise) | No |
| Modifica el observable | No | Sí (absorbe anti-noise) |
| QPU overhead ejecución | ~50% (noise learning + amplified circuits) | 0% (solo noise learning) |
| Bias teórico | Depende de linealidad extrapolación | Depende de truncation budget |
| Requiere | qiskit-aer (PEA simulation) | samplomatic + NoiseLearnerV3 + Executor |

#### Cuándo sería útil

1. **Verificación cruzada post-Kingston**: Si PEA-ZNE da 0.37% en hardware real,
   ejecutar PNA con el mismo noise model y comparar resultados fortalece la
   credibilidad. Consistency entre dos métodos independientes = evidencia fuerte.

2. **Reducción de QPU budget**: Si el crédito IBM es limitado, PNA no requiere
   circuitos amplificados — solo el circuito original + noise learning (que PEA
   también necesita). Esto reduce el total de shots en ~33%.

3. **Circuitos fuera del régimen PEA**: Si en el futuro p=2 AQC-compressed da
   27 CZ (actualmente marginal para PEA), PNA podría funcionar donde PEA pierde
   linealidad, porque no depende de la extrapolabilidad del noise scaling.

4. **Observable multi-term expandido**: Si se miden observables locales (per-site
   magnetización, correladores ⟨Z_i Z_j⟩), PNA es natural — el observable expandido
   tiene terms 1-local y 2-local que crecen moderadamente bajo propagation.

#### Esfuerzo de integración

| Componente | Esfuerzo | Dependencias |
|-----------|:--------:|-------------|
| Instalar `qiskit-addon-pna` + `samplomatic` | 1h | pip, optional dep |
| Adaptar noise learning al workflow actual | 1 día | NoiseLearnerV3 (reemplaza PEA NoiseLearner) |
| Generar Õ para H_TFIM (19 Paulis) | 2h CPU | Pauli propagation (trivial para 18 CZ) |
| Integrar con Executor primitive | 1-2 días | Reemplaza EstimatorV2 actual |
| Validación end-to-end en FakeTorino | 1 día | Comparar vs PEA resultado |
| **Total** | **~4 días** | samplomatic, NoiseLearnerV3, Executor |

#### Prerrequisitos

- IBM credentials + acceso a NoiseLearnerV3 (requiere qiskit-ibm-runtime reciente)
- `samplomatic` package (IBM internal, disponible via pip desde 2025)
- Executor primitive (prototype en qiskit-ibm-runtime, alternativa a EstimatorV2)

#### Estimación de resultado esperado

Para HVA p=1 N=10 (18 CZ, depth_2q=14):
- Observable: H_TFIM ≈ 19 Pauli strings (ZZ nearest-neighbor + X single-site)
- Anti-noise generators por layer: ~19 × 2 (fwd/bwd) = ~38 propagations
- Circuito total a propagar: 14 layers de 2Q → max ~500 evolved terms
- `max_err_terms=1000`, `max_obs_terms=1000` → trivial para esta escala
- Resultado esperado: comparable a PEA (0.3-1% ΔE/gap)

---

### 9.2 Shaded Lightcones + PEC — `qiskit-addon-slc`

**Docs**: https://qiskit.github.io/qiskit-addon-slc/
**Paper**: arXiv:2409.04401 — "Lightcone shading for classically accelerated QEM"
**Versión**: v0.1.0

#### Qué hace

Reduce el sampling overhead de PEC (Probabilistic Error Cancellation) calculando
bounds clásicos sobre el impacto de cada error en el observable. Los errores con
bajo impacto se excluyen → overhead exponencialmente menor. Provee **error bounds
formales** sobre el bias residual.

```
Pipeline: Noise model Paulis → Forward/Backward bounds → Merge + Tighten
        → Prioritize (budget allocation) → Execute with anti-noise → Post-process
```

#### La propuesta de valor única

PEC es la ÚNICA técnica de mitigación que provee **bounds demostrables** en el
error. PEA-ZNE extrapola (sin garantía formal). PNA trunca (sin bound estricto
en el error de truncation en la práctica). SLC+PEC dice:

> "El bias residual de esta medición es ≤ ε con probabilidad 1 − δ"

Esto es fundamental para aplicaciones que requieren **certificación** (e.g.,
quantum chemistry accuracy, compliance con chemical accuracy de 1 kcal/mol).

#### Cuándo sería útil

1. **Certificación rigurosa de energía**: Si la tesis o un paper futuro necesita
   demostrar formalmente que |E_measured − E_exact| ≤ ε, SLC+PEC es el camino.
   PEA no provee esta garantía.

2. **Auditoría de PEA-ZNE**: Si los resultados de PEA en hardware real son
   sospechosamente buenos (o malos), SLC+PEC con bounds certificados confirma o
   refuta la validez.

3. **Circuitos con más qubits y local observable**: SLC escala particularmente
   bien cuando el observable es local (e.g., ⟨Z_0 Z_1⟩ en un circuito de 50 qubits).
   El lightcone limita la influencia del ruido lejano → overhead constante vs N.
   Para nuestro observable global (H_TFIM = Σ terms over ALL sites), el beneficio
   es menor pero aún significativo.

4. **Comparación con overhead de PEA para publicación**: Mostrar que SLC+PEC
   con γ²~500 da resultados equivalentes a PEA (50% overhead) pero con garantías
   formales = argumento fuerte para un paper.

#### Overhead estimado para nuestro circuito

Para HVA p=1 N=10 heavy_hex (18 CZ):
- Full PEC sin shading: γ² = 9^18 ≈ 10^17 → **imposible**
- Con lightcone convencional: γ² ≈ 9^8 ≈ 43M → **imposible**
- Con SLC (bounds + prioritization): γ² ≈ 200-800 (estimación basada en tutorial 20q)
- En shots: 16K × 500 = 8M shots → ~$200-500 USD en ibm_kingston (viable para 1 h-point)

Nota: El tutorial de IBM usa 4096 randomizations × 64 shots = 262K total shots
para un circuito 20q × 40 layers (mucho más profundo). Nuestro caso sería más
barato por la baja profundidad.

#### Esfuerzo de integración

| Componente | Esfuerzo | Dependencias |
|-----------|:--------:|-------------|
| Instalar `qiskit-addon-slc` + `samplomatic` | 1h | pip, optional dep |
| Generar noise_model_paulis para HVA boxed | 2h | boxing pass manager |
| Compute forward bounds (CPU-intensive) | 4h ejecución, 1h código | Multi-process, timeout tuning |
| Compute backward bounds | 4h ejecución, 1h código | Idem |
| Merge + tighten bounds | 30min | Funciones utility |
| Noise learning (NoiseLearnerV3) | 1 día QPU | Shared con PNA |
| compute_local_scales (prioritization) | 30min CPU | Después de noise learning |
| Integrar con Executor (anti-noise injection) | 1-2 días | samplomatic boxes |
| Post-processing (TREX + post-selection) | 1 día | qiskit-addon-utils |
| Validación en FakeTorino | 1 día | Comparar γ² y bias vs PEA |
| **Total** | **~7-10 días** | samplomatic, NoiseLearnerV3, Executor, cluster CPU |

#### Prerrequisitos

- `qiskit-addon-slc>=0.1.0` + `samplomatic>=0.16` + `qiskit-addon-utils>=0.3`
- NoiseLearnerV3 (noise learning hardware job)
- CPU cluster recomendado (8+ cores) para bounds computation (128 threads ideal)
- Executor primitive (alternativa a EstimatorV2)
- Conocimiento de Pauli propagation framework (`pauli-prop` package)

---

### 9.3 Operator Backpropagation (OBP) — `qiskit-addon-obp`

**Docs**: https://qiskit.github.io/qiskit-addon-obp/
**Versión**: v0.3.0

#### Qué hace

Absorbe slices del final de un circuito en el observable medido, reduciendo la
profundidad a ejecutar a cambio de un observable expandido (más Pauli groups).

#### Por qué NO aplica hoy

- Nuestro circuito (18 CZ, depth_2q=14) ya es más corto que el umbral donde OBP
  tiene sentido (>40 layers típicamente).
- PEA-ZNE funciona perfectamente a esta profundidad.
- El observable expandido requiere más grupos de medición → más shots.

#### Cuándo podría ser útil

**Escenario**: Si en el futuro se despliega HVA p≥3 (o AQC-compressed desde p=4)
con 50+ CZ gates donde PEA pierde linealidad y SLC es demasiado costoso.

OBP permitiría recortar las últimas 5-10 layers del circuito → reducir a ~30 CZ →
re-habilitar PEA-ZNE. Trade-off: observable pasa de 19 a ~50-100 Pauli groups,
pero eso es manejable con EstimatorV2 grouping.

**Esfuerzo**: ~2 días (slicing + budget tuning + validation).
**Probabilidad de necesitarlo**: Baja (<10%) dado p≤2 constraint de Mele et al.

---

### 9.4 Circuit Cutting — `qiskit-addon-cutting`

**Docs**: https://qiskit.github.io/qiskit-addon-cutting/
**Versión**: v0.10.0

#### Qué hace

Corta gates entanglantes → subcircuitos más pequeños → recombina via QPD.
Overhead: ×9 por cada CZ cortado.

#### Por qué NO aplica hoy

- N=10 qubits en QPU de 156q → sobra espacio.
- Mapomatic da layouts SWAP-free → 0 gates no-locales.
- Overhead ×9/gate es prohibitivo para 18 CZ.

#### Cuándo podría ser útil

**Escenario**: N=50-80 en hardware real donde el transpiler introduce 50+ SWAPs
para routing. Si se identifican 2-3 "cuellos de botella" (gates entre clusters
lejanos), cortarlos con overhead ×9 cada uno (total ×729) podría ser viable si
el circuito original es inviable por coherence time.

**Escenario alternativo**: Si IBM lanza QPUs con topología modular (multiple chips
connected via coherent links), cutting entre chips sería la técnica natural.

**Esfuerzo**: ~3-5 días (cut finding + subcircuit generation + reconstruction).
**Probabilidad de necesitarlo**: Muy baja (<5%) con topologías Heron actuales.

---

### 9.5 Técnicas descartadas definitivamente

| Addon | Razón de descarte |
|-------|-------------------|
| **qiskit-addon-opt-mapper** | Mapea problemas combinatorios (QUBO/MaxCut) a QPU. Nuestro Hamiltoniano TFIM ya ES un operador cuántico nativo — no hay paso de traducción. Dominio completamente incompatible (optimization vs condensed matter). |
| **qiskit-addon-mpf** | Reduce error de Trotter en evolución temporal e^{-iHt}. Nuestro VQE es estacionario — no hay Trotterización, no hay error de Trotter. Los ángulos θ son variacionales, no discretizaciones temporales. |

---

### 9.6 Matriz de decisión: cuándo activar cada técnica

```
¿PEA-ZNE da ΔE/gap < 5%?
├── SÍ → Usar PEA-ZNE (primary). Done.
│         └── ¿Necesitás verificación cruzada? → PNA (4 días)
│         └── ¿Necesitás error bounds formales? → SLC+PEC (10 días)
│
└── NO → ¿Por qué falla?
          ├── Circuito muy profundo (>30 CZ, R²<0.90)
          │   ├── OBP para recortar depth (2 días)
          │   └── Luego PEA-ZNE sobre circuito reducido
          │
          ├── Observable muy no-local
          │   └── SLC con lightcone parcial (7 días)
          │
          ├── QPU budget insuficiente para amplificación
          │   └── PNA — zero QPU overhead en ejecución (4 días)
          │
          └── Hardware topología fragmentada (multi-chip)
              └── Circuit cutting entre chips (5 días)
```

---

### 9.7 Referencias

- **PNA**: qiskit-addon-pna v0.2.0, https://qiskit.github.io/qiskit-addon-pna/
- **SLC**: qiskit-addon-slc v0.1.0, arXiv:2409.04401, https://qiskit.github.io/qiskit-addon-slc/
- **OBP**: qiskit-addon-obp v0.3.0, arXiv:2502.01897, https://qiskit.github.io/qiskit-addon-obp/
- **Cutting**: qiskit-addon-cutting v0.10.0, arXiv:2205.00016, https://qiskit.github.io/qiskit-addon-cutting/
- **PEC formal bounds**: Temme et al., arXiv:1612.02058
- **Pauli-Lindblad sparse models**: van den Berg et al., arXiv:2201.09866
- **PEA (nuestra primary)**: Kim et al., Nature 618 (2023)
- **Samplomatic framework**: IBM Qiskit ecosystem (2025)
- **NoiseLearnerV3**: qiskit-ibm-runtime (2025+)
