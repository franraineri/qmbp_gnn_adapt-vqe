# Future Work: Advanced Techniques for GNN-HVA Scaling

> Techniques identified as promising extensions beyond the scope of this thesis.
> Included for Chapter 7 (Future Work) discussion with literature references.
>
> **Date**: 2026-06-08

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
