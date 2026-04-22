# Hybrid GNN-HVA Framework for Topological Phase Characterization

## 🔬 Project Overview

This repository contains the codebase and theoretical framework for a Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project aims to accelerate Variational Quantum Eigensolvers (VQE) to characterize Quantum Topological Phases (like Quantum Spin Liquids) in frustrated many-body systems.

To overcome the limitations of Noisy Intermediate-Scale Quantum (NISQ) devices—specifically the "barren plateau" problem and noise-induced truncation—we abandon pure Quantum Neural Networks (QCNN) and standard deep VQEs. Instead, we propose a  **predictive hybrid architecture** : A classical Graph Neural Network (GNN) trained on Tensor Network data acts as an "Intelligent Warm-Start" to initialize a shallow, physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

## 📄 Theoretical Paradigm Shift (Critical Context for AI Agents)

**ATTENTION KIRO (AI Agent):** All code generation and architectural decisions MUST comply with the findings of the recent literature:

> *Mele, A. A., et al. "Noise-induced shallow circuits and the absence of barren plateaus" (Nature Physics, 2026).*

**Key takeaways from this paper dictating our codebase:**

1. **Depth Truncation:** Non-unital noise effectively truncates quantum circuits to logarithmic depth $\mathcal{O}(\log n)$. Deep circuits are classically simulable and lose quantum advantage. **Rule:** All our quantum circuits (HVAs) MUST be strictly shallow (e.g., $p=1$ or $p=2$ layers).
2. **Local Observables Only:** Global cost functions still suffer from barren plateaus under noise. **Rule:** We must extract and monitor *local observables* (e.g., local magnetization $\langle X_i \rangle$, local correlation $\langle Z_i Z_{i+1} \rangle$) to characterize phases, rather than relying on global state fidelity in the quantum hardware execution.
3. **Absence of Barren Plateaus for Local Costs:** By using shallow circuits and local observables, we guarantee stable gradients. Our GNN exploits this by providing the perfect starting seed, allowing instantaneous convergence before noise destroys the signal.

## 🗺️ The 4-Phase Roadmap

The project is strictly divided into four operational phases. Kiro must contextualize any task within this specific pipeline:

### PHASE 1: Classical Ground Truth Generation

* **Goal:** Solve parameterized Hamiltonians (e.g., 1D Transverse Field Ising Model, 2D Spin Ladders) classically.
* **Tools:** Exact Diagonalization (for PoC < 15 qubits), DMRG / TeNPy (for quasi-1D), NetKet (Neural Quantum States for 2D).
* **Output:** Dataset mapping Hamiltonian parameters (e.g., $h, J$) to exact ground state vectors and local observable expectation values.

### PHASE 2: Symmetry-Aware Ansatz & Compilation

* **Goal:** Translate the classical ground states into optimal parameters ($\theta_{opt}$) for a quantum circuit.
* **Architecture:** Use a  **Hamiltonian Variational Ansatz (HVA)** . Never use Hardware-Efficient Ansätze (HEA).
* **Constraint:** The HVA must be shallow ($p \le 2$).
* **Optimization Strategy:** Use  **Warm Start** . The optimized $\theta$ for Hamiltonian $H_i$ must be used as the initial guess for $H_{i+1}$ to ensure physical continuity and fast convergence.

### PHASE 3: Graph Neural Network (GNN) Predictive Model

* **Goal:** Train a classical model to predict $\theta_{opt}$ from the Hamiltonian graph.
* **Tools:** PyTorch (`torch.nn`).
* **PoC Approach:** For the 1D TFIM with uniform $J$, the graph structure is fixed and only $h$ varies. A simple MLP ($h \to \theta_{pred}$) suffices as the PoC predictor. Upgrade to full GNN when extending to non-uniform couplings or 2D lattices.
* **Training:** MSE loss on $\theta_{opt}$ with `ReduceLROnPlateau` scheduling. Physics validation callback every N epochs feeds $\theta_{pred}$ into `StatevectorEstimator` to verify predicted energies match exact diagonalization.
* **Generalization:** Always validate on at least one interpolation point (unseen $h$ value) to verify the model generalizes between training grid points.

### PHASE 4: Deployment & Restricted Adaptive Refinement

* **Goal:** Execute on real IBM Hardware (e.g., IBM Heron) using the trained GNN for inference.
* **Workflow:** Unseen Hamiltonian -> GNN predicts $\theta_{pred}$ -> Initialize HVA (Warm-Start) -> Execute VQE.
* **Adaptive Step:** If using `AdaptVQE`, strictly limit to `max_iterations=2` to prevent the circuit from growing into the noise-truncation regime. When the warm-start is near-optimal, AdaptVQE terminates at iteration 0 (all gradients below threshold) — this is the ideal outcome.
* **Phase Characterization:** Measure local observables ($\langle X_i \rangle$, $\langle Z_i Z_{i+1} \rangle$) to classify the quantum phase. Use the observable crossover from Phase 1 exact data as the finite-size critical point, not the thermodynamic limit $h_c = 1.0$.

## 💻 Tech Stack & Code Practices (Qiskit 2.x Standard)

**KIRO INSTRUCTIONS:** You must write code adhering to the **Qiskit 2.x ecosystem** (and modern 1.x). Deprecated Qiskit 0.4x syntax is strictly forbidden.

### Mandatory Coding Rules:

1. **Operators:** ALWAYS use `qiskit.quantum_info.SparsePauliOp` for building Hamiltonians and observables. NEVER use `PauliSumOp` or `opflow` (they are deprecated).
   * *Correct:* `SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=N)`
2. **Execution/Primitives:** ALWAYS use  **Qiskit Primitives V2** .
   * Use `qiskit.primitives.StatevectorEstimator` for exact local simulations (PoC).
   * Use `qiskit_ibm_runtime.EstimatorV2` for hardware execution.
   * *Never* use `qiskit.execute`, `Aer.get_backend()`, or Primitives V1.
3. **Algorithms:** Import algorithms from the standalone package `qiskit_algorithms`, NOT from `qiskit.algorithms` (deprecated).
4. **Data Binding:** Use `circuit.assign_parameters()` to bind predicted angles before passing them to the Estimator.

## 🚧 Contingencies & Scope

* If 2D Tensor Network simulations hit memory limits, fallback to quasi-1D cylindrical Spin Ladders.
* If hardware noise is too high even for shallow HVAs, target Symmetry-Protected Topological (SPT) phases (which require constant-depth circuits) instead of pure QSLs.

## 📊 Validation Metrics (Priority Order)

Metrics are ordered by physical relevance. The top metrics are what matter on real hardware; the bottom ones are noiseless-only diagnostics.

| Priority | Metric | What it tells you | Threshold | Hardware? |
|----------|--------|-------------------|-----------|-----------|
| **1** | **ΔE / gap** | Are we resolving the quantum phase? Energy error relative to the spectral gap determines if the pipeline can distinguish the ground state from the first excited state. | < 5% | ✅ |
| **2** | **⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩** | Phase characterization via local observables. These are the order parameters that classify ferromagnetic vs paramagnetic. The ⟨X⟩ = ⟨ZZ⟩ crossover defines the finite-size critical point. | error < 1e-2 | ✅ |
| **3** | **ΔE** | Absolute energy accuracy. Useful but less informative than ΔE/gap — a ΔE of 0.01 means nothing without knowing the gap scale. | < 1e-2 (aspirational for p=2) | ✅ |
| **4** | **Fidelity** | Full state overlap with exact ground state. Powerful for noiseless validation but **forbidden on hardware** (global cost → barren plateaus under noise per Mele et al.). | ≥ 99.5% (noiseless only) | ❌ |
| **5** | **ADAPT iterations** | Circuit depth compliance. Must stay ≤ 2 to respect the O(log n) noise-truncation bound. Termination at 0 iterations (AlgorithmError) is the ideal outcome. | ≤ 2 | ✅ |

**Key insight from PoC:** The ΔE threshold (1e-2) is aspirational — it is bounded by the HVA expressibility ceiling at each h value. At h=1.5 with p=2, the VQE itself achieves ΔE≈1.9e-2, so the MLP+AdaptVQE pipeline cannot beat this. The ΔE/gap metric (1.3% at h=1.5) correctly shows the pipeline resolves the physics despite the absolute ΔE exceeding 1e-2.

## 📚 Documentation

Detailed technical documentation is available in the [`documentation/`](documentation/) directory:

* **[Project Summary (English)](documentation/qmbp_doc_summary.md)** — Comprehensive overview of the physics problem, the classical bottleneck, the hybrid GNN-HVA solution, implementation techniques by phase, and full bibliography.
* **[Project Summary (Español)](documentation/qmbp_doc_summary_es.md)** — Resumen completo del proyecto en español: problema físico, solución híbrida, hoja de ruta operativa, técnicas de implementación y bibliografía.
* **[Architectural Document (ES/EN)](documentation/architectural_doc_es_en.md)** — Bilingual technical document covering the GNN data strategy, noise resilience justification, and the spin systems vs. quantum chemistry design rationale.
* **[Bibliography](documentation/bibliography.md)** — Complete APA-formatted reference list: foundational physics, NISQ theory, VQE algorithms, ML parameter prediction, tensor networks, and software frameworks.

## 🚀 Quick Start (PoC)

The current Proof of Concept (PoC V3.0) implements all 4 phases using the 1D Transverse Field Ising Model (TFIM) for $N=6$ qubits:

* **`poc_v3_phases1_2.ipynb`** — Phases 1-2: Exact diagonalization sweep + HVA warm-start VQE optimization (descending sweep h=2→0) over $h/J \in [0, 2]$.
* **`poc_v3_phases3_4.ipynb`** — Phases 3-4: MLP predictor training (with fidelity-filtered data and physics validation callback) + AdaptVQE deployment on unseen $h=1.5$ in the paramagnetic regime.

Data flows between notebooks via `phase1_phase2_tfim_N6_p2.npz`. Run notebook 1 first to generate the dataset.

---

# 🇪🇸 Versión en Español

# Arquitectura Híbrida GNN-HVA para Caracterización de Fases Topológicas

## 🔬 Descripción del Proyecto

Este repositorio contiene el código y el marco teórico para un Trabajo Final de Máster (TFM) en Computación Cuántica y Física de la Materia Condensada. El proyecto busca acelerar los Variational Quantum Eigensolvers (VQE) para caracterizar Fases Cuánticas Topológicas (como los Líquidos de Espín Cuánticos) en sistemas frustrados de muchos cuerpos.

Para superar las limitaciones de los dispositivos cuánticos ruidosos de escala intermedia (NISQ) — específicamente el problema de "barren plateaus" y la truncación inducida por ruido — abandonamos las Redes Neuronales Cuánticas (QCNN) puras y los VQE profundos estándar. En su lugar, proponemos una **arquitectura híbrida predictiva**: Una Red Neuronal de Grafos (GNN) clásica entrenada con datos de Redes Tensoriales actúa como un "Warm-Start Inteligente" para inicializar un circuito cuántico superficial e informado por la física (Hamiltonian Variational Ansatz - HVA).

## 📄 Cambio de Paradigma Teórico (Contexto Crítico para Agentes IA)

**ATENCIÓN KIRO (Agente IA):** Toda generación de código y decisión arquitectónica DEBE cumplir con los hallazgos de la literatura reciente:

> *Mele, A. A., et al. "Noise-induced shallow circuits and the absence of barren plateaus" (Nature Physics, 2026).*

**Conclusiones clave de este paper que dictan nuestro código:**

1. **Truncamiento de Profundidad:** El ruido no unital trunca efectivamente los circuitos cuánticos a profundidad logarítmica $\mathcal{O}(\log n)$. Los circuitos profundos son clásicamente simulables y pierden la ventaja cuántica. **Regla:** Todos nuestros circuitos cuánticos (HVAs) DEBEN ser estrictamente superficiales (ej: $p=1$ o $p=2$ capas).
2. **Solo Observables Locales:** Las funciones de costo globales aún sufren de barren plateaus bajo ruido. **Regla:** Debemos extraer y monitorear *observables locales* (ej: magnetización local $\langle X_i \rangle$, correlación local $\langle Z_i Z_{i+1} \rangle$) para caracterizar fases, en lugar de depender de la fidelidad global del estado en la ejecución en hardware cuántico.
3. **Ausencia de Barren Plateaus para Costos Locales:** Al usar circuitos superficiales y observables locales, garantizamos gradientes estables. Nuestra GNN explota esto proporcionando la semilla de inicio perfecta, permitiendo convergencia instantánea antes de que el ruido destruya la señal.

## 🗺️ Hoja de Ruta de 4 Fases

El proyecto está estrictamente dividido en cuatro fases operativas:

### FASE 1: Generación de Ground Truth Clásico

* **Objetivo:** Resolver Hamiltonianos parametrizados (ej: Modelo de Ising 1D con Campo Transverso, Escaleras de Espín 2D) clásicamente.
* **Herramientas:** Diagonalización Exacta (para PoC < 15 qubits), DMRG / TeNPy (para quasi-1D), NetKet (Estados Cuánticos Neuronales para 2D).
* **Salida:** Dataset que mapea parámetros del Hamiltoniano (ej: $h, J$) a vectores de estado fundamental exactos y valores esperados de observables locales.

### FASE 2: Ansatz con Simetría y Compilación

* **Objetivo:** Traducir los estados fundamentales clásicos en parámetros óptimos ($\theta_{opt}$) para un circuito cuántico.
* **Arquitectura:** Usar un **Hamiltonian Variational Ansatz (HVA)**. Nunca usar Hardware-Efficient Ansätze (HEA).
* **Restricción:** El HVA debe ser superficial ($p \le 2$).
* **Estrategia de Optimización:** Usar **Warm Start**. El $\theta$ optimizado para el Hamiltoniano $H_i$ debe usarse como estimación inicial para $H_{i+1}$ para asegurar continuidad física y convergencia rápida.

### FASE 3: Modelo Predictivo con Red Neuronal de Grafos (GNN)

* **Objetivo:** Entrenar un modelo clásico para predecir $\theta_{opt}$ a partir del grafo del Hamiltoniano.
* **Herramientas:** PyTorch (`torch.nn`).
* **Enfoque PoC:** Para el TFIM 1D con $J$ uniforme, la estructura del grafo es fija y solo varía $h$. Un MLP simple ($h \to \theta_{pred}$) es suficiente como predictor del PoC. Escalar a GNN completa al extender a acoplamientos no uniformes o redes 2D.
* **Entrenamiento:** Pérdida MSE sobre $\theta_{opt}$ con scheduling `ReduceLROnPlateau`. Callback de validación física cada N épocas alimenta $\theta_{pred}$ al `StatevectorEstimator` para verificar que las energías predichas coincidan con la diagonalización exacta.
* **Generalización:** Siempre validar en al menos un punto de interpolación (valor de $h$ no visto) para verificar que el modelo generaliza entre puntos de la grilla de entrenamiento.

### FASE 4: Despliegue y Refinamiento Adaptativo Restringido

* **Objetivo:** Ejecutar en hardware IBM real (ej: IBM Heron) usando la GNN entrenada para inferencia.
* **Flujo de trabajo:** Hamiltoniano no visto -> GNN predice $\theta_{pred}$ -> Inicializar HVA (Warm-Start) -> Ejecutar VQE.
* **Paso Adaptativo:** Si se usa `AdaptVQE`, limitar estrictamente a `max_iterations=2` para evitar que el circuito crezca hacia el régimen de truncación por ruido. Cuando el warm-start es casi óptimo, AdaptVQE termina en la iteración 0 (todos los gradientes bajo el umbral) — este es el resultado ideal.
* **Caracterización de Fase:** Medir observables locales ($\langle X_i \rangle$, $\langle Z_i Z_{i+1} \rangle$) para clasificar la fase cuántica. Usar el cruce de observables de los datos exactos de Fase 1 como punto crítico de tamaño finito, no el límite termodinámico $h_c = 1.0$.

## 💻 Stack Tecnológico y Prácticas de Código (Estándar Qiskit 2.x)

**INSTRUCCIONES KIRO:** El código debe adherirse al **ecosistema Qiskit 2.x** (y 1.x moderno). La sintaxis deprecada de Qiskit 0.4x está estrictamente prohibida.

### Reglas de Código Obligatorias:

1. **Operadores:** SIEMPRE usar `qiskit.quantum_info.SparsePauliOp` para construir Hamiltonianos y observables. NUNCA usar `PauliSumOp` u `opflow` (están deprecados).
   * *Correcto:* `SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=N)`
2. **Ejecución/Primitivas:** SIEMPRE usar **Qiskit Primitives V2**.
   * Usar `qiskit.primitives.StatevectorEstimator` para simulaciones locales exactas (PoC).
   * Usar `qiskit_ibm_runtime.EstimatorV2` para ejecución en hardware.
   * *Nunca* usar `qiskit.execute`, `Aer.get_backend()`, o Primitives V1.
3. **Algoritmos:** Importar algoritmos del paquete independiente `qiskit_algorithms`, NO de `qiskit.algorithms` (deprecado).
4. **Vinculación de Datos:** Usar `circuit.assign_parameters()` para vincular ángulos predichos antes de pasarlos al Estimator.

## 🚧 Contingencias y Alcance

* Si las simulaciones de Redes Tensoriales 2D alcanzan límites de memoria, recurrir a Escaleras de Espín cilíndricas quasi-1D.
* Si el ruido del hardware es demasiado alto incluso para HVAs superficiales, apuntar a fases Topológicas Protegidas por Simetría (SPT) (que requieren circuitos de profundidad constante) en lugar de QSLs puros.

## 📊 Métricas de Validación (Orden de Prioridad)

Las métricas están ordenadas por relevancia física. Las primeras son lo que importa en hardware real; las últimas son diagnósticos solo para simulación sin ruido.

| Prioridad | Métrica | Qué nos dice | Umbral | ¿Hardware? |
|-----------|---------|--------------|--------|------------|
| **1** | **ΔE / gap** | ¿Estamos resolviendo la fase cuántica? El error energético relativo al gap espectral determina si el pipeline puede distinguir el estado fundamental del primer estado excitado. | < 5% | ✅ |
| **2** | **⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩** | Caracterización de fase vía observables locales. Son los parámetros de orden que clasifican ferromagnético vs paramagnético. El cruce ⟨X⟩ = ⟨ZZ⟩ define el punto crítico de tamaño finito. | error < 1e-2 | ✅ |
| **3** | **ΔE** | Precisión energética absoluta. Útil pero menos informativo que ΔE/gap — un ΔE de 0.01 no significa nada sin conocer la escala del gap. | < 1e-2 (aspiracional para p=2) | ✅ |
| **4** | **Fidelidad** | Solapamiento total con el estado fundamental exacto. Potente para validación sin ruido pero **prohibido en hardware** (costo global → barren plateaus bajo ruido según Mele et al.). | ≥ 99.5% (solo noiseless) | ❌ |
| **5** | **Iteraciones ADAPT** | Cumplimiento de profundidad del circuito. Debe mantenerse ≤ 2 para respetar el límite de truncación por ruido O(log n). Terminación en 0 iteraciones (AlgorithmError) es el resultado ideal. | ≤ 2 | ✅ |

**Hallazgo clave del PoC:** El umbral ΔE (1e-2) es aspiracional — está acotado por el techo de expresibilidad del HVA en cada valor de h. A h=1.5 con p=2, el VQE mismo alcanza ΔE≈1.9e-2, por lo que el pipeline MLP+AdaptVQE no puede superar esto. La métrica ΔE/gap (1.3% a h=1.5) muestra correctamente que el pipeline resuelve la física a pesar de que el ΔE absoluto exceda 1e-2.

## 📚 Documentación

La documentación técnica detallada está disponible en el directorio [`documentation/`](documentation/):

* **[Resumen del Proyecto (English)](documentation/qmbp_doc_summary.md)** — Visión general completa del problema físico, el cuello de botella clásico, la solución híbrida GNN-HVA, técnicas de implementación por fase y bibliografía completa.
* **[Resumen del Proyecto (Español)](documentation/qmbp_doc_summary_es.md)** — Resumen completo del proyecto en español: problema físico, solución híbrida, hoja de ruta operativa, técnicas de implementación y bibliografía.
* **[Documento Arquitectónico (ES/EN)](documentation/architectural_doc_es_en.md)** — Documento técnico bilingüe que cubre la estrategia de datos de la GNN, la justificación de resiliencia al ruido y la fundamentación del diseño de sistemas de espines vs. química cuántica.
* **[Bibliografía](documentation/bibliography.md)** — Lista completa de referencias en formato APA: física fundamental, teoría NISQ, algoritmos VQE, predicción de parámetros con ML, redes tensoriales y frameworks de software.

## 🚀 Inicio Rápido (PoC)

La Prueba de Concepto actual (PoC V3.0) implementa las 4 fases usando el Modelo de Ising 1D con Campo Transverso (TFIM) para $N=6$ qubits:

* **`poc_v3_phases1_2.ipynb`** — Fases 1-2: Barrido de diagonalización exacta + optimización VQE con warm-start del HVA (barrido descendente h=2→0) sobre $h/J \in [0, 2]$.
* **`poc_v3_phases3_4.ipynb`** — Fases 3-4: Entrenamiento del predictor MLP (con datos filtrados por fidelidad y callback de validación física) + despliegue de AdaptVQE en $h=1.5$ no visto en el régimen paramagnético.

Los datos fluyen entre notebooks vía `phase1_phase2_tfim_N6_p2.npz`. Ejecutar el notebook 1 primero para generar el dataset.