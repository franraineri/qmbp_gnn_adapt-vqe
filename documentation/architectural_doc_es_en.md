# Documentación Técnica: Arquitectura Híbrida y Estrategia de Simulación

Esta documentación detalla las decisiones de diseño, la fundamentación física y la arquitectura del marco híbrido  **GNN-HVA** . El objetivo es proporcionar una base técnica sólida sobre la viabilidad del proyecto en hardware NISQ y las ventajas competitivas de utilizar inteligencia artificial clásica para guiar la computación cuántica.

## 1. Estrategia de Datos y Arquitectura de Aprendizaje (GNN)

La arquitectura se fundamenta en la premisa de que la estructura de un Hamiltoniano de muchos cuerpos es inherentemente un **grafo** $\mathcal{G} = (V, E)$. En esta representación, los qubits se identifican como nodos ($V$) y las interacciones físicas (como el intercambio $J$ o el campo magnético $h$) como aristas o atributos de nodo ($E$).

### 1.1 Justificación de la GNN (Message Passing Neural Networks)

A diferencia de técnicas de aprendizaje profundo convencionales o técnicas de  *Reservoir Computing* , las  **Redes Neuronales de Grafos (GNN)** , específicamente las de tipo *Message Passing* (MPNN), ofrecen ventajas críticas:

* **Invarianza y Covarianza:** Respetan las simetrías espaciales y la topología del sistema físico.
* **Generalización Extensible:** Una vez entrenado, el modelo puede inferir parámetros para sistemas con un número de qubits distinto al del entrenamiento, permitiendo escalar de sistemas pequeños a regímenes de utilidad cuántica (40-100 qubits) sin reentrenamiento masivo.
* **Mapeo de Parámetros:** La GNN realiza una regresión directa desde la topología del Hamiltoniano hacia los ángulos óptimos ($\theta$) del circuito variacional.

### 1.2 Generación de Ground Truth (Fase 1)

Para evitar el sesgo de datos ruidosos del hardware actual, la Fase 1 utiliza algoritmos clásicos de alta fidelidad:

* **DMRG (Density Matrix Renormalization Group):** Utilizado para sistemas cuasi-1D (cadenas y escaleras de espines) para obtener estados fundamentales exactos mediante Redes Tensoriales (MPS).
* **NQS (Neural Quantum States):** Utilizado para mallas 2D frustradas donde el "Problema del Signo" impide el uso de Monte Carlo tradicional.
* **Resultado:** Un dataset limpio que empareja Hamiltonianos con sus vectores de estado y observables locales ideales.

## 2. Resiliencia al Ruido y Erradicación de Barren Plateaus

El proyecto soluciona las barreras fundamentales de los algoritmos variacionales (VQE) mediante dos pilares teóricos de vanguardia (2021-2026):

### 2.1 Truncamiento de Profundidad inducido por Ruido

La investigación reciente ( *Mele et al., 2026* ) demuestra que el ruido no unital (relajación térmica y amortiguamiento de amplitud) actúa como un filtro que **trunca la profundidad efectiva** de los circuitos cuánticos a $O(\log n)$.

* **Estrategia:** En lugar de luchar contra el ruido con circuitos profundos inútiles, implementamos un **Ansatz Variacional Hamiltoniano (HVA)** estrictamente superficial ($p \le 2$).
* **Ventaja:** Al ser superficial, el circuito mantiene la coherencia necesaria para capturar la física del sistema antes de que el ruido degrade el estado hacia un punto fijo trivial.

### 2.2 Observables Locales vs. Barren Plateaus

Los gradientes de las funciones de costo globales (como la fidelidad total del estado) desaparecen exponencialmente con el número de qubits, imposibilitando el entrenamiento en hardware real.

* **Solución:** Nuestra arquitectura utiliza **observables locales** (correlaciones de primeros vecinos $\langle Z_i Z_{i+1} \rangle$). Se ha demostrado que las funciones de costo locales en circuitos superficiales bajo ruido no unital  **no sufren de barren plateaus** , garantizando que el optimizador siempre encuentre una dirección de descenso válida.


## 3. Justificación: Sistemas de Espines vs. Química Cuántica

Una decisión estratégica es la orientación hacia la **Física de la Materia Condensada** (sistemas de espines) para maximizar la eficiencia del hardware NISQ.

### 3.1 Mapeo Isomórfico (Eficiencia de Compuertas)

* **Espines:** Existe un isomorfismo natural entre un espín-1/2 y un qubit. Las interacciones $Z_i Z_j$ del Hamiltoniano se traducen directamente en compuertas nativas $R_{ZZ}$. Esto permite que el HVA sea extremadamente compacto.
* **Moléculas (Química):** Los electrones son fermiones. Su simulación requiere transformaciones como  **Jordan-Wigner** , que mapean operadores locales a cadenas de Pauli largas y no locales. Esto aumenta drásticamente la profundidad del circuito y la tasa de error, haciendo inviable el uso de circuitos superficiales.

### 3.2 Evasión del "Active Space" y Precisión

La química cuántica requiere una "precisión química" de $10^{-3}$ Hartree y una selección manual compleja de orbitales ( *Active Spaces* ). Los modelos de espín permiten centrarse en el descubrimiento de  **fases topológicas y fenómenos colectivos** , que son objetivos más robustos y significativos para demostrar la utilidad cuántica en dispositivos ruidosos.


# Technical Documentation: Hybrid Architecture and Simulation Strategy (English Translation)

This documentation provides an expanded technical overview of the **GNN-HVA** hybrid framework.

## 1. Data Strategy & Learning Architecture (GNN)

The architecture treats a many-body Hamiltonian as a **graph** $\mathcal{G} = (V, E)$, where qubits are nodes and physical interactions are edges or node attributes.

* **MPNN Justification:** Message Passing Neural Networks respect spatial symmetries and physical topology, allowing the model to generalize control parameters ($\theta$) to larger systems without retraining.
* **Ground Truth Generation:** Phase 1 utilizes DMRG (Matrix Product States) for quasi-1D systems and NQS (Neural Quantum States) for 2D frustrated lattices to ensure the network learns exact physics before facing real-world noise.

## 2. Noise Resilience & Barren Plateau Eradication

We address VQE barriers through two key theoretical pillars (2021-2026):

* **Noise-Induced Truncation:** Recent research ( *Mele et al., 2026* ) shows that non-unital noise effectively truncates circuit depth to $O(\log n)$. Our strategy uses a strictly shallow **Hamiltonian Variational Ansatz (HVA)** ($p \le 2$) to maintain coherence.
* **Local Observables:** Unlike global cost functions, local observables (nearest-neighbor correlations) in shallow noisy circuits  **do not exhibit barren plateaus** , ensuring stable and trainable gradients.

## 3. Justification: Spin Systems vs. Quantum Chemistry

We focus on **Condensed Matter Physics** (spin systems) to maximize hardware efficiency.

### 3.1 Isomorphic Mapping

Spin-1/2 systems map naturally to qubits. Hamiltonian $Z_i Z_j$ terms translate directly into native $R_{ZZ}$ gates, keeping the HVA compact. Conversely, chemistry requires **Jordan-Wigner** transforms, which turn local interactions into long, non-local Pauli strings, drastically increasing circuit depth and error rates.

### 3.2 Avoiding "Active Space"

Quantum chemistry demands extreme precision and complex orbital selection. Spin models allow us to focus on  **topological phases and collective phenomena** , which are more robust targets for NISQ hardware utility.

---

> **Full bibliography / Bibliografía completa:** All references cited in this document are consolidated in [documentation/bibliography.md](bibliography.md).


---

# QPU Execution Analysis: Simulator vs. Real Hardware

## 4. Classical Simulation vs. Quantum Hardware Deployment (English)

### 4.1 What Runs Where

The pipeline is designed so that **only Phase 4 touches quantum hardware**. Everything else is classical:

| Phase | Execution | Why |
|-------|-----------|-----|
| Phase 1: Ground Truth | Classical CPU (exact diag / DMRG) | Generates noise-free reference data |
| Phase 2: VQE Sweep | Classical CPU (StatevectorEstimator) | Needs hundreds of iterations — too expensive on QPU |
| Phase 3: MPNN Training | Classical CPU (PyTorch) | Pure machine learning, no quantum |
| Phase 4: Deployment | **QPU** (EstimatorV2 + ZNE) | Only 0–2 AdaptVQE iterations needed thanks to warm-start |

The key insight: Phases 1–3 prepare a near-optimal parameter prediction (θ_pred) so that Phase 4 needs minimal quantum resources — just 0–2 circuit evaluations on hardware.

### 4.2 QPU Time Estimates

For a single test point deployment on IBM Torino (133 qubits, Eagle r3):

| System | Circuit gates (native) | Shots | Observables | Est. QPU time | Est. wall time |
|--------|----------------------|-------|-------------|---------------|----------------|
| N=6, p=2 | ~30 gates | 4096 | 11 (6 X + 5 ZZ) | ~10s | 2–5 min (queue) |
| N=10, p=2 | ~50 gates | 4096 | 19 (10 X + 9 ZZ) | ~15s | 2–5 min (queue) |
| N=20, p=2 | ~100 gates | 8192 | 39 | ~30s | 3–10 min (queue) |

Wall time is dominated by job queue, not circuit execution. The actual QPU time per circuit is microseconds — the overhead is classical communication, compilation, and scheduling.

### 4.3 Would Real QPU Improve Results?

**No — real hardware makes results worse, not better.** The simulator provides the upper bound on pipeline quality. Hardware adds three sources of degradation:

1. **Gate errors** (~0.1–0.5% per two-qubit gate). For N=6 with ~12 RZZ gates, total error accumulation is ~1–6%. This directly reduces fidelity from our simulated 0.992 to approximately 0.95–0.98.

2. **Shot noise** (statistical uncertainty from finite measurements). With 4096 shots, each observable has uncertainty ~1/√4096 ≈ 0.016. This is comparable to our ⟨X⟩ error (8.4e-03 at N=10) — shot noise would dominate the signal.

3. **Readout errors** (~1–3% per qubit). Mitigated by readout error correction (TREX), but adds systematic bias.

### 4.4 What QPU Execution Demonstrates

Despite worse numerical results, hardware execution is essential for the thesis because it proves:

1. **The Mele et al. principle works in practice:** shallow HVA (p≤2) survives noise while deeper circuits would not.
2. **The warm-start advantage is real:** AdaptVQE converges in 0–2 iterations on hardware (vs 10+ without warm-start), saving precious coherence time.
3. **Local observables are measurable on hardware:** ⟨X_i⟩ and ⟨Z_iZ_{i+1}⟩ can be extracted with reasonable shot budgets, unlike global fidelity which requires exponential tomography.
4. **The pipeline is end-to-end functional:** classical training → quantum deployment → phase classification.

### 4.5 Error Mitigation Strategy

To bridge the gap between simulator and hardware:

- **ZNE (Zero Noise Extrapolation):** Run the circuit at noise levels 1x, 2x, 3x (by inserting identity-equivalent gate pairs), then extrapolate to 0x noise. Adds 3x overhead but significantly reduces systematic bias.
- **TREX (Twirled Readout Error eXtinction):** Randomizes readout errors to make them symmetric, then corrects statistically.
- **Dynamical Decoupling:** Inserts identity pulses during idle periods to suppress decoherence. Free (no extra shots).

### 4.6 The Correct Thesis Narrative

1. **Noiseless simulation** (what we've done) establishes the pipeline's theoretical ceiling.
2. **Hardware execution** shows realistic performance with noise + mitigation.
3. **The gap between them** quantifies the noise impact and validates the shallow-circuit strategy.
4. **Success criterion on hardware:** ΔE/gap < 5% and correct phase classification — NOT fidelity ≥ 99.5% (which is a noiseless-only metric).

---

## 5. Pipeline Techniques and Parameters — Complete Reference (English)

This section explains every technique and parameter used in the pipeline, designed for readers who want to understand what each component does and why.

### 5.1 The Transverse Field Ising Model (TFIM)

The physical system we study:

$$H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i$$

- **J** (coupling constant): strength of spin-spin interaction. We use J=1.0 (uniform).
- **h** (transverse field): external magnetic field that competes with spin alignment.
- **Phase transition at h/J ≈ 1.0:** below this, spins align (ferromagnetic); above, they point along the field (paramagnetic).
- **Why this model:** simplest system exhibiting a quantum phase transition, maps directly to qubits without Jordan-Wigner overhead.

### 5.2 Hamiltonian Variational Ansatz (HVA)

The quantum circuit that approximates the ground state:

$$|\psi(\theta)\rangle = \prod_{l=1}^{p} e^{-i\theta_{l,x} H_X} \cdot e^{-i\theta_{l,zz} H_{ZZ}} |+\rangle^{\otimes N}$$

- **p_layers** (circuit depth): number of alternating ZZ/X layers. Fixed at p=2 (Mele et al. constraint).
- **Initial state |+⟩^N:** all qubits in superposition. This is the exact ground state at h→∞, making it the natural starting point for a descending sweep.
- **2θ scaling:** gates use `RZZ(2θ)` and `RX(2θ)` to match the physical Hamiltonian evolution operator.
- **Why HVA over HEA:** HVA respects the Hamiltonian's symmetry structure, giving it better expressibility per parameter than generic hardware-efficient ansätze.

### 5.3 VQE (Variational Quantum Eigensolver) Parameters

The classical optimizer that finds optimal circuit parameters:

- **n_restarts** (default 5): number of random perturbations around the warm-start. Each restart explores a different region of the energy landscape. More restarts = better chance of finding the global minimum, but with diminishing returns beyond 5.
- **restart_sigma** (default 0.1): standard deviation of the random perturbation. Controls how far each restart explores from the current best. σ=0.1 stays close (local refinement); σ=0.2 explores more broadly but risks instability.
- **maxiter** (default 1000): maximum L-BFGS-B iterations per optimization run. The optimizer typically converges in 10–50 iterations for well-initialized problems.
- **ftol** (default 1e-14): convergence tolerance. The optimizer stops when the energy change between iterations is smaller than this.
- **bounds** ([-π, π]): parameter search range. Expanded from the original ±0.1 to allow the optimizer to explore the full physical parameter space.
- **Descending sweep (h=2→0):** we optimize from high h (where |+⟩^N is nearly exact) to low h, propagating each θ_opt as the initial guess for the next point. This warm-start propagation produces smooth θ landscapes that the MPNN can learn.

### 5.4 MPNN (Message Passing Neural Network) Parameters

The graph neural network that predicts optimal parameters:

- **hidden_dim** (64 for N=6, 128 for N=10): width of the message passing layers. Must scale with graph complexity — too small underfits, too large overfits on limited training data.
- **n_layers** (3): depth of message passing. Each layer aggregates information from one hop of neighbors. 3 layers means each node "sees" up to 3 hops away — sufficient for a 6–10 node chain.
- **GINConv** (Graph Isomorphism Network): the specific message passing operator. Treats all neighbors equally (isotropic). Optimal for uniform lattices where all edges have the same coupling J.
- **global_mean_pool:** after message passing, averages all node embeddings into a single fixed-size vector. This makes the model lattice-agnostic — it can accept graphs of any size.
- **n_epochs** (6000): training iterations. The model converges by ~4000 epochs; extra epochs provide stability.
- **lr** (1e-3): Adam optimizer learning rate. Lower (5e-4) causes instability; higher (3e-3) causes divergence.
- **patience** (150): epochs without improvement before the learning rate is halved. Prevents oscillation in late training.
- **fidelity_threshold** (0.93): minimum VQE fidelity to include a training point. Points below this have unreliable θ_opt (the VQE didn't find the true ground state).

### 5.5 AdaptVQE Parameters

The hardware deployment algorithm:

- **max_iterations** (2): maximum circuit layers AdaptVQE can add. Enforces the Mele et al. depth constraint.
- **gradient_threshold** (1e-3): if all Pauli pool gradients are below this, AdaptVQE stops (declares convergence). When this happens at iteration 0, it means the warm-start was already optimal — the ideal outcome.
- **Pauli pool:** the set of operators AdaptVQE can add to the circuit. We use the Hamiltonian's own terms (ZZ bonds + X sites) — physically motivated and local.

### 5.6 QRC (Quantum Reservoir Computing) Parameters

The fallback deployment route:

- **Fixed reservoir:** an HVA circuit with random parameters that are NEVER optimized. This eliminates barren plateaus by construction — no quantum training loop exists.
- **Rx(h) encoding:** the transverse field value is encoded into the reservoir by applying Rx(h) rotations on each qubit. This creates h-dependent quantum states without optimization.
- **Linear readout:** a classical linear regression maps reservoir output features (local observable measurements) to predicted phase observables. Trained on exact Phase 1 data.
- **R² ≈ 0.97:** the linear readout achieves excellent fit on 27 training points, demonstrating that the reservoir creates sufficiently rich feature representations.

### 5.7 Validation Metrics (in priority order)

1. **ΔE/gap < 5%:** "Does the pipeline resolve the physics?" — measures whether the predicted energy is close enough to the ground state to distinguish it from the first excited state. This is the primary metric because it directly answers whether phase classification is possible.

2. **⟨X⟩, ⟨ZZ⟩ errors < 1e-2:** "Can we characterize the phase from observables?" — these are the quantities measured on hardware. If they're accurate, we can classify ferromagnetic vs paramagnetic.

3. **ΔE < 1e-2:** "Is the absolute energy accurate?" — aspirational metric bounded by HVA expressibility. Never passes at h=1.25 because the circuit fundamentally cannot represent the ground state to this precision with p=2.

4. **Fidelity ≥ 99.5%:** "Is the quantum state correct?" — only meaningful in noiseless simulation. On hardware, fidelity cannot be measured without exponential-cost tomography.

5. **ADAPT iterations ≤ 2:** "Does the circuit stay shallow?" — enforces the Mele et al. constraint. Always passes because we cap max_iterations=2.

---

## 4. Análisis de Ejecución en QPU: Simulador vs. Hardware Real (Español)

### 4.1 Qué se ejecuta dónde

El pipeline está diseñado para que **solo la Fase 4 toque hardware cuántico**. Todo lo demás es clásico:

| Fase | Ejecución | Por qué |
|------|-----------|---------|
| Fase 1: Ground Truth | CPU clásica (diag. exacta / DMRG) | Genera datos de referencia sin ruido |
| Fase 2: Barrido VQE | CPU clásica (StatevectorEstimator) | Necesita cientos de iteraciones — demasiado costoso en QPU |
| Fase 3: Entrenamiento MPNN | CPU clásica (PyTorch) | Machine learning puro, sin cuántica |
| Fase 4: Despliegue | **QPU** (EstimatorV2 + ZNE) | Solo 0–2 iteraciones de AdaptVQE gracias al warm-start |

### 4.2 Estimación de tiempo en QPU

Para un punto de test en IBM Torino (133 qubits, Eagle r3):

| Sistema | Compuertas nativas | Shots | Observables | Tiempo QPU | Tiempo total |
|---------|-------------------|-------|-------------|------------|--------------|
| N=6, p=2 | ~30 | 4096 | 11 | ~10s | 2–5 min (cola) |
| N=10, p=2 | ~50 | 4096 | 19 | ~15s | 2–5 min (cola) |
| N=20, p=2 | ~100 | 8192 | 39 | ~30s | 3–10 min (cola) |

### 4.3 ¿Mejoraría el QPU real los resultados?

**No — el hardware real empeora los resultados.** El simulador proporciona el límite superior de calidad. El hardware añade:

1. **Errores de compuerta** (~0.1–0.5% por compuerta de dos qubits). Para N=6 con ~12 RZZ, la acumulación total es ~1–6%.
2. **Ruido de muestreo** (incertidumbre estadística). Con 4096 shots, cada observable tiene error ~1/√4096 ≈ 0.016.
3. **Errores de lectura** (~1–3% por qubit). Mitigados con TREX.

### 4.4 Qué demuestra la ejecución en QPU

A pesar de peores números, la ejecución en hardware es esencial para la tesis:

1. **El principio de Mele et al. funciona en la práctica:** HVA superficial (p≤2) sobrevive al ruido.
2. **La ventaja del warm-start es real:** AdaptVQE converge en 0–2 iteraciones en hardware.
3. **Los observables locales son medibles:** ⟨X_i⟩ y ⟨Z_iZ_{i+1}⟩ se extraen con presupuestos razonables de shots.
4. **El pipeline es funcional de extremo a extremo:** entrenamiento clásico → despliegue cuántico → clasificación de fase.

### 4.5 Estrategia de mitigación de errores

- **ZNE (Zero Noise Extrapolation):** Ejecutar el circuito a niveles de ruido 1x, 2x, 3x y extrapolar a 0x. Añade 3x de overhead pero reduce significativamente el sesgo sistemático.
- **TREX (Twirled Readout Error eXtinction):** Aleatoriza errores de lectura para hacerlos simétricos y corregirlos estadísticamente.
- **Dynamical Decoupling:** Inserta pulsos de identidad durante periodos ociosos para suprimir la decoherencia. Sin costo adicional.

---

## 5. Técnicas y Parámetros del Pipeline — Referencia Completa (Español)

### 5.1 El Modelo de Ising en Campo Transversal (TFIM)

$$H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i$$

- **J** (constante de acoplamiento): intensidad de la interacción espín-espín. Usamos J=1.0 (uniforme).
- **h** (campo transversal): campo magnético externo que compite con el alineamiento de espines.
- **Transición de fase en h/J ≈ 1.0:** por debajo, los espines se alinean (ferromagnético); por encima, apuntan al campo (paramagnético).

### 5.2 Hamiltonian Variational Ansatz (HVA)

El circuito cuántico que aproxima el estado fundamental:

- **p_layers** (profundidad): número de capas alternantes ZZ/X. Fijado en p=2 (restricción de Mele et al.).
- **Estado inicial |+⟩^N:** todos los qubits en superposición. Es el estado fundamental exacto cuando h→∞.
- **Escalado 2θ:** las compuertas usan `RZZ(2θ)` y `RX(2θ)` para corresponder con el operador de evolución del Hamiltoniano.
- **Por qué HVA y no HEA:** el HVA respeta la estructura de simetría del Hamiltoniano, dando mejor expresibilidad por parámetro.

### 5.3 Parámetros del VQE (Variational Quantum Eigensolver)

- **n_restarts** (5): perturbaciones aleatorias alrededor del warm-start. Más restarts = mejor probabilidad de encontrar el mínimo global.
- **restart_sigma** (0.1): desviación estándar de la perturbación. Controla cuán lejos explora cada restart.
- **maxiter** (1000): iteraciones máximas de L-BFGS-B por optimización.
- **ftol** (1e-14): tolerancia de convergencia.
- **bounds** ([-π, π]): rango de búsqueda de parámetros.
- **Barrido descendente (h=2→0):** optimizamos desde h alto (donde |+⟩^N es casi exacto) hacia h bajo, propagando cada θ_opt como guess inicial para el siguiente punto. Esta propagación warm-start produce paisajes suaves de θ que la MPNN puede aprender.

### 5.4 Parámetros de la MPNN (Message Passing Neural Network)

- **hidden_dim** (64 para N=6, 128 para N=10): ancho de las capas de message passing. Debe escalar con la complejidad del grafo.
- **n_layers** (3): profundidad del message passing. Cada capa agrega información de un salto de vecinos.
- **GINConv** (Graph Isomorphism Network): operador de message passing isotrópico. Óptimo para redes uniformes.
- **global_mean_pool:** promedia todos los embeddings de nodos en un vector de tamaño fijo. Hace al modelo agnóstico a la topología.
- **n_epochs** (6000): iteraciones de entrenamiento.
- **lr** (1e-3): tasa de aprendizaje del optimizador Adam.
- **patience** (150): épocas sin mejora antes de reducir el learning rate a la mitad.
- **fidelity_threshold** (0.93): fidelidad mínima del VQE para incluir un punto de entrenamiento.

### 5.5 Parámetros de AdaptVQE

- **max_iterations** (2): capas máximas que AdaptVQE puede añadir. Impone la restricción de Mele et al.
- **gradient_threshold** (1e-3): si todos los gradientes del Pauli pool están por debajo, AdaptVQE se detiene. Cuando esto ocurre en iteración 0, significa que el warm-start ya era óptimo — el resultado ideal.
- **Pauli pool:** operadores que AdaptVQE puede añadir al circuito. Usamos los términos del propio Hamiltoniano (enlaces ZZ + sitios X).

### 5.6 Parámetros del QRC (Quantum Reservoir Computing)

- **Reservorio fijo:** circuito HVA con parámetros aleatorios que NUNCA se optimizan. Elimina barren plateaus por construcción.
- **Codificación Rx(h):** el campo transversal se codifica aplicando rotaciones Rx(h) en cada qubit.
- **Readout lineal:** regresión lineal clásica que mapea features del reservorio a observables predichos.
- **R² ≈ 0.97:** el readout lineal logra excelente ajuste con 27 puntos de entrenamiento.

### 5.7 Métricas de Validación (en orden de prioridad)

1. **ΔE/gap < 5%:** "¿Resuelve el pipeline la física?" — mide si la energía predicha está suficientemente cerca del estado fundamental para distinguirlo del primer estado excitado.
2. **Errores ⟨X⟩, ⟨ZZ⟩ < 1e-2:** "¿Podemos caracterizar la fase desde observables?" — son las cantidades medidas en hardware.
3. **ΔE < 1e-2:** "¿Es precisa la energía absoluta?" — métrica aspiracional limitada por la expresibilidad del HVA.
4. **Fidelidad ≥ 99.5%:** "¿Es correcto el estado cuántico?" — solo significativa en simulación sin ruido.
5. **Iteraciones ADAPT ≤ 2:** "¿Se mantiene superficial el circuito?" — impone la restricción de Mele et al.

---

> **Full bibliography / Bibliografía completa:** [documentation/bibliography.md](bibliography.md)
