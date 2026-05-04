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
