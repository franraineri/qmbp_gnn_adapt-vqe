# 07 — Divide & Conquer: Descomposición de Topologías de Lattice

**Fecha**: 2026-07-27
**Estado**: Investigación inicial
**Objetivo**: Evaluar técnicas de divide-and-conquer para descomponer problemas de N partículas en subproblemas más pequeños, reutilizando la periodicidad de las topologías.

---

## Motivación

Las topologías de nuestro proyecto (chain_1d, ladder, square, triangular, heavy_hex, kagome) son estructuras periódicas con patrones repetitivos. Si podemos:

1. **Dividir** el lattice de N sitios en fragmentos más pequeños
2. **Resolver** cada fragmento de forma independiente (o casi)
3. **Rearmar** la solución global a partir de las parciales

...entonces podemos escalar a sistemas mucho más grandes sin necesidad de más qubits, y aprovechar la MPNN como predictor de parámetros para cada fragmento.

---

## Preguntas Fundamentales

| # | Pregunta | Status |
|---|----------|--------|
| 0 | ¿Se puede rearmar la solución? | ✅ Sí, con costo proporcional al entanglement en la frontera |
| 1 | ¿Cómo se comporta la materia al dividir? | Depende del régimen (ver §1) |
| 2 | ¿Qué estudios existen? | 7 familias de técnicas identificadas (ver §2) |
| 3 | ¿Cómo dividir nuestras topologías? | Estrategia por topología (ver §3) |

---

## §1 — Comportamiento Físico al Fragmentar

### Regímenes y viabilidad

| Régimen | Entanglement cruzando frontera | Viabilidad D&C | Relevancia para nosotros |
|---------|-------------------------------|----------------|--------------------------|
| Paramagnético (h >> h_c) | Exponencialmente bajo | Excelente | TFIM h>1 |
| Crítico (h ≈ h_c) | Log(L) en 1D, algebraico en 2D | Difícil | Punto de transición h_c≈1 |
| Ordenado (h << h_c) | Area-law pero GSD | Bueno con simetría | TFIM h<1 |

### Resultado clave: Area Law

Para estados fundamentales de Hamiltonianos con gap (gapped), el entanglement entropy S(A) entre un fragmento A y el resto escala con el **área** de la frontera, no con el volumen:

- **1D**: S(A) = const (independiente de L) → D&C funciona muy bien
- **1D crítico**: S(A) ~ (c/3)·log(L) donde c es la carga central CFT
- **2D**: S(A) ~ L_boundary (perímetro del corte)

Esto implica que la dificultad del reensamblado es proporcional al número de bonds cortados.

### Para nuestras topologías

| Topología | Bonds cortados (por partición) | Dificultad de reensamblado |
|-----------|-------------------------------|---------------------------|
| chain_1d | 1 | Mínima |
| ladder | 2 | Baja |
| square (strip) | √N | Moderada |
| triangular (strip) | ~1.5×√N | Moderada-alta |
| heavy_hex (celda) | 3 | Baja (estructura natural IBM) |
| kagome | 6 (por estrella) | Alta |

---

## §2 — Técnicas Existentes (Estado del Arte)

### 2.1 Deep VQE (Fujii et al., 2020)

**Concepto**: Divide el lattice en subsistemas, resuelve cada uno con VQE, construye un Hamiltoniano efectivo en base reducida para las interacciones inter-fragmento, y resuelve el Hamiltoniano efectivo con otro VQE.

**Relevancia para nosotros**: ★★★★★ — Directamente aplicable a nuestro pipeline HVA+MPNN.

**Papers**:
- Fujii et al., "Deep VQE: a divide-and-conquer method", [arXiv:2007.10917](https://arxiv.org/abs/2007.10917) (2020, publicado 2022)
- Mizukami et al., "Deep VQE for excited states and periodic materials", [arXiv:2104.00855](https://arxiv.org/abs/2104.00855) (2021)
- Yoshioka et al., "Constructing Local Bases for Deep VQE", [arXiv:2202.08473](https://arxiv.org/abs/2202.08473) (2022)

**Repositorios**:
- ⚠️ No se encontró implementación pública oficial del grupo de Fujii
- Implementación propia necesaria, basada en nuestro `VQEOptimizer` + `HVACircuitBuilder`

**Algoritmo**:
1. Dividir lattice en subsistemas {A₁, A₂, ...}
2. Para cada Aᵢ: resolver VQE → obtener estados base {|ψ_k⟩}
3. Proyectar H_inter (interacciones entre subsistemas) en la base reducida
4. Resolver H_eff con VQE en espacio reducido

**Integración con nuestro proyecto**:
- Usar MPNN para predecir θ_init de cada subsistema (warm-start fragmentos)
- Extender `PipelineRunner` con un modo `deep_vqe=True`
- Los clusters de NLCE (`NLCERunner`) ya implementan la infraestructura de cluster solving

---

### 2.2 NLCE — Numerical Linked-Cluster Expansion (ya implementado parcialmente)

**Concepto**: Descompone propiedades del bulk como sumas sobre clusters finitos con sustracción de Euler (inclusion-exclusion).

**Relevancia para nosotros**: ★★★★★ — Ya tenemos `NLCERunner` para 1D.

**Papers**:
- Rigol et al., "Numerical Linked-Cluster Algorithms. I. Spin systems on square, triangular, and kagomé lattices", [arXiv:0706.3254](https://arxiv.org/abs/0706.3254) (2007)
- Rigol, "A Short Introduction to NLCEs", [arXiv:1207.3366](https://arxiv.org/abs/1207.3366) (2012)
- Tang et al., "NLCEs for 2D spin models with continuous disorder", [arXiv:2402.00931](https://arxiv.org/abs/2402.00931) (2024, publicado Junio 2025)

**Repositorios**:
- No hay repos públicos genéricos standalone; las implementaciones suelen estar en código de grupo
- Nuestro código: `src/qmbp_simulation/analysis/nlce.py` (1D chains con OBC)

**Extensión necesaria para 2D**:
- Enumerar clusters embebibles en square/triangular/kagome
- Calcular multiplicidades de embedding (lattice constants)
- Generalizar la sustracción de Euler para grafos 2D

---

### 2.3 DMET — Density Matrix Embedding Theory

**Concepto**: Embeds un fragmento en un "bath" (baño) construido a partir de la solución mean-field del sistema completo. El fragmento + bath se resuelve exactamente, y se itera hasta autoconsistencia.

**Relevancia para nosotros**: ★★★★☆ — Muy potente para 2D, requiere implementación sustancial.

**Papers**:
- Knizia & Chan, "Density Matrix Embedding Theory", PRL 109, 186404 (2012)
- Wouters et al., "Block Product DMET for spin systems (J1-J2 square lattice)", [arXiv:1702.04285](https://arxiv.org/pdf/1702.04285v2.pdf) (2017)
- Nusspickel & Booth, "New perspectives on DMET", [arXiv:2503.09881](https://arxiv.org/html/2503.09881v1) (2025)
- Oinam et al., "Multireference Embedding and Fragmentation", [arXiv:2505.13394](https://arxiv.org/abs/2505.13394) (Mayo 2025)
- Van Voorhis et al., "Localized Orbital-Based Embedding from Exact DFT", [arXiv:2507.19591](https://arxiv.org/html/2507.19591v1) (Julio 2025)

**Repositorios**:
- **Vayesta** (BoothGroup): https://github.com/BoothGroup/Vayesta — Python, DMET para moléculas, sólidos, y lattice models. Activamente mantenido.
- **QC-DMET** (Wouters): https://github.com/SebWouters/QC-DMET — Python, DMET para quantum chemistry.
- **libdmet_preview** (Cui/Chan): https://github.com/gkclab/libdmet_preview — DMET para lattice models y sólidos periódicos.
- **pDMET** (Pham): https://github.com/hungpham2017/pDMET — DMET para sistemas periódicos.
- **PyDMET** (Wouters): https://github.com/SebWouters/PyDMET — Implementación más simple.

**Integración con nuestro proyecto**:
- Usar Vayesta como backend DMET para validación de ground truth en 2D
- Adaptar el solver de fragmentos para usar nuestro VQE+HVA
- Comparar E_DMET vs E_exact como validation del approach

---

### 2.4 Circuit Cutting / Wire Cutting (Qiskit Addon)

**Concepto**: Corta el circuito cuántico (no el Hamiltoniano) en sub-circuitos más pequeños que se ejecutan independientemente. Reconstruye el resultado via postprocesamiento clásico con overhead de sampling.

**Relevancia para nosotros**: ★★★★☆ — Directamente usable para hardware deployment en IBM Torino.

**Papers**:
- Peng et al., "CutQC: Using Small Quantum Computers for Large Problems", ASPLOS 2020
- Lowe et al., "Cutting circuits with multiple two-qubit unitaries", [arXiv:2312.11638](https://arxiv.org/abs/2312.11638) (2024)
- Bechtold et al., "Wire cutting with Non-Maximally Entangled States", [arXiv:2403.09690](https://arxiv.org/html/2403.09690v2) (2024)
- Brandhofer et al., "Distributed Circuit Cutting for HPC", [arXiv:2505.01184](https://arxiv.org/html/2505.01184v2) (Mayo 2025)
- Schmitt et al., "Improved sampling bounds beyond bipartitions", [arXiv:2506.18031](https://arxiv.org/pdf/2506.18031v2) (Junio 2025)
- Beyer et al., "Bridging wire and gate cutting with ZX-calculus", [arXiv:2503.11494](https://arxiv.org/abs/2503.11494v3) (2025)

**Repositorios**:
- **qiskit-addon-cutting** (IBM oficial): https://github.com/Qiskit/qiskit-addon-cutting — Wire cutting + gate cutting con Qiskit. Activamente mantenido.
- **CutQC** (Tang/MIT): https://github.com/weiT1993/CutQC — Implementación original del paper CutQC.
- **QCut**: https://github.com/JooNiv/QCut — Wire cuts y gate cuts sobre Qiskit, resetless.
- Documentación IBM: https://qiskit.github.io/qiskit-addon-cutting/

**Integración con nuestro proyecto**:
- Instalar `qiskit-addon-cutting` y usarlo con nuestros HVA circuits
- Particionar circuitos HVA N=20+ en sub-circuitos de ~10 qubits
- Costo: overhead exponencial en número de cortes → minimizar cortes
- Ideal para heavy_hex donde la estructura natural permite pocos cortes

---

### 2.5 Entanglement Forging (IBM, 2021-2024)

**Concepto**: Descompone el estado cuántico via Schmidt decomposition. Simula cada mitad con N/2 qubits y recombina clásicamente. Duplica el tamaño simulable.

**Relevancia para nosotros**: ★★★☆☆ — Requiere bipartición natural, bueno para ladder.

**Papers**:
- Eddins et al., "Doubling the size of quantum simulators by entanglement forging", Nature 2022
- Huembeli et al., "Entanglement Forging with generative neural network models", [arXiv:2205.00933](https://arxiv.org/abs/2205.00933) (2022)
- Haug & Kim, "Entropy-driven entanglement forging", [arXiv:2409.04510](https://arxiv.org/abs/2409.04510) (Sept 2024)

**Repositorios**:
- **prototype-entanglement-forging** (Qiskit Community): https://github.com/qiskit-community/prototype-entanglement-forging — VQE + forging. ⚠️ Puede estar deprecado, ver si migró a circuit-knitting-toolbox.
- **circuit-knitting-toolbox** (Qiskit): https://github.com/Qiskit-Extensions/circuit-knitting-toolbox — Incluía forging module (renombrado a `circuit_knitting.forging`).

**Integración con nuestro proyecto**:
- Aplicable a ladder (bipartición natural: chain A + chain B)
- Para heavy_hex: particionar en sub-grafos con poca conectividad
- Limita el overhead si el entanglement entropy es bajo (nuestro régimen h>h_c)

---

### 2.6 Quantum Bootstrap Embedding (QBE)

**Concepto**: Divide el sistema en fragmentos superpuestos (overlapping). Optimiza un Lagrangiano compuesto para que las regiones de overlap coincidan entre fragmentos adyacentes. Iterativo, self-consistent.

**Relevancia para nosotros**: ★★★☆☆ — Interesante para 2D, paper reciente (Junio 2025).

**Papers**:
- Liu et al., "Quantum Bootstrap Embedding", [arXiv:2301.01457](https://arxiv.org/abs/2301.01457) (2023, MIT)
- Li et al., "VQE-Based Quantum Bootstrap Embedding for Molecules", [arXiv:2606.17095](https://arxiv.org/html/2606.17095v1) (Junio 2025)

**Repositorios**:
- **QBootstrapEmbedding** (MIT): https://github.com/yuanliu1/QBootstrapEmbedding
- **QuEmb** (Troy Van Voorhis group): https://github.com/troyvvgroup/quemb — Bootstrap embedding para moléculas y sólidos 1D/2D. Activamente desarrollado (último commit Junio 2026).

**Integración con nuestro proyecto**:
- Los fragmentos overlapping son más robustos que particiones disjuntas
- Podríamos adaptar QuEmb para Hamiltonians de spin (actualmente usa fermiones)
- La estrategia de matching en boundaries es relevante para nuestro reensamblado

---

### 2.7 MERA — Multiscale Entanglement Renormalization Ansatz (Trotterized)

**Concepto**: Usa un esquema jerárquico de coarse-graining con disentanglers + isometries. En la versión Trotterized, los tensores son circuitos cuánticos parametrizados. El coarse-graining ES la descomposición.

**Relevancia para nosotros**: ★★★☆☆ — Conceptualmente elegante, costoso en circuit depth.

**Papers**:
- Barthel et al., "Quantum-classical eigensolver using MERA", [arXiv:2108.13401](https://arxiv.org/abs/2108.13401) (2021, publicado 2025)
- Barthel et al., "Convergence and quantum advantage of Trotterized MERA", [arXiv:2303.08910](https://arxiv.org/abs/2303.08910) (2023, publicado 2025)

**Repositorios**:
- ⚠️ No se encontró implementación pública del grupo de Barthel
- Implementación genérica de MERA tensors: https://github.com/HaoranLiao/dephased_ttn_mera (TensorFlow, clasificación)

**Integración con nuestro proyecto**:
- La idea de coarse-graining jerárquico complementa Deep VQE
- Los disentanglers podrían aprenderse con nuestra MPNN (predecir parámetros de cada capa)
- Requiere circuitos log-depth → factible para IBM Torino

---

## §3 — Técnicas Adicionales Descubiertas

### 3.1 Max-Cut Graph Partition para Inicialización VQE (2025)

**Concepto**: Usa max-cut del grafo del lattice para particionar qubits en clusters, diseña circuito ansatz basado en esa partición. Evita barren plateaus.

**Paper**: "Max-Cut graph-driven quantum circuit design for planar spin glasses", [arXiv:2504.12096](https://arxiv.org/abs/2504.12096) (Abril 2025)

**Relevancia**: ★★★★☆ — Directamente aplicable como estrategia de partición para nuestras topologías frustradas.

**Repositorio**: ⚠️ No encontrado público. Implementación propia con NetworkX sería sencilla.

---

### 3.2 Sample-based Quantum Diagonalization (SQD) + Krylov

**Concepto**: Prepara estados con circuitos cortos (Trotter evolution), muestrea, y diagonaliza clásicamente en el subespacio muestreado. No es D&C per se, pero permite usar circuitos más cortos que VQE completo.

**Papers**:
- IBM, "SKQD for Heisenberg models", [arXiv:2512.17141](https://arxiv.org/html/2512.17141v1) (Dic 2025) — 18 y 30 qubits en hardware real
- Piccinelli et al., "Quantum chemistry with SKQD", [arXiv:2508.02578](https://arxiv.org/html/2508.02578) (2025)

**Repositorios**:
- **qiskit-addon-sqd** (IBM oficial): https://github.com/Qiskit/qiskit-addon-sqd
- **qiskit-addon-sqd-hpc**: https://github.com/Qiskit/qiskit-addon-sqd-hpc (C++/MPI para HPC)

**Integración**: Complementario — podríamos usar SQD como solver de fragmentos en lugar de VQE.

---

### 3.3 Meta-VQE (Aspuru-Guzik, 2020)

**Concepto**: Entrena un circuito que aprende el perfil de energía E(h) para todo un rango de parámetros del Hamiltoniano, en vez de optimizar punto por punto.

**Papers**: Cervera-Lierta et al., "Meta-VQE: Learning energy profiles", [arXiv:2009.13545](https://arxiv.org/abs/2009.13545) (2020)

**Repositorios**:
- https://github.com/AlbaCL/Meta-VQE
- https://github.com/aspuru-guzik-group/Meta-VQE

**Relevancia para D&C**: ★★★☆☆ — No es D&C directamente, pero es complementario: una vez particionado, Meta-VQE puede resolver cada fragmento para todo el sweep de h simultáneamente (como hace nuestra MPNN).

---

### 3.4 Divide-and-Conquer del Espacio de Hilbert (Schulz et al., 2013)

**Concepto**: Para modelos con simetría traslacional, construye la base del espacio de Hilbert usando D&C + FFT. Reduce el costo de exact diag.

**Paper**: Schulz et al., "Divide and conquer the Hilbert space of translation-symmetric spin systems", [arXiv:1210.1701](https://arxiv.org/abs/1210.1701) (2013)

**Repositorio**:
- **Divide-and-Conquer-solves-localization**: https://github.com/lluisher/Divide-and-Conquer-solves-localization — D&C para eigenstates localizados en sistemas grandes.

**Relevancia**: ★★☆☆☆ — Más relevante para classical solver optimizations que para VQE.

---

### 3.5 MBE — Many-Body Expansion + VQE (Liu et al., 2022)

**Concepto**: Fragmentación tipo inclusion-exclusion (como NLCE pero para orbitales). Divide en fragmentos + calcula correcciones de 2-body, 3-body, etc.

**Paper**: Liu et al., "Divide-and-conquer variational quantum algorithms for large-scale simulations", [arXiv:2208.14789](https://arxiv.org/abs/2208.14789) (2022)

**Repositorio**: ⚠️ No encontrado público.

**Relevancia**: ★★★☆☆ — Complementario a NLCE con diferente esquema de truncación.

---

## §4 — Tabla Resumen de Repositorios

| Técnica | Repositorio | Lenguaje | Mantenido | Usable directo |
|---------|-------------|----------|-----------|----------------|
| Circuit Cutting | https://github.com/Qiskit/qiskit-addon-cutting | Python/Qiskit | ✅ Activo | ✅ Sí |
| SQD (Krylov) | https://github.com/Qiskit/qiskit-addon-sqd | Python/Qiskit | ✅ Activo | ✅ Sí |
| Entanglement Forging | https://github.com/qiskit-community/prototype-entanglement-forging | Python/Qiskit | ⚠️ Archivado? | 🔶 Parcial |
| DMET (Vayesta) | https://github.com/BoothGroup/Vayesta | Python | ✅ Activo | 🔶 Adaptar |
| DMET (libdmet) | https://github.com/gkclab/libdmet_preview | Python | ✅ Activo | 🔶 Adaptar |
| QC-DMET | https://github.com/SebWouters/QC-DMET | Python | ⚠️ Legacy | 🔶 Referencia |
| Bootstrap Embedding | https://github.com/troyvvgroup/quemb | Python | ✅ Activo (2026) | 🔶 Adaptar |
| QBootstrapEmbedding | https://github.com/yuanliu1/QBootstrapEmbedding | Python | ⚠️ 2023 | 🔶 Referencia |
| Meta-VQE | https://github.com/AlbaCL/Meta-VQE | Python | ⚠️ 2020 | 🔶 Ideas |
| CutQC | https://github.com/weiT1993/CutQC | Python | ⚠️ 2020 | 🔶 Referencia |
| QCut | https://github.com/JooNiv/QCut | Python/Qiskit | ? | 🔶 Alternativa |
| D&C Localization | https://github.com/lluisher/Divide-and-Conquer-solves-localization | Python | ⚠️ 2022 | 🔶 Ideas |
| QuSpin (ED) | https://github.com/QuSpin/QuSpin | Python | ✅ Activo | ✅ Validación |

---

## §5 — Plan de Integración con Nuestro Proyecto

### Fase 1: Quick wins (1-2 semanas)

1. **Circuit Cutting con qiskit-addon-cutting**
   - Instalar addon, crear PoC cortando un HVA circuit de N=20 en 2×N=10
   - Validar que E_cut ≈ E_exact para TFIM chain_1d
   - Medir sampling overhead

2. **Extender NLCE a square/triangular**
   - Enumerar clusters 2D hasta L=6 sitios
   - Implementar lattice constants para square
   - Comparar con exact diag N=16

### Fase 2: Deep VQE prototype (2-4 semanas)

3. **Implementar Deep VQE sobre nuestro pipeline**
   - Dividir chain N=20 en 2 subsistemas de N=10
   - Resolver cada subsistema con VQE (ya funciona)
   - Construir H_eff y resolver
   - Warm-start con MPNN predictions

4. **DMET benchmark con Vayesta**
   - Instalar Vayesta, configurar para lattice TFIM
   - Comparar E_DMET vs nuestro E_exact para square N=16
   - Evaluar si sirve como ground truth alternativo a DMRG

### Fase 3: Hardware-ready D&C (4-8 semanas)

5. **Circuit cutting para IBM Torino deployment**
   - Particionar HVA circuits N=40 heavy_hex en sub-circuitos ~20 qubits
   - Ejecutar sub-circuitos con PEA-ZNE
   - Reconstruir energía global

6. **Graph-based partitioning**
   - Implementar max-cut del lattice graph (NetworkX)
   - Usar como heurística de partición para todas las topologías
   - Comparar con strip-based partitioning

---

## §6 — Conexión con Infraestructura Existente

| Componente existente | Uso en D&C |
|---------------------|------------|
| `NLCERunner` | Base para NLCE 2D extension |
| `HamiltonianBuilder` | Construir H de cada fragmento |
| `make_lattice()` | Crear sub-lattices de fragmentos |
| `VQEOptimizer` | Solver de fragmentos |
| `MPNNPredictor` | Warm-start θ_init por fragmento |
| `HVACircuitBuilder` | Circuitos de cada fragmento |
| `NoisyBackend` + PEA-ZNE | Mitigación por sub-circuito |
| `ClassicalSolver` | Ground truth de fragmentos pequeños |
| `AQCCircuitCompressor` | Comprimir circuitos antes de cortar |

---

## §7 — Riesgos y Limitaciones

| Riesgo | Mitigación |
|--------|-----------|
| Overhead exponencial en circuit cutting | Minimizar número de cortes; usar topologías con pocos bonds cruzados |
| Pérdida de correlaciones en punto crítico | No aplicar D&C cerca de h_c; usar NLCE con L_max grande |
| Complejidad de implementación DMET | Usar Vayesta como backend, no reimplementar |
| Deep VQE requiere elegir base reducida | Empezar con low-energy states de cada subsistema |
| Entanglement forging limitado a biparticiones | Solo ladder y heavy_hex con partición natural |

---

## §8 — Referencias Adicionales Útiles

- **QuSpin** (ED para validación): https://github.com/QuSpin/QuSpin — Exact diag para spin chains, sirve como ground truth para validar D&C.
- IBM Quantum tutorials de circuit cutting: https://qiskit.github.io/qiskit-addon-cutting/tutorials/
- IBM tutorial de wire cutting para PBC: https://quantum.cloud.ibm.com/docs/tutorials/periodic-boundary-conditions-with-circuit-cutting
- IBM SQD tutorial para lattice models: https://quantum.cloud.ibm.com/docs/tutorials/krylov-quantum-diagonalization

---

*Content was rephrased for compliance with licensing restrictions. All sources cited inline.*
