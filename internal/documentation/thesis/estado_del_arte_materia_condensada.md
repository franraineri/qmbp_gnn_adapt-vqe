# Estado del Arte: Simulación Computacional de Materia Condensada Fuertemente Correlacionada

**Fecha**: Agosto 2026
**Contexto**: Revisión para la tesis "Hybrid GNN-HVA Framework for Topological Phase Characterization"

---

## 1. El Problema Fundamental

El espacio de Hilbert crece exponencialmente con el número de partículas (2^N para spines-1/2). No existe un método universal, insesgado y eficientemente escalable para resolver problemas genéricos de muchos cuerpos cuánticos. Cada familia de métodos impone aproximaciones que determinan sus límites de aplicabilidad.

---

## 2. Software Especializado — Programas Principales

### 2.1 Tensor Networks / DMRG

| Paquete | Lenguaje | Enfoque | Referencia |
|---------|----------|---------|-----------|
| **ITensor** (v4+) | Julia | General MPS/MPO, simetrías | [itensor.org](https://itensor.org) |
| **TeNPy** | Python | DMRG/TEBD/TDVP, 2D cilindros | [tenpy.readthedocs.io](https://tenpy.readthedocs.io) |
| **Block2** | C++/Python | ab initio DMRG, MPO paralelo | [arXiv:2310.03920](https://arxiv.org/abs/2310.03920) |
| **SyTen** | C++ | Simetrías no-abelianas (SU(2)) | Múnich group |
| **TensorKit.jl** | Julia | Infraestructura tensorial con simetrías anyónicas | [arXiv:2508.10076](https://arxiv.org/abs/2508.10076) |
| **TNRKit.jl** | Julia | Renormalización TN (TRG, HOTRG, LoopTNR) | [arXiv:2604.06922](https://arxiv.org/abs/2604.06922) |
| **ITensorNetworks.jl** | Julia | PEPS/redes tensoriales generales | [github.com/ITensor](https://github.com/ITensor/ITensorNetworks.jl) |
| **Tenet.jl** | Julia | TN en supercomputadores (BSC) | [github.com/bsc-quantic/Tenet.jl](https://github.com/bsc-quantic/Tenet.jl) |
| **DMRJtensor.jl** | Julia | Propósito general | [github.com/bakerte/DMRJtensor.jl](https://github.com/bakerte/DMRJtensor.jl) |
| **Quimb** | Python | TN generales, contracción | [github.com/jcmgray/quimb](https://github.com/jcmgray/quimb) |
| **TensorMixedStates** | Julia | MPS para estados puros y mixtos | [arXiv:2505.11377](https://arxiv.org/abs/2505.11377) |

Un survey de junio 2025 mapea **37 paquetes DMRG** existentes. Un benchmark de julio 2026 muestra diferencias de rendimiento de hasta 2 órdenes de magnitud entre implementaciones.

> Refs: [arXiv:2506.12629](https://arxiv.org/abs/2506.12629) "The Software Landscape for the Density Matrix Renormalization Group" (2025); [arXiv:2607.28369](https://arxiv.org/abs/2607.28369) "Performance Benchmarking: Software for the DMRG" (2026).

### 2.2 Quantum Monte Carlo

| Paquete | Método | Referencia |
|---------|--------|-----------|
| **SmoQyDQMC.jl** | DQMC con electrón-fonón | [arXiv:2311.09395](https://arxiv.org/abs/2311.09395) |
| **ALF** | Determinant QMC generalizado | [alf.physik.uni-wuerzburg.de](https://alf.physik.uni-wuerzburg.de) |
| **QMCPACK** | Diffusion/AFQMC para materiales | [qmcpack.org](https://qmcpack.org) |
| **QUEST** | DQMC para modelos de Hubbard | DOE funded |
| **mVMC** | VMC para fermiones correlacionados | [github.com/issp-center-dev/mVMC](https://github.com/issp-center-dev/mVMC) |

### 2.3 Neural Quantum States

| Paquete | Enfoque | Referencia |
|---------|---------|-----------|
| **NetKet** | VMC + NQS (JAX) | [netket.readthedocs.io](https://netket.readthedocs.io) |
| **jVMC** | VMC JAX-based para spines/fermiones | [github.com/markusschmitt/vmc_jax](https://github.com/markusschmitt/vmc_jax) |
| **FermiNet** | Ab initio VMC con redes profundas | [github.com/google-deepmind/ferminet](https://github.com/google-deepmind/ferminet) |
| **DeepQMC** | VMC molecular (JAX/PyTorch) | [github.com/deepqmc/deepqmc](https://github.com/deepqmc/deepqmc) |

### 2.4 DFT y Campo Medio

| Paquete | Método | Escala |
|---------|--------|--------|
| **VASP** | DFT | ~10⁶ átomos |
| **Quantum ESPRESSO** | DFT | ~10⁵ átomos |
| **TRIQS** | DFT+DMFT | ∞ lattice, ~5-10 orb. correl. |
| **w2dynamics** | CT-QMC impurity solver | Impurity ~5-10 orbitales |

### 2.5 Computación Cuántica

| Paquete | Enfoque | Referencia |
|---------|---------|-----------|
| **Qiskit** (IBM) | VQE, QPE, dinámica | [qiskit.org](https://qiskit.org) |
| **Cirq** (Google) | QAOA, simulación | [quantumai.google/cirq](https://quantumai.google/cirq) |
| **PennyLane** (Xanadu) | VQE, ML cuántico | [pennylane.ai](https://pennylane.ai) |
| **QESEM** (Qedma) | Error mitigation | [docs.qedma.io](https://docs.qedma.io) |

---

## 3. Métodos Numéricos: Capacidades y Límites (2024–2026)

### 3.1 Diagonalización Exacta (ED / Lanczos)

- **Principio**: Almacena el vector de estado completo; sin aproximaciones.
- **Límite actual**: ~40–50 spines-1/2 (con simetrías + supercomputadores >1 TB RAM). Rutinariamente ~20–30 sitios.
- **Escaleo**: Exponencial insalvable (2^N).
- **Software**: QuSpin, HΦ, PETSc/SLEPc.

> Ref: [arXiv:1802.10052](https://arxiv.org/abs/1802.10052) "Exact diagonalization methods revisited" — hasta 50 spines-1/2 con simetrías.

### 3.2 DMRG / MPS (1D)

- **Límite actual**: ~1000+ sitios con D~1000–4000. Con TPUs: D=65,536.
- **Escaleo**: O(N·D³); polinomial para sistemas gapped 1D.
- **Limitaciones**: Mal escaleo en 2D, problemas críticos, dinámicas a tiempos largos.

> Ref: [arXiv:2204.05693](https://arxiv.org/abs/2204.05693) "DMRG with Tensor Processing Units" (Google, 2022) — D=65,536 en 1D.

### 3.3 DMRG 2D (Cilindros)

- **Límite actual**: 6–12 sitios de ancho × largo (~100–200 sitios totales).
- **Escaleo**: D requerido crece exponencialmente con el ancho del cilindro.
- **Avance 2025**: Gutzwiller-guided DMRG resuelve mínimos locales en 2D, accede a fases topológicas exóticas.

> Refs: [arXiv:2503.18374](https://arxiv.org/abs/2503.18374) "Gutzwiller-guided DMRG" (Springer 2025); modelo de Hubbard en cilindros de ancho 8: [arXiv:2511.18644](https://arxiv.org/abs/2511.18644).

### 3.4 Transcorrelated DMRG

- **Principio**: Correlador Jastrow/Gutzwiller absorbido en el Hamiltoniano → reduce D necesario.
- **Límite actual**: **12×12 = 144 sitios** (4× más que estudios previos).
- **Escaleo**: Mejora significativa sobre DMRG estándar para esfuerzo computacional equivalente.

> Ref: [arXiv:2506.07441](https://arxiv.org/abs/2506.07441) "Scaling up the transcorrelated DMRG" (PRB 2025).

### 3.5 PEPS / iPEPS (Redes Tensoriales 2D)

- **Límite actual (fPEPS finito)**: D=28, superando DMRG con m=32,000 multipletes SU(2) en 8-leg ladders.
- **iPEPS**: Trabaja directamente en el límite termodinámico con D≤16–20.
- **Escaleo**: Contracción D^10–D^12.

> Ref: [arXiv:2502.13454](https://arxiv.org/abs/2502.13454) "Accurate Simulation of the Hubbard Model with Finite Fermionic PEPS" (PRL 2025) — hito: fPEPS D=28 > DMRG m=32,000 en Hubbard 8-leg.

### 3.6 Quantum Monte Carlo (QMC)

#### Sin sign problem (bosones, spines no frustrados, half-filling)

- **DQMC**: **10,368 sitios** (honeycomb Hubbard, 2026).
- **DQMC (submatrix update)**: **8,000 sitios** (Hubbard 3D half-filling, 2024).
- **SSE/World-line**: >10,000 sitios para bosones/spines.

> Refs: [arXiv:2602.03656](https://arxiv.org/abs/2602.03656) "Resolving Quantum Criticality in the Honeycomb Hubbard Model" (2026, 10,368 sitios); [arXiv:2404.09989](https://arxiv.org/abs/2404.09989) "Boosting DQMC with Submatrix Updates" (2024, 8,000 sitios).

#### Con sign problem (fermiones dopados, frustración)

- **Límite práctico**: ~100–300 sitios a temperaturas bajas.
- **Escaleo**: Señal decae exponencialmente con β·N.

#### AFQMC ab initio (sólidos)

- **Avance 2026**: Acceso directo al límite termodinámico con scaling O(N³) mediante tensor hypercontraction + k-point symmetry.
- **Resultado**: Primer cálculo AFQMC de sólidos sin embedding ni correcciones finite-size empíricas.

> Ref: [arXiv:2602.16679](https://arxiv.org/abs/2602.16679) "Ab Initio AFQMC in the Thermodynamic Limit" (2026).

### 3.7 Neural Quantum States (NQS) + VMC

- **Récord 2026**: **42×42 = 1,764 sitios** en magneto frustrado J₁-J₂ triangular.
- **Avance algorítmico**: "Constant-time local updates" eliminan el cuello de botella O(N).
- **Limitación**: Entrenamiento costoso, validación difícil, expresividad acotada por O(log N) para redes finitas.

> Refs: [arXiv:2602.02665](https://arxiv.org/abs/2602.02665) "Approaching the Thermodynamic Limit with NQS" (2026, 42×42 sitios); [arXiv:2603.11189](https://arxiv.org/abs/2603.11189) "Constant-Time Local Updates for NQS" (2026); [arXiv:2505.03466](https://arxiv.org/abs/2505.03466) "Design principles of deep NQS for frustrated magnets" (2025).

### 3.8 DFT + DFT+DMFT

- **DFT**: ~10⁶ átomos (lineal-scaling). No captura correlaciones fuertes.
- **DFT+DMFT**: Lattice infinito, ~5–10 orbitales correlacionados. Excelente para óxidos, materiales f-electron.
- **Avance 2026**: DFT+DMFT con quantum computers como impurity solvers.

> Refs: [Nature npj Comp. Mat. 2026](https://www.nature.com/articles/s41524-026-02289-2) "Efficient quantum implementation of DMFT"; [arXiv:2601.16401](https://arxiv.org/abs/2601.16401) "Accelerating DMFT convergence by preconditioning" (2025).

---

## 4. Tabla Resumen: Límites de Partículas por Método (2026)

| Método | Dim. óptima | Tamaño máximo actual | Limitación | Software principal |
|--------|------------|---------------------|------------|-------------------|
| ED (Lanczos) | Cualquiera | ~40–50 spines | Exponencial en N | QuSpin, HΦ, PETSc |
| DMRG 1D | 1D | ~1000+ sitios (D~4000) | Polinomial si gapped | ITensor, TeNPy, Block2 |
| DMRG 2D (cilindro) | Quasi-2D | ~100–200 (ancho ≤8–12) | Exponencial en ancho | ITensor, TeNPy, SyTen |
| Gutzwiller-guided DMRG | 2D | Mejora DMRG 2D | Aún limitado por ancho | Custom codes |
| Transcorrelated DMRG | 2D | 12×12 = 144 sitios | Jastrow complexity | Block2 + custom |
| fPEPS finito | 2D | 8-leg ladders (D=28) | D^10+ contracción | Custom, PEPSKit.jl |
| iPEPS | 2D (∞) | Límite termodinámico (D≤20) | Contracción aproximada | TeNPy, custom |
| QMC sin sign | Cualquiera | **10,368 sitios** | Solo sin frustración | SmoQyDQMC, ALF |
| QMC con sign | Cualquiera | ~100–300 sitios | Exponencial en β·N | ALF, QUEST |
| AFQMC ab initio | 3D sólidos | Límite termodinámico (O(N³)) | Phaseless bias | QMCPACK |
| NQS/VMC | 2D frustrado | **1,764 sitios** (42×42) | Entrenamiento, validación | NetKet, jVMC |
| FermiNet/DeepQMC | Moléculas | ~30–50 electrones | GPU-intensive | FermiNet, DeepQMC |
| DFT | 3D real | ~10⁶ átomos | No correlaciones fuertes | VASP, QE |
| DFT+DMFT | 3D real | ∞ lattice, ~5–10 orb. | Impurity solver | TRIQS, w2dynamics |
| Quantum hardware | Dinámica | **74 qubits** (ventaja) | Errores, profundidad | Qiskit, QESEM |

---

## 5. Quantum Advantage: IBM + Qedma (julio 2026)

### 5.1 El Experimento (arXiv:2607.24937)

**Paper**: "Resolving Structure in Prethermal Floquet Dynamics with Precision Quantum Computation"
**Autores**: Leviatan, Watad, Perry et al. (Qedma + IBM + RIKEN + BlueQubit)
**Publicado**: 27 julio 2026

#### Modelo físico

**Floquet Mixed-Field Ising Model** en topología heavy-hex. El unitario de un ciclo:

```
U_F = ∏(r=1→3) [ e^{-iθ_zz/2 · C_r} · e^{-iθ_z/2 · Z_Σ} · e^{-iθ_x/2 · X_Σ} ]
```

Donde C_r = Σ_{⟨i,j⟩∈E_r} Z_i·Z_j son las interacciones ZZ sobre las 3 capas de color del heavy-hex.

#### Parámetros seleccionados

- **θ_x ≈ π/6** (campo transverso)
- **θ_z ≈ π/27** (campo longitudinal)
- **θ_zz = π/3** (interacción Ising)
- **Estado inicial**: |0⟩^N (ferromagnético)
- Punto seleccionado: **(3θ_x, 3θ_z, θ_zz) = (1.57, 0.34, 1.05)**

Estos parámetros son **fijos** — no se barren. Lo que varía es el número de ciclos Floquet (tiempo).

#### Topología y tamaños

- **Topología**: Exclusivamente **heavy-hex** (nativa IBM Heron r3)
- **Tamaños**: 21, 28, 35 qubits (exacto) → 51, 74 qubits (QPU con QESEM)
- **Hardware**: IBM `ibm_boston` (Heron r3, 156 qubits), también validación en Quantinuum H2 y Helios

#### Observable principal

Magnetización promedio: M = (1/N_q) Σ_q ⟨Z_q⟩ en función del ciclo Floquet (tiempo discreto).

#### Por qué los métodos clásicos fallan

1. **PEPS-BP** (D=512, D=700): Converge hasta ~12 ciclos Floquet, luego drift no-físico.
2. **Sparse Pauli-Path** (ORQA en Fugaku, ~10¹² strings): Divergencia de fase a partir de ciclo ~15. Estimación: ~10³⁰ Pauli strings necesarios para convergencia.
3. **PEPO-BP**: Pierde convergencia a ~7 ciclos (operator entanglement crece más rápido).
4. **Heuristic-Corrected TEBD** (MPS χ=4096): Amplitudes no-monotónicas con sistema size.
5. **Statevector**: Imposible más allá de 35 qubits (2³⁵ amplitudes).

**Razón fundamental**: La dinámica Floquet genera entrelazamiento volume-law y complejidad de operador que ninguna representación clásica eficiente puede capturar después de suficientes ciclos.

#### Resultado físico descubierto

Oscilaciones pretermales sub-armónicas de larga vida (período ~4× el drive) en la magnetización. Finite-size scaling con los datos cuánticos (51, 74 qubits) provee evidencia fuerte de que estas oscilaciones persisten en el límite termodinámico de heavy-hex ladders.

#### Validación de confianza (Trusted Quantum Computation)

1. QESEM-Unbiased coincide con clásicos en su rango convergido (ciclos 1–12).
2. QESEM-Extrapolated coincide con QESEM-Unbiased en su ventana común.
3. Validación cruzada en Quantinuum H2/Helios (hardware independiente).
4. Modelo de ruido validado con Z-scores ≈ N(0,1).
5. Resultados publicados en el Quantum Advantage Tracker (abierto a refutación).

### 5.2 Las 3 Demostraciones Simultáneas (30 julio 2026)

| Colaboración | Problema | Qubits | Ref |
|---|---|---|---|
| IBM + Qedma | Dinámica KTFIM prethermal | 74 | [arXiv:2607.24937](https://arxiv.org/abs/2607.24937) |
| IBM + Algorithmiq | Materia heterogénea | ~50-100 | [IBM Newsroom](https://newsroom.ibm.com/2026-07-30-ibm-and-algorithmiq-demonstrate-quantum-advantage,-establishing-a-framework-for-trusted-quantum-computation-beyond-classical-verification) |
| IBM + U. Chicago | Circuitos lógicos error-corrected | 70 lógicos | [IBM Newsroom](https://newsroom.ibm.com/2026-07-30-ibm-and-the-university-of-chicago-demonstrate-quantum-advantage,-establishing-trusted-quantum-computation-on-logical-circuits) |

---

## 6. Comparación: IBM+Qedma vs GNN-HVA (Este Proyecto)

### 6.1 Paradigmas Cuánticos

| | IBM (Simulación Hamiltoneana) | GNN-HVA (Variacional) |
|---|---|---|
| **Circuito** | Fijo, determinado por la física | Parametrizado, θ a optimizar |
| **Profundidad** | Alta (30 Floquet cycles) | Baja (p=1–3 capas HVA) |
| **Output** | ⟨M(t)⟩ = magnetización vs tiempo | E₀(h) = energía mínima vs campo |
| **Lo que querés saber** | Comportamiento dinámico (no-equilibrio) | Diagrama de fases estático (equilibrio) |
| **Entrelazamiento** | Crece con t (volume law) | Acotado (area law) |
| **¿QPU es necesaria?** | Sí (clásico no puede) | No (clásico puede, QPU es alternativa) |
| **Rol del ML** | Ninguno | Central (GNN elimina VQE iterativo) |

### 6.2 Parámetros

| Tu proyecto | IBM+Qedma |
|-------------|-----------|
| h (campo transverso), barrido 0→2 | θ_x, θ_z, θ_zz (fijos en un punto) |
| J=1 (fijo) | Implícito en θ_zz |
| Estado inicial: \|+⟩^N | Estado inicial: \|0⟩^N |
| Lo que varía: campo h | Lo que varía: tiempo (# ciclos) |

### 6.3 Topología y Tamaño

| | IBM+Qedma | GNN-HVA |
|---|---|---|
| Topologías | Solo heavy-hex | 6 (chain, heavy_hex, ladder, square, triangular, kagome) |
| N máximo exacto | 35 | 20 (ED) / 200 (MPS extrapolación) |
| N máximo QPU | 74 | 10 (hardware, Phase 4) |

### 6.4 Convergencias Estratégicas

1. **Ambos usan heavy-hex**: El hallazgo de este proyecto (heavy_hex ≈ chain_1d en expresividad) valida el diseño de IBM.
2. **Ambos usan TFIM**: El modelo base es esencialmente el mismo Hamiltoniano. IBM lo evoluciona dinámicamente; nosotros calculamos su ground state.
3. **Error mitigation**: QESEM (ellos) y ZNE+PEA+GNN-QEM (nosotros) atacan el mismo problema.
4. **Puente natural**: GNN-HVA prepara |ψ₀⟩ eficientemente → protocolo tipo IBM evoluciona ese estado → quantum advantage en la dinámica.

### 6.5 ¿Por qué IBM logra quantum advantage y este proyecto no?

1. **Ground states satisfacen area law** → representables por MPS/DMRG → clásicamente resolubles.
2. **La dinámica temporal genera volume law** → no representable clásicamente → QPU necesaria.
3. **Circuitos shallow (p≤3) son simulables clásicamente**; circuitos con 30 Floquet cycles no.
4. **Este proyecto aporta eficiencia (29-500× speedup) y generalización (cross-N), no ventaja cuántica intrínseca.**

---

## 7. Métodos Híbridos Emergentes (2025–2026)

| Método | Descripción | Referencia |
|--------|-------------|-----------|
| Gutzwiller-guided DMRG | Funciones de onda proyectadas como estado inicial para DMRG 2D | [arXiv:2503.18374](https://arxiv.org/abs/2503.18374) |
| Transcorrelated DMRG | Jastrow en Hamiltoniano → reduce D | [arXiv:2506.07441](https://arxiv.org/abs/2506.07441) |
| Quantum-assisted VMC | Circuito cuántico genera muestras para VMC clásico | [arXiv:2502.20799](https://arxiv.org/abs/2502.20799) |
| ORQA | Álgebra cuántica OR-representada para dinámica escalable | [arXiv:2506.13241](https://arxiv.org/abs/2506.13241) |
| PhysVEC | AI + verificadores para simulación auto-correctiva | [arXiv:2604.00149](https://arxiv.org/abs/2604.00149) |
| LLM-assisted DMRG | LLMs generan código DMRG desde especificaciones LaTeX | [arXiv:2604.04089](https://arxiv.org/abs/2604.04089) |

---

## 8. Problemas Abiertos (2026)

Problemas donde ningún método clásico es confiable:

1. **Modelo de Hubbard dopado en 2D** (superconductividad de alta Tc): ~100 sitios con incertidumbre significativa entre métodos.
2. **Spin liquids en kagome/triangular**: NQS y DMRG dan respuestas distintas.
3. **Dinámica fuera de equilibrio a tiempos largos**: Todos los métodos clásicos colapsan (→ quantum advantage demostrada).
4. **Fermiones en 3D con frustración**: Prácticamente inaccesible.
5. **Transiciones de fase topológicas**: Requieren observables no-locales difíciles de medir.

---

## 9. Tendencias del Ecosistema

1. **Julia está dominando** el nuevo desarrollo en tensor networks y QMC (TensorKit, TNRKit, SmoQyDQMC, Tenet, DMRJtensor).
2. **GPU/TPU acceleration** está empujando D a valores sin precedentes (D=65,536 en TPUs).
3. **NQS como aproximador universal** se consolida para frustrados 2D (42×42 = 1,764 sitios).
4. **Quantum advantage existe para dinámica** (IBM julio 2026), no para ground states estáticos.
5. **AFQMC llega al TDL** eliminando la barrera de superceldas finitas en sólidos reales.
6. **PEPS supera DMRG en 2D genuino** (fPEPS D=28 > DMRG m=32,000 en Hubbard 8-leg).

---

## 10. Referencias Completas

### Reviews y Surveys

- [arXiv:2506.09308](https://arxiv.org/abs/2506.09308) — "Quantum Algorithm Software for Condensed Matter Physics" (2025). Survey + benchmark reproducible.
- [arXiv:2506.12629](https://arxiv.org/abs/2506.12629) — "The Software Landscape for the DMRG" (2025). Compara 37 paquetes.
- [arXiv:2607.28369](https://arxiv.org/abs/2607.28369) — "Performance Benchmarking: Software for the DMRG" (2026).
- [Nature Comms 2024](https://www.nature.com/articles/s41467-024-46402-9) — "Quantum many-body simulations on digital quantum computers: State-of-the-art".
- [arXiv:2304.13395](https://arxiv.org/abs/2304.13395) — "Density-matrix renormalization group" (tutorial/review 2023).
- [arXiv:2402.09402](https://arxiv.org/abs/2402.09402) — "Neural Quantum States: From Architectures to Applications" (2024).

### Récords de Tamaño (2024–2026)

- [arXiv:2602.03656](https://arxiv.org/abs/2602.03656) — DQMC 10,368 sitios (honeycomb Hubbard, 2026).
- [arXiv:2404.09989](https://arxiv.org/abs/2404.09989) — DQMC 8,000 sitios (submatrix updates, 2024).
- [arXiv:2602.02665](https://arxiv.org/abs/2602.02665) — NQS 42×42 sitios (frustrated 2D, 2026).
- [arXiv:2204.05693](https://arxiv.org/abs/2204.05693) — DMRG D=65,536 en TPUs (2022).
- [arXiv:2502.13454](https://arxiv.org/abs/2502.13454) — fPEPS D=28 supera DMRG (Hubbard 8-leg, 2025).
- [arXiv:2506.07441](https://arxiv.org/abs/2506.07441) — Transcorrelated DMRG 12×12 (2025).
- [arXiv:2602.16679](https://arxiv.org/abs/2602.16679) — AFQMC en límite termodinámico (2026).

### Quantum Advantage

- [arXiv:2607.24937](https://arxiv.org/abs/2607.24937) — "Resolving Structure in Prethermal Floquet Dynamics" (IBM+Qedma, 2026). **Paper principal**.
- [arXiv:2508.10997](https://arxiv.org/abs/2508.10997) — "Reliable high-accuracy error mitigation for utility-scale circuits" (QESEM paper, 2025).
- [arXiv:2608.05202](https://arxiv.org/abs/2608.05202) — "Quantum Error Management in Practice" (comparación IBM/QESEM/Q-CTRL, 2026).
- [arXiv:2603.18825](https://arxiv.org/abs/2603.18825) — "Quantum Advantage: a Tensor Network Perspective" (2026).
- [IBM Newsroom 2026-07-30](https://newsroom.ibm.com/2026-07-30-ibm-and-qedma-demonstrate-quantum-advantage,-modeling-physics-beyond-classical-capabilities-through-trusted-quantum-computation) — Anuncio oficial IBM+Qedma.

### Neural Quantum States

- [arXiv:2406.01017](https://arxiv.org/abs/2406.01017) — "Neural Quantum States in VMC Method" (review 2024).
- [arXiv:2505.03466](https://arxiv.org/abs/2505.03466) — "Design principles of deep NQS for frustrated magnets" (2025).
- [arXiv:2603.11189](https://arxiv.org/abs/2603.11189) — "Constant-Time Local Updates for NQS" (2026).

### Métodos Híbridos y Nuevos

- [arXiv:2503.18374](https://arxiv.org/abs/2503.18374) — Gutzwiller-guided DMRG (2025).
- [arXiv:2502.20799](https://arxiv.org/abs/2502.20799) — Quantum-assisted VMC (2025).
- [arXiv:2506.13241](https://arxiv.org/abs/2506.13241) — ORQA para dinámica cuántica escalable (2025).
- [arXiv:2604.00149](https://arxiv.org/abs/2604.00149) — PhysVEC: AI auto-correctiva para simulación (2026).
- [arXiv:2604.04089](https://arxiv.org/abs/2604.04089) — LLM-assisted DMRG development (2026).

### Modelos y Benchmarks

- [arXiv:2606.03147](https://arxiv.org/abs/2606.03147) — "Quantum Optimization Algorithms for Strongly Correlated Many-Body Systems" (2026).
- [Nature npj Comp. Mat. 2026](https://www.nature.com/articles/s41524-026-02289-2) — DFT+DMFT con quantum solvers.
- [arXiv:2311.09395](https://arxiv.org/abs/2311.09395) — SmoQyDQMC.jl (Julia DQMC, 2024).

### Software Landscape

- [github.com/PerSehlstedt/DMRG-software](https://github.com/PerSehlstedt/DMRG-software) — Lista curada de implementaciones DMRG.
- [tensor4all.org](https://tensor4all.org/) — Ecosistema tensor4all (C++/Python/Julia/Rust).
- [ALPS](https://alps.comp-phys.org/) — ALPS software package.

---

*Documento generado para la tesis "Hybrid GNN-HVA Framework for Topological Phase Characterization". Última actualización: agosto 2026.*


---

## 11. Direcciones hacia Quantum Advantage desde GNN-HVA

### Premisa

El proyecto actual opera en un régimen (ground states, area-law, N=4-200) donde la computación clásica es superior. Las siguientes direcciones extienden el paradigma hacia problemas donde la QPU es genuinamente necesaria.

### Direcciones Identificadas

| # | Dirección | Idea | Factibilidad | Impacto |
|---|-----------|------|:---:|:---:|
| 1 | **Quench dynamics post-preparación** | GNN prepara \|ψ₀(h)⟩ → Trotter evolution U(t)\|ψ₀⟩ genera volume-law → clásico colapsa | ★★★★ | ★★★★★ |
| 2 | **2D frustrated N>20** | Triangular/kagome N=20-30 donde DMRG y NQS no acuerdan; QPU como árbitro | ★★ | ★★★★★ |
| 3 | **Quantum training data** | QPU genera datos en regímenes inaccesibles clásicamente → reentrenar GNN → modelo clásico informado cuánticamente | ★★★ | ★★★ |
| 4 | **Excited states y gaps** | Variational deflation para excited states con entrelazamiento mayor; gap espectral en transiciones QPT | ★ | ★★★★ |
| 5 | **Verificación cuántica** | QPU verifica DMRG en regímenes grade F; publicar en Quantum Advantage Tracker | ★★★★★ | ★★★ |

### Recomendación: Combinación 1+5

**Corto plazo (tesis)**: Dirección 5 — verificar DMRG con QPU+QESEM en regímenes donde tenemos grade F (heavy_hex/triangular N=10-16).

**Extensión post-tesis**: Dirección 1 — agregar Trotter steps post-HVA. GNN prepara ground state (costo cero) → evolución Floquet 15-30 cycles → régimen de quantum advantage demostrada (IBM, arXiv:2607.24937).

### Diferenciador Único

Ningún otro grupo tiene un GNN que prepare ground states de TFIM en heavy-hex instantáneamente. Esto convierte la preparación de estado (VQE iterativo costoso) en costo cero, liberando todo el QPU-time para dinámica (donde la ventaja existe).


---

## 12. Condiciones para Quantum Advantage en Materia Condensada

### 12.1 Definición Operativa (2026)

Quantum advantage ocurre cuando un computador cuántico produce un resultado que ninguna computadora clásica puede reproducir con recursos comparables, **y** el resultado es verificable. Requiere dos criterios simultáneos:

1. **Superioridad computacional**: El problema excede a los mejores métodos clásicos disponibles.
2. **Confianza (Trust)**: El resultado cuántico es verificablemente correcto (no garbage from noise).

> Ref: [arXiv:2506.20658](https://arxiv.org/abs/2506.20658) "A Framework for Quantum Advantage" (IBM, 2025). Cinco propiedades: predictability, typicality, robustness, verifiability, usefulness.

### 12.2 Las 3 Condiciones Necesarias (derivadas de la evidencia 2024-2026)

**Condición 1: Entrelazamiento volume-law**

- Ground states de Hamiltonians locales satisfacen area-law → MPS/DMRG los representan eficientemente → **NO hay quantum advantage para ground states estáticos**.
- La dinámica temporal (Trotter, Floquet, quench) genera volume-law → los clásicos colapsan → ventaja posible.
- Excepción potencial: ground states de sistemas altamente frustrados en 2D (area-law se cumple pero el coeficiente es tan grande que DMRG necesita D impracticable).

**Condición 2: Dimensionalidad >1D**

- En 1D, tensor networks (DMRG/TDVP) son extraordinariamente eficientes. Incluso la dinámica es simulable hasta tiempos considerables con GPU modernas.
- **Ejemplo concreto (junio 2026)**: Q-CTRL clamó 3,000× speedup en Fermi-Hubbard 1D (L=60, 120 qubits). Un mes después, Multiverse Computing ([arXiv:2606.04771](https://arxiv.org/abs/2606.04771)) resolvió el **mismo problema completamente** con TDVP + GPU H200 + simetrías, alcanzando χ=60,000 (15× más que Q-CTRL), incluyendo el régimen que la QPU dejó sin verificar.
- Heavy-hex es quasi-2D (tiene loops) → belief propagation pierde garantías → sweet spot para ventaja cuántica.
- 2D genuino (square, triangular, kagome) → ventaja más robusta, pero requiere más qubits y profundidad.

**Condición 3: Observable sensible a componentes "difíciles"**

- No basta que el estado sea difícil de representar. El observable medido debe depender de las componentes que los clásicos truncan.
- En IBM+Qedma: las oscilaciones pretermales dependen de Pauli strings de peso ~10-20% del espacio total (4^N). Truncar por debajo de eso destruye la señal.
- Energía del ground state es un observable "fácil" (DMRG la aproxima bien incluso con D modesto). Correlaciones temporales a largo alcance son "difíciles".

### 12.3 La Carrera Claims-Refutaciones (Patrón Histórico)

| Fecha | Claim | Refutación | Lección |
|-------|-------|-----------|---------|
| Jun 2023 | IBM 127-qubit kicked Ising (Nature 618) | Tindall et al. (PRX Quantum 2024): BP + TN reproduce con más precisión | Heavy-hex tree-like → BP funciona para ciertos parámetros |
| May 2026 | Q-CTRL 120-qubit Fermi-Hubbard 1D, 3000× | Multiverse (arXiv:2606.04771): GPU TDVP χ=60k resuelve todo | **1D nunca es seguro** para quantum advantage |
| Jul 2026 | IBM+Qedma 74-qubit Floquet prethermal | 8+ meses en Quantum Advantage Tracker sin refutación | Parámetros far-from-Clifford + loops + volume-law = robusto |

> Refs: [arXiv:2606.04771](https://arxiv.org/abs/2606.04771) "Pushing the Classical Frontier of 1D Fermi-Hubbard Quench Dynamics Beyond Current Quantum Simulations"; [arXiv:2603.18825](https://arxiv.org/abs/2603.18825) "Quantum Advantage: a Tensor Network Perspective".

### 12.4 Límites Fundamentales de Error Mitigation

- Sampling overhead de error mitigation crece **exponencialmente** con profundidad del circuito ([arXiv:2109.04457](https://arxiv.org/abs/2109.04457), Nature npj QI 2022).
- Error mitigation **sola** no puede producir ventaja cuántica exponencial (resultado teórico: [arXiv:2503.17243](https://arxiv.org/abs/2503.17243)).
- Pero: para problemas de tamaño polinomial fijo donde solo se necesita ventaja constante/polinomial → error mitigation es suficiente (caso IBM+Qedma).
- La ventaja real viene de que el **problema** escala exponencialmente para los clásicos, y el overhead de mitigation escala más lento que ese crecimiento.

### 12.5 Dónde la Computación Cuántica ya Gana (agosto 2026)

| Régimen | Por qué gana | Evidencia |
|---------|-------------|-----------|
| Dinámica Floquet quasi-2D (heavy-hex), >15 cycles | Volume-law + loops + far-from-Clifford | IBM+Qedma (arXiv:2607.24937) |
| Fermi-Hubbard 2D quench, N>60 | Volume-law + fermion sign → QMC y TN fallan | Quantinuum (arXiv:2510.26300) |
| Thermal states frustrated kagome, 79 spines | Frustración + temperatura finita + sign problem | IBM 139-qubit (arXiv:2605.26245) |

### 12.6 Dónde NO Gana (y por qué es relevante para nosotros)

| Régimen | Por qué NO | Competidor clásico |
|---------|-----------|-------------------|
| Ground states 1D (cualquier N) | Area-law → DMRG D~4000 | ITensor, TeNPy |
| Ground states 2D (N<100, gapped) | Area-law → PEPS/iPEPS | fPEPS D=28 |
| Dinámica 1D (hasta t~10) | GPU TDVP χ=60,000 | H200 GPU clusters |
| Energía molecular (few electrons) | CCSD(T), AFQMC | Gaussian, QMCPACK |

### 12.7 Implicaciones para GNN-HVA

**Lo que nuestro proyecto PUEDE aportar hacia quantum advantage:**
- Preparación de estado no-trivial (ground state a h arbitrario) a costo O(1) → **enabling technology** para protocolos donde otros ya demostraron ventaja.
- La preparación con VQE iterativo es el bottleneck principal en protocolos de quench dynamics. Eliminar ese bottleneck con GNN es una contribución real.

**Lo que nuestro proyecto NO PUEDE hacer:**
- Clamar quantum advantage en ground states (area-law, siempre resoluble clásicamente).
- Mostrar ventaja en chain_1d a cualquier N (será refutado con tensor networks en GPU).
- Competir con IBM/Q-CTRL sin hardware comparable y QESEM.

**Posicionamiento correcto:**
"GNN-HVA es una enabling technology que reduce el costo de preparación de estados cuánticos no-triviales, habilitando la exploración sistemática de regímenes dinámicos donde la quantum advantage está demostrada."

---

## 13. Otros Claims de Quantum Advantage (2025-2026)

### Q-CTRL: Fermi-Hubbard 1D (mayo 2026)

- **Paper**: [arXiv:2605.04025](https://arxiv.org/abs/2605.04025) "Fast, accurate, high-resolution simulation of large-scale Fermi-Hubbard models on a digital quantum processor"
- **Resultado**: 120 qubits, L=60 fermiones, hasta t=6, 30 Trotter steps en IBM Heron
- **Claim**: 3,000× speedup vs TDVP clásico con χ=4096
- **Parcialmente refutado**: [arXiv:2606.04771](https://arxiv.org/abs/2606.04771) mostró que con GPU H200, simetrías U(1), y χ hasta 60,000, el clásico converge completamente incluyendo el régimen high-entanglement. El speedup se reduce significativamente.
- **Lección**: En 1D, la competencia clásica mejora rápidamente con hardware y algoritmos.

### IBM Kagome: Thermal States (junio 2026)

- **Paper**: [arXiv:2605.26245](https://arxiv.org/abs/2605.26245) "Preparing thermal states of frustrated quantum spin systems using 139 qubits"
- **Resultado**: Antiferromagneto en kagome, 79 spines + 60 environment qubits, 1000+ capas de 2Q gates
- **Significado**: Preparación de thermal states de sistema frustrado a temperatura ajustable. Señal robusta pese a profundidad extrema.
- **Relevancia**: La preparación de estados no-triviales es exactamente lo que nuestro GNN hace (pero para ground states, no thermal states).

### Sample-based Krylov Quantum Diagonalization (SKQD)

- **Paper**: [arXiv:2605.29521](https://arxiv.org/abs/2605.29521) "Ground-state estimation of the Heisenberg model on frustrated lattices"
- **Resultado**: Kagome lattice hasta 72 spines usando muestras de QPU + diagonalización Krylov clásica
- **Relevancia**: Método híbrido donde QPU genera bases de Krylov que el clásico usa para ground states. Distinto a VQE puro.

---

*Última actualización: agosto 2026. Incluye análisis de condiciones para quantum advantage y posicionamiento del proyecto.*
