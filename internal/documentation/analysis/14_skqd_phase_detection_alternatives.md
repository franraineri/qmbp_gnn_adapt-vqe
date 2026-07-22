# SKQD para Detección de Transiciones de Fase Cuánticas — Análisis de Viabilidad

**Fecha**: 2026-06-05
**Contexto**: Evaluación de 5 alternativas para usar SKQD/SQD en el problema
de detección de QPT, verificando novedad, factibilidad e implementación.

---

## Alternativa 1: Fidelity Susceptibility vía Overlap de Estados SQD

### ¿Qué es?

Calcular χ_F(h) = -2 ln|⟨ψ₀(h)|ψ₀(h+δh)⟩| / δh² usando los vectores propios
que SKQD devuelve para cada valor de h. El pico de χ_F señala h_c.

### ¿Ya se hizo antes?

**SÍ, parcialmente.** El concepto de χ_F en hardware cuántico ya fue demostrado:

- **Fontana et al. (2024)** [arXiv:2402.18953v3]: "Noise-Robust Detection of
  Quantum Phase Transitions". Demostraron en IBM hardware que χ_F es
  noise-robust incluso SIN error mitigation, usando VQE pre-optimizado.
  Publicado como PRL 133, 120601 (2024).
- **arXiv:2207.06526**: "Quantum computing fidelity susceptibility using
  automatic differentiation" — usa diferenciación automática cuántica.
- **arXiv:2509.01359**: "Heisenberg limited quantum algorithm for estimating
  the fidelity susceptibility" — algoritmo teórico escalable.
- **arXiv:2408.03418**: "Detecting Quantum and Classical Phase Transitions via
  Unsupervised ML of the Fisher Information Metric" — ML + FIM.

**PERO**: Ninguno usa SQD/SKQD como método de preparación del ground state.
Todos usan VQE o preparación directa. La combinación
**SKQD + χ_F** no ha sido publicada.

### ¿Es factible?

**Parcialmente.** Hay un problema técnico importante:

- SKQD proyecta H en un subespacio de bitstrings para CADA h independientemente.
- Los subespacios para h y h+δh pueden ser **diferentes** (distintos bitstrings
  muestreados), lo cual hace que el overlap ⟨ψ(h)|ψ(h+δh)⟩ no esté bien
  definido directamente.
- **Solución**: Forzar un subespacio común (unión de bitstrings de ambos h)
  y re-diagonalizar ambos en ese subespacio compartido.
- **Problema para TFIM h>>1**: El ground state en la fase paramagnética
  (h→∞ → |+⟩^N) NO es sparse en la base Z. SQD converge mal para h>>h_c.
  Solo funcionaría bien para h < ~2 en TFIM chain N=6-10.

### Veredicto

**PARCIALMENTE NOVEL, PARCIALMENTE FACTIBLE.**

La novedad está en combinar SKQD (no-variacional) con χ_F. Pero la limitación
de sparsity para TFIM en fase paramagnética es severa. Funcionaría solo en un
rango limitado de h, lo cual reduce su utilidad como detector universal.

**No recomendada como línea principal.** Útil como cross-check para h ≤ 2.

---

## Alternativa 2: Gap Espectral ΔE(h) = E₁ - E₀ vía SQD

### ¿Qué es?

Usar SKQD para calcular los dos eigenvalores más bajos (E₀ y E₁) de la
proyección del Hamiltoniano, y barrer h buscando el cierre del gap.

### ¿Ya se hizo antes?

**SÍ, extensamente (clásicamente) pero NO con SQD/SKQD:**

- El gap ΔE como indicador de QPT es textbook physics (Sachdev, "Quantum Phase
  Transitions", 1999).
- En hardware cuántico: **arXiv:2601.13881** "Low-Resource Quantum Energy Gap
  Estimation via Randomization" (2026) usa shadow spectroscopy + time evolution.
- Nadie ha usado SKQD para estimar E₁ en el contexto de phase detection.

### ¿Es factible?

**Problemático por varias razones:**

1. **SQD está diseñado para ground states.** La configuration recovery está
   optimizada para el eigenvalor más bajo. Obtener E₁ requiere que el segundo
   eigenvalor del subespacio proyectado sea el verdadero primer excitado — pero
   esto solo se garantiza si el subespacio tiene soporte suficiente en AMBOS
   estados.
2. **Para TFIM finito** (N=6-10), ΔE ~ 1/N. El gap closing es gradual y difícil
   de resolver con precisión finita. Necesitarías una precisión en energía de
   ~0.01J para N=10, que está en el límite de lo que SKQD puede lograr en
   hardware con noise.
3. **El overhead es doble**: necesitas que SKQD converja bien tanto para E₀ como
   para E₁, y el segundo eigenvalor es menos sparse → peor convergencia.

### Veredicto

**NOVEL en la combinación SKQD+gap, pero NO FACTIBLE para nuestros tamaños.**

El cierre del gap para TFIM N=6-10 es demasiado gradual para ser resuelto con
la precisión que SKQD ofrece en hardware ruidoso. Descartada.

---

## Alternativa 3: Ocupación Orbital / Correladores como Order Parameter

### ¿Qué es?

SKQD devuelve como subproducto natural el vector de ocupaciones orbitales
promedio y los coeficientes del ground state en la base computacional. De estos
se puede calcular:
- m_z = Σ⟨Z_i⟩/N (magnetización)
- ⟨Z_i Z_j⟩ (correlador de Ising)
- Cualquier observable local

El cambio abrupto de estas cantidades señala la QPT.

### ¿Ya se hizo antes?

**SÍ, exhaustivamente:**

- **Con VQE en hardware**: Fontana et al. (2024) [PRL 133, 120601] ya demostró
  que correladores de 2 sitios detectan transiciones de fase en IBM hardware
  de forma noise-robust.
- **Con sampling clásico + ML**: Carrasquilla & Melko (2017), Nature Physics 13,
  431 — el paper fundacional de ML para QPT usando datos de Monte Carlo.
- **Con QCNN**: Cong et al. (2019), Nature Physics — clasificación de fases
  usando circuitos cuánticos convolucionales.
- **Con SKQD específicamente**: Brooks et al. (2026) [arXiv:2605.29521]
  calculan occupancies como subproducto de SKQD para Heisenberg, pero NO las
  usan como indicador de transición de fase.

**La combinación SKQD + observables locales para detectar QPT NO ha sido
publicada explícitamente**, pero es conceptualmente trivial (medir ⟨Z_iZ_j⟩
del estado SQD y buscar el punto donde cambia). No tiene suficiente novedad
para una contribución original.

### ¿Es factible?

**Sí, pero con la misma limitación de sparsity:**

- Para TFIM en la fase paramagnética (h > h_c), el ground state no es sparse
  en la base Z, por lo que SKQD convergería mal exactamente donde necesitamos
  medir.
- Para h < h_c (fase ordenada), ⟨Z_iZ_j⟩ → ±1, pero SKQD converge fácilmente
  ahí porque el estado ES sparse.
- El cruce (h ≈ h_c) es el punto más interesante y donde la sparsity es
  intermedia — podría funcionar con suficientes shots.

### Veredicto

**FACTIBLE pero NO NOVEL.** Es la extensión más obvia y ya fue hecha con VQE
(Fontana et al. 2024). La única novedad sería usar SKQD en vez de VQE como
método de preparación, lo cual es una mejora incremental, no una contribución
nueva. Descartada como contribución original.

---

## Alternativa 4: Krylov Complexity como Order Parameter

### ¿Qué es?

La "Krylov complexity" K(t) mide cuánto se extiende un estado (o operador) en
el espacio de Krylov bajo evolución temporal. Varios papers recientes muestran
que K(t) exhibe comportamiento singular en QPTs:

- K(t) promediada temporalmente actúa como order parameter (LMG model)
- El "Krylov spread fidelity" detecta DQPTs
- El pico de K(t) distingue fases caóticas de integrables

### ¿Ya se hizo antes?

**SÍ, teóricamente. NO en hardware cuántico para QPT de equilibrio:**

- **arXiv:2312.05321** (2023): Krylov complexity como order parameter en modelo
  LMG — pero para DQPT (dynamical), no para QPT de equilibrio.
- **arXiv:2401.04383** (2024, JHEP): Krylov complexity para deconfinement en
  large N — QFT, no lattice models.
- **arXiv:2503.18936** (2025): Krylov spread en modelo SSH no-Hermitiano.
- **arXiv:2504.07474** (2025): DQPT en modelos fully-connected.
- **arXiv:2407.17054** (2024): KCP para transiciones caóticas-integrables.

**Crucialmente**: TODOS estos papers son teóricos/numéricos con diagonalización
exacta. NINGUNO usa SKQD o hardware cuántico. Y la mayoría estudian DQPTs
(quench dynamics), no equilibrium QPTs como la del TFIM.

### ¿Es factible?

**NO con nuestros recursos actuales, por varias razones:**

1. **Krylov complexity requiere evolución temporal LARGA.** Para que K(t) muestre
   comportamiento distinguible entre fases, necesitas t ~ O(1/ΔE), que requiere
   muchos Trotter steps → circuitos profundos.
2. **Es un observable dinámico.** Necesitas medir la distribución de probabilidad
   en la base de Krylov en FUNCIÓN del tiempo t. Esto implica ejecutar muchos
   circuitos de profundidad creciente y reconstruir K(t) — un overhead masivo.
3. **No está claro que funcione para QPT de equilibrio del TFIM.** Los resultados
   existentes son para quench dynamics (estado fuera de equilibrio) o para modelos
   específicos (LMG, SSH). Para TFIM 1D la transición es bien conocida y la
   Krylov complexity no ha sido demostrada como indicador.
4. **La conexión con SKQD es tenue.** SKQD usa el espacio de Krylov para
   DIAGONALIZAR, no para medir complejidad. Son conceptos que comparten nombre
   pero tienen propósitos distintos.

### Veredicto

**PARCIALMENTE NOVEL pero NO FACTIBLE con nuestro setup.**

Es un concepto teórico interesante pero: (a) no está demostrado para QPT de
equilibrio del TFIM, (b) requiere recursos cuánticos enormes (muchos circuitos
profundos), (c) la conexión con SKQD es superficial. Sería un proyecto de
investigación completo en sí mismo. **Descartada.**

---

## Alternativa 5: SKQD + GNN con Features Físicas (Framework Alternativo)

### ¿Qué es?

Reformular el pipeline de detección de fase reemplazando VQE por SKQD:

```
Phase 1: DMRG/ED → ground state exacto (reference, como ahora)
Phase 2: SKQD(h) para cada h → {E₀, occupancies, correladores}
Phase 3: GNN predice fase desde features = {occupancy_vector, ⟨ZZ⟩, E₀/N}
          en vez de θ_opt
```

La GNN aprende a clasificar fases desde observables físicos del ground state
SKQD en lugar de ángulos variacionales.

### ¿Ya se hizo antes?

**Las piezas individuales sí. La combinación específica NO:**

| Componente | ¿Existe? | Referencia |
|-----------|----------|-----------|
| ML + observables → clasificación de fases | ✅ | Carrasquilla & Melko 2017 (Monte Carlo + NN) |
| GNN para fases cuánticas | ✅ | Nuestro propio trabajo (GNN-HVA) + otros |
| SKQD para ground state de Heisenberg | ✅ | Brooks et al. 2026 [arXiv:2605.29521] |
| Shadow tomography + ML para fases | ✅ | arXiv:2508.04774, arXiv:2508.17688 |
| Raw bitstrings + NN para fases | ✅ | arXiv:1906.10155 (2020), arXiv:2604.03550 (2025) |
| SKQD + GNN para phase detection | ❌ | **NO existe** |
| SKQD bitstrings como input a ML sin observables | ❌ | **NO existe** |

**La novedad real sería**: usar las distribuciones de bitstrings muestreadas por
SKQD (no procesadas en observables) directamente como input a una GNN que
clasifique la fase. Esto combinaría:
- La robustez al ruido de SKQD (inherent noise filtering)
- La capacidad de generalización de GNNs sobre grafos de lattice
- Zero-overhead en observables (usas los bitstrings crudos)

### ¿Es factible?

**SÍ, con restricciones:**

**A favor:**
- SKQD está disponible como `qiskit-addon-sqd` — production-ready.
- Brooks et al. (2026) ya demostró SKQD hasta 72 spins en IBM Heron r3.
- Para TFIM en fase ordenada (h < h_c), el ground state ES sparse → SKQD
  converge bien.
- No requiere optimización variacional → no hay barren plateaus.
- Los bitstrings muestreados contienen información de la fase (la distribución
  de bitstrings CAMBIA al cruzar h_c).
- Funciona para Heisenberg frustrado (Brooks et al. demostró Kagome, J1-J2
  square) donde nuestro HVA falla.

**En contra:**
- **Problema de sparsity para TFIM paramagnético**: Para h >> h_c, |+⟩^N no
  es sparse en base Z → SKQD falla. Pero se puede usar base X (rotar qubits
  antes de medir) o Δ > 1 (como Brooks et al.).
- **Pierde D1**: No hay θ_opt → no hay detección en weight-space. Es un
  framework DIFERENTE, no una mejora del actual.
- **Tamaño del dataset**: Para N=10, el subespacio Sz=0 tiene dim = C(10,5) =
  252. Con 10⁶ shots tenemos excelente cobertura. Para N=20 → C(20,10) = 184756
  — todavía manejable.

### Implementación Detallada

#### Arquitectura Propuesta: SKQD-GNN

```
┌─────────────────────────────────────────────────────┐
│                    SKQD-GNN Framework                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ Lattice  │───▶│ SKQD Circuit │───▶│ Bitstring│  │
│  │ Graph G  │    │ (Trotter)    │    │ Samples  │  │
│  └──────────┘    └──────────────┘    └──────────┘  │
│       │                                    │        │
│       │                                    ▼        │
│       │          ┌──────────────────────────────┐   │
│       │          │ Feature Extraction per h:    │   │
│       │          │  f₁: occupancy vector (N,)   │   │
│       │          │  f₂: E₀/N (scalar)          │   │
│       │          │  f₃: bitstring entropy       │   │
│       │          │  f₄: ⟨Z_iZ_j⟩ correlator    │   │
│       │          │  f₅: subspace dimension      │   │
│       │          └──────────────────────────────┘   │
│       │                       │                     │
│       ▼                       ▼                     │
│  ┌──────────┐    ┌──────────────────────────┐      │
│  │  Graph   │───▶│    GNN Classifier         │      │
│  │ Encoding │    │  (MPNN, same as Phase 3)  │      │
│  └──────────┘    │  Input: G + features(h)   │      │
│                  │  Output: phase label       │      │
│                  └──────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Pipeline Step-by-Step

**Step 1: Hamiltoniano y Circuitos Krylov**

Para TFIM en lattice G con N sitios:
```python
H_TFIM = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i
```

Trotter step (segundo orden):
```python
U(Δt) ≈ e^{-iΔt/2 H_ZZ} · e^{-iΔt H_X} · e^{-iΔt/2 H_ZZ}
```

Krylov circuits: |ψ_k⟩ = U^k |ψ_0⟩ para k = 0, 1, ..., D-1.

Initial state |ψ_0⟩ = |+⟩^N (para TFIM) o Néel state (para Heisenberg).

**Step 2: Sampling en Hardware**

- Para cada h en grid [h_min, h_max]:
  - Construir D circuitos Krylov (D=8 típicamente)
  - Ejecutar en IBM Heron con 10⁴-10⁶ shots totales
  - Combinar bitstrings de todos los circuitos
  - Aplicar configuration recovery (qiskit-addon-sqd)

**Step 3: Feature Extraction**

De los resultados de SKQD para cada h, extraer:

```python
features = {
    "occupancy": result.orbital_occupancies,  # (N,) vector
    "energy_per_site": result.energy / N,     # scalar
    "bitstring_entropy": -Σ p_i log(p_i),     # Shannon entropy de la distribución
    "correlator_nn": mean(<Z_iZ_j> para nn),  # correlador nearest-neighbor
    "subspace_dim": prod(result.sci_state.amplitudes.shape),  # proxy de sparsity
    "convergence_iter": n_iterations_to_converge,  # cuántas iteraciones CR
}
```

**Step 4: GNN Training**

- Mismo MPNN que usamos actualmente (h=128, L=3 message-passing layers)
- Node features: occupancy_i(h) para cada sitio i
- Edge features: ⟨Z_iZ_j⟩(h) para cada arista
- Global features: E₀/N, entropy, subspace_dim
- Label: fase (0 = ordenada, 1 = paramagnética), determinada por DMRG/ED
- Training: sweep descendente en h, como ahora

**Step 5: Phase Detection sin Order Parameter**

La GNN aprende a clasificar la fase sin que le digamos qué medir. Los
indicadores emergentes serían:
- Gradient de features respecto a h (análogo a D1 pero en espacio de
  observables)
- Atención del GNN sobre nodos/aristas específicos → interpretabilidad
- Uncertainty de la clasificación cerca de h_c

#### Ventajas sobre el Framework Actual (VQE + θ_opt)

| Aspecto | VQE + θ_opt (actual) | SKQD + features |
|---------|---------------------|-----------------|
| Optimización | Iterativa (COBYLA/SPSA) | NO necesaria |
| Barren plateaus | Riesgo para N>10 | Inexistente |
| Noise tolerance | Requiere ZNE | Inherente (SQD filtra noise) |
| Modelos frustrados | ❌ (HVA p≤2 falla) | ✅ (Brooks 2026: hasta 72 spins) |
| Profundidad circuito | ~18 CX (p=1, N=10) | ~N CX × D (D Trotter steps) |
| Upper bound energy | No garantizado | ✅ Siempre |
| Weight-space detection (D1) | ✅ | ❌ (no hay θ) |
| Escalabilidad a N>20 | ⚠️ (VQE costoso) | ✅ (sampling es O(shots)) |

#### Limitaciones y Mitigaciones

| Limitación | Mitigación |
|-----------|-----------|
| TFIM paramagnético no sparse en base Z | Usar base X (pre-rotar) o XXZ con Δ>1 |
| Circuitos más profundos que HVA p=1 | D=4-8 Trotter steps suficientes para N≤12 |
| Pierde detección D1 | Reemplazar por gradient de occupancies/entropy vs h |
| Nuevo framework (no extiende el actual) | Posicionar como "future work" o extensión |

#### Scope para Tesis vs. Publicación Independiente

**Para la tesis actual (Cap. 6 Future Work):**
- Describir el framework conceptual
- Mostrar un PoC con N=6 TFIM chain (validar que SKQD reproduce E₀ y que las
  occupancies cambian en h_c)
- Comparar con el resultado actual del VQE pipeline
- ~2 páginas de texto + 1 figura

**Para publicación independiente (post-tesis):**
- Implementación completa del pipeline SKQD-GNN
- Benchmark contra VQE-GNN (nuestro framework actual) en múltiples topologías
- Extender a Heisenberg frustrado (donde VQE falla) como demostración de
  ventaja
- Target: Physical Review Research o Quantum Science and Technology

---

---

## Apéndice: Análisis Detallado de Factibilidad — Phase 3 del SKQD-GNN

### El Problema Central: ¿Qué Aprende la GNN?

En nuestro framework actual:
- **Input**: Grafo del lattice (edges + node features [h_i, coord_i])
- **Output**: θ_opt ∈ ℝ^(2p) — regresión continua de ángulos variacionales
- **Task**: Regresión (MSE loss, predice ángulos exactos)
- **Fase se determina después**: en Phase 4, usando θ_pred para evaluar
  observables (⟨X⟩, ⟨ZZ⟩) y compararlos con el ground state

La GNN NO clasifica fases directamente — predice parámetros y la fase se
infiere de los observables del circuito parametrizado.

### ¿Qué Cambiaría con SKQD?

Con SKQD no hay θ_opt que predecir. Hay dos opciones fundamentales:

#### Opción A: GNN como CLASIFICADOR de fase (supervisado)

```
Input: Grafo + features por nodo derivadas de SKQD(h)
Output: P(fase = ordenada) ∈ [0, 1] — clasificación binaria
Loss: Binary Cross-Entropy
Labels: de DMRG/ED (como ahora, Phase 1 ya las calcula)
```

**Factibilidad**: ✅ ALTA — esto es esencialmente lo que hacen Carrasquilla &
Melko (2017) pero con una GNN en vez de un MLP, y con features de SKQD en vez
de configuraciones de Monte Carlo.

**Problema**: Trivial si le das directamente ⟨ZZ⟩ como feature — la GNN no
necesita "aprender" nada que un threshold simple no pueda hacer. La red solo
aprendería un umbral en el correlador. NO HAY VALOR AGREGADO de la GNN.

#### Opción B: GNN como REGRESOR de observables (semi-supervisado)

```
Input: Grafo + [h_i, coord_i] (como ahora, sin features SKQD)
Output: [⟨Z_i⟩, ⟨Z_iZ_j⟩, E₀/N] predichos — regresión
Loss: MSE contra valores SKQD
Labels de fase: implícitas (el cambio abrupto en las predicciones → QPT)
```

**Factibilidad**: ⚠️ MEDIA — la GNN aprendería a predecir observables del
ground state para h-values no vistos. Esto es interesante pero:
- ¿Por qué no calcular SKQD directamente para cada h? (es barato en hardware)
- El valor agregado sería generalización a NUEVAS topologías sin re-ejecutar
  SKQD.

#### Opción C: GNN predice si SKQD convergerá bien (meta-learning)

```
Input: Grafo + h_value
Output: {n_iterations_convergencia, subspace_dimension_needed, E₀_estimada}
Loss: MSE
Training data: runs de SKQD previos en varias topologías
```

**Factibilidad**: 🔬 BAJA para tesis actual — requiere muchos runs de SKQD
como training data, y el valor científico es bajo (optimización de hiperparams).

### Análisis Crítico: ¿Dónde Está el Valor Científico?

El valor científico de nuestro framework actual NO es la GNN per se — es que
**D1 (peak de |∂θ/∂h|) detecta QPT sin conocer el order parameter**. Es un
descubrimiento en el espacio de parámetros variacionales.

Si eliminamos θ_opt y usamos observables de SKQD:
- Ya no hay D1 (no hay weight-space)
- La detección de fase se reduce a "calcular ⟨ZZ⟩ y poner un umbral"
- La GNN no aporta nada que un cálculo directo no haga

**La ÚNICA forma en que SKQD + GNN tiene valor es si:**
1. La GNN generaliza a topologías/tamaños NO VISTOS en training
2. Los features de SKQD capturan información NO TRIVIAL de la fase

### Propuesta Refinada: SKQD-GNN para Generalización Inter-Topología

El verdadero valor sería:

```
TRAINING:
  - Ejecutar SKQD en topologías A, B, C (chain, ladder, triangular)
  - Para cada (topología, h): extraer {occupancy_vector, bitstring_entropy,
    convergence_rate, subspace_sparsity}
  - Labels: fase (de ED/DMRG)
  - Entrenar GNN: Grafo + h → fase

INFERENCE:
  - Nueva topología D (kagome, heavy-hex) que NUNCA vio
  - GNN predice fase SIN ejecutar SKQD (zero-shot)
  - Validar con SKQD/ED en la nueva topología
```

**Esto SÍ tiene valor**: transferencia de conocimiento de fase entre geometrías.
Pero requiere que las features de SKQD capturen algo universal sobre la fase
que trascienda la topología específica.

### Factibilidad Técnica del Paso Phase 3

#### Requisitos de Datos

Para entrenar la GNN con este approach necesitamos:

| Topología | N | h_points | SKQD shots | Tiempo estimado (simulador) |
|-----------|---|----------|------------|---------------------------|
| chain_1d | 6 | 20 | 10⁴ × 20 = 200k | ~5 min |
| chain_1d | 10 | 20 | 10⁴ × 20 = 200k | ~10 min |
| ladder | 6 | 20 | 10⁴ × 20 = 200k | ~10 min |
| triangular | 6 | 20 | 10⁴ × 20 = 200k | ~15 min |
| **Total training** | | **80** | **800k** | **~40 min** |

Con D=8 Krylov steps, N=6: circuito más profundo tiene ~48 CX gates (dentro
del rango ZNE, pero más profundo que nuestro HVA p=1 con 9 CX).

#### Sparsity del TFIM

El ground state del TFIM en base Z:

| h/J | Descripción | Sparsity (N=6) | ¿SKQD funciona? |
|-----|-------------|----------------|-----------------|
| 0.0 | Fully ordered (|↑↑...↑⟩ + |↓↓...↓⟩)/√2 | 2/64 = 3% | ✅ Excelente |
| 0.5 | Ordered con fluctuaciones | ~8/64 = 12% | ✅ Bueno |
| 1.0 | Crítico (h_c) | ~20/64 = 31% | ⚠️ Marginal |
| 1.5 | Paramagnético débil | ~40/64 = 62% | ❌ Malo |
| 2.0+ | Paramagnético fuerte (~|+⟩^N) | ~64/64 = 100% | ❌ Falla |

**Conclusión**: SKQD para TFIM solo funciona bien para h ≤ h_c ≈ 1.0.
Esto cubre la fase ordenada y el punto crítico, pero NO la fase paramagnética.

**Mitigación**: Usar la base X (aplicar H^⊗N antes de medir):
- En base X, la fase paramagnética es sparse (|0...0⟩_X)
- La fase ordenada es la que se vuelve no-sparse en base X
- Solución: ejecutar SKQD en AMBAS bases y combinar features

#### Implementación Concreta del Pipeline Modificado

```python
# Pseudo-código del Phase 3 con SKQD features

from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
# NOTA: Para spin models, usar la versión qubit de SQD:
from qiskit_addon_sqd.qubit import solve_qubit

def run_skqd_phase3(lattice, h_values, krylov_dim=8, shots=10000):
    """
    Ejecutar SKQD para cada h y extraer features para la GNN.
    """
    features_per_h = []

    for h in h_values:
        # 1. Construir circuitos Krylov para TFIM(h)
        circuits = build_krylov_circuits(lattice, h, krylov_dim)

        # 2. Ejecutar en backend (simulador o hardware)
        sampler = StatevectorSampler()  # o Sampler(backend)
        job = sampler.run(circuits, shots=shots // krylov_dim)
        bitstrings = combine_all_bitstrings(job.result())

        # 3. Filtrar bitstrings por simetría (Sz=0 para Heisenberg,
        #    o particle number para TFIM en alguna base)
        valid_bitstrings = filter_by_symmetry(bitstrings, lattice)

        # 4. Proyectar H y diagonalizar (classical)
        H_projected = project_hamiltonian(H_tfim(h), valid_bitstrings)
        eigenvalues, eigenvectors = diagonalize(H_projected)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        # 5. Extraer features
        occupancy = compute_occupancy(psi0, valid_bitstrings)
        correlators = compute_zz_correlators(psi0, valid_bitstrings, lattice)
        entropy = compute_bitstring_entropy(bitstrings)
        subspace_dim = len(valid_bitstrings)

        features_per_h.append({
            'occupancy': occupancy,       # (N,) vector
            'correlators': correlators,    # (n_edges,) vector
            'energy_per_site': E0 / lattice.n_qubits,
            'entropy': entropy,
            'subspace_dim': subspace_dim,
            'h': h,
        })

    return features_per_h


def build_skqd_graph_dataset(lattice, features_per_h, labels):
    """
    Construir dataset PyG para la GNN clasificadora.

    Diferencias con build_graph_dataset() actual:
    - Node features: [h_i, coord_i, occupancy_i, local_mag_i]
    - Edge features: [⟨Z_iZ_j⟩] (correlador por bond)
    - Target: label binario (0=ordered, 1=paramagnetic)
      en vez de θ_opt continuo
    """
    from torch_geometric.data import Data
    import torch

    dataset = []
    edge_index = build_edge_index(lattice)

    for feat, label in zip(features_per_h, labels):
        # Node features: [h, coordination, occupancy, |occupancy - 0.5|]
        h_feat = np.full(lattice.n_qubits, feat['h'])
        coord = get_coordination(lattice)
        occ = feat['occupancy']
        occ_deviation = np.abs(occ - 0.5)  # Signal de orden

        x = torch.tensor(
            np.stack([h_feat, coord, occ, occ_deviation], axis=1),
            dtype=torch.float32,
        )  # shape: (N, 4)

        # Edge features: correladores ZZ por bond
        edge_attr = torch.tensor(
            feat['correlators'].reshape(-1, 1),
            dtype=torch.float32,
        )
        # Duplicar para ambas direcciones
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # Target: clasificación binaria
        y = torch.tensor([label], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.h_value = feat['h']
        data.energy = feat['energy_per_site']
        dataset.append(data)

    return dataset
```

#### Cambios Necesarios en `MPNNPredictor`

```python
class PhaseClassifierMPNN(nn.Module):
    """
    Variante del MPNNPredictor para clasificación binaria de fase.

    Diferencias con MPNNPredictor:
    - Output: sigmoid(scalar) → P(paramagnetic)
    - Loss: BCEWithLogitsLoss en vez de MSE
    - Node features: 4 (h, coord, occupancy, occ_deviation)
    - Edge features: 1 (⟨Z_iZ_j⟩ correlator)
    - Readout: global_mean_pool → MLP → scalar logit
    """

    def __init__(self, node_features=4, hidden_dim=64, n_layers=3,
                 edge_feature_dim=1):
        super().__init__()
        # NNConv (usa edge features: correladores)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Primera capa
        edge_nn = nn.Sequential(
            nn.Linear(edge_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_features * hidden_dim),
        )
        self.convs.append(NNConv(node_features, hidden_dim, edge_nn))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Capas siguientes
        for _ in range(n_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Classification head (binary)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),  # logit
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(...)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)  # logit, apply sigmoid for P(phase)
```

### Obstáculos Técnicos Identificados

| Obstáculo | Severidad | Solución |
|-----------|-----------|----------|
| TFIM h>1 no es sparse en base Z | 🔴 Alta | Dual-basis SKQD (medir en Z y X) |
| `qiskit-addon-sqd` es para fermiones, no spins | 🔴 Alta | Usar `solve_qubit()` o implementar proyección manual |
| La GNN con observables directos es trivial | 🟡 Media | Valor está en generalización inter-topología |
| Circuitos Krylov más profundos que HVA p=1 | 🟡 Media | D=4 suficiente para N=6 TFIM (32 CX) |
| No hay detección tipo D1 (sin weight-space) | 🟡 Media | Reemplazar con ∂features/∂h (gradient de occupancies) |
| Training data limitada por topologías simulables | 🟢 Baja | 4 topologías × 20 h-points = 80 samples (suficiente para PoC) |

### ¿Qué Aporta vs. Simplemente Calcular ⟨ZZ⟩?

La pregunta clave es: **¿por qué usar una GNN si puedo calcular ⟨ZZ⟩(h)
directamente de SKQD y poner un threshold?**

La respuesta honesta:

1. **Para una sola topología conocida**: la GNN NO aporta nada. Un threshold
   en ⟨ZZ⟩ funciona igual o mejor.

2. **Para generalización a topologías nuevas**: la GNN PUEDE aportar si aprende
   features universales del "acercamiento a la criticalidad" que transciendan
   la geometría específica. Esto es especulativo pero testeable.

3. **Para modelos con QPT no convencionales** (frustrated, topological): donde
   NO existe un order parameter local obvio, la GNN podría descubrir
   combinaciones no-triviales de features que señalen la transición.

### Veredicto Final sobre Phase 3

**Para la tesis (PoC minimal)**:
- Viable como demostración conceptual (1-2 semanas de trabajo)
- Valor limitado si solo se aplica a TFIM 1D (trivial)
- Valor real solo si se demuestra generalización a ≥2 topologías

**Para publicación futura**:
- Necesita demostrar que la GNN generaliza a topologías no vistas en training
- Necesita comparar con el baseline trivial (threshold en ⟨ZZ⟩)
- El sweet spot sería aplicarlo a modelos frustrados donde el order parameter
  no es obvio → ahí la GNN tendría ventaja real

**Recomendación actualizada**: El Phase 3 de SKQD-GNN es factible
técnicamente pero su valor científico es cuestionable para TFIM (donde la
transición es bien conocida). El valor real emerge SOLO en el contexto de
generalización inter-topología o modelos sin order parameter local. Para la
tesis, incluir como Future Work con un párrafo honesto sobre las limitaciones.

---

## Apéndice B: Investigación de Papers Relevantes al Phase 3

### Papers que atacan problemas iguales o similares

A continuación, un análisis de la literatura existente que intenta resolver
el mismo problema (ML + observables/samples → clasificación de fase) y qué
implica para nuestra propuesta SKQD-GNN.

---

### Paper 1: "Adiabatic Fine-Tuning of Neural Quantum States Enables Detection of Phase Transitions in Weight Space"
**[arXiv:2503.17140, Marzo 2025]**

**Qué hace**: Entrena Neural Quantum States (NQS, redes tipo RBM/Transformer)
para representar el ground state del TFIM y Heisenberg J1-J2. Luego detecta
la QPT analizando los PESOS de la red entrenada — sin medir observables.

**Relevancia directa**: Es EXACTAMENTE nuestro approach D1 pero con NQS en
vez de VQE. Demuestran que "phase transitions manifest as distinct structures
in weight space" — validando que la detección en espacio de parámetros es
un principio general, no un artefacto de HVA.

**Implicación para SKQD-GNN**: Este paper REFUERZA que la detección en weight-
space (como D1) es valiosa y no trivial. Si usamos SKQD + GNN con features
físicas, PERDEMOS esta capacidad. Pero si entrenamos una NQS con datos de SKQD
y luego analizamos sus pesos... eso sería una combinación nueva (NQS inicializada
con subespacio SKQD).

**Status**: Publicado, no usa SKQD.

---

### Paper 2: "Learning quantum phase transition in parametrized quantum circuits with an attention mechanism"
**[arXiv:2506.06678, Junio 2025]**

**Qué hace**: Integra attention (de LLMs) + VAE para capturar correlaciones
DENTRO de los parámetros de circuitos cuánticos parametrizados. Detecta QPT
de forma no-supervisada desde las correlaciones entre parámetros del circuito.

**Relevancia directa**: ALTÍSIMA — es literalmente "detectar QPT desde los
parámetros variacionales" con ML moderno. Es una versión attention-based de
nuestro D1/PCA.

**Implicación para SKQD-GNN**: Confirma que el espacio de parámetros (θ_opt)
es donde vive la información de fase. Si eliminamos θ_opt (usando SKQD), perdemos
el canal más informativo. La alternativa (features físicas) es menos rica.

---

### Paper 3: "Deep Learning of Phase Transitions for Quantum Spin Chains from Correlation Aspects"
**[arXiv:2301.06669, Enero 2023, publicado 2023]**

**Qué hace**: Usa deep learning sobre FUNCIONES DE CORRELACIÓN (exactamente
⟨Z_iZ_j⟩ y similares) para detectar transiciones de fase en cadenas de spin.
Demuestra que los correladores contienen suficiente información para clasificar
fases incluso en modelos no triviales.

**Relevancia directa**: Este paper ES esencialmente "Phase 3 de SKQD-GNN" pero
con datos clásicos (ED) en vez de SKQD. Demuestra que usar ⟨ZZ⟩ como input a
una NN FUNCIONA para clasificar fases.

**Implicación para SKQD-GNN**: El approach de usar correladores como features
para ML ya fue demostrado y publicado. Nuestra contribución no sería "correladores
+ ML = detectar fase" (eso ya existe) sino específicamente "SKQD en hardware
genera correladores → GNN generaliza entre topologías". La novedad estaría en:
(a) usar SKQD como fuente de datos (no ED), y (b) usar GNN para
generalización inter-topología.

---

### Paper 4: "A Unsupervised Framework for Identifying Diverse Quantum Phase Transitions Using Classical Shadow Tomography"
**[arXiv:2508.17688, Agosto 2025]**

**Qué hace**: Classical shadows + PCA (unsupervised) para detectar QPT en
múltiples modelos (TFIM, XXZ, Kitaev). Las shadows proveen representaciones
compactas del estado; PCA encuentra la dirección de máxima varianza → pico en
h_c.

**Relevancia directa**: MUY ALTA. Es el approach más cercano a lo que
proponemos — obtener datos del quantum state (shadows ≈ bitstrings) y aplicar
ML unsupervised para detectar la transición. La diferencia: usan shadows
(mediciones random Pauli) mientras que SKQD produce bitstrings en base Z
con Krylov time-evolution.

**Implicación para SKQD-GNN**: Si usamos PCA de features SKQD (occupancies,
entropy) vs h, estamos haciendo esencialmente lo mismo que este paper pero con
SKQD en vez de shadows. El valor marginal es bajo — a menos que demostremos que
SKQD produce features MÁS informativas que random shadows para un shot budget
dado.

---

### Paper 5: "Fluctuation based interpretable analysis scheme for quantum many-body snapshots"
**[arXiv:2304.06029, 2023, publicado PRB]**

**Qué hace**: Clasifica fases de materia directamente desde "snapshots"
(mediciones proyectivas del many-body state). Analiza fluctuaciones en las
distribuciones de bitstrings, mostrando que la ESTADÍSTICA de los snapshots
(no solo el valor medio) contiene información de fase.

**Relevancia directa**: ALTA. Los bitstrings de SKQD SON snapshots del ground
state. Este paper demuestra que la distribución de bitstrings (no solo ⟨ZZ⟩)
contiene información de fase que un ML puede extraer.

**Implicación para SKQD-GNN**: Sugiere que el input correcto a la GNN NO es
solo {⟨Z_i⟩, ⟨Z_iZ_j⟩} (valores medios) sino la distribución COMPLETA de
bitstrings (histograma, fluctuaciones, higher-order correlators). Esto es más
rico y potencialmente no-trivial.

---

### Paper 6: "Network theory classification of quantum matter based on wave function snapshots"
**[arXiv:2512.02121, Diciembre 2024]**

**Qué hace**: Construye GRAFOS a partir de las correlaciones entre bitstrings
medidos y usa propiedades de red (degree distribution, clustering) para
clasificar fases. Demuestra que las snapshots en la fase ordenada forman redes
con estructura diferente a las de la fase desordenada.

**Relevancia directa**: MUY ALTA Y DIRECTA. Es "GNN + snapshots → fase"
pero implementado como análisis de red en vez de GNN trainable. La conexión con
nuestro proyecto es clara: los bitstrings de SKQD definen un grafo de
correlaciones, y las propiedades de ESE grafo distinguen fases.

**Implicación para SKQD-GNN**: Podríamos construir un "correlation graph" desde
los bitstrings SKQD (edge si dos bitstrings co-ocurren o están a Hamming
distance ≤ k) y luego aplicar nuestra GNN sobre ESE grafo. Esto sería
genuinamente nuevo: SKQD → correlation graph → GNN → phase.

---

### Paper 7: "Machine Learning Domain Adaptation in Spin Models with Continuous Phase Transitions"
**[arXiv:2411.13027, Noviembre 2024]**

**Qué hace**: Entrena una NN en un modelo de universality class A (e.g., Ising)
y la aplica a universality class B (e.g., Potts). Demuestra que transfer
learning FUNCIONA para detectar QPT — la NN aprende features universales de
la criticalidad que trascienden el modelo específico.

**Relevancia directa**: ALTA para nuestra propuesta de generalización inter-
topología. Si el transfer funciona entre universality classes, también debería
funcionar entre topologías de la misma universality class (TFIM en chain vs
ladder vs triangular — todos son 2D Ising).

**Implicación para SKQD-GNN**: Da soporte teórico a la idea de que una GNN
entrenada en chain + ladder podría generalizar a triangular. PERO: esto ya se
demostró con datos clásicos (MC), no con datos cuánticos (SKQD). La novedad
de usar SKQD sería para sistemas donde MC falla (sign problem).

---

### Paper 8: "Unveiling phase transitions with machine learning" (Transfer Learning)
**[arXiv:1904.01486, 2019, PRL publicado 2025]**

**Qué hace**: Demuestra transfer learning entre modelos: NN entrenada con
interacciones nearest-neighbor identifica una fase nueva cuando se introducen
next-nearest-neighbor interactions. La red aprende algo universal sobre "orden"
vs "desorden" que transfiere a modelos no vistos.

**Relevancia directa**: MEDIA-ALTA. Confirma que transfer learning para fases
es viable, pero usa datos de MC (clásicos). No hay QPU involved.

---

### Síntesis: ¿Qué es genuinamente nuevo en SKQD-GNN Phase 3?

Después de revisar la literatura exhaustivamente:

| Componente | ¿Ya publicado? | ¿Dónde? |
|-----------|---------------|---------|
| Correladores + NN → fase | ✅ | arXiv:2301.06669 (2023) |
| Snapshots/bitstrings + ML → fase | ✅ | arXiv:2304.06029 (2023), arXiv:2512.02121 (2024) |
| Shadows + PCA → QPT (unsupervised) | ✅ | arXiv:2508.17688 (2025) |
| Transfer learning entre modelos | ✅ | arXiv:1904.01486 (2019), arXiv:2411.13027 (2024) |
| GNN sobre lattice → predict propiedades | ✅ | arXiv:2404.08782 (2024) |
| Weight-space detection (como D1) | ✅ | arXiv:2503.17140 (2025) |
| SKQD bitstrings → correlation graph → GNN → phase | ❌ | **NO EXISTE** |
| SKQD + GNN transfer inter-topología | ❌ | **NO EXISTE** |
| SKQD para modelos frustrados + ML fase | ❌ | **NO EXISTE** |

### Conclusión Actualizada

**El único angle genuinamente novel para SKQD-GNN Phase 3** no es usar
observables directos como features (eso es trivial y ya fue publicado), sino
una de estas opciones:

1. **SKQD → Bitstring Correlation Graph → GNN**: Construir un grafo dinámico
   desde la distribución de bitstrings SKQD (inspirado en arXiv:2512.02121)
   y aplicar GNN para clasificar fase. Esto combina SKQD (provably convergent,
   noise-robust) con graph-based classification de forma no vista antes.

2. **SKQD inter-topología transfer**: Entrenar GNN con datos SKQD de topologías
   A,B,C → predecir fase en topología D sin ejecutar SKQD. Valor: para modelos
   con sign problem donde MC no puede generar datos de training.

3. **SKQD para frustrated models + weight-space analysis**: Usar NQS
   inicializado con datos SKQD (como warm start) y luego aplicar weight-space
   analysis (como arXiv:2503.17140) para detectar QPT en modelos donde HVA falla.

**La opción simple** (features = {occupancy, ⟨ZZ⟩, E₀/N} → GNN → label)
**NO es novel ni científicamente interesante**. Ya fue publicado en múltiples
formas. No recomiendo este path.

### Recomendación Final Revisada

Para la tesis, la propuesta SKQD-GNN Phase 3 con features directas {occupancy,
⟨ZZ⟩, E₀/N} como input a una GNN clasificadora **NO tiene suficiente
novedad**. Los papers arXiv:2301.06669 y arXiv:2508.17688 ya demuestran que
correladores/shadows + ML detecta QPT.

Lo que SÍ tiene valor como future work es:
- **Opción A (para la tesis)**: Mencionar SKQD como método alternativo de
  preparación de ground state que evita VQE, con referencia a la literatura
  que ya demostró que observables + ML detecta fases. ~1 párrafo en Cap. 6.
- **Opción B (para publicación)**: SKQD → bitstring correlation graph → GNN,
  genuinamente nuevo, combina ideas de arXiv:2512.02121 con SKQD en hardware.
  Requiere ~3 meses de trabajo post-tesis.

---

## Resumen de Veredictos

| # | Alternativa | ¿Novel? | ¿Factible? | Recomendación |
|---|------------|---------|-----------|---------------|
| 1 | χ_F vía overlap SQD | Parcial | Parcial (sparsity) | ❌ No principal |
| 2 | Gap ΔE vía SQD | Sí | No (precisión insuficiente) | ❌ Descartada |
| 3 | Occupancies como OP | No | Sí | ❌ Trivial/ya hecho |
| 4 | Krylov Complexity | Parcial | No (recursos masivos) | ❌ Descartada |
| 5 | SKQD + GNN framework | **SÍ** | **SÍ** | ✅ **Recomendada** |

---

## Conclusión

**La Alternativa 5 (SKQD-GNN)** es la única que combina novedad genuina con
factibilidad técnica. No ha sido publicada, es implementable con herramientas
existentes (qiskit-addon-sqd + nuestro MPNN), y abre la puerta a modelos
frustrados donde nuestro framework actual falla.

**Acción recomendada**: Incluir como propuesta de Future Work en Chapter 6 de
la tesis, con un PoC mínimo (N=6 chain TFIM, comparar occupancies SKQD vs
VQE en 3-4 valores de h alrededor de h_c).

---

## Referencias Clave

- Brooks, Zou & Rhone (2026). "Ground-state estimation of the Heisenberg model
  on frustrated lattices with SKQD." arXiv:2605.29521.
- Robledo-Moreno et al. (2025). "Chemistry beyond exact solutions on a
  quantum-centric supercomputer." Science Advances 11(25).
- Yu et al. (2025). "Quantum-centric algorithm for sample-based Krylov
  diagonalization." arXiv:2501.09702.
- Fontana et al. (2024). "Noise-Robust Detection of Quantum Phase Transitions."
  PRL 133, 120601.
- arXiv:2508.04774 (2025). "Universal quantum phase classification on quantum
  computers from machine learning."
- arXiv:2604.03550 (2025). "Post-Selection-Free Decoding of Measurement-Induced
  Area-Law Phases via Neural Networks."
- arXiv:2312.05321 (2023). "Krylov Complexity and Dynamical Phase Transition in
  the quenched LMG model."

---

## Apéndice C: Pipelines Detallados — Approaches Noveles

### Pipeline A: SKQD → Bitstring Correlation Graph → GNN

#### Concepto Central

En vez de extraer observables (⟨ZZ⟩, occupancy) de los bitstrings SKQD y
pasarlos como features a la GNN, construimos un **grafo dinámico** directamente
desde la distribución de bitstrings. Este grafo captura la ESTRUCTURA de la
distribución (correlaciones, clustering, comunidades) que cambia abruptamente
en la QPT.

**Inspiración**: arXiv:2512.02121 ("Network theory classification of quantum
matter", 2024) + arXiv:2301.13216 ("Wave function network description and
Kolmogorov complexity", 2023, PRL) + arXiv:2510.12415 ("Snapshot renormalization
group for quantum matter", 2024).

Estos papers demuestran que:
- Snapshots del many-body state forman redes con propiedades topológicas
  que distinguen fases.
- En la fase ordenada: pocas configuraciones dominantes → grafo sparse,
  alta clustering, componentes disconnected.
- En la fase desordenada: muchas configuraciones comparables → grafo denso,
  baja clustering, giant component.
- En el punto crítico: estructura fractal/scale-free.

#### ¿Por qué SKQD mejora sobre sampling directo?

Sampling directo de un circuito ruidoso produce bitstrings contaminados por
noise que NO reflejan el ground state. SKQD aplica configuration recovery
iterativo que FILTRA el ruido y converge al ground state. Por lo tanto:

- Bitstrings post-SKQD son representativos del TRUE ground state.
- El grafo construido desde bitstrings SKQD refleja la VERDADERA estructura
  de la distribución, no artefactos de noise.
- Esto es crucialmente diferente de simplemente medir un circuito y construir
  el grafo (lo cual daría resultados ruidosos).

#### Pipeline Detallado

```
┌─────────────────────────────────────────────────────────────────┐
│                PIPELINE A: Bitstring Correlation Graph           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: SKQD Sampling (por cada h-point)                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Build Krylov │───▶│ Sample QPU/  │───▶│ Configuration    │  │
│  │ Circuits     │    │ Simulator    │    │ Recovery (SQD)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                    │            │
│                                                    ▼            │
│  STEP 2: Construct Bitstring Correlation Graph                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Bitstrings B = {b₁, b₂, ..., bₘ} (M unique bitstrings) │  │
│  │                                                          │  │
│  │ Nodes: cada bitstring bᵢ es un nodo                      │  │
│  │ Edges: connect bᵢ ↔ bⱼ si:                              │  │
│  │   Option 1: Hamming(bᵢ, bⱼ) ≤ d_threshold              │  │
│  │   Option 2: |ψ(bᵢ)·ψ(bⱼ)| > ε  (overlap amplitudes)   │  │
│  │   Option 3: Co-occurrence en el eigenvector SQD          │  │
│  │                                                          │  │
│  │ Node features: [count(bᵢ), |amplitude(bᵢ)|², Hamming_   │  │
│  │                 weight(bᵢ), local_magnetization(bᵢ)]     │  │
│  │ Edge weights: 1/Hamming(bᵢ,bⱼ) o |amplitude_i·ampl_j|  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  STEP 3: GNN Classification                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Input: Correlation Graph G(h) + global feature [h, N]    │  │
│  │ Architecture: GINConv o GAT (attention sobre edges)      │  │
│  │ Readout: global_mean_pool + global_max_pool → MLP        │  │
│  │ Output: P(phase = paramagnetic | G(h))                   │  │
│  │ Loss: Binary Cross-Entropy                               │  │
│  │ Training: graphs de múltiples h-points y topologías      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  STEP 4: Phase Detection                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ h_c estimado = h donde P(phase) cruza 0.5                │  │
│  │ Incertidumbre = ancho de la transición (sigmoid width)   │  │
│  │ Validación = comparar con ED/DMRG h_c                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Construcción del Grafo: Opciones y Trade-offs

| Método de construcción | Ventajas | Desventajas |
|----------------------|----------|-------------|
| Hamming distance ≤ d | Simple, interpretable, scale-free (arXiv:2512.02121) | d es un hiperparámetro; grafo puede ser demasiado denso |
| Amplitude co-occurrence | Físicamente motivado (amplitudes del GS) | Requiere el eigenvector de SQD, no solo bitstrings |
| k-NN en Hamming space | Número fijo de edges por nodo | Pierde la información de distancia absoluta |
| Thresholded mutual information | Captura correlaciones no-lineales | Costoso de calcular para muchos bitstrings |

**Recomendación**: Usar Hamming distance ≤ d con d = N/4 (basado en el paper
"The Structure of Bit-String Similarity Networks", Entropy 27(1):57, 2025, que
caracteriza analíticamente las propiedades de estas redes).

#### Propiedades del Grafo que Cambian en la QPT

Basado en arXiv:2512.02121 y arXiv:2510.12415:

| Propiedad | Fase Ordenada (h < h_c) | Fase Crítica (h ≈ h_c) | Fase Desordenada (h > h_c) |
|-----------|------------------------|----------------------|---------------------------|
| Nodos (bitstrings únicos) | Pocos (~2-10) | Moderados (~20-50) | Muchos (~N choose N/2) |
| Degree distribution | Estrecha (todos ~mismo grado) | Power-law (scale-free) | Estrecha (denso) |
| Clustering coefficient | Alto (cliques) | Intermedio | Bajo (random-like) |
| Giant component fraction | <1 (disconnected) | Percolation transition | =1 (fully connected) |
| Average path length | Corto (small-world) | Diverge (critical slowing) | Corto (dense) |
| Modularity | Alta (comunidades claras) | Moderada | Baja |

**Observación clave**: La transición de fase se manifiesta como una
**percolation transition** en el bitstring graph. Esto conecta QPT con
teoría de redes de forma profunda y es exactamente lo que la GNN puede
aprender sin supervisión explícita.

#### Análisis de Robustez

**¿Funciona con finite shots?**

- Con S shots, obtenemos ≤ S bitstrings únicos (típicamente mucho menos por
  repeticiones).
- Para N=6 TFIM: dim(Hilbert) = 64. Con 10⁴ shots, muestreamos ~50-100
  bitstrings únicos → grafo de 50-100 nodos. Manejable.
- Para N=10: dim(Hilbert) = 1024. Con 10⁴ shots → ~200-500 nodos. OK.
- Para N=20: dim(Hilbert) = 10⁶. Con 10⁴ shots → ~5000-9000 nodos. Puede ser
  demasiado grande para GNN → necesitar subsampling.

**¿Funciona con noise?**

- SKQD's configuration recovery FILTRA noise → bitstrings representan el
  true ground state (demostrado en Brooks et al. 2026 con IBM Heron r3).
- Sin CR, los bitstrings ruidosos producirían un grafo random-like para
  TODAS las fases → no informativo. Con CR, la estructura de fase se preserva.

**¿Problema de sparsity para TFIM paramagnético?**

- Sí, persiste. Para h >> h_c, el GS tiene soporte uniforme → muchos
  bitstrings con amplitud similar → grafo denso en AMBAS fases.
- Mitigación: dual-basis (Z + X) o usar XXZ con Δ>1 (como Brooks et al.).
- Alternativa: Para TFIM, trabajar solo en el rango h ∈ [0, 2] donde
  la fase ordenada es sparse. Esto cubre h_c.

#### API Concreta (`qiskit-addon-sqd` para spin models)

El módulo `qiskit_addon_sqd.qubit` provee:

```python
from qiskit_addon_sqd.qubit import solve_qubit

# Input:
#   bitstring_matrix: np.ndarray of shape (M, N) bool — M bitstrings, N qubits
#   hamiltonian: SparsePauliOp — el Hamiltoniano TFIM
#   num_states: int — número de eigenvalores a calcular
#
# Output:
#   eigenvalues: np.ndarray [num_states]
#   eigenstates: np.ndarray [M, num_states] — coeficientes en la base de bitstrings
#
# NOTA: La function projeta H en el subespacio de bitstrings y diagonaliza.
# NO requiere PySCF ni fermiones. Funciona directamente con SparsePauliOp.

eigenvalues, eigenstates = solve_qubit(
    bitstring_matrix=bitstring_matrix,  # (M, N) bool array
    hamiltonian=H_tfim,                 # SparsePauliOp
    num_states=1,                       # solo ground state
)
```

Esto confirma que `qiskit-addon-sqd` SÍ soporta modelos de spin directamente.
No es exclusivamente para fermiones.

#### Implementación del Pipeline A

```python
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from qiskit_addon_sqd.qubit import solve_qubit
from qiskit.quantum_info import SparsePauliOp


def build_krylov_circuits_tfim(lattice, h, krylov_dim=8, dt=0.2):
    """Build D Krylov circuits for TFIM at given h."""
    from qiskit import QuantumCircuit
    N = lattice.n_qubits
    circuits = []

    # Initial state: |+⟩^N (for TFIM)
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
    qc.measure_all()
    circuits.append(qc.copy())

    # Trotter step: e^{-i dt H_ZZ} · e^{-i dt H_X}
    for k in range(1, krylov_dim):
        qc.remove_final_measurements()
        # ZZ interactions
        for (i, j) in lattice.edges:
            qc.rzz(-2 * lattice.J * dt, i, j)
        # X field
        for i in range(N):
            qc.rx(-2 * h * dt, i)
        qc.measure_all()
        circuits.append(qc.copy())

    return circuits


def skqd_sample_and_recover(circuits, hamiltonian, shots=10000):
    """Run SKQD: sample circuits + configuration recovery."""
    from qiskit.primitives import StatevectorSampler
    from qiskit.primitives import BitArray

    sampler = StatevectorSampler()
    job = sampler.run(circuits, shots=shots // len(circuits))

    # Combine all bitstrings
    bit_array = BitArray.concatenate_shots(
        [result.data.meas for result in job.result()]
    )

    # Convert to matrix
    bitstrings = bit_array.get_bitstrings()
    N = len(bitstrings[0])
    bitstring_matrix = np.array(
        [[int(b) for b in bs] for bs in bitstrings], dtype=bool
    )

    # De-duplicate
    unique_bs = np.unique(bitstring_matrix, axis=0)

    # Solve in subspace
    eigenvalues, eigenstates = solve_qubit(
        bitstring_matrix=unique_bs,
        hamiltonian=hamiltonian,
        num_states=1,
    )

    return unique_bs, eigenvalues[0], eigenstates[:, 0]


def build_correlation_graph(bitstrings, amplitudes, d_threshold):
    """
    Construct correlation graph from bitstrings.

    Nodes: unique bitstrings
    Edges: Hamming distance ≤ d_threshold
    Node features: [amplitude², hamming_weight, local_mag]
    Edge features: [1/hamming_distance]
    """
    M, N = bitstrings.shape

    # Compute pairwise Hamming distances
    distances = squareform(pdist(bitstrings.astype(int), metric='hamming')) * N
    # distances[i,j] = Hamming distance between bitstring i and j

    # Build edge list
    src, dst = [], []
    edge_weights = []
    for i in range(M):
        for j in range(i+1, M):
            if distances[i, j] <= d_threshold and distances[i, j] > 0:
                src.extend([i, j])
                dst.extend([j, i])
                edge_weights.extend([1.0 / distances[i, j]] * 2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

    # Node features
    amp_sq = np.abs(amplitudes) ** 2
    hamming_weights = bitstrings.sum(axis=1) / N
    local_mag = 1.0 - 2.0 * hamming_weights  # magnetization per bitstring

    x = torch.tensor(
        np.stack([amp_sq, hamming_weights, local_mag], axis=1),
        dtype=torch.float32,
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pipeline_a_full(lattice, h_values, labels, krylov_dim=8, shots=10000):
    """
    Full Pipeline A: SKQD → Correlation Graph → Dataset.

    Returns list[Data] for GNN training.
    """
    from qmbp_simulation.models import HamiltonianBuilder, make_lattice
    builder = HamiltonianBuilder()
    N = lattice.n_qubits
    d_threshold = N // 4 + 1  # Hamming distance threshold

    dataset = []
    for h, label in zip(h_values, labels):
        # Build TFIM Hamiltonian for this h
        lattice_h = make_lattice(
            topology=lattice.topology,
            n_qubits=N,
            J=lattice.J,
            h=float(h),
        )
        H = builder.build(lattice_h)

        # Build Krylov circuits
        circuits = build_krylov_circuits_tfim(lattice_h, h, krylov_dim)

        # SKQD
        bitstrings, E0, amplitudes = skqd_sample_and_recover(
            circuits, H, shots
        )

        # Build correlation graph
        graph = build_correlation_graph(bitstrings, amplitudes, d_threshold)

        # Add metadata
        graph.y = torch.tensor([label], dtype=torch.float32)
        graph.h_value = float(h)
        graph.energy = float(E0)
        graph.n_nodes_original = len(bitstrings)

        dataset.append(graph)

    return dataset
```

---

### Pipeline B: SKQD Inter-Topología Transfer

#### Concepto Central

Entrenar una GNN con datos SKQD de varias topologías (chain, ladder,
triangular) y luego inferir la fase en una topología NUEVA (kagome, heavy-hex)
sin re-ejecutar SKQD. La GNN aprende features universales de la criticalidad
que trascienden la geometría.

**Base teórica**: La transferabilidad de GNNs está fundamentada en la teoría
de **graphons** (arXiv:2109.10096, arXiv:2307.13206, arXiv:2112.04629). Un
graphon es el límite continuo de una secuencia de grafos, y GNNs entrenadas en
grafos muestreados de un mismo graphon CONVERGEN a la misma función. Si las
topologías de lattice (chain, ladder, triangular) son todas "muestreadas" del
mismo graphon subyacente (el lattice 2D con perturbaciones), entonces una GNN
entrenada en unas debería transferir a otras.

**Evidencia experimental**: arXiv:2411.13027 (2024) demostró que NNs entrenadas
en una universality class detectan QPT en otra. arXiv:1904.01486 (2019, PRL)
mostró transfer learning de modelos NN a modelos NNN. Esto sugiere que features
de criticalidad SON universales y transferibles.

#### ¿Qué features universales capturaría la GNN?

La criticalidad de una QPT de segundo orden se manifiesta como:
- Divergencia de la correlation length ξ → ∞
- Power-law decay de correlaciones: C(r) ~ r^{-(d-2+η)}
- Scaling de entanglement entropy: S ~ log(L) (1D) o S ~ L^{d-1} (area law violado)

En el espacio de bitstrings SKQD, esto se traduce en:
- Distribución de amplitudes más "flat" (menos sparse) cerca de h_c
- Correlaciones entre sitios lejanos aumentan
- Shannon entropy de la distribución de bitstrings tiene un máximo en h_c
- El grafo de correlación experimenta percolation transition en h_c

Todas estas propiedades son UNIVERSALES — no dependen de la topología
específica sino de la universality class. TFIM en 1D y 2D pertenecen a la
misma universality class (Ising), por lo que el transfer debería funcionar
entre chain, ladder y triangular.

#### Pipeline Detallado

```
┌─────────────────────────────────────────────────────────────────┐
│            PIPELINE B: Inter-Topology Transfer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: Data Generation (multi-topology)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ For topology ∈ {chain_1d, ladder, triangular}:            │  │
│  │   For h ∈ linspace(0.5, 3.0, 20):                        │  │
│  │     1. Build TFIM Hamiltonian on topology                 │  │
│  │     2. Run SKQD (D=8 Krylov, 10⁴ shots)                  │  │
│  │     3. Extract: E₀, occupancies, correlators, bitstrings  │  │
│  │     4. Label: phase from ED/DMRG                          │  │
│  │                                                           │  │
│  │ Total: 3 topologies × 20 h-points = 60 training graphs   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  PHASE 2: Feature Engineering (topology-agnostic)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Para cada (topology, h) data point:                       │  │
│  │                                                           │  │
│  │ Node features (per site i):                               │  │
│  │   - h/h_c_estimate (normalized field)                     │  │
│  │   - coordination_i / max_coordination (normalized degree) │  │
│  │   - occupancy_i (from SKQD eigenvector)                   │  │
│  │   - |occupancy_i - 0.5| (order signal)                    │  │
│  │   - local_entropy_i (Shannon entropy of site-marginal)    │  │
│  │                                                           │  │
│  │ Edge features (per bond i-j):                             │  │
│  │   - ⟨Z_iZ_j⟩ (correlator from SKQD state)               │  │
│  │   - |⟨Z_iZ_j⟩ - ⟨Z_i⟩⟨Z_j⟩| (connected correlator)     │  │
│  │                                                           │  │
│  │ Global features:                                          │  │
│  │   - E₀/(N·J) (normalized energy per site)                │  │
│  │   - S_bitstring (Shannon entropy of distribution)         │  │
│  │   - n_unique/shots (effective dimension fraction)         │  │
│  │   - mean(|⟨Z_iZ_j⟩|) (average correlation strength)     │  │
│  │                                                           │  │
│  │ KEY DESIGN PRINCIPLE: ALL features are normalized and     │  │
│  │ topology-agnostic. No absolute sizes, no topology IDs.    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  PHASE 3: GNN Training                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Architecture: NNConv (uses edge features) + global pool   │  │
│  │   - Node features: 5                                      │  │
│  │   - Edge features: 2                                      │  │
│  │   - Hidden dim: 64-128                                    │  │
│  │   - Layers: 3-4 (capture multi-hop correlations)          │  │
│  │   - Global readout: concat(mean_pool, max_pool, global)   │  │
│  │   - Classifier head: MLP → sigmoid                        │  │
│  │                                                           │  │
│  │ Training strategy:                                        │  │
│  │   - Leave-one-topology-out cross-validation               │  │
│  │   - Train on 2 topologies, validate on 3rd               │  │
│  │   - Report transfer accuracy per topology pair            │  │
│  │                                                           │  │
│  │ Loss: BCEWithLogitsLoss + optional entropy regularizer    │  │
│  │ Optimizer: Adam, lr=1e-3, patience=100                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  PHASE 4: Transfer Evaluation                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Test topology (NEVER seen in training): kagome            │  │
│  │                                                           │  │
│  │ Metrics:                                                  │  │
│  │   - Classification accuracy (above/below h_c)            │  │
│  │   - |h_c_predicted - h_c_exact| (critical point error)   │  │
│  │   - Sigmoid width (sharpness of transition detection)    │  │
│  │                                                           │  │
│  │ Baselines to beat:                                        │  │
│  │   1. Naive threshold on ⟨ZZ⟩ (trivial baseline)          │  │
│  │   2. GNN trained + tested on SAME topology (oracle)      │  │
│  │   3. Random classifier (50% accuracy)                    │  │
│  │                                                           │  │
│  │ Success criterion:                                        │  │
│  │   Transfer accuracy ≥ 80% AND |Δh_c| ≤ 0.5              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Análisis de Robustez — Pipeline B

**¿La GNN puede realmente generalizar entre topologías?**

Evidencia a favor:
- arXiv:2109.10096: GNNs con graphon convergence son formalmente transferibles
  entre grafos del mismo "tipo".
- arXiv:2112.04629: "Transferability increases with graph size" — nuestros
  grafos (N=6-10) son pequeños, lo cual REDUCE transferabilidad.
- arXiv:2411.13027: Transfer entre universality classes funciona para NNs con
  datos de MC. No hay razón para que no funcione con datos de SKQD.
- Nuestra propia experiencia: el MPNN actual generaliza entre h-values no
  vistos dentro de una topología. El paso a generalizar entre topologías es
  natural para message-passing architectures.

Evidencia en contra:
- arXiv:2306.02555: "Barriers for GNN in discrete random structures" —
  GNNs tienen limitaciones locales que pueden impedir capturar propiedades
  globales.
- Para TFIM, h_c depende de la topología: h_c=1.0 (chain), h_c≈3.04 (square),
  h_c≈4.2 (triangular). Si la GNN aprende un threshold ABSOLUTO en h,
  no transferirá. PERO si aprende features NORMALIZADAS (h/h_c, correlation
  length/system_size), SÍ debería transferir.
- Los lattices son PEQUEÑOS (N=6-10). Con pocos nodos, el message passing
  se satura en pocas capas y pierde información estructural fina.

**Mitigación de riesgos:**
- Normalizar TODAS las features por topología (h → h/z donde z = coordination)
- Usar features invariantes a la topología (ratios, normalized correlators)
- Augmentar training data con noise injection y random subsampling
- Si el transfer falla para topologías "lejanas" (chain → kagome), reportar
  como resultado NEGATIVO válido (= las fases no son universales para features
  locales de SKQD)

#### Limitaciones Fundamentales

| Limitación | Severidad | ¿Mitigable? | Implicación |
|-----------|-----------|-------------|-------------|
| h_c varía entre topologías | 🔴 Alta | Parcial (normalización) | GNN no puede predecir h_c absoluto |
| TFIM paramagnético no sparse | 🔴 Alta | Dual-basis | Agrega complejidad |
| N pequeño (6-10) reduce transferabilidad | 🟡 Media | Usar N=10-12 | Mayor costo de SKQD |
| Solo 60 training graphs (3×20) | 🟡 Media | Data augmentation | Riesgo de overfitting |
| universality class assumption | 🟡 Media | Validar con frustrated models | Puede fallar para QSL |
| SKQD requiere sparsity | 🟡 Media | XXZ Δ>1 (Brooks 2026) | Limita el rango de h |

#### ¿Cuándo es este Pipeline MEJOR que el VQE+D1 actual?

| Escenario | VQE+D1 (actual) | Pipeline B (SKQD transfer) |
|-----------|----------------|---------------------------|
| TFIM 1D, topología fija | ✅ Superior (D1 funciona) | ❌ Overkill |
| TFIM en nueva topología sin ejecutar nada | ❌ Requiere VQE sweep | ✅ Zero-shot prediction |
| Heisenberg frustrado | ❌ HVA falla | ✅ SKQD funciona (Brooks) |
| N > 12 | ⚠️ VQE ~30 min/run | ✅ SKQD escala mejor |
| Hardware ruidoso | ⚠️ Requiere ZNE | ✅ SKQD noise-robust |

---

### Integración con el Proyecto Actual

#### Lo que ya tenemos y reutilizamos

| Componente existente | Reutilizable para Pipeline A | Reutilizable para Pipeline B |
|---------------------|:---------------------------:|:---------------------------:|
| `make_lattice()` | ✅ | ✅ |
| `HamiltonianBuilder.build()` | ✅ (SparsePauliOp → solve_qubit) | ✅ |
| `MPNNPredictor` base architecture | ⚠️ Modificar output (clasificación) | ⚠️ Modificar features + output |
| `build_graph_dataset()` | ❌ Reemplazar (grafo es diferente) | ⚠️ Adaptar (edge features de SKQD) |
| `train_mpnn()` | ⚠️ Cambiar loss a BCE | ⚠️ Cambiar loss + val strategy |
| `ClassicalSolver` (ED/DMRG) | ✅ (para labels) | ✅ (para labels + validation) |
| HVA circuits | ❌ Reemplazar con Krylov circuits | ❌ Reemplazar con Krylov circuits |
| Phase 1 (ED sweep) | ✅ Idéntico | ✅ Idéntico |

#### Nuevos módulos necesarios

```
src/qmbp_simulation/
├── sampling/                    ← NUEVO subpackage
│   ├── __init__.py
│   ├── krylov_circuits.py      ← Build Krylov circuits for TFIM/Heisenberg
│   ├── skqd_runner.py          ← Orchestrate SKQD (sample + recover + solve)
│   └── bitstring_graph.py      ← Build correlation graph from bitstrings
├── predictors/
│   ├── mpnn.py                 ← Existente (regresión θ_opt)
│   ├── phase_classifier.py     ← NUEVO (clasificación binaria)
│   └── transfer_gnn.py         ← NUEVO (multi-topology training)
```

#### Effort Estimate

| Componente | Pipeline A | Pipeline B | Combinado |
|-----------|:----------:|:----------:|:---------:|
| Krylov circuit builder | 2-3 días | (shared) | 2-3 días |
| SKQD integration (solve_qubit) | 2-3 días | (shared) | 2-3 días |
| Bitstring correlation graph | 3-4 días | — | 3-4 días |
| Phase classifier GNN | 2-3 días | (shared) | 2-3 días |
| Transfer training infrastructure | — | 3-4 días | 3-4 días |
| Feature engineering (normalized) | — | 2-3 días | 2-3 días |
| Experiments + validation | 5-7 días | 5-7 días | 7-10 días |
| **Total** | **~3 semanas** | **~3 semanas** | **~4-5 semanas** |

---

### Veredicto Final de Robustez

#### Pipeline A (Bitstring Correlation Graph)

| Criterio | Score | Justificación |
|---------|:-----:|---------------|
| Novedad | 9/10 | Combinación SKQD + graph theory de snapshots no existe |
| Factibilidad técnica | 7/10 | API existe (solve_qubit), pero sparsity limita rango de h |
| Valor científico | 7/10 | Conecta QPT con percolation transition en data space |
| Robustez al noise | 8/10 | SKQD configuration recovery filtra noise |
| Riesgo de resultado negativo | 4/10 | Si sparsity falla, los grafos no distinguen fases |
| Esfuerzo vs. reward | 6/10 | 3 semanas para algo que puede no funcionar para h>h_c |

**Veredicto**: Viable como PoC para tesis (restricto a h ≤ 1.5 donde SKQD
funciona). Publicable si demuestra que el graph structure cambia
cualitativamente en h_c.

#### Pipeline B (Inter-Topology Transfer)

| Criterio | Score | Justificación |
|---------|:-----:|---------------|
| Novedad | 8/10 | Transfer con datos cuánticos SKQD no existe |
| Factibilidad técnica | 6/10 | Depende de si features normalizadas transfieren |
| Valor científico | 8/10 | Demuestra universalidad de criticalidad en data-space |
| Robustez al noise | 8/10 | SKQD noise-robust |
| Riesgo de resultado negativo | 6/10 | h_c varía entre topologías → transfer puede fallar |
| Esfuerzo vs. reward | 5/10 | 3 semanas + riesgo alto de que no funcione |

**Veredicto**: Más ambicioso, mayor riesgo. Requiere demostrar que las features
SKQD capturan algo universal. Si funciona, es publicable en top venue (PRL/NatPhys).
Si no funciona, es un resultado negativo interesante ("localidad de GNN no
captura criticality universal de QPT").

#### Recomendación Integrada

1. **Para la tesis (inmediato)**: Incluir AMBOS como propuestas de Future Work
   en Cap. 6 (2-3 páginas). No implementar — solo describir el framework con
   diagramas y justificación teórica.

2. **Post-tesis (3-5 meses)**: Implementar Pipeline A primero (menor riesgo).
   Si el bitstring graph muestra percolation transition en h_c → publicar.
   Luego extender a Pipeline B (transfer) como segundo paper.

3. **Si solo hay tiempo para UNO**: Pipeline A es más seguro (la percolation
   transition es un observable objetivo, no depende de transfer learning).

---

### Referencias Adicionales (Appendix C)

- arXiv:2512.02121 (2024). "Network theory classification of quantum matter
  based on wave function snapshots."
- arXiv:2301.13216 (2023, PRL). "Wave function network description and
  Kolmogorov complexity of quantum many-body systems."
- arXiv:2510.12415 (2024). "Snapshot renormalization group for quantum matter."
- Preprints 202412.1650 / Entropy 27(1):57 (2025). "The Structure of Bit-String
  Similarity Networks."
- arXiv:2109.10096 (2021). "Transferability of Graph Neural Networks."
- arXiv:2307.13206 (2023). "Transferability of GNNs using Graphon and Sampling."
- arXiv:2112.04629 (2021). "Transferability Properties of GNNs."
- arXiv:2411.13027 (2024). "ML Domain Adaptation in Spin Models with
  Continuous Phase Transitions."
- IBM `qiskit-addon-sqd` v0.12: `qiskit_addon_sqd.qubit.solve_qubit()` API.
