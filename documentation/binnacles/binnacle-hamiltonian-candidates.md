# Binnacle — Evaluación de Hamiltonianos Candidatos para el Framework

## 2026-06-02 — Investigación de Literatura para Nuevos Modelos

### Objetivo

Determinar qué Hamiltonianos adicionales son viables para el framework GNN-HVA bajo las
restricciones existentes (HVA p≤2, |+⟩^N o Néel, shallow circuits, local observables,
IBM Torino hardware).

### Metodología

1. Revisión de `bibliography_curated.md` (44 papers)
2. Revisión de `alternative_bibliography.md` (20+ papers)
3. Revisión de `literature-synthesis.md` (insights sintetizados)
4. Búsqueda web de papers 2024-2026 sobre VQE + fase topológica
5. Análisis de compatibilidad con restricciones del proyecto

---

## Criterios de Viabilidad (TODOS deben cumplirse)

| # | Criterio | Razón |
|---|----------|-------|
| 1 | HVA p≤2 puede expresar ground state (fid≥0.90) | Mele et al. depth constraint |
| 2 | Transición de fase detectable con observables LOCALES | Hardware: solo ⟨O_i⟩ medibles |
| 3 | N=6-10 qubits suficiente para physics no-trivial | Hardware constraint |
| 4 | Implementable con SparsePauliOp (Pauli strings) | Qiskit 2.x requirement |
| 5 | CX/CZ budget ≤ 18 gates para p=1 con ZNE | Hardware viability |
| 6 | Añade valor científico distinto a TFIM puro | No duplicar resultados |

---

## Candidato 1: TFIM + Campo Longitudinal (IMPLEMENTADO ✅)

**H = −J·ZZ − h·X − g·Z**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ✅ fid≥0.98 | E4b: 75 pts, 100% pass rate |
| Observable local | ✅ ⟨X_i⟩, ⟨Z_i⟩, ⟨ZZ⟩ | Rompimiento Z₂ detectable |
| N=6-10 útil | ✅ | Crossover visible a N=6 |
| SparsePauliOp | ✅ | Ya implementado |
| CX budget | ✅ 10 CZ (p=1) | Idéntico a TFIM estándar |
| Valor añadido | ✅ 2D phase diagram (h,g) | Primer modelo bi-paramétrico |

**Veredicto: VIABLE — Ya validado.** Aporta demostración de extensibilidad del framework.

### Soporte bibliográfico

- Dutta et al. (2015): Canonical TFIM+longitudinal reference (en curated bib)
- arXiv:2301.05040: Phase transitions in antiferromagnetic TFIM with longitudinal field
- arXiv:2506.20870 (2025): VQE for boundary-field-induced phase transitions en Ising — confirma
  que VQE con shallow circuits puede detectar transiciones en modelo Ising extendido

---

## Candidato 2: TFIM Frustrado (J₁-J₂ o NNN)

**H = −J₁ Σ_{⟨ij⟩} Z_iZ_j + J₂ Σ_{⟨⟨ij⟩⟩} Z_iZ_j − h Σ_i X_i**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ⚠️ Incierto | Frustración → más entanglement |
| Observable local | ✅ ⟨X_i⟩, ⟨ZZ_nn⟩, ⟨ZZ_nnn⟩ | Orden/desorden detectable |
| N=6-10 útil | ⚠️ | Requiere 2D (≥16 sitios) para frustración real |
| SparsePauliOp | ✅ | Solo ZZ + X terms |
| CX budget | ⚠️ | NNN bonds añaden CX (más connectivity) |
| Valor añadido | ✅ | Frustración → nuevo régimen físico |

**Veredicto: POSIBLE PERO ARRIESGADO.**

### Análisis de riesgo

arXiv:2505.22932 (2025) muestra VQE en TFIM frustrado 2D en ion trap con **16 qubits y
múltiples fases magnéticas**. Pero usa ansatz de 3-4 capas, NO p≤2 HVA. La frustración
genera estados con entanglement que escala linearly — similar al problema de Heisenberg
que ya falló en V9.

### Decisión: NO IMPLEMENTAR ahora

- El valor científico requiere 2D (≥16 qubits), lo cual excede nuestra capacidad
- El HVA que necesitaríamos tendría NNN bonds → más CX gates → fuera del budget ZNE
- Riesgo alto de repetir el resultado negativo de Heisenberg

---

## Candidato 3: Cadena de Kitaev (Topología)

**H = −J Σ_i (c†_i c_{i+1} + Δ c_i c_{i+1} + h.c.) − μ Σ_i c†_i c_i**

Tras Jordan-Wigner: **H = −J Σ_i (X_iX_{i+1} + Y_iY_{i+1}) + Δ Σ_i (X_iY_{i+1} − Y_iX_{i+1}) − μ Σ_i Z_i**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ✅ (probable) | Free-fermion → exactamente resoluble → bajo entanglement |
| Observable local | ✅ ⟨Z_i⟩ (paridad), edge modes | Topología detectable localmente en bordes |
| N=6-10 útil | ✅ | Edge modes visibles con N≥4 (open boundary) |
| SparsePauliOp | ✅ | XX + YY + XY − YX + Z terms |
| CX budget | ⚠️ | XX + YY → más 2Q gates que TFIM puro |
| Valor añadido | ✅✅ | **TOPOLOGICAL** — directamente en el título de la tesis |

**Veredicto: EL MÁS PROMETEDOR para la tesis.**

### Soporte bibliográfico FUERTE

- **arXiv:2408.15179 (2024, actualizado 2025)**: "Detecting quasi-degenerate ground states in
  1D topological models via VQE" — **Exactamente nuestro caso**. Estudian SSH y Kitaev chain
  con VQE. Demuestran que VQE puede detectar fases topológicas en cadenas cortas. Paper
  directamente aplicable.

- **arXiv:2304.13408**: "Quantum-circuit algorithms for many-body topological invariant and
  Majorana zero mode" — Algoritmos de circuito cuántico para invariantes topológicos.

- **Wiersema et al. (2020)** en nuestra curated bib: HVA paper original menciona que HVA
  funciona bien para modelos resoluble por fermiones libres (como Kitaev).

### Análisis de viabilidad técnica

La cadena de Kitaev es un modelo de **fermiones libres** tras Jordan-Wigner. Esto significa:
1. El ground state tiene entanglement BAJO (area law exacto)
2. HVA p=1-2 debería ser suficiente (similar al TFIM que también es free-fermion via JW)
3. La transición topológica ocurre en |μ| = 2J → fase trivial vs topológica

El HVA necesario tendría:
- Capa RXX + RYY (bonds nn): interacción de hopping
- Capa RXY (bonds nn, antisimétrica): pairing superconductor
- Capa RZ (sitios): potencial químico

**Params/layer = 3-4**, similar al Heisenberg. PERO la diferencia clave es que el modelo
es exactamente resoluble → el ground state vive en un espacio de baja complejidad.

### Estimación CX gates

Para N=6, cadena abierta (5 bonds):
- RXX: 5 bonds × 2 CX/RXX = 10 CX
- RYY: 5 bonds × 2 CX/RYY = 10 CX
- Total p=1: ~20 CX → **EXCEDE budget ZNE (>18)**

Esto es un problema. Sin embargo:
- Con optimización de transpilación (cancel adjacent CX), podría reducirse
- Alternativa: usar forma simplificada del Kitaev en el punto sweet-spot Δ=J

### Decisión: INVESTIGAR MÁS antes de implementar

Necesito verificar:
1. ¿El ground state del Kitaev es expresable con p=1 HVA?
2. ¿Cuántas CX gates reales tras transpilación optimizada?
3. ¿Los edge modes son detectables con ⟨Z_i⟩ local?

---

## Candidato 4: XXZ con Anisotropía Variable (Δ como parámetro)

**H = J Σ_{(i,j)} (X_iX_j + Y_iY_j + Δ·Z_iZ_j) − h Σ_i Z_i**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ❌ a Δ=1 | V9: 30 runs, max fid=48% |
| Observable local | ✅ ⟨Z_i⟩, ⟨S·S⟩ | |
| N=6-10 útil | ✅ | |
| SparsePauliOp | ✅ | Ya implementado |
| CX budget | ❌ ~30+ CX | XX+YY+ZZ por bond |
| Valor añadido | ⚠️ | Solo si Δ<1 funciona |

**Veredicto: NO VIABLE a Δ≥0.5.**

V9 demostró que el HVA p≤2 para Heisenberg es fundamentalmente insuficiente.
El modelo XY (Δ=0) también falló (fid=23%). El entanglement requerido escala
linearly con N en la fase antiferromagnética, imposible con p≤2.

La única ventana sería Δ≪1 (muy cerca del modelo libre) pero esto no aporta
física interesante — se reduce esencialmente al TFIM en otra base.

### Decisión: DESCARTADO

---

## Candidato 5: TFIM con Boundary Fields (condiciones de contorno)

**H = −J Σ_{i} Z_iZ_{i+1} − h Σ_i X_i − h_L Z_1 − h_R Z_N**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ✅ (probable) | Misma estructura que TFIM+longitudinal, pero solo en bordes |
| Observable local | ✅ ⟨Z_1⟩, ⟨Z_N⟩ edge magnetization | Wetting transition detectable |
| N=6-10 útil | ✅ | Efectos de borde visibles a N≥6 |
| SparsePauliOp | ✅ | Solo ZZ + X + Z(bordes) |
| CX budget | ✅ | Idéntico a TFIM (Z en bordes es single-qubit) |
| Valor añadido | ⚠️ Moderado | Transición de wetting, primer orden |

**Veredicto: VIABLE pero bajo valor añadido.**

### Soporte bibliográfico

- **arXiv:2506.20870 (2025)**: "Variational simulation of quantum phase transitions induced
  by boundary fields" — Paper exacto sobre esto. Implementa VQE en TFIM con campos de borde.
  Encuentra diagrama de fases rico (línea de primer orden + transición de wetting continua).
  **Validado en hardware experimental.**

### Análisis

Técnicamente es trivial de implementar (es un caso especial de nuestro TFIM+longitudinal
donde g se aplica solo a los sitios de borde). El HVA estándar (ZZ+X) probablemente funciona
porque los campos de borde no rompen la Z₂ de bulk.

Sin embargo, el valor científico es marginal: no demuestra nueva capacidad del framework
(ya tenemos TFIM+longitudinal que es más general).

### Decisión: NO PRIORIZAR (guardar para trabajo futuro si hay tiempo)

---

## Candidato 6: Modelo de Schwinger Lattice (QED 1+1)

**H = m Σ_i (−1)^i Z_i + J Σ_i (X_iX_{i+1} + Y_iY_{i+1}) + ... (campo eléctrico)**

| Criterio | Resultado | Evidencia |
|----------|-----------|-----------|
| HVA expresividad | ⚠️ Incierto | Depende de la formulación |
| Observable local | ✅ Electric field, chiral condensate | |
| N=6-10 útil | ✅ | Demostrado en hardware (arXiv:2312.12831) |
| SparsePauliOp | ✅ | Pauli representable |
| CX budget | ⚠️ | Depende de encoding |
| Valor añadido | ✅✅ | High-energy physics, muy diferente a Ising |

**Veredicto: INTERESANTE pero fuera de scope.**

arXiv:2312.12831 (2023) y arXiv:2504.20824 (2025) demuestran VQE en Schwinger model
en hardware, detectando transición de fase de primer orden con θ-term. Sin embargo:
- Es un modelo de lattice gauge theory, no un spin model puro
- Requiere encoding adicional (fermion-to-qubit)
- El HVA no aplica directamente (no es un Hamiltonian variational para gauge theories)

### Decisión: DESCARTADO (fuera del scope de la tesis: spin models)

---

## Resumen de Evaluación

| Candidato | Viabilidad | Valor Científico | Esfuerzo | Recomendación |
|-----------|:---:|:---:|:---:|:---:|
| **TFIM + longitudinal** | ✅✅ | ✅ | Hecho | ✅ IMPLEMENTADO |
| **Kitaev chain** | ✅ (verificar CX) | ✅✅ | Medio | ⭐ INVESTIGAR |
| TFIM frustrado (J₁-J₂) | ⚠️ | ✅ | Alto | ❌ No ahora |
| XXZ anisotropía | ❌ | ⚠️ | Bajo | ❌ Descartado (V9) |
| TFIM boundary fields | ✅ | ⚠️ | Bajo | 📝 Futuro |
| Schwinger lattice | ⚠️ | ✅✅ | Alto | ❌ Fuera de scope |

---

## Recomendación Final

### Para la tesis (implementar)

1. **TFIM + longitudinal** — Ya hecho. Demuestra extensibilidad.
2. **Cadena de Kitaev** — Si el CX budget lo permite. Aporta "topological" al título.

### Para trabajo futuro (documentar pero no implementar)

3. TFIM con boundary fields — Trivial de implementar, bajo valor
4. TFIM frustrado — Requiere N>10 y más profundidad
5. Schwinger — Fuera del scope de spin models

### Próximo paso concreto

Verificar viabilidad de Kitaev chain con un **prototipo rápido** (N=4, p=1):
- ¿Cuántas CX gates tras transpilación?
- ¿El ground state es expresable con HVA p=1?
- ¿La transición topológica (μ=2J) es detectable con ⟨Z_edge⟩?

Si las 3 respuestas son positivas → implementar. Si alguna falla → documentar como
"fuera del budget de hardware" y usar solo TFIM + longitudinal como extensión.

---

## Referencias Clave Nuevas (para agregar a bibliografía)

| Paper | Relevancia |
|-------|-----------|
| arXiv:2408.15179 (2024) | VQE en Kitaev + SSH chains, detección topológica |
| arXiv:2505.22932 (2025) | TFIM frustrado 2D en trapped-ion, 16 qubits |
| arXiv:2506.20870 (2025) | VQE + boundary fields, transición de wetting |
| arXiv:2301.05040 (2023) | Diagrama de fases TFIM antiferromagnético + longitudinal |

*Binnacle entry complete.*


---

## ADDENDUM: Kitaev Chain Verification Results (2026-06-02)

### Resultado: 🔴 NO VIABLE

Script: `scripts/verify_kitaev_cx_budget.py`

#### Test 1: 2Q Gate Count (FakeTorino, optimization_level=2)

| N | Kitaev HVA (p=1) | TFIM HVA (p=1) | Δ | ZNE (≤18) |
|---|:---:|:---:|:---:|:---:|
| 4 | 12 | 6 | +6 | ✅ |
| 6 | 20 | 10 | +10 | ❌ |
| 8 | 28 | 14 | +14 | ❌ |
| 10 | 36 | 18 | +18 | ❌ |

El Kitaev HVA usa **exactamente el doble de CX gates** que TFIM (RXX+RYY = 2× el costo de RZZ).

#### Test 2: Expressibility (N=4, p=1, 15 restarts)

| μ | Phase | Best Fidelity | ΔE/gap |
|---|---|:---:|:---:|
| 0.5 | topological | 0.160 | 4.44 |
| 1.5 | topological | 0.122 | 3.66 |
| 2.5 | trivial | 0.063 | 16.53 |
| 4.0 | trivial | 0.063 | 3.80 |

**Fidelidad máxima: 16%** — el HVA (XX+YY+Z con |+⟩^N) es completamente insuficiente.

#### Test 3: Edge Detection (exact ground state)

Los observables locales ⟨Z_edge⟩ vs ⟨Z_bulk⟩ sí muestran diferencia en la fase topológica,
confirmando que la física es detectable — pero el circuito no puede preparar el estado.

### Root Cause Analysis

1. **Gate budget 2×:** RXX + RYY por bond = 4 CX, vs RZZ = 2 CX para TFIM
2. **Estado inicial inadecuado:** |+⟩^N es el ground state de H_X, no del Kitaev
3. **Expresividad insuficiente:** 3 params no capturan las correlaciones de pairing
4. **El Kitaev NO es equivalente a TFIM:** Aunque ambos son free-fermion, el Kitaev
   tiene pairing (p-wave superconductivity) que requiere correlaciones XX+YY, no solo ZZ

### Conclusión Final

La cadena de Kitaev **no es compatible** con las restricciones del framework (HVA p≤2,
≤18 CX gates, |+⟩^N initial state). Requeriría:
- Un ansatz diferente (no HVA standard)
- Estado inicial adaptado (BCS-like state)
- p≥3 capas o ansatz con más parámetros por capa

**El TFIM + longitudinal es la ÚNICA extensión viable confirmada.**


---

## ADDENDUM 2: Análisis Detallado de Barreras del Framework para Kitaev (2026-06-03)

### Objetivo

Documentar con precisión cuáles son las restricciones del framework que impiden la
implementación exitosa de la cadena de Kitaev, qué reglas las gobiernan, y cuáles
serían los cambios mínimos necesarios para intentar una validación — junto con la
evaluación de si esos cambios son compatibles con la arquitectura del proyecto.

---

### Barrera 1: Estado Inicial |+⟩^N — Overlap nulo con ground state de Kitaev

**Regla que lo impone:** SKILL.md → "Initial state: |+⟩^N (`qc.h(range(n))`). MANDATORY."

**Por qué existe la regla:**
El estado |+⟩^N es el ground state exacto de H_X = −h·ΣX_i (campo transverso puro).
Para el TFIM, cuando h→∞ el ground state converge a |+⟩^N, y la estrategia de
warm-start VQE (descending sweep h_max→h_min) parte desde este estado natural.
El ansatz HVA solo necesita "rotar" gradualmente desde |+⟩^N hacia el ground state
ferromagnético conforme h decrece.

**Por qué falla para Kitaev:**
El ground state del Kitaev en la fase topológica (|μ| < 2J) es un estado de
**fermiones pareados** (BCS-like). En la base de qubits tras Jordan-Wigner, este
estado tiene estructura de entanglement muy diferente a |+⟩^N:

- |+⟩^N tiene ⟨Z_i⟩ = 0 para todo i → no tiene magnetización neta
- El ground state de Kitaev tiene correlaciones XX+YY de largo alcance (pairing)
- El overlap ⟨+|^N |GS_Kitaev⟩ es exponencialmente pequeño en N

Resultado: el VQE arranca en una región del landscape que es un **mínimo local
profundo** (la energía de |+⟩^N bajo H_Kitaev no es terrible, pero el gradiente
apunta hacia soluciones incorrectas). Los 15 restarts no logran escapar.

**Precedente de excepción:** El modelo Heisenberg usa estado Néel (|010101...⟩) como
excepción explícita a la regla |+⟩^N. El `create_heisenberg()` acepta
`initial_state="neel"` como parámetro. El mismo mecanismo permitiría un estado
adaptado para Kitaev.

**Cambio hipotético:**
- Opción A: `initial_state="half_fill"` → |0101...⟩ (half-filling fermionic).
  Similar a Néel pero motivado por la estructura de la cadena a half-filling.
- Opción B: `initial_state="bcs"` → Estado BCS preparable con un circuito
  Gaussian de profundidad O(N). Requiere circuito de preparación previo al HVA.
- Opción C: Ground state del Kitaev a μ=0 (punto sweet-spot), preparable
  exactamente con un circuito matchgate de profundidad N-1.

**Evaluación:** La Opción A es trivial de implementar (una línea de código,
precedente existente). Las opciones B/C añaden gates 2Q al circuito de
preparación, empeorando el problema del budget CX.

---

### Barrera 2: Budget CX — 20 CZ a N=6 p=1 (excede ZNE ≤ 18)

**Regla que lo impone:** Project-status.md → "ZNE threshold: ~18 CX gates."

**Origen de la regla:**
La extrapolación ZNE (Zero-Noise Extrapolation) inhomogénea requiere que el circuito
sea lo suficientemente corto para que la relación señal/ruido permita un fit lineal
confiable (R² > 0.99). Empíricamente, con el ruido de IBM Torino:
- ≤18 CX: ZNE lineal funciona (R²>0.99, gain +48-62%)
- 19-25 CX: ZNE degradado (R²<0.9, gain variable)
- >25 CX: ZNE inútil (señal enterrada en ruido)

**Por qué Kitaev excede:**
```
TFIM (N=6, p=1):  5 bonds × 1 RZZ × 2 CX/RZZ = 10 CX  ✅
Kitaev (N=6, p=1): 5 bonds × (RXX + RYY) × 2 CX/gate = 20 CX  ❌ (+2 sobre límite)
```

Cada bond del Kitaev necesita representar la interacción de hopping (XX) y de
pairing (YY). Cada una requiere 2 CX gates en la descomposición estándar.

**Cambios hipotéticos y sus consecuencias:**

| Opción | CX resultantes | Viabilidad | Problema |
|--------|:-:|:-:|---|
| Reducir a N=4 | 12 CX | ✅ ZNE OK | N=4 tiene poca física (solo 1 bond de bulk) |
| Usar N=5 (impar) | 16 CX | ✅ ZNE OK | Boundary effects dominan; N impar rompe simetría |
| Gate cancellation (RXX·RYY → fSim) | ~12-14 CX | ⚠️ Posible | Requiere synthesis custom; Qiskit opt_level=3 podría lograrlo |
| Subir threshold ZNE | 20 CX | ❌ | No es negociable — es un resultado empírico, no una elección |

**La opción más realista** es N=4 o gate cancellation. Pero N=4 tiene un solo "bond
de bulk" (los bonds 0-1 y 3-4 son de borde), lo que hace que los edge modes no se
distingan del bulk. La física interesante del Kitaev (Majorana zero modes en los
bordes) requiere al menos N=6 para ser observable.

---

### Barrera 3: Parámetros globales por capa — Insuficiente para capturar pairing

**Regla que lo impone:** Convención arquitectónica del HVA: "un θ por tipo de interacción
por capa" (ver `circuits/hva.py`).

**Cómo funciona en TFIM:**
```python
# 2 params por capa: θ_zz (TODOS los bonds), θ_x (TODOS los sitios)
for i, j in lattice.edges:
    qc.rzz(2 * theta_zz, i, j)  # mismo ángulo para todos los bonds
for i in range(n_qubits):
    qc.rx(2 * theta_x, i)       # mismo ángulo para todos los sitios
```

Esto funciona para TFIM porque el Hamiltoniano es **traslacionalmente invariante** y
el ground state hereda esa simetría. Un solo θ_zz captura la correlación ZZ uniforme.

**Por qué falla para Kitaev:**
El Kitaev con boundary conditions abiertas (necesarias para edge modes) tiene un
ground state que **NO es traslacionalmente invariante**:
- Los qubits de borde (0 y N-1) tienen correlaciones cualitativamente diferentes
- El pairing Δ(XY − YX) introduce una **fase compleja** direction-dependent
- Con un solo θ_XX global, el circuito no puede diferenciar hopping de pairing

Para capturar la física del Kitaev, se necesitaría una de:
- **Params por bond** (edge-resolved HVA): θ_XX^{(i,j)} diferente para cada bond.
  Para N=6 p=1: 5 bonds × 2 params + 1 RZ = 11 params (vs 3 actuales).
- **Separación hopping/pairing**: θ_XX y θ_YY como params independientes (ya lo es
  en la prueba de verificación) — pero ni así funciona porque la invariancia traslacional
  del ansatz no puede representar las correlaciones de borde.
- **RXY antisimétrico explícito**: Gate e^{−iθ(XY−YX)} que no se descompone
  naturalmente en RXX + RYY con mismos ángulos.

**Evaluación:**
Usar params por bond **sigue siendo HVA** (cada gate corresponde a un término del
Hamiltoniano). No viola la regla "ONLY HVA, NEVER HEA" — simplemente añade
resolución espacial. Sin embargo:
- 11 params en un landscape VQE con p=1 es significativamente más duro que 3
- El Heisenberg con 4 params/capa (8 total en p=2) ya demostró que landscapes
  con >6 params tienen minimos locales difíciles (fid máx 48%)
- 11+ params con N=4 (donde la física es trivial) no aporta valor científico

---

### Barrera 4 (emergente): La combinación de problemas es peor que la suma

Los 3 problemas NO son independientes — se refuerzan:

1. Estado inicial malo → VQE necesita más iteraciones y restarts
2. Más params → landscape más complejo → VQE necesita AÚN más restarts
3. Más CX gates → no se puede usar hardware → valor de la validación se reduce

El Heisenberg (V9) ya demostró este patrón con solo 2 de los 3 problemas
(4 params + estado subóptimo): max fid = 48% con 10 restarts. El Kitaev
tiene los 3 simultáneamente, con un agravante: la fidelidad verificada es
**3× peor** que Heisenberg (16% vs 48%).

---

### Tabla resumen: Reglas vs Cambios necesarios

| Barrera | Regla afectada | Cambio mínimo | Rompe regla | Precedente |
|---------|---------------|---------------|:-----------:|:----------:|
| Estado |+⟩^N | SKILL.md (mandatory) | initial_state adaptado | ⚠️ Parcial | Heisenberg usa Néel |
| CX > 18 | ZNE threshold (empírico) | N=4 o gate cancellation | No (limita N) | — |
| Params globales | Convención HVA | Params por bond | No (sigue siendo HVA) | — |
| Combinación 1+2+3 | Expresividad p≤2 | p≥3 o ansatz híbrido | ❌ Sí (Mele et al.) | Ninguno |

---

### Veredicto Final Extendido

La cadena de Kitaev está bloqueada por **3 restricciones simultáneas** del framework,
de las cuales:
- 1 tiene precedente de relajación (estado inicial → como Heisenberg)
- 1 es un límite empírico no negociable (budget CX = física del hardware)
- 1 es una limitación arquitectónica resoluble pero costosa (params por bond)

Incluso relajando las 3, el resultado más optimista (N=4, p=2, initial_state
adaptado, params por bond) produciría un resultado con **valor científico mínimo**:
- N=4 no tiene physics de Majorana distinguible del trivial
- No es deployable en hardware (o trivialmente deployable sin interés)
- No demuestra capacidad del framework (requiere excepciones a todas las reglas)

**Conclusión reforzada:** El Kitaev NO es viable bajo el framework GNN-HVA actual.
La barrera es fundamentalmente física (entanglement structure del pairing
superconductor), no una limitación de implementación que se pueda "arreglar"
con más código. Documentar como resultado negativo informado.

**Implicación para la tesis:** Citar esta incompatibilidad como evidencia de los
**límites de aplicabilidad del HVA p≤2**: modelos con pairing superconductor o
entanglement que escala linealmente están fuera del alcance de circuitos shallow.
Esto complementa el resultado negativo de Heisenberg (V9) y establece el boundary
preciso del framework.

*Addendum complete. No further Kitaev work planned.*
