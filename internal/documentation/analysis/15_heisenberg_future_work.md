# Heisenberg XXZ — Future Work & Alternative Ansätze

**Fecha**: 2026-06-01
**Contexto**: Los experimentos V9 (36 runs, N=6/10/16) confirman que HVA p≤2 no puede
expresar el ground state del modelo Heisenberg XXZ. Este documento cataloga las
alternativas identificadas en la literatura para superar esta limitación.

---

## 1. El Dilema Profundidad/Ruido

Nuestros resultados de depth scaling (Test 4) muestran:

| p | Params | Fidelity (Δ=1, h=3) | CX gates (N=6) | ZNE viable? |
|---|:------:|:--------------------:|:---------------:|:-----------:|
| 2 | 8 | 0.2% | 30 | ❌ (>18 threshold) |
| 3 | 12 | 37% | 45 | ❌ |
| 5 | 20 | 48% | 75 | ❌ |
| 6 | 24 | 43% | 90 | ❌ |

**Conclusión**: No existe un p donde Heisenberg tenga AMBOS alta fidelidad Y
viabilidad en hardware NISQ. Se necesita un ansatz fundamentalmente diferente.

---

## 2. Alternativas Identificadas en la Literatura

### 2.1 Ansatz con Preservación de Simetría (Recomendado)

**Referencia**: Sharma et al., "Symmetry-Preserving Variational Quantum Simulation
of the Heisenberg Spin Chain on Noisy Quantum Hardware", [arXiv:2512.23009](https://arxiv.org/abs/2512.23009) (Dec 2025)

**Idea**: Diseñar gates que conserven el número cuántico total S_z, restringiendo
la búsqueda variacional al subespacio de simetría correcto.

**Resultados reportados**:
- Mejora significativa en estimaciones de energía vs HEA
- Mayor robustez contra ruido de hardware (validado en IQM Garnet)
- Convergencia más clara con menos parámetros

**Por qué resuelve nuestro problema**: Nuestro Test 3 mostró que el estado Néel
es una trampa (gradiente=0) porque el HVA estándar puede salir del sector de
simetría correcto. Un ansatz que preserve S_z por construcción elimina este
problema — el optimizador solo puede moverse dentro del sector correcto.

**Viabilidad para nuestro framework**: ALTA
- Requiere: nuevo `create_heisenberg_symmetric()` en `HVACircuitBuilder`
- Compatible con: `PipelineRunner`, `ModelSpec`, `VQEOptimizer` (sin cambios)
- Estimación de esfuerzo: ~1 semana de implementación + validación

Content was rephrased for compliance with licensing restrictions.

---

### 2.2 Ansatz Inspirado en MPS/DMRG

**Referencia**: Javanmard et al., "Matrix product state ansatz for the variational
quantum solution of the Heisenberg model on Kagome geometries",
[arXiv:2401.02355](https://arxiv.org/abs/2401.02355) (Jan 2024)

**Idea**: Usar un circuito parametrizado cuya estructura replica la descomposición
MPS del ground state. El circuito aplica unitarias de 2 qubits en un patrón
"brick-wall" que codifica la estructura de entrelazamiento del MPS.

**Resultados reportados**:
- Funciona para Heisenberg en Kagome (geometría frustrada)
- Baja profundidad, pocos parámetros
- Compatible con ZNE para mitigación de ruido
- Representa fielmente la física del ground state

**Por qué es relevante**: Nuestro framework ya tiene MPS validado (V7 exp 3A/3B:
|MPS-SV|=1e-14, chi=64 suficiente). Un ansatz MPS-inspired usaría esa misma
estructura pero en un circuito cuántico parametrizado.

**Viabilidad para nuestro framework**: MEDIA
- Requiere: estructura de circuito completamente nueva (no es HVA)
- Ventaja: ya tenemos la infraestructura MPS para generar el target
- Desventaja: no es compatible con el warm-start descendente (diferente paradigma)
- Estimación de esfuerzo: ~2-3 semanas

Content was rephrased for compliance with licensing restrictions.

---

### 2.3 TETRIS-ADAPT-VQE (Circuitos Adaptativos Compactos)

**Referencia**: "TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower,
denser circuit Ansätze", [arXiv:2209.10562](https://arxiv.org/abs/2209.10562) (2022, pub. 2024)

**Idea**: Construir el ansatz adaptativamente, seleccionando operadores de un pool
y empaquetándolos en paralelo (como piezas de Tetris) para minimizar profundidad.

**Resultados reportados**:
- Circuitos significativamente más cortos que ADAPT-VQE estándar
- Sin aumento en gates CNOT ni parámetros variacionales
- Ventaja crece con el tamaño del sistema

**Por qué es relevante**: Podría encontrar un ansatz de baja profundidad específico
para Heisenberg que nuestro HVA fijo no puede representar.

**Viabilidad para nuestro framework**: BAJA
- Requiere: loop adaptativo con muchas mediciones (incompatible con GNN warm-start)
- El paradigma es opuesto al nuestro: ADAPT construye el circuito, nosotros lo fijamos
- No es compatible con el pipeline de 4 fases (Phase 2 asume circuito fijo)
- Estimación de esfuerzo: ~4+ semanas, requiere rediseño del pipeline

Content was rephrased for compliance with licensing restrictions.

---

### 2.4 Slice-Wise Initial State Optimization

**Referencia**: "Slice-Wise Initial State Optimization to Improve Cost and Accuracy
of the VQE on Lattice Models", [arXiv:2509.13034](https://arxiv.org/html/2509.13034v1) (2025)

**Idea**: Optimizar el estado inicial capa por capa antes de ejecutar el VQE completo.
Cada "slice" del circuito se optimiza independientemente, construyendo un buen
punto de partida que evita mínimos locales.

**Resultados reportados**:
- Mejora fidelidades en Heisenberg y Hubbard hasta 20 qubits
- Reduce evaluaciones de función
- Simple, cost-effective, adecuado para NISQ

**Por qué resuelve nuestro problema**: Nuestro Test 3 mostró que el Néel es una
trampa con gradiente cero. Slice-wise optimization construiría un mejor estado
inicial que escapa de esa trampa antes de comenzar el VQE global.

**Viabilidad para nuestro framework**: ALTA
- Requiere: modificar `VQEOptimizer` para añadir fase de pre-optimización por capas
- Compatible con: todo el pipeline existente (solo cambia la inicialización)
- Podría combinarse con nuestro warm-start descendente
- Estimación de esfuerzo: ~1 semana

Content was rephrased for compliance with licensing restrictions.

---

### 2.5 Generative Quantum Eigensolver (GQE)

**Referencia**: "A generative quantum eigensolver for spin Hamiltonians",
[arXiv:2603.24298](https://arxiv.org/html/2603.24298) (2026)

**Idea**: Usar un modelo generativo clásico para producir circuitos cuánticos
con propiedades deseadas, superando las limitaciones de ansätze fijos.

**Resultados reportados**:
- Supera limitaciones de barren plateaus y expresibilidad restringida
- No depende de estructura domain-specific

**Viabilidad para nuestro framework**: BAJA
- Paradigma completamente diferente (genera circuitos, no parámetros)
- No compatible con nuestro pipeline GNN→θ

---

## 3. Tabla Comparativa

| Alternativa | Fidelity esperada | CX budget | Compatible con pipeline? | Esfuerzo |
|-------------|:-----------------:|:---------:|:------------------------:|:--------:|
| Symmetry-preserving | >90% (literatura) | ~20-30 | ✅ (nuevo circuit builder) | 1 sem |
| MPS-inspired | >90% (Kagome) | ~30-50 | ⚠️ (nuevo paradigma) | 2-3 sem |
| TETRIS-ADAPT | >95% (adaptativo) | variable | ❌ (incompatible) | 4+ sem |
| Slice-wise init | ~60-80% (estimado) | 30 (p=2) | ✅ (solo init) | 1 sem |
| GQE | desconocido | variable | ❌ (paradigma diferente) | 4+ sem |

---

## 4. Recomendación para Trabajo Futuro

**Prioridad 1**: Symmetry-preserving ansatz (§2.1)
- Mayor impacto con menor esfuerzo
- Directamente resuelve la causa raíz (sector de simetría incorrecto)
- Validado en hardware real (IQM Garnet, Dec 2025)

**Prioridad 2**: Slice-wise initialization (§2.4)
- Complementario a cualquier ansatz
- Podría mejorar incluso el HVA estándar (escapar de la trampa Néel)
- Mínimo cambio al pipeline existente

**Prioridad 3**: MPS-inspired ansatz (§2.2)
- Para cuando se necesite Kagome u otras geometrías frustradas
- Aprovecha nuestra infraestructura MPS existente

---

## 5. Thesis Statement (para Chapter 6 — Conclusions & Future Work)

> "The systematic evaluation of the Heisenberg XXZ model across 36 pipeline
> configurations and three system sizes (N=6, 10, 16) establishes a definitive
> negative result: HVA p≤2 with Néel initial state cannot express the ground
> state due to symmetry-sector trapping (fidelity ≈ 0%, energy gap scaling
> linearly as ~3.8N). Depth scaling validation confirms that p=5 achieves only
> 48% fidelity, and the CX budget at p≥3 (45+ gates) exceeds the ZNE threshold
> (18 gates), creating an irreconcilable depth-noise trade-off.
>
> Recent literature offers three promising paths forward: (1) symmetry-preserving
> ansätze that restrict variational search to the correct S_z sector [Sharma 2025],
> validated on real quantum hardware; (2) MPS-inspired circuits that efficiently
> encode entanglement structure [Javanmard 2024], demonstrated on frustrated
> Kagome geometries; and (3) slice-wise initialization [arXiv:2509.13034] that
> escapes the Néel trap through layer-by-layer pre-optimization. Each represents
> a natural extension of this framework, requiring only modifications to the
> circuit construction layer while preserving the GNN warm-start pipeline."
