# Utilidad Práctica del Framework GNN-HVA

**Fecha**: 2026-07-21  
**Basado en**: Investigación de mercado, literatura 2024-2026, ecosistema IBM Quantum

---

## Premisa Honesta

Este proyecto NO es una herramienta comercial ni pretende competir con software de
producción. Es un **framework de investigación** que resuelve problemas concretos para
audiencias específicas. Su valor está en la combinación única de componentes, no en
ninguno individual.

---

## Audiencia 1: Grupos Experimentales con Acceso a QPU

### ¿Quiénes son?

- **Oak Ridge + IBM + Purdue + UIUC** (acaban de publicar KCuF₃ en marzo 2026 — simulación
  de materiales magnéticos en QPU IBM 127-qubit, comparando con datos de scattering de
  neutrones)
- **IQM Garnet users** (Sharma 2026, TFIM en hardware real)
- **Grupos IBM Quantum Network** (675+ papers publicados, 180+ instituciones)
- **Grupos con acceso a IBM Heron** vía IBM Quantum Platform

### ¿Qué problema tienen?

Cuando van a correr VQE en hardware real, necesitan responder **antes de gastar QPU time**:

1. "¿Mi ansatz HVA p=2 puede resolver este campo h en esta topología?"
2. "¿Cuántos CX gates tengo? ¿ZNE va a funcionar o no?"
3. "¿Qué método de mitigación uso: gate-folding, CES, o PEA?"
4. "¿Puedo transferir mi modelo entrenado en cadena 1D a heavy_hex sin reentrenar?"

### ¿Qué les ofreces?

| Capacidad | Forma de entrega | Costo QPU |
|-----------|------------------|:---------:|
| Atlas h_min(N,p,topo): "¿puedo?" | Tabla/API consulta | 0 |
| Predicción θ_opt via GNN: warm-start | Forward pass (~1ms) | 0 |
| GNN-QEM post-procesamiento: corrección | Forward pass sobre resultado ruidoso | 0 |
| Diagnóstico θ_smoothness: "¿MPNN aprenderá?" | Análisis sobre datos VQE | 0 |
| PEA-ZNE vs GF-ZNE decision: "¿qué mitigación?" | Regla: CX < 18 → GF, else PEA | 0 |

### ¿Cómo se lo entregas?

**Opción A — Paper + código reproducible**:
Publicar Paper A (GNN-QEM) y Paper B (Atlas) con enlace a repositorio GitHub.
Los grupos experimentales lo citan y adaptan.

**Opción B — Módulo pip instalable** (más útil):
```bash
pip install qmbp-simulation
```
```python
from qmbp_simulation import expressibility_check, GNNQEMCorrector

# Antes de correr en QPU:
viable = expressibility_check(model="tfim", topology="heavy_hex", N=10, p=2, h=1.5)
# → True/False + h_min estimado + CX count + ZNE recommendation

# Después de correr en QPU (post-procesamiento):
corrector = GNNQEMCorrector.pretrained("tfim_chain_ladder")
e_corrected = corrector.correct(e_noisy=−4.2, h=1.5, topology="heavy_hex")
```

### ¿Por qué lo usarían?

- **Ahorro de QPU time**: IBM cobra por shot y por tiempo de reserva. Un check previo
  que evita 1 run fallido a 8192 shots × 30 circuitos = ahorro de ~$50-200 en créditos.
- **Paper de Oak Ridge (marzo 2026)** simuló KCuF₃ (cadena de espín Heisenberg). Tu framework
  les diría inmediatamente: "Heisenberg con HVA p≤2 = 0% éxito, necesitan UCCSD o p≥5".
  Eso les hubiera ahorrado exploración.
- **Referencia citable**: Grupos necesitan justificar sus elecciones de ansatz y mitigación.
  Citar tu atlas es más fuerte que "elegimos p=3 porque nos pareció".

---

## Audiencia 2: Educación Universitaria en Quantum Computing

### ¿Quiénes son?

- **IBM Quantum Learning** (cursos de VQE para chemistry ya existentes en quantum.cloud.ibm.com)
- **CERN** (ya enseñan VQE en spin chains a estudiantes de PhD — presentación S2_VQE_Lecture)
- **QSEEC** (Quantum Science and Engineering Education Conference, IEEE, anual)
- **Universidades en Latinoamérica**: UBA QuICC (Buenos Aires), CINVESTAV (México),
  Universidad de la República (Uruguay), USP/UNICAMP (Brasil)
- **ASQC 2026** — Simposio de Quantum Computing en JAIIO, La Plata, Argentina

### ¿Qué problema tienen?

Los cursos de VQE enseñan el algoritmo en abstracto (H₂ molecule, 2 qubits) pero NO enseñan:
- Cuándo VQE **falla** y por qué (expresividad, landscape, barren plateaus)
- Cómo elegir ansatz para un problema concreto
- Cómo escala en la práctica (no solo en teoría)
- Workflow completo: desde Hamiltoniano hasta resultado mitigado en hardware

### ¿Qué les ofreces?

Un **laboratorio completo con datos reales** donde el estudiante puede:

1. Elegir modelo (TFIM, Heisenberg, XY) y topología
2. Ver por qué Heisenberg falla (no es un bug, es física)
3. Explorar la frontera h_min interactivamente
4. Entrenar una GNN y ver qué aprende (interpretabilidad)
5. Simular ruido y ver efecto de ZNE (noisy mode)
6. Comparar su resultado con ground truth exacto

### ¿Cómo se lo entregas?

**Opción A — Jupyter notebooks tutoriales** (menor esfuerzo):
```
notebooks/
├── 01_hamiltonian_construction.ipynb
├── 02_vqe_warmstart_demo.ipynb
├── 03_expressibility_limits.ipynb
├── 04_gnn_training.ipynb
├── 05_noisy_simulation_zne.ipynb
└── 06_phase_detection_unsupervised.ipynb
```
Ejecutables en Google Colab con `pip install qmbp-simulation`.

**Opción B — Módulo para IBM Quantum Learning** (mayor impacto):
IBM acepta contribuciones externas a su plataforma educativa. Un módulo
"VQE for Spin Models: From Theory to Practice" encajaría perfectamente.

### ¿Por qué lo usarían?

- **Gap educativo real**: IBM tiene cursos de VQE para química pero NO para materia
  condensada/spin models. Tu framework llena ese vacío exacto.
- **2025 fue el Año Internacional de Quantum** (UNESCO). Hay demanda activa de materiales.
- **Latinoamérica específicamente** tiene pocos recursos avanzados en español. Tu framework
  con documentación en español sería el primero en la región para este nivel.

---

## Audiencia 3: Desarrolladores de Benchmarks de Hardware Cuántico

### ¿Quiénes son?

- **QED-C (Quantum Economic Development Consortium)**: define benchmarks estándar
- **Equipos de IBM, IonQ, Quantinuum, Rigetti** que publican benchmarks de sus chips
- **arXiv:2111.00044**: ya propusieron benchmark basado en Fermi-Hubbard 1D (24 qubits,
  3 vendors). Tu TFIM es más simple y escalable.
- **arXiv:2607.11637 (julio 2026)**: benchmark de optimización en hardware real

### ¿Qué problema tienen?

Los benchmarks actuales (quantum volume, CLOPS, random circuit sampling) miden el hardware
pero NO miden "¿qué tan útil es para física?". El paper de benchmark Fermi-Hubbard
(arXiv:2111.00044) demostró que benchmarks basados en aplicación son más informativos,
pero:
- Fermi-Hubbard es más complejo de implementar
- No existe un benchmark de spin models estandarizado con dificultad calibrada
- No hay una "escala de dificultad" continua como la que ofrece h en TFIM

### ¿Qué les ofreces?

Un **protocolo de benchmark con dificultad ajustable**:

```
Benchmark TFIM-HVA:
- Modelo: TFIM 1D (o heavy_hex para topología real)
- Dificultad: h ∈ [5.0, 1.0] (trivial → máxima)
- Tamaño: N = 4, 6, 8, 10, 16, 20 (escalable)
- Métrica: ΔE/gap < 5% = PASS
- Referencia: ground truth DMRG exacto (provisto)
- Comparación: tu resultado vs atlas h_min
```

**Score del hardware**: "Este QPU resuelve TFIM N=10 hasta h=1.5 con ZNE"
→ Comparable entre vendors sin ambigüedad.

### ¿Cómo se lo entregas?

**Opción A — Paper de benchmark** con protocolo estandarizado + resultados de referencia.
Publicar en **Quantum Science and Technology** como "Application-Level Benchmark".

**Opción B — Paquete benchmark ejecutable**:
```python
from qmbp_simulation.benchmark import TFIMBenchmark

bench = TFIMBenchmark(N=10, topology="heavy_hex")
results = bench.run(backend=your_backend, shots=8192)
bench.score()  # → "PASS up to h=1.8, FAIL below"
bench.compare_with_atlas()  # → "Consistent with expressibility limit"
```

### ¿Por qué lo usarían?

- **Necesidad no cubierta**: No existe benchmark de spin models estandarizado.
- **Dificultad ajustable**: h es un knob continuo (a diferencia de random circuits donde
  solo puedes variar profundidad discretamente).
- **Ground truth provisto**: DMRG exacto elimina ambigüedad en la evaluación.
- **El paper de KCuF₃ (IBM, marzo 2026)** demuestra que la comunidad QUIERE benchmarks
  basados en materiales magnéticos. Tu TFIM es el stepping stone más simple.

---

## Audiencia 4: Investigadores de Materia Condensada Computacional

### ¿Quiénes son?

- Grupos que usan DMRG/tensor networks pero quieren explorar QPU como alternativa
- Grupos que estudian transiciones de fase cuánticas en modelos de espín
- Específicamente: los ~50 grupos que citan Hauschild2018 (TeNPy DMRG) y están interesados
  en el puente clásico-cuántico

### ¿Qué problema tienen?

- DMRG escala exponencialmente en 2D (bond dimension χ crece rápido)
- Quieren evaluar si QPU puede ser útil para sus problemas antes de invertir
- No tienen expertise en Qiskit/circuitos cuánticos
- Necesitan ground truth clásico + comparación cuántica en un solo workflow

### ¿Qué les ofreces?

Un pipeline que conecta su mundo (Hamiltoniano + lattice) directamente con QPU sin que
necesiten saber de circuitos:

```python
from qmbp_simulation import PipelineRunner, LatticeConfig

# Su lenguaje familiar:
lattice = LatticeConfig(topology="triangular", N=10)
runner = PipelineRunner(model="tfim", h_values=[0.5, 1.0, 1.5, 2.0, 2.5])

# Todo automático:
results = runner.run_full_pipeline(lattice, mode="noiseless")
# → Phase 1: DMRG ground truth
# → Phase 2: VQE warm-start
# → Phase 3: GNN predictor
# → Phase 4: Report con métricas
```

### ¿Por qué lo usarían?

- **Bajo costo de entrada**: No necesitan aprender Qiskit, solo especificar el Hamiltoniano
- **Resultado inmediato**: "¿QPU sirve para mi problema?" → respuesta en minutos (simulación)
- **Publicable**: Pueden comparar DMRG vs QPU-pipeline en su paper como "quantum benchmarking"

---

## Audiencia 5: Startups y Empresas de QEM-as-a-Service

### ¿Quiénes son?

- **Qedma** (Israel): QESEM — servicio comercial de quantum error mitigation
- **Riverlane** (Cambridge): Deltaflow — error correction middleware
- **Q-CTRL** (Australia): Fire Opal — optimización y supresión de errores
- Equipos internos de IBM, Google, Amazon que desarrollan capas de mitigación

### ¿Qué problema tienen?

Necesitan **benchmarks contra los cuales medir sus servicios**. Cuando un cliente pregunta
"¿tu servicio de QEM funciona para mi VQE de spin model?", necesitan datos de referencia.

### ¿Qué les ofreces?

- **Dataset de referencia**: 93 runs ruidosos con ZNE, PEA, GF comparados
- **Reglas de decisión documentadas**: "CX < 18 → GF funciona, CX > 18 → PEA obligatorio"
- **GNN-QEM como baseline**: su servicio debería superar tu corrección zero-shot gratuita

### ¿Realista?

BAJA probabilidad de adopción directa (tienen sus propios datos). Pero como **referencia
académica** que citan para justificar sus claims de superioridad, sí funciona.

---

## Formato de Entrega: Análisis Costo-Beneficio

| Formato | Esfuerzo | Impacto | Audiencia primaria |
|---------|:--------:|:-------:|-------------------|
| Papers (A+B) en arxiv | 3-6 semanas | Alto (citaciones) | Investigadores |
| PyPI package (`qmbp-simulation`) | 2-4 semanas | Medio-alto (adopción) | Desarrolladores + grupos exp |
| Jupyter notebooks tutoriales | 1-2 semanas | Medio (educación) | Estudiantes + docentes |
| Benchmark protocol + dataset | 2-3 semanas | Alto (estándar) | Hardware vendors |
| IBM Quantum Learning module | 4-8 semanas | Muy alto (alcance) | Global (IBM tiene 40K students) |
| Presentación ASQC 2026 La Plata | 1 semana | Medio (regional) | Comunidad LATAM |

---

## Estrategia Recomendada (por orden cronológico)

### Fase 1: Inmediata (julio-agosto 2026)

1. **Paper A (GNN-QEM)** → arXiv preprint + enviar a PRResearch
   - Ya tienes todos los datos. Solo falta escribir.
   - Impacto: establece prioridad antes de que alguien publique algo similar a GEM
     pero para Hamiltonianos (el campo se mueve rápido)

2. **Paper B (Atlas)** → arXiv preprint + enviar a NJP o PRA
   - Compilar tablas existentes en formato paper
   - Incluir fits y predicciones testeables

### Fase 2: Corto plazo (septiembre-octubre 2026)

3. **Open-source GitHub** con documentación mínima:
   - README con install + quickstart
   - 3 notebooks de demo (express check, GNN-QEM, pipeline completo)
   - Tests que pasen en CI
   - License MIT

4. **Presentar en ASQC 2026** (La Plata, JAIIO)
   - Poster o talk sobre el pipeline
   - Contacto directo con comunidad Argentina/LATAM

### Fase 3: Medio plazo (nov 2026 - feb 2027)

5. **PyPI package** con API limpia
6. **Jupyter notebooks** para educación (Colab-compatible)
7. **Proponer a IBM Quantum Learning** como módulo educativo

---

## ¿Qué hace que ESTE proyecto sea útil vs los 6+ papers de GNN-warm-start?

La diferencia no es ningún componente individual. Es la **integración verificada a escala**:

| Lo que otros tienen | Lo que TÚ tienes (y nadie más) |
|--------------------|---------------------------------|
| GNN predice θ | GNN predice θ + sabes CUÁNDO falla + sabes POR QUÉ |
| VQE en simulación | VQE + DMRG + MPS hasta N=250 validado |
| ZNE genérico | PEA > GF documentado con 81 evaluaciones |
| Demo en 1 topología | 5 topologías × 4 profundidades × 8 modelos |
| Funciona/no funciona | Mapa cuantitativo de DÓNDE funciona y dónde no |
| Resultado positivo | 22 findings (15 positivos + 7 negativos validados) |

**Tu valor es el MAPA, no el VEHÍCULO**. Otros construyen vehículos (GNN, Flow, RL).
Tú construiste el mapa del terreno: qué se puede resolver, qué no, y por qué.
Eso es lo que un experimentalista necesita ANTES de gastar $10K en QPU time.

---

## Conclusión

La utilidad práctica se concentra en **tres productos tangibles**:

1. **Para investigadores**: Dos papers publicables con contribución original verificada
   (GNN-QEM cross-topology y Atlas de expresividad).

2. **Para experimentalistas**: Una herramienta de "pre-flight check" que ahorra QPU time
   al diagnosticar viabilidad ANTES de ejecutar (gratis, clásico, instantáneo).

3. **Para educadores**: El primer laboratorio completo en español de VQE aplicado a spin
   models que cubre el ciclo completo incluyendo fracasos (Heisenberg, frustración, ruido).

Ninguno de estos productos existe actualmente en el ecosistema. La combinación de
datos a escala + hallazgos negativos documentados + multi-topología validada es única.

---

*Investigado con búsqueda de literatura arxiv 2024-2026, ecosistema IBM Quantum,
y análisis del mercado de QEM/educación cuántica a julio 2026.*
