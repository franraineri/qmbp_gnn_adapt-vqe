# GNN-HVA v6.0 — Changes and Architecture Upgrade

V6.0 is a full modular rewrite of the pipeline, extracting all logic from monolithic notebooks into reusable Python modules under `src/poc/v6/`. The architecture preserves V4's proven single descending sweep with pure energy cost — the only approach that produced smooth θ landscapes — while addressing the three structural limitations that blocked V5.x: the MLP's inability to generalize beyond 1D uniform-coupling TFIM, the lack of optimization diagnostics, and the absence of a barren-plateau-free deployment alternative. The key insight from V5.x failures is that pipeline phases are tightly coupled: changing Phase 2's cost function without updating Phase 3 breaks the entire pipeline. V6 respects this by keeping Phase 2's pure energy cost intact and focusing structural changes on Phase 3 (MLP→MPNN) and Phase 4 (adding QRC).

The most significant change is replacing the V4 MLP predictor with a Message Passing Neural Network (MPNN) built on PyTorch Geometric. The MPNN takes graph-structured input — node features `[h_i, coordination_number_i]` with edge connectivity from the lattice topology — and produces θ_pred via GINConv layers + global mean pooling. This makes the predictor lattice-agnostic: the same trained model can accept graphs of different sizes and topologies (chain, ladder, Kagome, triangular) without retraining, which was impossible with the fixed-input MLP. The training loop includes energy-driven validation every 50 epochs via StatevectorEstimator, plus automatic divergence detection that halts training when MSE converges but ΔE stagnates — the diagnostic signature of the V5.x failure mode.

Phase 1 now supports arbitrary lattice topologies through `HamiltonianBuilder` (with `make_lattice()` factory for chain_1d, ladder, triangular, Kagome) and scales beyond N=14 via DMRG/TeNPy integration in `ClassicalSolver` (auto-selects exact diag for N<15, DMRG for N≥15, with memory fallback for 2D lattices). Phase 2 adds `OptimizationCallback` for full trajectory logging (energy, gradient proxy, parameters at every iteration) and expanded bounds [-π, π] with warm-start seeding (θ=0 for h=0). The VQE optimizer enforces the descending sweep direction and never wraps angles — preserving the smooth θ landscape that V4 demonstrated is essential for Phase 3 learnability.

Phase 4 introduces dual-route deployment: the main route (Adapt-VQE with max_iterations=2, catching AlgorithmError at iteration 0 as the ideal warm-start outcome) and a new QRC fallback route that uses a fixed random HVA circuit as a quantum reservoir with classical linear regression readout — eliminating barren plateaus by construction since no quantum parameters are ever optimized. Phase classification uses a data-driven ⟨X⟩ = ⟨ZZ⟩ crossover instead of the hardcoded h_c=1.0 threshold. Pipeline integrity safeguards include dataset metadata (`cost_function="energy"`, version, library versions), Phase 3 loading validation that rejects mismatched cost functions, and observable locality assertions ensuring all hardware-path operators act on ≤2 adjacent qubits.

New dependencies: `torch_geometric>=2.5` (MPNN), `physics-tenpy>=1.0` (DMRG), `scikit-learn>=1.4` (QRC readout). All modules are fully tested and the end-to-end pipeline has been validated on N=6 TFIM, producing all 5 validation metrics with a 2/6 checklist baseline matching V4 (with reduced training parameters for speed — full notebook runs with 27 h-points and 4000 MPNN epochs are expected to improve on this).

---

# GNN-HVA v6.0 — Cambios y Actualización de Arquitectura (Español)

V6.0 es una reescritura modular completa del pipeline. En las versiones anteriores (V3–V5), toda la lógica vivía dentro de notebooks monolíticos: construir el Hamiltoniano, resolver el estado fundamental, optimizar el circuito cuántico, entrenar la red neuronal y desplegar en hardware — todo mezclado en celdas de Jupyter. V6 extrae cada una de esas responsabilidades en módulos Python independientes (`hamiltonian_builder.py`, `classical_solver.py`, `hva_builder.py`, `vqe_optimizer.py`, `mpnn_predictor.py`, `qrc_pipeline.py`, `hardware_deployer.py`, `pipeline_utils.py`), con interfaces claras y dataclasses compartidas. Los notebooks ahora solo orquestan estos módulos — no contienen lógica propia. Esto hace que el código sea reutilizable tanto para el PoC (N=6) como para la implementación final de la tesis (N=12 ladder, N=40 DMRG, hardware real).

El cambio más importante es el reemplazo del MLP (perceptrón multicapa) por una MPNN (Red Neuronal de Paso de Mensajes) construida con PyTorch Geometric. El MLP de V4 recibía un solo número (el campo transversal h) y predecía los 4 parámetros del circuito. Esto funcionaba para la cadena 1D, pero no podía generalizar a otras topologías porque no "veía" la estructura del grafo. La MPNN recibe el grafo completo del Hamiltoniano — cada qubit es un nodo con features (h_i, número de coordinación), y las interacciones son aristas — y produce θ_pred mediante capas de convolución sobre grafos (GINConv) seguidas de un pooling global que colapsa el grafo a un vector de tamaño fijo. Esto significa que el mismo modelo entrenado puede recibir grafos de distinto tamaño y topología (cadena, escalera, Kagome, triangular) sin reentrenamiento. Además, el entrenamiento incluye validación energética cada 50 épocas y detección automática de divergencia — si el MSE baja pero el error energético se estanca, el entrenamiento se detiene con un diagnóstico que apunta a problemas en los datos de la Fase 2.

La Fase 1 ahora soporta topologías arbitrarias mediante `HamiltonianBuilder` (con una fábrica `make_lattice()` que genera automáticamente aristas y números de coordinación para cada topología) y escala más allá de N=14 gracias a la integración de DMRG vía TeNPy en `ClassicalSolver` (selección automática: diagonalización exacta para N<15, DMRG para N≥15, con fallback de memoria para redes 2D). La Fase 2 agrega `OptimizationCallback` para registrar la trayectoria completa de optimización (energía, norma del gradiente, parámetros en cada iteración), bounds expandidos a [-π, π], y seeding de warm-start (θ=0 para h=0). El optimizador VQE mantiene el sweep descendente único de V4 y nunca aplica wrapping de ángulos — preservando el paisaje suave de θ que V4 demostró es esencial para que la Fase 3 pueda aprender.

La Fase 4 introduce despliegue de doble ruta: la ruta principal (Adapt-VQE restringido con max_iterations=2, capturando AlgorithmError en iteración 0 como resultado ideal del warm-start) y una nueva ruta alternativa QRC (Quantum Reservoir Computing) que usa un circuito HVA con parámetros aleatorios fijos como reservorio cuántico, con un readout de regresión lineal clásica. La ventaja del QRC es que elimina los barren plateaus por construcción, ya que ningún parámetro cuántico se optimiza jamás. La clasificación de fase usa el cruce data-driven ⟨X⟩ = ⟨ZZ⟩ de los datos exactos de la Fase 1, en lugar del umbral hardcodeado h_c=1.0. Las salvaguardas de integridad del pipeline incluyen metadatos en el dataset (`cost_function="energy"`, versión, versiones de librerías), validación al cargar datos en la Fase 3 que rechaza funciones de costo incompatibles (previniendo el modo de fallo de V5.x), y aserciones de localidad que verifican que todos los observables para hardware actúan sobre ≤2 qubits adyacentes.

Dependencias nuevas: `torch_geometric>=2.5` (MPNN), `physics-tenpy>=1.0` (DMRG), `scikit-learn>=1.4` (readout QRC). Todos los módulos están testeados individualmente y el pipeline end-to-end fue validado sobre TFIM 1D N=6, produciendo las 5 métricas de validación con un checklist de 2/6 que iguala la línea base de V4 (con parámetros de entrenamiento reducidos por velocidad — las ejecuciones completas con 27 puntos de h y 4000 épocas de MPNN deberían mejorar este resultado).

---

# GNN-HVA v6.1 — Hardware Deployment & MPNN Enhancements

V6.1 extends the V6.0 modular architecture with a production-ready hardware deployment path and three MPNN enhancements. All new code lives in separate modules (`hardware_deployer_v61.py`, `config_v61.py`, `analysis_utils.py`) or extends `mpnn_predictor.py` — stable V6.0 modules remain untouched. The design is literature-founded: six error mitigation techniques from five 2024–2026 papers, validated against IBM hardware benchmarks (Kiiamov 2026, Larrucea 2026, Sharma 2026).

## New Modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `config_v61.py` | ~200 | Constants (shot budgets, ZNE thresholds, NN config) + dataclasses (`DeployResultV61`, `LayoutResult`, `GradientAnalysisResult`, `MPNNCheckpoint`) |
| `hardware_deployer_v61.py` | ~1300 | Full deployment orchestrator: `HardwareDeployerV61`, `LayoutSelector`, `ObservableGrouper`, `NNExtrapolator`, `build_estimator_options()`, `compute_shot_budget()` |
| `analysis_utils.py` | ~200 | `WeightGradientAnalyzer` — purely classical post-training analysis for unsupervised phase detection (zero QPU cost) |

## Phase 4: Hardware Deployment (HIGH PRIORITY)

### Error Mitigation Stack (5 layers, applied in order)

1. **Dynamical Decoupling (DD)**: XpXm pulse sequences during idle periods. Suppresses T₂ decoherence at zero shot overhead. Configured via `EstimatorV2.options.dynamical_decoupling`.

2. **Pauli Twirling**: 32 randomizations × 256 shots. Converts coherent gate errors into stochastic Pauli noise (easier to mitigate statistically). Configured via `EstimatorV2.options.twirling`.

3. **TREX (Twirled Readout Error eXtinction)**: Symmetrizes readout errors for statistical correction. Configured via `EstimatorV2.options.resilience.measure_mitigation`.

4. **Inhomogeneous ZNE** (Uvarov et al. 2024): Multiple qubit layouts with diverse Circuit Error Sums (CES). Linear regression on (CES, observable) pairs extrapolates to CES=0. No gate folding — exploits natural error rate variation across IBM heavy-hex topology.

5. **NN-Enhanced Extrapolation** (Sun et al. 2025, optional): When ≥5 data points available, a 2-layer MLP (16, 8) replaces linear regression for non-linear noise-energy relationships. Falls back to linear for <5 points.

### Key Design Decisions

- **Inhomogeneous ZNE over gate folding**: Gate folding amplifies noise uniformly; IBM processors have highly non-uniform error rates (0.1%–2%). Different qubit mappings provide natural noise scaling without circuit depth increase.
- **COBYLA over L-BFGS-B on hardware**: Gradient-free optimizer tolerates shot noise. With MPNN warm-start, typically 0 iterations needed anyway.
- **"Indeterminate" phase label**: When |⟨X⟩ - |⟨ZZ⟩|| ≤ σ (1/√shots), classification is uncertain. Forcing a label would be scientifically dishonest (Sharma 2026 confirms noise broadens the critical crossover).
- **3 PUBs per layout**: X-observables (list), ZZ-observables (list), full Hamiltonian (single op). Lists return per-term arrays; single op returns scalar energy.

### Layout Selection (`LayoutSelector`)

- Extracts gate error rates from `backend.target` (Qiskit 2.x API)
- BFS-based connected subset search on heavy-hex topology
- Selects 3–5 layouts maximizing CES spread (min ratio ≥ 2.0)
- Uses topology CES for ranking (fast heuristic), circuit CES for ZNE axis (true value)
- Seeded RNG for reproducibility; caches results per session

### Shot Budget Scaling

| System size | Shots | σ |
|-------------|-------|---|
| N ≤ 6 | 8,192 | 1.1e-2 |
| 7 ≤ N ≤ 10 | 16,384 | 7.8e-3 |
| N > 10 | 32,768 | 5.5e-3 |

User override allowed (minimum 4,096).

### Success Criterion

**ΔE/gap < 5% AND correct phase label** — not fidelity ≥ 99.5% (unmeasurable on hardware).

## Phase 3: MPNN Enhancements (MEDIUM PRIORITY)

### Per-Parameter Output Heads (Task 9)

Replaces the single `Linear(hidden, 2p)` head with two specialized heads:
- `head_zz`: `Linear(hidden, p)` → θ_zz (ZZ entangling parameters)
- `head_x`: `Linear(hidden, p)` → θ_x (X rotation parameters)

Physics motivation: θ_zz and θ_x have distinct optimization landscapes. Separate heads allow specialization. Outputs concatenated as `[θ_zz, θ_x]` for backward compatibility. Training reports per-head MSE for diagnosis.

### Edge Feature Encoding via NNConv (Task 10)

For non-uniform couplings (ladders with J_leg ≠ J_rung, 2D lattices):
- `build_graph_dataset(include_edge_features=True)` adds `edge_attr` tensors with per-bond J_ij
- `MPNNPredictor(use_edge_features=True)` replaces GINConv with NNConv layers
- Edge MLP: `Linear(1, 32) → ReLU → Linear(32, in_dim * out_dim)`
- Sum aggregation (`aggr="add"`) for WL-test equivalence (Xu et al. 2019)
- Falls back to GINConv for uniform J (no information gain from constant edge features)

### Weight Gradient Analysis (Task 11)

`WeightGradientAnalyzer` (Hernandes et al. 2025): detects phase transitions from trained MPNN weight structure.
- Computes ∂L/∂W for each h-value via forward+backward pass
- Reports per-layer gradient norms (each GINConv layer + MLP head)
- Peak detection via `scipy.signal.find_peaks` in critical region h ∈ [0.8, 1.4]
- **Validated**: smoke test detected peak at h=1.20 with 9 training points
- Zero QPU cost — purely classical post-training analysis

### Model Checkpoint with Architecture Metadata

`MPNNCheckpoint` dataclass stores:
- `state_dict` + architecture type (`"ginconv"` or `"nnconv"`)
- `per_parameter_heads`, `use_edge_features`, `hidden_dim`, `n_layers`, `output_dim`
- Training metadata (epoch, loss, dataset info)

`load_mpnn_checkpoint()` reconstructs the correct architecture from metadata before loading weights. Handles legacy V6.0 checkpoints gracefully (assumes single-head GINConv defaults).

## Critical Implementation Lessons

1. **EstimatorV2 returns scalar for multi-term SparsePauliOp, array for list of ops.** For per-site measurements, always submit as a list of individual operators.

2. **Never reconstruct energy manually** (`-J*sum(ZZ) - h*sum(X)`). Submit the full Hamiltonian as a PUB — the Estimator handles coefficient weighting correctly.

3. **Two types of CES**: topology CES (heuristic for layout ranking) vs circuit CES (true value for ZNE extrapolation). Never mix them.

4. **Modern IBM backends may not expose calibration timestamps.** Default to assuming fresh calibration when `backend.properties()` returns None.

5. **No reusable libraries exist** for inhomogeneous ZNE, heavy-hex layout selection, or weight gradient analysis. Mitiq does gate-folding only. DD/twirling/TREX are native to Qiskit Runtime.

## Test Coverage

- 33 tests total (18 V6.0 + 15 V6.1 integration tests)
- Smoke test: `scripts/smoke_test_v61.py` — 12 h-points, multi-point deployment, gradient analysis (~16s)
- All tests pass on Python 3.12 with Qiskit 2.x, PyTorch Geometric 2.7+

## Dependencies Added

- `scikit-learn>=1.4` (MLPRegressor for NN extrapolation)
- `scipy>=1.11` (find_peaks for gradient analysis)
- `qiskit-ibm-runtime>=0.30` (EstimatorV2, QiskitRuntimeService — hardware mode only)

---

# GNN-HVA v6.1 — Despliegue en Hardware y Mejoras al MPNN (Español)

V6.1 extiende la arquitectura modular de V6.0 con un camino de despliegue en hardware listo para producción y tres mejoras al MPNN. Todo el código nuevo vive en módulos separados (`hardware_deployer_v61.py`, `config_v61.py`, `analysis_utils.py`) o extiende `mpnn_predictor.py` — los módulos estables de V6.0 no se modifican. El diseño está fundamentado en la literatura: seis técnicas de mitigación de errores de cinco papers 2024–2026, validadas contra benchmarks de hardware IBM.

## Fase 4: Despliegue en Hardware

La pila de mitigación de errores aplica 5 capas en orden: (1) Desacoplamiento Dinámico (DD) con secuencias XpXm, (2) Pauli Twirling con 32 randomizaciones, (3) TREX para errores de lectura, (4) ZNE Inhomogéneo usando múltiples layouts de qubits con diferentes Circuit Error Sums (CES), y (5) extrapolación NN opcional con MLP de 2 capas. Las capas 1–3 son nativas de Qiskit Runtime (configuración declarativa). Las capas 4–5 son implementación propia — no existe biblioteca reutilizable.

La selección de layouts usa BFS sobre la topología heavy-hex del backend IBM para encontrar 3–5 subconjuntos conectados con CES diverso. La extrapolación lineal de (CES, observable) a CES=0 da el valor mitigado. El criterio de éxito es **ΔE/gap < 5% Y etiqueta de fase correcta** — no fidelidad ≥ 99.5% (imposible de medir en hardware).

## Fase 3: Mejoras al MPNN

- **Cabezas por parámetro**: Cabezas MLP separadas para θ_zz y θ_x, permitiendo especialización basada en la física distinta de cada tipo de parámetro.
- **Features de arista via NNConv**: Para acoplamientos no uniformes (escaleras con J_leg ≠ J_rung), NNConv procesa atributos de arista J_ij a través de un MLP aprendido. Agregación por suma (`aggr="add"`) para equivalencia con el test de Weisfeiler-Lehman.
- **Análisis de gradientes de pesos** (Hernandes et al. 2025): Detecta transiciones de fase desde la estructura interna del MPNN entrenado. Cero costo QPU — análisis puramente clásico post-entrenamiento. Validado: detectó pico en h=1.20 con solo 9 puntos de entrenamiento.

## Lecciones Críticas de Implementación

1. `EstimatorV2` devuelve escalar para `SparsePauliOp` multi-término, array para lista de ops. Para mediciones por sitio, siempre enviar como lista.
2. Nunca reconstruir energía manualmente — enviar el Hamiltoniano completo como PUB.
3. Dos tipos de CES: topología (heurística para ranking) vs circuito (valor real para extrapolación ZNE).
4. Backends IBM modernos pueden no exponer timestamps de calibración — asumir calibración fresca por defecto.
5. No existen bibliotecas reutilizables para ZNE inhomogéneo, selección de layouts en heavy-hex, ni análisis de gradientes de pesos.
