# Integration Plan 04: Qracle Unified Hamiltonian+Circuit Graph

**Paper:** Zhang et al. (2025) — Qracle: A GNN-based Parameter Initializer for VQEs
**arXiv:** 2505.01236
**Code:** ✅ `https://github.com/chizhang24/Qracle`
**Priority:** MEDIUM (3-4 days, strongest benefit for `tfim_bond_resolved`)
**Status:** ✅ IMPLEMENTED — awaiting B.1 cross-topology validation

## What It Does

Qracle encodes BOTH the Hamiltonian structure AND the ansatz circuit structure
into a single unified graph. Hamiltonian terms become nodes/edges, AND circuit
gates become additional nodes with connectivity reflecting the circuit topology.
The GNN processes this combined representation to predict optimal parameters.

Our current approach: graph = Hamiltonian only (qubits=nodes, interactions=edges).
Qracle's approach: graph = Hamiltonian + Circuit (adds gate nodes + parameter nodes).

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ✅ Implemented |
| Requires new dependencies? | ❌ PyTorch Geometric (already have) |
| Reuses existing modules? | ✅ `MPNNPredictor` (same GINConv backbone) |
| Addresses a real problem? | ⚠️ For global-param HVA: NO (circuit is uniform) |
| For bond-resolved HVA? | ✅ YES — each bond has different θ, circuit structure matters |
| Publishable? | ✅ If improves bond-resolved cross-N generalization |

## Implementation Status

### ✅ Step 1: Graph Construction (`predictors/unified_graph.py`)

- `build_unified_bond_resolved_graph()` — Constructs heterogeneous graph
- `build_unified_dataset()` — Batch dataset builder
- `validate_unified_graph()` — Structural validation (p>1 aware)
- `compute_graph_metrics()` — Expansion ratios, density, node counts

Node types:
  - type=0: Qubit nodes (N) — features: [h_i, coord_i, N/100, 0]
  - type=1: ZZ gate nodes (n_edges × p) — features: [layer/p, bond/n_e, N/100, 1]
  - type=2: RX gate nodes (N × p) — features: [layer/p, qubit/N, N/100, 2]

Edge types (encoded via connectivity):
  - Hamiltonian: qubit ↔ qubit (lattice.edges, bidirectional)
  - Gate-qubit: gate_node ↔ qubit(s) it acts on (bidirectional)
  - Intra-layer: ZZ gate → RX gate (circuit causal order)
  - Inter-layer: RX gate (layer l) → ZZ gate (layer l+1)

### ✅ Step 2: Type-Aware Architecture (`predictors/unified_mpnn.py`)

- `UnifiedMPNN` class — Type-conditioned GNN with:
  - Learned type embedding (nn.Embedding, configurable dim, default 16)
  - Shared GINConv backbone (all node types participate in message passing)
  - Gate-node readout: θ_zz predicted from ZZ gate node embeddings
  - Qubit/RX-node readout: θ_x from qubit (p=1) or RX gate (p>1) embeddings
  - Fallback mode: edge concatenation (gate_readout=False)
  - Full p>1 multi-layer support
- `train_unified_mpnn()` — Training loop with:
  - Weight decay (default 1e-4, higher than BondResolvedMPNN)
  - Gradient clipping (max norm 1.0)
  - Train/val split with periodic validation
  - p>1 target layout rearrangement (interleaved → grouped)
  - Early stopping on LR exhaustion

### ✅ Step 3: Integration & Helpers

- `experiments/helpers/graph_utils.py::train_unified_mpnn_variant()` — Reusable
  training wrapper matching `train_bond_resolved_variant` interface
- `evaluate_bond_resolved_variant()` — Already works with UnifiedMPNN (same
  forward interface: Data → [1, n_params])
- Package exports in `predictors/__init__.py`

### ✅ Step 4: Benchmark Runner

- `scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py`
  - Compares Variant E (BondResolvedMPNN + unified graph) vs Variant F (UnifiedMPNN)
  - Multi-topology support (chain_1d, ladder, square, triangular)
  - Paired t-test + Cohen's d effect size analysis
  - Train/test split on h-grid (even/odd indices)

### ✅ Step 5: Initial Evaluation (chain_1d, from EXPERIMENT_PLAN_04_06.md)

Results from the `run_noise_aware_comparison.py` 2×2 ablation:

| Variant | Graph | θ target | Final MSE | Mean ΔE/gap | Pass@5% |
|---------|-------|----------|:---------:|:-----------:|:-------:|
| A (baseline) | Ham-only | noiseless | 2.21e-04 | 0.0604 | 63% |
| B (unified) | Unified | noiseless | **8.51e-05** | 0.0634 | 58% |
| C (noisy) | Ham-only | noisy | 4.72e-03 | 0.0862 | 47% |
| D (combined) | Unified | noisy | 6.03e-04 | 0.0924 | 37% |

**Key findings (chain_1d N=10 p=1):**
1. Unified graph improves training MSE by 61% (8.5e-5 vs 2.2e-4)
2. But deployment ΔE/gap is **neutral** (Cohen's d = -0.30, not significant)
3. On chain_1d, all gates are equivalent by translational symmetry → no heterogeneity to exploit
4. The graph memorizes better without generalizing better

## Extended Plan: Unified Graph for Cross-N Scaling & Deep HVA

**Meta-objetivo:** Usar la MPNN entrenada en configuraciones pequeñas/fáciles para
predecir θ_opt en sistemas más grandes (N mayor) y regímenes más difíciles (h menor
con p mayor), minimizando el VQE necesario. Reutilizar todos los objetos y modelos
creados en cada paso como insumo para el siguiente.

### Contexto: Por qué el unified graph es clave para scaling

El cross-N con global HVA (2 params) ya funciona (30/30 PASS, train N=40+80 → predict N=100).
Pero con bond-resolved (19-79 params), la MPNN necesita entender la **estructura del circuito**
para extrapolar correctamente a grafos más grandes. El unified graph provee exactamente eso:
los gate nodes codifican la topología del circuito de forma invariante al tamaño.

### What was already tested (DO NOT REPEAT)

| Idea | Resultado | Referencia |
|------|-----------|------------|
| Más h-points (>20) | Diminishing returns extremos (k=5 ya pasa a N=10) | G1, S4 |
| hidden_dim ablation (64/128/256) | Todos rinden igual; 128 = sweet spot | Thesis variants |
| Data augmentation (θ noise) | DAÑA a N=10; marginal a N=6 | poc-results.md |
| Model más grande (>256) | Sin sentido: 500-1000× overparameterized para 20 pts | Análisis de capacidad |

---

### Phase B.1 — Cross-Topology Validation (READY TO EXECUTE)

**Hipótesis:** En topologías no-simétricas (ladder, square), los gate nodes tienen
conectividad heterogénea. UnifiedMPNN debería explotar esta información.

**Métrica clave:** `gate_neighborhood_cv` (CV del vecindario de los ZZ gates).
- chain_1d: 0.092 (bajo), ladder: 0.120 (medio), square: 0.154 (alto)

**Comando:**
```bash
.venv/bin/python scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py \
    --topology ladder --n-qubits 10 --p-layers 1 --h-min 1.3 --h-max 3.0 \
    --h-points 20 --mpnn-epochs 2000 --hidden-dim 256 --type-embedding-dim 16 \
    --test-fraction 0.4 --verbose
```

**Criterio de decisión:**
- Cohen's d ≥ 0.3 (F vs A) → Proceder a B.2
- Cohen's d < 0.2 → Hallazgo negativo documentable; saltar a B.3 directamente

**Tiempo estimado:** ~5 min (ladder N=10)

---

### Phase B.2 — Cross-N Transfer con Unified Graph (PRINCIPAL)

**Objetivo:** Entrenar UnifiedMPNN en N=10 (bond-resolved, ~19 params) y predecir
θ_opt para N=20, N=40, N=60 SIN reentrenar. El unified graph es size-agnostic
(per-node/per-edge prediction) — solo cambia el número de nodos del grafo.

**Estrategia:**
1. Entrenar en N=10 (chain_1d, 20 h-points, VQE noiseless) → modelo M₁
2. Generar VQE ground truth en N=20 (5-10 h-points solo, para validación)
3. Predecir θ(N=20) usando M₁ con grafo de N=20 (29 nodos unified)
4. Evaluar ΔE/gap en N=20 — ¿pasa sin reentrenar?
5. Si no pasa: fine-tune M₁ con los 5-10 puntos de N=20 → M₂
6. Predecir N=40, N=60 con M₂

**Reutilización de objetos:**
- M₁ (checkpoint del B.1) → warm-start para B.2
- Dataset de VQE N=10 del B.1 → training base
- `evaluate_bond_resolved_variant()` → evaluación en cualquier N
- `build_unified_bond_resolved_graph()` → construye grafos para N arbitrario
- `save_mpnn_checkpoint()` / `load_mpnn_checkpoint()` → persistencia entre pasos

**Infraestructura necesaria:**
- [ ] Adaptar `UnifiedMPNN` para multi-N batching (grafos de distintos tamaños en un batch)
- [ ] Crear helper `fine_tune_unified_mpnn(model, new_dataset, lr=1e-4, epochs=500)`
- [ ] Validar que `BondResolvedMPNN` con unified graph funciona cross-N (norm_type="none")

---

### Phase B.3 — Deep HVA (p=3-5) para h Más Bajos

**Objetivo:** Con p≤2, h_min ≈ 1.57 + 0.005·N (no podemos predecir debajo).
Con p=3-5, h_min baja a ~1.0-1.4, accediendo a la transición de fase real.
La MPNN debería predecir θ para circuitos más profundos usando el unified graph.

**Estrategia:**
1. Ejecutar VQE con p=3 en N=10, chain_1d, h ∈ [0.8, 3.0] (30 params bond-resolved)
2. Construir unified graph con p=3 (más gate nodes: 9×3=27 ZZ + 10×3=30 RX)
3. Entrenar UnifiedMPNN en p=3 → modelo M_p3
4. Evaluar: ¿predice bien en h < 1.3 donde p=2 fallaba?
5. Si sí: transfer M_p3 a N=20 p=3 (cross-N + deep)

**Reutilización:**
- Misma arquitectura UnifiedMPNN (ya soporta p>1)
- `build_unified_bond_resolved_graph(p_layers=3)` → ya implementado
- `HVACircuitBuilder.create_bond_resolved(N, p=3, lattice)` → ya existe
- Modelo M₁ de B.1 → inicialización parcial de pesos (transfer learning)

**Restricción hardware:** p=3-5 solo es viable en noiseless o con ZNE agresivo.
Para QPU real, p≤3 con PEA-ZNE (ya implementado).

---

### Phase B.4 — Multi-Topology Transfer

**Objetivo:** Entrenar UN solo modelo en múltiples topologías y predecir en
topologías no vistas. El unified graph codifica la topología explícitamente
→ el modelo debería generalizar.

**Estrategia:**
1. Entrenar en {chain_1d, ladder} N=10 (dataset combinado)
2. Predecir en {square, triangular} N=10 sin VQE adicional
3. Si funciona: es "zero-shot cross-topology" — muy publicable

**Reutilización:**
- Modelos M₁/M₂ de pasos anteriores → fine-tune con dataset mixto
- `run_cross_topology.py` (ya existe) → adaptar para unified graph
- `evaluate_bond_resolved_variant()` con lattice distinto → ya funciona

---

### Principio de Reutilización Progresiva

Cada fase produce artefactos que alimentan la siguiente:

```
B.1 (ladder N=10 p=1) → checkpoint M₁ + dataset D₁
    ↓
B.2 (cross-N: M₁ → predict N=20,40) → fine-tuned M₂ + validation data
    ↓
B.3 (deep HVA: p=3, h<1.3) → M_p3 (pesos inicializados desde M₁)
    ↓
B.4 (cross-topology) → modelo universal M_multi
```

**Reglas:**
1. NUNCA entrenar desde cero si hay un modelo previo relevante (siempre warm-start)
2. SIEMPRE guardar checkpoints con `save_mpnn_checkpoint()` + metadata del training
3. SIEMPRE reutilizar datasets de VQE previos (el VQE es el cuello de botella, no la MPNN)
4. SIEMPRE evaluar con `evaluate_bond_resolved_variant()` (interfaz unificada)
5. Documentar cada resultado en el JSON de la run (para el ResultIndex)

---

### Métricas de Éxito (Post-hoc, sin umbrales fijos)

| Fase | Métrica principal | Qué mide |
|------|-------------------|----------|
| B.1 | Cohen's d (F vs A) por topología | ¿La arquitectura type-aware ayuda? |
| B.2 | ΔE/gap en N_target sin reentrenar | ¿El unified graph generaliza cross-N? |
| B.3 | h_min alcanzable con p=3-5 | ¿Accedemos a la transición de fase? |
| B.4 | ΔE/gap en topología no vista | ¿Zero-shot cross-topology funciona? |
| Global | VQE-points necesarios / calidad | ¿Cuánto VQE ahorramos con la MPNN? |

## Architecture Details

```
src/qmbp_simulation/
└── predictors/
    ├── mpnn.py              # ✅ MPNNPredictor, BondResolvedMPNN
    ├── unified_graph.py     # ✅ Graph construction (Hamiltonian+Circuit)
    └── unified_mpnn.py      # ✅ Type-aware GNN architecture
```

### UnifiedMPNN vs BondResolvedMPNN

| Feature | BondResolvedMPNN | UnifiedMPNN |
|---------|:----------------:|:-----------:|
| Node type awareness | Mask at readout | Learned embedding |
| θ_zz prediction | Edge concatenation | Gate-node readout |
| Type embedding | None | nn.Embedding(3, 16) |
| p>1 support | Via edge_list repeat | Native (all gate nodes) |
| Cross-topology benefit | Limited | Expected (heterogeneity) |
| Parameters (hidden=256) | ~400K | ~426K (+6%) |

### When to Use

- **UnifiedMPNN**: Non-symmetric topologies (square, ladder, heavy_hex, triangular)
  where qubit degree varies → gate nodes carry structural info
- **BondResolvedMPNN**: chain_1d or any topology with translational symmetry
  → simpler model is sufficient, no benefit from unified graph

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting on larger graph (2.5-3× nodes) | weight_decay=1e-4, val_fraction=0.2 |
| No benefit on symmetric topologies | Confirmed: chain_1d is neutral. Focus on 2D. |
| p>1 target layout mismatch | Fixed: rearrange interleaved → grouped in loss |
| Training time increase | ~1.3× in practice (more nodes, same epochs) |
| Gate-node redundancy (same circuit every h) | Type embedding learns to downweight redundant info |

## References

- Zhang et al. (2025) "Qracle" arXiv:2505.01236
- EXPERIMENT_PLAN_04_06.md — Combined #04 + #06 experiment results
- Meng et al. (2025) arXiv:2504.00464 — GNN > CNN for circuit properties
- Xu et al. (ICLR 2019) — GINConv theoretical expressiveness

---

## Confirmed Findings (2026-07-28)

### F1. MPNN replaces VQE within the valid regime
- Chain_1d N=10 p=1: MPNN direct prediction ΔE/gap = 6.6%, VQE (10 restarts) = 7.0%
- Ratio MPNN/VQE = 1.02× (filtered reanalysis)
- Trained on 5 anchor points only

### F2. 3× VQE speedup with MPNN warm-start
- Baseline: 339s (20 pts × 10 restarts). Accelerated: 109s (5 anchor + MPNN + 15 warm-start)
- 100% quality preservation (15/15 target points within 1.5× baseline)
- MPNN training cost (45s) amortizes immediately

### F3. Unified graph does NOT improve deployment on symmetric topologies
- Chain_1d: Training MSE improves 61% but deployment ΔE/gap is neutral (d = -0.30)
- All ZZ gates are equivalent by translational symmetry → circuit structure is redundant

### F4. Type-aware architecture (UnifiedMPNN) gives statistically significant but operationally irrelevant improvement on ladder
- Ladder N=10 p=1: Cohen's d = +1.10 (F vs E), but absolute improvement = 1.2% on 500% error base
- The error base is dominated by VQE non-convergence, not MPNN architecture

### F5. VQE bond-resolved convergence limit is topology-dependent
- Chain_1d p=1: converges to h ≈ 1.3 (ΔE/gap < 20%)
- Ladder p=1: converges only to h ≈ 2.2
- Scales with lattice coordination number

### F6. Extrapolation works UP (h > h_max) but fails DOWN (h < h_min)
- h > 3.0: ΔE/gap < 1% (paramagnetic phase trivial)
- h < 1.3: ΔE/gap = 40-1400% (ansatz not expressive, warm-start cannot help)

### F7. Noise-aware training with Gaussian shot noise is counterproductive
- Cohen's d = -1.47, 0/19 wins. COBYLA + shot noise → randomly displaced minima
- Needs real hardware noise (FakeTorino/T1-T2) to capture relevant phenomenon
