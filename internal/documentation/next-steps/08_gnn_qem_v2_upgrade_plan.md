# Plan 08: GNN-QEM V2 — Upgrade para Escalabilidad a N Grande

**Fecha**: 2026-08-03
**Objetivo**: Refactorear GNNQEMCorrector para escalar de N=6-10 a N=50-127+
    **Módulo target**: `src/qmbp_simulation/predictors/gnn_qem.py`
    **Prerequisitos**: Data existente en Zoo + GroundTruthCache + FakeBackend disponible

---

## Resumen Ejecutivo

El GNN-QEM actual (GINConv, hidden=64, 3 capas, global_mean_pool) funciona a N≤10 pero
no escala. Este plan implementa 5 mejoras incrementales, cada una con su test A/B contra
la versión actual, para construir un `GNNQEMCorrectorV2` que funcione hasta N=127.

---

## Repos externos a reutilizar

| Repo | URL | Licencia | Qué copiar/importar |
|------|-----|----------|---------------------|
| **IBM ML-QEM** | https://github.com/qiskit-community/ml-qem | Apache-2.0 | `docs/tutorials/gnn.py` (GNN model), circuit features, 100Q TFIM dataset |
| **Trans-GNN Circuit** | https://github.com/Prof-it/trans-gnn-QuantumCircuitPrediction | MIT | Transformer-GNN con noise node features, encoding de circuito como grafo |
| **PyG GATv2Conv** | https://github.com/pyg-team/pytorch_geometric (ya instalado) | MIT | `torch_geometric.nn.conv.GATv2Conv` — drop-in con edge_attr |
| **PyG VirtualNode** | https://github.com/pyg-team/pytorch_geometric (ya instalado) | MIT | `torch_geometric.nn.VirtualNode` transform |
| **GNN Hardware Predictor** | https://github.com/antotu/GNN-Model-Quantum-Predictor | — | Referencia de DAG encoding (no copiar directamente) |

---

## Paso 0: Baseline — Medir rendimiento actual (1 día)

**Qué hacer**:
- Ejecutar `run_gnn_qem_training.py` con config estándar
- Guardar métricas: val_MAE, val_improvement_pct, per-N breakdown
- Generar dataset de test holdout (N=6, N=10) para comparar en todos los pasos

**Métricas baseline**:
```
val_MAE_baseline = ?
improvement_pct_baseline = ?
N=6 MAE, N=10 MAE (separados)
```

**Criterio de éxito para cada paso**: mejora > 10% en val_MAE vs paso anterior.

---

## Paso 1: Training data con θ_opt del Zoo (2 días)

**Prioridad**: 🔴 ALTA
**Impacto esperado**: ALTO (circuitos representativos del deployment real)
**Esfuerzo**: BAJO (data ya existe)

### Qué cambiar

`generate_qem_training_data()` actualmente usa `theta = rng.uniform(-1, 1, n_params)`.
Reemplazar con θ_opt cargados del model_zoo/caché:

```python
# NUEVO: cargar θ_opt del zoo en vez de random
from qmbp_simulation.predictors.model_zoo import load_pretrained
mpnn = load_pretrained(model="tfim", topology=topo, n_qubits=n_qubits)
theta_pred = mpnn.predict(graph_data)  # θ representativo del deployment
bound = qc.assign_parameters(theta_pred)
```

Para h-points sin MPNN disponible, usar θ del GroundTruthCache (VQE noiseless).

### Test A/B

| Variante | Training data | Modelo |
|----------|---------------|--------|
| A (baseline) | θ random | GNNQEMCorrector actual |
| B (este paso) | θ_opt del Zoo | GNNQEMCorrector actual |

**Métricas**: val_MAE, improvement_pct sobre mismo test set holdout.
**Pass si**: B mejora > 10% sobre A en val_MAE.

### Fuentes de θ_opt concretas (prioridad de uso)

1. **NPZ datasets** (`data/*.npz`): Contienen `theta_opt` array directamente.
   Load con `load_phase12_dataset(path)["theta_opt"]`. Ya hay sweeps para
   chain_1d, ladder, heavy_hex a N=6,10,16,20.

2. **Model Zoo MPNN** (`load_pretrained()`): Para h-points sin NPZ, predecir
   θ con el MPNN. Disponible para todas las topologías/N del zoo.

3. **EvalCache lookup**: Para verificar E_exact sin re-computar. Key format:
   `"GT|{model}|{topology}|{n_qubits}|{h:.6f}"`.

### Datos adicionales a incorporar en el context

Del GroundTruthCache, agregar como features de contexto:
- `gap / 10.0` — normalizado, informa sensibilidad al ruido (H1 de la investigación)
- Separar CES en `CES_2q` y `CES_readout` (H2)

### Código reutilizable
- `model_zoo.load_pretrained()` — ya existe
- `GroundTruthCache.get()` — ya existe
- `load_phase12_dataset()` — ya existe
- `EvalCache.get_ground_truth()` — ya existe
- No se necesita código externo para este paso

---

## Paso 2: Per-edge 2Q gate errors + GATv2Conv (3 días)

**Prioridad**: 🔴 ALTA
**Impacto esperado**: ALTO (captura heterogeneidad de errores entre enlaces)
**Esfuerzo**: MEDIO

### Qué cambiar

1. **`build_qem_graph()`**: Agregar `Data.edge_attr` con error 2Q por arista
2. **`GNNQEMCorrectorV2`**: Reemplazar GINConv por GATv2Conv con `edge_dim=1`

```python
# En build_qem_graph():
# ANTES: gate_errors promediados al nodo
# DESPUÉS: per-edge error como edge_attr
edge_attr = torch.zeros(edge_index.size(1), 1)
for i, (src, dst) in enumerate(edge_index.T):
    key = f"{min(layout[src], layout[dst])}-{max(layout[src], layout[dst])}"
    edge_attr[i, 0] = cal_snap.gate_errors_2q.get(key, 0.005)

data.edge_attr = edge_attr
```

```python
# En el modelo:
from torch_geometric.nn import GATv2Conv

# Reemplaza GINConv:
self.convs = nn.ModuleList()
self.convs.append(GATv2Conv(
    in_channels=node_feature_dim,
    out_channels=hidden_dim // n_heads,
    heads=n_heads,
    edge_dim=1,  # ← per-edge 2Q gate error
    concat=True,
))
# Siguientes capas: hidden_dim → hidden_dim
for _ in range(n_layers - 1):
    self.convs.append(GATv2Conv(
        in_channels=hidden_dim,
        out_channels=hidden_dim // n_heads,
        heads=n_heads,
        edge_dim=1,
        concat=True,
    ))
```

### Test A/B

| Variante | Graph | Modelo |
|----------|-------|--------|
| A (paso 1) | node features + edge_index | GINConv (actual) |
| B (este paso) | node features + edge_index + **edge_attr** | **GATv2Conv** |

**Métricas**: val_MAE, improvement_pct. **Además**: comparar per-h performance para
detectar si edges importan más cerca de h_c (donde circuitos son más profundos).
**Pass si**: B mejora > 15% sobre A.

### Código reutilizable
- **PyG GATv2Conv** (`torch_geometric.nn.conv.GATv2Conv`): ya instalado, API compatible
- **IBM ml-qem `gnn.py`** (https://github.com/qiskit-community/ml-qem/blob/research/docs/tutorials/gnn.py):
  referencia de cómo IBM estructura su GNN para QEM. Estudiar features y encoding.
- **Trans-GNN** (https://github.com/Prof-it/trans-gnn-QuantumCircuitPrediction):
  implementación funcional con noise features por nodo.

### Data augmentation de calibración (incorporar aquí, no como paso separado)

Per arXiv:2509.12933 (H5 de la investigación), agregar en `build_qem_graph()`:
```python
if augment:
    # Perturbar calibración ±20% para simular drift temporal
    noise_scale = 0.2
    t1_norm *= (1 + noise_scale * torch.randn_like(t1_norm))
    t2_norm *= (1 + noise_scale * torch.randn_like(t2_norm))
    readout *= (1 + noise_scale * torch.randn_like(readout))
    edge_attr *= (1 + noise_scale * torch.randn_like(edge_attr))
```
Esto es GRATIS (no requiere regenerar data) y mejora generalización.

---

## Paso 3: Node features expandidas + Gate count per qubit (1 día)

**Prioridad**: 🟡 MEDIA
**Impacto esperado**: MEDIO
**Esfuerzo**: BAJO

### Qué cambiar

Expandir node_feature_dim de 4 a 6:

```python
# ANTES: [T1/100, T2/100, readout_err, gate_err_max]
# DESPUÉS: [T1/100, T2/100, readout_err, gate_err_max, n_cx_local/max_cx, degree/max_degree]

# n_cx_local: cuántos 2Q gates tocan este qubit en el circuito transpilado
# degree: grado del nodo en el coupling map (connectivity del qubit)
```

Calcular `n_cx_local` en `generate_qem_training_data()`:
```python
cx_per_qubit = np.zeros(n_qubits)
for inst in transpiled.data:
    if inst.operation.num_qubits == 2:
        for q in inst.qubits:
            idx = transpiled.find_bit(q).index
            cx_per_qubit[idx] += 1
max_cx = max(cx_per_qubit.max(), 1)
# Agregar como 5to feature: cx_per_qubit / max_cx
```

### Test A/B

| Variante | Node features |
|----------|---------------|
| A (paso 2) | [T1, T2, readout, gate_err] (4 dims) |
| B (este paso) | [T1, T2, readout, gate_err, **n_cx_local, degree**] (6 dims) |

**Pass si**: mejora > 5% en val_MAE (marginal improvement esperada pero valiosa a N grande).

### Código reutilizable
- Cálculo de gate count per qubit: adaptable de `execution.noisy_utils.compute_circuit_ces()`
- No requiere código externo

---

## Paso 4: Virtual Global Node (2 días)

**Prioridad**: 🟡 MEDIA-ALTA
**Impacto esperado**: ALTO a N grande (resuelve diámetro sin más capas)
**Esfuerzo**: MEDIO

### Qué cambiar

Agregar un nodo virtual (#N) conectado a todos los qubits. Su feature inicial es el
context vector [h, n_2q/50, CES, E_noisy/N]. Participa en message-passing.

```python
from torch_geometric.transforms import VirtualNode

# Opción 1: usar PyG VirtualNode transform
transform = VirtualNode()
data = transform(data)  # agrega nodo virtual con edges bidireccionales a todos

# Opción 2: implementación manual (más control sobre features del virtual node)
# Features del virtual node: [h, n_2q/50, CES_2q, CES_readout, E_noisy/N, gap/10, sign_bias]
virtual_feat = torch.tensor([[h, n_2q/50, ces_2q, ces_readout, e_noisy/N, gap/10.0, -1.0]],
                            dtype=torch.float32)
# Pad to node_feature_dim con zeros (o usar linear projection)
virtual_feat_padded = self.virtual_node_proj(virtual_feat)  # project to hidden_dim
```

**Cambio en readout**: En vez de global_mean_pool + concat(context), usar directamente
el embedding del virtual node como representación global:
```python
# ANTES:
pooled = global_mean_pool(x, batch)
out = torch.cat([pooled, context], dim=1)

# DESPUÉS:
# El virtual node ya absorbió toda la info global
virtual_node_emb = x[virtual_node_indices]  # [batch_size, hidden_dim]
out = virtual_node_emb  # no necesita concat — ya tiene context como feature
```

### Test A/B

| Variante | Readout | Capas necesarias para cubrir grafo |
|----------|---------|:---:|
| A (paso 3) | global_mean_pool + concat(context) | 3 (insuficiente a N>10) |
| B (este paso) | **virtual node embedding** | 1 hop extra = alcanza todo |

**Test especial**: Probar a N=20 (donde tenés DMRG como ground truth) y comparar.
Si el virtual node ayuda, la mejora será más visible a N=20 que a N=6.

**Pass si**: mejora > 10% a N=20 (puede ser neutral a N=6, eso es OK).

### Código reutilizable
- **PyG VirtualNode**: `from torch_geometric.transforms import VirtualNode` (ya instalado)
- **GTranQEM** (https://openreview.net/forum?id=XnVttczoAV): paper de referencia
  para el "virtual quantum-representative node". No tiene código público pero la
  implementación con PyG VirtualNode es directa.

---

## Paso 5: Scaling vía pseudo-labels ZNE para N>20 (3 días)

**Prioridad**: 🟡 MEDIA
**Impacto esperado**: ALTO (extiende el modelo a N donde no hay ground truth exacto)
**Esfuerzo**: MEDIO-ALTO

### Concepto: "Three-Stair Scaling"

Inspirado en arXiv:2411.16354. Tres regímenes de training:

| Régimen | N | Label (target) | Fuente |
|---------|---|----------------|--------|
| **Tier 1** (supervisado) | 6, 10, 16, 20 | E_exact (solver/DMRG) | GroundTruthCache |
| **Tier 2** (semi-supervisado) | 30, 50 | E_zne (PEA-ZNE como proxy) | Hardware rehearsal data |
| **Tier 3** (deployment) | 100, 127 | — (solo inferencia) | FakeBackend calibration |

### Qué cambiar

1. **Ampliar `generate_qem_training_data()`** con modo `pseudo_label="zne"`:
```python
if pseudo_label == "exact":
    target = e_exact  # ground truth del solver
elif pseudo_label == "zne":
    # Usar PEA-ZNE como pseudo-label (mejor estimación disponible)
    target = run_pea_zne(transpiled, H, backend, noise_factors=[1,3,5]).extrapolated
```

2. **Loss ponderada por confianza del label**:
```python
# Tier 1 (exacto): weight = 1.0
# Tier 2 (ZNE): weight = 0.5 (menos confiable)
loss = weighted_mse(delta_e_pred, target, weights=label_confidence)
```

### Test A/B

| Variante | Training data |
|----------|---------------|
| A (paso 4) | Solo N=6,10 con E_exact |
| B (este paso) | N=6,10 (exact) + N=20,30 (ZNE pseudo-labels) |

**Test especial**: Evaluar a N=20 con ground truth exacto (DMRG) para ver si
pseudo-labels de ZNE a N=30 MEJORAN la predicción a N=20 (transfer knowledge).

**Pass si**: MAE mejora en N=20 holdout comparado con modelo entrenado solo en N=6,10.

### Código reutilizable
- **`run_pea_zne()`** de `execution.noisy_utils` — ya existe
- **IBM ml-qem** 100Q TFIM dataset (https://github.com/qiskit-community/ml-qem/blob/research/docs/tutorials/):
  Contiene data real de 100Q con ZNE que puede servir como referencia para validación
  cruzada (¿nuestro modelo entrenado en N≤20 predice correctamente su data de 100Q?)
- **arXiv:2411.16354** three-stair pattern: concepto, no código (implementar sobre nuestro framework)

---

## Paso 6 (Opcional): Laplacian Positional Encoding (1 día)

**Prioridad**: 🟢 BAJA (solo si pasos 1-5 muestran que N-generalization falla)
**Impacto esperado**: MEDIO (hace la representación scale-invariant)
**Esfuerzo**: BAJO

### Qué cambiar

Agregar positional encoding basado en eigenvectores del Laplaciano del grafo:

```python
from torch_geometric.transforms import AddLaplacianEigenvectorPE

transform = AddLaplacianEigenvectorPE(k=4)  # top-4 eigenvectors
data = transform(data)
# Esto agrega data.laplacian_eigenvector_pe de shape [n_nodes, 4]
# Concatenar con node features: [T1, T2, readout, gate_err, n_cx, degree, pe_0..pe_3]
```

Esto da a cada nodo una "posición" en el grafo que es invariante al tamaño.
Un qubit en el centro de un chain_1d N=10 tiene PE similar a uno en el centro
de chain_1d N=50.

### Test A/B
Solo relevante si Paso 5 muestra que el modelo NO generaliza bien entre N distintos.
Si Pasos 1-5 ya generalizan bien → SKIP.

### Código reutilizable
- **PyG** `AddLaplacianEigenvectorPE` — ya instalado, 1 línea

---

## Convivencia V1 / V2 y Backward Compatibility

### Principio: V2 coexiste con V1, no la reemplaza

V1 sigue siendo funcional para N≤10 y para los 6 experiment runners que ya la usan.
V2 es una clase SEPARADA en el mismo módulo, con su propio config dataclass.

### Estructura de código en `gnn_qem.py`

```python
# Clases existentes (NO se tocan):
class GNNQEMConfig: ...        # V1 config (node_feat=4, hidden=64, etc.)
class GNNQEMCorrector: ...     # V1 model (GINConv, global_mean_pool)

# Clases NUEVAS (se agregan al final del archivo):
@dataclass
class GNNQEMConfigV2:
    """V2 config — escalable a N grande."""
    node_feature_dim: int = 6      # T1, T2, readout, gate_err, n_cx_local, degree
    edge_feature_dim: int = 1      # per-edge 2Q gate error
    context_dim: int = 7           # h, n_2q, CES_2q, CES_readout, E_noisy/N, gap, sign_bias
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.15
    use_virtual_node: bool = True
    augment_calibration: bool = True  # ±20% perturbation during training
    augment_scale: float = 0.2
    lr: float = 5e-4
    epochs: int = 3000
    patience: int = 300

class GNNQEMCorrectorV2(nn.Module):
    """V2 — GATv2Conv + edge features + virtual node."""
    ...
```

### Checkpoint versionado

Los checkpoints V2 se guardan con un header de versión:

```python
def save_qem_v2_checkpoint(model, path, train_result=None, metadata=None):
    torch.save({
        "version": "2.0",               # ← versión del modelo
        "config": asdict(model.config),  # Para reconstruir la arquitectura
        "state_dict": model.state_dict(),
        "train_result": train_result,
        "metadata": metadata,
    }, path)
```

### Auto-detección en `HardwareBackend.load_gnn_qem()`

```python
def load_gnn_qem(self, checkpoint_path: str | Path) -> None:
    """Load GNN-QEM model — auto-detects V1 or V2 from checkpoint."""
    raw = torch.load(Path(checkpoint_path), map_location="cpu")

    if isinstance(raw, dict) and raw.get("version", "1.0") == "2.0":
        # V2 checkpoint
        config = GNNQEMConfigV2(**raw["config"])
        model = GNNQEMCorrectorV2(config)
        model.load_state_dict(raw["state_dict"])
        self._gnn_qem_version = 2
    else:
        # V1 checkpoint (legacy format)
        model, _, _ = load_qem_checkpoint(Path(checkpoint_path))
        self._gnn_qem_version = 1

    self._gnn_qem_model = model
```

### build_qem_graph() vs build_qem_graph_v2()

La función `build_qem_graph()` existente NO se modifica. Se crea una nueva:

```python
def build_qem_graph_v2(sample: QEMSampleV2) -> Data:
    """V2 graph builder — per-edge attrs + expanded node features."""
    ...
```

El `correct_energy()` existente sigue funcionando para V1. Para V2:

```python
def correct_energy_v2(model: GNNQEMCorrectorV2, sample: QEMSampleV2, ...) -> QEMCorrectionResult:
    """V2 correction — same return type as V1 for API compatibility."""
    data = build_qem_graph_v2(sample)
    ...
    return QEMCorrectionResult(...)  # Mismo tipo de retorno que V1
```

El `QEMCorrectionResult` es compartido entre V1 y V2 (misma interfaz de salida).

---

## Data Generation: Extensión, no reemplazo

### Función existente: `generate_qem_training_data()` — NO SE MODIFICA

Sigue generando data con θ random para V1 y como fallback.

### Función nueva: `generate_qem_training_data_v2()`

Extensión que usa θ_opt del proyecto:

```python
def generate_qem_training_data_v2(
    topologies: list[str] | None = None,
    n_qubits_list: list[int] | None = None,
    h_values: list[float] | None = None,
    p_layers: int = 1,
    shots: int = 8192,
    model_name: str = "tfim",
    theta_source: Literal["zoo", "npz", "random"] = "zoo",
    npz_paths: list[Path] | None = None,
    include_gap: bool = True,
) -> list[QEMSampleV2]:
    """Generate training data with realistic θ_opt and expanded features.

    Parameters
    ----------
    theta_source : {"zoo", "npz", "random"}
        - "zoo": Predict θ with MPNN from model_zoo (default)
        - "npz": Load θ_opt from existing NPZ datasets
        - "random": Fallback to V1 behavior
    npz_paths : list[Path] | None
        Paths to .npz files (required if theta_source="npz")
    include_gap : bool
        If True, fetch gap from GroundTruthCache for context features.
    """
    ...
```

**Estrategia de carga de θ_opt (cascada)**:

```python
# Intento 1: NPZ directo (más preciso — VQE real optimizado)
if theta_source == "npz" and npz_paths:
    for path in npz_paths:
        data = load_phase12_dataset(path)
        # Extraer theta_opt para cada h_value que coincida
        ...

# Intento 2: MPNN del Zoo (inferencia rápida para cualquier h)
elif theta_source == "zoo":
    mpnn = load_pretrained(model=model_name, topology=topo, n_qubits=n)
    for h in h_values:
        graph = build_graph_for_h(h, topo, n)
        theta = mpnn(graph).detach().numpy()
        ...

# Intento 3: Fallback a random (backward-compat)
else:
    theta = rng.uniform(-1, 1, n_params)
```

### QEMSampleV2 — Extensión del dataclass existente

```python
@dataclass
class QEMSampleV2(QEMSample):
    """Extended sample with V2 features. Inherits all V1 fields."""
    # Nuevos campos (V2):
    gap: float = 0.0                          # Spectral gap (from GroundTruthCache)
    n_cx_per_qubit: list[float] = field(default_factory=list)  # CX count per qubit
    qubit_degree: list[int] = field(default_factory=list)      # Coupling map degree
    ces_2q: float = 0.0                       # CES solo 2Q gates
    ces_readout: float = 0.0                  # CES solo readout
```

Al heredar de `QEMSample`, todos los scripts V1 que construyen `QEMSample` siguen
funcionando. `QEMSampleV2` simplemente agrega campos opcionales.

---

## Arquitectura Final: GNNQEMCorrectorV2

```python
class GNNQEMCorrectorV2(nn.Module):
    """
    Mejoras sobre V1:
    - GATv2Conv con per-edge 2Q gate errors (Paso 2)
    - Node features expandidas [6 dims] (Paso 3)
    - Virtual global node (Paso 4)
    - hidden_dim=128, heads=4, n_layers=4 (escalado)
    - Residual connections entre capas
    - global_add_pool (no mean — preserva magnitud)
    """
    config: GNNQEMConfigV2  # node_feat=6, edge_feat=1, hidden=128, heads=4, layers=4
```

### Comparación V1 vs V2

| Aspecto | V1 (actual) | V2 (target) |
|---------|:-----------:|:-----------:|
| Conv layer | GINConv | GATv2Conv |
| Hidden dim | 64 | 128 |
| Heads | — | 4 |
| Layers | 3 | 4 |
| Node features | 4 | 6 |
| Edge features | ❌ (promediado) | ✅ per-edge |
| Virtual node | ❌ | ✅ |
| Readout | global_mean_pool + concat | virtual_node + global_add_pool |
| Scale tested | N≤10 | N≤127 (target) |
| Training data | θ random | θ_opt (Zoo) + multi-N |
| Parameters (est.) | ~25K | ~200K |

---

## Timeline

| Semana | Paso | Días | Entregable |
|:------:|------|:----:|------------|
| 1 | 0: Baseline | 1 | Métricas de referencia |
| 1 | 1: θ_opt training data | 2 | Dataset representativo + test A/B |
| 1-2 | 2: GATv2Conv + edge_attr | 3 | GNNQEMCorrectorV2 backbone |
| 2 | 3: Node features extra | 1 | Features finales |
| 2-3 | 4: Virtual node | 2 | Readout escalable |
| 3 | 5: Pseudo-labels ZNE | 3 | Modelo multi-N |
| — | 6: Laplacian PE | 1 | Solo si necesario |
| **Total** | | **~13 días** | Modelo escalable validado |

---

## Criterios de Éxito Global

| Métrica | V1 (baseline) | V2 (target) |
|---------|:-------------:|:-----------:|
| val_MAE (N=6-10) | X | < 0.7X |
| val_MAE (N=20) | No probado | < 2X (extrapolación) |
| Improvement_pct | Y% | > Y+20% |
| Confidence accuracy | ~70% | > 85% |
| Inference time (N=127) | — | < 10ms |

---

## Dependencias de pip (ya satisfechas)

```
torch>=2.0
torch-geometric>=2.4  # GATv2Conv, VirtualNode, LaplacianEigenvectorPE
qiskit-ibm-runtime    # FakeBackend para data generation
```

No se requieren dependencias nuevas.

---

## Notas sobre reutilización de código externo

### IBM ml-qem (Apache-2.0)

Archivos relevantes para estudiar/adaptar:
- `docs/tutorials/gnn.py` — GNN architecture para QEM (PyG-based)
- `docs/tutorials/mlp.py` — Circuit-level features engineering
- `docs/demos/demo1_rf_mimic_zne_100q_twirl.ipynb` — 100Q TFIM workflow

**Cómo reutilizar**: No importar directamente (API legacy Qiskit, incompatible).
Adaptar patrones: circuit feature extraction, GNN architecture choices, scaling patterns.
La librería `blackwater` que envuelve Estimator es interesante conceptualmente pero
nuestra integración via `HardwareBackend.load_gnn_qem()` es más limpia.

### PyG built-ins (MIT, ya instalado)

```python
from torch_geometric.nn import GATv2Conv          # Paso 2
from torch_geometric.transforms import VirtualNode  # Paso 4
from torch_geometric.transforms import AddLaplacianEigenvectorPE  # Paso 6
from torch_geometric.nn import global_add_pool      # Readout
```

### Trans-GNN-QuantumCircuitPrediction (MIT)

Útil como referencia para:
- Cómo codifican noise features como node attributes
- Transformer + GNN hybrid architecture
- Dataset generation con Qiskit + FakeBackend

URL: https://github.com/Prof-it/trans-gnn-QuantumCircuitPrediction

---

---

## Datos Disponibles en el Proyecto (Inventario para Training)

### Fuente 1: GroundTruthCache (`data/ground_truth_cache.json`)

Contiene por cada (topology, N, model, h):
- `energy` — E₀ exacto (ExactDiag o DMRG)
- `gap` — Gap espectral
- `method` — "exact_diag" o "dmrg"
- `mag_x` — Magnetización ⟨X⟩ bulk
- `corr_zz` — Correlación ⟨ZZ⟩ bulk

**Uso para GNN-QEM**: E_exact es el target del modelo. El gap es útil como feature
de contexto (gap pequeño → circuito más sensible al ruido, corrección mayor esperada).

### Fuente 2: NPZ Datasets (Phase 1+2, en `data/` y `results/`)

Cada `.npz` contiene arrays para un sweep completo:
- `h_values` — grid de h
- `theta_opt` — **θ óptimos VQE por h-point** (shape: [n_h, n_params])
- `ground_energies` — E₀ por h
- `gaps` — gap espectral por h
- `vqe_energies` — energía VQE lograda
- `fidelities` — fidelidad del estado
- `mag_x`, `corr_zz` — observables

**Uso para GNN-QEM Paso 1**: `theta_opt` es exactamente lo que necesitamos para generar
training data representativa. Bind `theta_opt[i]` al circuito HVA → transpila →
ejecuta en FakeBackend → obtener E_noisy. El target es `ground_energies[i]`.

### Fuente 3: Model Zoo (checkpoints + metadata)

`ZooEntry` contiene: model, topology, n_qubits, p_layers, h_range, pass_rate, seeds.
El checkpoint `.pt` contiene el MPNN entrenado que puede predecir θ para *cualquier h*.

**Uso para GNN-QEM**: Para h-points donde no hay NPZ disponible, usar el MPNN del zoo
para predecir θ_opt(h) → genera training data on-the-fly sin correr VQE.

### Fuente 4: EvalCache (`data/eval_cache.json`)

Cache de energías evaluadas, indexada por (model, topology, N, p, J, h, sha256(θ)).
Contiene miles de evaluaciones exactas.

**Uso para GNN-QEM**: Si generamos E_noisy para un (circuito, θ), podemos buscar la
energía noiseless exacta en el EvalCache sin re-computar.

### Fuente 5: CalibrationSnapshot (de FakeBackend)

`take_calibration_snapshot(backend)` devuelve:
- `qubit_t1: dict[int, float]` — T1 en µs (FakeTorino: ~100-200 µs)
- `qubit_t2: dict[int, float]` — T2 en µs (FakeTorino: ~50-150 µs)
- `gate_errors_2q: dict[str, float]` — Error 2Q por enlace "q0-q1" (FakeTorino: 0.003-0.03)
- `readout_errors: dict[int, float]` — Error readout por qubit (FakeTorino: 0.005-0.05)

**Uso**: Node features + Edge features directamente. El `gate_errors_2q` KEYED por
enlace ("q0-q1") mapea 1:1 a `edge_attr` en el grafo.

### Fuente 6: Datos de Hardware Rehearsal (results/hardware/)

Runs existentes en FakeTorino/FakeSherbrooke con energías medidas, CES, layouts usados,
y métricas de ZNE. Estos son training data GRATIS para GNN-QEM:
- `e_zne` (energía post-ZNE) como target intermedio
- `e_exact` cuando disponible (lookup en GT cache)
- Layout, CES, n_2q_gates como features

---

## Información de la Investigación Inicial — Hallazgos Aplicables al Plan

De la investigación realizada al inicio de esta sesión:

### H1: El gap espectral es informativo como feature de contexto

Legnini & Berberich (2026) muestran que Δθ ∝ (gap)⁻¹. Circuitos con gap pequeño
(cerca de h_c) son más sensibles al ruido. Agregar `gap/max_gap` como context feature
da al modelo una señal directa de "cuánta corrección esperar".

**Acción**: Agregar `gap` al context del virtual node (Paso 4). Disponible gratis del
GroundTruthCache para todo N≤22.

### H2: CES (Circuit Error Score) ya captura exposición total

`CES = 1 - Π(1-ε_gate)` ya está en el context. Pero podemos descomponer:
- `CES_2q` — solo errores 2Q (dominante)
- `CES_readout` — solo readout
- `CES_1q` — solo 1Q (despreciable, omitir)

**Acción**: En vez de un solo CES, separar en CES_2q + CES_readout como 2 context dims.

### H3: El paper de Cantori (2024) muestra que E_noisy como input es CRÍTICO

Sin el valor ruidoso medido, la NN no puede predecir la corrección (no sabe
"cuánto se desvió"). E_noisy/N ya está en nuestro context — validado.

### H4: La dirección (over/under estimation) importa

En arXiv:2604.16815, el GNN predice ΔE = E_exact - E_noisy (puede ser + o -).
Nuestro modelo ya hace esto. Pero podemos agregar un feature que señale la
*dirección esperada* del error: para TFIM, ruido incoherente siempre SUBE la
energía (E_noisy > E_exact). Si el modelo sabe esto a priori → convergencia más rápida.

**Acción**: Agregar `sign_bias = -1.0` como feature constante al context (para TFIM,
el ruido siempre produce E_noisy ≥ E_exact). Para otros modelos, puede variar.

### H5: Data augmentation de calibración es barata y efectiva

arXiv:2509.12933 muestra que modelos entrenados en UNA sola calibración no generalizan.
Perturbar T1/T2/gate_err ±20% durante training (sin re-ejecutar circuitos) es
equivalente a simular drift temporal. Solo implica modificar `build_qem_graph()` con
un flag `augment=True` que aplique ruido gaussiano a los features.

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|:------------:|------------|
| GATv2Conv más lento que GINConv a N=127 | MEDIA | Reducir heads (4→2), usar sparse attention |
| Virtual node domina y colapsa información local | BAJA | Monitorear attention weights; residual sin virtual |
| Pseudo-labels ZNE son ruidosas → entrenan mal | MEDIA | Weight 0.5 + dropout 0.2 + más data Tier 1 |
| θ_opt data no mejora sobre random | BAJA | Si falla, el random sigue siendo backup |
| Modelo sobreajusta a FakeTorino | MEDIA | Augment calibration ±20%, test en FakeSherbrooke |


---

## Apéndice: Features Disponibles del FakeBackend (Referencia Completa)

### Acceso via `take_calibration_snapshot(FakeTorino())`

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qmbp_simulation.execution.noisy_utils import take_calibration_snapshot

backend = FakeTorino()
snap = take_calibration_snapshot(backend)

# Per-qubit (dict[int, float]):
snap.qubit_t1       # T1 en µs. FakeTorino range: ~80-300 µs
snap.qubit_t2       # T2 en µs. FakeTorino range: ~30-200 µs
snap.readout_errors # Prob readout error. FakeTorino range: 0.003-0.05

# Per-edge (dict[str, float] keyed "q0-q1"):
snap.gate_errors_2q # Error 2Q gate. FakeTorino range: 0.003-0.035

# Derived:
snap.mean_t1_us     # Promedio global T1
snap.mean_t2_us     # Promedio global T2
snap.mean_2q_error  # Promedio global error 2Q
```

### Acceso via `backend.target` (BackendV2 Target API)

```python
target = backend.target

# Qubit properties (T1, T2):
for q in range(backend.num_qubits):
    props = target.qubit_properties[q]
    t1_seconds = props.t1   # en segundos (multiplicar ×1e6 para µs)
    t2_seconds = props.t2

# Gate errors (per-gate, per-qubit(s)):
for op_name in target.operation_names:
    for qargs in target.qargs_for_operation_name(op_name):
        gate_props = target[op_name][qargs]
        error = gate_props.error      # probabilidad de error
        duration = gate_props.duration # duración del gate en seconds
```

### Features derivadas que podemos computar

| Feature | Fórmula | Dónde va | Justificación |
|---------|---------|----------|---------------|
| `t1_norm` | T1_qubit / 100 µs | Node | Normalizado a escala típica |
| `t2_norm` | T2_qubit / 100 µs | Node | Normalizado a escala típica |
| `t2_over_t1` | T2/T1 ∈ [0, 2] | Node | Ratio indica tipo de decoherencia dominante |
| `readout_err` | raw (0-0.05) | Node | Directo |
| `gate_err_2q` | raw (0.003-0.035) | Edge | PER-EDGE, no promediado |
| `n_cx_local` | count(CX on qubit)/max | Node | Exposición acumulada al error |
| `degree` | n_neighbors/max_degree | Node | Conectividad del qubit |
| `CES_2q` | 1 - Π(1-ε_2q) | Context/Virtual | Score de error total 2Q |
| `CES_readout` | 1 - Π(1-ε_ro) | Context/Virtual | Score de error readout |
| `gap_norm` | gap/10.0 | Context/Virtual | Sensibilidad al ruido (H1) |
| `n_2q_total` | total 2Q gates / 50 | Context/Virtual | Profundidad normalizada |
| `e_noisy_norm` | E_noisy / N | Context/Virtual | Energía medida extensiva |
| `sign_bias` | -1 for TFIM | Context/Virtual | Dirección esperada del error |
| `gate_duration_2q` | duration × 1e9 (ns) | Edge (futuro) | Gates más largos → más decoherencia |

### Nota sobre FakeBackend gate durations

FakeTorino también expone `gate_props.duration` para cada gate. Gates 2Q más largos
sufren más decoherencia durante ejecución. Esto es un feature de SEGUNDO ORDEN que
podría ser edge_attr[1] en futuro (no incluido en V2 inicial para mantener simple).
Considerar para V3 si V2 no alcanza accuracy target.

---

## Checklist de Implementación

- [ ] Paso 0: Medir baseline actual
- [ ] Paso 1: Generar training data con θ_opt (NPZ + Zoo)
- [ ] Paso 1: Test A/B vs random θ
- [ ] Paso 2: Implementar GATv2Conv + edge_attr
- [ ] Paso 2: Implementar data augmentation de calibración
- [ ] Paso 2: Test A/B vs GINConv
- [ ] Paso 3: Expandir node features a 6 dims
- [ ] Paso 3: Test A/B vs 4 dims
- [ ] Paso 4: Implementar virtual global node con context expandido
- [ ] Paso 4: Test a N=20 (DMRG ground truth disponible)
- [ ] Paso 5: Generar pseudo-labels ZNE para N>20
- [ ] Paso 5: Fine-tune y test de generalización
- [ ] Final: Comparar V2 completa vs V1 baseline en todas las métricas
