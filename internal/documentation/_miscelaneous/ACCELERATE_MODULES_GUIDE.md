# Accelerate Modules — Guía de Uso

## Qué es

El sistema acelerado predice parámetros θ para circuitos HVA bond-resolved
sin ejecutar VQE completo en cada h-point. Usa una UnifiedMPNN entrenada en
pocos puntos anchor para predecir el resto.

**Solo aplica a bond-resolved HVA** (19+ params por layer). Para TFIM global
(2 params), scipy interpolation funciona igual de bien — no necesitás GNN.

**Componentes:**

| Módulo | Función |
|--------|---------|
| `AcceleratedVQE` | Pipeline completo: anchors + train + predict + refine |
| `model_zoo` | Registro de checkpoints con SHA256 verification |
| `QualityPredictor` | Estima si VQE va a converger antes de ejecutar |

---

## Cómo Funciona

```
┌─────────────────────────────────────────────────────────┐
│  AcceleratedVQE.run(h_values)                           │
├─────────────────────────────────────────────────────────┤
│  1. Buscar modelo en zoo → si existe, skip a paso 4    │
│  2. VQE completo en K=5 anchor points (warm-start)     │
│  3. Entrenar UnifiedMPNN con esos K puntos             │
│  4. Predecir θ para los N-K puntos restantes           │
│  5. Evaluar + refinar puntos con ΔE/gap > threshold    │
│  6. Auto-export al zoo si pass_rate > 80%              │
└─────────────────────────────────────────────────────────┘
```

Si ya hay un modelo en el zoo (de un run previo), el pipeline salta
directamente al paso 4: predicción instantánea para todos los h-points.

---

## Uso Básico

### Desde Python (cualquier runner o notebook):

```python
from qmbp_simulation import make_lattice
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.pipeline.accelerated import AcceleratedVQE, AcceleratedConfig
import numpy as np

# Setup
spec = get_model_spec("tfim_bond_resolved")
lattice = make_lattice("chain_1d", 10, J=1.0, h=2.0)
hva = HVACircuitBuilder()
circuit, _ = hva.create_bond_resolved(10, 1, lattice)
backend = NoiselessBackend()

# Run (zero-config)
accel = AcceleratedVQE(lattice, circuit, spec, backend)
result = accel.run(np.linspace(1.3, 3.0, 20), seed=42, p_layers=1)

print(f"Pass rate: {result.pass_rate:.0%}")
print(f"Speedup: {result.speedup_estimate:.1f}×")
print(f"Methods: {set(result.method)}")
```

### Desde CLI:

```bash
# Pipeline completo con auto-export al zoo
.venv/bin/python scripts/.../run_noiseless_pipeline.py \
  --topology chain_1d --n-qubits 10 --export-zoo

# Re-usar modelo existente (instantáneo)
.venv/bin/python scripts/.../run_noiseless_pipeline.py \
  --accelerate --topology chain_1d --n-qubits 10

# Con refinamiento de puntos inciertos (P4 active learning)
.venv/bin/python scripts/.../run_noiseless_pipeline.py \
  --accelerate --active-rounds 2 --topology chain_1d
```

---

## Caso de Uso: Cross-N Transfer (Bond-Resolved)

### Objetivo
Entrenar UnifiedMPNN con VQE de N=10 y predecir θ para N=20, N=40 sin
re-entrenar. Solo tiene sentido para **bond-resolved** donde hay 19+ params
y scipy interpolation no escala.

### Por qué funciona
UnifiedMPNN predice per-node (θ_x) y per-gate (θ_zz), produciendo un output
de tamaño variable que se adapta al número de qubits del grafo. Un modelo
entrenado en N=10 (29 nodos en el unified graph) puede recibir un grafo de
N=20 (49 nodos) y producir las 39 predicciones correspondientes.

### Requisitos
- `norm_type="none"` en el modelo (BatchNorm destruye cross-N en chain_1d)
- Misma topología obligatoria (ladder→ladder, no chain→ladder)
- Solo bond-resolved (el global HVA con 2 params no necesita GNN)

### Status: VALIDADO EXPERIMENTALMENTE

Cross-N con bond-resolved UnifiedMPNN ha sido ejecutado y validado:

#### Resultado: N=10 → N=20 (14 h-points, h ∈ [2.0, 3.5])

| Métrica | Valor |
|---------|-------|
| Pass rate @5% (ΔE/gap < 0.05) | **64% (9/14)** |
| Pass rate @10% (ΔE/gap < 0.10) | **93% (13/14)** |
| Mean ΔE/gap | 0.046 |
| Tiempo training N=10 | 197s |
| Tiempo cross-N predict N=20 | 577s (dominado por exact_diag validación) |
| Modelo exportado al zoo | ✅ unified_tfim_br_chain_n10_p1.pt |

Breakdown por región h:
- **h > 2.8: 100% pass @5%** — generalización perfecta
- **2.0 ≤ h ≤ 2.8: degradación gradual** — ΔE/gap crece monótonamente
- **h = 2.0: ΔE/gap = 0.12** — único punto que falla @10%

#### Resultado: N=10 → N=25 (parcial, DMRG ground truth)

| Métrica | Valor |
|---------|-------|
| h=2.83: ΔE/gap | 0.046 ✓ |
| Fidelity | N/A (N>22, DMRG no provee statevector) |
| Active learning | Impracticable (VQE O(2^25) por eval) |

**Hallazgo clave:** El GNN aprende la estructura física de θ(h) independiente de N.
Para h > 2.5, la predicción cross-N es confiable sin VQE adicional.

### Limitaciones automáticas (implementadas):
- **Fidelity:** Auto-skip para N > STATEVECTOR_MAX_N (22). Reporta `F=N/A(N>22)`.
- **Active learning:** Auto-skip para N > 22. Log de warning explica el motivo.
- **MPSBackend:** Para N>22 con refinamiento, usar runners MPS-based en vez de statevector.

### Plan:
```python
# 1. Entrenar en N=10 (ya hecho en experiments anteriores)
accel = AcceleratedVQE(lattice_10, circuit_10, spec, backend)
result_10 = accel.run(h_values, p_layers=1)
accel.save_model("models/unified_chain_n10_p1.pt")

# 2. Cargar y predecir en N=20 (pendiente de ejecutar)
from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint
model = load_unified_checkpoint("models/unified_chain_n10_p1.pt", eval_mode=True)

# Construir grafo para N=20
lattice_20 = make_lattice("chain_1d", 20, J=1.0, h=2.0)
from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
import torch

for h in [3.0, 2.5, 2.0, 1.5]:
    g = build_unified_bond_resolved_graph(lattice_20, h_value=h, p_layers=1,
                                          include_circuit_nodes=True)
    with torch.no_grad():
        theta_pred = model(g).numpy().flatten()
    # theta_pred shape: (19,) para N=10, (29,) para N=20 — automático
```

---

## Reutilización de Datos

| Recurso | Dónde | Cómo reutilizar |
|---------|-------|-----------------|
| Checkpoint UnifiedMPNN | `data/model_zoo/checkpoints/` | `load_unified_checkpoint()` o `AcceleratedVQE(config=AcceleratedConfig(use_zoo=True))` |
| Datos θ_opt (VQE) | `results/experiments/` (per-point data en JSON) | Extraer con `ResultIndex`, re-entrenar MPNN |
| Training history | `ResultIndex` (521+ runs) | `QualityPredictor` lo usa automáticamente |

---

## Límites Conocidos (Findings F1-F7)

| Límite | Qué pasa | Workaround |
|--------|----------|------------|
| h < h_min (F6) | MPNN extrapola mal, VQE no converge | No intentar. Usar p más alto o aceptar el límite |
| Topologías simétricas (F3) | Unified graph no aporta vs Ham-only | Usar BondResolvedMPNN simple (más rápido) |
| VQE bond-resolved en ladder p=1 (F5) | Solo converge para h > 2.2 | Subir h_min o usar p≥2 |
| Noise-aware con shot noise (F7) | Empeora los resultados | No usar. Esperar datos de hardware real |

---

## Comandos Rápidos

```bash
make zoo-list          # Ver modelos disponibles
make zoo-validate      # Verificar SHA256 de checkpoints
make quality-check     # Predecir viabilidad antes de correr
```
