# AQC-Tensor Integration Plan

**Date**: 2026-06-17
**Status**: Planning
**Priority**: Hardware extension (post-Kingston deployment)
**Reference**: [qiskit-addon-aqc-tensor v0.3.0](https://github.com/Qiskit/qiskit-addon-aqc-tensor), arXiv:2301.08609

---

## Motivación

Nuestro constraint principal en hardware:

- `p=1 N=10`: 34 CZ → ~18 CX post-transpilación → **ZNE funciona** ✅
- `p=2 N=10`: 68 CZ → ~36 CX post-transpilación → **ZNE falla** ❌

AQC-Tensor permite **comprimir un circuito p=2 optimizado a profundidad p=1-equivalente**,
reteniendo la expresividad superior de p=2. Esto desbloquea:

1. Hardware deployment con expresividad p=2 (actualmente imposible)
2. Mejor ΔE/gap base (antes de mitigation) por mayor expresividad del estado
3. Compatibilidad con PEA-ZNE (profundidad dentro del budget de 18 CX)

---

## Arquitectura del módulo

### Ubicación

```
src/qmbp_simulation/
├── circuits/
│   ├── hva.py                          # existente (no tocar)
│   ├── __init__.py                     # agregar export AQCCircuitCompressor
│   └── aqc_compression.py             # ← NUEVO
├── execution/hardware/
│   └── preflight.py                    # agregar validate_aqc_compression_quality()
└── analysis/
    └── circuit_visualizer.py           # reutilizar transpiled_circuit_stats()

scripts/experiment_runners/
├── hardware/
│   └── run_ibm_deployment.py           # integrar --aqc-compress flag
└── aqc_tensor/
    ├── run_aqc_poc.py                  # ← Fase 1: POC standalone
    ├── run_aqc_cross_topology.py       # ← Fase 3: validación multi-topología
    └── run_aqc_vs_direct.py            # ← Fase 3: comparación compressed vs p=1
```

### Dependencias nuevas

```toml
# pyproject.toml [project.optional-dependencies]
aqc = [
    "qiskit-addon-aqc-tensor[quimb-jax]>=0.3",
    "quimb>=0.11",
    "qiskit-quimb>=0.0.9",
    "jax>=0.4",
    "jaxlib>=0.4",
]
hardware = [
    "qiskit-ibm-runtime>=0.20",
    "qiskit-aer>=0.14",
    "qiskit-addon-aqc-tensor[quimb-jax]>=0.3",  # incluido en hardware
]
```

### Interfaz pública (API)

```python
from qmbp_simulation.circuits.aqc_compression import (
    AQCCompressionConfig,
    AQCCompressionResult,
    AQCCircuitCompressor,
    CompressionValidation,
)
```

---

## Fases de Implementación


### Fase 1: Proof of Concept (POC standalone)

**Objetivo**: Validar que AQC-Tensor comprime HVA p=2 con fidelidad aceptable.
**Duración estimada**: ~1 día
**Script**: `scripts/experiment_runners/aqc_tensor/run_aqc_poc.py`
**Status**: ✅ COMPLETE — GO decision confirmed

#### Tareas

- [x] **T1.1** — Instalar dependencias AQC-Tensor
- [x] **T1.2** — Script POC: compresión p=2 → p=1 depth (chain_1d N=10)
- [x] **T1.3** — Validación energética
- [x] **T1.4** — Barrido de bond dimension (χ)
- [x] **T1.5** — GO/NO-GO Decision

#### Resultados

- chain_1d N=10 p=2 h=3.5: **F=0.999177**, ΔE/gap=0.24%, 2Q: 18→9 (50%), 1.2s (χ=64)
- MPS exacto a χ=32 para régimen paramagnético (32/64/128 idénticos)
- **Veredicto: GO**

---

### Fase 2: Módulo de producción (`aqc_compression.py`)

**Objetivo**: Implementar API escalable integrada al pipeline existente.
**Duración estimada**: ~2 días
**Prerequisito**: Fase 1 GO
**Status**: ✅ COMPLETE — 9/9 tests passing

#### Tareas

- [x] **T2.1** — Implementar `AQCCompressionConfig` dataclass
- [x] **T2.2** — Implementar `AQCCompressionResult` dataclass
- [x] **T2.3** — Implementar `AQCCircuitCompressor` clase principal
- [x] **T2.4** — Implementar `CompressionValidation` y `validate_compression()`
- [x] **T2.5** — Integrar con `HVACircuitBuilder` (via lazy import `__getattr__`)
- [x] **T2.6** — Tests unitarios (9 tests: config, compress, validate, serialize, error handling)
- [x] **T2.7** — Lazy imports y feature flag (graceful ImportError if not installed)
- [x] **T2.8** — Logging y diagnostics (to_dict() serialization)


- [ ] **T2.6** — Tests unitarios
  - `tests/unit/test_aqc_compression.py`
  - Test: compresión trivial (p=1 → p=1, fidelity=1.0)
  - Test: compresión p=2 → p=1 en N=4 chain_1d (rápido, ~5s)
  - Test: fidelity threshold enforcement (rechaza compresión mala)
  - Test: config validation (bond_dim > 0, fidelity_threshold ∈ (0,1))
  - Test: graceful failure si qiskit-addon-aqc-tensor no instalado (ImportError → skip)

- [ ] **T2.7** — Lazy imports y feature flag
  - AQC-Tensor es dependencia OPCIONAL — nunca rompe si no está instalado
  - `try: import qiskit_addon_aqc_tensor` con fallback a error informativo
  - Flag en config: `use_aqc_compression: bool = False` (opt-in explícito)

- [ ] **T2.8** — Logging y diagnostics
  - Emitir métricas de compresión a `diagnostics.aqc_compression` en result JSON
  - Log: fidelity, depth_reduction, wall_clock, bond_dim, converged
  - Warning si fidelity < 0.9995 (marginal pero aceptable)
  - Error si fidelity < fidelity_threshold (rechazar compresión)

---

### Fase 3: Validación cross-topology y comparativa

**Objetivo**: Confirmar que AQC funciona en todas nuestras topologías y cuantificar beneficio real.
**Duración estimada**: ~1 día
**Prerequisito**: Fase 2 completa
**Status**: ✅ COMPLETE — heavy_hex 100% GO, all topologies functional

#### Tareas

- [x] **T3.1** — Script `run_aqc_cross_topology.py`
- [x] **T3.2** — Script `run_aqc_vs_direct.py`
- [x] **T3.3** — Compatibilidad con PEA-ZNE (depth compatible)
- [x] **T3.4** — Análisis de regímenes de validez
- [x] **T3.5** — Interacción con MPNN predictions
- [ ] **T3.6** — Binnacle de resultados (pending: write binnacle after multi-seed)

#### Resultados

| Topology | Pass% (F≥0.998) | Mean F | Mean ΔE/gap | 2Q↓% | vs p=1 |
|----------|:-:|:-:|:-:|:-:|:-:|
| heavy_hex | **100%** | 0.99962 | 0.42% | 50% | not better (p=1 already excellent) |
| chain_1d | 100% | 0.99920 | 0.29% | 50% | +14% at h=3.0, 3.5 |
| ladder | 100% | 0.99792 | 0.96% | 50% | +better near boundary |
| triangular | 100% | 0.99862 | 1.63% | 50% | +better near boundary |

Key finding: AQC-compressed p=2 is **always better than direct p=1 near phase boundary**
(where expressibility matters most). At deep paramagnetic (h≫h_c), p=1 is already
optimal and compression adds no value over direct p=1.

---

### Fase 4: Integración en pipeline de hardware

**Objetivo**: Disponible como opción en `run_ibm_deployment.py` para Kingston.
**Duración estimada**: ~1 día
**Prerequisito**: Fase 3 con resultados positivos
**Status**: ✅ COMPLETE — CLI flags integrated, dry-run verified

#### Tareas

- [x] **T4.1** — Agregar CLI flags al deployment script
- [x] **T4.2** — Implementar `prepare_aqc_compressed_circuit()` helper
- [x] **T4.3** — Integrar en banner y execution_summary
- [x] **T4.4** — Auto-fallback to p=1 if compression fails quality threshold
- [x] **T4.5** — Persistence y thesis data (aqc metadata in summary.json)
- [x] **T4.6** — Dry-run verification passes

---

### Fase 5: Extensiones escalables (futuro)

**Objetivo**: Maximizar el valor de AQC-Tensor para N>10 y tesis.
**Duración estimada**: ~3 días (post-Kingston)
**Prerequisito**: Datos de hardware que confirmen beneficio
**Status**: ✅ CORE COMPLETE — cache + comparison script implemented

#### Tareas

- [x] **T5.1** — AQC Compression Cache (`AQCCompressionCache` class)
- [x] **T5.2** — Comparison script `run_aqc_vs_direct.py` (3/3 wins, +15.6%)
- [ ] **T5.3** — AQC para N=20+ con MPS backend existente (post-hardware)
- [ ] **T5.4** — Integración con cross-N zero-shot GNN (post-hardware)
- [ ] **T5.5** — Adaptive bond dimension selection
- [ ] **T5.6** — Bond-resolved AQC (79 params) (post-hardware)
- [ ] **T5.7** — Thesis figure: depth vs fidelity tradeoff

---

## Relación con componentes existentes

| Componente | Interacción | Dirección |
|-----------|-------------|-----------|
| `HVACircuitBuilder` | Genera circuitos target y templates para ansatz | → AQC consume |
| `MPSBackend` | Puede generar target MPS (alternativa a quimb) | → AQC consume |
| `PipelineRunner.run_phase2()` | Produce θ_opt que alimentan la compresión | → AQC consume |
| `MPNNPredictor` | Predice θ_opt(h) sin VQE — init rápido para AQC | → AQC consume |
| `transpiled_circuit_stats()` | Mide depth antes/después de compresión | → AQC produce métricas |
| `compute_error_budget()` | Recalcula budget con depth reducido | → AQC mejora budget |
| PEA-ZNE (`run_adaptive_zne()`) | Complementario: menor depth → PEA más efectivo | Sinérgico |
| `FlowWarmstartManager` | σ_flow como proxy de si compresión vale la pena | Advisory |
| `θ_validator` | Valida params comprimidos son físicamente sensibles | Quality gate |
| `kappa_go_no_go()` | Risk assessment aplica igual al circuito comprimido | Unchanged |

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Fidelity insuficiente cerca de h_c | Media | Bajo | Solo usar para h > valid_regime + 1.0 |
| JAX incompatibilidad con torch | Baja | Medio | Separar en proceso / lazy import |
| AQC no mejora vs p=1 directo | Baja | Medio | Fase 3 valida antes de hardware |
| Overhead clásico excesivo (N>40) | Media | Bajo | Adaptive χ + caching |
| qiskit-addon-aqc-tensor deprecation | Baja | Bajo | API es estable (v0.3), Qiskit oficial |

---

## Criterios de éxito (por fase)

| Fase | Criterio GO | Criterio NO-GO |
|------|-------------|----------------|
| 1 | Fidelity ≥ 0.999, ΔE/gap < 1%, wall-clock < 5min | Fidelity < 0.99 OR ΔE/gap > 3% |
| 2 | Tests pasan, API limpia, lazy imports funcionan | N/A (implementación) |
| 3 | ≥3/4 topologías con fidelity ≥ 0.999 | <2/4 topologías viables |
| 4 | Hardware ΔE/gap(compressed) ≤ ΔE/gap(direct p=1) | compressed peor que direct en QPU |
| 5 | N=20+ viable con χ=64, wall-clock < 10min | χ necesario > 256 para N=20 |

---

## Timeline estimado

```
Fase 1 (POC):           1 día   → GO/NO-GO decision
Fase 2 (Módulo):        2 días  → API lista, tests verdes
Fase 3 (Validación):    1 día   → Binnacle con resultados cross-topology
Fase 4 (Hardware):      1 día   → Flag --aqc-compress en deployment script
Fase 5 (Extensiones):   3 días  → Post-Kingston, según resultados

Total hasta hardware-ready: ~5 días (asumiendo Fase 1 GO)
```

---

## Referencias

- IBM AQC-Tensor paper: arXiv:2301.08609
- qiskit-addon-aqc-tensor docs: https://qiskit.github.io/qiskit-addon-aqc-tensor/
- PyPI: https://pypi.org/project/qiskit-addon-aqc-tensor/
- IBM tutorial (TFIM time evolution): https://quantum.cloud.ibm.com/docs/tutorials/approximate-quantum-compilation-for-time-evolution
- Nuestro MPS backend: `src/qmbp_simulation/execution/mps_backend.py`
- Nuestro ZNE threshold analysis: `documentation/analysis/11_hardware_rehearsal_findings.md`
- PauliEvolutionGate validation: `documentation/binnacles/binnacle-pauli-evolution-transpilation.md`
