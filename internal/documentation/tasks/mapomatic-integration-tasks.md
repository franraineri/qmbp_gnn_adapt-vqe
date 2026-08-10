# Mapomatic Integration — Layout Optimizer Module

**Fecha**: 2026-06-17
**Objetivo**: Integrar mapomatic (VF2 subgraph isomorphism) como módulo de selección de layouts noise-aware para hardware deployment, complementando el BFS existente con filtrado multi-capa.
**Referencia**: [qiskit-community/mapomatic](https://github.com/qiskit-community/mapomatic) v0.14 (Apache-2.0)

---

## Contexto y Motivación

### Problema actual
- `find_layouts_bfs()` genera subgrafos conectados por BFS aleatorio → no garantiza que el subgrafo sea isomorfo al grafo de interacción del circuito → puede requerir SWAPs extra.
- `select_layouts_low_ces()` transpila TODOS los candidatos para calcular CES → O(n_candidates × transpile_time).
- No hay selección multi-backend (ibm_kingston vs ibm_boston).

### Qué aporta mapomatic
- **VF2 exhaustivo**: Encuentra TODOS los subgrafos isomorfos al circuito → layouts SWAP-free garantizados.
- **Custom cost functions**: Scoring con calibración live (gate error + readout + T1/T2).
- **Multi-backend**: Rankear entre múltiples QPUs con una sola llamada.
- **Performance**: VF2 es O(ms) para N=10 en chips de 156+ qubits.

### Por qué filtrado multi-capa
Para N=10 en ibm_kingston (156 qubits), VF2 puede devolver miles de layouts. Sin filtrado previo, evaluarlos todos es costoso e innecesario. La estrategia: podar el espacio ANTES de VF2, limitar resultados DURANTE, y filtrar por calidad DESPUÉS.

---

## Fase 0: Dependencia y Configuración

### Tarea 0.1: Agregar mapomatic como dependencia opcional
- **Archivo**: `pyproject.toml`
- **Cambio**: Agregar `"mapomatic>=0.14"` al grupo `[project.optional-dependencies].hardware`
- **Criterio**: `pip install -e ".[hardware]"` instala mapomatic sin romper deps existentes
- **Graceful degradation**: Si no está instalado, el módulo expone un flag `MAPOMATIC_AVAILABLE = False` y toda la funcionalidad cae al BFS existente

### Tarea 0.2: Verificar compatibilidad en CI
- **Archivo**: `.github/workflows/ci.yml`
- **Cambio**: El CI no instala `[hardware]` (solo `[test,dev]`), confirmar que los imports condicionales no rompen nada
- **Test**: Import guard `try: import mapomatic except ImportError` no falla en CI

---

## Fase 1: Módulo Core — `layout_optimizer.py`

### Tarea 1.1: Crear el módulo base
- **Archivo**: `src/qmbp_simulation/execution/hardware/layout_optimizer.py`
- **Responsabilidad**: Wrapper sobre mapomatic con filtrado multi-capa y scoring BackendV2

### Tarea 1.2: Implementar Capa 0 — Podado pre-VF2 del CouplingMap
```python
def build_filtered_coupling_map(
    backend,
    *,
    max_2q_error: float = 0.01,
    min_t1_us: float = 50.0,
    exclude_qubits: set[int] | None = None,
) -> CouplingMap:
```
- Extraer edges del `backend.target` filtrando por:
  - Error de gate 2Q > `max_2q_error` → excluir edge
  - T1 del qubit < `min_t1_us` → excluir qubit (y todas sus edges)
  - Qubits en `exclude_qubits` (known-bad, manual blacklist)
- **Retorna**: `CouplingMap` reducido (solo high-quality subgraph)
- **Test**: Verificar que el CouplingMap resultante es conexo o tiene componentes ≥ N qubits

### Tarea 1.3: Implementar Capa 1 — VF2 con call_limit controlado
```python
def find_vf2_layouts(
    circuit: QuantumCircuit,
    backend_or_cmap,
    *,
    call_limit: int = 100_000,
    strict_direction: bool = True,
    max_layouts: int = 200,
) -> list[list[int]]:
```
- Wrapper sobre `mm.deflate_circuit()` + `mm.matching_layouts()`
- Parámetros de control:
  - `call_limit`: Reduce de 3e7 (default mapomatic) a 1e5 — suficiente para N=10
  - `strict_direction`: Respeta dirección nativa de CZ/ECR
  - `max_layouts`: Corte post-VF2 (si VF2 devuelve >200, truncar)
- **Fallback**: Si `MAPOMATIC_AVAILABLE = False`, delegate a `find_layouts_bfs()`

### Tarea 1.4: Implementar Capa 2 — Custom cost function BackendV2
```python
def compute_layout_fidelity_cost(
    circuit: QuantumCircuit,
    layouts: list[list[int]],
    backend,
    *,
    defective_edge_threshold: float = 0.10,
    include_readout: bool = True,
) -> list[tuple[list[int], float]]:
```
- Scoring usando `backend.target` (NO `backend.properties()` V1):
  - Fidelidad producto: `Π(1 - ε_gate)` para todas las 2Q gates del circuito
  - Readout error: `Π(1 - ε_readout)` para qubits medidos
  - Penalización defective edge: error += 1.0 si alguna edge > threshold
- **Retorna**: Lista `[(layout, error)]` ordenada ascending (mejor primero)
- Compatible como `cost_function` para `mm.evaluate_layouts()` y standalone

### Tarea 1.5: Implementar Capa 3 — Selección final + transpilación
```python
def select_optimal_layouts(
    circuit: QuantumCircuit,
    backend,
    *,
    n_select: int = 3,
    max_ces: float = 0.5,
    max_2q_error: float = 0.01,
    min_t1_us: float = 50.0,
    optimization_level: int = 2,
    call_limit: int = 100_000,
    exclude_qubits: set[int] | None = None,
    strategy: Literal["lowest_cost", "ces_spread", "hybrid"] = "lowest_cost",
) -> LayoutSelection:
```
- Orquesta el pipeline completo: Capa 0 → Capa 1 → Capa 2 → Capa 3
- Transpila solo los top-N finales (no todos los candidatos)
- Calcula CES real post-transpilación
- Filtra por `max_ces` (como `select_layouts_low_ces`)
- **Retorna**: `LayoutSelection` (mismo dataclass que noisy_utils — zero breaking changes)
- **Estrategias**:
  - `"lowest_cost"`: Top-N por fidelity cost (para PEA primary)
  - `"ces_spread"`: Maximizar spread CES (para GF-ZNE inhomogéneo)
  - `"hybrid"`: Top-2 lowest cost + 1 high-CES (para adaptive ZNE)

### Tarea 1.6: Implementar multi-backend ranking
```python
def rank_backends(
    circuit: QuantumCircuit,
    backends: list,
    *,
    n_top: int = 3,
    max_2q_error: float = 0.01,
    cost_function: Callable | None = None,
) -> list[tuple[list[int], str, float]]:
```
- Para cada backend: filtrar CouplingMap → VF2 → score → best layout
- **Retorna**: `[(best_layout, backend_name, cost)]` ordenado por costo
- Uso: seleccionar entre ibm_kingston (Heron R2) e ibm_boston (Heron R3) para thesis data

---

## Fase 2: Dataclasses y Modelos de Resultado

### Tarea 2.1: Crear LayoutOptimizationResult
```python
@dataclass
class LayoutOptimizationResult:
    """Complete result of noise-aware layout optimization."""
    selected_layouts: list[list[int]]
    fidelity_costs: list[float]           # Per-layout fidelity cost
    ces_values: list[float]               # Per-layout CES post-transpilación
    transpiled_circuits: list[QuantumCircuit]
    total_vf2_layouts_found: int          # Cuántos encontró VF2 (antes de filtrar)
    filtered_coupling_map_edges: int      # Edges en el CMap filtrado
    original_coupling_map_edges: int      # Edges en el CMap original
    filtering_stats: dict[str, Any]       # {excluded_qubits, excluded_edges, ...}
    backend_name: str
    strategy_used: str
    elapsed_s: float
```
- **Archivo**: Dentro de `layout_optimizer.py` o en `config.py` si crece

### Tarea 2.2: Extensión de HardwareConfig
- **Archivo**: `src/qmbp_simulation/execution/hardware/config.py`
- Agregar campos opcionales:
```python
    # Layout optimizer settings (mapomatic integration)
    use_mapomatic: bool = True              # Enable VF2 layout selection
    layout_max_2q_error: float = 0.01       # Capa 0: edge quality threshold
    layout_min_t1_us: float = 50.0          # Capa 0: qubit T1 threshold
    layout_call_limit: int = 100_000        # Capa 1: VF2 search limit
    layout_exclude_qubits: list[int] = field(default_factory=list)  # Manual blacklist
    layout_strategy: str = "lowest_cost"    # Capa 3: selection strategy
```
- **Backward compatible**: Defaults = comportamiento actual (BFS) si `use_mapomatic=False`

---

## Fase 3: Integración con Pipeline Existente

### Tarea 3.1: Adaptar `select_layouts_for_hardware()`
- **Archivo**: `src/qmbp_simulation/execution/hardware/submission.py`
- **Cambio**: Agregar branch que usa `select_optimal_layouts()` cuando `config.use_mapomatic=True`
- **Fallback**: Si mapomatic no está instalado, log warning y usar BFS

```python
def select_layouts_for_hardware(bound_circuit, backend, config, logger):
    if config.use_mapomatic and MAPOMATIC_AVAILABLE:
        return select_optimal_layouts(
            bound_circuit, backend,
            n_select=config.n_layouts,
            max_ces=config.max_ces,
            max_2q_error=config.layout_max_2q_error,
            min_t1_us=config.layout_min_t1_us,
            optimization_level=config.optimization_level,
            call_limit=config.layout_call_limit,
            exclude_qubits=set(config.layout_exclude_qubits),
            strategy=config.layout_strategy,
        )
    else:
        # Existing BFS path (unchanged)
        ...
```

### Tarea 3.2: Integrar en `run_ibm_deployment.py`
- **Archivo**: `scripts/experiment_runners/hardware/run_ibm_deployment.py`
- **Cambios**:
  - `HardwareConfig` usa `use_mapomatic=True` por defecto
  - CLI flag `--no-mapomatic` para desactivar
  - Loggear stats comparativas: VF2 layouts found vs BFS candidates
  - Persistir `LayoutOptimizationResult` en el JSON de resultados

### Tarea 3.3: Integrar en preflight
- **Archivo**: `src/qmbp_simulation/execution/hardware/preflight.py`
- **Cambio**: `validate_transpiled_circuit_quality()` ya es compatible (recibe transpiled circuit + layout)
- Solo verificar que los layouts de mapomatic pasan el preflight existente sin cambios

### Tarea 3.4: Wiring en `__init__.py`
- **Archivo**: `src/qmbp_simulation/execution/hardware/__init__.py`
- **Exports nuevos**:
```python
from qmbp_simulation.execution.hardware.layout_optimizer import (
    MAPOMATIC_AVAILABLE,
    build_filtered_coupling_map,
    find_vf2_layouts,
    compute_layout_fidelity_cost,
    select_optimal_layouts,
    rank_backends,
    LayoutOptimizationResult,
)
```

---

## Fase 4: Tests

### Tarea 4.1: Tests unitarios del módulo
- **Archivo**: `tests/unit/test_layout_optimizer.py`
- **Casos**:
  1. `build_filtered_coupling_map` excluye edges con error alto (mock backend)
  2. `build_filtered_coupling_map` excluye qubits con T1 bajo
  3. `find_vf2_layouts` devuelve layouts válidos para circuito HVA N=6
  4. `find_vf2_layouts` fallback a BFS cuando mapomatic no disponible
  5. `compute_layout_fidelity_cost` ordena correctamente (layout con menos error primero)
  6. `compute_layout_fidelity_cost` penaliza defective edges
  7. `select_optimal_layouts` produce LayoutSelection compatible
  8. `select_optimal_layouts` respeta max_ces
  9. `select_optimal_layouts` con strategy="ces_spread" da layouts diversos
  10. `rank_backends` ordena backends por costo

### Tarea 4.2: Test de integración con FakeTorino
- **Archivo**: `tests/integration/test_layout_optimizer_integration.py`
- **Casos**:
  1. Pipeline completo: HVA N=10 p=1 → mapomatic → transpile → validate
  2. Comparar resultados VF2 vs BFS (layouts, CES, depth_2q)
  3. Verificar que `select_layouts_for_hardware` con `use_mapomatic=True` produce resultado válido
- **Requiere**: `qiskit-aer` (FakeTorino). Marcar con `@pytest.mark.integration`

### Tarea 4.3: Test de graceful degradation
- Mock `import mapomatic` failure → verificar que todo cae a BFS sin error

---

## Fase 5: Documentación y Persistencia

### Tarea 5.1: Actualizar code-style.md
- **Archivo**: `.kiro/steering/code-style.md`
- Agregar sección de imports para layout_optimizer en el bloque de execution/hardware

### Tarea 5.2: Actualizar project-status.md
- **Archivo**: `.kiro/steering/project-status.md`
- Agregar entrada en "Active Development" y "References"

### Tarea 5.3: Binnacle de integración
- **Archivo**: `documentation/binnacles/binnacle-mapomatic-integration.md`
- Documentar: decisión de diseño, resultados de validación, comparativa VF2 vs BFS

### Tarea 5.4: Actualizar hardware deployment spec
- **Archivo**: `HARDWARE_DEPLOYMENT_SPEC.md`
- Agregar sección sobre layout optimization con mapomatic

---

## Fase 6: Extensiones Futuras (post-validación)

### Tarea 6.1: T1/T2 decoherence-aware scoring (scheduling)
- Integrar idle-time decoherence en la cost function (como el ejemplo del README de mapomatic)
- Requiere: `transpile()` con `scheduling_method='alap'` → medir delays → calcular idle error
- **Prioridad**: Baja (el costo extra de transpilación por layout puede no justificarse para N=10)

### Tarea 6.2: Integración con κ (landscape curvature)
- Correlacionar fidelity cost del layout con κ per-h
- Si un layout tiene alto costo Y el h-point tiene alto κ → flag como "high risk"
- **Prioridad**: Media (informativo para go/no-go decision)

### Tarea 6.3: Integración con GNN-QEM circuit selection
- El GNN predictive mode (Spearman ρ=0.945) puede rankear circuitos por dificultad esperada
- Combinar ranking GNN con ranking mapomatic para selección óptima pre-ejecución
- **Prioridad**: Baja (ya tenemos κ como proxy)

### Tarea 6.4: Dynamic recalibration monitoring
- Si `check_calibration_drift()` detecta T1 drift > 20%, re-ejecutar `build_filtered_coupling_map()`
- Nuevo layout selection mid-experiment si calibración degrada
- **Prioridad**: Media (para runs largos de Tier 2/3)

### Tarea 6.5: Layout caching
- Cache de layouts válidos por backend+circuito hash (evitar re-ejecutar VF2 en cada h-point)
- Invalidar cache si calibración cambia (timestamp check)
- **Prioridad**: Baja (VF2 es O(ms), no bottleneck real)

---

## Diagrama de Arquitectura

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    LAYOUT OPTIMIZER MODULE                                │
│              src/qmbp_simulation/execution/hardware/layout_optimizer.py   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐                 │
│  │   Capa 0    │    │    Capa 1    │    │   Capa 2    │                 │
│  │  Pre-filter │───→│  VF2 Search  │───→│  Scoring    │                 │
│  │  CouplingMap│    │  (mapomatic) │    │  (Target)   │                 │
│  └─────────────┘    └──────────────┘    └─────────────┘                 │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  Exclude bad edges   call_limit=1e5    gate_error+readout               │
│  Exclude bad qubits  strict_direction  defective penalty                │
│  Exclude blacklist   max_layouts=200   sort ascending                   │
│                                                                          │
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │     Capa 3      │                                   │
│                    │  Top-N + Transp │                                   │
│                    └─────────────────┘                                   │
│                             │                                            │
│                             ▼                                            │
│                    ┌─────────────────┐                                   │
│                    │LayoutSelection  │ (same dataclass as noisy_utils)   │
│                    │ .layouts        │                                   │
│                    │ .ces_values     │                                   │
│                    │ .transpiled_qcs │                                   │
│                    └─────────────────┘                                   │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│  FALLBACK: if mapomatic not installed → find_layouts_bfs() + low_ces    │
└──────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  submission.py :: select_layouts_for_hardware()                           │
│  → Llama select_optimal_layouts() si use_mapomatic=True                  │
│  → Llama find_layouts_bfs() + select_layouts_low_ces() si False          │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  preflight.py :: validate_transpiled_circuit_quality()                    │
│  → Valida error_budget, depth_2q, defective_edges                        │
│  → Sin cambios (ya compatible con cualquier source de layouts)           │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  run_ibm_deployment.py :: run_tier_0/1/2/3()                             │
│  → HardwareConfig(use_mapomatic=True, layout_max_2q_error=0.01, ...)    │
│  → Persiste LayoutOptimizationResult en summary.json                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Criterios de Aceptación Global

1. **Zero breaking changes**: Todo el pipeline existente funciona idéntico con `use_mapomatic=False`
2. **Graceful degradation**: Si mapomatic no está instalado, warning + fallback a BFS
3. **Typing completo**: Módulo pasa `mypy --strict`
4. **Tests**: ≥10 unit tests + ≥3 integration tests
5. **Performance**: Overhead de VF2 < 1s para N=10 en backends de 156 qubits
6. **Compatibilidad BackendV2**: Cost function usa SOLO `backend.target`, nunca `backend.properties()`
7. **Persistencia**: Resultados de layout optimization guardados en JSON (via `json_serialize`)
8. **CI green**: No rompe lint + mypy + tests existentes

---

## Orden de Ejecución Recomendado

```
Fase 0 (deps)        → 0.1, 0.2
Fase 1 (core)        → 1.2, 1.3, 1.4, 1.5, 1.1(crear archivo), 1.6
Fase 2 (dataclasses) → 2.1, 2.2
Fase 3 (integración) → 3.4, 3.1, 3.2, 3.3
Fase 4 (tests)       → 4.1, 4.3, 4.2
Fase 5 (docs)        → 5.1, 5.2, 5.3, 5.4
Fase 6 (extensiones) → post-hardware-validation, según necesidad
```

**Estimación total**: ~4-6 horas de implementación (Fases 0-4), ~1h documentación (Fase 5).

---

## Estado de Ejecución

| Fase | Estado | Fecha |
|------|--------|-------|
| 0 — Deps | ✅ Completada | 2026-06-17 |
| 1 — Core module | ✅ Completada | 2026-06-17 |
| 2 — Dataclasses | ✅ Completada | 2026-06-17 |
| 3 — Integración pipeline | ✅ Completada | 2026-06-17 |
| 4 — Tests (48 total) | ✅ Completada | 2026-06-17 |
| 5 — Documentación | ✅ Completada | 2026-06-17 |
| 6 — Extensiones futuras | ⏳ Post-hardware-validation |

### Resultados de Validación
- VF2 mean CES = 0.029 vs BFS mean CES = 0.171 (6× improvement)
- 2614 VF2 layouts found in <1s on FakeTorino (N=10 chain)
- 0 extra SWAPs in transpiled VF2 circuits (isomorphism guarantee)
- 488 pre-existing tests unaffected
