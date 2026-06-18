---
inclusion: fileMatch
fileMatchPattern: "**/layout_optimizer*,**/submission.py,**/test_layout_optimizer*"
---

# Layout Optimizer — Mapomatic VF2 Integration

## Module Location
`src/qmbp_simulation/execution/hardware/layout_optimizer.py`

## What It Does

Multi-layer noise-aware layout optimization for IBM QPU deployment:
- **Layer 0**: Pre-filter the CouplingMap (remove bad qubits by T1, bad edges by error)
- **Layer 1**: VF2 subgraph isomorphism via mapomatic (finds SWAP-free layouts)
- **Layer 2**: Score layouts by fidelity cost (BackendV2 Target API)
- **Layer 3**: Select top-N + transpile (strategy-aware)

## Key Advantage Over BFS

VF2 finds subgraphs that are **isomorphic to the circuit's interaction graph** →
layouts are guaranteed SWAP-free. BFS just finds connected subgraphs that may
need SWAPs. Validated: VF2 gives ~6× lower CES than BFS on FakeTorino N=10.

## API Quick Reference

```python
from qmbp_simulation.execution.hardware import (
    MAPOMATIC_AVAILABLE,
    build_filtered_coupling_map,
    find_vf2_layouts,
    compute_layout_fidelity_cost,
    select_optimal_layouts,
    rank_backends,
    LayoutOptimizationResult,
)
```

## Configuration (HardwareConfig fields)

| Field | Default | Description |
|-------|---------|-------------|
| `use_mapomatic` | `True` | Enable VF2 (graceful fallback to BFS if unavailable) |
| `layout_max_2q_error` | `0.01` | Layer 0: exclude edges above this error rate |
| `layout_min_t1_us` | `50.0` | Layer 0: exclude qubits with T1 below this |
| `layout_call_limit` | `100_000` | Layer 1: VF2 search depth limit |
| `layout_exclude_qubits` | `[]` | Layer 0: manual qubit blacklist |
| `layout_strategy` | `"lowest_cost"` | Layer 3: selection strategy |

## Strategies

| Strategy | Use When | Behavior |
|----------|----------|----------|
| `lowest_cost` | PEA primary (default) | Top-N with lowest fidelity cost |
| `ces_spread` | GF-ZNE inhomogeneous | Maximize CES diversity across layouts |
| `hybrid` | Adaptive ZNE | Top-(N-1) lowest + 1 higher-CES |

## Integration Points

1. **`submission.py :: select_layouts_for_hardware()`** — calls `select_optimal_layouts()` when enabled
2. **`backend.py :: _get_cached_layouts()`** — caches layout selection per circuit structure
3. **`run_ibm_deployment.py`** — CLI: `--no-mapomatic`, `--layout-strategy`, `--layout-max-2q-error`
4. **`runner_base.py :: HardwareValidationRunner`** — CLI: `--no-mapomatic`, `--layout-strategy`

## Constraints (ALWAYS enforce)

- **BackendV2 only**: Cost function uses `backend.target`, never `backend.properties()` (V1 API deprecated)
- **Graceful degradation**: If `mapomatic` not installed → BFS fallback, no error
- **Output compatibility**: Returns `LayoutSelection` (same dataclass as noisy_utils)
- **Defensive scoring**: Layouts shorter than circuit are skipped, not crashed
- **Defective edge penalty**: Any edge > 10% error gets +1.0 cost penalty (pushed to bottom)

## Dependency

- `mapomatic>=0.14` in `[project.optional-dependencies].hardware`
- Import guarded: `try: import mapomatic; MAPOMATIC_AVAILABLE = True`
- CI does NOT install hardware extras — import guard prevents CI failure

## Testing

- Unit tests: `tests/unit/test_layout_optimizer.py` (19 tests)
- Extended tests: `tests/unit/test_layout_optimizer_extended.py` (21 tests)
- Integration: `tests/integration/test_layout_optimizer_integration.py` (8 tests, needs FakeTorino)
- Sanity check: `check_layout_optimizer_integration` in `project_health/analysis/sanity_check.py`

## Project Health Integration

### Analyzer
```bash
python -m project_health.analysis.layout_optimizer_analyzer              # Summary
python -m project_health.analysis.layout_optimizer_analyzer --verbose    # Per-record detail
python -m project_health.analysis.layout_optimizer_analyzer --benchmark  # Live VF2 vs BFS on FakeTorino
python -m project_health.analysis.layout_optimizer_analyzer --json report.json  # JSON export
```

Scans `results/hardware/run_*/execution_log.json` and `results/experiments/exp_hw_rehearsal_v*/` for layout selection events. Reports VF2 vs BFS CES comparison and generates recommendations.

### Sanity Check (auto-registered)
Runs within `python -m project_health.analysis.sanity_check` under category `hardware_readiness`:
- `layout_optimizer_importable` — module loads without error
- `hw_config_mapomatic_fields` — HardwareConfig has all 6 mapomatic fields
- `mapomatic_installed` — optional dep available (warning if not)

### Structured Log Events (produced by submission.py)
- `layout_method`: `{method, strategy, reason}` — which path was chosen
- `layout_selection`: `{n_selected, ces_values, method}` — final selection result
- `layout_fallback`: `{reason}` — logged when VF2 fails and BFS is used

## Performance Notes

- VF2 is O(ms) for N=10 on 156-qubit backends
- Found 2614 candidate layouts in <1s on FakeTorino
- Scoring 200 layouts = negligible (no transpilation in Layer 2)
- Only top-N final layouts get transpiled (expensive step)
- For HVA circuits, all qubits are active → `deflate_circuit` is identity (no edge case)
