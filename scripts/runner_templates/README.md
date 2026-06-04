# Runner Templates

Este directorio contiene templates para crear nuevos runner scripts siguiendo
el estándar del framework. Los tres tipos de runner cubren todos los patrones
de ejecución del proyecto.

## Tipos de Runner

| Tipo | Cuándo usar | Ejemplo existente |
|------|-------------|-------------------|
| `ExperimentRunner` | Script que envuelve un solo `BaseExperiment` | `run_e4b_e4c_standard.py` |
| `ValidationRunner` | Suite multi-sección con tablas y métricas | `run_mps_pseudo_hardware.py` |
| `HardwareValidationRunner` | Ejecución en IBM Torino o FakeTorino | Deployment scripts |
| `VariantPipelineRunner` | Muchas variantes del pipeline en lote | `run_p1_pipeline_variants.py` |

## Garantías del framework

Cada runner base enforce automáticamente:

1. **Preflight validation** — Estructura (runner_id, hypothesis, sections) + IDs duplicados.
2. **Structured logging** — `StructuredLogger` captura eventos para análisis post-hoc.
3. **Result saving** — Dual-compatible con `result_io` + `digest/scanner.py` + `compare.py`.
4. **Error isolation** — Una sección fallida no aborta las demás (excepto `--stop-on-failure`).
5. **CLI estándar** — `--section`, `--skip-preflight`, `--verbose`, `--dry-run`, `--stop-on-failure`.
6. **Exit codes** — 0 si todo pasa, 1 si hay fallos (para CI/automation).
7. **Log independiente** — Archivo `log_*.json` separado para timing analysis post-hoc.

## Utilidades integradas en ValidationRunner

Métodos heredados que eliminan boilerplate repetitivo:

```python
# VQE descending sweep con warm-start
theta_map = self.vqe_descending_sweep(
    topology="chain_1d", n_qubits=10,
    h_values=[2.5, 2.0, 1.5], seed=42,
    p_layers=1, n_restarts=1,
    model="tfim_longitudinal", model_kwargs={"g": 0.3},
)

# Exact energy + gap (auto-dispatch: exact diag N≤15, DMRG N>15)
e_exact, gap = self.exact_ground_state("heavy_hex", 10, h=3.25)

# State fidelity |⟨ψ_exact|ψ_vqe⟩|²
fid = self.compute_fidelity(circuit, theta_opt, exact_state_vector)

# MPS truncation (noise proxy — low chi ≈ hardware decoherence)
psi_trunc = self.truncate_statevector_mps(psi, n_qubits=10, chi_max=16)
```

### Backend resolution

Los utility methods reutilizan automáticamente el backend de la subclase:
- Si `self.noiseless` existe → lo usa.
- Si `self.backend` existe → lo usa.
- Si ninguno → crea `NoiselessBackend()`.

Esto evita instanciar backends duplicados cuando `setup()` ya los creó.

## Compatibilidad con digest/compare

El output JSON de todo runner es **parseable por**:
- `project_health/digest/` (scanner.py → `ExperimentResult`)
- `project_health/compare.py` (ResultStore → verdicts)
- `project_health/analysis/` scripts

Requisitos del `build_config()` para compatibilidad:
```python
def build_config(self) -> dict:
    return {
        "experiment_id": "E4b",       # REQUIRED: maps to exp_<id> folder
        "category": "E",              # REQUIRED: first letter
        "hypothesis": "...",          # REQUIRED: for compare output
        "description": "...",         # REQUIRED: for reports
        "system": {                   # REQUIRED: for digest scanner
            "n_qubits": 6,
            "p_layers": 1,
            "topology": "chain_1d",
            "model": "tfim_longitudinal",
        },
        "seeds": [42, 43, 44],        # REQUIRED: for analysis.n_seeds
    }
```

## Cross-section data sharing

Pattern para pasar datos entre secciones:
```python
def setup(self):
    # Declare shared state in setup()
    self._calibrated_chi: int | None = None
    self._phase2_data: dict | None = None

def section_1(self) -> dict:
    # Populate shared state
    self._calibrated_chi = 16
    return {"chi": 16, "pass": True}

def section_2(self) -> dict:
    # Use shared state (with guard)
    chi = self._calibrated_chi or 16
    ...
```

**Nota**: Las secciones se ejecutan secuencialmente en orden de `define_sections()`.
Si una sección depende de otra, el guard con fallback es suficiente.

## Cómo crear un nuevo runner

1. Identifica el tipo: `ValidationRunner` (multi-sección), `ExperimentRunner` (wrapper),
   o `VariantPipelineRunner` (batch).
2. Copia el template correspondiente.
3. Define los 4 atributos obligatorios: `runner_id`, `experiment_id`, `description`, `hypothesis`.
4. Si es nuevo experiment_id, agrégalo a `src/qmbp_simulation/framework/criteria.py`.
5. Implementa `define_sections()` y cada sección.
6. Implementa `build_config()` con los campos requeridos para digest.
7. Ejecuta con `python scripts/mi_runner.py --dry-run` para verificar estructura.
8. Ejecuta con `python scripts/mi_runner.py` para la ejecución real.

## CLI flags disponibles

| Flag | Efecto |
|------|--------|
| `--section 1 3` | Ejecuta solo secciones 1 y 3 |
| `--dry-run` | Lista secciones sin ejecutar |
| `--stop-on-failure` | Aborta al primer fallo |
| `--skip-preflight` | Salta validación de estructura |
| `--verbose` / `-v` | Logging DEBUG + tracebacks completos |

## Archivos

- `template_validation_runner.py` — Template para suites de validación multi-sección.
- `template_experiment_runner.py` — Template para wrappers de BaseExperiment.
- `template_variant_runner.py` — Template para pipeline variant runners.

## Módulo base

`src/qmbp_simulation/framework/runner_base.py` — Contiene:
- `ValidationRunner` (ABC) — Multi-sección con preflight + logging + result saving.
- `ExperimentRunner` (ABC) — Wrapper de BaseExperiment con CLI.
- `VariantPipelineRunner` (ABC) — Batch execution con preflight sincronizado.
- `Section` / `SectionResult` — Data models.
- `resolve_project_root()` — Utility para encontrar el root desde cualquier profundidad.

## Tests

`tests/test_runner_base.py` — 29 tests cubriendo:
- Preflight (structural checks, duplicate IDs, empty runner).
- Lifecycle (run, dry-run, stop-on-failure, section filter, skip-preflight).
- Result envelope (structure, digest compatibility, structured log).
- Utility methods (VQE sweep, exact energy, fidelity, MPS truncation).
- Error handling (import errors, missing config, None returns).
- Backend resolution.
- Criteria registration.
