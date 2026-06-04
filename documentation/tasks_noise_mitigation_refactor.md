# Tasks — Noise Suppression Pipeline Refactoring

**Created**: 2026-06-04
**Goal**: Corregir inconsistencias estructurales en el pipeline de mitigación de errores
y preparar la infraestructura para deployment confiable en IBM Torino.
**Total estimated time**: ~4 horas
**Prerequisito**: PEA implementado (2026-06-04, completado).

---

## Contexto del Problema

La auditoría del pipeline de noise suppression identificó 6 issues:

| # | Issue | Severidad | Impacto |
|---|-------|-----------|---------|
| 1 | Doble ZNE en hardware mode (IBM server-side + client CES-extrap) | 🔴 Crítico | Extrapolación sobre datos ya mitigados viola el modelo lineal |
| 2 | `NoisyBackend` ignora `MitigationOptions` silenciosamente | 🟡 Medio | Confusión API — usuario cree que aplica mitigación cuando no |
| 3 | `shots_per_randomization` inconsistente (512 en HardwareConfig vs 128 en dataclass) | 🟡 Medio | Overhead 4× innecesario en noise learning |
| 4 | PEA local simula depolarizing, no Pauli-Lindblad completo | 🟡 Menor | Resultados locales ligeramente optimistas vs hardware real |
| 5 | Gate-folding multi-layout no integrado en `HardwareBackend.evaluate()` | 🟡 Mejora | Falta la mejor estrategia (GF-ZNE + layout averaging) |
| 6 | No hay fallback automático GF→PEA por R² threshold | 🟡 Mejora | Requiere intervención manual para cambiar amplifier |

---

## Principios de Diseño (para todas las tasks)

1. **No romper backward compat** — scripts existentes deben seguir funcionando.
2. **Modular** — cada task se puede implementar y testear independientemente.
3. **Observable** — cada cambio produce output verificable (JSON, logs, o test results).
4. **Un solo source of truth** — configuración fluye `MitigationOptions` → `HardwareConfig` → `evaluate()`.

---

## Task NM-1: Resolver Doble ZNE en Hardware Mode

**Prioridad**: 🔴 BLOQUEANTE para hardware deployment
**Esfuerzo**: 1.5 horas
**Archivos**: `src/qmbp_simulation/execution/hardware/backend.py`, `submission.py`

### Problema

`HardwareBackend.evaluate()` aplica DOS niveles de ZNE simultáneamente en `mode="hardware"`:
1. IBM Runtime server-side ZNE (configurado por `build_estimator_options()` → `zne_mitigation=True`)
2. Client-side CES-ZNE (en `evaluate()` → `linear_zne(ces_values, energies)`)

Los datos per-layout que recibe `linear_zne()` ya están mitigados por IBM,
rompiendo la suposición de que son "raw noisy at different CES".

### Solución

Refactorizar `HardwareBackend.evaluate()` con lógica condicional:

```python
def evaluate(self, circuit, hamiltonian, params):
    bound = circuit.assign_parameters(params)
    layout_selection = self._get_cached_layouts(bound)
    raw_results = submit_all_then_collect(...)

    energies = [r["energy"] for r in raw_results]

    if self._config.mitigation.zne_enabled:
        # IBM Runtime ya aplica ZNE server-side → solo promediar layouts
        # (variance reduction por √n_layouts, sin re-extrapolación)
        return float(np.mean(energies))
    else:
        # Sin IBM ZNE → usar CES-ZNE del lado cliente (legacy behavior)
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
        zne_result = linear_zne(np.array(ces_used), np.array(energies))
        return zne_result.extrapolated_value
```

### Subtareas

- [ ] **NM-1.1**: Modificar `HardwareBackend.evaluate()` con branching por `zne_enabled`
- [ ] **NM-1.2**: Modificar `HardwareBackend.run_deployment()` con la misma lógica
  - Guardar tanto el promedio como los valores individuales per-layout en `HardwareRunResult`
  - Calcular `zne_r2` solo cuando se usa CES-ZNE; para layout-averaging, calcular `std` entre layouts
- [ ] **NM-1.3**: Actualizar `HardwareRunResult` dataclass:
  - Agregar campo `mitigation_strategy: str` → "ibm_zne_layout_avg" | "ces_zne" | "gate_folding_local"
  - Agregar campo `layout_std: float | None` → std entre layouts (cuando se usa averaging)
- [ ] **NM-1.4**: Actualizar `README.md` del hardware module con la nueva lógica

### Verificación

- [ ] Test unitario: `HardwareBackend` con `zne_enabled=True` retorna `mean(energies)` (no `linear_zne`)
- [ ] Test unitario: `HardwareBackend` con `zne_enabled=False` retorna `linear_zne(ces, energies)`
- [ ] Test de regresión: `mode="fake_backend"` sigue usando CES-ZNE (build_estimator_options no envía ZNE)
- [ ] Ejecutar `run_hardware_rehearsal.py` en modo fake_backend → mismos resultados que antes

### Riesgo

Si `linear_zne()` se elimina para el path hardware+zne_enabled, pierde la métrica R².
**Mitigación**: Guardar las energías per-layout y CES en el resultado aunque no se use para extrapolación — permite diagnóstico post-hoc.

---

## Task NM-2: Integrar Gate-Folding ZNE + Layout Averaging en HardwareBackend

**Prioridad**: 🟡 ALTA — la mejor estrategia validada para heavy_hex
**Esfuerzo**: 1 hora
**Archivos**: `src/qmbp_simulation/execution/hardware/backend.py`
**Depende de**: NM-1 (branch logic must exist)

### Problema

La estrategia óptima validada (GF-ZNE en cada layout + promedio) existe como
utility en `noisy_utils.py` (`run_gate_folding_zne_deployment(multi_layout=True)`)
pero NO está integrada en `HardwareBackend`. Los runners la usan directamente,
bypassing el backend ABC.

### Solución

Agregar un tercer branch en `evaluate()` para modo `fake_backend` + gate-folding local:

```python
def evaluate(self, circuit, hamiltonian, params):
    bound = circuit.assign_parameters(params)
    layout_selection = self._get_cached_layouts(bound)

    if self._config.mode == "hardware" and self._config.mitigation.zne_enabled:
        # Path A: IBM Runtime server-side ZNE + layout averaging
        raw_results = submit_all_then_collect(...)
        energies = [r["energy"] for r in raw_results]
        return float(np.mean(energies))

    elif self._config.mode == "fake_backend" and self._config.mitigation.zne_enabled:
        # Path B: Local gate-folding/PEA ZNE + layout averaging
        amplifier = self._config.mitigation.zne_amplifier
        noise_factors = tuple(self._config.mitigation.zne_noise_factors or [1, 3, 5])

        if amplifier == "pea":
            from qmbp_simulation.execution.noisy_utils import run_pea_zne_deployment
            result = run_pea_zne_deployment(
                bound, hamiltonian, self.backend, layout_selection,
                self._noisy_config, noise_factors=noise_factors, multi_layout=True,
            )
            return result.energy_layout_avg or result.energy_pea_zne.extrapolated_value
        else:
            from qmbp_simulation.execution.noisy_utils import run_gate_folding_zne_deployment
            result = run_gate_folding_zne_deployment(
                bound, hamiltonian, self.backend, layout_selection,
                self._noisy_config, noise_factors=noise_factors, multi_layout=True,
            )
            return result.energy_layout_avg or result.energy_gf_zne.extrapolated_value

    else:
        # Path C: Legacy CES-ZNE (no IBM ZNE, no local amplification)
        raw_results = submit_all_then_collect(...)
        energies = [r["energy"] for r in raw_results]
        ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
        zne_result = linear_zne(np.array(ces_used), np.array(energies))
        return zne_result.extrapolated_value
```

### Subtareas

- [ ] **NM-2.1**: Agregar `_noisy_config` property a HardwareBackend (NoisyEstimatorConfig desde HardwareConfig)
- [ ] **NM-2.2**: Implementar el triple-branch en `evaluate()`
- [ ] **NM-2.3**: Implementar el mismo branching en `run_deployment()` (con métricas extendidas)
- [ ] **NM-2.4**: Agregar campo `amplifier_used: str` a `HardwareRunResult`

### Verificación

- [ ] Test: `mode="fake_backend"` + `zne_enabled=True` + `zne_amplifier="gate_folding"` → usa `run_gate_folding_zne_deployment`
- [ ] Test: `mode="fake_backend"` + `zne_enabled=True` + `zne_amplifier="pea"` → usa `run_pea_zne_deployment`
- [ ] Test: `mode="fake_backend"` + `zne_enabled=False` → usa CES-ZNE legacy
- [ ] Test: `mode="hardware"` + `zne_enabled=True` → layout averaging (no local ZNE)
- [ ] Ejecutar `run_pea_hardware_readiness.py` → resultados consistentes con ejecución directa

---

## Task NM-3: Resolver `NoisyBackend` — Clarificar como "Raw-Only"

**Prioridad**: 🟡 MEDIA
**Esfuerzo**: 20 minutos
**Archivos**: `src/qmbp_simulation/execution/backends.py`

### Problema

`NoisyBackend` acepta `MitigationOptions` en su constructor pero NUNCA las consume:
```python
class NoisyBackend:
    def __init__(self, mitigation=MitigationOptions(), ...):
        self._mitigation = mitigation  # ← stored, never read
```

Esto viola el principio de menor sorpresa — un usuario que pasa `MitigationOptions(zne_enabled=True)`
espera que el backend aplique ZNE, pero no lo hace.

### Solución: Deprecar el parámetro + Documentar

```python
class NoisyBackend(ExecutionBackend):
    """Shot-noise simulation — RAW noise only, no mitigation applied.

    For mitigated noisy simulation, use the utility functions directly:
    - Gate-folding ZNE: run_gate_folding_zne() from noisy_utils
    - PEA-ZNE: run_pea_zne() from noisy_utils
    - CES-ZNE: run_zne_deployment() from noisy_utils

    This backend is intended for:
    - Generating "noisy raw" baselines (factor=1, no mitigation)
    - VQE training with shot noise approximation
    - Quick noise-level estimation without full mitigation stack
    """

    def __init__(
        self,
        shots: int = 8192,
        noise_model=None,
        mitigation: MitigationOptions | None = None,  # DEPRECATED — ignored
        seed_simulator: int | None = None,
    ) -> None:
        if mitigation is not None and mitigation.zne_enabled:
            import warnings
            warnings.warn(
                "NoisyBackend does not apply mitigation options. "
                "Use run_gate_folding_zne() or run_pea_zne() from "
                "qmbp_simulation.execution.noisy_utils for mitigated estimation.",
                DeprecationWarning,
                stacklevel=2,
            )
        ...
```

### Subtareas

- [ ] **NM-3.1**: Agregar `DeprecationWarning` cuando se pasa `MitigationOptions` con cualquier flag enabled
- [ ] **NM-3.2**: Actualizar docstring de `NoisyBackend` para documentar que es "raw-only"
- [ ] **NM-3.3**: Buscar y verificar que ningún script existente depende de mitigación vía `NoisyBackend`
  - Grep: `NoisyBackend.*mitigation` en `scripts/` y `tests/`
- [ ] **NM-3.4**: Actualizar `code-style.md` steering: agregar nota "NoisyBackend = raw noise only"

### Verificación

- [ ] `NoisyBackend()` sin args → no warning, funciona igual
- [ ] `NoisyBackend(mitigation=MitigationOptions(zne_enabled=True))` → emite DeprecationWarning
- [ ] Todos los tests existentes pasan sin warnings inesperados
- [ ] `ruff check` limpio

---

## Task NM-4: Alinear `shots_per_randomization` Default

**Prioridad**: 🟡 MEDIA
**Esfuerzo**: 10 minutos
**Archivos**: `src/qmbp_simulation/execution/hardware/config.py`

### Problema

Inconsistencia entre defaults:
- `MitigationOptions` dataclass: `shots_per_randomization = 128` (correcto, alineado con IBM default)
- `HardwareConfig` default factory: `shots_per_randomization=512` (4× más de lo necesario)

512 shots per randomization genera ~4× overhead en la fase de noise learning de PEA
sin beneficio significativo para nuestros circuitos de 18 CZ gates.

### Solución

```python
# hardware/config.py — alinear con IBM defaults y MitigationOptions
mitigation: MitigationOptions = field(
    default_factory=lambda: MitigationOptions(
        dd_enabled=True,
        trex_enabled=True,
        twirling_enabled=True,
        zne_enabled=True,
        num_randomizations=32,
        shots_per_randomization=128,  # ← Fix: era 512, IBM default es 128
    )
)
```

### Subtareas

- [ ] **NM-4.1**: Cambiar `shots_per_randomization=512` → `128` en `HardwareConfig` default
- [ ] **NM-4.2**: Agregar comentario explicando la razón: "IBM LayerNoiseLearning default=128"
- [ ] **NM-4.3**: Verificar que ningún test hardcodea `512` como valor esperado

### Verificación

- [ ] `from qmbp_simulation.execution import HardwareConfig; c = HardwareConfig(); assert c.mitigation.shots_per_randomization == 128`
- [ ] `ruff check` limpio
- [ ] Tests de runner_base.py pasan

---

## Task NM-5: Implementar Fallback Automático GF→PEA

**Prioridad**: 🟡 MEJORA — reduce intervención manual en hardware
**Esfuerzo**: 45 minutos
**Archivos**: `src/qmbp_simulation/execution/noisy_utils.py`
**Depende de**: NM-2 (backend integration)

### Problema

Actualmente, si gate-folding ZNE da mal R², el usuario debe manualmente
cambiar `--zne-amplifier pea` y re-ejecutar. En un run de hardware con
créditos limitados, perder un h-point por no hacer fallback es costoso.

### Solución

Crear una función de deployment "adaptativa" que intenta GF primero y
cae a PEA si la calidad de extrapolación es insuficiente:

```python
@dataclass
class AdaptiveZNEResult:
    """Result from adaptive GF→PEA ZNE strategy."""
    extrapolated_value: float
    r_squared: float
    amplifier_used: str  # "gate_folding" | "pea"
    gf_result: GateFoldingZNEResult | None
    pea_result: PEAResult | None
    fallback_triggered: bool


def run_adaptive_zne(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...] = (1, 3, 5),
    r2_threshold: float = 0.90,
    extrapolator: str = "linear",
    seed_offset: int = 0,
) -> AdaptiveZNEResult:
    """Run GF-ZNE first; if R² < threshold, fall back to PEA-ZNE.

    Cost-aware: only pays PEA overhead when GF fails.
    """
    # Step 1: Try gate-folding
    gf_result = run_gate_folding_zne(
        transpiled_circuit, observable, backend, config,
        noise_factors=tuple(int(nf) for nf in noise_factors),
        extrapolator=extrapolator, seed_offset=seed_offset,
    )

    if gf_result.r_squared >= r2_threshold:
        return AdaptiveZNEResult(
            extrapolated_value=gf_result.extrapolated_value,
            r_squared=gf_result.r_squared,
            amplifier_used="gate_folding",
            gf_result=gf_result, pea_result=None,
            fallback_triggered=False,
        )

    # Step 2: Fallback to PEA
    _logger.warning(
        f"[adaptive_zne] GF R²={gf_result.r_squared:.3f} < {r2_threshold}, "
        f"falling back to PEA"
    )
    pea_result = run_pea_zne(
        transpiled_circuit, observable, backend, config,
        noise_factors=noise_factors,
        extrapolator=extrapolator, seed_offset=seed_offset + 5000,
    )

    return AdaptiveZNEResult(
        extrapolated_value=pea_result.extrapolated_value,
        r_squared=pea_result.r_squared,
        amplifier_used="pea",
        gf_result=gf_result, pea_result=pea_result,
        fallback_triggered=True,
    )
```

### Subtareas

- [ ] **NM-5.1**: Implementar `AdaptiveZNEResult` dataclass y `run_adaptive_zne()` en `noisy_utils.py`
- [ ] **NM-5.2**: Exportar desde `execution/__init__.py`
- [ ] **NM-5.3**: Agregar `--zne-amplifier adaptive` como tercera opción en CLI (`framework/cli.py` y `runner_base.py`)
- [ ] **NM-5.4**: Integrar en `HardwareBackend.evaluate()` Path B (fake_backend + zne_enabled)
- [ ] **NM-5.5**: Agregar campo `fallback_triggered: bool` a `HardwareRunResult`

### Verificación

- [ ] Test: con R² artificial < 0.90 (circuito corto donde GF no tiene spread) → fallback triggered
- [ ] Test: con R² natural > 0.90 (circuito normal) → no fallback, usa GF
- [ ] Test: `--zne-amplifier adaptive` funciona desde CLI
- [ ] Log muestra "falling back to PEA" cuando se activa
- [ ] `HardwareRunResult.fallback_triggered` se guarda en JSON output

### Parámetro configurable

El threshold `r2_threshold=0.90` debe ser configurable:
- CLI: `--zne-r2-threshold 0.90`
- En `MitigationOptions`: agregar campo `zne_r2_fallback_threshold: float = 0.90`

---

## Task NM-6: Documentar Limitación PEA Local (Depolarizing vs Pauli-Lindblad)

**Prioridad**: 🟡 MENOR — correctness documental
**Esfuerzo**: 15 minutos
**Archivos**: `src/qmbp_simulation/execution/noisy_utils.py`, `hardware/README.md`

### Problema

La implementación local de PEA (`_build_amplified_noise_model`) usa
`depolarizing_error(rate * factor, 2)` para simular la amplificación.
El PEA real de IBM aprende un modelo Pauli-Lindblad completo con 15
generadores por par de qubits (XX, XY, XZ, ...).

Esto significa que nuestra simulación local es una aproximación isotrópica
del noise channel real. Para FakeTorino (cuyo modelo de ruido ya ES
mayormente depolarizing), la aproximación es buena. Pero si se compara
contra hardware real, podría haber discrepancia.

### Solución: Documentar en código y README

### Subtareas

- [ ] **NM-6.1**: Agregar nota en docstring de `_build_amplified_noise_model()`:
  ```
  Note: This is an approximation. Real PEA (IBM Runtime) learns a full
  Pauli-Lindblad model (15 generators per 2Q pair) and amplifies each
  channel independently. Our local simulation uses isotropic depolarizing
  as a simplification, which is accurate for FakeTorino but may differ
  from real hardware by ~5-10% in extrapolated values.
  ```
- [ ] **NM-6.2**: Agregar sección "Limitations" en `hardware/README.md` bajo la sección de PEA
- [ ] **NM-6.3**: En `run_pea_hardware_readiness.py`, agregar footnote en el verdict:
  "Results are from depolarizing PEA approximation — hardware may differ by ±10%"

### Verificación

- [ ] `ruff check` limpio
- [ ] README.md tiene sección de limitaciones
- [ ] Docstring actualizado en `_build_amplified_noise_model`

---

## Orden de Ejecución y Dependencias

```
NM-4 (10min) ─── independiente, quick fix, hacer primero
     │
NM-3 (20min) ─── independiente, clarificación API
     │
NM-1 (1.5h) ──── CRÍTICO: resolver doble-ZNE
     │
     └── NM-2 (1h) ─── depende de NM-1 (usa el branch logic)
              │
              └── NM-5 (45min) ─── depende de NM-2 (adaptive ZNE usa el backend)
     │
NM-6 (15min) ─── independiente, documentación pura
```

**Sesión recomendada**:
1. NM-4 (10min) — quick win, fix default
2. NM-3 (20min) — deprecation warning, documentar NoisyBackend
3. NM-6 (15min) — documentar limitación PEA
4. NM-1 (1.5h) — refactorizar evaluate() con branching
5. NM-2 (1h) — integrar GF/PEA deployment en backend
6. NM-5 (45min) — adaptive fallback

**Total**: ~4 horas

---

## Verificación Final (después de todas las tasks)

### Tests de regresión (TODOS deben pasar)

```bash
# Core tests (no deben romperse)
python -m pytest tests/test_runner_base.py -q

# Import chain verification
python -c "
from qmbp_simulation.execution import (
    MitigationOptions, HardwareConfig, HardwareBackend,
    run_pea_zne, run_gate_folding_zne, PEAResult, GateFoldingZNEResult,
)
from qmbp_simulation.framework import add_noisy_args
print('All imports OK')
"

# Hardware config consistency
python -c "
from qmbp_simulation.execution import HardwareConfig
c = HardwareConfig()
assert c.mitigation.shots_per_randomization == 128, f'Expected 128, got {c.mitigation.shots_per_randomization}'
assert c.mitigation.zne_amplifier == 'gate_folding'
print('Config defaults OK')
"

# Lint
ruff check src/qmbp_simulation/execution/ src/qmbp_simulation/framework/
```

### Tests funcionales (requieren FakeTorino)

```bash
# PEA integration (si qiskit-aer instalado)
python -m pytest tests/test_pea_integration.py -v

# Hardware rehearsal en fake_backend (verifica que no hay doble-ZNE)
python scripts/experiment_runners/run_hardware_rehearsal.py --section 2 --mode fake_backend
```

### Invariantes post-refactor (verificar manualmente)

| Invariante | Cómo verificar |
|-----------|---------------|
| `mode="fake_backend"` no aplica IBM ZNE | `build_estimator_options()` con mode=fake_backend retorna sin `zne_mitigation` |
| `mode="hardware"` + `zne_enabled=True` no hace CES-extrap | `evaluate()` retorna `mean(energies)`, no `linear_zne(...)` |
| `mode="hardware"` + `zne_enabled=False` hace CES-extrap | `evaluate()` retorna `linear_zne(ces, energies)` (legacy) |
| `NoisyBackend(mitigation=MitigationOptions(zne_enabled=True))` emite warning | Capturar con `warnings.catch_warnings()` |
| PEA local produce valores finitos para N=4 p=1 | `assert np.isfinite(result.extrapolated_value)` |
| Gate-folding local produce R²>0.5 para N=6 p=1 chain_1d | Validado en runs existentes |
| Adaptive ZNE usa GF si R²>0.90, PEA si R²<0.90 | Test con mock data |
| `HardwareRunResult.mitigation_strategy` se guarda en JSON | Check `json.load(result_file)["mitigation_strategy"]` |
| `--zne-amplifier adaptive` es opción válida de CLI | `parser.parse_args(["--zne-amplifier", "adaptive"])` no error |

---

## Errores a Evitar

| Trampa | Cómo caer | Prevención |
|--------|-----------|-----------|
| Romper mode="fake_backend" al refactorizar evaluate() | Olvidar que fake_backend usa `BackendEstimatorV2` local (no IBM) | Test explícito para fake_backend path |
| Doble conteo de shots en layout averaging | Contar shots per-layout × n_layouts × noise_factors | Mantener `_total_shots` accounting correcto |
| PEA fallback en loop infinito | R² siempre bajo porque circuito es too shallow | PEA fallback se ejecuta MAX 1 vez, sin re-retry |
| Mezclar GateFoldingZNEResult con PEAResult | Son dataclasses diferentes con campos distintos | Usar `AdaptiveZNEResult` como wrapper unificado |
| Perder backward compat en `build_estimator_options` | Cambiar behavior para configs que ya funcionan | Agregar tests para cada combinación de flags |
| Olvidar exportar nuevos tipos desde `__init__.py` | `from qmbp_simulation.execution import AdaptiveZNEResult` → ImportError | Checklist: agregar a `__init__.py` Y a `__all__` |

---

## Definición de Done

- [ ] `HardwareBackend.evaluate()` NO aplica doble-ZNE en ningún modo
- [ ] `mode="hardware"` + `zne_enabled=True` → layout averaging (not CES-extrap)
- [ ] `mode="fake_backend"` + `zne_enabled=True` → local GF/PEA via noisy_utils
- [ ] `NoisyBackend` emite DeprecationWarning si se pasa mitigation con flags activos
- [ ] `HardwareConfig` default `shots_per_randomization == 128`
- [ ] `run_adaptive_zne()` implementado y exportado
- [ ] `--zne-amplifier adaptive` funcional en CLI
- [ ] Limitación depolarizing de PEA local documentada
- [ ] Todos los tests existentes pasan (17/18 en test_runner_base)
- [ ] `ruff check` limpio en archivos modificados
- [ ] hardware/README.md actualizado con nueva lógica de branching
