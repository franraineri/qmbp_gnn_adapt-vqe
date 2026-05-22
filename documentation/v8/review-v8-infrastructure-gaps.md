# Revisión: Brechas de Infraestructura para V8

> Análisis de lo que falta en el código existente para ejecutar los 19 experimentos V8,
> más errores y mejoras detectados en las prácticas actuales.
>
> Fecha: 2026-05-22
> **Última actualización: 2026-05-22** — Parte 3 y Parte 5 actualizadas para reflejar
> el estado real del filesystem. Ver `documentation/STATUS-V8.md` como fuente de verdad.

---

## Parte 1: Brechas Bloqueantes para V8

### 1.1 Módulo MPS faltante (`experiment_mps_simulation.py`)

**Problema:** El script `experiment_p1_scaling.py` importa `run_vqe_mps`, `create_mps_backend`,
`evaluate_energy_mps` desde `experiment_mps_simulation`, pero estas funciones no están
disponibles como API reutilizable. Están enterradas en un script de experimento.

**Impacto en V8:** Experimentos E1 (N=30 pipeline) y B3 (LCC) necesitan evaluación MPS.

**Fix necesario:** Extraer las funciones MPS a `scripts/experiments_v8/techniques/mps_backend.py`
como API limpia, o mejor aún, integrarlas en `src/poc/v6/vqe_optimizer.py` como backend alternativo.

---

### 1.2 H-grid customizable en scripts de pipeline

**Problema:** Los scripts de pipeline (`run_v61_parametric.py`) usan grids fijos.
No hay forma estándar de especificar rangos custom (e.g., h∈[2.25, 4.0] para N=20 p=1).
Nota: `pipeline_core.py` está documentado como código muerto (zero imports) — el h-grid
se maneja directamente en los scripts runners.

**Impacto en V8:** Experimentos A3 (scaling law con N=4,8,14), E1 (N=30, h≥2.5),
E4 (2D phase diagram h×g), y E3 (active learning con grid adaptivo).

**Fix necesario:** El framework V8 ya resuelve esto con `ExperimentConfig.system.h_values`
que acepta listas arbitrarias. No se requiere modificar código V6 estable.

---

### 1.3 Gap computation robusta para N≥15

**Problema:** `ClassicalSolver._solve_dmrg()` retorna `gap=0` cuando el excited-state DMRG
converge al ground state. Downstream, `ΔE/gap` produce `inf`.

**Impacto en V8:** Experimento A1 (orthogonal projection DMRG) existe precisamente para
resolver esto. Pero mientras tanto, A3 (scaling law) y E1 (N=30) necesitan gaps válidos.

**Fix inmediato:** Usar gap analítico como fallback: `gap = max(2*abs(J-h), 2*pi/N)` cuando
DMRG retorna 0. Esto ya se hace en los binnacles pero no está codificado en `ClassicalSolver`.

---

### 1.4 Dataset vacío no manejado

**Problema:** Si el fidelity filter elimina TODOS los puntos, `build_graph_dataset()` retorna
lista vacía. `train_mpnn()` crashea con error críptico de PyTorch.

**Impacto en V8:** Experimentos con N=20 donde solo h≥2.0 pasa el filtro pueden producir
datasets muy pequeños (3-6 puntos). Con active learning (E3), el dataset empieza con 5 puntos.

**Fix necesario:** Validación en `train_mpnn()`:
```python
if len(dataset) < 3:
    raise ValueError(f"Dataset too small ({len(dataset)} points). Need ≥3 for training.")
```

---

### 1.5 Falta soporte para modelos extendidos en `HamiltonianBuilder`

**Problema:** `HamiltonianBuilder.build()` solo construye TFIM estándar (H = -J·ZZ - h·X).
El campo longitudinal (E4: H = -J·ZZ - h·X - g·Z) requiere extensión.

**Impacto en V8:** Experimento E4 (TFIM + longitudinal field) necesita un builder extendido.

**Fix necesario:** Ya existe `build_heisenberg()` como precedente. Agregar:
```python
def build_tfim_longitudinal(self, lattice, g: float = 0.0) -> SparsePauliOp:
    """TFIM with longitudinal field: H = -J·ΣZZ - h·ΣX - g·ΣZ"""
```

---

## Parte 2: Errores y Problemas Detectados en Prácticas Actuales

### 2.1 VQE Optimizer: Convergencia no validada

**Archivo:** `src/poc/v6/vqe_optimizer.py`

**Problema:** Después de `scipy.optimize.minimize()`, el código no verifica `result.success`.
Si L-BFGS-B no converge (e.g., alcanza maxiter), el resultado se acepta silenciosamente.
Esto puede producir θ_opt subóptimos que contaminan el dataset de Phase 3.

**Evidencia:** En binnacle-N10, seed 42 produce MSE 10× peor que seed 43. Parte de esto
puede ser VQE no-convergido en puntos difíciles (h≈1.0) que pasan el fidelity filter.

**Fix:**
```python
if not result.success:
    logger.warning(f"VQE did not converge at h={h:.3f}: {result.message}")
    # Optionally: mark this point as low-confidence
```

---

### 2.2 DMRG Gap: Fallo silencioso produce `gap=0`

**Archivo:** `src/poc/v6/classical_solver.py`, líneas 195-210

**Problema:** Cuando el excited-state DMRG converge al ground state, `gap=0.0` se retorna
sin warning. Downstream, `ΔE/gap` = `inf` o `nan`. El código que usa `gap` no siempre
verifica `gap > 0`.

**Evidencia:** En binnacle-p1-scaling, N=20 usa "approximate analytical gap" porque DMRG
falla. Esto está documentado pero no codificado como fallback automático.

**Fix:** En `_solve_dmrg()`, después de `gap = 0.0`:
```python
if gap == 0.0:
    # Analytical fallback for 1D TFIM
    h_val = float(lattice.h) if np.isscalar(lattice.h) else float(np.mean(lattice.h))
    j_val = float(lattice.J) if np.isscalar(lattice.J) else float(np.mean(lattice.J))
    gap = max(2 * abs(j_val - h_val), 2 * np.pi / lattice.n_qubits)
    gap_source = "analytical"  # Traceability: mark how gap was obtained
    logger.warning(
        f"DMRG excited state converged to GS. Using analytical gap={gap:.4f} "
        f"(valid for 1D TFIM far from h_c)"
    )
else:
    gap_source = "dmrg"

# Return both gap and source for downstream traceability
# result.gap_source allows consumers to know if gap is exact or approximate
```
Nota: Agregar `gap_source: Literal["dmrg", "analytical", "exact_diag"]` al resultado
permite que downstream (ΔE/gap reporting) indique la confiabilidad del denominador.

---

### 2.3 MPNN: Divergence detection con threshold hardcodeado

**Archivo:** `src/poc/v6/mpnn_predictor.py`, training loop

**Problema:** El threshold de divergencia (`mean_de > 0.01`) está hardcodeado. Para N=20
donde ΔE=0.04 es aceptable, este threshold dispararía false positives. Para N=6 donde
ΔE=0.001 es esperado, nunca se dispara.

**Fix:** Hacer configurable:
```python
def train_mpnn(..., divergence_threshold: float = 0.01, ...):
```

---

### 2.4 Pipeline Core: Sin validación entre fases

**Archivo:** `src/poc/v6/pipeline_core.py`

**Problema:** Si Phase 1 falla parcialmente (e.g., DMRG timeout en un punto), Phase 2
procede con datos incompletos. Si Phase 2 produce 0 puntos con fidelity≥0.93, Phase 3
recibe dataset vacío.

**Evidencia:** En los runs de N=20 (binnacle-hamed-v7), el primer intento incluyó h<1.5
donde VQE produce "garbage θ". No hubo warning de que esos puntos eran inútiles.

**Fix:** Agregar validación inter-fase:
```python
# After Phase 2
n_valid = np.sum(fidelities >= cfg.fidelity_threshold)
if n_valid == 0:
    raise ValueError("Phase 2 produced 0 valid points. Adjust h-grid or VQE config.")
if n_valid < 5:
    logger.warning(f"Only {n_valid} valid points for Phase 3. Results may be unreliable.")
```

---

### 2.5 Shared Runners: `evaluate_energy_statevector` crea nuevo Estimator cada vez

**Archivo:** `scripts/experiments_hamed_v7/experiment_utils.py`

**Problema:** `evaluate_energy_statevector()` instancia un nuevo `StatevectorEstimator()`
en cada llamada. Para un VQE con 500 iteraciones × 5 restarts × 27 h-points = 67,500 calls,
esto crea 67,500 estimator objects. El overhead es pequeño pero innecesario.

**Fix:** Usar dependency injection o `functools.lru_cache` (evitar global mutable):
```python
import functools

@functools.lru_cache(maxsize=1)
def _get_default_estimator():
    return StatevectorEstimator()

def evaluate_energy_statevector(circuit, hamiltonian, params, estimator=None):
    if estimator is None:
        estimator = _get_default_estimator()
    ...
```
Nota: `lru_cache` es thread-safe para la creación y evita el anti-pattern de `global`.

---

### 2.6 Sign Symmetry no manejada en training data

**Archivo:** `src/poc/v6/mpnn_predictor.py` (build_graph_dataset)

**Problema:** Cuando diferentes seeds de VQE encuentran θ con diferentes convenciones de
signo (Z₂ symmetry), el MPNN recibe targets inconsistentes. Esto fue identificado en
binnacle-p1-scaling pero NO está corregido en el código de producción.

**Impacto:** A N=20 p=1, seed 44 encuentra θ con signo opuesto a seeds 42/43. Si se
entrena el MPNN con datos de múltiples seeds sin canonicalizar, el MSE será alto.

**Fix:** Ya implementado en `scripts/experiments_v8/techniques/sign_equivariant.py`.
Necesita integrarse en `build_graph_dataset()`:
```python
if canonicalize_signs:
    theta_opt = canonicalize_dataset(theta_opt, reference_index=-1)
```

---

### 2.7 Experiment scripts: Path management frágil

**Archivo:** Todos los scripts en `scripts/experiments_hamed_v7/`

**Problema:** Cada script hace `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`.
Si el script se ejecutado desde un directorio diferente, o si la estructura cambia, los
imports fallan. No hay validación de que el path insertado es correcto.

**Fix en V8:** El `BaseExperiment` ya resuelve esto con detección de `pyproject.toml`:
```python
if not (PROJECT_ROOT / "pyproject.toml").exists():
    _p = Path(__file__).resolve()
    while _p != _p.parent:
        if (_p / "pyproject.toml").exists():
            PROJECT_ROOT = _p
            break
        _p = _p.parent
```

---

### 2.8 Falta de timeout en VQE para N≥20

**Problema:** VQE a N=20 con StatevectorEstimator puede tomar 50+ minutos. Si un restart
se queda en un loop (e.g., L-BFGS-B oscilando), no hay mecanismo de timeout.

**Evidencia:** En binnacle-hamed-v7, "Phase 2 took 60 min despite reduced config."

**Fix:** Usar `threading.Timer` para portabilidad (funciona en macOS, Linux, y threads):
```python
import threading

class VQETimeout(Exception):
    pass

def _timeout_callback(thread_id):
    import ctypes
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(VQETimeout)
    )

# Alternative simpler approach: use scipy's maxiter + callback with wall-clock check
def make_timeout_callback(max_seconds: float):
    """Create a scipy callback that raises after max_seconds."""
    import time
    start = time.monotonic()
    def callback(xk):
        if time.monotonic() - start > max_seconds:
            raise VQETimeout(f"VQE restart exceeded {max_seconds}s")
    return callback

# Usage in VQE:
result = minimize(..., callback=make_timeout_callback(max_seconds_per_restart))
```
Nota: `signal.alarm()` solo funciona en Unix main thread. La alternativa con callback
de scipy es portable y funciona en threads secundarios.

---

## Parte 3: Lo que Falta en el Framework V8 para Cubrir Todos los Experimentos

> **Actualizado 2026-05-22:** La mayoría de módulos técnicos ya fueron creados.
> Se marca el estado real del filesystem.

### 3.1 Técnicas (modules en `techniques/`)

| Experimento | Técnica necesaria | Estado |
|-------------|-------------------|--------|
| A1 | `orthogonal_dmrg.py` — TeNPy OrthogonalExcitations wrapper | ❌ No creado |
| A2 | `tci.py` — Tensor Cross Interpolation (necesita `xfac` o custom) | ❌ No creado |
| B1 | `analytical_init.py` — Perturbation theory initial guess | ✅ Creado |
| B2 | `parameter_freezing.py` — TITAN-style trajectory analysis | ✅ Creado |
| B3 | `light_cone.py` — LCC sub-circuit extraction | ❌ No creado |
| B4 | `hessian_restart.py` — Hessian-guided restart logic | ✅ Creado |
| C1 | `physics_loss.py` — Energy-validated MPNN training | ✅ Creado |
| C3 | `sign_equivariant.py` — Sign canonicalization strategies | ✅ Creado |
| D1 | (usa `WeightGradientAnalyzer` existente en `analysis_utils.py`) | ✅ Existe |
| D3 | `tensor_completion.py` — Low-rank tensor reconstruction | ❌ No creado |
| E3 | `active_learning.py` — Ensemble + acquisition functions | ✅ Creado |
| F1 | `dypp.py` — Dynamic parameter prediction (extrapolation) | ✅ Creado |

**Resumen:** 8/12 técnicas creadas. Faltan 4 (A1, A2, B3, D3) — todas excluidas
del plan final V8 por alto esfuerzo de implementación.

### 3.2 Experimentos (modules en `experiments/`)

| Archivo | Estado | Notas |
|---------|--------|-------|
| `exp_a3_scaling_law.py` | ✅ Creado + ejecutado | Resultados en `results/exp_a3/` |
| `exp_b1_analytical.py` | ✅ Creado + ejecutado | Resultados en `results/exp_b1/` |
| `exp_b2_freezing.py` | ✅ Creado | Pendiente de ejecución |
| `exp_b4_hessian.py` | ✅ Creado + ejecutado | Resultados en `results/exp_b4/` |
| `exp_c3_sign.py` | ✅ Creado | Pendiente de ejecución |
| `exp_f1_dypp.py` | ✅ Creado + ejecutado | Resultados en `results/exp_f1/` |
| `exp_f3_fluctuation.py` | ✅ Creado + ejecutado | Resultados en `results/exp_f3/` |
| `exp_c1_physics_loss.py` | ❌ No creado | Registrado en `__init__.py` → ImportError |
| `exp_d1_weight_space.py` | ❌ No creado | Registrado en `__init__.py` → ImportError |
| `exp_e3_active.py` | ❌ No creado | Registrado en `__init__.py` → ImportError |
| `exp_e4_longitudinal.py` | ❌ No creado | Registrado en `__init__.py` → ImportError |

**⚠️ PROBLEMA:** El registry (`experiments/__init__.py`) registra 16 experimentos,
pero solo 7 tienen archivo implementado. Los 9 restantes (A1, A2, B3, C1, D1, D3, E1, E3, E4)
causarán `ImportError` al intentar ejecutarlos. Esto no es bloqueante (solo falla al
invocar esos IDs específicos) pero debería limpiarse.

### 3.3 Dependencias externas necesarias

| Experimento | Dependencia | Disponible? |
|-------------|-------------|-------------|
| A2 (TCI) | `xfac` o implementación custom | ❌ Necesita instalación |
| D3 (Tensor completion) | `tensorly` o `scipy` (ALS) | ⚠️ scipy tiene `sparse`, no ALS nativo |
| F2 (Flow-VQE) | `normflows` o `nflows` | ❌ Necesita instalación |
| D2 (Attention) | `torch` (ya disponible) | ✅ |
| E3 (Active learning) | `torch` ensemble (ya disponible) | ✅ |

### 3.4 Baseline caching

El `ResultStore` tiene soporte para baselines pero no hay script que genere los baselines
V6.1 de referencia. Necesitamos un script `generate_baselines.py` que ejecute:
- N=6, h=1.5, seed=43: baseline para comparación
- N=10, h=1.5, seed=43: baseline para comparación
- N=20, h=2.0: baseline para comparación

Este script debería ser independiente (no embebido en `setup()` de cada experimento)
para evitar re-generación redundante entre experimentos.

---

## Parte 4: Mejoras de Prácticas Recomendadas

### 4.1 Logging estructurado

**Actual:** `logger.info(f"Phase 2: avg fid={...}")` — texto libre, no parseable.

**Recomendado:** Usar logging estructurado para análisis automático:
```python
logger.info("phase2_complete", extra={
    "avg_fidelity": float(np.mean(fidelities)),
    "n_valid": int(np.sum(fidelities >= threshold)),
    "elapsed_s": elapsed,
})
```

### 4.2 Reproducibilidad: Seed management centralizado

**Actual:** Seeds se setean en múltiples lugares (scripts, pipeline_core, vqe_optimizer).

**Recomendado:** Un único punto de seed management:
```python
def set_global_seed(seed: int):
    """Set seed for all RNGs used in the pipeline."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # TeNPy uses numpy's RNG, so this covers DMRG too
```

### 4.3 Validación de resultados: Sanity checks automáticos

**Actual:** Los resultados se guardan sin validación. Un ΔE/gap negativo o una fidelidad > 1.0
se guardaría sin warning.

**Recomendado:** Agregar sanity checks en `V8Metrics`:
```python
def validate(self) -> list[str]:
    issues = []
    if self.relative_error < 0:
        issues.append(f"Negative DE/gap: {self.relative_error}")
    if self.fidelity is not None and (self.fidelity < 0 or self.fidelity > 1.001):
        issues.append(f"Invalid fidelity: {self.fidelity}")
    if self.energy > 0 and self.exact_energy < 0:
        issues.append(f"Energy has wrong sign: {self.energy} vs exact {self.exact_energy}")
    return issues
```

### 4.4 Comparación automática con resultados previos

**Actual:** Cada binnacle compara manualmente con resultados anteriores.

**Recomendado:** El `ResultStore.compare_experiments()` ya hace esto. Pero necesita
integrarse en el lifecycle de `BaseExperiment`:
```python
def analyze(self, results):
    analysis = super().analyze(results)
    # Auto-compare with previous runs of the same experiment
    store = ResultStore()
    previous = store.load_all_runs(self.config.experiment_id)
    if previous:
        analysis["regression_check"] = self._check_regression(results, previous[-1])
    return analysis
```

---

## Parte 5: Resumen de Acciones

> **Actualizado 2026-05-22** con estado real del filesystem.

### Inmediatas (antes de ejecutar V8)

1. ✅ Framework V8 creado (`core/`, `techniques/`, `experiments/`, CLI)
2. ⬜ Agregar gap analítico como fallback en `ClassicalSolver` (2.2) — incluir `gap_source` para trazabilidad
3. ⬜ Agregar validación de dataset vacío en `train_mpnn()` (1.4)
4. ⬜ Agregar validación inter-fase en `pipeline_core.py` (2.4) — nota: pipeline_core es dead code, aplicar en scripts runners
5. ✅ ~~Crear `techniques/mps_backend.py`~~ — resuelto: MPS funciones disponibles en experiment scripts

### Para cada experimento (al implementar)

6. ⬜ Crear scripts faltantes: `exp_c1_physics_loss.py`, `exp_d1_weight_space.py`, `exp_e3_active.py`, `exp_e4_longitudinal.py`
7. ⬜ Crear técnicas faltantes: `orthogonal_dmrg.py`, `tci.py`, `light_cone.py`, `tensor_completion.py` (solo si se ejecutan A1/A2/B3/D3)
8. ⬜ Crear `generate_baselines.py` como script independiente
9. ⬜ Limpiar registry: remover entradas de experimentos no implementados o agregar stubs con `NotImplementedError`

### Mejoras de calidad (post-V8)

10. ⬜ Integrar sign canonicalization en `build_graph_dataset()` (2.6)
11. ⬜ Agregar convergence validation en VQE optimizer (2.1)
12. ⬜ Agregar timeout por restart en VQE usando callback de scipy (2.8)
13. ⬜ Usar `lru_cache` para StatevectorEstimator (2.5)
14. ⬜ Agregar tests unitarios para cada fix aplicado

---

## Parte 6: Testing Recomendado para Fixes

Cada fix de la Parte 2 debería tener al menos un test unitario:

| Fix | Test sugerido |
|-----|---------------|
| 2.1 (VQE convergence) | Test que `result.success=False` produce warning en logs |
| 2.2 (DMRG gap fallback) | Test que gap=0 → analytical fallback con `gap_source="analytical"` |
| 2.3 (divergence threshold) | Test que threshold configurable se respeta |
| 2.4 (inter-phase validation) | Test que dataset vacío → `ValueError` con mensaje claro |
| 2.5 (estimator cache) | Test que `_get_default_estimator()` retorna misma instancia |
| 2.6 (sign canonicalization) | Test que `canonicalize_dataset()` produce θ_x > 0 siempre |
| 2.8 (VQE timeout) | Test que callback lanza `VQETimeout` después de N segundos |

**Ubicación sugerida:** `tests/test_v8_fixes.py` o integrar en tests existentes.

**Framework:** pytest (ya configurado en `pyproject.toml`).
