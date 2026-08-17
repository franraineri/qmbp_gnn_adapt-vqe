inclusion: always

# Reuse-First Workflow (MANDATORY)

## Context-Efficient Code Reading (MANDATORY)

0. usar MCP tool context-mode / ctx_batch_execute
1. **Consultar `module-index.md` primero** — Ya tiene firmas de todas las funciones/clases
2. **Usar `read_code`** — Para archivos >200 líneas, obtiene solo firmas (AST)
   - `read_code path="archivo.py"` → firmas si archivo es grande
   - `read_code path="archivo.py" selector="mi_funcion"` → solo esa función
   - `read_code path="archivo.py" selector="Clase.metodo"` → método específico
3. **Usar `grep_search`** — Localizar antes de leer
   - `grep_search query="def nombre_funcion"` → encuentra ubicación exacta
4. **Usar `read_file` con rangos** — Solo si necesitás líneas específicas
   - `read_file path="archivo.py" start_line=X end_line=Y`

**Anti-patterns:**
- ❌ Leer archivo completo de 500+ líneas innecesariamente
- ❌ Usar `read_file` sin rangos
- ❌ No consultar module-index

---

## CRITICAL RULE: Implement in the most general module first

When adding ANY new functionality (helper, utility, pattern, fix):

1. **Ask**: "Could more than one runner/script benefit from this?"
2. **If YES** → implement in the appropriate shared module:
   - Runner patterns → `src/qmbp_simulation/framework/runner_base.py` (ValidationRunner methods)
   - CLI args → `src/qmbp_simulation/framework/cli.py`
   - Analysis utilities → `src/qmbp_simulation/analysis/metrics.py`
   - Model persistence → `src/qmbp_simulation/predictors/model_zoo.py`
   - Physics helpers → appropriate `src/qmbp_simulation/` subpackage
3. **If NO** → implement in the specific runner, but keep it extractable (no inline lambdas, clear interface)

Every time new Python code is about to be created (script, helper, module, plotter, analyzer):

## Step A — Check Index
1. Read `.kiro/steering/module-index.md`
2. Search for existing modules with similar functionality by:
   - Matching class/function names
   - Matching docstring descriptions
   - Matching category (LIB, SCRIPT, HEALTH, EXP, etc.)
3. If a match exists → prefer extending it (add function, subclass, new flag)

## Step B — Implement with Reuse
- Import from existing modules rather than copy-pasting code
- If extending: add to the same file or create a minimal subclass
- If truly novel: create new file following existing patterns in same category

## Step C — Update Index
- After implementation, run: `python scripts/general_project_maintenance/generate_module_index.py`
- This is automated via the `refresh-module-index` hook on fileCreated events

## Step D — Verify Integration
- Run relevant tests or at minimum `python -c "import <new_module>"`
- Check no circular imports introduced

## Anti-patterns (NEVER do these)
- Creating a new script when an existing one accepts `--flags` to do the same
- Duplicating utility functions that exist in `qmbp_simulation.utils`
- Creating analysis helpers that replicate `project_health/analysis/` functionality
- Writing JSON serialization logic (use `json_serialize` from utils)
- Duplicating experiment criteria (use `framework/criteria.py`)

## Runner Reuse Patterns (MANDATORY for experiment runners)

When writing `section_*()` methods in any `ValidationRunner` subclass:

### Ground Truth Access
```python
# ✅ CORRECT — uses 2-level cache (in-memory + disk-persistent GroundTruthCache)
e_exact, gap = self.exact_ground_state(topology, n_qubits, h, model="tfim")

# ❌ WRONG — manual GroundTruthCache.get/put
gt_cache = GroundTruthCache()
cached = gt_cache.get(topo, n, model, h)
```

### Circuit Evaluation with Cache
```python
# ✅ CORRECT — CachedBackend wraps backend + EvalCache transparently
eval_backend = self.get_cached_backend(topology=topo, n_qubits=N, model="tfim", p_layers=p)
eval_backend.set_h(h)
energy = eval_backend.evaluate(circuit, H, theta)

# ❌ WRONG — manual EvalCache.make_key/get/put
eval_cache = EvalCache()
key = eval_cache.make_key(topo, N, h, theta)
cached = eval_cache.get(key)
```

### Backend Selection by System Size
```python
# ✅ CORRECT — auto-selects Statevector (N≤22) or MPS (N>22)
backend = self.select_backend(n_qubits)
# For VQE loops (stricter threshold):
backend = self.select_backend(n_qubits, for_vqe_loop=True)

# ❌ WRONG — inline if/else
if n > STATEVECTOR_MAX_N:
    backend = MPSBackend(chi_max=64)
```

### MPNN Model Persistence (via model_zoo)
```python
# ✅ CORRECT — load from zoo (auto-detects model/topo/N/p from self._args)
mpnn = self.load_mpnn_from_zoo()
# With explicit params or cross-N:
mpnn = self.load_mpnn_from_zoo(model="tfim", n_qubits=10, allow_cross_n=True)

# ✅ CORRECT — save to zoo after training
self.save_mpnn_to_zoo(predictor, n_training_points=len(h_values), notes=f"mse={mse:.2e}")
```

### VQE Checkpoint Resume
```python
# ✅ CORRECT — typed contract with param validation
checkpoint = self.load_vqe_checkpoint(topology, n_params=circuit.num_parameters)
if checkpoint is not None:
    results, prev_theta = checkpoint

# ❌ WRONG — manual load_checkpoint + inline key access
cp = self.load_checkpoint(f"vqe_{topology}")
results = cp["results"]
theta = np.array(cp["current_theta"])
```

### Fidelity Computation (safe, N-guarded)
```python
# ✅ CORRECT — handles N-check, solver call, and errors automatically
fidelity = self.safe_compute_fidelity(circuit, theta, topology, n_qubits, h, model="tfim")

# ✅ ALSO CORRECT — when you already have the ground state vector
fidelity = self.compute_fidelity(circuit, theta, exact_ground_state_vector)
```

### Per-H Result Dict (standardized for compute_deploy_summary)
```python
# ✅ CORRECT — ensures all required keys, float-typed, ΔE/gap computed
result = self.build_per_h_result(h, e_pred, e_exact, gap, fidelity=fid, method="warm")

# ❌ WRONG — manual dict construction with inconsistent key names
results.append({"h_test": h, "de_gap": de_gap, ...})  # "h_test" vs "h" inconsistency
```

### Per-H Deployment Summary Statistics
```python
# ✅ CORRECT — reusable utility for pass rates, mean ΔE/gap, fidelity stats
from qmbp_simulation.analysis.metrics import compute_deploy_summary
summary = compute_deploy_summary(per_h_results)
# Returns: n_points, pass_rate_5pct, pass_rate_10pct, mean_de_gap, mean_fidelity, etc.

# ❌ WRONG — manual aggregation
n_pass = sum(1 for r in results if r["de_gap"] < 0.05)
pass_rate = n_pass / len(results)
mean_fidelity = np.mean([r["fidelity"] for r in results if r["fidelity"]])
```

### H-Grid Generation
```python
# ✅ CORRECT — non-uniform (dense near h_critical)
self._h_values = self.generate_h_grid()

# ✅ CORRECT — uniform (for bond-resolved or when h_critical unknown)
self._h_values = self.generate_h_grid(uniform=True)
```

### Quality Check
```python
# ✅ CORRECT — uses base helper with auto-detect or explicit configs
qc = self.run_quality_check()
qc = self.run_quality_check(configs=[{"model": "tfim_bond_resolved", ...}])

# ❌ WRONG — inline QualityPredictor instantiation
from qmbp_simulation.analysis.quality_predictor import QualityPredictor
predictor = QualityPredictor()
report = predictor.predict(...)
```

### Per-H Result Dict Structure (for compute_deploy_summary compatibility)
Each per-h result dict MUST contain at minimum `"de_gap"` (float).
Optional enrichment: `"abs_error"`, `"fidelity"`, `"e_pred"`, `"e_exact"`, `"gap"`.

## Data Persistence Patterns (CRITICAL for data integrity)

### CachedBackend with Context Manager (auto-flush)
```python
# ✅ CORRECT — cache automatically flushed on exit, even on exception
with self.get_cached_backend(topology=topo, n_qubits=N, model="tfim", p_layers=p) as eval_backend:
    eval_backend.set_h(h)
    energy = eval_backend.evaluate(circuit, H, theta)
# Cache flushed here automatically

# ❌ WRONG — manual flush (easy to forget, lost on exception)
eval_backend = self.get_cached_backend(...)
energy = eval_backend.evaluate(circuit, H, theta)
eval_backend.flush()  # Can be skipped if exception occurs
```

### Immediate Persistence of Computed Data
```python
# ✅ CORRECT — persist as soon as data is computed
for h in h_values:
    theta_opt, e_vqe = run_vqe(h)
    _upsert_npz(npz_path, h, theta_opt, e_vqe, ...)  # Immediate persist

# ❌ WRONG — accumulate in memory, persist at end
results = []
for h in h_values:
    theta_opt, e_vqe = run_vqe(h)
    results.append((h, theta_opt, e_vqe))  # Lost if process crashes
np.savez(npz_path, ...)  # Only saved at end
```

### Ground Truth Cache Flush After Computation
```python
# ✅ CORRECT — flush GT cache immediately after computing new values
gt_cache = GroundTruthCache()
for h in h_values_missing:
    e, gap = compute_ground_truth(h)
    gt_cache.put(topo, n, model, h, e, gap)
gt_cache.flush()  
```

### NPZ Anti-Regression Pattern
```python
# ✅ CORRECT — only update if new energy is better
def _upsert_npz(npz_path, h_new, theta_new, e_vqe_new, ...):
    if npz_path.exists():
        existing = np.load(npz_path)
        # Only replace if e_vqe_new < e_existing
        ...
    np.savez(npz_path, ...)
```
