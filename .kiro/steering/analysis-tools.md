---
inclusion: always
---

# Analysis Tools — Cross-Integration Utilities

Herramientas de análisis, calidad y escalabilidad. Usar siempre que se trabaje con datos NPZ, entrenamiento MPNN, extrapolación, o validación de pipeline.

---

## 1. Quality Tier System (`analysis/metrics.py`)

### is_point_failure(de_gap, abs_error, fidelity)
Dual criterion: ΔE/gap < 5% AND |ΔE| < 0.10 AND fidelity > 0.97.
```python
from qmbp_simulation.analysis.metrics import is_point_failure
fail = is_point_failure(de_gap=0.06, abs_error=0.08)  # True (de_gap > 5%)
```

### identify_failures(per_h_results)
Bulk failure detection sobre lista de resultados.
```python
from qmbp_simulation.analysis.metrics import identify_failures
failures = identify_failures(per_h_results)  # [0, 3, 7] indices
```

### compute_refinement_priority(de_gap, abs_error, gap, n_params, ...)
Ordena fallos por prioridad para refinamiento VQE. Evita gastar compute en puntos sin esperanza.
```python
from qmbp_simulation.analysis.metrics import compute_refinement_priority
priority, should_skip, reason = compute_refinement_priority(
    de_gap=0.06, abs_error=0.12, gap=2.0, n_params=30, n_prev_attempts=1
)
```

### classify_training_utility(n_points, pass_rate_dual, pass_rate_5pct)
Clasifica NPZ en: "useful", "insufficient_signal", "not_useful".
```python
from qmbp_simulation.analysis.metrics import classify_training_utility
category, reason = classify_training_utility(n_points=40, pass_rate_dual=0.55, pass_rate_5pct=0.70)
```

### validate_training_dataset(per_n_points, max_de_gap, min_total_points, min_n_values)
Validación pre-training: verifica que data multi-N sea viable.
```python
from qmbp_simulation.analysis.metrics import validate_training_dataset
viable, report = validate_training_dataset(agg._data_by_n, min_total_points=10)
```

---

## 2. Scalability & Extrapolation (`analysis/metrics.py`)

### compute_scalability_score(topology, n_max_viable, pass_rate_dual, h_frontier)
Score 0-1 de qué tan bien escala el pipeline para una topología.
```python
from qmbp_simulation.analysis.metrics import compute_scalability_score
score, reason = compute_scalability_score("chain_1d", 20, 1.0, 3.04)
```

### compute_extrapolation_viability(topology, n_max_viable, mean_de_gap_per_n, target_n)
Predice si extrapolación a target_n funcionará.
```python
from qmbp_simulation.analysis.metrics import compute_extrapolation_viability
viable, reason, prediction = compute_extrapolation_viability(
    "chain_1d", n_max_viable=20, mean_de_gap_per_n={6: 0.01, 10: 0.02}, target_n=30
)
```

### compute_training_readiness(tier_breakdown, utility_partition)
¿Los datos están listos para entrenar MPNN?
```python
from qmbp_simulation.analysis.metrics import compute_training_readiness, get_usable_training_configs
partition = get_usable_training_configs(dashboard)
ready, reason, stats = compute_training_readiness(tier_breakdown, partition)
```

### generate_model_quality_dashboard(output_path)
Genera dashboard completo desde todos los NPZ.
```python
from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard
dashboard = generate_model_quality_dashboard()  # → data/model_quality_dashboard.json
```

### generate_unified_scaling_report(dashboard, tier_breakdown, target_n_values)
Reporte unificado de escalabilidad.
```python
from qmbp_simulation.analysis.metrics import generate_unified_scaling_report
report = generate_unified_scaling_report(dashboard, tier_breakdown=tb, target_n_values=[30, 40])
```

---

## 3. Model Zoo (`predictors/model_zoo.py`)

### get_runner_tag(runner_id) / make_date_tag()
Tags de trazabilidad: 2 letras runner + DDMMYY fecha.
```python
from qmbp_simulation.predictors.model_zoo import get_runner_tag, make_date_tag
tag = get_runner_tag("accelerated_cross_n_v1")  # "AC"
date = make_date_tag()  # "100826"
```

### load_best_for_cross_n(model, topology, n_target, p_layers)
Carga el mejor modelo para predecir a N=n_target.
```python
from qmbp_simulation.predictors.model_zoo import load_best_for_cross_n
model, meta = load_best_for_cross_n("tfim_bond_resolved", "chain_1d", n_target=20, p_layers=1)
```

### load_best_for_cross_n_quality_aware(model, topology, n_target, p_layers)
Igual que arriba pero incluye reporte de calidad de datos de entrenamiento.
```python
from qmbp_simulation.predictors.model_zoo import load_best_for_cross_n_quality_aware
model, meta, quality = load_best_for_cross_n_quality_aware(
    "tfim_bond_resolved", "chain_1d", n_target=20, p_layers=1
)
# quality: {"verified_ratio": 0.52, "quality_score": 0.84, "warnings": [...]}
```

### get_training_data_quality(topology, n_qubits, model, p_layers)
Calidad de datos NPZ para un modelo específico.
```python
from qmbp_simulation.predictors.model_zoo import get_training_data_quality
quality = get_training_data_quality("chain_1d", 10, "tfim_bond_resolved", 1)
```

### register_checkpoint(model, entry, overwrite=False)
Registra modelo en zoo con metadata completa (runner_tag, date_tag incluidos).
```python
from qmbp_simulation.predictors.model_zoo import register_checkpoint, ZooEntry
entry = ZooEntry(model="tfim_bond_resolved", topology="chain_1d", n_qubits=0, p_layers=1,
    checkpoint_file="unified_...", h_range=(2.0, 5.0), pass_rate=0.95,
    n_training_points=50, seeds=[42], created="...",
    runner_tag=get_runner_tag(self.runner_id), date_tag=make_date_tag())
register_checkpoint(model, entry, overwrite=True)
```

### validate_zoo()
Integridad completa: SHA256 checksums, orphans, manifest consistency.
```python
from qmbp_simulation.predictors.model_zoo import validate_zoo
report = validate_zoo()  # {"n_corrupted": 0, "n_missing": 1, ...}
```

---

## 4. Data Persistence (`framework/result_io.py`)

### upsert_theta_npz(npz_path, h_new, theta_new, e_vqe_new, e_exact_new, gaps_new, method_new, quality_tier_new)
Atomic write con anti-regression. SIEMPRE usar con `quality_tier_new`.
```python
from qmbp_simulation.framework.result_io import upsert_theta_npz
n_upd, n_add = upsert_theta_npz(
    npz_path, h_new=h_arr, theta_new=theta_arr,
    e_vqe_new=e_pred, e_exact_new=e_exact, gaps_new=gaps,
    method_new=["vqe_refined", "mpnn_pred"],
    quality_tier_new=["verified", "approximate"],
)
```

**Tier assignment pattern**:
```python
from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR
tier = "verified" if method in ("vqe_refined",) and de_gap < DE_GAP_THRESHOLD and abs_err < MAX_ABS_ERROR \
    else "approximate" if de_gap < DE_GAP_THRESHOLD and abs_err < MAX_ABS_ERROR \
    else "unverified"
```

### load_theta_npz(npz_path)
Load con validación (no NaN/Inf).
```python
from qmbp_simulation.framework.result_io import load_theta_npz
data = load_theta_npz(npz_path)  # {"h_values", "theta_opt", "e_vqe", "e_exact", "gaps", ...}
```

---

## 5. ValidationRunner Methods (`framework/runner_base.py`)

### self.get_npz_quality_tiers(npz_path)
```python
stats = ValidationRunner.get_npz_quality_tiers(npz_path)
# {"n_verified": 30, "n_approximate": 15, "n_unverified": 5, "quality_score": 0.84}
```

### self.load_best_mpnn_for_cross_n(n_target, model, topology, p_layers, ...)
Carga mejor modelo con selección jerárquica: multi-N → single-N → cross-N.
```python
model = self.load_best_mpnn_for_cross_n(
    n_target=20, model="tfim_bond_resolved", topology="chain_1d", p_layers=1
)
```

---

## 6. Runners Principales

### run_accelerated_cross_n.py (bond_resolved/)
Pipeline completo: train → multi-N train → cross-N predict → iterative improve.
```bash
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py \
    --topology chain_1d --train-n 6 --target-n 10 12 --mode iterative --force-retrain
```

### run_large_n_extrapolation.py (scaling/)
Extrapolación zero-shot a N=30-100.
```bash
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology chain_1d --target-n 30 40 60 --h-min 2.5 --h-max 5.0
```

---

## 7. Scripts de Mantenimiento

| Script | Uso |
|--------|-----|
| `update_cross_n_coverage.py` | Actualiza coverage doc con quality tier breakdown |
| `generate_scaling_report.py` | Reporte unificado escalabilidad → JSON |
| `upgrade_npz_quality_tiers.py` | Agrega quality_tier a NPZ legacy |
| `run_full_validation.py` | Validación completa pipeline (7 pasos) |
| `quick_health_check.py` | Check rápido: zoo, NPZ, imports |
| `inspect_data_stores.py` | Inventario: GT cache, NPZ, zoo, eval_cache |

---

## 8. Cuándo Usar Cada Herramienta

| Pregunta | Función/Script |
|----------|----------------|
| ¿Este punto es failure? | `is_point_failure()` |
| ¿Qué puntos refinar primero? | `compute_refinement_priority()` |
| ¿Mi NPZ es útil para training? | `classify_training_utility()` |
| ¿Puedo entrenar ahora? | `compute_training_readiness()` |
| ¿Data multi-N es viable? | `validate_training_dataset()` |
| ¿Escala bien esta topología? | `compute_scalability_score()` |
| ¿Funcionará extrapolación N=X? | `compute_extrapolation_viability()` |
| ¿El modelo zoo es confiable? | `load_best_for_cross_n_quality_aware()` |
| ¿Mis NPZ tienen tiers? | `get_npz_quality_tiers()` / `upgrade_npz_quality_tiers.py` |
| Reporte completo de estado | `generate_model_quality_dashboard()` |
| Persistir θ con calidad | `upsert_theta_npz(..., quality_tier_new=[...])` |
