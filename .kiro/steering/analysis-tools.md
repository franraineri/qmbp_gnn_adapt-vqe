---
inclusion: fileMatch
fileMatchPattern: "**/analysis/**,**/metrics*,**/quality*,**/model_zoo*,**/model_registry*"
---

# Analysis Tools — Guía de Uso

Cuándo y cómo usar las herramientas de análisis, calidad y ModelRegistryDB.
Para listado completo de funciones/clases → ver `module-index.md`.

---

## Cuándo Usar Cada Herramienta

| Pregunta | Módulo | Función/CLI |
|----------|--------|-------------|
| ¿Punto es failure? | `analysis.metrics` | `is_point_failure()` |
| ¿Qué refinar primero? | `analysis.metrics` | `compute_refinement_priority()` |
| ¿NPZ útil para training? | `analysis.metrics` | `classify_training_utility()` |
| ¿Puedo entrenar ahora? | `analysis.metrics` | `compute_training_readiness()` |
| ¿Data multi-N viable? | `analysis.metrics` | `validate_training_dataset()` |
| ¿Escala bien? | `analysis.metrics` | `compute_scalability_score()` |
| Dashboard completo | `analysis.metrics` | `generate_model_quality_dashboard()` |
| Persistir θ con calidad | `framework.result_io` | `upsert_theta_npz(..., quality_tier_new=[...])` |
| ¿Mejor modelo para N=X? | `predictors.model_zoo` | `load_best_for_cross_n(reject_contaminated=True)` |
| ¿Modelo zoo confiable? | `predictors.model_zoo` | `load_best_for_cross_n_quality_aware()` |
| ¿Qué modelos tengo? | `predictors.model_registry_db` | `ModelRegistryDB.query()` / CLI `list` |
| ¿Modelo tiene problemas? | `predictors.model_registry_db` | `db.run_failure_diagnostics()` / CLI `diagnose` |
| ¿Modelo contaminado? | `predictors.model_registry_db` | `db.get_models_by_failure_mode("contaminated_training")` |
| ¿Checkpoint intacto? | `predictors.model_registry_db` | `db.validate_integrity()` / CLI `validate` |
| ¿Hubo regresiones? | `predictors.model_registry_db` | `db.detect_regressions()` / CLI `regressions` |
| Health completo | `predictors.model_registry_db` | `db.get_comprehensive_health()` / CLI `comprehensive-health` |
| ¿Training data cambió? | `predictors.model_registry_db` | `db.check_training_data_changed()` |
| ¿Qué modelos reentrenar? | `predictors.model_registry_db` | `db.get_models_needing_retrain()` |
| Registro post-training | `predictors.model_zoo` | `register_checkpoint_with_training_metrics(..., auto_diagnose=True)` |

---

## Dual Criterion (Definición central)

```python
from qmbp_simulation.analysis.metrics import is_point_failure
# Punto es FAILURE si: ΔE/gap ≥ 5% OR |ΔE| ≥ 0.10 OR fidelity < 0.97
```

---

## Model Zoo + Registry: Flujo Post-Training

Tras `train_unified_mpnn()`, usar **una sola función** que hace todo:

```python
from qmbp_simulation.predictors.model_zoo import register_checkpoint_with_training_metrics

path = register_checkpoint_with_training_metrics(
    model, entry, training_result=train_result,
    architecture_config={"hidden_dim": 64, "n_conv_layers": 3, "n_heads": 1},
    optimizer_config={"learning_rate": 1e-3, "weight_decay": 1e-4},
    auto_diagnose=True,       # Corre failure diagnostics automáticamente
    auto_sync_dashboard=True, # Sincroniza dashboard quality
    overwrite=True,
)
```

Esto automáticamente:
1. Guarda checkpoint + metadata en zoo manifest
2. Registra en ModelRegistryDB con métricas completas (MSE, epochs, stop_reason)
3. Computa y almacena `training_data_hash` (detecta staleness futuro)
4. Corre failure diagnostics → auto-tag (clean/contaminated/gap-masked/...)
5. Sincroniza dashboard quality → needs_retrain, training_utility

---

## ModelRegistryDB: Patrones Clave

```python
from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

db = ModelRegistryDB()

# CRUD
models = db.query(topology="chain_1d", min_training_points=50)
record = db.get_model("unified_tfim_br_chain_1d_multiN_6+8+10+12+20_p1.pt")

# Failure diagnostics (auto-runs on registration, or manually)
diag = db.run_failure_diagnostics("model.pt", force=True)
# diag.primary_mode: "healthy" | "contaminated_training" | "gap_masking" | ...

# Staleness detection
changed, reason = db.check_training_data_changed("model.pt")
stale = db.detect_stale_models()  # Marca needs_retrain + evento training_data_changed

# Health
health = db.get_comprehensive_health("model.pt")
# health["status"]: "healthy" | "warning" | "critical"
# health["recommendation"]: "deploy" | "investigate" | "retrain" | "do_not_use"

# Retrain queue
needing = db.get_models_needing_retrain()  # Ordenados: contaminated > stale > other

# Sync from zoo manifest (para modelos legacy)
db.sync_from_manifest()
db.enrich_points_per_n()
db.enrich_from_dashboard()
```

---

## ModelRegistryDB: Dataclasses

| Clase | Campos clave |
|-------|-------------|
| `ModelRecord` | model_id, topology, training, evaluations, tags, status, version |
| `TrainingProvenance` | n_values_used, training_data_hash, architecture_config, optimizer_config, training_metrics |
| `TrainingMetrics` | final_mse, final_val_mse, epochs, stop_reason, convergence_status, weight_distribution |
| `ModelArchitectureConfig` | hidden_dim, n_conv_layers, n_heads, dropout, activation |
| `OptimizerConfig` | learning_rate, weight_decay, scheduler, scheduler_patience, layerwise_lr |
| `FailureDiagnosticSummary` | primary_mode, confidence, contamination_severity, gap_masked_fraction |

---

## Failure Modes & Auto-Tags

| Mode | Auto-tag | Acción |
|------|----------|--------|
| `healthy` | `clean` | Deploy |
| `gap_masking` | `gap-masked` | Investigate (métricas infladas) |
| `contaminated_training` | `contaminated` | Retrain (reject en load_best_for_cross_n) |
| `intrinsic_vqe_error` | `ansatz-limited` | Retrain con más p_layers |
| `generalization_failure` | `cross-n-degraded` | Retrain con más N values |

---

## History Events

| Evento | Trigger |
|--------|---------|
| `registered` | Modelo nuevo en registry |
| `retrained` | Overwrite con nueva versión |
| `evaluated` | add_evaluation() |
| `regression_detected` | pass_rate bajó >5% vs prev evaluation |
| `training_data_changed` | NPZ hash difiere del almacenado |
| `auto_retrain_triggered` | Retrain automático disparado |
| `quality_degraded` | Métricas empeoraron sin retrain |
| `failure_diagnosed` | Diagnóstico ejecutado |
| `needs_retrain_flagged/cleared` | Flag set/cleared |

---

## CLI: query_model_registry.py

```bash
# Uso general
python scripts/maintenance/query_model_registry.py <subcommand> [options]

# Subcommands principales:
list [--topology T] [--json]     # Listar modelos
get "pattern*"                   # Info detallada
summary                          # Estadísticas
sync                             # Sync desde zoo manifest
diagnose <model_id> [--force]    # Failure diagnostics
diagnostics [--topology T]       # Batch diagnostics
comprehensive-health <model_id>  # Health completo
health-dashboard                 # Dashboard global
best -t <topo> -n <N>           # Mejor modelo para deployment
validate [<model_id>]           # Integridad
regressions                      # Detectar regresiones
history [--model-id M] [--event-type E]
tag <model_id> --add/--remove TAG
versions [--topology T]
```

---

## MPNN Diagnostics Consolidados (`analysis/metrics.py`)

```python
from qmbp_simulation.analysis.metrics import compute_mpnn_diagnostics

diag = compute_mpnn_diagnostics(
    mpnn_results_by_n={10: {...}, 20: {...}},
    topology="chain_1d", model_name="tfim_bond_resolved", p_layers=1,
)
# diag["summary"]["overall_health"]: "healthy" | "investigate" | "retrain"
# Incluye: theta_smoothness, variational_violations, scaling_fit, training_data_quality
```

---

## Scripts de Mantenimiento

| Script | Uso |
|--------|-----|
| `query_model_registry.py` | CLI completo ModelRegistryDB |
| `update_cross_n_coverage.py` | Actualiza coverage con quality tiers |
| `generate_scaling_report.py` | Reporte escalabilidad → JSON |
| `upgrade_npz_quality_tiers.py` | Agrega quality_tier a NPZ legacy |
| `run_full_validation.py` | Validación completa (7 pasos) |
| `quick_health_check.py` | Check rápido: zoo, NPZ, imports |
| `inspect_data_stores.py` | Inventario: GT cache, NPZ, zoo |
