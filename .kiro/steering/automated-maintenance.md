inclusion: always

# Automated Maintenance (NO AI tokens needed)

## Post-Run Hooks (automáticos, 0 intervención)

| Hook | Trigger | Qué hace |
|------|---------|----------|
| `zoo-coherence-check` | agentStop | Diagnóstico: zoo↔dashboard↔GT coherencia |
| `validate-zoo-registration` | postTaskExecution | Previene n_qubits=10 bug |

## Auto-Integrations (built into runner_base.run())

| Qué | Cuándo | Cómo |
|-----|--------|------|
| Zoo pass_rate update | Post-run, if sections produced pass_rate_dual | `_log_data_quality_feedback` → `_extract_best_pass_rate_dual` → `auto_update_zoo_pass_rate` |
| Exclusion-policy N-level filter | During `load_best_mpnn_for_cross_n` training branch | Removes N-values with `contaminated_training`/`gap_masking` failure modes |
| Auto-detect exclusions | Post-run, fire-and-forget subprocess | `auto_detect_exclusions()` |
| Dashboard regeneration | Post-run, fire-and-forget subprocess | `generate_model_quality_dashboard()` |
| ResultIndex + project-status.md | Post-run, fire-and-forget subprocess | `ResultIndex.rebuild()` + `refresh_status()` |

## Scripts de Mantenimiento (correr manualmente, sin AI)

```bash
# Audit profundo + coherencia (subsume check_zoo_coherence checks)
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --coherence

# Aplicar fixes: tags + exclusion sync
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --sync-exclusions

# Limpieza completa (prune + archive)
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --prune-stale --archive-orphans

# Solo coherencia (rápido, hook-friendly)
.venv/bin/python scripts/maintenance/check_zoo_coherence.py

# Ver qué modelos necesitan retrain
.venv/bin/python scripts/maintenance/check_zoo_coherence.py --retrain-queue

# Evaluar modelos + actualizar zoo pass_rate
.venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo

# Regenerar coverage docs
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py

# Actualizar project-status.md
.venv/bin/python scripts/maintenance/update_project_status.py
```

## Cuándo Correr Qué

| Evento | Acción automática | Acción manual recomendada |
|--------|-------------------|---------------------------|
| Después de VQE/training | Hooks + auto pass_rate update | `check_zoo_coherence.py` |
| Antes de un retrain | — | `check_zoo_coherence.py --retrain-queue` |
| Después de agregar NPZ data | — | `evaluate_zoo_models.py --update-zoo` |
| Limpieza periódica | — | `audit_and_fix_model_zoo.py --fix --archive-orphans --sync-exclusions` |
| Exclusiones desincronizadas | — | `audit_and_fix_model_zoo.py --coherence` (detecta drift) |

## Funciones Clave (import desde src/, no duplicar)

| Necesidad | Import |
|-----------|--------|
| Evaluar calidad de NPZ | `from qmbp_simulation.analysis.metrics import get_usable_training_configs` |
| Dataset limpio para training | `from qmbp_simulation.framework.result_io import build_clean_training_dataset` |
| Validar antes de registrar | `from qmbp_simulation.predictors.model_zoo import _validate_zoo_entry` |
| Cola de retrain | `from qmbp_simulation.predictors.model_zoo import compute_retrain_queue` |
| Actualizar pass_rate post-eval | `runner.auto_update_zoo_pass_rate(pass_rate_dual)` (auto desde run()) |
| Exclusion policy filter | `from qmbp_simulation.analysis.metrics import load_training_exclusions, get_excluded_files` |
| Sync exclusions → registry | `from qmbp_simulation.analysis.metrics import auto_detect_exclusions` |
| Zoo integrity check | `from qmbp_simulation.predictors.model_zoo import validate_zoo` |
