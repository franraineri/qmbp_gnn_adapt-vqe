#!/usr/bin/env bash
#
# Ver los mejores modelos predictores de angulos HVA en la ventana critica
# h=[0.8,1.8], usando el critical_ranking del model_zoo (|dE| + fidelity).
#
# Todos los comandos estan verificados y funcionando.
#
# Uso:
#   ./show_critical_models.sh              # corre todo
#   ./show_critical_models.sh 3            # corre solo el comando N (1..5)
#
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

ONLY="${1:-all}"

run() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

# ── 1. Mejor modelo (seleccion automatica por ranking empirico) ──────────────
if run 1; then
echo "═══ 1. Mejor modelo (load_best_model_for h_regime=critical) ═══"
"$PY" -c "
from qmbp_simulation.predictors.model_zoo import load_best_model_for, _critical_window_key
model, entry, source = load_best_model_for('chain_1d', model='tfim_bond_resolved', p_layers=1, h_regime='critical')
crit = entry.critical_ranking.get(_critical_window_key(), {})
ae, fid = crit.get('abs_error_mean'), crit.get('fidelity_mean')
print('MEJOR:', entry.checkpoint_file)
print(f'  |dE|_mean={ae:.3f}  fidelity_mean={fid}  grade={crit.get(\"grade\")}  source={source}')
"
echo
fi

# ── 2 + 3. Ranking completo + tabla por-N en h=1.0 (script dedicado) ─────────
if run 2 || run 3; then
echo "═══ 2+3. Ranking critico + |dE|/fidelity por N en h=1.0 ═══"
"$PY" scripts/maintenance/show_critical_models.py --p 1 --n 10 20 30 --h 1.0
echo
fi

# ── 4. Refrescar el ranking + rellenar fidelidades faltantes ─────────────────
if run 4; then
echo "═══ 4. Refresh ranking + backfill de fidelidades ═══"
"$PY" -c "from qmbp_simulation.predictors.model_zoo import backfill_critical_ranking_from_evals; print('ranking updated:', backfill_critical_ranking_from_evals())"
"$PY" -c "from qmbp_simulation.predictors.model_zoo import backfill_missing_fidelities; print('fidelity backfill:', backfill_missing_fidelities())"
echo
fi

# ── 5. Validar coherencia (detectar drift ranking-empirico vs metadata) ──────
if run 5; then
echo "═══ 5. Validacion de coherencia (drift) ═══"
"$PY" -c "
from qmbp_simulation.analysis.metrics import validate_data_consistency
r = validate_data_consistency()
issues = r.get('critical_ranking_issues', [])
print(f'{len(issues)} critical_ranking drift issue(s)')
for i in issues:
    print(' DRIFT:', i['checkpoint'][:45], '->', i['issue'][:80])
"
echo
fi
