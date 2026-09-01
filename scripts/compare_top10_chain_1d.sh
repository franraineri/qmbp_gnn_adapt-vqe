#!/usr/bin/env bash
#
# Comparativa de los mejores 10 modelos chain_1d (p=1) en el régimen crítico.
#
# La lista de modelos NO está hardcodeada: se arma en runtime con la API del
# model_zoo (_load_manifest + _sort_score, el mismo ranking que usa el zoo para
# seleccionar el "best"). Luego corre run_large_n_extrapolation por cada uno.
#
# Uso:
#   scripts/compare_top10_chain_1d.sh
#   TOPN=10 TARGET_N=10 H_MIN=1.0 H_MAX=1.5 H_POINTS=6 scripts/compare_top10_chain_1d.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python

# ── Parámetros (overridables por env) ────────────────────────────────────────
TOPN="${TOPN:-10}"
TOPOLOGY="${TOPOLOGY:-chain_1d}"
P_LAYERS="${P_LAYERS:-1}"
TARGET_N="${TARGET_N:-10}"      # N<=16 => fidelity exacta (statevector)
H_MIN="${H_MIN:-0.5}"
H_MAX="${H_MAX:-2.0}"
H_POINTS="${H_POINTS:-3}"

# ── Paso 1: armar la lista top-N con la funcionalidad del zoo ────────────────
# _sort_score() es el ranking canónico (training_quality_score, fallback pass_rate).
# Incluye modelos multi_topology (que también sirven para chain_1d) además de los
# específicos de la topología. Imprime un checkpoint por línea.
MODELS=$(
  "$PY" - "$TOPOLOGY" "$P_LAYERS" "$TOPN" <<'PYEOF'
import sys
from qmbp_simulation.predictors.model_zoo import _load_manifest, _sort_score, _checkpoint_available

topology, p_layers, topn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

entries = [
    e for e in _load_manifest()
    if e.p_layers == p_layers
    and e.topology in (topology, "multi_topology")
    and _checkpoint_available(e.checkpoint_file)
]
# Ranking canónico del zoo (desc). Desempate: pass_rate, luego más reciente.
entries.sort(key=lambda e: (_sort_score(e), e.pass_rate, e.created), reverse=True)

for e in entries[:topn]:
    print(e.checkpoint_file)
PYEOF
)

if [ -z "$MODELS" ]; then
  echo "No se encontraron modelos para topology=$TOPOLOGY p=$P_LAYERS" >&2
  exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "Top-$TOPN modelos $TOPOLOGY p=$P_LAYERS (ranking del zoo):"
echo "$MODELS" | nl -w2 -s'. '
echo "Config eval: N=$TARGET_N  h=[$H_MIN, $H_MAX] ($H_POINTS pts)"
echo "═══════════════════════════════════════════════════════════════"
echo

# ── Paso 2: evaluar cada modelo ──────────────────────────────────────────────
# --force-recompute es obligatorio: los runs comparten el NPZ large-N y el
# cache del run previo contaminaría las predicciones del siguiente modelo.
i=0
while IFS= read -r ckpt; do
  [ -z "$ckpt" ] && continue
  i=$((i + 1))
  # quitar sufijo .pt para el matcher fuzzy de --checkpoint
  ckpt_arg="${ckpt%.pt}"
  echo "───────────────────────────────────────────────────────────────"
  echo "[$i/$TOPN] $ckpt_arg"
  echo "───────────────────────────────────────────────────────────────"
  "$PY" scripts/experiment_runners/scaling/run_large_n_extrapolation.py \
    --topology "$TOPOLOGY" --p-layers 2 --target-n 30 \
    --h-min "$H_MIN" --h-max "$H_MAX" --h-points "$H_POINTS" \
    --skip-random-baseline --force-recompute \
    --checkpoint "$ckpt_arg" \
    # --refine-failing --vqe-maxiter 300
    2>&1 | grep -E "Model:|N=$TARGET_N: ΔE/gap|Grade|Fuzzy-matched|Evaluation report" || true
  echo
done <<< "$MODELS"

echo "═══════════════════════════════════════════════════════════════"
echo "Listo. Reportes por modelo en:"
echo "  results/extrapolation_evals/${TOPOLOGY}_p${P_LAYERS}/eval_${TOPOLOGY}_*.md"
echo "═══════════════════════════════════════════════════════════════"
