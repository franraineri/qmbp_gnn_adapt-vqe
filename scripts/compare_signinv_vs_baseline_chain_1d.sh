#!/usr/bin/env bash
#
# A/B directo: nuevo modelo (loss Z2 sign_invariant) vs baseline chain_1d p=1.
#
# Usa run_model_comparison.py sobre DOS checkpoints explícitos, en condiciones
# identicas, con un h-grid denso (30 pts por defecto) para reducir el ruido
# estadistico del run previo (que solo tenia 3 pts -> CI enormes).
#
# Uso:
#   scripts/compare_signinv_vs_baseline_chain_1d.sh
#   H_POINTS=40 TARGET_N="8 10 12 16 20" scripts/compare_signinv_vs_baseline_chain_1d.sh
#   NEW_CKPT=... BASE_CKPT=... scripts/compare_signinv_vs_baseline_chain_1d.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
CKPT_DIR=data/model_zoo/checkpoints

# ── Parametros (overridables por env) ────────────────────────────────────────
TOPOLOGY="${TOPOLOGY:-chain_1d}"
TARGET_N="${TARGET_N:-8 10 12 16 20}"
H_MIN="${H_MIN:-1.0}"
H_MAX="${H_MAX:-3.0}"
H_POINTS="${H_POINTS:-20}"

NEW_CKPT="${NEW_CKPT:-$CKPT_DIR/unifMPNN__chain_1d_p1_signinv_v2.pt}"
BASE_CKPT="${BASE_CKPT:-$CKPT_DIR/unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt}"

# ── Validacion de existencia (falla temprano con mensaje claro) ──────────────
for f in "$NEW_CKPT" "$BASE_CKPT"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint no encontrado: $f" >&2
    exit 1
  fi
done

echo "═══════════════════════════════════════════════════════════════"
echo "A/B sign_invariant vs baseline  |  $TOPOLOGY p=1"
echo "  NEW : $(basename "$NEW_CKPT")"
echo "  BASE: $(basename "$BASE_CKPT")"
echo "  Eval: N=[$TARGET_N]  h=[$H_MIN, $H_MAX] ($H_POINTS pts)"
echo "═══════════════════════════════════════════════════════════════"
echo

# shellcheck disable=SC2086  # TARGET_N debe expandirse en multiples argumentos
"$PY" scripts/experiment_runners/cross_topology/run_model_comparison.py \
  --topology "$TOPOLOGY" --target-n $TARGET_N \
  --checkpoints "$NEW_CKPT" "$BASE_CKPT" \
  --h-min "$H_MIN" --h-max "$H_MAX" --h-points "$H_POINTS" \
  --save-report 

echo
echo "═══════════════════════════════════════════════════════════════"
echo "Listo. Reportes por modelo en:"
echo "  results/model_comparison/${TOPOLOGY}_p1/evaluation_${TOPOLOGY}_p1_*.md"
echo "Envelope JSON en:"
echo "  results/experiments/exp_model_comparison/tfim_bond_resolved/${TOPOLOGY}/run_*.json"
echo "═══════════════════════════════════════════════════════════════"
