#!/usr/bin/env bash
# heavy_hex critical window: h=1.0..2.0, 10 h-points, N=10 20 30.
# Random VQE baseline ON (default). --force-recompute so GT uses the
# robustened DMRG instead of inheriting stale gap-floor cache values.
# No `set -e`: a failing model must not abort the remaining evaluations.
set -uo pipefail

PY=.venv/bin/python
RUNNER=scripts/experiment_runners/scaling/run_large_n_extrapolation.py
CKPT_DIR=data/model_zoo/checkpoints

COMMON="--topology heavy_hex --model-name tfim_bond_resolved --p-layers 1 \
  --h-min 1.0 --h-max 2.5 --h-points 15 --target-n 8 10 20 --force-recompute --skip-random-baseline"

MODELS=(
  "unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt"
  "unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt"
  "unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt"
  "unified_tfim_br_MT_residual+film_p1.pt"
  "unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1_v1.pt"
  "unifMPNN__heavy_hex_p1_res_mse.pt"
  "unifMPNN__heavy_hex_p1_res_film_mse.pt"
  "unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18+20+21+26+30+40_p1.pt"
  "unified_multiN_heavyhex_p1.pt"
  "unifMPNN__heavy_hex_p1_res_mse_v2.pt"
  "unifMPNN__heavy_hex_p1_res_film_mse_v2.pt"
  "unifMPNN__heavy_hex_p1_plain_energyw.pt"
  "unifMPNN__heavy_hex_p1_res_energyw.pt"
  "unifMPNN__heavy_hex_p1_res_film_energyw.pt"
  "unifMPNN__heavy_hex_p1_res_physics05.pt"
  "unifMPNN__heavy_hex_p1_full_stack.pt"
  "unifMPNN__heavy_hex_p1_res_film_energyw_critical.pt"
  "unifMPNN__heavy_hex_p1_res_film_deploy.pt"
  "unifMPNN__MT_p1_res_film_base.pt"
)

n_ok=0; n_fail=0
for m in "${MODELS[@]}"; do
  echo "=== Evaluating $m ==="
  if $PY $RUNNER $COMMON --checkpoint "$CKPT_DIR/$m"; then
    n_ok=$((n_ok+1))
  else
    echo "!!! FAILED: $m (exit $?) - continuing"
    n_fail=$((n_fail+1))
  fi
done
echo "=== DONE: $n_ok ok, $n_fail failed of ${#MODELS[@]} ==="
