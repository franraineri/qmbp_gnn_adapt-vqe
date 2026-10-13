#!/usr/bin/env bash
# N=20 h=0.5 follow-up prune sweep (serial, background-friendly).
# Runs AFTER the earlier refinement finished. Two stages, 3 restarts each:
#   1) p3_half_nn_rx re-pruned at tol 0.2 / 0.25 / 0.3 (the full reference is
#      converged ONCE as the shared high-fidelity donor, then each tol prunes
#      from it) — more aggressive bond removal, keep only the strategic ones.
#   2) diag_axial50 top-8 nnn (reduce the geometric hybrid to its 8 strongest
#      nnn bonds by donor |theta|).
# target_fid + ftol=1e-9 keep each converge out of the flat degenerate-manifold
# refinement trap (gap ~1e-5) so points close in minutes, not hours.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

echo "[pruneq] stage 1: p3_half_nn_rx prune 0.2 / 0.25 / 0.3 (restarts=3)"
$PY -u scripts/analysis/vl_vs_hva/run_variant_topk.py \
  --n 20 --h 0.5 --variant p3_half_nn_rx \
  --prune-tols 0.2 0.25 0.3 \
  --two-pass --micro-descent -1 \
  --restarts 3 --maxiter 400 --ftol 1e-9 --target-fid 0.98 --no-sync \
  > "$LOGDIR/N20_followup_p3halfnnrx_prune_025_03.log" 2>&1
echo "[pruneq] stage 1 done (exit $?)"

echo "[pruneq] stage 2: diag_axial50 top-8 (restarts=3)"
$PY -u scripts/analysis/vl_vs_hva/run_n20_hybrid_prune.py \
  --n 20 --h 0.5 --keep 8 \
  --restarts 3 --maxiter 400 --ftol 1e-9 --target-fid 0.95 \
  > "$LOGDIR/N20_followup_diagaxial_top8.log" 2>&1
echo "[pruneq] stage 2 done (exit $?)"

echo "[pruneq] refreshing ansatz summary"
$PY scripts/analysis/vl_vs_hva/generate_ansatz_summary.py --n 20 --h 0.5 \
  >> "$LOGDIR/N20_followup_summary.log" 2>&1
echo "[pruneq] N20 PRUNE FOLLOW-UP QUEUE COMPLETE"
