#!/usr/bin/env bash
# N=20 h=1.0 — push fidelity past the 0.714 ceiling of the small ansatze.
# Two tasks in series:
#   1) extended fixed ansatze (deeper nnn: nnn2, nnn2_half)
#   2) short ADAPT (max_layers 3, restarts 1, no lookahead) — grows selectively.
# restarts=1 + no lookahead because h=1.0 has a large gap (0.227, easy regime) and
# N=20 dense statevector is ~2-11s/iter: the expensive extras do not pay off here.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

echo "[queue] task 1/2 — extended fixed ansatze"
$PY -u scripts/analysis/vl_vs_hva/run_n20_small_ansatze.py \
  --n 20 --h 1.0 --set extended --restarts 1 --maxiter 400 \
  > "$LOGDIR/N20_extended_h10.log" 2>&1
echo "[queue] done task 1 (exit $?)"

echo "[queue] task 2/2 — short ADAPT (max_layers 3)"
$PY -u scripts/analysis/vl_vs_hva/run_adapt_poolA.py \
  --n 20 --h-list 1.0 --restarts 1 --maxiter 400 --target-fid 0.95 \
  --max-layers 3 --grow-step 2 --prerank-micro --warm-seed \
  --analytic-seed-new --tiebreak-abs-fid --stop-on-fpc-drop \
  --fpc-drop-min-fid 0.70 --pareto-min-fid 0.60 \
  --out-suffix short_n20_h10 --no-sync \
  > "$LOGDIR/N20_adapt_short_h10.log" 2>&1
echo "[queue] done task 2 (exit $?)"
echo "[queue] N20 PUSH COMPLETE"
