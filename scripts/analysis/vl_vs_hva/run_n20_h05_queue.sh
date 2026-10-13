#!/usr/bin/env bash
# N=20 h=0.5 — best methods on the HARD (near-degenerate) regime.
# N=20 h=0.5 has a near-degenerate ground DOUBLET (gap E1-E0 ~ 7e-5, physical gap
# E2-E0 ~ 0.407). The runner measures SUBSPACE fidelity (over the doublet) and
# ΔE/gap against the physical gap — a single-vector fidelity would be meaningless.
# restarts=1, no lookahead: dense statevector at dim 2^20 is ~2-11s/iter.
#
# Three tasks in series:
#   1) small fixed ansatze  (nn1..nnn1_half)
#   2) extended fixed ansatze (nnn2, nnn2_half)
#   3) short ADAPT (max_layers 3)
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

echo "[queue] task 1/3 — small fixed ansatze (h=0.5)"
$PY -u scripts/analysis/vl_vs_hva/run_n20_small_ansatze.py \
  --n 20 --h 0.5 --set small --restarts 1 --maxiter 400 \
  > "$LOGDIR/N20_small_h05.log" 2>&1
echo "[queue] done task 1 (exit $?)"

echo "[queue] task 2/3 — extended fixed ansatze (h=0.5)"
$PY -u scripts/analysis/vl_vs_hva/run_n20_small_ansatze.py \
  --n 20 --h 0.5 --set extended --restarts 1 --maxiter 400 \
  > "$LOGDIR/N20_extended_h05.log" 2>&1
echo "[queue] done task 2 (exit $?)"

echo "[queue] task 3/3 — short ADAPT (h=0.5, max_layers 3)"
$PY -u scripts/analysis/vl_vs_hva/run_adapt_poolA.py \
  --n 20 --h-list 0.5 --restarts 1 --maxiter 400 --target-fid 0.95 \
  --max-layers 3 --grow-step 2 --prerank-micro --warm-seed \
  --analytic-seed-new --tiebreak-abs-fid --stop-on-fpc-drop \
  --fpc-drop-min-fid 0.70 --pareto-min-fid 0.50 \
  --out-suffix short_n20_h05 --no-sync \
  > "$LOGDIR/N20_adapt_short_h05.log" 2>&1
echo "[queue] done task 3 (exit $?)"
echo "[queue] N20 h=0.5 COMPLETE"
