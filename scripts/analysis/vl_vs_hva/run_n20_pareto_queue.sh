#!/usr/bin/env bash
# N=20 — close the Pareto frontier (option C): deepest fixed ansatz (p3) AND
# free-growth ADAPT, on BOTH regimes (h=0.5 degenerate, h=1.0 easy).
#
# Rationale:
#   - p3 full+prunes: can fidelity go past nnn2_half's 0.965 (h0.5) / 0.866 (h1.0)
#     and at what 2q cost? Closes the frontier from above.
#   - ADAPT max_layers 6 (not 3): does adaptive growth reach the p3 fidelity with
#     fewer 2q? The "ADAPT vs fixed structure" question at large N.
#
# Config: restarts=1, no lookahead — dense statevector at dim 2^20 is ~5s/grad for
# p3; the h=1.0 gap is large and h=0.5 warm-starts are strong, so random restarts
# and the quadratic lookahead do not pay off. Pipeline upgrades (auto_donors +
# CachedBackend) are active inside the runners. Sequential; crash-safe partials.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

# ── p3 fixed-ansatz Pareto sweep, both regimes ───────────────────────────────
for h in 0.5 1.0; do
  log="$LOGDIR/N20_p3_h${h}.log"
  echo "[queue] p3_half_nn_rx sweep h=$h -> $log"
  $PY -u scripts/analysis/vl_vs_hva/run_variant_topk.py \
    --n 20 --h "$h" --variant p3_half_nn_rx \
    --prune-tols 0.1 0.2 --restarts 1 --maxiter 300 --no-sync \
    > "$log" 2>&1
  echo "[queue] done p3 h=$h (exit $?)"
done

# ── free-growth ADAPT (max_layers 6), both regimes ───────────────────────────
for h in 0.5 1.0; do
  log="$LOGDIR/N20_adapt_free_h${h}.log"
  echo "[queue] ADAPT free h=$h -> $log"
  $PY -u scripts/analysis/vl_vs_hva/run_adapt_poolA.py \
    --n 20 --h-list "$h" --restarts 1 --maxiter 400 --target-fid 0.95 \
    --max-layers 6 --grow-step 2 --prerank-micro --warm-seed \
    --analytic-seed-new --tiebreak-abs-fid --stop-on-fpc-drop \
    --fpc-drop-min-fid 0.80 --pareto-min-fid 0.50 \
    --out-suffix free_n20_h${h} --no-sync \
    > "$log" 2>&1
  echo "[queue] done ADAPT free h=$h (exit $?)"
done
echo "[queue] N20 PARETO QUEUE COMPLETE"
