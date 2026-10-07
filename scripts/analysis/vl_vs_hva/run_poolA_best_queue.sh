#!/usr/bin/env bash
# ADAPT Option-A "best" queue: the three MEASURED-winning levers accumulated —
# warm-seed (B) + layer-growth-penalty (C) + analytic-seed-new (2c).
# Excludes the levers the smoke rejected (fast_rank, block_precondition).
# N10 then N14, hard h. run_adapt_poolA.py persists partial+final itself.
#
# Logs: results/hva_vl_study/poolA_logs/best_<timestamp>/<run>.log
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/poolA_logs/best_${TS}"
mkdir -p "${LOGDIR}"

HLIST="0.3 0.5 0.7 0.9"
RESTARTS=3
MAXITER=1500
TARGET_FID=0.95
MAX_LAYERS=6
MIN_DFID=0.01
PENALTY=1.0          # layer-growth penalty (C)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }

run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then log "DONE  ${name} (rc=0)"; else log "FAIL  ${name} (rc=${rc}) — continuing"; fi
  return 0
}

log "ADAPT Option-A BEST queue (warm-seed + penalty + analytic-seed-new) — ${LOGDIR}"
log "h_list=${HLIST} restarts=${RESTARTS} maxiter=${MAXITER} penalty=${PENALTY}"

for N in 10 14; do
  run_step "poolA_best_N${N}" "${PY}" "${GATE_DIR}/run_adapt_poolA.py" \
    --n ${N} --h-list ${HLIST} --restarts ${RESTARTS} --maxiter ${MAXITER} \
    --target-fid ${TARGET_FID} --max-layers ${MAX_LAYERS} --min-dfid ${MIN_DFID} \
    --layer-growth-penalty ${PENALTY} --warm-seed --analytic-seed-new --no-sync
done

run_step scoreboard "${PY}" "${GATE_DIR}/update_scoreboard.py"
log "QUEUE COMPLETE"
