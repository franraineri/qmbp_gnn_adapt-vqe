#!/usr/bin/env bash
# ADAPT Option-A queue: unrestricted-p growth (Δfid/Δ2q), N10 then N14, hard h.
#
# run_adapt_poolA.py persists partial+final itself (StudyPersister), so a crash
# never loses completed points. This wrapper runs N10 then N14 sequentially with
# production budgets, logs each to its own file, and never lets one failure abort
# the other (set +e per command).
#
# Logs: results/hva_vl_study/poolA_logs/<timestamp>/<run>.log
# Run via the background-process tool (survives the session).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/poolA_logs/${TS}"
mkdir -p "${LOGDIR}"

HLIST="0.3 0.5 0.7 0.9"
RESTARTS=3           # >=2 required so repeat-layer's new layer can activate
MAXITER=1500
TARGET_FID=0.95
MAX_LAYERS=6
MIN_DFID=0.01        # keep growing while absolute Δfid clears this (stop-rule fix)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }

run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then log "DONE  ${name} (rc=0)"; else log "FAIL  ${name} (rc=${rc}) — continuing"; fi
  return 0
}

log "ADAPT Option-A queue — logs in ${LOGDIR}"
log "h_list=${HLIST} restarts=${RESTARTS} maxiter=${MAXITER} target_fid=${TARGET_FID} max_layers=${MAX_LAYERS}"

run_step poolA_N10 "${PY}" "${GATE_DIR}/run_adapt_poolA.py" \
  --n 10 --h-list ${HLIST} --restarts ${RESTARTS} --maxiter ${MAXITER} \
  --target-fid ${TARGET_FID} --max-layers ${MAX_LAYERS} --min-dfid ${MIN_DFID} --no-sync

run_step poolA_N14 "${PY}" "${GATE_DIR}/run_adapt_poolA.py" \
  --n 14 --h-list ${HLIST} --restarts ${RESTARTS} --maxiter ${MAXITER} \
  --target-fid ${TARGET_FID} --max-layers ${MAX_LAYERS} --min-dfid ${MIN_DFID} --no-sync

run_step scoreboard "${PY}" "${GATE_DIR}/update_scoreboard.py"

log "QUEUE COMPLETE"
