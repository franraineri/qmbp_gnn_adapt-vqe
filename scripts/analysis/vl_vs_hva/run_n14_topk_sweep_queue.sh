#!/usr/bin/env bash
# N14 recorte sweep: exhaustively generate top-k / prune variants at N=14 h=0.5
# to close the comparison vs ADAPT Pool-A (0.954 @ 192 CX). Reuses the existing
# run_variant_topk.py (keep-fracs + prune-tols sweeps) — no new runner.
#
# Two structural variants (the strong families), swept over many k and tolerances,
# at a few hard h. Each run persists its own artifacts under bond_ablation/.
#
# Logs: results/hva_vl_study/n14_topk_logs/<timestamp>/<run>.log
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/n14_topk_logs/${TS}"
mkdir -p "${LOGDIR}"

N=14
RESTARTS=3
MAXITER=1500
KEEP_FRACS="0.2 0.3 0.4 0.5 0.6 0.8 1.0"
PRUNE_TOLS="0.05 0.1 0.2"
VARIANTS="p2_half_nn_rx p2_base"
HLIST="0.5 0.7"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }

run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then log "DONE  ${name} (rc=0)"; else log "FAIL  ${name} (rc=${rc}) — continuing"; fi
  return 0
}

log "N14 top-k/prune sweep — ${LOGDIR}"
log "variants=${VARIANTS} keep_fracs=${KEEP_FRACS} prune_tols=${PRUNE_TOLS} h=${HLIST}"

for h in ${HLIST}; do
  for v in ${VARIANTS}; do
    run_step "topk_${v}_h${h}" "${PY}" "${GATE_DIR}/run_variant_topk.py" \
      --n ${N} --h ${h} --variant ${v} \
      --keep-fracs ${KEEP_FRACS} --prune-tols ${PRUNE_TOLS} \
      --restarts ${RESTARTS} --maxiter ${MAXITER} --no-sync
  done
done

log "QUEUE COMPLETE"
