#!/usr/bin/env bash
# Overnight symmetry campaign — build short, high-fidelity ansätze for N=20 h=0.5
# square frustrated, respecting the lattice D2 symmetry FROM THE START.
#
# Central lesson (mem 388): the existing pruned optima (prune_th) BREAK orbits —
# their by-|θ|-threshold selection keeps some bonds of an orbit and drops others,
# so parameter-tying over them is invalid (compressed seed → fidelity ≈ 0). To
# build genuinely tied, short, high-fidelity ansätze the SELECTION itself must be
# orbit-complete (select_orbit_complete), not a threshold prune.
#
# Two sound paths, run SEQUENTIALLY (each N=20 statevector job saturates cores):
#
#   A) ORBIT-COMPLETE TIED FROM SCRATCH — select whole orbits (nn backbone +
#      nnn), tie by symmetry, seed from the ANALYTIC orbit-constant seed (valid
#      because the selection respects orbits). Sweep orbit budgets to map the
#      fid-vs-2q frontier with FEW params. This creates our own symmetric ansätze.
#      Uses --rank-by theta (|θ| of the stored full spec — instant, no full-circuit
#      re-convergence).
#
#   B) PER-BOND DEEP REFINEMENT of the known optima — the prune_th specs can only
#      be improved in their own per-bond space (174/188 params; tying is invalid).
#      Generous night budget + symmetrized seed (projects onto the symmetric
#      subspace KEEPING per-bond width — no orbit breaking) to squeeze more
#      fidelity at FIXED 2q (148 / 176).
#
# Usage:  bash scripts/analysis/vl_vs_hva/run_n20_symmetry_campaign.sh
set -u

PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"
HERE="scripts/analysis/vl_vs_hva"

SPEC_A="p3_half_nn_rx_prune0.3_th_nn11_nnn10"      # 0.8988 @ 148 2q (per-bond)
SPEC_B="p3_half_nn_rx_prune0.25_th_nn13_nnn12"     # 0.9491 @ 176 2q (per-bond)

echo "===== SYMMETRY CAMPAIGN START $(date) ====="

# ── PATH A: orbit-complete tied from scratch (our own symmetric ansätze) ──────
# p3_half_nn_rx structure, prune the nn backbone AND nnn by WHOLE orbits across a
# grid, 3-way (loose_indep / loose_tied / orbit_complete_tied) at equal 2q. The
# orbit_complete_tied band is the short symmetric ansatz we want. Generous budget.
# Budgets chosen for SHORT ansätze (<~200 2q): prune the nn backbone hard by
# whole orbits (keep 4-6 of 10) + few nnn orbits. keep-nn-orbits=8 already gives
# 260 2q (smoke), so stay at 4-6 for the short regime.
echo ">>> [A] orbit-complete tied frontier (p3_half_nn_rx, short budgets)  $(date)"
$PY -u "$HERE/run_n20_frontier_symmetry.py" \
    --n 20 --h 0.5 --structure p3_half_nn_rx \
    --keep-nn-orbits 4 5 6 --keep-orbits 1 2 3 \
    --rank-by theta \
    --restarts 6 --maxiter 1000 --target-fid 0.97 --ftol 1e-10 --micro-descent 8 \
    > "$LOGDIR/CAMPAIGN_A_orbit_complete_frontier.log" 2>&1
echo ">>> [A] done  $(date)"

# ── PATH A2: symmetry-LEVEL sweep (trivial → reflection → D2) at fixed 2q ─────
# For a fixed orbit-complete selection, how much does each level of the point
# group buy in params vs fidelity? One representative subgroup per order.
for KO in 2 3; do
  echo ">>> [A2] symmetry-level sweep keep-orbits=$KO  $(date)"
  $PY -u "$HERE/run_symmetry_sweep.py" \
      --n 20 --h 0.5 --keep-orbits "$KO" \
      --restarts 6 --maxiter 1000 --target-fid 0.995 --ftol 1e-10 \
      --micro-descent 8 \
      > "$LOGDIR/CAMPAIGN_A2_symsweep_ko${KO}.log" 2>&1
  echo ">>> [A2] keep-orbits=$KO done  $(date)"
done

# ── PATH B: per-bond deep refinement of the known optima (symmetrized seed) ───
# The prune_th specs break orbits, so refine them in their own per-bond space.
# symmetrized seed projects onto the symmetric subspace (keeps per-bond width).
# Night budget: many restarts, long maxiter. Same 2q (148 / 176) as baseline.
echo ">>> [B] per-bond deep refinement (stored + symmetrized)  $(date)"
$PY -u "$HERE/run_n20_refine_pruned.py" \
    --n 20 --h 0.5 --specs "$SPEC_A" "$SPEC_B" \
    --arms stored symmetrized \
    --restarts 8 --maxiter 1500 --target-fid 0.999 --ftol 1e-11 \
    --save-specs \
    > "$LOGDIR/CAMPAIGN_B_perbond_deep.log" 2>&1
echo ">>> [B] done  $(date)"

echo "===== SYMMETRY CAMPAIGN DONE $(date) ====="
