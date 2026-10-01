# Classical→Quantum Crossover — 2D Frustrated Square TFIM (quench dynamics)

Where a bond-dimension-χ MPS stops tracking the exact post-quench state.
Model: `tfim_frustrated`, J = 1.0, J₂ = 0.5, square lattice. Initial state:
exact ground state of H(h₁) (no GNN — this isolates *where the classical method
dies*, independent of state-preparation fidelity).

## Setup

- Quench: prepare in the ordered phase (h₁ = 0.3, small gap) → evolve under
  H(h₂ = 2.5) (paramagnet). This crosses the ordered→paramagnet transition,
  which for J₂ = 0.5 sits around h ~ 0.5–1.0 (the gap grows monotonically with
  h: h = 0.5 → gap 0.224, h = 3.0 → gap 4.54).
- Time: 2nd-order Suzuki–Trotter, dt = 0.02, 125 steps (t = 2.5).
- Reference: exact time evolution (`expm_multiply`), N ≤ 22.
- Metric: per step, truncate the exact state to χ (variational SVD) and measure
  infidelity 1 − |⟨ψ_exact|ψ_χ⟩|². Crossover = first step with infidelity > 0.01
  (99% fidelity floor). This is measured **against ground truth**, not the MPS's
  own energy-conservation drift (see Methodology below).

## Result — crossover Trotter step (t = step × 0.02)

| N  | χ=8 | χ=16 | χ=32 | χ=64 | χ=128 | S_max (spatial) |
|----|-----|------|------|------|-------|-----------------|
| 10 | 15  | 34   | —    | —    | —     | 2.36            |
| 14 | 12  | 17   | 30   | —    | —     | 3.73            |
| 18 | 8   | 13   | 17   | 26   | 49    | 5.17            |

`—` = the MPS never crosses the 1% infidelity threshold within 125 steps
(χ is large enough to be effectively exact at that N).

## Physics

1. **Entanglement grows with N**: S_max 2.36 → 3.73 → 5.17. The post-quench
   state carries more entanglement as the system grows, so a fixed χ holds it
   for less time.
2. **At fixed χ, the crossover advances with N** (χ=8: step 15 → 12 → 8). This
   is the signature of the classical→quantum crossover: the larger the system,
   the sooner a bounded-χ MPS fails.
3. **At fixed N, more χ lasts longer** (strictly monotone). At N = 10 any χ ≥ 32
   is effectively exact (central Schmidt rank ≤ 2⁵ = 32), so it never crosses.
4. **N = 18 is the interesting size**: even χ = 128 fails at step 49 (t = 0.98),
   and χ = 64 fails at step 26 (t = 0.52). The post-quench dynamics exceed
   χ = 128 at moderate times — the regime where a classical MPS at accessible
   bond dimension can no longer follow the state.

## Implication for the GNN

This maps *where* the classical method dies. It uses the exact ground state as
the initial state, so it is independent of the GNN's preparation fidelity. The
next step (separate experiment) is to replace the exact initial state with the
GNN-prepared |ψ(θ_GNN)⟩ and ask whether the same crossover holds when the input
carries the GNN's preparation error — i.e. whether the GNN is a good-enough
input for the dynamics to still leave the classically-simulable region.

## Methodology guards applied (why the numbers are trustworthy)

- **Trotter error is not the crossover.** With dt = 0.1 the energy drift at
  χ-exact was 0.15 (pure Trotter, ∝ dt²) — it would trip any energy-based
  threshold for the wrong reason. Fixed with dt = 0.02 (Trotter drift < 0.006 at
  t = 2.5).
- **Crossover is measured vs the exact state, not MPS self-consistency.** The
  earlier energy-conservation-drift criterion was non-monotone in χ (χ = 8 crossed
  *before* χ = 4) because Aer's greedy truncation over a deep circuit accumulates
  error non-variationally. The current criterion (infidelity vs the exact state,
  via variational SVD truncation) is monotone in χ by construction.
- **2D-aware entanglement cut.** S_max uses a spatial (left/right column)
  bipartition of the square lattice, not a qubit-index cut. An index cut is only
  a spatial cut in 1D; in 2D it mixes the two halves and overstates S.
- **Trotter nn/nnn grouping is exact.** All ZZ terms (nn −J₁ and nnn +J₂) commute
  (diagonal in Z), so grouping them in one ZZ block introduces no extra Trotter
  error beyond the standard ZZ↔X splitting (verified: Trotter-vs-exact fidelity
  1.000000 at small dt).

## Reproduce

```bash
for N in 10 14 18; do
  .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --topology square --n-qubits $N --model tfim_frustrated --j2 0.5 \
    --h1 0.3 --h2 2.5 --n-trotter 125 --dt 0.02 --chi-values 8 16 32 64 128 --seeds 42
done
```

Envelopes: `results/experiments/exp_frustrated/qd1/tfim_frustrated/square_p1/`
(section_2 = MPS crossover, `method="exact_reference_fidelity"`).
Engine: `scripts/experiment_runners/scaling/run_quench_dynamics_study.py`.
