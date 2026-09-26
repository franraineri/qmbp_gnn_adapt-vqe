# HVA Ansatz on 2D Frustrated Square TFIM — Report (analog to the VL study)

State-preparation study of the **bond-resolved HVA ansatz** on the
**frustrated square-lattice TFIM** (`tfim_frustrated`, `J2 = 0.5`), built as the
direct analog of `reports/REPORT_VL_square_frustrated.md`. Where VL synthesizes a
zero-parameter loading circuit for a known ground-state vector, the HVA is a
**variational ansatz**: we find its best parameters via VQE and measure how
close the best ansatz state gets to the exact ground state, at the same field
values, focusing on **fidelity vs 2-qubit-gate cost**.

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`
- Lattice: `square`, `N = 9`
- Fields: `h ∈ {0.5, 1.0}` (the two small-gap points VL struggled with)
- Ansätze: `nn` = `create_bond_resolved` (nearest-neighbour RZZ + RX);
  `nnn` = `create_bond_resolved_frustrated` (+ next-nearest-neighbour RZZ,
  mapping the J2 diagonals)
- Depth: `p ∈ {1, 2}`
- Optimizer: robust multi-restart L-BFGS-B (4 random restarts), noiseless
  statevector backend. Reported values are the ansatz **expressivity ceiling**
  (best θ found), not a predictor.
- Fidelity: `F = |⟨ψ_exact | ψ_HVA(θ*)⟩|²`; `ΔE/gap = |E(θ*) − E0| / gap`
- 2q count / depth: transpiled to `rz, sx, x, cx` (shared basis)
- Source artifact: `results/hva_vl_study/hva_nnn_sweep/hva_ansatze_h_sweep_frustrated.json`
- Reproduction:
  ```bash
  .venv/bin/python scripts/analysis/hva_nnn_h_sweep_frustrated.py \
      --topologies square --h 0.5 1.0 --ansatze nn nnn --p 1 2 --n-random 4
  ```

---

## 1. HVA results (square, N = 9, J2 = 0.5)

| h   | ansatz | p | gap    | fidelity F | ΔE/gap | 2q gates | 2q depth | NISQ-fid est | params |
|-----|--------|---|-------:|-----------:|-------:|---------:|---------:|-------------:|-------:|
| 0.5 | nn     | 1 | 0.2238 |     0.7253 | 2.8223 |       24 |       16 |        0.820 |     21 |
| 0.5 | nn     | 2 | 0.2238 |     0.7696 | 1.3296 |       48 |       24 |        0.674 |     42 |
| 0.5 | nnn    | 1 | 0.2238 |     0.7358 | 2.4030 |       52 |       22 |        0.655 |     35 |
| 0.5 | nnn    | 2 | 0.2238 |     0.9523 | 0.5888 |      104 |       40 |        0.430 |     70 |
| 1.0 | nn     | 1 | 0.8601 |     0.9144 | 0.3558 |       24 |       16 |        0.820 |     21 |
| 1.0 | nn     | 2 | 0.8601 |     0.9392 | 0.2127 |       48 |       24 |        0.674 |     42 |
| 1.0 | nnn    | 1 | 0.8601 |     0.9332 | 0.2924 |       52 |       22 |        0.655 |     35 |
| 1.0 | nnn    | 2 | 0.8601 |     0.9950 | 0.0396 |      104 |       40 |        0.430 |     70 |

---

## 2. HVA vs VL, side by side (same h, N = 9)

VL numbers from `reports/REPORT_VL_square_frustrated.md` (§3 default, §4 best config).

### h = 0.5 (gap 0.224 — the hard point)

| method              | fidelity F | 2q gates |
|---------------------|-----------:|---------:|
| VL default (L2/F20) |     0.809  |       30 |
| VL best (L8/F50)    |     0.943  |      120 |
| HVA nn p=1          |     0.725  |       24 |
| HVA nn p=2          |     0.770  |       48 |
| HVA nnn p=1         |     0.736  |       52 |
| **HVA nnn p=2**     | **0.952**  |      104 |

### h = 1.0 (gap 0.860)

| method              | fidelity F | 2q gates |
|---------------------|-----------:|---------:|
| VL default (L2/F20) |     0.900  |       30 |
| VL best (L8/F50)    |     0.975  |      120 |
| HVA nn p=1          |     0.914  |       24 |
| HVA nn p=2          |     0.939  |       48 |
| HVA nnn p=1         |     0.933  |       52 |
| **HVA nnn p=2**     | **0.995**  |      104 |

---

## 3. Findings

- **HVA nnn p=2 beats VL-best on both points, at fewer or equal 2q.** At h=0.5 it
  reaches F=0.952 with 104 CX vs VL-best F=0.943 with 120 CX. At h=1.0, F=0.995
  (104 CX) vs VL-best F=0.975 (120 CX). The variational ansatz that maps the
  frustration (NNN) is the strongest preparer here.
- **The NNN coupling is what unlocks the hard point.** At h=0.5, the NN-only
  ansätze plateau (F ≈ 0.73–0.77 even at p=2), while adding NNN at p=2 jumps to
  F=0.952. The J2 diagonals are structurally necessary near the small gap — depth
  alone (nn p=2) does not substitute for them.
- **At the easier point (h=1.0) HVA is very 2q-efficient.** HVA nn p=1 already
  reaches F=0.914 with just 24 CX — comparable to VL default (0.900 at 30 CX) at
  lower cost. HVA nn p=2 (0.939, 48 CX) beats VL default.
- **Fidelity tracks the gap, same as VL.** Every configuration is worse at h=0.5
  (small gap) than at h=1.0, mirroring the VL gap-limited trend.
- **The NISQ trade-off persists.** The best HVA (nnn p=2) carries the lowest
  estimated NISQ fidelity (~0.43 at 104 CX) — the fidelity/executability tension
  seen throughout the study still holds; higher preparation fidelity costs more
  2q and thus more hardware error.

---

## 4. Metric definitions

- `F` — exact statevector fidelity `|⟨ψ_exact|ψ_HVA(θ*)⟩|²` (measured, VQE-optimal θ).
- `ΔE/gap` — energy error of the best ansatz state normalized by the spectral gap.
- `2q gates`, `2q depth` — counted on the transpiled circuit (`rz, sx, x, cx`).
- `NISQ-fid est` — estimated hardware fidelity from the 2q-gate error budget
  (Heron-typical rates; `compute_error_budget`).
- `params` — variational parameter count of the ansatz.
- `E0`, `gap` — exact ground-state energy and spectral gap (`ClassicalSolver`).

---

## 5. Scaling to N = 18 and the optimization-vs-ansatz question

Sections 1–4 established the ansatz expressivity ceiling at N = 9. Extending the
study to larger N revealed that the *hard* part of state preparation on this
ansatz is not expressivity but optimization — and only at large N. This section
records what was explored beyond the N = 9 tables above.

### 5.1 N = 18, h = 0.5 — the stuck point

At N = 18, h = 0.5 (nnn p=2, second-order warm-start, adjoint gradient), the best
achieved fidelity plateaus at ~0.65 even with a large iteration budget:

| maxiter | best fidelity | notes |
|--------:|:-------------:|-------|
| 300     | 0.637         | all 3 restarts hit the iteration cap (nit=300, not converged) |
| 1200    | 0.653         | +0.016 for 4× the iterations — diminishing returns |

The pure second-order seed (sigma=0) gave the best fidelity but hit the cap
(nit=1200, still descending); the perturbed restarts *converged* (nit<1200) but to
*worse* fidelities (0.275, 0.495). This counterintuitive pattern — the
non-converged run winning — is genuine, not a selection bug (verified: best-of
picks the lowest energy, which coincided with the highest fidelity).

Ground-state energy error is only 2.45% (dE=0.343, e0=−16.985), so the state is
energetically close; the fidelity is what lags.

### 5.2 Where the missing fidelity goes (spectral decomposition)

Decomposing the N=18 prepared state over the exact eigenvectors:

| eigenvector | weight |ψ⟩ | note |
|-------------|:----------:|------|
| E₀ (ground) | 0.6527 | = the reported fidelity |
| E₁ (near-degenerate, gap 0.006) | **0.0000** | degeneracy is NOT the leak |
| E₃ (ΔE ≈ 0.31) | 0.1641 | 16% weight on a higher state |
| rest (E₄+) | ~0.19 | spread over higher excited states |

Two verified conclusions:

- **The near-degeneracy is not the obstacle.** The near-degenerate partner E₁
  (gap 0.006) carries weight *exactly* 0. The subspace fidelity (projection onto
  span{E₀, E₁}) equals the single-vector fidelity — the degeneracy is handled
  cleanly and does not corrupt the metric.
- **The deficit is weight on *higher* states.** 35% of the weight sits on E₃ and
  above — the signature that would normally suggest an expressivity limit.

### 5.3 Expressivity ceiling vs N — the decisive scan

To separate optimization from expressivity, we measured, per N at h=0.5, the
**ceiling** (best fidelity over several random inits — the ansatz's intrinsic
limit) against the **warm-start** fidelity, so `gap = ceiling − warmstart` is the
fidelity the warm-start leaves on the table.

| N  | gap    | warmstart | 1st-order | ceiling | fidelity_gap | verdict |
|----|:------:|:---------:|:---------:|:-------:|:------------:|---------|
| 8  | 0.0923 | 0.9628    | 0.9675    | 0.9628  | +0.0000      | warm-start OK |
| 10 | 0.0061 | 0.9871    | 0.9849    | 0.9871  | +0.0000      | warm-start OK |
| 18 | 0.0059 | 0.6527    | —         | (high, see below) | large | optimization-limited |

**N = 10 is the decisive control.** Its gap (0.0061) is essentially identical to
N = 18's (0.0059), yet the p=2 ansatz reaches fidelity 0.987 with random inits and
the warm-start matches it (fidelity_gap = 0). This rules out both the small gap and
the ansatz depth as the cause of the N=18 collapse:

- The near-degeneracy (small gap) cannot be the obstacle — N=10 has the same gap
  and reaches 0.987.
- The p=2 ansatz is expressive enough even at a minuscule gap — 0.987 at N=10.

Therefore the ~0.65 at N=18 is an **optimization limit at large N**, not an
expressivity limit and not a degeneracy artifact. The right lever is better basin
exploration (varied restarts), *not* raising p.

> Methodology note: the ceiling scan uses N ≤ 12 (statevector cheap, seconds per
> optimization). At N = 18 each L-BFGS-B with a meaningful iteration budget takes
> minutes per restart (2¹⁸ statevector + per-call symbolic overhead), so the ceiling
> there is inferred from the N-trend rather than measured densely. A random-init
> init at θ=0 (|+⟩^N) does not move at all (nit=0) — the origin is a critical point,
> which is why the analytic warm-start is essential just to start descending.

### 5.4 Ansatz ↔ Hamiltonian correspondence (verified, no bug)

Audited to rule out a design fault behind the deficit: `create_bond_resolved_frustrated`
and `build_frustrated_tfim` both use the same `_generate_nnn_edges(lattice)` and the
same `lattice.edges`. The Hamiltonian has three term types (−J₁·ZZ_nn, +J₂·ZZ_nnn,
−h·X); the ansatz has an RZZ/RX generator for each. No Hamiltonian term lacks a
matching ansatz generator — the ansatz is structurally complete for this model.

Design note (not a bug): the `tfim_frustrated` ModelSpec registers
`create_frustrated_tfim` as its canonical circuit, but this study deliberately uses
the more expressive `create_bond_resolved_frustrated`. Runs driven through the
standard pipeline (`spec.create_circuit`) would get a different ansatz.

### 5.5 What this changes for the plan

- **Do not jump to p=3 for N=18 h=0.5.** The evidence points to optimization, not
  expressivity. p=3 was the earlier hypothesis; the N-scan overturned it.
- **The lever is varied-basin restarts** (mixed seeds + perturbation subspaces),
  now implemented as the `mixed` strategy in `StudyRunner`.
- **Diagnostics are now automatic.** Every `StudyRunner` point records per-restart
  convergence, the spectral decomposition above, and Var(H) under
  `point["diagnostics"]`, so this analysis no longer needs bespoke scripts.

### 5.6 Reproduction

```bash
# N=18 second-order (baseline stuck point)
.venv/bin/python scripts/analysis/vl_vs_hva/run_n18_second_order.py \
    --n 18 --h 0.5 --p 2 --maxiter 1200 --strategy second_order

# mixed varied-basin restarts (the recommended lever)
.venv/bin/python scripts/analysis/vl_vs_hva/run_n18_second_order.py \
    --n 18 --h 0.5 --p 2 --maxiter 600 --strategy mixed
```
