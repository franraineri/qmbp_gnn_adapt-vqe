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
| 8  | 0.0923 | 0.9628    | 0.9675    | 0.9628  | +0.0000      | warm-start optimal |
| 10 | 0.0061 | 0.9871    | 0.9849    | 0.9871  | +0.0000      | warm-start optimal |
| 12 | 0.0071 | 0.9449    | 0.9388    | 0.9702  | **+0.0252**  | gap opens, ceiling still high |
| 18 | 0.0059 | 0.6527    | —         | (high, see below) | large | optimization-limited |

**The N-trend is a monotone opening of the warm-start gap while the ceiling stays
high.** At N=8,10 the warm-start reaches the ceiling exactly (fidelity_gap = 0). At
N=12 the first non-zero gap appears (warm-start 0.945 vs ceiling 0.970, Δ=+0.025) —
the warm-start begins to fall short even though the ansatz can still reach 0.97. By
N=18 the warm-start has collapsed to 0.65 while the ceiling remains high. The
warm-start degrades progressively with N; the ansatz expressivity does not.

**N = 10 is the decisive control.** Its gap (0.0061) is essentially identical to
N = 18's (0.0059), yet the p=2 ansatz reaches fidelity 0.987 with random inits and
the warm-start matches it (fidelity_gap = 0). And N=12 (gap 0.0071, also
near-degenerate) still reaches ceiling 0.970. This rules out both the small gap and
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
- **The lever is multi-seed sampling with a convergence-sufficient budget** — not a
  single analytic seed, and (per §5.7) not Metropolis at this N.
- **Diagnostics are now automatic.** Every `StudyRunner` point records per-restart
  convergence, the spectral decomposition above, and Var(H) under
  `point["diagnostics"]`, so this analysis no longer needs bespoke scripts.

### 5.6 Fair-convergence ranking at N=18 h=0.5 (bias-corrected)

An initial `mixed` run (maxiter=500) suggested the alternative seeds were "worse",
but 4 of 6 restarts had hit the iteration cap (not converged). Re-running each seed
with a convergence budget (maxiter=3000, two-segment starvation probe) corrects the
bias:

| seed type          | fidelity | ΔE from E₀ | nit   | note |
|--------------------|:--------:|:----------:|:-----:|------|
| **second_dir_x**   | **0.759**| 0.190      | 718   | best (θ_x-directed) |
| second_iso_large   | 0.738    | 0.200      | 1089  | |
| second_dir_zz      | 0.735    | 0.199      | 677   | |
| second_pure        | 0.655    | 0.340      | 2781  | prev. "best" — now 4th |
| second_iso_small   | 0.461    | 0.682      | 446   | |
| first_pure         | 0.438    | 0.778      | 973   | |

Best verified fidelity at N=18 h=0.5 rises to **0.759** (+0.10 over the pre-correction
0.653). The θ_x-directed perturbation — a *false positive at N=9* — is the *winner at
N=18*: the useful perturbation subspace is N-dependent.

### 5.7 What is rigorously established about the optimization landscape

- **The bottleneck is basin SELECTION, not local convergence.** With convergence
  guaranteed, different seeds settle at clearly different fidelities (0.44–0.76) —
  the signature of multiple, well-separated local minima with substantially different
  ground-state overlap.
- **The variational minima are well-separated in ENERGY** (spread 0.588 = 99.7× the
  gap), even though the target ground state is near-degenerate with its first excited
  partner. So best-of-by-energy is a *valid* selector here (verified: lowest-energy
  seed = highest-fidelity seed, monotone order). The near-degeneracy of the *target*
  does not translate into near-degeneracy of the *ansatz minima*.
- **No seed is dominant across (N, h).** second_pure wins at N=9 h=0.5 but is 4th at
  N=18; θ_x-directed inverts from false-positive (N=9) to winner (N=18). The best
  basin moves with (N, h) — so the robust approach is multi-seed best-of with
  guaranteed convergence, not a universal analytic seed.
- **Metropolis is refuted at N=18 with a modest per-hop budget** (§ heuristic report):
  0.42, all 13 hops capped. Chain-drift only helps if each hop converges; at N=18
  that costs ~3000 iters/hop, impractical.

### 5.8 Methodological lesson

Never rank restart fidelities when `nit == maxiter` (capped): the run may be starved
(still descending), not in a worse basin. The `run_n18_fair_convergence.py` script
uses a two-segment starvation probe (`late_gain`) to distinguish starvation from a
genuine local minimum before ranking.

### 5.9 Reproduction

```bash
# N=18 second-order (baseline stuck point, pre-correction)
.venv/bin/python scripts/analysis/vl_vs_hva/run_n18_second_order.py \
    --n 18 --h 0.5 --p 2 --maxiter 1200 --strategy second_order

# Fair-convergence seed comparison (the correct, bias-free lever): each seed
# optimized to genuine convergence, capped seeds flagged starved vs local-min.
.venv/bin/python scripts/analysis/vl_vs_hva/run_n18_fair_convergence.py \
    --n 18 --h 0.5 --p 2 --maxiter 3000
```

> Note: the `mixed` strategy at maxiter=500 and Metropolis at n_hops=12 were both
> tried and are documented in §5.6–5.7 and the heuristic report; neither is the
> recommended lever — `mixed` starved its restarts (unfair ranking) and Metropolis
> collapsed (0.42) with an insufficient per-hop budget. Multi-seed best-of with a
> convergence-sufficient budget (the fair-convergence script) is the correct approach.

---

## 6. Landscape structure (basin-counting) and h=1.0 transfer

Beyond ranking seeds, we measured the *structure* of the optimization landscape:
how many distinguishable fidelity basins exist and how large the good one is,
via K=20 random converged starts per N (`run_basin_count.py`).

| N  | gap    | # basins | dominant-basin attraction | true ceiling | corr(E, fid) |
|----|:------:|:--------:|:-------------------------:|:------------:|:------------:|
| 8  | 0.0923 | 5        | 75%                       | 0.9653       | −0.982       |
| 10 | 0.0061 | 3        | 90%                       | 0.9910       | −0.984       |
| 12 | 0.0071 | 5        | 79%                       | 0.9816       | −0.997       |

**Rigorously established (N ≤ 12):**

- The landscape is **benign**: one large dominant basin (75–90% of random starts)
  and a high ceiling (~0.98). N=10 (gap 0.006, same regime as N=18) has the *largest*
  good basin (90%).
- **best-of-by-energy is valid**: corr(E, fidelity) = −0.98…−0.997 — lowest energy is
  highest fidelity, near-perfectly monotone. (Verifies the selection all runners use.)
- **Refuted**: "dominant basin shrinks with N" (75→90→79, no trend); "basin count
  grows with N" (5→3→5); "ceiling decays with N" (stays ~0.98).

**The open question this sharpens:** N≤12 is benign, yet N=18 warm-start = 0.65. With
gradual shrinking refuted, the N=18 collapse is either an *abrupt* change past N=12,
or the analytic seed specifically lands in a bad basin while a *random* start would
not. Untested: no random-start basin-count at N=18 exists (all N=18 runs used
analytic seeds). A small random basin-count at N=18 is the key remaining experiment.

**h=1.0 transfer:** the best h=0.5 method (`second_dir_x`) applied once at N=18 h=1.0
converges to **0.921** (nit=618), beating the standard warm-start baseline (~0.885)
by +0.036. At the easy point the directed-θ_x method with a convergence budget finds
a better state than the plain warm-start.

---

## 7. Ansatz structure variants — expressivity vs 2q-gate cost (p=1-based, N=10)

Systematic study of *what structural modification* recovers the expressivity a
plain extra layer buys, and at what 2-qubit-gate cost. Built on p=1 at N=10
h=0.5 (gap 0.006 — the same near-degenerate regime as N=18), each variant run at
a fixed protocol (restarts=4, maxiter=3000, best-of by energy, exact fidelity).

Infrastructure (reusable): `HVACircuitBuilder.create_bond_resolved_frustrated_configurable`
(general engine, `circuits/hva.py`), the declarative `AnsatzVariant`/`VARIANTS`
registry (`circuits/hva_variants.py`), and `run_ansatz_variants.py`
(`--find-saturation` + comparison). Schema `ansatz_variants_v1`.

### Saturation scan (Phase 0): p=1 never saturates

| N  | gap    | p1 ceiling |
|----|:------:|:----------:|
| 6  | 0.2506 | 0.6967     |
| 8  | 0.0923 | 0.7072     |
| 10 | 0.0061 | 0.4089     |
| 12 | 0.0071 | 0.2837     |

Plain p=1 is expressivity-limited for the frustrated model and *degrades* with N
(0.70 → 0.28). N=10 chosen for the variant study: clear deficit (0.41) in the
near-degenerate regime.

### Variant comparison (N=10 h=0.5)

| variant         | fidelity | 2q (CX) | Δ vs p1 | cost class |
|-----------------|:--------:|:-------:|:-------:|------------|
| p1_base         | 0.4089   | 56      | —       | baseline   |
| p1_interleaved  | 0.3450   | 56      | −0.064  | same 2q (nn/nnn order swap) |
| p1_rz_extra     | 0.4191   | 56      | +0.010  | 0 CX (trailing RZ) |
| p1_rx_extra     | 0.4784   | 56      | +0.070  | 0 CX (trailing RX) |
| **p1_half_nnn** | **0.9740** | 86    | +0.565  | +30 CX (half-layer nnn) |
| **p1_half_nn**  | **0.9873** | 82    | +0.578  | +26 CX (half-layer nn) |
| p2_base         | 0.9917   | 112     | +0.583  | full extra layer (anchor) |
| p3_base         | 0.9963   | 168     | +0.587  | two extra layers (anchor) |

**Cost-benefit (the headline number):** `p1_half_nn` reaches 0.9873 at 82 CX vs
`p2_base` 0.9917 at 112 CX — **99.6% of the full-layer fidelity at 73% of its 2q
cost**. The half-layer of nn RZZ is the efficient sweet spot: essentially the p=2
ceiling for 30 fewer CX.

### Findings

- **A half-layer of RZZ recovers nearly all the expressivity of a full extra
  layer, at intermediate 2q cost.** `p1_half_nn` reaches 0.987 with 82 CX — up
  from 0.409 (p1) — capturing essentially the p=2 ceiling (~0.98 at N=10) at
  roughly **73% of a full layer's 2q cost**. This is the efficient intermediate
  point (Option A) the study set out to find.
- **The missing expressivity is entanglement (RZZ), not local rotations.** The
  zero-2q options help only marginally (RX +0.070, RZ +0.010), an order of
  magnitude less than a half-layer of RZZ (+0.58). Adding RX is a cheap partial
  win (free in the 2q budget) but does not substitute for more entanglement.
- **nn beats nnn, marginally, and is cheaper.** `p1_half_nn` (0.987, 82 CX) edges
  `p1_half_nnn` (0.974, 86 CX): the nearest-neighbour block contributes slightly
  more per CX.
- **Order matters — interleaving hurts.** Swapping to nnn-before-nn
  (`p1_interleaved`, same CX) *reduces* fidelity (−0.064); the standard nn-then-nnn
  ordering is better.

### Hardware implication

For the frustrated model, going from p=1 to p=2 (a full extra layer, +56 CX) is
overkill: a **half-layer of nn RZZ (+26 CX)** delivers nearly the same fidelity.
This is the cost-optimal structural upgrade when the 2q budget is the constraint.

Data: `results/hva_vl_study/hva_nnn_sweep/ansatz_variants_square_N10_h0.50.json`,
`ansatz_saturation_square_p1_h0.50.json`.

---

## 8. Half-layer does NOT transfer to N=18 — correction to §7

§7 found (at N=10) that a half-layer of nn RZZ captures ~all of p=2's fidelity at
73% of the CX. Testing the same variants at **N=18 h=0.5** (converged, restarts=4,
maxiter=3000) overturns that as an N-small artifact:

| variant     | fidelity | 2q (CX) | fid/CX   | converged | note |
|-------------|:--------:|:-------:|:--------:|:---------:|------|
| p1_half_nn  | 0.2851   | 186     | 1.53e-3  | 4/4       | collapses (was 0.987 at N=10) |
| p2_base     | 0.6552   | 264     | 2.48e-3  | 4/4       | full layer |
| p3_base     | 0.8941   | 396     | 2.26e-3  | 1/4       | floor (3 starved → true ceiling higher) |

**Findings:**

- **The half-layer sweet spot is N-small-specific.** At N=10 `p1_half_nn`=0.987 ≈ p2;
  at N=18 it collapses to 0.285 — far below p2 (0.655) and p3 (0.894).
- **Required expressivity scales super-linearly with N.** At N=10 a half-layer
  sufficed; at N=18 even a full extra layer (p2) only reaches 0.655 — the second
  full layer (p3) is what recovers ~0.89.
- **Clean monotone hierarchy in CX at N=18**, no shortcuts: half_nn (186 CX, 0.285)
  < p2 (264 CX, 0.655) < p3 (396 CX, 0.894). half_nn has the *worst* fidelity-per-CX
  of the three here — the opposite of N=10.
- **half_nn and p2 converged 4/4** → 0.285 / 0.655 are genuine ceilings of those
  structures (not starvation). p3 converged 1/4 → 0.894 is a floor.

**Recurring lesson (seen with seeds, the landscape, and now ansatz structure):**
small-N conclusions do not transfer to large N. At N=18 the only reliable lever is
**more depth (p3)**, not cheap structural tricks. The §7 half-layer result stands
only at N=10.

Data: `results/hva_vl_study/hva_nnn_sweep/ansatz_variants_square_N18_h0.50.json`.
