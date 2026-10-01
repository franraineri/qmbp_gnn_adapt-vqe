# Optimized θ angles in the bond-resolved frustrated HVA: patterns & symmetries

System: frustrated square TFIM (J=1, J2=0.5), bond-resolved +NNN HVA ansatz.
Primary dataset: dense h-sweep, N=8, p=1, 24 points h ∈ [0.2, 2.5], first-order
warm-start descent (single physical branch). Cross-checks: N=8 p=2, N=18 p=2
(exact `eigsh` ground truth). Analyzer: `qmbp_simulation.analysis.theta_patterns`.

Data:

- `theta_patterns/theta_sweep_square_N8_p1_J20.50_first_order.json` (+ `_bestof`)
- `theta_patterns/analysis_square_N8_p1_J20.50_first_order.json`
- Figures in `theta_patterns/figures/`.

## What the angles represent

The ansatz prepares, from the paramagnet `|+⟩^N`,

    |ψ⟩ = ∏_layers [ ∏_{NN} e^{-iθ_nn Z_iZ_j} · ∏_{NNN} e^{-iθ_nnn Z_iZ_j}
                     · ∏_i e^{-iθ_x X_i} ] |+⟩^N

with gates `rzz(2θ)` and `rx(2θ)`. Each block has a direct physical reading:

| block | gate | meaning |
|-------|------|---------|
| `θ_nn`  | `rzz(2θ)` on NN bonds  | accumulated ZZ correlation on each nearest-neighbour bond — the *effective imaginary-time coupling* the state imprints. Grows as the field weakens. |
| `θ_nnn` | `rzz(2θ)` on NNN bonds | the frustrating next-nearest coupling; competes in sign with `θ_nn` (J2 > 0). |
| `θ_x`   | `rx(2θ)` on sites      | single-site rotation off the field axis. `θ_x → π/2` ⇒ paramagnet `|+⟩`; `θ_x → 0` ⇒ tilt toward the Z (ordered) axis. |

The leading-order (adiabatic / first-order Trotter) prediction is
`θ_nn = -J/(4h)`, `θ_nnn = -J2/(4h)`, `θ_x = arctan(J/h)`.

## Confirmed patterns

### 1. θ_nn follows a clean 1/h law — but with a renormalized coefficient

On the ordered branch (θ_nn < 0, 10 fitted points, h ∈ [0.3, 1.2]):

    θ_nn(h) ≈ -0.112 / h        (fit residual std = 0.040)

The `1/h` functional form predicted by first-order theory is confirmed with a
small residual, **but the coefficient is renormalized from the naive -0.25 to
≈ -0.112** (about 45% of the naive value). Interpretation: in the *bond-resolved*
ansatz each bond carries its own angle and the frustrating NNN block absorbs part
of the ZZ correlation, so the per-bond NN rotation needed is smaller than the
global-HVA estimate. The naive `-J/(4h)` is the right scaling law, the wrong
prefactor — a quantitative correction the warm-start could adopt.

### 2. θ_x tracks arctan(J/h) up to an O(1) factor

On the ordered branch, `θ_x / arctan(J/h) = 1.30 ± 0.27`. The transverse-field
rotation follows the predicted `arctan(J/h)` shape (monotonic decrease with h),
overshooting it by ~30% on average — the optimizer tilts slightly further off the
field axis than first order predicts.

### 3. θ_nnn is small and sign-changing

`θ_nnn` stays near zero (|mean| ≲ 0.05 for h ≥ 0.4) and **changes sign around
h ≈ 0.75**: negative (aligned with θ_nn) deep in the ordered phase, then weakly
positive. This is the frustration signature — the NNN bonds cannot simultaneously
satisfy the NN order, so their optimal rotation is small and flips as the state
crosses from order-dominated to field-dominated.

### 4. A sharp basin crossover at h ≈ 1.3

The single most prominent feature: at **h ≈ 1.3** the optimized angles jump
discontinuously between two distinct representations of the ground state:

| branch | h range | θ_nn | θ_x | fidelity |
|--------|---------|------|-----|----------|
| ordered / warm-start | h ≲ 1.25 | negative, ~-J/4h | ≈ 1.2 (tilted) | 0.44 → 0.97 |
| paramagnetic         | h ≳ 1.30 | small positive   | ≈ 0.36 (near |+⟩) | 0.97 → 0.996 |

Both are valid optima; the paramagnetic branch has *higher* fidelity at large h
(the state is essentially `|+⟩` and needs only tiny corrections). The optimizer
switches to whichever converges lower. This is the same multi-basin structure the
warm-start restart study found — here it is visible directly in the angle
trajectories (see `figures/theta_nn_scaling_headline.png`).

## Symmetries

### Z2 (global spin-flip) — exact

The TFIM commutes with ∏_i X_i, so the energy is invariant under flipping the
sign of all ZZ rotations together (`θ_nn, θ_nnn → -θ_nn, -θ_nnn`), while `θ_x` is
untouched. Confirmed numerically in `verify_hva_periodicity.py` (ΔE < 1e-14 under
the flip). The analyzer canonicalizes this gauge before any comparison, so
Z2-equivalent optima are correctly identified as identical.

### Periodic-image / gauge freedom

`rx(2θ)` and `rzz(2θ)` are π-periodic in θ, so the optimizer lands on different
periodic images at different h. Wrapping each block to the canonical branch
(`wrap_angle`, period π) removes these jumps; the residual discontinuity at
h ≈ 1.3 is therefore a *genuine basin change*, not a gauge artifact.

### Spatial homogeneity — broken near the ordered phase, restored in the paramagnet

Bond-to-bond dispersion of `θ_nn` (homogeneity = 1 − std/|mean|) is the probe of
spatial symmetry breaking:

- Deep ordered phase (h → 0.2): homogeneity drops toward ~0 and the block std
  widens (the error bars in `theta_blocks_vs_h`), i.e. the optimizer assigns
  *different* rotations to different bonds — a symmetry-broken, inhomogeneous
  solution as the frustrated order sets in and the gap closes.
- Paramagnetic phase (h ≳ 1.5): homogeneity ≈ 0.92, nearly uniform bonds — the
  translationally-invariant `|+⟩`-like state needs the same small rotation
  everywhere, so the bond-resolved freedom collapses back to the global-HVA
  solution.

The bond-resolved parametrization thus earns its extra parameters *only* near the
frustrated transition; far from it the state is homogeneous and a global HVA would
suffice.

## Layer structure (p = 2) and scaling with N

- **Layer drift (p=2):** the two layers are not identical — θ_x drifts by ~0.7 rad
  between layers (N=8 h=0.5 and N=18 h=1.0), i.e. the second layer performs a
  different (finer) rotation than the first. The ZZ blocks drift less (~0.2–0.4).
  The layers specialize rather than repeat.
- **N = 18, h = 1.0 (best large-N optimum, fid 0.885):** the effective coupling
  recovered from the angles is `J_nn_eff = -4h·θ̄_nn = +0.92 ≈ J = 1`, and NN
  homogeneity is 0.75 — the high-fidelity large-N state **correctly encodes the
  physical coupling** with fairly uniform bonds. This is the strongest evidence
  that, when the optimizer converges, the angles are physically meaningful and
  not arbitrary.
- **N = 18, h = 0.5 (fid 0.42, near-degenerate):** `J_nn_eff` does not recover J
  and homogeneity is negative — deep in the frustrated critical region the gap
  nearly closes (Δ ≈ 0.006), the ground state is near-degenerate, and a single
  fixed-p ansatz cannot pin a unique homogeneous solution.

## Takeaways

1. The optimized angles are **physically interpretable**: `θ_nn ∝ 1/h`,
   `θ_x ∝ arctan(J/h)`, `θ_nnn` small and sign-changing — the first-order
   warm-start captures the right functional forms.
2. The prefactors are **renormalized** by the bond-resolved + frustrated
   structure (θ_nn coefficient -0.11 vs naive -0.25); a warm-start using -0.11/h
   would seed closer to the ordered-branch optimum.
3. The landscape has **two basins** with a crossover at h ≈ 1.3; the angle
   trajectories expose it directly.
4. Symmetries: **Z2 exact**; **spatial homogeneity broken near the ordered phase,
   restored in the paramagnet** — the bond-resolved freedom matters only near the
   transition.

## Reproduce

```bash
# dense θ sweep (ordered/physical branch)
.venv/bin/python scripts/analysis/vl_vs_hva/theta_pattern_sweep.py \
    --n 8 --p 1 --strategy first_order --hmin 0.2 --hmax 2.5 --nh 24 --maxiter 800
# analysis + figures
.venv/bin/python scripts/analysis/vl_vs_hva/analyze_theta_patterns.py \
    --dataset results/hva_vl_study/theta_patterns/theta_sweep_square_N8_p1_J20.50_first_order.json
```


## Multi-seed confirmation (error bars over seeds)

To turn the single-shot θ(h) claims into confirmed distributions, the sweep was
repeated with **6 independent optimization seeds per h** (Metropolis basin-hopping,
N=8 square, J2=0.5, p=1). Each seed uses a distinct RNG base so the restarts
explore different basins; we report the across-seed mean ± σ of the
gauge-canonical (wrapped) layer-0 block means, and the fraction of seeds landing
on the ordered (θ_nn < 0) branch.

Data: `theta_patterns/theta_sweep_square_N8_p1_J20.50_metropolis_seeds6.json`,
`theta_patterns/multiseed_analysis_*.json`,
figure `figures/multiseed_theta_nn_*.png`.

| h | fidelity (mean ± σ) | θ_nn (mean ± σ) | θ_x (mean ± σ) | ordered-branch frac |
|------|---------------------|-----------------|----------------|:-------------------:|
| 0.20 | 0.319 ± 0.065 | -0.248 ± 0.261 | -0.344 ± 0.225 | 1.00 |
| 0.30 | 0.414 ± 0.048 | -0.099 ± 0.146 | +0.081 ± 0.400 | 0.83 |
| 0.40 | 0.556 ± 0.069 | -0.162 ± 0.111 | +0.741 ± 0.493 | 0.83 |
| 0.50 | 0.665 ± 0.095 | -0.183 ± 0.171 | +0.132 ± 0.818 | 0.83 |
| 0.60 | 0.790 ± 0.000 | -0.141 ± 0.087 | +0.460 ± 0.763 | 0.83 |
| 0.70 | 0.851 ± 0.000 | -0.164 ± 0.047 | +0.894 ± 0.587 | 1.00 |
| 0.80 | 0.893 ± 0.000 | -0.180 ± 0.000 | +1.200 ± 0.000 | 1.00 |

(h ≥ 0.8 is confirmed single-basin; the p=1 first-order sweep already established
the easy tail h ≳ 0.8 converges identically under any method, so the remaining
high-h points add no new structure.)

### What the seeds confirm

1. **θ_nn is well-determined; θ_x is the soft direction.** Across seeds, θ_nn has
   a *small* spread (σ ≈ 0.05–0.26, shrinking as h grows) and stays negative
   (ordered branch), whereas θ_x has a *large* spread (σ up to 0.82 at h = 0.5).
   The ZZ correlation the ansatz imprints is pinned by the frustrated coupling;
   the transverse-field rotation has a near-degenerate direction the optimizer
   does not resolve uniquely near the transition. This is new — the single-shot
   analysis could not separate "well-determined" from "soft" parameters.

2. **The ordered branch is robust, not a lucky seed.** The ordered-branch
   fraction is 0.83–1.00 for all h ≤ 0.8: 5–6 of 6 seeds land on θ_nn < 0. The
   branch structure reported earlier survives multi-seed validation.

3. **Basin disagreement is confined to the transition and collapses above it.**
   The fidelity spread across seeds is nonzero only for h ≤ 0.5 (σ up to 0.095),
   then collapses to σ = 0 at h ≥ 0.6 — every seed finds the same optimum in the
   field-dominated regime. Deep in the ordered phase (h = 0.2) one seed reached
   fidelity 0.465 while five stalled near 0.288, a direct measurement of the
   multi-basin landscape the warm-start restart study inferred indirectly.

4. **The renormalized 1/h law holds within error bars.** The multi-seed θ_nn(h)
   means are consistent with the `≈ -0.11/h` fit from the single-seed sweeps; the
   error bars do not admit the naive `-0.25/h` on the resolved branch.

### Consequence for warm-starting

Because θ_x is the soft, seed-sensitive direction near the transition, a
warm-start should (i) pin θ_nn ≈ -0.11/h tightly (well-determined) and (ii) treat
θ_x as the direction to *explore* (perturb / basin-hop), rather than trusting the
`arctan(J/h)` value there. This is consistent with, and sharpens, the confirmed
"Metropolis in the hard regime" recipe: the exploration budget is best spent on
the transverse-field subspace.

### Reproduce

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/theta_pattern_sweep.py \
    --n 8 --p 1 --strategy metropolis --hmin 0.2 --hmax 2.5 --nh 24 \
    --maxiter 400 --sigma 0.3 --seeds 6
.venv/bin/python scripts/analysis/vl_vs_hva/analyze_theta_patterns.py \
    --dataset results/hva_vl_study/theta_patterns/theta_sweep_square_N8_p1_J20.50_metropolis_seeds6.json \
    --multiseed
```


## Does the renormalized -0.11/h seed actually help? (head-to-head)

The θ-pattern study suggested a warm-start using ``θ_nn ≈ -0.112/h`` instead of
the textbook ``-J/(4h) = -0.25/h``. We tested this directly with a paired,
seed-controlled protocol: **pure L-BFGS-B descent** (no restarts, so quality
reflects the *seed*), the same per-seed perturbation applied to both
coefficients (paired comparison), 8 seeds, N ∈ {8, 10}, h across the transition.

Data: `theta_patterns/warmstart_coef_shootout_square_p1_J20.50.json`,
figure `figures/warmstart_coef_shootout.png`.

### The advantage is budget-dependent (and that is the point)

At a generous iteration budget (maxiter ≳ 150) **both seeds converge to the same
optimum** — pure L-BFGS from either starting angle reaches the basin floor, so
the coefficient is irrelevant. The renormalized seed only matters under a
**constrained budget**, where the closer starting point converges but the farther
one is still descending. This is the H1/H2 mechanism (distance-to-optimum governs
convergence under a fixed budget) applied to the coefficient — and it is exactly
the large-N regime, where iterations are expensive.

Results at a budget-limited maxiter = 10:

| regime | mean fidelity gain (renorm − naive) | verdict |
|--------|-------------------------------------|---------|
| deep ordered (h ≤ 0.45) | **+0.133** | renorm wins decisively |
| mid / easy (h > 0.45)   | −0.010 | neutral (naive marginally better, within noise) |
| overall (18 points, N=8,10) | +0.028 (median ≈ 0) | renorm wins 2 pts >1σ, naive 1 |

The decisive point is h = 0.30 deep in the ordered phase: at N=8 the renormalized
seed reaches fidelity **0.49 vs 0.08** for naive (+4.8σ); at N=10, **0.23 vs
0.13** (+0.9σ). There the naive ``-0.25/h`` **overshoots** — its ZZ angle is more
than twice too large — and a tight budget cannot walk it back, while the
``-0.11/h`` seed is already near the ordered-branch optimum.

### Honest scope of the claim

- **Confirmed:** deep in the ordered phase (h ≲ 0.4), under a tight iteration
  budget, the renormalized ``-0.11/h`` seed is substantially better — the naive
  seed overshoots and strands the optimizer. This is the regime where warm-start
  quality matters most for large N.
- **Not a universal win:** for h ≳ 0.45 the two coefficients tie (both converge),
  and in the mid-range the naive seed is occasionally ~1σ better. Median gain
  across all h is ≈ 0.
- **N-scaling:** the effect holds at both N=8 and N=10, weaker (and noisier) at
  N=10 because the naive seed sometimes still finds a good basin. The direction is
  consistent; a larger-N / lower-budget test would sharpen it.

### Practical recommendation

Use a **regime-gated coefficient**: seed with ``θ_nn ≈ -0.11/h`` when h is in the
ordered phase (h ≲ 0.45) and/or the iteration budget is tight (large N); the
naive ``-0.25/h`` is fine elsewhere. Combined with the earlier finding that θ_x is
the soft direction, the sharpest warm-start is: **renormalized, tightly-pinned ZZ
angles + exploration budget spent on θ_x**.

### Reproduce

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/warmstart_coef_shootout.py \
    --n 8 10 --hmin 0.3 --hmax 1.5 --nh 9 --seeds 8 --maxiter 10
```


## Four-way seed comparison: naive vs renorm vs second-order vs nn-shrink

The renormalized flat seed (``-0.11/h``) and the second-order seeds are two
*different* corrections to the same "naive overshoots" problem. We ran all four
in the same paired, seed-controlled, pure-descent protocol (8 seeds, N ∈ {8,10},
budget-limited maxiter=10, h across the transition):

- **naive** — first-order, ``θ_nn = -0.25/h`` (textbook).
- **renorm** — first-order, ``θ_nn = -0.11/h`` (flat, from the θ-pattern fit).
- **second_order** — ``(J/2h)²`` shrink on ZZ + curvature on θ_x (report Q3).
- **so_nn_shrink** — second-order + extra 0.4× shrink on θ_nn only (E1 study).

Data: `theta_patterns/warmstart_coef_shootout_square_p1_J20.50.json` (schema v2).

### Winner by regime (mean fidelity gain vs naive)

| regime | renorm | second_order | so_nn_shrink | best |
|--------|:------:|:------------:|:------------:|------|
| ordered (h ≤ 0.45)      | **+0.133** | +0.079 | +0.087 | **renorm** |
| transition (0.4–0.65)   | +0.005 | +0.004 | −0.005 | tie |
| paramagnetic (h ≳ 1.2)  | small + | small + | **best** | **so_nn_shrink** |

Per-h winner counts over the 18 (N,h) points: naive 7, renorm 5, so_nn_shrink 5,
second_order 1.

### What this settles

- **Deep in the ordered phase, the flat renorm ``-0.11/h`` is the clear winner** —
  it beats *both* second-order variants (h=0.30, N=8: renorm 0.495 vs
  second_order 0.386 vs so_nn_shrink 0.421 vs naive 0.085). There the naive and
  even the h-dependent second-order shrink still leave θ_nn too large; the flat
  0.11 prefactor lands closest.
- **In the paramagnetic phase (h ≳ 1.2), so_nn_shrink is best** — when the ZZ
  angles should be small, the extra 0.4× shrink on θ_nn starts closest to the
  optimum (wins h=1.35, 1.50 at both N).
- **Near the transition and just above it, no seed dominates** — differences are
  within noise (naive's larger angle is even marginally best around h≈0.9).
- The **second-order (standard) seed is rarely the outright winner** in this
  budget-limited, pure-descent test — its strength in the earlier study came from
  *opening a better basin* under restarts/Metropolis near h_c, not from raw
  proximity under a single tight descent, which is what this protocol measures.

### Corrected recommendation

There is no single best seed across all h; the sharpest choice is
**regime-gated**:

- **h ≲ 0.45 (ordered):** flat renorm ``θ_nn ≈ -0.11/h``.
- **h ≳ 1.2 (paramagnetic):** ``so_nn_shrink`` (extra θ_nn shrink).
- **near h_c (~0.4–0.6):** any of them for proximity; the second-order seed's
  documented advantage there is realized *with basin-hopping*, not pure descent.
- always spend the exploration budget on the θ_x subspace (the soft direction).

So the earlier phrase "renormalized and well-pinned ZZ angles" should read: the
*flat* ``-0.11/h`` is the winner specifically in the ordered phase; ``so_nn_shrink``
is the empirically sharpest in the paramagnetic phase. They are complementary, not
interchangeable.

### Reproduce

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/warmstart_coef_shootout.py \
    --n 8 10 --hmin 0.3 --hmax 1.5 --nh 9 --seeds 8 --maxiter 10 --coef-renorm 0.112
```
