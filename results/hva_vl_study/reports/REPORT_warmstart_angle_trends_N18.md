# Warm-start angle trends at N=18 — findings + experiment plan

Analysis of the analytic warm-start that has led to our best N=18 fidelities, and
what the OPTIMIZED angles reveal about how to improve it. Every number here comes
from saved artifacts (re-simulated / read via `analyze_angles.py blocks` and the
fair-convergence JSONs) — no invented data.

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`, square lattice, p = 2.
- Key sources: `n18_square_N18_p2_faircov_mi3000.json` (h=1.0, 6 seeds converged),
  `analyze_angles.py blocks`, `qmbp_simulation.analysis.warmstart`.

---

## 1. The analytic warm-start ladder (how we got here)

| seed | formula (per block, per layer) | role |
|------|--------------------------------|------|
| first-order | `nn=-J/4h`, `nnn=-J2/4h`, `x=arctan(J/h)` | Trotter/adiabatic baseline |
| second-order | first-order + `(J/2h)²` shrink on ZZ, curvature on X | best analytic *pure* seed |
| second-order + perturbation | second-order seed + Gaussian noise σ, best-of | best RESULT so far |

At N=18 h=1.0 (fair-convergence, all converged):

| seed type | σ | block | F |
|-----------|---|-------|------:|
| second_iso_large | 0.3 | all | **0.9314** |
| second_iso_small | 0.1 | all | 0.9255 |
| second_dir_zz | 0.2 | zz | 0.9105 |
| second_dir_x | 0.2 | x | 0.9058 |
| second_pure | 0.0 | all | 0.8859 |
| first_pure | 0.0 | all | 0.8859 |

**The pure analytic seed is NOT optimal** — a large isotropic perturbation (σ=0.3)
lifts fidelity by +0.045 over the pure seed. The seed lands in a decent basin but
not the best one; noise is needed to escape it.

---

## 2. What the optimized angles reveal — RIGOROUS (E1, 8 configs)

Method: for each config `(N, h, p=2)`, take the HIGHEST-fidelity persisted θ,
re-verify its fidelity by re-simulation, and compare it to the analytic
second-order seed block-by-block, layer-by-layer, **controlling for the ZZ
sign-degeneracy** (report `|opt|/|seed|` magnitude ratio and `sign_flip_frac`,
not just signed means). Tool: `analyze_seed_vs_optimum.py` →
`seed_vs_optimum_square.json`; logic in `warmstart.seed_vs_optimum_report`.

> ⚠️ Two earlier hypotheses were REFUTED by the multi-config data:
> 1. "Optimal angles are heterogeneous / need a bigger uniform shrink" — artifact
>    of reading the σ=0.3-perturbed best (`second_iso_large`) whose std is residual
>    noise, not structure.
> 2. "There is a between-layer X ladder (layer-0 low-X, layer-1 high-X)" — that was
>    ONE sample (`second_pure`, N=18 h=1.0). Across 8 configs it does NOT hold: in
>    the high-fidelity cases the two layers' θ_x are similar (e.g. N=18 h=1.0 best:
>    L0 |x|=0.86, L1 |x|=0.95; N=9 h=1.0: L0 0.39, L1 0.43). No systematic ladder.

### 2.1 The sign-degeneracy control was essential

θ_x has LOW sign-flip vs the seed (6–22%) across configs — the transverse-field
direction is stable and matches the seed's sign. The large sign flips (33–100%)
are all in the ZZ blocks: RZZ(2θ)~RZZ(−2θ) up to a state phase, so signed ZZ means
are unreliable. **All trends below use magnitudes** `|opt|/|seed|`.

### 2.2 The one systematic, sign-safe trend

Across the high-fidelity configs (F > 0.93), by magnitude:

| block | `|opt|/|seed|` trend | reading |
|-------|----------------------|---------|
| θ_nn  | **0.32–0.96× (consistently < 1)** | seed OVER-estimates nn entanglement — optimum shrinks it |
| θ_nnn | 0.19–2.66× (erratic) | seed models the J2 diagonal WORST — no stable factor |
| θ_x   | 0.51–1.26× (near 1) | seed gets the transverse rotation ~right |

Concrete high-F examples (`|opt|/|seed|`, layer 0):

| config | F | nn | nnn | x |
|--------|---|---:|----:|--:|
| N=9 h=1.0 p2 | 0.9954 | 0.32× | 2.57× | 0.51× |
| N=18 h=1.0 p2 | 0.9314 | 0.96× | 0.92× | 1.15× |
| N=9 h=0.5 p2 | 0.9523 | 0.35× | 1.11× | 0.39× |
| N=18 h=0.5 p2 | 0.6552 | 0.50× | 1.53× | 0.89× |

### 2.3 The seed still helps and layers are similar

`‖θ_opt − θ_seed‖ ≈ ‖θ_seed‖` (e.g. N=18 h=1.0: 6.29 vs 4.92) — a real but bounded
move; θ=0 doesn't even start descending, so the analytic seed is genuinely useful.
In the good configs the two layers behave similarly (no ladder) — the seed's
layer-uniformity is NOT the problem.

### 2.4 Outlier: N=6 p=2

Ratios explode (×4.7–×10.65, flip 71–100%) — at N=6 the tiny Hilbert space has
many equivalent optima on different sign branches. Treated as an outlier, excluded
from trend fitting.

---

## 3. What this does and does NOT support

- **DOES support**: the seed over-estimates θ_nn (a stronger nn shrink would start
  closer to the optimum) — a single-parameter, testable calibration.
- **Does NOT support**: a per-layer schedule / X-ladder (refuted), or a uniform
  global shrink (nnn is erratic, x is fine).
- **Honest limit**: only θ_nn shows a clean, direction-consistent trend. nnn (the
  frustration term) is the least predictable and is likely where the real
  difficulty lives — consistent with the N=18 h=0.5 diagnosis (balanced,
  frustration-driven entanglement).

---

## 4. Experiment plan (revised after E1)

E1 (seed-vs-optimum, 8 configs) is DONE and refuted the layer-schedule idea. The
remaining plan is narrower and data-driven. All runs use the robust warm-start
(analytic seed + restarts, best-of by energy) and feed the scoreboard.

### E1 — Seed vs optimum, per config ✅ DONE

`analyze_seed_vs_optimum.py` → `seed_vs_optimum_square.json`. Findings in §2:
only θ_nn shows a clean, direction-consistent trend (optimum shrinks it,
`|opt|/|seed|` ≈ 0.3–0.5 in the strong cases); nnn erratic; x ≈ right; no layer
ladder. Sign-degeneracy controlled via magnitude + `sign_flip_frac`.

### E1b — Widen the sample (cheap, N≤12) — improves confidence

Several configs lack a clean persisted p=2 θ (N=10, N=12). Regenerate them at
N≤12 with the robust warm-start (fast), then re-run E1 so the θ_nn-shrink trend is
backed by more points across h ∈ {0.5, 1.0}. **Launch when machine load is low**
(currently two N=18 jobs are running — do not oversubscribe).

### E2 — Recalibrate the nn shrink — implemented + cheap-eval DONE

`second_order_nn_shrink_theta` (shrinks θ_nn only by `nn_extra_shrink`, default
0.4; nnn/x untouched). Cheap evaluation — fidelity AT the seed (statevector, no
VQE) scanning the shrink factor — via `eval_nn_shrink_seed.py` →
`nn_shrink_seed_eval_square.json`.

**Result (fidelity at the seed, best shrink factor per config):**

| config | std seed F | best-shrink F | best factor |
|--------|-----------:|--------------:|:-----------:|
| N=6 h=1.0 | 0.551 | 0.754 | 0.2 |
| N=8 h=1.0 | 0.406 | 0.681 | 0.2 |
| N=10 h=1.0 | 0.253 | 0.569 | 0.2 |
| N=12 h=1.0 | 0.156 | 0.518 | 0.2 |
| N=6 h=0.5 | 0.183 | 0.438 | 0.2 |
| N=8 h=0.5 | 0.044 | 0.235 | 0.2 |
| N=10 h=0.5 | 0.049 | 0.049 | 1.0 (no gain) |
| N=12 h=0.5 | 0.022 | 0.024 | 0.8 (marginal) |

**Findings:**
- At **h=1.0 (gapped)** the nn-shrink lifts the seed fidelity 2–3× at every N, with
  a consistent optimum near `s≈0.2` — a strong, systematic confirmation of E1's
  "seed over-estimates θ_nn" trend.
- At **h=0.5 (near-degenerate/frustrated)** the shrink helps only at small N
  (6, 8); at N=10, 12 it gives ~no gain. The frustrated regime's difficulty is NOT
  in θ_nn — consistent with E1 (nnn erratic) and the N=18 dirty-state diagnosis.
- Caveat: this measures the SEED fidelity (starting point), not the final
  optimized fidelity. A better seed reduces restart dependence; the full VQE
  comparison (below) is the confirming test.

### E2-full — VQE from the nn-shrink seed (heavy, when load drops)

Run the robust optimizer seeded with `second_order_nn_shrink_theta(s≈0.2)` vs the
standard seed at N≤12, both h. **Goal:** does the better seed reach equal/higher
final F with fewer restarts (esp. at h=1.0)? Deferred while N=18 jobs run.

### E3 — Directed vs isotropic perturbation at N=18 (medium)

`second_iso_large` (σ=0.3, all blocks) currently wins at N=18. Test whether the
E2 recalibrated seed + SMALL σ reaches the same F with fewer restarts — replacing
"lucky big random noise" with a deterministic informed seed. Run only after E2
shows promise at small N.

### Deliberately DROPPED (refuted by E1)

- Per-layer scheduled seed / X-ladder — not supported across configs.
- Uniform global shrink — nnn is erratic and x is already ~right.

### Priority

1. **E1b** (cheap, when load drops) — widen the θ sample at N≤12.
2. **E2** (cheap) — recalibrate nn shrink, the only data-supported change.
3. **E3** — validate at N=18 only if E2 helps at small N.

---

## 5. Open caveats

- The θ_nn-shrink trend is clean but the nnn (frustration) term is erratic — the
  seed's real weakness is the J2 diagonal, which is also where the hard physics
  lives (N=18 h=0.5 diagnosis: balanced, frustration-driven entanglement). A
  better seed may not fix nnn without addressing that entanglement structure.
- Sign-degeneracy is handled by magnitude comparison; if a future analysis needs
  signed structure, canonicalize signs by state overlap first.
- N=6 is an outlier (tiny Hilbert space, many equivalent optima) — excluded.
- E1b/E2/E3 must cover both h=0.5 (near-degenerate) and h=1.0 (gapped) — they
  behave differently and a seed tuned on one may hurt the other (the reason the
  second-order seed is already regime-gated).


---

## 6. Result: partial-3rd-layer beats full p3 at lower 2q (N=18 h=0.5)

Using the **compose-extend warm-start** (seed the p=2 prefix from the best known
p2 θ = 0.6552, extra blocks at near-identity, so the optimizer only refines the
new layer), with `--restarts 1 --maxiter 1500`:

| variant | F | 2q (CX) | converged | note |
|---------|------:|--------:|:---------:|------|
| p2_half_nn_rx | **0.9268** | **318** | 0/1 (floor) | p2 + half-nn RZZ + free RX |
| nnn p=3 (fair-conv best) | 0.9241 | 396 | 2/6 | best full p3 |
| p2_half_nn | 0.9124 | 318 | 0/1 (floor) | p2 + half-nn RZZ |
| p3_base | 0.8941 | 396 | 1/4 | full p3, weak warm-start |
| p2_base | 0.6552 | 264 | 4/4 | anchor |

**Findings:**
- **A partial third layer (half-layer of nn RZZ) beats a FULL third layer at ~20%
  fewer 2q.** `p2_half_nn_rx` reaches F=0.927 with 318 CX vs the best p3's 0.924
  with 396 CX — more fidelity, 78 fewer CX. The full third layer wastes 2q.
- **The free-RX trick holds at N=18 too**: `p2_half_nn_rx` (0.927) > `p2_half_nn`
  (0.912) at the SAME 318 CX (RX is 0-CX). Consistent with N=10.
- **The compose-extend warm-start was essential**: both started from the p2 floor
  (0.6552) and only improved; random restarts at this hostile gap-0.006 point
  land in bad basins.
- **Caveat**: both partial-p3 runs are FLOORS (converged 0/1 at maxiter=1500) —
  the true optima are ≥ these values, so the gap over p3 is a lower bound. More
  iterations would likely raise them further.

**Takeaway for "more fidelity, fewer 2q":** at N=18 h=0.5 the cost-optimal HVA
upgrade is **p2 + a half-layer of nn RZZ + a free RX block** (318 CX), not a full
p3 (396 CX). Persisted in the scoreboard; regenerate the consolidated report to
surface it in the iso-2q view.
