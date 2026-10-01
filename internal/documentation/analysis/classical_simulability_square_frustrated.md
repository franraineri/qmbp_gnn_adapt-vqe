# Classical simulability limits — 2D square frustrated TFIM (quench dynamics)

Concise reference. All figures measured on our runs (ED + sparse `expm_multiply`,
infidelity threshold 0.01 vs the exact state) unless a source is cited.
Last updated: 2026-10-02.

Evidence tags used throughout: **[SOLID]** = measured + cross-checked;
**[WEAK]** = extrapolation / few points / unverified assumption. The hypothesis
(§6) is only as strong as its [WEAK] items — they are the open work, listed
explicitly in §7 so nothing is over-claimed.

---

## Critical point (J2 = 0.5) — [SOLID]

- Our data (N=10): minimum gap at h ≈ 0.3, max |d²E/dh²| at h ≈ 0.6.
- Literature: **Γc ≈ 0.28**, 2nd-order VBS → paramagnet transition
  ([arXiv:1901.00278](https://arxiv.org/html/1901.00278v1)).
- **h = 0.5 sits essentially on the critical region**, not far from it.

So our default quench `h_prep=0.5 → h₂(high)` **leaves** the critical region
toward the paramagnet; it does not cross the transition inward. To study a real
phase crossing, quench **toward h ≈ 0.3** (from the paramagnet). Cross-checked by
both our data and the cited literature.

---

## Two definitions of "classical limit"

| | Microstate (fidelity → 1) | Macrostate (observables ~1%) |
|---|---|---|
| χ required | explodes with t (volume law) | small, saturates |
| Reference | [arXiv:2310.08567](https://arxiv.org/html/2310.08567v3) | — |

The microstate crossover is the hard limit; observables stay cheap because the
half-chain entropy saturates (finite size + 2D area law in the bulk).

---

## Measured crossover — N=10, J2=0.5, dt=0.05 (χ up to 64) — [SOLID]

t* = Trotter step where a χ-MPS first exceeds infidelity 0.01 vs the exact state.
These quenches evolve the **exact** ground state (no ansatz), so they are a clean
measurement of the MPS microstate limit, independent of any preparation error.

| Quench | Crosses transition? | S_sat | t*(χ=16) | t*(χ=32) |
|--------|--------------------|-------|----------|----------|
| 3.0 → 0.3 | yes (strong) | 2.75 | **9** | never |
| 2.5 → 0.3 | yes | 2.71 | **10** | never |
| 0.5 → 3.0 | no (leaves crit.) | 1.91 | 27 | never |

**N=10 verdict: no real classical limit, even crossing the transition.**
Crossing raises entanglement (S_sat 1.9 → 2.75) and pulls the χ=16 crossover from
27 to ~9 steps, but **χ=32 tracks the exact microstate for the whole window
(t = 6.0, 120 steps)**. The exact statevector is 16 KB (2¹⁰·16 B). The system is
too small for the volume law to explode: half-chain (5 qubits) caps S at
5·ln2 ≈ 3.47 and finite-size saturation keeps it at 2.75.

> **Caveat on χ vs S.** The χ needed at crossover is NOT `e^{S_sat}`: our own data
> show χ=32 ≫ e^{2.75} ≈ 16 and χ=8 already crosses early. The crossover χ depends
> on the full Schmidt spectrum, not the entropy alone, so `e^S` is only a loose
> lower-bound heuristic — do not quote it as the required χ.

---

## Projection to larger N — [WEAK] (two-point extrapolation, do not over-trust)

S_sat(N) ≈ 0.14·N + 0.5 is fit from **only two measured points (N=10, 14)**. Two
points define a line trivially — there is **no evidence yet that S_sat(N) is
linear**. Treat the N≥16 row as a hypothesis to test, not a result.

| N | S_sat (extrap.) | χ exact 1D = 2^(N/2) | exact SV | status |
|---|-----------------|----------------------|----------|--------|
| 14 | 2.48 (measured) | 128 | 0.3 MB | measured |
| 16 | ~2.76 (extrap.) | 256 | 1 MB | **unverified** |
| 18 | ~3.04 (extrap.) | 512 | 4 MB | **unverified** |
| 20 | ~3.32 (extrap.) | 1024 | 17 MB | **unverified** |

Columns deliberately omit a "χ for observables" number: per the §4 caveat, we do
not have a defensible S→χ map yet. ED (exact SV) stays trivial through N=20.

2D penalty [SOLID, from literature]: a 1D MPS over a square lattice pays the 2D
area law → χ must grow ~exp(width) for fixed fidelity ([iMPS→iPEPS crossover ~11
sites, Heisenberg, arXiv:1705.03222](https://arxiv.org/html/1705.03222v1)).

---

## 5. Conclusions (what the data supports today)

1. **[SOLID] ED stays trivial** through N=20 (17 MB) — the classical limit is NOT
   statevector memory. We always have the exact reference in ideal simulation.
2. **[SOLID] The microstate crossover t* falls when crossing the transition**
   (χ=16: 27 → 9 steps at N=10) — crossing h_c is the right regime to stress MPS.
3. **[SOLID] N=10 exposes no real classical limit**: χ=32 tracks the microstate
   for the whole window; finite-size caps S_sat.
4. **[WEAK] S_sat grows with N** (two points) and the 2D area law should amplify
   the collapse of t* — but neither the growth law nor the N≥16 crossover is
   measured yet.

---

## 6. Hypothesis (stated conservatively)

> In the 2D square frustrated TFIM, a 1D MPS of moderate χ reproduces the **exact
> microstate** of quench dynamics only up to a finite t* that **shrinks when the
> quench crosses the transition** (measured at N=10) and is expected to **collapse
> with N** via time-linear entanglement growth, aggravated by the 2D area law
> (χ ~ exp(width)). The classically-hard, quantum-relevant regime is therefore:
> **N ≥ 16, quenching through h ≈ 0.3, judged on the microstate (not averaged
> observables, not quenches that leave the critical region).**

Scope limits we will NOT over-claim:
- Nothing here claims hardware quantum advantage — t_noise is a separate budget
  (see `../hardware/nighthawk_r2_gate_limits_and_noise.md`); ideal simulation only.
- Averaged observables (M_z, S) stay classically cheap; the claim is about the
  microstate, which is the honest hard target.

---

## 7. Open assumptions to verify before the hypothesis is firm

1. **Preparation ceiling near h_c [WEAK, risk].** The bond-resolved ansatz has a
   fidelity ceiling ≈0.96 near the critical point at p=2 (gap~0.01-0.02): no seed
   reaches >0.96 there (measured, N=10). A quench prepared near h_c inherits that
   4% infidelity from t=0, which may contaminate the dynamics. The crossing quench
   is prepared in the **paramagnet (h=3.0)** where fidelity is high — this must be
   **measured, not assumed**. The clean microstate crossovers in §3 avoid this by
   evolving the EXACT ground state; a prepared-state run is the realistic test.
2. **S_sat(N) growth law [WEAK].** Need N=12 and N=16 measured to replace the
   two-point line with a credible trend (linear? saturating?).
3. **Crossover χ vs N [WEAK].** Need t*(χ) at N=12/16/18 with χ up to 128 to show
   that χ=64/128 finally fails to track the microstate at larger N (the limit that
   does NOT exist at N=10).
4. **S→χ map [WEAK].** Replace the `e^S` heuristic with the measured crossover χ
   as a function of S / Schmidt spectrum.

Items 1-3 are exactly what the overnight crossing-quench sweep (N=12/16/18,
h_prep=3.0 → h₂≈0.3, χ up to 128, recording prep fidelity) is designed to settle.

<!-- Links -->
[arxiv-1901]: https://arxiv.org/html/1901.00278v1
[arxiv-2310]: https://arxiv.org/html/2310.08567v3
[arxiv-imps]: https://arxiv.org/html/1705.03222v1
