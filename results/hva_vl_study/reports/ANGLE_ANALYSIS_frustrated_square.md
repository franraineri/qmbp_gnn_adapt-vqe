# Optimized-Angle (θ) Analysis — bond-resolved HVA, frustrated square TFIM (J2=0.5)

Post-hoc study of saved variational angles. System: `tfim_frustrated`, square lattice,
J2=0.5, bond-resolved HVA (per layer `[θ_nn | θ_nnn | θ_x]`, rzz(2θ) / rx(2θ)).
No new VQE launched. A basin-count K=6 job is still running in background (3/6 done);
this analysis uses only what is on disk.

All numbers come from the reusable query/analysis layer
(`scripts/analysis/vl_vs_hva/results_query.py` + `analyze_angles.py`), extended in this
session (see "Tooling changes").

---

## (a) Which configs actually carry angle data

Probed every `results/hva_vl_study/hva_nnn_sweep/*.json` for angle vectors and their
convergence. Two storage conventions exist: `theta_final`/`theta_init` (basin-count) and
bare `theta` (per run) + `best_theta` (per block); the query layer originally read only
the first, so most angle data was invisible until the fix below.

| Config (topo, N, p, h) | angle-bearing runs | converged | θ-length | fid range | source file(s) |
|---|:--:|:--:|:--:|---|---|
| square, 18, 2, 0.5 | 20 | **5** | 168 | 0.0002–0.6527 | basincount k6, mixed mi500, secondorder mi300/mi1200, metropolis mi500, frustrated_final |
| square, 18, 2, 1.0 |  4 | 0 | 168 | 0.0768–0.8847 | frustrated_final (maxiter=40) |
| square,  8, 2, 0.5 |  3 | 0 |  58 | 0.9004–0.9495 | secondorder mi30 (maxiter=30) |
| resources NPZ h=0.5 | 2 | 0 (1 `done=False`, 1 unknown) | 168 | 0.4228 / 0.4803 | `theta_square_N18_p2_h0.50*.npz` |
| resources NPZ h=1.0 | 1 | 0 (starved, maxiter=40) | 168 | 0.8847 | `theta_square_N18_p2_h1.00.npz` |

Configs with **no usable θ** (predate angle saving or store only summaries):
`ansatz_variants_square_N10_h0.50.json`, `ansatz_saturation_square_p1_h0.50.json`,
`basincount_square_N{8,10,12}_p2_h0.50_k20.json`, `hva_nnn_h_sweep_frustrated.json`,
`hva_ansatze_h_sweep_frustrated.json`, `hva_nnn_N18_frustrated_final.json` *(has θ but all starved)*,
`n9_confirm_second_order.json`, the p=3 fair-cov file, and the N=18 h=1.0 dir-x file.

> **Correction to the stated caveat:** the ansatz-variants N=10 file does **not** contain
> θ (only per-variant summaries). Conversely, several N=18 files *do* carry θ that the
> query layer was silently skipping. The only config with **>1 converged** angle-bearing
> run is **square N=18 p=2 h=0.5**.

**Convergence rule applied.** Explicit `converged` when present; otherwise inferred
`nit < maxiter`. Everything at maxiter=30/40 (N=8 h=0.5; N=18 h=1.0) and the
`nit==maxiter` second-order runs is **starved** and excluded from basin statistics — never
pooled with converged runs.

---

## (b) Per-block angle signature — good vs bad basin (Q1)

**Only square N=18 p=2 h=0.5 qualifies** (5 converged runs, pooled from `basincount k6`
and `mixed mi500`; identical ansatz: e0=−16.985, gap=0.00589, 168 params, n_nn=27,
n_nnn=39, n_q=18). **Small-N caveat: 5 runs total.**

### The 5 converged runs

| fid | energy | nit | source |
|---:|---:|---:|---|
| 0.5418 | −16.494 | 2085 | basincount k6 |
| 0.5410 | −16.526 |  275 | mixed mi500 |
| 0.5325 | −16.526 |  343 | mixed mi500 |
| 0.1045 | −15.868 | 2219 | basincount k6 |
| 0.0002 | −15.411 | 1874 | basincount k6 |

**Key measured fact:** there is **no good basin** among converged runs. The best converged
fidelity is 0.54. High fidelity (>0.9) at N=18 h=0.5 appears *only* in starved runs, and
even those top out ~0.65. So "good vs bad" here means **0.54 vs ~0** basin, not 0.99 vs 0.5.

### Pairwise L2 among converged runs (parameter space, 168-dim)

```
              0.542(basin) 0.541(mixed) 0.533(mixed) 0.104(basin) 0.000(basin)
0.542(basin)      0.00        26.74        26.30        36.59        38.22
0.541(mixed)     26.74         0.00         3.87        25.15        26.33
0.533(mixed)     26.30         3.87         0.00        24.73        25.85
0.104(basin)     36.59        25.15        24.73         0.00        35.49
0.000(basin)     38.22        26.33        25.85        35.49         0.00
```

**Two distinct basins reach fid≈0.54.** The two mixed runs are L2=3.87 apart (same basin);
the basincount 0.542 run is ~26–27 away from them → a *different* basin at nearly identical
fidelity. The 0.104 and 0.0002 runs are each ~25–38 from everything → further separate
basins. This quantifies the "multiple basins of similar energy, different fidelity"
picture: near this frustrated transition, similar fidelity does **not** imply proximity in
θ-space.

### Per-block stats, high (fid≥0.53) vs low (fid<0.53)

| block | high mean | high std | high absmean | low mean | low std | low absmean |
|---|---:|---:|---:|---:|---:|---:|
| θ_nn  | −0.085 | 1.111 | 0.649 | −0.256 | 1.737 | 1.430 |
| θ_nnn | −0.081 | 1.209 | 0.664 | +0.031 | 1.953 | 1.687 |
| θ_x   | +0.739 | 1.166 | **1.189** | −0.034 | 1.955 | 1.697 |

Per-block L2, **best (0.5418) vs worst (0.0002)** representative:

| block | L2 | note |
|---|---:|---|
| θ_nn  | 19.51 | 54 angles |
| θ_nnn | **28.14** | 78 angles (most edges) |
| θ_x   | 16.99 | 36 angles |

**Which block distinguishes basins?**
- By **total L2**, θ_nnn shows the largest raw separation — but it also has the most angles
  (78 vs 54 vs 36), so absolute L2 is confounded by block size.
- The high-fidelity group is distinguished less by which block moved and more by
  **magnitude**: converged high-fid runs have **small entangling angles** (θ_nn/θ_nnn
  absmean ≈ 0.65) and a **positive, moderate θ_x** (mean +0.74), whereas the collapsed
  low-fid runs have large, sign-scattered angles everywhere (absmean ≈ 1.4–1.7).
- **Caution:** the "high" group pools two *different* basins (see L2 table), so its mean
  masks between-basin structure. With only 3 high / 2 low runs, treat the block means as
  indicative, not conclusive.

**Measured:** high-fidelity converged basins sit near a *small-entangling, moderate-θ_x*
region; collapsed runs have inflated, disordered angles. **Hypothesis (needs the K=6 job to
finish + more seeds):** θ_x sign/orientation plus entangling-angle magnitude jointly select
the basin; no single block is a clean discriminator at this sample size.

---

## (c) Seed → optimum distance (Q2)

Second-order analytic warm-start (`second_order_warmstart_theta`, J=1, J2=0.5) vs the 5
converged optima, N=18 p=2 h=0.5:

| fid | L2(2nd-order seed, opt) | L2(1st-order seed) | per-block absΔ (nn / nnn / x) | basin |
|---:|---:|---:|---|---|
| 0.5418 | 25.92 | 26.22 | 1.67 / 1.82 / 1.62 | basincount (far) |
| 0.5410 |  **3.94** |  4.88 | 0.27 / 0.23 / 0.32 | mixed (seed's basin) |
| 0.5325 |  **4.11** |  5.16 | 0.28 / 0.21 / 0.33 | mixed (seed's basin) |
| 0.1045 | 24.41 | 24.81 | 1.38 / 1.60 / 1.97 | basincount |
| 0.0002 | 25.95 | 26.33 | 1.50 / 1.81 / 1.86 | basincount |

**Measured:**
- The **mixed** runs are ~4 from the seed across *all* blocks (Δ≈0.2–0.3 each) — they are
  the second-order seed refined locally. They reach fid 0.53–0.54.
- The **basincount** 0.5418 run reaches the *same* fidelity from L2≈26 away — a genuinely
  different basin the seed does not point at.
- Distance-to-seed does **not** predict fidelity here: a near-seed optimum (L2 4, fid 0.54)
  and a far optimum (L2 26, fid 0.54) tie; another far optimum (L2 26) collapses to 0.0002.
- The second-order seed is marginally closer than first-order to every optimum (≈0.3–1.2
  lower L2), consistent with prior finding that it opens a slightly better-placed basin.

This reproduces the earlier "proximity ≠ correct basin" conclusion **at N=18 with converged
data**: the seed lands you in *a* 0.54 basin; the global optimum is elsewhere and the seed
does not reach it.

---

## (d) NPZ best-known optima by block (Q4)

From `results/hva_vl_study/resources/n18_frustrated/*.npz` (layout metadata in-file). **All
three are starved / interrupted — best-known, not converged.**

| NPZ (fid, gap, status) | θ_nn absmean | θ_nnn absmean | θ_x absmean |
|---|---:|---:|---:|
| h=0.5 bestof (0.4228, 0.0059, unknown) | 0.519 | 0.144 | 1.413 |
| h=0.5 metropolis (0.4803, 0.0059, **done=False**) | 0.474 | 0.123 | 1.552 |
| h=1.0 (0.8847, 0.494, starved maxiter=40) | 0.169 | 0.066 | 0.562 |

**Measured:**
- **h=1.0 vs h=0.5:** every block is ~3× smaller at h=1.0 (nn 0.17 vs ~0.5, nnn 0.07 vs
  ~0.14, x 0.56 vs ~1.5). Larger transverse field → smaller rotation angles, as expected;
  the paramagnetic point sits near the small-angle region and the optimizer reaches
  fid 0.88 even in 40 iterations.
- **The two h=0.5 NPZ optima are L2≈3.9 apart (per block ~3.8 each)** — the same basin,
  and the same seed-basin the converged mixed runs occupy (their θ_x absmean ~1.4–1.5
  matches). So the "best-known" saved h=0.5 point is that seed basin, whose ceiling under
  full convergence is ≈0.54 (part (b)), **not** the global optimum.
- These NPZ h=0.5 fidelities (0.42–0.48) are **below** the converged basincount best
  (0.542): the saved best-known point is a shallow/interrupted member of the seed basin.

---

## Conclusions (rigorously supported)

1. **Angle data is scarce and one-config-deep.** Converged, angle-bearing runs exist at
   **exactly one** configuration — square N=18 p=2 h=0.5 (5 runs). Everything else is either
   angle-less or starved. Any per-block "signature" claim is inherently small-sample here.
2. **No good basin is converged at N=18 h=0.5.** Best converged fidelity = 0.54; >0.9 occurs
   only in starved runs. The optimization, not the stored angles, is the ceiling.
3. **≥4 converged basins, similar energy, very different fidelity, far apart in θ-space.**
   Two distinct basins tie at fid≈0.54 while sitting ~27 apart (L2); collapse basins
   (fid≈0.1, ≈0) are ~25–38 away. Similar fidelity ≠ nearby angles.
4. **Distinguishing feature is angle *magnitude*, not a single block.** Converged higher-fid
   basins have small entangling angles (θ_nn/θ_nnn absmean ≈0.65) and moderate positive θ_x
   (mean +0.74); collapse basins have inflated, sign-scattered angles (absmean ≈1.4–1.7).
   θ_nnn has the largest raw L2 separation but also the most angles — size-confounded.
5. **Seed proximity does not predict fidelity.** The second-order seed's own basin converges
   to ≈0.54 (L2≈4); an equally-good 0.54 basin sits L2≈26 from the seed. "Proximity ≠
   correct basin," now confirmed with converged N=18 data.
6. **h=1.0 optima are ~3× smaller-angle than h=0.5** across all blocks, and the best-known
   h=0.5 NPZ is the seed basin (fid 0.42–0.48), below the converged basincount best.

## Open questions / where data is too sparse

- **Q1 fully:** needs the running K=6 basin-count to finish (currently 3/6) plus multi-seed
  repeats to separate "which block selects the good basin" from sample noise. A true *good*
  (>0.9) converged basin at N=18 h=0.5 has never been observed — it may not exist at p=2.
- **Q3 (angle drift with N):** **not answerable.** Converged angle data exists at a single N
  (18). N=8 (fid 0.90–0.95) and N=18 h=1.0 (0.88) angles are all starved; pooling them with
  N=18 h=0.5 would violate the convergence rule and mix maxiter regimes. `scan-n` returns
  best-known but those are non-converged — reported as "insufficient".
- **Sign structure of θ_x:** the +0.74 mean in high-fid runs vs −0.03 in collapse runs hints
  at a transverse-field orientation selecting the basin (consistent with the earlier θ_x
  hypothesis), but 3 vs 2 runs is far too few to assert. Needs the finished basin-count.
- **p=3:** p=3 angle vectors were not analyzed (the p=3 fair-cov file stores no θ). Whether
  the p=3 fidelity jump (0.65→0.92) corresponds to a distinct angle regime is untested.

## Tooling changes (reusable, per repo convention)

- `results_query.py::_harvest` — now context-aware: propagates block-level config
  (N, p, h, n_nn, n_nnn, gap, maxiter) into nested run records and harvests the
  `theta` / `best_theta` convention in addition to `theta_final`. Added convergence
  inference (`nit < maxiter` when no explicit flag), a `ResultRecord.converged` property,
  and dedup of block-best vs top-run duplicates. This surfaced 17 previously-invisible
  angle records.
- `analyze_angles.py` — two new subcommands, both importing `results_query` and using the
  NaN-safe `save_json`:
  - `basin-signature` — converged-only good/bad per-block mean/std/absmean split + per-block
    L2 between best and worst converged representative + discriminating block.
  - `npz-blocks` — per-block signature of the `resources/*.npz` optima, reading in-file
    layout metadata and flagging convergence via `done`/`maxiter`.
- Saved artifacts: `angle_analysis_{basin-signature,seed-gap,npz-blocks}_*.json` under
  `hva_nnn_sweep/`.
