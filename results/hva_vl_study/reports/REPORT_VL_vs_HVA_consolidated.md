# VL vs HVA — consolidated comparison (frustrated square TFIM, J2=0.5)

Auto-generated from the JSON artifacts across all N, h, and method variants — do not edit by hand. Regenerate with `generate_vl_hva_consolidated_report.py`. Descriptive (which method fits which scenario), not a competition.

- Only runs with **F > 0.5** are shown (weaker runs hidden per scenario).
- Per row: `F`, `|ΔE|`, `ΔE/gap`, 2q gates, total gates, depth, loader/χ, converged.
- HVA gate counts: transpiled `rz, sx, x, cx`, θ-independent structural count (recomputed deterministically from the saved θ where the sweep omitted them).
- VL gate counts: as returned by Haiqu. `loader` = vector_loading vs mps_loading; `χ` = MPS bond cap (mps only).
- HVA `converged`: per-restart contract where available (e.g. 4/4); `ceiling` = best-of-by-energy; `n/a (random)` = random-restart sweep. VL is a fixed zero-parameter circuit → `n/a (fixed circuit)`.
- θ-search cost and hardware executability are out of scope.
- ⚠️ At N=18 VL uses `mps_loading` (target = χ-capped MPS), not the `vector_loading` used at N≤10 — flagged per row.
- Where available, a **dual-target** matrix scores both methods against the exact ground state AND the MPS χ=64 reference (all re-simulated locally), quantifying that the target choice barely moves F.
- An **iso-2q** view compares fidelity at matched 2q budgets (VL interpolated on its quality curve) — the fair resource-normalized comparison.
- Scenarios where VL was only run at its default config (no quality sweep) are moved to an **appendix** — the effort is asymmetric, so they don't belong in the primary same-effort comparison.

## N = 6, h = 0.50 (gap 0.2506, E0 -5.0158)

> ⚠️ Only **HVA** present above F>0.5 at this scenario — no same-scenario cross-method comparison possible.

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | p2_base | — | 0.9957 | 0.0140 | 0.056 | 52 | — | — | 0/2 |
| HVA | nnn p=1 (fair-conv best) | — | 0.9088 | 0.1998 | 0.797 | 26 | 87 | 29 | 6/6 seeds |

## N = 9, h = 0.50 (gap 0.2238, E0 -7.7532)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=2 (ceiling) | — | 0.9574 | 0.1281 | 0.572 | 104 | 273 | 73 | ceiling (best-of by energy) |
| HVA | nnn p=2 | — | 0.9523 | 0.1318 | 0.589 | 104 | 273 | 73 | n/a (random best-of) |
| HVA | nn p=2 | — | 0.7696 | 0.2975 | 1.330 | 48 | 189 | 49 | n/a (random best-of) |
| HVA | nnn p=1 | — | 0.7358 | 0.5378 | 2.403 | 52 | 150 | 41 | n/a (random best-of) |
| HVA | nn p=1 | — | 0.7253 | 0.6316 | 2.822 | 24 | 108 | 32 | n/a (random best-of) |
| VL | L8/F50 | vector | 0.9431 | 0.2750 | 1.229 | 120 | — | — | n/a (fixed circuit) |
| VL | L8/F20 | vector | 0.9200 | 0.4033 | 1.802 | 120 | — | — | n/a (fixed circuit) |
| VL | L4/F50 | vector | 0.9085 | 0.4210 | 1.881 | 60 | — | — | n/a (fixed circuit) |
| VL | L4/F20 | vector | 0.8844 | 0.5229 | 2.337 | 60 | — | — | n/a (fixed circuit) |
| VL | L2/F50 | vector | 0.8359 | 0.6290 | 2.811 | 30 | — | — | n/a (fixed circuit) |
| VL | L2/F20 | vector | 0.8093 | 0.7165 | 3.202 | 30 | — | — | n/a (fixed circuit) |
| VL | default L2/F20 | vector | 0.8086 | 0.7165 | 3.202 | 30 | 207 | 53 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 24 | 0.7253 | 0.8086 | VL |
| 30 | 0.7364 | 0.8359 | VL |
| 48 | 0.7696 | 0.8650 | VL |
| 52 | 0.7358 | 0.8714 | VL |
| 60 | 0.7691 | 0.9085 | VL |
| 104 | 0.9574 | 0.9170 | HVA |
| 120 | 0.9574 | 0.9431 | HVA |

## N = 9, h = 1.00 (gap 0.8601, E0 -11.1802)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=2 (ceiling) | — | 0.9964 | 0.0263 | 0.031 | 104 | 273 | 73 | ceiling (best-of by energy) |
| HVA | nnn p=2 | — | 0.9954 | 0.0302 | 0.035 | 104 | 273 | 73 | n/a (random best-of) |
| HVA | nn p=2 | — | 0.9393 | 0.1830 | 0.213 | 48 | 189 | 49 | n/a (random best-of) |
| HVA | nnn p=1 | — | 0.9332 | 0.2515 | 0.292 | 52 | 150 | 41 | n/a (random best-of) |
| HVA | nn p=1 | — | 0.9144 | 0.3060 | 0.356 | 24 | 108 | 32 | n/a (random best-of) |
| VL | L8/F50 | vector | 0.9754 | 0.1981 | 0.230 | 120 | — | — | n/a (fixed circuit) |
| VL | L4/F50 | vector | 0.9665 | 0.2432 | 0.283 | 60 | — | — | n/a (fixed circuit) |
| VL | L8/F20 | vector | 0.9656 | 0.2599 | 0.302 | 120 | — | — | n/a (fixed circuit) |
| VL | L4/F20 | vector | 0.9547 | 0.3093 | 0.360 | 60 | — | — | n/a (fixed circuit) |
| VL | L2/F50 | vector | 0.9082 | 0.4741 | 0.551 | 30 | — | — | n/a (fixed circuit) |
| VL | default L2/F20 | vector | 0.8999 | 0.5399 | 0.628 | 30 | 215 | 53 | n/a (fixed circuit) |
| VL | L2/F20 | vector | 0.8999 | 0.5399 | 0.628 | 30 | — | — | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 24 | 0.9144 | 0.8999 | HVA |
| 30 | 0.9207 | 0.9082 | HVA |
| 48 | 0.9393 | 0.9361 | HVA |
| 52 | 0.9332 | 0.9423 | VL |
| 60 | 0.9428 | 0.9665 | VL |
| 104 | 0.9964 | 0.9658 | HVA |
| 120 | 0.9964 | 0.9754 | HVA |

## N = 10, h = 0.50 (gap 0.0061, E0 -9.3693)

> ⚠️ Only **HVA** present above F>0.5 at this scenario — no same-scenario cross-method comparison possible.

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | p2_full_ref | — | 0.9893 | 0.0221 | 3.599 | 112 | — | — | 1/2 |
| HVA | p2_full_ref | — | 0.9892 | 0.0221 | 3.598 | 112 | — | — | 2/2 |
| HVA | p2_full_ref (nnn=15) | — | 0.9892 | — | — | 112 | — | — | masked |
| HVA | p1_half_nn_rx | — | 0.9891 | 0.0234 | 3.818 | 82 | — | — | 4/4 |
| HVA | p2_topk_f0.7 (nnn=10) | — | 0.9865 | — | — | 92 | — | — | masked |
| HVA | p2_topk_f0.5 (nnn=8) | — | 0.9861 | — | — | 84 | — | — | masked |
| HVA | p2_topk_f0.3 (nnn=4) | — | 0.9670 | — | — | 68 | — | — | masked |

_2 run(s) hidden (F ≤ 0.5)._

## N = 18, h = 0.50 (gap 0.0059, E0 -16.9850)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=3 (fair-conv best) | — | 0.9241 | 0.0819 | 13.897 | 396 | 918 | 147 | 2/6 seeds |
| HVA | p3_base | — | 0.9132 | 0.0937 | 15.890 | 396 | — | — | 0/1 |
| VL | L8/F50 | mps χ=64 | 0.8625 | 0.5382 | 91.326 | 264 | 1768 | 181 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 264 | 0.9241 | 0.8625 | HVA |
| 396 | 0.9241 | 0.8625 | HVA |

**Dual-target fidelity** — both methods re-simulated locally against the same reference. `⟨exact|MPS χ=64⟩² = 0.999933`, so the two references coincide to that overlap.

| method | F vs exact | F vs MPS χ | source θ/QPY |
|--------|-----------:|-----------:|--------------|
| HVA | 0.6552 | 0.6554 | ansatz_variants_square_N18_h0.50.json:p2_base |
| VL | 0.8605 | 0.8611 | square_N18_h0.50_J20.50_L8_F50_mps.qpy |

## N = 18, h = 1.00 (gap 0.4937, E0 -23.0280)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=2 (fair-conv best) | — | 0.9314 | 0.2608 | 0.528 | 264 | 630 | 106 | 6/6 seeds |
| VL | L8/F50 | mps χ=64 | 0.7924 | 1.3939 | 2.824 | 264 | 1767 | 181 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 264 | 0.9314 | 0.7924 | HVA |

**Dual-target fidelity** — both methods re-simulated locally against the same reference. `⟨exact|MPS χ=64⟩² = 0.999876`, so the two references coincide to that overlap.

| method | F vs exact | F vs MPS χ | source θ/QPY |
|--------|-----------:|-----------:|--------------|
| HVA | 0.9314 | 0.9314 | n18_square_N18_p2_faircov_mi3000.json:second_iso_large |
| VL | 0.7885 | 0.7891 | square_N18_h1.00_J20.50_L8_F50_mps.qpy |

## Appendix — VL default-only scenarios (effort-asymmetric)

Here VL has ONLY its default config (`L2/F20`, no quality sweep), so HVA's tuned ceiling is compared against VL's floor. Not a same-effort comparison; shown for completeness. Re-run the VL quality sweep at these h to make them primary.

## N = 9, h = 2.00 (gap 2.6252, E0 -19.3383)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=2 (ceiling) | — | 0.9997 | 0.0043 | 0.002 | 104 | 273 | 73 | ceiling (best-of by energy) |
| VL | default L2/F20 | vector | 0.9613 | 0.3887 | 0.148 | 30 | 216 | 53 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 30 | 0.9997 | 0.9613 | HVA |
| 104 | 0.9997 | 0.9613 | HVA |

**Dual-target fidelity** — both methods re-simulated locally against the same reference. `⟨exact|MPS χ=64⟩² = 1.000000`, so the two references coincide to that overlap.

| method | F vs exact | F vs MPS χ | source θ/QPY |
|--------|-----------:|-----------:|--------------|
| HVA | 0.9997 | 0.9997 | compare_hva_nnn_vs_vl_square_N9_p2.json |
| VL | 0.9613 | 0.9613 | tfim_frustrated_square_n9_p2_h2.00_j20.50_vl.qpy |

## N = 9, h = 3.00 (gap 4.5394, E0 -27.9787)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | nnn p=2 (ceiling) | — | 1.0000 | 0.0010 | 0.000 | 104 | 273 | 73 | ceiling (best-of by energy) |
| VL | default L2/F20 | vector | 0.9870 | 0.1721 | 0.038 | 30 | 216 | 53 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 30 | 1.0000 | 0.9870 | HVA |
| 104 | 1.0000 | 0.9870 | HVA |

**Dual-target fidelity** — both methods re-simulated locally against the same reference. `⟨exact|MPS χ=64⟩² = 1.000000`, so the two references coincide to that overlap.

| method | F vs exact | F vs MPS χ | source θ/QPY |
|--------|-----------:|-----------:|--------------|
| HVA | 1.0000 | 1.0000 | compare_hva_nnn_vs_vl_square_N9_p2.json |
| VL | 0.9870 | 0.9870 | tfim_frustrated_square_n9_p2_h3.00_j20.50_vl.qpy |

## N = 10, h = 1.00 (gap 0.5023, E0 -12.4739)

| method | variant | loader/χ | F | \|ΔE\| | ΔE/gap | 2q | total | depth | converged |
|--------|---------|----------|------:|-------:|-------:|---:|------:|------:|-----------|
| HVA | p2_full_ref | — | 0.9621 | 0.1274 | 0.254 | 112 | — | — | 2/2 |
| HVA | adapt step6 (25 bonds) | — | 0.8723 | 0.4223 | 0.841 | 50 | — | — | masked |
| HVA | adapt step5 (23 bonds) | — | 0.8654 | 0.4192 | 0.835 | 46 | — | — | masked |
| HVA | adapt step3 (19 bonds) | — | 0.8615 | 0.4541 | 0.904 | 38 | — | — | masked |
| HVA | adapt step2 (17 bonds) | — | 0.8421 | 0.4622 | 0.920 | 34 | — | — | masked |
| HVA | p1_topk_nnn (nnn=8) | — | 0.8356 | 0.4734 | 0.942 | 42 | — | — | 2/2 |
| HVA | adapt step4 (21 bonds) | — | 0.8348 | 0.5984 | 1.191 | 42 | — | — | masked |
| HVA | p1_topk_nnn (nnn=8) | — | 0.8289 | 0.6770 | 1.348 | 42 | — | — | 2/2 |
| HVA | adapt step1 (15 bonds) | — | 0.8059 | 0.6832 | 1.360 | 30 | — | — | masked |
| HVA | adapt step0 (13 bonds) | — | 0.7715 | 0.7945 | 1.582 | 26 | — | — | masked |
| HVA | adapt step7 (27 bonds) | — | 0.5956 | 2.4739 | 4.925 | 54 | — | — | masked |
| HVA | p1_pruned (nnn=15) | — | 0.5956 | 2.4739 | 4.925 | 56 | — | — | 2/2 |
| HVA | p1_full_ref (nnn=15) | — | 0.5956 | 2.4739 | 4.925 | 56 | — | — | 2/2 |
| VL | default L2/F20 | vector | 0.8966 | 0.6180 | 1.230 | 34 | 237 | 53 | n/a (fixed circuit) |

**Iso-2q view** — fidelity at matched 2q budgets (VL interpolated on its quality curve; HVA from its runs).

| 2q budget | HVA F | VL F | better |
|----------:|------:|-----:|--------|
| 26 | 0.7715 | 0.8966 | VL |
| 30 | 0.8059 | 0.8966 | VL |
| 34 | 0.8421 | 0.8966 | VL |
| 38 | 0.8615 | 0.8966 | VL |
| 42 | 0.8356 | 0.8966 | VL |
| 46 | 0.8654 | 0.8966 | VL |
| 50 | 0.8723 | 0.8966 | VL |
| 54 | 0.5956 | 0.8966 | VL |
| 56 | 0.5956 | 0.8966 | VL |
| 112 | 0.9621 | 0.8966 | HVA |

## Coverage matrix (methods present per scenario)

| N | h | HVA variants | VL variants |
|---|------|-------------:|------------:|
| 6 | 0.50 | 2 | 0 |
| 9 | 0.50 | 5 | 7 |
| 9 | 1.00 | 5 | 7 |
| 9 | 2.00 | 1 | 1 |
| 9 | 3.00 | 1 | 1 |
| 10 | 0.50 | 9 | 0 |
| 10 | 1.00 | 13 | 1 |
| 18 | 0.50 | 2 | 1 |
| 18 | 1.00 | 1 | 1 |

