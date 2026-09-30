# VL vs HVA — same-scenario comparison (square, N=9, p=2, J2=0.5)

Auto-generated from the JSON artifacts — do not edit by hand. Regenerate with `generate_vl_hva_comparison_report.py`. This is a descriptive side-by-side (which method fits which scenario), not a competition.

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`
- Lattice: `square`, `N = 9`, HVA depth `p = 2`
- HVA fidelity is the best-of-by-energy VQE ceiling (θ captured in the artifact).
- VL numbers are Haiqu-measured (default settings unless noted).
- 2q/1q/total/depth counted on the transpiled circuit (`rz, sx, x, cx`).
- θ-search cost and hardware executability are intentionally out of scope.

- Source artifact: `results/hva_vl_study/hva_nnn_sweep/compare_hva_nnn_vs_vl_square_N9_p2.json`

## 1. Same-scenario table

| h | method | ansatz | F | ΔE/gap | total gates | 2q | 1q | depth |
|------|--------|--------|------:|-------:|------------:|---:|---:|------:|
| 0.50 | HVA | nnn (create_bond_resolved_frustrated) | 0.9574 | 0.5722 | 273 | 104 | 169 | 73 |
| 0.50 | VL | default L2/F20 (Haiqu-measured) | 0.8086 | 3.2019 | 207 | 30 | 177 | 53 |
| 1.00 | HVA | nnn (create_bond_resolved_frustrated) | 0.9964 | 0.0306 | 273 | 104 | 169 | 73 |
| 1.00 | VL | default L2/F20 (Haiqu-measured) | 0.8999 | 0.6278 | 215 | 30 | 185 | 53 |

## 2. VL quality curve per scenario (num_layers × fine_tuning)

VL fidelity/2q as its two knobs vary (from the quality sweep). Shows the range VL spans at each scenario, next to the single HVA nnn p=2 point.

### h = 0.50 (gap 0.2238)

| method | config | F | 2q |
|--------|--------|------:|---:|
| HVA nnn | p=2 | 0.9574 | 104 |
| VL | L2/F20 | 0.8093 | 30 |
| VL | L2/F50 | 0.8359 | 30 |
| VL | L4/F20 | 0.8844 | 60 |
| VL | L4/F50 | 0.9085 | 60 |
| VL | L8/F20 | 0.9200 | 120 |
| VL | L8/F50 | 0.9431 | 120 |

### h = 1.00 (gap 0.8601)

| method | config | F | 2q |
|--------|--------|------:|---:|
| HVA nnn | p=2 | 0.9964 | 104 |
| VL | L2/F20 | 0.8999 | 30 |
| VL | L2/F50 | 0.9082 | 30 |
| VL | L4/F20 | 0.9547 | 60 |
| VL | L8/F20 | 0.9656 | 120 |
| VL | L4/F50 | 0.9665 | 60 |
| VL | L8/F50 | 0.9754 | 120 |

## 3. Per-scenario read (descriptive)

### h = 0.50 (gap 0.2238)

- HVA nnn (create_bond_resolved_frustrated): F = 0.9574, 104 CX, 273 total gates, best warm-start = warmstart:mixed.
- VL (default L2/F20 (Haiqu-measured)): F = 0.8086, 30 CX, 207 total gates.
- VL best (L8/F50): F = 0.9431, 120 CX.

### h = 1.00 (gap 0.8601)

- HVA nnn (create_bond_resolved_frustrated): F = 0.9964, 104 CX, 273 total gates, best warm-start = warmstart:metropolis.
- VL (default L2/F20 (Haiqu-measured)): F = 0.8999, 30 CX, 215 total gates.
- VL best (L8/F50): F = 0.9754, 120 CX.

## 4. Metric definitions

- `F` — exact statevector fidelity `|⟨ψ_exact|ψ_prep⟩|²` (measured).
- `ΔE/gap` — energy error of the prepared state normalized by the spectral gap.
- `total/2q/1q/depth` — counted on the transpiled circuit; HVA structural
  count uses fixed non-zero angles (θ-independent), VL as returned by Haiqu.
- `ansatz` — HVA generator set: nnn maps the J2 diagonals; nn is NN-only.
