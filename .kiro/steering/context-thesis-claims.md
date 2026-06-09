---
inclusion: fileMatch
fileMatchPattern: "**/thesis*,documentation/analysis/09*,documentation/analysis/08*,**/*.tex,documentation/analysis/12*"
---

# Thesis Claims Context (invoke with #context-thesis-claims)

> Pre-digested context for thesis writing: validated claims, table index, evidence chain.

## Core Thesis Contributions (Chapter 5)

### Contribution 1: GNN-Warm-Started HVA Pipeline
- Pipeline achieves ΔE/gap < 5% for h ≥ h_min across all tested topologies.
- MPNN generalizes across h-values (interpolation, not extrapolation).
- Zero additional QPU cost for warm-start (classical prediction only).

### Contribution 2: Unsupervised Phase Detection from θ_opt
- PCA of θ_opt(h) detects h_c with Δh=0.25 (zero extra QPU runs).
- |∂θ/∂h| corroborates D1 weight gradient (Δh=0.18 agreement).
- Works for chain_1d; requires h-grid covering h_c.

### Contribution 3: PEA-ZNE as Universal Mitigation
- +94.4% error reduction, R²=0.998, 18/18 wins vs gate-folding.
- Validated on ALL 4 topologies: chain (+97%), ladder (+91%), heavy_hex (+98%), triangular (+97%).
- ~50% QPU overhead justified by 4.6× improvement over GF.

### Contribution 4: GNN-QEM Zero-Shot Transfer
- 100% improvement rate on unseen heavy_hex (+72.3% error reduction).
- Graph IS essential for predictive mode (GNN 100% vs MLP 67% vs Linear 0%).
- NOT composable with PEA (alternative, not complement).

### Contribution 5: Cross-N Generalization
- Train N=40+80 → predict N=50,60,70,100: 25/25 PASS (0.16% mean ΔE/gap).
- BatchNorm harmful for cross-N on symmetric topologies (discovery + fix).
- GNN extrapolates to N=100 beyond training (0.18%), beats scipy 2.6×.

## Thesis Tables Index (Chapter 5)

| Table | Content | Source |
|-------|---------|--------|
| 5.1 | Best configs per topology (N=6, p=2) | `09_thesis_tables.md` |
| 5.2 | Best configs per topology (N=10, p=2) | `09_thesis_tables.md` |
| 5.3 | Cross-topology comparison (top-15 per topology) | `09_thesis_tables.md` |
| 5.4 | p=1 vs p=2 comparison | `09_thesis_tables.md` |
| 5.5 | ZNE boundary analysis | `09_thesis_tables.md` |
| 5.6 | Experiment verdicts (V8 confirmed/rejected) | `09_thesis_tables.md` |
| 5.7 | Heisenberg failure analysis (V9) | `binnacle-heisenberg-extension.md` |
| 5.8 | S-series results | `binnacle-s-series-results.md` |
| 5.9 | Tier 1 extensions | `12_tier1_session_results.md` |
| 5.10 | Hardware rehearsal findings | `11_hardware_rehearsal_findings.md` |
| 5.11 | Gate-folding vs PEA-ZNE | `binnacle-gate-folding-zne.md` |
| 5.12 | PEA cross-topology (6 experiments) | ZNE_CROSS_TOPO results |
| 5.13 | GNN-QEM cross-topology | `binnacle-gnn-qem-validation.md` |
| 5.14 | GNN-QEM ablation | `ablation_no_enoisy_results.json` |
| 5.15 | Unsupervised phase detection (PCA + derivative) | `binnacle-theta-pca-unsupervised-detection.md` |
| 5.16 | MPS scaling (N=40/50/80) | `binnacle-mps-scaling.md` |
| 5.17 | Cross-N zero-shot GNN | `binnacle-cross-n-zero-shot.md` |
| 5.18 | Failure mode summary (174 runs) | `project-status.md` |
| 5.19 | Scaling law validation | `binnacle-mps-scaling.md` |
| 5.20 | Hardware deployment tiers | `HARDWARE_DEPLOYMENT_SPEC.md` |
| 5.21 | Negative results (scientific contributions) | `validated-decisions.md` |

## Key Statistics (for claims)

| Claim | Value | Evidence | Stat sig |
|-------|-------|----------|----------|
| Experiments completed | 22 confirmed, 8 rejected, 2 failed | project-status.md | — |
| Pipeline runs | 210+ across 5 topologies | project-status.md | — |
| Useful-outcome rate | 93% (28/30) | project-status.md | — |
| PEA vs GF | 4.6× better | ZNE_CROSS_TOPO | t=46.32, p<10⁻¹⁹ |
| GNN-QEM zero-shot | +72.3% | cross_topology_results.json | t=13.28, p<10⁻⁶ |
| PEA on triangular | +96.8% | PEA_TRIANGULAR | t=111.22, 9/9 wins |
| Cross-N GNN | 25/25 PASS, 0.16% | zero_shot_v3 results | 3 seeds |
| Affine overshoot | 0% in 102 records | Affine audit | — |
| MPS N=80 | 0.08% ΔE/gap | scaling_N80 | 5/5 h-points |

## Negative Results (valid scientific contributions)

| Finding | Why it matters |
|---------|---------------|
| HVA is TFIM-specific (V9: 30 runs) | Delineates ansatz applicability |
| No cross-topology transfer (S2) | Each topology needs own training |
| DyPP redundant (F1) | Warm-start already near-optimal |
| Naive ensemble UQ not calibrated (G2) | Motivates MC-Dropout |
| Physics-informed loss HURTS at N=10 (C1) | Only helps with full h-range |
| GNN-QEM not composable with PEA | Clarifies deployment strategy |
| Noise-aware training fails (V7 5B) | Shot noise corrupts targets |

## Figures (21 vector PDFs in `documentation/thesis_figures/`)

Generate with: `make figures-thesis` (PDF 300dpi) or `make figures` (PNG).

## DO NOT

- Cite worklog files (superseded data, incorrect counts).
- Claim fidelity-based success on hardware (use ΔE/gap + phase label).
- Claim GNN correction is non-linear (it's 99.96% linear with E_noisy).
- Present h_min shifts as tunable (they're physics limits).
- Mix CES-ZNE results with PEA-ZNE results without noting the strategy.
- Omit negative results — they are valid scientific contributions.

## Source Files

- #[[file:documentation/analysis/09_thesis_tables.md]]
- #[[file:documentation/analysis/08_summary.md]]
- #[[file:.kiro/knowledge/validated-decisions.md]]
- #[[file:analysis/10_key_findings_corrected.md]]
- #[[file:documentation/thesis_figures/]]
