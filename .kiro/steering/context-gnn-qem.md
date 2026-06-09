---
inclusion: fileMatch
fileMatchPattern: "**/gnn_qem*,**/predictors/gnn*,results/gnn_qem/**"
---

# GNN-QEM Context (invoke with #context-gnn-qem)

> Pre-digested context for GNN-based quantum error mitigation work.

## What's Done

- In-distribution: +99.4% error reduction (chain_1d + ladder, N=6/10).
- Cross-topology zero-shot: +72.3% on unseen heavy_hex (t=13.28, p<10⁻⁶).
- Ablation: Graph IS essential without E_noisy (GNN 100% vs MLP 67% vs Linear 0%).
- With E_noisy: correction is 99.96% linear — graph adds +11% precision only.
- Circuit selection (predictive mode): Spearman ρ=0.945, 100% binary accuracy.
- **NOT composable with PEA**: Post-ZNE, GNN regresses 15/15 points (over-corrects).

## Architecture

```python
# GINConv(3 layers, hidden=64), ~30K params
# Input: circuit graph (nodes=qubits, edges=2Q gates)
# Node features: [qubit_error_rate, degree, position_encoding]
# Edge features: [gate_error_rate, gate_type_onehot]
# Output: E_corrected (scalar)
```

## Two Operating Modes

| Mode | Input includes E_noisy? | Use case |
|------|------------------------|----------|
| **Correction** | YES | Post-execution error reduction (replaces ZNE) |
| **Predictive** | NO | Pre-execution circuit ranking/selection |

## Deployment Rules (CRITICAL)

```
PEA available?
  ├── YES → Use PEA (primary) + affine (always). Skip GNN-QEM entirely.
  └── NO  → Use GNN-QEM (correction mode) + affine (always).
```

- PEA and GNN-QEM are ALTERNATIVES, not complements.
- Both remove structured noise. After one removes structure, residual is shot noise.
- GNN trained on large errors (10-25 units) OVER-CORRECTS post-PEA residuals (0.01 units).

## Claim Reframing (from ablation)

- **Without E_noisy**: Graph captures noise propagation topology → essential for RANKING.
- **With E_noisy**: Correction is linear (E_corrected ≈ a·E_noisy + b). Graph adds regularization only.
- Thesis contribution: "GNN captures noise topology for predictive selection" (not "GNN corrects errors better than linear").

## DO NOT

- Apply GNN-QEM after PEA-ZNE (0% improvement, risk of regression).
- Claim GNN correction is non-linear (it's 99.96% linear with E_noisy).
- Train on post-ZNE residuals (domain mismatch — errors too small).
- Use MLP when topology varies (GNN wins 100% vs 67% without E_noisy).

## Source Files

- #[[file:src/qmbp_simulation/predictors/gnn_qem.py]]
- #[[file:documentation/binnacles/binnacle-gnn-qem-validation.md]]
- #[[file:results/gnn_qem/cross_topology_results.json]]
- #[[file:results/gnn_qem/ablation_no_enoisy_results.json]]
- #[[file:results/gnn_qem/post_zne_validation.json]]
- #[[file:results/gnn_qem/vqe_realistic_results.json]]
