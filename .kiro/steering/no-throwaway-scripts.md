---
inclusion: fileMatch
fileMatchPattern: "**/scripts/**,**/analysis/**,**/project_health/**"
---

# Rule: Use Existing Scripts — Never Create One-Off Analysis Files

## CRITICAL RULE (ALWAYS ENFORCE)

When asked to analyze results, compute metrics, or investigate data:

1. **NEVER create a new Python script** for ad-hoc analysis.
2. **ALWAYS use the existing canonical scripts** with appropriate CLI parameters.
3. If the existing scripts don't cover the need, **extend them** (add a flag/mode) rather than creating a new file.
4. For quick one-off computations, use `/tmp/kiro_snippet.py` (ephemeral, never committed).

## Available Analysis Scripts (use these)

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/analysis/compute_h_frontier.py` | h_frontier vs N (chain_1d, linear fits) | `--threshold 0.05 --min-n 20 --json` |
| `scripts/analysis/compute_h_frontier_all.py` | h_frontier across ALL topologies/models | `--model tfim --topology heavy_hex` |
| `scripts/analysis/compute_h_frontier_models.py` | Per-model frontier (N=10 chain_1d) | (no args needed) |
| `scripts/analysis/compute_h_frontier_topologies.py` | Per-topology frontier (N=10) | (no args needed) |
| `scripts/analysis/analyze_all_phase3.py` | Phase3 MPNN results + anomalies | `--date 20260717 --n-qubits 40 --json` |
| `scripts/analysis/extract_theta_trajectories.py` | Extract θ_opt(h) from all runs | `--include-scaling` or `--only-scaling` |
| `scripts/analysis/theta_pca_phase_detection.py` | PCA + clustering on θ(h) | (no args) |
| `scripts/analysis/theta_derivative_analysis.py` | ∂θ/∂h peaks vs D1 weight gradient | (no args) |
| `scripts/analysis/extract_delta_e_fidelity.py` | ΔE + Fidelity summary table | (no args) |
| `scripts/analysis/check_delta_e_by_topo_p.py` | |ΔE| absolute by topology × p | (no args) |
| `scripts/analysis/check_matrix_gaps.py` | Which (N,p) data is missing | `--target-ns 20 40 60 --json` |
| `scripts/analysis/reanalyze_p2_filtered.py` | Re-analyze with h≥1.3 filter | (no args) |
| `scripts/analysis/verify_hva_periodicity.py` | Verify θ periodicities numerically | (no args) |

## How to Run

```bash
.venv/bin/python scripts/analysis/<script>.py [options]
```

## If You Need Something New

1. **Can an existing script do it with a new flag?** → Add `--new-flag` to the existing script.
2. **Is it a one-liner?** → Write to `/tmp/kiro_snippet.py`, execute, delete.
3. **Is it genuinely new reusable functionality?** → Add it to `project_health/analysis/` as a module, NOT as a standalone script.

## Rationale

Previously, 13+ one-off scripts accumulated in `scripts/analysis/` — each hardcoding specific file paths, duplicating functionality from canonical scripts, and becoming immediately stale. This creates maintenance debt and confusion about which script is authoritative.
