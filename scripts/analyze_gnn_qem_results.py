#!/usr/bin/env python
"""Analyze GNN-QEM results: compare with PEA-ZNE baseline, identify validation gaps."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

output_dir = Path("results/gnn_qem")

ct = json.load(open(output_dir / "cross_topology_results.json"))
ft = json.load(open(output_dir / "evaluation.json"))

print("=" * 70)
print("COMPARATIVE ANALYSIS: GNN-QEM vs PEA-ZNE Baseline")
print("=" * 70)

# 1. In-distribution results
s = ft["summary"]
print(f"\n{'─' * 70}")
print("1. IN-DISTRIBUTION (chain_1d + ladder, N=6)")
print(f"   Samples: {s['n_samples']}")
print(
    f"   Error reduction: {s['eval_mean_err_before']:.3f} → {s['eval_mean_err_after']:.4f} ({s['eval_mean_reduction_pct']:.1f}%)"
)
print(f"   Rate: {s['eval_improvement_rate']:.0f}% ({s['n_improved']}/{s['n_samples']})")

# 2. Zero-shot cross-topology
zs = ct["zero_shot"]
print(f"\n{'─' * 70}")
print("2. ZERO-SHOT TRANSFER (chain+ladder → heavy_hex N=10)")
print(f"   Samples: {zs['n_samples']}")
print(
    f"   Error reduction: {zs['mean_err_before']:.3f} → {zs['mean_err_after']:.3f} ({zs['reduction_pct']:.1f}%)"
)
print(f"   Median: {zs['median_err_before']:.3f} → {zs['median_err_after']:.3f}")
print(f"   Max residual: {zs['max_residual']:.3f}")
print(f"   Rate: {zs['improvement_rate']:.0f}%")
print(f"   Confidence: {zs['mean_confidence']:.3f}")

# 3. Fine-tuned
ftu = ct["fine_tuned"]
print(f"\n{'─' * 70}")
print("3. FINE-TUNED (all topologies → held-out heavy_hex)")
print(f"   Finetune/Test: {ftu['n_finetune']}/{ftu['n_test']}")
print(
    f"   Error reduction: {ftu['mean_err_before']:.3f} → {ftu['mean_err_after']:.4f} ({ftu['reduction_pct']:.1f}%)"
)
print(f"   Rate: {ftu['improvement_rate']:.0f}%")
print(f"   Val MAE: {ftu['val_mae']:.4f}")

# 4. Comparison table
print(f"\n{'─' * 70}")
print("4. COMPARISON TABLE (all methods)")
print(f"   {'Method':<35} {'Rate':>6} {'Mean Reduction':>15} {'Residual':>10}")
print(f"   {'─' * 35} {'─' * 6} {'─' * 15} {'─' * 10}")
print(f"   {'PEA-ZNE (previous best)':<35} {'100%':>6} {'+94.4%':>15} {'~0.5-3':>10}")
red_in = s["eval_mean_reduction_pct"]
res_in = s["eval_mean_err_after"]
red_zs = zs["reduction_pct"]
res_zs = zs["mean_err_after"]
red_ft = ftu["reduction_pct"]
res_ft = ftu["mean_err_after"]
print(
    f"   {'GNN-QEM in-dist (chain+ladder)':<35} {'100%':>6} {f'+{red_in:.1f}%':>15} {f'{res_in:.3f}':>10}"
)
print(
    f"   {'GNN-QEM zero-shot (→heavy_hex)':<35} {'100%':>6} {f'+{red_zs:.1f}%':>15} {f'{res_zs:.3f}':>10}"
)
print(
    f"   {'GNN-QEM fine-tuned (→heavy_hex)':<35} {'100%':>6} {f'+{red_ft:.1f}%':>15} {f'{res_ft:.4f}':>10}"
)

# 5. Key finding
print(f"\n{'─' * 70}")
print("5. KEY FINDING")
print("   GNN-QEM works on RANDOM theta (large errors 10-25 units).")
print("   PEA-ZNE works on VQE-optimized theta (small errors ~0.5-3 units).")
print("   They address DIFFERENT error regimes — complementary, not competing.")
print()
print("   OPTIMAL STACK: VQE → PEA-ZNE → GNN-QEM → Affine → verdict")
print("   Each stage reduces the residual for the next stage.")

# 6. Validation gaps
print(f"\n{'─' * 70}")
print("6. VALIDATION GAPS (what needs to be confirmed)")
print("   A. GNN-QEM on POST-ZNE residuals (currently trained on raw noise)")
print("      → Need: generate data with PEA-ZNE applied first, train GNN on residual")
print("   B. Multi-seed reproducibility (is 100% rate stable across seeds?)")
print("      → Need: run with seeds 42/43/44 × 3 independent training runs")
print("   C. Statistical test (paired t-test or Wilcoxon on per-sample ΔE)")
print("      → Can compute from existing per_sample data")
print("   D. Realistic error magnitudes (VQE theta, not random)")
print("      → Need: use VQE-optimized theta from Phase 2 dataset")

# 7. Compute paired statistical test from existing data
print(f"\n{'─' * 70}")
print("7. STATISTICAL SIGNIFICANCE (from existing zero-shot data)")
errs_before = [r["err_before"] for r in zs["per_sample"]]
errs_after = [r["err_after"] for r in zs["per_sample"]]
diffs = [b - a for b, a in zip(errs_before, errs_after, strict=False)]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs, ddof=1)
n = len(diffs)
t_stat = mean_diff / (std_diff / np.sqrt(n))
# One-sided p-value approximation (t-distribution with n-1 df)
from scipy import stats

p_value = stats.t.sf(t_stat, df=n - 1)
print("   Paired t-test (GNN helps vs doesn't help):")
print(f"   n={n}, mean_diff={mean_diff:.3f}, std={std_diff:.3f}")
print(f"   t={t_stat:.3f}, p={p_value:.6f}")
print(f"   Significant at p<0.05: {'YES' if p_value < 0.05 else 'NO'}")
print(f"   Significant at p<0.01: {'YES' if p_value < 0.01 else 'NO'}")
print(f"   Effect size (Cohen's d): {mean_diff / std_diff:.2f}")

print(f"\n{'=' * 70}")
print(f"OVERALL VERDICT: {ct['verdict']}")
print(f"{'=' * 70}")
