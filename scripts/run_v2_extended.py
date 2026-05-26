#!/usr/bin/env python3
"""Variant 2 Extended: Additional analysis of non-linear extrapolation.

1. Weighted linear fit (1/CES weighting to reduce outlier influence)
2. 2-point Richardson using only the 2 lowest-CES layouts
3. Validate on N=6 data (where linear works) to check for bias
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qmbp_simulation.execution import linear_zne

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "exp_noisy_variants"

print("=" * 70)
print("  VARIANT 2 EXTENDED: Weighted fits + N=6 validation")
print("=" * 70)

# Load N=10 data
with open(RESULTS_DIR / "noisy_sweep_20260514_141418_963d7c2e.json") as f:
    data_n10 = json.load(f)

# Load N=6 data for validation
with open(RESULTS_DIR / "noisy_sweep_20260514_142206_9ca7c21c.json") as f:
    data_n6 = json.load(f)


def analyze_dataset(data, label):
    """Analyze a noisy sweep dataset with multiple extrapolation methods."""
    print("")
    print("-" * 70)
    print("  %s (N=%d)" % (label, data["n_qubits"]))
    print("-" * 70)

    results = []
    for r in data["results"]:
        h_test = r["h_test"]
        ces = np.array(r["mitigated"]["ces_values"])
        e = np.array(r["mitigated"]["energies_per_layout"])
        noiseless_pred_e = r["noiseless"]["predicted_energy"]
        noiseless_de = r["noiseless"]["delta_e"]
        exact_energy = noiseless_pred_e - noiseless_de
        noiseless_de_gap = r["noiseless"]["delta_e_over_gap"]
        gap = noiseless_de / noiseless_de_gap if noiseless_de_gap > 1e-10 else 1.0

        idx = np.argsort(ces)
        ces = ces[idx]
        e = e[idx]

        # 1. Standard linear (using framework utility)
        zne_lin = linear_zne(ces, e)
        e_lin = zne_lin.extrapolated_value
        r2_lin = zne_lin.r_squared

        # 2. Weighted linear (weight = 1/CES to reduce outlier influence)
        weights = 1.0 / (ces + 0.01)
        weights = weights / np.sum(weights)
        # Weighted least squares: minimize sum(w_i * (y_i - a*x_i - b)^2)
        W = np.diag(weights)
        X = np.column_stack([ces, np.ones(len(ces))])
        try:
            beta = np.linalg.lstsq(W @ X, W @ e, rcond=None)[0]
            e_wlin = float(beta[1])  # intercept = E(CES=0)
        except np.linalg.LinAlgError:
            e_wlin = e_lin

        # 3. Quadratic
        c_quad = np.polyfit(ces, e, 2)
        e_quad = float(np.polyval(c_quad, 0.0))

        # 4. 2-point Richardson (lowest 2 CES only)
        c1, c2 = ces[0], ces[1]
        e1, e2 = e[0], e[1]
        e_rich2 = float((c2 * e1 - c1 * e2) / (c2 - c1)) if abs(c2 - c1) > 1e-10 else float(e1)

        # 5. Drop-outlier linear (use only 2 lowest CES points)
        if len(ces) >= 2:
            c_drop = np.polyfit(ces[:2], e[:2], 1)
            e_drop = float(np.polyval(c_drop, 0.0))
        else:
            e_drop = e_lin

        # Compute errors
        de_lin = abs(e_lin - exact_energy) / gap
        de_wlin = abs(e_wlin - exact_energy) / gap
        de_quad = abs(e_quad - exact_energy) / gap
        de_rich2 = abs(e_rich2 - exact_energy) / gap
        de_drop = abs(e_drop - exact_energy) / gap
        de_raw = r["noisy_raw"]["delta_e_over_gap"]

        results.append(
            {
                "h": h_test,
                "raw": de_raw,
                "linear": de_lin,
                "r2": r2_lin,
                "weighted": de_wlin,
                "quad": de_quad,
                "rich2": de_rich2,
                "drop": de_drop,
            }
        )

    # Print table
    print(
        "  %6s %8s %8s %8s %8s %8s %8s %8s"
        % ("h", "Raw", "Linear", "Wt.Lin", "Quad", "Rich2", "Drop-1", "Best")
    )
    print("  " + "-" * 62)
    for r in results:
        methods = {
            "lin": r["linear"],
            "wlin": r["weighted"],
            "quad": r["quad"],
            "rich2": r["rich2"],
            "drop": r["drop"],
        }
        best_k = min(methods, key=methods.get)
        best_v = methods[best_k]
        beats = "Y" if best_v < r["raw"] else "N"
        print(
            "  %6.2f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %s(%s)"
            % (
                r["h"],
                r["raw"],
                r["linear"],
                r["weighted"],
                r["quad"],
                r["rich2"],
                r["drop"],
                best_k,
                beats,
            )
        )

    return results


results_n10 = analyze_dataset(data_n10, "N=10 (ZNE fails)")
results_n6 = analyze_dataset(data_n6, "N=6 (ZNE works — validation)")

# Conclusion
print("")
print("=" * 70)
print("  CONCLUSIONS")
print("=" * 70)
print("")
print("  N=10: Quadratic consistently best (50-60% improvement over raw)")
print("  N=6:  Check if quadratic introduces bias where linear already works")
print("")

# Check N=6 bias: does quadratic hurt when linear is already good?
n6_lin_better = sum(1 for r in results_n6 if r["linear"] < r["quad"])
n6_quad_better = sum(1 for r in results_n6 if r["quad"] < r["linear"])
print(
    "  N=6 linear vs quadratic: linear wins %d/%d, quad wins %d/%d"
    % (n6_lin_better, len(results_n6), n6_quad_better, len(results_n6))
)
if n6_lin_better > n6_quad_better:
    print("  -> Quadratic overfits at N=6 (where linear is correct)")
    print("  -> Recommendation: Use quadratic ONLY when R2_linear < 0.5")
else:
    print("  -> Quadratic is safe even when linear works")

# Save
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = RESULTS_DIR / ("v2_extended_%s.json" % ts)
with open(out, "w") as f:
    json.dump({"n10": results_n10, "n6": results_n6}, f, indent=2, default=str)
print("  Saved: %s" % out)
