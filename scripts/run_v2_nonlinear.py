#!/usr/bin/env python3
"""Variant 2: Non-linear extrapolation of existing N=10 ZNE data.

Hypothesis: E(CES) may be exponential at N=10. Linear fit gives R²<0.05
but exponential/quadratic/Richardson may recover meaningful extrapolation.

This is pure post-processing — no quantum simulation needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qmbp_simulation.execution import linear_zne

RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "exp_noisy_variants"

data_path = RESULTS_DIR / "noisy_sweep_20260514_141418_963d7c2e.json"
with open(data_path) as f:
    data = json.load(f)

assert data["n_qubits"] == 10
print("=" * 70)
print("  VARIANT 2: Non-Linear Extrapolation of Existing N=10 ZNE Data")
print("  Hypothesis: Exponential/quadratic fit may work where linear fails")
print("=" * 70)

results = []
for r in data["results"]:
    h_test = r["h_test"]
    ces_values = np.array(r["mitigated"]["ces_values"])
    energies = np.array(r["mitigated"]["energies_per_layout"])

    # Exact energy from noiseless data
    noiseless_pred_e = r["noiseless"]["predicted_energy"]
    noiseless_de = r["noiseless"]["delta_e"]
    exact_energy = noiseless_pred_e - noiseless_de

    # Gap
    noiseless_de_gap = r["noiseless"]["delta_e_over_gap"]
    gap = noiseless_de / noiseless_de_gap if noiseless_de_gap > 1e-10 else 1.0

    # Sort by CES
    idx = np.argsort(ces_values)
    ces = ces_values[idx]
    e = energies[idx]

    # --- Linear fit (using framework utility) ---
    zne_lin = linear_zne(ces, e)
    e_lin = zne_lin.extrapolated_value
    r2_lin = zne_lin.r_squared

    # --- Quadratic fit (exact with 3 points) ---
    coeffs_quad = np.polyfit(ces, e, 2)
    e_quad = float(np.polyval(coeffs_quad, 0.0))

    # --- Exponential: E(CES) = a * exp(b * CES) + c ---
    try:

        def exp_model(x, a, b, c):
            return a * np.exp(b * x) + c

        p0 = [1.0, 0.1, np.min(e) - 1.0]
        popt, _ = curve_fit(exp_model, ces, e, p0=p0, maxfev=5000)
        e_exp = float(exp_model(0.0, *popt))
        exp_ok = True
    except (RuntimeError, ValueError):
        e_exp = float("nan")
        popt = [float("nan")] * 3
        exp_ok = False

    # --- Richardson (2-point, lowest CES pair) ---
    c1, c2 = ces[0], ces[1]
    e1, e2 = e[0], e[1]
    e_rich = float((c2 * e1 - c1 * e2) / (c2 - c1)) if abs(c2 - c1) > 1e-10 else float(e1)

    # ΔE/gap for each method
    de_lin = abs(e_lin - exact_energy) / gap
    de_quad = abs(e_quad - exact_energy) / gap
    de_exp = abs(e_exp - exact_energy) / gap if exp_ok else float("nan")
    de_rich = abs(e_rich - exact_energy) / gap
    noisy_raw_de_gap = r["noisy_raw"]["delta_e_over_gap"]

    print(f"\n  h={h_test:.2f}: CES={ces.tolist()}")
    print(f"    Exact E={exact_energy:.4f}, Gap={gap:.4f}")
    print(f"    Noisy raw:   ΔE/gap={noisy_raw_de_gap:.4f}")
    print(f"    Linear:      E(0)={e_lin:.4f}, ΔE/gap={de_lin:.4f}, R²={r2_lin:.4f}")
    print(f"    Quadratic:   E(0)={e_quad:.4f}, ΔE/gap={de_quad:.4f}")
    print(f"    Exponential: E(0)={e_exp:.4f}, ΔE/gap={de_exp:.4f} {'OK' if exp_ok else 'FAIL'}")
    print(f"    Richardson:  E(0)={e_rich:.4f}, ΔE/gap={de_rich:.4f}")

    results.append(
        {
            "h_test": h_test,
            "ces": ces.tolist(),
            "energies": e.tolist(),
            "exact_energy": exact_energy,
            "gap": gap,
            "noisy_raw_de_gap": noisy_raw_de_gap,
            "linear": {"e": e_lin, "de_gap": de_lin, "r2": r2_lin},
            "quadratic": {"e": e_quad, "de_gap": de_quad},
            "exponential": {"e": e_exp, "de_gap": de_exp, "ok": exp_ok},
            "richardson": {"e": e_rich, "de_gap": de_rich},
        }
    )

# Summary
print(f"\n{'─' * 70}")
print(f"  {'h':<6} {'Raw':<10} {'Linear':<10} {'Quad':<10} {'Exp':<10} {'Rich':<10} {'Best'}")
print(f"  {'─' * 64}")
for r in results:
    methods = {
        "linear": r["linear"]["de_gap"],
        "quad": r["quadratic"]["de_gap"],
        "exp": r["exponential"]["de_gap"],
        "rich": r["richardson"]["de_gap"],
    }
    valid = {k: v for k, v in methods.items() if np.isfinite(v)}
    best = min(valid, key=valid.get) if valid else "none"
    best_val = valid[best] if valid else float("nan")
    # Does best beat noisy raw?
    beats_raw = best_val < r["noisy_raw_de_gap"]
    marker = "✅" if beats_raw else "❌"
    print(
        f"  {r['h_test']:<6.2f} "
        f"{r['noisy_raw_de_gap']:<10.4f} "
        f"{r['linear']['de_gap']:<10.4f} "
        f"{r['quadratic']['de_gap']:<10.4f} "
        f"{r['exponential']['de_gap']:<10.4f} "
        f"{r['richardson']['de_gap']:<10.4f} "
        f"{best} {marker}"
    )

# Conclusion
print(f"\n{'─' * 70}")
any_improvement = any(
    min(r["linear"]["de_gap"], r["quadratic"]["de_gap"], r["richardson"]["de_gap"])
    < r["noisy_raw_de_gap"]
    for r in results
)
if any_improvement:
    print("  CONCLUSION: Non-linear extrapolation IMPROVES over noisy raw at some h-values.")
    print("  However, check if improvement is meaningful (ΔE/gap < 5% threshold).")
else:
    print("  CONCLUSION: No extrapolation method beats noisy raw.")
    print("  The failure is fundamental — no monotonic E(CES) relationship exists at N=10.")

# Save
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = RESULTS_DIR / f"v2_nonlinear_{ts}.json"
with open(out_path, "w") as f:
    json.dump({"variant": "V2", "results": results}, f, indent=2, default=str)
print(f"\n  Saved: {out_path}")
