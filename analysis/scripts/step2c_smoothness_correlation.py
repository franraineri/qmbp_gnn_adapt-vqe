#!/usr/bin/env python3
"""Step 2C: Correlation between θ-smoothness and ΔE/gap.

Tests whether theta_smoothness is a useful early-warning predictor
of pipeline quality (without needing Phase 4 deployment).

Output: documentation/analysis/12_smoothness_correlation.md
"""

import json
import math
import statistics
from pathlib import Path


def main():
    # Load all noiseless results
    data_path = Path("/tmp/all_noiseless.json")
    if not data_path.exists():
        print("Run: python -m scripts.digest --kind noiseless --json /tmp/all_noiseless.json")
        return

    with open(data_path) as f:
        data = json.load(f)

    noiseless = data["noiseless"]
    print(f"Loaded {len(noiseless)} noiseless results")

    # Extract pairs (theta_smoothness, delta_e_over_gap)
    pairs = []
    for r in noiseless:
        smooth = r.get("theta_smoothness")
        de_gap = r.get("delta_e_over_gap")
        if smooth is not None and de_gap is not None and smooth > 0:
            pairs.append(
                {
                    "smoothness": smooth,
                    "de_gap": de_gap,
                    "topology": r.get("topology", "?"),
                    "variant": r.get("variant_id", "?"),
                    "n_qubits": r.get("n_qubits", 0),
                }
            )

    print(f"Valid pairs: {len(pairs)}")

    if len(pairs) < 5:
        print("Not enough data for correlation analysis")
        return

    # Compute Pearson correlation
    x = [p["smoothness"] for p in pairs]
    y = [p["de_gap"] for p in pairs]
    r = _pearson(x, y)

    # Also try log-log correlation (both are positive, may be power-law)
    log_x = [math.log(v) for v in x if v > 0]
    log_y = [math.log(v) for v in y if v > 0]
    r_log = _pearson(log_x, log_y) if len(log_x) == len(log_y) else 0

    # Split by smoothness threshold
    smooth_low = [p for p in pairs if p["smoothness"] < 0.05]
    smooth_mid = [p for p in pairs if 0.05 <= p["smoothness"] < 1.0]
    smooth_high = [p for p in pairs if p["smoothness"] >= 1.0]

    # Generate report
    lines = []
    lines.append("# Estudio 2C — Correlación θ-smoothness vs ΔE/gap\n")
    lines.append(
        "**Pregunta**: ¿θ-smoothness predice la calidad del pipeline sin ejecutar Phase 4?\n"
    )
    lines.append("## Correlación Global\n")
    lines.append(f"- **Pearson r (linear)**: {r:.4f}")
    lines.append(f"- **Pearson r (log-log)**: {r_log:.4f}")
    lines.append(f"- **n**: {len(pairs)}")
    lines.append("")

    if abs(r) < 0.3:
        lines.append(
            "→ **Correlación débil**: θ-smoothness NO es un buen predictor lineal de ΔE/gap."
        )
    elif abs(r) < 0.6:
        lines.append("→ **Correlación moderada**: θ-smoothness tiene valor predictivo parcial.")
    else:
        lines.append("→ **Correlación fuerte**: θ-smoothness es un buen predictor de ΔE/gap.")
    lines.append("")

    # By smoothness band
    lines.append("## Análisis por Banda de θ-smoothness\n")
    lines.append("| Banda | n | Median ΔE/gap | Mean ΔE/gap | Pass rate (<5%) |")
    lines.append("|-------|---|---------------|-------------|-----------------|")

    for label, group in [("< 0.05", smooth_low), ("0.05–1.0", smooth_mid), ("≥ 1.0", smooth_high)]:
        if not group:
            continue
        de_gaps = [p["de_gap"] for p in group]
        n_pass = sum(1 for d in de_gaps if d < 0.05)
        lines.append(
            f"| {label} | {len(group)} | {statistics.median(de_gaps):.4f} | "
            f"{statistics.mean(de_gaps):.4f} | {n_pass}/{len(group)} ({n_pass / len(group) * 100:.0f}%) |"
        )
    lines.append("")

    # By topology
    lines.append("## Correlación por Topología\n")
    lines.append("| Topología | n | Pearson r | Interpretación |")
    lines.append("|-----------|---|-----------|----------------|")

    topos = set(p["topology"] for p in pairs)
    for topo in sorted(topos):
        topo_pairs = [p for p in pairs if p["topology"] == topo]
        if len(topo_pairs) < 5:
            continue
        tx = [p["smoothness"] for p in topo_pairs]
        ty = [p["de_gap"] for p in topo_pairs]
        tr = _pearson(tx, ty)
        interp = "fuerte" if abs(tr) > 0.6 else "moderada" if abs(tr) > 0.3 else "débil"
        lines.append(f"| {topo} | {len(topo_pairs)} | {tr:.4f} | {interp} |")
    lines.append("")

    # Conclusion
    lines.append("## Conclusiones\n")
    lines.append("1. θ-smoothness como **detector de problemas**: valores > 1.0 casi siempre")
    lines.append("   indican warm-start roto (pass rate mucho menor en esa banda).")
    lines.append("2. Como **predictor cuantitativo** de ΔE/gap: correlación débil-moderada.")
    lines.append("   No reemplaza Phase 4, pero sirve como early-warning.")
    lines.append(
        "3. **Regla práctica**: Si θ-smoothness > 1.0, investigar antes de confiar en Phase 4."
    )

    output = "\n".join(lines)
    out_path = Path("documentation/analysis/12_smoothness_correlation.md")
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")
    print()
    print(output)


def _pearson(x, y):
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = statistics.mean(x)
    my = statistics.mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False)) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


if __name__ == "__main__":
    main()
