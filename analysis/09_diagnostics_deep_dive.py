#!/usr/bin/env python3
"""Deep dive into pipeline diagnostics across all variants.

This is the MOST IMPORTANT next analysis: correlating theta_smoothness,
generalization_gap, and convergence_rate with final ΔE/gap outcome.

Answers the key question: "Can we PREDICT failure from Phase 2/3 diagnostics
without running Phase 4?"

Also completes the missing data from ladder N=6 and chain_1d variants.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "results" / "thesis"
OUTPUT = ROOT / "analysis"


@dataclass
class DiagnosticRecord:
    """Full diagnostic record for one variant."""

    folder: str
    variant_name: str
    topology: str
    n_qubits: int
    delta_e_over_gap: float | None
    # Phase 2 diagnostics
    convergence_rate: float | None
    theta_smoothness: float | None
    # Phase 3 diagnostics
    generalization_gap: float | None
    # Phase 4 diagnostics
    error_from_circuit: float | None
    error_from_mpnn: float | None
    # Config
    n_restarts: int | None = None
    hidden_dim: int | None = None
    n_epochs: int | None = None
    seed: int | None = None
    h_test: float | None = None


def scan_all_pipeline_results() -> list[DiagnosticRecord]:
    """Scan ALL pipeline_run files from all variant folders."""
    records = []

    variant_folders = [
        ("variants_N6_N10_1D_linnear", "chain_1d", 6),
        ("variants_N6_ladder", "ladder", 6),
        ("variants_N6_triangular", "triangular", 6),
        ("variants_N10_ladder", "ladder", 10),
        ("variants_N10_triangular", "triangular", 10),
    ]

    for folder_name, topology, n_qubits in variant_folders:
        folder_path = THESIS / folder_name
        if not folder_path.exists():
            continue

        # Scan all subdirectories for pipeline results
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            if subdir.name == "checkpoints":
                continue

            # Find latest pipeline_run file
            pipeline_files = sorted(subdir.glob("pipeline_run_*.json"), reverse=True)
            if not pipeline_files:
                continue

            try:
                with open(pipeline_files[0]) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Extract Phase 4 results
            p4 = data.get("phase4_results", [])
            de_gap = None
            h_test = None
            if p4:
                de_gap = p4[0].get("delta_e_over_gap")
                h_test = p4[0].get("h_test")

            # Extract diagnostics
            diag = data.get("diagnostics", {})
            p2 = diag.get("phase2", {})
            p3 = diag.get("phase3", {})
            p4_diag = diag.get("phase4", {})

            # Energy decomposition
            decomp = p4_diag.get("energy_decomposition", {})

            # Config
            config = data.get("config", {})
            mpnn_config = config.get("mpnn", {})

            record = DiagnosticRecord(
                folder=folder_name,
                variant_name=subdir.name,
                topology=topology,
                n_qubits=n_qubits,
                delta_e_over_gap=de_gap,
                convergence_rate=p2.get("convergence_rate"),
                theta_smoothness=p2.get("theta_smoothness"),
                generalization_gap=p3.get("generalization_gap"),
                error_from_circuit=decomp.get("error_from_circuit"),
                error_from_mpnn=decomp.get("error_from_mpnn"),
                n_restarts=config.get("n_restarts"),
                hidden_dim=mpnn_config.get("hidden_dim"),
                n_epochs=mpnn_config.get("n_epochs"),
                seed=config.get("seed"),
                h_test=h_test,
            )
            records.append(record)

    return records


def analyze_smoothness_correlation(records: list[DiagnosticRecord]) -> str:
    """Core analysis: does theta_smoothness predict ΔE/gap?"""
    lines = []
    lines.append("# Análisis de Correlación: theta_smoothness vs ΔE/gap\n")
    lines.append(
        "**Pregunta**: ¿Podemos predecir el fracaso del pipeline desde "
        "Phase 2 sin ejecutar Phase 3+4?\n"
    )

    # Filter records with both metrics
    valid = [
        r for r in records if r.theta_smoothness is not None and r.delta_e_over_gap is not None
    ]

    lines.append(f"**Datos**: {len(valid)} variants con ambas métricas\n")

    if len(valid) < 5:
        lines.append("⚠️ Insuficientes datos para análisis de correlación.")
        return "\n".join(lines)

    # Categorize by smoothness threshold
    lines.append("## Distribución por Umbral de Smoothness\n")
    lines.append(
        "| Rango θ_smoothness | N | Pass (<5%) | Marginal | Fail (>10%) "
        "| Pass Rate | Mediana ΔE/gap |"
    )
    lines.append(
        "|-------------------|---|------------|----------|-------------|"
        "-----------|----------------|"
    )

    thresholds = [
        ("< 0.05 (excelente)", lambda s: s < 0.05),
        ("0.05 - 0.10 (bueno)", lambda s: 0.05 <= s < 0.10),
        ("0.10 - 1.0 (sospechoso)", lambda s: 0.10 <= s < 1.0),
        ("> 1.0 (chain break)", lambda s: s >= 1.0),
    ]

    for label, pred in thresholds:
        group = [r for r in valid if pred(r.theta_smoothness)]
        if not group:
            continue
        n = len(group)
        n_pass = sum(1 for r in group if r.delta_e_over_gap < 0.05)
        n_marg = sum(1 for r in group if 0.05 <= r.delta_e_over_gap < 0.10)
        n_fail = sum(1 for r in group if r.delta_e_over_gap >= 0.10)
        de_values = sorted([r.delta_e_over_gap for r in group])
        median = de_values[len(de_values) // 2]
        pass_rate = n_pass / n if n > 0 else 0
        lines.append(
            f"| {label} | {n} | {n_pass} | {n_marg} | {n_fail} | {pass_rate:.0%} | {median:.4f} |"
        )

    # Pearson correlation (manual, no numpy dependency)
    x = [r.theta_smoothness for r in valid]
    y = [r.delta_e_over_gap for r in valid]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    r_pearson = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    # Spearman (rank correlation — more robust to outliers)
    rank_x = [sorted(x).index(xi) for xi in x]
    rank_y = [sorted(y).index(yi) for yi in y]
    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n
    cov_rxy = (
        sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y, strict=False)) / n
    )
    std_rx = (sum((rx - mean_rx) ** 2 for rx in rank_x) / n) ** 0.5
    std_ry = (sum((ry - mean_ry) ** 2 for ry in rank_y) / n) ** 0.5
    r_spearman = cov_rxy / (std_rx * std_ry) if std_rx > 0 and std_ry > 0 else 0

    lines.append("\n## Correlación\n")
    lines.append(
        f"- **Pearson r** = {r_pearson:.3f} "
        f"({'fuerte' if abs(r_pearson) > 0.7 else 'moderada' if abs(r_pearson) > 0.4 else 'débil'})"
    )
    lines.append(
        f"- **Spearman ρ** = {r_spearman:.3f} "
        f"({'fuerte' if abs(r_spearman) > 0.7 else 'moderada' if abs(r_spearman) > 0.4 else 'débil'})"
    )

    # Decision rule
    lines.append("\n## Regla de Decisión Propuesta\n")
    lines.append("```")
    lines.append("IF theta_smoothness > 1.0:")
    lines.append("    ABORT Phase 3+4 (warm-start chain broke)")
    lines.append("    ACTION: reduce restarts or increase h-grid density")
    lines.append("ELIF theta_smoothness > 0.10:")
    lines.append("    WARNING: elevated risk of MPNN failure")
    lines.append("    ACTION: check gen_gap after Phase 3")
    lines.append("ELSE:")
    lines.append("    PROCEED normally")
    lines.append("```")

    # Validate the rule
    lines.append("\n## Validación de la Regla\n")
    above_1 = [r for r in valid if r.theta_smoothness >= 1.0]
    below_005 = [r for r in valid if r.theta_smoothness < 0.05]
    if above_1:
        fail_rate = sum(1 for r in above_1 if r.delta_e_over_gap >= 0.10) / len(above_1)
        lines.append(
            f"- θ_smoothness ≥ 1.0: {len(above_1)} cases, "
            f"{fail_rate:.0%} fail rate → regla es {'efectiva' if fail_rate > 0.8 else 'parcial'}"
        )
    if below_005:
        pass_rate = sum(1 for r in below_005 if r.delta_e_over_gap < 0.05) / len(below_005)
        lines.append(
            f"- θ_smoothness < 0.05: {len(below_005)} cases, "
            f"{pass_rate:.0%} pass rate → buen predictor de éxito"
        )

    return "\n".join(lines)


def analyze_gen_gap_correlation(records: list[DiagnosticRecord]) -> str:
    """Correlation between generalization_gap and ΔE/gap."""
    lines = []
    lines.append("\n---\n")
    lines.append("# Análisis de Correlación: generalization_gap vs ΔE/gap\n")
    lines.append("**Pregunta**: ¿El gen_gap de Phase 3 predice el resultado de Phase 4?\n")

    valid = [
        r for r in records if r.generalization_gap is not None and r.delta_e_over_gap is not None
    ]

    lines.append(f"**Datos**: {len(valid)} variants con ambas métricas\n")

    if len(valid) < 5:
        lines.append("⚠️ Insuficientes datos.")
        return "\n".join(lines)

    # Categorize
    lines.append("| Rango gen_gap | N | Pass | Marginal | Fail | Pass Rate |")
    lines.append("|---------------|---|------|----------|------|-----------|")

    thresholds = [
        ("< 1e-4 (excelente)", lambda g: g < 1e-4),
        ("1e-4 - 1e-3 (bueno)", lambda g: 1e-4 <= g < 1e-3),
        ("1e-3 - 1e-2 (sospechoso)", lambda g: 1e-3 <= g < 1e-2),
        ("> 1e-2 (overfitting)", lambda g: g >= 1e-2),
    ]

    for label, pred in thresholds:
        group = [r for r in valid if pred(r.generalization_gap)]
        if not group:
            continue
        n = len(group)
        n_pass = sum(1 for r in group if r.delta_e_over_gap < 0.05)
        n_marg = sum(1 for r in group if 0.05 <= r.delta_e_over_gap < 0.10)
        n_fail = sum(1 for r in group if r.delta_e_over_gap >= 0.10)
        pass_rate = n_pass / n if n > 0 else 0
        lines.append(f"| {label} | {n} | {n_pass} | {n_marg} | {n_fail} | {pass_rate:.0%} |")

    # Correlation
    x = [r.generalization_gap for r in valid]
    y = [r.delta_e_over_gap for r in valid]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    r_pearson = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

    lines.append(f"\n**Pearson r** = {r_pearson:.3f}")

    return "\n".join(lines)


def analyze_error_decomposition(records: list[DiagnosticRecord]) -> str:
    """Where does the error come from? Circuit vs MPNN."""
    lines = []
    lines.append("\n---\n")
    lines.append("# Descomposición del Error: Circuito vs MPNN\n")
    lines.append(
        "**Pregunta**: ¿Cuánto error viene del HVA (expresibilidad) vs del MPNN (predicción)?\n"
    )

    valid = [
        r for r in records if r.error_from_circuit is not None and r.error_from_mpnn is not None
    ]

    lines.append(f"**Datos**: {len(valid)} variants con descomposición\n")

    if len(valid) < 3:
        lines.append("⚠️ Insuficientes datos con energy_decomposition.")
        return "\n".join(lines)

    # Group by topology
    by_topo = defaultdict(list)
    for r in valid:
        by_topo[(r.topology, r.n_qubits)].append(r)

    lines.append(
        "| Topología | N | N pts | Mean Circuit Error | Mean MPNN Error | % Circuit | Bottleneck |"
    )
    lines.append(
        "|-----------|---|-------|--------------------|-----------------|-----------|------------|"
    )

    for key in sorted(by_topo.keys()):
        topo, n = key
        group = by_topo[key]
        circuit_errors = [r.error_from_circuit for r in group]
        mpnn_errors = [r.error_from_mpnn for r in group]
        mean_c = statistics.mean(circuit_errors)
        mean_m = statistics.mean(mpnn_errors)
        total = mean_c + mean_m
        pct_circuit = mean_c / total * 100 if total > 0 else 0
        bottleneck = (
            "Circuit (HVA)" if pct_circuit > 60 else ("MPNN" if pct_circuit < 40 else "Balanced")
        )
        lines.append(
            f"| {topo} | {n} | {len(group)} | {mean_c:.4f} | {mean_m:.4f} | "
            f"{pct_circuit:.0f}% | {bottleneck} |"
        )

    lines.append("\n## Interpretación\n")
    lines.append(
        "- Si Circuit >> MPNN: el HVA no puede expresar el ground state "
        "(límite físico, no mejorable con ML)"
    )
    lines.append(
        "- Si MPNN >> Circuit: el predictor es el cuello de botella "
        "(mejorable con más datos/epochs/capacity)"
    )
    lines.append("- Si balanced: ambos contribuyen — mejora requiere ambos frentes")

    return "\n".join(lines)


def analyze_topology_diagnostics(records: list[DiagnosticRecord]) -> str:
    """Diagnostic distributions by topology."""
    lines = []
    lines.append("\n---\n")
    lines.append("# Distribución de Diagnósticos por Topología\n")

    by_topo = defaultdict(list)
    for r in records:
        if r.theta_smoothness is not None:
            by_topo[(r.topology, r.n_qubits)].append(r)

    lines.append(
        "| Topología | N | N pts | Med. Smoothness | Med. Gen Gap | "
        "Med. ΔE/gap | Chain Breaks (>1.0) |"
    )
    lines.append(
        "|-----------|---|-------|-----------------|--------------|"
        "------------|---------------------|"
    )

    for key in sorted(by_topo.keys()):
        topo, n = key
        group = by_topo[key]
        smoothness_vals = [r.theta_smoothness for r in group if r.theta_smoothness is not None]
        gen_gaps = [r.generalization_gap for r in group if r.generalization_gap is not None]
        de_vals = [r.delta_e_over_gap for r in group if r.delta_e_over_gap is not None]

        med_s = statistics.median(smoothness_vals) if smoothness_vals else None
        med_g = statistics.median(gen_gaps) if gen_gaps else None
        med_d = statistics.median(de_vals) if de_vals else None
        n_breaks = sum(1 for s in smoothness_vals if s >= 1.0)

        ms = f"{med_s:.4f}" if med_s is not None else "N/A"
        mg = f"{med_g:.2e}" if med_g is not None else "N/A"
        md = f"{med_d:.4f}" if med_d is not None else "N/A"

        lines.append(
            f"| {topo} | {n} | {len(group)} | {ms} | {mg} | {md} | "
            f"{n_breaks}/{len(smoothness_vals)} |"
        )

    return "\n".join(lines)


def generate_complete_data_table(records: list[DiagnosticRecord]) -> str:
    """Generate the complete cross-topology table with ALL data."""
    lines = []
    lines.append("\n---\n")
    lines.append("# Tabla Completa Cross-Topología (Datos Corregidos)\n")
    lines.append(
        "Incluye TODOS los pipeline results encontrados "
        "(resuelve el problema de datos faltantes).\n"
    )

    # Group by topology/N
    by_topo = defaultdict(list)
    for r in records:
        if r.delta_e_over_gap is not None:
            by_topo[(r.topology, r.n_qubits)].append(r)

    lines.append(
        "| Topología | N | Total Variants | PASS | MARGINAL | FAIL | "
        "Mejor | Mediana | Peor | Pass Rate |"
    )
    lines.append(
        "|-----------|---|----------------|------|----------|------|"
        "-------|---------|------|-----------|"
    )

    for key in sorted(by_topo.keys()):
        topo, n = key
        group = by_topo[key]
        de_vals = sorted([r.delta_e_over_gap for r in group])
        n_pass = sum(1 for v in de_vals if v < 0.05)
        n_marg = sum(1 for v in de_vals if 0.05 <= v < 0.10)
        n_fail = sum(1 for v in de_vals if v >= 0.10)
        total = len(de_vals)
        best = de_vals[0]
        worst = de_vals[-1]
        median = de_vals[total // 2]
        pass_rate = n_pass / total

        lines.append(
            f"| {topo} | {n} | {total} | {n_pass} | {n_marg} | {n_fail} | "
            f"{best:.4f} | {median:.4f} | {worst:.4f} | {pass_rate:.0%} |"
        )

    # Total
    all_de = [r.delta_e_over_gap for r in records if r.delta_e_over_gap is not None]
    total = len(all_de)
    n_pass = sum(1 for v in all_de if v < 0.05)
    lines.append(
        f"\n**Total**: {total} variants con datos, {n_pass}/{total} "
        f"({n_pass / total:.0%}) pasan ΔE/gap < 5%"
    )

    return "\n".join(lines)


def main():
    """Run the diagnostics deep dive."""
    import sys

    print("=" * 70, file=sys.stderr)
    print("DIAGNOSTICS DEEP DIVE — Pipeline Failure Prediction", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    print("\n[1/5] Scanning all pipeline results...", file=sys.stderr)
    records = scan_all_pipeline_results()
    print(f"  → {len(records)} pipeline results found", file=sys.stderr)

    # Count by topology
    by_topo = defaultdict(int)
    for r in records:
        by_topo[(r.topology, r.n_qubits)] += 1
    for key in sorted(by_topo.keys()):
        print(f"    {key[0]} N={key[1]}: {by_topo[key]} results", file=sys.stderr)

    # Count with diagnostics
    n_with_smooth = sum(1 for r in records if r.theta_smoothness is not None)
    n_with_gap = sum(1 for r in records if r.generalization_gap is not None)
    n_with_decomp = sum(1 for r in records if r.error_from_circuit is not None)
    print(f"  → With theta_smoothness: {n_with_smooth}", file=sys.stderr)
    print(f"  → With generalization_gap: {n_with_gap}", file=sys.stderr)
    print(f"  → With energy_decomposition: {n_with_decomp}", file=sys.stderr)

    # Save raw diagnostic data
    raw_diag = []
    for r in records:
        raw_diag.append(
            {
                "folder": r.folder,
                "variant_name": r.variant_name,
                "topology": r.topology,
                "n_qubits": r.n_qubits,
                "delta_e_over_gap": r.delta_e_over_gap,
                "theta_smoothness": r.theta_smoothness,
                "generalization_gap": r.generalization_gap,
                "convergence_rate": r.convergence_rate,
                "error_from_circuit": r.error_from_circuit,
                "error_from_mpnn": r.error_from_mpnn,
                "n_restarts": r.n_restarts,
                "hidden_dim": r.hidden_dim,
                "seed": r.seed,
                "h_test": r.h_test,
            }
        )

    raw_path = OUTPUT / "raw_data" / "all_diagnostics.json"
    with open(raw_path, "w") as f:
        json.dump(raw_diag, f, indent=2)
    print(f"  → Saved to {raw_path}", file=sys.stderr)

    # Run analyses
    print("\n[2/5] Smoothness correlation...", file=sys.stderr)
    result1 = analyze_smoothness_correlation(records)

    print("[3/5] Gen gap correlation...", file=sys.stderr)
    result2 = analyze_gen_gap_correlation(records)

    print("[4/5] Error decomposition...", file=sys.stderr)
    result3 = analyze_error_decomposition(records)

    print("[5/5] Topology diagnostics + complete table...", file=sys.stderr)
    result4 = analyze_topology_diagnostics(records)
    result5 = generate_complete_data_table(records)

    # Write combined output
    output_path = OUTPUT / "09_diagnostics_deep_dive.md"
    with open(output_path, "w") as f:
        f.write(result1)
        f.write(result2)
        f.write(result3)
        f.write(result4)
        f.write(result5)

    print(f"\n✅ Written to {output_path}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()
