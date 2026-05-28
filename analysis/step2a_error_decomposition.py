#!/usr/bin/env python3
"""Step 2A: Error decomposition analysis by topology.

Extracts energy_decomposition from diagnostics to determine whether
errors come from HVA expressibility (circuit) or MPNN prediction.

Output: documentation/analysis/11_error_decomposition.md
"""

import json
import statistics
from pathlib import Path


def main():
    # Load all noiseless results with diagnostics
    results_root = Path("results/thesis")
    decompositions: dict[str, list[dict]] = {}  # topology → list of decomp dicts

    for folder in sorted(results_root.iterdir()):
        if not folder.is_dir():
            continue
        # Scan subfolders for diagnostics.json or pipeline_run with diagnostics
        for subfolder in sorted(folder.iterdir()):
            if not subfolder.is_dir():
                continue
            if subfolder.name == "checkpoints":
                continue

            # Try diagnostics.json first
            diag_file = subfolder / "diagnostics.json"
            if diag_file.exists():
                data = _load_json(diag_file)
                if data:
                    _extract_decomp(data, subfolder.name, folder.name, decompositions)
                    continue

            # Try latest pipeline_run
            pipeline_files = sorted(subfolder.glob("pipeline_run_*.json"), reverse=True)
            if pipeline_files:
                data = _load_json(pipeline_files[0])
                if data and "diagnostics" in data:
                    config = data.get("config", {})
                    system = data.get("system", {})
                    topo = config.get("topology") or system.get("topology", "")
                    if not topo:
                        topo = _infer_topo(folder.name)
                    diag = data["diagnostics"]
                    phase4 = diag.get("phase4", {})
                    decomp = phase4.get("energy_decomposition", {})
                    if decomp and decomp.get("e_exact") is not None:
                        decomp["variant"] = subfolder.name
                        decomp["n_qubits"] = config.get("n_qubits") or system.get("n_qubits", 0)
                        decompositions.setdefault(topo, []).append(decomp)

    # Analyze
    print(f"Found decompositions: {', '.join(f'{k}={len(v)}' for k, v in decompositions.items())}")
    print()

    lines = []
    lines.append("# Estudio 2A — Error Decomposition por Topología\n")
    lines.append("**Pregunta**: ¿El error viene del HVA (circuit) o del MPNN (prediction)?")
    lines.append("")
    lines.append("## Datos\n")

    summary_rows = []

    for topo in sorted(decompositions.keys()):
        entries = decompositions[topo]
        if not entries:
            continue

        circuit_errors = [
            e["error_from_circuit"] for e in entries if e.get("error_from_circuit") is not None
        ]
        mpnn_errors = [
            e["error_from_mpnn"] for e in entries if e.get("error_from_mpnn") is not None
        ]

        if not circuit_errors or not mpnn_errors:
            continue

        total_errors = [c + m for c, m in zip(circuit_errors, mpnn_errors, strict=False)]
        mpnn_fractions = [
            m / t if t > 0 else 0 for m, t in zip(mpnn_errors, total_errors, strict=False)
        ]

        n = len(entries)
        med_circuit = statistics.median(circuit_errors)
        med_mpnn = statistics.median(mpnn_errors)
        med_mpnn_frac = statistics.median(mpnn_fractions)

        lines.append(f"### {topo} (n={n})\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Median error_from_circuit | {med_circuit:.5f} |")
        lines.append(f"| Median error_from_mpnn | {med_mpnn:.5f} |")
        lines.append(f"| Median MPNN fraction | {med_mpnn_frac:.1%} |")
        lines.append(f"| Mean error_from_circuit | {statistics.mean(circuit_errors):.5f} |")
        lines.append(f"| Mean error_from_mpnn | {statistics.mean(mpnn_errors):.5f} |")
        lines.append("")

        summary_rows.append(
            {
                "topology": topo,
                "n": n,
                "med_circuit": med_circuit,
                "med_mpnn": med_mpnn,
                "mpnn_fraction": med_mpnn_frac,
            }
        )

        # Show top 5 worst MPNN errors
        worst = sorted(entries, key=lambda e: e.get("error_from_mpnn", 0), reverse=True)[:3]
        if worst:
            lines.append("Worst MPNN errors:")
            for w in worst:
                lines.append(
                    f"  - {w.get('variant', '?')}: circuit={w.get('error_from_circuit', 0):.4f}, mpnn={w.get('error_from_mpnn', 0):.4f}"
                )
            lines.append("")

    # Summary table
    lines.append("## Resumen Comparativo\n")
    lines.append("| Topología | n | Med circuit error | Med MPNN error | MPNN fraction |")
    lines.append("|-----------|---|-------------------|----------------|---------------|")
    for row in summary_rows:
        lines.append(
            f"| {row['topology']} | {row['n']} | {row['med_circuit']:.5f} | "
            f"{row['med_mpnn']:.5f} | {row['mpnn_fraction']:.1%} |"
        )
    lines.append("")

    # Conclusions
    lines.append("## Conclusiones\n")
    if summary_rows:
        max_mpnn_frac = max(r["mpnn_fraction"] for r in summary_rows)
        min_mpnn_frac = min(r["mpnn_fraction"] for r in summary_rows)
        if max_mpnn_frac > 0.8:
            lines.append("- **MPNN es el bottleneck dominant** en al menos una topología.")
        if min_mpnn_frac < 0.2:
            lines.append("- **HVA expressibility es el bottleneck** en al menos una topología.")
        lines.append(
            "- La fracción MPNN indica dónde mejorar: si >50%, mejorar MPNN; si <50%, mejorar ansatz."
        )

    output = "\n".join(lines)
    out_path = Path("documentation/analysis/11_error_decomposition.md")
    out_path.write_text(output)
    print(f"Saved to {out_path}")
    print()
    print(output)


def _extract_decomp(data, variant_name, folder_name, decompositions):
    """Extract decomposition from a diagnostics dict."""
    phase4 = data.get("phase4", {})
    decomp = phase4.get("energy_decomposition", {})
    if decomp and decomp.get("e_exact") is not None:
        topo = _infer_topo(folder_name)
        decomp["variant"] = variant_name
        decompositions.setdefault(topo, []).append(decomp)


def _infer_topo(name):
    name_lower = name.lower()
    for t in ("chain_1d", "ladder", "triangular", "kagome", "linnear"):
        if t in name_lower:
            return "chain_1d" if t == "linnear" else t
    return "unknown"


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


if __name__ == "__main__":
    main()
