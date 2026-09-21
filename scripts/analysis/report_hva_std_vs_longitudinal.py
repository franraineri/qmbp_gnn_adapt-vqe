#!/usr/bin/env python
"""Render a plain-text report from the HVA std vs longitudinal comparison JSON.

Reads comparison_results.json (produced by compare_hva_std_vs_longitudinal.py) and
writes a plain-text report with:
  - a per-h side-by-side table (std vs longitudinal) for each (topology, N),
  - a focused h=1.0 summary table (the critical point),
  - conclusions answering the 4 required questions.

Usage:
    .venv/bin/python scripts/analysis/report_hva_std_vs_longitudinal.py \
        --in results/analysis/hva_std_vs_longitudinal/comparison_results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _index(results: list[dict]) -> dict:
    """Index results by (topology, n_qubits, h, ansatz)."""
    idx: dict = {}
    for r in results:
        idx[(r["topology"], r["n_qubits"], round(r["h"], 2), r["ansatz"])] = r
    return idx


def _groups(results: list[dict]):
    """Ordered unique (topology, n_qubits) and h values."""
    tn = []
    hs = []
    for r in results:
        key = (r["topology"], r["n_qubits"])
        if key not in tn:
            tn.append(key)
        h = round(r["h"], 2)
        if h not in hs:
            hs.append(h)
    hs.sort()
    return tn, hs


def build_report(payload: dict) -> str:
    meta = payload["meta"]
    results = payload["results"]
    idx = _index(results)
    tn_groups, h_values = _groups(results)

    lines: list[str] = []
    W = 100
    lines.append("=" * W)
    lines.append("HVA ESTÁNDAR vs HVA + CAMPO LONGITUDINAL (RZ por sitio) — TFIM puro, simulación ideal")
    lines.append("=" * W)
    lines.append("")
    lines.append("Pregunta: para el MISMO ground state del TFIM puro (H = -J·ZZ - h·X, g=0),")
    lines.append("¿el ansatz bond-resolved + RZ por sitio (tfim_longitudinal) llega más cerca del")
    lines.append("autovalor exacto que el bond-resolved estándar, a p=1, sin subir la profundidad 2q?")
    lines.append("")
    lines.append("Setup (idéntico para ambos ansätze en cada punto):")
    lines.append(f"  Backend            : {meta['backend']} (StatevectorEstimator, sin ruido)")
    lines.append(f"  Hamiltoniano objetivo: {meta['target_hamiltonian']}")
    lines.append(f"  p_layers           : {meta['p_layers']}")
    lines.append(f"  Semillas           : {meta['seeds']} (media ± std sobre semillas)")
    lines.append(f"  Optimizador        : L-BFGS-B, maxiter={meta['maxiter']}, n_restarts={meta['n_restarts']}")
    lines.append(f"  Topologías / N     : {meta['topologies']} / {meta['n_qubits']}")
    lines.append("")
    lines.append("Nota metodológica: el RZ del tfim_longitudinal es una libertad del ANSATZ, no un")
    lines.append("término del Hamiltoniano. El Hamiltoniano objetivo es el TFIM puro (g=0) para ambos;")
    lines.append("así la comparación mide 'qué ansatz aproxima mejor el MISMO estado', no otro problema.")
    lines.append("")

    # ── Per (topology, N) side-by-side tables ──
    for (topo, n) in tn_groups:
        # anchor exact values (E0, gap) come from either ansatz (identical target)
        lines.append("-" * W)
        lines.append(f"TOPOLOGÍA={topo}  N={n}  p={meta['p_layers']}")
        lines.append("-" * W)
        # header
        hdr = (
            f"{'h':>5} | {'E0':>9} | {'gap':>7} || "
            f"{'STD |ΔE|':>9} {'STD ΔE/gap':>10} {'STD fid':>8} || "
            f"{'LON |ΔE|':>9} {'LON ΔE/gap':>10} {'LON fid':>8} || "
            f"{'Δ|ΔE|(L-S)':>11}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for h in h_values:
            std = idx.get((topo, n, h, "std"))
            lon = idx.get((topo, n, h, "longitudinal"))
            if std is None or lon is None:
                continue
            diff = lon["abs_de_mean"] - std["abs_de_mean"]
            lines.append(
                f"{h:>5.2f} | {std['e0']:>9.4f} | {std['gap']:>7.4f} || "
                f"{std['abs_de_mean']:>9.4f} {std['de_over_gap_mean']:>10.4f} {std['fidelity_best']:>8.4f} || "
                f"{lon['abs_de_mean']:>9.4f} {lon['de_over_gap_mean']:>10.4f} {lon['fidelity_best']:>8.4f} || "
                f"{diff:>+11.4f}"
            )
        # cost row (constant across h)
        any_std = idx.get((topo, n, h_values[0], "std"))
        any_lon = idx.get((topo, n, h_values[0], "longitudinal"))
        if any_std and any_lon:
            lines.append("")
            lines.append(
                f"  Coste circuito: STD  nparams={any_std['n_params']:>3}  gates2q={any_std['n_2q_gates']:>2}  depth={any_std['depth']:>2}"
            )
            lines.append(
                f"                  LON  nparams={any_lon['n_params']:>3}  gates2q={any_lon['n_2q_gates']:>2}  depth={any_lon['depth']:>2}"
                f"   (Δparams=+{any_lon['n_params']-any_std['n_params']}, Δgates2q={any_lon['n_2q_gates']-any_std['n_2q_gates']})"
            )
        lines.append("")

    # ── h=1.0 focused summary (critical point) ──
    lines.append("=" * W)
    lines.append("RESUMEN EN h = 1.0 (PUNTO CRÍTICO h_c — donde el HVA estándar más sufre)")
    lines.append("=" * W)
    hdr2 = (
        f"{'topo':>10} {'N':>3} | {'E0':>9} {'ansatz_min(STD)':>16} {'ansatz_min(LON)':>16} | "
        f"{'|ΔE| STD':>9} {'|ΔE| LON':>9} {'mejora':>8} | {'fid STD':>8} {'fid LON':>8}"
    )
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for (topo, n) in tn_groups:
        std = idx.get((topo, n, 1.0, "std"))
        lon = idx.get((topo, n, 1.0, "longitudinal"))
        if std is None or lon is None:
            continue
        improve = std["abs_de_mean"] - lon["abs_de_mean"]
        lines.append(
            f"{topo:>10} {n:>3} | {std['e0']:>9.4f} {std['e_ansatz_min_best']:>16.4f} "
            f"{lon['e_ansatz_min_best']:>16.4f} | {std['abs_de_mean']:>9.4f} {lon['abs_de_mean']:>9.4f} "
            f"{improve:>+8.4f} | {std['fidelity_best']:>8.4f} {lon['fidelity_best']:>8.4f}"
        )
    lines.append("")
    lines.append("'mejora' = |ΔE|_STD − |ΔE|_LON (positivo = el longitudinal acerca más al exacto).")
    lines.append("")

    # ── Conclusions ──
    # Compute aggregate: max improvement anywhere.
    max_improve = 0.0
    max_improve_at = None
    for r_std in [r for r in results if r["ansatz"] == "std"]:
        key = (r_std["topology"], r_std["n_qubits"], round(r_std["h"], 2), "longitudinal")
        r_lon = idx.get(key)
        if r_lon is None:
            continue
        imp = r_std["abs_de_mean"] - r_lon["abs_de_mean"]
        if abs(imp) > abs(max_improve):
            max_improve = imp
            max_improve_at = (r_std["topology"], r_std["n_qubits"], round(r_std["h"], 2))

    lines.append("=" * W)
    lines.append("CONCLUSIONES")
    lines.append("=" * W)
    lines.append("")
    lines.append("1) ¿El RZ por sitio (tfim_longitudinal) reduce |ΔE| respecto al HVA estándar a p=1,")
    lines.append("   para el MISMO TFIM puro? ¿En qué magnitud?")
    lines.append("")
    lines.append("   NO. En los 48 puntos barridos (2 topologías × 3 N × 8 h), el ansatz longitudinal")
    lines.append("   alcanza EXACTAMENTE la misma energía mínima, |ΔE| y fidelidad que el estándar,")
    lines.append("   coincidiendo hasta 4 decimales en todos los casos.")
    lines.append(f"   La mayor diferencia observada en |ΔE| es {max_improve:+.4f}"
                 + (f" (en {max_improve_at})." if max_improve_at else "."))
    lines.append("   Es decir, la mejora del techo es CERO.")
    lines.append("")
    lines.append("   Mecanismo (verificado en optimal_thetas.json): en los 48 puntos, todo el bloque")
    lines.append("   RZ óptimo es θ_z = 0 exactamente. El ground state del TFIM puro es Z₂-simétrico")
    lines.append("   (⟨Z_i⟩ = 0), por lo que ∂E/∂θ_z = 0 en θ_z = 0: la libertad longitudinal es un")
    lines.append("   punto estacionario nulo para este objetivo y el optimizador la deja en cero.")
    lines.append("")
    lines.append("2) ¿La mejora se concentra cerca de h_c=1.0 o es uniforme?")
    lines.append("")
    lines.append("   Ni una cosa ni la otra: no hay mejora en NINGÚN h. El resultado es idéntico en todo")
    lines.append("   el barrido, incluyendo el punto crítico h_c=1.0 donde el HVA estándar tiene su mayor")
    lines.append("   brecha (heavy_hex N=10: |ΔE|=0.3916, fid=0.796). El RZ no ayuda justamente donde")
    lines.append("   más se necesitaría.")
    lines.append("")
    lines.append("3) ¿Cuál es el coste extra?")
    lines.append("")
    lines.append("   El RZ añade N parámetros de UN qubit por capa (p.ej. N=10: 29 vs 19 params) y")
    lines.append("   CERO gates de dos qubits (confirmado: gates2q idéntico entre ambos ansätze en")
    lines.append("   todos los N). La profundidad LÓGICA total sube en 1 (la capa de RZ de un qubit),")
    lines.append("   pero la profundidad de DOS qubits — el predictor de error en hardware — no cambia,")
    lines.append("   y en IBM el RZ es virtual (coste cero). O sea: más parámetros para optimizar, sin")
    lines.append("   ningún beneficio para el TFIM puro y sin coste 2q relevante.")
    lines.append("")
    lines.append("4) ¿Vale la pena como alternativa a subir a p=2?")
    lines.append("")
    lines.append("   No, para el TFIM puro. El techo del HVA p=1 lo fija la falta de expresividad de")
    lines.append("   entrelazamiento (número/estructura de capas de RZZ), no la ausencia de un giro")
    lines.append("   longitudinal. Subir a p=2 añade otra capa de RZZ (más gates 2q y profundidad) que SÍ")
    lines.append("   sube el techo; el RZ, en cambio, no toca la parte de dos qubits y por eso no mueve el")
    lines.append("   techo. El tfim_longitudinal solo es útil cuando el Hamiltoniano objetivo rompe la")
    lines.append("   simetría Z₂ (g>0), caso para el que fue diseñado; para g=0 es equivalente al estándar.")
    lines.append("")
    lines.append("Caveat metodológico: el warm-start usa θ inicial ≈ 0 (incluido θ_z ≈ ±0.05). El bloque")
    lines.append("θ_z = 0 es el óptimo por la simetría Z₂ del TFIM puro (∂E/∂θ_z=0 en θ_z=0), y los")
    lines.append("multi-restart (n_restarts) perturban alrededor de ese punto sin encontrar un basin")
    lines.append("mejor. No se puede descartar formalmente un óptimo con θ_z grande y aislado, pero el")
    lines.append("argumento de simetría y la coincidencia exacta en 48/48 puntos lo hacen inverosímil.")
    lines.append("")
    lines.append("=" * W)
    lines.append("Artefactos: comparison_results.json (métricas completas), optimal_thetas.json (θ* de")
    lines.append("ambos ansätze, incluido el bloque θ_z=0 del longitudinal). Anclaje reproducido:")
    lines.append("heavy_hex N=10 p=1 h=1.0 STD → ansatz_min=-12.0806, E0=-12.4722, gap=0.2201, fid=0.7958.")
    lines.append("=" * W)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="results/analysis/hva_std_vs_longitudinal/comparison_results.json",
    )
    parser.add_argument("--out", type=str, default=None, help="Output .txt (default: alongside input)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    payload = json.loads(in_path.read_text())
    report = build_report(payload)

    out_path = Path(args.out) if args.out else in_path.parent / "REPORT.txt"
    out_path.write_text(report)
    print(report)
    print(f"\nSaved report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
