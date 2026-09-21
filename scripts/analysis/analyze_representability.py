#!/usr/bin/env python
"""Consolidate the classical-representability study into SUMMARY.md + figures.

Reads every study_*.json in results/classical_representability/ (any method),
computes the chi needed to converge per (method, topology, N, h), and renders
the diagnostic figures. Safe to run with partial data (only MPS present, etc.).

Usage:
    python scripts/analysis/analyze_representability.py
    python scripts/analysis/analyze_representability.py --conv-tol 1e-6
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from representability_io import RESULTS_DIR, load_all_rows  # noqa: E402

FIGDIR = RESULTS_DIR / "figures"


def chi_needed(rows_for_cell: list[dict], conv_tol: float) -> int | None:
    """Smallest chi_requested whose abs_error <= conv_tol (rows of one cell)."""
    ok = [r for r in rows_for_cell
          if r.get("abs_error") is not None and r["abs_error"] <= conv_tol]
    if not ok:
        return None
    return min(int(r["chi_requested"]) for r in ok)


def build_summary(rows: list[dict], conv_tol: float) -> str:
    # group by (method, topology, N, h)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["method"], r.get("topology"), r["N"], r["h"])].append(r)

    lines = ["# Classical representability — resumen\n"]
    lines.append(f"Tolerancia de convergencia (|ΔE| ≤): {conv_tol:g}\n")
    lines.append("chi necesario para converger la energía, por método/N/h.\n")
    lines.append("Un guion (—) = no converge dentro de los chi probados.\n")

    methods = sorted({k[0] for k in cells})
    for method in methods:
        lines.append(f"\n## {method}\n")
        topos = sorted({k[1] for k in cells if k[0] == method})
        for topo in topos:
            Ns = sorted({k[2] for k in cells if k[0] == method and k[1] == topo})
            hs = sorted({k[3] for k in cells if k[0] == method and k[1] == topo})
            header = "| N \\ h | " + " | ".join(f"h={h:g}" for h in hs) + " |"
            sep = "|" + "---|" * (len(hs) + 1)
            lines.append(f"### {topo}\n")
            lines.append("Celda = chi necesario (S_vN máx en paréntesis).\n")
            lines.append(header)
            lines.append(sep)
            for N in Ns:
                cells_row = []
                for h in hs:
                    rr = cells.get((method, topo, N, h), [])
                    cn = chi_needed(rr, conv_tol)
                    s_vals = [r.get("entanglement_entropy") for r in rr
                              if r.get("entanglement_entropy") is not None]
                    s_max = max(s_vals) if s_vals else None
                    cell = "—" if cn is None else str(cn)
                    if s_max is not None:
                        cell += f" (S={s_max:.2f})"
                    cells_row.append(cell)
                lines.append(f"| {N} | " + " | ".join(cells_row) + " |")
            lines.append("")

    lines.append("\n## Veredicto de viabilidad\n")
    lines.append(
        "El chi necesario crece con la entropía de entrelazamiento S. Estados con "
        "S baja (lejos del punto crítico) son representables con chi chico → viable. "
        "S alta (crítico y/o N grande) empuja chi hacia valores impracticables → el "
        "método clásico deja de escalar. Los guiones y los chi grandes marcan esa "
        "frontera."
    )
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return [f"(figuras omitidas: {exc})"]

    FIGDIR.mkdir(parents=True, exist_ok=True)
    saved = []

    # E vs chi (una linea por (method,N,h)) — solo con abs_error disponible
    fig, ax = plt.subplots(figsize=(8, 5))
    series: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        if r.get("abs_error") is None:
            continue
        series[(r["method"], r["N"], r["h"])].append(
            (int(r["chi_requested"]), max(r["abs_error"], 1e-18)))
    for (m, N, h), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=4, label=f"{m} N={N} h={h:g}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("chi (bond dimension)")
    ax.set_ylabel("|ΔE| vs referencia")
    ax.set_title("Convergencia de energía vs bond dimension")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    p = FIGDIR / "E_vs_chi.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # entropy vs N (una linea por (method,h))
    fig, ax = plt.subplots(figsize=(8, 5))
    ent: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in rows:
        s = r.get("entanglement_entropy")
        if s is None:
            continue
        key = (r["method"], r["h"])
        ent[key][int(r["N"])] = max(ent[key].get(int(r["N"]), 0.0), float(s))
    for (m, h), byN in sorted(ent.items()):
        xs = sorted(byN)
        ax.plot(xs, [byN[n] for n in xs], marker="s", ms=5, label=f"{m} h={h:g}")
    ax.set_xlabel("N (qubits)")
    ax.set_ylabel("S_vN máx (corte central)")
    ax.set_title("Entropía de entrelazamiento vs tamaño del sistema")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    p = FIGDIR / "entropy_vs_N.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # mem vs chi
    fig, ax = plt.subplots(figsize=(8, 5))
    mem: dict[tuple, list[tuple]] = defaultdict(list)
    for r in rows:
        if r.get("mem_bytes_est") is None:
            continue
        mem[(r["method"], r["N"])].append((int(r["chi_requested"]), int(r["mem_bytes_est"])))
    for (m, N), pts in sorted(mem.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] / 1e6 for p in pts],
                marker="^", ms=4, label=f"{m} N={N}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("chi (bond dimension)")
    ax.set_ylabel("memoria estimada (MB)")
    ax.set_title("Costo de memoria del MPS vs bond dimension")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p = FIGDIR / "mem_vs_chi.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    return saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conv-tol", type=float, default=1e-6,
                    help="umbral |ΔE| para 'convergido' (default 1e-6)")
    args = ap.parse_args(argv)

    rows = load_all_rows()
    if not rows:
        print(f"No hay datos en {RESULTS_DIR}. Corré primero un runner de estudio.")
        return 1

    summary = build_summary(rows, args.conv_tol)
    (RESULTS_DIR / "SUMMARY.md").write_text(summary)
    print(f"escrito -> {RESULTS_DIR / 'SUMMARY.md'}  ({len(rows)} filas)")

    figs = _plot(rows)
    print("figuras:", ", ".join(figs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
