#!/usr/bin/env python
"""Circuit comparison: HVA (p=2) vs Haiqu VL for square N=10 at h=1.0.

Saves a PNG of each circuit and a structural analysis of how each is built.
- HVA: bond-resolved ansatz, built locally (Qiskit).
- VL: Haiqu vector_loading on the exact ground state, retrieved as qpy.

Requires HAIQU_API_KEY for the VL branch.

Usage:
    HAIQU_API_KEY=... .venv/bin/python scripts/analysis/compare_circuits_square_n10_p2.py
"""

from __future__ import annotations

import os
from pathlib import Path

from hva_vl_study_common import (
    circuit_energy,
    circuit_stats,
    exact_ground_state_vector,
    find_repo_root,
    haiqu_session,
    run_vl_job,
    save_circuit_qpy,
    save_json,
    study_dir,
    transpile_hw,
)

# Move-resilient repo-root resolution (never count parents[N] — that silently
# broke when this dir moved one level deeper).
_REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
OUTDIR = study_dir("circuit_comparison")


def _analyze(qc, label: str) -> dict:
    stats = circuit_stats(qc)
    stats["label"] = label
    return stats


def _save_png(qc, path: Path, title: str, fold: int = 40) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = qc.draw("mpl", fold=fold, idle_wires=False)
        fig.suptitle(title, fontsize=10)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  PNG failed for {title}: {type(exc).__name__}: {exc}")
        return False


def compare(topology: str, n: int, p: int, h: float, model: str = "tfim", j2: float = 0.0) -> dict:
    from qmbp_simulation import HVACircuitBuilder, make_lattice

    tag = f"{model}_{topology}_n{n}_p{p}_h{h:.2f}" + (f"_j2{j2:.2f}" if j2 else "")

    lat = make_lattice(topology, n, J=1.0, h=h)
    # Shared solve: cached (E0, gap) via GroundTruthCache + exact vector + H
    # (frustrated forwards J2 so the cache key stays distinct from J2=0).
    psi, e0, gap, H_op = exact_ground_state_vector(
        topology, n, h, model=model, j2=j2, return_hamiltonian=True
    )

    analysis: dict = {"topology": topology, "N": n, "p_layers": p, "h": h,
                      "model": model, "J2": j2,
                      "E0": e0, "gap": gap,
                      "n_edges": len(lat.edges)}

    hva, _ = HVACircuitBuilder().create_bond_resolved(n, p, lat)
    analysis["hva_logical"] = _analyze(hva, f"HVA p={p} (logical)")
    _save_png(hva, OUTDIR / f"{tag}_hva_logical.png",
              f"HVA bond-resolved p={p} — {topology} N={n} (logical, {hva.num_parameters} params)")
    hva_t = transpile_hw(hva, optimization_level=1)
    analysis["hva_transpiled"] = _analyze(hva_t, f"HVA p={p} (transpiled cx/rz/sx/x)")
    _save_png(hva_t, OUTDIR / f"{tag}_hva_transpiled.png",
              f"HVA p={p} transpiled (cx/rz/sx/x) — {topology} N={n}")

    if os.environ.get("HAIQU_API_KEY"):
        try:
            with haiqu_session(f"VL_vs_HVA_{tag}"):
                fid, status, vl, cid, vl_meta = run_vl_job(psi, n, num_layers=2)
            analysis["vl_fidelity"] = fid
            analysis["vl_status"] = status
            analysis["vl_circuit_id"] = cid
            analysis["vl_meta"] = vl_meta
            if vl is not None:
                analysis["vl"] = _analyze(vl, "Haiqu VL circuit")
                # Energy of the VL-prepared state and its error vs the exact GS.
                vl_energy = circuit_energy(vl, H_op)
                analysis["vl_energy"] = vl_energy
                analysis["vl_abs_error"] = abs(vl_energy - e0)
                # Persist the circuit locally (QPY) for later re-analysis without Haiqu.
                qpy_path = OUTDIR / "circuits" / f"{tag}_vl.qpy"
                save_circuit_qpy(vl, qpy_path)
                analysis["vl_circuit_qpy"] = str(qpy_path.relative_to(_REPO_ROOT))
                _save_png(vl, OUTDIR / f"{tag}_vl.png",
                          f"Haiqu VL circuit — {topology} N={n} (fid={fid})")
        except Exception as exc:  # noqa: BLE001
            analysis["vl_error"] = f"{type(exc).__name__}: {exc}"
    else:
        analysis["vl_error"] = "HAIQU_API_KEY not set"

    result_path = save_json(analysis, "circuit_comparison", f"{tag}_analysis.json")
    _print_report(analysis)
    print(f"\n→ Figuras + análisis en {OUTDIR}/ (prefix {tag})")
    print(f"→ JSON de resultados: {result_path}")
    return analysis


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Compare HVA ansatz vs Haiqu VL circuit")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--model", default="tfim")
    ap.add_argument("--j2", type=float, default=0.0)
    args = ap.parse_args(argv)
    compare(args.topology, args.n, args.p, args.h, model=args.model, j2=args.j2)
    return 0


def _print_report(a: dict) -> None:
    print("=" * 74)
    print(f"Circuit comparison — {a['topology']} N={a['N']} p={a['p_layers']} h={a['h']}")
    print(f"E0={a['E0']:.4f}  gap={a['gap']:.4f}  lattice edges={a['n_edges']}")
    print("=" * 74)
    for k in ("hva_logical", "hva_transpiled", "vl"):
        s = a.get(k)
        if not s:
            continue
        print(f"\n{s['label']}:")
        print(f"  qubits={s['num_qubits']}  params={s['num_parameters']}  "
              f"total_gates={s['total_gates']}")
        print(f"  depth={s['depth']}  depth_2q={s['depth_2q']}  "
              f"2q_gates={s['n_2q_gates']}  1q_gates={s['n_1q_gates']}")
        print(f"  gate_counts={s['gate_counts']}")
    if "vl_fidelity" in a:
        print(f"\nVL fidelity: {a['vl_fidelity']}  status={a.get('vl_status')}")
    if "vl_error" in a:
        print(f"\nVL branch skipped: {a['vl_error']}")


if __name__ == "__main__":
    raise SystemExit(main())
