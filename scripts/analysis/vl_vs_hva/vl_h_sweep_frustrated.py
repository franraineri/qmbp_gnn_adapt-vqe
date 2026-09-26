#!/usr/bin/env python
"""h-sweep of Haiqu VL vs HVA in the frustrated regime (tfim_frustrated, J2>0).

Sweeps h for fixed J2 to map where the VL loader degrades in the (h, J2) plane.
Reuses compare() from compare_circuits_square_n10_p2 (VL real circuit + fidelity,
HVA structural stats, per-config PNGs). Consolidates every scenario into one
JSON, persisted incrementally after each point (crash-safe).

Requires HAIQU_API_KEY.

Usage:
    HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_h_sweep_frustrated.py \
        --topologies square triangular --n 9 --p 2 --j2 0.5 --h 0.5 1.0 2.0 3.0
"""

from __future__ import annotations

import argparse

from compare_circuits_square_n10_p2 import compare  # noqa: E402 (same dir)
from hva_vl_study_common import save_json


def run(topologies, n, p, j2, h_values, model="tfim_frustrated") -> list[dict]:
    rows: list[dict] = []
    for topo in topologies:
        for h in h_values:
            try:
                a = compare(topo, n, p, h, model=model, j2=j2)
                vl = a.get("vl") or {}
                rows.append(
                    {
                        "topology": topo, "N": n, "p_layers": p, "model": model,
                        "J2": j2, "h": h,
                        "E0": a.get("E0"), "gap": a.get("gap"), "n_edges": a.get("n_edges"),
                        "vl_fidelity": a.get("vl_fidelity"),
                        "vl_status": a.get("vl_status"),
                        "vl_energy": a.get("vl_energy"), "vl_abs_error": a.get("vl_abs_error"),
                        "vl_depth": vl.get("depth"), "vl_2q": vl.get("n_2q_gates"),
                        "hva_transpiled_2q": (a.get("hva_transpiled") or {}).get("n_2q_gates"),
                        "vl_error": a.get("vl_error"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — never abort the batch
                rows.append({"topology": topo, "N": n, "h": h, "J2": j2,
                             "error": f"{type(exc).__name__}: {exc}"})
            save_json(
                {"rows": rows},
                "vl_h_sweep", "vl_h_sweep_frustrated.json",
                params={"N": n, "p_layers": p, "J2": j2, "model": model},
                description="h-sweep of Haiqu VL vs HVA in frustrated 2D",
            )
    return rows


def format_rows(rows) -> str:
    out = ["=" * 78,
           f"{'topo':<11}{'h':<6}{'gap':<9}{'VL fid':<10}{'VL 2q':<7}{'HVA 2q':<7}"]
    out.append("-" * 78)
    for r in rows:
        if "error" in r:
            out.append(f"{r['topology']:<11}{r['h']:<6}ERROR: {r['error'][:40]}")
            continue
        fid = r.get("vl_fidelity")
        fs = f"{fid:.4f}" if isinstance(fid, (int, float)) else str(fid)
        gap = r.get("gap")
        gs = f"{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)
        out.append(
            f"{r['topology']:<11}{r['h']:<6}{gs:<9}{fs:<10}"
            f"{str(r.get('vl_2q')):<7}{str(r.get('hva_transpiled_2q')):<7}"
        )
    out.append("=" * 78)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="h-sweep VL vs HVA, frustrated regime")
    p.add_argument("--topologies", nargs="+", default=["square", "triangular"])
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run(args.topologies, args.n, args.p, args.j2, args.h)
    print("\n" + format_rows(rows))
    from hva_vl_study_common import study_dir

    print(f"\n→ Resultados guardados en: {study_dir('vl_h_sweep') / 'vl_h_sweep_frustrated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
