#!/usr/bin/env python
"""E1: analytic seed vs highest-fidelity optimum, per configuration.

For every saved configuration (N, h, p=2) with a persisted optimized θ, this
loads the BEST (highest-fidelity) θ, re-verifies its fidelity by re-simulation,
and contrasts it against the analytic second-order seed — block by block, layer
by layer, with sign-degeneracy control — via
:func:`qmbp_simulation.analysis.warmstart.seed_vs_optimum_report`.

Goal: is the "layer-0 low-X / layer-1 high-X + ZZ shrink" pattern systematic
across N and h, or specific to one point? Free (no VQE — reads saved θ only).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_seed_vs_optimum.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def _iter_theta_candidates(study: Path):
    """Yield (N, h, p, fidelity, theta_list, source) from every artifact."""
    root = study / "hva_nnn_sweep"
    for f in sorted(glob.glob(str(root / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        name = Path(f).name
        rows = d.get("rows")
        if isinstance(rows, list):
            for r in rows:
                th = r.get("theta_final") or r.get("best_theta_final")
                fid = r.get("fidelity") or r.get("best_fidelity")
                if th and fid is not None:
                    yield (
                        d.get("N", r.get("N")),
                        d.get("h", r.get("h")),
                        r.get("p_layers", d.get("p_layers", 2)),
                        float(fid),
                        th,
                        f"{name}:{r.get('variant', r.get('seed_type', '?'))}",
                    )
        # dual_target style (single best per method) — skip (no per-param theta list here)


def main(argv=None) -> int:
    from qiskit.quantum_info import Statevector

    from qmbp_simulation.analysis.warmstart import seed_vs_optimum_report
    from qmbp_simulation.framework.runner_base import resolve_project_root
    from qmbp_simulation.framework.study_core import ground_state
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    root = resolve_project_root(Path(__file__))
    study = root / "results" / "hva_vl_study"

    # Best theta per (N, h, p) by re-verified fidelity, standard [nn,nnn,x]*p layout only.
    best: dict[tuple, dict] = {}
    for N, h, p, fid, th, src in _iter_theta_candidates(study):
        if N is None or h is None:
            continue
        try:
            n = int(N)
            hf = round(float(h), 2)
            pp = int(p)
        except (TypeError, ValueError):
            continue
        theta = np.asarray(th, float)
        # Only the standard layout: len == p*(n_nn+n_nnn+n). Compute expected len.
        lat_probe = None
        try:
            from qmbp_simulation.models import make_lattice

            lat_probe = make_lattice("square", n, J=1.0, h=hf)
        except Exception:
            continue
        n_nn = len(lat_probe.edges)
        n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat_probe))
        expected = pp * (n_nn + n_nnn + n)
        if len(theta) != expected:
            continue  # variant / non-standard layout → skip (seed compares to standard)
        key = (n, hf, pp)
        if key not in best or fid > best[key]["fid"]:
            best[key] = {"fid": fid, "theta": theta, "src": src, "n_nn": n_nn, "n_nnn": n_nnn}

    results = {
        "schema": "seed_vs_optimum_v1",
        "topology": "square",
        "model": "tfim_frustrated",
        "J2": 0.5,
        "configs": [],
    }

    print(f"{'N':>3} {'h':>5} {'p':>2} {'F':>7}  per-layer θ_x (seed→opt) | ZZ ratio_abs")
    print("-" * 78)
    for n, hf, pp in sorted(best):
        rec = best[(n, hf, pp)]
        # Re-verify fidelity by re-simulation (trust but verify).
        lat, qc, H, psi, e0, gap, _, _ = ground_state("square", n, hf, 0.5, pp)
        if qc.num_parameters != len(rec["theta"]):
            continue
        sv = np.asarray(Statevector(qc.assign_parameters(rec["theta"])).data)
        fid_check = float(abs(np.vdot(psi, sv)) ** 2)
        report = seed_vs_optimum_report(rec["theta"], rec["n_nn"], rec["n_nnn"], n, pp, hf, J=1.0, J2=0.5)
        report["N"] = n
        report["fidelity_reported"] = rec["fid"]
        report["fidelity_reverified"] = fid_check
        report["source"] = rec["src"]
        report["gap"] = gap
        results["configs"].append(report)

        # Compact console line: θ_x seed→opt per layer + ZZ ratio.
        xs = " ".join(
            f"L{i}:{ly['x']['seed']:+.2f}→{ly['x']['opt_mean']:+.2f}" for i, ly in enumerate(report["layers"])
        )
        zzr = " ".join(
            f"L{i}:nn{ly['nn']['ratio_abs']:.2f}/nnn{ly['nnn']['ratio_abs']:.2f}"
            for i, ly in enumerate(report["layers"])
        )
        print(f"{n:>3} {hf:>5.2f} {pp:>2} {fid_check:>7.4f}  {xs}  | {zzr}")

    save_json(
        results,
        "hva_nnn_sweep",
        "seed_vs_optimum_square.json",
        params={"experiment": "seed_vs_optimum"},
        description="E1: analytic 2nd-order seed vs highest-fidelity optimized "
        "θ per (N,h,p), per-layer per-block, sign-degeneracy-aware",
    )
    print(f"\n[E1] {len(results['configs'])} configs analyzed → seed_vs_optimum_square.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
