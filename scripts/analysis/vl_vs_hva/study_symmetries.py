#!/usr/bin/env python
"""Deterministic symmetry study of a frustrated-TFIM lattice — thin CLI.

A command-line front-end over the reusable symmetry module
(:mod:`qmbp_simulation.circuits.symmetry`): it builds the lattice + Hamiltonian,
delegates ALL symmetry logic to :func:`analyze_symmetry` (group detection via
graph automorphisms, bond/site orbits, parameter reduction, ground-state
invariance), prints a readable summary, and persists the resulting
:class:`SymmetryReport` as JSON for downstream analysis.

No symmetry logic lives here — the script only wires inputs/outputs. The orbit
and invariance primitives are shared with the ansatz builder so the study and the
circuit agree by construction.

Reuses: make_lattice + HamiltonianBuilder._generate_nnn_edges (lattice/edges),
get_model_spec + degenerate_ground_manifold (exact ground manifold for the
physical check), analysis.symmetry.analyze_symmetry (everything else).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/study_symmetries.py --n 20 --h 0.5 --gs --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT  # noqa: E402

from qmbp_simulation.circuits.symmetry import analyze_symmetry  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402


def _exact_ground_state(topology: str, n: int, h: float, j2: float, model: str):
    """Exact ground manifold for the physical invariance check — reuses the
    shared degenerate_ground_manifold (same source of truth as the study/specs).

    Returns ``(psi0, manifold_dim)`` where psi0 is one ground-manifold vector.
    """
    from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(model)
    hk = dict(getattr(spec, "hamiltonian_kwargs", {}))
    hk["J2"] = j2
    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **hk)
    eigvecs, _e0, _gap, m = degenerate_ground_manifold(H)
    return np.asarray(eigvecs[:, 0], complex), m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Symmetry study (thin CLI over analysis.symmetry)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--topology", default="square")
    ap.add_argument("--model", default="tfim_frustrated")
    ap.add_argument(
        "--gs",
        action="store_true",
        help="Also check exact ground-state invariance (eigsh, dim 2^N; fine up to ~N=20, adds a minute).",
    )
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    lat = make_lattice(args.topology, args.n, J=1.0, h=args.h)
    nn = [(int(i), int(j)) for i, j in lat.edges]
    nnn = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]

    psi0, manifold_dim = (None, None)
    if args.gs:
        psi0, manifold_dim = _exact_ground_state(args.topology, args.n, args.h, args.j2, args.model)

    group, report = analyze_symmetry(
        nn,
        nnn,
        args.n,
        topology=args.topology,
        h=args.h,
        j2=args.j2,
        ground_state=psi0,
        manifold_dim=manifold_dim,
    )

    print(f"=== SYMMETRY STUDY — {args.topology} N={args.n} h={args.h} J2={args.j2} ===\n")
    print(f"nn edges: {report.n_nn}   nnn edges: {report.n_nnn}")
    print(f"\nsymmetry group order: {report.group_order} ({len(group.perms)} non-identity + identity)")
    print("\norbits (bonds/sites that must share one tied parameter):")
    print(f"  nn  : {report.n_nn} → {report.nn_orbits} orbits (sizes {dict(sorted(report.nn_orbit_sizes.items()))})")
    print(f"  nnn : {report.n_nnn} → {report.nnn_orbits} orbits (sizes {dict(sorted(report.nnn_orbit_sizes.items()))})")
    print(f"  site: {args.n} → {report.site_orbits} orbits")
    print(
        f"\nPER-LAYER params: {report.params_per_layer_independent} independent "
        f"→ {report.params_per_layer_tied} tied (×{report.reduction_factor:.2f} fewer)"
    )

    if report.gs_invariance is not None:
        print(f"\nground-state invariance (|⟨ψ|P|ψ⟩|, 1.0 = invariant; manifold_dim={report.manifold_dim}):")
        for name, val in report.gs_invariance.items():
            flag = "(invariant)" if val > 0.99 else "(NOT invariant — manifold mixes sectors?)"
            print(f"  {name:8s} = {val:.4f}  {flag}")

    if args.save:
        import json

        p = STUDY_ROOT / "reports" / f"symmetry_study_{args.topology}_N{args.n}_h{args.h:.2f}.json"
        p.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n[saved] {p.relative_to(STUDY_ROOT.parent.parent)}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
