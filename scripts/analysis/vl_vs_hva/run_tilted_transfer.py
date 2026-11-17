#!/usr/bin/env python
"""Cross-N transfer on tilted PBC: seed N=16 tied with the converged N=9 tied θ.

The tilted N=16 p-sweep stalled (p2=0.30, p3/p4 worse) — a pure OPTIMIZATION wall,
not a physical one: the calibrated seed starts at init~0.065 at N=16 and L-BFGS-B
with a modest budget never reaches the GS basin (p4 ended 10 gaps above E0). N=9
by contrast converged to 0.996.

Key enabler: the TIED ansatz has the SAME parameter count at every N (one scalar
per orbit per block, and every block is a single orbit on both the N=9 and N=16
tilted clusters). So the converged N=9 tied θ (p=3: 10 params @0.9826, p=4: 13
params @0.9960) transfers VERBATIM as the N=16 seed at the same depth — no bond
remapping, an exact drop-in. Physically motivated: the optimal per-orbit angles
of the frustrated GS should be close across N (same frustration, same orbit
structure). This tests whether a GOOD seed (vs the poor calibrated one) lets N=16
reach the basin the calibrated start could not.

Reuses: the N=9 θ from the pareto JSON, build_frustrated_h / _tilted_lattice,
cached_degenerate_ground_manifold, the tied builder, converge_circuit,
cx_and_params, subspace_fidelity, StudyPersister.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_transfer.py \
        --src-n 9 --dst-n 16 --h 0.5 --p-layers 3 4 --restarts 3 --maxiter 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT, StudyPersister  # noqa: E402
from run_tilted_tied_ab import _tilted_lattice, build_frustrated_h  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    make_cost_fid,
)
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
TOPO_KEY = "square_tilted"
_FULL_SYMS = 100000


def optimize_fidelity(qc, H, psi, seed, *, restarts, maxiter, seed0, ftol):
    """Minimize INFIDELITY 1-|<psi|ψ(θ)>|² (not energy) via multi-seed L-BFGS-B.

    The transfer study found converge_circuit (energy objective) DEGRADES a
    high-fidelity transferred seed at N=16: the tied-ansatz energy minimum does
    not coincide with GS fidelity. This optimizes the fidelity directly. Uses the
    shared make_cost_fid fid(x) and the shared _lbfgsb (finite-difference grad —
    no analytic fidelity gradient). Restart 0 is the warm seed; the rest are small
    perturbations of it (we want to stay near the good seed, not random-restart
    the whole box). Returns (best_fid, best_theta, runs).
    """
    _cost, fid, _grad, _bk = make_cost_fid(qc, H, psi)

    def infid(x):
        return 1.0 - fid(x)

    rng = np.random.default_rng(seed0)
    seed = np.asarray(seed, float)
    runs = []
    best_fid, best_theta = -1.0, None
    for r in range(restarts):
        x0 = seed if r == 0 else np.clip(seed + rng.normal(scale=0.15, size=seed.size), -np.pi, np.pi)
        xf, _inf, _nit = _lbfgsb(infid, x0, maxiter=maxiter, grad=None, ftol=ftol)
        f = fid(xf)
        runs.append({"theta_final": xf.tolist(), "fidelity": f, "energy": 1.0 - f})
        if f > best_fid:
            best_fid, best_theta = f, xf
    return best_fid, np.asarray(best_theta, float), runs


def _load_src_tied_theta(src_n, h, p_layers):
    """The converged N=src tied θ for a full [nn,nnn,x]*p (+rx_final) at depth p.

    Reads the pareto JSON; the full-family row at depth p has structure_sig with p
    'nnn' blocks. Returns the theta list (length 3*p+1) or None.
    """
    path = STUDY_ROOT / SUBDIR / f"tilted_pareto_2q_N{src_n}_h{h:.2f}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for r in data.get("rows", []):
        sig = r.get("structure_sig", "")
        if sig.count("nnn") == p_layers and r.get("theta_best"):
            return np.asarray(r["theta_best"], float), r.get("report_fidelity")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tilted cross-N transfer: N=src tied θ → N=dst seed")
    ap.add_argument("--src-n", type=int, default=9)
    ap.add_argument("--dst-n", type=int, default=16)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--p-layers", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--rx-final", action="store_true", default=True)
    ap.add_argument("--no-rx-final", dest="rx_final", action="store_false")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=99900)
    ap.add_argument("--target-fid", type=float, default=0.999)
    ap.add_argument("--ftol", type=float, default=1e-12)
    ap.add_argument(
        "--objective",
        choices=["energy", "fidelity", "both"],
        default="both",
        help="energy = converge_circuit (⟨H⟩); fidelity = minimize 1-|<GS|psi>|²; "
        "both = run both arms from the same transferred seed for the A/B.",
    )
    args = ap.parse_args(argv)

    for n in (args.src_n, args.dst_n):
        if n not in MAGIC_TILTS:
            print(f"N={n} not magic; available {sorted(MAGIC_TILTS)}")
            return 1

    from qmbp_simulation.execution import NoiselessBackend

    t_start = time.time()
    l1, l2 = MAGIC_TILTS[args.dst_n]
    cl = generate_square_tilted(l1, l2)
    nn_edges, nnn_edges = cl.nn_edges, cl.nnn_edges
    H = build_frustrated_h(nn_edges, nnn_edges, cl.n_qubits, args.h, args.j2)
    manifold, e0, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=TOPO_KEY, n_qubits=args.dst_n, model=MODEL, h=args.h, j2=args.j2
    )
    psi = manifold[:, 0]
    degenerate = m_dim > 1
    mref = manifold if degenerate else manifold[:, :1]
    group = detect_symmetry_group(nn_edges, nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()
    print(
        f"[transfer] N={args.src_n}→{args.dst_n} tilt=({l1},{l2}) h={args.h} "
        f"group={group.order} md={m_dim} gap={phys_gap:.5f} e0={e0:.5f} "
        f"(setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    def report_fid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), mref)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"tilted_transfer_N{args.src_n}to{args.dst_n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.dst_n, "p_layers": None, "model": MODEL, "topology": TOPO_KEY, "h": args.h},
        extra={
            "topology": TOPO_KEY,
            "tilt": [l1, l2],
            "src_n": args.src_n,
            "dst_n": args.dst_n,
            "h": args.h,
            "J2": args.j2,
            "schema": "tilted_transfer_v1",
            "manifold_dim": m_dim,
            "group_order": group.order,
            "phys_gap": phys_gap,
            "e0": e0,
        },
        params={"experiment": "tilted_transfer", "src_n": args.src_n, "dst_n": args.dst_n, "h": args.h},
        description=f"Tilted cross-N transfer: N={args.src_n} tied θ seeds N={args.dst_n}",
        rows_ref=rows,
    )

    print(f"\n=== transfer N={args.src_n} tied θ → N={args.dst_n} seed (per depth) ===", flush=True)
    for p in args.p_layers:
        blocks = ["nn", "nnn", "x"] * p
        qc, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=args.rx_final
        )
        n2q, npar = cx_and_params(qc)

        src = _load_src_tied_theta(args.src_n, args.h, p)
        if src is None:
            print(f"  p={p}: no N={args.src_n} tied θ for this depth — skip", flush=True)
            continue
        seed, src_fid = src
        if seed.size != npar:
            print(f"  p={p}: src θ len {seed.size} != dst params {npar} — skip", flush=True)
            continue

        init_fid = report_fid(qc, seed)  # how good is the N=src θ AS-IS on N=dst?
        arms = ["energy", "fidelity"] if args.objective == "both" else [args.objective]

        for obj in arms:
            t0 = time.time()
            if obj == "energy":
                _f, e_best, runs = converge_circuit(
                    qc,
                    H,
                    psi,
                    restarts=args.restarts,
                    maxiter=args.maxiter,
                    seed0=args.seed0,
                    warm_theta=seed,
                    backend=backend,
                    target_fid=args.target_fid,
                    ftol=args.ftol,
                )
                theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
                delta_e = float(e_best) - e0
            else:  # fidelity — minimize infidelity directly
                _bf, theta_best, runs = optimize_fidelity(
                    qc, H, psi, seed, restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, ftol=args.ftol
                )
                delta_e = None
            rfid = report_fid(qc, theta_best)
            row = {
                "arm": f"transfer_{obj}",
                "objective": obj,
                "p_layers": p,
                "src_n": args.src_n,
                "dst_n": args.dst_n,
                "h": round(float(args.h), 2),
                "topology": TOPO_KEY,
                "e0": e0,
                "delta_e": delta_e,
                "manifold_dim": m_dim,
                "src_fidelity": float(src_fid) if src_fid is not None else None,
                "transfer_init_fidelity": float(init_fid),
                "report_fidelity": float(rfid),
                "n_2q_transpiled": n2q,
                "n_params": npar,
                "fidelity_per_cx": (rfid / n2q) if n2q else None,
                "theta_best": theta_best.tolist(),
            }
            rows.append(row)
            persister.persist(rows)
            de_s = f" ΔE={delta_e:+.4f}" if delta_e is not None else ""
            print(
                f"  p={p} [{obj:9s}] 2q={n2q:3d} params={npar:2d}  srcN{args.src_n}={src_fid:.4f}  "
                f"init={init_fid:.4f} →{rfid:.4f}{de_s} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    persister.persist(rows, status="final")

    print("\n=== SUMMARY: energy vs fidelity objective from the transferred seed ===", flush=True)
    print(f"  calibrated p-sweep N={args.dst_n} (energy, stalled): p2=0.3031 p3=0.2793 p4=0.1000", flush=True)
    for r in rows:
        print(
            f"  p={r['p_layers']} [{r['objective']:9s}]: init={r['transfer_init_fidelity']:.4f} "
            f"→ {r['report_fidelity']:.4f} @ {r['n_2q_transpiled']} 2q",
            flush=True,
        )
    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
