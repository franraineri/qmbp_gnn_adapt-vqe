#!/usr/bin/env python
"""2q-minimization Pareto sweep on a tilted PBC cluster, symmetry-tied ansatz.

The warm-start study established that on the tilted cluster (symmetric GS) the
TIED ansatz beats per-bond at a fraction of the parameters (N=9 p=2: 0.90 tied vs
0.63 per-bond). This runner now asks: what is the FEWEST 2q gates that still
reaches high fidelity? Because the point group is so rich that every nn bond is a
single orbit (and every nnn a single orbit), the only symmetry-preserving levers
are STRUCTURAL — which blocks, and how many layers. A partial orbit would break
the symmetry, so bond selection is binary per block.

It sweeps tied structures over {block sequence × depth}, each with the validated
warm-start machinery (calibrated seed → prepare_warmstart micro-descent → gap-
scaled restarts, cascade tiling across depth within a family), and traces the
2q↔fidelity Pareto front. Only arms NOT already in the JSON are run (idempotent:
re-running skips structures whose (blocks, rx_final) signature is recorded).

Structures (all tied, orbit-complete, so all symmetric):
  nn_only_p{p}  : [nn, x] * p              — drops the nnn block (half the 2q)
  full_p{p}     : [nn, nnn, x] * p         — the baseline family
Only the structures whose signature is new vs the existing JSON are executed; the
[nn,nnn,x]*1 and *2 tied results from run_tilted_warmstart are recognized and
reused (not re-run) when that JSON is present.

Reuses: tied_analytic_seed, _tile_tied, build_frustrated_h, _tilted_lattice,
cached_degenerate_ground_manifold, prepare_warmstart, restarts_for_gap,
converge_circuit, cx_and_params, subspace_fidelity, StudyPersister.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_pareto_2q.py \
        --n 9 --h 0.5 --j2 0.5 --families nn_only full --p-max 4
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
from run_tilted_warmstart import _tile_tied  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.analysis.warmstart import restarts_for_gap  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    make_cost_fid,
    prepare_warmstart,
)
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
TOPO_KEY = "square_tilted"
_FULL_SYMS = 100000

# Block sequence per family (one layer unit; repeated p times).
FAMILY_LAYER = {
    "nn_only": ["nn", "x"],  # drops nnn → half the 2q of full
    "full": ["nn", "nnn", "x"],  # baseline
    "nnn_only": ["nnn", "x"],  # diagnostic: nnn alone
}


def _sig(blocks, rx_final):
    """Signature string for a structure (for idempotent skip)."""
    return "+".join(blocks) + ("|rxf" if rx_final else "")


def _tied_seed_for(family, p, h, j2, rx_final):
    """Calibrated orbit-constant seed for a family layer repeated p times.

    tied_analytic_seed assumes the [nn,nnn,x] triple; for nn_only / nnn_only we
    build the per-layer scalars directly (same calibrated values, minus the
    absent block) so the seed length matches the circuit.
    """
    from qmbp_simulation.analysis.warmstart import calibrated_x_scale, calibrated_zz_coef

    zz = calibrated_zz_coef(h)
    xs = calibrated_x_scale(h)
    arctan_x = float(np.arctan(1.0 / h))
    nn_v, nnn_v, x_v = -zz, -(zz * j2), arctan_x * xs
    layer_map = {"nn": nn_v, "nnn": nnn_v, "x": x_v}
    layer = [layer_map[b] for b in FAMILY_LAYER[family]]
    seed = layer * p
    if rx_final:
        seed = seed + [x_v]
    return np.clip(np.asarray(seed, float), -np.pi, np.pi)


def _load_existing_sigs(path):
    """Signatures already converged, from this runner's JSON AND the warmstart JSON.

    Recognizes the [nn,nnn,x]*p tied results from run_tilted_warmstart so the full_p1
    / full_p2 structures are not re-run when that artifact exists.
    """
    sigs: dict[str, float] = {}
    for p in (path, path.parent / path.name.replace("pareto_2q", "warmstart")):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for row in data.get("rows", []):
            if row.get("arm", "").startswith("tied") and "structure_sig" in row:
                sigs[row["structure_sig"]] = row.get("report_fidelity")
            # warmstart JSON rows: reconstruct the full_p{p} signature. That runner
            # always used rx_final=True, so the signature carries the |rxf suffix.
            elif row.get("arm") == "tied-warm" and "p_layers" in row:
                blocks = ["nn", "nnn", "x"] * int(row["p_layers"])
                sigs[_sig(blocks, True)] = row.get("report_fidelity")
    return sigs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tilted PBC 2q-Pareto sweep (tied, symmetric)")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--families", nargs="+", default=["nn_only", "full"], choices=list(FAMILY_LAYER))
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--rx-final", action="store_true", default=True)
    ap.add_argument("--no-rx-final", dest="rx_final", action="store_false")
    ap.add_argument("--base-restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=99500)
    ap.add_argument("--target-fid", type=float, default=0.999)
    ap.add_argument("--ftol", type=float, default=1e-12)
    ap.add_argument("--force", action="store_true", help="re-run even if signature exists")
    args = ap.parse_args(argv)

    if args.n not in MAGIC_TILTS:
        print(f"N={args.n} not magic; available {sorted(MAGIC_TILTS)}")
        return 1

    from qmbp_simulation.execution import NoiselessBackend

    t_start = time.time()
    l1, l2 = MAGIC_TILTS[args.n]
    cl = generate_square_tilted(l1, l2)
    nn_edges, nnn_edges = cl.nn_edges, cl.nnn_edges
    H = build_frustrated_h(nn_edges, nnn_edges, cl.n_qubits, args.h, args.j2)
    manifold, e0, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=TOPO_KEY, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    psi = manifold[:, 0]
    degenerate = m_dim > 1
    mref = manifold if degenerate else manifold[:, :1]
    group = detect_symmetry_group(nn_edges, nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    n_restarts = restarts_for_gap(phys_gap, base_restarts=args.base_restarts)
    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()

    out_file = f"tilted_pareto_2q_N{args.n}_h{args.h:.2f}.json"
    out_path = STUDY_ROOT / SUBDIR / out_file
    existing = {} if args.force else _load_existing_sigs(out_path)
    print(
        f"[pareto2q] N={args.n} tilt=({l1},{l2}) h={args.h} group={group.order} md={m_dim} "
        f"gap={phys_gap:.5f} restarts={n_restarts} known_sigs={len(existing)} "
        f"(setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    def report_fid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), mref)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": TOPO_KEY, "h": args.h},
        extra={
            "topology": TOPO_KEY,
            "tilt": [l1, l2],
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "tilted_pareto_2q_v1",
            "manifold_dim": m_dim,
            "group_order": group.order,
            "phys_gap": phys_gap,
            "e0": e0,
            "families": list(args.families),
        },
        params={"experiment": "tilted_pareto_2q", "N": args.n, "h": args.h},
        description="Tilted PBC 2q-Pareto: tied symmetric structures, warm-start cascade",
        rows_ref=rows,
    )

    def _warm_seed(qc, raw):
        cost, fid, grad, _bk = make_cost_fid(qc, H, psi, backend=backend)
        ws = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=len(nn_edges),
            n_nnn=len(nnn_edges),
            n_qubits=args.n,
            p_layers=1,
            h=args.h,
            J=1.0,
            J2=args.j2,
            topology=TOPO_KEY,
            model=MODEL,
            gap=phys_gap,
            backend=backend,
            cost=cost,
            fid=fid,
            grad=grad,
            target_len=qc.num_parameters,
            auto_donors=False,
            extra_candidates=[(np.asarray(raw, float), "calibrated_tied")],
        )
        seed = np.asarray(ws["seed"], float)
        if seed.size != qc.num_parameters or not np.all(np.isfinite(seed)):
            seed = np.asarray(raw, float)
        return seed, ws.get("init_fidelity")

    def _n2q_of_sig(sig):
        """Transpiled 2q of a known structure, rebuilt from its block signature."""
        body, _, _ = sig.partition("|")
        blocks = body.split("+")
        rxf = sig.endswith("|rxf")
        qc, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=rxf
        )
        return cx_and_params(qc)[0]

    results: list[dict] = []
    # record known structures (for the Pareto table) without re-running them
    for sig, fid in existing.items():
        if fid is not None:
            results.append(
                {"structure_sig": sig, "report_fidelity": fid, "n_2q_transpiled": _n2q_of_sig(sig), "reused": True}
            )

    print("\n=== 2q-Pareto sweep (tied, symmetric) ===", flush=True)
    for family in args.families:
        prev_theta = None
        for p in range(1, args.p_max + 1):
            blocks = FAMILY_LAYER[family] * p
            sig = _sig(blocks, args.rx_final)
            qc, _ = builder.create_bond_resolved_symmetry_tied(
                cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=args.rx_final
            )
            n2q, npar = cx_and_params(qc)
            if sig in existing and not args.force:
                print(f"  [skip] {family}_p{p} ({sig}) 2q={n2q} — known fid={existing[sig]:.4f}", flush=True)
                # still thread the cascade: reconstruct nothing (no theta stored) → use calibrated
                continue

            if prev_theta is not None and prev_theta.size and family == "full":
                raw = _tile_tied(prev_theta, p - 1, p, args.rx_final)
                seed_kind = f"cascade<{family}_p{p - 1}>"
                if raw.size != npar:
                    raw = _tied_seed_for(family, p, args.h, args.j2, args.rx_final)
                    seed_kind = "calibrated"
            else:
                raw = _tied_seed_for(family, p, args.h, args.j2, args.rx_final)
                seed_kind = "calibrated"

            seed, init_fid = _warm_seed(qc, raw)
            t0 = time.time()
            fid, e_best, runs = converge_circuit(
                qc,
                H,
                psi,
                restarts=n_restarts,
                maxiter=args.maxiter,
                seed0=args.seed0,
                warm_theta=seed,
                backend=backend,
                target_fid=args.target_fid,
                ftol=args.ftol,
            )
            theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
            rfid = report_fid(qc, theta_best)
            row = {
                "arm": "tied",
                "family": family,
                "p_layers": p,
                "structure_sig": sig,
                "blocks": blocks,
                "rx_final": args.rx_final,
                "n_qubits": args.n,
                "tilt": [l1, l2],
                "h": round(float(args.h), 2),
                "topology": TOPO_KEY,
                "e0": e0,
                "delta_e": float(e_best) - e0,
                "manifold_dim": m_dim,
                "init_fidelity": (float(init_fid) if init_fid is not None else None),
                "report_fidelity": float(rfid),
                "n_2q_transpiled": n2q,
                "n_params": npar,
                "fidelity_per_cx": (rfid / n2q) if n2q else None,
                "seed_provenance": seed_kind,
                "theta_best": theta_best.tolist(),
            }
            rows.append(row)
            persister.persist(rows)
            results.append(row)
            prev_theta = theta_best
            print(
                f"  {family}_p{p:<2} ({sig}) fid={rfid:.4f} 2q={n2q:3d} p={npar:3d} "
                f"ΔE={row['delta_e']:+.4f} fpc={row['fidelity_per_cx']:.5f} "
                f"({seed_kind}, {time.time() - t0:.1f}s)",
                flush=True,
            )

    persister.persist(rows, status="final")

    # ── Pareto table: sort by 2q ascending; mark front (max fid at <= that 2q) ──
    print("\n=== 2q↔fidelity PARETO (ascending 2q) ===", flush=True)
    tbl = sorted([r for r in results if r.get("n_2q_transpiled")], key=lambda r: r["n_2q_transpiled"])
    best_so_far = -1.0
    for r in tbl:
        f = r["report_fidelity"]
        front = f > best_so_far + 1e-9
        best_so_far = max(best_so_far, f)
        mark = "◆ PARETO" if front else ""
        tag = r.get("structure_sig", "?")
        reused = " (reused)" if r.get("reused") else ""
        print(f"  2q={r['n_2q_transpiled']:>4}  fid={f:.4f}  {tag:<18}{reused}  {mark}", flush=True)

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
