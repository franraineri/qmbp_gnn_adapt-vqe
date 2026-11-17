#!/usr/bin/env python
"""Phase-1b warm-start on tilted PBC clusters: calibrated seed + micro-descent,
tied vs per-bond, with a p=1→p=2 cascade. No cold-start.

The seed probe (probe_tilted_seeds.py) showed the CALIBRATED analytic seed is the
best launch point on the tied tilted ansatz (3-50x over first-order), but its raw
init-fidelity falls with N (N=9 ~0.32, N=16 ~0.065). So the seed alone is not
enough at scale — this runner combines it with the study's standard machinery:

  - calibrated orbit-constant seed (tied_analytic_seed, the probe's winner),
  - prepare_warmstart's difficulty-adaptive micro-descent + candidate ranking
    (auto_donors OFF — the OBC donor corpus does not apply to this PBC model),
  - gap-scaled restarts (restarts_for_gap) refining FROM the seed (never cold),
  - a p=1→p=2 cascade: the converged tied θ at p=1 tiles into the p=2 seed,
  - per-bond arm seeded by EXPANDING the converged tied θ (the exact inverse of
    the tying), so per-bond starts from the symmetric optimum and can only refine.

Caching: the degenerate manifold is read from / written to the shared NPZ cache
under topology key "square_tilted" (NOT "square" — the OBC square manifold at the
same N/h is a DIFFERENT physical model: OBC N=16 is degenerate md=2 e0=-16.25,
tilted N=16 is md=1 e0=-17.13; sharing the key would silently load the wrong GS).

Comparison honesty: PBC is a different model than OBC. The valid quantitative
comparison is tied-warm vs per-bond-warm vs the earlier cold-start (same tilted
cluster). The OBC scoreboard at the same N is reported only as a qualitative
contrast (and at N=16 the OBC corpus has no h=0.5 entry at all).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_warmstart.py \
        --n 9 --h 0.5 --j2 0.5 --p-layers 1 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import StudyPersister  # noqa: E402
from probe_tilted_seeds import tied_analytic_seed  # noqa: E402  (the probe's winning seed)
from run_tilted_tied_ab import (  # noqa: E402
    _expand_tied_to_perbond,
    _tilted_lattice,
    build_frustrated_h,
)

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
TOPO_KEY = "square_tilted"  # distinct cache key from OBC "square" (different model)
_FULL_SYMS = 100000


def _tile_tied(theta_p1, p_from, p_to, rx_final):
    """Tile a converged tied θ at depth p_from into a seed at depth p_to.

    The tied layout is [nn, nnn, x] per layer (3 values) + one rx_final value.
    Repeating the per-layer triple p_to times and keeping the single rx_final
    value reuses the p_from solution as the p_to launch point (cascade).
    """
    theta_p1 = np.asarray(theta_p1, float)
    rx = [float(theta_p1[-1])] if rx_final else []
    body = theta_p1[:-1] if rx_final else theta_p1
    per = 3  # nn, nnn, x
    layers = [body[i * per : (i + 1) * per] for i in range(p_from)]
    # repeat the last converged layer to fill up to p_to
    out: list[float] = []
    for layer in range(p_to):
        src = layers[min(layer, p_from - 1)]
        out.extend(float(v) for v in src)
    return np.asarray(out + rx, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tilted PBC warm-start: calibrated seed + descent, tied vs perbond")
    ap.add_argument("--n", type=int, default=9, help="magic N (in MAGIC_TILTS)")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--p-layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--rx-final", action="store_true", default=True)
    ap.add_argument("--no-rx-final", dest="rx_final", action="store_false")
    ap.add_argument("--base-restarts", type=int, default=2, help="base restart count; scaled up by restarts_for_gap")
    ap.add_argument("--maxiter", type=int, default=800)
    ap.add_argument("--seed0", type=int, default=99000)
    ap.add_argument("--target-fid", type=float, default=0.999)
    ap.add_argument("--ftol", type=float, default=1e-12)
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

    # Shared manifold cache under the tilted-specific key (never the OBC "square").
    manifold, e0, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=TOPO_KEY, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    psi = manifold[:, 0]
    degenerate = m_dim > 1
    mref = manifold if degenerate else manifold[:, :1]
    group = detect_symmetry_group(nn_edges, nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    n_restarts = restarts_for_gap(phys_gap, base_restarts=args.base_restarts)
    print(
        f"[tilted_ws] N={args.n} tilt=({l1},{l2}) h={args.h} J2={args.j2} "
        f"group={group.order} md={m_dim} gap={phys_gap:.5f} e0={e0:.5f} "
        f"restarts={n_restarts} (setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()

    def report_fid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), mref)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"tilted_warmstart_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": TOPO_KEY, "h": args.h},
        extra={
            "topology": TOPO_KEY,
            "tilt": [l1, l2],
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "tilted_warmstart_v1",
            "manifold_dim": m_dim,
            "group_order": group.order,
            "phys_gap": phys_gap,
            "e0": e0,
            "p_sweep": list(args.p_layers),
            "rx_final": args.rx_final,
            "seed": "calibrated+micro_descent",
            "cold_start": False,
        },
        params={"experiment": "tilted_warmstart", "N": args.n, "h": args.h},
        description="Tilted PBC warm-start: calibrated seed + descent, tied vs perbond (no cold-start)",
        rows_ref=rows,
    )

    def _warm_seed(qc, raw_seed):
        """Refine raw_seed via prepare_warmstart's micro-descent + ranking.

        auto_donors OFF (OBC donor corpus does not apply to this PBC model); the
        calibrated orbit-constant seed is injected as the extra candidate so the
        ranked winner is never a length-mismatched OBC fallback.
        """
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
            extra_candidates=[(np.asarray(raw_seed, float), "calibrated_tied")],
        )
        seed = np.asarray(ws["seed"], float)
        if seed.size != qc.num_parameters or not np.all(np.isfinite(seed)):
            seed = np.asarray(raw_seed, float)
        return seed, ws.get("init_fidelity")

    def _record(arm, p_layers, qc, fid, e_best, runs, seed_prov, init_fid):
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        n2q, npar = cx_and_params(qc)
        rfid = report_fid(qc, theta_best)
        row = {
            "arm": arm,
            "p_layers": p_layers,
            "n_qubits": args.n,
            "tilt": [l1, l2],
            "h": round(float(args.h), 2),
            "topology": TOPO_KEY,
            "e0": e0,
            "phys_gap": phys_gap,
            "manifold_dim": m_dim,
            "degenerate": degenerate,
            "e_best": float(e_best),
            "delta_e": float(e_best) - e0,
            "init_fidelity": (float(init_fid) if init_fid is not None else None),
            "best_fidelity": float(fid),
            "report_fidelity": float(rfid),
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "fidelity_per_cx": (rfid / n2q) if n2q else None,
            "seed_provenance": seed_prov,
            "theta_best": theta_best.tolist(),
        }
        rows.append(row)
        persister.persist(rows)
        init_s = f" init={init_fid:.4f}" if init_fid is not None else ""
        print(
            f"  p={p_layers} {arm:12s} fid={rfid:.4f}{init_s} 2q={n2q:3d} p={npar:3d} "
            f"ΔE={row['delta_e']:+.4f} ({seed_prov})",
            flush=True,
        )
        return row, theta_best

    print("\n=== Warm-start p-sweep (no cold-start): tied then perbond ===", flush=True)
    summary: list[tuple] = []
    prev_tied_theta = None
    for p_layers in args.p_layers:
        blocks = ["nn", "nnn", "x"] * p_layers
        print(f"\n [p={p_layers}] blocks={blocks}", flush=True)

        qc_tied, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=args.rx_final
        )

        # Tied seed: cascade (tile previous p) if available, else calibrated analytic.
        if prev_tied_theta is not None:
            raw = _tile_tied(prev_tied_theta, p_layers - 1, p_layers, args.rx_final)
            seed_kind = f"cascade<p{p_layers - 1}>"
        else:
            raw = tied_analytic_seed("calibrated", p_layers, args.h, 1.0, args.j2, args.rx_final)
            seed_kind = "calibrated"
        if raw.size != qc_tied.num_parameters:
            raw = tied_analytic_seed("calibrated", p_layers, args.h, 1.0, args.j2, args.rx_final)
            seed_kind = "calibrated"
        seed_tied, init_t = _warm_seed(qc_tied, raw)

        t0 = time.time()
        fid_t, e_t, runs_t = converge_circuit(
            qc_tied,
            H,
            psi,
            restarts=n_restarts,
            maxiter=args.maxiter,
            seed0=args.seed0,
            warm_theta=seed_tied,
            backend=backend,
            target_fid=args.target_fid,
            ftol=args.ftol,
        )
        r_tied, theta_tied = _record("tied-warm", p_layers, qc_tied, fid_t, e_t, runs_t, seed_kind, init_t)
        print(f"    (tied {time.time() - t0:.1f}s)", flush=True)
        prev_tied_theta = theta_tied

        # Per-bond arm: expand the converged tied θ (exact inverse of the tying).
        qc_pb, _ = builder.create_bond_resolved_masked(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, rx_final=args.rx_final
        )
        seed_pb = _expand_tied_to_perbond(theta_tied, blocks, nn_edges, nnn_edges, cl.n_qubits, group, args.rx_final)
        assert seed_pb.size == qc_pb.num_parameters
        init_pb = report_fid(qc_pb, seed_pb)
        t0 = time.time()
        fid_p, e_p, runs_p = converge_circuit(
            qc_pb,
            H,
            psi,
            restarts=n_restarts,
            maxiter=args.maxiter,
            seed0=args.seed0,
            warm_theta=seed_pb,
            backend=backend,
            target_fid=args.target_fid,
            ftol=args.ftol,
        )
        r_pb, _ = _record("perbond-warm", p_layers, qc_pb, fid_p, e_p, runs_p, "tied_expanded+refine", init_pb)
        print(f"    (perbond {time.time() - t0:.1f}s)", flush=True)

        summary.append((p_layers, r_tied, r_pb))

    persister.persist(rows, status="final")

    print("\n=== SUMMARY (per depth) ===", flush=True)
    for p_layers, r_tied, r_pb in summary:
        d = r_tied["report_fidelity"] - r_pb["report_fidelity"]
        winner = "TIED" if d > 1e-4 else ("perbond" if d < -1e-4 else "tie")
        red = r_pb["n_params"] / max(r_tied["n_params"], 1)
        print(
            f"  p={p_layers} 2q={r_tied['n_2q_transpiled']:3d}  "
            f"tied={r_tied['report_fidelity']:.4f}({r_tied['n_params']}p)  "
            f"perbond={r_pb['report_fidelity']:.4f}({r_pb['n_params']}p)  "
            f"Δ={d:+.4f} → {winner} ({red:.0f}x fewer params)",
            flush=True,
        )

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
