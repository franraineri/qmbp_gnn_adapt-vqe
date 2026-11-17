#!/usr/bin/env python
"""A/B: does a SYMMETRY-PROJECTED warm-start beat the raw analytic seed?

The converged optimum of the full bond-resolved ansatz is nearly orbit-constant
(within-orbit |θ| spread ~0.01-0.1 at N=9), i.e. it lives near the symmetric
subspace. Hypothesis: projecting the analytic seed onto that subspace (orbit-
average, KEEPING per-bond width — no tying, no expressivity loss) starts the
optimizer closer to the optimum, so it converges to higher fidelity / in fewer
iterations than the raw seed.

Robust design: for K independent perturbations (gaussian noise, distinct RNG
seeds) applied to the SAME base seed, run ONE L-BFGS-B (restarts=1) from each of
the two arms — raw seed vs symmetrized seed — on the identical circuit. A single
restart makes the START POINT the dominant factor (many restarts would wash out
the warm-start). Reports per-arm fidelity mean/std, L-BFGS-B iteration count, and
the init fidelity of the seed itself.

Reuses: ground_state, make_cost_fid, prepare_warmstart (structure-aware seed),
symmetrize_seed, detect_symmetry_group, subspace_fidelity. scipy L-BFGS-B directly
(to expose the iteration count that converge_circuit hides).
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

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group, symmetrize_seed  # noqa: E402
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
BLOCKS = ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"]
RX_FINAL = True


def _lbfgsb_with_iters(cost, grad, x0, maxiter, ftol):
    """Run L-BFGS-B; return (x_opt, n_iters). Exposes the iteration count."""
    from scipy.optimize import minimize

    res = minimize(
        cost,
        np.asarray(x0, float),
        method="L-BFGS-B",
        jac=grad,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": 1e-8},
    )
    return res.x, int(res.nit)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A/B symmetrized vs raw warm-start")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--trials", type=int, default=12, help="independent noise perturbations (robustness).")
    p.add_argument("--noise", type=float, default=0.15, help="gaussian noise std added to the base seed per trial.")
    p.add_argument("--maxiter", type=int, default=150, help="L-BFGS-B iters per single-restart run.")
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--micro-descent", type=int, default=60)
    p.add_argument("--seed0", type=int, default=30000)
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    lat, _q, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn = [(int(i), int(j)) for i, j in lat.edges]
    nnn = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1
    group = detect_symmetry_group(nn, nnn, args.n)

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    sel = BondSelection(nn_edges=nn, nnn_edges=nnn)
    qc, _ = builder.create_bond_resolved_masked(args.n, lat, blocks=BLOCKS, bond_selection=sel, rx_final=RX_FINAL)
    cost, fid_fn, grad, _ = make_cost_fid(qc, H, psi, backend=backend)
    npar = qc.num_parameters

    # Base analytic seed (structure-aware), and its symmetry-projected version.
    ws = prepare_warmstart(
        qc,
        H,
        psi,
        n_nn=n_nn,
        n_nnn=n_nnn,
        n_qubits=args.n,
        p_layers=1,
        h=args.h,
        J2=args.j2,
        backend=backend,
        topology=args.topology,
        model=MODEL,
        gap=gap,
        target_len=npar,
        auto_donors=False,
        micro_descent=args.micro_descent,
        blocks=BLOCKS,
        rx_final=RX_FINAL,
    )
    base_raw = np.asarray(ws["seed"], float)
    base_sym = symmetrize_seed(base_raw, BLOCKS, nn, nnn, args.n, group, rx_final=RX_FINAL)

    def _fid(theta):
        s = subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None
        return float(s) if s is not None else float(fid_fn(theta))

    init_raw, init_sym = _fid(base_raw), _fid(base_sym)
    dist = float(np.linalg.norm(base_raw - base_sym))
    print(f"[ab] N={args.n} h={args.h} manifold_dim={m_dim} group_order={group.order} params={npar}", flush=True)
    print(
        f"[ab] init fidelity: raw={init_raw:.4f}  symmetrized={init_sym:.4f}  "
        f"||raw-sym||={dist:.3f}  trials={args.trials} noise={args.noise}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed0)
    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"symmetrized_warmstart_ab_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "symmetrized_warmstart_ab_v1",
            "structure": "p3_half_nn_rx",
            "init_fid_raw": init_raw,
            "init_fid_sym": init_sym,
            "seed_distance": dist,
            "group_order": group.order,
            "trials": args.trials,
            "noise": args.noise,
        },
        params={"experiment": "symmetrized_warmstart_ab", "N": args.n, "h": args.h},
        description="A/B: symmetry-projected vs raw analytic warm-start (equal circuit)",
        rows_ref=rows,
    )

    # The analytic seed is ALREADY orbit-constant (same angle per bond type), so
    # projecting it is a no-op. The projection only matters for a seed that has
    # DRIFTED off the symmetric subspace — a transferred/noisy/donor seed. We model
    # that: each trial perturbs the base with noise (a realistic off-subspace seed);
    # the "raw" arm starts from the NOISY seed, the "symmetrized" arm starts from
    # that same noisy seed PROJECTED back onto the subspace. This measures whether
    # re-symmetrizing a drifted seed helps — the actual use case.
    for t in range(args.trials):
        noisy = base_raw + rng.normal(0.0, args.noise, size=npar)
        projected = symmetrize_seed(noisy, BLOCKS, nn, nnn, args.n, group, rx_final=RX_FINAL)
        for arm, x0 in (("raw", noisy), ("symmetrized", projected)):
            t0 = time.time()
            x_opt, n_it = _lbfgsb_with_iters(cost, grad, x0, args.maxiter, args.ftol)
            f = _fid(x_opt)
            rows.append(
                {
                    "trial": t,
                    "arm": arm,
                    "n_qubits": args.n,
                    "h": round(float(args.h), 2),
                    "final_fidelity": f,
                    "iters": n_it,
                    "init_fidelity": _fid(x0),
                    "seconds": round(time.time() - t0, 2),
                }
            )
            persister.persist(rows)
        r = rows[-2]
        s = rows[-1]
        print(
            f"  trial {t:2d}: raw(noisy) fid={r['final_fidelity']:.4f}({r['iters']}it) init={r['init_fidelity']:.3f}  "
            f"sym(proj) fid={s['final_fidelity']:.4f}({s['iters']}it) init={s['init_fidelity']:.3f}  "
            f"Δfid={s['final_fidelity'] - r['final_fidelity']:+.4f}",
            flush=True,
        )

    persister.persist(rows, status="final")
    backend.flush()

    # Paired summary.
    raw = [r for r in rows if r["arm"] == "raw"]
    sym = [r for r in rows if r["arm"] == "symmetrized"]
    rf = np.array([r["final_fidelity"] for r in raw])
    sf = np.array([r["final_fidelity"] for r in sym])
    ri = np.array([r["iters"] for r in raw])
    si = np.array([r["iters"] for r in sym])
    wins = int(np.sum(sf > rf + 1e-6))
    print("\n=== A/B SUMMARY (paired, single-restart) ===", flush=True)
    print(f"  raw:         fid {rf.mean():.4f} ± {rf.std():.4f}   iters {ri.mean():.0f} ± {ri.std():.0f}", flush=True)
    print(f"  symmetrized: fid {sf.mean():.4f} ± {sf.std():.4f}   iters {si.mean():.0f} ± {si.std():.0f}", flush=True)
    print(f"  Δfid mean = {(sf - rf).mean():+.4f}   symmetrized wins {wins}/{len(rf)} trials", flush=True)
    print(
        f"  iter change: {si.mean() - ri.mean():+.0f} ({100 * (si.mean() - ri.mean()) / max(ri.mean(), 1):+.0f}%)",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
