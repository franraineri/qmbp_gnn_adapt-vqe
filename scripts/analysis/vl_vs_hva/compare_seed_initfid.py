#!/usr/bin/env python
"""Compare seed init-fidelity: per-layer PROFILE seed vs the current methods.

The per-layer θ_x profile is strongly N-invariant (CV 0.02-0.06), and the
current calibrated seed uses one UNIFORM x_scale. This measures whether a seed
built from the per-layer template starts at a HIGHER init-fidelity (NO
optimization) than the existing seeds, for the half_nn_rx family at N=10/14.

Seeds compared (all evaluated un-optimized, exact fidelity):
  - analytic  : variant_warmstart_theta (the family's second-order seed)
  - calibrated: calibrated_warmstart_theta (uniform x_scale) — only if length fits
  - transfer  : converged full θ from a SMALLER N (cross-N) onto this layout
  - profile   : the per-layer template seed built here (θ_x per layer, nn/nnn per
                layer means), with the TARGET N held out of the template fit
                (leave-one-N-out) so the comparison is not circular.

Pure evaluation — builds circuits, computes exact init-fidelity, runs NO optimizer.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/compare_seed_initfid.py \
        --variant p2_half_nn_rx --n 14 --h 0.5
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

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    calibrated_warmstart_theta,
    transfer_theta_for_blocks,
    variant_warmstart_theta,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant  # noqa: E402
from qmbp_simulation.framework.study_core import cx_and_params, ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SPECS_DIR = STUDY_ROOT / "ansatz_specs"


def _layout(blocks, n_nn, n_nnn, nq, rx_final, rz_final):
    size = {"nn": n_nn, "nnn": n_nnn, "x": nq, "z": nq}
    counts: dict[str, int] = {}
    out = []
    off = 0
    for b in blocks:
        counts[b] = counts.get(b, 0) + 1
        out.append((f"{b}#{counts[b]}", b, off, off + size[b]))
        off += size[b]
    if rx_final:
        out.append(("rx_final", "x", off, off + nq))
        off += nq
    if rz_final:
        out.append(("rz_final", "z", off, off + nq))
    return out


def _full_specs(variant, topology, h, exclude_n=None):
    out = []
    for f in sorted(SPECS_DIR.glob(f"{variant}_{topology}_N*_h{h:.2f}.spec.json")):
        if "prune" in f.name or "topk" in f.name:
            continue
        try:
            s = AnsatzSpec.load(str(f))
        except Exception:
            continue
        if exclude_n is not None and s.n_qubits == exclude_n:
            continue
        out.append(s)
    return out


def _build_profile_template(variant, topology, h, exclude_n):
    """Per-block-label template (signed mean for x, mean|θ| for zz) from the
    full specs, EXCLUDING ``exclude_n`` (leave-one-N-out → non-circular)."""
    specs = _full_specs(variant, topology, h, exclude_n=exclude_n)
    acc: dict[str, list] = {}
    for s in specs:
        th = np.asarray(s.theta, float)
        for lab, btype, a, b in _layout(s.blocks, len(s.nn_edges), len(s.nnn_edges),
                                        s.n_qubits, s.rx_final, s.rz_final):
            seg = th[a:b]
            val = float(np.mean(seg)) if btype == "x" else float(np.mean(np.abs(seg)))
            acc.setdefault(lab, []).append(val)
    return {lab: float(np.mean(v)) for lab, v in acc.items()}, len(specs)


def _profile_seed(template, blocks, n_nn, n_nnn, nq, rx_final, rz_final):
    """Assemble a seed vector from the per-label template for the TARGET layout."""
    parts = []
    for lab, btype, _a, _b in _layout(blocks, n_nn, n_nnn, nq, rx_final, rz_final):
        sz = {"nn": n_nn, "nnn": n_nnn, "x": nq, "z": nq}[btype]
        val = template.get(lab, 0.0)
        parts.append(np.full(sz, val, dtype=float))
    return np.clip(np.concatenate(parts), -np.pi, np.pi)


def main(argv=None) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    p = argparse.ArgumentParser(description="Compare seed init-fidelity")
    p.add_argument("--variant", default="p2_half_nn_rx")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    args = p.parse_args(argv)

    base = VARIANTS[args.variant]
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()
    qc, _ = build_variant(builder, args.n, lat, base)
    n_2q, npar = cx_and_params(qc)
    _cost, fid, _grad, _ = make_cost_fid(qc, H, psi, backend=backend)

    print(f"=== Seed init-fidelity — {args.variant} N={args.n} h={args.h} "
          f"(gap={gap:.4f}, npar={npar}) ===")
    seeds = {}

    # analytic
    an = variant_warmstart_theta(base.blocks, n_nn, n_nnn, args.n, args.h,
                                 J=1.0, J2=args.j2, rx_final=base.rx_final,
                                 rz_final=base.rz_final)
    if len(an) == npar:
        seeds["analytic"] = np.asarray(an, float)

    # calibrated (standard layout length — usually != half_nn_rx, so often skipped)
    cal = calibrated_warmstart_theta(n_nn, n_nnn, args.n, 1, args.h, J=1.0, J2=args.j2)
    if len(cal) == npar:
        seeds["calibrated"] = np.asarray(cal, float)

    # transfer — converged full from the nearest SMALLER N (cross-N)
    donors = _full_specs(args.variant, args.topology, args.h)
    donors = [s for s in donors if s.n_qubits < args.n and s.theta]
    if donors:
        from qmbp_simulation.models import make_lattice
        d = max(donors, key=lambda s: s.n_qubits)  # nearest smaller N
        ld = make_lattice(args.topology, d.n_qubits, J=1.0, h=args.h)
        tr = transfer_theta_for_blocks(
            np.asarray(d.theta, float), list(base.blocks),
            donor_nnn_edges=HamiltonianBuilder._generate_nnn_edges(ld),
            target_nnn_edges=HamiltonianBuilder._generate_nnn_edges(lat),
            n_nn=n_nn, n_qubits=args.n, rx_final=base.rx_final, rz_final=base.rz_final,
            donor_n_nn=len(ld.edges), donor_n_qubits=d.n_qubits,
            donor_nn_edges=list(ld.edges), target_nn_edges=list(lat.edges))
        if tr is not None and tr.size == npar:
            seeds[f"transfer<N{d.n_qubits}>"] = tr

    # profile (leave-this-N-out template)
    template, n_used = _build_profile_template(args.variant, args.topology, args.h, args.n)
    if template:
        prof = _profile_seed(template, base.blocks, n_nn, n_nnn, args.n,
                             base.rx_final, base.rz_final)
        if prof.size == npar:
            seeds[f"profile(LOO,{n_used}N)"] = prof

    print(f"{'seed':22s} {'init_fid':>9}")
    ranked = sorted(seeds.items(), key=lambda kv: -float(fid(kv[1])))
    for name, th in ranked:
        print(f"{name:22s} {float(fid(th)):>9.4f}")
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
