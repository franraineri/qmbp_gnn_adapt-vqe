#!/usr/bin/env python
"""t_noise (prep + evolution) for the Kitaev chain on native heavy-hex.

Kitaev H = -J·Σ(XX+YY) - μ·Σ Z is frustrated by BOND direction (not geometry),
so it needs only native heavy-hex edges — no non-native nnn, hence no SWAP
routing. This measures whether that low-2q native circuit pushes t_noise (the
Trotter step where predicted hardware fidelity drops below 1/e) high enough to
exceed a classical crossover.

t_noise here is a PESSIMISTIC floor: the exp(−Σ nᵢ·εᵢ) model includes no circuit
compression and no error suppression (ZNE/QESEM), both of which raise it.

Prep = HVA Kitaev ansatz (create_kitaev) optimized to the exact ground state of
H(μ₁). Evolution = 2nd-order Trotter of H(μ₂). Both transpiled to FakeTorino.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_kitaev_heavyhex_noise.py \
        --n 18 --mu1 0.5 --mu2 3.0 --p 2 --dt 0.1 --n-trotter 40 \
        --restarts 6 --maxiter 1500
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from qmbp_simulation import make_lattice
from qmbp_simulation.analysis.dynamics import (
    NIGHTHAWK_CZ_ERROR,
    NOISE_FLOOR,
    classical_crossover_sparse,
    count_2q_gates,
    tnoise_curve,
)
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.circuits.trotter import build_kitaev_trotter_step
from qmbp_simulation.framework.study_core import make_cost_fid, optimize_bestof
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
from qmbp_simulation.utils.helpers import write_json_atomic

_REPO_ROOT = Path(__file__).resolve().parents[3]
builder = HamiltonianBuilder()
hva = HVACircuitBuilder()


_DENSE_LIMIT = 12  # dense eigh above this needs >100s of GB; use sparse eigsh


def _ground_state(topology, n, mu, delta):
    lat = make_lattice(topology, n, J=1.0, h=mu)
    H = builder.build_kitaev(lat, delta=delta)
    if n <= _DENSE_LIMIT:
        w, v = np.linalg.eigh(np.asarray(H.to_matrix()))
        gs, e0, gap = v[:, 0], float(w[0]), float(w[1] - w[0])
    else:
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
        idx = np.argsort(w)
        gs = v[:, idx[0]]
        e0, gap = float(w[idx[0]]), float(w[idx[1]] - w[idx[0]])
    gs = gs / np.linalg.norm(gs)
    return lat, H, gs, e0, gap


def _get_backend():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        return FakeTorino(), "FakeTorino"
    except Exception:  # noqa: BLE001
        return None, "typical_fallback"


def run(args) -> int:
    floor = NOISE_FLOOR
    backend, bname = _get_backend()
    lat, H, psi, e0, gap = _ground_state(args.topology, args.n, args.mu1, args.delta)
    edges = list(lat.edges)

    print(f"[kitaev_noise] {args.topology} N={args.n} mu1={args.mu1}->mu2={args.mu2} "
          f"p={args.p} delta={args.delta} e0={e0:.4f} gap={gap:.5f} edges={len(edges)}", flush=True)

    # Prepare ground state of H(mu1) with the bond-resolved Kitaev HVA ansatz.
    # Initial state |0⟩^N (the field −μZ favors it); bond-resolved per-edge θ
    # recovers high fidelity on the irregular heavy-hex graph.
    from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec

    spec_path_out = None
    if args.load_spec:
        spec = AnsatzSpec.load(args.load_spec)
        qc, theta = spec.build(builder=hva, lattice=lat)
        if theta is None:
            raise ValueError(f"spec {args.load_spec} has no θ")
        from qiskit.quantum_info import Statevector
        f_prep = float(abs(np.vdot(
            psi, np.asarray(Statevector(qc.assign_parameters(theta)).data))) ** 2)
        print(f"  [loaded spec {args.load_spec}]", flush=True)
    else:
        qc, _ = hva.create_kitaev_bond_resolved(args.n, args.p, lat, initial_state="zero")
        cost, fid_fn, grad, _ = make_cost_fid(qc, H, psi)
        f_prep, e_prep, runs = optimize_bestof(
            cost, fid_fn, grad, qc.num_parameters,
            restarts=args.restarts, maxiter=args.maxiter, seed0=0)
        theta = min(runs, key=lambda r: r["energy"])["theta_final"]
    prep = qc.assign_parameters(theta)
    prep_2q_logical = count_2q_gates(qc)

    # Persist the portable Kitaev AnsatzSpec (edges + θ + metrics).
    if not args.load_spec:
        spec = AnsatzSpec.from_kitaev(
            name=f"kitaev_bondres_p{args.p}", topology=args.topology,
            n_qubits=args.n, h=args.mu1, p_layers=args.p, lattice=lat,
            theta=np.asarray(theta, float), initial_state="zero", delta=args.delta,
            fidelity=f_prep, n_2q=prep_2q_logical, n_params=len(theta))
        spec_dir = _REPO_ROOT / "results/hva_vl_study/ansatz_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path_out = str((spec_dir /
            f"kitaev_bondres_{args.topology}_N{args.n}_p{args.p}_mu{args.mu1:.2f}.spec.json")
            .relative_to(_REPO_ROOT))
        spec.save(_REPO_ROOT / spec_path_out)
        print(f"  saved ansatz spec → {spec_path_out}", flush=True)

    step = build_kitaev_trotter_step(args.n, edges, args.mu2, args.dt,
                                     delta=args.delta, order=2)
    step_2q_logical = count_2q_gates(step)

    # Fix the output path up front so the t_noise checkpoint and the final
    # result share one file — the sparse crossover at N=18 is the costly part,
    # so a crash during it keeps the already-computed t_noise curve.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = (_REPO_ROOT / "results" / "experiments" / "exp_frustrated"
              / "kitaev_heavyhex_noise" / "kitaev"
              / f"{args.topology}_N{args.n}_p{args.p}_mu{args.mu1:.2f}to{args.mu2:.2f}")
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"kitaev_noise_{timestamp}.json"

    eps_2q = args.eps_2q if args.noise_model == "analytic" else None
    curve = tnoise_curve(prep, step, args.n_trotter, backend, floor=floor, eps_2q=eps_2q)
    fcurve = curve["fidelity_curve"]
    czcum = curve["cz_cumulative"]
    t_noise = curve["t_noise_step"]
    early = curve["early_stopped"]
    source = curve["source"]

    print(f"  F_prep={f_prep:.4f} prep_2q(log)={prep_2q_logical} prep_2q(transp)={czcum[0]} "
          f"step_2q(log)={step_2q_logical}", flush=True)
    print(f"  t_noise = step {t_noise} (t={None if t_noise is None else t_noise*args.dt})", flush=True)

    def _build_doc(status, crossover, window):
        return {
            "schema": "kitaev_heavyhex_noise_v1",
            "experiment": "kitaev_prep_evolution_noise",
            "status": status,
            "ansatz_spec_path": spec_path_out,
            "ansatz_spec_loaded": args.load_spec,
            "specs": {
                "model": "kitaev", "ansatz": "kitaev_bond_resolved", "initial_state": "zero",
                "topology": args.topology, "delta": args.delta,
                "n_qubits": args.n, "p_layers": args.p, "mu1": args.mu1, "mu2": args.mu2,
                "dt": args.dt, "n_trotter": args.n_trotter, "backend": bname,
                "fidelity_floor": floor, "e0": e0, "gap": gap, "n_edges": len(edges),
            },
            "result": {
                "prep_fidelity": float(f_prep),
                "prep_2q_logical": prep_2q_logical,
                "prep_2q_transpiled": int(czcum[0]) if czcum else None,
                "step_2q_logical": step_2q_logical,
                "fidelity_curve": fcurve,
                "cz_cumulative": czcum,
                "t_noise_step": t_noise,
                "t_noise_time": (t_noise * args.dt if t_noise is not None else None),
                "early_stopped": early,
                "error_rates_source": source,
                "classical_crossover": crossover,
                "window": window,
                "note": "pessimistic floor — no circuit compression, no ZNE/QESEM",
            },
            "timestamp": timestamp,
        }

    def _write(doc):
        # Atomic (tmp+rename) + NaN/Inf-safe: the fidelity curve and crossover
        # can hold non-finite entries, and the pre-crossover checkpoint must
        # never corrupt if the sparse evolution crashes at N=18.
        write_json_atomic(out, doc)

    # Checkpoint: persist the t_noise result before the costly sparse crossover.
    _write(_build_doc("partial_tnoise", crossover=None, window=None))

    # Classical crossover: where a χ-MPS can no longer track the exact evolution
    # under H(mu2). Only for N within exact-evolution reach.
    chi_values = [4, 8, 16, 32]
    crossover = None
    window = None
    if args.n <= 20:
        lat2 = make_lattice(args.topology, args.n, J=1.0, h=args.mu2)
        H_mu2 = builder.build_kitaev(lat2, delta=args.delta)
        crossover = classical_crossover_sparse(
            psi, H_mu2.to_matrix(sparse=True), args.dt, args.n_trotter,
            args.n, chi_values)
        t_star = crossover.get(16)  # χ=16 reference
        if t_star is not None and t_noise is not None:
            window = {"t_star_chi16": t_star, "t_noise": t_noise,
                      "has_advantage_window": t_noise > t_star}
        print(f"  classical crossover t*(χ): {crossover}", flush=True)
        if window:
            verdict = "WINDOW ✅" if window["has_advantage_window"] else "no window ❌"
            print(f"  t_noise={t_noise} vs t*(χ=16)={t_star}: {verdict}", flush=True)

    _write(_build_doc("complete", crossover, window))
    print(f"\n→ {out.relative_to(_REPO_ROOT)}", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", default="heavy_hex")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--mu1", type=float, default=0.5, help="prep chemical potential (topological phase |mu|<2J)")
    p.add_argument("--mu2", type=float, default=3.0, help="quench chemical potential (trivial phase |mu|>2J)")
    p.add_argument("--delta", type=float, default=1.0, help="pairing asymmetry (1.0 = pure XX)")
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n-trotter", type=int, default=40)
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--load-spec", type=str, default=None,
                   help="Path to a Kitaev AnsatzSpec JSON — reuse that exact "
                   "ansatz (edges + converged θ) instead of re-optimizing.")
    p.add_argument("--noise-model", choices=["fake", "analytic"], default="fake",
                   help="t_noise model: 'fake' = Qiskit fake-backend calibration "
                   "(default); 'analytic' = exp(-N2q·eps_2q) on the transpiled 2q "
                   "count (transparent, Nighthawk/Heron median CZ error).")
    p.add_argument("--eps-2q", type=float, default=NIGHTHAWK_CZ_ERROR,
                   help=f"Per-2q-gate error for --noise-model analytic "
                   f"(default {NIGHTHAWK_CZ_ERROR}, Nighthawk r2 median CZ).")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
