#!/usr/bin/env python
"""Quench dynamics from the prune0.3 ansatz state — full observable fan.

Prepares the frustrated-square ground state with the low-2q prune0.3 ansatz
(nn7/nnn4, 36 CZ, F≈0.96), quenches under H(h2), and tracks the full set of
dynamical observables: M_z(t), M_x(t), staggered M_z(t), half-chain entropy
S(t), Loschmidt echo L(t) + rate function r(t) (DQPT), and a representative
two-point correlator C(r,t). Sweeps several quenches h1→h2.

Key comparison: dynamics from the prune0.3 PREPARED state (what hardware can
make, F≈0.96) vs from the EXACT ground state (F=1.0) — quantifies how the 4%
preparation infidelity propagates into the measured observables.

Reuses: selection_from_variant_theta + build_variant (prune0.3 ansatz),
observables.py (all dynamical observables), exact / sparse time evolution.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_prune03_dynamics.py \
        --n 10 --h-prep 0.5 --h2-list 1.0 1.5 2.5 --dt 0.05 --n-trotter 60
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from qmbp_simulation import make_lattice
from qmbp_simulation.analysis.dynamics import (
    NIGHTHAWK_CZ_ERROR,
    classical_crossover_sparse,
    correlator_zz,
    tnoise_on_backend,
)
from qmbp_simulation.analysis.observables import (
    detect_dqpt_critical_times,
    detect_rate_function_peaks,
    half_chain_entropy,
    loschmidt_echo,
    magnetization_x,
    magnetization_z,
    rate_function,
    staggered_magnetization_z,
)
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.circuits.bond_mask import full_selection, selection_from_variant_theta
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant
from qmbp_simulation.circuits.trotter import build_frustrated_trotter_step
from qmbp_simulation.framework.study_core import make_cost_fid, optimize_bestof
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
from qmbp_simulation.utils.helpers import write_json_atomic

_REPO = Path(__file__).resolve().parents[3]
_ART = _REPO / "results/hva_vl_study/bond_ablation/variant_topk_p1_half_nn_rx_square_N10_h0.50.json"
_ART_N = 10  # the artifact's θ is optimized at N=10; valid only at this size
_BLOCKS = ["nn", "nnn", "x", "nn", "x"]
builder = HamiltonianBuilder()
hva = HVACircuitBuilder()


def _prepare_prune03(n, h_prep, j2, restarts, maxiter, spec_path=None, derive_prep=False):
    """Prepare GS(h_prep) with the prune0.3 ansatz; return state + exact GS + variant.

    When ``spec_path`` is given, the ansatz (edges + converged θ) is loaded from a
    portable AnsatzSpec instead of being reconstructed from the study artifact and
    re-optimized — the whole point of the portable format.
    """
    from qiskit.quantum_info import Statevector
    from scipy.sparse.linalg import eigsh

    lat = make_lattice("square", n, J=1.0, h=h_prep)
    nn_e, nnn_e = list(lat.edges), builder._generate_nnn_edges(lat)
    Hprep = builder.build_frustrated_tfim(lat, J2=j2)
    w, v = eigsh(Hprep.to_matrix(sparse=True), k=2, which="SA")
    psi_exact = v[:, np.argsort(w)[0]].astype(complex)

    if spec_path is not None:
        from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec

        spec = AnsatzSpec.load(spec_path)
        variant = spec.to_variant()
        qc, theta = spec.build(builder=hva, lattice=lat)
        if theta is None:
            raise ValueError(f"spec {spec_path} has no θ — cannot reuse as a prep")
        f = (abs(np.vdot(psi_exact, np.asarray(
            Statevector(qc.assign_parameters(theta)).data))) ** 2)
        sel = spec.to_bond_selection()
    elif derive_prep:
        # Derive the prune0.3 selection from scratch AT THIS N: optimize the full
        # [nn,nnn,x,nn,x]+rx_final variant here, then prune by |θ| (tol=0.3) with
        # the same criterion as the N=10 artifact. This generalizes the experiment
        # to any N (the selection emerges from this N's θ, not the N=10 one).
        full_sel = full_selection(nn_e, nnn_e)
        full_variant = make_masked_variant(
            "p1_half_nn_rx_full", "full reference for prune derivation",
            _BLOCKS, full_sel, rx_final=True, tags=("reference",))
        qc_full, _ = build_variant(hva, n, lat, full_variant)
        cost, fid, grad, _ = make_cost_fid(qc_full, Hprep, psi_exact)
        _, _, runs_full = optimize_bestof(cost, fid, grad, qc_full.num_parameters,
                                          restarts=restarts, maxiter=maxiter, seed0=0)
        theta_full = min(runs_full, key=lambda r: r["energy"])["theta_final"]
        sel = selection_from_variant_theta(theta_full, _BLOCKS, nn_e, nnn_e, n,
                                           rx_final=True, method="prune", tol=0.3)
        variant = make_masked_variant(
            "prune0.3", f"derived prune0.3 @ N={n}", _BLOCKS, sel,
            rx_final=True, tags=("T1", "derived"))
        qc, _ = build_variant(hva, n, lat, variant)
        cost, fid, grad, _ = make_cost_fid(qc, Hprep, psi_exact)
        f, _, runs = optimize_bestof(cost, fid, grad, qc.num_parameters,
                                     restarts=restarts, maxiter=maxiter, seed0=0)
        theta = min(runs, key=lambda r: r["energy"])["theta_final"]
    else:
        # The study artifact's θ is specific to the N it was optimized at (N=10):
        # its length matches that lattice's bond counts. Rebuilding at another N
        # would feed an N=10 θ into an N≠10 block layout (a cryptic length error
        # deep in bond_weights_for_blocks). Fail early with the actionable fix.
        if n != _ART_N:
            raise ValueError(
                f"--n {n} but the prune0.3 study artifact is N={_ART_N}-specific "
                f"({_ART.name}). Reconstructing the ansatz from the artifact only "
                f"works at N={_ART_N}. To run at N={n}, pass --load-spec with an "
                f"AnsatzSpec generated for N={n} (run this script once at that N to "
                f"emit one under results/hva_vl_study/ansatz_specs/)."
            )
        with open(_ART) as _fh:
            d = json.load(_fh)
        theta_full = next(np.asarray(r["best_theta_final"], float)
                          for r in d["rows"] if r["variant"] == "p1_half_nn_rx_full")
        sel = selection_from_variant_theta(theta_full, _BLOCKS, nn_e, nnn_e, n,
                                           rx_final=True, method="prune", tol=0.3)
        variant = make_masked_variant("prune0.3", "nn7/nnn4", _BLOCKS, sel,
                                      rx_final=True, tags=("T1",))
        qc, _ = build_variant(hva, n, lat, variant)
        cost, fid, grad, _ = make_cost_fid(qc, Hprep, psi_exact)
        f, _, runs = optimize_bestof(cost, fid, grad, qc.num_parameters,
                                     restarts=restarts, maxiter=maxiter, seed0=0)
        theta = min(runs, key=lambda r: r["energy"])["theta_final"]

    prep_bound = qc.assign_parameters(theta)
    psi_prep = np.asarray(Statevector(prep_bound).data)
    n_2q = sum(1 for i in qc.data if i.operation.num_qubits == 2)
    return psi_prep, psi_exact, float(f), n_2q, sel, prep_bound, lat, variant, np.asarray(theta, float)


def _evolve_and_measure(psi0, H_sparse, dt, n_steps, n):
    """Exact evolution via sparse ``expm_multiply``; per-step observables.

    Evolves with the sparse matrix action (no dense 2ᴺ×2ᴺ propagator), so this
    scales to N≈18-22 where a dense ``expm`` would need >1 TB. Same step function
    as dynamics.evolve_states(evolver=...).
    """
    from scipy.sparse.linalg import expm_multiply

    generator = -1j * H_sparse * dt
    psi = np.asarray(psi0, dtype=complex).copy()
    obs = {k: [] for k in ("t", "mz", "mx", "stag_mz", "entropy", "loschmidt",
                           "rate", "corr_0_mid")}
    mid = n // 2
    for step in range(n_steps + 1):
        obs["t"].append(step * dt)
        obs["mz"].append(magnetization_z(psi, n))
        obs["mx"].append(magnetization_x(psi, n))
        obs["stag_mz"].append(staggered_magnetization_z(psi, n))
        obs["entropy"].append(half_chain_entropy(psi, n))
        L = loschmidt_echo(psi0, psi)
        obs["loschmidt"].append(L)
        obs["rate"].append(rate_function(L, n))
        obs["corr_0_mid"].append(correlator_zz(psi, 0, mid, n))
        if step < n_steps:
            psi = expm_multiply(generator, psi)
            psi /= np.linalg.norm(psi)
    return obs


def run(args) -> int:
    (psi_prep, psi_exact, f_prep, n_2q, sel, prep_bound, lat,
     variant, theta) = _prepare_prune03(
        args.n, args.h_prep, args.j2, args.restarts, args.maxiter,
        spec_path=args.load_spec, derive_prep=args.derive_prep)
    nn_e, nnn_e = list(lat.edges), builder._generate_nnn_edges(lat)
    src = (f"loaded spec {args.load_spec}" if args.load_spec
           else f"derived @ N={args.n}" if args.derive_prep
           else "reconstructed from artifact")
    print(f"[prune03_dyn] N={args.n} h_prep={args.h_prep} J2={args.j2} "
          f"F_prep={f_prep:.4f} 2q={n_2q} nn={len(sel.nn_edges)} nnn={len(sel.nnn_edges)} "
          f"[{src}]", flush=True)

    # Persist the portable AnsatzSpec of this prep (edges + θ + metrics) so any
    # other runner can reuse this exact ansatz without re-deriving it.
    spec_path_out = None
    if not args.load_spec:
        from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec

        spec = AnsatzSpec.from_variant(
            variant, topology="square", n_qubits=args.n, h=args.h_prep, j2=args.j2,
            theta=theta, fidelity=f_prep, n_2q=n_2q, n_params=len(theta),
            seed_kind="analytic+descent")
        spec_dir = (_REPO / "results/hva_vl_study/ansatz_specs")
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path_out = str((spec_dir / f"prune0.3_square_N{args.n}_h{args.h_prep:.2f}.spec.json")
                            .relative_to(_REPO))
        spec.save(_REPO / spec_path_out)
        print(f"  saved ansatz spec → {spec_path_out}", flush=True)

    # Fix the output path up front so per-quench checkpoints and the final
    # result share one file — a crash at large N keeps the completed quenches.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = (_REPO / "results/experiments/exp_frustrated/prune03_dynamics"
              / "tfim_frustrated" / f"square_N{args.n}_hprep{args.h_prep:.2f}")
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"prune03_dynamics_{timestamp}.json"

    base_specs = {
        "model": "tfim_frustrated", "ansatz": "prune0.3_nn7_nnn4",
        "topology": "square", "n_qubits": args.n, "j2": args.j2,
        "h_prep": args.h_prep, "F_prep": f_prep, "prep_2q": n_2q,
        "dt": args.dt, "n_trotter": args.n_trotter, "h2_list": args.h2_list,
        "nn_edges": [list(e) for e in sorted(sel.nn_edges)],
        "nnn_edges": [list(e) for e in sorted(sel.nnn_edges)],
    }

    def _write(doc):
        # Atomic (tmp+rename) + NaN/Inf-safe: observable curves (rate=−ln L/N)
        # can be non-finite, and checkpoints must never corrupt on a crash.
        write_json_atomic(out, doc)

    quenches = []
    for h2 in args.h2_list:
        lat2 = make_lattice("square", args.n, J=1.0, h=h2)
        H_sparse = builder.build_frustrated_tfim(lat2, J2=args.j2).to_matrix(sparse=True)
        obs_prep = _evolve_and_measure(psi_prep, H_sparse, args.dt, args.n_trotter, args.n)
        obs_exact = _evolve_and_measure(psi_exact, H_sparse, args.dt, args.n_trotter, args.n)

        dqpt_prep = detect_dqpt_critical_times(obs_prep["t"], obs_prep["loschmidt"])
        dqpt_exact = detect_dqpt_critical_times(obs_exact["t"], obs_exact["loschmidt"])
        peaks_prep = detect_rate_function_peaks(obs_prep["t"], obs_prep["rate"])

        # How much does the 4% prep infidelity distort each observable?
        max_dev = {k: float(np.max(np.abs(np.array(obs_prep[k]) - np.array(obs_exact[k]))))
                   for k in ("mz", "mx", "stag_mz", "entropy", "corr_0_mid")}
        s_max = float(np.max(obs_prep["entropy"]))

        quenches.append({
            "h2": h2, "s_max_prep": s_max,
            "dqpt_times_prep": dqpt_prep, "dqpt_times_exact": dqpt_exact,
            "n_rate_peaks_prep": len(peaks_prep),
            "max_obs_deviation_prep_vs_exact": max_dev,
            "observables_prep": obs_prep,
            "observables_exact": obs_exact,
        })
        print(f"  h2={h2}: S_max={s_max:.3f} DQPT(prep)={len(dqpt_prep)} "
              f"DQPT(exact)={len(dqpt_exact)} maxdev_mz={max_dev['mz']:.3f} "
              f"maxdev_S={max_dev['entropy']:.3f}", flush=True)

        # Checkpoint: persist completed quenches immediately (crash-safe).
        _write({
            "schema": "prune03_dynamics_v1", "status": "partial",
            "quenches_done": len(quenches), "quenches_total": len(args.h2_list),
            "ansatz_spec_path": spec_path_out, "ansatz_spec_loaded": args.load_spec,
            "specs": base_specs, "quenches": quenches, "timestamp": timestamp,
        })

    # Classical crossover + hardware t_noise for the max-entanglement quench.
    best_q = max(quenches, key=lambda q: q["s_max_prep"])
    h2_star = best_q["h2"]
    H_sparse_star = builder.build_frustrated_tfim(
        make_lattice("square", args.n, J=1.0, h=h2_star), J2=args.j2).to_matrix(sparse=True)
    chi_values = args.chi_values
    crossover = classical_crossover_sparse(psi_exact, H_sparse_star, args.dt,
                                           args.n_trotter, args.n, chi_values)
    step_qc = build_frustrated_trotter_step(args.n, nn_e, nnn_e, h2_star, args.dt,
                                            j2=args.j2, order=2)
    eps_2q = args.eps_2q if args.noise_model == "analytic" else None
    tnoise = {}
    for bname in ("nighthawk", "heron"):
        res = tnoise_on_backend(prep_bound, step_qc, min(args.n_trotter, 25), bname,
                                eps_2q=eps_2q)
        if res:
            tnoise[bname] = res
    print(f"\n  max-entanglement quench h2={h2_star} (S_max={best_q['s_max_prep']:.3f})")
    print(f"  classical crossover t*(χ): {crossover}")
    for bn, r in tnoise.items():
        print(f"  t_noise ({bn}): step {r['t_noise_step']}  (prep_2q_transp={r['cz_cumulative'][0]})")
    window = {
        "h2_star": h2_star,
        "t_star_chi16": crossover.get(16),
        "t_noise_nighthawk": tnoise.get("nighthawk", {}).get("t_noise_step"),
        "t_noise_heron": tnoise.get("heron", {}).get("t_noise_step"),
    }

    _write({
        "schema": "prune03_dynamics_v1", "status": "complete",
        "ansatz_spec_path": spec_path_out,
        "ansatz_spec_loaded": args.load_spec,
        "classical_crossover": crossover,
        "t_noise": tnoise,
        "window": window,
        "specs": base_specs,
        "quenches": quenches,
        "timestamp": timestamp,
    })
    print(f"\n→ {out.relative_to(_REPO)}", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--h-prep", type=float, default=0.5)
    p.add_argument("--h2-list", type=float, nargs="+", default=[1.0, 1.5, 2.5])
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--n-trotter", type=int, default=60)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=600)
    p.add_argument("--load-spec", type=str, default=None,
                   help="Path to an AnsatzSpec JSON — reuse that exact ansatz "
                   "(edges + converged θ) instead of reconstructing from the artifact.")
    p.add_argument("--derive-prep", action="store_true",
                   help="Derive the prune0.3 selection from scratch at the given N "
                   "(optimize full variant + prune by |θ|, tol=0.3) instead of "
                   "reconstructing from the N=10 artifact. Use for N != 10.")
    p.add_argument("--noise-model", choices=["fake", "analytic"], default="fake",
                   help="t_noise model: 'fake' = Qiskit fake-backend calibration "
                   "(default); 'analytic' = exp(-N2q·eps_2q) on the transpiled 2q "
                   "count (transparent, Nighthawk median CZ error).")
    p.add_argument("--eps-2q", type=float, default=NIGHTHAWK_CZ_ERROR,
                   help=f"Per-2q-gate error for --noise-model analytic "
                   f"(default {NIGHTHAWK_CZ_ERROR}, Nighthawk r2 median CZ).")
    p.add_argument("--chi-values", type=int, nargs="+", default=[4, 8, 16, 32],
                   help="MPS bond dimensions for the classical crossover t*(χ) "
                   "(default: 4 8 16 32). Use higher χ to probe larger N.")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
