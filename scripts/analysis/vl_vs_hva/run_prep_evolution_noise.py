#!/usr/bin/env python
"""Resource/noise budget of PREPARATION + EVOLUTION for a quench experiment.

Measures t_noise (the Trotter step where the predicted circuit fidelity on real
heavy-hex hardware drops below 1/e) for the FULL circuit = state preparation +
k Trotter steps, so the choice of preparation ansatz actually enters the budget.

Motivation: the quench runner's Section 4 measures t_noise with a trivial |+⟩
preparation, so an efficient low-2q ground-state ansatz does not change it. Here
we quantify what a cheap, high-fidelity ansatz (e.g. p2_topk_f0.3: top-4 nnn,
~68 2q, F≈0.967) buys for the full experiment, versus the full p2 ansatz.

The EVOLUTION Hamiltonian is always the full frustrated TFIM (all nn+nnn): the
masked ansatz is only a cheaper way to PREPARE the ground state of the full H,
not a different physics. Prep is appended before the Trotter layers; the whole
thing is transpiled to FakeTorino (Heron, heavy-hex) per step and scored with
compute_error_budget (F ≈ exp(−Σ nᵢ·εᵢ), real calibration rates).

Persists a JSON per run under:
  results/experiments/exp_frustrated/prep_evolution_noise/{model}/
    {topology}_N{n}_p{p}_h{h1}to{h2}/prep_evolution_noise_{ts}.json

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_prep_evolution_noise.py \
        --n 10 --h1 0.5 --h2 2.5 --j2 0.5 --p 2 --nnn-keep 4 \
        --dt 0.1 --n-trotter 40 --restarts 6 --maxiter 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from qiskit.circuit import QuantumCircuit

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qmbp_simulation.analysis.dynamics import NOISE_FLOOR, tnoise_curve  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_from_theta,
    full_selection,
    top_k_by_weight,
)
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant  # noqa: E402
from qmbp_simulation.circuits.trotter import build_frustrated_trotter_step  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.utils.helpers import write_json_atomic  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _optimize(qc, H, psi, *, restarts, maxiter, seed0=0, backend=None):
    """Thin alias over the shared converge_circuit (make_cost_fid + best-of)."""
    return converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                            seed0=seed0, backend=backend)


def _get_backend():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        return FakeTorino(), "FakeTorino"
    except Exception:  # noqa: BLE001
        return None, "typical_fallback"


def run(args) -> int:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h1, args.j2, args.p)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()
    backend, backend_name = _get_backend()  # hardware fake backend for t_noise
    floor = NOISE_FLOOR

    from qmbp_simulation.execution import NoiselessBackend
    vqe_backend = NoiselessBackend()  # shared across prep variants (cached dense H)

    print(f"[prep_evo_noise] N={args.n} h1={args.h1}->h2={args.h2} J2={args.j2} "
          f"p={args.p} e0={e0:.4f} gap={gap:.5f} n_nn={n_nn} n_nnn={n_nnn}", flush=True)

    # Reference θ (full p) → per-bond weights for the top-k selection.
    ref_variant = make_masked_variant(
        f"p{args.p}_full", f"p={args.p} full frustrated reference",
        ["nn", "nnn", "x"] * args.p,
        full_selection(nn_edges, nnn_edges), tags=("reference",))
    t0 = time.time()
    ref_fid, ref_e, ref_runs = _optimize(
        build_variant(builder, args.n, lat, ref_variant)[0], H, psi,
        restarts=args.restarts, maxiter=args.maxiter, backend=vqe_backend)
    ref_theta = min(ref_runs, key=lambda r: r["energy"])["theta_final"]
    w_nn, w_nnn = bond_weights_from_theta(ref_theta, n_nn, n_nnn, args.n, args.p)

    variants_out = []
    trotter_step = build_frustrated_trotter_step(
        args.n, nn_edges, nnn_edges, args.h2, args.dt, j2=args.j2, order=2)

    # Candidate preparations: trivial |+⟩, full p, and masked top-k.
    candidates = [("plus_trivial", None), (f"p{args.p}_full", ref_variant)]
    k = args.nnn_keep
    sel_topk = top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn, k_nn=None, k_nnn=k)
    topk_variant = make_masked_variant(
        f"p{args.p}_topk_nnn{k}", f"p={args.p} top-{k} nnn by |θ|",
        ["nn", "nnn", "x"] * args.p, sel_topk, tags=("T2", "subset"))
    candidates.append((f"p{args.p}_topk_nnn{k}", topk_variant))

    for name, variant in candidates:
        if variant is None:
            prep = QuantumCircuit(args.n)
            prep.h(range(args.n))
            prep_fid, prep_2q_logical = None, 0
        else:
            qc, _ = build_variant(builder, args.n, lat, variant)
            n2q_logical, _ = cx_and_params(qc)
            vf, ve, vruns = _optimize(qc, H, psi, restarts=args.restarts,
                                      maxiter=args.maxiter, backend=vqe_backend)
            theta = min(vruns, key=lambda r: r["energy"])["theta_final"]
            prep = qc.assign_parameters(theta)
            prep_fid, prep_2q_logical = float(vf), int(n2q_logical)

        curve = tnoise_curve(prep, trotter_step, args.n_trotter, backend, floor=floor)
        variants_out.append({
            "preparation": name,
            "prep_fidelity": prep_fid,
            "prep_2q_logical": prep_2q_logical,
            "prep_2q_transpiled": curve["prep_2q"],
            "t_noise_step": curve["t_noise_step"],
            "t_noise_time": (curve["t_noise_step"] * args.dt
                             if curve["t_noise_step"] is not None else None),
            "fidelity_curve": curve["fidelity_curve"],
            "cz_cumulative": curve["cz_cumulative"],
            "early_stopped": curve["early_stopped"],
        })
        tn = curve["t_noise_step"]
        print(f"  {name:18s} F_prep={prep_fid if prep_fid is None else round(prep_fid,4)} "
              f"prep_2q(log/transp)={prep_2q_logical}/{curve['prep_2q']} "
              f"t_noise=step {tn}", flush=True)

    doc = {
        "schema": "prep_evolution_noise_v1",
        "experiment": "prep_evolution_noise",
        "status": "complete",
        "specs": {
            "model": "tfim_frustrated", "topology": args.topology, "j2": args.j2,
            "n_qubits": args.n, "p_layers": args.p, "h1": args.h1, "h2": args.h2,
            "dt": args.dt, "n_trotter": args.n_trotter, "nnn_keep": args.nnn_keep,
            "backend": backend_name, "fidelity_floor": floor,
            "e0": e0, "gap": gap, "n_nn": n_nn, "n_nnn": n_nnn,
        },
        "reference": {"full_fidelity": float(ref_fid), "seconds": round(time.time() - t0, 1)},
        "variants": variants_out,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    subdir = (_REPO_ROOT / "results" / "experiments" / "exp_frustrated"
              / "prep_evolution_noise" / "tfim_frustrated"
              / f"{args.topology}_N{args.n}_p{args.p}_h{args.h1:.2f}to{args.h2:.2f}")
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"prep_evolution_noise_{doc['timestamp']}.json"
    # Atomic + NaN/Inf-safe write (fidelity curves can contain non-finite values).
    write_json_atomic(out, doc)
    print(f"\n→ {out.relative_to(_REPO_ROOT)}", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--h1", type=float, default=0.5)
    p.add_argument("--h2", type=float, default=2.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--nnn-keep", type=int, default=4, help="top-k nnn bonds kept")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--n-trotter", type=int, default=40)
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=2000)
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
