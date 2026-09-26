#!/usr/bin/env python
"""Haiqu VL via mps_loading at large N (beyond the dense-vector N≤20 limit).

For system sizes where the exact ground-state *vector* is no longer available
(N > statevector limit), the dense ``vector_loading`` path cannot be used. This
script takes the MPS route: DMRG at a bounded bond dimension (χ ≤ 64, Haiqu's
cap) → ``mps_loading`` → the loaded circuit → energy error |ΔE| via the MPS
backend.

It also reports the **input-MPS quality** (how much the χ-cap costs): exact
fidelity when N is small enough, otherwise the DMRG truncation error.

Pipeline per (topology, h):
  1. Build the project Hamiltonian (spec-based, frustrated J2 forwarded).
  2. dmrg_mps_capped(H, N, chi_max) → plr MPS + quality.
  3. run_vl_mps_job(mps, num_layers, fine_tuning) → VL circuit (Haiqu cloud).
  4. |ΔE| = |⟨ψ_VL|H|ψ_VL⟩ − E0| via the MPS backend (select_backend(N)).
  5. Persist row + save VL circuit as QPY + circuit_id + vl_meta.

Requires HAIQU_API_KEY.

Usage:
    HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_mps_large_n.py \
        --topology square --n 18 --j2 0.5 --h 0.5 1.0 \
        --num-layers 8 --fine-tuning 50 --chi-max 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402
    find_repo_root,
    haiqu_session,
    n_2q,
    run_vl_mps_job,
    save_circuit_qpy,
    save_json,
    study_dir,
)
from mps_extraction import capped_mps_vidal, dmrg_mps_capped, mps_input_quality  # noqa: E402

# Move-resilient repo-root resolution (never count parents[N]).
_REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)


def _build_hamiltonian(topology, n, h, model, j2):
    from qmbp_simulation import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2
    lat = make_lattice(topology, n, J=1.0, h=h)
    return spec.build_hamiltonian(lat, **ham_kwargs), lat


def _circuit_energy_mps(circuit, hamiltonian, n) -> float:
    """⟨ψ_VL|H|ψ_VL⟩ for a parameter-free circuit via the N-appropriate backend."""
    from qmbp_simulation.execution.backends import select_backend

    if circuit.num_parameters:
        raise ValueError(f"expected parameter-free circuit, got {circuit.num_parameters} params")
    backend = select_backend(n)
    return float(backend.evaluate(circuit, hamiltonian, np.empty(0)))


def _circuits_dir() -> Path:
    d = study_dir("vl_mps_n18") / "circuits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _config_tag(num_layers, fine_tuning, truncation_cutoff, mps_form) -> str:
    """Compact, filesystem-safe tag identifying a VL config (keeps runs distinct)."""
    return f"L{num_layers}_F{fine_tuning}_tc{truncation_cutoff:.0e}_{mps_form}"


def run(topology, n, h_values, model, j2, num_layers, fine_tuning, chi_max, timeout_s,
        *, truncation_cutoff=1e-6, mps_form="standard") -> list[dict]:
    rows: list[dict] = []
    session = None
    cfg_tag = _config_tag(num_layers, fine_tuning, truncation_cutoff, mps_form)
    out_name = f"vl_mps_{topology}_N{n}_{cfg_tag}.json"
    try:
        for h in h_values:
            row = {"topology": topology, "N": n, "model": model, "J2": j2, "h": h,
                   "num_layers": num_layers, "fine_tuning_iterations": fine_tuning,
                   "chi_max": chi_max, "truncation_cutoff": truncation_cutoff,
                   "mps_form": mps_form}
            try:
                H, _lat = _build_hamiltonian(topology, n, h, model, j2)
                capped = dmrg_mps_capped(H, n, chi_max=chi_max)
                quality = mps_input_quality(capped, topology=topology, h=h, model=model,
                                            j2=j2, hamiltonian=H)
                row["E0_dmrg"] = capped.energy
                row["mps_input"] = quality
                row["max_bond_realized"] = capped.max_bond_realized

                # Choose the MPS input form: standard site tensors or Vidal (Γ/Λ).
                if mps_form == "vidal":
                    gammas, lambdas = capped_mps_vidal(capped)
                    mps_input = (gammas, lambdas)
                else:
                    mps_input = capped.tensors

                if session is None:
                    session = haiqu_session(f"VL_mps_{topology}_N{n}")
                    session.__enter__()
                # shape="plr" describes site-tensor axis order (Γ and standard
                # tensors share it); Haiqu auto-detects Vidal vs standard by the
                # input structure (tuple vs list).
                fid, status, vl, cid, vl_meta = run_vl_mps_job(
                    mps_input, num_layers=num_layers, max_time=timeout_s,
                    fine_tuning_iterations=fine_tuning, truncation_cutoff=truncation_cutoff,
                    shape="plr",
                )
                row["vl_fidelity"] = fid
                row["vl_status"] = status
                row["vl_circuit_id"] = cid
                row["vl_meta"] = vl_meta
                if vl is not None:
                    row["vl_2q"] = n_2q(vl)
                    row["vl_depth"] = int(vl.depth())
                    row["vl_total_gates"] = int(len(vl.data))
                    vl_energy = _circuit_energy_mps(vl, H, n)
                    row["vl_energy"] = vl_energy
                    row["vl_abs_error"] = abs(vl_energy - capped.energy)
                    qpy_path = _circuits_dir() / f"{topology}_N{n}_h{h:.2f}_J2{j2:.2f}_{cfg_tag}.qpy"
                    save_circuit_qpy(vl, qpy_path)
                    row["vl_circuit_qpy"] = str(qpy_path.relative_to(_REPO_ROOT))
            except Exception as exc:  # noqa: BLE001 — never abort the batch
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            _print_row(row)
            save_json(
                {"rows": rows}, "vl_mps_n18", out_name,
                params={"topology": topology, "N": n, "model": model, "J2": j2,
                        "num_layers": num_layers, "fine_tuning_iterations": fine_tuning,
                        "chi_max": chi_max, "truncation_cutoff": truncation_cutoff,
                        "mps_form": mps_form, "loader": "mps_loading"},
                description="Haiqu VL via mps_loading at large N (bond-capped DMRG input)",
            )
    finally:
        if session is not None:
            session.__exit__(None, None, None)
    return rows


def _print_row(r: dict) -> None:
    if "error" in r:
        print(f"  {r['topology']} N={r['N']} h={r['h']}: ERROR {r['error'][:60]}", flush=True)
        return
    q = r.get("mps_input", {})
    mps_fid = q.get("mps_fidelity")
    if isinstance(mps_fid, (int, float)):
        mps_fid_s = f"{mps_fid:.4f}"
    else:
        mps_fid_s = f"n/a (trunc={q.get('trunc_err', float('nan')):.1e})"
    print(f"  {r['topology']} N={r['N']} h={r['h']} L{r['num_layers']}F{r['fine_tuning_iterations']} "
          f"χ={r.get('max_bond_realized')}/{r['chi_max']} | MPS_in_fid={mps_fid_s} | "
          f"VL_fid={r.get('vl_fidelity')} |dE|={r.get('vl_abs_error')} 2q={r.get('vl_2q')}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Haiqu VL via mps_loading at large N")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--fine-tuning", type=int, default=50)
    p.add_argument("--chi-max", type=int, default=64, help="DMRG bond cap (Haiqu accepts ≤64)")
    p.add_argument("--truncation-cutoff", type=float, default=1e-6,
                   help="Haiqu VL entanglement cutoff (smaller = more faithful)")
    p.add_argument("--mps-form", choices=["standard", "vidal"], default="standard",
                   help="MPS input form passed to Haiqu mps_loading")
    p.add_argument("--job-timeout", type=float, default=900.0,
                   help="Haiqu compute budget (max_time); client polls max_time+300s")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_tag = _config_tag(args.num_layers, args.fine_tuning, args.truncation_cutoff, args.mps_form)
    print(f"[vl_mps_large_n] {args.topology} N={args.n} h={args.h} J2={args.j2} "
          f"L{args.num_layers}F{args.fine_tuning} chi_max={args.chi_max} "
          f"tc={args.truncation_cutoff:.0e} form={args.mps_form}", flush=True)
    rows = run(args.topology, args.n, args.h, args.model, args.j2,
               args.num_layers, args.fine_tuning, args.chi_max, args.job_timeout,
               truncation_cutoff=args.truncation_cutoff, mps_form=args.mps_form)
    print(f"\n→ Resultados: {study_dir('vl_mps_n18') / f'vl_mps_{args.topology}_N{args.n}_{cfg_tag}.json'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
