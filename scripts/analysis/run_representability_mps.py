#!/usr/bin/env python
"""MPS/DMRG bond-dimension study for the TFIM ground state (chain_1d).

Sweeps the MPS bond dimension chi and records, per (N, h, chi): energy, |ΔE| vs
reference, truncation error, entanglement entropy, chi actually reached, time and
estimated memory. Answers "how viable is a classical MPS representation as N grows
and near the critical point?".

- Reference energy: exact diag via ClassicalSolver for N <= 16; else the largest
  chi in the sweep (self-reference for convergence).
- DMRG via TeNPy TFIChain with max_trunc_err=None so small chi truncates (does
  NOT abort) — that regime is exactly what we study.

Usage:
    python scripts/analysis/run_representability_mps.py --smoke   # N=4,10 quick
    python scripts/analysis/run_representability_mps.py           # full N=4,10,50,100
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from representability_io import Row, mem_estimate_mps, save_run  # noqa: E402

TOPO = "chain_1d"
J = 1.0


def dmrg_chi(N: int, h: float, chi: int) -> dict:
    """Run finite DMRG at fixed chi_max; return energy, S_vN, trunc err, chi, time."""
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.tf_ising import TFIChain
    from tenpy.networks.mps import MPS

    # TeNPy TFIChain: H = -J sum X_i X_{i+1} - g sum Z_i. Same spectrum &
    # entanglement as our H = -J sum Z_i Z_{i+1} - h sum X_i (X<->Z relabel),
    # so ground energy, S_vN and chi-scaling are identical for this chi study.
    model = TFIChain({"L": N, "J": J, "g": h, "bc_MPS": "finite", "conserve": None})
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * N, bc="finite")
    params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-14},
        "max_trunc_err": None,  # degrade consistency check to a warning
        "max_sweeps": 60, "min_sweeps": 4, "max_E_err": 1e-12, "mixer": True,
    }
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, params)
    t0 = time.time()
    e0, psi_out = eng.run()
    dt = time.time() - t0

    S = psi_out.entanglement_entropy()
    s_max = float(np.max(S)) if len(S) else 0.0
    chi_actual = int(max(psi_out.chi)) if psi_out.chi else 1
    trunc = None
    ss = getattr(eng, "sweep_stats", None)
    if ss and "max_trunc_err" in ss and len(ss["max_trunc_err"]):
        trunc = float(ss["max_trunc_err"][-1])
    return {"energy": float(e0), "S": s_max, "chi_actual": chi_actual,
            "trunc": trunc, "time_s": dt}


def exact_energy(N: int, h: float) -> float | None:
    """Exact ground energy via the repo solver (N<=16); None otherwise."""
    from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

    if N > STATEVECTOR_MAX_N:
        return None
    lat = make_lattice(TOPO, N, J=J, h=h)
    H = HamiltonianBuilder().build(lat)
    gt = ClassicalSolver().solve(H, lat, method="exact")
    return float(gt.ground_energy)


def run(N_values, h_values, chi_values) -> list[Row]:
    rows: list[Row] = []
    for N in N_values:
        for h in h_values:
            e_ref = exact_energy(N, h)  # None for N>16 -> self-reference below
            per_chi = {}
            for chi in chi_values:
                try:
                    r = dmrg_chi(N, h, chi)
                    per_chi[chi] = r
                    status = "ok"
                except Exception as exc:  # noqa: BLE001
                    print(f"  N={N} h={h} chi={chi} FAILED: {type(exc).__name__}: {exc}")
                    rows.append(Row("mps_dmrg", N, h, chi, status="failed", note=str(exc)[:120]))
                    continue

                # reference: exact if available, else the largest successful chi
                ref = e_ref
                mark_ref = ref
                if ref is None:
                    mark_ref = min((v["energy"] for v in per_chi.values()), default=r["energy"])
                abs_err = abs(r["energy"] - mark_ref) if mark_ref is not None else None
                rows.append(Row(
                    method="mps_dmrg", N=N, h=h,
                    chi_requested=chi, chi_actual=r["chi_actual"],
                    energy=r["energy"], e_ref=ref,
                    abs_error=abs_err, trunc_error=r["trunc"],
                    entanglement_entropy=r["S"],
                    time_s=r["time_s"], mem_bytes_est=mem_estimate_mps(N, chi),
                    status=status,
                ))
            # For N>16 (self-reference), recompute abs_error vs the max-chi energy.
            if e_ref is None and per_chi:
                best_chi = max(per_chi)
                best_e = per_chi[best_chi]["energy"]
                for row in rows:
                    if row.N == N and row.h == h and row.energy is not None:
                        row.e_ref = best_e
                        row.abs_error = abs(row.energy - best_e)
            print(f"  N={N} h={h}: done ({len([c for c in per_chi])} chi points)")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="caso corto: N=4,10 con pocos chi (verificacion)")
    ap.add_argument("--N", type=int, nargs="+", default=None)
    ap.add_argument("--h", type=float, nargs="+", default=None)
    ap.add_argument("--chi", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        N_values = args.N or [4, 10]
        h_values = args.h or [0.5, 1.0, 2.0]
        chi_values = args.chi or [2, 4, 8, 16]
    else:
        N_values = args.N or [4, 10, 50, 100]
        h_values = args.h or [0.5, 1.0, 2.0]
        chi_values = args.chi or [2, 4, 8, 16, 32, 64, 128]

    print(f"[mps] N={N_values} h={h_values} chi={chi_values}")
    rows = run(N_values, h_values, chi_values)
    path = save_run("mps_dmrg", TOPO, rows,
                    N_values=N_values, h_values=h_values, chi_values=chi_values, J=J)
    print(f"[mps] guardado -> {path}")
    print(f"[mps] {len(rows)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
