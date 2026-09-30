#!/usr/bin/env python
"""Dual-target fidelity: score VL and HVA against BOTH reference states.

For a scenario where the exact ground-state vector still fits (N ≤ ~22), a state
preparation can be scored two ways:

- vs the **exact** ground state |ψ_exact⟩ (eigsh) — the true physical objective.
- vs the **MPS χ=64** state |ψ_MPS⟩ (bond-capped DMRG, the input VL actually
  loads) — VL's native target.

Both VL and HVA are scored against both, giving a 2×2 fidelity matrix per
scenario. This makes the target-asymmetry explicit and quantifies how much the
choice of reference moves each method's fidelity.

Both circuits are re-simulated locally (statevector), so every number here is
recomputed under identical conditions — no reliance on the fidelity the Haiqu
cloud reported (which was vs the MPS).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/dual_target_fidelity.py \
        --n 18 --h 0.5 --p 2 --j2 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402
from mps_extraction import dmrg_mps_capped, mps_to_statevector  # noqa: E402


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2)


def _load_qpy_statevector(qpy_path: Path) -> np.ndarray:
    from qiskit import qpy
    from qiskit.quantum_info import Statevector

    with open(qpy_path, "rb") as f:
        circuits = qpy.load(f)
    return np.asarray(Statevector(circuits[0]).data)


def _hva_statevector(qc, theta: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(qc.assign_parameters(theta)).data)


def main(argv=None) -> int:
    from qmbp_simulation.framework.runner_base import resolve_project_root
    from qmbp_simulation.framework.study_core import ground_state

    ap = argparse.ArgumentParser(description="Dual-target (exact vs MPS χ) fidelity")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--chi-max", type=int, default=64)
    args = ap.parse_args(argv)

    root = resolve_project_root(Path(__file__))
    study = root / "results" / "hva_vl_study"
    hk = round(float(args.h), 2)

    # ── Reference states ─────────────────────────────────────────────────
    # Exact ground state (also gives the exact HVA circuit + H).
    lat, qc_hva, H, psi_exact, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, args.p)
    Hm = H.to_matrix(sparse=True)

    # MPS χ=64 state (VL's native target), contracted to a statevector.
    capped = dmrg_mps_capped(H, args.n, chi_max=args.chi_max)
    psi_mps = mps_to_statevector(capped)
    # Overlap between the two references — how faithful the MPS cap is.
    ref_overlap = _fidelity(psi_exact, psi_mps)

    def energy(sv):
        return float(np.real(np.vdot(sv, Hm @ sv)))

    print(f"[dual-target] N={args.n} h={hk} p={args.p} J2={args.j2}", flush=True)
    print(f"  e0(exact)={e0:.5f} gap={gap:.6f} | E0(mps χ={args.chi_max})={capped.energy:.5f}", flush=True)
    print(f"  ⟨exact|MPS⟩² = {ref_overlap:.6f}  (MPS-cap faithfulness)", flush=True)

    results = {
        "topology": args.topology,
        "N": args.n,
        "h": hk,
        "p_layers": args.p,
        "J2": args.j2,
        "chi_max": args.chi_max,
        "e0_exact": e0,
        "gap": gap,
        "e0_mps": float(capped.energy),
        "reference_overlap_exact_vs_mps": ref_overlap,
        "schema": "dual_target_fidelity_v1",
        "methods": {},
    }

    # ── VL: re-simulate its saved QPY circuit ────────────────────────────
    vl_candidates = [
        study / "vl_mps_n18" / "circuits" / f"{args.topology}_N{args.n}_h{hk:.2f}_J2{args.j2:.2f}_L8_F50_mps.qpy",
        study
        / "circuit_comparison"
        / "circuits"
        / f"tfim_frustrated_{args.topology}_n{args.n}_p{args.p}_h{hk:.2f}_j2{args.j2:.2f}_vl.qpy",
    ]
    vl_qpy = next((p for p in vl_candidates if p.exists()), None)
    if vl_qpy is not None:
        sv_vl = _load_qpy_statevector(vl_qpy)
        results["methods"]["VL"] = {
            "source_qpy": str(vl_qpy.relative_to(root)),
            "fidelity_vs_exact": _fidelity(psi_exact, sv_vl),
            "fidelity_vs_mps": _fidelity(psi_mps, sv_vl),
            "energy": energy(sv_vl),
            "de_gap_vs_exact": abs(energy(sv_vl) - e0) / gap if gap > 0 else None,
        }
        v = results["methods"]["VL"]
        print(
            f"  VL : F_vs_exact={v['fidelity_vs_exact']:.4f}  "
            f"F_vs_MPS={v['fidelity_vs_mps']:.4f}  (from {vl_qpy.name})",
            flush=True,
        )
    else:
        print(f"  VL : no QPY found ({[p.name for p in vl_candidates]})", flush=True)

    # ── HVA: pick the BEST saved θ across sources (re-sim to verify) ─────
    # The θ persisted in resources/ may be an early run, not the ceiling; the
    # ansatz_variants / compare artifacts often hold a better θ. Collect every
    # candidate θ of the right length, re-simulate, and keep the highest fidelity.
    candidates: list[tuple[np.ndarray, str]] = []
    npar = qc_hva.num_parameters

    theta_npz = (
        study / "resources" / f"n{args.n}_frustrated" / f"theta_{args.topology}_N{args.n}_p{args.p}_h{hk:.2f}.npz"
    )
    if theta_npz.exists():
        th = np.asarray(np.load(theta_npz)["theta"], float)
        if len(th) == npar:
            candidates.append((th, str(theta_npz.relative_to(root))))

    av_json = study / "hva_nnn_sweep" / f"ansatz_variants_{args.topology}_N{args.n}_h{hk:.2f}.json"
    if av_json.exists():
        import json as _json

        for r in _json.loads(av_json.read_text()).get("rows", []):
            th = r.get("best_theta_final")
            if th and len(th) == npar:
                candidates.append((np.asarray(th, float), f"{av_json.name}:{r.get('variant')}"))

    cmp_json = study / "hva_nnn_sweep" / f"compare_hva_nnn_vs_vl_{args.topology}_N{args.n}_p{args.p}.json"
    if cmp_json.exists():
        import json as _json

        for r in _json.loads(cmp_json.read_text()).get("rows", []):
            if round(float(r.get("h", -1)), 2) != hk:
                continue
            th = r.get("hva_nnn", {}).get("theta_final")
            if th and len(th) == npar:
                candidates.append((np.asarray(th, float), f"{cmp_json.name}"))

    # Fair-convergence artifacts hold the best warm-start seeds' θ (often the
    # true ceiling at large N); scan every seed row of matching (N, h, len).
    import glob as _glob
    import json as _json

    for fc in _glob.glob(str(study / "hva_nnn_sweep" / "n*_faircov*.json")):
        d = _json.loads(Path(fc).read_text())
        if int(d.get("N", -1)) != args.n or round(float(d.get("h", -1)), 2) != hk:
            continue
        for r in d.get("rows", []):
            th = r.get("theta_final")
            if th and len(th) == npar:
                candidates.append((np.asarray(th, float), f"{Path(fc).name}:{r.get('seed_type', '?')}"))

    if candidates:
        scored = [(th, src, _fidelity(psi_exact, _hva_statevector(qc_hva, th))) for th, src in candidates]
        best_th, best_src, best_f = max(scored, key=lambda c: c[2])
        sv_hva = _hva_statevector(qc_hva, best_th)
        results["methods"]["HVA"] = {
            "source_theta": best_src,
            "n_candidates": len(candidates),
            "fidelity_vs_exact": _fidelity(psi_exact, sv_hva),
            "fidelity_vs_mps": _fidelity(psi_mps, sv_hva),
            "energy": energy(sv_hva),
            "de_gap_vs_exact": abs(energy(sv_hva) - e0) / gap if gap > 0 else None,
        }
        h_ = results["methods"]["HVA"]
        print(
            f"  HVA: F_vs_exact={h_['fidelity_vs_exact']:.4f}  "
            f"F_vs_MPS={h_['fidelity_vs_mps']:.4f}  "
            f"(best of {len(candidates)} θ, from {best_src})",
            flush=True,
        )
    else:
        print("  HVA: no θ of matching length found in any source", flush=True)

    # ── 2×2 matrix printout ──────────────────────────────────────────────
    print("\n  Fidelity matrix (rows = method, cols = target):", flush=True)
    print(f"  {'':6s} {'vs exact':>10s} {'vs MPS χ':>10s}", flush=True)
    for m in ("VL", "HVA"):
        mm = results["methods"].get(m)
        if mm:
            print(f"  {m:6s} {mm['fidelity_vs_exact']:>10.4f} {mm['fidelity_vs_mps']:>10.4f}", flush=True)

    save_json(
        results,
        "hva_nnn_sweep",
        f"dual_target_fidelity_{args.topology}_N{args.n}_p{args.p}_h{hk:.2f}.json",
        params={"experiment": "dual_target_fidelity", "N": args.n, "h": hk, "p_layers": args.p},
        description="VL and HVA fidelity scored against BOTH the exact ground "
        "state and the MPS χ=64 reference (2×2 matrix), all "
        "re-simulated locally under identical conditions",
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
