#!/usr/bin/env python
"""Diagnose why HVA loses fidelity at N=18 — h=0.5 (hard) vs h=1.0 (easier).

For each h, using the exact ground state and the best saved HVA θ, reports:

- **Ground-state entanglement**: half-chain von Neumann entropy S and the
  Schmidt spectrum decay — how entangled the target is (a proxy for how many
  layers/χ any preparer needs).
- **Spectral decomposition of the prepared HVA state**: where the missing
  fidelity goes (near-degenerate partner vs spread over higher excited states),
  via ``compute_state_spectral_decomposition``.
- **Energy variance** Var(H) of the prepared state — how far from an eigenstate.

The contrast between h=0.5 (near-degenerate, gap≈0.006) and h=1.0 (gap≈0.49)
localizes whether the N=18 h=0.5 deficit is expressivity or basin/optimization.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/diagnose_n18.py --n 18 --p 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def _schmidt_spectrum(psi: np.ndarray, n: int) -> np.ndarray:
    """Half-chain Schmidt values (descending) of a statevector (little-endian)."""
    half = n // 2
    mat = psi.reshape(2 ** (n - half), 2**half)
    s = np.linalg.svd(mat, compute_uv=False)
    return np.sort(s)[::-1]


def _best_hva_theta(study: Path, topology, n, h, j2, p, npar) -> tuple[np.ndarray | None, str]:
    """Highest-fidelity saved θ of the right length across all sources."""
    from qiskit.quantum_info import Statevector

    from qmbp_simulation.framework.study_core import ground_state

    lat, qc, H, psi, e0, gap, _, _ = ground_state(topology, n, h, j2, p)
    cands: list[tuple[np.ndarray, str]] = []
    hk = round(float(h), 2)

    npz = study / "resources" / f"n{n}_frustrated" / f"theta_{topology}_N{n}_p{p}_h{hk:.2f}.npz"
    if npz.exists():
        th = np.asarray(np.load(npz)["theta"], float)
        if len(th) == npar:
            cands.append((th, npz.name))
    for fc in glob.glob(str(study / "hva_nnn_sweep" / "n*_faircov*.json")):
        d = json.loads(Path(fc).read_text())
        if int(d.get("N", -1)) != n or round(float(d.get("h", -1)), 2) != hk:
            continue
        for r in d.get("rows", []):
            th = r.get("theta_final")
            if th and len(th) == npar:
                cands.append((np.asarray(th, float), f"{Path(fc).name}:{r.get('seed_type')}"))
    for av in glob.glob(str(study / "hva_nnn_sweep" / f"ansatz_variants_{topology}_N{n}_h{hk:.2f}.json")):
        d = json.loads(Path(av).read_text())
        for r in d.get("rows", []):
            th = r.get("best_theta_final")
            if th and len(th) == npar:
                cands.append((np.asarray(th, float), f"{Path(av).name}:{r.get('variant')}"))

    if not cands:
        return None, "none"

    def fid(th):
        sv = np.asarray(Statevector(qc.assign_parameters(th)).data)
        return float(abs(np.vdot(psi, sv)) ** 2)

    best = max(cands, key=lambda c: fid(c[0]))
    return best[0], best[1]


def main(argv=None) -> int:
    from scipy.sparse.linalg import eigsh

    from qmbp_simulation.analysis.fidelity import compute_state_spectral_decomposition
    from qmbp_simulation.analysis.observables import half_chain_entropy
    from qmbp_simulation.framework.runner_base import resolve_project_root
    from qmbp_simulation.framework.study_core import ground_state

    ap = argparse.ArgumentParser(description="Diagnose HVA fidelity at N=18")
    ap.add_argument("--topology", default="square")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--n-low", type=int, default=6, help="How many low eigenvectors to resolve for the decomposition.")
    args = ap.parse_args(argv)

    root = resolve_project_root(Path(__file__))
    study = root / "results" / "hva_vl_study"

    results = {
        "topology": args.topology,
        "N": args.n,
        "p_layers": args.p,
        "J2": args.j2,
        "schema": "diagnose_n18_v1",
        "per_h": {},
    }

    for h in args.h:
        lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, h, args.j2, args.p)
        Hm = H.to_matrix(sparse=True)

        # Ground-state entanglement + Schmidt decay.
        S = half_chain_entropy(psi, args.n)
        sv = _schmidt_spectrum(psi, args.n)
        sv2 = sv**2
        # Effective Schmidt rank at 99% weight (proxy for required χ).
        cum = np.cumsum(sv2 / sv2.sum())
        chi_99 = int(np.searchsorted(cum, 0.99) + 1)

        # Low spectrum for the decomposition.
        k = max(args.n_low, 2)
        ev, evec = eigsh(Hm, k=k, which="SA")
        order = np.argsort(ev)
        eigvecs = evec[:, order].astype(complex)

        theta, src = _best_hva_theta(study, args.topology, args.n, h, args.j2, args.p, qc.num_parameters)
        h_block = {
            "gap": gap,
            "e0": e0,
            "half_chain_entropy": S,
            "schmidt_chi_99pct": chi_99,
            "schmidt_top5": [float(x) for x in sv2[:5] / sv2.sum()],
        }
        if theta is not None:
            from qiskit.quantum_info import Statevector

            psi_hva = np.asarray(Statevector(qc.assign_parameters(theta)).data)
            fid = float(abs(np.vdot(psi, psi_hva)) ** 2)
            e_hva = float(np.real(np.vdot(psi_hva, Hm @ psi_hva)))
            var_h = float(np.real(np.vdot(psi_hva, Hm @ (Hm @ psi_hva))) - e_hva**2)
            decomp = compute_state_spectral_decomposition(
                qc, np.asarray(theta), eigvecs, n_low=2, energy_variance=var_h, gap=gap
            )
            h_block.update(
                {
                    "hva_fidelity": fid,
                    "hva_theta_source": src,
                    "hva_energy_variance": var_h,
                    "ground_weight": decomp.get("ground_weight"),
                    "subspace_fidelity_E0E1": decomp.get("subspace_fidelity"),
                    "weight_outside_low2": decomp.get("weight_outside_low_subspace"),
                    "low_weights": decomp.get("low_weights"),
                    "infidelity_dominant_factor": decomp.get("infidelity_dominant_factor"),
                }
            )
        results["per_h"][f"{round(float(h), 2):.2f}"] = h_block

        print(f"\n=== N={args.n} h={round(float(h), 2):.2f} ===", flush=True)
        print(f"  gap={gap:.5f}  half-chain S={S:.4f}  Schmidt χ(99%)={chi_99}", flush=True)
        print(f"  Schmidt top-5 weights: {[f'{x:.3f}' for x in h_block['schmidt_top5']]}", flush=True)
        if theta is not None:
            print(f"  HVA F={h_block['hva_fidelity']:.4f} (θ from {src})", flush=True)
            print(
                f"  ground_weight={h_block['ground_weight']:.4f}  "
                f"E0+E1 subspace={h_block['subspace_fidelity_E0E1']:.4f}  "
                f"outside={h_block['weight_outside_low2']:.4f}",
                flush=True,
            )
            print(
                f"  Var(H)={h_block['hva_energy_variance']:.4f}  "
                f"dominant infidelity factor: {h_block.get('infidelity_dominant_factor')}",
                flush=True,
            )
            lw = h_block.get("low_weights") or []
            print(f"  low-eigvec weights: {[f'{x:.3f}' for x in lw]}", flush=True)

    save_json(
        results,
        "hva_nnn_sweep",
        f"diagnose_n{args.n}_{args.topology}_p{args.p}.json",
        params={"experiment": "diagnose_n18", "N": args.n, "p_layers": args.p},
        description="Why HVA loses fidelity at N=18: entanglement + spectral "
        "decomposition of the prepared state, per h",
    )
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
