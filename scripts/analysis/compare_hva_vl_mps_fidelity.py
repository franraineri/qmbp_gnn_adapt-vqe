#!/usr/bin/env python
"""State-preparation fidelity: VL-ideal (Haiqu) vs HVA+GNN vs HVA-ansatz-ceiling.

Comparison (b): three DISTINCT ways of preparing the target ground state, by
exact statevector fidelity F = |⟨ψ_exact|ψ⟩|² at N ≤ statevector limit.

- VL ideal (Haiqu Variational Loading): loads the exact ground state by
  construction → F = 1.0 (reference ceiling, not a variational ansatz).
- HVA + GNN: θ predicted by the MPNN (NO VQE). What the pipeline produces.
- HVA ansatz ceiling: best θ the p=1 ansatz can reach, via a short VQE. The gap
  1.0 − F_ceiling is the ansatz expressivity limit (not a predictor failure).

Usage:
    .venv/bin/python scripts/analysis/compare_hva_vl_mps_fidelity.py \
        --topology heavy_hex --n-qubits 10 16 --h 2.5 --vqe-maxiter 150
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


def compare_state_preparations(
    topology: str = "heavy_hex",
    n_qubits: list[int] | None = None,
    h: float = 2.5,
    p_layers: int = 1,
    checkpoint: str = "unifMPNN__heavy_hex_p1_signinv_fid_v1.pt",
    vqe_maxiter: int = 150,
    vqe_restarts: int = 3,
    include_ceiling: bool = True,
) -> list[dict]:
    """Compare VL-ideal / HVA+GNN / HVA-ceiling by exact fidelity per N.

    Reusable entry point (callable from notebooks). Returns one row-dict per N
    with keys: N, E0, gap, vl_fid (=1.0), gnn_fid, gnn_E, gnn_dE, ceil_fid,
    ceil_E, ceil_dE. Rows are only produced for N whose exact ground-state
    vector is available (N ≤ statevector limit). No VQE runs when
    include_ceiling=False.
    """
    import torch
    from hva_vl_study_common import exact_ground_state_vector, state_fidelity_exact

    from qmbp_simulation import (
        HamiltonianBuilder,
        HVACircuitBuilder,
        VQEConfig,
        VQEOptimizer,
        make_lattice,
    )
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.predictors.model_zoo import _smart_load_checkpoint, resolve_checkpoint_fuzzy
    from qmbp_simulation.predictors.unified_graph import build_graph_for_model

    n_qubits = n_qubits or [10, 16]
    backend = NoiselessBackend()

    ckpt_path = resolve_checkpoint_fuzzy(checkpoint, topology=topology, p_layers=p_layers)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not resolvable: {checkpoint}")
    model = _smart_load_checkpoint(str(ckpt_path))
    model.eval()

    rows: list[dict] = []
    for n in n_qubits:
        # Shared solve: cached (E0, gap) via GroundTruthCache + exact vector.
        psi_exact, e0, gap = exact_ground_state_vector(topology, n, h, model="tfim")
        if psi_exact is None:
            continue
        lattice = make_lattice(topology, n, J=1.0, h=h)
        # tfim spec's build_hamiltonian IS HamiltonianBuilder().build, so the
        # GNN energy is measured against the same operator used for the GT.
        H_op = HamiltonianBuilder().build(lattice)
        circuit, _ = HVACircuitBuilder().create_bond_resolved(n, p_layers, lattice)

        graph = build_graph_for_model(model, lattice, h_value=h, p_layers=p_layers)
        with torch.no_grad():
            theta_gnn = np.clip(model(graph).numpy().flatten(), -np.pi, np.pi)
        if theta_gnn.size != circuit.num_parameters:
            continue
        e_gnn = backend.evaluate(circuit, H_op, theta_gnn)
        fid_gnn = state_fidelity_exact(circuit, theta_gnn, psi_exact)

        fid_ceil = e_ceil = None
        if include_ceiling:
            cfg = VQEConfig(n_restarts=vqe_restarts, maxiter=vqe_maxiter, p_layers=p_layers)
            res = VQEOptimizer(config=cfg, seed=42).optimize(H_op, circuit, theta_gnn.copy())
            e_ceil = float(res.energy)
            fid_ceil = state_fidelity_exact(circuit, res.theta_opt, psi_exact)

        rows.append(
            {
                "N": n, "E0": e0, "gap": gap,
                "vl_fid": 1.0, "vl_E": e0,
                "gnn_fid": fid_gnn, "gnn_E": e_gnn, "gnn_dE": abs(e_gnn - e0),
                "ceil_fid": fid_ceil, "ceil_E": e_ceil,
                "ceil_dE": (abs(e_ceil - e0) if e_ceil is not None else None),
            }
        )
    return rows


def format_rows(rows: list[dict]) -> str:
    """Human-readable table from compare_state_preparations() rows."""
    out = ["=" * 66]
    for r in rows:
        out.append(f"N={r['N']}  E0={r['E0']:+.4f}  gap={r['gap']:.4f}")
        out.append(f"  VL ideal (Haiqu)   : E={r['E0']:+.4f}  |dE|=0.0000  fid=1.0000")
        if r["ceil_fid"] is not None:
            out.append(
                f"  HVA ceiling (VQE)  : E={r['ceil_E']:+.4f}  "
                f"|dE|={r['ceil_dE']:.4f}  fid={r['ceil_fid']:.4f}"
            )
        out.append(
            f"  HVA + GNN (no VQE) : E={r['gnn_E']:+.4f}  "
            f"|dE|={r['gnn_dE']:.4f}  fid={r['gnn_fid']:.4f}"
        )
        out.append("=" * 66)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VL vs HVA+GNN vs HVA-ceiling fidelity")
    p.add_argument("--topology", default="heavy_hex")
    p.add_argument("--n-qubits", type=int, nargs="+", default=[10, 16])
    p.add_argument("--h", type=float, default=2.5)
    p.add_argument("--p-layers", type=int, default=1)
    p.add_argument("--checkpoint", default="unifMPNN__heavy_hex_p1_signinv_fid_v1.pt")
    p.add_argument("--vqe-maxiter", type=int, default=150)
    p.add_argument("--vqe-restarts", type=int, default=3)
    p.add_argument("--no-ceiling", action="store_true", help="Skip the VQE ansatz ceiling")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Config: {args.topology} p={args.p_layers} h={args.h}  (F via exact statevector)\n")
    rows = compare_state_preparations(
        topology=args.topology,
        n_qubits=args.n_qubits,
        h=args.h,
        p_layers=args.p_layers,
        checkpoint=args.checkpoint,
        vqe_maxiter=args.vqe_maxiter,
        vqe_restarts=args.vqe_restarts,
        include_ceiling=not args.no_ceiling,
    )
    print(format_rows(rows))
    path = save_json(
        {"schema": "state_prep_fidelity_v1", "rows": rows},
        "state_prep_fidelity", f"state_prep_fidelity_{args.topology}_p{args.p_layers}_h{args.h:.2f}.json",
        config={"topology": args.topology, "n_qubits": args.n_qubits, "h": args.h,
                "p_layers": args.p_layers, "checkpoint": args.checkpoint},
        description="VL-ideal vs HVA+GNN vs HVA-ansatz-ceiling state-prep fidelity",
    )
    print(f"\n→ Resultados: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
