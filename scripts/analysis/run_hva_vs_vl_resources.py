#!/usr/bin/env python
"""GNN-HVA vs Vector-Loading (VL) circuit-resource study for TFIM ground states.

Compares two ways to put a QPU into the TFIM ground state |ψ₀(h)⟩, focusing on
**circuit depth and 2-qubit gate count vs bond dimension χ**:

- **HVA (GNN pipeline):** a shallow bond-resolved ansatz, depth/gates constant in
  the state's entanglement. Approximate (expressibility-limited) but cheap.
- **VL (Haiqu-style):** load an MPS approximation of the exact ground state, bond
  dimension χ. Fidelity → 1 as χ grows, but circuit depth grows with χ. Realized
  here with the `mps-to-circuit` library (exact = Schön 2006, approximate = Ran
  2019 brick-wall), the same MPS→circuit synthesis Haiqu VL uses.

Per (h, χ, method) we report the MEASURED fidelity and the MEASURED transpiled
depth/2q, plus the algorithm's depth FORMULA as a cross-check. HVA is one
constant reference line.

Reuses: ClassicalSolver/HamiltonianBuilder/make_lattice (GS), quimb (χ
truncation), mps_to_circuit (VL), HVACircuitBuilder (HVA). Ground state and MPS
truncation validated on chain_1d.

Usage:
    python scripts/analysis/run_hva_vs_vl_resources.py            # N=10, h=1.0,2.5
    python scripts/analysis/run_hva_vs_vl_resources.py --smoke    # quick, fewer chi
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from hva_vl_study_common import BASIS_GATES, save_json, study_dir  # noqa: E402

TOPO = "chain_1d"
J = 1.0
BASIS = BASIS_GATES
TWO_Q = {"cx", "cz", "ecr"}


@dataclass
class Row:
    """One (method, h, chi) circuit-resource measurement."""

    method: str  # "hva" | "vl_exact" | "vl_approx"
    h: float
    chi_requested: int | None
    chi_real: int | None = None
    fidelity: float | None = None
    energy: float | None = None
    e_exact: float | None = None
    gap: float | None = None
    abs_error: float | None = None
    de_gap: float | None = None
    depth_raw: int | None = None
    depth_transpiled: int | None = None
    depth_2q: int | None = None
    n_2q_transpiled: int | None = None
    n_2q_formula: int | None = None
    parallelism_ratio: float | None = None
    nisq_fidelity_est: float | None = None
    n_params: int | None = None
    status: str = "ok"
    note: str = ""


def _n2q(qc) -> int:
    return sum(v for k, v in qc.count_ops().items() if k in TWO_Q)


def _quimb_to_lpr_arrays(mps, n: int) -> list[np.ndarray]:
    """Normalize a quimb MPS to uniform (bond_L, phys, bond_R) tensors.

    quimb stores boundary tensors as 2D and interior as (bond_L, phys, bond_R);
    mps_to_circuit needs uniform 3D with trivial bond=1 at the open boundaries
    and shape="lpr".
    """
    arrs = []
    for i in range(n):
        t = mps[i]
        phys = f"k{i}"
        bonds = [ix for ix in t.inds if ix != phys]
        if i == 0:
            a = t.transpose(phys, bonds[0]).data.reshape(1, 2, -1)
        elif i == n - 1:
            a = t.transpose(bonds[0], phys).data.reshape(-1, 2, 1)
        else:
            a = t.transpose(bonds[0], phys, bonds[1]).data
        arrs.append(np.asarray(a))
    return arrs


def _n2q_formula_exact(n: int, chi: int) -> int:
    """Generic MPS→circuit exact synthesis upper bound (Shende k-qubit unitary).

    Each site tensor spanning k=ceil(log2 χ) bond qubits needs a generic
    (k+1)-qubit unitary; its CNOT count is (4^(k+1) - 3(k+1) - 1)/4. Times N.
    This is an upper bound, cross-checked against the measured transpiled count.
    """
    k = max(1, math.ceil(math.log2(max(chi, 1)))) + 1
    cnot_generic = (4**k - 3 * k - 1) // 4
    return n * max(cnot_generic, 1)


def _n2q_formula_approx(n: int, num_layers: int) -> int:
    """Brick-wall (Ran 2019) approximate prep: num_layers × (N-1) 2q gates."""
    return num_layers * (n - 1)


def _fidelity_to_state(qc, target: np.ndarray) -> float:
    from qiskit.quantum_info import Statevector

    sv = Statevector(qc).data
    return float(abs(np.vdot(target, sv)) ** 2)


def _energy_metrics(bound_qc, H, e_exact: float, gap: float) -> dict:
    """⟨H⟩ of the prepared circuit and its ΔE, ΔE/gap vs the exact ground state."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(bound_qc)
    e = float(np.real(sv.expectation_value(H if isinstance(H, SparsePauliOp) else H)))
    abs_err = abs(e - e_exact)
    return {"energy": e, "abs_error": abs_err,
            "de_gap": abs_err / gap if gap and gap > 0 else None}


def _hw_metrics(transpiled_qc) -> dict:
    """depth_2q, parallelism_ratio, and estimated NISQ fidelity (Heron typical rates)."""
    from qmbp_simulation.analysis.circuit_visualizer import (
        compute_error_budget,
        transpiled_circuit_stats,
    )

    stats = transpiled_circuit_stats(transpiled_qc)
    budget = compute_error_budget(transpiled_qc, backend=None)
    return {"depth_2q": stats.get("depth_2q"),
            "parallelism_ratio": stats.get("parallelism_ratio"),
            "nisq_fidelity_est": budget.get("fidelity_estimate")}


def _ground_state(n: int, h: float):
    from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

    lat = make_lattice(TOPO, n, J=J, h=h)
    H = HamiltonianBuilder().build(lat)
    gt = ClassicalSolver().solve(H, lat)
    psi = np.asarray(gt.ground_state, dtype=complex)
    psi /= np.linalg.norm(psi)
    return lat, H, gt, psi


def _hva_row(n: int, h: float, lat, H, gt, psi_exact) -> Row:
    """HVA bond-resolved p=1 with a quick VQE-optimized θ. Constant-depth reference."""
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from scipy.optimize import minimize

    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend

    circuit, _ = HVACircuitBuilder().create_bond_resolved(n, 1, lat)
    n_params = circuit.num_parameters
    backend = NoiselessBackend()

    def cost(theta):
        return backend.evaluate(circuit, H, theta)

    rng = np.random.default_rng(42)
    theta0 = rng.uniform(-0.05, 0.05, n_params)
    res = minimize(cost, theta0, method="COBYLA", options={"maxiter": 400})
    bound = circuit.assign_parameters(res.x)

    fid = float(abs(np.vdot(psi_exact, Statevector(bound).data)) ** 2)
    em = _energy_metrics(bound, H, float(gt.ground_energy), float(gt.gap))
    pm = generate_preset_pass_manager(optimization_level=2, basis_gates=BASIS, seed_transpiler=42)
    t = pm.run(bound)
    hw = _hw_metrics(t)
    return Row(
        method="hva", h=h, chi_requested=None, fidelity=fid,
        energy=em["energy"], e_exact=float(gt.ground_energy), gap=float(gt.gap),
        abs_error=em["abs_error"], de_gap=em["de_gap"],
        depth_raw=bound.depth(), depth_transpiled=t.depth(), depth_2q=hw["depth_2q"],
        n_2q_transpiled=_n2q(t), parallelism_ratio=hw["parallelism_ratio"],
        nisq_fidelity_est=hw["nisq_fidelity_est"], n_params=n_params,
        note="bond-resolved p=1, COBYLA θ",
    )


def _vl_rows(n: int, h: float, H, gt, psi_exact, chi_values) -> list[Row]:
    import quimb.tensor as qtn
    from qiskit import transpile
    from mps_to_circuit import mps_to_circuit

    e0, gap = float(gt.ground_energy), float(gt.gap)
    rows: list[Row] = []
    mps_full = qtn.MatrixProductState.from_dense(psi_exact, dims=[2] * n)

    def _vl_row(method, chi, chi_real, qc, note, formula, v_trunc):
        t = transpile(qc, basis_gates=BASIS, optimization_level=2, seed_transpiler=42)
        em = _energy_metrics(qc, H, e0, gap)
        hw = _hw_metrics(t)
        return Row(
            method=method, h=h, chi_requested=chi, chi_real=chi_real,
            fidelity=_fidelity_to_state(qc, v_trunc), energy=em["energy"],
            e_exact=e0, gap=gap, abs_error=em["abs_error"], de_gap=em["de_gap"],
            depth_raw=qc.depth(), depth_transpiled=t.depth(), depth_2q=hw["depth_2q"],
            n_2q_transpiled=_n2q(t), n_2q_formula=formula,
            parallelism_ratio=hw["parallelism_ratio"],
            nisq_fidelity_est=hw["nisq_fidelity_est"], note=note,
        )

    for chi in chi_values:
        m = mps_full.copy()
        m.compress(max_bond=chi, cutoff=0.0)
        chi_real = int(m.max_bond())
        _v_trunc = m.to_dense().flatten()
        _v_trunc /= np.linalg.norm(_v_trunc)
        fid_trunc = float(abs(np.vdot(psi_exact, _v_trunc)) ** 2)
        arrs = _quimb_to_lpr_arrays(m, n)

        # exact synthesis (Schön): needs chi >= 4 (chi=2,3 degenerate)
        if chi >= 4:
            try:
                qc = mps_to_circuit(arrs, method="exact", shape="lpr")
                rows.append(_vl_row("vl_exact", chi, chi_real, qc,
                                    "Schön 2006 exact", _n2q_formula_exact(n, chi), _v_trunc))
            except Exception as exc:  # noqa: BLE001
                rows.append(Row("vl_exact", h, chi, chi_real, status="failed", note=str(exc)[:80]))
        else:
            rows.append(Row("vl_exact", h, chi, chi_real, status="skipped",
                            note="exact requires chi>=4"))

        # approximate synthesis (Ran brick-wall): all chi, realistic VL mode
        try:
            qc = mps_to_circuit(arrs, method="approximate", shape="lpr", num_layers=chi)
            rows.append(_vl_row("vl_approx", chi, chi_real, qc,
                                f"Ran 2019 brick-wall, num_layers={chi}",
                                _n2q_formula_approx(n, chi), _v_trunc))
        except Exception as exc:  # noqa: BLE001
            rows.append(Row("vl_approx", h, chi, chi_real, status="failed", note=str(exc)[:80]))

        # pure MPS-truncation fidelity as the VL ceiling
        rows.append(Row(
            method="vl_ceiling", h=h, chi_requested=chi, chi_real=chi_real,
            fidelity=fid_trunc, note="MPS truncation fidelity (VL best case)",
        ))
    return rows


def run(n: int, h_values, chi_values) -> list[Row]:
    rows: list[Row] = []
    for h in h_values:
        lat, H, gt, psi = _ground_state(n, h)
        print(f"  h={h}: GS ready (E0={gt.ground_energy:.4f}, gap={gt.gap:.4f})", flush=True)
        rows.append(_hva_row(n, h, lat, H, gt, psi))
        rows.extend(_vl_rows(n, h, H, gt, psi, chi_values))
        print(f"  h={h}: done", flush=True)
    return rows


def save_run(n: int, h_values, chi_values, rows) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema": "hva_vs_vl_resources_v1",
        "topology": TOPO, "n_qubits": n, "J": J,
        "config": {"h_values": list(h_values), "chi_values": list(chi_values),
                   "basis_gates": list(BASIS)},
        "rows": [asdict(r) for r in rows],
    }
    return save_json(payload, "resources", f"hva_vs_vl_{TOPO}_N{n}_{ts}.json",
                     description="HVA vs VL circuit-resource sweep vs bond dimension")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--h", type=float, nargs="+", default=None)
    ap.add_argument("--chi", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true", help="quick: fewer chi")
    args = ap.parse_args(argv)

    n = args.n
    h_values = args.h or [1.0, 2.5]
    if args.smoke:
        chi_values = args.chi or [2, 4, 8]
    else:
        chi_values = args.chi or [2, 4, 8, 16]

    print(f"[hva_vs_vl] N={n} h={h_values} chi={chi_values}", flush=True)
    rows = run(n, h_values, chi_values)
    path = save_run(n, h_values, chi_values, rows)
    print(f"[hva_vs_vl] saved -> {path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
