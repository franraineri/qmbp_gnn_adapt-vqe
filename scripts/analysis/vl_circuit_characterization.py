#!/usr/bin/env python
"""Characterize Haiqu Vector Loading (VL) circuits vs the HVA ansatz.

For each (topology, N, h) scenario this runs Haiqu's REAL vector_loading on the
exact ground state and records what the VL circuit actually costs — beyond its
(high) fidelity: circuit depth, 2-qubit gate count, gate diversity — then puts
it side by side with the bond-resolved HVA circuit for the same system.

Answers: "besides high fidelity, what depth / characteristics does the VL
circuit take, and how does it compare to our HVA?"

Requires HAIQU_API_KEY in the environment. VL runs in the Haiqu cloud (no IBM
QPU credits). Only N ≤ 20 (vector_loading caps at 2**20 amplitudes) and
N ≤ statevector limit (need the exact ground-state vector).

Usage:
    HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_circuit_characterization.py \
        --topology heavy_hex --n-qubits 6 10 --h 2.5 1.0 --out results/analysis/vl_char.json
"""

from __future__ import annotations

import argparse

from hva_vl_study_common import (
    circuit_stats,
    exact_ground_state_vector,
    haiqu_session,
    run_vl_job,
    save_json,
    study_dir,
    transpile_hw,
)


def _hva_stats(topology: str, n: int, p: int, h: float) -> dict:
    from qmbp_simulation import HVACircuitBuilder, make_lattice

    lat = make_lattice(topology, n, J=1.0, h=h)
    circ, _ = HVACircuitBuilder().create_bond_resolved(n, p, lat)
    # Transpile to the shared hardware-like basis so depth/2q are comparable to VL.
    return circuit_stats(transpile_hw(circ, optimization_level=1))


def characterize_vl(
    topology: str,
    n_qubits: list[int],
    h_values: list[float],
    p_layers: int = 1,
    num_layers: int = 2,
    job_timeout_s: float = 900.0,
    filename: str | None = None,
) -> list[dict]:
    """Run VL on the exact ground state for each (N, h); compare vs HVA.

    Returns one row per scenario with VL fidelity + circuit stats and the HVA
    circuit stats. Per-scenario failures are captured (never abort the batch).
    Persists incrementally to results/hva_vl_study/vl_characterization/.
    """
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

    filename = filename or f"vl_char_{topology}.json"
    rows: list[dict] = []
    # Single Haiqu session for the whole batch (login/init once, not per point).
    with haiqu_session("VL_vs_HVA_characterization"):
        for n in n_qubits:
            for h in h_values:
                row = {"topology": topology, "N": n, "h": h, "p_layers": p_layers}
                try:
                    if n > STATEVECTOR_MAX_N or n > 20:
                        row["error"] = f"N={n} exceeds VL/statevector limit (<=min(20,{STATEVECTOR_MAX_N}))"
                        rows.append(row)
                        continue
                    # Shared solve: cached (E0, gap) + exact vector (None above limit).
                    psi, e0, gap = exact_ground_state_vector(topology, n, h, model="tfim")
                    if psi is None:
                        row["error"] = "no exact ground_state vector"
                        rows.append(row)
                        continue

                    fid, status, vl = run_vl_job(psi, n, num_layers=num_layers, timeout_s=job_timeout_s)
                    row["vl_status"] = status
                    row["vl_fidelity"] = fid
                    if vl is not None:
                        row["vl"] = circuit_stats(vl)
                    row["hva"] = _hva_stats(topology, n, p_layers, h)
                    row["E0"] = e0
                    row["gap"] = gap
                except Exception as exc:  # noqa: BLE001 — robustness: never abort batch
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                save_json(
                    {"rows": rows}, "vl_characterization", filename,
                    topology=topology, p_layers=p_layers, vl_num_layers=num_layers,
                )
    return rows


def format_rows(rows: list[dict]) -> str:
    out = ["=" * 78]
    for r in rows:
        head = f"N={r['N']:>2} h={r['h']:<4}"
        if "error" in r:
            out.append(f"{head}  ERROR: {r['error']}")
            out.append("-" * 78)
            continue
        vl = r.get("vl") or {}
        hva = r.get("hva") or {}
        fid = r.get("vl_fidelity")
        fid_s = f"{fid:.4f}" if isinstance(fid, (int, float)) else str(fid)
        out.append(f"{head}  VL status={r.get('vl_status')}  VL fidelity={fid_s}")
        out.append(
            f"    VL circuit : depth={vl.get('depth')}  2q_gates={vl.get('n_2q_gates')}  "
            f"qubits={vl.get('num_qubits')}"
        )
        out.append(
            f"    HVA circuit: depth={hva.get('depth')}  2q_gates={hva.get('n_2q_gates')}  "
            f"params={hva.get('num_parameters')}"
        )
        if vl.get("n_2q_gates") is not None and hva.get("n_2q_gates"):
            ratio = vl["n_2q_gates"] / hva["n_2q_gates"] if hva["n_2q_gates"] else float("inf")
            out.append(f"    → VL uses {ratio:.1f}× the HVA 2q-gate count")
        out.append("-" * 78)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Characterize Haiqu VL circuits vs HVA")
    p.add_argument("--topology", default="heavy_hex")
    p.add_argument("--n-qubits", type=int, nargs="+", default=[6, 10])
    p.add_argument("--h", type=float, nargs="+", default=[2.5, 1.0])
    p.add_argument("--p-layers", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=2, help="VL variational layers")
    p.add_argument("--job-timeout", type=float, default=900.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = characterize_vl(
        topology=args.topology,
        n_qubits=args.n_qubits,
        h_values=args.h,
        p_layers=args.p_layers,
        num_layers=args.num_layers,
        job_timeout_s=args.job_timeout,
    )
    print(format_rows(rows))
    print(f"\n→ Resultados: {study_dir('vl_characterization') / f'vl_char_{args.topology}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
