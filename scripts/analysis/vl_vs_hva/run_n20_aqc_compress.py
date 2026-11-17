#!/usr/bin/env python
"""AQC compression experiment — compress the full p3_half_nn_rx (0.9939 @ 524 2q)
down to a shallow p1 template (< 170 2q) and see what fidelity survives.

A different attack on the low-2q goal than bond pruning: instead of removing nnn
bonds from the full structure (which hits the nn-backbone floor), AQC-Tensor takes
the full's STATE (as an MPS) and re-optimizes a SMALL ansatz to maximize fidelity
with it. The compressed circuit's 2q count ≈ the template's, so a p1 template gives
a sub-170 2q circuit — if the state is representable at that depth.

Reuses: AnsatzSpec (load+build the full target), AQCCircuitCompressor (the
tensor-network compressor), cx_and_params, ground_state (exact E0/gap for
validation). No new physics.

Caveat: the N=20 h=0.5 ground state is near-degenerate (gap 7e-5, near criticality)
→ its MPS needs a HIGH bond dim. We sweep χ to find the minimum that represents the
target faithfully (a low-χ MPS of the target would make the whole comparison moot).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_aqc_compress.py \
        --target-spec results/hva_vl_study/ansatz_specs/p3_half_nn_rx_square_N20_h0.50.spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.aqc_compression import (  # noqa: E402
    AQCCircuitCompressor,
    AQCCompressionConfig,
)
from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant  # noqa: E402
from qmbp_simulation.framework.study_core import cx_and_params  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402


def _template(builder, n, lat, kind, H):
    """Build a BOUND 'good_circuit' template (AQC parametrizes it from this seed).

    generate_ansatz_from_circuit needs a circuit with NO free parameters — it reads
    the structure and the bound values as the ansatz + initial point. So we bind the
    template with a quick VQE (same recipe as the compressor's own default helper:
    COBYLA against the pure TFIM Hamiltonian). The 2q count of the eventual
    compressed circuit mirrors this template's.

    - 'hva_p1'    : standard shared-parameter HVA p=1 (~62 2q, parameter-tied).
    - '<variant>' : a registry variant (e.g. p1_rx_extra ~154 2q, bond-resolved).
    Returns (bound_circuit, n_2q, n_params).
    """
    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize as scipy_minimize

    if kind == "hva_p1":
        qc, _ = builder.create(n, 1, lat)
    else:
        qc, _ = build_variant(builder, n, lat, VARIANTS[kind])
    n2q, npar = cx_and_params(qc)
    estimator = StatevectorEstimator()

    def cost(params):
        bound = qc.assign_parameters(params)
        return estimator.run([(bound, H)]).result()[0].data.evs.item()

    rng = np.random.default_rng(42)
    init = rng.uniform(-0.05, 0.05, qc.num_parameters)
    # Binding is only a SEED for AQC (which re-optimizes by state fidelity), so a
    # modest COBYLA budget is enough; a large one is the bottleneck for the
    # bond-resolved templates (100+ params × statevector evals).
    res = scipy_minimize(cost, init, method="COBYLA", options={"maxiter": 150})
    return qc.assign_parameters(res.x), n2q, npar


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AQC-compress the full to a sub-170 2q template")
    p.add_argument("--target-spec", required=True, help="AnsatzSpec of the high-fidelity full to compress.")
    p.add_argument(
        "--templates",
        nargs="+",
        default=["hva_p1", "p1_rx_extra"],
        help="Template kinds: hva_p1 (62 2q) and/or registry variants.",
    )
    p.add_argument(
        "--chis",
        nargs="+",
        type=int,
        default=[64, 128, 256],
        help="MPS bond dims to sweep (near-critical state needs high χ).",
    )
    p.add_argument("--max-iters", type=int, default=400)
    p.add_argument("--fid-threshold", type=float, default=0.90, help="Report-only acceptance floor (not a hard stop).")
    args = p.parse_args(argv)

    spec = AnsatzSpec.load(args.target_spec)
    lat = make_lattice(spec.topology, spec.n_qubits, J=1.0, h=spec.h)
    target_qc, theta = spec.build()
    if theta is None:
        print("target spec has no stored theta — cannot compress", flush=True)
        return 1
    target_bound = target_qc.assign_parameters(theta)
    tgt_2q = sum(1 for inst in target_bound.data if inst.operation.num_qubits == 2)
    print(
        f"[aqc] target: {Path(args.target_spec).stem} fid={spec.fidelity:.4f} "
        f"native_2q={tgt_2q} N={spec.n_qubits} h={spec.h}",
        flush=True,
    )

    # Pure TFIM Hamiltonian for the quick template-binding VQE (same target the
    # compressor's own default uses — frustration enters via the lattice edges).
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    H_template = HamiltonianBuilder().build(lat)

    builder = HVACircuitBuilder()
    results = []
    for tkind in args.templates:
        good, g2q, gpar = _template(builder, spec.n_qubits, lat, tkind, H_template)
        print(f"\n[aqc] template={tkind} native_2q={g2q} params={gpar}", flush=True)
        for chi in args.chis:
            cfg = AQCCompressionConfig(
                max_bond_dim=chi, max_iterations=args.max_iters, fidelity_threshold=args.fid_threshold
            )
            try:
                compressor = AQCCircuitCompressor(cfg)
                t0 = time.time()
                res = compressor.compress_circuit(target_bound, lat, good_circuit=good)
                dt = time.time() - t0
            except Exception as exc:  # noqa: BLE001
                print(f"  χ={chi}: FAILED ({type(exc).__name__}: {exc})", flush=True)
                continue
            fpc = res.fidelity / res.n_2q_compressed if res.n_2q_compressed else 0.0
            print(
                f"  χ={chi:>3d}: fid={res.fidelity:.4f} 2q_compressed={res.n_2q_compressed} "
                f"fid/2q={fpc:.3e} iters={res.n_iterations} ({dt:.1f}s)",
                flush=True,
            )
            results.append(
                {
                    "template": tkind,
                    "chi": chi,
                    "fidelity": res.fidelity,
                    "n_2q_compressed": res.n_2q_compressed,
                    "n_2q_original": res.n_2q_original,
                    "n_2q_reduction_pct": res.n_2q_reduction_pct,
                    "fid_per_2q": fpc,
                    "n_params": res.n_params,
                    "n_iterations": res.n_iterations,
                    "converged": res.converged,
                    "wall_s": round(dt, 1),
                }
            )

    out = STUDY_ROOT / "bond_ablation" / f"aqc_compress_{spec.topology}_N{spec.n_qubits}_h{spec.h:.2f}.json"
    out.write_text(
        json.dumps(
            {
                "schema": "aqc_compress_v1",
                "topology": spec.topology,
                "N": spec.n_qubits,
                "h": spec.h,
                "target_spec": Path(args.target_spec).name,
                "target_fidelity": spec.fidelity,
                "target_native_2q": tgt_2q,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n[aqc] wrote {out}", flush=True)

    # Headline: best sub-170 2q point.
    sub170 = [r for r in results if r["n_2q_compressed"] < 170]
    if sub170:
        best = max(sub170, key=lambda r: r["fidelity"])
        print(
            f"[aqc] BEST <170 2q: template={best['template']} χ={best['chi']} "
            f"fid={best['fidelity']:.4f} @ {best['n_2q_compressed']} 2q",
            flush=True,
        )
    else:
        print("[aqc] no sub-170 2q result produced", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
