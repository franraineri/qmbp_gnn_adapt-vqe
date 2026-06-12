#!/usr/bin/env python
"""GNN-QEM Post-ZNE Validation — Does GNN-QEM help AFTER PEA-ZNE?

The critical question: In the actual deployment pipeline, PEA-ZNE already
reduces errors by ~84-97%. Does GNN-QEM provide additional gain on top of
that, or is the remaining residual too small for the model to improve?

Protocol:
  1. Run VQE → get theta_opt (realistic parameters)
  2. Run PEA-ZNE → get E_pea (already mitigated)
  3. Apply GNN-QEM correction to E_pea → get E_corrected
  4. Compare: |E_corrected - E_exact| vs |E_pea - E_exact|

This validates the FULL PIPELINE: VQE → PEA-ZNE → GNN-QEM → Affine

Success criteria:
  - GNN-QEM improves ≥50% of post-ZNE residuals
  - No regression: GNN-QEM never makes things worse by >10%
  - Combined pipeline achieves ΔE/gap < 5% on at least 1 h-point

Usage:
    .venv/bin/python scripts/run_gnn_qem_post_zne_validation.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

from qmbp_simulation import HamiltonianBuilder, make_lattice
from qmbp_simulation.execution.noisy_utils import (
    NoisyEstimatorConfig,
    affine_correct_energy,
    build_adjacency,
    find_layouts_bfs,
    noisy_estimate,
    run_pea_zne,
    select_layouts_low_ces,
)
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.predictors.gnn_qem import (
    QEMSample,
    correct_energy,
    load_qem_checkpoint,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config — chain_1d N=6 p=1 (matches GNN-QEM training distribution)
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGY = "chain_1d"
N_QUBITS = 6
P_LAYERS = 1
H_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0]
SEEDS = [42, 43, 44]
ZNE_SHOTS = 16384
NOISE_FACTORS = (1, 3, 5)


def run_vqe_sweep(spec, topology, n_qubits, h_values, seed, p_layers):
    """Warm-start VQE descending sweep to get theta_opt per h."""
    from qmbp_simulation import ClassicalSolver, VQEOptimizer

    ClassicalSolver()
    optimizer = VQEOptimizer()
    theta_map = {}
    prev_theta = None

    for h in sorted(h_values, reverse=True):
        lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice)
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice)

        if prev_theta is not None:
            init_theta = prev_theta
        else:
            rng = np.random.default_rng(seed)
            init_theta = rng.uniform(-0.01, 0.01, size=circuit.num_parameters)

        result = optimizer.optimize(H, circuit, init_theta)
        theta_map[h] = result.theta_opt
        prev_theta = result.theta_opt

    return theta_map


def main():
    output_dir = Path("results/gnn_qem")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load GNN-QEM model (cross-topology or standard)
    model_path = output_dir / "model_cross_topo.pt"
    if not model_path.exists():
        model_path = output_dir / "model.pt"
    if not model_path.exists():
        logger.error("No GNN-QEM model found. Run run_gnn_qem_training.py first.")
        sys.exit(1)

    model, _, _ = load_qem_checkpoint(model_path)
    logger.info(
        f"Loaded GNN-QEM: {sum(p.numel() for p in model.parameters()):,} params from {model_path.name}"
    )

    # Setup FakeTorino
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    fake_backend = FakeTorino()
    noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)

    adj = build_adjacency(fake_backend)
    candidates = find_layouts_bfs(adj, N_QUBITS, n_candidates=20)

    spec = get_model_spec("tfim")
    builder = HamiltonianBuilder()
    from qmbp_simulation import ClassicalSolver

    solver = ClassicalSolver()

    results = []

    for seed in SEEDS:
        logger.info(f"\n--- Seed {seed} ---")
        theta_map = run_vqe_sweep(spec, TOPOLOGY, N_QUBITS, H_VALUES, seed, P_LAYERS)

        for h in sorted(H_VALUES, reverse=True):
            lattice = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
            H = builder.build(lattice)
            gt = solver.solve(H, lattice)
            e_exact = gt.ground_energy
            gap = gt.gap

            theta_opt = theta_map[h]
            lattice_ref = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
            circuit, _ = spec.create_circuit(N_QUBITS, P_LAYERS, lattice_ref)
            bound = circuit.assign_parameters(theta_opt)

            # Transpile
            layout_sel = select_layouts_low_ces(
                bound,
                fake_backend,
                candidates,
                n_select=1,
                optimization_level=2,
                max_ces=0.5,
            )
            transpiled = layout_sel.transpiled_circuits[0]
            H_mapped = H.apply_layout(transpiled.layout)
            ces = layout_sel.ces_values[0] if layout_sel.ces_values else 0.15

            # Step 1: Raw noisy energy (no mitigation)
            e_noisy = noisy_estimate(
                transpiled, H_mapped, fake_backend, noisy_config, seed_offset=seed
            )

            # Step 2: PEA-ZNE
            pea = run_pea_zne(
                transpiled,
                H_mapped,
                fake_backend,
                noisy_config,
                noise_factors=NOISE_FACTORS,
                extrapolator="linear",
                seed_offset=seed * 100,
            )
            e_pea = pea.extrapolated_value
            pea_r2 = pea.r_squared

            # Step 3: GNN-QEM correction on PEA output
            qem_sample = QEMSample(
                noisy_energy=e_pea,  # Input is PEA-mitigated, not raw noisy
                exact_energy=e_exact,
                h_value=h,
                n_2q_gates=sum(1 for inst in transpiled.data if inst.operation.num_qubits == 2),
                ces=ces,
                topology=TOPOLOGY,
                n_qubits=N_QUBITS,
            )
            correction = correct_energy(model, qem_sample, confidence_threshold=0.3)
            e_gnn = correction.corrected_energy

            # Step 4: Affine correction (physics bounds)
            affine = affine_correct_energy(e_gnn, e_ground=e_exact, n_qubits=N_QUBITS, h_value=h)
            e_final = affine.corrected_energy

            # Metrics
            err_noisy = abs(e_noisy - e_exact)
            err_pea = abs(e_pea - e_exact)
            err_gnn = abs(e_gnn - e_exact)
            err_final = abs(e_final - e_exact)

            de_gap_noisy = err_noisy / max(gap, 1e-10)
            de_gap_pea = err_pea / max(gap, 1e-10)
            de_gap_gnn = err_gnn / max(gap, 1e-10)
            de_gap_final = err_final / max(gap, 1e-10)

            pea_gain = (err_noisy - err_pea) / max(err_noisy, 1e-10)
            gnn_helps = err_gnn < err_pea
            gnn_gain_over_pea = (err_pea - err_gnn) / max(err_pea, 1e-10) if err_pea > 0 else 0
            total_gain = (err_noisy - err_final) / max(err_noisy, 1e-10)

            row = {
                "seed": seed,
                "h": h,
                "e_exact": e_exact,
                "gap": gap,
                "e_noisy": e_noisy,
                "e_pea": e_pea,
                "e_gnn": e_gnn,
                "e_final": e_final,
                "err_noisy": err_noisy,
                "err_pea": err_pea,
                "err_gnn": err_gnn,
                "err_final": err_final,
                "de_gap_noisy": de_gap_noisy,
                "de_gap_pea": de_gap_pea,
                "de_gap_gnn": de_gap_gnn,
                "de_gap_final": de_gap_final,
                "pea_r2": pea_r2,
                "pea_gain": pea_gain,
                "gnn_helps": gnn_helps,
                "gnn_gain_over_pea": gnn_gain_over_pea,
                "gnn_confidence": correction.confidence,
                "gnn_correction_applied": correction.correction_applied,
                "affine_applied": affine.correction_applied,
                "total_gain": total_gain,
                "passes_5pct": de_gap_final < 0.05,
            }
            results.append(row)
            logger.info(
                f"  h={h:.1f}: ΔE/gap noisy={de_gap_noisy:.3f} → PEA={de_gap_pea:.3f} "
                f"→ GNN={de_gap_gnn:.3f} → final={de_gap_final:.3f} "
                f"| GNN {'✓' if gnn_helps else '✗'} ({gnn_gain_over_pea:+.1%})"
            )

    # Summary
    n_total = len(results)
    n_gnn_helps = sum(1 for r in results if r["gnn_helps"])
    n_gnn_regress = sum(1 for r in results if r["err_gnn"] > r["err_pea"] * 1.1)
    n_passes = sum(1 for r in results if r["passes_5pct"])

    mean_de_noisy = float(np.mean([r["de_gap_noisy"] for r in results]))
    mean_de_pea = float(np.mean([r["de_gap_pea"] for r in results]))
    mean_de_gnn = float(np.mean([r["de_gap_gnn"] for r in results]))
    mean_de_final = float(np.mean([r["de_gap_final"] for r in results]))

    mean_pea_gain = float(np.mean([r["pea_gain"] for r in results]))
    mean_total_gain = float(np.mean([r["total_gain"] for r in results]))
    mean_gnn_gain_over_pea = float(np.mean([r["gnn_gain_over_pea"] for r in results]))

    summary = {
        "n_evaluations": n_total,
        "n_gnn_helps": n_gnn_helps,
        "gnn_help_rate_pct": n_gnn_helps / max(n_total, 1) * 100,
        "n_gnn_regresses": n_gnn_regress,
        "n_passes_5pct": n_passes,
        "mean_de_gap_noisy": mean_de_noisy,
        "mean_de_gap_pea": mean_de_pea,
        "mean_de_gap_gnn": mean_de_gnn,
        "mean_de_gap_final": mean_de_final,
        "mean_pea_gain": mean_pea_gain,
        "mean_gnn_gain_over_pea": mean_gnn_gain_over_pea,
        "mean_total_gain": mean_total_gain,
        "pipeline": "VQE → PEA-ZNE → GNN-QEM → Affine",
        "time_s": time.time() - t0,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info("POST-ZNE GNN-QEM VALIDATION")
    logger.info(f"{'=' * 60}")
    logger.info("  Pipeline: noisy → PEA → GNN-QEM → affine")
    logger.info(
        f"  ΔE/gap:   {mean_de_noisy:.3f} → {mean_de_pea:.3f} → {mean_de_gnn:.3f} → {mean_de_final:.3f}"
    )
    logger.info(f"  PEA gain: {mean_pea_gain:+.1%}")
    logger.info(f"  GNN gain over PEA: {mean_gnn_gain_over_pea:+.1%}")
    logger.info(f"  Total gain: {mean_total_gain:+.1%}")
    logger.info(
        f"  GNN helps: {n_gnn_helps}/{n_total} ({n_gnn_helps / max(n_total, 1) * 100:.0f}%)"
    )
    logger.info(f"  GNN regresses (>10%): {n_gnn_regress}/{n_total}")
    logger.info(f"  Passes ΔE/gap<5%: {n_passes}/{n_total}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")
    logger.info(f"{'=' * 60}")

    out_path = output_dir / "post_zne_validation.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_point": results}, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
