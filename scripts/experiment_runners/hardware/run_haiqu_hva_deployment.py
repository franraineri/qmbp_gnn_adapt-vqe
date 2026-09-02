#!/usr/bin/env python
"""Run a bond-resolved HVA on real QPU via Haiqu, with GNN-predicted angles.

Loads the trained cross-N model
``unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1``,
predicts the bond-resolved θ for a target h-value, builds the corresponding
HVA circuit, and runs it on a real QPU through the Haiqu middleware stack
(state compression optional, Error Shield optional).

Pipeline
--------
    1. Load the named cross-N UnifiedMPNN from the model zoo.
    2. Predict bond-resolved θ at h (build_graph_for_model → forward → clip).
    3. Build the bond-resolved HVA circuit + TFIM Hamiltonian + ground truth.
    4. Estimate QPU cost/time with Haiqu (dry run — spends no credits).
    5. (Optional) Refine θ with VQE on the Haiqu backend (150 iter, 2 restarts).
    6. Evaluate on QPU and collect ALL Haiqu metrics + derived physics.
    7. Save everything to a structured JSON file.

Cost warning
------------
When ``--refine`` is enabled, the VQE loop calls the QPU once PER energy
evaluation: up to ``VQE_MAXITER × VQE_RESTARTS`` submissions. On real hardware
this is expensive. The Haiqu dry-run estimate (step 4) reports the *per-shot*
QPU cost; multiply by the expected number of evaluations for the refined run.
Always start with ``--dry-run`` to inspect the estimate before committing.

Examples
--------
    # Dry run only — estimate QPU cost/time, no execution
    python scripts/experiment_runners/hardware/run_haiqu_hva_deployment.py --dry-run

    # Simulator, no mitigation, no VQE (credential-free smoke test)
    python scripts/experiment_runners/hardware/run_haiqu_hva_deployment.py \
        --device fake_torino --no-mitigation --no-refine

    # Real hardware, Error Shield on, VQE refinement on
    python scripts/experiment_runners/hardware/run_haiqu_hva_deployment.py \
        --device ibm_kingston --mitigation --refine
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Global configuration (edit here or override via CLI where exposed)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CHECKPOINT = "unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1"
TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1
J = 1.0
H_VALUE = 1.0

# VQE refinement — server-side via haiqu.variational_optimization.
# NFT (default) is analytic per-parameter; scipy methods (cobyla/powell/...) also available.
VQE_OPTIMIZER = "nft"
VQE_MAXFEV = 300  # max circuit evaluations (server-side budget)
VQE_MAXITER = 500  # max parameter-update iterations
VQE_RESTARTS = 0  # extra seeded restarts beyond the GNN warm-start (best min_loss wins)

# Haiqu execution defaults
DEFAULT_DEVICE = "ibm_pittsburgh"
DEFAULT_SHOTS = 16384

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("haiqu_hva_deployment")

_ROOT = Path(__file__).resolve().parents[3]


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run bond-resolved HVA (GNN-predicted θ) on QPU via Haiqu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Haiqu device id (e.g. ibm_kingston, fake_torino)."
    )
    p.add_argument("--n-qubits", type=int, default=N_QUBITS, help="Number of qubits.")
    p.add_argument("--h", type=float, default=H_VALUE, help="Transverse-field value.")
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per circuit.")
    p.add_argument(
        "--checkpoint", default=MODEL_CHECKPOINT, help="Zoo checkpoint name/pattern to load."
    )

    # Error mitigation toggle (default ON)
    mit = p.add_mutually_exclusive_group()
    mit.add_argument(
        "--mitigation", dest="mitigation", action="store_true", help="Enable Haiqu Error Shield."
    )
    mit.add_argument(
        "--no-mitigation", dest="mitigation", action="store_false", help="Disable error mitigation."
    )
    p.set_defaults(mitigation=True)

    # VQE refinement toggle (default ON)
    ref = p.add_mutually_exclusive_group()
    ref.add_argument(
        "--refine", dest="refine", action="store_true", help="Refine θ with VQE on the QPU."
    )
    ref.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Skip VQE (deploy predicted θ as-is).",
    )
    p.set_defaults(refine=True)

    p.add_argument(
        "--vqe-optimizer",
        default=VQE_OPTIMIZER,
        choices=["nft", "cobyla", "nelder-mead", "powell", "cobyqa"],
        help="Server-side optimizer for --refine (NFT is analytic, ideal for HVA).",
    )
    p.add_argument(
        "--vqe-maxfev", type=int, default=VQE_MAXFEV, help="Max circuit evaluations (server-side)."
    )
    p.add_argument(
        "--vqe-maxiter", type=int, default=VQE_MAXITER, help="Max parameter-update iterations."
    )

    # Compression toggle (default OFF — p=1 heavy_hex is shallow)
    p.add_argument(
        "--compress", action="store_true", help="Apply Haiqu state_compression before running."
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only estimate QPU cost/time with Haiqu; do not execute or refine.",
    )
    p.add_argument(
        "--output-dir",
        default=str(_ROOT / "results" / "haiqu"),
        help="Directory for the collected-data JSON.",
    )
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Steps
# ═══════════════════════════════════════════════════════════════════════════


def load_model(checkpoint: str):
    """Resolve and load the named cross-N UnifiedMPNN from the zoo."""
    from qmbp_simulation.predictors.model_zoo import (
        _smart_load_checkpoint,
        resolve_checkpoint_fuzzy,
    )

    path = resolve_checkpoint_fuzzy(checkpoint, topology=TOPOLOGY, p_layers=P_LAYERS)
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve checkpoint {checkpoint!r} in the model zoo.\n"
            f"  Check data/model_zoo/checkpoints/ for the exact filename."
        )
    model = _smart_load_checkpoint(str(path))
    model.eval()
    logger.info("Loaded model: %s", path.name)
    return model, path


def predict_theta_bond_resolved(model, lattice, h: float) -> np.ndarray:
    """Predict bond-resolved θ for the given h (p=1 layout matches circuit)."""
    import torch

    from qmbp_simulation.predictors.unified_graph import build_graph_for_model

    graph = build_graph_for_model(model, lattice, h_value=h, p_layers=P_LAYERS)
    with torch.no_grad():
        theta = np.clip(model(graph).numpy().flatten(), -np.pi, np.pi)
    logger.info("Predicted θ: %d params (%s)", theta.size, np.round(theta[:6], 4))
    return theta


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ── Imports (kept local so --help works without heavy deps) ──────────
    from qmbp_simulation import (
        ClassicalSolver,
        HamiltonianBuilder,
        HVACircuitBuilder,
        make_lattice,
    )
    from qmbp_simulation.execution.hardware.haiqu_backend import (
        HaiquBackend,
        HaiquConfig,
    )

    n_qubits = args.n_qubits
    h = args.h

    # ── 1. Load model ────────────────────────────────────────────────────
    model, ckpt_path = load_model(args.checkpoint)

    # ── 2. Lattice, Hamiltonian, ground truth ────────────────────────────
    lattice = make_lattice(TOPOLOGY, n_qubits, J=J, h=h)
    hamiltonian = HamiltonianBuilder().build(lattice)
    gt = ClassicalSolver().solve(hamiltonian, lattice)
    e_exact, gap = gt.ground_energy, gt.gap
    logger.info("Ground truth @ h=%.2f: E_exact=%+.4f gap=%.4f", h, e_exact, gap)

    # ── 3. Predict θ + build bond-resolved circuit ───────────────────────
    theta_pred = predict_theta_bond_resolved(model, lattice, h)
    circuit, theta_vec = HVACircuitBuilder().create_bond_resolved(n_qubits, P_LAYERS, lattice)
    if theta_pred.size != circuit.num_parameters:
        raise ValueError(
            f"θ size mismatch: predicted {theta_pred.size} but circuit expects "
            f"{circuit.num_parameters}. Check n_qubits / topology / p_layers."
        )
    logger.info(
        "HVA bond-resolved circuit: %d qubits, %d params",
        circuit.num_qubits,
        circuit.num_parameters,
    )

    # ── 4. Haiqu backend + cost/time estimate (dry run) ──────────────────
    # Compute the output paths up front so the append-only sidecar shares the
    # run's timestamp and lives beside the final JSON. Every operation record
    # (cost estimate, run, refinement, evaluation) is flushed there the instant
    # it happens, so a crash mid-run still leaves job ids + collected energy on
    # disk — recoverable without the final save_collected_data.
    out_dir = Path(args.output_dir)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    mit_tag = "mit" if args.mitigation else "nomit"
    stem = f"haiqu_{TOPOLOGY}_n{n_qubits}_p{P_LAYERS}_h{h:.2f}_{mit_tag}_{args.device}_{ts}"
    out_path = out_dir / f"{stem}.json"
    sidecar_path = out_dir / f"{stem}.jsonl"

    cfg = HaiquConfig(
        device_id=args.device,
        shots=args.shots,
        use_mitigation=args.mitigation,
        use_compression=args.compress,
        vqe_optimizer=args.vqe_optimizer,
        vqe_maxfev=args.vqe_maxfev,
        vqe_maxiter=args.vqe_maxiter,
        vqe_restarts=VQE_RESTARTS,
        experiment_name=f"HVA {TOPOLOGY} N{n_qubits} p{P_LAYERS} h{h:.2f}",
        sidecar_path=str(sidecar_path),
    )
    backend = HaiquBackend(cfg)
    logger.info("Incremental sidecar: %s", sidecar_path)
    logger.info(
        "Haiqu backend: device=%s mitigation=%s compression=%s (simulator=%s)",
        cfg.device_id,
        cfg.use_mitigation,
        cfg.use_compression,
        cfg.is_simulator(),
    )

    logger.info("Estimating QPU cost/time with Haiqu (dry run — no credits spent)...")
    try:
        if args.refine:
            # Refinement runs the whole optimizer server-side, so estimate that
            # path (loop cost) rather than a single-evaluation run.
            cost = backend.estimate_cost_variational(circuit, hamiltonian, theta_pred)
            logger.info("Estimated QPU cost (server-side VQE refinement): %s", cost)
        else:
            cost = backend.estimate_cost(circuit, hamiltonian, theta_pred)
            logger.info("Estimated QPU cost (single evaluation): %s", cost)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cost estimation unavailable (%s: %s).", type(exc).__name__, exc)

    if args.refine:
        logger.warning(
            "VQE refinement is ON (server-side, optimizer=%s): up to ~%d circuit "
            "evaluations run inside the Haiqu cloud as a single tracked job. On "
            "real hardware, total QPU cost ≈ single-evaluation estimate above × "
            "number of evaluations. Consider --dry-run first.",
            args.vqe_optimizer,
            args.vqe_maxfev,
        )

    if args.dry_run:
        logger.info("Dry run complete — no execution. Saving estimate record.")
        backend.save_collected_data(
            out_path,
            extra=_context(args, ckpt_path, n_qubits, h, e_exact, gap, theta_pred, refined=False),
        )
        logger.info("Saved → %s", out_path)
        return 0

    # ── 5. Optional VQE refinement — SERVER-SIDE via Haiqu ───────────────
    # Uses haiqu.variational_optimization: the whole optimization loop runs in
    # the Haiqu cloud (one tracked job on the dashboard) rather than a local
    # loop that would round-trip to the QPU on every evaluation. Warm-started
    # with the GNN prediction; default optimizer is NFT.
    theta_final = theta_pred
    if args.refine:
        job_name = f"VQE-HVA-{TOPOLOGY}-N{n_qubits}-p{P_LAYERS}-h{h:.2f}-{args.vqe_optimizer}"
        job_desc = (
            f"Server-side {args.vqe_optimizer.upper()} refinement of bond-resolved HVA "
            f"warm-started from {args.checkpoint} at h={h:.2f} "
            f"(maxfev={args.vqe_maxfev}, maxiter={args.vqe_maxiter}, "
            f"mitigation={args.mitigation}, compression={args.compress})."
        )
        logger.info(
            "Refining θ server-side via Haiqu (optimizer=%s, maxfev=%d, maxiter=%d)...",
            args.vqe_optimizer,
            args.vqe_maxfev,
            args.vqe_maxiter,
        )
        theta_final, vqe_record = backend.refine_variational(
            circuit,
            hamiltonian,
            theta_pred,
            h=h,
            job_name=job_name,
            job_description=job_desc,
        )
        logger.info("VQE refined min_loss=%+.4f", vqe_record["min_loss"])

    # ── 6. Final QPU evaluation with full data collection ────────────────
    logger.info("Final QPU evaluation with full data collection...")
    record = backend.evaluate_full(
        circuit,
        hamiltonian,
        theta_final,
        h=h,
        e_exact=e_exact,
        gap=gap,
    )
    tag = "✅ PASS" if record["pass_5pct"] else "❌ FAIL"
    logger.info(
        "Result @ h=%.2f: E=%+.4f |ΔE|=%.4f ΔE/gap=%s %s",
        h,
        record["energy"],
        record["abs_delta_e"],
        f"{record['de_gap']:.2%}" if record["de_gap"] is not None else "n/a",
        tag,
    )

    # ── 7. Save everything ───────────────────────────────────────────────
    backend.save_collected_data(
        out_path,
        extra=_context(args, ckpt_path, n_qubits, h, e_exact, gap, theta_pred, refined=args.refine),
    )
    logger.info("Saved %d Haiqu records → %s", len(backend.records), out_path)
    return 0


def _context(args, ckpt_path, n_qubits, h, e_exact, gap, theta_pred, *, refined):
    """Build the provenance context merged into the saved JSON."""
    return {
        "model_checkpoint": ckpt_path.name,
        "topology": TOPOLOGY,
        "n_qubits": n_qubits,
        "p_layers": P_LAYERS,
        "J": J,
        "h": h,
        "e_exact": e_exact,
        "gap": gap,
        "theta_predicted": theta_pred.tolist(),
        "circuit_type": "bond_resolved",
        "vqe_refined": refined,
        "vqe_optimizer": args.vqe_optimizer if refined else None,
        "vqe_maxfev": args.vqe_maxfev if refined else None,
        "vqe_maxiter": args.vqe_maxiter if refined else None,
        "vqe_restarts": VQE_RESTARTS if refined else None,
        "mitigation": args.mitigation,
        "compression": args.compress,
        "shots": args.shots,
        "device": args.device,
    }


if __name__ == "__main__":
    sys.exit(main())
