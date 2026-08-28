#!/usr/bin/env python3
"""Fidelity Decomposition Diagnostic — why does fidelity drop near h_c?

For a given topology (default chain_1d), sweeps h across the critical region
and, for each h, optimizes a VQE state and decomposes its ground-state
infidelity into interpretable factors:

- Var(H) = ⟨H²⟩ − ⟨H⟩²        (how far |ψ⟩ is from an eigenstate — attackable)
- gap Δ = E₁ − E₀              (physics: vanishes at criticality)
- Var(H)/gap²                  (Eckart term: the infidelity budget)
- half-chain entanglement entropy of |E₀⟩ (criticality signature)
- overlap spectrum |⟨E_k|ψ⟩|²  (WHERE the lost fidelity goes: first excited
  vs spread over many states)
- dominant factor: dirty_state | small_gap | clean

This answers "is low fidelity near h_c an optimization problem (dirty state)
or a physics ceiling (small gap / critical entanglement)?" — which decides
whether more restarts/warm-start help or whether deeper p is required.

Reuses (no physics duplicated):
- HamiltonianBuilder / make_lattice / HVACircuitBuilder / model_registry
- ClassicalSolver (ground state + gap)
- analysis.observables.half_chain_entropy
- analysis.fidelity.compute_exact_fidelity + classify_infidelity_factor
- execution.select_backend (VQE energy evaluation)

Only supports N ≤ STATEVECTOR_MAX_N (needs the exact statevector for the
overlap spectrum and exact fidelity).

Usage:
    .venv/bin/python scripts/analysis/fidelity_decomposition_diagnostic.py
    .venv/bin/python scripts/analysis/fidelity_decomposition_diagnostic.py \
        --topology chain_1d --n-qubits 10 --p-layers 1 \
        --h-min 0.5 --h-max 1.5 --n-points 11 --n-restarts 8 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "results" / "analysis"


@dataclass
class HPointDiagnostic:
    """Fidelity decomposition at a single h-point."""

    h: float
    e_pred: float
    e_exact: float
    abs_error: float
    gap: float
    de_gap: float
    fidelity: float
    energy_variance: float
    variance_over_gap2: float
    entanglement_entropy: float
    overlap_ground: float  # |⟨E_0|ψ⟩|²  (== fidelity, sanity)
    overlap_first_excited: float  # |⟨E_1|ψ⟩|²
    infidelity_first_excited_fraction: float  # (1−F) attributable to |E_1⟩
    dominant_factor: str  # dirty_state | small_gap | clean


def _optimize_vqe(circuit, H, backend, n_params, n_restarts, maxiter, seed):
    """Multi-restart VQE; returns (best_theta, best_energy). Reuses backend.evaluate."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_theta = None
    best_e = float("inf")
    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        res = minimize(
            lambda t: backend.evaluate(circuit, H, t),
            x0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        if res.fun < best_e:
            best_e = float(res.fun)
            best_theta = res.x
    return best_theta, best_e


def run_diagnostic(
    *,
    topology: str = "chain_1d",
    n_qubits: int = 10,
    p_layers: int = 1,
    model_name: str = "tfim_bond_resolved",
    h_min: float = 0.5,
    h_max: float = 1.5,
    n_points: int = 11,
    n_restarts: int = 8,
    maxiter: int = 400,
    seed: int = 42,
) -> list[HPointDiagnostic]:
    """Run the fidelity-decomposition sweep across the critical region."""
    from qmbp_simulation import make_lattice
    from qmbp_simulation.analysis.fidelity import (
        classify_infidelity_factor,
        compute_exact_fidelity,
    )
    from qmbp_simulation.analysis.observables import half_chain_entropy
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import select_backend
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.classical import ClassicalSolver

    if n_qubits > STATEVECTOR_MAX_N:
        raise ValueError(
            f"Diagnostic needs the exact statevector (overlap spectrum): "
            f"N={n_qubits} > STATEVECTOR_MAX_N={STATEVECTOR_MAX_N}."
        )

    spec = get_model_spec(model_name)
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()
    backend = select_backend(n_qubits, for_vqe_loop=True)

    # Circuit structure is fixed across h (features depend on h at graph level;
    # for the bare VQE diagnostic we rebuild H per h and reuse one circuit).
    lat_ref = make_lattice(topology, n_qubits, J=1.0, h=2.0)
    circuit, _ = hva.create_bond_resolved(n_qubits, p_layers, lat_ref)
    n_params = circuit.num_parameters

    h_values = np.linspace(h_min, h_max, n_points)
    results: list[HPointDiagnostic] = []

    for h in h_values:
        h = float(h)
        lat_h = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

        # Exact ground truth: E0, gap, ground-state vector.
        gt = solver.solve(H, lat_h, method="exact")
        e_exact = float(gt.ground_energy)
        gap = float(gt.gap)
        psi_gs = gt.ground_state  # (2^N,)

        # Full spectrum for overlap decomposition (N small → dense eigh is fine).
        H_mat = np.asarray(H.to_matrix())
        evals, evecs = np.linalg.eigh(H_mat)

        # VQE optimize at this h.
        theta, e_pred = _optimize_vqe(circuit, H, backend, n_params, n_restarts, maxiter, seed)

        # Exact fidelity |⟨E0|ψ⟩|² (reuses shared implementation).
        fidelity = compute_exact_fidelity(circuit, theta, psi_gs)
        if fidelity is None:
            fidelity = 0.0

        # State vector of the VQE circuit for the overlap spectrum.
        from qiskit.quantum_info import Statevector

        psi_vqe = np.asarray(Statevector(circuit.assign_parameters(theta)).data)

        overlaps = np.abs(evecs.conj().T @ psi_vqe) ** 2  # |⟨E_k|ψ⟩|² for all k
        overlap_ground = float(overlaps[0])
        overlap_first_excited = float(overlaps[1]) if len(overlaps) > 1 else 0.0
        infidelity = max(1.0 - fidelity, 0.0)
        infid_frac_e1 = float(overlap_first_excited / infidelity) if infidelity > 1e-12 else 0.0

        # Energy variance Var(H) = ⟨H²⟩ − ⟨H⟩² directly from the statevector.
        h_psi = H_mat @ psi_vqe
        exp_h = float(np.real(np.vdot(psi_vqe, h_psi)))
        exp_h2 = float(np.real(np.vdot(h_psi, h_psi)))
        energy_variance = max(exp_h2 - exp_h**2, 0.0)

        # Entanglement entropy of the EXACT ground state (criticality signature).
        entropy = half_chain_entropy(np.asarray(psi_gs), n_qubits)

        # Dominant infidelity factor (reuses the pipeline classifier).
        decomp = classify_infidelity_factor(energy_variance, gap)

        abs_error = abs(e_pred - e_exact)
        de_gap = abs_error / max(gap, 1e-10)

        results.append(
            HPointDiagnostic(
                h=h,
                e_pred=e_pred,
                e_exact=e_exact,
                abs_error=abs_error,
                gap=gap,
                de_gap=de_gap,
                fidelity=fidelity,
                energy_variance=energy_variance,
                variance_over_gap2=(decomp["variance_over_gap2"] or 0.0),
                entanglement_entropy=entropy,
                overlap_ground=overlap_ground,
                overlap_first_excited=overlap_first_excited,
                infidelity_first_excited_fraction=infid_frac_e1,
                dominant_factor=decomp["infidelity_dominant_factor"],
            )
        )

    return results


def format_markdown(
    results: list[HPointDiagnostic],
    *,
    topology: str,
    n_qubits: int,
    p_layers: int,
    model_name: str,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Fidelity Decomposition Diagnostic — {topology}",
        "",
        f"**Generated**: {now}",
        f"**Config**: N={n_qubits}, p={p_layers}, model={model_name}",
        "**Question**: near h_c, is low fidelity a *dirty state* (attackable via "
        "optimization) or a *small gap / critical entanglement* physics ceiling?",
        "",
        "> `Var(H)` = ⟨H²⟩−⟨H⟩² (0 for an eigenstate). `Var(H)/gap²` is the Eckart "
        "infidelity budget. `S_ent` = half-chain entropy of the exact ground state "
        "(peaks at h_c). `1−F via E₁` = fraction of the infidelity carried by the "
        "first excited state (near-degeneracy signature).",
        "",
        "| h | F | 1−F | Var(H) | gap | Var/gap² | S_ent | 1−F via E₁ | Factor |",
        "|--:|--:|----:|-------:|----:|--------:|------:|-----------:|:------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.h:.3f} | {r.fidelity:.4f} | {1 - r.fidelity:.4f} | "
            f"{r.energy_variance:.4f} | {r.gap:.4f} | {r.variance_over_gap2:.4f} | "
            f"{r.entanglement_entropy:.4f} | {r.infidelity_first_excited_fraction:.2%} | "
            f"{r.dominant_factor} |"
        )

    # Summary interpretation
    n_dirty = sum(1 for r in results if r.dominant_factor == "dirty_state")
    n_small_gap = sum(1 for r in results if r.dominant_factor == "small_gap")
    n_clean = sum(1 for r in results if r.dominant_factor == "clean")
    worst = min(results, key=lambda r: r.fidelity)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Points by dominant factor: **{n_dirty} dirty_state** "
            f"(attackable via optimization), **{n_small_gap} small_gap** "
            f"(physics ceiling), **{n_clean} clean**.",
            f"- Worst fidelity: F={worst.fidelity:.4f} at h={worst.h:.3f} "
            f"(factor: {worst.dominant_factor}, gap={worst.gap:.4f}, "
            f"Var(H)={worst.energy_variance:.4f}).",
        ]
    )
    if n_dirty >= n_small_gap and n_dirty > 0:
        lines.append(
            "- **Verdict**: infidelity is dominated by dirty states → more VQE "
            "restarts / warm-start / a Var(H)-penalized objective should help "
            "*without* increasing the ansatz."
        )
    elif n_small_gap > 0:
        lines.append(
            "- **Verdict**: infidelity is dominated by the small gap / critical "
            "entanglement → a physics ceiling at this p. Raising p only in the "
            "critical window is the lever that lifts the ceiling."
        )
    lines.append("")
    lines.append("*Generated by `scripts/analysis/fidelity_decomposition_diagnostic.py`*")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topology", default="chain_1d")
    p.add_argument("--n-qubits", type=int, default=10)
    p.add_argument("--p-layers", type=int, default=1)
    p.add_argument("--model-name", default="tfim_bond_resolved")
    p.add_argument("--h-min", type=float, default=0.5)
    p.add_argument("--h-max", type=float, default=1.5)
    p.add_argument("--n-points", type=int, default=11)
    p.add_argument("--n-restarts", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true", help="Also write JSON output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = run_diagnostic(
        topology=args.topology,
        n_qubits=args.n_qubits,
        p_layers=args.p_layers,
        model_name=args.model_name,
        h_min=args.h_min,
        h_max=args.h_max,
        n_points=args.n_points,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
        seed=args.seed,
    )

    md = format_markdown(
        results,
        topology=args.topology,
        n_qubits=args.n_qubits,
        p_layers=args.p_layers,
        model_name=args.model_name,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = (
        OUTPUT_DIR / f"fidelity_decomposition_{args.topology}_N{args.n_qubits}_p{args.p_layers}.md"
    )
    md_path.write_text(md)
    print(f"  📊 Fidelity decomposition: {md_path.relative_to(ROOT)}")

    if args.json:
        json_path = md_path.with_suffix(".json")
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "topology": args.topology,
            "n_qubits": args.n_qubits,
            "p_layers": args.p_layers,
            "model_name": args.model_name,
            "points": [asdict(r) for r in results],
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"  📄 JSON: {json_path.relative_to(ROOT)}")

    # Console summary
    print(f"\n  {args.topology} N={args.n_qubits} p={args.p_layers} — near h_c:")
    for r in results:
        print(
            f"    h={r.h:.3f}: F={r.fidelity:.4f} gap={r.gap:.4f} "
            f"Var(H)={r.energy_variance:.4f} → {r.dominant_factor}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
