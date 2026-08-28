#!/usr/bin/env python
"""Evaluate GNN State Preparation Fidelity.

Measures the fidelity F = |<ψ_exact|ψ_HVA(θ_GNN)>|² of the GNN-predicted
state against the exact ground state. This is the key metric for hardware
viability: if F > F_min, the QPU experiment will produce valid results.

Methods:
A) Direct fidelity (N ≤ 22): exact statevector overlap via NoiselessBackend
B) Energy-gap lower bound (N > 22): F ≥ 1 - (E_pred - E₀) / gap (first-order,
   variational-principle bound; weaker than and not comparable to the Eckart
   variance bound F ≥ 1 - Var(H)/gap²). From large_n_extrapolation NPZ data.

Usage:
    # Direct fidelity for N=10-20 at h=3.0 (hardware operating point)
    python scripts/analysis/evaluate_gnn_fidelity.py \
        --topology heavy_hex --n-qubits 10 12 14 16 20 \
        --h-values 3.0 2.5 --save

    # Energy bound from extrapolation data (N>22)
    python scripts/analysis/evaluate_gnn_fidelity.py \
        --topology heavy_hex --n-qubits 20 30 40 \
        --from-extrapolation --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


@dataclass
class FidelityResult:
    """Fidelity measurement for a single (N, h) configuration."""

    n_qubits: int
    h: float
    fidelity: float
    method: str  # "direct_statevector" or "energy_gap_bound"
    e_pred: float | None = None
    e_exact: float | None = None
    gap: float | None = None
    de_gap: float | None = None


@dataclass
class FidelityReport:
    """Complete fidelity evaluation report."""

    topology: str
    results: list[FidelityResult] = field(default_factory=list)
    f_min_reference: float | None = None  # From fidelity threshold analysis
    hardware_viable: bool = False


def evaluate_direct_fidelity(
    topology: str,
    n_qubits: int,
    h: float,
    p_layers: int = 1,
) -> FidelityResult | None:
    """Compute exact fidelity via statevector overlap (N ≤ 22).

    F = |<ψ_exact|ψ_HVA(θ_GNN)>|²

    Uses NoiselessBackend.get_statevector() for the HVA state and
    exact diagonalization for the ground state.
    """
    import torch

    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice
    from qmbp_simulation.predictors.model_zoo import load_best_model_for
    from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    # Load MPNN model
    try:
        model, entry, source = load_best_model_for(
            topology,
            model="tfim_bond_resolved",
            p_layers=p_layers,
            n_target=n_qubits,
        )
        model.eval()
    except Exception as e:
        logger.warning(f"  No model for {topology}: {e}")
        return None

    # Predict θ
    lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
    graph = build_unified_bond_resolved_graph(lattice, h_value=h, p_layers=p_layers)
    with torch.no_grad():
        theta_pred = model(graph).cpu().numpy().flatten()

    # Build HVA circuit and get statevector
    hva = HVACircuitBuilder()
    circuit, _ = hva.create_bond_resolved(n_qubits, p_layers, lattice)
    n_params = circuit.num_parameters
    if len(theta_pred) != n_params:
        if len(theta_pred) < n_params:
            theta_pred = np.pad(theta_pred, (0, n_params - len(theta_pred)))
        else:
            theta_pred = theta_pred[:n_params]

    backend = NoiselessBackend()
    psi_gnn = backend.get_statevector(circuit, theta_pred)

    # Get exact ground state
    builder = HamiltonianBuilder()
    H_op = builder.build(lattice)
    if n_qubits <= 14:
        from scipy.linalg import eigh

        H_dense = np.asarray(H_op.to_matrix())
        eigenvalues, eigenvectors = eigh(H_dense)
        psi_exact = eigenvectors[:, 0]
        e_exact = eigenvalues[0]
        gap = eigenvalues[1] - eigenvalues[0]
    else:
        from scipy.sparse.linalg import eigsh

        H_sparse = H_op.to_matrix(sparse=True)
        eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which="SA")
        idx = np.argsort(eigenvalues)
        psi_exact = eigenvectors[:, idx[0]]
        e_exact = eigenvalues[idx[0]]
        gap = eigenvalues[idx[1]] - eigenvalues[idx[0]]

    psi_exact /= np.linalg.norm(psi_exact)

    # Compute fidelity via the canonical helper: it routes both states through
    # Qiskit's Statevector/state_fidelity, keeping a single qubit-ordering
    # convention and avoiding a raw np.vdot overlap that is exposed to endianness
    # mismatches between the circuit and the solver eigenvector.
    from qmbp_simulation.analysis.fidelity import compute_exact_fidelity

    fidelity = compute_exact_fidelity(circuit, theta_pred, psi_exact)
    if fidelity is None:
        logger.warning(f"  Exact fidelity failed for {topology} N={n_qubits} h={h}")
        return None

    # Also compute energy of GNN state
    H_mat = H_op.to_matrix(sparse=True) if n_qubits > 14 else np.asarray(H_op.to_matrix())
    e_pred = float(np.real(psi_gnn.conj() @ (H_mat @ psi_gnn)))
    de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)

    # Persist GT to cache (free data)
    try:
        gt_cache = GroundTruthCache()
        if gt_cache.get(topology, n_qubits, "tfim_bond_resolved", h) is None:
            gt_cache.put(
                topology=topology,
                n_qubits=n_qubits,
                model="tfim_bond_resolved",
                h=h,
                energy=float(e_exact),
                gap=float(gap),
                method="fidelity_eval",
            )
            gt_cache.flush()
    except Exception:
        pass

    return FidelityResult(
        n_qubits=n_qubits,
        h=h,
        fidelity=fidelity,
        method="direct_statevector",
        e_pred=e_pred,
        e_exact=float(e_exact),
        gap=float(gap),
        de_gap=de_gap,
    )


def evaluate_energy_bound(
    topology: str,
    n_qubits: int,
    p_layers: int = 1,
) -> list[FidelityResult]:
    """First-order (energy-gap) fidelity lower bound from extrapolation NPZ.

    F ≥ 1 − (E_pred − E₀) / gap

    This is the variational-principle bound: writing |ψ⟩ = Σ c_k|E_k⟩, the
    excess energy is E_pred − E₀ = Σ|c_k|²(E_k − E₀) ≥ (1 − F)·gap, hence
    F = |c₀|² ≥ 1 − (E_pred − E₀)/gap. It is a RIGOROUS but WEAKER bound than
    the Eckart / variance bound (F ≥ 1 − Var(H)/gap²) used by the canonical
    ``compute_variance_fidelity_bound``; the two use different information
    (mean energy gap here vs. energy variance there) and are NOT directly
    comparable. Method tag ``"energy_gap_bound"`` marks this provenance.

    Validity: like Eckart, this only bounds GROUND-state fidelity while the
    state lies below the midpoint between E₀ and E₁ (E_pred < E₀ + gap/2).
    Points that fail this are skipped (the bound would not certify ground
    fidelity there).

    Uses existing evaluated NPZ data — zero compute cost (no circuit rebuild).
    """
    npz_path = (
        _project_root / "data" / "large_n_extrapolation" / f"{topology}_N{n_qubits}_p{p_layers}.npz"
    )
    if not npz_path.exists():
        # Try training data
        npz_path = (
            _project_root / "data" / "multi_n_training" / f"{topology}_N{n_qubits}_p{p_layers}.npz"
        )
    if not npz_path.exists():
        return []

    data = np.load(npz_path, allow_pickle=True)
    h_values = np.asarray(data["h_values"], dtype=float)

    has_e_vqe = "e_vqe" in data
    has_e_exact = "e_exact" in data
    has_gaps = "gaps" in data

    if not (has_e_vqe and has_e_exact and has_gaps):
        return []

    e_vqe = np.asarray(data["e_vqe"], dtype=float)
    e_exact = np.asarray(data["e_exact"], dtype=float)
    gaps = np.asarray(data["gaps"], dtype=float)

    results = []
    for i in range(len(h_values)):
        if not (np.isfinite(e_vqe[i]) and np.isfinite(e_exact[i]) and gaps[i] > 1e-10):
            continue
        de = e_vqe[i] - e_exact[i]
        if de < 0:
            de = 0  # Variational principle: E_pred >= E_exact (numerical noise)
        # Validity: skip points where the state sits past the E₀↔E₁ midpoint —
        # there the bound no longer certifies ground-state fidelity.
        if de >= 0.5 * gaps[i]:
            continue
        f_bound = 1.0 - de / gaps[i]
        f_bound = max(0.0, min(1.0, f_bound))  # Clamp to [0, 1]
        de_gap = abs(e_vqe[i] - e_exact[i]) / gaps[i]

        results.append(
            FidelityResult(
                n_qubits=n_qubits,
                h=float(h_values[i]),
                fidelity=f_bound,
                method="energy_gap_bound",
                e_pred=float(e_vqe[i]),
                e_exact=float(e_exact[i]),
                gap=float(gaps[i]),
                de_gap=de_gap,
            )
        )

    return results


def run_fidelity_evaluation(
    topology: str,
    n_qubits_list: list[int],
    h_values: list[float] | None = None,
    p_layers: int = 1,
    from_extrapolation: bool = False,
) -> FidelityReport:
    """Run complete fidelity evaluation.

    Parameters
    ----------
    topology : str
        Lattice topology.
    n_qubits_list : list[int]
        System sizes to evaluate.
    h_values : list[float] | None
        Field values (for direct method). If None, uses [3.0, 2.5].
    p_layers : int
        HVA depth.
    from_extrapolation : bool
        If True, use energy bound from NPZ (no compute needed).

    Returns
    -------
    FidelityReport
    """
    if h_values is None:
        h_values = [3.0, 2.5]

    report = FidelityReport(topology=topology)

    # Load F_min from threshold analysis (if available)
    threshold_path = (
        _project_root / "results" / "analysis" / f"dqpt_fidelity_threshold_{topology}_N10.json"
    )
    if threshold_path.exists():
        try:
            with open(threshold_path) as f:
                threshold_data = json.load(f)
            report.f_min_reference = threshold_data.get("f_min")
        except Exception:
            pass

    for n in n_qubits_list:
        if from_extrapolation or n > 22:
            # Energy bound method (zero cost)
            bounds = evaluate_energy_bound(topology, n, p_layers)
            report.results.extend(bounds)
            if bounds:
                avg_f = np.mean([r.fidelity for r in bounds])
                logger.info(f"  N={n}: {len(bounds)} pts, mean F_bound={avg_f:.4f} (energy bound)")
        else:
            # Direct statevector method
            for h in h_values:
                result = evaluate_direct_fidelity(topology, n, h, p_layers)
                if result is not None:
                    report.results.append(result)
                    logger.info(
                        f"  N={n}, h={h:.1f}: F={result.fidelity:.4f}, ΔE/gap={result.de_gap:.4f}"
                    )

    # Hardware viability check
    if report.results:
        f_min = report.f_min_reference if report.f_min_reference is not None else 0.50
        all_above = all(r.fidelity > f_min for r in report.results)
        report.hardware_viable = all_above

    return report


def print_report(report: FidelityReport) -> None:
    """Pretty-print the fidelity report."""
    print(f"\n{'=' * 70}")
    print(f"  GNN Fidelity Evaluation: {report.topology}")
    print(f"{'=' * 70}")

    if not report.results:
        print("  No results.")
        return

    print(f"\n  {'N':>3} | {'h':>4} | {'F':>6} | {'ΔE/gap':>7} | {'Method':>16} | Go?")
    print(f"  {'-' * 55}")

    f_min = report.f_min_reference or 0.50
    for r in sorted(report.results, key=lambda x: (x.n_qubits, x.h)):
        go = "✓" if r.fidelity > f_min else "✗"
        print(
            f"  {r.n_qubits:>3} | {r.h:>4.1f} | {r.fidelity:>6.4f} | "
            f"{r.de_gap:>7.4f} | {r.method:>16} | {go}"
        )

    print(f"\n  F_min reference: {f_min:.2f}")
    print(f"  Hardware viable (all F > F_min): {report.hardware_viable}")

    # Summary by N
    by_n = {}
    for r in report.results:
        by_n.setdefault(r.n_qubits, []).append(r.fidelity)
    print("\n  Summary by N:")
    for n in sorted(by_n.keys()):
        fs = by_n[n]
        print(f"    N={n}: mean F={np.mean(fs):.4f}, min F={min(fs):.4f} ({len(fs)} pts)")


def save_report(report: FidelityReport, out_path: Path) -> None:
    """Save report to JSON."""
    from qmbp_simulation.utils.helpers import json_serialize

    output = {
        "topology": report.topology,
        "f_min_reference": report.f_min_reference,
        "hardware_viable": report.hardware_viable,
        "results": [
            {
                "n_qubits": r.n_qubits,
                "h": r.h,
                "fidelity": r.fidelity,
                "method": r.method,
                "e_pred": r.e_pred,
                "e_exact": r.e_exact,
                "gap": r.gap,
                "de_gap": r.de_gap,
            }
            for r in report.results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN state preparation fidelity")
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--n-qubits", type=int, nargs="+", default=[10, 12, 14, 16, 20])
    parser.add_argument("--h-values", type=float, nargs="+", default=[3.0, 2.5])
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument(
        "--from-extrapolation",
        action="store_true",
        help="Use energy bound from existing NPZ (zero compute)",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    report = run_fidelity_evaluation(
        topology=args.topology,
        n_qubits_list=args.n_qubits,
        h_values=args.h_values,
        p_layers=args.p_layers,
        from_extrapolation=args.from_extrapolation,
    )

    print_report(report)

    if args.save:
        suffix = "bound" if args.from_extrapolation else "direct"
        out_path = (
            _project_root / "results" / "analysis" / f"gnn_fidelity_{args.topology}_{suffix}.json"
        )
        save_report(report, out_path)


if __name__ == "__main__":
    main()
