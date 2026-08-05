"""VQEzy Benchmark Evaluator — Evaluate our MPNN on external VQE data.

This module implements two evaluation modes:

1. **Energy-based benchmark** (primary): For each VQEzy Hamiltonian instance,
   run our full pipeline (ground truth → VQE → MPNN) and compare the
   resulting energy against VQEzy's reported optimum. This validates that
   our HVA + MPNN approach achieves competitive ground-state accuracy.

2. **Efficiency benchmark**: Compare the number of VQE iterations needed.
   VQEzy uses 2000 Adam steps per instance. Our pipeline uses warm-start
   sweep + MPNN prediction (typically ~50-200 L-BFGS-B iterations total
   amortized across a sweep). This quantifies the practical speedup.

Usage:
    from qmbp_simulation.predictors.external_benchmarks import (
        VQEzyBenchmarkEvaluator, load_vqezy_tfi
    )

    dataset = load_vqezy_tfi("path/to/ti_8_qubit.h5", h_min=0.5, h_max=3.0)
    evaluator = VQEzyBenchmarkEvaluator(n_qubits=8, p_layers=1, topology="square")
    results = evaluator.evaluate(dataset)
    print(results.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    """Result of evaluating a single VQEzy instance.

    Attributes
    ----------
    instance_id : str
        VQEzy instance identifier.
    h : float
        Transverse field value.
    j : float
        Coupling constant.
    e_vqezy : float
        Energy reported by VQEzy (their optimizer + their ansatz).
    e_exact : float
        Exact ground state energy (our Phase 1).
    e_our_vqe : float
        Energy from our VQE (Phase 2, HVA warm-start).
    e_our_mpnn : float | None
        Energy from MPNN-predicted θ (Phase 3, zero-shot).
    de_gap_vqezy : float
        VQEzy's ΔE/gap (how close they got to exact).
    de_gap_our_vqe : float
        Our VQE ΔE/gap.
    de_gap_our_mpnn : float | None
        Our MPNN ΔE/gap (zero-shot deployment).
    abs_error_vqezy : float
        |E_vqezy - E_exact| absolute energy error.
    abs_error_ours : float
        |E_our_vqe - E_exact| absolute energy error.
    abs_error_per_site_ours : float
        |E_our_vqe - E_exact| / N — extensive scaling diagnostic.
    spectral_gap : float
        Spectral gap used for ΔE/gap normalization.
    gap_is_floor : bool
        Whether the gap is a conservative floor estimate (< 1e-6).
    fidelity_vqe : float
        Fidelity of our VQE solution.
    fidelity_mpnn : float | None
        Fidelity of our MPNN prediction.
    our_vqe_iters : int
        Number of VQE iterations (our pipeline).
    vqezy_iters : int
        Number of iterations (VQEzy, typically 2000).
    pass_5pct : bool
        Whether our pipeline achieves ΔE/gap < 5%.
    we_beat_vqezy : bool
        Whether our energy is better (lower) than VQEzy's.
    regime : str
        Classified regime: 'paramagnetic' (h/j > 2), 'critical' (0.8 < h/j < 1.2),
        'ordered' (h/j < 0.8), or 'intermediate'.
    variational_violation : bool
        True if our VQE energy is below E_exact (physics bug indicator).
    """

    instance_id: str = ""
    h: float = 0.0
    j: float = 0.0
    e_vqezy: float = 0.0
    e_exact: float = 0.0
    e_our_vqe: float = 0.0
    e_our_mpnn: float | None = None
    de_gap_vqezy: float = 0.0
    de_gap_our_vqe: float = 0.0
    de_gap_our_mpnn: float | None = None
    abs_error_vqezy: float = 0.0
    abs_error_ours: float = 0.0
    abs_error_per_site_ours: float = 0.0
    spectral_gap: float = 0.0
    gap_is_floor: bool = False
    fidelity_vqe: float = 0.0
    fidelity_mpnn: float | None = None
    our_vqe_iters: int = 0
    vqezy_iters: int = 2000
    pass_5pct: bool = False
    we_beat_vqezy: bool = False
    regime: str = "unknown"
    variational_violation: bool = False


@dataclass
class BenchmarkResult:
    """Aggregate results from VQEzy benchmark evaluation.

    Attributes
    ----------
    instance_results : list[InstanceResult]
        Per-instance detailed results.
    n_instances : int
        Total instances evaluated.
    n_pass_5pct : int
        Instances where our pipeline achieves ΔE/gap < 5%.
    n_beat_vqezy : int
        Instances where our energy < VQEzy's energy.
    pass_rate : float
        Fraction passing 5% threshold.
    beat_rate : float
        Fraction where we get better energy than VQEzy.
    mean_de_gap_ours : float
        Mean ΔE/gap from our VQE.
    mean_de_gap_vqezy : float
        Mean ΔE/gap from VQEzy.
    mean_de_gap_mpnn : float | None
        Mean ΔE/gap from MPNN zero-shot (if evaluated).
    median_de_gap_ours : float
        Median ΔE/gap from our VQE.
    total_vqe_iters_ours : int
        Total VQE iterations across all instances (our pipeline).
    total_vqe_iters_vqezy : int
        Total iterations by VQEzy.
    speedup_factor : float
        Ratio: VQEzy iterations / our iterations.
    elapsed_s : float
        Total wall-clock time for evaluation.
    config : dict
        Configuration used for this benchmark run.
    """

    instance_results: list[InstanceResult] = field(default_factory=list)
    n_instances: int = 0
    n_pass_5pct: int = 0
    n_beat_vqezy: int = 0
    pass_rate: float = 0.0
    beat_rate: float = 0.0
    mean_de_gap_ours: float = 0.0
    mean_de_gap_vqezy: float = 0.0
    mean_de_gap_mpnn: float | None = None
    median_de_gap_ours: float = 0.0
    total_vqe_iters_ours: int = 0
    total_vqe_iters_vqezy: int = 0
    speedup_factor: float = 0.0
    elapsed_s: float = 0.0
    config: dict = field(default_factory=dict)
    # ── Additional metrics ──
    mean_abs_error_ours: float = 0.0
    mean_abs_error_vqezy: float = 0.0
    mean_abs_error_per_site_ours: float = 0.0
    n_variational_violations: int = 0
    n_gap_floor_used: int = 0
    # Per-regime breakdown
    regime_stats: dict = field(default_factory=dict)
    # ── Success criteria (thresholds) ──
    success_criteria_met: bool = False
    criteria_details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "═══ VQEzy Benchmark Results ═══",
            f"Instances evaluated: {self.n_instances}",
            f"Pass rate (ΔE/gap < 5%): {self.n_pass_5pct}/{self.n_instances} "
            f"({self.pass_rate * 100:.1f}%)",
            f"Beat VQEzy rate: {self.n_beat_vqezy}/{self.n_instances} "
            f"({self.beat_rate * 100:.1f}%)",
            f"",
            f"Mean ΔE/gap — Ours: {self.mean_de_gap_ours:.4f} | "
            f"VQEzy: {self.mean_de_gap_vqezy:.4f}",
            f"Median ΔE/gap — Ours: {self.median_de_gap_ours:.4f}",
            f"Mean |ΔE| — Ours: {self.mean_abs_error_ours:.4f} | "
            f"VQEzy: {self.mean_abs_error_vqezy:.4f}",
            f"Mean |ΔE|/N — Ours: {self.mean_abs_error_per_site_ours:.6f}",
        ]
        if self.mean_de_gap_mpnn is not None:
            lines.append(f"Mean ΔE/gap — MPNN zero-shot: {self.mean_de_gap_mpnn:.4f}")
        lines.extend([
            f"",
            f"Iterations — Ours total: {self.total_vqe_iters_ours} | "
            f"VQEzy total: {self.total_vqe_iters_vqezy}",
            f"Speedup factor: {self.speedup_factor:.1f}×",
            f"Wall-clock: {self.elapsed_s:.1f}s",
        ])
        # Diagnostics
        if self.n_variational_violations > 0:
            lines.append(
                f"⚠️  Variational violations: {self.n_variational_violations}/{self.n_instances}"
            )
        if self.n_gap_floor_used > 0:
            lines.append(
                f"ℹ️  Gap floor used (near-degenerate): {self.n_gap_floor_used}/{self.n_instances}"
            )
        # Regime breakdown
        if self.regime_stats:
            lines.append("")
            lines.append("Per-regime breakdown:")
            for regime, stats in sorted(self.regime_stats.items()):
                lines.append(
                    f"  {regime}: n={stats['n']}, pass_rate={stats['pass_rate']:.0%}, "
                    f"beat_rate={stats['beat_rate']:.0%}, mean_ΔE/gap={stats['mean_de_gap']:.4f}"
                )
        # Success criteria
        lines.append("")
        verdict = "✅ PASS" if self.success_criteria_met else "❌ FAIL"
        lines.append(f"Success criteria: {verdict}")
        for criterion, detail in self.criteria_details.items():
            status = "✅" if detail["passed"] else "❌"
            lines.append(f"  {status} {criterion}: {detail['message']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        from qmbp_simulation.utils.helpers import json_serialize
        return {
            "n_instances": self.n_instances,
            "n_pass_5pct": self.n_pass_5pct,
            "n_beat_vqezy": self.n_beat_vqezy,
            "pass_rate": self.pass_rate,
            "beat_rate": self.beat_rate,
            "mean_de_gap_ours": self.mean_de_gap_ours,
            "mean_de_gap_vqezy": self.mean_de_gap_vqezy,
            "mean_de_gap_mpnn": self.mean_de_gap_mpnn,
            "median_de_gap_ours": self.median_de_gap_ours,
            "mean_abs_error_ours": self.mean_abs_error_ours,
            "mean_abs_error_vqezy": self.mean_abs_error_vqezy,
            "mean_abs_error_per_site_ours": self.mean_abs_error_per_site_ours,
            "total_vqe_iters_ours": self.total_vqe_iters_ours,
            "total_vqe_iters_vqezy": self.total_vqe_iters_vqezy,
            "speedup_factor": self.speedup_factor,
            "elapsed_s": self.elapsed_s,
            "n_variational_violations": self.n_variational_violations,
            "n_gap_floor_used": self.n_gap_floor_used,
            "success_criteria_met": self.success_criteria_met,
            "criteria_details": self.criteria_details,
            "regime_stats": self.regime_stats,
            "config": self.config,
            "per_instance": [
                {
                    "id": r.instance_id,
                    "h": r.h,
                    "j": r.j,
                    "regime": r.regime,
                    "e_exact": json_serialize(r.e_exact),
                    "e_vqezy": json_serialize(r.e_vqezy),
                    "e_our_vqe": json_serialize(r.e_our_vqe),
                    "e_our_mpnn": json_serialize(r.e_our_mpnn),
                    "de_gap_vqezy": json_serialize(r.de_gap_vqezy),
                    "de_gap_our_vqe": json_serialize(r.de_gap_our_vqe),
                    "de_gap_our_mpnn": json_serialize(r.de_gap_our_mpnn),
                    "abs_error_ours": json_serialize(r.abs_error_ours),
                    "abs_error_vqezy": json_serialize(r.abs_error_vqezy),
                    "abs_error_per_site_ours": json_serialize(r.abs_error_per_site_ours),
                    "spectral_gap": json_serialize(r.spectral_gap),
                    "gap_is_floor": r.gap_is_floor,
                    "fidelity_vqe": json_serialize(r.fidelity_vqe),
                    "fidelity_mpnn": json_serialize(r.fidelity_mpnn),
                    "our_vqe_iters": r.our_vqe_iters,
                    "pass_5pct": r.pass_5pct,
                    "we_beat_vqezy": r.we_beat_vqezy,
                    "variational_violation": r.variational_violation,
                }
                for r in self.instance_results
            ],
        }



class VQEzyBenchmarkEvaluator:
    """Evaluate our GNN-HVA pipeline against VQEzy dataset.

    This evaluator runs our full pipeline (Phase 1-3) on VQEzy Hamiltonians
    and compares energy accuracy and computational cost.

    Two evaluation strategies:

    1. **Per-instance**: Each VQEzy instance is a separate (j, h) point.
       We compute exact ground truth, run single-point VQE, and optionally
       deploy a pre-trained MPNN.

    2. **Sweep-based** (recommended): Group instances by coupling j, then
       run our standard descending sweep over h-values for each j-group.
       This leverages our warm-start advantage (VQEzy optimizes each point
       independently with 2000 iterations).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match VQEzy instances).
    p_layers : int
        HVA ansatz depth (default 1).
    topology : str
        Lattice topology for our framework (default "square" for 2D TFI).
    n_restarts : int
        VQE restarts per point (default 3).
    maxiter : int
        VQE max iterations per restart (default 200).
    seed : int
        Random seed for reproducibility.
    mpnn_checkpoint : str | Path | None
        Path to pre-trained MPNN checkpoint for zero-shot evaluation.
        If None, skips MPNN evaluation (VQE-only comparison).
    """

    def __init__(
        self,
        n_qubits: int = 8,
        p_layers: int = 1,
        topology: str = "square",
        n_restarts: int = 3,
        maxiter: int = 200,
        seed: int = 42,
        mpnn_checkpoint: str | Path | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.topology = topology
        self.n_restarts = n_restarts
        self.maxiter = maxiter
        self.seed = seed
        self.mpnn_checkpoint = mpnn_checkpoint

        # Lazy-loaded components
        self._solver = None
        self._optimizer = None
        self._circuit = None
        self._mpnn = None

    def _init_components(self):
        """Initialize pipeline components (lazy, once)."""
        from qmbp_simulation import (
            ClassicalSolver,
            HVACircuitBuilder,
            VQEConfig,
            VQEOptimizer,
        )
        from qmbp_simulation.execution import NoiselessBackend

        self._solver = ClassicalSolver()
        self._backend = NoiselessBackend()

        vqe_config = VQEConfig(
            n_restarts=self.n_restarts,
            maxiter=self.maxiter,
            method="L-BFGS-B",
        )
        self._optimizer = VQEOptimizer(config=vqe_config, backend=self._backend, seed=self.seed)
        self._circuit_builder = HVACircuitBuilder()

        # Load MPNN if checkpoint provided
        if self.mpnn_checkpoint is not None:
            from qmbp_simulation.predictors import load_mpnn_checkpoint
            self._mpnn = load_mpnn_checkpoint(str(self.mpnn_checkpoint))
            self._mpnn.eval()
            logger.info(f"Loaded MPNN checkpoint: {self.mpnn_checkpoint}")

    def _build_lattice_for_instance(self, instance):
        """Build a LatticeConfig matching the VQEzy instance topology."""
        from qmbp_simulation.predictors.external_benchmarks.vqezy_loader import (
            _build_rectangular_edges,
        )
        from qmbp_simulation.models.data_models import LatticeConfig

        n_qubits = instance.n_qubits if hasattr(instance, 'n_qubits') else self.n_qubits
        j_val = instance.j if hasattr(instance, 'j') else 1.0
        h_val = instance.h if hasattr(instance, 'h') else 1.0

        # Build correct edges for VQEzy's grid
        grid_shape = getattr(instance, 'grid_shape', None)
        if grid_shape is not None:
            rows, cols = grid_shape[0], grid_shape[1]
            edges = _build_rectangular_edges(rows, cols)
        else:
            # Default: use our framework's square generation
            from qmbp_simulation.models.hamiltonian import generate_square
            edges = generate_square(n_qubits)

        # Compute coordination numbers
        coord = np.zeros(n_qubits)
        for i, j_site in edges:
            coord[i] += 1
            coord[j_site] += 1

        return LatticeConfig(
            topology=self.topology,
            n_qubits=n_qubits,
            J=j_val,
            h=h_val,
            edges=edges,
            coordination_numbers=coord,
            periodic=False,
        )

    def _get_circuit_for_lattice(self, lattice):
        """Get or build HVA circuit for a given lattice."""
        circuit, _ = self._circuit_builder.create(
            lattice.n_qubits, self.p_layers, lattice
        )
        return circuit

    def evaluate(
        self,
        dataset,
        *,
        mode: str = "sweep",
        j_tolerance: float = 0.05,
        rescale_h_by_j: bool = False,
    ) -> BenchmarkResult:
        """Run full benchmark evaluation on VQEzy dataset.

        Parameters
        ----------
        dataset : VQEzyDataset
            Loaded VQEzy instances to evaluate.
        mode : str
            Evaluation mode:
            - "sweep": Group by j-value, run descending h-sweep per group
              (leverages warm-start, recommended).
            - "per_instance": Evaluate each instance independently
              (fair 1:1 comparison but misses our sweep advantage).
        j_tolerance : float
            Tolerance for grouping instances by coupling j (default 0.05).
        rescale_h_by_j : bool
            If True, feed h/j (instead of h) to the MPNN for prediction.
            Tests whether the MPNN learned θ(h/j) generalization. Only
            affects MPNN predictions, not VQE evaluation.

        Returns
        -------
        BenchmarkResult
            Aggregate and per-instance results.
        """
        if self._solver is None:
            self._init_components()

        # ── Input validation ──
        if len(dataset) == 0:
            logger.warning("Empty dataset provided. Returning empty result.")
            return BenchmarkResult(elapsed_s=0.0)

        # Validate n_qubits match
        dataset_n = dataset.n_qubits
        if dataset_n != 0 and dataset_n != self.n_qubits:
            raise ValueError(
                f"Dataset n_qubits={dataset_n} does not match evaluator "
                f"n_qubits={self.n_qubits}. Use matching configuration."
            )

        t_start = time.perf_counter()

        if mode == "sweep":
            instance_results = self._evaluate_sweep(
                dataset, j_tolerance=j_tolerance, rescale_h_by_j=rescale_h_by_j
            )
        elif mode == "per_instance":
            instance_results = self._evaluate_per_instance(dataset)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'sweep' or 'per_instance'.")

        elapsed = time.perf_counter() - t_start

        # Compute aggregate statistics
        return self._compute_aggregate(instance_results, elapsed)

    def _evaluate_sweep(
        self,
        dataset,
        j_tolerance: float = 0.05,
        rescale_h_by_j: bool = False,
    ) -> list[InstanceResult]:
        """Evaluate using sweep-based strategy (groups by j).

        Groups instances with similar j-values, sorts h descending within
        each group, and runs our warm-start VQE sweep. This is our natural
        advantage over VQEzy's independent per-instance optimization.
        """
        from qmbp_simulation import HamiltonianBuilder
        from qmbp_simulation.models.data_models import LatticeConfig

        # Group instances by j-value (within tolerance)
        groups = self._group_by_j(dataset.instances, j_tolerance)
        logger.info(
            f"Grouped {len(dataset)} instances into {len(groups)} j-groups "
            f"(tolerance={j_tolerance})"
        )

        all_results: list[InstanceResult] = []
        builder = HamiltonianBuilder()

        for j_val, instances in groups.items():
            # Sort by h descending (our sweep direction)
            instances_sorted = sorted(instances, key=lambda x: x.h, reverse=True)
            h_values = np.array([inst.h for inst in instances_sorted])

            if len(h_values) < 2:
                # Single-point group — evaluate independently
                for inst in instances_sorted:
                    result = self._evaluate_single(inst, builder)
                    all_results.append(result)
                continue

            logger.info(
                f"  j={j_val:.2f}: {len(h_values)} h-points, "
                f"h∈[{h_values.min():.2f}, {h_values.max():.2f}]"
            )

            # Build lattice template for this j-group (use first instance for grid)
            ref_inst = instances_sorted[0]
            lattice = self._build_lattice_for_instance(ref_inst)
            # Override J to the group's j-value
            lattice = LatticeConfig(
                topology=lattice.topology,
                n_qubits=lattice.n_qubits,
                J=j_val,
                h=float(h_values[0]),
                edges=lattice.edges,
                coordination_numbers=lattice.coordination_numbers,
                periodic=lattice.periodic,
            )

            # Build circuit for this lattice
            circuit = self._get_circuit_for_lattice(lattice)

            # Phase 1: Exact ground truth for all h-points in this group
            exact_data = []
            for h in h_values:
                lat_h = LatticeConfig(
                    topology=lattice.topology,
                    n_qubits=lattice.n_qubits,
                    J=j_val,
                    h=float(h),
                    edges=lattice.edges,
                    coordination_numbers=lattice.coordination_numbers,
                    periodic=lattice.periodic,
                )
                H = builder.build(lat_h)
                gt = self._solver.solve(H, lat_h)
                exact_data.append(gt)

            # Phase 2: Warm-start descending sweep
            vqe_results = self._optimizer.descending_sweep(
                h_values=h_values,
                circuit=circuit,
                lattice=lattice,
                exact_data=exact_data,
            )

            # Phase 3: MPNN zero-shot prediction (if available)
            mpnn_energies = None
            mpnn_fidelities = None
            if self._mpnn is not None:
                j_rescale_val = j_val if rescale_h_by_j else None
                mpnn_energies, mpnn_fidelities = self._mpnn_predict_sweep(
                    lattice, h_values, exact_data, circuit, j_rescale=j_rescale_val
                )

            # Build per-instance results
            for i, (inst, gt, vqe_r) in enumerate(
                zip(instances_sorted, exact_data, vqe_results, strict=False)
            ):
                e_exact = gt.ground_energy
                # Gap floor 0.1: conservative for VQEzy benchmark to avoid
                # ΔE/gap→∞ at near-degenerate points. The gap_is_floor flag
                # marks these for post-hoc filtering. Standard pipeline uses
                # max(gap, 1e-10) which can inflate ΔE/gap near QPT.
                gap = gt.gap if gt.gap and gt.gap > 1e-10 else 0.1
                gap_is_floor = gt.gap is None or gt.gap < 1e-6

                e_mpnn = None
                fid_mpnn = None
                if mpnn_energies is not None:
                    e_mpnn = mpnn_energies[i]
                    fid_mpnn = mpnn_fidelities[i] if mpnn_fidelities is not None else None

                all_results.append(self._build_instance_result(
                    inst=inst,
                    e_exact=e_exact,
                    gap=gap,
                    gap_is_floor=gap_is_floor,
                    vqe_energy=vqe_r.energy,
                    vqe_fidelity=vqe_r.fidelity,
                    vqe_iters=vqe_r.n_iterations,
                    e_mpnn=e_mpnn,
                    fid_mpnn=fid_mpnn,
                ))

        return all_results

    def _evaluate_per_instance(self, dataset) -> list[InstanceResult]:
        """Evaluate each instance independently (no warm-start advantage)."""
        from qmbp_simulation import HamiltonianBuilder

        builder = HamiltonianBuilder()
        results: list[InstanceResult] = []

        for idx, inst in enumerate(dataset.instances):
            if idx % 50 == 0:
                logger.info(f"  Evaluating instance {idx}/{len(dataset)}...")
            result = self._evaluate_single(inst, builder)
            results.append(result)

        return results

    def _evaluate_single(self, inst, builder) -> InstanceResult:
        """Evaluate a single VQEzy instance."""
        from qmbp_simulation.models.data_models import LatticeConfig

        lattice = self._build_lattice_for_instance(inst)
        circuit = self._get_circuit_for_lattice(lattice)
        n_params = circuit.num_parameters

        # Phase 1: Exact ground truth
        H = builder.build(lattice)
        gt = self._solver.solve(H, lattice)
        e_exact = gt.ground_energy
        gap = gt.gap if gt.gap and gt.gap > 1e-10 else 0.1
        gap_is_floor = gt.gap is None or gt.gap < 1e-6

        # Phase 2: Single-point VQE
        initial_guess = np.random.default_rng(self.seed).uniform(
            -0.01, 0.01, n_params
        )
        vqe_result = self._optimizer.optimize(
            H, circuit, initial_guess,
            exact_energy=e_exact,
            exact_state=gt.ground_state,
        )

        return self._build_instance_result(
            inst=inst,
            e_exact=e_exact,
            gap=gap,
            gap_is_floor=gap_is_floor,
            vqe_energy=vqe_result.energy,
            vqe_fidelity=vqe_result.fidelity,
            vqe_iters=vqe_result.n_iterations,
        )

    def _build_instance_result(
        self,
        *,
        inst,
        e_exact: float,
        gap: float,
        gap_is_floor: bool,
        vqe_energy: float,
        vqe_fidelity: float,
        vqe_iters: int,
        e_mpnn: float | None = None,
        fid_mpnn: float | None = None,
    ) -> InstanceResult:
        """Build an InstanceResult with all metrics and checks."""
        abs_error_vqezy = abs(inst.e_optimal - e_exact)
        abs_error_ours = abs(vqe_energy - e_exact)
        n_qubits = inst.n_qubits if hasattr(inst, 'n_qubits') else self.n_qubits

        de_gap_vqezy = abs_error_vqezy / gap
        de_gap_ours = abs_error_ours / gap

        # Variational principle check: E_VQE should never be below E_exact
        variational_violation = (
            np.isfinite(vqe_energy) and vqe_energy < e_exact - 1e-8
        )
        if variational_violation:
            logger.warning(
                f"Variational violation at j={inst.j:.2f}, h={inst.h:.2f}: "
                f"E_VQE={vqe_energy:.6f} < E_exact={e_exact:.6f} "
                f"(Δ={e_exact - vqe_energy:.2e})"
            )

        # MPNN metrics
        de_gap_mpnn = None
        if e_mpnn is not None:
            de_gap_mpnn = abs(e_mpnn - e_exact) / gap

        # Regime classification
        regime = self._classify_regime(inst.h, inst.j)

        return InstanceResult(
            instance_id=inst.instance_id,
            h=inst.h,
            j=inst.j,
            e_vqezy=inst.e_optimal,
            e_exact=e_exact,
            e_our_vqe=vqe_energy,
            e_our_mpnn=e_mpnn,
            de_gap_vqezy=de_gap_vqezy,
            de_gap_our_vqe=de_gap_ours,
            de_gap_our_mpnn=de_gap_mpnn,
            abs_error_vqezy=abs_error_vqezy,
            abs_error_ours=abs_error_ours,
            abs_error_per_site_ours=abs_error_ours / n_qubits,
            spectral_gap=gap,
            gap_is_floor=gap_is_floor,
            fidelity_vqe=vqe_fidelity,
            fidelity_mpnn=fid_mpnn,
            our_vqe_iters=vqe_iters,
            vqezy_iters=inst.n_vqe_iterations,
            pass_5pct=de_gap_ours < 0.05,
            we_beat_vqezy=vqe_energy < inst.e_optimal,
            regime=regime,
            variational_violation=variational_violation,
        )

    def _mpnn_predict_sweep(
        self,
        lattice,
        h_values: np.ndarray,
        exact_data: list,
        circuit,
        j_rescale: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Deploy MPNN zero-shot on a sweep of h-values.

        Parameters
        ----------
        lattice : LatticeConfig
        h_values : np.ndarray
        exact_data : list[GroundTruthResult]
        circuit : QuantumCircuit
        j_rescale : float | None
            If provided, rescale h-values as h_input = h / j_rescale before
            feeding to the MPNN. This tests the hypothesis that the MPNN
            learned θ(h/j) rather than θ(h) specifically.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            (energies, fidelities) from MPNN-predicted θ.
        """
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import HamiltonianBuilder
        from qmbp_simulation.models.data_models import LatticeConfig

        builder = HamiltonianBuilder()
        edge_index_np, coord = builder.build_graph_data(lattice)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        energies = np.zeros(len(h_values))
        fidelities = np.zeros(len(h_values))

        self._mpnn.eval()
        with torch.no_grad():
            for i, h in enumerate(h_values):
                # Apply h→h/j rescaling if requested
                h_input = float(h) / j_rescale if j_rescale else float(h)

                # Build graph for this h-point (using rescaled h as feature)
                h_feat = np.full(lattice.n_qubits, h_input)
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float)], axis=1),
                    dtype=torch.float32,
                )
                data = Data(x=x, edge_index=edge_index)
                data.batch = torch.zeros(x.size(0), dtype=torch.long)

                # Predict θ
                theta_pred = self._mpnn(data).numpy().flatten()

                # Evaluate energy with predicted θ (using REAL Hamiltonian, not rescaled)
                lat_h = LatticeConfig(
                    topology=lattice.topology,
                    n_qubits=lattice.n_qubits,
                    J=lattice.J,
                    h=float(h),
                    edges=lattice.edges,
                    coordination_numbers=lattice.coordination_numbers,
                    periodic=lattice.periodic,
                )
                H = builder.build(lat_h)
                energy = self._backend.evaluate(circuit, H, theta_pred)
                energies[i] = energy

                # Compute fidelity if ground state available
                if exact_data[i].ground_state is not None:
                    fid = self._backend.compute_fidelity(
                        circuit, theta_pred, exact_data[i].ground_state
                    )
                    fidelities[i] = float(fid)

        return energies, fidelities

    @staticmethod
    def _group_by_j(
        instances: list,
        tolerance: float = 0.05,
    ) -> dict[float, list]:
        """Group instances by coupling j within tolerance."""
        groups: dict[float, list] = {}

        for inst in instances:
            # Find existing group within tolerance
            matched = False
            for j_key in groups:
                if abs(inst.j - j_key) <= tolerance:
                    groups[j_key].append(inst)
                    matched = True
                    break
            if not matched:
                groups[inst.j] = [inst]

        return groups

    def _compute_aggregate(
        self,
        instance_results: list[InstanceResult],
        elapsed: float,
    ) -> BenchmarkResult:
        """Compute aggregate statistics from per-instance results."""
        n = len(instance_results)
        if n == 0:
            return BenchmarkResult(elapsed_s=elapsed)

        de_gaps_ours = np.array([r.de_gap_our_vqe for r in instance_results])
        de_gaps_vqezy = np.array([r.de_gap_vqezy for r in instance_results])
        abs_errors_ours = np.array([r.abs_error_ours for r in instance_results])
        abs_errors_vqezy = np.array([r.abs_error_vqezy for r in instance_results])
        abs_errors_per_site = np.array([r.abs_error_per_site_ours for r in instance_results])

        n_pass = sum(1 for r in instance_results if r.pass_5pct)
        n_beat = sum(1 for r in instance_results if r.we_beat_vqezy)
        n_violations = sum(1 for r in instance_results if r.variational_violation)
        n_gap_floor = sum(1 for r in instance_results if r.gap_is_floor)
        total_iters_ours = sum(r.our_vqe_iters for r in instance_results)
        total_iters_vqezy = sum(r.vqezy_iters for r in instance_results)

        # MPNN aggregate if available
        mean_de_gap_mpnn = None
        mpnn_results = [r for r in instance_results if r.de_gap_our_mpnn is not None]
        if mpnn_results:
            mean_de_gap_mpnn = float(np.mean([r.de_gap_our_mpnn for r in mpnn_results]))

        speedup = total_iters_vqezy / total_iters_ours if total_iters_ours > 0 else 0.0

        # ── Per-regime breakdown ──
        regime_stats = self._compute_regime_stats(instance_results)

        # ── Success criteria evaluation ──
        criteria_details = self._evaluate_success_criteria(
            n, n_pass, n_beat, de_gaps_ours, speedup, n_violations
        )
        success_criteria_met = all(d["passed"] for d in criteria_details.values())

        return BenchmarkResult(
            instance_results=instance_results,
            n_instances=n,
            n_pass_5pct=n_pass,
            n_beat_vqezy=n_beat,
            pass_rate=n_pass / n,
            beat_rate=n_beat / n,
            mean_de_gap_ours=float(np.mean(de_gaps_ours)),
            mean_de_gap_vqezy=float(np.mean(de_gaps_vqezy)),
            mean_de_gap_mpnn=mean_de_gap_mpnn,
            median_de_gap_ours=float(np.median(de_gaps_ours)),
            mean_abs_error_ours=float(np.mean(abs_errors_ours)),
            mean_abs_error_vqezy=float(np.mean(abs_errors_vqezy)),
            mean_abs_error_per_site_ours=float(np.mean(abs_errors_per_site)),
            total_vqe_iters_ours=total_iters_ours,
            total_vqe_iters_vqezy=total_iters_vqezy,
            speedup_factor=speedup,
            elapsed_s=elapsed,
            n_variational_violations=n_violations,
            n_gap_floor_used=n_gap_floor,
            regime_stats=regime_stats,
            success_criteria_met=success_criteria_met,
            criteria_details=criteria_details,
            config={
                "n_qubits": self.n_qubits,
                "p_layers": self.p_layers,
                "topology": self.topology,
                "n_restarts": self.n_restarts,
                "maxiter": self.maxiter,
                "seed": self.seed,
                "mpnn_checkpoint": str(self.mpnn_checkpoint) if self.mpnn_checkpoint else None,
            },
        )

    @staticmethod
    def _classify_regime(h: float, j: float) -> str:
        """Classify the (j, h) point into a physical regime."""
        from qmbp_simulation.analysis.metrics import classify_regime
        return classify_regime(h, j)

    @staticmethod
    def _compute_regime_stats(instance_results: list[InstanceResult]) -> dict:
        """Compute per-regime statistics."""
        from collections import defaultdict
        regime_groups: dict[str, list[InstanceResult]] = defaultdict(list)
        for r in instance_results:
            regime_groups[r.regime].append(r)

        stats = {}
        for regime, results in regime_groups.items():
            n_r = len(results)
            n_pass_r = sum(1 for r in results if r.pass_5pct)
            n_beat_r = sum(1 for r in results if r.we_beat_vqezy)
            de_gaps = [r.de_gap_our_vqe for r in results]
            stats[regime] = {
                "n": n_r,
                "pass_rate": n_pass_r / n_r if n_r > 0 else 0.0,
                "beat_rate": n_beat_r / n_r if n_r > 0 else 0.0,
                "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else 0.0,
                "median_de_gap": float(np.median(de_gaps)) if de_gaps else 0.0,
            }
        return stats

    @staticmethod
    def _evaluate_success_criteria(
        n: int,
        n_pass: int,
        n_beat: int,
        de_gaps_ours: np.ndarray,
        speedup: float,
        n_violations: int,
    ) -> dict:
        """Evaluate whether the benchmark meets publishable success criteria.

        Criteria (from 01_vqezy_external_benchmark.md):
        - C1: PassRate ≥ 50% on VQEzy instances → strong generalization claim
        - C2: BeatRate ≥ 70% → our pipeline is better than VQEzy's approach
        - C3: Speedup ≥ 10× → practical efficiency advantage
        - C4: No variational principle violations → physics consistency
        - C5: Median ΔE/gap < 0.10 → overall quality

        Returns dict[criterion_name, {"passed": bool, "message": str}]
        """
        pass_rate = n_pass / n if n > 0 else 0.0
        beat_rate = n_beat / n if n > 0 else 0.0
        median_de_gap = float(np.median(de_gaps_ours)) if len(de_gaps_ours) > 0 else 1.0

        criteria = {}

        # C1: PassRate ≥ 50%
        criteria["pass_rate_50pct"] = {
            "passed": pass_rate >= 0.50,
            "message": f"PassRate={pass_rate:.1%} (threshold ≥50%)",
        }

        # C2: BeatRate ≥ 70%
        criteria["beat_rate_70pct"] = {
            "passed": beat_rate >= 0.70,
            "message": f"BeatRate={beat_rate:.1%} (threshold ≥70%)",
        }

        # C3: Speedup ≥ 10×
        criteria["speedup_10x"] = {
            "passed": speedup >= 10.0,
            "message": f"Speedup={speedup:.1f}× (threshold ≥10×)",
        }

        # C4: No variational violations
        criteria["no_variational_violations"] = {
            "passed": n_violations == 0,
            "message": f"Violations={n_violations}/{n} (threshold: 0)",
        }

        # C5: Median ΔE/gap < 10%
        criteria["median_de_gap_10pct"] = {
            "passed": median_de_gap < 0.10,
            "message": f"Median ΔE/gap={median_de_gap:.4f} (threshold <0.10)",
        }

        return criteria
