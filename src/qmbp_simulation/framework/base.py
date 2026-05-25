"""Abstract base class for all experiments.

Provides the standard lifecycle:
    1. setup() — build circuits, load ground truth, validate config
    2. run_single(seed) — execute one seed of the experiment
    3. run() — orchestrate across all seeds with checkpointing
    4. analyze() — compute summary statistics and comparisons
    5. report() — generate human-readable summary
    6. save() — persist results to JSON
    7. execute() — full lifecycle convenience method

Subclasses implement run_single() and optionally override setup()/analyze().
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.execution import ExecutionBackend, NoiselessBackend
from qmbp_simulation.framework.config import ExperimentConfig
from qmbp_simulation.framework.logging import StructuredLogger
from qmbp_simulation.framework.metrics import ExperimentMetrics, WarmColdComparison

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Abstract base for all experiments.

    Lifecycle:
        exp = MyExperiment(config)
        exp.setup()
        results = exp.run()
        analysis = exp.analyze(results)
        exp.report(analysis)
        exp.save(results, analysis)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        backend: ExecutionBackend | None = None,
        results_dir: Path | None = None,
    ):
        self.config = config
        self.backend = backend or NoiselessBackend()
        self.results_dir = results_dir or (
            Path("results/experiments") / f"exp_{config.experiment_id.lower()}"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.slog = StructuredLogger(config.experiment_id)

        # Populated during setup()
        self.circuit = None
        self.builder = None
        self.solver = None
        self.hva = None

        # Warm-cold comparison results (populated by subclasses via
        # run_warm_cold_comparison() when config.auto_warm_cold_comparison=True)
        self.warm_cold_results: list[WarmColdComparison] = []

    # ── Lifecycle Methods ────────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialize shared resources (circuit, solver, etc.).

        Override in subclass for experiment-specific setup.
        Always call super().setup() first.
        """
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models import HamiltonianBuilder, make_lattice
        from qmbp_simulation.solvers import ClassicalSolver

        # Validate config
        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

        # Build shared infrastructure
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        topology = self.config.system.topology

        base_lattice = make_lattice(topology, N, J=self.config.system.J, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)

        logger.info(
            f"Setup complete: {self.config.experiment_id} | "
            f"N={N}, p={p}, topology={topology}, "
            f"n_params={self.circuit.num_parameters}"
        )

    @abstractmethod
    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Execute the experiment for a single seed.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list[ExperimentMetrics]
            One ExperimentMetrics per h-value or per test point.
        """
        ...

    def run(self) -> dict[int, list[ExperimentMetrics]]:
        """Run experiment across all configured seeds with checkpointing.

        Returns
        -------
        dict[int, list[ExperimentMetrics]]
            Mapping seed -> list of metrics.
        """
        all_results: dict[int, list[ExperimentMetrics]] = {}
        total_start = time.time()

        logger.info(
            f"Starting experiment {self.config.experiment_id}: {len(self.config.seeds)} seeds"
        )

        for i, seed in enumerate(self.config.seeds):
            # Check for existing checkpoint
            checkpoint = self._load_checkpoint(seed)
            if checkpoint is not None:
                logger.info(f"  Seed {seed}: loaded from checkpoint")
                all_results[seed] = checkpoint
                continue

            logger.info(f"  Seed {seed} ({i + 1}/{len(self.config.seeds)}): running...")
            np.random.seed(seed)
            self.slog.log("seed_start", seed=seed, data={"index": i})
            self.slog.start_timer(f"seed_{seed}")

            try:
                t0 = time.time()
                metrics = self.run_single(seed)
                elapsed = time.time() - t0

                # Sanity-check all metrics
                for m in metrics:
                    issues = m.validate()
                    if issues:
                        for issue in issues:
                            logger.warning(f"  Seed {seed}, h={m.h_value}: {issue}")

                logger.info(f"  Seed {seed}: done in {elapsed:.1f}s ({len(metrics)} points)")
                self.slog.stop_timer(
                    f"seed_{seed}",
                    event_type="seed_complete",
                    seed=seed,
                    data={
                        "n_points": len(metrics),
                        "elapsed_s": round(elapsed, 2),
                    },
                )
                all_results[seed] = metrics
                self._save_checkpoint(seed, metrics)
            except Exception as e:
                logger.error(f"  Seed {seed}: FAILED — {e}")
                self.slog.log(
                    "seed_failed",
                    seed=seed,
                    data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                if self.config.debug:
                    raise
                all_results[seed] = []

        total_elapsed = time.time() - total_start
        logger.info(f"Experiment {self.config.experiment_id} complete: {total_elapsed:.1f}s total")

        # Clean up checkpoints after successful completion
        n_failed = sum(1 for m in all_results.values() if not m)
        if n_failed == 0:
            self._cleanup_checkpoints()
        elif n_failed == len(all_results):
            logger.error("ALL seeds failed. Check configuration and inputs.")

        return all_results

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict[str, Any]:
        """Compute summary statistics.

        Parameters
        ----------
        results : dict[int, list[ExperimentMetrics]]
            Output from run().

        Returns
        -------
        dict with keys: summary, per_seed
        """
        analysis: dict[str, Any] = {
            "experiment_id": self.config.experiment_id,
            "category": self.config.category,
            "hypothesis": self.config.hypothesis,
            "n_seeds": len(results),
            "timestamp": self._run_id,
        }

        all_metrics: list[ExperimentMetrics] = []
        per_seed_summary: dict[int, dict[str, Any]] = {}
        for seed, metrics in results.items():
            if not metrics:
                continue
            all_metrics.extend(metrics)
            per_seed_summary[seed] = {
                "mean_de_gap": float(np.mean([m.relative_error for m in metrics])),
                "min_de_gap": float(np.min([m.relative_error for m in metrics])),
                "max_de_gap": float(np.max([m.relative_error for m in metrics])),
                "n_passing": sum(1 for m in metrics if m.passes_threshold()),
                "n_total": len(metrics),
                "total_time_s": sum(m.wall_time_s for m in metrics),
            }

        analysis["per_seed"] = per_seed_summary

        if all_metrics:
            de_gaps = [m.relative_error for m in all_metrics]
            analysis["summary"] = {
                "mean_de_gap": float(np.mean(de_gaps)),
                "std_de_gap": float(np.std(de_gaps)),
                "median_de_gap": float(np.median(de_gaps)),
                "min_de_gap": float(np.min(de_gaps)),
                "max_de_gap": float(np.max(de_gaps)),
                "pass_rate": (sum(1 for d in de_gaps if d < 0.05) / len(de_gaps)),
                "n_total_points": len(all_metrics),
                "total_time_s": sum(m.wall_time_s for m in all_metrics),
                "convergence_rate": (sum(1 for m in all_metrics if m.converged) / len(all_metrics)),
                "mean_evaluations": float(
                    np.mean([m.n_evaluations for m in all_metrics if m.n_evaluations > 0])
                )
                if any(m.n_evaluations > 0 for m in all_metrics)
                else 0.0,
            }
        else:
            analysis["summary"] = {"error": "No successful runs"}

        return analysis

    def report(self, analysis: dict[str, Any]) -> str:
        """Generate human-readable report string.

        Parameters
        ----------
        analysis : dict
            Output from analyze().

        Returns
        -------
        str
            Formatted report for logging/printing.
        """
        lines = [
            f"{'=' * 60}",
            f"EXPERIMENT {self.config.experiment_id}: {self.config.description}",
            f"{'=' * 60}",
            f"Hypothesis: {self.config.hypothesis}",
            "",
        ]

        summary = analysis.get("summary", {})
        if "error" in summary:
            lines.append(f"RESULT: FAILED — {summary['error']}")
        else:
            lines.extend(
                [
                    f"Results ({analysis['n_seeds']} seeds):",
                    f"  Mean ΔE/gap: {summary['mean_de_gap']:.4f} ± {summary['std_de_gap']:.4f}",
                    f"  Pass rate:   {summary['pass_rate'] * 100:.1f}% (threshold 5%)",
                    f"  Total time:  {summary['total_time_s']:.1f}s",
                    "",
                ]
            )

            # Per-seed breakdown
            lines.append("Per-seed breakdown:")
            for seed, data in analysis.get("per_seed", {}).items():
                status = "✅" if data["n_passing"] == data["n_total"] else "⚠️"
                lines.append(
                    f"  Seed {seed}: {status} "
                    f"mean={data['mean_de_gap']:.4f}, "
                    f"pass={data['n_passing']}/{data['n_total']}, "
                    f"time={data['total_time_s']:.1f}s"
                )

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    def save(
        self,
        results: dict[int, list[ExperimentMetrics]],
        analysis: dict[str, Any],
    ) -> Path:
        """Save full results + analysis + structured logs to JSON.

        Returns path to saved file.
        """
        output = {
            "config": self.config.to_dict(),
            "backend": self.backend.name,
            "analysis": analysis,
            "results": {
                str(seed): [m.to_dict() for m in metrics] for seed, metrics in results.items()
            },
            "structured_log": self.slog.to_dict(),
            "environment": self._get_environment(),
        }

        # Include warm-cold comparison results if any were collected
        if self.warm_cold_results:
            output["warm_cold_comparisons"] = [wc.to_dict() for wc in self.warm_cold_results]

        filepath = self.results_dir / f"run_{self._run_id}.json"
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=self._json_default)

        # Also save structured log separately
        log_path = self.results_dir / f"log_{self._run_id}.json"
        self.slog.save(log_path)

        logger.info(f"Results saved to {filepath}")
        return filepath

    # ── Convenience: Full Pipeline ───────────────────────────────────────────

    def execute(self) -> dict[str, Any]:
        """Full lifecycle: setup → run → analyze → report → save.

        Returns the analysis dict.
        """
        self.setup()
        results = self.run()
        analysis = self.analyze(results)
        report_str = self.report(analysis)
        print(report_str)
        self.save(results, analysis)
        return analysis

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_exact_solution(self, h: float, N: int | None = None) -> dict:
        """Get exact solution for given h-value.

        Returns dict with keys: lattice, hamiltonian, exact (GroundTruthResult).
        """
        from qmbp_simulation.models import HamiltonianBuilder, make_lattice
        from qmbp_simulation.solvers import ClassicalSolver

        if N is None:
            N = self.config.system.n_qubits

        builder = self.builder or HamiltonianBuilder()
        solver = self.solver or ClassicalSolver()

        lattice = make_lattice(
            self.config.system.topology,
            N,
            J=self.config.system.J,
            h=h,
        )
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)
        return {"lattice": lattice, "hamiltonian": H, "exact": exact}

    def evaluate_energy(self, params: np.ndarray, hamiltonian) -> float:
        """Evaluate energy using the configured backend.

        Parameters
        ----------
        params : np.ndarray
            Circuit parameters.
        hamiltonian : SparsePauliOp
            Hamiltonian operator.

        Returns
        -------
        float
            Expectation value ⟨H⟩.
        """
        return self.backend.evaluate(self.circuit, hamiltonian, params)

    def run_warm_cold_comparison(
        self,
        h: float,
        seed: int,
        warm_init: np.ndarray,
        hamiltonian,
        exact_energy: float,
        gap: float,
        n_params: int | None = None,
        maxiter: int = 500,
    ) -> WarmColdComparison:
        """Run VQE from warm-start and cold-start, return comparison.

        Compares warm-start initialization (e.g., from MPNN or previous h-point)
        against random cold-start to quantify the benefit of warm-starting.

        Results are automatically stored in self.warm_cold_results and included
        in saved JSON output (gain_pct, iteration_savings_pct).

        Parameters
        ----------
        h : float
            Transverse field value.
        seed : int
            Random seed for cold-start initialization.
        warm_init : np.ndarray
            Warm-start parameters (from previous h-point or predictor).
        hamiltonian : SparsePauliOp
            Hamiltonian operator for energy evaluation.
        exact_energy : float
            Exact ground state energy for ΔE/gap computation.
        gap : float
            Spectral gap for normalization.
        n_params : int | None
            Number of parameters (inferred from warm_init if None).
        maxiter : int
            Maximum optimizer iterations (default 500).

        Returns
        -------
        WarmColdComparison
            Comparison result with gain_pct and iteration_savings_pct.

        Example
        -------
        >>> comparison = self.run_warm_cold_comparison(
        ...     h=1.5, seed=42, warm_init=theta_warm,
        ...     hamiltonian=H, exact_energy=e_exact, gap=gap,
        ... )
        >>> print(f"Gain: {comparison.gain_pct:.1f}%")
        """
        from scipy.optimize import minimize

        if n_params is None:
            n_params = len(warm_init)

        def cost_fn(params: np.ndarray) -> float:
            return self.backend.evaluate(self.circuit, hamiltonian, params)

        # Warm-start VQE
        result_warm = minimize(
            cost_fn,
            warm_init,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": maxiter, "ftol": 1e-14},
        )
        warm_de_gap = abs(result_warm.fun - exact_energy) / max(gap, 1e-10)

        # Cold-start VQE (random init)
        rng = np.random.default_rng(seed + 1000)
        cold_init = rng.uniform(-0.5, 0.5, n_params)
        result_cold = minimize(
            cost_fn,
            cold_init,
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": maxiter, "ftol": 1e-14},
        )
        cold_de_gap = abs(result_cold.fun - exact_energy) / max(gap, 1e-10)

        comparison = WarmColdComparison.compute(
            h_value=h,
            seed=seed,
            warm_init=warm_init,
            warm_energy=result_warm.fun,
            warm_de_gap=warm_de_gap,
            warm_nit=result_warm.nit,
            cold_init=cold_init,
            cold_energy=result_cold.fun,
            cold_de_gap=cold_de_gap,
            cold_nit=result_cold.nit,
        )

        # Auto-store for inclusion in saved results
        self.warm_cold_results.append(comparison)

        return comparison

    def compute_fidelity(self, params: np.ndarray, exact_state: np.ndarray) -> float:
        """Compute state fidelity |<psi_exact|psi_vqe>|^2."""
        from qiskit.quantum_info import Statevector, state_fidelity

        bound = self.circuit.assign_parameters(params)
        sv_vqe = Statevector(bound)
        sv_exact = Statevector(exact_state)
        return float(state_fidelity(sv_vqe, sv_exact))

    # ── Private ──────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        level = (
            logging.DEBUG
            if self.config.debug
            else (logging.INFO if self.config.verbose else logging.WARNING)
        )
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _save_checkpoint(self, seed: int, metrics: list[ExperimentMetrics]) -> None:
        """Save checkpoint for crash recovery."""
        cp_dir = self.results_dir / "checkpoints"
        cp_dir.mkdir(exist_ok=True)
        cp_path = cp_dir / f"checkpoint_{self._run_id}_seed{seed}.json"
        data = [m.to_dict() for m in metrics]
        with open(cp_path, "w") as f:
            json.dump(data, f, default=self._json_default)

    def _load_checkpoint(self, seed: int) -> list[ExperimentMetrics] | None:
        """Load checkpoint if exists for current run_id + seed."""
        cp_path = self.results_dir / "checkpoints" / f"checkpoint_{self._run_id}_seed{seed}.json"
        if cp_path.exists():
            try:
                with open(cp_path) as f:
                    data = json.load(f)
                return [ExperimentMetrics(**d) for d in data]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Checkpoint corrupted for seed {seed}: {e}. Restarting from scratch."
                )
                cp_path.unlink(missing_ok=True)
                return None
        return None

    def _cleanup_checkpoints(self) -> None:
        """Remove checkpoints after successful completion."""
        cp_dir = self.results_dir / "checkpoints"
        if cp_dir.exists():
            for f in cp_dir.glob(f"checkpoint_{self._run_id}_*.json"):
                f.unlink()

    def _get_environment(self) -> dict[str, str]:
        """Capture environment info for reproducibility."""
        import platform

        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
            "run_id": self._run_id,
        }
        try:
            import qiskit

            env["qiskit"] = qiskit.__version__
        except ImportError:
            pass
        try:
            import torch

            env["torch"] = torch.__version__
        except ImportError:
            pass
        return env

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
