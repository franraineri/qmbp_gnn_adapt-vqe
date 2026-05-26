"""Performance benchmarking suite for qmbp_simulation components.

Provides programmatic access to component benchmarks for regression
detection and timing baselines. Used by scripts/benchmark.py as CLI
wrapper, and importable for CI/CD integration.

Usage:
    from qmbp_simulation.framework.benchmarking import BenchmarkSuite

    suite = BenchmarkSuite(n_qubits=[4, 6, 8], n_repeats=3)
    results = suite.run(components=["solver", "vqe"])
    suite.print_summary(results)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes
    ----------
    component : str
        Name of the benchmarked component.
    n_qubits : int
        System size used for the benchmark.
    elapsed_s : float
        Median elapsed time in seconds.
    details : dict
        Additional timing details (min, max, repeats).
    """

    component: str
    n_qubits: int
    elapsed_s: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)


class BenchmarkSuite:
    """Configurable benchmark suite for pipeline components.

    Benchmarks: solver (exact diag), circuit (HVA construction),
    vqe (single-point optimization), mpnn (forward pass inference).

    Parameters
    ----------
    n_qubits : list[int]
        System sizes to benchmark.
    n_repeats : int
        Number of repeats per benchmark (median is reported).
    verbose : bool
        Print progress during execution.
    """

    AVAILABLE_COMPONENTS = ("solver", "circuit", "vqe", "mpnn")

    def __init__(
        self,
        n_qubits: list[int] | None = None,
        n_repeats: int = 3,
        verbose: bool = False,
    ) -> None:
        self.n_qubits = n_qubits or [4, 6, 8]
        self.n_repeats = n_repeats
        self.verbose = verbose

    def run(
        self,
        components: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run benchmarks for specified components.

        Parameters
        ----------
        components : list[str] | None
            Components to benchmark. If None, runs all available.

        Returns
        -------
        list[BenchmarkResult]
            Results for each (component, n_qubits) combination.
        """
        targets = components or list(self.AVAILABLE_COMPONENTS)
        results: list[BenchmarkResult] = []

        for component in targets:
            bench_fn = _BENCH_FUNCTIONS.get(component)
            if bench_fn is None:
                if self.verbose:
                    print(f"  Skipping unknown component: {component}")
                continue

            if self.verbose:
                print(f"  [{component}]")

            for n in self.n_qubits:
                try:
                    result = bench_fn(n, self.n_repeats)
                    results.append(result)
                    if self.verbose:
                        print(f"    N={n:2d}: {result.elapsed_s * 1000:8.2f} ms (median)")
                except Exception as e:
                    if self.verbose:
                        print(f"    N={n:2d}: ERROR — {e}")

        return results

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        """Print a formatted summary table.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Results from run().
        """
        print("-" * 60)
        print(f"{'Component':<12} {'N':<6} {'Time (ms)':<12} {'Notes'}")
        print("-" * 60)
        for r in results:
            print(f"{r.component:<12} {r.n_qubits:<6} {r.elapsed_s * 1000:<12.2f}")
        print("-" * 60)

    def to_dict(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Serialize results to a JSON-compatible dict.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Results from run().

        Returns
        -------
        dict
            Serializable results with timestamp.
        """
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "n_qubits": self.n_qubits,
                "n_repeats": self.n_repeats,
            },
            "results": [r.to_dict() for r in results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual benchmark functions
# ─────────────────────────────────────────────────────────────────────────────


def _bench_solver(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark exact diagonalization solver."""
    from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.5)
    H = builder.build(lattice)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        solver.solve(H, lattice)
        times.append(time.perf_counter() - t0)

    return BenchmarkResult(
        component="solver",
        n_qubits=n_qubits,
        elapsed_s=float(np.median(times)),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def _bench_circuit(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark HVA circuit construction."""
    from qmbp_simulation import HVACircuitBuilder, make_lattice

    hva = HVACircuitBuilder()
    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.5)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        hva.create(n_qubits, 2, lattice)
        times.append(time.perf_counter() - t0)

    return BenchmarkResult(
        component="circuit",
        n_qubits=n_qubits,
        elapsed_s=float(np.median(times)),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def _bench_vqe(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark single-point VQE optimization."""
    from qmbp_simulation import (
        ClassicalSolver,
        HamiltonianBuilder,
        HVACircuitBuilder,
        VQEOptimizer,
        make_lattice,
    )
    from qmbp_simulation.models import VQEConfig

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.5)
    H = builder.build(lattice)
    exact = solver.solve(H, lattice)
    circuit, _ = hva.create(n_qubits, 2, lattice)

    config = VQEConfig(p_layers=2, n_restarts=1, maxiter=200, enable_callbacks=False)

    times = []
    for _ in range(n_repeats):
        optimizer = VQEOptimizer(config=config)
        t0 = time.perf_counter()
        optimizer.descending_sweep(
            h_values=np.array([1.5]),
            circuit=circuit,
            lattice=lattice,
            exact_data=[exact],
        )
        times.append(time.perf_counter() - t0)

    return BenchmarkResult(
        component="vqe",
        n_qubits=n_qubits,
        elapsed_s=float(np.median(times)),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def _bench_mpnn(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark MPNN forward pass (inference only)."""
    import torch

    from qmbp_simulation import HamiltonianBuilder, MPNNPredictor, make_lattice

    builder = HamiltonianBuilder()
    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.5)
    node_feat, edge_index = builder.build_graph_data(lattice)

    from torch_geometric.data import Data

    graph = Data(
        x=torch.tensor(node_feat, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
    model.eval()

    # Warmup
    with torch.no_grad():
        model(graph)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(graph)
        times.append(time.perf_counter() - t0)

    return BenchmarkResult(
        component="mpnn",
        n_qubits=n_qubits,
        elapsed_s=float(np.median(times)),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


_BENCH_FUNCTIONS: dict[str, Any] = {
    "solver": _bench_solver,
    "circuit": _bench_circuit,
    "vqe": _bench_vqe,
    "mpnn": _bench_mpnn,
}
