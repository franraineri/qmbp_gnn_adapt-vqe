#!/usr/bin/env python3
"""Performance benchmarking CLI for qmbp_simulation.

Benchmarks key pipeline components at various system sizes to track
performance regressions and establish timing baselines.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --components vqe solver
    python scripts/benchmark.py --n-qubits 4 6 8
    python scripts/benchmark.py --output results/benchmarks/run.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    component: str
    n_qubits: int
    elapsed_s: float
    details: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for qmbp_simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                             # Run all benchmarks
    %(prog)s --components solver vqe     # Benchmark specific components
    %(prog)s --n-qubits 4 6 8 10        # Custom system sizes
    %(prog)s --output bench.json         # Save results to file
        """,
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["solver", "vqe", "circuit", "mpnn", "pipeline"],
        default=["solver", "vqe", "circuit", "mpnn"],
        help="Components to benchmark (default: solver vqe circuit mpnn)",
    )
    parser.add_argument(
        "--n-qubits",
        nargs="+",
        type=int,
        default=[4, 6, 8],
        help="System sizes to benchmark (default: 4 6 8)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeats per benchmark (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def bench_solver(n_qubits: int, n_repeats: int) -> BenchmarkResult:
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
        elapsed_s=np.median(times),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def bench_circuit(n_qubits: int, n_repeats: int) -> BenchmarkResult:
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
        elapsed_s=np.median(times),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def bench_vqe(n_qubits: int, n_repeats: int) -> BenchmarkResult:
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
        elapsed_s=np.median(times),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


def bench_mpnn(n_qubits: int, n_repeats: int) -> BenchmarkResult:
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
        elapsed_s=np.median(times),
        details={"min_s": min(times), "max_s": max(times), "repeats": n_repeats},
    )


BENCH_FUNCTIONS = {
    "solver": bench_solver,
    "circuit": bench_circuit,
    "vqe": bench_vqe,
    "mpnn": bench_mpnn,
}


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("BENCHMARK: qmbp_simulation")
    print("=" * 60)
    print(f"  Components: {', '.join(args.components)}")
    print(f"  System sizes: {args.n_qubits}")
    print(f"  Repeats: {args.n_repeats}")
    print()

    all_results: list[BenchmarkResult] = []

    for component in args.components:
        bench_fn = BENCH_FUNCTIONS.get(component)
        if bench_fn is None:
            print(f"  Skipping unknown component: {component}")
            continue

        print(f"  [{component}]")
        for n in args.n_qubits:
            try:
                result = bench_fn(n, args.n_repeats)
                all_results.append(result)
                print(f"    N={n:2d}: {result.elapsed_s * 1000:8.2f} ms (median)")
            except Exception as e:
                print(f"    N={n:2d}: ERROR — {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
        print()

    # Summary table
    print("-" * 60)
    print(f"{'Component':<12} {'N':<6} {'Time (ms)':<12} {'Notes'}")
    print("-" * 60)
    for r in all_results:
        print(f"{r.component:<12} {r.n_qubits:<6} {r.elapsed_s * 1000:<12.2f}")
    print("-" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": [asdict(r) for r in all_results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
