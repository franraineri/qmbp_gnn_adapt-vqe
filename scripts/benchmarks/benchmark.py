#!/usr/bin/env python3
"""Performance benchmarking CLI for qmbp_simulation.

Thin CLI wrapper around BenchmarkSuite from the framework.
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
from pathlib import Path


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
        choices=["solver", "vqe", "circuit", "mpnn", "layout", "aqc", "mitiq"],
        default=None,
        help="Components to benchmark (default: all)",
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


def main() -> None:
    args = parse_args()

    from qmbp_simulation.framework import BenchmarkSuite
    from qmbp_simulation.framework.result_io import save_benchmark_result

    print("=" * 60)
    print("BENCHMARK: qmbp_simulation")
    print("=" * 60)
    print(f"  Components: {args.components or 'all'}")
    print(f"  System sizes: {args.n_qubits}")
    print(f"  Repeats: {args.n_repeats}")
    print()

    suite = BenchmarkSuite(
        n_qubits=args.n_qubits,
        n_repeats=args.n_repeats,
        verbose=True,
    )
    results = suite.run(components=args.components)

    # Summary table
    suite.print_summary(results)

    # Save results
    if args.output:
        data = suite.to_dict(results)
        path = save_benchmark_result(data, output_path=Path(args.output))
        print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
