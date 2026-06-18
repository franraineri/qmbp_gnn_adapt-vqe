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

    AVAILABLE_COMPONENTS = ("solver", "circuit", "vqe", "mpnn", "layout", "aqc", "mitiq")

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


def _bench_layout(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark layout optimization: VF2 (mapomatic) vs BFS quality and timing.

    Measures TWO dimensions for fair comparison:
    1. Time: search + scoring (VF2) vs search only (BFS)
    2. Quality: CES post-transpilation, SWAP count, depth_2q

    The quality metrics are critical because VF2 is slightly slower in
    pure search time, but produces dramatically better layouts (6× lower CES,
    zero SWAPs). A time-only comparison misses the value.

    Both methods are run through full transpilation to measure real CES,
    ensuring an apples-to-apples quality comparison.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qmbp_simulation import HVACircuitBuilder, make_lattice
    from qmbp_simulation.execution.hardware.layout_optimizer import (
        MAPOMATIC_AVAILABLE,
        build_filtered_coupling_map,
        compute_layout_fidelity_cost,
        find_vf2_layouts,
    )
    from qmbp_simulation.execution.noisy_utils import (
        build_adjacency,
        compute_circuit_ces,
        find_layouts_bfs,
    )

    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        backend = FakeTorino()
    except ImportError:
        return BenchmarkResult(
            component="layout",
            n_qubits=n_qubits,
            elapsed_s=0.0,
            details={"skipped": True, "reason": "FakeTorino not available"},
        )

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.5)
    qc, _ = HVACircuitBuilder().create(n_qubits, 1, lattice)
    theta = np.random.uniform(-0.5, 0.5, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    # ── Warmup (exclude from timing) ──
    # First transpiler call loads Rust extensions (~100ms); exclude from benchmark
    pm_warmup = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pm_warmup.run(bound)

    # ── VF2 benchmark (if available) ──
    vf2_times: list[float] = []
    vf2_ces_values: list[float] = []
    vf2_n_2q_gates: list[int] = []
    vf2_n_layouts_found = 0
    depth_2q_vf2: int = 0

    if MAPOMATIC_AVAILABLE:
        import mapomatic as mm

        # Warmup VF2
        filtered_cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)
        deflated = mm.deflate_circuit(bound)
        find_vf2_layouts(deflated, filtered_cmap, max_layouts=10)

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            filtered_cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)
            deflated = mm.deflate_circuit(bound)
            layouts = find_vf2_layouts(deflated, filtered_cmap, max_layouts=100)
            if layouts:
                scored = compute_layout_fidelity_cost(deflated, layouts[:50], backend)
                best_layout = scored[0][0]
            else:
                best_layout = None
            vf2_times.append(time.perf_counter() - t0)
            vf2_n_layouts_found = len(layouts)

        # Quality measurement (outside timing loop for accuracy)
        if layouts:
            best_layout = compute_layout_fidelity_cost(deflated, layouts[:50], backend)[0][0]
            pm = generate_preset_pass_manager(
                optimization_level=2, backend=backend, initial_layout=best_layout
            )
            transpiled_vf2 = pm.run(bound)
            ces_vf2, _ = compute_circuit_ces(transpiled_vf2, backend)
            n_2q_vf2 = sum(1 for inst in transpiled_vf2.data if inst.operation.num_qubits == 2)
            depth_2q_vf2 = transpiled_vf2.depth(
                filter_function=lambda x: x.operation.num_qubits == 2
            )
            vf2_ces_values.append(ces_vf2)
            vf2_n_2q_gates.append(n_2q_vf2)

    # ── BFS benchmark (always) ──
    bfs_times: list[float] = []
    bfs_ces_values: list[float] = []
    bfs_n_2q_gates: list[int] = []
    bfs_n_candidates = 0
    depth_2q_bfs: int = 0

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(adj, n_qubits, n_candidates=40, seed=42)
        bfs_times.append(time.perf_counter() - t0)
        bfs_n_candidates = len(candidates)

    # BFS quality (transpile best candidate)
    if candidates:
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, initial_layout=candidates[0]
        )
        transpiled_bfs = pm.run(bound)
        ces_bfs, _ = compute_circuit_ces(transpiled_bfs, backend)
        n_2q_bfs = sum(1 for inst in transpiled_bfs.data if inst.operation.num_qubits == 2)
        depth_2q_bfs = transpiled_bfs.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        bfs_ces_values.append(ces_bfs)
        bfs_n_2q_gates.append(n_2q_bfs)

    # ── Assemble results ──
    primary_times = vf2_times if vf2_times else bfs_times
    median_time = float(np.median(primary_times))

    details: dict[str, Any] = {
        "min_s": min(primary_times),
        "max_s": max(primary_times),
        "repeats": n_repeats,
        "method": "vf2" if vf2_times else "bfs",
        "mapomatic_available": MAPOMATIC_AVAILABLE,
        "warmed_up": True,
    }

    # VF2 metrics
    if vf2_times:
        details["vf2_median_s"] = float(np.median(vf2_times))
        details["vf2_n_layouts_found"] = vf2_n_layouts_found
        if vf2_ces_values:
            details["vf2_best_ces"] = round(vf2_ces_values[0], 5)
        if vf2_n_2q_gates:
            details["vf2_n_2q_gates"] = vf2_n_2q_gates[0]
            details["vf2_swaps_added"] = vf2_n_2q_gates[0] - (n_qubits - 1)
            details["vf2_depth_2q"] = depth_2q_vf2

    # BFS metrics
    details["bfs_median_s"] = float(np.median(bfs_times))
    details["bfs_n_candidates"] = bfs_n_candidates
    if bfs_ces_values:
        details["bfs_best_ces"] = round(bfs_ces_values[0], 5)
    if bfs_n_2q_gates:
        details["bfs_n_2q_gates"] = bfs_n_2q_gates[0]
        details["bfs_swaps_added"] = bfs_n_2q_gates[0] - (n_qubits - 1)
        details["bfs_depth_2q"] = depth_2q_bfs

    # Comparative ratios
    if vf2_ces_values and bfs_ces_values:
        details["ces_improvement_ratio"] = round(bfs_ces_values[0] / vf2_ces_values[0], 2)
    if vf2_n_2q_gates and bfs_n_2q_gates:
        details["swap_reduction"] = bfs_n_2q_gates[0] - vf2_n_2q_gates[0]

    return BenchmarkResult(
        component="layout",
        n_qubits=n_qubits,
        elapsed_s=median_time,
        details=details,
    )


def _bench_aqc(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark AQC-Tensor circuit compression (full pipeline).

    Measures end-to-end time for: build p=2 circuit → bind θ_opt → MPS target →
    generate ansatz → L-BFGS-B optimization → compressed circuit.

    Requires qiskit-addon-aqc-tensor. Skips gracefully if not installed.
    Uses chain_1d topology at h=3.5 (deep paramagnetic, fast MPS).
    """
    try:
        from qmbp_simulation.circuits.aqc_compression import (
            AQCCircuitCompressor,
            AQCCompressionConfig,
        )
    except ImportError:
        return BenchmarkResult(
            component="aqc",
            n_qubits=n_qubits,
            elapsed_s=0.0,
            details={"error": "qiskit-addon-aqc-tensor not installed", "repeats": 0},
        )

    from qmbp_simulation import HVACircuitBuilder, VQEOptimizer, make_lattice
    from qmbp_simulation.models import VQEConfig
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=3.5)
    builder = HamiltonianBuilder()
    H = builder.build(lattice)
    hva = HVACircuitBuilder()
    circuit_p2, _ = hva.create(n_qubits, 2, lattice)

    # One-time setup: get θ_opt via quick VQE (not included in timing)
    vqe_config = VQEConfig(n_restarts=1, maxiter=200)
    optimizer = VQEOptimizer(config=vqe_config, seed=42)
    rng = np.random.default_rng(42)
    init = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
    vqe_result = optimizer.optimize(H, circuit_p2, init)
    target_circuit = circuit_p2.assign_parameters(vqe_result.theta_opt)

    # Benchmark the compression (the part that runs pre-QPU)
    config = AQCCompressionConfig(max_bond_dim=64, max_iterations=100)
    compressor = AQCCircuitCompressor(config)

    # Warmup (JAX JIT compilation on first call)
    compressor.compress_circuit(target_circuit, lattice)

    times = []
    fidelities = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = compressor.compress_circuit(target_circuit, lattice)
        times.append(time.perf_counter() - t0)
        fidelities.append(result.fidelity)

    return BenchmarkResult(
        component="aqc",
        n_qubits=n_qubits,
        elapsed_s=float(np.median(times)),
        details={
            "min_s": min(times),
            "max_s": max(times),
            "repeats": n_repeats,
            "bond_dim": 64,
            "p_source": 2,
            "mean_fidelity": float(np.mean(fidelities)),
            "n_2q_compressed": result.n_2q_compressed,
            "n_2q_original": result.n_2q_original,
        },
    )


def _bench_mitiq(n_qubits: int, n_repeats: int) -> BenchmarkResult:
    """Benchmark Mitiq error mitigation methods: ZNE, CDR, DDD+ZNE.

    Measures per-method execution time and accuracy on a TFIM circuit
    with depolarizing noise. Requires mitiq and qiskit-aer.

    Returns timing breakdown per method and ΔE/gap improvement.
    """
    try:
        from qmbp_simulation.execution.mitiq_utils import (
            is_mitiq_available,
            make_noiseless_executor,
            run_mitiq_cdr,
            run_mitiq_ddd_zne,
            run_mitiq_zne,
        )

        if not is_mitiq_available():
            return BenchmarkResult(
                component="mitiq",
                n_qubits=n_qubits,
                elapsed_s=0.0,
                details={"error": "mitiq not installed", "skipped": True},
            )
    except ImportError:
        return BenchmarkResult(
            component="mitiq",
            n_qubits=n_qubits,
            elapsed_s=0.0,
            details={"error": "mitiq_utils import failed", "skipped": True},
        )

    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        return BenchmarkResult(
            component="mitiq",
            n_qubits=n_qubits,
            elapsed_s=0.0,
            details={"error": "qiskit-aer not installed", "skipped": True},
        )

    from qiskit.circuit import QuantumCircuit

    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=2.0)
    H = HamiltonianBuilder().build(lattice)

    # Build a Mitiq-compatible circuit (uses only gates Mitiq can fold cleanly).
    # Our HVA uses cx-rz-cx for ZZ, which Mitiq's Cirq converter handles.
    # Use explicit H+CX+Rz+Rx structure that Mitiq processes correctly.
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(0.3, i + 1)
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.rx(0.5, i)
    bound = qc

    # Noisy backend
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    backend = AerSimulator(noise_model=nm)
    config = NoisyEstimatorConfig(shots=2048, seed_simulator=42)

    # Get exact for ΔE/gap
    exact = make_noiseless_executor(H)(bound)

    timings: dict[str, list[float]] = {"zne": [], "cdr": [], "ddd_zne": []}
    de_gaps: dict[str, float] = {}

    for _ in range(n_repeats):
        # ZNE
        t0 = time.time()
        r_zne = run_mitiq_zne(
            bound,
            H,
            backend,
            config,
            scale_factors=(1.0, 1.5, 2.0, 3.0),
            factory_name="linear",
            folding_method="global",
        )
        timings["zne"].append(time.time() - t0)

        # CDR
        t0 = time.time()
        r_cdr = run_mitiq_cdr(bound, H, backend, config, n_training_circuits=5)
        timings["cdr"].append(time.time() - t0)

        # DDD+ZNE
        t0 = time.time()
        r_ddd = run_mitiq_ddd_zne(
            bound,
            H,
            backend,
            config,
            ddd_rule="xx",
            scale_factors=(1.0, 1.5, 2.0),
        )
        timings["ddd_zne"].append(time.time() - t0)

    # Compute ΔE/gap for the last run
    gap = 0.5  # approximate
    de_gaps["zne"] = abs(r_zne.extrapolated_value - exact) / gap
    de_gaps["cdr"] = abs(r_cdr.mitigated_value - exact) / gap
    de_gaps["ddd_zne"] = abs(r_ddd.extrapolated_value - exact) / gap

    median_total = float(
        np.median(
            [sum(x) for x in zip(timings["zne"], timings["cdr"], timings["ddd_zne"], strict=False)]
        )
    )

    return BenchmarkResult(
        component="mitiq",
        n_qubits=n_qubits,
        elapsed_s=round(median_total, 3),
        details={
            "timings_median_s": {k: round(float(np.median(v)), 3) for k, v in timings.items()},
            "de_gaps": {k: round(v, 4) for k, v in de_gaps.items()},
            "n_repeats": n_repeats,
            "best_method": min(de_gaps, key=lambda k: de_gaps[k]),
        },
    )


_BENCH_FUNCTIONS: dict[str, Any] = {
    "solver": _bench_solver,
    "circuit": _bench_circuit,
    "vqe": _bench_vqe,
    "mpnn": _bench_mpnn,
    "layout": _bench_layout,
    "aqc": _bench_aqc,
    "mitiq": _bench_mitiq,
}
