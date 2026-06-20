"""Quick integration test for noisy_utils module."""

import numpy as np

from qmbp_simulation.execution import (
    LayoutSelection,
    NoisyEstimatorConfig,
    ZNEResult,
    build_adjacency,
    compute_circuit_ces,
    find_layouts_bfs,
    linear_zne,
    select_layouts_by_circuit_ces,
)


def test_noisy_estimator_config_precision():
    config = NoisyEstimatorConfig(shots=16384, seed_simulator=42)
    assert abs(config.precision - 1.0 / np.sqrt(16384)) < 1e-10


def test_linear_zne_perfect_linear():
    ces = np.array([0.1, 0.3, 0.5])
    vals = np.array([-5.0, -4.8, -4.6])  # slope=1.0, intercept=-5.1
    result = linear_zne(ces, vals)
    assert isinstance(result, ZNEResult)
    assert abs(result.extrapolated_value - (-5.1)) < 1e-10
    assert abs(result.r_squared - 1.0) < 1e-10
    assert abs(result.slope - 1.0) < 1e-10


def test_linear_zne_single_point():
    result = linear_zne(np.array([0.1]), np.array([-5.0]))
    assert result.r_squared == 0.0
    assert result.extrapolated_value == -5.0


def test_linear_zne_identical_ces():
    result = linear_zne(np.array([0.1, 0.1, 0.1]), np.array([-5.0, -4.9, -5.1]))
    assert result.r_squared == 0.0


def test_layout_selection_dataclass():
    ls = LayoutSelection(
        layouts=[[0, 1, 2], [3, 4, 5]],
        ces_values=[0.1, 0.3],
        transpiled_circuits=[],
    )
    assert len(ls.layouts) == 2
    assert ls.ces_values[1] == 0.3


def test_build_adjacency_and_find_layouts():
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    adj = build_adjacency(backend)
    assert len(adj) > 100

    layouts = find_layouts_bfs(adj, n_qubits=6, n_candidates=10, seed=42)
    assert len(layouts) >= 5
    assert all(len(layout) == 6 for layout in layouts)


def test_select_layouts_by_circuit_ces():
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    from qmbp_simulation import HVACircuitBuilder, make_lattice

    backend = FakeTorino()
    adj = build_adjacency(backend)
    layouts = find_layouts_bfs(adj, n_qubits=6, n_candidates=10, seed=42)

    hva = HVACircuitBuilder()
    lattice = make_lattice("chain_1d", 6, J=1.0, h=2.0)
    qc, _ = hva.create(6, 1, lattice)
    theta = np.random.default_rng(42).uniform(-0.1, 0.1, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    selection = select_layouts_by_circuit_ces(bound, backend, layouts, n_select=3)
    assert isinstance(selection, LayoutSelection)
    assert len(selection.layouts) == 3
    assert len(selection.ces_values) == 3
    assert len(selection.transpiled_circuits) == 3
    # CES should have diversity (min != max)
    assert min(selection.ces_values) < max(selection.ces_values)


def test_compute_circuit_ces():
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    from qmbp_simulation import HVACircuitBuilder, make_lattice

    backend = FakeTorino()
    adj = build_adjacency(backend)
    layouts = find_layouts_bfs(adj, n_qubits=6, n_candidates=5, seed=42)

    hva = HVACircuitBuilder()
    lattice = make_lattice("chain_1d", 6, J=1.0, h=2.0)
    qc, _ = hva.create(6, 1, lattice)
    theta = np.random.default_rng(42).uniform(-0.1, 0.1, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    selection = select_layouts_by_circuit_ces(bound, backend, layouts, n_select=2)
    ces_val, n_2q = compute_circuit_ces(selection.transpiled_circuits[0], backend)
    assert ces_val > 0
    assert n_2q > 0
