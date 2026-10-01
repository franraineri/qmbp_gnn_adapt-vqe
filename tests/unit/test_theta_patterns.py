"""Unit tests for the optimized-θ physical analyzer (analysis.theta_patterns)."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis.theta_patterns import (
    analyze_theta,
    block_statistics,
    canonicalize_z2,
    compare_to_warmstart,
    decompose_theta,
    effective_couplings,
    layer_size,
)
from qmbp_simulation.analysis.warmstart import first_order_warmstart_theta

N_NN, N_NNN, N_QUBITS, P = 10, 11, 8, 2
PER = N_NN + N_NNN + N_QUBITS


class TestDecompose:
    def test_layer_size(self):
        assert layer_size(N_NN, N_NNN, N_QUBITS) == PER

    def test_shapes(self):
        theta = np.arange(PER * P, dtype=float)
        layers = decompose_theta(theta, N_NN, N_NNN, N_QUBITS, P)
        assert len(layers) == P
        assert layers[0]["nn"].shape == (N_NN,)
        assert layers[0]["nnn"].shape == (N_NNN,)
        assert layers[0]["x"].shape == (N_QUBITS,)

    def test_ordering_preserved(self):
        theta = np.arange(PER, dtype=float)
        lay = decompose_theta(theta, N_NN, N_NNN, N_QUBITS, 1)[0]
        np.testing.assert_array_equal(lay["nn"], np.arange(N_NN))
        np.testing.assert_array_equal(lay["nnn"], np.arange(N_NN, N_NN + N_NNN))
        np.testing.assert_array_equal(lay["x"], np.arange(N_NN + N_NNN, PER))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            decompose_theta(np.zeros(5), N_NN, N_NNN, N_QUBITS, P)


class TestBlockStats:
    def test_uniform_block_is_homogeneous(self):
        theta = np.zeros(PER)
        theta[:N_NN] = -0.3  # uniform NN block
        stats = block_statistics(decompose_theta(theta, N_NN, N_NNN, N_QUBITS, 1))
        assert stats[0]["nn"]["std"] == pytest.approx(0.0, abs=1e-12)
        assert stats[0]["nn"]["homogeneity"] == pytest.approx(1.0)

    def test_dispersed_block_lower_homogeneity(self):
        rng = np.random.default_rng(0)
        theta = np.zeros(PER)
        theta[:N_NN] = -0.3 + rng.normal(0, 0.2, N_NN)
        stats = block_statistics(decompose_theta(theta, N_NN, N_NNN, N_QUBITS, 1))
        assert stats[0]["nn"]["homogeneity"] < 1.0


class TestZ2:
    def test_flip_zz_blocks_when_positive(self):
        # NN block summing positive → canonical flips ZZ blocks negative.
        theta = np.zeros(PER)
        theta[:N_NN] = 0.2                       # positive NN sum
        theta[N_NN:N_NN + N_NNN] = 0.1
        theta[N_NN + N_NNN:] = 0.5               # θ_x untouched by Z2
        c = canonicalize_z2(theta, N_NN, N_NNN, N_QUBITS, 1)
        lay = decompose_theta(c, N_NN, N_NNN, N_QUBITS, 1)[0]
        assert lay["nn"].sum() <= 0
        np.testing.assert_allclose(lay["x"], 0.5)  # x unchanged

    def test_already_canonical_unchanged(self):
        theta = np.zeros(PER)
        theta[:N_NN] = -0.2
        c = canonicalize_z2(theta, N_NN, N_NNN, N_QUBITS, 1)
        np.testing.assert_allclose(c, theta)


class TestWarmStartComparison:
    def test_warmstart_matches_itself(self):
        ws = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.5, J=1.0, J2=0.5)
        cmp = compare_to_warmstart(ws, N_NN, N_NNN, N_QUBITS, P, 0.5,
                                   J=1.0, J2=0.5, order="first")
        assert cmp["l2_total"] == pytest.approx(0.0, abs=1e-9)

    def test_z2_flipped_optimum_is_close(self):
        ws = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, 0.5, J=1.0, J2=0.5)
        flipped = ws.copy()
        # flip ZZ blocks in both layers (a Z2 relabeling) — should stay close
        for layer in range(P):
            o = layer * PER
            flipped[o:o + N_NN + N_NNN] *= -1
        cmp = compare_to_warmstart(flipped, N_NN, N_NNN, N_QUBITS, P, 0.5,
                                   J=1.0, J2=0.5, order="first")
        assert cmp["l2_total"] == pytest.approx(0.0, abs=1e-9)


class TestEffectiveCouplings:
    def test_recovers_input_couplings_from_warmstart(self):
        h, J, J2 = 0.8, 1.0, 0.5
        ws = first_order_warmstart_theta(N_NN, N_NNN, N_QUBITS, P, h, J=J, J2=J2)
        layers = decompose_theta(ws, N_NN, N_NNN, N_QUBITS, P)
        eff = effective_couplings(layers, h, J=J)
        # J_eff = -4h·mean(θ_nn); warm-start θ_nn = -J/(4h) → J_eff = J
        assert eff["J_nn_eff"] == pytest.approx(J, abs=1e-9)
        assert eff["J_nnn_eff"] == pytest.approx(J2, abs=1e-9)
        assert eff["theta_x_over_pred"] == pytest.approx(1.0, abs=1e-9)


class TestAnalyzeTheta:
    def test_full_analysis_json_safe(self):
        import json
        rng = np.random.default_rng(1)
        theta = rng.normal(0, 0.3, PER * P)
        res = analyze_theta(theta, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS,
                            p_layers=P, h=0.5, J=1.0, J2=0.5)
        json.dumps(res)  # must be serializable
        assert res["layer_drift"] is not None  # p=2
        assert set(res["block_stats"][0]) == {"nn", "nnn", "x"}

    def test_p1_has_no_layer_drift(self):
        theta = np.zeros(PER)
        res = analyze_theta(theta, n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS,
                            p_layers=1, h=1.0, J=1.0, J2=0.5)
        assert res["layer_drift"] is None


class TestCompressedCircuitComparison:
    """decompose_spec_theta + compare_compressed_circuits (gate-level analysis)."""

    def test_decompose_spec_theta_roles_and_edges(self):
        from qmbp_simulation.analysis.theta_patterns import decompose_spec_theta

        # blocks nn,nnn,x with 2 nn edges, 1 nnn edge, 3 qubits, rx_final
        theta = [0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2]
        nn_edges = [(0, 1), (1, 2)]
        nnn_edges = [(0, 2)]
        gates = decompose_spec_theta(theta, ["nn", "nnn", "x"], nn_edges, nnn_edges, 3,
                                     rx_final=True)
        assert len(gates) == 2 + 1 + 3 + 3  # nn + nnn + x + rx_final
        nn = [g for g in gates if g["role"] == "nn"]
        assert nn[0]["key"] == (0, 1) and nn[0]["angle"] == pytest.approx(0.1)
        assert nn[1]["key"] == (1, 2) and nn[1]["angle"] == pytest.approx(0.2)
        nnn = [g for g in gates if g["role"] == "nnn"]
        assert nnn[0]["key"] == (0, 2) and nnn[0]["angle"] == pytest.approx(0.3)

    def test_decompose_spec_theta_wrong_length_raises(self):
        from qmbp_simulation.analysis.theta_patterns import decompose_spec_theta

        with pytest.raises(ValueError):
            decompose_spec_theta([0.1, 0.2], ["nn", "x"], [(0, 1)], [], 3)

    def test_compare_detects_invariant_shared_gates(self):
        from qmbp_simulation.analysis.theta_patterns import compare_compressed_circuits

        # two circuits sharing nn edges with (near) identical angles → invariant
        base = {"blocks": ["nn", "x"], "nn_edges": [(0, 1), (1, 2)], "nnn_edges": [],
                "n_qubits": 3, "rx_final": False, "fidelity": 0.99}
        a = dict(base, name="A", theta=[-0.30, -0.20, 0.5, 0.5, 0.5])
        b = dict(base, name="B", theta=[-0.31, -0.19, 0.5, 0.5, 0.5])
        res = compare_compressed_circuits([a, b], min_fidelity=0.9)
        assert res["n_circuits"] == 2
        assert res["per_role"]["nn"]["frac_invariant"] == pytest.approx(1.0)
        assert res["per_role"]["nn"]["mean_dispersion"] < 0.15

    def test_compare_needs_two_specs(self):
        from qmbp_simulation.analysis.theta_patterns import compare_compressed_circuits

        one = {"name": "A", "blocks": ["nn"], "nn_edges": [(0, 1)], "nnn_edges": [],
               "n_qubits": 2, "fidelity": 0.99, "theta": [-0.3]}
        res = compare_compressed_circuits([one], min_fidelity=0.9)
        assert "error" in res

    def test_compare_z2_gauge_aligned(self):
        from qmbp_simulation.analysis.theta_patterns import compare_compressed_circuits

        # same circuit but Z2-flipped ZZ sign → should still read as invariant
        base = {"blocks": ["nn", "x"], "nn_edges": [(0, 1), (1, 2)], "nnn_edges": [],
                "n_qubits": 3, "rx_final": False, "fidelity": 0.99}
        a = dict(base, name="A", theta=[-0.30, -0.20, 0.5, 0.5, 0.5])
        b = dict(base, name="B", theta=[0.30, 0.20, 0.5, 0.5, 0.5])  # ZZ flipped
        res = compare_compressed_circuits([a, b], min_fidelity=0.9)
        assert res["per_role"]["nn"]["frac_invariant"] == pytest.approx(1.0)


class TestAngleSpectrum:
    """angle_spectrum_by_role: N-comparable sorted |θ| fingerprint per role."""

    def test_spectrum_sorted_descending_and_counts(self):
        from qmbp_simulation.analysis.theta_patterns import angle_spectrum_by_role

        # blocks nn,x : 3 nn edges, 2 qubits
        theta = [-0.5, -0.1, -0.3, 1.2, 1.2]
        spec = angle_spectrum_by_role(theta, ["nn", "x"], [(0, 1), (1, 2), (0, 2)], [], 2)
        nn = spec["nn"]["spectrum"]
        assert list(nn) == sorted(nn, reverse=True)   # descending
        assert nn[0] == pytest.approx(0.5)
        assert spec["nn"]["n_total"] == 3
        assert spec["x"]["mean"] == pytest.approx(1.2)

    def test_n_active_counts_above_tenth_of_max(self):
        from qmbp_simulation.analysis.theta_patterns import angle_spectrum_by_role

        # one big nn, two near-zero → only 1 active
        theta = [-0.8, -0.02, -0.01, 1.0, 1.0]
        spec = angle_spectrum_by_role(theta, ["nn", "x"], [(0, 1), (1, 2), (0, 2)], [], 2)
        assert spec["nn"]["n_active"] == 1
