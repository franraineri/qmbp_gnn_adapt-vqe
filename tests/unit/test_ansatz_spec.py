"""Tests for the portable AnsatzSpec codec (complete, reusable ansatz definition).

Covers the round-trip contract: a spec captures system + structure + EXACT bond
edges + θ, serializes to JSON, reloads, and rebuilds the IDENTICAL circuit — the
portability guarantee that lets any runner select and reuse an ansatz.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.circuits.ansatz_spec import (
    AnsatzSpec,
    decode_ansatz_spec,
    encode_ansatz_spec,
)
from qmbp_simulation.circuits.bond_mask import BondSelection
from qmbp_simulation.circuits.hva_variants import make_masked_variant


def _masked_variant():
    nn = [(0, 4), (1, 5), (2, 3)]
    nnn = [(1, 6), (6, 9)]
    sel = BondSelection(nn_edges=nn, nnn_edges=nnn, provenance="prune_by_theta(tol=0.3)")
    v = make_masked_variant("demo_prune", "x", ["nn", "nnn", "x", "nn", "x"], sel, rx_final=True)
    return v, nn, nnn


class TestAnsatzSpecFromVariant:
    def test_captures_exact_edges_and_structure(self):
        v, nn, nnn = _masked_variant()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5, j2=0.5, theta=[0.1, 0.2, 0.3])
        assert spec.nn_edges == [list(e) for e in nn]
        assert spec.nnn_edges == [list(e) for e in nnn]
        assert spec.blocks == ["nn", "nnn", "x", "nn", "x"]
        assert spec.rx_final is True
        assert spec.selection_provenance == "prune_by_theta(tol=0.3)"
        assert spec.theta == [0.1, 0.2, 0.3]

    def test_metrics_routed_to_fields_and_extra(self):
        v, _, _ = _masked_variant()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5, fidelity=0.98, n_2q=36, custom="z")
        assert spec.fidelity == 0.98
        assert spec.n_2q == 36
        assert spec.metrics["custom"] == "z"  # unknown kwargs land in metrics

    def test_full_variant_records_passed_edges(self):
        from qmbp_simulation.circuits.hva_variants import AnsatzVariant

        full = AnsatzVariant("p1_base", "x", blocks=["nn", "nnn", "x"])
        spec = AnsatzSpec.from_variant(
            full, topology="square", n_qubits=6, h=0.5, nn_edges=[(0, 1)], nnn_edges=[(0, 2)]
        )
        assert spec.selection_provenance == "full"
        assert spec.nn_edges == [[0, 1]] and spec.nnn_edges == [[0, 2]]


class TestAnsatzSpecRoundTrip:
    def test_dict_codec_roundtrip(self):
        v, _, _ = _masked_variant()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5, theta=[0.1, 0.2])
        spec2 = decode_ansatz_spec(encode_ansatz_spec(spec))
        assert spec2.to_dict() == spec.to_dict()
        assert spec2.schema == "ansatz_spec_v1"

    def test_save_load_roundtrip(self, tmp_path):
        v, _, _ = _masked_variant()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5, theta=[0.1, 0.2])
        p = spec.save(tmp_path / "demo.spec.json")
        assert p.exists()
        loaded = AnsatzSpec.load(p)
        assert loaded.to_dict() == spec.to_dict()

    def test_from_dict_ignores_unknown_keys(self):
        v, _, _ = _masked_variant()
        d = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5).to_dict()
        d["some_future_field"] = 123  # forward-compat
        spec = AnsatzSpec.from_dict(d)
        assert spec.name == "demo_prune"


class TestAnsatzSpecBuild:
    def test_rebuilds_identical_circuit(self):
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.circuits.hva_variants import build_variant
        from qmbp_simulation.models import make_lattice

        v, _, _ = _masked_variant()
        lat = make_lattice("square", 10, J=1.0, h=0.5)
        qc0, _ = build_variant(HVACircuitBuilder(), 10, lat, v)
        theta = np.random.default_rng(0).uniform(-1, 1, qc0.num_parameters)

        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5, theta=theta)
        qc1, theta1 = spec.build()
        assert qc1.num_parameters == qc0.num_parameters
        # same gate op counts → identical structure
        assert dict(qc1.count_ops()) == dict(qc0.count_ops())
        np.testing.assert_allclose(theta1, theta)

    def test_build_without_theta_returns_none(self):
        v, _, _ = _masked_variant()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=10, h=0.5)
        qc, theta = spec.build()
        assert qc.num_parameters > 0
        assert theta is None


class TestAnsatzSpecSymmetryTied:
    """A symmetry-tied variant round-trips as a tied spec and rebuilds the tied
    circuit (one param per orbit). N=20 square is the only completed lattice with
    a non-trivial point group (D2, order 4) — the regime the lever targets."""

    def _tied_variant_n20(self):
        from dataclasses import replace

        from qmbp_simulation.circuits.bond_mask import full_selection
        from qmbp_simulation.circuits.hva_variants import make_masked_variant
        from qmbp_simulation.circuits.symmetry import detect_symmetry_group
        from qmbp_simulation.models import make_lattice
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        lat = make_lattice("square", 20, J=1.0, h=0.5)
        nn = [(int(i), int(j)) for i, j in lat.edges]
        nnn = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
        g = detect_symmetry_group(nn, nnn, 20)
        v = make_masked_variant("tied20", "x", ["nn", "nnn", "x"], full_selection(nn, nnn))
        return replace(v, symmetry_group=g), nn, nnn

    def test_from_variant_captures_tied_metadata(self):
        v, nn, nnn = self._tied_variant_n20()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=20, h=0.5, j2=0.5)
        assert spec.symmetry is not None
        assert spec.symmetry["tied"] is True
        assert spec.symmetry["group_order"] == 4
        # D2 orbits at N=20: 10 nn, 13 nnn, 6 site → reduction (31+46+20)/29 ≈ 3.34
        assert spec.symmetry["nn_orbits"] == 10
        assert spec.symmetry["nnn_orbits"] == 13
        assert spec.symmetry["site_orbits"] == 6
        assert spec.symmetry["reduction_factor"] == pytest.approx(3.3448, abs=1e-3)

    def test_tied_spec_roundtrip_rebuilds_tied_circuit(self):
        v, _, _ = self._tied_variant_n20()
        spec = AnsatzSpec.from_variant(v, topology="square", n_qubits=20, h=0.5, j2=0.5, theta=[0.1] * 29)
        reloaded = AnsatzSpec.from_dict(spec.to_dict())  # JSON round-trip
        qc, theta = reloaded.build()
        # tied layout: 10 + 13 + 6 = 29 params (vs 97 independent), same 2q
        assert qc.num_parameters == 29
        assert len(theta) == 29
        n2q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        assert n2q == 77


class TestAnsatzSpecKitaev:
    """Kitaev bond-resolved ansatz (RXX/RYY per edge + RZ), initial_state=zero."""

    def _kitaev_lattice(self, n=6):
        from qmbp_simulation.models import make_lattice

        return make_lattice("chain_1d", n, J=1.0, h=0.5)

    def test_from_kitaev_captures_edges_and_kind(self):
        lat = self._kitaev_lattice(6)
        spec = AnsatzSpec.from_kitaev(
            name="kitaev_p1", topology="chain_1d", n_qubits=6, h=0.5, p_layers=1, lattice=lat, delta=0.3
        )
        assert spec.ansatz_kind == "kitaev_xxyy"
        assert spec.initial_state == "zero"
        assert spec.nn_edges == [list(e) for e in lat.edges]
        assert spec.nnn_edges == []
        assert spec.selection_provenance == "kitaev_all_edges"
        assert spec.p_layers == 1

    def test_build_matches_direct_builder(self):
        from qmbp_simulation.circuits import HVACircuitBuilder

        lat = self._kitaev_lattice(6)
        builder = HVACircuitBuilder()
        qc_direct, _ = builder.create_kitaev_bond_resolved(6, 1, lat, initial_state="zero")
        theta = np.random.default_rng(1).uniform(-1, 1, qc_direct.num_parameters)

        spec = AnsatzSpec.from_kitaev(
            name="kitaev_p1", topology="chain_1d", n_qubits=6, h=0.5, p_layers=1, lattice=lat, theta=theta, delta=0.3
        )
        qc_spec, theta_spec = spec.build(builder=builder, lattice=lat)

        assert qc_spec.num_parameters == qc_direct.num_parameters
        assert dict(qc_spec.count_ops()) == dict(qc_direct.count_ops())
        np.testing.assert_allclose(theta_spec, theta)

    def test_kitaev_roundtrip_preserves_kind_and_state(self, tmp_path):
        lat = self._kitaev_lattice(6)
        spec = AnsatzSpec.from_kitaev(
            name="kitaev_p1",
            topology="chain_1d",
            n_qubits=6,
            h=0.5,
            p_layers=1,
            lattice=lat,
            theta=[0.1, 0.2, 0.3],
            delta=0.3,
        )
        loaded = AnsatzSpec.load(spec.save(tmp_path / "kitaev.spec.json"))
        assert loaded.to_dict() == spec.to_dict()
        assert loaded.ansatz_kind == "kitaev_xxyy"
        assert loaded.initial_state == "zero"
