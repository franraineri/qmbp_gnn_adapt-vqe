"""Portable ansatz specification — the standard codec for saving/loading an
ansatz's COMPLETE definition so any runner can pick one and reuse it.

Motivation
----------
Study artifacts stored θ and bond COUNTS but not the exact edges, so an ansatz
could not be replicated without re-deriving the selection. ``AnsatzSpec`` closes
that gap: it is the single, self-contained, JSON-portable description of an
ansatz — system (topology/N/h/J2), structure (blocks + rx/rz finals), the exact
bond selection (nn/nnn edges), optional converged θ, and metrics/provenance.

Reuse-first
-----------
This is a thin codec ON TOP of the existing primitives — it does not
re-implement circuit construction or bond logic:
  - :class:`qmbp_simulation.circuits.bond_mask.BondSelection` (edge selection)
  - :func:`qmbp_simulation.circuits.hva_variants.make_masked_variant` /
    :func:`...build_variant` (circuit build)
  - :class:`...hva_variants.AnsatzVariant` (structure)

Round-trip
----------
    spec = AnsatzSpec.from_variant(variant, topology="square", n_qubits=10,
                                   h=0.5, j2=0.5, theta=theta, metrics={...})
    spec.save(path)                         # → JSON on disk
    spec2 = AnsatzSpec.load(path)           # ← exact reconstruction
    qc, theta = spec2.build()               # ready-to-run circuit (+ bound θ)

``build()`` reconstructs the EXACT circuit (same edges, blocks, finals) via the
existing masked/configurable builder, so a spec is portable across every runner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "ansatz_spec_v1"


def _as_edge_list(edges) -> list[list[int]]:
    """Normalize an edge iterable to a JSON-friendly list of [i, j] int pairs."""
    return [[int(i), int(j)] for i, j in (edges or [])]


@dataclass
class AnsatzSpec:
    """Complete, portable definition of one bond-resolved frustrated HVA ansatz.

    Everything needed to rebuild the exact circuit and (optionally) its converged
    state. ``nn_edges``/``nnn_edges`` are the bonds that carry an RZZ — the piece
    that was previously missing from artifacts. ``theta`` is the converged
    parameter vector (optional; a spec is still valid as a structure-only recipe).
    """

    # ── identity / system ──
    name: str
    topology: str
    n_qubits: int
    h: float
    j2: float = 0.5
    # ── structure (passed verbatim to the configurable/masked builder) ──
    blocks: list[str] = field(default_factory=list)
    rx_final: bool = False
    rz_final: bool = False
    # ── exact bond selection (the portability fix) ──
    nn_edges: list[list[int]] = field(default_factory=list)
    nnn_edges: list[list[int]] = field(default_factory=list)
    selection_provenance: str = "full"
    # ── optional converged state + metrics ──
    theta: list[float] | None = None
    fidelity: float | None = None
    n_2q: int | None = None
    n_params: int | None = None
    seed_kind: str | None = None
    base_variant: str | None = None
    # ── gate family: which builder reconstructs the circuit ──
    # "frustrated_zz" (default): bond-resolved RZZ via the masked builder (blocks).
    # "kitaev_xxyy":  RXX/RYY per edge + RZ via create_kitaev_bond_resolved.
    ansatz_kind: str = "frustrated_zz"
    initial_state: str = "plus"
    p_layers: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    # ── construction from existing objects ─────────────────────────────────
    @classmethod
    def from_variant(cls, variant, *, topology: str, n_qubits: int, h: float,
                     j2: float = 0.5, theta=None, nn_edges=None, nnn_edges=None,
                     **metrics) -> AnsatzSpec:
        """Build a spec from an :class:`AnsatzVariant` (+ system + optional θ).

        If the variant carries a ``bond_selection`` its edges are used; otherwise
        pass the full-lattice ``nn_edges``/``nnn_edges`` explicitly so the spec is
        self-contained (a full ansatz still records which edges it entangles).
        """
        sel = getattr(variant, "bond_selection", None)
        if sel is not None:
            nn = _as_edge_list(sel.nn_edges)
            nnn = _as_edge_list(sel.nnn_edges)
            prov = getattr(sel, "provenance", "full") or "full"
        else:
            nn = _as_edge_list(nn_edges)
            nnn = _as_edge_list(nnn_edges)
            prov = "full"
        known = {"fidelity", "n_2q", "n_params", "seed_kind", "base_variant"}
        return cls(
            name=variant.name, topology=topology, n_qubits=int(n_qubits),
            h=float(h), j2=float(j2), blocks=list(variant.blocks),
            rx_final=bool(variant.rx_final), rz_final=bool(variant.rz_final),
            nn_edges=nn, nnn_edges=nnn, selection_provenance=prov,
            theta=(list(map(float, theta)) if theta is not None else None),
            fidelity=metrics.get("fidelity"), n_2q=metrics.get("n_2q"),
            n_params=metrics.get("n_params"), seed_kind=metrics.get("seed_kind"),
            base_variant=metrics.get("base_variant", variant.name),
            metrics={k: v for k, v in metrics.items() if k not in known},
        )

    # ── reconstruction ─────────────────────────────────────────────────────
    def to_bond_selection(self):
        """The exact :class:`BondSelection` this spec entangles."""
        from qmbp_simulation.circuits.bond_mask import BondSelection

        return BondSelection(
            nn_edges=[tuple(e) for e in self.nn_edges],
            nnn_edges=[tuple(e) for e in self.nnn_edges],
            provenance=self.selection_provenance,
        )

    def to_variant(self):
        """Reconstruct the :class:`AnsatzVariant` (masked by the exact edges)."""
        from qmbp_simulation.circuits.hva_variants import make_masked_variant

        return make_masked_variant(
            self.name, f"{self.name} (from AnsatzSpec)", list(self.blocks),
            self.to_bond_selection(), rx_final=self.rx_final,
            rz_final=self.rz_final, tags=("spec",))

    def build(self, builder=None, lattice=None):
        """Rebuild the EXACT circuit. Returns ``(qc, theta)``.

        ``theta`` is the stored converged vector when present (ready to bind),
        else ``None`` (structure-only recipe). The lattice is created from the
        spec's system when not supplied, so a caller only needs the spec.
        Routes to the Kitaev bond-resolved builder when ``ansatz_kind`` says so.
        """
        import numpy as np

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models import make_lattice

        if lattice is None:
            lattice = make_lattice(self.topology, self.n_qubits, J=1.0, h=self.h)
        builder = builder or HVACircuitBuilder()

        if self.ansatz_kind == "kitaev_xxyy":
            p = self.p_layers or 1
            qc, _ = builder.create_kitaev_bond_resolved(
                self.n_qubits, p, lattice, initial_state=self.initial_state)
        else:
            from qmbp_simulation.circuits.hva_variants import build_variant

            qc, _ = build_variant(builder, self.n_qubits, lattice, self.to_variant())
        theta = (np.asarray(self.theta, float) if self.theta is not None else None)
        return qc, theta

    @classmethod
    def from_kitaev(cls, *, name, topology, n_qubits, h, p_layers, lattice,
                    theta=None, initial_state="zero", delta=1.0, **metrics) -> AnsatzSpec:
        """Spec for a Kitaev bond-resolved ansatz (RXX/RYY per edge + RZ).

        The Kitaev ansatz is not block-structured, so it records the lattice
        edges (all of them, since RXX+RYY act on every edge) and ``p_layers``;
        ``build()`` reconstructs it via ``create_kitaev_bond_resolved``.
        """
        edges = _as_edge_list(getattr(lattice, "edges", []))
        known = {"fidelity", "n_2q", "n_params", "seed_kind", "base_variant"}
        return cls(
            name=name, topology=topology, n_qubits=int(n_qubits), h=float(h),
            j2=float(delta), blocks=[], rx_final=False, rz_final=False,
            nn_edges=edges, nnn_edges=[], selection_provenance="kitaev_all_edges",
            theta=(list(map(float, theta)) if theta is not None else None),
            ansatz_kind="kitaev_xxyy", initial_state=initial_state,
            p_layers=int(p_layers),
            fidelity=metrics.get("fidelity"), n_2q=metrics.get("n_2q"),
            n_params=metrics.get("n_params"), seed_kind=metrics.get("seed_kind"),
            base_variant=metrics.get("base_variant", name),
            metrics={k: v for k, v in metrics.items() if k not in known},
        )

    # ── serialization ──────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (self-describing, carries ``schema``)."""
        d = asdict(self)
        d["schema"] = SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnsatzSpec:
        """Reconstruct from a dict, ignoring unknown keys (forward-compatible)."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str | Path) -> Path:
        """Write the spec as JSON (parent dirs created, atomic rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.rename(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> AnsatzSpec:
        """Load a spec previously written with :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── codec free functions (encode/decode) ───────────────────────────────────
def encode_ansatz_spec(spec: AnsatzSpec) -> dict[str, Any]:
    """Encode an :class:`AnsatzSpec` to a JSON-safe dict (the compressor)."""
    return spec.to_dict()


def decode_ansatz_spec(data: dict[str, Any]) -> AnsatzSpec:
    """Decode a dict back into an :class:`AnsatzSpec` (the decompressor)."""
    return AnsatzSpec.from_dict(data)
