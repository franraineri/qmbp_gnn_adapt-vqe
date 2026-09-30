"""Ansatz structure variants for the bond-resolved frustrated HVA (exploratory).

Declarative registry of p=1-based structure variants used to study which
modification recovers the expressivity that a plain extra layer (p=2, p=3) buys,
and at what 2-qubit-gate cost. Each variant is a block sequence for
:meth:`HVACircuitBuilder.create_bond_resolved_frustrated_configurable` plus
metadata (description, theoretical 2q cost) so a results file explains itself.

Variants (all p=1-based unless noted):
- ``p1_base``       : one standard frustrated layer [nn, nnn, x]        (baseline)
- ``p1_half_nn``    : p=1 + a half-layer of only nn RZZ  (Option A_nn, +nn CX)
- ``p1_half_nnn``   : p=1 + a half-layer of only nnn RZZ (Option A_nnn, +nnn CX)
- ``p1_rx_extra``   : p=1 + a trailing RX block          (Option B, +0 CX)
- ``p1_rz_extra``   : p=1 + a trailing RZ block          (Option C, +0 CX)
- ``p1_interleaved``: p=1 with nn/nnn interleaved per edge-pair  (Option D, same CX)
- ``p2_base``       : two standard layers   (anchor)
- ``p3_base``       : three standard layers (anchor)

2q cost is reported both theoretically (here, ``cx_per_rzz=2``) and, at runtime,
measured on the transpiled circuit via
``analysis.circuit_visualizer.circuit_summary`` / ``transpiled_circuit_stats``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnsatzVariant:
    """One ansatz-structure variant: block sequence + metadata.

    ``blocks``/``rx_final``/``rz_final`` are passed straight to
    ``create_bond_resolved_frustrated_configurable``. ``rzz_blocks`` counts the
    RZZ blocks (each costs ``2 * n_edges_of_that_type`` CX), used for the
    theoretical 2q estimate; the runtime measures the real transpiled count.
    """

    name: str
    description: str
    blocks: list[str]
    rx_final: bool = False
    rz_final: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    # When set, this variant is ``extends``'s block sequence followed by
    # ``extra_blocks`` — enabling the extend-from-base warm-start (seed the shared
    # prefix from the base variant's best known θ, tail analytically).
    extends: str | None = None
    extra_blocks: tuple[str, ...] = field(default_factory=tuple)

    def theoretical_cx(self, n_nn: int, n_nnn: int, cx_per_rzz: int = 2) -> int:
        """Theoretical 2q-gate count = cx_per_rzz × (#nn RZZ + #nnn RZZ)."""
        cx = 0
        for b in self.blocks:
            if b == "nn":
                cx += cx_per_rzz * n_nn
            elif b == "nnn":
                cx += cx_per_rzz * n_nnn
        return cx


# ── Registry ──────────────────────────────────────────────────────────────────
# Order matters for the results table (baseline → cheap variants → anchors).
VARIANTS: dict[str, AnsatzVariant] = {
    "p1_base": AnsatzVariant(
        "p1_base",
        "One standard frustrated layer [nn, nnn, x]",
        blocks=["nn", "nnn", "x"],
        tags=("baseline",),
    ),
    "p1_rx_extra": AnsatzVariant(
        "p1_rx_extra",
        "p=1 + trailing RX block (Option B, +0 CX)",
        blocks=["nn", "nnn", "x"],
        rx_final=True,
        tags=("zero_2q", "option_B"),
    ),
    "p1_rz_extra": AnsatzVariant(
        "p1_rz_extra",
        "p=1 + trailing RZ block, breaks Z2 (Option C, +0 CX)",
        blocks=["nn", "nnn", "x"],
        rz_final=True,
        tags=("zero_2q", "option_C"),
    ),
    "p1_half_nn": AnsatzVariant(
        "p1_half_nn",
        "p=1 + half-layer of only nn RZZ then x (Option A_nn)",
        blocks=["nn", "nnn", "x", "nn", "x"],
        tags=("intermediate_2q", "option_A"),
    ),
    "p1_half_nnn": AnsatzVariant(
        "p1_half_nnn",
        "p=1 + half-layer of only nnn RZZ then x (Option A_nnn)",
        blocks=["nn", "nnn", "x", "nnn", "x"],
        tags=("intermediate_2q", "option_A"),
    ),
    "p1_interleaved": AnsatzVariant(
        "p1_interleaved",
        "p=1 with nn/nnn order swapped (Option D, same CX)",
        blocks=["nnn", "nn", "x"],
        tags=("same_2q", "option_D"),
    ),
    "p1_half_nn_rx": AnsatzVariant(
        "p1_half_nn_rx",
        "p=1 + half-layer nn RZZ + trailing RX (entangle + free rotations)",
        blocks=["nn", "nnn", "x", "nn", "x"],
        rx_final=True,
        tags=("intermediate_2q",),
    ),
    "p2_base": AnsatzVariant(
        "p2_base",
        "Two standard frustrated layers (anchor)",
        blocks=["nn", "nnn", "x", "nn", "nnn", "x"],
        tags=("anchor",),
    ),
    # p=2 + a PARTIAL third entanglement layer. Motivated by the N=18 h=0.5
    # diagnosis: the deficit is balanced entanglement (RZZ), not local rotations,
    # and p2→p3 (a full extra layer, +132 CX at N=18) is overkill. These add only
    # ONE RZZ block on top of p2, targeting fidelity-per-2q between p2 and p3.
    "p2_half_nn": AnsatzVariant(
        "p2_half_nn",
        "p=2 + half-layer of only nn RZZ then x (partial 3rd layer)",
        blocks=["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"],
        tags=("intermediate_2q", "partial_p3"),
        extends="p2_base",
        extra_blocks=("nn", "x"),
    ),
    "p2_half_nnn": AnsatzVariant(
        "p2_half_nnn",
        "p=2 + half-layer of only nnn RZZ then x (partial 3rd layer)",
        blocks=["nn", "nnn", "x", "nn", "nnn", "x", "nnn", "x"],
        tags=("intermediate_2q", "partial_p3"),
        extends="p2_base",
        extra_blocks=("nnn", "x"),
    ),
    "p2_half_nn_rx": AnsatzVariant(
        "p2_half_nn_rx",
        "p=2 + half-layer nn RZZ then x + trailing RX (partial 3rd + free rot)",
        blocks=["nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"],
        rx_final=True,
        tags=("intermediate_2q", "partial_p3"),
        extends="p2_base",
        extra_blocks=("nn", "x"),
    ),
    "p3_base": AnsatzVariant(
        "p3_base",
        "Three standard frustrated layers (anchor)",
        blocks=["nn", "nnn", "x"] * 3,
        tags=("anchor",),
    ),
}


def build_variant(builder, n_qubits: int, lattice, variant: AnsatzVariant):
    """Build the circuit for a variant via the configurable engine."""
    return builder.create_bond_resolved_frustrated_configurable(
        n_qubits,
        lattice,
        blocks=variant.blocks,
        rx_final=variant.rx_final,
        rz_final=variant.rz_final,
    )
