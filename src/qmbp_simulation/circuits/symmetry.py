"""Lattice symmetry analysis for symmetry-tied ansätze — general & reusable.

The single source of truth for "what symmetries does this lattice have, and how
do they partition the bonds/sites into parameter-sharing orbits?". Built so a
symmetry-tied ansatz can share one parameter per orbit instead of one per bond,
shrinking the optimization space without changing the 2q gate count.

Design
------
A symmetry is a qubit permutation that maps the nn edge set onto itself AND the
nnn edge set onto itself — exactly the permutations that commute with the
frustrated Hamiltonian ``H = J·Σ ZZ_nn + J2·Σ ZZ_nnn − h·Σ X``. We find them as
AUTOMORPHISMS of the combined edge-typed graph (networkx VF2), which needs no
coordinate embedding — robust across the irregular qubit numbering make_lattice
produces at different N. (``nnn`` is the graph-distance-2 set, itself invariant
under any nn automorphism, so preserving nn implies preserving nnn; we tag both
anyway for a direct, self-documenting check.)

Everything here is pure (edges + integers in, dataclasses out) and persistable:
:class:`SymmetryReport` round-trips to JSON so a study can save the symmetry
metadata alongside its results for later analysis.

Public API
----------
- :func:`detect_symmetry_group`  — the valid symmetry group as qubit permutations
- :func:`bond_orbits` / :func:`site_orbits` — orbit partitions under a group
- :func:`orbit_index_map` — bond → orbit-id (the tying contract a builder consumes)
- :func:`ground_state_invariance` — |⟨ψ|P|ψ⟩| per symmetry (the physical check)
- :func:`analyze_symmetry` — one call → a full :class:`SymmetryReport`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

Edge = tuple[int, int]
Perm = list[int]

SCHEMA = "symmetry_report_v1"


# ── data contracts ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LatticeSymmetryGroup:
    """The symmetry group of a lattice as qubit permutations.

    ``perms`` are the NON-identity symmetries (each a list where ``perm[q]`` is the
    image of qubit ``q``). The full group order is ``len(perms) + 1`` (identity is
    implicit). ``names`` labels them (generic ``sym1..symK`` unless a caller
    supplies geometric labels). ``provenance`` records how they were derived.
    """

    n_qubits: int
    perms: list[Perm] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    provenance: str = "graph_automorphism(nn+nnn)"

    @property
    def order(self) -> int:
        """Group order including the identity."""
        return len(self.perms) + 1

    @property
    def is_trivial(self) -> bool:
        return len(self.perms) == 0


@dataclass
class SymmetryReport:
    """Persistable summary of a lattice symmetry analysis (JSON round-trips).

    Carries the group order, orbit counts per role (nn/nnn/site), the resulting
    per-layer parameter reduction, and — when computed — the exact ground-state
    invariance under each symmetry (the physical confirmation that tying is valid).
    """

    topology: str
    n_qubits: int
    h: float
    j2: float
    n_nn: int
    n_nnn: int
    group_order: int
    nn_orbits: int
    nnn_orbits: int
    site_orbits: int
    params_per_layer_independent: int
    params_per_layer_tied: int
    reduction_factor: float
    nn_orbit_sizes: dict[int, int] = field(default_factory=dict)
    nnn_orbit_sizes: dict[int, int] = field(default_factory=dict)
    gs_invariance: dict[str, float] | None = None
    manifold_dim: int | None = None
    provenance: str = "graph_automorphism(nn+nnn)"
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON keys must be strings
        d["nn_orbit_sizes"] = {str(k): v for k, v in self.nn_orbit_sizes.items()}
        d["nnn_orbit_sizes"] = {str(k): v for k, v in self.nnn_orbit_sizes.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymmetryReport:
        fields = cls.__dataclass_fields__
        dd = {k: v for k, v in d.items() if k in fields}
        for key in ("nn_orbit_sizes", "nnn_orbit_sizes"):
            if key in dd and dd[key]:
                dd[key] = {int(k): v for k, v in dd[key].items()}
        return cls(**dd)


# ── helpers ──────────────────────────────────────────────────────────────────
def _canon(edge: Edge) -> Edge:
    i, j = int(edge[0]), int(edge[1])
    return (i, j) if i <= j else (j, i)


def _edge_set(edges) -> set[Edge]:
    return {_canon(e) for e in edges}


def _image_edge(edge: Edge, perm: Perm) -> Edge:
    return _canon((perm[edge[0]], perm[edge[1]]))


# ── symmetry detection (graph automorphisms) ─────────────────────────────────
def detect_symmetry_group(
    nn_edges,
    nnn_edges,
    n_qubits: int,
    *,
    max_syms: int = 16,
) -> LatticeSymmetryGroup:
    """The valid symmetry group as qubit permutations, via graph automorphisms.

    A permutation is a symmetry iff it maps the nn graph onto itself AND the nnn
    graph onto itself. We find these as automorphisms of the combined graph whose
    edges are tagged ``nn`` / ``nnn`` (VF2 with a categorical edge match), so both
    edge sets are preserved simultaneously. The identity is excluded from the
    returned perms (it is implicit; see :attr:`LatticeSymmetryGroup.order`).

    ``max_syms`` caps the enumeration defensively (the largest 2D point group, D4,
    has 8 elements → 7 non-identity). Requires networkx.
    """
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher, categorical_edge_match

    nn_s = _edge_set(nn_edges)
    nnn_s = _edge_set(nnn_edges)
    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    for i, j in nn_s:
        g.add_edge(i, j, kind="nn")
    for i, j in nnn_s:
        if not g.has_edge(i, j):
            g.add_edge(i, j, kind="nnn")

    gm = GraphMatcher(g, g, edge_match=categorical_edge_match("kind", None))
    perms: list[Perm] = []
    names: list[str] = []
    identity = list(range(n_qubits))
    for mapping in gm.isomorphisms_iter():
        perm = [mapping[q] for q in range(n_qubits)]
        if perm == identity:
            continue
        perms.append(perm)
        names.append(f"sym{len(perms)}")
        if len(perms) >= max_syms:
            break
    return LatticeSymmetryGroup(n_qubits=n_qubits, perms=perms, names=names)


# ── orbits ───────────────────────────────────────────────────────────────────
def bond_orbits(edges, group: LatticeSymmetryGroup) -> list[list[Edge]]:
    """Partition ``edges`` into orbits under the group (edges mapped to each other
    by some symmetry share an orbit → they must share one tied parameter).

    Uses the group's generators (the stored perms) and takes closure by repeated
    application until each orbit stops growing. A trivial group gives singleton
    orbits (no tying). Orbits are returned sorted, each a sorted edge list.
    """
    es = _edge_set(edges)
    if group.is_trivial:
        return [[e] for e in sorted(es)]
    seen: set[Edge] = set()
    orbits: list[list[Edge]] = []
    for e in sorted(es):
        if e in seen:
            continue
        orbit = {e}
        frontier = {e}
        while frontier:
            nxt: set[Edge] = set()
            for f in frontier:
                for p in group.perms:
                    g = _image_edge(f, p)
                    if g in es and g not in orbit:
                        nxt.add(g)
            orbit |= nxt
            frontier = nxt
        seen |= orbit
        orbits.append(sorted(orbit))
    return orbits


def site_orbits(n_qubits: int, group: LatticeSymmetryGroup) -> list[list[int]]:
    """Partition qubits into orbits under the group (for tied single-qubit blocks)."""
    if group.is_trivial:
        return [[q] for q in range(n_qubits)]
    seen: set[int] = set()
    orbits: list[list[int]] = []
    for q in range(n_qubits):
        if q in seen:
            continue
        orbit = {q}
        frontier = {q}
        while frontier:
            nxt: set[int] = set()
            for f in frontier:
                for p in group.perms:
                    g = p[f]
                    if g not in orbit:
                        nxt.add(g)
            orbit |= nxt
            frontier = nxt
        seen |= orbit
        orbits.append(sorted(orbit))
    return orbits


def orbit_index_map(edges, group: LatticeSymmetryGroup) -> list[int]:
    """Bond → orbit-id, aligned with the ``edges`` order (the tying contract).

    Returns a list ``idx`` where ``idx[k]`` is the orbit id (0-based, contiguous)
    of ``edges[k]``. A builder assigns one shared parameter per distinct orbit id,
    so bonds in the same orbit get the same angle. Orbit ids are numbered by first
    appearance in ``edges`` order (stable, reproducible).
    """
    orbits = bond_orbits(edges, group)
    edge_to_orbit: dict[Edge, int] = {}
    for oid, orbit in enumerate(orbits):
        for e in orbit:
            edge_to_orbit[e] = oid
    # Renumber by first appearance in the given edges order (stable contract).
    remap: dict[int, int] = {}
    out: list[int] = []
    for e in edges:
        oid = edge_to_orbit[_canon(e)]
        if oid not in remap:
            remap[oid] = len(remap)
        out.append(remap[oid])
    return out


def site_index_map(n_qubits: int, group: LatticeSymmetryGroup) -> list[int]:
    """Qubit → orbit-id (the site-block analogue of :func:`orbit_index_map`)."""
    orbits = site_orbits(n_qubits, group)
    q_to_orbit: dict[int, int] = {}
    for oid, orbit in enumerate(orbits):
        for q in orbit:
            q_to_orbit[q] = oid
    remap: dict[int, int] = {}
    out: list[int] = []
    for q in range(n_qubits):
        oid = q_to_orbit[q]
        if oid not in remap:
            remap[oid] = len(remap)
        out.append(remap[oid])
    return out


# ── ground-state invariance (physical check) ─────────────────────────────────
def apply_permutation_to_state(psi: np.ndarray, perm: Perm, n_qubits: int) -> np.ndarray:
    """Apply a qubit permutation to a statevector by reindexing basis states.

    Qiskit little-endian: qubit ``q`` is bit ``q`` of the basis index. The symmetry
    moves the bit at position ``q`` to position ``perm[q]``. Pure; returns a new
    array. O(2^N · N) — fine for N ≤ ~22.
    """
    dim = 1 << n_qubits
    out = np.zeros_like(psi)
    for idx in range(dim):
        amp = psi[idx]
        if amp == 0:
            continue
        new = 0
        for q in range(n_qubits):
            if (idx >> q) & 1:
                new |= 1 << perm[q]
        out[new] = amp
    return out


def ground_state_invariance(psi: np.ndarray, group: LatticeSymmetryGroup) -> dict[str, float]:
    """|⟨ψ|P|ψ⟩| for each symmetry P — 1.0 means ψ is invariant under P.

    The physical confirmation that parameter-tying by this group is valid: if the
    target ground state is invariant under every P, then an orbit-tied ansatz loses
    no relevant expressivity. Values well below 1 warn the manifold mixes symmetry
    sectors (tie with care). Pure.
    """
    out: dict[str, float] = {}
    for name, p in zip(group.names, group.perms, strict=True):
        pp = apply_permutation_to_state(psi, p, group.n_qubits)
        out[name] = float(abs(np.vdot(psi, pp)))
    return out


# ── one-call analysis → persistable report ───────────────────────────────────
def analyze_symmetry(
    nn_edges,
    nnn_edges,
    n_qubits: int,
    *,
    topology: str = "square",
    h: float = 0.5,
    j2: float = 0.5,
    ground_state: np.ndarray | None = None,
    manifold_dim: int | None = None,
) -> tuple[LatticeSymmetryGroup, SymmetryReport]:
    """Detect the group, compute orbits + reduction, optionally check GS invariance.

    Returns ``(group, report)``. Pass ``ground_state`` (a statevector) to include
    the exact invariance check in the report — the physical validation. The report
    is JSON-persistable for downstream analysis.
    """
    from collections import Counter

    group = detect_symmetry_group(nn_edges, nnn_edges, n_qubits)
    nn_orb = bond_orbits(nn_edges, group)
    nnn_orb = bond_orbits(nnn_edges, group)
    s_orb = site_orbits(n_qubits, group)
    n_nn, n_nnn = len(_edge_set(nn_edges)), len(_edge_set(nnn_edges))
    indep = n_nn + n_nnn + n_qubits
    tied = len(nn_orb) + len(nnn_orb) + len(s_orb)

    gs_inv = None
    if ground_state is not None:
        gs_inv = ground_state_invariance(np.asarray(ground_state, complex), group)

    report = SymmetryReport(
        topology=topology,
        n_qubits=n_qubits,
        h=float(h),
        j2=float(j2),
        n_nn=n_nn,
        n_nnn=n_nnn,
        group_order=group.order,
        nn_orbits=len(nn_orb),
        nnn_orbits=len(nnn_orb),
        site_orbits=len(s_orb),
        params_per_layer_independent=indep,
        params_per_layer_tied=tied,
        reduction_factor=(indep / tied if tied else 1.0),
        nn_orbit_sizes=dict(Counter(len(o) for o in nn_orb)),
        nnn_orbit_sizes=dict(Counter(len(o) for o in nnn_orb)),
        gs_invariance=gs_inv,
        manifold_dim=manifold_dim,
    )
    return group, report


# ── tied warm-start (orbit compression of an independent seed) ───────────────
def compress_seed_to_orbits(
    indep_seed,
    blocks,
    nn_edges,
    nnn_edges,
    n_qubits: int,
    group: LatticeSymmetryGroup,
    *,
    rx_final: bool = False,
    rz_final: bool = False,
) -> np.ndarray:
    """Compress a per-bond warm-start to the tied (one-per-orbit) layout.

    The independent (masked) seed gives one value per bond/site per block; the
    symmetry-tied circuit has one parameter per ORBIT per block. Because the
    target state is group-invariant, bonds in an orbit share the same ideal
    angle, so each orbit's tied value is the MEAN of its members' independent
    values (robust to tiny asymmetries). Walks ``blocks`` in order, then the
    trailing ``rx_final`` / ``rz_final`` site blocks — the exact parameter order
    the tied builder (:meth:`HVACircuitBuilder.create_bond_resolved_symmetry_tied`)
    emits, so the compressed seed aligns slot-for-slot.

    Reuses the single-source orbit contract (:func:`orbit_index_map` /
    :func:`site_index_map`), so study runner and circuit never diverge.
    """
    nn_o = orbit_index_map(list(nn_edges), group) if len(nn_edges) else []
    nnn_o = orbit_index_map(list(nnn_edges), group) if len(nnn_edges) else []
    site_o = site_index_map(n_qubits, group)
    sizes = {"nn": len(nn_edges), "nnn": len(nnn_edges), "x": n_qubits, "z": n_qubits}
    orbmap = {"nn": nn_o, "nnn": nnn_o, "x": site_o, "z": site_o}
    seq = list(blocks)
    if rx_final:
        seq.append("x")
    if rz_final:
        seq.append("z")
    indep = np.asarray(indep_seed, float).ravel()
    expected = sum(sizes[b] for b in seq)
    if indep.size != expected:
        raise ValueError(
            f"independent seed length {indep.size} does not match the block layout "
            f"{seq} (expected {expected}); cannot compress to orbits"
        )
    out: list[float] = []
    off = 0
    for b in seq:
        seg = indep[off : off + sizes[b]]
        om = orbmap[b]
        n_orb = (max(om) + 1) if om else 0
        tied = np.zeros(n_orb)
        counts = np.zeros(n_orb)
        for k, oid in enumerate(om):
            tied[oid] += seg[k]
            counts[oid] += 1
        out.extend((tied / np.maximum(counts, 1)).tolist())
        off += sizes[b]
    return np.asarray(out, float)


# ── symmetry-projected warm-start (keep per-bond width, average within orbit) ─
def symmetrize_seed(
    seed,
    blocks,
    nn_edges,
    nnn_edges,
    n_qubits: int,
    group: LatticeSymmetryGroup,
    *,
    rx_final: bool = False,
    rz_final: bool = False,
) -> np.ndarray:
    """Project a per-bond seed onto the symmetric subspace, KEEPING per-bond width.

    The group-invariance of the target state means the optimum θ* is (nearly)
    constant within each orbit (verified: the converged full θ has ~0.01-0.1 rad
    within-orbit spread). This replaces every bond's seed value with its ORBIT
    MEAN but leaves the vector at full per-bond length — so the circuit stays the
    bond-resolved one (no parameter tying, no expressivity loss), while the
    starting point already lives on the symmetric manifold where the optimum sits.

    Contrast with :func:`compress_seed_to_orbits`, which SHRINKS to one value per
    orbit (for the tied circuit). Here the output length equals the input length:
    it is a projection, not a compression. Idempotent (re-symmetrizing is a no-op)
    and a no-op for a trivial group (every orbit is a singleton).

    Walks ``blocks`` then the trailing ``rx_final`` / ``rz_final`` site blocks,
    matching the bond-resolved parameter order. Reuses the orbit contract
    (:func:`orbit_index_map` / :func:`site_index_map`) so it never diverges from
    the tied path.
    """
    nn_o = orbit_index_map(list(nn_edges), group) if len(nn_edges) else []
    nnn_o = orbit_index_map(list(nnn_edges), group) if len(nnn_edges) else []
    site_o = site_index_map(n_qubits, group)
    sizes = {"nn": len(nn_edges), "nnn": len(nnn_edges), "x": n_qubits, "z": n_qubits}
    orbmap = {"nn": nn_o, "nnn": nnn_o, "x": site_o, "z": site_o}
    seq = list(blocks)
    if rx_final:
        seq.append("x")
    if rz_final:
        seq.append("z")
    vec = np.asarray(seed, float).ravel()
    expected = sum(sizes[b] for b in seq)
    if vec.size != expected:
        raise ValueError(
            f"seed length {vec.size} does not match the block layout {seq} (expected {expected}); cannot symmetrize"
        )
    out = vec.copy()
    off = 0
    for b in seq:
        seg = out[off : off + sizes[b]]
        om = orbmap[b]
        if om:
            n_orb = max(om) + 1
            sums = np.zeros(n_orb)
            counts = np.zeros(n_orb)
            for k, oid in enumerate(om):
                sums[oid] += seg[k]
                counts[oid] += 1
            means = sums / np.maximum(counts, 1)
            for k, oid in enumerate(om):
                seg[k] = means[oid]
        off += sizes[b]
    return out


# ── subgroup enumeration (symmetry-level sweep) ──────────────────────────────
def _compose(p: Perm, q: Perm) -> Perm:
    """Compose two permutations: (p∘q)[i] = p[q[i]]."""
    return [p[q[i]] for i in range(len(q))]


def _closure(generators: list[Perm], n_qubits: int) -> list[Perm]:
    """Closed set of permutations generated by ``generators`` (incl. identity).

    Repeatedly composes until the set stops growing — the subgroup they span.
    Each element is stored as a tuple key for dedup; returns NON-identity perms
    as lists (identity excluded, matching LatticeSymmetryGroup's convention).
    """
    identity = tuple(range(n_qubits))
    seen = {identity}
    for g in generators:
        seen.add(tuple(g))
    changed = True
    elems = {tuple(g) for g in generators} | {identity}
    while changed:
        changed = False
        current = [list(e) for e in elems]
        for a in current:
            for b in current:
                c = tuple(_compose(a, b))
                if c not in elems:
                    elems.add(c)
                    changed = True
    return [list(e) for e in elems if e != identity]


def generate_subgroups(group: LatticeSymmetryGroup) -> list[LatticeSymmetryGroup]:
    """Enumerate the distinct closed subgroups of ``group`` (for a symmetry sweep).

    Returns one :class:`LatticeSymmetryGroup` per distinct subgroup (including the
    trivial group and the full group), sorted by ascending order. Built by taking
    the closure of every subset of the group's non-identity perms and deduping by
    element set — so a caller can measure how much each LEVEL of symmetry (trivial
    → a single reflection → D2 → … → the full D4) contributes to a tied ansatz's
    parameter reduction and fidelity.

    Small groups only (point groups ≤ D4 → ≤ 8 elements): the subset enumeration
    is 2^(order-1), fine for the lattice point groups this study uses. The
    provenance of each subgroup records its order.
    """
    from itertools import combinations

    n = group.n_qubits
    perms = group.perms
    seen_sets: set[frozenset] = set()
    subgroups: list[LatticeSymmetryGroup] = []
    # include the empty generator set (trivial group) through full set
    for r in range(len(perms) + 1):
        for combo in combinations(perms, r):
            closed = _closure(list(combo), n)
            key = frozenset(tuple(p) for p in closed)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            order = len(closed) + 1
            subgroups.append(
                LatticeSymmetryGroup(
                    n_qubits=n,
                    perms=closed,
                    names=[f"sym{i + 1}" for i in range(len(closed))],
                    provenance=f"subgroup(order={order})",
                )
            )
    subgroups.sort(key=lambda g: g.order)
    return subgroups


# ── strategy recommendation (empirical, from the N=9/12/20 symmetry study) ────
@dataclass
class SymmetryStrategy:
    """Recommended ansatz-symmetry strategy for a (lattice, h) point.

    ``strategy`` is ``"orbit_tied"`` (select whole orbits + one parameter per
    orbit) or ``"loose"`` (bond-resolved, one parameter per bond). ``rationale``
    explains the choice; ``expected_param_reduction`` is the ×-factor fewer
    parameters orbit-tying would use (1.0 when loose). Advisory only — the caller
    decides; this encodes the study's measured rule, it does not run anything.
    """

    strategy: str
    rationale: str
    expected_param_reduction: float
    group_order: int


def recommend_symmetry_strategy(
    group: LatticeSymmetryGroup,
    h: float,
    *,
    gap: float | None = None,
    nn_edges=None,
    nnn_edges=None,
    h_order_max: float = 0.5,
    h_para_min: float = 1.0,
) -> SymmetryStrategy:
    """Recommend ``orbit_tied`` vs ``loose`` for a (lattice point group, h).

    Encodes the empirical rule measured across the square TFIM-frustrated study
    (N=9 D4, N=12 D2, N=20 D2; h-sweep 0.3→1.5):

    - **Trivial group** (order 1): ``loose`` — there is no symmetry to exploit.
    - **h ≤ ``h_order_max``** (ordered / frustrated side): ``orbit_tied`` — the
      ground state is strongly symmetric, so selecting whole orbits + tying
      MATCHES or BEATS loose fidelity (e.g. +0.34 at h=0.3, N=9) at a large
      parameter reduction. This is the robust default where symmetry is present.
    - **h ≥ ``h_para_min``** (paramagnet): ``loose`` — the per-bond freedom edges
      out tied fidelity slightly; keep orbit_tied only if parameters are the
      priority (it stays within ~1-2% and uses far fewer).
    - **in between**: ``orbit_tied`` — fidelity is a near-tie and the parameter
      reduction is free.

    The distinction that actually matters (from the study): ``orbit_tied`` means
    whole-orbit SELECTION + tying; the failure mode ``loose_tied`` (tying over
    loosely-pruned bonds) collapses and is never recommended here.

    ``expected_param_reduction`` is computed from the orbit counts when the edge
    lists are supplied (else from the group order as a proxy). Pure/advisory.
    """
    order = group.order
    if group.is_trivial:
        return SymmetryStrategy(
            strategy="loose",
            rationale="trivial symmetry group (order 1): no orbits to tie",
            expected_param_reduction=1.0,
            group_order=order,
        )

    # Parameter-reduction estimate: independent / tied counts from the orbits.
    reduction = float(order)  # proxy
    if nn_edges is not None and nnn_edges is not None:
        n_ind = len(nn_edges) + len(nnn_edges)
        n_tied = len(bond_orbits(nn_edges, group)) + len(bond_orbits(nnn_edges, group))
        if n_tied:
            reduction = round(n_ind / n_tied, 2)

    if h <= h_order_max:
        strat, why = (
            "orbit_tied",
            (
                f"h={h:g} ≤ {h_order_max:g} (ordered/frustrated): the GS is strongly "
                f"symmetric, orbit-tied matches or beats loose fidelity at ×{reduction} "
                f"fewer params"
            ),
        )
    elif h >= h_para_min:
        strat, why = (
            "loose",
            (
                f"h={h:g} ≥ {h_para_min:g} (paramagnet): per-bond freedom edges out "
                f"tied fidelity slightly; use orbit_tied only if params are the priority"
            ),
        )
    else:
        strat, why = (
            "orbit_tied",
            (
                f"h={h:g} near h_c: fidelity is a near-tie, orbit-tying's ×{reduction} "
                f"param reduction is effectively free"
            ),
        )
    return SymmetryStrategy(strategy=strat, rationale=why, expected_param_reduction=reduction, group_order=order)
