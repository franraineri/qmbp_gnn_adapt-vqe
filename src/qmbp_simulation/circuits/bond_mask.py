"""Bond selection for masked bond-resolved HVA ansätze.

Shared primitive for exploring ansätze that entangle only a SUBSET of the
lattice bonds, rather than the all-or-nothing nn/nnn blocks of
``create_bond_resolved_frustrated_configurable``. Every RZZ dropped saves two
CX, so choosing WHICH bonds to keep is the lever for "same fidelity, fewer 2q
gates".

Three selection signals, one contract (:class:`BondSelection`):

- :func:`prune_by_theta` — keep bonds whose optimized ``|θ_zz|`` exceeds a
  tolerance (a near-zero RZZ is a near-identity; dropping it is nearly free).
  This is technique **T1** (per-bond pruning).
- :func:`top_k_by_weight` — keep the ``k`` highest-weight bonds by some score
  (``|θ|`` or gradient magnitude). This is technique **T2** (nnn subset).
- :func:`rank_by_gradient` — order bonds by energy-gradient magnitude at a seed
  point. This is the growth signal an **ADAPT-VQE** loop consumes to add bonds
  one at a time; the loop extends a :class:`BondSelection` rather than
  re-implementing circuit construction.

All functions are pure (lists of edges + numpy arrays in, a ``BondSelection``
out) so they unit-test without any circuit build or ``scripts/`` import. The
selection is consumed by ``HVACircuitBuilder.create_bond_resolved_masked``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

Edge = tuple[int, int]


def _normalize_edges(edges) -> list[Edge]:
    """Coerce an edge iterable to a list of ``(int, int)`` tuples."""
    return [(int(i), int(j)) for i, j in edges]


@dataclass(frozen=True)
class BondSelection:
    """A chosen subset of nn / nnn RZZ bonds for a masked ansatz.

    ``nn_edges`` / ``nnn_edges`` are the bonds that WILL carry an RZZ gate (each
    an independent bond-resolved parameter). ``provenance`` records how the
    selection was derived (e.g. ``"prune_by_theta(tol=0.05)"``) so a result
    artifact explains itself. The 2q cost of the selection is
    ``2 * (len(nn_edges) + len(nnn_edges))`` CX.

    A selection holding EVERY nn and nnn bond reproduces the standard frustrated
    layer exactly — so the masked builder is a strict generalization of the
    configurable one (verified by an equivalence test).
    """

    nn_edges: list[Edge] = field(default_factory=list)
    nnn_edges: list[Edge] = field(default_factory=list)
    provenance: str = ""

    def __post_init__(self) -> None:
        # Freeze-friendly normalization: store plain (int,int) tuples.
        object.__setattr__(self, "nn_edges", _normalize_edges(self.nn_edges))
        object.__setattr__(self, "nnn_edges", _normalize_edges(self.nnn_edges))

    @property
    def n_bonds(self) -> int:
        """Total RZZ bonds carried by this selection."""
        return len(self.nn_edges) + len(self.nnn_edges)

    @property
    def n_2q(self) -> int:
        """Theoretical 2q-gate (CX) count: two CX per RZZ."""
        return 2 * self.n_bonds

    def is_empty(self) -> bool:
        """Whether no bond is selected (a rotation-only ansatz)."""
        return self.n_bonds == 0


def full_selection(nn_edges, nnn_edges, *, provenance: str = "full") -> BondSelection:
    """Select every nn and nnn bond — reproduces the standard frustrated layer."""
    return BondSelection(
        nn_edges=_normalize_edges(nn_edges),
        nnn_edges=_normalize_edges(nnn_edges),
        provenance=provenance,
    )


def prune_by_theta(
    nn_edges,
    nnn_edges,
    theta_nn: np.ndarray,
    theta_nnn: np.ndarray,
    *,
    tol: float = 0.05,
) -> BondSelection:
    """T1 — keep only bonds whose optimized ``|θ_zz|`` exceeds ``tol``.

    A bond whose converged angle is ~0 contributes a near-identity RZZ; pruning
    it removes 2 CX at negligible fidelity cost. ``theta_nn`` / ``theta_nnn`` are
    the per-bond angles for ONE layer, aligned with ``nn_edges`` / ``nnn_edges``.
    Non-finite angles are treated as prunable (dropped).

    Raises ``ValueError`` on a length mismatch between edges and angles.
    """
    import numpy as np

    nn = _normalize_edges(nn_edges)
    nnn = _normalize_edges(nnn_edges)
    t_nn = np.asarray(theta_nn, dtype=float)
    t_nnn = np.asarray(theta_nnn, dtype=float)
    if t_nn.shape[0] != len(nn):
        raise ValueError(f"theta_nn length {t_nn.shape[0]} != n_nn {len(nn)}")
    if t_nnn.shape[0] != len(nnn):
        raise ValueError(f"theta_nnn length {t_nnn.shape[0]} != n_nnn {len(nnn)}")

    keep_nn = [e for e, t in zip(nn, t_nn, strict=True) if np.isfinite(t) and abs(t) > tol]
    keep_nnn = [e for e, t in zip(nnn, t_nnn, strict=True) if np.isfinite(t) and abs(t) > tol]
    return BondSelection(
        nn_edges=keep_nn,
        nnn_edges=keep_nnn,
        provenance=f"prune_by_theta(tol={tol})",
    )


def top_k_by_weight(
    nn_edges,
    nnn_edges,
    weight_nn: np.ndarray,
    weight_nnn: np.ndarray,
    *,
    k_nn: int | None = None,
    k_nnn: int | None = None,
) -> BondSelection:
    """T2 — keep the highest-weight bonds by ``|weight|`` (θ or gradient).

    ``k_nn`` / ``k_nnn`` cap how many nn / nnn bonds to keep (``None`` = keep all
    of that type). Bonds are ranked by descending ``|weight|``; ties break by
    original edge order (stable). Non-finite weights sort last (least important).

    This is how the "nnn subset of highest weight" ansatz is built: e.g.
    ``k_nn=all, k_nnn=len//2`` keeps every nn bond plus the strongest half of nnn.
    """
    import numpy as np

    def _select(edges, weights, k) -> list[Edge]:
        edges = _normalize_edges(edges)
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(edges):
            raise ValueError(f"weight length {w.shape[0]} != n_edges {len(edges)}")
        if k is None or k >= len(edges):
            return edges
        if k <= 0:
            return []
        # Non-finite → -inf so they rank last; stable order via index tiebreak.
        score = np.where(np.isfinite(w), np.abs(w), -np.inf)
        order = sorted(range(len(edges)), key=lambda idx: (-score[idx], idx))
        chosen = sorted(order[:k])  # restore original edge order for the kept set
        return [edges[idx] for idx in chosen]

    return BondSelection(
        nn_edges=_select(nn_edges, weight_nn, k_nn),
        nnn_edges=_select(nnn_edges, weight_nnn, k_nnn),
        provenance=f"top_k_by_weight(k_nn={k_nn}, k_nnn={k_nnn})",
    )


def rank_by_gradient(
    nn_edges,
    nnn_edges,
    grad_nn: np.ndarray,
    grad_nnn: np.ndarray,
) -> list[tuple[str, Edge, float]]:
    """ADAPT growth signal — rank all bonds by ``|energy gradient|`` (descending).

    Returns a flat list of ``(kind, edge, |grad|)`` where ``kind`` is ``"nn"`` or
    ``"nnn"``, ordered most-important first. An ADAPT-VQE loop starts from an
    empty (or minimal) :class:`BondSelection` and appends the top-ranked bond(s)
    each iteration, re-building the circuit via ``create_bond_resolved_masked`` —
    reusing this ranking rather than re-deriving circuit structure. Non-finite
    gradients rank last.
    """
    import numpy as np

    nn = _normalize_edges(nn_edges)
    nnn = _normalize_edges(nnn_edges)
    g_nn = np.asarray(grad_nn, dtype=float)
    g_nnn = np.asarray(grad_nnn, dtype=float)
    if g_nn.shape[0] != len(nn):
        raise ValueError(f"grad_nn length {g_nn.shape[0]} != n_nn {len(nn)}")
    if g_nnn.shape[0] != len(nnn):
        raise ValueError(f"grad_nnn length {g_nnn.shape[0]} != n_nnn {len(nnn)}")

    ranked: list[tuple[str, Edge, float]] = []
    for edge, g in zip(nn, g_nn, strict=True):
        ranked.append(("nn", edge, float(abs(g)) if np.isfinite(g) else -1.0))
    for edge, g in zip(nnn, g_nnn, strict=True):
        ranked.append(("nnn", edge, float(abs(g)) if np.isfinite(g) else -1.0))
    ranked.sort(key=lambda t: -t[2])
    return ranked


def bond_weights_from_theta(
    theta,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
):
    """Per-bond importance ``max|θ|`` across the p layers of a standard ansatz.

    Aggregates the per-bond RZZ angle magnitude over all ``p_layers`` of the
    STANDARD ``[nn, nnn, x]`` repeated layout, returning ``(w_nn, w_nnn)`` aligned
    with the nn / nnn edge lists. A bond that is strong in ANY layer is kept, so
    this is the ranking signal for :func:`prune_by_theta` (T1) and
    :func:`top_k_by_weight` (T2). Consolidates the ``_bond_weights`` helper that
    was duplicated across the bond-ablation / topk / prep-evolution runners.

    For a non-standard block sequence (extra half-layers, trailing rotations) use
    :func:`bond_weights_for_blocks`, which walks the exact block offsets.
    """
    import numpy as np

    theta = np.asarray(theta, float)
    per = n_nn + n_nnn + n_qubits
    w_nn = np.zeros(n_nn)
    w_nnn = np.zeros(n_nnn)
    for layer in range(p_layers):
        o = layer * per
        w_nn = np.maximum(w_nn, np.abs(theta[o : o + n_nn]))
        w_nnn = np.maximum(w_nnn, np.abs(theta[o + n_nn : o + n_nn + n_nnn]))
    return w_nn, w_nnn


def bond_weights_for_blocks(
    theta,
    blocks: list[str],
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    *,
    rx_final: bool = False,
    rz_final: bool = False,
):
    """Per-bond ``max|θ|`` for an ARBITRARY block sequence (structure variants).

    Walks ``blocks`` in order with the same offset convention the configurable /
    masked builders use (nn/nnn RZZ blocks, x/z single-qubit blocks, then
    rx_final/rz_final), accumulating the maximum ``|θ|`` each nn / nnn bond
    reaches in ANY of its blocks. Needed for variants like ``p2_half_nn_rx``
    (``[nn,nnn,x, nn,nnn,x, nn,x]`` + rx_final) where nn appears more times than
    nnn, so the uniform-layer :func:`bond_weights_from_theta` would mis-slice.

    Returns ``(w_nn, w_nnn)`` aligned with the nn / nnn edge lists. Raises if the
    θ length does not match the block sequence. Pure.
    """
    import numpy as np

    theta = np.asarray(theta, float)
    size = {"nn": n_nn, "nnn": n_nnn, "x": n_qubits, "z": n_qubits}
    expected = sum(size[b] for b in blocks)
    expected += n_qubits if rx_final else 0
    expected += n_qubits if rz_final else 0
    if theta.size != expected:
        raise ValueError(
            f"theta length {theta.size} != expected {expected} for this lattice "
            f"(n_nn={n_nn} n_nnn={n_nnn} n_qubits={n_qubits} blocks={blocks} "
            f"rx_final={rx_final} rz_final={rz_final}). The most common cause is a "
            f"size mismatch: a θ optimized at one N is being applied to a lattice "
            f"of a different N (bond counts scale with N). Use a θ/AnsatzSpec built "
            f"for n_qubits={n_qubits}, or rebuild the lattice to match the θ."
        )
    w_nn = np.zeros(n_nn)
    w_nnn = np.zeros(n_nnn)
    off = 0
    for b in blocks:
        if b == "nn" and n_nn:
            w_nn = np.maximum(w_nn, np.abs(theta[off : off + n_nn]))
        elif b == "nnn" and n_nnn:
            w_nnn = np.maximum(w_nnn, np.abs(theta[off + 0 : off + n_nnn]))
        off += size[b]
    return w_nn, w_nnn


def selection_from_variant_theta(
    theta,
    blocks: list[str],
    nn_edges,
    nnn_edges,
    n_qubits: int,
    *,
    rx_final: bool = False,
    rz_final: bool = False,
    method: str = "top_k",
    keep_frac: float = 0.5,
    tol: float = 0.05,
) -> BondSelection:
    """Derive a bond :class:`BondSelection` from a converged structure-variant θ.

    The one bridge that connects an ARBITRARY ansatz structure (``blocks`` +
    ``rx_final``/``rz_final``, e.g. ``p2_half_nn_rx``) to bond selection. It reads
    the per-bond importance from the variant's θ with the structure-aware
    :func:`bond_weights_for_blocks` (so asymmetric layouts are sliced correctly),
    then applies either technique:

    - ``method="top_k"`` (T2): keep every nn bond + the strongest ``keep_frac`` of
      nnn bonds by ``|θ|`` (:func:`top_k_by_weight`).
    - ``method="prune"`` (T1): drop any nn/nnn bond with ``|θ| < tol``
      (:func:`prune_by_theta`).

    Previously T1/T2 only worked on the uniform ``[nn,nnn,x]*p`` layout because
    the runners sliced θ by hand assuming that layout; this makes both techniques
    available to any registry structure variant. Pure (edges + θ in, selection
    out). Raises on an unknown ``method``.
    """
    nn_edges = _normalize_edges(nn_edges)
    nnn_edges = _normalize_edges(nnn_edges)
    w_nn, w_nnn = bond_weights_for_blocks(
        theta, blocks, len(nn_edges), len(nnn_edges), n_qubits, rx_final=rx_final, rz_final=rz_final
    )
    if method == "top_k":
        k = max(1, int(round(keep_frac * len(nnn_edges))))
        return top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn, k_nn=None, k_nnn=k)
    if method == "prune":
        return prune_by_theta(nn_edges, nnn_edges, w_nn, w_nnn, tol=tol)
    raise ValueError(f"unknown method {method!r}; expected 'top_k' or 'prune'")


def rank_correlation(weight_a, weight_b) -> float | None:
    """Spearman rank correlation between two per-bond importance signals.

    The decision metric for the ADAPT Gate 0 question: does the energy-gradient
    ranking of the candidate bonds agree with the ``|θ|`` ranking of the fully
    converged ansatz? ``weight_a`` / ``weight_b`` are aligned per-bond arrays
    (e.g. ``|∂E/∂θ|`` and ``|θ_full|`` over the same nnn edges). Returns
    Spearman ρ in ``[-1, 1]`` (Pearson on the rank-transformed values, averaging
    tied ranks), or ``None`` when either input is degenerate (fewer than three
    points, or all values equal) so the caller treats it as "no signal".

    Pure numpy — no scipy — so it runs anywhere the bond helpers do. Mirrors the
    ``_spearman`` previously duplicated in ``analyze_bond_importance.py``.
    """
    import numpy as np

    a = np.asarray(weight_a, dtype=float)
    b = np.asarray(weight_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"weight_a shape {a.shape} != weight_b shape {b.shape}")
    if a.size < 3 or np.allclose(a, a.flat[0]) or np.allclose(b, b.flat[0]):
        return None

    def _rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        start = csum - counts
        avg = (start + csum - 1) / 2.0  # average rank within each tie group
        return avg[inv]

    ra = _rank(a) - _rank(a).mean()
    rb = _rank(b) - _rank(b).mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else None


def rank_agreement_topk(weight_a, weight_b, k: int) -> float | None:
    """Overlap fraction of the top-``k`` bonds selected by two importance signals.

    Complements :func:`rank_correlation` with the metric that matters for bond
    GROWTH: of the ``k`` bonds each signal ranks highest, what fraction do they
    agree on? ``1.0`` means the two signals would grow the exact same ``k`` bonds
    (ADAPT by gradient ≡ top-k by ``|θ|``); ``0.0`` means fully disjoint choices.

    ``k`` is clamped to ``[1, n_bonds]``. Returns ``None`` only when there are no
    bonds. Ties at the ``k``-th boundary follow the stable descending-``|weight|``
    order used by :func:`top_k_by_weight`, so this is consistent with the actual
    selection. Pure numpy.
    """
    import numpy as np

    a = np.asarray(weight_a, dtype=float)
    b = np.asarray(weight_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"weight_a shape {a.shape} != weight_b shape {b.shape}")
    n = a.size
    if n == 0:
        return None
    k = max(1, min(int(k), n))

    def _topk_set(w):
        score = np.where(np.isfinite(w), np.abs(w), -np.inf)
        order = sorted(range(n), key=lambda idx: (-score[idx], idx))
        return set(order[:k])

    sa, sb = _topk_set(a), _topk_set(b)
    return len(sa & sb) / float(k)
