"""Physical analysis of optimized HVA angles (bond-resolved frustrated ansatz).

Decomposes an optimized parameter vector ``θ`` into its physical blocks and
quantifies what the angles represent, the patterns they follow, and the
symmetries they obey. Pure numpy; unit-testable without a quantum backend.

Ansatz recap (``create_bond_resolved_frustrated``), initial state ``|+⟩^N``::

    |ψ⟩ = ∏_layers [ ∏_{NN edges} e^{-iθ_nn Z_iZ_j}
                     ∏_{NNN edges} e^{-iθ_nnn Z_iZ_j}
                     ∏_{sites}    e^{-iθ_x X_i} ] |+⟩^N

Parameter layout per layer: ``[θ_nn (n_nn), θ_nnn (n_nnn), θ_x (n_qubits)]``,
with gates ``rzz(2θ)`` and ``rx(2θ)`` — the factor of 2 is folded into the
angle, so the leading-order (adiabatic/Trotter) relations are

    θ_nn ≈ -J/(4h),   θ_nnn ≈ -J2/(4h),   θ_x ≈ arctan(J/h).

Physical reading of each block:

- ``θ_nn`` / ``θ_nnn`` — the accumulated ZZ-rotation on each nearest / next-
  nearest bond. It is the *effective imaginary-time coupling* the state applies
  to that bond; its magnitude grows as the field ``h`` weakens (the ordered
  phase needs stronger ZZ correlation). NN and NNN carry opposite roles under
  frustration (J2 competes with J1).
- ``θ_x`` — the single-site rotation about X. ``θ_x → π/2`` leaves the site in
  the paramagnetic ``|+⟩`` (field-dominated); ``θ_x → 0`` tilts it toward the
  Z-axis (order-dominated). It interpolates the site between the two phases.

Symmetries this module measures:

- **Z2 (global spin-flip)**: the TFIM ground state is invariant under
  ∏_i X_i; the energy is invariant under ``θ → -θ`` on the ZZ blocks. We
  canonicalize the sign so comparisons are Z2-invariant.
- **Spatial homogeneity**: on a translationally-invariant lattice the optimizer
  *could* pick a uniform θ per block; the bond-to-bond dispersion measures how
  far the state departs from the homogeneous (global-HVA) solution — a probe of
  symmetry breaking near the frustrated transition.
"""

from __future__ import annotations

import numpy as np

from qmbp_simulation.analysis.warmstart import (
    first_order_warmstart_theta,
    second_order_warmstart_theta,
)


def layer_size(n_nn: int, n_nnn: int, n_qubits: int) -> int:
    """Number of parameters in one layer of the bond-resolved frustrated ansatz."""
    return n_nn + n_nnn + n_qubits


def decompose_theta(theta, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int) -> list[dict]:
    """Split a flat θ vector into per-layer physical blocks.

    Returns a list of length ``p_layers``; each entry is a dict with numpy
    arrays ``{"nn": (n_nn,), "nnn": (n_nnn,), "x": (n_qubits,)}``. Raises if the
    length does not match ``(n_nn + n_nnn + n_qubits) * p_layers``.
    """
    theta = np.asarray(theta, dtype=float)
    per = layer_size(n_nn, n_nnn, n_qubits)
    expected = per * p_layers
    if theta.size != expected:
        raise ValueError(
            f"theta length {theta.size} != expected {expected} "
            f"(n_nn={n_nn}, n_nnn={n_nnn}, n_qubits={n_qubits}, p={p_layers})"
        )
    layers = []
    for layer in range(p_layers):
        o = layer * per
        layers.append({
            "nn": theta[o:o + n_nn].copy(),
            "nnn": theta[o + n_nn:o + n_nn + n_nnn].copy(),
            "x": theta[o + n_nn + n_nnn:o + per].copy(),
        })
    return layers


def _block_stats(arr: np.ndarray) -> dict:
    """Mean / std / min / max / homogeneity for one block of angles."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "abs_mean": None, "homogeneity": None}
    mean = float(arr.mean())
    std = float(arr.std())
    abs_mean = float(np.abs(arr).mean())
    # Homogeneity ∈ (-inf, 1]: 1 = all angles identical; lower = more dispersed
    # relative to the block's scale. Uses abs_mean to stay meaningful when the
    # mean is near zero but magnitudes are not.
    homogeneity = float(1.0 - std / abs_mean) if abs_mean > 1e-12 else None
    return {
        "n": int(arr.size), "mean": mean, "std": std,
        "min": float(arr.min()), "max": float(arr.max()),
        "abs_mean": abs_mean, "homogeneity": homogeneity,
    }


def block_statistics(layers: list[dict]) -> list[dict]:
    """Per-layer, per-block statistics (mean/std/range/homogeneity)."""
    return [{blk: _block_stats(layer[blk]) for blk in ("nn", "nnn", "x")}
            for layer in layers]


def canonicalize_z2(theta, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int):
    """Return a Z2-canonical copy of θ (sign fixed by the ZZ blocks).

    The energy is invariant under flipping the sign of the ZZ rotations
    together (a Z2 relabeling). We fix a canonical sign by requiring the sum of
    the first-layer NN angles to be ≤ 0 (matching the analytic warm-start, whose
    NN angles are negative), so two Z2-equivalent optima compare equal.
    """
    theta = np.asarray(theta, dtype=float).copy()
    layers = decompose_theta(theta, n_nn, n_nnn, n_qubits, p_layers)
    nn_sum = float(np.sum(layers[0]["nn"])) if n_nn else 0.0
    if nn_sum > 0:
        # flip ZZ blocks (nn + nnn) across all layers; θ_x is unaffected by Z2.
        per = layer_size(n_nn, n_nnn, n_qubits)
        for layer in range(p_layers):
            o = layer * per
            theta[o:o + n_nn + n_nnn] *= -1.0
    return theta


def compare_to_warmstart(theta, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                         h: float, *, J: float = 1.0, J2: float = 0.0,
                         order: str = "first") -> dict:
    """Compare an optimized θ to the analytic warm-start it should resemble.

    Returns the L2 distance overall and per block, plus the per-block mean of
    the optimized angles against the (scalar) analytic prediction. ``order`` ∈
    {"first", "second"}. Both θ are Z2-canonicalized before comparison so a
    spin-flipped optimum is not spuriously "far".
    """
    if order == "second":
        ws = second_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    else:
        ws = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    th_c = canonicalize_z2(theta, n_nn, n_nnn, n_qubits, p_layers)
    ws_c = canonicalize_z2(ws, n_nn, n_nnn, n_qubits, p_layers)
    opt_layers = decompose_theta(th_c, n_nn, n_nnn, n_qubits, p_layers)
    ws_layers = decompose_theta(ws_c, n_nn, n_nnn, n_qubits, p_layers)

    per_block = {}
    for blk in ("nn", "nnn", "x"):
        opt_all = np.concatenate([lay[blk] for lay in opt_layers]) if any(
            lay[blk].size for lay in opt_layers) else np.array([])
        ws_all = np.concatenate([lay[blk] for lay in ws_layers]) if any(
            lay[blk].size for lay in ws_layers) else np.array([])
        if opt_all.size:
            per_block[blk] = {
                "opt_mean": float(opt_all.mean()),
                "ws_mean": float(ws_all.mean()),
                "l2": float(np.linalg.norm(opt_all - ws_all)),
                "mean_abs_dev": float(np.abs(opt_all - ws_all).mean()),
            }
        else:
            per_block[blk] = {"opt_mean": None, "ws_mean": None, "l2": 0.0,
                              "mean_abs_dev": None}
    return {
        "order": order, "h": float(h),
        "l2_total": float(np.linalg.norm(th_c - ws_c)),
        "per_block": per_block,
    }


def wrap_angle(theta, period: float = np.pi, center: float = 0.0):
    """Wrap angle(s) into ``(center - period/2, center + period/2]``.

    The single-qubit ``rx(2θ)`` gate is π-periodic in θ (``rx(2θ+2π)=rx(2θ)``),
    and the ZZ rotation ``rzz(2θ)`` is likewise π-periodic. Optimizers land on
    different periodic images / Z2 gauge copies at different h, which makes the
    RAW block means jump discontinuously even when the physical state is smooth.
    Wrapping to a canonical branch removes that gauge noise so the true θ(h)
    pattern is visible. Default period π, centered at 0.
    """
    arr = np.asarray(theta, dtype=float)
    return (arr - center + period / 2.0) % period - period / 2.0 + center


def effective_couplings(layers: list[dict], h: float, *, J: float = 1.0) -> dict:
    """Invert the leading-order warm-start relations to read off effective scales.

    Treating layer-1 block means as a first-order seed, recover the effective
    couplings the state encodes::

        J_nn_eff  = -4h · mean(θ_nn)
        J_nnn_eff = -4h · mean(θ_nnn)
        field_angle = mean(θ_x)          (compare to arctan(J/h))

    These are diagnostic, not a fit; they show whether the optimized state's
    average bond rotation matches the physical ``J/(4h)`` scaling and whether the
    NNN block carries the opposite (frustrating) sign expected for J2 > 0.
    """
    if not layers:
        raise ValueError("need at least one layer")
    l0 = layers[0]
    nn_mean = float(l0["nn"].mean()) if l0["nn"].size else 0.0
    nnn_mean = float(l0["nnn"].mean()) if l0["nnn"].size else None
    x_mean = float(l0["x"].mean()) if l0["x"].size else None
    return {
        "J_nn_eff": -4.0 * h * nn_mean,
        "J_nnn_eff": (-4.0 * h * nnn_mean) if nnn_mean is not None else None,
        "field_angle": x_mean,
        "field_angle_pred": float(np.arctan(J / h)),
        "theta_x_over_pred": (x_mean / np.arctan(J / h))
        if (x_mean is not None and h > 0) else None,
    }


def analyze_theta(theta, *, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                  h: float, J: float = 1.0, J2: float = 0.0) -> dict:
    """Full physical analysis of one optimized θ vector.

    Combines decomposition, per-block statistics, warm-start comparison (first
    and second order), effective couplings, and layer-to-layer drift into one
    JSON-safe dict. This is the single entry point the sweep analysis and the
    report generation call.
    """
    layers = decompose_theta(theta, n_nn, n_nnn, n_qubits, p_layers)
    stats = block_statistics(layers)
    # Gauge-canonical block stats: wrap each block to the π-periodic branch so
    # periodic-image / Z2 jumps between h-points do not masquerade as physics.
    wrapped_layers = [{blk: wrap_angle(layer[blk]) for blk in ("nn", "nnn", "x")}
                      for layer in layers]
    stats_wrapped = block_statistics(wrapped_layers)
    cmp1 = compare_to_warmstart(theta, n_nn, n_nnn, n_qubits, p_layers, h,
                                J=J, J2=J2, order="first")
    cmp2 = compare_to_warmstart(theta, n_nn, n_nnn, n_qubits, p_layers, h,
                                J=J, J2=J2, order="second")
    eff = effective_couplings(layers, h, J=J)

    # Layer-to-layer drift (only meaningful for p >= 2): how much block means
    # change between layers — a uniform-per-layer ansatz would show ~0.
    layer_drift = None
    if p_layers >= 2:
        drift = {}
        for blk in ("nn", "nnn", "x"):
            means = [float(lay[blk].mean()) if lay[blk].size else 0.0 for lay in layers]
            drift[blk] = float(np.max(means) - np.min(means))
        layer_drift = drift

    return {
        "h": float(h), "J": float(J), "J2": float(J2),
        "n_nn": n_nn, "n_nnn": n_nnn, "n_qubits": n_qubits, "p_layers": p_layers,
        "block_stats": stats,
        "block_stats_wrapped": stats_wrapped,
        "warmstart_first": cmp1,
        "warmstart_second": cmp2,
        "effective_couplings": eff,
        "layer_drift": layer_drift,
    }


def extract_theta_metrics(theta, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int,
                          h: float, *, J: float = 1.0, J2: float = 0.0,
                          seed_theta=None) -> dict:
    """Flat, tidy per-θ metrics row — the automation-friendly view.

    Where :func:`analyze_theta` returns a nested physical report, this returns a
    single flat dict of the handful of numbers we keep recomputing by hand when
    comparing runs across N and h: block magnitudes, the layer-specialization
    ratio, the θ_x mean, and (if a ``seed_theta`` is given) the distance of the
    optimized θ to that seed. All on the Z2-canonical, π-wrapped θ so values are
    gauge-free and comparable across points. Pure.

    Columns:
      nn_absmean, nnn_absmean, x_absmean : mean |θ| per block (intensive, N-comparable)
      nn_L1_mean, nn_L2_mean             : per-layer nn means (p>=2; else L2=0)
      l2_l1_nn_ratio                     : |nn_L2|/|nn_L1| — layer specialization
      theta_x_mean                       : overall θ_x mean (phase feature)
      d_theta_to_seed                    : L2 distance to seed_theta (canonical), or None
      phase                              : 'ordered' | 'near_hc' | 'paramag' (by h window)
    """
    th = wrap_angle(canonicalize_z2(theta, n_nn, n_nnn, n_qubits, p_layers))
    layers = decompose_theta(th, n_nn, n_nnn, n_qubits, p_layers)

    def _absmean(blk):
        allb = np.concatenate([lay[blk] for lay in layers]) if any(
            lay[blk].size for lay in layers) else np.array([])
        return float(np.abs(allb).mean()) if allb.size else 0.0

    nn_l1 = float(layers[0]["nn"].mean()) if (layers and layers[0]["nn"].size) else 0.0
    nn_l2 = float(layers[1]["nn"].mean()) if (p_layers >= 2 and layers[1]["nn"].size) else 0.0
    x_all = np.concatenate([lay["x"] for lay in layers]) if any(
        lay["x"].size for lay in layers) else np.array([])
    theta_x_mean = float(x_all.mean()) if x_all.size else 0.0

    d_seed = None
    if seed_theta is not None:
        s = wrap_angle(canonicalize_z2(np.asarray(seed_theta, float),
                                       n_nn, n_nnn, n_qubits, p_layers))
        if s.size == th.size:
            d_seed = float(np.linalg.norm(th - s))

    # Phase by the calibrated h windows (kept local to avoid importing warmstart,
    # which imports this module — would be circular).
    phase = "ordered" if h <= 0.45 else ("near_hc" if h < 1.2 else "paramag")

    return {
        "N": int(n_qubits), "h": float(h), "p_layers": int(p_layers), "phase": phase,
        "nn_absmean": _absmean("nn"), "nnn_absmean": _absmean("nnn"),
        "x_absmean": _absmean("x"),
        "nn_L1_mean": nn_l1, "nn_L2_mean": nn_l2,
        "l2_l1_nn_ratio": float(abs(nn_l2) / (abs(nn_l1) + 1e-9)),
        "theta_x_mean": theta_x_mean,
        "d_theta_to_seed": d_seed,
    }


# ── Compressed-circuit angle comparison ──────────────────────────────────────
# The analyzers above assume the fixed [nn|nnn|x] * p layout. Compressed circuits
# (top-k masks, pruned bonds, half_nn_rx variants, extra rx/rz-final) have an
# ARBITRARY block sequence and different bond sets. The functions below decode an
# AnsatzSpec-style θ by its block sequence into role+edge-tagged angles, then
# compare several high-fidelity compressed circuits ON THEIR SHARED gates — so we
# can see which angles are INVARIANT across compressions (predictable) vs which
# the compression lets drift. Divide & conquer at the gate level.

def decompose_spec_theta(theta, blocks, nn_edges, nnn_edges, n_qubits, *,
                         rx_final: bool = False, rz_final: bool = False) -> list[dict]:
    """Decode a block-sequence θ into role+edge-tagged gate angles.

    Mirrors the masked builder's parameter order: each entry of ``blocks`` consumes
    a slab of θ — ``"nn"`` → one angle per ``nn_edges``, ``"nnn"`` → one per
    ``nnn_edges``, ``"x"``/``"z"`` → one per qubit — followed by an optional
    ``rx_final`` / ``rz_final`` single-qubit layer. Returns a flat list of
    ``{"role", "key", "angle", "layer"}`` where ``key`` is the sorted edge tuple
    (ZZ roles) or the qubit index (single-qubit roles), so two circuits can be
    aligned by gate identity regardless of block ordering or masking. Pure.
    """
    theta = np.asarray(theta, float)
    nn = [tuple(sorted((int(a), int(b)))) for a, b in nn_edges]
    nnn = [tuple(sorted((int(a), int(b)))) for a, b in nnn_edges]
    # Validate the declared structure consumes exactly len(theta) before indexing.
    _sizes = {"nn": len(nn), "nnn": len(nnn), "x": n_qubits, "z": n_qubits}
    expected = sum(_sizes[b] for b in blocks)
    expected += n_qubits if rx_final else 0
    expected += n_qubits if rz_final else 0
    if theta.size != expected:
        raise ValueError(
            f"θ length {theta.size} != expected {expected} for blocks={blocks} "
            f"(nn={len(nn)}, nnn={len(nnn)}, n_q={n_qubits}, "
            f"rx_final={rx_final}, rz_final={rz_final})")
    gates: list[dict] = []
    o = 0
    layer = 0
    for blk in list(blocks) + (["rx"] if rx_final else []) + (["rz"] if rz_final else []):
        if blk == "nn":
            for e in nn:
                gates.append({"role": "nn", "key": e, "angle": float(theta[o]), "layer": layer}); o += 1
        elif blk == "nnn":
            for e in nnn:
                gates.append({"role": "nnn", "key": e, "angle": float(theta[o]), "layer": layer}); o += 1
        elif blk in ("x", "rx"):
            role = "x"
            for q in range(n_qubits):
                gates.append({"role": role, "key": q, "angle": float(theta[o]), "layer": layer}); o += 1
            if blk == "x":
                layer += 1  # an interior x block closes a Trotter layer
        elif blk in ("z", "rz"):
            for q in range(n_qubits):
                gates.append({"role": "z", "key": q, "angle": float(theta[o]), "layer": layer}); o += 1
        else:
            raise ValueError(f"unknown block '{blk}'")
    if o != theta.size:
        raise ValueError(f"θ length {theta.size} != consumed {o} for blocks={blocks}")
    return gates


def _spec_zz_sign(gates: list[dict]) -> float:
    """Z2 canonical sign: +1 if the summed nn-ZZ angle is ≤ 0, else -1 (to flip)."""
    s = sum(g["angle"] for g in gates if g["role"] == "nn")
    return -1.0 if s > 0 else 1.0


def compare_compressed_circuits(specs: list[dict], *, min_fidelity: float = 0.9) -> dict:
    """Compare high-fidelity compressed circuits on their SHARED gates.

    Each spec dict is an AnsatzSpec-style record (``blocks``, ``nn_edges``,
    ``nnn_edges``, ``n_qubits``/``n_qubits``, ``theta``, ``rx_final``/``rz_final``,
    ``fidelity``, ``name``). Only specs with ``fidelity >= min_fidelity`` and a
    stored ``theta`` enter. Every θ is decoded to role+edge-tagged angles,
    Z2-canonicalized (ZZ sign) and π-wrapped so gauge copies align, then angles
    are grouped by ``(role, key)`` across circuits.

    Returns, per shared gate, the across-circuit mean/std of the angle, plus
    per-role aggregates: how many gates are SHARED by all circuits, the fraction
    that are "invariant" (std < ``invariant_tol``), and the mean dispersion. This
    reveals which gate angles the different compressions agree on (predictable,
    good warm-start targets) vs which drift freely (compression slack). Pure.
    """
    invariant_tol = 0.15
    decoded = []
    for s in specs:
        if (s.get("fidelity") or 0.0) < min_fidelity or not s.get("theta"):
            continue
        nq = int(s.get("n_qubits") or s.get("N"))
        gates = decompose_spec_theta(
            s["theta"], s["blocks"], s["nn_edges"], s["nnn_edges"], nq,
            rx_final=bool(s.get("rx_final", False)),
            rz_final=bool(s.get("rz_final", False)))
        sign = _spec_zz_sign(gates)
        for g in gates:
            a = g["angle"] * (sign if g["role"] in ("nn", "nnn") else 1.0)
            # π-wrap (ZZ and RX are π-periodic in θ)
            a = (a + np.pi / 2.0) % np.pi - np.pi / 2.0
            g["angle"] = a
        decoded.append({"name": s.get("name", "?"), "fidelity": s.get("fidelity"),
                        "gates": gates})

    if len(decoded) < 2:
        return {"n_circuits": len(decoded), "error": "need >=2 high-fidelity specs with theta"}

    # group angles by (role, key, layer) across circuits
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)
    present: dict[tuple, int] = defaultdict(int)
    for d in decoded:
        seen = set()
        for g in d["gates"]:
            gk = (g["role"], g["key"], g["layer"])
            groups[gk].append(g["angle"])
            if gk not in seen:
                present[gk] += 1
                seen.add(gk)

    n = len(decoded)
    shared = {gk: v for gk, v in groups.items() if present[gk] == n}
    per_role: dict[str, dict] = {}
    gate_rows = []
    for role in ("nn", "nnn", "x", "z"):
        stds, means = [], []
        n_inv = 0
        role_gk = [gk for gk in shared if gk[0] == role]
        for gk in role_gk:
            arr = np.array(shared[gk], float)
            st = float(arr.std()); mn = float(arr.mean())
            stds.append(st); means.append(mn)
            if st < invariant_tol:
                n_inv += 1
            gate_rows.append({"role": role, "key": gk[1], "layer": gk[2],
                              "mean": mn, "std": st,
                              "invariant": bool(st < invariant_tol)})
        if role_gk:
            per_role[role] = {
                "n_shared": len(role_gk),
                "frac_invariant": float(n_inv / len(role_gk)),
                "mean_dispersion": float(np.mean(stds)),
                "angle_mean": float(np.mean(means)),
                "angle_absmean": float(np.mean(np.abs(means))),
            }
    return {
        "n_circuits": n,
        "circuits": [{"name": d["name"], "fidelity": d["fidelity"]} for d in decoded],
        "n_shared_gates": len(shared),
        "invariant_tol": invariant_tol,
        "per_role": per_role,
        "gates": gate_rows,
    }


def angle_spectrum_by_role(theta, blocks, nn_edges, nnn_edges, n_qubits, *,
                           rx_final: bool = False, rz_final: bool = False) -> dict:
    """Per-role sorted |θ| spectrum (gauge-canonical) — the N-comparable fingerprint.

    Reduces a (possibly compressed) circuit's θ to, for each role, the descending
    sorted array of per-gate ``|angle|`` (max over layers), plus summary
    percentiles. Because it is sorted and intensive, the spectrum can be compared
    ACROSS N (different gate counts) to test whether the angle structure is
    N-invariant — the signature of a scalable construction rule. Z2-canonical +
    π-wrapped. Pure.

    Returns ``{role: {"spectrum": np.ndarray, "mean", "p50", "p75", "max",
    "n_active"}}`` where ``n_active`` counts gates with |angle| > 0.1·max (the
    structurally relevant ones).
    """
    gates = decompose_spec_theta(theta, blocks, nn_edges, nnn_edges, n_qubits,
                                 rx_final=rx_final, rz_final=rz_final)
    sign = _spec_zz_sign(gates)
    by_role: dict[str, dict] = {}
    agg: dict[str, dict] = {}
    for g in gates:
        a = g["angle"] * (sign if g["role"] in ("nn", "nnn") else 1.0)
        a = abs((a + np.pi / 2.0) % np.pi - np.pi / 2.0)
        agg.setdefault(g["role"], {}).setdefault(g["key"], []).append(a)
    for role, keymap in agg.items():
        spec = np.array(sorted((max(v) for v in keymap.values()), reverse=True))
        if spec.size == 0:
            continue
        mx = float(spec.max())
        by_role[role] = {
            "spectrum": spec,
            "mean": float(spec.mean()),
            "p50": float(np.percentile(spec, 50)),
            "p75": float(np.percentile(spec, 75)),
            "max": mx,
            "n_active": int((spec > 0.1 * mx).sum()),
            "n_total": int(spec.size),
        }
    return by_role
