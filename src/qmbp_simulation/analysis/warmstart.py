"""Analytic warm-start seeds for the bond-resolved frustrated HVA ansatz.

Experiment-agnostic, pure-numpy core promoted out of the ``vl_vs_hva`` study
scripts so it is testable without importing from ``scripts/`` (repo testing
rule) and reusable by any runner that seeds a VQE on this ansatz.

Parameter layout (per layer, repeated ``p_layers`` times)::

    [theta_nn (n_nn edges), theta_nnn (n_nnn edges), theta_x (n_qubits)]

The circuit applies ``rzz(2*theta)`` on each edge and ``rx(2*theta)`` on each
qubit, so the analytic angles below already fold in that factor of two — the
``1/(4h)`` (not ``1/(2h)``) coefficient is the leading-order Trotter/adiabatic
form for the ``rzz(2*theta)`` convention. Changing the gate convention would
change these formulas.

Two seeds are provided:

- :func:`first_order_warmstart_theta` — exact in the ``h -> 0`` / ``h -> inf``
  limits, first order in between. The reliable general seed.
- :func:`second_order_warmstart_theta` — adds the Trotter/BCH ``(J/2h)^2``
  corrections (ZZ-angle shrink + transverse-field curvature). Confirmed strong
  ONLY in a narrow window around the frustrated transition; it must be
  regime-gated via :func:`second_order_regime_gate` (it *hurts* deep in the
  ordered phase, where the shrink over-corrects).

See ``results/hva_vl_study/REPORT_HEURISTIC_warmstart_restarts.md`` for the
multi-seed calibration behind the coefficients and the gate window.
"""

from __future__ import annotations

import numpy as np

# Calibrated second-order coefficients (report section Q3/P4). The shrink
# coefficient sits at the broad optimum c ~ 1/3 (fid ~0.957 at h=0.5); the
# curvature coefficient is 1/6. Either correction alone recovers most of the
# gain (they are nearly redundant), so both defaults are exposed for override.
DEFAULT_SHRINK_COEF: float = 1.0 / 3.0
DEFAULT_CURV_COEF: float = 1.0 / 6.0

# Confirmed regime where the second-order warm-start helps (report section P3).
# Below this it over-shrinks and hurts (h=0.3: -0.166); above it is neutral.
SECOND_ORDER_H_MIN: float = 0.4
SECOND_ORDER_H_MAX: float = 0.6


def _layer_layout(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int) -> int:
    """Parameters per layer for the bond-resolved (+NNN) HVA ansatz."""
    if min(n_nn, n_nnn, n_qubits, p_layers) < 0 or p_layers == 0:
        raise ValueError(f"invalid layout: n_nn={n_nn} n_nnn={n_nnn} n_qubits={n_qubits} p_layers={p_layers}")
    return n_nn + n_nnn + n_qubits


def first_order_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
) -> np.ndarray:
    """Leading-order Trotter/adiabatic warm-start for the bond-resolved HVA.

    Per layer, with layout ``[theta_nn, theta_nnn, theta_x]``:

        theta_nn  = -J  / (4h)
        theta_nnn = -J2 / (4h)
        theta_x   = arctan(J / h)

    Exact in the ``h -> 0`` and ``h -> inf`` limits, first order in between.
    Returns a flat array of length ``(n_nn + n_nnn + n_qubits) * p_layers``.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    per = _layer_layout(n_nn, n_nnn, n_qubits, p_layers)
    theta = np.zeros(per * p_layers)
    theta_nn = -J / (4 * h)
    theta_nnn = -J2 / (4 * h)
    theta_x = np.arctan(J / h)
    for layer in range(p_layers):
        o = layer * per
        theta[o : o + n_nn] = theta_nn
        theta[o + n_nn : o + n_nn + n_nnn] = theta_nnn
        theta[o + n_nn + n_nnn : o + per] = theta_x
    return theta


def second_order_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    shrink_coef: float = DEFAULT_SHRINK_COEF,
    curv_coef: float = DEFAULT_CURV_COEF,
) -> np.ndarray:
    """Frustration-aware second-order warm-start (Trotter/BCH corrections).

    Extends :func:`first_order_warmstart_theta` with two ``(J/2h)^2`` corrections
    (report section Q3):

        shrink = 1 - (J/2h)^2 * shrink_coef        # applied to ZZ angles
        theta_nn  = (-J  / 4h) * shrink
        theta_nnn = (-J2 / 4h) * shrink
        theta_x   = arctan(J/h) * (1 - (J/2h)^2 * curv_coef)

    Near the frustrated transition (``h ~ 0.5``) this opens a better basin than
    both the first-order seed and Metropolis basin-hopping, converging with a
    single optimization. It is NOT universal — gate it with
    :func:`second_order_regime_gate` (deep in the ordered phase the shrink
    over-corrects and *hurts*). Same layout/length as the first-order seed.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    per = _layer_layout(n_nn, n_nnn, n_qubits, p_layers)
    theta = np.zeros(per * p_layers)
    a_nn = J / (4 * h)
    a_nnn = J2 / (4 * h)
    shrink = 1.0 - (J / (2 * h)) ** 2 * shrink_coef
    theta_x = np.arctan(J / h) * (1.0 - (J / (2 * h)) ** 2 * curv_coef)
    for layer in range(p_layers):
        o = layer * per
        theta[o : o + n_nn] = -a_nn * shrink
        theta[o + n_nn : o + n_nn + n_nnn] = -a_nnn * shrink
        theta[o + n_nn + n_nnn : o + per] = theta_x
    return theta


def second_order_nn_shrink_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    nn_extra_shrink: float = 0.4,
    shrink_coef: float = DEFAULT_SHRINK_COEF,
    curv_coef: float = DEFAULT_CURV_COEF,
) -> np.ndarray:
    """Second-order seed with a STRONGER shrink applied to θ_nn only.

    Motivated by the E1 seed-vs-optimum study (``seed_vs_optimum_square.json``):
    across the high-fidelity configs the optimum consistently uses SMALLER nn ZZ
    angles than the analytic seed (``|opt_nn|/|seed_nn|`` ≈ 0.3–0.5), while θ_nnn
    is erratic and θ_x is already about right. This variant keeps the standard
    second-order θ_nnn and θ_x but multiplies θ_nn by ``nn_extra_shrink`` (default
    0.4, the middle of the measured range), starting the optimizer closer to the
    observed optimum without touching the terms the seed already gets right.

    ``nn_extra_shrink=1.0`` reproduces the standard second-order seed exactly.
    Same layout/length as :func:`second_order_warmstart_theta`.
    """
    theta = second_order_warmstart_theta(
        n_nn,
        n_nnn,
        n_qubits,
        p_layers,
        h,
        J=J,
        J2=J2,
        shrink_coef=shrink_coef,
        curv_coef=curv_coef,
    )
    per = _layer_layout(n_nn, n_nnn, n_qubits, p_layers)
    for layer in range(p_layers):
        o = layer * per
        theta[o : o + n_nn] *= nn_extra_shrink
    return theta


def variant_warmstart_theta(
    blocks: list[str],
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    rx_final: bool = False,
    rz_final: bool = False,
    shrink_coef: float = DEFAULT_SHRINK_COEF,
    curv_coef: float = DEFAULT_CURV_COEF,
) -> np.ndarray:
    """Second-order warm-start for an ARBITRARY block sequence (ansatz variants).

    Structure variants (``create_bond_resolved_frustrated_configurable``) declare
    an arbitrary block order like ``["nn","nnn","x","nn","x"]``; the standard
    per-layer seed does not fit them. This fills each block with its analytic
    second-order angle, in the exact parameter order the configurable circuit
    uses (blocks in sequence, then rx_final, then rz_final):

        nn  → (-J  / 4h) * shrink
        nnn → (-J2 / 4h) * shrink
        x   → arctan(J/h) * (1 - (J/2h)^2 * curv_coef)
        z   → 0 (symmetry-breaking block; neutral seed)

    Trailing rx_final/rz_final blocks are seeded at 0 (identity) so they only
    help if the optimizer moves them. Gives every variant the SAME informed
    starting point the base ansatz gets — the robust seed for a fair comparison.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    a_nn = (J / (4 * h)) * (1.0 - (J / (2 * h)) ** 2 * shrink_coef)
    a_nnn = (J2 / (4 * h)) * (1.0 - (J / (2 * h)) ** 2 * shrink_coef)
    theta_x = np.arctan(J / h) * (1.0 - (J / (2 * h)) ** 2 * curv_coef)
    block_val = {"nn": -a_nn, "nnn": -a_nnn, "x": theta_x, "z": 0.0}
    block_size = {"nn": n_nn, "nnn": n_nnn, "x": n_qubits, "z": n_qubits}

    parts: list[np.ndarray] = []
    for b in blocks:
        if b not in block_val:
            raise ValueError(f"unknown block {b!r}; expected nn|nnn|x|z")
        parts.append(np.full(block_size[b], block_val[b], dtype=float))
    if rx_final:
        parts.append(np.zeros(n_qubits))
    if rz_final:
        parts.append(np.zeros(n_qubits))
    return np.concatenate(parts) if parts else np.zeros(0)


def compose_extend_theta(
    base_theta: np.ndarray,
    extra_blocks: list[str],
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    extra_rx_final: bool = False,
    extra_rz_final: bool = False,
    shrink_coef: float = DEFAULT_SHRINK_COEF,
    curv_coef: float = DEFAULT_CURV_COEF,
) -> np.ndarray:
    """Warm-start for an EXTENDED variant: known base θ + analytic tail.

    When a variant's block sequence is a base variant's sequence followed by
    ``extra_blocks`` (e.g. p2 + a partial third layer), the best possible seed is
    the base variant's already-optimized θ for the shared prefix, plus the
    analytic second-order angles for the extra blocks. This starts the optimizer
    from the best known near-ground-state point and only asks it to tune the new
    layer — the strongest warm-start for these structure extensions.

    The tail (extra blocks) is seeded at **near-identity (all zeros)**, NOT the
    analytic angles: an RZZ block at its analytic angle strongly perturbs the
    already-optimized base state (pushing fidelity to ~0), whereas zeros make the
    extra layer start as the identity so the initial state equals the base state
    (its known fidelity) and the optimizer only has to *improve* from there — the
    correct warm-start for a structure extension. ``J``/``J2``/``shrink``/``curv``
    are accepted for signature symmetry but unused (tail is zeros).
    """
    n_extra = 0
    size = {"nn": n_nn, "nnn": n_nnn, "x": n_qubits, "z": n_qubits}
    for b in extra_blocks:
        if b not in size:
            raise ValueError(f"unknown block {b!r}; expected nn|nnn|x|z")
        n_extra += size[b]
    if extra_rx_final:
        n_extra += n_qubits
    if extra_rz_final:
        n_extra += n_qubits
    return np.concatenate([np.asarray(base_theta, float), np.zeros(n_extra)])


def _block_slices(n_nn: int, n_nnn: int, n_qubits: int, layer: int):
    """(nn, nnn, x) index slices for a given layer in the standard layout."""
    per = n_nn + n_nnn + n_qubits
    o = layer * per
    return (slice(o, o + n_nn), slice(o + n_nn, o + n_nn + n_nnn), slice(o + n_nn + n_nnn, o + per))


def seed_vs_optimum_report(
    theta_opt: np.ndarray,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
) -> dict:
    """Rigorous per-layer per-block comparison: analytic seed vs optimized θ.

    For one configuration, contrasts what the analytic second-order seed PREDICTS
    against where the highest-fidelity optimization ARRIVED, block by block and
    layer by layer. Controls for the ZZ sign-degeneracy (RZZ(2θ) ~ RZZ(-2θ) up to
    a state phase on this ansatz family) by reporting BOTH signed means and
    magnitude means, so a sign flip is not mistaken for a magnitude change.

    Returns a dict with, per layer and block:
    - ``seed`` (the uniform analytic value),
    - ``opt_mean`` / ``opt_absmean`` / ``opt_std`` (optimized distribution),
    - ``ratio_abs`` = |opt|.mean / |seed| (magnitude change, sign-safe),
    - ``sign_flip_frac`` (fraction of angles whose sign differs from the seed),
    plus global ``distance`` = ‖θ_opt − θ_seed‖ and ‖θ_seed‖, ‖θ_opt‖.
    """
    theta_opt = np.asarray(theta_opt, float)
    seed = second_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    if len(theta_opt) != len(seed):
        raise ValueError(
            f"theta length {len(theta_opt)} != seed length {len(seed)} "
            f"(n_nn={n_nn} n_nnn={n_nnn} n_qubits={n_qubits} p={p_layers})"
        )

    out: dict = {
        "h": round(float(h), 2),
        "p_layers": p_layers,
        "n_nn": n_nn,
        "n_nnn": n_nnn,
        "n_qubits": n_qubits,
        "distance": float(np.linalg.norm(theta_opt - seed)),
        "norm_seed": float(np.linalg.norm(seed)),
        "norm_opt": float(np.linalg.norm(theta_opt)),
        "layers": [],
    }
    for layer in range(p_layers):
        sl_nn, sl_nnn, sl_x = _block_slices(n_nn, n_nnn, n_qubits, layer)
        layer_rec = {}
        for name, sl in (("nn", sl_nn), ("nnn", sl_nnn), ("x", sl_x)):
            seed_val = float(seed[sl][0]) if seed[sl].size else 0.0
            opt = theta_opt[sl]
            abs_seed = abs(seed_val)
            layer_rec[name] = {
                "seed": seed_val,
                "opt_mean": float(opt.mean()),
                "opt_absmean": float(np.abs(opt).mean()),
                "opt_std": float(opt.std()),
                "ratio_abs": (float(np.abs(opt).mean() / abs_seed) if abs_seed > 1e-12 else None),
                "sign_flip_frac": (float(np.mean(np.sign(opt) != np.sign(seed_val))) if abs_seed > 1e-12 else None),
            }
        out["layers"].append(layer_rec)
    return out


def second_order_correction_magnitude(h: float, *, J: float = 1.0, shrink_coef: float = DEFAULT_SHRINK_COEF) -> float:
    """The scalar ``(J/2h)^2 * shrink_coef`` correction magnitude at ``h``.

    Grows as ``h`` falls; used both to calibrate the regime gate and to record
    per-h provenance. At the gate window edges (h=0.4..0.6, J=1, c=1/3) this is
    roughly 0.52 .. 0.23.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    return (J / (2 * h)) ** 2 * shrink_coef


def second_order_regime_gate(h: float, *, h_min: float = SECOND_ORDER_H_MIN, h_max: float = SECOND_ORDER_H_MAX) -> bool:
    """Selection rule: is the second-order warm-start expected to help at ``h``?"""
    return h_min <= h <= h_max


def aggregate_seed_stats(fid_first: list[float] | np.ndarray, fid_second: list[float] | np.ndarray) -> dict:
    """Aggregate per-seed first- vs second-order fidelities into mean/std/sep.

    Returns the multi-seed statistics used by the confirmation study: means,
    stds, the mean delta, and the separation in sigma units

        sep = (mean_2 - mean_1) / sqrt(std_1^2 + std_2^2)

    plus a boolean ``second_order_helps`` (delta > 0.01 AND sep > 1sigma),
    matching the report's significance rule. Pure math — no I/O.
    """
    a1 = np.asarray(fid_first, dtype=float)
    a2 = np.asarray(fid_second, dtype=float)
    if a1.size == 0 or a2.size == 0:
        raise ValueError("aggregate_seed_stats requires non-empty inputs")
    sep = float((a2.mean() - a1.mean()) / np.sqrt(a1.std() ** 2 + a2.std() ** 2 + 1e-12))
    delta = float(a2.mean() - a1.mean())
    return {
        "n_seeds": int(min(a1.size, a2.size)),
        "fid1_mean": float(a1.mean()),
        "fid1_std": float(a1.std()),
        "fid2_mean": float(a2.mean()),
        "fid2_std": float(a2.std()),
        "delta_mean": delta,
        "separation_sigma": sep,
        "second_order_helps": bool(delta > 0.01 and sep > 1.0),
    }
