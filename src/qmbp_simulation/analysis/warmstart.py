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
    zz_coef: float = 0.25,
) -> np.ndarray:
    """Leading-order Trotter/adiabatic warm-start for the bond-resolved HVA.

    Per layer, with layout ``[theta_nn, theta_nnn, theta_x]``:

        theta_nn  = -J  * zz_coef / h
        theta_nnn = -J2 * zz_coef / h
        theta_x   = arctan(J / h)

    ``zz_coef`` is the ZZ-angle prefactor. The naive leading-order value is
    ``1/4`` (giving the textbook ``-J/(4h)``); an empirically renormalized value
    (~0.11 for the bond-resolved frustrated square, per the θ-pattern study) can
    be passed to test a better-calibrated seed. Default 0.25 preserves the exact
    original behavior. Exact in the ``h -> 0`` / ``h -> inf`` limits, first order
    in between.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    per = _layer_layout(n_nn, n_nnn, n_qubits, p_layers)
    theta = np.zeros(per * p_layers)
    theta_nn = -J * zz_coef / h
    theta_nnn = -J2 * zz_coef / h
    theta_x = np.arctan(J / h)
    for layer in range(p_layers):
        o = layer * per
        theta[o : o + n_nn] = theta_nn
        theta[o + n_nn : o + n_nn + n_nnn] = theta_nnn
        theta[o + n_nn + n_nnn : o + per] = theta_x
    return theta


# ── Calibrated analytic warm-start (square frustrated J2=0.5) ─────────────────
# zz_coef(h) and x_scale(h) fitted to MAXIMIZE the raw init-fidelity (no
# optimization) of the first-order seed, from a 2D (zz_coef × x_scale) scan at
# N=10 p=2. The default seed (zz=0.25, x_scale=1) has near-zero init-fid in the
# ordered/transition region; these calibrated coefficients lift it by up to ~20×
# (e.g. h=0.5: 0.009 → 0.19 raw). Calibrated on N=10, validated cross-N.
_CAL_ZZ_A: float = 0.103  # tanh baseline
_CAL_ZZ_B: float = 0.036  # tanh amplitude
_CAL_ZZ_HC: float = 1.17  # tanh center (zz rises toward the paramagnet)
_CAL_ZZ_W: float = 0.24  # tanh width
_CAL_XS_BREAK: float = 1.0  # x_scale regime break (ordered/transition vs paramagnet)


def calibrated_zz_coef(h: float) -> float:
    """Calibrated ZZ prefactor zz_coef(h) maximizing the raw-seed init-fidelity.

    Smooth tanh in h: ~0.07 in the ordered/transition region, rising to ~0.14
    deep in the paramagnet. Replaces the textbook 0.25 (which gives near-zero
    init-fid near h_c). Pure, bounded to (0, 0.25].
    """
    val = _CAL_ZZ_A + _CAL_ZZ_B * np.tanh((float(h) - _CAL_ZZ_HC) / _CAL_ZZ_W)
    return float(np.clip(val, 1e-3, 0.25))


def calibrated_x_scale(h: float, *, h_break: float = _CAL_XS_BREAK) -> float:
    """Calibrated θ_x multiplier x_scale(h) maximizing the raw-seed init-fidelity.

    Piecewise-linear with a regime break at ``h_break`` (≈1.0): below it the
    transverse-field angle must be AMPLIFIED as h rises toward the paramagnet
    (x_scale ~1.0 → ~1.55); above it the analytic arctan(J/h) is already close,
    starting near ~0.85 and drifting up slowly. The break reflects a real regime
    change (not noise): the θ_x correction flips sign across h≈1. Pure, clipped.
    """
    h = float(h)
    if h <= h_break:
        xs = 1.0 + 0.70 * (h - 0.3)  # ~1.0 at h=0.3 → ~1.5 at h=1.0
    else:
        xs = 0.85 + 0.37 * (h - h_break)  # ~0.85 at h=1.0 → rises slowly
    return float(np.clip(xs, 0.7, 1.6))


def calibrated_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
) -> np.ndarray:
    """Analytic warm-start with CALIBRATED zz_coef(h) and x_scale(h).

    Builds the first-order seed with :func:`calibrated_zz_coef` for the ZZ angles
    and multiplies the θ_x block by :func:`calibrated_x_scale` — the two knobs the
    2D scan showed matter for the raw init-fidelity. This is the best PURELY
    analytic seed for this model/topology: it maximizes how close the un-optimized
    state sits to the ground state (strong in the paramagnet, a better launch
    point near h_c). Calibrated on N=10 p=2 J2=0.5; validated cross-N. Same layout
    and length as :func:`first_order_warmstart_theta`.
    """
    zz = calibrated_zz_coef(h)
    xs = calibrated_x_scale(h)
    theta = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, zz_coef=zz)
    per = n_nn + n_nnn + n_qubits
    for layer in range(p_layers):
        o = layer * per + n_nn + n_nnn
        theta[o : o + n_qubits] *= xs
    return np.clip(theta, -np.pi, np.pi)


# Structural seed: the N-invariant θ_x constant + sparse ZZ blocks, from the
# cross-N angle-spectrum study (θ_x spectrum is N-invariant ~arctan(J/h)·x_scale;
# θ_nn/θ_nnn are SPARSE — a few large bonds, the rest ~0, with the nnn median
# near zero). Fraction of nnn kept "active" (the rest suppressed toward 0).
STRUCTURAL_NNN_SUPPRESS: float = 0.35


def structural_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    nnn_suppress: float = STRUCTURAL_NNN_SUPPRESS,
) -> np.ndarray:
    """Analytic seed built from the measured ANGLE STRUCTURE (no donor needed).

    From the cross-N spectrum study (mem: angle-structure):
    - **θ_x** is strongly N-invariant and sits at ``arctan(J/h)·x_scale(h)`` for
      every qubit — a near-universal constant. We use the calibrated value.
    - **θ_nn / θ_nnn** are SPARSE: a few large bonds carry the correlation and the
      rest (especially nnn, whose median is ~0) are near zero. Without a donor we
      cannot know WHICH bonds are large, so we encode the sparsity by SUPPRESSING
      the nnn block toward zero (``nnn_suppress`` × the calibrated nnn), keeping
      the calibrated nn magnitude. This matches the measured shape (nnn mostly
      inactive) instead of the uniform ``-J2/4h`` that over-populates every nnn.

    Same layout/length as :func:`first_order_warmstart_theta`. Pure. Its value is
    as an extra cascade candidate: where the uniform nnn hurts (near h_c), the
    suppressed-nnn structural seed can fall into a cleaner basin.
    """
    zz = calibrated_zz_coef(h)
    xs = calibrated_x_scale(h)
    theta = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, zz_coef=zz)
    per = n_nn + n_nnn + n_qubits
    for layer in range(p_layers):
        o = layer * per
        # suppress nnn toward 0 (sparsity: most nnn are inactive)
        theta[o + n_nn : o + n_nn + n_nnn] *= nnn_suppress
        # θ_x constant, calibrated
        theta[o + n_nn + n_nnn : o + per] *= xs
    return np.clip(theta, -np.pi, np.pi)


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


# Renormalized flat-seed ZZ prefactor for the ordered phase (θ-pattern study):
# θ_nn = -0.11/h works better than the textbook -0.25/h deep in the ordered
# phase, where the second-order shrink over-corrects.
FLAT_RENORM_ZZ_COEF: float = 0.11
# Regime boundaries (square frustrated J2=0.5). Below ORDERED_H_MAX use the flat
# renormalized seed; inside the second-order window use second-order; well into
# the paramagnet use the nn-shrink variant.
ORDERED_H_MAX: float = 0.45
PARAMAGNETIC_H_MIN: float = 1.2


def select_regime_seed(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    use_calibrated: bool = False,
) -> tuple[np.ndarray, str]:
    """Pick the sharpest warm-start seed for the field ``h`` (regime-aware).

    Encodes the calibrated seed-by-regime table for the frustrated square
    (J2=0.5), returning ``(theta_seed, seed_name)``:

    - ``h <= ORDERED_H_MAX`` (ordered phase) → **flat renorm**: first-order seed
      with the renormalized ZZ prefactor ``FLAT_RENORM_ZZ_COEF`` (θ_nn=-0.11/h),
      which beats the textbook -0.25/h and the over-shrinking second-order seed
      deep in the ordered phase.
    - ``ORDERED_H_MAX < h < PARAMAGNETIC_H_MIN`` (near h_c) → **second-order**:
      the Trotter/BCH shrink + transverse-field curvature, strong in this window.
    - ``h >= PARAMAGNETIC_H_MIN`` (paramagnet) → **so_nn_shrink**: second-order
      with an extra 0.4× shrink on θ_nn (the optimum uses smaller nn angles here).

    ``use_calibrated=True`` overrides the regime table with the continuously
    calibrated seed :func:`calibrated_warmstart_theta` (zz_coef(h) + x_scale(h)),
    which has a much higher raw init-fidelity across all phases. Off by default
    so the established regime-gated behavior and its tests are preserved.

    Pure; returns the seed vector (standard layout) and its name for provenance.
    """
    if use_calibrated:
        return (calibrated_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2), "calibrated")
    if h <= ORDERED_H_MAX:
        seed = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, zz_coef=FLAT_RENORM_ZZ_COEF)
        return seed, "flat_renorm(zz_coef=0.11)"
    if h < PARAMAGNETIC_H_MIN:
        seed = second_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
        return seed, "second_order"
    seed = second_order_nn_shrink_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, nn_extra_shrink=0.4)
    return seed, "so_nn_shrink(0.4)"


# Gap below which the optimization landscape is near-degenerate (ordered phase
# deep / close to h_c). From the θ-scaling study: d(θ_opt, regime_seed) grows
# with N here, so the budget must scale and fewer params converge cleaner.
SMALL_GAP: float = 1e-2


# Critical field of the frustrated square (J2=0.5): the difficulty study placed
# the hard regime at h~0.5 (dfid/dN steepest, d(θ,seed) grows with N there),
# while h=0.3 (ordered) and h=1.3 (paramagnet) are easy at any N. Proximity is
# therefore peaked at H_CRITICAL, not spread over the whole second-order window.
H_CRITICAL: float = 0.5
# Half-width (in h) of the hard region around h_c. Beyond ~this distance the
# phase is "easy" (seed suffices): h=0.3 and h=1.3 are ~0.8 away → weight ~0.
TRANSITION_HALF_WIDTH: float = 0.35


def phase_proximity(h: float, *, h_c: float = H_CRITICAL, half_width: float = TRANSITION_HALF_WIDTH) -> float:
    """Weight in [0, 1] for how close ``h`` is to the frustrated transition h_c.

    The difficulty study found the hard regime is concentrated AROUND h_c≈0.5
    (steepest dfid/dN, d(θ,seed) growing with N), while the ordered (h≈0.3) and
    paramagnetic (h≈1.3) phases are easy at ANY N. This is a triangular bump
    peaked at ``h_c`` (weight 1) decaying linearly to 0 at ``h_c ± half_width``,
    so the easy phases get weight ~0 and only near-h_c points accrue difficulty.
    Pure.
    """
    d = abs(float(h) - h_c)
    return float(max(0.0, 1.0 - d / half_width))


def _phase_of(h: float) -> int:
    """Phase bucket for the square-frustrated J2=0.5 line: 0 ordered, 1 near-h_c, 2 paramagnet."""
    if h <= ORDERED_H_MAX:
        return 0
    if h < PARAMAGNETIC_H_MIN:
        return 1
    return 2


def _npz_donor_candidates(topology, n_qubits, h, p_layers, *, model, frustrated, same_phase, max_de_gap, root):
    """(|ΔN|, |Δh|, donor) candidates from the NPZ training corpus.

    Reads ``data/multi_n_training/<model>/[frustrated/]{topology}_N*_p{p}.npz``
    (the data the pipeline itself generates). For each OTHER-N file, picks the
    single best point (closest h, lowest ΔE/gap ≤ ``max_de_gap``, same phase when
    ``same_phase``) as a donor dict. Pure (NPZ + lattice reads only).
    """
    import numpy as np

    from qmbp_simulation.framework.result_io import TRAINING_DATA_ROOT, training_npz_read_globs
    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    _root = root if root is not None else TRAINING_DATA_ROOT
    tgt_phase = _phase_of(h)
    out: list[tuple[int, float, dict]] = []
    for d, pattern in training_npz_read_globs(topology, p_layers, model=model, frustrated=frustrated, root=_root):
        if not d.exists():
            continue
        for npz in sorted(d.glob(pattern)):
            try:
                dn = int(npz.stem.split("_N")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if dn == n_qubits:
                continue  # exclude self-N
            try:
                data = np.load(npz, allow_pickle=True)
            except Exception:
                continue
            hs = np.asarray(data.get("h_values", []), dtype=np.float64)
            th = data.get("theta_opt")
            if hs.size == 0 or th is None or len(th) == 0:
                continue
            de = np.asarray(data["de_gaps"], dtype=np.float64) if "de_gaps" in data else np.full(hs.shape, np.inf)
            best_idx, best_key = None, None
            for i in range(len(hs)):
                if same_phase and _phase_of(float(hs[i])) != tgt_phase:
                    continue
                if np.isfinite(de[i]) and de[i] > max_de_gap:
                    continue
                key = (abs(float(hs[i]) - h), float(de[i]) if np.isfinite(de[i]) else 1e9)
                if best_key is None or key < best_key:
                    best_key, best_idx = key, i
            if best_idx is None:
                continue
            theta_i = np.asarray(th[best_idx], dtype=np.float64)
            if not np.all(np.isfinite(theta_i)):
                continue
            lat = make_lattice(topology, dn, J=1.0, h=float(hs[best_idx]))
            nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat) if frustrated else []
            n_nn_d = len(lat.edges)
            per = len(theta_i) // max(p_layers, 1)
            n_nnn_d = max(0, per - n_nn_d - dn)
            donor = {
                "theta": theta_i,
                "n_nn": n_nn_d,
                "n_nnn": n_nnn_d,
                "p": p_layers,
                "n_qubits": dn,
                "nnn_edges": nnn_edges,
                "h": float(hs[best_idx]),
                "label": f"npz<N{dn}h{hs[best_idx]:.2f}>",
            }
            out.append((abs(dn - n_qubits), abs(float(hs[best_idx]) - h), donor))
    return out


def _bond_ablation_donor_candidates(topology, n_qubits, h, p_layers, *, same_phase, min_fid, variant, root):
    """(|ΔN|, |Δh|, donor) candidates from the bond-ablation study JSON corpus.

    Reads ``results/hva_vl_study/bond_ablation/bond_topk_regime_{topo}_N*_p{p}_h*.json``
    (the ``vl_vs_hva`` study artifacts). For each OTHER-N file matching the target
    (p, phase), extracts the ``variant`` row's ``best_theta_final`` + fidelity
    (must be ≥ ``min_fid``) as a donor dict with the same shape as the NPZ
    backend. This lets the frustrated study runners reuse the shared discoverer
    while keeping their richer N/h coverage (e.g. N=9/14/18) that the NPZ corpus
    may lack. Pure (JSON + lattice reads only).
    """
    import glob as _glob
    import json as _json
    import re as _re
    from pathlib import Path as _P

    import numpy as np

    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    base = _P(root) if root is not None else _P("results/hva_vl_study/bond_ablation")
    tgt_phase = _phase_of(h)
    out: list[tuple[int, float, dict]] = []
    pat = str(base / f"bond_topk_regime_{topology}_N*_p{p_layers}_h*.json")
    for fp in _glob.glob(pat):
        m = _re.search(rf"_N(\d+)_p{p_layers}_h([0-9.]+)\.json$", _P(fp).name)
        if not m:
            continue
        dn, dh = int(m.group(1)), float(m.group(2))
        if dn == n_qubits and abs(dh - h) < 1e-9:
            continue  # exclude self (N, h)
        if dn == n_qubits:
            continue  # exclude self-N (cross-h at same N handled by continuation donor)
        if same_phase and _phase_of(dh) != tgt_phase:
            continue
        try:
            d = _json.loads(_P(fp).read_text())
        except Exception:
            continue
        row = next((r for r in d.get("rows", []) if r.get("variant") == variant), None)
        if row is None:
            continue
        th = row.get("best_theta_final") or row.get("theta_final")
        fid = row.get("best_fidelity")
        if not th or fid is None or fid < min_fid:
            continue
        theta_i = np.asarray(th, dtype=np.float64)
        if not np.all(np.isfinite(theta_i)):
            continue
        lat = make_lattice(topology, dn, J=1.0, h=dh)
        nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
        donor = {
            "theta": theta_i,
            "n_nn": int(row.get("n_nn_bonds", len(lat.edges))),
            "n_nnn": int(row.get("n_nnn_bonds", len(nnn_edges))),
            "p": p_layers,
            "n_qubits": dn,
            "nnn_edges": nnn_edges,
            "h": dh,
            "label": f"ablation<N{dn}h{dh:.2f}>@{fid:.3f}",
        }
        out.append((abs(dn - n_qubits), abs(dh - h), donor))
    return out


def discover_donors(
    topology: str,
    n_qubits: int,
    h: float,
    p_layers: int,
    *,
    model: str = "tfim_bond_resolved",
    frustrated: bool = False,
    max_donors: int = 4,
    same_phase: bool = True,
    max_de_gap: float = 0.10,
    sources: tuple[str, ...] = ("npz",),
    npz_root=None,
    ablation_root=None,
    ablation_min_fid: float = 0.9,
    ablation_variant: str = "p2_full_ref",
):
    """Shared cross-N / cross-h warm-start donor discoverer (source-pluggable).

    The single ``src``-level donor finder for BOTH the accelerated cross-N runner
    and the frustrated study runners, replacing the private per-script scanners.
    It unifies the same-phase, cross-N/cross-h selection logic and can draw from
    one or more corpora via ``sources``:

    - ``"npz"``: the canonical training corpus
      (``data/multi_n_training/<model>/[frustrated/]{topology}_N*_p{p}.npz``) — the
      data the pipeline itself generates, so the loop is self-reinforcing.
    - ``"bond_ablation"``: the ``vl_vs_hva`` study JSON artifacts
      (``results/hva_vl_study/bond_ablation/bond_topk_regime_*``), which carry
      richer N/h coverage (e.g. N=9/14/18) than the NPZ corpus.

    Candidates from all requested sources are merged, de-duplicated by (N, h)
    [NPZ wins ties — it is the verified corpus], ordered by closeness in
    (|ΔN|, |Δh|), and capped at ``max_donors``. Each donor dict is compatible with
    :func:`best_combined_warmstart` / :func:`transfer_theta`:
    ``{theta, n_nn, n_nnn, p, n_qubits, nnn_edges, h, label}``. The target's own N
    is excluded (same-N cross-h is the caller's continuation-donor job).

    Returns a (possibly empty) list of donor dicts. Pure w.r.t. the circuit.
    """
    cands: list[tuple[int, float, dict]] = []
    if "npz" in sources:
        cands.extend(
            _npz_donor_candidates(
                topology,
                n_qubits,
                h,
                p_layers,
                model=model,
                frustrated=frustrated,
                same_phase=same_phase,
                max_de_gap=max_de_gap,
                root=npz_root,
            )
        )
    if "bond_ablation" in sources:
        cands.extend(
            _bond_ablation_donor_candidates(
                topology,
                n_qubits,
                h,
                p_layers,
                same_phase=same_phase,
                min_fid=ablation_min_fid,
                variant=ablation_variant,
                root=ablation_root,
            )
        )

    # Closest-first, with NPZ winning (N, h) ties (sort is stable and NPZ
    # candidates were appended first).
    cands.sort(key=lambda t: (t[0], t[1]))
    seen: set[tuple[int, float]] = set()
    out: list[dict] = []
    for dn, dh, donor in cands:
        key = (int(donor["n_qubits"]), round(float(donor["h"]), 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(donor)
        if len(out) >= max_donors:
            break
    return out


def discover_npz_donors(
    topology: str,
    n_qubits: int,
    h: float,
    p_layers: int,
    *,
    model: str = "tfim_bond_resolved",
    frustrated: bool = False,
    max_donors: int = 4,
    same_phase: bool = True,
    max_de_gap: float = 0.10,
    root=None,
):
    """NPZ-only donor discovery (back-compat alias of :func:`discover_donors`).

    Preserved so existing callers keep their exact behavior (``sources=("npz",)``).
    New callers that also want the bond-ablation study corpus should call
    :func:`discover_donors` with ``sources=("npz", "bond_ablation")``.
    """
    return discover_donors(
        topology,
        n_qubits,
        h,
        p_layers,
        model=model,
        frustrated=frustrated,
        max_donors=max_donors,
        same_phase=same_phase,
        max_de_gap=max_de_gap,
        sources=("npz",),
        npz_root=root,
    )


def difficulty_index(
    n_qubits: int,
    h: float,
    gap: float | None = None,
    *,
    J2: float = 0.0,
    h_c: float = H_CRITICAL,
    half_width: float = TRANSITION_HALF_WIDTH,
) -> float:
    """Scalar difficulty D = N · phase_proximity(h) — the warm-start hardness.

    From the scaling study: fidelity falls with N ONLY near the transition
    (dfid/dN ≈ −0.033 near h_c vs ~0 in the ordered/paramagnetic phases), and
    the raw gap does NOT predict difficulty (corr(fid, log gap) ≈ 0). So the
    hardness is the number of qubits weighted by how close h sits to h_c — NOT
    the gap, and NOT the θ_x/arctan feature (the cheap check showed θ_x only
    tracks arctan in the paramagnet). ``gap`` is accepted for API symmetry and
    optional logging but intentionally not used in the score.

    Returns ~0 in the ordered/paramagnetic phases (seed suffices at any N) and
    grows with N as h approaches the transition. Pure.
    """
    return float(n_qubits) * phase_proximity(h, h_c=h_c, half_width=half_width)


def budget_for_difficulty(
    n_qubits: int,
    h: float,
    gap: float | None = None,
    *,
    base_restarts: int = 1,
    base_frac: float = 0.5,
    J2: float = 0.0,
    max_restarts: int = 4,
) -> tuple[int, float, float]:
    """Map (N, h) difficulty → (restarts, keep_frac, difficulty_index).

    Unifies the gap-adaptive knobs under the calibrated difficulty model:

    - In the TRANSITION window, restarts scale with ``difficulty_index`` (N·prox):
      D>=12 → +2, D>=8 → +1 restart over base (clipped to ``max_restarts``). This
      is where the N=18 collapse lives, so budget goes here.
    - In the ordered/paramagnetic phases (D≈0) restarts stay at base.
    - ``keep_frac`` is only tightened in the near-degenerate small-gap case
      (reuses :func:`topk_frac_for_gap`), which is orthogonal to phase.

    Returns ``(restarts, keep_frac, D)``. Pure. This supersedes calling
    :func:`restarts_for_gap` directly: it is gap-aware for the mask but
    difficulty-aware (not gap-aware) for the restart count, matching the finding
    that the gap alone mis-scales the budget.
    """
    D = difficulty_index(n_qubits, h, gap, J2=J2)
    base = max(1, int(base_restarts))
    if D >= 12:
        restarts = base + 2
    elif D >= 8:
        restarts = base + 1
    else:
        restarts = base
    restarts = min(restarts, max_restarts)
    keep_frac = topk_frac_for_gap(gap, base_frac) if gap is not None else base_frac
    return restarts, keep_frac, D


# Micro-descent iteration budget bounds. The budget-fair diagnostic
# (diagnose_warmstart_budget_fair.py) showed a single full L-BFGS-B is the best
# use of the budget (θ_x best-of / block schemes don't beat it at equal compute),
# and that the OLD default of 12 iters left large fidelity on the table in the
# hard phases (e.g. 0.65→0.99 at h=0.3, 0.60→0.68 at N12 h=0.5 going 12→48 iters).
# So the only genuine lever is spending MORE descent where it is hard. These
# bound the difficulty-adaptive budget.
MICRO_DESCENT_MIN: int = 24
# Raised ceiling: the fast adjoint gradient (8-22× over the Qiskit reverse
# estimator) makes large budgets affordable, and the budget-vs-N calibration
# (calibrate_budget_vs_n.py) showed the transition needs far more than 60 iters
# at large N — N10 h0.5 crosses f>0.9 at ~24 iters, N12 at ~160, N14 at >400.
MICRO_DESCENT_MAX: int = 400

# N below which the easy-phase budget (min_iters) already suffices in the
# transition, and the super-linear scaling anchor + per-(N-N0)² coefficient.
# Calibrated so phase_proximity=1 gives ≈ min + COEF·(N-N0)²:
#   N10→24, N12→~160, N14→~424 (matches the measured crossing budgets).
MICRO_DESCENT_N0: int = 10
MICRO_DESCENT_COEF: float = 25.0

# Phase-INDEPENDENT N floor. The h=1.0 diagnosis (diag_h1) showed that even away
# from h_c (phase_proximity≈0) large N needs more than the 24-iter floor: the raw
# seed quality falls with N (N10 raw 0.68 → N14 0.55) and the micro-descent keeps
# improving to ~60-120 iters (N14 h1.0: 0.949@24 → 0.966@120, then saturates).
# So the floor itself grows linearly with N (independent of phase), capped where
# the h=1.0 curve saturates. Combined additively with the transition N² term.
MICRO_DESCENT_FLOOR_COEF: float = 12.0
MICRO_DESCENT_FLOOR_MAX: int = 120


def micro_descent_budget(
    n_qubits: int,
    h: float,
    gap: float | None = None,
    *,
    J2: float = 0.0,
    min_iters: int = MICRO_DESCENT_MIN,
    max_iters: int = MICRO_DESCENT_MAX,
    n0: int = MICRO_DESCENT_N0,
    coef: float = MICRO_DESCENT_COEF,
    floor_coef: float = MICRO_DESCENT_FLOOR_COEF,
    floor_max: int = MICRO_DESCENT_FLOOR_MAX,
) -> int:
    """Difficulty-adaptive micro-descent budget that SCALES WITH N (ad-hoc N, h).

    Two additive N-scaling terms, calibrated to the measured fidelity-vs-budget
    curves, so the budget tracks genuine difficulty and nothing more::

        budget = floor_N + phase_proximity(h) · coef · (N − n0)²

    1. **Phase-independent N floor** ``floor_N = min_iters + floor_coef·(N − n0)``
       (capped at ``floor_max``). Away from h_c the raw seed still degrades with N
       and the descent keeps gaining to ~60-120 iters (h=1.0 N14: 0.949@24 →
       0.966@120). The floor gives large N enough budget AT ANY h — fixing the
       old behaviour where h=1.0 (``phase_proximity≈0``) was starved at 24 iters.
    2. **Transition N² term** ``phase_proximity(h)·coef·(N − n0)²`` adds the heavy
       budget ONLY near h_c, where the crossing budget grows super-linearly
       (N10→24, N12→~160, N14→>400).

    Small N (≤ ``n0``) stays at ``min_iters`` everywhere. The whole thing is
    clamped to ``max_iters``. Affordable thanks to the fast adjoint gradient
    (8-22× cheaper per iter). Every runner gets it via ``prepare_warmstart``
    (``micro_descent=None``). Pure; returns an int in ``[min_iters, max_iters]``.
    """
    n_excess = max(0, int(n_qubits) - int(n0))
    floor_n = min(float(floor_max), min_iters + floor_coef * n_excess)
    prox = phase_proximity(h)  # 0 away from h_c, 1 at the transition
    budget = floor_n + prox * coef * (n_excess**2)
    return int(round(max(min_iters, min(max_iters, budget))))


# Two-pass selector (M2) pre-rank budget. The selector_budget diagnostic
# (p2_half_nn_rx N18 h0.5) showed the micro-descent ranking only becomes
# TRUSTWORTHY once the budget is long enough: at K=100 it picked the wrong donor
# (donor_N8 0.848) because the true winner (donor_N10) starts slow (0.653) but
# lands in the best basin, revealed only at K>=400 (0.924). So a cheap short pass
# cannot DECIDE, but it can PRE-RANK to drop the obvious losers before the full
# (K>=400) pass settles the winner on the survivors. This fraction of the full
# budget is the short-pass length; the floor keeps it meaningful at small N.
SHORT_DESCENT_FRAC: float = 0.25
SHORT_DESCENT_MIN: int = 24


def short_descent_budget(
    full_iters: int, *, frac: float = SHORT_DESCENT_FRAC, min_iters: int = SHORT_DESCENT_MIN
) -> int:
    """Pre-rank budget for the two-pass selector — a fraction of the full budget.

    The two-pass early-exit (M2) runs this SHORT descent on every candidate to
    pre-rank them, then the full :func:`micro_descent_budget` descent only on the
    top-k survivors. The short pass is deliberately cheap (``frac`` of the full,
    floored at ``min_iters``) — the selector_budget study showed it must not be
    trusted to DECIDE the winner (a slow-starting donor with the best basin loses
    a 100-iter race), only to prune candidates that are clearly off the basin.
    Pure; returns an int in ``[min_iters, full_iters]``.
    """
    full = int(full_iters)
    # frac of the full budget, floored at min_iters, then never above the full
    # budget (a short pass can't be longer than the full one).
    short = int(round(min(full, max(min_iters, frac * full))))
    return short


# ── Analytic-seed cache (M5) ─────────────────────────────────────────────────
# The analytic seeds (calibrated / structural / regime) are PURE functions of
# (n_nn, n_nnn, n_qubits, p_layers, h, J, J2) [+ a couple of per-seed knobs].
# best_combined_warmstart rebuilds all of them on EVERY call — and the two-pass
# selector / multi-restart loops call it repeatedly for the SAME config, so the
# identical analytic vectors are recomputed many times. These bounded caches
# memoize them. The whole point is a speed-up that NEVER changes a value, so two
# rules are enforced without exception:
#
#   1. The key encodes EVERY argument that affects the output (no "data in the
#      wrong place"): the full layout, h/J/J2 by their EXACT float bits
#      (``float.hex()`` — never a rounded h, which would alias two distinct
#      fields onto one entry), and each seed's extra knob (nnn_suppress /
#      use_calibrated / zz_coef). A different model/topology changes n_nn/n_nnn,
#      so it is already captured by the layout part of the key.
#   2. Entries are stored and returned as COPIES. The underlying builders mutate
#      their array in place (``theta[x] *= xs``); handing back the cached object
#      would let a caller corrupt every future hit. ``.copy()` on store AND on
#      read keeps the cache immutable and callers independent.
_ANALYTIC_SEED_CACHE_MAX: int = 256
_CALIBRATED_SEED_CACHE: dict[tuple, np.ndarray] = {}
_STRUCTURAL_SEED_CACHE: dict[tuple, np.ndarray] = {}
_REGIME_SEED_CACHE: dict[tuple, tuple[np.ndarray, str]] = {}
_VARIANT_SEED_CACHE: dict[tuple, np.ndarray] = {}


def _seed_cache_put(cache: dict, key, value) -> None:
    """FIFO-bounded insert (mirrors unified_mpnn._cache_put_bounded)."""
    if key not in cache and len(cache) >= _ANALYTIC_SEED_CACHE_MAX:
        cache.pop(next(iter(cache)), None)  # evict oldest (insertion order)
    cache[key] = value


def _hx(x: float) -> str:
    """Exact, collision-free float key component (full precision, no rounding)."""
    return float(x).hex()


def clear_analytic_seed_caches() -> None:
    """Empty all analytic-seed caches (test isolation / long-running processes)."""
    _CALIBRATED_SEED_CACHE.clear()
    _STRUCTURAL_SEED_CACHE.clear()
    _REGIME_SEED_CACHE.clear()
    _VARIANT_SEED_CACHE.clear()


def cached_calibrated_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
) -> np.ndarray:
    """Memoized :func:`calibrated_warmstart_theta` (returns a fresh copy).

    Same value as the pure builder, byte-for-byte; only recomputation is saved.
    The key is the full layout plus exact-bit (h, J, J2), so no two distinct
    configurations ever share an entry. The returned array is always a copy, so
    the caller may mutate it freely without touching the cache.
    """
    key = (int(n_nn), int(n_nnn), int(n_qubits), int(p_layers), _hx(h), _hx(J), _hx(J2))
    hit = _CALIBRATED_SEED_CACHE.get(key)
    if hit is None:
        hit = calibrated_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
        _seed_cache_put(_CALIBRATED_SEED_CACHE, key, hit.copy())
    return hit.copy()


def cached_structural_warmstart_theta(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    nnn_suppress: float = STRUCTURAL_NNN_SUPPRESS,
) -> np.ndarray:
    """Memoized :func:`structural_warmstart_theta` (returns a fresh copy).

    ``nnn_suppress`` is part of the key — a different suppression gives a
    different seed and must not reuse another's entry.
    """
    key = (int(n_nn), int(n_nnn), int(n_qubits), int(p_layers), _hx(h), _hx(J), _hx(J2), _hx(nnn_suppress))
    hit = _STRUCTURAL_SEED_CACHE.get(key)
    if hit is None:
        hit = structural_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, nnn_suppress=nnn_suppress)
        _seed_cache_put(_STRUCTURAL_SEED_CACHE, key, hit.copy())
    return hit.copy()


def cached_select_regime_seed(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    use_calibrated: bool = False,
) -> tuple[np.ndarray, str]:
    """Memoized :func:`select_regime_seed` (returns a fresh copy + its name).

    ``use_calibrated`` is in the key: it switches the whole seed family, so the
    two branches must never share an entry.
    """
    key = (int(n_nn), int(n_nnn), int(n_qubits), int(p_layers), _hx(h), _hx(J), _hx(J2), bool(use_calibrated))
    hit = _REGIME_SEED_CACHE.get(key)
    if hit is None:
        seed, name = select_regime_seed(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2, use_calibrated=use_calibrated)
        _seed_cache_put(_REGIME_SEED_CACHE, key, (seed.copy(), name))
        return seed.copy(), name
    seed, name = hit
    return seed.copy(), name


def cached_variant_warmstart_theta(
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
    """Memoized :func:`variant_warmstart_theta` (returns a fresh copy).

    The structure-variant analytic seed is pure in (block sequence, layout, h, J,
    J2, rx/rz flags, shrink/curv). The runners build it once per masked variant of
    the SAME (N, h) config, so memoizing it skips the redundant rebuild. ``blocks``
    enters the key as a tuple (a different sequence is a different seed); every
    knob is keyed so no two variants collide. Returned array is always a copy.
    """
    key = (
        tuple(blocks),
        int(n_nn),
        int(n_nnn),
        int(n_qubits),
        _hx(h),
        _hx(J),
        _hx(J2),
        bool(rx_final),
        bool(rz_final),
        _hx(shrink_coef),
        _hx(curv_coef),
    )
    hit = _VARIANT_SEED_CACHE.get(key)
    if hit is None:
        hit = variant_warmstart_theta(
            blocks,
            n_nn,
            n_nnn,
            n_qubits,
            h,
            J=J,
            J2=J2,
            rx_final=rx_final,
            rz_final=rz_final,
            shrink_coef=shrink_coef,
            curv_coef=curv_coef,
        )
        _seed_cache_put(_VARIANT_SEED_CACHE, key, hit.copy())
    return hit.copy()


def restarts_for_gap(gap: float, base_restarts: int = 1, *, small_gap: float = SMALL_GAP, max_restarts: int = 4) -> int:
    """Scale restarts up as the spectral gap shrinks (gap-proportional budget).

    The θ-scaling study showed ``d(θ_opt, regime_seed)`` grows with N in the
    small-gap regime (ordered-deep / near h_c), which is exactly where a single
    restart under-converges (the N=18 h=0.5 collapse). Where the gap is large
    (paramagnet), the regime seed already lands in the basin, so one restart is
    enough. This maps the gap to a restart count:

    - ``gap >= small_gap``             → ``base_restarts`` (seed is close enough)
    - ``small_gap/10 <= gap <small_gap`` → ``base_restarts + 1``
    - ``gap < small_gap/10`` (near-degenerate) → ``base_restarts + 2``

    Clipped to ``max_restarts``. Pure; the runner multiplies this by its own
    budget knobs.
    """
    base = max(1, int(base_restarts))
    if gap is None or gap >= small_gap:
        out = base
    elif gap >= small_gap / 10.0:
        out = base + 1
    else:
        out = base + 2
    return min(out, max_restarts)


def topk_frac_for_gap(
    gap: float, base_frac: float, *, small_gap: float = SMALL_GAP, aggressive_frac: float = 0.33
) -> float:
    """Shrink the nnn keep-fraction in the near-degenerate (small-gap) regime.

    The θ-scaling study found that at very small gap (ordered-deep, e.g. N=12
    h=0.3, gap~2e-5) the masked top-k ansatz *beat* the full reference: fewer
    parameters give a cleaner landscape. So when the gap is tiny we keep fewer
    nnn bonds (more aggressive mask); otherwise we keep ``base_frac``. Returns a
    fraction in ``(0, 1]``. Pure.
    """
    if not 0.0 < base_frac <= 1.0:
        raise ValueError(f"base_frac must be in (0, 1], got {base_frac}")
    if gap is not None and gap < small_gap / 10.0:
        return min(base_frac, aggressive_frac)
    return base_frac


def crosses_transition(
    h_from: float, h_to: float, *, h_lo: float = ORDERED_H_MAX, h_hi: float = PARAMAGNETIC_H_MIN
) -> bool:
    """True if moving from ``h_from`` to ``h_to`` crosses a phase boundary.

    The cross-h study showed θ transfers well *within* a phase but poorly across
    the frustrated transition (0.5↔1.3 gave <0.28 init-fid either way). An
    h-sweep should therefore DROP the previous-h donor and reset to the regime
    seed when it steps across ``h_lo`` (ordered↔near-h_c) or ``h_hi``
    (near-h_c↔paramagnet). Pure boundary check.
    """

    def regime(h: float) -> int:
        if h <= h_lo:
            return 0  # ordered
        if h < h_hi:
            return 1  # near h_c
        return 2  # paramagnet

    return regime(h_from) != regime(h_to)


def bond_resolved_regime_seed(
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
    J2: float = 0.0,
    donor_theta: np.ndarray | None = None,
    donor_n_nn: int | None = None,
    donor_n_nnn: int | None = None,
    donor_p: int | None = None,
    strength: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Regime seed whose uniform ZZ blocks are modulated per-bond by a donor.

    The θ-scaling study found that near h_c the nn block stops being uniform (it
    becomes genuinely bond-resolved) and the scalar regime seed drifts away from
    the optimum as N grows. This builds the standard regime seed, then rescales
    each nn/nnn angle by the donor's RELATIVE per-bond pattern (its angle divided
    by the donor block mean), so the seed inherits the shape — which bonds are
    strong/weak — while keeping the regime's calibrated magnitude.

    ``strength`` in [0, 1] blends between the flat regime seed (0) and the fully
    modulated one (1). θ_x is left at the regime value (the donor's x is handled
    separately by the x-subspace explorer). When no usable donor is given,
    returns the plain regime seed. The donor is Z2-canonicalized first so its
    pattern is in the same gauge. Returns ``(seed, name)``.
    """
    seed, name = select_regime_seed(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    if donor_theta is None or donor_n_nn is None or donor_p is None:
        return seed, name
    d_nnn = donor_n_nnn if donor_n_nnn is not None else n_nnn
    donor = np.asarray(donor_theta, float)
    if donor.size != (donor_n_nn + d_nnn + n_qubits) * donor_p:
        return seed, name
    # Only modulate when the bond counts match (same layout) — a per-bond shape
    # transfer needs aligned blocks; cross-layout cases fall back to flat.
    if donor_n_nn != n_nn or d_nnn != n_nnn or donor_p != p_layers:
        return seed, name
    s = float(np.clip(strength, 0.0, 1.0))
    donor = _wrap_pi(_canonicalize_z2_flat(donor, n_nn, n_nnn, n_qubits, p_layers))
    seed = _canonicalize_z2_flat(seed, n_nn, n_nnn, n_qubits, p_layers)
    per = n_nn + n_nnn + n_qubits
    out = seed.copy()
    for layer in range(p_layers):
        o = layer * per
        for blk_off, blk_len in ((0, n_nn), (n_nn, n_nnn)):
            if blk_len == 0:
                continue
            sl = slice(o + blk_off, o + blk_off + blk_len)
            d_blk = donor[sl]
            d_mean = float(np.mean(d_blk))
            if abs(d_mean) < 1e-9:
                continue  # donor block ~0 carries no shape
            shape = d_blk / d_mean  # relative per-bond pattern (mean 1)
            modulated = seed[sl] * shape  # keep regime magnitude, donor shape
            out[sl] = (1.0 - s) * seed[sl] + s * modulated
    out = np.clip(out, -np.pi, np.pi)
    return out, f"{name}+bondshape"


def theta_x_indices(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int) -> list[int]:
    """Indices of the θ_x (transverse-field RX) parameters in the flat θ vector.

    The exploration budget is best spent on this soft subspace — the direction
    the analytic seeds least determine. Layout per layer:
    ``[nn (n_nn), nnn (n_nnn), x (n_qubits)]``.
    """
    per = _layer_layout(n_nn, n_nnn, n_qubits, p_layers)
    idx: list[int] = []
    for layer in range(p_layers):
        o = layer * per + n_nn + n_nnn
        idx.extend(range(o, o + n_qubits))
    return idx


def _split_layers(theta: np.ndarray, n_nn: int, n_nnn: int, n_qubits: int, p: int):
    """Per-layer (nn, nnn, x) slices of a bond-resolved θ. Returns list of dicts."""
    theta = np.asarray(theta, float)
    per = n_nn + n_nnn + n_qubits
    layers = []
    for layer in range(p):
        o = layer * per
        layers.append(
            {
                "nn": theta[o : o + n_nn],
                "nnn": theta[o + n_nn : o + n_nn + n_nnn],
                "x": theta[o + n_nn + n_nnn : o + per],
            }
        )
    return layers


def _canonicalize_z2_flat(theta: np.ndarray, n_nn: int, n_nnn: int, n_qubits: int, p: int) -> np.ndarray:
    """Z2-canonical copy of a flat bond-resolved θ (sign fixed by the ZZ blocks).

    The energy is invariant under flipping the sign of ALL ZZ rotations together
    (a Z2 relabeling); θ_x is unaffected. We fix the gauge by requiring the sum
    of the first-layer NN angles to be ≤ 0 (matching the analytic seeds, whose
    NN angles are negative). Pure-numpy twin of
    :func:`qmbp_simulation.analysis.theta_patterns.canonicalize_z2`, inlined here
    to avoid a circular import (theta_patterns imports this module).
    """
    theta = np.asarray(theta, float).copy()
    if n_nn == 0:
        return theta
    per = n_nn + n_nnn + n_qubits
    if float(np.sum(theta[:n_nn])) > 0:
        for layer in range(p):
            o = layer * per
            theta[o : o + n_nn + n_nnn] *= -1.0
    return theta


def _wrap_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angle(s) into ``(-π/2, π/2]`` — the π-periodic branch of rx/rzz(2θ)."""
    arr = np.asarray(theta, float)
    return (arr + np.pi / 2.0) % np.pi - np.pi / 2.0


def lattice_coords(topology: str, n_qubits: int) -> dict[int, tuple[int, int]] | None:
    """Qubit index → (row, col) for regular-grid topologies, else ``None``.

    Cross-N donor transfer matches bonds by qubit index, but on a grid the index
    numbering shifts with N (e.g. square ``cols = ceil(sqrt(N))`` changes 4→5
    between N14 and N18), so the SAME physical bond gets different indices at
    different N — collapsing cross-N overlap to ~15%. These coordinates let the
    transfer align bonds by PHYSICAL POSITION instead, which measured ~74%
    coverage N14→N18 and converged a p2_half_nn_rx N18 to 0.92 vs 0.79 (and ~30×
    faster) with the raw-index match.

    Returns ``None`` for topologies without a trivial grid embedding
    (triangular/kagome/heavy_hex) — callers then fall back to index matching
    (full back-compat). Mirrors the index→cell mapping of the generators in
    :mod:`qmbp_simulation.models.hamiltonian`.
    """
    import math

    if topology == "square":
        cols = math.ceil(math.sqrt(n_qubits))
        return {s: (s // cols, s % cols) for s in range(n_qubits)}
    if topology == "chain_1d":
        return {s: (0, s) for s in range(n_qubits)}
    if topology == "ladder":
        # generate_ladder numbers the two legs; cell = (leg, rung).
        cols = n_qubits // 2
        return {s: (s // cols, s % cols) for s in range(n_qubits)}
    return None


def remap_edges_by_coords(edges, donor_coords, target_coords):
    """Translate donor edges to target-index edges sharing the same grid cells.

    Returns a list the SAME length/order as ``edges`` (to stay aligned with the
    donor θ). An edge maps to the target edge at the same two (row,col) cells;
    cells absent in the target become the sentinel ``(-1, -1)`` — a valid
    int-tuple that never equals a real target edge, so the per-bond match simply
    misses and that θ slot is regime-filled. Returns ``edges`` unchanged when
    either coord map is ``None`` (non-grid topology → index matching).
    """
    if donor_coords is None or target_coords is None:
        return edges
    rc2idx = {rc: s for s, rc in target_coords.items()}
    out = []
    for a, b in edges:
        rc_a = donor_coords.get(int(a))
        rc_b = donor_coords.get(int(b))
        if rc_a in rc2idx and rc_b in rc2idx:
            out.append((rc2idx[rc_a], rc2idx[rc_b]))
        else:
            out.append((-1, -1))
    return out


def _grid_degree(coords):
    """Orthogonal nn-degree per qubit from grid coords (corner=2/edge=3/bulk=4).

    A square-grid site's nn-degree is the number of its 4 orthogonal neighbours
    that exist in the lattice. Used by the cross-N θ_x fill (M4): θ_x depends on
    the local coordination (bulk qubits carry a larger transverse angle than edge
    ones), so broadcasting the donor's θ_x MEAN PER DEGREE is sharper than a
    single global mean. Returns ``{idx: degree}`` or ``None`` if coords is None.
    """
    if coords is None:
        return None
    cells = set(coords.values())
    deg = {}
    for idx, (r, c) in coords.items():
        deg[idx] = sum(((r + dr, c + dc) in cells) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return deg


def transfer_theta(
    donor_theta: np.ndarray,
    *,
    donor_n_nn: int,
    donor_n_nnn: int,
    donor_p: int,
    target_n_nn: int,
    target_n_nnn: int,
    target_p: int,
    n_qubits: int,
    donor_nnn_edges=None,
    target_nnn_edges=None,
    fill_value: float = 0.0,
    fill_theta: np.ndarray | None = None,
    canonicalize: bool = False,
    donor_n_qubits: int | None = None,
    donor_blocks: tuple[str, ...] = ("nn", "nnn", "x"),
    donor_coords=None,
    target_coords=None,
) -> np.ndarray | None:
    """Transfer a converged bond-resolved θ onto a (possibly masked/different-p)
    target layout, aligning angles by physical role and bond identity.

    - nn and x blocks copy layer-by-layer (nn is the full lattice in both).
    - nnn copies per-bond: for each target nnn edge, the donor's angle for that
      SAME edge (matched via ``donor_nnn_edges`` / ``target_nnn_edges``) is used;
      edges not present in the donor get the regime-gated angle from
      ``fill_theta`` (per-bond) when provided, else the scalar ``fill_value``.
    - Layer count: the first ``min(donor_p, target_p)`` layers are copied; extra
      target layers are filled from the donor's LAST layer (near-identity-ish
      continuation), extra donor layers are dropped.
    - ``canonicalize=True`` Z2-canonicalizes the donor (and wraps angles to the
      π branch) BEFORE splitting, so a spin-flipped / periodic-image donor
      transfers into the same gauge as the analytic fill instead of silently
      cancelling against it. Off by default to preserve verbatim copy.

    - ``donor_n_qubits`` (default = target ``n_qubits``) enables CROSS-N
      continuation: when the donor comes from a different N, its θ_x block has a
      different length, so it is replaced by the donor's θ_x MEAN broadcast to
      the target width (N-invariant per the θ-symmetry study); nn/nnn still align
      by bond, with regime fill for bonds the donor lacks.

    This is how a p3 reference θ (0.91) seeds a p2 top-k ansatz: the shared
    structure starts in the good basin instead of a blind analytic guess.
    Returns the target-length seed, or ``None`` if the donor can't be split.
    """
    donor_theta = np.asarray(donor_theta, float)
    # Donor qubit count defaults to the target's (same-N transfer). Supplying a
    # different donor_n_qubits enables CROSS-N continuation: the θ_x block (which
    # is per-qubit, so its length changes with N) is filled with the donor's
    # θ_x MEAN — N-invariant per the θ-symmetry study — rather than copied.
    d_nq = donor_n_qubits if donor_n_qubits is not None else n_qubits
    donor_per = donor_n_nn + donor_n_nnn + d_nq
    if donor_theta.size != donor_per * donor_p:
        return None
    if canonicalize:
        donor_theta = _wrap_pi(_canonicalize_z2_flat(donor_theta, donor_n_nn, donor_n_nnn, d_nq, donor_p))
    donor_layers = _split_layers(donor_theta, donor_n_nn, donor_n_nnn, d_nq, donor_p)

    # Per-bond regime fill (improvement c): missing nn/nnn bonds take the
    # analytic regime angle for that layer instead of a flat fill_value.
    fill_layers = None
    if fill_theta is not None:
        ft = np.asarray(fill_theta, float)
        if ft.size == (target_n_nn + target_n_nnn + n_qubits) * target_p:
            fill_layers = _split_layers(ft, target_n_nn, target_n_nnn, n_qubits, target_p)

    # Geometric alignment (cross-N): remap donor nnn edges to target indices by
    # grid cell so the per-bond match aligns physically, not by raw index. No-op
    # without coords (non-grid topology / same-N transfer) — full back-compat.
    if donor_coords is not None and target_coords is not None and donor_nnn_edges is not None:
        donor_nnn_edges = remap_edges_by_coords(donor_nnn_edges, donor_coords, target_coords)

    # M4: per-degree θ_x fill for cross-N. Precompute donor θ_x mean grouped by
    # qubit degree, and the target qubit degrees, so a cross-N x-block is filled
    # per degree (bulk vs edge) instead of one global mean. Only when coords +
    # degrees are available and the N actually differs.
    cross_n_x = donor_coords is not None and target_coords is not None and d_nq != n_qubits
    donor_deg = _grid_degree(donor_coords) if cross_n_x else None
    target_deg = _grid_degree(target_coords) if cross_n_x else None

    # Per-bond nnn map donor→angle, keyed by sorted edge tuple.
    nnn_angle = {}
    if donor_nnn_edges is not None and target_nnn_edges is not None:
        for layer_idx in range(donor_p):
            d_nnn = donor_layers[layer_idx]["nnn"]
            for e, a in zip(donor_nnn_edges, d_nnn, strict=False):
                nnn_angle[(layer_idx, tuple(sorted((int(e[0]), int(e[1])))))] = float(a)

    seed_parts = []
    for layer in range(target_p):
        src = donor_layers[min(layer, donor_p - 1)]
        fl = fill_layers[layer] if fill_layers is not None else None
        nn_fill = fl["nn"] if fl is not None else np.full(target_n_nn, fill_value)
        nnn_fill = fl["nnn"] if fl is not None else np.full(target_n_nnn, fill_value)
        # Selective per-block transfer (divide & conquer): a block NOT in
        # ``donor_blocks`` is taken from the fill (regime) instead of the donor.
        # The cross-N study found the donor's nnn block is unreliable near h_c
        # (second-neighbor geometry shifts with N), so a caller can keep nn+x
        # from the donor and leave nnn at the regime value.
        # nn: donor nn block if sizes match AND nn is a donor block, else fill.
        if "nn" in donor_blocks and donor_n_nn == target_n_nn:
            nn_block = src["nn"]
        else:
            nn_block = nn_fill
        # nnn: per-bond lookup; donor angle where the edge exists, else regime fill
        if "nnn" not in donor_blocks:
            nnn_block = nnn_fill
        elif target_nnn_edges is not None and nnn_angle:
            d_layer = min(layer, donor_p - 1)
            nnn_block = np.array(
                [
                    nnn_angle.get(
                        (d_layer, tuple(sorted((int(e[0]), int(e[1]))))),
                        float(nnn_fill[j]) if j < nnn_fill.size else fill_value,
                    )
                    for j, e in enumerate(target_nnn_edges)
                ]
            )
        elif donor_n_nnn == target_n_nnn:
            nnn_block = src["nnn"]
        else:
            nnn_block = nnn_fill
        # x: copy verbatim when the qubit count matches; for cross-N transfer
        # use the donor's θ_x MEAN (N-invariant) broadcast to the target width,
        # falling back to the regime x fill, then the scalar fill_value.
        if "x" not in donor_blocks:
            x_block = fl["x"] if (fl is not None and fl["x"].size == n_qubits) else np.full(n_qubits, fill_value)
        elif len(src["x"]) == n_qubits:
            x_block = src["x"]
        elif src["x"].size and donor_deg is not None and target_deg is not None and len(src["x"]) == len(donor_deg):
            # M4: donor θ_x mean PER DEGREE, broadcast to target qubits by their
            # degree; degrees absent in the donor fall back to the global mean.
            glob = float(np.mean(src["x"]))
            by_deg: dict[int, list] = {}
            for q, xv in enumerate(src["x"]):
                by_deg.setdefault(donor_deg[q], []).append(float(xv))
            deg_mean = {d: float(np.mean(v)) for d, v in by_deg.items()}
            x_block = np.array([deg_mean.get(target_deg[q], glob) for q in range(n_qubits)], dtype=float)
        elif src["x"].size:
            x_block = np.full(n_qubits, float(np.mean(src["x"])))
        elif fl is not None and fl["x"].size == n_qubits:
            x_block = fl["x"]
        else:
            x_block = np.full(n_qubits, fill_value)
        seed_parts.extend([nn_block, nnn_block, x_block])
    return np.clip(np.concatenate(seed_parts), -np.pi, np.pi)


def best_warm_start_seed(
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    J: float = 1.0,
    J2: float = 0.0,
    fid_fn=None,
    donors=None,
    target_nnn_edges=None,
    descent_fn=None,
) -> tuple[np.ndarray, str, float | None]:
    """Single, runner-agnostic warm-start chooser — the one entry point to reuse.

    Builds candidate seeds and returns ``(seed, provenance, init_fidelity)`` for
    the BEST one. Candidates, in order:

    1. **Transferred θ** from each converged high-fidelity donor (``donors``),
       mapped onto this target layout via :func:`transfer_theta`. The donor is
       Z2-canonicalized + π-wrapped first (``canonicalize=True``) and bonds the
       donor lacks are filled with the REGIME analytic angle for that bond
       (``fill_theta=regime_seed``), not a flat zero — so a partial donor still
       lands in the right phase.
    2. **Regime analytic seed** (:func:`select_regime_seed`) — always included
       as the robust fallback.

    Candidate selection:

    - With ``descent_fn`` (θ→(θ_refined, fid_after)): each candidate gets a SHORT
      micro-descent and is ranked by its POST-descent fidelity — proximity of a
      raw seed is not the same as the quality of the basin it falls into, so we
      score the basin, not the seed. The returned seed is the refined θ.
    - Else with ``fid_fn`` (θ→fid): candidates ranked by raw seed fidelity.
    - Else: the first donor transfer, else the regime seed.

    ``donors`` is a list of dicts, each:
        {"theta", "n_nn", "n_nnn", "p", "label", "nnn_edges"(optional)}

    Any runner can call this with its circuit's ``fid_fn``/``descent_fn`` to get
    a precise, history-aware seed without re-implementing transfer/cascade logic.
    """
    regime_seed, regime_name = select_regime_seed(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)

    candidates: list[tuple[np.ndarray, str]] = []
    for d in donors or []:
        seed = transfer_theta(
            d["theta"],
            donor_n_nn=d["n_nn"],
            donor_n_nnn=d["n_nnn"],
            donor_p=d["p"],
            target_n_nn=n_nn,
            target_n_nnn=n_nnn,
            target_p=p_layers,
            n_qubits=n_qubits,
            donor_nnn_edges=d.get("nnn_edges"),
            target_nnn_edges=target_nnn_edges,
            fill_theta=regime_seed,
            canonicalize=True,
            donor_n_qubits=d.get("n_qubits"),
        )
        if seed is not None and seed.size == (n_nn + n_nnn + n_qubits) * p_layers:
            candidates.append((seed, f"transfer<{d.get('label', 'donor')}>"))

    candidates.append((regime_seed, regime_name))

    if descent_fn is not None:
        # Rank by the basin each seed falls into (post micro-descent), not the
        # raw seed fidelity — a closer seed can relax into a worse optimum.
        best_seed, best_prov, best_fid = None, None, -1.0
        for seed, prov in candidates:
            try:
                refined, f = descent_fn(seed)
                f = float(f)
            except Exception:
                continue
            if f > best_fid:
                best_seed, best_prov, best_fid = np.asarray(refined, float), f"{prov}+descent", f
        if best_seed is not None:
            return best_seed, best_prov, best_fid
        # fall through to fid_fn / priority if every descent failed

    if fid_fn is None:
        # No evaluator: prefer the first donor transfer, else the regime seed.
        seed, prov = candidates[0]
        return seed, prov, None

    best_seed, best_prov, best_fid = None, None, -1.0
    for seed, prov in candidates:
        try:
            f = float(fid_fn(seed))
        except Exception:
            continue
        if f > best_fid:
            best_seed, best_prov, best_fid = seed, prov, f
    if best_seed is None:  # all evaluations failed → regime seed, unevaluated
        return regime_seed, regime_name, None
    return best_seed, best_prov, best_fid


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


def warmstart_init_fidelity(
    donor_theta: np.ndarray,
    *,
    donor_n_nn: int,
    donor_n_nnn: int,
    donor_p: int,
    target_n_nn: int,
    target_n_nnn: int,
    target_p: int,
    n_qubits: int,
    fid_fn,
    donor_nnn_edges=None,
    target_nnn_edges=None,
    fill_theta: np.ndarray | None = None,
    donor_n_qubits: int | None = None,
) -> tuple[float | None, np.ndarray | None]:
    """Init-fidelity of a donor θ transferred onto a target layout — NO reoptimize.

    This is the cheap probe behind the continuation-in-N idea: transfer a
    converged donor (e.g. from a smaller N at the same h) onto the target layout
    with :func:`transfer_theta` (Z2-canonicalized, regime-filled), then evaluate
    the caller-supplied ``fid_fn`` (θ → fidelity against the target ground state)
    on the transferred seed. No optimization is run, so one call costs a single
    state evaluation — orders of magnitude cheaper than a full convergence.

    Returns ``(init_fid, transferred_theta)``; ``(None, None)`` if the donor
    cannot be transferred (bad length / layout). ``fid_fn`` is injected (not
    built here) so this stays decoupled from circuit construction and testable
    with a stub.
    """
    seed = transfer_theta(
        donor_theta,
        donor_n_nn=donor_n_nn,
        donor_n_nnn=donor_n_nnn,
        donor_p=donor_p,
        target_n_nn=target_n_nn,
        target_n_nnn=target_n_nnn,
        target_p=target_p,
        n_qubits=n_qubits,
        donor_nnn_edges=donor_nnn_edges,
        target_nnn_edges=target_nnn_edges,
        fill_theta=fill_theta,
        canonicalize=True,
        donor_n_qubits=donor_n_qubits,
    )
    if seed is None or seed.size != (target_n_nn + target_n_nnn + n_qubits) * target_p:
        return None, None
    try:
        return float(fid_fn(seed)), seed
    except Exception:
        return None, seed


def theta_x_arctan_deviation(
    theta: np.ndarray,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    *,
    J: float = 1.0,
) -> dict:
    """Compare the measured θ_x block mean to the analytic ``arctan(J/h)``.

    The θ-symmetry study found θ_x is the N-invariant, phase-defining block and
    tracks the leading-order ``arctan(J/h)``. This quantifies how well: it
    returns the per-layer and overall θ_x mean (on the π-wrapped canonical θ),
    the analytic prediction, and the relative deviation. A small, bounded
    deviation across h means θ_x can be used as an ANALYTIC feature of the
    difficulty index without any measurement. Pure.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    pred = float(np.arctan(J / h))
    th = _wrap_pi(_canonicalize_z2_flat(theta, n_nn, n_nnn, n_qubits, p_layers))
    per = n_nn + n_nnn + n_qubits
    layer_means = []
    for layer in range(p_layers):
        o = layer * per + n_nn + n_nnn
        xb = th[o : o + n_qubits]
        layer_means.append(float(xb.mean()) if xb.size else 0.0)
    overall = float(np.mean(layer_means)) if layer_means else 0.0
    rel_dev = abs(overall - pred) / (abs(pred) + 1e-12)
    return {
        "h": float(h),
        "arctan_pred": pred,
        "theta_x_mean": overall,
        "theta_x_mean_per_layer": layer_means,
        "rel_deviation": float(rel_dev),
        "tracks_arctan": bool(rel_dev < 0.15),
    }


def _regime_of(h: float, *, h_lo: float = ORDERED_H_MAX, h_hi: float = PARAMAGNETIC_H_MIN) -> int:
    """Phase bucket for ``h``: 0 ordered, 1 near-h_c, 2 paramagnet.

    Same boundaries as :func:`crosses_transition`; factored out so the ensemble
    can group donors by phase. Pure.
    """
    if h < h_lo:
        return 0
    if h < h_hi:
        return 1
    return 2


def _circular_mean(seeds: list[np.ndarray]) -> np.ndarray:
    """Per-angle circular mean of equal-length angle vectors.

    Averages via atan2(Σsin, Σcos) so wrapped angles (e.g. +π and −π) combine
    correctly instead of cancelling to 0. Returns a vector in (−π, π]. Pure.
    """
    arr = np.stack([np.asarray(s, float) for s in seeds], axis=0)
    return np.arctan2(np.sin(arr).sum(axis=0), np.cos(arr).sum(axis=0))


def ensemble_donor_seed(
    donors,
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    J: float = 1.0,
    J2: float = 0.0,
    target_nnn_edges=None,
    fill_theta: np.ndarray | None = None,
    target_len: int | None = None,
    min_donors: int = 2,
) -> tuple[np.ndarray, int] | None:
    """One ensemble warm-start: the circular mean of same-phase donors.

    Hypothesis (tested, NOT confirmed): a single donor can land in an
    idiosyncratic basin, so averaging several converged θ from the SAME phase
    (ordered / near-h_c / paramagnet) — each Z2-canonicalized and transferred
    onto the target layout — might sit in a more robust shared basin. In practice
    (validate_ensemble_donor.py) the circular mean washed out the bond-resolved
    structure and LOST to the best single donor at every tested (N, h). The
    helper is retained for auditability / re-checking at new layouts, but
    :func:`best_combined_warmstart` leaves it off by default. Only same-phase
    donors are pooled — mixing phases averages across a transition.

    Each donor is routed through :func:`transfer_theta` with ``canonicalize=True``
    (so all donors share the Z2/π gauge before averaging) and the SAME per-donor
    block policy as :func:`best_combined_warmstart` (full layout → nn+nnn+x;
    cross-N/cross-nnn → nn+x only, nnn/x regime-filled). The resulting
    target-length seeds are combined with a circular mean.

    Returns ``(seed, n_pooled)`` or ``None`` when fewer than ``min_donors``
    same-phase donors transfer successfully (then the caller simply skips it).
    Pure w.r.t. the backend. ``fill_theta`` defaults to the regime seed.
    """
    std_len = (n_nn + n_nnn + n_qubits) * p_layers
    target_len = std_len if target_len is None else int(target_len)
    if fill_theta is None:
        fill_theta, _ = select_regime_seed(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
    target_phase = _regime_of(h)

    transferred: list[np.ndarray] = []
    for d in donors or []:
        d_h = d.get("h")
        if d_h is None or _regime_of(float(d_h)) != target_phase:
            continue
        d_nq = d.get("n_qubits", n_qubits)
        d_nnn = d["n_nnn"]
        if "blocks" in d:
            blocks = tuple(d["blocks"])
        elif d_nq == n_qubits and d_nnn == n_nnn:
            blocks = ("nn", "nnn", "x")
        else:
            blocks = ("nn", "x")  # cross layout → only reliable blocks
        seed = transfer_theta(
            d["theta"],
            donor_n_nn=d["n_nn"],
            donor_n_nnn=d_nnn,
            donor_p=d["p"],
            target_n_nn=n_nn,
            target_n_nnn=n_nnn,
            target_p=p_layers,
            n_qubits=n_qubits,
            donor_nnn_edges=d.get("nnn_edges"),
            target_nnn_edges=target_nnn_edges,
            fill_theta=fill_theta,
            canonicalize=True,
            donor_n_qubits=d.get("n_qubits"),
            donor_blocks=blocks,
        )
        if seed is not None and seed.size == target_len:
            transferred.append(seed)

    if len(transferred) < max(2, int(min_donors)):
        return None
    return _circular_mean(transferred), len(transferred)


# Block names in the standard bond-resolved layout, in order.
_BLOCK_NAMES: tuple[str, ...] = ("nn", "nnn", "x")

# Per-phase per-block source policy for the hybrid "best-of-each-tool" seed.
# Derived from the per-block oracle (oracle_block_sources.py) over the square
# frustrated J2=0.5 corpus: for each block, the source whose block lands closest
# (sign/wrap-invariant) to the optimized θ, aggregated by phase. nn is calibrated
# everywhere (strong backbone); nnn/x shift by phase (structural nnn in ordered,
# regime/donor for θ_x). "donor" entries resolve to the best transferred donor
# when one is available, else fall back to the default source.
BLOCK_MIX_POLICY: dict[str, dict[str, str]] = {
    "ordered": {"nn": "calibrated", "nnn": "structural", "x": "donor"},
    "near_hc": {"nn": "calibrated", "nnn": "calibrated", "x": "regime"},
    "paramag": {"nn": "calibrated", "nnn": "regime", "x": "calibrated"},
}


def block_mix_policy_for(h: float, *, h_lo: float = ORDERED_H_MAX, h_hi: float = PARAMAGNETIC_H_MIN) -> dict[str, str]:
    """Per-block source policy for ``h`` (phase-gated). See :data:`BLOCK_MIX_POLICY`."""
    phase = "ordered" if h < h_lo else ("near_hc" if h < h_hi else "paramag")
    return dict(BLOCK_MIX_POLICY[phase])


# Models whose single-qubit term is a transverse field on a ZZ backbone (Ising
# family) — the family the analytic seeds (calibrated_zz_coef / calibrated_x_scale
# / select_regime_seed) were calibrated on. Other models (Heisenberg XX+YY+ZZ, XY,
# Kitaev) have a different param-per-layer structure, so those Ising-calibrated
# analytic seeds do not transfer; only the general regime seed + data-driven
# donors are trustworthy there.
_TFIM_FAMILY: frozenset[str] = frozenset(
    {
        "tfim",
        "tfim_frustrated",
        "tfim_bond_resolved",
        "tfim_longitudinal",
        "tfim_bond_resolved_longitudinal",
    }
)


def warmstart_profile(
    *,
    model: str | None = None,
    topology: str | None = None,
    J2: float = 0.0,
) -> dict:
    """Decide which warm-start techniques are trustworthy for this structure.

    The best technique is structure-dependent. The analytic seeds (calibrated,
    structural) are calibrated for the TFIM/Ising family (ZZ backbone + transverse
    field); applying them to Heisenberg/XY/Kitaev — different param-per-layer and
    no single ZZ coefficient — would be a blind guess. The regime seed and
    transferred donors are general and stay on everywhere.

    Returns the flags :func:`best_combined_warmstart` expects::

        {"include_calibrated": bool, "include_structural": bool,
         "include_regime": True, "include_ensemble": False,
         "include_block_mix": False, "calibrated_family": bool, "notes": str}

    - ``model=None`` (the historical default): assume the TFIM-frustrated family
      the runners were built on — full analytic stack ON (back-compatible).
    - ensemble / block_mix are ALWAYS off: validated negative results
      (validate_ensemble_donor.py, validate_block_mix.py, validate_coupled_hybrid.py).

    Pure — a lookup, no state. ``topology``/``J2`` are accepted for future
    structure-specific gating and recorded in ``notes``.
    """
    is_ising = model is None or str(model).lower() in _TFIM_FAMILY
    if is_ising:
        notes = (
            f"ising-family model={model or 'tfim_frustrated(default)'} "
            f"topology={topology or 'any'} J2={J2}: full analytic stack"
        )
    else:
        notes = f"non-ising model={model}: calibrated/structural OFF (Ising-only calibration); regime+donors ON"
    return {
        "include_calibrated": is_ising,
        "include_structural": is_ising,
        "include_regime": True,
        "include_ensemble": False,  # validated negative
        "include_block_mix": False,  # validated negative
        "calibrated_family": is_ising,
        "notes": notes,
    }


def _block_sign_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sign/Z2- and π-wrap-invariant L2 distance between two angle blocks.

    Both blocks are π-wrapped (the rzz(2θ)/rx branch) and the sign that minimizes
    the distance is chosen (the ZZ Z2 degeneracy flips a whole block's sign
    without changing energy). Returns mean per-angle L2 so blocks of different
    length are comparable. Pure.
    """
    a = _wrap_pi(np.asarray(a, float))
    b = _wrap_pi(np.asarray(b, float))
    if a.size == 0 or a.size != b.size:
        return float("inf")
    d_pos = np.linalg.norm(a - b)
    d_neg = np.linalg.norm(a + b)
    return float(min(d_pos, d_neg) / np.sqrt(a.size))


def block_source_distances(
    theta_opt: np.ndarray,
    sources: dict[str, np.ndarray],
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
) -> dict:
    """Per-block distance of each candidate source to the optimized θ.

    For every block (nn, nnn, x) in every layer, measures the sign/wrap-invariant
    distance (:func:`_block_sign_distance`) from each source's block to the
    optimized θ's block, and reports which source is closest. This is the
    "per-block oracle" used to decide whether a hybrid seed — taking each block
    from its best source — could beat any single whole-θ candidate.

    ``sources`` maps a label to a full-length θ (same layout as ``theta_opt``;
    sources whose length differs are skipped). Returns::

        {"per_block": {(layer, block): {"best": label, "dist": {label: d, ...}}},
         "winners": {block: {label: win_count}},   # aggregated over layers
         "sources": [labels actually compared]}

    Pure — no backend, no optimization. Reuses the standard block slicing.
    """
    theta_opt = np.asarray(theta_opt, float)
    tgt_len = (n_nn + n_nnn + n_qubits) * p_layers
    usable = {lbl: np.asarray(th, float) for lbl, th in sources.items() if np.asarray(th, float).size == tgt_len}
    per_block: dict = {}
    winners: dict = {b: {} for b in _BLOCK_NAMES}
    for layer in range(p_layers):
        sl = dict(zip(_BLOCK_NAMES, _block_slices(n_nn, n_nnn, n_qubits, layer), strict=True))
        for b in _BLOCK_NAMES:
            opt_blk = theta_opt[sl[b]]
            dists = {lbl: _block_sign_distance(th[sl[b]], opt_blk) for lbl, th in usable.items()}
            if not dists:
                continue
            best = min(dists, key=dists.get)
            per_block[(layer, b)] = {"best": best, "dist": dists}
            winners[b][best] = winners[b].get(best, 0) + 1
    return {"per_block": per_block, "winners": winners, "sources": list(usable.keys())}


def block_mix_warmstart(
    sources: dict[str, np.ndarray],
    policy: dict[str, str],
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    default_source: str | None = None,
) -> np.ndarray | None:
    """Assemble a hybrid seed, taking each block from the source named by ``policy``.

    ``policy`` maps a block name (``"nn"``/``"nnn"``/``"x"``) to a source label in
    ``sources``; blocks absent from the policy fall back to ``default_source``
    (or the first available source). Every layer uses the same per-block source
    (the oracle found the winning source is layer-stable within a phase).

    This is the "best of each tool" seed: e.g. nnn from ``structural``, nn from a
    ``donor``, x from ``calibrated``. Reuses :func:`_block_slices`; introduces no
    new angle math. Returns the hybrid θ, or ``None`` if a required source is
    missing or has the wrong length.
    """
    tgt_len = (n_nn + n_nnn + n_qubits) * p_layers
    usable = {lbl: np.asarray(th, float) for lbl, th in sources.items() if np.asarray(th, float).size == tgt_len}
    if not usable:
        return None
    fallback = default_source if default_source in usable else next(iter(usable))
    out = np.array(usable[fallback], float, copy=True)
    for layer in range(p_layers):
        sl = dict(zip(_BLOCK_NAMES, _block_slices(n_nn, n_nnn, n_qubits, layer), strict=True))
        for b in _BLOCK_NAMES:
            src = policy.get(b, fallback)
            if src in usable:
                out[sl[b]] = usable[src][sl[b]]
    return out


def block_indices(n_nn: int, n_nnn: int, n_qubits: int, p_layers: int) -> dict[str, np.ndarray]:
    """Flat parameter indices for each block (nn/nnn/x), across all layers.

    Returns ``{"nn": idx, "nnn": idx, "x": idx}`` with the integer positions of
    every nn / nnn / θ_x angle in the standard bond-resolved layout. Reuses
    :func:`_block_slices`; the complement of one block's indices is "everything
    else" (what a coordinate step freezes). Pure.
    """
    idx: dict[str, list[int]] = {b: [] for b in _BLOCK_NAMES}
    for layer in range(p_layers):
        sl = dict(zip(_BLOCK_NAMES, _block_slices(n_nn, n_nnn, n_qubits, layer), strict=True))
        for b in _BLOCK_NAMES:
            idx[b].extend(range(sl[b].start, sl[b].stop))
    return {b: np.asarray(v, dtype=int) for b, v in idx.items()}


def block_coordinate_descent(
    theta0: np.ndarray,
    cost,
    grad,
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    lbfgsb,
    order: tuple[str, ...] = ("nnn", "nn", "x"),
    sweeps: int = 1,
    maxiter_per_block: int = 8,
) -> np.ndarray:
    """Coordinate descent over θ BLOCKS: relax one block with the rest frozen.

    Unlike :func:`block_mix_warmstart` (which splices blocks from different
    sources and breaks inter-block coherence), this keeps a single coherent θ and
    refines it block by block AGAINST THE FULL STATE — each block is optimized
    with the others held fixed, so the coupling between blocks is respected. This
    is the "coupled hybrid": the best-of-each-tool improvement that fidelity's
    non-separability actually allows.

    For each block in ``order`` (default nnn → nn → x: the oracle's hardest-to-
    predict blocks first), builds a reduced objective over ONLY that block's
    coordinates (the complement frozen at the current θ) and runs ``lbfgsb`` for
    ``maxiter_per_block`` iterations. Repeats for ``sweeps`` passes.

    ``cost(θ)`` / ``grad(θ)`` act on the FULL vector; ``lbfgsb(cost, x0, maxiter,
    grad)`` is injected (reuse the project's :func:`study_runner._lbfgsb`) so this
    stays decoupled from the backend and testable with a stub. Returns the
    refined full-length θ. Pure w.r.t. its inputs (θ0 is not mutated).

    Validation note (validate_coupled_hybrid.py): at an EQUAL optimization budget
    this did NOT beat a full L-BFGS-B micro-descent from the same seed — it tied
    where the seed was already good and regressed in the ordered phase (0.32 vs
    0.59 at h=0.3). Freezing blocks discards the cross-block (off-diagonal
    Hessian) curvature that the full descent exploits, which matters most in the
    hard regime. Kept as reusable infrastructure; NOT wired into
    :func:`best_combined_warmstart`.
    """
    theta = np.array(theta0, float, copy=True)
    blk = block_indices(n_nn, n_nnn, n_qubits, p_layers)
    for _ in range(max(1, int(sweeps))):
        for b in order:
            active = blk.get(b)
            if active is None or active.size == 0:
                continue

            def _sub_cost(x_active, _idx=active, _base=theta):
                full = _base.copy()
                full[_idx] = x_active
                return cost(full)

            def _sub_grad(x_active, _idx=active, _base=theta):
                full = _base.copy()
                full[_idx] = x_active
                g = np.asarray(grad(full), float)
                return g[_idx]

            x0 = theta[active]
            xr, _e, _nit = lbfgsb(_sub_cost, x0, maxiter=maxiter_per_block, grad=_sub_grad)
            theta[active] = np.asarray(xr, float)
    return theta


def best_combined_warmstart(
    *,
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    J: float = 1.0,
    J2: float = 0.0,
    fid_fn=None,
    descent_fn=None,
    donors=None,
    target_nnn_edges=None,
    extra_candidates=None,
    include_calibrated: bool = True,
    include_regime: bool = True,
    include_structural: bool = True,
    include_ensemble: bool = False,
    include_block_mix: bool = False,
    target_len: int | None = None,
    topology: str | None = None,
    descent_fn_short=None,
    two_pass_top_k: int = 2,
    warm_restart_full: bool = False,
) -> dict:
    """The one-call combined warm-start cascade — the best seed we can give.

    Encapsulates the four complementary techniques validated across the study
    into a single, modular, reusable chooser:

    1. **Calibrated analytic seed** (:func:`calibrated_warmstart_theta`) — the
       best purely-analytic seed (zz_coef(h) + x_scale(h)); strong far from h_c.
    2. **Regime seed** (:func:`select_regime_seed`) — the phase-gated fallback.
    3. **Transferred donors** (:func:`transfer_theta`) — converged θ from prior
       runs, including CROSS-N continuation (each donor dict may carry its own
       ``n_qubits``); edge-aligned, Z2-canonicalized, regime-filled.
    4. **Micro-descent selection** — when ``descent_fn`` (θ→(θ_refined, fid)) is
       given, every candidate gets a short relaxation and the one that falls into
       the BEST basin is returned (its refined θ). This is the multiplier: a
       mediocre raw seed that relaxes into a great basin wins over a closer seed
       that doesn't.

    Modularity / extensibility:
    - ``extra_candidates``: an optional list of ``(theta, label)`` the caller
      injects (e.g. a problem-specific ansatz guess). They compete on equal terms.
    - ``include_calibrated`` / ``include_regime`` / ``include_structural``:
      toggle the built-in analytic candidates (first two on by default).
    - ``include_ensemble``: add the circular mean of same-phase donors as one
      extra candidate (:func:`ensemble_donor_seed`). Off by default and left off:
      validation (validate_ensemble_donor.py) showed it loses to the best single
      donor in every tested regime. Kept as an opt-in audited negative result.
    - ``include_block_mix`` (OFF by default): the hybrid "best-of-each-tool" seed
      (:func:`block_mix_warmstart`) that takes each block from the source the
      per-block oracle found closest to θ_opt (:data:`BLOCK_MIX_POLICY`). The
      oracle is per-block and INDEPENDENT, but fidelity is NOT separable across
      blocks: splicing a donor's θ_x onto a different source's nn/nnn breaks the
      inter-block coherence that made each piece good, so the hybrid regressed vs
      the best single candidate in 5/6 tested regimes (validate_block_mix.py).
      Left off; the helpers are kept for analysis and a possible future *coupled*
      hybrid.
    - ``donors``: list of dicts ``{theta, n_nn, n_nnn, p, label, nnn_edges?, n_qubits?}``.
    - ``descent_fn_short`` / ``two_pass_top_k``: the two-pass pre-rank (M2). The
      short descent runs ONCE per distinct candidate (duplicates inherit the
      score), a pure dedup that never changes the ranking.
    - ``warm_restart_full`` (OFF by default): continue each survivor's FULL descent
      from its short-refined θ instead of the raw seed, reusing the pre-rank work.
      This is NOT neutral — on a multi-basin landscape a different start can
      converge to a different basin and change the winner — so it is opt-in and
      must be A/B-validated per regime. The default (raw-seed start) reproduces
      the exhaustive behaviour exactly.

    Scalability: no state is built here — ``fid_fn`` / ``descent_fn`` are injected
    (the caller owns the backend), so this stays O(#candidates) evaluations and
    works at any N the caller can evaluate.

    Returns a dict::

        {"seed": np.ndarray,          # the chosen warm-start θ (refined if descent)
         "provenance": str,           # which candidate won
         "init_fidelity": float|None, # its (post-descent) fidelity, if evaluated
         "report": [                  # every candidate, for audit/metrics
             {"label", "raw_fid", "descent_fid", "selected"}, ...]}

    Without ``fid_fn`` and ``descent_fn`` it returns the first available candidate
    (calibrated → regime → first donor) unevaluated. This never raises on a bad
    candidate; it skips it and records the failure in the report.
    """
    # ``target_len`` defaults to the standard [nn,nnn,x]*p layout length, but a
    # non-standard structure (extra half-layers, rx_final — e.g. p2_half_nn_rx)
    # has a different parameter count; the caller passes the real
    # ``qc.num_parameters`` so its ``extra_candidates`` (structure-aware seeds)
    # survive the length filter while the built-in standard-layout candidates
    # (calibrated/regime) are harmlessly skipped.
    std_len = (n_nn + n_nnn + n_qubits) * p_layers
    target_len = std_len if target_len is None else int(target_len)
    # Cached analytic seeds (M5): same values, recomputation skipped. Each cached
    # wrapper returns a fresh copy, so the in-place mutation below (and in the
    # callers) stays isolated from the cache.
    regime_seed, regime_name = cached_select_regime_seed(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)

    # ── 1-3) assemble candidates (label, theta) ──────────────────────────────
    candidates: list[tuple[str, np.ndarray]] = []
    if include_calibrated:
        candidates.append(
            ("calibrated", cached_calibrated_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2))
        )
    if include_structural:
        candidates.append(
            ("structural", cached_structural_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2))
        )
    if include_regime:
        candidates.append((regime_name, regime_seed))
    for d in donors or []:
        d_nq = d.get("n_qubits", n_qubits)
        d_nnn = d["n_nnn"]
        # Rigorous divide-&-conquer: decide which block-subsets to try PER DONOR
        # from its layout relation to the target, not from a caller flag.
        #  - full ("nn","nnn","x") only when the layout matches (same N, same nnn)
        #    — then the donor's nnn is bond-aligned and reliable.
        #  - a cross-N or cross-nnn donor ALSO (or only) contributes "nn","x":
        #    its second-neighbor block does not transfer cleanly (bond geometry
        #    shifts with N near h_c), so nnn is left at the regime value.
        if "blocks" in d:  # explicit caller override wins
            block_sets = [tuple(d["blocks"])]
        elif d_nq == n_qubits and d_nnn == n_nnn:
            block_sets = [("nn", "nnn", "x")]  # same layout → full is safe
        else:
            block_sets = [("nn", "nnn", "x"), ("nn", "x")]  # cross: try both
        # Cross-N geometric alignment: when the donor comes from a different N on
        # a grid topology, map its edges by physical cell so the per-bond match
        # aligns (raw indices shift with the grid width). No-op for same-N or
        # non-grid topologies (coords None → index match, back-compat).
        d_coords = t_coords = None
        if topology is not None and d_nq != n_qubits:
            d_coords = lattice_coords(topology, d_nq)
            t_coords = lattice_coords(topology, n_qubits)
        for blocks in block_sets:
            seed = transfer_theta(
                d["theta"],
                donor_n_nn=d["n_nn"],
                donor_n_nnn=d_nnn,
                donor_p=d["p"],
                target_n_nn=n_nn,
                target_n_nnn=n_nnn,
                target_p=p_layers,
                n_qubits=n_qubits,
                donor_nnn_edges=d.get("nnn_edges"),
                target_nnn_edges=target_nnn_edges,
                fill_theta=regime_seed,
                canonicalize=True,
                donor_n_qubits=d.get("n_qubits"),
                donor_blocks=blocks,
                donor_coords=d_coords,
                target_coords=t_coords,
            )
            if seed is not None and seed.size == target_len:
                suffix = "" if blocks == ("nn", "nnn", "x") else "|nn+x"
                candidates.append((f"transfer<{d.get('label', 'donor')}>{suffix}", seed))
    # Ensemble candidate: circular mean of same-phase donors (opt-in). Low risk —
    # it competes as one extra candidate and is simply absent when <2 same-phase
    # donors transfer. Empirically it did NOT beat the best single donor in any
    # tested regime (averaging across N/h washes out the bond-resolved structure
    # that makes a matched donor good), so it is OFF by default and kept only as
    # an audited negative result the caller can re-check at new layouts.
    if include_ensemble:
        ens = ensemble_donor_seed(
            donors,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=n_qubits,
            p_layers=p_layers,
            h=h,
            J=J,
            J2=J2,
            target_nnn_edges=target_nnn_edges,
            fill_theta=regime_seed,
            target_len=target_len,
        )
        if ens is not None:
            ens_seed, n_pooled = ens
            if ens_seed.size == target_len:
                candidates.append((f"ensemble<phase{_regime_of(h)},k={n_pooled}>", ens_seed))

    # Hybrid "best-of-each-tool" candidate (opt-in, OFF by default). Takes each
    # block from the source the per-block oracle found closest to θ_opt for this
    # phase (BLOCK_MIX_POLICY), reusing the already-assembled candidates as the
    # source pool. VALIDATED NEGATIVE: fidelity is not separable across blocks, so
    # splicing blocks from different sources breaks inter-block coherence and the
    # hybrid regressed vs the best single candidate in 5/6 regimes. Kept as an
    # audited opt-in; see validate_block_mix.py.
    if include_block_mix:
        src_pool: dict[str, np.ndarray] = {}
        for lbl, th in candidates:
            if th.size != target_len:
                continue
            if lbl in ("calibrated", "structural"):
                src_pool.setdefault(lbl, th)
            elif lbl == regime_name:
                src_pool.setdefault("regime", th)
            elif lbl.startswith("transfer<") and "donor" not in src_pool:
                src_pool["donor"] = th
        policy = block_mix_policy_for(h)
        default_src = "calibrated" if "calibrated" in src_pool else ("regime" if "regime" in src_pool else None)
        mix = block_mix_warmstart(
            src_pool, policy, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n_qubits, p_layers=p_layers, default_source=default_src
        )
        if mix is not None and mix.size == target_len:
            used = {b: (policy[b] if policy[b] in src_pool else default_src) for b in _BLOCK_NAMES}
            tag = "+".join(f"{b}:{used[b]}" for b in _BLOCK_NAMES)
            candidates.append((f"block_mix<{tag}>", mix))

    for theta, label in extra_candidates or []:
        theta = np.asarray(theta, float)
        if theta.size == target_len:
            candidates.append((str(label), theta))

    if not candidates:  # degenerate guard — regime seed is always valid
        candidates.append((regime_name, regime_seed))

    # ── 3.5) dedup map for the micro-descent (M3) ────────────────────────────
    # Several analytic seeds coincide at some h (structural ≈ second_order), and a
    # donor's two block-sets can collapse to the same vector. The micro-descent is
    # the expensive step, so we evaluate each DISTINCT vector once and let its
    # duplicates inherit the result. ALL candidates stay in the report / selection
    # (semantics unchanged) — ``_rep_of[i]`` points to the representative index
    # whose descent result candidate ``i`` reuses. Pure speed, no decision change.
    _rep_of = list(range(len(candidates)))
    if len(candidates) > 1:
        kept: list[tuple[int, np.ndarray]] = []
        for i, (_lbl, th) in enumerate(candidates):
            tw = _wrap_pi(np.asarray(th, float))
            dup_of = next(
                (
                    j
                    for j, kv in kept
                    if kv.size == tw.size and min(float(np.linalg.norm(tw - kv)), float(np.linalg.norm(tw + kv))) < 1e-9
                ),
                None,
            )
            if dup_of is None:
                kept.append((i, tw))
            else:
                _rep_of[i] = dup_of

    # ── 4) selection ─────────────────────────────────────────────────────────
    report = [{"label": lbl, "raw_fid": None, "descent_fid": None, "selected": False} for lbl, _ in candidates]

    def _finish(idx, seed, prov, init_fid):
        if 0 <= idx < len(report):
            report[idx]["selected"] = True
        return {"seed": np.asarray(seed, float), "provenance": prov, "init_fidelity": init_fid, "report": report}

    if descent_fn is not None:
        # M2 two-pass early-exit (opt-in): a SHORT descent on every candidate
        # pre-ranks them, then only the top-k get the FULL descent. Saves the
        # expensive full micro-descent on obviously-losing candidates at large N
        # (budget 400 × 5 candidates → 400 × 2). Off by default (descent_fn_short
        # is None) so the exhaustive behaviour and its results are unchanged.
        full_idxs = list(range(len(candidates)))
        # Short-pass refined θ per REPRESENTATIVE. The pre-rank runs the short
        # descent ONCE per distinct vector (duplicates inherit the score via
        # _rep_of) — this is fix 1, a pure dedup that saves redundant short passes
        # and never changes the ranking (identical vectors score identically).
        _short_refined: dict = {}  # rep index → short-refined θ (for warm_restart_full)
        if descent_fn_short is not None and len(candidates) > two_pass_top_k:
            pre_by_rep: dict = {}  # rep index → short fid
            for i in range(len(candidates)):
                rep = _rep_of[i]
                if rep not in pre_by_rep:
                    try:
                        r_ref, f = descent_fn_short(candidates[rep][1])
                        pre_by_rep[rep] = float(f)
                        _short_refined[rep] = np.asarray(r_ref, float)
                    except Exception:
                        pre_by_rep[rep] = -1.0
            pre = [(pre_by_rep.get(_rep_of[i], -1.0), i) for i in range(len(candidates))]
            pre.sort(reverse=True)
            full_idxs = [i for _f, i in pre[:two_pass_top_k]]

        best = (-1.0, None, None, -1)
        _descent_cache: dict = {}  # rep index → (refined, fid)
        for i in full_idxs:
            lbl, th = candidates[i]
            rep = _rep_of[i]
            if rep in _descent_cache:  # M3: reuse the representative's descent
                refined, f = _descent_cache[rep]
            else:
                try:
                    # fix 2 (OPT-IN, warm_restart_full): continue the full descent
                    # from the short-refined θ instead of the raw seed, so the
                    # pre-rank work is reused. NOT neutral in general — on a
                    # multi-basin landscape a different start point can converge to
                    # a different basin, so this CAN change which candidate wins.
                    # Off by default: the default full descent starts from the raw
                    # seed, exactly reproducing the validated exhaustive behaviour.
                    start = _short_refined.get(rep, candidates[rep][1]) if warm_restart_full else candidates[rep][1]
                    refined, f = descent_fn(start)
                    f = float(f)
                except Exception:
                    continue
                _descent_cache[rep] = (refined, f)
            report[i]["descent_fid"] = f
            if f > best[0]:
                best = (f, np.asarray(refined, float), f"{lbl}+descent", i)
        if best[1] is not None:
            return _finish(best[3], best[1], best[2], best[0])
        # all descents failed → fall through to fid_fn / priority

    if fid_fn is not None:
        best = (-1.0, None, None, -1)
        for i, (lbl, th) in enumerate(candidates):
            try:
                f = float(fid_fn(th))
            except Exception:
                continue
            report[i]["raw_fid"] = f
            if f > best[0]:
                best = (f, th, lbl, i)
        if best[1] is not None:
            return _finish(best[3], best[1], best[2], best[0])
        return _finish(0, regime_seed, regime_name, None)

    # No evaluator: first candidate by priority (calibrated → regime → donor).
    return _finish(0, candidates[0][1], candidates[0][0], None)


def transfer_theta_for_blocks(
    donor_theta,
    blocks: list[str],
    *,
    donor_nnn_edges,
    target_nnn_edges,
    n_nn: int,
    n_qubits: int,
    rx_final: bool = False,
    rz_final: bool = False,
    fill_value: float = 0.0,
    donor_n_nn: int | None = None,
    donor_n_qubits: int | None = None,
    donor_nn_edges=None,
    target_nn_edges=None,
    donor_coords=None,
    target_coords=None,
) -> np.ndarray | None:
    """Transfer a converged structure-variant θ onto a same-structure target θ.

    The structure-aware analogue of :func:`transfer_theta` for an arbitrary block
    sequence (``blocks`` + ``rx_final``/``rz_final``, e.g. ``p2_half_nn_rx``).
    Handles two cases with the same per-block, per-bond logic:

    - **Same-N bond masking** (default): the target keeps the same ``n_nn`` and
      ``n_qubits`` as the donor but only ``target_nnn_edges`` ⊆ ``donor_nnn_edges``
      carry an RZZ. nn/x copied verbatim; nnn copied per-bond by sorted-tuple.
    - **Cross-N continuation** (donor from a SMALLER N): pass ``donor_n_nn`` /
      ``donor_n_qubits`` (and optionally ``donor_nn_edges`` / ``target_nn_edges``)
      so nn/nnn are matched per shared edge and the rest regime-filled, while θ_x
      is broadcast from the donor's N-invariant MEAN (the θ-symmetry finding).
      This lets an N=10 converged full seed an N=14/N=18 full in its good basin.

    ``donor_coords`` / ``target_coords`` (idx→(row,col), from
    :func:`lattice_coords`) enable GEOMETRIC cross-N matching: the donor nn/nnn
    edges are remapped to target indices sharing the same grid cell BEFORE the
    per-bond match, so bonds align by physical position rather than raw index.
    This is the N14→N18 fix (coverage 17%→74%, converged fidelity 0.79→0.92).
    When omitted (or a non-grid topology → ``None``), the raw-index match is used
    (full back-compat).

    Edges not present in the donor get ``fill_value``. Returns the target-length θ,
    or ``None`` if the donor length doesn't match its declared structure.
    """
    donor = np.asarray(donor_theta, float)
    d_nn = donor_n_nn if donor_n_nn is not None else n_nn
    d_nq = donor_n_qubits if donor_n_qubits is not None else n_qubits
    # Geometric alignment: remap donor edges to target indices by grid cell so
    # the per-bond match below (keyed by sorted tuple) aligns physically. No-op
    # when coords are absent (non-grid topology or same-N verbatim transfer).
    if donor_coords is not None and target_coords is not None:
        donor_nnn_edges = remap_edges_by_coords(donor_nnn_edges, donor_coords, target_coords)
        if donor_nn_edges is not None:
            donor_nn_edges = remap_edges_by_coords(donor_nn_edges, donor_coords, target_coords)
    donor_size = {"nn": d_nn, "nnn": len(list(donor_nnn_edges)), "x": d_nq, "z": d_nq}
    expected = sum(donor_size[b] for b in blocks)
    expected += d_nq if rx_final else 0
    expected += d_nq if rz_final else 0
    if donor.size != expected:
        return None

    donor_nnn = [tuple(sorted((int(e[0]), int(e[1])))) for e in donor_nnn_edges]
    target_nnn = [tuple(sorted((int(e[0]), int(e[1])))) for e in target_nnn_edges]
    nnn_idx = {e: j for j, e in enumerate(donor_nnn)}

    # nn per-bond map only when both edge lists are given AND counts differ
    # (cross-N); otherwise nn is copied verbatim (same-N, identical backbone).
    cross_n = d_nq != n_qubits or d_nn != n_nn
    nn_idx = None
    tgt_nn = None
    if cross_n and donor_nn_edges is not None and target_nn_edges is not None:
        d_nn_t = [tuple(sorted((int(e[0]), int(e[1])))) for e in donor_nn_edges]
        tgt_nn = [tuple(sorted((int(e[0]), int(e[1])))) for e in target_nn_edges]
        nn_idx = {e: j for j, e in enumerate(d_nn_t)}
    # θ_x N-invariant mean (cross-N fill for the per-qubit block).
    x_mean = fill_value

    out_parts: list[np.ndarray] = []
    off = 0
    for b in blocks:
        d_blk = donor[off : off + donor_size[b]]
        if b == "nnn":
            out_parts.append(
                np.array([d_blk[nnn_idx[e]] if e in nnn_idx else fill_value for e in target_nnn], dtype=float)
            )
        elif b == "nn":
            if nn_idx is not None:  # cross-N per-bond nn match
                out_parts.append(
                    np.array([d_blk[nn_idx[e]] if e in nn_idx else fill_value for e in tgt_nn], dtype=float)
                )
            elif d_nn == n_nn:  # same backbone → verbatim
                out_parts.append(np.asarray(d_blk, float))
            else:  # counts differ, no edge map → regime fill
                out_parts.append(np.full(n_nn, fill_value))
        else:  # x / z single-qubit blocks
            if d_nq == n_qubits:
                out_parts.append(np.asarray(d_blk, float))
            else:  # cross-N: broadcast donor mean (N-invariant)
                x_mean = float(np.mean(d_blk)) if d_blk.size else fill_value
                out_parts.append(np.full(n_qubits, x_mean))
        off += donor_size[b]
    for trailing in (rx_final, rz_final):
        if trailing:
            tail = donor[off : off + d_nq]
            if d_nq == n_qubits:
                out_parts.append(np.asarray(tail, float))
            else:
                out_parts.append(np.full(n_qubits, float(np.mean(tail)) if tail.size else fill_value))
            off += d_nq
    out = np.concatenate(out_parts) if out_parts else np.zeros(0)
    size_tgt = {"nn": n_nn, "nnn": len(target_nnn), "x": n_qubits, "z": n_qubits}
    expected_tgt = sum(size_tgt[b] for b in blocks) + (n_qubits if rx_final else 0) + (n_qubits if rz_final else 0)
    if out.size != expected_tgt:
        return None
    return np.clip(out, -np.pi, np.pi)
