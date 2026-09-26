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
        raise ValueError(
            f"invalid layout: n_nn={n_nn} n_nnn={n_nnn} n_qubits={n_qubits} "
            f"p_layers={p_layers}"
        )
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
        theta[o:o + n_nn] = theta_nn
        theta[o + n_nn:o + n_nn + n_nnn] = theta_nnn
        theta[o + n_nn + n_nnn:o + per] = theta_x
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
        theta[o:o + n_nn] = -a_nn * shrink
        theta[o + n_nn:o + n_nn + n_nnn] = -a_nnn * shrink
        theta[o + n_nn + n_nnn:o + per] = theta_x
    return theta


def second_order_correction_magnitude(
    h: float, *, J: float = 1.0, shrink_coef: float = DEFAULT_SHRINK_COEF
) -> float:
    """The scalar ``(J/2h)^2 * shrink_coef`` correction magnitude at ``h``.

    Grows as ``h`` falls; used both to calibrate the regime gate and to record
    per-h provenance. At the gate window edges (h=0.4..0.6, J=1, c=1/3) this is
    roughly 0.52 .. 0.23.
    """
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    return (J / (2 * h)) ** 2 * shrink_coef


def second_order_regime_gate(
    h: float, *, h_min: float = SECOND_ORDER_H_MIN, h_max: float = SECOND_ORDER_H_MAX
) -> bool:
    """Selection rule: is the second-order warm-start expected to help at ``h``?
    """
    return h_min <= h <= h_max


def aggregate_seed_stats(
    fid_first: list[float] | np.ndarray, fid_second: list[float] | np.ndarray
) -> dict:
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
