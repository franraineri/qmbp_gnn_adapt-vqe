"""Phase classification — pure, deterministic, testable.

Classifies quantum phase based on per-site magnetization and
per-bond correlation observables with statistical uncertainty.
"""

from __future__ import annotations

import numpy as np


def classify_phase(
    x_values: list[float],
    zz_values: list[float],
    shots: int,
) -> tuple[str, float, float, float]:
    """Classify phase based on |<X>| vs |<ZZ>|.

    Pure function — no external state dependency. Deterministic for
    identical inputs.

    Parameters
    ----------
    x_values : list[float]
        Per-site <X_i> values (ZNE-extrapolated).
    zz_values : list[float]
        Per-bond <Z_iZ_j> values (ZNE-extrapolated).
    shots : int
        Total shots used (for computing sigma).

    Returns
    -------
    tuple[str, float, float, float]
        (label, mag_x, corr_zz, sigma)
        label in {"paramagnetic", "ordered", "indeterminate"}
    """
    sigma = float(1.0 / np.sqrt(shots))

    if len(x_values) == 0 and len(zz_values) == 0:
        return "indeterminate", 0.0, 0.0, sigma

    mag_x = float(np.mean(np.abs(x_values))) if x_values else 0.0
    corr_zz = float(np.mean(np.abs(zz_values))) if zz_values else 0.0

    if abs(mag_x - corr_zz) < sigma:
        label = "indeterminate"
    elif mag_x > corr_zz:
        label = "paramagnetic"
    else:
        label = "ordered"

    return label, mag_x, corr_zz, sigma
