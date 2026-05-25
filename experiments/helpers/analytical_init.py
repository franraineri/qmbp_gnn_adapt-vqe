"""B1: Analytical Initial Guess from Perturbation Theory.

For h >> h_c (deep paramagnetic), derives theta_opt analytically:
- The ground state is approximately |+>^N with small ZZ corrections.
- theta_x ~ pi/2 (rotate |0> to |+>)
- theta_zz ~ J/(2h) (perturbative ZZ correction)

This eliminates seed sensitivity at large h and provides a deterministic
initialization for the descending sweep.

References:
    - Mele et al. (2022): Parameter transferability in HVA.
    - Wiersema et al. (2020): HVA expressibility analysis.
"""

from __future__ import annotations

import numpy as np


def analytical_init_p1(h: float, J: float = 1.0) -> np.ndarray:
    """Analytical initial guess for HVA p=1 at given h.

    Parameters
    ----------
    h : float
        Transverse field strength.
    J : float
        Coupling constant (default 1.0).

    Returns
    -------
    np.ndarray
        [theta_zz, theta_x] — 2 parameters.

    Notes
    -----
    For h >> 1:
        theta_x ≈ 3*pi/8 ≈ 1.178 (empirical from p=1 scaling experiments)
        theta_zz ≈ pi - J/(2h) (approaches pi as h → ∞)

    For h ~ 1 (near critical):
        Falls back to small random (perturbation theory breaks down).
    """
    if h < 1.2:
        # Near critical: perturbation theory invalid, use small random
        return np.random.uniform(-0.1, 0.1, 2)

    # Empirical from binnacle-p1-scaling: theta_x ≈ ±1.178 (= 3π/8)
    theta_x = 3 * np.pi / 8

    # theta_zz approaches pi as h → ∞, with correction ~ J/h
    theta_zz = np.pi - J / (2 * h)

    return np.array([theta_zz, theta_x])


def analytical_init_p2(h: float, J: float = 1.0) -> np.ndarray:
    """Analytical initial guess for HVA p=2 at given h.

    Parameters
    ----------
    h : float
        Transverse field strength.
    J : float
        Coupling constant (default 1.0).

    Returns
    -------
    np.ndarray
        [theta_zz1, theta_x1, theta_zz2, theta_x2] — 4 parameters.

    Notes
    -----
    Uses empirical observations from V6.1 VQE data:
    - At large h, theta_x dominates (close to pi/4)
    - theta_zz scales as ~J/h (perturbative correction)
    - Second layer is a small refinement of the first

    The formula is calibrated against VQE-optimized theta at N=6.
    """
    if h < 1.0:
        # Deep ferromagnetic: no analytical form, use small random
        return np.random.uniform(-0.1, 0.1, 4)

    # Empirical fit from VQE data (N=6, p=2):
    # theta_x1 ~ pi/4 at large h, decreases toward h_c
    # theta_zz1 ~ J/(2h) at large h
    # Layer 2 is ~30% of layer 1 magnitude

    # First layer (dominant)
    theta_zz1 = 0.5 * J / h  # Perturbative ZZ coupling
    theta_x1 = np.pi / 4 * (1 - 0.5 / h)  # Approaches pi/4 from below

    # Second layer (refinement — empirically ~30% of first)
    theta_zz2 = 0.15 * J / h
    theta_x2 = 0.3 * theta_x1

    return np.array([theta_zz1, theta_x1, theta_zz2, theta_x2])


def validate_analytical_init(
    circuit,
    hamiltonian,
    h: float,
    exact_energy: float,
    evaluate_fn,
    p: int = 2,
    J: float = 1.0,
) -> dict:
    """Validate analytical guess against exact VQE result.

    Parameters
    ----------
    circuit : QuantumCircuit
    hamiltonian : SparsePauliOp
    h : float
    exact_energy : float
    evaluate_fn : callable(circuit, hamiltonian, params) -> float
    p : int
    J : float

    Returns
    -------
    dict with keys: analytical_theta, analytical_energy, energy_error,
                    relative_to_exact, quality (good/fair/poor)
    """
    theta_analytical = analytical_init_p1(h, J) if p == 1 else analytical_init_p2(h, J)

    energy = evaluate_fn(circuit, hamiltonian, theta_analytical)
    error = abs(energy - exact_energy)

    # Quality assessment
    if error < 0.01:
        quality = "excellent"
    elif error < 0.05:
        quality = "good"
    elif error < 0.2:
        quality = "fair"
    else:
        quality = "poor"

    return {
        "analytical_theta": theta_analytical.tolist(),
        "analytical_energy": float(energy),
        "exact_energy": float(exact_energy),
        "energy_error": float(error),
        "quality": quality,
        "h": h,
        "p": p,
    }
