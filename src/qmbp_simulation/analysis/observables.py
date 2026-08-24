"""Shared physics observables for time evolution and phase detection.

This module provides vectorized, reusable implementations of quantum
observables needed by the quench dynamics runner, QPT detection, and
future hardware post-processing.

All functions operate on statevectors (numpy arrays of shape (2^N,)).
For MPS-based observables, use the MPSBackend.evaluate() with SparsePauliOp
operators from HamiltonianBuilder.build_local_observables().

Functions
---------
half_chain_entropy : Von Neumann entropy of half-chain bipartition
magnetization_z : <M_z> = (1/N) sum <Z_i>
magnetization_x : <M_x> = (1/N) sum <X_i>
loschmidt_echo : L(t) = |<psi_0|psi_t>|^2
rate_function : r(t) = -(1/N) * ln(L(t))
order_parameter_x : Alias for magnetization_x (TFIM order parameter)
staggered_magnetization_z : Neel order parameter for AFM phases
"""

from __future__ import annotations

import numpy as np


def half_chain_entropy(psi: np.ndarray, n_qubits: int) -> float:
    """Von Neumann entropy of the half-chain bipartition via SVD.

    Computes S_vN = -Tr(rho_A * log(rho_A)) where A is the left half of
    the chain (qubits 0..N/2-1).

    Parameters
    ----------
    psi : np.ndarray
        Statevector of shape (2^N,).
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Entanglement entropy in nats (natural log).
    """
    n_a = n_qubits // 2
    psi_matrix = psi.reshape(2**n_a, 2 ** (n_qubits - n_a))
    sv = np.linalg.svd(psi_matrix, compute_uv=False)
    probs = sv**2
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log(probs)))


def magnetization_z(psi: np.ndarray, n_qubits: int) -> float:
    """Compute <M_z> = (1/N) sum_i <Z_i> in the computational basis.

    Vectorized implementation using bitwise popcount.

    Parameters
    ----------
    psi : np.ndarray
        Statevector of shape (2^N,).
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Magnetization per site in [-1, 1].
    """
    dim = 2**n_qubits
    probs = np.abs(psi) ** 2
    basis_states = np.arange(dim, dtype=np.int64)

    # Vectorized popcount: count number of set bits per basis state
    # Uses Brian Kernighan's bit trick via lookup table for efficiency
    if dim <= 65536:
        # Small enough for 16-bit lookup table (most common case: N≤16)
        table = np.array([bin(i).count("1") for i in range(min(dim, 65536))], dtype=np.int32)
        if dim <= 65536:
            popcount = table[basis_states]
        else:
            popcount = table[basis_states & 0xFFFF] + table[(basis_states >> 16) & 0xFFFF]
    else:
        # Fallback for very large dims: shift-and-count
        popcount = np.zeros(dim, dtype=np.int32)
        tmp = basis_states.copy()
        for _ in range(n_qubits):
            popcount += (tmp & 1).astype(np.int32)
            tmp >>= 1

    # Z eigenvalue: +1 for |0>, -1 for |1> → per-site average
    z_eigenvalues = (n_qubits - 2 * popcount) / n_qubits
    return float(np.dot(probs, z_eigenvalues))


def magnetization_x(psi: np.ndarray, n_qubits: int) -> float:
    """Compute <M_x> = (1/N) sum_i <X_i>.

    X flips a single bit: <psi|X_i|psi> = sum_k conj(psi[k]) * psi[k ^ (1<<i)].
    Vectorized over all sites.

    Parameters
    ----------
    psi : np.ndarray
        Statevector of shape (2^N,).
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Transverse magnetization per site.
    """
    dim = 2**n_qubits
    mx_total = 0.0
    for i in range(n_qubits):
        # X_i maps |k> -> |k XOR (1<<i)>
        flipped_indices = np.arange(dim, dtype=np.int64) ^ (1 << i)
        # <X_i> = sum_k conj(psi[k]) * psi[flipped_k]
        mx_total += float(np.real(np.dot(np.conj(psi), psi[flipped_indices])))
    return mx_total / n_qubits


def loschmidt_echo(psi_0: np.ndarray, psi_t: np.ndarray) -> float:
    """Compute Loschmidt echo L(t) = |<psi_0|psi_t>|^2."""
    overlap = np.vdot(psi_0, psi_t)
    return float(np.abs(overlap) ** 2)


def rate_function(loschmidt: float, n_qubits: int) -> float:
    """Compute the rate function r(t) = -(1/N) * ln(L(t)).


    The rate function is the intensive free energy analog for DQPTs.
    Cusps (non-analyticities) in r(t) signal dynamical phase transitions.
    """
    if loschmidt <= 0:
        return float("inf")
    return -np.log(max(loschmidt, 1e-300)) / n_qubits


def order_parameter_x(psi: np.ndarray, n_qubits: int) -> float:
    """TFIM order parameter: transverse magnetization <M_x>.


    In the TFIM, the order parameter for the paramagnetic phase is <M_x>.
    This is an alias for magnetization_x for semantic clarity.
    """
    return magnetization_x(psi, n_qubits)


def staggered_magnetization_z(psi: np.ndarray, n_qubits: int) -> float:
    """Neel order parameter: (1/N) sum_i (-1)^i <Z_i>.

    Detects antiferromagnetic ordering in frustrated/Heisenberg models.

    Parameters
    ----------
    psi : np.ndarray
        Statevector.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Staggered magnetization per site.
    """
    dim = 2**n_qubits
    probs = np.abs(psi) ** 2
    basis_states = np.arange(dim, dtype=np.int64)

    # Compute staggered Z expectation vectorized over all sites
    # For each basis state k, sum_i (-1)^i * Z_i(k) where Z_i = +1 if bit_i=0, -1 if bit_i=1
    # Rearrange: Z_i = 1 - 2*bit_i, so sum = sum_i (-1)^i * (1 - 2*bit_i)
    #   = sum_i (-1)^i - 2 * sum_i (-1)^i * bit_i
    # First term is a constant for even/odd N; second requires per-state computation.

    # Pre-compute staggered sign pattern
    signs = np.array([(-1) ** i for i in range(n_qubits)], dtype=np.float64)
    constant_term = float(signs.sum())  # sum_i (-1)^i

    # For each basis state, compute sum_i (-1)^i * bit_i
    staggered_bits = np.zeros(dim, dtype=np.float64)
    for i in range(n_qubits):
        bit_i = ((basis_states >> i) & 1).astype(np.float64)
        staggered_bits += signs[i] * bit_i

    # Full staggered magnetization per basis state
    staggered_per_state = (constant_term - 2 * staggered_bits) / n_qubits
    return float(np.dot(probs, staggered_per_state))


def detect_dqpt_critical_times(
    times: list[float] | np.ndarray,
    loschmidt_values: list[float] | np.ndarray,
    *,
    threshold: float = 0.1,
    adaptive: bool = True,
) -> list[float]:
    """Detect DQPT critical times from Loschmidt echo minima.

    Identifies local minima of L(t) as DQPT signatures. Uses both:
    - Absolute threshold: L(t*) < threshold (default 0.1)
    - Adaptive: L(t*) < median(L) * 0.5 (catches DQPTs even when
      L never drops below 0.1, common for short evolutions or large gaps)

    Parameters
    ----------
    times : array-like
        Time points.
    loschmidt_values : array-like
        Loschmidt echo at each time point.
    threshold : float
        Maximum L(t) value to count as "near zero" (DQPT). Default 0.1.
    adaptive : bool
        If True, also detect minima below median(L)*0.5 (catches DQPTs
        in short evolutions where L doesn't reach the absolute threshold).

    Returns
    -------
    list[float]
        Critical times t* where DQPTs are detected.
    """
    L_arr = np.asarray(loschmidt_values)
    t_arr = np.asarray(times)
    critical_times = []

    if len(L_arr) < 3:
        return critical_times

    # Adaptive threshold: use median * 0.5 if it's more permissive
    effective_threshold = threshold
    if adaptive and len(L_arr) > 5:
        # Exclude t=0 (always 1.0) from median calculation
        L_interior = L_arr[1:]
        median_L = float(np.median(L_interior))
        adaptive_threshold = median_L * 0.5
        effective_threshold = max(threshold, adaptive_threshold)

    for i in range(1, len(L_arr) - 1):
        if L_arr[i] < L_arr[i - 1] and L_arr[i] < L_arr[i + 1]:
            if L_arr[i] < effective_threshold:
                critical_times.append(float(t_arr[i]))

    return critical_times


def detect_rate_function_peaks(
    times: list[float] | np.ndarray,
    rate_values: list[float] | np.ndarray,
) -> list[dict[str, float]]:
    """Detect peaks in the rate function r(t).

    Peaks correspond to DQPTs (non-analytic points in the return rate).

    Parameters
    ----------
    times : array-like
        Time points.
    rate_values : array-like
        Rate function values r(t) at each time point.

    Returns
    -------
    list[dict]
        Each dict contains {"t": critical_time, "r": peak_value}.
    """
    r_arr = np.asarray(rate_values)
    t_arr = np.asarray(times)
    peaks = []

    if len(r_arr) < 3:
        return peaks

    for i in range(1, len(r_arr) - 1):
        if r_arr[i] > r_arr[i - 1] and r_arr[i] > r_arr[i + 1]:
            peaks.append({"t": float(t_arr[i]), "r": float(r_arr[i])})

    return peaks
