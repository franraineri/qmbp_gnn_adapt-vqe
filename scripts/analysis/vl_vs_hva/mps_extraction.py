#!/usr/bin/env python
"""Bond-capped MPS extraction from DMRG — general, backend-agnostic.

Produces a matrix-product-state ground-state approximation with a *bounded* bond
dimension and returns its site tensors in a portable ``plr`` layout (physical,
left, right), plus a quality assessment of the truncation. This is deliberately
NOT tied to any consumer: the tensors feed anything that ingests an MPS — Haiqu
``mps_loading``, an Aer MPS backend, entanglement analysis, etc.

The DMRG runs on the *exact* project Hamiltonian: any ``SparsePauliOp`` is
converted to a TeNPy MPO term-by-term (Pauli Z=2·Sz, X=2·Sx, Y=2·Sy), so the
extracted MPS is the ground state of the identical operator the rest of the
project uses — no re-implemented physics, no sign/edge mismatch. Works for TFIM,
frustrated TFIM, Heisenberg, XY, etc.

Public API:

- ``dmrg_mps_capped(hamiltonian, n, chi_max=64, ...)`` → :class:`CappedMPS`.
- ``mps_to_statevector(capped)`` → dense vector (small N only).
- ``mps_input_quality(capped, ...)`` → exact fidelity (small N) or truncation
  error proxy (large N).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pauli → TeNPy spin-1/2 operator (Z=2·Sz, X=2·Sx, Y=2·Sy). Identity handled inline.
_PAULI_TO_SPIN = {"X": ("Sx", 2.0), "Y": ("Sy", 2.0), "Z": ("Sz", 2.0)}


@dataclass
class CappedMPS:
    """A bond-capped MPS plus provenance for downstream loaders.

    ``tensors`` are rank-3 arrays in ``plr`` order (physical, left, right), one
    per site, site ``i`` → qubit ``q_i`` (Qiskit ordering). ``bond_dims`` are the
    realized virtual bond sizes (≤ ``chi_max``). ``trunc_err`` is the DMRG
    discarded weight (sum of squared truncated singular values).
    """

    tensors: list[np.ndarray]
    n_qubits: int
    chi_max: int
    max_bond_realized: int
    bond_dims: list[int]
    energy: float
    trunc_err: float
    meta: dict = field(default_factory=dict)


def _pauli_op_to_termlist(hamiltonian):
    """Convert a Qiskit SparsePauliOp to a TeNPy TermList (Pauli → spin-1/2).

    Each Pauli string term ``c · P`` becomes ``c · Π_i (2·S_axis)`` on the
    non-identity sites. Qiskit labels are big-endian (leftmost = highest qubit),
    so we index sites as ``q = n-1-pos`` to match site i → qubit i.
    """
    from tenpy.networks.terms import TermList

    n = hamiltonian.num_qubits
    terms: list[list[tuple[str, int]]] = []
    strengths: list[complex] = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs, strict=True):
        label = pauli.to_label()  # e.g. "IZZI"
        ops: list[tuple[str, int]] = []
        factor = 1.0
        for pos, ch in enumerate(label):
            if ch == "I":
                continue
            site = n - 1 - pos  # big-endian label → site index
            spin_op, scale = _PAULI_TO_SPIN[ch]
            ops.append((spin_op, site))
            factor *= scale
        c = complex(coeff) * factor
        if not ops:  # global identity term → constant energy offset, skip
            continue
        terms.append(ops)
        strengths.append(c.real if abs(c.imag) < 1e-12 else c)
    return TermList(terms, strengths), n


def dmrg_mps_capped(hamiltonian, n: int, *, chi_max: int = 64,
                    max_sweeps: int = 100, meta: dict | None = None) -> CappedMPS:
    """Run TeNPy DMRG on ``hamiltonian`` at bounded bond dimension; extract MPS.

    ``hamiltonian`` is a Qiskit ``SparsePauliOp`` (any project model). ``chi_max``
    bounds the virtual bond dimension (Haiqu ``mps_loading`` accepts ≤ 64).
    Returns a :class:`CappedMPS` with plr-ordered site tensors, realized bond
    dimensions, energy and DMRG truncation error.
    """
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.lattice import Chain
    from tenpy.models.model import MPOModel
    from tenpy.networks.mpo import MPOGraph
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    term_list, n_terms_qubits = _pauli_op_to_termlist(hamiltonian)
    if n_terms_qubits != n:
        raise ValueError(f"Hamiltonian has {n_terms_qubits} qubits, expected {n}.")

    site = SpinHalfSite(conserve=None)
    sites = [site] * n
    graph = MPOGraph.from_term_list(term_list, sites, bc="finite")
    H_mpo = graph.build_MPO()
    chain = Chain(L=n, site=site, bc_MPS="finite")
    model_obj = MPOModel(chain, H_mpo)

    psi = MPS.from_lat_product_state(chain, [["up"]] * n)
    engine = tenpy_dmrg.TwoSiteDMRGEngine(
        psi, model_obj,
        {
            "mixer": True,
            "max_E_err": 1e-12,
            "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12, "trunc_cut": 1e-10},
            "max_sweeps": max_sweeps,
        },
    )
    e0, psi = engine.run()

    tensors: list[np.ndarray] = []
    bond_dims: list[int] = []
    for i in range(n):
        b = psi.get_B(i, form="B")
        arr = b.transpose(["p", "vL", "vR"]).to_ndarray()
        tensors.append(np.ascontiguousarray(arr))
        bond_dims.append(int(arr.shape[2]))

    te = getattr(engine, "trunc_err", None)
    trunc_err = float(te.eps) if te is not None else 0.0

    out = CappedMPS(
        tensors=tensors, n_qubits=n, chi_max=chi_max,
        max_bond_realized=max(bond_dims), bond_dims=bond_dims,
        energy=float(e0), trunc_err=trunc_err, meta=meta or {},
    )
    # Retain the TeNPy MPS so the Vidal (Γ/Λ) form can be extracted on demand
    # without re-running DMRG. Kept out of the dataclass fields (not serializable).
    out._tenpy_psi = psi  # type: ignore[attr-defined]
    return out


def capped_mps_vidal(capped: CappedMPS):
    """Extract the Vidal (central-canonical) form of a CappedMPS for Haiqu.

    Returns ``(gammas, lambdas)`` where ``gammas`` are the plr-ordered Γ site
    tensors (form 'G') and ``lambdas`` are the N-1 internal Λ bond vectors
    (singular values). This is Haiqu ``mps_loading``'s Vidal input: a tuple of
    (site tensors, bond tensors). Requires the CappedMPS to carry its TeNPy MPS
    (set by ``dmrg_mps_capped``); raises otherwise.
    """
    psi = getattr(capped, "_tenpy_psi", None)
    if psi is None:
        raise ValueError("CappedMPS has no retained TeNPy MPS; re-run dmrg_mps_capped.")
    n = capped.n_qubits
    gammas: list[np.ndarray] = []
    for i in range(n):
        g = psi.get_B(i, form="G")
        arr = g.transpose(["p", "vL", "vR"]).to_ndarray()
        gammas.append(np.ascontiguousarray(arr))
    # Internal bonds: Λ_1 .. Λ_{n-1} (singular values at each internal cut).
    lambdas: list[np.ndarray] = []
    for i in range(1, n):
        s = psi.get_SL(i)  # left Schmidt values of site i = bond (i-1,i)
        lambdas.append(np.ascontiguousarray(np.asarray(s, dtype=float)))
    return gammas, lambdas


def mps_to_statevector(capped: CappedMPS) -> np.ndarray:
    """Contract a plr MPS to a dense statevector (Qiskit little-endian).

    Feasible only for small N (2**N amplitudes). Used for exact fidelity checks.
    """
    psi = capped.tensors[0][:, 0, :]  # (p0, r0) — collapse trivial left leg
    for t in capped.tensors[1:]:
        psi = np.tensordot(psi, t, axes=([psi.ndim - 1], [1]))  # (..., p, r)
    psi = psi[..., 0]  # collapse trivial right leg
    n = capped.n_qubits
    # site i is physical index i (big-endian row-major); reverse to Qiskit little-endian.
    vec = psi.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 0 else vec


def mps_input_quality(capped: CappedMPS, *, topology: str, h: float,
                      model: str = "tfim", j2: float = 0.0,
                      hamiltonian=None, chi_ref: int | None = None) -> dict:
    """Assess the capped MPS quality.

    - N ≤ statevector limit: exact fidelity |⟨ψ_exact|ψ_MPS⟩|² (primary metric).
    - Otherwise: a bond-cap energy proxy. TeNPy's ``trunc_err`` is unreliable when
      the state saturates ``chi_max`` (it can report 0), so when ``hamiltonian``
      is given we re-run DMRG at ``chi_ref`` (default 4× chi_max) and report
      ``energy_vs_chi_ref`` = |E(chi_max) − E(chi_ref)| as the honest cap cost.

    ``bond_saturated`` flags whether the realized bond hit the cap.
    """
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

    out = {
        "n_qubits": capped.n_qubits,
        "chi_max": capped.chi_max,
        "max_bond_realized": capped.max_bond_realized,
        "bond_saturated": capped.max_bond_realized >= capped.chi_max,
        "trunc_err": capped.trunc_err,
        "mps_energy": capped.energy,
    }
    if capped.n_qubits <= STATEVECTOR_MAX_N:
        try:
            from hva_vl_study_common import exact_ground_state_vector

            psi_exact, e0, _ = exact_ground_state_vector(
                topology, capped.n_qubits, h, model=model, j2=j2,
            )
            if psi_exact is not None:
                psi_mps = mps_to_statevector(capped)
                out["mps_fidelity"] = float(abs(np.vdot(psi_exact, psi_mps)) ** 2)
                out["mps_energy_error"] = abs(capped.energy - e0)
                return out
        except Exception as exc:  # noqa: BLE001
            out["mps_fidelity"] = None
            out["fidelity_note"] = f"exact-fidelity failed: {type(exc).__name__}: {exc}"
            return out

    out["mps_fidelity"] = None
    out["fidelity_note"] = (
        f"N={capped.n_qubits} > statevector limit ({STATEVECTOR_MAX_N}); "
        f"no exact vector — using an energy-vs-higher-χ proxy"
    )
    if hamiltonian is not None:
        chi_ref = chi_ref or capped.chi_max * 4
        try:
            ref = dmrg_mps_capped(hamiltonian, capped.n_qubits, chi_max=chi_ref)
            out["chi_ref"] = chi_ref
            out["energy_ref"] = ref.energy
            out["energy_vs_chi_ref"] = abs(capped.energy - ref.energy)
            out["max_bond_at_chi_ref"] = ref.max_bond_realized
        except Exception as exc:  # noqa: BLE001
            out["fidelity_note"] += f" (chi_ref check failed: {type(exc).__name__})"
    return out
