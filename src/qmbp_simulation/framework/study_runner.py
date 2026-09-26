"""StudyRunner — a reusable lifecycle base for warm-start VQE study scripts.

The ``vl_vs_hva`` study grew ~13 argparse runners that each re-implemented the
same loop: solve the ground state, build the bond-resolved HVA circuit, seed a
warm-start, optimize, score fidelity/ΔE, and persist crash-safely. This module
factors that loop into a base class so a subclass declares *what* to run, not
*how*:

    setup(h) → seed() → optimize(strategy) → record() → persist() → summarize()

Design notes
------------
- **Composition over the section runner.** ``ValidationRunner`` is section-
  oriented and calls ``os._exit()`` in ``run()``; StudyRunner instead owns a
  simple per-unit loop and reuses the *helpers* that make sense (ground-state
  cache, checkpointing semantics) via the shared ``src`` modules. It never
  calls ``os._exit`` so it is safe to drive from tests or notebooks.
- **Pluggable optimizer strategy.** The warm-start method is a *parameter*, not
  a fork: ``first_order``, ``second_order`` (regime-gated), ``bestof``,
  ``metropolis``. :func:`resolve_strategy` returns the callable; ``second_order``
  auto-falls back to ``metropolis`` outside the confirmed window
  (:func:`~qmbp_simulation.analysis.warmstart.second_order_regime_gate`).
- **Pure, testable strategy layer.** The strategies live here in ``src`` (they
  only need the warm-start core + scipy + a backend), so they are unit-testable
  without importing from ``scripts/``. The ``vl_vs_hva`` service delegates its
  ``warmstart_*_vqe`` functions to these, keeping a single source of truth for
  the optimizer loops as well as the formulas.
- **Crash-safe by construction.** Checkpointing/resume is built in via
  :class:`~qmbp_simulation.framework.study_checkpoint.StudyCheckpoint`; the loop
  persists after every unit of work and re-raises on error with partial state on
  disk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qmbp_simulation.analysis.warmstart import (
    first_order_warmstart_theta,
    second_order_regime_gate,
    second_order_warmstart_theta,
)

# Strategy names accepted across the study runners and CLIs.
STRATEGIES = ("first_order", "second_order", "bestof", "metropolis", "mixed")

# Restart recipes for the ``mixed`` strategy. Each entry explores a DIFFERENT
# basin (varied seed + varied perturbation subspace), not just a different noise
# magnitude — so a smaller per-restart iteration budget can still find a better
# optimum than one long run from a single seed. Fields:
#   name       : label recorded per restart
#   seed_order : "second" (regime seed) or "first" (leading-order seed)
#   sigma      : perturbation magnitude (0.0 = pure seed, no perturbation)
#   block      : which parameter block to perturb — "all", "x" (theta_x only),
#                or "zz" (theta_nn + theta_nnn). Ignored when sigma == 0.
MIXED_RESTART_TYPES: tuple[dict, ...] = (
    {"name": "second_pure", "seed_order": "second", "sigma": 0.0, "block": "all"},
    {"name": "second_iso_small", "seed_order": "second", "sigma": 0.10, "block": "all"},
    {"name": "second_iso_large", "seed_order": "second", "sigma": 0.30, "block": "all"},
    {"name": "second_dir_x", "seed_order": "second", "sigma": 0.20, "block": "x"},
    {"name": "second_dir_zz", "seed_order": "second", "sigma": 0.20, "block": "zz"},
    {"name": "first_pure", "seed_order": "first", "sigma": 0.0, "block": "all"},
)

# Number of restarts the ``mixed`` strategy runs. Cycles through
# ``MIXED_RESTART_TYPES`` if larger than the tuple (each extra pass re-seeds the
# RNG so repeats explore new noise draws). Override this global to add/remove
# restarts without changing the recipe list.
MIXED_N_RESTARTS: int = len(MIXED_RESTART_TYPES)


@dataclass
class WarmStartResult:
    """Outcome of a warm-start optimization at one (h, unit).

    Mirrors the 5-tuple contract the ``vl_vs_hva`` service exposes
    (``energy, bound, fidelity, runs, best_theta``) while also carrying the
    resolved strategy label so callers can record which method actually ran
    (e.g. ``second_order`` that fell back to ``metropolis``).
    """

    energy: float
    fidelity: float
    best_theta: np.ndarray
    runs: list[dict] = field(default_factory=list)
    strategy: str = ""

    def as_tuple(self, bound_circuit) -> tuple:
        """Return the legacy 5-tuple ``(energy, bound, fid, runs, best_theta)``."""
        return (self.energy, bound_circuit, self.fidelity, self.runs, self.best_theta)


def make_adjoint_gradient(circuit, hamiltonian, backend=None):
    """Return an exact-gradient callable ``grad(theta)->np.ndarray`` for
    ``d<H>/dtheta``, or ``None`` if unavailable.

    Delegates to the backend's own ``gradient()`` (the single source of truth —
    e.g. ``NoiselessBackend`` returns the exact adjoint/reverse gradient, noisy
    backends return None). Defaults to ``NoiselessBackend`` when no backend is
    given, matching the study's statevector-exact setting. Kept as a thin helper
    so existing study callers need not change.
    """
    if backend is None:
        from qmbp_simulation.execution import NoiselessBackend

        backend = NoiselessBackend()
    try:
        return backend.gradient(circuit, hamiltonian)
    except Exception:
        return None


def _lbfgsb(cost: Callable[[np.ndarray], float], x0: np.ndarray, *, maxiter: int,
            grad: Callable[[np.ndarray], np.ndarray] | None = None):
    """Single L-BFGS-B run in the standard ``[-pi, pi]`` box. Returns (x, fun, nit).

    ``grad`` is an optional analytic gradient (jac). When provided (e.g. the exact
    adjoint gradient), L-BFGS-B uses it instead of its internal finite-difference
    approximation — same optimum, far fewer circuit evaluations (the FD jac costs
    ~n_params extra cost() calls per step). When ``None``, behaviour is unchanged.
    """
    from scipy.optimize import minimize

    r = minimize(
        cost, x0, method="L-BFGS-B", jac=grad,
        bounds=[(-np.pi, np.pi)] * len(x0),
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    return r.x, float(r.fun), int(r.nit)


# ── Optimizer strategies (pure; operate on a cost fn + fidelity fn) ────────────
#
# Each strategy takes:
#   cost(theta)->float, fidelity(theta)->float, th_ws (seed), n_params,
#   plus its own knobs. It returns (best_energy, best_theta, runs).
# This keeps them backend-agnostic and unit-testable with a stub cost function.


def strategy_single(cost, fidelity, th_ws, *, maxiter=80, grad=None):
    """One L-BFGS-B from the warm-start seed (used by first_order / second_order)."""
    x, e, nit = _lbfgsb(cost, th_ws, maxiter=maxiter, grad=grad)
    runs = [{"restart": 0, "sigma": 0.0, "energy": e,
             "fidelity": float(fidelity(x)), "nit": nit}]
    return e, x, runs


def strategy_bestof(cost, fidelity, th_ws, *, maxiter=80,
                    sigmas=(0.0, 0.1, 0.1, 0.3), seed0=7000, mask=None, grad=None):
    """Perturbed best-of from the warm-start seed (isotropic by default).

    Reproduces ``warmstart_bestof_vqe``: sigma=0 is the pure seed floor, then
    small/large-sigma perturbations, best-of by energy. ``mask`` optionally
    restricts perturbation to a parameter subspace.
    """
    n_params = len(th_ws)
    if mask is None:
        mask = np.ones(n_params, bool)
    best_e, best_x, runs = None, None, []
    for i, sigma in enumerate(sigmas):
        if sigma == 0.0:
            x0 = th_ws
        else:
            rng = np.random.default_rng(seed0 + i)
            noise = np.zeros(n_params)
            noise[mask] = rng.normal(0, sigma, int(mask.sum()))
            x0 = np.clip(th_ws + noise, -np.pi, np.pi)
        x, e, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        runs.append({"sigma": float(sigma), "energy": e,
                     "fidelity": float(fidelity(x)), "nit": nit, "theta": x.tolist()})
        if best_e is None or e < best_e:
            best_e, best_x = e, x
    return best_e, best_x, runs


def strategy_metropolis(cost, fidelity, th_ws, *, maxiter=80, sigma=0.3, n_hops=8,
                        temperature=1.0, seed0=13000, grad=None):
    """Analytic warm-start + Metropolis basin-hopping (the confirmed general winner).

    Reproduces ``warmstart_metropolis_vqe``: seed, then hop with Gaussian
    perturbations, accepting uphill moves with prob ``exp(-dE/T)`` so the chain
    can drift to a better basin. ``runs`` logs each hop.
    """
    n_params = len(th_ws)
    T = max(temperature, 1e-6)
    rng = np.random.default_rng(seed0)
    x_cur, e_cur, nit0 = _lbfgsb(cost, th_ws, maxiter=maxiter, grad=grad)
    x_best, e_best = x_cur.copy(), e_cur
    runs = [{"hop": 0, "energy": e_cur, "fidelity": float(fidelity(x_cur)),
             "accepted": True, "nit": nit0}]
    for k in range(n_hops):
        x0 = np.clip(x_cur + rng.normal(0, sigma, n_params), -np.pi, np.pi)
        x_new, e_new, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        de = e_new - e_cur
        accepted = de < 0 or rng.random() < np.exp(-de / T)
        if accepted:
            x_cur, e_cur = x_new, e_new
        if e_new < e_best:
            x_best, e_best = x_new.copy(), e_new
        runs.append({"hop": k + 1, "energy": e_new,
                     "fidelity": float(fidelity(x_new)),
                     "accepted": bool(accepted), "nit": nit})
    return e_best, x_best, runs


def _block_mask(block: str, n_nn: int, n_nnn: int, n_qubits: int, p_layers: int) -> np.ndarray:
    """Boolean mask selecting a parameter block across all layers.

    Layout per layer is ``[theta_nn | theta_nnn | theta_x]``. ``block`` picks
    ``"all"``, ``"x"`` (theta_x only), or ``"zz"`` (theta_nn + theta_nnn).
    """
    per = n_nn + n_nnn + n_qubits
    mask = np.zeros(per * p_layers, bool)
    for layer in range(p_layers):
        o = layer * per
        if block == "all":
            mask[o:o + per] = True
        elif block == "x":
            mask[o + n_nn + n_nnn:o + per] = True
        elif block == "zz":
            mask[o:o + n_nn + n_nnn] = True
        else:
            raise ValueError(f"unknown block {block!r}; expected all|x|zz")
    return mask


def strategy_mixed(cost, fidelity, *, seeds, n_nn, n_nnn, n_qubits, p_layers,
                   maxiter=80, seed0=50000, grad=None,
                   restart_types=MIXED_RESTART_TYPES, n_restarts=None):
    """Best-of over restarts that explore DIFFERENT basins, not just noise levels.

    Each restart varies the *seed* (first- vs second-order) and the perturbed
    *subspace* (all / theta_x / ZZ), so a smaller per-restart ``maxiter`` still
    samples distinct optima. Every restart records ``nit`` and a ``converged``
    flag (``nit < maxiter``) so callers can see which basins actually settled.

    ``seeds`` is a mapping ``{"first": th1, "second": th2}`` of precomputed
    warm-start vectors. ``n_restarts`` defaults to :data:`MIXED_N_RESTARTS`;
    if it exceeds ``len(restart_types)`` the recipe list cycles, each extra pass
    drawing fresh noise. Returns ``(best_energy, best_theta, runs)``.
    """
    if n_restarts is None:
        n_restarts = MIXED_N_RESTARTS
    masks = {b: _block_mask(b, n_nn, n_nnn, n_qubits, p_layers) for b in ("all", "x", "zz")}
    best_e, best_x, runs = None, None, []
    for i in range(n_restarts):
        recipe = restart_types[i % len(restart_types)]
        th_ws = seeds[recipe["seed_order"]]
        sigma = float(recipe["sigma"])
        if sigma == 0.0:
            x0 = th_ws
        else:
            mask = masks[recipe["block"]]
            rng = np.random.default_rng(seed0 + i)
            noise = np.zeros(len(th_ws))
            noise[mask] = rng.normal(0, sigma, int(mask.sum()))
            x0 = np.clip(th_ws + noise, -np.pi, np.pi)
        x, e, nit = _lbfgsb(cost, x0, maxiter=maxiter, grad=grad)
        runs.append({
            "restart": i, "type": recipe["name"], "seed_order": recipe["seed_order"],
            "sigma": sigma, "block": recipe["block"], "energy": e,
            "fidelity": float(fidelity(x)), "nit": nit,
            "converged": bool(nit < maxiter), "theta": x.tolist(),
        })
        if best_e is None or e < best_e:
            best_e, best_x = e, x
    return best_e, best_x, runs


def resolve_strategy(name: str, h: float) -> str:
    """Resolve a strategy name for a given ``h``, applying the regime gate.

    ``second_order`` is enabled only inside the confirmed transition window;
    outside it, this returns ``"metropolis"`` (the confirmed general fallback).
    All other names pass through unchanged. Raises on an unknown name.
    """
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGIES}")
    if name == "second_order" and not second_order_regime_gate(h):
        return "metropolis"
    return name


def run_strategy(
    strategy: str,
    *,
    cost: Callable[[np.ndarray], float],
    fidelity: Callable[[np.ndarray], float],
    n_nn: int,
    n_nnn: int,
    n_qubits: int,
    p_layers: int,
    h: float,
    J: float = 1.0,
    J2: float = 0.0,
    maxiter: int = 80,
    sigma: float = 0.3,
    n_hops: int = 8,
    bestof_sigmas: tuple = (0.0, 0.1, 0.1, 0.3),
    temperature: float | None = None,
    shrink_coef: float | None = None,
    curv_coef: float | None = None,
    seed0: int | None = None,
    grad: Callable[[np.ndarray], np.ndarray] | None = None,
) -> WarmStartResult:
    """Dispatch to the resolved warm-start strategy and return a WarmStartResult.

    Builds the appropriate seed (first- vs second-order), runs the chosen
    optimizer, and reports the resolved strategy label (which may differ from
    ``strategy`` when the second-order gate falls back to metropolis).

    ``grad`` is an optional exact analytic gradient (jac) passed through to the
    optimizer. When supplied (e.g. the adjoint gradient), L-BFGS-B uses it in
    place of its finite-difference approximation — identical optimum, far fewer
    circuit evaluations. When ``None``, behaviour is unchanged.
    """
    resolved = resolve_strategy(strategy, h)

    # Seed: second-order form only when the resolved strategy needs it.
    if resolved in ("second_order", "mixed"):
        from qmbp_simulation.analysis.warmstart import DEFAULT_CURV_COEF, DEFAULT_SHRINK_COEF

        th_ws = second_order_warmstart_theta(
            n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2,
            shrink_coef=DEFAULT_SHRINK_COEF if shrink_coef is None else shrink_coef,
            curv_coef=DEFAULT_CURV_COEF if curv_coef is None else curv_coef,
        )
    else:
        th_ws = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)

    if resolved == "first_order":
        e, x, runs = strategy_single(cost, fidelity, th_ws, maxiter=maxiter, grad=grad)
    elif resolved == "second_order":
        # Second-order seed + small perturbed best-of for genuine variation.
        e, x, runs = strategy_bestof(
            cost, fidelity, th_ws, maxiter=maxiter,
            sigmas=(0.0, sigma, sigma), seed0=30000 if seed0 is None else seed0,
            grad=grad,
        )
    elif resolved == "bestof":
        e, x, runs = strategy_bestof(
            cost, fidelity, th_ws, maxiter=maxiter, sigmas=bestof_sigmas,
            seed0=7000 if seed0 is None else seed0, grad=grad,
        )
    elif resolved == "mixed":
        th_first = first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h, J=J, J2=J2)
        e, x, runs = strategy_mixed(
            cost, fidelity, seeds={"second": th_ws, "first": th_first},
            n_nn=n_nn, n_nnn=n_nnn, n_qubits=n_qubits, p_layers=p_layers,
            maxiter=maxiter, seed0=50000 if seed0 is None else seed0, grad=grad,
        )
    else:  # metropolis
        e, x, runs = strategy_metropolis(
            cost, fidelity, th_ws, maxiter=maxiter, sigma=sigma, n_hops=n_hops,
            temperature=1.0 if temperature is None else temperature,
            seed0=13000 if seed0 is None else seed0, grad=grad,
        )
    return WarmStartResult(energy=float(e), fidelity=float(fidelity(x)),
                           best_theta=np.asarray(x), runs=runs, strategy=resolved)


class StudyRunner:
    """Base lifecycle for warm-start VQE study points (frustrated bond-resolved HVA).

    Subclasses declare *what*; the base owns *how* (ground state, circuit, seed,
    optimize, score, persist, resume). A subclass typically overrides only
    :meth:`build_circuit` (default: ``create_bond_resolved_frustrated``) and the
    persistence wiring (``subdir`` / ``out_file`` / artifact dir), and calls
    :meth:`run_point` per (topology, h) or drives :meth:`run_sweep`.

    Parameters
    ----------
    topology, n_qubits, p_layers, j2, model : physics config.
    strategy : one of :data:`STRATEGIES`. ``second_order`` is regime-gated.
    maxiter, sigma, n_hops : optimizer knobs.
    ansatz : ``"nnn"`` (frustration-aware, default) or ``"nn"``.
    """

    def __init__(self, *, topology: str, n_qubits: int, p_layers: int = 2,
                 j2: float = 0.5, model: str = "tfim_frustrated",
                 strategy: str = "metropolis", maxiter: int = 150,
                 sigma: float = 0.3, n_hops: int = 8, ansatz: str = "nnn",
                 J: float = 1.0, use_adjoint_grad: bool = True):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected {STRATEGIES}")
        self.topology = topology
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.j2 = j2
        self.model = model
        self.strategy = strategy
        self.maxiter = maxiter
        self.sigma = sigma
        self.n_hops = n_hops
        self.ansatz = ansatz
        self.J = J
        # Use the exact adjoint (reverse) gradient as the optimizer jac when
        # available — same optimum, far fewer circuit evaluations. Falls back to
        # finite-difference (grad=None) if qiskit_algorithms is missing.
        self.use_adjoint_grad = use_adjoint_grad
        self._backend = None

    def _make_adjoint_grad(self, qc, H):
        """Return an exact-gradient callable ``grad(theta)->np.ndarray`` via the
        adjoint (reverse) method, or ``None`` to fall back to finite-difference.

        The reverse gradient is exact (validated to ~1e-10 vs central-difference)
        and its cost is one reverse pass rather than ~n_params forward evals, so
        the speedup grows with parameter count (measured ~13x at N=16).
        """
        if not self.use_adjoint_grad:
            return None
        return make_adjoint_gradient(qc, H, backend=self.backend)

    # ── lifecycle: setup ─────────────────────────────────────────────────────
    @property
    def backend(self):
        """Lazily-created NoiselessBackend (statevector-exact)."""
        if self._backend is None:
            from qmbp_simulation.execution import NoiselessBackend

            self._backend = NoiselessBackend()
        return self._backend

    def ground_state(self, h: float):
        """Exact ground state ``(psi, e0, gap, H)`` — cached E0/gap, exact vector.

        Uses the statevector-exact path for ``N <= STATEVECTOR_MAX_N`` and sparse
        ``eigsh`` above it, matching the study service. E0/gap come from the
        shared GroundTruthCache (``h:.2f`` keys); the vector is re-solved.
        """
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

        if self.n_qubits <= STATEVECTOR_MAX_N:
            psi, e0, gap, H = _exact_ground_state_vector(
                self.topology, self.n_qubits, h, model=self.model, j2=self.j2
            )
            return psi, e0, gap, H, "cached/exact"
        psi, e0, gap, H = _eigsh_ground_state(
            self.topology, self.n_qubits, h, model=self.model, j2=self.j2
        )
        return psi, e0, gap, H, "eigsh_k2"

    def build_circuit(self, lattice):
        """Build the ansatz circuit and return ``(qc, n_nn, n_nnn)``.

        Default: bond-resolved frustrated (+NNN) for ``ansatz="nnn"``, plain
        bond-resolved for ``"nn"``. Override for a different ansatz.
        """
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        hva = HVACircuitBuilder()
        if self.ansatz == "nnn":
            qc, _ = hva.create_bond_resolved_frustrated(self.n_qubits, self.p_layers, lattice)
            n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lattice))
        else:
            qc, _ = hva.create_bond_resolved(self.n_qubits, self.p_layers, lattice)
            n_nnn = 0
        return qc, len(lattice.edges), n_nnn

    # ── lifecycle: optimize ──────────────────────────────────────────────────
    def optimize_point(self, h: float) -> dict:
        """Run one point end-to-end: ground state → circuit → seed → optimize → score.

        Returns a per-h result dict (energy, abs_error, de_gap, fidelity, the
        resolved strategy, per-restart runs, and best_theta), plus the live
        circuit under the ``_qc`` key (drop it before JSON serialization).
        Crash-safe persistence + resume across a sweep is provided by
        :meth:`run_sweep`; callers that need bespoke per-hop artifact layout
        (e.g. the Metropolis point runner) drive :meth:`optimize_point` directly
        and wire their own :class:`StudyCheckpoint`.
        """
        from qmbp_simulation.models import make_lattice

        psi, e0, gap, H, gs_method = self.ground_state(h)
        lat = make_lattice(self.topology, self.n_qubits, J=self.J, h=h)
        qc, n_nn, n_nnn = self.build_circuit(lat)

        def cost(x):
            return self.backend.evaluate(qc, H, x)

        def fidelity(x):
            return _state_fidelity_exact(qc, x, psi)

        grad = self._make_adjoint_grad(qc, H)

        res = run_strategy(
            self.strategy, cost=cost, fidelity=fidelity,
            n_nn=n_nn, n_nnn=n_nnn, n_qubits=self.n_qubits, p_layers=self.p_layers,
            h=h, J=self.J, J2=(self.j2 if self.ansatz == "nnn" else 0.0),
            maxiter=self.maxiter, sigma=self.sigma, n_hops=self.n_hops,
            temperature=gap, grad=grad,
        )
        point = {
            "topology": self.topology, "h": h, "n_qubits": self.n_qubits,
            "p_layers": self.p_layers, "J2": self.j2, "model": self.model,
            "ansatz": self.ansatz, "gs_method": gs_method,
            "e_vqe": res.energy, "e0_exact": float(e0), "gap": float(gap),
            "abs_error": float(abs(res.energy - e0)),
            "de_gap": float(abs(res.energy - e0) / gap) if gap > 0 else None,
            "fidelity": res.fidelity, "n_params": int(qc.num_parameters),
            "n_nn": n_nn, "n_nnn": n_nnn,
            "strategy_requested": self.strategy, "strategy_resolved": res.strategy,
            "runs": res.runs, "best_theta": res.best_theta,
            "_qc": qc,
        }
        point["diagnostics"] = self._diagnostic_metrics(qc, res, H, psi, gap)
        return point

    def _diagnostic_metrics(self, qc, res, H, psi, gap: float) -> dict:
        """Auto-computed diagnostics attached to every point (cheap, reused helpers).

        Always records per-restart convergence (from ``res.runs``). When the
        ground-state vector is available it adds the energy variance and the
        spectral decomposition of the prepared state (where the missing fidelity
        goes + dominant infidelity factor), reusing the shared fidelity helpers.
        """
        from qmbp_simulation.analysis.metrics import compute_restart_convergence

        diag: dict = {"convergence": compute_restart_convergence(res.runs, self.maxiter)}
        if psi is None:
            return diag
        try:
            from qmbp_simulation.analysis.fidelity import compute_state_spectral_decomposition

            theta = np.asarray(res.best_theta)
            var_h = self.backend.compute_energy_variance(qc, H, theta)
            # Low eigenvectors: reuse psi (ground) and derive the first-excited
            # partner cheaply from H only when the statevector is tractable.
            eigvecs = self._low_eigvecs(H, psi)
            diag["spectral"] = compute_state_spectral_decomposition(
                qc, theta, eigvecs, n_low=2,
                energy_variance=var_h, gap=gap,
            )
            if var_h is not None:
                diag["energy_variance"] = float(var_h)
        except Exception as exc:  # noqa: BLE001 - diagnostics never break a run
            diag["spectral_error"] = str(exc)
        return diag

    def _low_eigvecs(self, H, psi):
        """Return low eigenvectors as columns [ground | first_excited].

        Uses the already-solved ground vector ``psi`` for column 0 and a sparse
        ``eigsh(k=2)`` for the first-excited partner. Only called at tractable N
        (guarded by the caller's statevector path).
        """
        from scipy.sparse.linalg import eigsh

        evals, evecs = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
        order = np.argsort(evals)
        cols = evecs[:, order]
        cols[:, 0] = np.asarray(psi, dtype=complex)  # exact ground from solver
        return cols

    # ── lifecycle: persist a single point's reusable artifacts ───────────────
    def persist_point(self, point: dict, artifact_dir, *, save_theta: bool = True,
                      qpy_saver=None) -> dict:
        """Write a point's reusable artifacts (θ NPZ, optional bound QPY) and
        return a JSON-safe copy of ``point`` (``_qc`` dropped, ``best_theta``
        listified, artifact paths recorded).

        This is the ``persist`` step of the lifecycle for the common case; the
        traceable JSON itself is written by the caller's ``StudyCheckpoint``
        (so the results-tree layout stays in the study service, not in src).
        ``artifact_dir`` is created if missing.

        QPY serialization is not a ``src`` concern (the QPY helper lives in the
        study service), so it is injected: pass ``qpy_saver(circuit, path)`` to
        also write the bound circuit. When ``None`` (default), no QPY is written
        — avoiding a hard dependency on a serializer that lives outside ``src``.
        """
        from pathlib import Path

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in point.items() if k != "_qc"}
        theta = np.asarray(point["best_theta"])
        out["best_theta"] = theta.tolist()
        tag = f"{self.topology}_N{self.n_qubits}_p{self.p_layers}_h{point['h']:.2f}"

        if save_theta:
            theta_npz = artifact_dir / f"theta_{tag}.npz"
            np.savez(theta_npz, theta=theta, h=point["h"], e_vqe=point["e_vqe"],
                     e0=point["e0_exact"], gap=point["gap"], fidelity=point["fidelity"],
                     n_nn=point.get("n_nn"), n_nnn=point.get("n_nnn"),
                     n_qubits=self.n_qubits, p_layers=self.p_layers)
            out["theta_npz"] = str(theta_npz)
        if qpy_saver is not None and point.get("_qc") is not None:
            bound = point["_qc"].assign_parameters(theta)
            qpy_path = artifact_dir / f"circuit_{tag}.qpy"
            qpy_saver(bound, qpy_path)
            out["circuit_qpy"] = str(qpy_path)
        return out

    # ── lifecycle: full sweep with built-in checkpoint / resume ──────────────
    def run_sweep(self, h_values, checkpoint, *, artifact_dir=None,
                  extra=None, on_point=None, qpy_saver=None) -> dict:
        """Run an h-sweep with crash-safe per-h checkpointing and resume.

        The base owns the loop; the caller supplies only *what* to sweep and
        *where* to persist:

        - ``checkpoint`` — a :class:`StudyCheckpoint` (units_key e.g. ``"per_h"``)
          wired to the study's ``save_json`` writer. Already-completed h are
          skipped on resume.
        - ``artifact_dir`` — optional dir for θ NPZ / QPY per point.
        - ``extra`` — merged into every checkpoint write (schema, params, ...).
        - ``on_point`` — optional ``callback(point) -> None`` for extra recording.

        Persists after EVERY h (crash-safe) and re-raises on error with partial
        state already on disk. Returns the completed-unit map.
        """
        checkpoint.resume()
        for h in h_values:
            key = f"{h:.2f}"
            if checkpoint.is_done(key):
                continue
            point = self.optimize_point(h)
            if artifact_dir is not None:
                record = self.persist_point(point, artifact_dir, qpy_saver=qpy_saver)
            else:
                record = {k: v for k, v in point.items() if k != "_qc"}
                record["best_theta"] = np.asarray(point["best_theta"]).tolist()
            if on_point is not None:
                on_point(record)
            checkpoint.record(key, record)
            checkpoint.persist(extra=extra)
        return checkpoint.units


# ── src-side helpers duplicated minimally (avoid scripts/ import) ──────────────
# These mirror the study service's ground-state / fidelity helpers using only
# src modules so StudyRunner stays importable from src and tests. The study
# service keeps its own copies for direct script use; both call the same
# underlying GroundTruthCache / compute_exact_fidelity, so numbers are identical.


def _exact_ground_state_vector(topology, n, h, *, model="tfim", j2=0.0):
    from qmbp_simulation import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers import ClassicalSolver
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    model_kwargs = None
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2
        model_kwargs = {"J2": j2} if j2 else None
    solver = ClassicalSolver()
    e0, gap = GroundTruthCache().get_or_compute(
        topology, n, model, h, model_kwargs=model_kwargs, solver=solver
    )
    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **ham_kwargs)
    gt = solver.solve(H, lat)
    psi = None
    if gt.ground_state is not None:
        psi = np.asarray(gt.ground_state, dtype=complex)
        psi /= np.linalg.norm(psi)
    return psi, float(e0), float(gap), H


def _eigsh_ground_state(topology, n, h, *, model="tfim", j2=0.0):
    from scipy.sparse.linalg import eigsh

    from qmbp_simulation import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2
    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **ham_kwargs)
    cache = GroundTruthCache()
    cached = cache.get(topology, n, model, h)
    evals, evecs = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
    idx = np.argsort(evals)
    psi = evecs[:, idx[0]].astype(complex)
    psi /= np.linalg.norm(psi)
    e0 = float(evals[idx[0]])
    gap = float(evals[idx[1]] - evals[idx[0]])
    if cached is not None:
        return psi, float(cached["energy"]), float(cached["gap"]), H
    cache.put(topology, n, model, h, energy=e0, gap=gap, method="eigsh_k2")
    cache.flush()
    return psi, e0, gap, H


def _state_fidelity_exact(circuit, theta, psi_exact) -> float:
    from qmbp_simulation.analysis.fidelity import compute_exact_fidelity

    fid = compute_exact_fidelity(circuit, np.asarray(theta), psi_exact)
    return float(fid) if fid is not None else 0.0
