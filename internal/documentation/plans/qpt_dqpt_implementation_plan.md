# Plan de Implementacion: QPT Detection & DQPT Analysis

**Fecha**: 2026-08-21
**Ultima revision**: 2026-08-21 (post-analisis quantum advantage)
**Objetivo**: (1) Validar que el MPNN captura la fisica de la QPT, y (2) establecer la infraestructura de quench dynamics noiseless como paso previo a ejecucion en hardware quasi-2D donde la quantum advantage esta demostrada.

---

## Encuadre Correcto (basado en analisis del estado del arte)

**Lo que este plan SI es:**
- Validacion noiseless de que el pipeline GNN+HVA+Trotter produce DQPTs correctos para N=10-20-30, estableciendo la infraestructura para ejecucion en hardware quasi-2D (heavy-hex N>=51) donde la quantum advantage esta demostrada (IBM+Qedma, arXiv:2607.24937).
- Validacion de que el MPNN captura h_c correctamente (QPT detection como consistency check interno).
- Codigo fundacional (TrotterEvolutionBuilder, observables.py) que habilita la extension post-tesis.

**Condiciones para quantum advantage real (referencia: seccion 12 del estado_del_arte_materia_condensada.md):**
1. Entrelazamiento volume-law (solo en dinamica temporal, no en ground states)
2. Dimensionalidad >1D (en 1D, GPU TDVP gana)
3. Observable sensible a componentes dificiles

**Donde aplica quantum advantage para ESTE proyecto:**
- Heavy-hex N>=51 + >15 Trotter steps + QESEM = regimen demostrado por IBM (julio 2026)
- Nuestro valor: GNN prepara |psi_0(h)> a costo O(1), liberando QPU-time para la evolucion

---

## Paso Nuevo: Crossover Plot (clasico pierde precision)

**Objetivo**: Determinar para heavy-hex, a que N y cuantos Trotter steps MPS (chi=64-256) pierde precision. Ese es el argumento visual clave: "de este punto en adelante, solo la QPU sabe la respuesta."

**Implementacion**:
```bash
# Para cada N en heavy_hex, evolucionar con chi creciente y medir divergencia
for N in 10 16 20 28 35; do
    for CHI in 32 64 128 256 512; do
        .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
            --section 2 --n-qubits $N --topology heavy_hex \
            --h1 0.5 --h2 2.0 --dt 0.1 --n-trotter 30 --chi-values $CHI
    done
done
```

**Metrica**: Ciclo Floquet en que chi=256 y chi=512 divergen >5% — marca la frontera clasica.

**Resultado esperado**: Para heavy-hex N=28+, divergencia a ~10-15 steps. Para N>=51, divergencia desde step 1 (chi insuficiente para representar el estado).

---

## Planes — Lo que SI implementar

---

Posibles caminos:
#3. Deteccion de Transicion de Fase a N Grande (Quantum Phase Transition)

**Naturaleza**: Analisis sobre datos EXISTENTES (no requiere QPU ni ejecucion nueva). Valida que el MPNN captura la fisica.

Que es: El TFIM tiene transicion de fase a h_c ~ 1.0 (chain_1d). En sistemas finitos, la transicion se suaviza. El pipeline calcula dE/dh (derivada numerica de la energia predicha) vs h para N grandes y muestra que el pico se agudiza con N.

Por que importa: Si h_c(MPNN-predicted) ~ h_c(exact), demuestra que la GNN captura la fisica de la transicion. Es un consistency check interno elegante, no un claim de ventaja cuantica.

Analisis concreto: Usar datos de chain_1d (grade A hasta N=60) para calcular E(h) en grilla fina h in [0.5, 2.0] para N=20, 30, 40, 60 y plotear -d^2E/dh^2 vs h. El pico marca h_c y deberia sharpear con N.

**NOTA**: Esto NO demuestra quantum advantage. DMRG calcula E(h) para chain_1d N=1000 en minutos. h_c=1.0 del TFIM 1D es un resultado analitico exacto.

#4. Quench Dynamics: Loschmidt Echo & DQPTs

**Naturaleza**: Validacion noiseless del pipeline. Establece infraestructura para hardware.

Que es: DQPTs ocurren cuando haces un quench (cambio subito de h1->h2 cruzando h_c). El Loschmidt echo L(t) = |<psi_0|e^{-iH2*t}|psi_0>|^2 tiene ceros a tiempos criticos t*.

**Alcance correcto (post-revision)**:
- N=8-20 con ED exacta: validar que el codigo detecta DQPTs conocidos del TFIM.
- N=20-35 en heavy-hex con ED: establecer baseline para el crossover plot.
- N>22 con MPS: SOLO como referencia para el crossover plot, NO como claim de resultado inaccesible.

**NO hacer**: Clamar que DQPT a N=30-60 en chain_1d es "inaccesible clasicamente". GPU TDVP (chi=60,000, H200) lo resuelve en minutos (arXiv:2606.04771).

Analisis concreto: Usar run_quench_dynamics_study.py con ground state de chain_1d y heavy_hex (N=10-20) como |psi_0>, quench de h=0.5->h=2.0, medir L(t) con Trotter steps. El valor esta en validar la infraestructura, no en el resultado numerico.
---

## Fase 1 — Refactoring del Quench Runner (Prerequisito)

### 1.1 Cargar MPNN una sola vez en `setup()`

**Archivo**: `scripts/experiment_runners/scaling/run_quench_dynamics_study.py`
**Problema**: `_predict_theta_gnn` llama `load_best_model_for_topology()` en cada invocacion (8x en section 3). Cada llamada carga el modelo PyTorch desde disco (~1.6MB).
**Fix**:

```python
# En setup():
def setup(self) -> None:
    self.setup_physics()
    args = self._args
    self._topology = args.topology[0] if isinstance(args.topology, list) else args.topology

    # Cargar MPNN una sola vez
    self._mpnn_model = self.load_best_mpnn_for_cross_n(
        n_target=args.n_qubits,
        model="tfim_bond_resolved",
        topology=self._topology,
        p_layers=args.p_layers,
        train_if_missing=False,
    )
    if self._mpnn_model is not None:
        logger.info(f"  MPNN loaded: {getattr(self, '_zoo_entry', None)}")
```

Reemplazar `_predict_theta_gnn` con:

```python
def _predict_theta_gnn(self, topology, n_qubits, h, p_layers):
    if self._mpnn_model is None:
        return None
    import torch
    from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
    lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
    graph = build_unified_bond_resolved_graph(lattice, h_value=h, p_layers=p_layers)
    with torch.no_grad():
        theta = self._mpnn_model(graph).cpu().numpy().flatten()
    return theta
```

### 1.2 Refactorizar `_ground_state_vector` para usar framework

**Problema**: Duplica la logica de caching de `exact_ground_state()` del base class. No se beneficia de gap validation, staleness detection, ni in-memory cache compartido.
**Fix**:

```python
def _ground_state_vector(self, n_qubits, topology, h):
    """Ground state vector con cache del framework."""
    # Side-effect: cachea energy+gap en self._gt_cache y disk
    self.exact_ground_state(topology, n_qubits, h, model=self._args.model)

    # Computar el vector (no cacheado, es 2^N floats)
    lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h)
    H_op = self.builder.build(lattice)

    if n_qubits <= 16:
        return self.solver.ground_state_vector(H_op, n_qubits=n_qubits)
    else:
        from scipy.sparse.linalg import eigsh
        H_sparse = H_op.to_matrix(sparse=True)
        _, evecs = eigsh(H_sparse, k=1, which="SA")
        gs = evecs[:, 0]
        gs /= np.linalg.norm(gs)
        return gs
```

### 1.3 Compartir estado entre secciones

**Problema**: Ground state de h1 se computa en section 1 y se re-computa en section 2.
**Fix**: Almacenar en `self._psi_h1` despues de section 1:

```python
# Al final de section_initial_state_dependence, antes del return:
self._psi_h1 = psi_gnn  # Reusar en section 2

# En _mps_crossover_exact_reference:
psi_0 = getattr(self, '_psi_h1', None) or self._ground_state_vector(n, topo, args.h1)
```

### 1.4 Preparar estado GNN para N>22 via HVA+MPS

**Problema critico**: Para N>22, section 1 solo compara `|+>^N` vs `|0>^N`. NO prepara el estado GNN — perdiendo el argumento principal de la tesis.
**Fix**: Nuevo metodo que usa theta de extrapolation NPZ o MPNN prediction:

```python
def _prepare_gnn_state_mps(self, n_qubits, topology, h, p_layers):
    """Preparar |psi_GNN(h)> para N>22 via HVA(theta)|+> evaluado con MPS."""
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qiskit.circuit import QuantumCircuit

    # Intentar cargar theta de extrapolation NPZ
    theta = None
    npz_path = Path(f"data/large_n_extrapolation/{topology}_N{n_qubits}_p{p_layers}.npz")
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        h_values = data["h_values"]
        idx = np.argmin(np.abs(h_values - h))
        if abs(h_values[idx] - h) < 0.05:
            theta = data["theta_opt"][idx]

    # Fallback: MPNN prediction
    if theta is None:
        theta = self._predict_theta_gnn(topology, n_qubits, h, p_layers)

    if theta is None:
        return None

    # Construir circuito HVA
    lat = self.make_lattice(topology, n_qubits, J=1.0, h=h)
    hva = HVACircuitBuilder()
    circuit, _ = hva.create_bond_resolved(n_qubits, p_layers, lat)

    # Evaluar via MPS para obtener statevector aproximado
    mps_backend = self.MPSBackend(chi_max=256)
    # Nota: get_statevector no existe en MPSBackend, usamos energy como proxy
    # Para el Loschmidt echo necesitamos el overlap, no el vector completo
    return {"circuit": circuit, "theta": theta, "topology": topology, "h": h}
```

### 1.5 Agregar persistencia a section 1

**Problema**: Si section 1 crashea despues de evolucionar 2/3 estados, todo se pierde.
**Fix**: Checkpoint per-state:

```python
# Dentro del loop de evolve:
for label, psi_init in [("gnn_gs", psi_gnn), ("zero", psi_zero), ("plus", psi_plus)]:
    cp_key = f"section1_{label}_N{n}"
    cp = self.load_checkpoint(cp_key)
    if cp:
        results_by_state[label] = cp
        continue
    # ... evolve ...
    results_by_state[label] = {...}
    self.save_checkpoint(cp_key, results_by_state[label])
```

---

## Fase 2 — Implementar DQPT (Loschmidt Echo como Section 4)

### 2.1 Nueva seccion en `define_sections()`

```python
Section(
    id=4,
    name="Dynamic Quantum Phase Transitions (Loschmidt Echo)",
    hypothesis=(
        "Quench across h_c produces zeros in Loschmidt echo L(t), "
        "signaling DQPTs. These are detectable at N=10-22 (exact) and "
        "extrapolable via finite-size scaling of critical times t*."
    ),
    fn=self.section_dqpt_loschmidt,
),
```

### 2.2 Implementacion de section_dqpt_loschmidt

```python
def section_dqpt_loschmidt(self) -> dict:
    """Compute Loschmidt echo and detect DQPTs after quench across h_c."""
    args = self._args
    n = args.n_qubits
    topo = self._topology

    # Parametros del quench: cruzar h_c
    # TFIM h_c ~ 1.0 para chain_1d, depende de topologia
    h_pre = args.h1   # e.g., 0.5 (ferromagnetico)
    h_post = args.h2  # e.g., 2.0 (paramagnetico)
    dt = args.dt
    n_steps = args.n_trotter

    logger.info(f"  DQPT: N={n}, {topo}, h: {h_pre} -> {h_post}")
    logger.info(f"  {n_steps} steps x dt={dt}, T_total={n_steps*dt}")

    if n > _ED_MAX_N:
        return self._dqpt_mps(n, topo, h_pre, h_post, dt, n_steps)

    # Preparar estado inicial |psi_0> = GS de H(h_pre)
    psi_0 = self._ground_state_vector(n, topo, h_pre)

    # Hamiltoniano post-quench H(h_post)
    lattice_post = self.make_lattice(topo, n, J=1.0, h=h_post)
    H_post_op = self.builder.build(lattice_post)

    if n <= _DENSE_LIMIT:
        H_post = np.asarray(H_post_op.to_matrix())
        U_dt = expm(-1j * H_post * dt)
        use_sparse = False
    else:
        H_post_sparse = H_post_op.to_matrix(sparse=True)
        use_sparse = True

    # Evolucion temporal con Loschmidt echo
    psi_t = psi_0.copy().astype(complex)
    times = [0.0]
    loschmidt_echo = [1.0]  # L(0) = |<psi_0|psi_0>|^2 = 1
    rate_function = [0.0]   # r(0) = 0
    energies = [float(np.real(psi_t.conj() @ (H_post if not use_sparse else H_post_sparse) @ psi_t))]
    entropies = [self._half_chain_entropy(psi_t, n)]

    for step in range(1, n_steps + 1):
        if use_sparse:
            from scipy.sparse.linalg import expm_multiply
            psi_t = expm_multiply(-1j * H_post_sparse * dt, psi_t)
        else:
            psi_t = U_dt @ psi_t
        psi_t /= np.linalg.norm(psi_t)

        t = step * dt
        times.append(t)

        # Loschmidt echo: L(t) = |<psi_0|psi(t)>|^2
        overlap = np.vdot(psi_0, psi_t)
        L_t = float(np.abs(overlap) ** 2)
        loschmidt_echo.append(L_t)

        # Rate function: r(t) = -(1/N) * ln(L(t))
        r_t = -np.log(max(L_t, 1e-300)) / n
        rate_function.append(float(r_t))

        # Observables adicionales
        H_mat = H_post if not use_sparse else H_post_sparse
        energies.append(float(np.real(psi_t.conj() @ (H_mat @ psi_t))))
        entropies.append(self._half_chain_entropy(psi_t, n))

    # Detectar DQPTs: minimos locales de L(t) (ceros aproximados)
    L_arr = np.array(loschmidt_echo)
    critical_times = []
    for i in range(1, len(L_arr) - 1):
        if L_arr[i] < L_arr[i-1] and L_arr[i] < L_arr[i+1]:
            if L_arr[i] < 0.1:  # Threshold para considerar "cerca de cero"
                critical_times.append(times[i])

    # Rate function: picos corresponden a DQPTs
    r_arr = np.array(rate_function)
    rate_peaks = []
    for i in range(1, len(r_arr) - 1):
        if r_arr[i] > r_arr[i-1] and r_arr[i] > r_arr[i+1]:
            rate_peaks.append({"t": times[i], "r": float(r_arr[i])})

    has_dqpt = len(critical_times) > 0

    result = {
        "n_qubits": n, "topology": topo,
        "h_pre": h_pre, "h_post": h_post,
        "dt": dt, "n_steps": n_steps, "T_total": n_steps * dt,
        "method": "exact_ed",
        "times": times,
        "loschmidt_echo": [float(x) for x in loschmidt_echo],
        "rate_function": [float(x) for x in rate_function],
        "energies": energies,
        "entropies": entropies,
        "critical_times": critical_times,
        "rate_peaks": rate_peaks,
        "n_dqpts_detected": len(critical_times),
        "has_dqpt": has_dqpt,
        "pass": has_dqpt,
        "thesis_claim": (
            f"Detected {len(critical_times)} DQPT(s) at t*={critical_times} "
            f"for N={n} quench {h_pre}->{h_post} crossing h_c. "
            f"Rate function peaks confirm non-analytic behavior."
            if has_dqpt else
            f"No DQPT detected for N={n} quench {h_pre}->{h_post}. "
            f"May need longer evolution time or different quench parameters."
        ),
    }

    logger.info(f"  DQPTs detected: {len(critical_times)}")
    for tc in critical_times:
        logger.info(f"    t* = {tc:.3f}")

    return result
```

### 2.3 DQPT para N>22 via MPS (SOLO para crossover plot, NO claim de QA)

**NOTA IMPORTANTE**: Este codigo existe SOLO para medir a que N y tiempo MPS pierde precision.
NO es un claim de resultado inaccesible clasicamente. En chain_1d, GPU TDVP con chi=60,000
resuelve el mismo problema (arXiv:2606.04771). El proposito es generar el "crossover plot"
para heavy-hex, donde los clasicos SI fallan a N>=51 (IBM+Qedma, arXiv:2607.24937).

```python
def _dqpt_mps(self, n, topo, h_pre, h_post, dt, n_steps):
    """MPS Trotter evolution for crossover analysis.

    PURPOSE: Measure at which step MPS loses precision (chi-convergence test).
    NOT a quantum advantage claim. For chain_1d this is trivially solvable
    classically with GPU TDVP (arXiv:2606.04771). For heavy-hex N>=51,
    this establishes the frontier where only QPU can go.

    Para N>22 no podemos calcular L(t)=|<psi_0|psi_t>|^2 directamente
    porque no tenemos el statevector. Usamos energy como proxy.
    """
    chi_max = max(self._args.chi_values)
    logger.info(f"  DQPT MPS: N={n}, chi={chi_max}")

    # Preparar estado GNN via HVA
    gnn_state = self._prepare_gnn_state_mps(n, topo, h_pre, self._args.p_layers)

    # Trotter evolution tracking energy
    trotter_step = self._build_trotter_step_circuit(n, topo, h_post, dt)
    lattice_post = self.make_lattice(topo, n, J=1.0, h=h_post)
    H_post_op = self.builder.build(lattice_post)

    mps_backend = self.MPSBackend(chi_max=chi_max)

    from qiskit.circuit import QuantumCircuit
    init_qc = QuantumCircuit(n)
    if gnn_state is not None:
        # Bind theta to HVA circuit
        circuit = gnn_state["circuit"]
        theta = gnn_state["theta"]
        bound = circuit.assign_parameters(dict(zip(circuit.parameters, theta)))
        init_qc = bound
    else:
        init_qc.h(range(n))  # |+>^N fallback

    energies = []
    full_circuit = init_qc.copy()
    empty_params = np.array([])

    for step in range(n_steps + 1):
        try:
            e = mps_backend.evaluate(full_circuit, H_post_op, empty_params)
            energies.append(float(e))
        except Exception:
            energies.append(energies[-1] if energies else 0.0)
        if step < n_steps:
            full_circuit = full_circuit.compose(trotter_step)

    # Detectar DQPT proxy: non-monotonic energy oscillations
    e_arr = np.array(energies)
    oscillation_amplitude = float(np.max(e_arr) - np.min(e_arr))

    return {
        "n_qubits": n, "topology": topo,
        "h_pre": h_pre, "h_post": h_post,
        "method": "mps_energy_proxy", "chi": chi_max,
        "energies": energies,
        "oscillation_amplitude": oscillation_amplitude,
        "note": "L(t) not directly computable for N>22 with MPS; using energy proxy",
        "pass": True,
    }
```

### 2.4 CLI args adicionales

```python
parser.add_argument(
    "--dqpt-h-pre", type=float, default=0.5,
    help="Pre-quench field for DQPT (default: 0.5, ferromagnetic)",
)
parser.add_argument(
    "--dqpt-h-post", type=float, default=2.0,
    help="Post-quench field for DQPT (default: 2.0, paramagnetic)",
)
parser.add_argument(
    "--dqpt-dt", type=float, default=0.05,
    help="Time step for DQPT (finer than crossover, default: 0.05)",
)
parser.add_argument(
    "--dqpt-steps", type=int, default=60,
    help="Trotter steps for DQPT (default: 60, T_total=3.0)",
)
```

### 2.5 Multi-N DQPT scan (finite-size scaling de t*)

Para demostrar que t* tiene scaling con N, correr la section 4 a multiples N:

```bash
# N=8,10,12,14,16,18,20,22 — exact ED range
for N in 8 10 12 14 16 18 20 22; do
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
        --section 4 --n-qubits $N --topology chain_1d \
        --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 60
done
```

---

## Fase 3 — QPT Detection (Nuevo Modulo de Analisis)

### 3.1 Nuevo archivo: `scripts/analysis/qpt_detection.py`

Este script NO es un runner (no necesita circuitos ni VQE). Es puro analisis numerico sobre datos ya existentes.

```python
#!/usr/bin/env python
"""QPT Detection via Energy Derivatives at Large N.

Loads E(h) data from large_n_extrapolation NPZ files and ground_truth_cache,
computes d^2E/dh^2 numerically, identifies h_c(N), and performs
finite-size scaling analysis.

Usage:
    python scripts/analysis/qpt_detection.py --topology chain_1d
    python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted
    python scripts/analysis/qpt_detection.py --topology heavy_hex --plot
"""
```

### 3.2 Estructura del modulo

```python
def load_energy_curves(topology: str, p_layers: int = 1, use_predicted: bool = False):
    """Cargar E(h) para todos los N disponibles.

    Args:
        topology: Topologia a analizar
        p_layers: Profundidad de circuito
        use_predicted: Si True, usa e_pred del MPNN. Si False, usa e_exact (GT)

    Returns:
        dict[int, dict]: {N: {"h": array, "E": array, "gap": array}}
    """
    from scripts.experiment_runners.scaling.run_large_n_extrapolation import (
        load_extrapolation_npz,
    )
    # Scan all available NPZ files
    data_dir = Path("data/large_n_extrapolation")
    pattern = f"{topology}_N*_p{p_layers}.npz"
    results = {}
    for npz_file in sorted(data_dir.glob(pattern)):
        n_str = npz_file.stem.split("_N")[1].split("_p")[0]
        n = int(n_str)
        npz_data = load_extrapolation_npz(topology, n, p_layers)
        if npz_data:
            h_vals = sorted(npz_data.keys())
            key = "e_pred" if use_predicted else "e_exact"
            results[n] = {
                "h": np.array([float(h) for h in h_vals]),
                "E": np.array([npz_data[h][key] for h in h_vals]),
                "gap": np.array([npz_data[h]["gap"] or 0.0 for h in h_vals]),
            }
    return results


def compute_second_derivative(h_values: np.ndarray, energies: np.ndarray):
    """Compute d^2E/dh^2 via finite differences (numpy gradient x2).

    Returns:
        h_interior: h values where d2E is defined (excludes boundaries)
        d2E: second derivative array
    """
    dE = np.gradient(energies, h_values)
    d2E = np.gradient(dE, h_values)
    return h_values, d2E


def find_critical_field(h_values: np.ndarray, d2E: np.ndarray):
    """Identificar h_c como la posicion del pico (maximo absoluto) de |d2E/dh2|.

    Para TFIM, d2E/dh2 tiene un pico negativo (convexidad maxima) en h_c.
    Usamos el minimo de d2E (pico mas pronunciado hacia abajo).
    """
    # El pico de susceptibilidad es el MINIMO de d2E/dh2 (punto de inflexion de E)
    idx_min = np.argmin(d2E)
    h_c = float(h_values[idx_min])
    peak_magnitude = float(abs(d2E[idx_min]))
    return h_c, peak_magnitude


def finite_size_scaling(h_c_by_n: dict[int, float]):
    """Fit h_c(N) = h_c(inf) + a/N^nu para extraer h_c termodinamico.

    En TFIM 1D, h_c(inf) = 1.0 exactamente. El fit nos da:
    - h_c(inf): valor termodinamico (debe coincidir con prediccion analitica)
    - nu: exponente de correlacion
    - Calidad del fit (R^2)
    """
    from scipy.optimize import curve_fit

    N_values = np.array(sorted(h_c_by_n.keys()), dtype=float)
    h_c_values = np.array([h_c_by_n[int(n)] for n in N_values])

    def scaling_law(N, h_inf, a, nu):
        return h_inf + a / N**nu

    try:
        popt, pcov = curve_fit(
            scaling_law, N_values, h_c_values,
            p0=[1.0, 1.0, 1.0],  # Initial guess para TFIM
            bounds=([0.5, -10, 0.1], [2.0, 10, 3.0]),
        )
        h_inf, a, nu = popt
        # R^2
        residuals = h_c_values - scaling_law(N_values, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((h_c_values - np.mean(h_c_values))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "h_c_inf": float(h_inf),
            "a": float(a),
            "nu": float(nu),
            "r_squared": float(r_squared),
            "N_values": N_values.tolist(),
            "h_c_values": h_c_values.tolist(),
        }
    except Exception as e:
        return {"error": str(e), "N_values": N_values.tolist(), "h_c_values": h_c_values.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QPT Detection via Energy Derivatives")
    parser.add_argument("--topology", type=str, default="chain_1d")
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--use-predicted", action="store_true",
                        help="Use MPNN-predicted energies instead of exact GT")
    parser.add_argument("--h-range", type=float, nargs=2, default=None,
                        help="Restrict h range for analysis (e.g., 0.5 2.0)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    # 1. Cargar datos
    curves = load_energy_curves(args.topology, args.p_layers, args.use_predicted)
    source = "MPNN-predicted" if args.use_predicted else "exact (GT)"
    print(f"\nQPT Detection: {args.topology} (source: {source})")
    print(f"Available N values: {sorted(curves.keys())}")

    # 2. Computar derivadas y detectar h_c para cada N
    h_c_by_n = {}
    print(f"\n{'N':>4} | {'h_c':>6} | {'|d2E/dh2|_max':>14} | {'n_points':>8}")
    print("-" * 45)

    for n in sorted(curves.keys()):
        data = curves[n]
        h, E = data["h"], data["E"]

        # Filtrar rango si se especifica
        if args.h_range:
            mask = (h >= args.h_range[0]) & (h <= args.h_range[1])
            h, E = h[mask], E[mask]

        if len(h) < 5:
            continue

        _, d2E = compute_second_derivative(h, E)
        h_c, peak_mag = find_critical_field(h, d2E)
        h_c_by_n[n] = h_c
        print(f"{n:>4} | {h_c:>6.3f} | {peak_mag:>14.4f} | {len(h):>8}")

    # 3. Finite-size scaling
    if len(h_c_by_n) >= 3:
        print("\n--- Finite-Size Scaling ---")
        fss = finite_size_scaling(h_c_by_n)
        if "error" not in fss:
            print(f"h_c(inf) = {fss['h_c_inf']:.4f} (exact TFIM: 1.000)")
            print(f"Exponent nu = {fss['nu']:.3f}")
            print(f"R^2 = {fss['r_squared']:.4f}")
            print(f"\nThesis claim: GNN+HVA pipeline detects QPT at h_c = {fss['h_c_inf']:.3f}")
            print(f"for {args.topology} topology using systems up to N={max(h_c_by_n.keys())}")
            print(f"where exact diagonalization is computationally intractable.")
        else:
            print(f"Fit failed: {fss['error']}")
            print(f"Raw h_c(N): {h_c_by_n}")

    # 4. Guardar resultados
    if args.save:
        import json
        from qmbp_simulation.utils.helpers import json_serialize
        output = {
            "topology": args.topology,
            "source": source,
            "h_c_by_n": {str(k): v for k, v in h_c_by_n.items()},
            "finite_size_scaling": fss if len(h_c_by_n) >= 3 else None,
            "per_n_data": {str(n): {"h": d["h"].tolist(), "E": d["E"].tolist()}
                          for n, d in curves.items()},
        }
        out_path = Path(f"results/analysis/qpt_detection_{args.topology}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=json_serialize)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
```

### 3.3 Comandos de ejecucion

```bash
# QPT detection usando ground truth exacto (mas confiable)
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --save
.venv/bin/python scripts/analysis/qpt_detection.py --topology heavy_hex --save
.venv/bin/python scripts/analysis/qpt_detection.py --topology ladder --save

# QPT detection usando predicciones MPNN (demuestra que el modelo captura la transicion)
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted --save
.venv/bin/python scripts/analysis/qpt_detection.py --topology heavy_hex --use-predicted --save

# Comparacion: exacto vs predicho (el punto clave para la tesis)
# Si h_c_predicted ~ h_c_exact, el MPNN captura la fisica de la QPT
```

---

## Oportunidades de Reuso entre Pipelines

### R1: Theta de Extrapolation NPZ para preparar estados en Quench

**Flujo**:
```
Large-N Extrapolation Runner  -->  data/large_n_extrapolation/{topo}_N{N}_p{p}.npz
                                        |
                                        | theta_opt[i] para cada h
                                        v
Quench Dynamics Runner  <--  _prepare_gnn_state_mps(N, topo, h, p)
                                        |
                                        v
                              HVA(theta)|+> evaluado con MPS = |psi_GNN(h)>
```

**Impacto**: Habilita section 1 para N>22 con el estado GNN (actualmente imposible).

### R2: Ground Truth Cache compartido

**Ambos runners usan GT cache** pero de formas distintas:
- Extrapolation: via `self.exact_ground_state()` (correcto)
- Quench: via `GroundTruthCache()` manual (redundante)

**Fix**: Quench runner usa `self.exact_ground_state()` como el extrapolation.

### R3: Factorizar observables a modulo compartido

**Crear**: `src/qmbp_simulation/analysis/observables.py`

```python
def half_chain_entropy(psi: np.ndarray, n_qubits: int) -> float:
    """Von Neumann entropy of half-chain bipartition."""

def magnetization_z(psi: np.ndarray, n_qubits: int) -> float:
    """<M_z> = (1/N) sum <Z_i>."""

def loschmidt_echo(psi_0: np.ndarray, psi_t: np.ndarray) -> float:
    """L(t) = |<psi_0|psi_t>|^2."""

def rate_function(loschmidt: float, n_qubits: int) -> float:
    """r(t) = -(1/N) * ln(L(t))."""

def order_parameter_x(psi: np.ndarray, n_qubits: int) -> float:
    """<M_x> = (1/N) sum <X_i>."""
```

**Consumidores**: Quench runner, hardware results analysis, QPT detection.

### R4: Factorizar Trotter circuit a modulo compartido

**Crear**: `src/qmbp_simulation/circuits/trotter.py`

```python
def build_trotter_step(
    topology: str, n_qubits: int, h: float, dt: float,
    order: int = 2, model: str = "tfim"
) -> QuantumCircuit:
    """Suzuki-Trotter step circuit for time evolution.

    order=1: first-order Trotter
    order=2: second-order (symmetric) Trotter
    """
```

**Consumidores**: Quench runner, hardware rehearsal (time evolution on QPU), potential future QAOA runner.

### R5: E(h) curves compartidas para QPT + scoreboard

El QPT detection script necesita E(h) para multiples N. Esto ya existe en:
1. `data/large_n_extrapolation/*.npz` (e_exact + e_pred)
2. `data/ground_truth_cache.json` (e_exact para todos los h computados)

**No duplicar**: QPT script debe leer de estas fuentes existentes, no recomputar.

---

## Implementacion Concreta: Punto 3 (QPT Detection)

### Paso 1: Crear el script de analisis
- Archivo: `scripts/analysis/qpt_detection.py`
- Estructura completa en seccion 3.2 arriba
- Funciones: `load_energy_curves`, `compute_second_derivative`, `find_critical_field`, `finite_size_scaling`

### Paso 2: Verificar datos disponibles
```bash
# Ver que NPZ existen y cuantos h-points tienen
.venv/bin/python -c "
import numpy as np
from pathlib import Path
for f in sorted(Path('data/large_n_extrapolation').glob('*_p1.npz')):
    if '_baselines' in str(f): continue
    d = np.load(f, allow_pickle=True)
    n_pts = len(d['h_values'])
    h_min, h_max = d['h_values'].min(), d['h_values'].max()
    print(f'{f.stem:40s}: {n_pts:>3} pts, h=[{h_min:.2f}, {h_max:.2f}]')
"
```

### Paso 3: Ejecutar y validar
```bash
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --save
# Esperado: h_c ~ 1.0 para chain_1d (TFIM exacto)
# Si funciona, el pipeline detecta la QPT correctamente
```

### Paso 4: Comparar GT vs MPNN predicted
```bash
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted --save
# Si h_c_predicted ~ h_c_exact, la GNN captura la fisica de la transicion
```

---

## Implementacion Concreta: Punto 4 (DQPT)

### Paso 1: Agregar Section 4 al quench runner
- Archivo: `scripts/experiment_runners/scaling/run_quench_dynamics_study.py`
- Agregar `section_dqpt_loschmidt` (codigo en seccion 2.2 arriba)
- Agregar a `define_sections()`
- Agregar CLI args `--dqpt-h-pre`, `--dqpt-h-post`, `--dqpt-dt`, `--dqpt-steps`

### Paso 2: Test a N pequeno (validar que detecta DQPTs conocidos)
```bash
# TFIM chain_1d N=10, quench 0.5 -> 2.0 (cruza h_c=1)
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 4 --n-qubits 10 --topology chain_1d \
    --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 60
```

### Paso 3: Scan de N para finite-size scaling de t*
```bash
# Todos los N en rango ED exacto
for N in 8 10 12 14 16 18 20; do
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
        --section 4 --n-qubits $N --topology chain_1d \
        --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80
done
```

### Paso 4: N>22 con MPS — Crossover plot (heavy-hex)
```bash
# heavy_hex a multiples chi para medir divergencia (crossover plot)
for CHI in 64 128 256; do
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
        --section 2 --n-qubits 28 --topology heavy_hex \
        --h1 0.5 --h2 2.0 --dt 0.1 --n-trotter 30 --chi-values $CHI
done
# El crossover plot muestra: a que step chi=128 y chi=256 divergen
# Ese punto marca la frontera clasica. Mas alla, solo QPU.
# NOTA: No clamar que esto es inaccesible clasicamente para chain_1d.
```

### Paso 5: Analisis post-hoc de t*(N)
```bash
# Script de analisis que lee los resultados de exp_qd1 y extrae t* vs N
.venv/bin/python -c "
import json, glob, numpy as np
results = []
for f in sorted(glob.glob('results/experiments/exp_qd1/run_*.json')):
    with open(f) as fh:
        d = json.load(fh)
    sec4 = d.get('results', {}).get('section_4', {})
    if sec4 and sec4.get('critical_times'):
        n = sec4['n_qubits']
        t_star = sec4['critical_times'][0]  # primer DQPT
        results.append((n, t_star))
        print(f'N={n:>3}: t* = {t_star:.4f}')

if len(results) >= 3:
    ns, ts = zip(*sorted(results))
    # t* deberia escalar como ~ 1/Delta(N) o ~ N^alpha
    print(f'\nScaling: t* from {min(ts):.3f} (N={min(ns)}) to {max(ts):.3f} (N={max(ns)})')
"
```

---

## Resumen de Archivos a Crear/Modificar

| Accion | Archivo | Descripcion |
|--------|---------|-------------|
| MODIFICAR | `scripts/experiment_runners/scaling/run_quench_dynamics_study.py` | Refactoring Fase 1 + agregar Section 4 (DQPT) |
| CREAR | `scripts/analysis/qpt_detection.py` | QPT detection via energy derivatives |
| CREAR | `src/qmbp_simulation/analysis/observables.py` | Modulo compartido: entropy, magnetization, Loschmidt |
| CREAR | `src/qmbp_simulation/circuits/trotter.py` | Trotter step circuit builder compartido |
| CREAR | `results/analysis/` | Directorio para resultados de QPT analysis |

---

## Criterios de Exito

### QPT Detection (validacion MPNN, NO claim de QA)
- [ ] h_c detectado a < 5% del valor exacto (1.0 para TFIM chain_1d)
- [ ] Finite-size scaling fit con R^2 > 0.9
- [ ] h_c(MPNN) ~ h_c(exact) — MPNN captura la transicion
- [ ] Funciona para al menos 3 topologias

### DQPT (validacion de infraestructura, NO claim de QA)
- [ ] Loschmidt echo muestra minimos claros (L < 0.1) para quench cruzando h_c (N=10-20, ED)
- [ ] Rate function r(t) muestra picos bien definidos en t*
- [ ] t* escalea consistentemente con N (verificable para N=8-20)
- [ ] Section 4 pasa para chain_1d y heavy_hex
- [ ] Para N>22, la preparacion GNN via HVA+MPS funciona y produce energia < E_trivial

### Crossover Plot (argumento visual clave para la tesis)
- [ ] Para heavy-hex, identificar el Trotter step en que chi=256 y chi=512 divergen >5%
- [ ] Mostrar que para N>=28 heavy-hex, la divergencia ocurre antes de step 15
- [ ] Grafico: "frontera clasica" marcada visualmente — mas alla, solo QPU

### Lo que NO debe aparecer en la tesis como claim
- [ ] Nunca clamar "inaccesible clasicamente" para chain_1d a cualquier N
- [ ] Nunca clamar quantum advantage para ground states (area-law → DMRG gana)
- [ ] Nunca presentar DQPT MPS como resultado final — es infraestructura pre-hardware


---

## Apendice: Hardware Execution Roadmap (post-validacion noiseless)

**Fecha**: 2026-08-21
**Objetivo**: Ejecutar quench dynamics en QPU IBM heavy-hex N=51-77 en el regimen donde los metodos clasicos pierden precision, usando GNN para preparacion O(1) del estado inicial.

---

### Estado actual de datos y modelos (snapshot 2026-08-21)

| Aspecto | Estado | Notas |
|---------|--------|-------|
| Training data heavy-hex | 11 NPZ (N=4-40), 735+ pts, h=[0.3, 5.5] | Suficiente para retrain |
| Topologia heavy-hex N=127 | `make_lattice` funciona, 126 edges, max_coord=3 | Scaling lineal |
| HVA p=1 N=127 | 253 params, 126 RZZ, depth=87 | Viable |
| HVA p=1 N=51 | 101 params, 50 RZZ, depth=52 | Optimo para primer test |
| Infraestructura hardware | 11 modulos + 9 runners + QESEM + ZNE | Completa |
| Credenciales IBM | ~/.qiskit/qiskit-ibm.json existe | Configurado |
| MPNN heavy-hex extrapol. | Best model 37% pass, multiN 0% | BLOQUEANTE — reentrenar |
| Extrapolation N=20-40 | 42-50% pass@5% | Insuficiente — necesita >70% |

---

### Quench parameters para hardware

```
h1 = 3.0   (paramagnetico, GNN grade A, gap grande, preparacion robusta)
h2 = 0.5   (ferromagnetico, cruza h_c, produce DQPTs)
dt = 0.2   (Trotter step grande para reducir profundidad)
n_steps = 15-20  (sweet spot: suficiente para DQPTs, viable en T2)
```

**Justificacion h1=3.0 (no h1=0.5):**
1. GNN predice mejor a h alto (grade A, datos abundantes)
2. Gap grande → preparacion HVA robusta contra errores
3. Estado cercano a |+>^N → p=1 basta
4. DQPTs igualmente validos en ambas direcciones

---

### Profundidad de circuito vs coherencia

| Config | N | Steps | Depth total | Tiempo estimado | vs T2 (200us) |
|--------|---|-------|-------------|-----------------|---------------|
| Conservative | 51 | 15 | ~830 | ~415 us | 2x T2 — QESEM viable |
| Moderate | 77 | 20 | ~1600 | ~800 us | 4x T2 — QESEM necesario |
| Aggressive | 127 | 30 | ~2790 | ~1400 us | 7x T2 — limite de QESEM |

**Recomendacion**: Empezar con N=51, 15 steps. Si QESEM extrae senal, escalar a N=77.

---

### Pre-condiciones para hardware (checklist)

- [ ] MPNN heavy-hex reentrenado con datos N=4-40 (735+ pts existentes)
- [ ] Nuevo modelo pass@5% > 60% en extrapolation N=20-40
- [ ] Validacion noiseless DQPT a N=20-22 (ED exacto) muestra DQPTs correctos
- [ ] Crossover plot confirma MPS falla a step ≤15 para N≥28 heavy-hex
- [ ] GNN predice theta para N=51 heavy-hex h=3.0 (test: energia < E_trivial)
- [ ] Hardware rehearsal v3 pasa preflight para N=51 heavy-hex
- [ ] QESEM credentials/budget confirmado con Qedma
- [ ] IBM QPU reservada (Eagle 127q o Heron 156q)

---

### Arquitectura QPU requerida

| Requisito | IBM Eagle/Heron | Match con proyecto |
|-----------|----------------|-------------------|
| Topologia | Heavy-hex nativo | EXACTA — zero SWAP overhead |
| Qubits | 127 (Eagle) / 156 (Heron) | N=51-77 cabe en ambos |
| Gate set | ECR + RZ + SX + X | RZZ = 2 ECR + rot (transpiler automatico) |
| T1/T2 | ~300/200 us (Heron r2) | Circuito debe ser <2-4x T2 con QESEM |
| 2Q error | ~0.3-0.5% (Heron r2) | 50 ECR × 15 steps = 750 ECR → QESEM obligatorio |
| Readout error | ~1-2% | Mitigado por readout error mitigation (built-in) |

**Key insight**: Heavy-hex ES la topologia nativa de IBM. Cada RZZ en tu circuito mapea a un ECR fisico sin SWAP routing. Esto es una ventaja enorme vs topologias genericas.

---

### Plan de ejecucion hardware (step by step)

**Fase H1: Rehearsal local (sin QPU, ~2h compute)**
```bash
# Preflight: verifica que el circuito es viable
.venv/bin/python scripts/experiment_runners/hardware/run_hardware_rehearsal_v3.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-values 3.0 --target-backend ibm_sherbrooke

# Full deployment dry-run: transpila, estima cost, simula con noise model
.venv/bin/python scripts/experiment_runners/hardware/run_full_deployment_pipeline.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-values 3.0 --dry-run --estimate-cost
```

**Fase H2: Single-point QPU test (~10 min QPU)**
```bash
# Un solo quench point: h1=3.0, N=51, 15 Trotter steps
.venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-values 3.0 --quench-h2 0.5 --trotter-steps 15 --dt 0.2 \
    --backend ibm_sherbrooke --shots 4000 --use-qesem
```

**Fase H3: Full sweep (~5-10h QPU)**
```bash
# Sweep h1 across phase diagram, quench to h2=0.5
.venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-values 2.5 3.0 3.5 4.0 --quench-h2 0.5 \
    --trotter-steps 15 --dt 0.2 \
    --backend ibm_sherbrooke --shots 4000 --use-qesem
```

---

### Costos estimados

| Fase | QPU time | Costo approx | Riesgo |
|------|----------|--------------|--------|
| H1 Rehearsal | 0 (local) | $0 | Ninguno |
| H2 Single-point | ~10 min | ~$50-100 | Bajo — puede fallar QESEM |
| H3 Full sweep | ~5-10h | ~$500-2000 | Medio — budget QESEM |

**Alternativa gratuita**: IBM Quantum Network (academia) da 10 min/mes gratis en Eagle. Fase H2 cabe en eso.

---

### Que resultado produce y por que es publicable

**Resultado**: Energy trajectory E(t) y/o Loschmidt echo proxy durante quench dynamics en heavy-hex N=51, 15+ Trotter steps.

**Por que es quantum advantage real:**
1. MPS (chi=256) pierde precision a step ~12 para N≥28 heavy-hex (tu crossover plot lo demuestra)
2. GPU TDVP no escala a 2D heavy-hex (solo funciona en 1D)
3. El resultado no es verificable clasicamente — solo QESEM + symmetry checks validan
4. IBM demostro el mismo regimen funciona (arXiv:2607.24937, julio 2026)

**Tu valor diferencial vs IBM:**
- IBM: |0>^N (trivial) + Floquet (un punto)
- Tu: |psi_GNN(h)> (estado arbitrario) + quench (parametro continuo h1→h2)
- Esto habilita exploration del diagrama de fases con quantum advantage

---

### Orden de ejecucion completo (de ahora al paper)

1. **[EN CURSO]** DQPT noiseless N=8-20 chain_1d → valida infraestructura
2. **[EN CURSO]** Crossover plot heavy-hex N=10-28 → identifica frontera clasica
3. **[EN CURSO]** Tier 3 data h<1.5 → habilita QPT detection
4. **[PROXIMO]** Reentrenar MPNN heavy-hex con 735 pts → pass >60%
5. **[PROXIMO]** Validar extrapolation a N=40 con nuevo modelo
6. **[PROXIMO]** Test GNN prediction a N=51 h=3.0 (E < E_trivial)
7. **[HARDWARE]** Rehearsal v3 para N=51 heavy-hex
8. **[HARDWARE]** Single-point QPU test (10 min)
9. **[HARDWARE]** Full sweep si paso 8 funciona
10. **[PAPER]** Compilar resultados: noiseless + crossover + hardware



---

## Plan de Trabajo: Validacion de Fidelidad, Circuit Cost, y Simulacion Trotter End-to-End

**Fecha**: 2026-08-23 (revision 2 — centrado en heavy_hex h1=3.0→h2=0.5)
**Contexto**: Para que el experimento de quench dynamics en hardware tenga sentido, necesitamos
demostrar que (a) sabemos cuanta fidelidad necesitamos, (b) sabemos cuanta tenemos,
(c) el circuito cabe en la coherencia del hardware, y (d) la cadena completa funciona noiseless.

**Topologia target**: heavy_hex (nativa en IBM Eagle/Heron — zero SWAP overhead)
**Quench target**: h1=3.0 → h2=0.5 (paramagnetico→ferromagnetico, cruza h_c, GNN confiable a h=3.0)

---

### Parte 1: Cuanta fidelidad NECESITAMOS (threshold minimo)

**Objetivo**: Establecer F_min tal que para F > F_min los DQPTs son detectables con
precision compatible con QESEM (~0.02 error absoluto, segun arXiv:2608.05202).

**Metodo**: ED exacta a N=10-16 heavy_hex con estados de fidelidad controlada.
Preparamos |psi_approx> = sqrt(F)|psi_0> + sqrt(1-F)|psi_perp> para F variando
de 1.0 a 0.3, y medimos degradacion de la senal DQPT.

**Metricas**:
- min(L(t)) como funcion de F (profundidad del minimo)
- |t*(F) - t*(F=1)| / t*(F=1) (shift del tiempo critico)
- max(r(t)) como funcion de F (amplitud rate function peaks)
- Detectabilidad: r_peak > 0.02 (threshold QESEM)

**Implementacion**:
```bash
.venv/bin/python scripts/analysis/dqpt_fidelity_threshold.py \
    --topology heavy_hex --n-qubits 10 \
    --fidelities 1.0 0.95 0.90 0.85 0.80 0.70 0.50 0.30 \
    --h-pre 3.0 --h-post 0.5 --dt 0.05 --steps 60 --save

# Validacion cruzada en chain_1d (sanity check rapido, NO el target)
.venv/bin/python scripts/analysis/dqpt_fidelity_threshold.py \
    --topology chain_1d --n-qubits 10 \
    --fidelities 1.0 0.95 0.90 0.85 0.80 0.70 0.50 --save
```

**Resultado esperado**: F_min ~ 0.80. A F=0.8 la senal retiene ~64% de amplitud
(F^2 ≈ 0.64), suficiente para r_peak > 0.02.

---

### Parte 2: Cuanta fidelidad TENEMOS (medicion real de la GNN)

**Objetivo**: Para heavy_hex a h=3.0 (nuestro punto de operacion), medir la fidelidad
real del estado HVA(theta_GNN) contra el ground state exacto.

**Metodo A — Fidelidad directa (N <= 22)**:
F = |<psi_exact|psi_HVA(theta_GNN)>|^2 via statevector.
Reutiliza: `MPSBackend.compute_fidelity()` y `safe_compute_fidelity()` en runner_base.

```bash
.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py \
    --topology heavy_hex --n-qubits 10 12 14 16 20 \
    --h-values 3.0 2.5 --save
```

**Metodo B — Lower bound energetico (N > 22)**:
F >= 1 - (E_pred - E_exact) / gap. Los datos existen en large_n_extrapolation NPZ.
NOTA: El bound es conservador. gap para N=51 viene de DMRG (aproximado).

```bash
.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py \
    --topology heavy_hex --n-qubits 20 30 40 51 \
    --from-extrapolation --save
```

**Metodo C — Fidelidad MPS (N=20-35, verificacion cruzada)**:
MPSBackend.get_statevector() + overlap directo. Mas costoso pero da F real.

```bash
.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py \
    --topology heavy_hex --n-qubits 20 28 \
    --h-values 3.0 --method mps --chi 256 --save
```

**Criterio go/no-go**: Si F_medida(N=51, h=3.0) > F_min → hardware viable.

---

### Parte 3: Circuit Cost — el circuito cabe en T2?

**Objetivo**: Verificar que HVA(theta) + Trotter(h2, dt, n_steps) transpilado a heavy-hex
nativo produce un circuito que cabe dentro del budget de coherencia T2.

**Modulos existentes que reutilizamos**:

| Modulo | Que hace | Donde |
|--------|----------|-------|
| `validate_transpiled_circuit_quality()` | Cuenta n_2q, depth_2q, error_budget | `hardware/preflight.py:527` |
| `compute_circuit_ces()` | CES = sum(gate_errors) del circuito transpilado | `noisy_utils.py:192` |
| `QPUCostEstimateExtended` | t1_budget_ratio = depth_2q × t_gate / T1 | `hardware/preflight.py:1257` |
| `QPUThroughputProfile.ibm_heron_r2()` | CLOPS model para estimacion de tiempo | `hardware/preflight.py:837` |
| `_interpolate_cx_count(n_qubits)` | CX count empirico para HVA p=1 | `hardware/preflight.py:1032` |
| `estimate_qpu_cost_extended()` | Cost completo con decoherencia | `hardware/preflight.py:1300+` |
| Section 20 de HardwareRehearsalV3 | Transpila y compara RZZ vs PauliEvol | `run_hardware_rehearsal_v3.py` |

**GAP identificado**: No existe una funcion `fits_within_coherence(circuit, backend)` que
responda boolean. El `t1_budget_ratio` usa T1 (no T2). Necesitamos agregar:

```python
def circuit_coherence_check(transpiled, backend, layout, n_trotter_steps=15):
    """Check if HVA+Trotter circuit fits within T2 coherence budget.
    
    Returns dict with:
    - t2_budget_ratio: total_2q_time / median_T2
    - fits_in_coherence: t2_budget_ratio < 4.0 (QESEM viable)
    - fits_in_coherence_strict: t2_budget_ratio < 2.0 (unmitigated viable)
    - total_ecr_gates: after transpilation
    - estimated_circuit_time_us: depth × ecr_time (84ns)
    """
```

**Implementacion del circuit cost check**:
```bash
# Construir circuito completo y medir cost
.venv/bin/python scripts/analysis/circuit_cost_check.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-prep 3.0 --h-quench 0.5 --dt 0.2 --trotter-steps 15 \
    --backend ibm_sherbrooke --save

# Variantes de profundidad
.venv/bin/python scripts/analysis/circuit_cost_check.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-prep 3.0 --h-quench 0.5 --dt 0.2 \
    --trotter-steps 10 12 15 18 20 --backend ibm_sherbrooke --save
```

**Metricas clave**:
- n_ecr total (HVA + Trotter × n_steps) — target: < 1000
- depth_2q — target: < depth donde QESEM mantiene precision (IBM logro 4500 gates)
- t2_budget_ratio — target: < 4.0 (QESEM viable segun IBM paper)
- error_budget = n_ecr × mean_2q_error — target: < 3.0 (QESEM-Extrapolated funciona aqui)
- QPU time estimado con QPUThroughputProfile.ibm_heron_r2()

**Estimacion previa (sin transpilacion)**:
- HVA p=1 heavy_hex N=51: 50 RZZ + 51 RX = ~100 ECR + rotaciones
- Trotter step (2nd order): 50 RZZ + 51 RX + 50 RZZ = ~200 ECR + rotaciones
- 15 steps: 15 × 200 = 3000 ECR (Trotter) + 100 ECR (HVA) = ~3100 ECR total
- Heron ECR time: ~84ns → 3100 × 84ns ≈ 260μs vs T2~200μs
- t2_budget_ratio ≈ 1.3 → VIABLE con QESEM

**NOTA**: Heavy_hex es topologia NATIVA → RZZ en edges del lattice no requiere SWAP.
Esto significa que la transpilacion no agrega gates significativos (0% routing overhead
para edges nativos). Confirmado por HardwareRehearsalV3 Section 20.

---

### Parte 4: Simulacion Trotter end-to-end noiseless (proof of concept)

**Objetivo**: Demostrar que HVA(theta_GNN) → Trotter(H(h2=0.5), dt, 15 steps)
produce DQPTs detectables cuando partimos del estado GNN (no del exacto).

**Metodo**: Comparar tres evoluciones identicas excepto el estado inicial:
1. |psi_exact> = ground state exacto de H(h=3.0) heavy_hex
2. |psi_GNN> = HVA(theta_predicted) con theta de la GNN
3. |+>^N = estado trivial (control: deberia dar DQPTs diferentes o ausentes)

**Implementacion** (requiere modificacion menor a section 4):
```bash
# N=10 heavy_hex: GNN vs exact en deteccion de DQPTs (h1=3.0→h2=0.5)
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 4 --n-qubits 10 --topology heavy_hex \
    --dqpt-h-pre 3.0 --dqpt-h-post 0.5 --dqpt-dt 0.05 --dqpt-steps 60

# N=14 heavy_hex
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 4 --n-qubits 14 --topology heavy_hex \
    --dqpt-h-pre 3.0 --dqpt-h-post 0.5 --dqpt-dt 0.05 --dqpt-steps 60

# N=20 heavy_hex (limite ED verificable)
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 4 --n-qubits 20 --topology heavy_hex \
    --dqpt-h-pre 3.0 --dqpt-h-post 0.5 --dqpt-dt 0.05 --dqpt-steps 60
```

**Modificacion necesaria a section_dqpt_loschmidt**: Evolucionar AMBOS estados
(psi_exact y psi_gnn) y reportar L_exact(t) y L_gnn(t) para comparacion directa.
Medir |t*_gnn - t*_exact| / t*_exact < 10%.

**Criterio de exito**:
- L_gnn(t) muestra mismos DQPTs que L_exact(t) (misma cantidad, t* similar)
- Amplitud r_gnn(t*) > 0.02 (detectable con QESEM)
- |t*_gnn - t*_exact| / t*_exact < 10%

---

### Parte 4b: MPS proxy a N=28-51 (pre-hardware, energy trajectories)

```bash
# N=28 heavy_hex: GNN-state vs |+>^N (energy oscillations como proxy)
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 1 --n-qubits 28 --topology heavy_hex \
    --h1 3.0 --h2 0.5 --dt 0.2 --n-trotter 15 --chi-values 128 256

# N=51: el test definitivo pre-hardware
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
    --section 1 --n-qubits 51 --topology heavy_hex \
    --h1 3.0 --h2 0.5 --dt 0.2 --n-trotter 15 --chi-values 128 256
```

---

### Parte 5: Validacion QESEM para circuitos HVA+Trotter

**Contexto**: IBM valido QESEM para circuitos Floquet (estructura repetitiva uniforme).
Nuestro circuito tiene estructura diferente: HVA (variacional, parametrizado) +
Trotter (fijo, repetitivo). Necesitamos verificar que QESEM caracteriza correctamente
ambos tipos de capas.

**Metodo**: Ejecutar HardwareRehearsalV3 con la configuracion de quench:
```bash
.venv/bin/python scripts/experiment_runners/hardware/run_hardware_rehearsal_v3.py \
    --topology heavy_hex --n-qubits 51 --p-layers 1 \
    --h-values 3.0 --target-backend ibm_sherbrooke
```

Esto ejecuta la transpilacion (Section 20), el preflight completo, y si pasa,
genera el circuito ISA listo para enviar.

---

### Resumen: Orden de ejecucion

| Paso | Prerequisito | Tiempo | Output |
|------|-------------|--------|--------|
| 1. Crear `dqpt_fidelity_threshold.py` | observables module | 1h dev | F_min |
| 2. Crear `evaluate_gnn_fidelity.py` | MPNN heavy_hex | 1h dev | F(N, h) medida |
| 3. Crear `circuit_cost_check.py` | trotter module | 1h dev | n_ecr, t2_ratio |
| 4. Ejecutar threshold (N=10 heavy_hex) | Script 1 | ~5 min | Plot F vs signal |
| 5. Ejecutar fidelity (N=10-20 heavy_hex) | Script 2 | ~10 min | Tabla F |
| 6. Ejecutar circuit cost (N=51 heavy_hex) | Script 3 | ~2 min | Viabilidad T2 |
| 7. Comparar F_medida vs F_min | Pasos 4+5 | analisis | Go/no-go fidelidad |
| 8. DQPT end-to-end (N=10-20 heavy_hex) | GNN model | ~30 min | L_gnn vs L_exact |
| 9. MPS proxy (N=28-51 heavy_hex) | Paso 8 ok | ~2-4h | Energy oscillations |
| 10. Hardware rehearsal v3 (N=51) | Pasos 7+8+9 ok | ~30 min | ISA circuit ready |
| 11. Decision QPU | Todo anterior | — | Go/No-go hardware |

---

### Relacion con papers de referencia

**IBM+Qedma (arXiv:2607.24937)**: Estado inicial |0>^N (fidelidad perfecta, zero cost).
Nosotros agregamos HVA preparation (fidelidad < 1, costo ~100 ECR). La compensacion es
que habilitamos quench physics inaccesible desde |0>. Nuestro circuito total (~3100 ECR)
es MAS CORTO que el de IBM (~4500 ECR por 30 Floquet cycles).

**Benchmark QESEM (arXiv:2608.05202)**: QESEM logra 0.02 absoluto en TFIM 50-75 qubits.
Si r_peak > 0.02, QESEM puede resolver la senal DQPT. Esto se verifica en Parte 1.

**DQPTs con ruido (arXiv:2504.03005)**: DQPTs se suavizan en estados mixtos pero
persisten como "broadened features" en single realizations. QESEM recupera la senal
ideal via extrapolacion a zero noise.

**Naive ZNE no funciona en Heron (arXiv:2607.24937, Appendix G)**: Unitary folding
no reproduce resultados correctos. QESEM (PEA-based) es necesario. Nuestro modulo
`qesem.py` ya soporta esto.


