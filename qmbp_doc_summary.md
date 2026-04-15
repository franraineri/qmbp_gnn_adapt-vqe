

### 1. The Quantum Many-Body Problem & The Limits of Classical Physics

In 1972, Nobel laureate P.W. Anderson published his famous essay *"More is Different"*, establishing that when billions of quantum particles (like electrons or spins) interact, they do not just act as a sum of their parts. Instead, entirely new physical laws emerge. This is the essence of the Quantum Many-Body Problem. The mathematical complexity arises because the particles do not exist in isolation; their wavefunctions overlap and interact, meaning the Schrödinger equation that governs them cannot be neatly separated into individual, solvable pieces. The system must be treated as a single, massive, inextricably linked mathematical entity. 

For decades, physicists classified the phases of matter using Landau’s Symmetry Breaking Theory. In this framework, water freezes into ice or iron becomes magnetic because the particles "break symmetry" and fall into a neat, locally ordered pattern as they cool down. Landau introduced the concept of the "local order parameter"—a measurable physical quantity, like a magnetization vector, that is zero in the disordered phase and non-zero in the ordered phase. If you look at one atom in a magnet, its orientation tells you the state of the whole material. Phase transitions were universally understood as a thermodynamic competition between energy (which favors order) and entropy (which favors chaos).

However, in the 1980s, physicists discovered materials that completely defied Landau's rules. Starting with the discovery of the Quantum Hall Effect, scientists observed states of matter that exhibited phase transitions without any symmetry breaking and without any local order parameter. At absolute zero (-273.15°C), where all thermal motion and classical entropy stop, these materials refused to freeze or order themselves. These became known as **Topological Quantum Phases**. In these phases, the "identity" of the material is not defined by the local arrangement of its atoms, but by global, topological invariants—properties of the quantum wavefunction that remain constant even if the system is smoothly deformed, much like how a donut and a coffee cup share the same topological property of having exactly one hole.

### 2. Geometric Frustration and Quantum Spin Liquids (QSLs)

The most sought-after topological phase is the Quantum Spin Liquid (QSL), a theoretical state of matter first proposed by Anderson in 1973. QSLs represent a radical departure from conventional solid-state physics because they remain fundamentally disordered, yet highly correlated, down to absolute zero.

QSLs are born from Geometric Frustration. Imagine a simple game with three magnetic atoms arranged in a triangle. The rules of their quantum interaction (antiferromagnetism) dictate that neighboring atoms must point in opposite directions to minimize their energy. 
* Atom 1 points UP.
* Atom 2 points DOWN.
* Atom 3 is now "frustrated." It is connected to both an UP and a DOWN atom. It cannot satisfy the rule for both neighbors simultaneously.

Classically, this frustration leads to a massive ground-state degeneracy—millions of equally valid, equally "frustrated" configurations with the exact same energy. But quantum mechanics does not allow the system to just pick one configuration and stop. 

When you scale this triangle into a massive 2D web—like a Kagome or triangular lattice—the entire system becomes intensely frustrated. Because no single "frozen" arrangement satisfies the energy requirements, strong quantum fluctuations take over. The spins remain in a constant, liquid-like state of motion even at absolute zero, constantly exploring the vast space of degenerate states.

Instead of freezing, the spins entangle with each other across macroscopic distances, forming what is known as a **Resonating Valence Bond (RVB)** state. A valence bond is a quantum singlet—a pair of spins inextricably entangled in a state where if one is UP, the other is strictly DOWN: 

$$\frac{|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle}{\sqrt{2}}$$

In a QSL, these valence bonds do not lock into a static pattern. The state "resonates," meaning the true ground state of the material is a massive quantum superposition of *all possible* singlet pairings across the entire lattice. 

Because of this intense, dynamic entanglement, there is zero local order. You cannot look at one spin, or even a small cluster of spins, to know the state of the system. Instead, the phase is characterized by **Topological Entanglement Entropy**—a measure of how deeply information is shared globally across the boundaries of the material. 

Furthermore, QSLs feature "fractionalized excitations." In a normal material, flipping a spin creates a localized magnetic wave (a magnon) carrying a spin of 1. In a QSL, because the lattice is a web of singlets, flipping a spin tears a singlet apart, creating two distinct, independent quasi-particles called **spinons**. Each spinon carries a fractional spin of 1/2 but has no electrical charge. This phenomenon, where the fundamental properties of an electron (spin and charge) literally split and move independently, completely defies classical intuition. These fractionalized excitations act as anyons, which hold immense potential for building fault-tolerant topological quantum memories, as local noise cannot easily destroy information that is stored non-locally across multiple fractional particles.


### 3. The Classical Bottleneck: The Curse of Dimensionality
To understand why simulating a QSL is so difficult, we must look at the mathematics of the **Hilbert Space**. 

In classical mechanics, describing the state of 50 coins is easy: you just write down 50 variables (Heads or Tails). 
In quantum mechanics, 50 spins exist in a superposition of all possible states simultaneously. To perfectly simulate 50 spins, a computer must track $2^{50}$ complex probability amplitudes. 

* 10 spins = 1,024 amplitudes.
* 30 spins = ~1 Billion amplitudes (16 Gigabytes of RAM).
* 50 spins = ~1.1 Quadrillion amplitudes (~18 Petabytes of RAM).

At around 50 interacting quantum particles, the memory capacity of the largest supercomputer on Earth (like the Frontier exascale supercomputer) is completely exhausted. This exponential scaling is the **Curse of Dimensionality**.

### 4. The Fatal Blow: The "Sign Problem"
To cheat the Curse of Dimensionality, classical physicists use statistical sampling methods, the most famous being **Quantum Monte Carlo (QMC)**. Instead of tracking every possible state, QMC samples a random subset of states and calculates the average, much like predicting an election by polling a thousand people.

However, QMC suffers a catastrophic failure known as the **Sign Problem** when applied to frustrated systems (like QSLs) or fermionic systems (like interacting electrons). 



Because quantum mechanics relies on wave functions that can be negative or complex (unlike classical probabilities, which are always positive), the quantum paths in frustrated systems often have opposite mathematical signs. When the algorithm tries to add up the probabilities to find the ground state, the positive and negative paths cancel each other out (destructive interference). 

To get a statistically significant answer out of this "noise" of cancellations, the classical computer requires exponentially more samples. In 2005, physicists proved that the Sign Problem is mathematically **NP-hard**, meaning no classical algorithm will ever be able to solve it efficiently for general cases. The classical path is a dead end.

### 5. The Quantum Savior and NISQ Limitations
In 1982, Richard Feynman proposed the ultimate solution: *"Nature isn't classical... and if you want to make a simulation of nature, you'd better make it quantum mechanical."* A quantum computer natively possesses an exponentially large Hilbert space. A 50-qubit processor inherently tracks $2^{50}$ amplitudes simply by existing. There is no Curse of Dimensionality, and there is no Sign Problem.

However, we are currently in the **Noisy Intermediate-Scale Quantum (NISQ)** era. The environment constantly interacts with the fragile qubits, causing them to lose their quantum state (Decoherence) and introducing gate errors. 
As proven in recent literature (e.g., *Mele et al.*), this noise doesn't just add slight inaccuracies; it fundamentally **truncates** the depth of the circuit. If a quantum algorithm requires 500 successive gate operations (a deep circuit) to simulate a QSL, the noise will effectively erase the information from the first 450 gates. The entanglement never reaches the macroscopic scale required to simulate a topological phase, forcing physicists to invent ultra-shallow, hybrid algorithms (like the GNN-HVA architecture in your thesis) to extract utility before the noise destroys the simulation.

---

## 3. Our Proposed Solution: The Hybrid GNN-HVA Architecture
To bypass these hardware roadblocks, your thesis proposes a paradigm-shifting hybrid architecture. Instead of forcing the quantum computer to do all the learning, we shift the heavy lifting to a classical Artificial Intelligence model. 

1. **The Hamiltonian Variational Ansatz (HVA):** We abandon generic, deep quantum circuits. Instead, we design a strictly shallow circuit ($p=1$ or $p=2$ layers) whose gates are directly derived from the physical equations of the target material. 
2. **The Graph Neural Network (GNN):** We train a classical GNN on data generated by advanced Tensor Networks. The GNN learns the complex topography of the quantum system.
3. **The "Intelligent Warm-Start":** When faced with a new, unseen material of 40-50 qubits, the GNN instantly predicts the optimal angles ($\theta_{opt}$) for the quantum gates. We inject these predicted angles directly into the shallow HVA circuit on the quantum hardware.

## 4. Why This Research is Groundbreaking (The "Why")
This methodology is not just an incremental improvement; it is a fundamental restructuring of how we approach quantum simulation:

* **It Respects Hardware Physics:** By enforcing a strict depth limit on our quantum circuits, we prevent the noise-induced information truncation that plagues other algorithms.
* **It Solves the Initialization Problem:** The GNN "Warm-Start" drops the quantum algorithm exactly at the bottom of the energy valley, completely bypassing the barren plateau problem that kills random-start VQEs.
* **Path to Quantum Utility:** By combining classical machine learning for parameter prediction with quantum hardware for the final state projection, this framework aims to successfully characterize complex quantum transitions (using local observables) on 50-qubit devices, demonstrating practical quantum advantage where purely classical or purely quantum methods fail.

## 5. The Operational Roadmap
To achieve this, the project is structured in four rigorous phases using state-of-the-art tooling (Qiskit 2.x and PyTorch):

* **Phase 1: Ground Truth Generation:** Solving systems classically using Exact Diagonalization ($N < 15$ for the PoC, currently $N=6$) or Tensor Networks (DMRG/NetKet for scaling to 20-40+ qubits) to find the true lowest energy states and extract local observable data.
* **Phase 2: Compilation & Optimization:** Using classical optimizers to find the exact gate angles that allow our shallow HVA circuit to mimic the exact physical states.
* **Phase 3: AI Training:** Training the PyTorch model (MLP for the PoC, GNN for scaling) to map the physical parameters directly to the optimized quantum gate angles, with physics-informed energy validation.
* **Phase 4: Hardware Deployment:** Using the trained model to predict parameters for an unseen system, deploying a highly restricted Adaptive VQE on IBM hardware (where convergence at iteration 0 is the ideal outcome), and validating the physics using local observable measurements and the finite-size critical point crossover.
---

# Implementation Stack & Techniques Context

## 1. Core Technology Stack
We operate strictly on modern, enterprise-grade, and open-source frameworks to ensure reproducibility and compatibility with current hardware.

* **Quantum Framework:** **Qiskit 2.x** (and modern 1.x). We strictly utilize the V2 ecosystem. Deprecated modules (`qiskit.opflow`, `PauliSumOp`, `qiskit.algorithms`) are banned from the codebase.
* **Machine Learning:** **PyTorch** (`torch.nn`). Used for building the Multi-Layer Perceptron (for the PoC) and the Graph Neural Network (GNN) for the final scaling.
* **Classical Numerical Solvers:** **NumPy** and **SciPy**. Used for exact diagonalization, matrix operations, and classical optimization (`L-BFGS-B`).
* **Advanced Tensor Networks (For scaling Phase 1):** **TeNPy** (for DMRG on quasi-1D structures) and **NetKet** (for Neural Quantum States on 2D lattices).

## 2. Advanced Implementation Techniques (By Phase)

### Phase 1: Ground Truth Generation (Classical Physics)
* **Dense Exact Diagonalization (For PoC):** For small systems ($N < 15$), we use `np.linalg.eigh(H.to_matrix())` rather than sparse solvers like `eigsh`. This guarantees numerical stability, avoids issues with degenerate ground states, and gives us the **Energy Gap ($\Delta = E_1 - E_0$)** for free, which is crucial for identifying phase transitions. The current PoC uses $N=6$; for scaling beyond $N \approx 14$, switch to DMRG/TeNPy.
* **Bulk Local Observables:** We do not rely on single-site measurements (e.g., measuring just qubit 0), which suffer from severe open-boundary artifacts. We calculate the bulk order parameters by averaging the expectation values across all sites in the lattice (e.g., $\frac{1}{N} \sum \langle X_i \rangle$).

### Phase 2: Quantum Compilation & HVA Design
* **Strict Hamiltonian Variational Ansatz (HVA):** We completely avoid Hardware-Efficient Ansätze. The HVA is built using exact time-evolution generators of our target Hamiltonian.
* **The $2\theta$ Physical Scaling:** Standard Qiskit rotation gates implement half-angle rotations. To correctly implement the physical time-evolution operator $e^{-i \theta H}$, we explicitly scale our parameters by a factor of 2 in the circuit definition (e.g., `qc.rzz(2 * theta, i, i+1)`). This ensures the neural network learns true physical time parameters.
* **Energy-Driven Optimization (The Mele et al. Compliance):** We *do not* use global state fidelity as the primary cost function, as this contradicts the findings that global costs cause barren plateaus under noise. Instead, we use `qiskit.primitives.StatevectorEstimator` to minimize the **Physical Energy** directly. Fidelity is calculated as a background validation metric in Phase 2 noiseless simulations (where it is safe and useful), but is never used as a cost function and is strictly forbidden on hardware paths.
* **Sequential Warm-Starting:** We do not initialize the classical optimizer with random noise. 
    1. The HVA circuit begins with a Hadamard layer preparing $|+\rangle^{\otimes N}$, the paramagnetic ground state (the $h \to \infty$ limit of the TFIM). At $\theta = 0$, the HVA therefore already produces the exact ground state for large $h$.
    2. We seed the optimizer with a small random perturbation ($\theta \sim \mathcal{U}(-0.01, 0.01)$) to escape the **symmetry saddle point** at $\theta = 0$. At exact zeros, the gradient vanishes by symmetry ($\sim 10^{-6}$) and L-BFGS-B declares convergence at iteration 0 without moving. This was a critical bug discovered during PoC development.
    3. We sweep **descending** from $h=2.0$ to $h=0.0$. At $h=2$, $|+\rangle^{\otimes N}$ is already near-exact, so $\theta \approx 0$ is near-optimal. Warm-start then carries the solution smoothly toward $h=0$.
    4. For every subsequent physical parameter ($h_{i+1}$), we use the optimized angles from the previous step ($\theta_{opt}$ at $h_i$) as the initial seed. This exploits the continuous nature of the physical wave function and accelerates convergence.
* **HVA Expressibility Limit (Key Thesis Finding):** The HVA with $|+\rangle^{\otimes N}$ initial state and $p=2$ layers has a fundamental expressibility ceiling: it **cannot reach** the deep ferromagnetic ground state ($h \to 0$, which is $|000...0\rangle$). Fidelity degrades below $h \approx 1.0$ (e.g., 22% at $h=0$ for $N=6$). This was verified with 50 random restarts over $[-\pi, \pi]$ — it is not an optimization failure but a structural limitation. The circuit lacks depth to concentrate all amplitude from an equal superposition onto a single basis state. The PoC pipeline is therefore validated for the **paramagnetic regime** ($h \geq 1.0$, fidelities $> 96\%$). The ferromagnetic side would require either more layers (violating the Mele et al. depth constraint) or a different initial state strategy. This expressibility-depth tradeoff is itself a significant thesis finding that directly illustrates the practical consequences of the noise-truncation theorem.

### Phase 3: Predictive ML (The GNN/MLP)
* **PoC: MLP Predictor.** For the 1D TFIM with uniform $J$, the graph structure is fixed and only $h$ varies. A simple Multi-Layer Perceptron ($h \to \theta_{pred}$) suffices as the PoC predictor. The full Graph Neural Network is reserved for scaling to non-uniform couplings or 2D lattices.
* **Curriculum Learning via Physical Tracking:** While the PyTorch model uses standard Mean Squared Error (MSE) to learn the $\theta_{opt}$ targets, we implement a custom validation callback. The network periodically feeds its predicted angles back into a Qiskit `StatevectorEstimator` to calculate the resulting quantum energy. We track this to ensure the network is learning the actual energy landscape, not just interpolating meaningless numbers.
* **Learning Rate Scheduling:** We use `ReduceLROnPlateau` (or `CosineAnnealingLR`) to avoid oscillation around the minimum on small datasets (21 training points in the PoC).
* **Interpolation Validation:** We always validate on at least one $h$ value not in the training set to verify the model generalizes between grid points, rather than only checking on training data.
* **Fidelity Filter (Critical):** Only train on Phase 2 data points where fidelity ≥ 96%. Points below this threshold have $\theta_{opt}$ that don't represent the true ground state — training on them poisons the model. The diagnostic signature of this failure mode is MSE converging to near-zero while the physics validation $\Delta E$ remains constant (the MLP faithfully learns garbage). In the PoC, this filter excludes the ferromagnetic regime ($h < 1.0$) where the HVA expressibility limit prevents convergence.
* **Test Point Selection:** The Phase 4 test point must be within the high-fidelity regime. Testing at $h = 1.5$ (where Phase 2 achieves fid ≈ 99.6%) gives meaningful results; testing at $h = 1.05$ (fid ≈ 97.6%) tests the pipeline at the edge of the expressibility limit, conflating ansatz limitations with pipeline quality.
* **Data Persistence:** All generated datasets (Hamiltonian parameters, energy gaps, observables, and optimized angles) are saved to disk as `.npz` files to decouple the quantum compilation from the PyTorch training loop.
* **Data Quality Gate:** Before training, verify that Phase 2 produced valid data: minimum fidelity should exceed 99% for the paramagnetic regime ($h \geq 1.0$). If the physics validation callback shows $\Delta E \approx$ constant throughout training (not decreasing), it is a sign that the training data itself is corrupted (e.g., all $\theta_{opt} \approx 0$ due to the saddle point bug). MSE converging to near-zero while $\Delta E$ remains large is the diagnostic signature of this failure mode.

### Phase 4: Hardware Deployment
* **Primitives V2 Native Execution:** We use `qiskit_ibm_runtime.EstimatorV2` for all final hardware executions, ensuring seamless integration with IBM's latest error suppression technologies.
* **Restricted Adapt-VQE:** When using adaptive refinement (`qiskit_algorithms.AdaptVQE`) to correct slight errors in the GNN's prediction, we strictly limit it via `max_iterations=2`. This enforces the $\mathcal{O}(\log n)$ depth limit dictated by the *Mele et al.* paper, ensuring we do not grow the circuit into the noise-truncation regime where quantum advantage is lost.
* **Convergence at Initialization:** When the warm-start is near-optimal, AdaptVQE evaluates all operator gradients and finds them below threshold on the first iteration, raising an `AlgorithmError`. This is the **ideal outcome** — it means 0 extra layers were needed and the MLP/GNN prediction was already sufficient. The code must catch this exception and treat it as success.
* **Phase Classification via Observable Crossover:** We classify the quantum phase by measuring local observables ($\langle X_i \rangle$, $\langle Z_i Z_{i+1} \rangle$) and comparing against the crossover point from Phase 1 exact data. For finite-size systems, the critical point shifts from the thermodynamic limit $h_c = 1.0$, so we use the data-driven $\langle X \rangle = \langle ZZ \rangle$ crossing rather than hardcoded thresholds.


---

### Key Sources & Bibliography

Here is a comprehensive and rigorously structured bibliography and source guide. This covers the foundational physics, the algorithmic theory, and the technical documentation that back every architectural decision we have made for your Master's Thesis. 

1. **Foundations of Many-Body Physics & QSLs:**
   * Anderson, P. W. (1972). *"More is different."* **Science**, 177(4047), 393-396. *(The philosophical and mathematical foundation of emergent many-body phenomena).*
   * Anderson, P. W. (1973). *"Resonating valence bonds: A new kind of insulator?"* **Materials Research Bulletin**, 8(2), 153-160. *(The original proposal of the Quantum Spin Liquid).*
   * Savary, L., & Balents, L. (2016). *"Quantum spin liquids: a review."* **Reports on Progress in Physics**, 80(1), 016502. *(A comprehensive modern review on geometric frustration and fractionalization).*

2. **The Sign Problem & Computational Complexity:**
   * Troyer, M., & Wiese, U. J. (2005). *"Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations."* **Physical Review Letters**, 94(17), 170201. *(The mathematical proof that the Sign Problem is NP-hard, cementing the necessity of quantum computers for frustrated systems).*

3. **Quantum Computing and NISQ Limitations:**
   * Feynman, R. P. (1982). *"Simulating physics with computers."* **International Journal of Theoretical Physics**, 21(6/7). *(The seminal paper proposing quantum computers to simulate quantum mechanics).*
   * Preskill, J. (2018). *"Quantum Computing in the NISQ era and beyond."* **Quantum**, 2, 79. *(Defines the current state of noisy quantum hardware and the necessity of hybrid algorithms).*
   * Mele, A. A., et al. (2026). *"Noise-induced shallow circuits and the absence of barren plateaus."* **Nature Physics**. *(The recent proof demonstrating depth-truncation due to non-unital hardware noise, establishing the limits of NISQ simulation).*


### 1. The Core Paradigm: Noise, Barren Plateaus, and Shallow Circuits
These papers justify *why* we cannot use deep circuits and *why* we must use local observables and Warm-Starts.

* **The Foundation of our Noise-Resilient Architecture:**
    * *Mele, A. A., Angrisani, A., Ghosh, S., Khatri, S., Eisert, J., Stilck França, D., & Quek, Y. (2026).* **"Noise-induced shallow circuits and the absence of barren plateaus."** *Nature Physics.* * **Relevance:** This is the critical paper uploaded during our session. It proves mathematically that non-unital hardware noise truncates quantum circuits to $\mathcal{O}(\log n)$ depth, making deep circuits classically simulable. It also proves that local cost functions under this noise do *not* suffer from barren plateaus, which is the exact theoretical justification for our Phase 2 and Phase 4 design.
* **Local vs. Global Cost Functions:**
    * *Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021).* **"Cost function dependent barren plateaus in shallow parametrized quantum circuits."** *Nature Communications, 12(1), 1791.*
    * **Relevance:** This paper established the rule that global state fidelity causes barren plateaus even in shallow circuits, directly prompting our architectural shift to measure local observables (like $\langle Z_i Z_{i+1} \rangle$) and minimize physical energy rather than state overlap.

### 2. Quantum Algorithms: HVA & ADAPT-VQE
These sources provide the mathematical blueprint for the quantum circuits we are building.

* **Hamiltonian Variational Ansatz (HVA):**
    * *Wiersema, R., Zhou, C., de Sereville, Y., Carrasquilla, J., & Kim, Y. B. (2020).* **"Exploring entanglement and optimization within the Hamiltonian Variational Ansatz."** *PRX Quantum, 1(2), 020319.*
    * **Relevance:** Details how to construct ansätze based directly on the non-commuting terms of the target Hamiltonian (Phase 2). It proves HVA is vastly superior to Hardware-Efficient Ansätze (HEA) for finding many-body ground states without requiring massive depth.
* **Adaptive Refinement:**
    * *Grimsley, H. R., Economou, S. E., Barnes, E., & Mayhall, N. J. (2019).* **"An adaptive variational algorithm for exact molecular simulations on a quantum computer."** *Nature Communications, 10(1), 3007.*
    * **Relevance:** The original Qubit-ADAPT-VQE paper. It justifies our Phase 4.3 step, where we allow the algorithm to dynamically add 1 or 2 operators to fix residual errors from the GNN's prediction.

### 3. Condensed Matter Physics & Data Generation (Ground Truth)
These are the sources for the physics of the materials we are simulating and the classical algorithms used in Phase 1 to generate the training data.

* **Tensor Networks & DMRG (For 1D and Quasi-1D):**
    * *Schollwöck, U. (2011).* **"The density-matrix renormalization group in the age of matrix product states."** *Annals of Physics, 326(1), 96-192.*
    * **Relevance:** The definitive bible on DMRG and Tensor Networks. It explains how to classically solve 1D and quasi-1D spin ladders to extract the exact ground truth we need for Phase 1.
* **Neural Quantum States (NQS for 2D Frustration):**
    * *Carleo, G., & Troyer, M. (2017).* **"Solving the quantum many-body problem with artificial neural networks."** *Science, 355(6325), 602-606.*
    * **Relevance:** The seminal paper demonstrating how classical neural networks (like RBMs) can approximate 2D frustrated wavefunctions where standard Monte Carlo fails due to the Sign Problem.
* **Topological Phases & Quantum Spin Liquids:**
    * *Savary, L., & Balents, L. (2016).* **"Quantum spin liquids: a review."** *Reports on Progress in Physics, 80(1), 016502.*
    * **Relevance:** The theoretical background on why QSLs lack local order and require highly entangled global states, defining the ultimate physical target of your thesis.

### 4. Technical Frameworks & Software Documentation
The actual codebase relies on the bleeding-edge versions of these specific libraries.

* **Qiskit 2.x & Primitives V2:**
    * *IBM Quantum Documentation (Current).* **"Migrating to Qiskit 1.0/2.0"** & **"Qiskit Primitives V2 Interface."**
    * **Relevance:** Dictates the use of `SparsePauliOp` for Hamiltonians and `StatevectorEstimator` / `EstimatorV2` for execution. (Available at: `docs.quantum.ibm.com`).
* **Qiskit Addon: AQC-Tensor:**
    * *IBM Quantum Addons GitHub Repository.* **`qiskit-addon-aqc-tensor`**
    * **Relevance:** The specific classical-to-quantum compiler tool we use in Phase 2 to translate Tensor Network states into quantum circuits.
* **Classical Simulation Libraries:**
    * **TeNPy (Tensor Network Python):** *Hauschild, J., & Pollmann, F. (2018).* (Available at: `github.com/tenpy/tenpy`). Used for DMRG generation.
    * **NetKet:** *Vicentini, F., et al. (2022).* (Available at: `netket.org`). Used for Neural Quantum States generation in 2D.
* **Machine Learning:**
    * **PyTorch Documentation:** Used for constructing the Multi-Layer Perceptron (MLP) and Graph Neural Networks (GNN) in Phase 3. 

