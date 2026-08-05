"""Quick test: DMRG vs Exact Diag comparison."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.models.model_registry import get_model_spec

spec = get_model_spec("tfim")
solver = ClassicalSolver()

for topology in ["chain_1d", "heavy_hex"]:
    for h in [1.0, 2.0, 4.0]:
        lattice = make_lattice(topology, 10, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        gt_exact = solver.solve(H, lattice, method="exact")
        gt_dmrg = solver.solve(H, lattice, method="dmrg")
        delta = abs(gt_exact.ground_energy - gt_dmrg.ground_energy)
        de_gap = delta / max(gt_exact.gap, 1e-10)
        print(f"{topology:12s} h={h:.1f}: E_exact={gt_exact.ground_energy:.8f} "
              f"E_dmrg={gt_dmrg.ground_energy:.8f} |ΔE|={delta:.2e} "
              f"ΔE/gap={de_gap:.6f}")
