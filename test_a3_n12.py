"""Quick test: run A3 for N=12 only to verify optimizations work."""

import sys
import time

sys.path.insert(0, ".")

from experiments.scaling.exp_a3_scaling_law import ExperimentA3
from qmbp_simulation.framework.config import AnalysisConfig, ExperimentConfig, SystemConfig

# Override config to test only N=12
config = ExperimentConfig(
    experiment_id="A3_test",
    category="A",
    description="Test A3 at N=12 with optimizations",
    hypothesis="Testing speedup",
    system=SystemConfig(n_qubits=6, p_layers=2),
    analysis=AnalysisConfig(scaling_n_values=[12]),
    seeds=[42],
    verbose=True,
)

exp = ExperimentA3(config)
exp.setup()

t0 = time.time()
metrics = exp.run_single(seed=42)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print("N=12 result:")
print(f"  h_min = {metrics[0].technique_metadata['h_min']}")
print(f"  boundary_found = {metrics[0].technique_metadata['boundary_found']}")
print(f"  wall_time = {elapsed:.1f}s")
print(f"{'=' * 60}")
