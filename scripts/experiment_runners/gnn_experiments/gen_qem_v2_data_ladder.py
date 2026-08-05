#!/usr/bin/env python
"""Generate GNN-QEM V2 training data for LADDER topology (parallel worker).

Run alongside the main training pipeline to pre-generate data for ladder.
Output: results/gnn_qem_v2/training_data_v2_ladder.json

Usage:
    python scripts/experiment_runners/gnn_experiments/gen_qem_v2_data_ladder.py
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s [ladder]: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

from pathlib import Path

from qmbp_simulation.predictors.gnn_qem import (
    generate_qem_training_data_v2,
    save_qem_samples_v2,
)

OUTPUT_PATH = Path("results/gnn_qem_v2/training_data_v2_ladder.json")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("Generating V2 data: ladder, N=6, 6 h-points, theta_source=zoo")
    samples = generate_qem_training_data_v2(
        topologies=["ladder"],
        n_qubits_list=[6],
        h_values=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        shots=4096,
        theta_source="zoo",
    )

    save_qem_samples_v2(samples, OUTPUT_PATH)
    elapsed = time.time() - t0
    logger.info(f"Done: {len(samples)} samples in {elapsed:.1f}s → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
