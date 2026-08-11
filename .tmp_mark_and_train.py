"""Mark bad NPZ as not_useful in dashboard, then retrain ladder multi-N."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Mark N=20, N=24, N=26 as not_useful in dashboard
# ═══════════════════════════════════════════════════════════════════════════

dashboard_path = Path("data/model_quality_dashboard.json")
dashboard = json.loads(dashboard_path.read_text())

# Files to mark as not_useful
mark_not_useful = [
    "ladder_N20_p1.npz",
    "ladder_N24_p1.npz",
    "ladder_N26_p1.npz",
    "triangular_N9_p1.npz",
    "triangular_N14_p1.npz",
    "triangular_N16_p1.npz",
]

existing_files = {c["file"] for c in dashboard["configs"]}
n_updated = 0
n_added = 0

for fname in mark_not_useful:
    if fname in existing_files:
        for c in dashboard["configs"]:
            if c["file"] == fname:
                if c.get("training_utility") != "not_useful":
                    c["training_utility"] = "not_useful"
                    c["training_utility_reason"] = (
                        "Irrecoverable with HVA p=1: 90%+ variational violations, "
                        "ansatz cannot express ground state at this N/h regime."
                    )
                    n_updated += 1
                break
    else:
        dashboard["configs"].append(
            {
                "file": fname,
                "training_utility": "not_useful",
                "training_utility_reason": (
                    "Irrecoverable with HVA p=1: 90%+ variational violations, "
                    "ansatz cannot express ground state at this N/h regime."
                ),
            }
        )
        n_added += 1

dashboard["generated_at"] = datetime.now(UTC).isoformat()
dashboard["n_configs"] = len(dashboard["configs"])
dashboard_path.write_text(json.dumps(dashboard, indent=2))
print(f"Step 1: Dashboard updated ({n_updated} updated, {n_added} added)")

# Also mark ladder_N14 and N=16 (already in dashboard as not_useful, confirm)
ladder_not_useful = [
    c
    for c in dashboard["configs"]
    if "ladder" in c.get("file", "") and c.get("training_utility") == "not_useful"
]
print(f"  Ladder not_useful files: {[c['file'] for c in ladder_not_useful]}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Re-train ladder multi-N with ONLY good data (N=4,6,8,10,12)
#          The aggregator will skip not_useful files automatically
# ═══════════════════════════════════════════════════════════════════════════

print("\nStep 2: Re-training ladder multi-N model...")
print("  (MultiNAggregator will skip not_useful files from dashboard)")

from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint
from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

agg = MultiNAggregator(topology="ladder", model="tfim_bond_resolved")
summary = agg.scan()
print(f"  Scanned N values: {agg.available_n_values()}")
print(f"  Points per N: {summary}")

# Build dataset with dual criterion
dataset = agg.build_combined_dataset(max_de_gap=0.10)
print(f"  Dataset: {len(dataset)} quality-filtered graphs")

if len(dataset) < 15:
    print(f"  ERROR: Only {len(dataset)} points, need more data.")
    exit(1)

# Train
sample = dataset[0]
n_feat = sample.x.shape[1]
model = UnifiedMPNN(
    node_features=n_feat,
    hidden_dim=256,
    n_layers=3,
    norm_type="none",
    dropout=0.1,
)

print(f"  Training: {len(dataset)} pts, hidden=256, layers=3, epochs=5000...")
t0 = time.perf_counter()
result = train_unified_mpnn(
    model,
    dataset,
    n_epochs=5000,
    lr=1e-3,
    patience=400,
    seed=42,
    mse_floor=1e-5,
)
elapsed = time.perf_counter() - t0

final_mse = result.get("final_mse", 0) if isinstance(result, dict) else 0
best_epoch = result.get("best_epoch", 0) if isinstance(result, dict) else 0
print(f"  Done: MSE={final_mse:.2e}, best_epoch={best_epoch}, time={elapsed:.1f}s")

# Register in zoo
n_values_str = "+".join(str(n) for n in agg.available_n_values())
entry = ZooEntry(
    model="tfim_bond_resolved",
    topology="ladder",
    n_qubits=0,
    p_layers=1,
    checkpoint_file=f"unified_tfim_br_ladder_multiN_{n_values_str}_p1.pt",
    h_range=(2.0, 5.5),
    pass_rate=0.0,
    n_training_points=len(dataset),
    seeds=[42],
    created=datetime.now(UTC).isoformat(),
    notes=f"Retrained with dashboard filter (not_useful excluded). N={agg.available_n_values()}, {len(dataset)} pts, MSE={final_mse:.2e}",
)
register_checkpoint(model, entry, overwrite=True)
print(f"  Registered: {entry.checkpoint_file}")
print("\nStep 2 COMPLETE")
