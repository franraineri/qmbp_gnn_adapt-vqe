import json
from pathlib import Path

thesis = Path("results/thesis")
for folder in sorted(thesis.iterdir()):
    if not folder.is_dir() or folder.name.startswith("."):
        continue
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            for f in sorted(sub.glob("pipeline_run_*.json"))[:1]:
                data = json.load(open(f))
                diag = data.get("diagnostics", {})
                p1 = diag.get("phase1", {})
                print(f"PHASE1 DATA (NOT EXTRACTED): {p1}")
                p2 = diag.get("phase2", {})
                print(f"PHASE2 per_h_timing count: {len(p2.get('per_h_timing_s', []))}")
                print(f"PHASE2 per_h_iterations: {p2.get('per_h_iterations', [])[:3]}")
                print(f"PHASE2 per_h_restart_spread: {p2.get('per_h_restart_spread', [])[:3]}")
                print(f"PHASE2 total_elapsed_s: {p2.get('total_elapsed_s')}")
                p3 = diag.get("phase3", {})
                per_h_mse = p3.get("per_h_mse", {})
                print(f"PHASE3 per_h_mse keys: {list(per_h_mse.keys())[:5]}")
                print(f"PHASE3 theta_x_mse: {p3.get('theta_x_mse')}")
                print(f"PHASE3 loss_curve_last100 len: {len(p3.get('loss_curve_last100', []))}")
                print(f"PHASE3 elapsed_s: {p3.get('elapsed_s')}")
                p4d = diag.get("phase4", {})
                print(f"PHASE4 DIAG KEYS: {list(p4d.keys())}")
                print(f"  snr_mag_x: {p4d.get('snr_mag_x')}")
                print(f"  snr_corr_zz: {p4d.get('snr_corr_zz')}")
                print(f"  classification_confidence: {p4d.get('classification_confidence')}")
                print(f"  ces_energy_pearson_r: {p4d.get('ces_energy_pearson_r')}")
                print(f"  energy_decomposition: {p4d.get('energy_decomposition')}")
                print(f"  total_shots: {p4d.get('total_shots')}")
                break
            break
    break

# Also check GNN-QEM
gnn_dir = Path("results/gnn_qem")
if gnn_dir.exists():
    for f in sorted(gnn_dir.glob("*.json"))[:1]:
        data = json.load(open(f))
        print(f"\nGNN-QEM: {f.name}")
        print(f"  TOP KEYS: {list(data.keys())[:12]}")
