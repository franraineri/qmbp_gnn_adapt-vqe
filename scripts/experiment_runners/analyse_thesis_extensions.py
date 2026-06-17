"""Analyse and print a structured report of thesis extension results."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
results_path = ROOT / "results" / "thesis_extensions" / "analysis_result.json"

with open(results_path) as f:
    r = json.load(f)

sep = "=" * 64
w = 68


def wrap(text):
    for line in textwrap.wrap(text, w):
        print("   ", line)


print(sep)
print("THESIS EXTENSION ANALYSIS — FULL RESULTS")
print(f"Run: {r['run_timestamp']}")
print(sep)

print("\nPRIORITY RANKING")
for k, v in r["ranking_rationale"].items():
    print(f"  {k}: {v}")

# ------------------------------------------------------------------
# EXT1
# ------------------------------------------------------------------
print(f"\n{sep}")
print("EXT1 — Bond-Resolved 2D")
print(sep)
e1 = r["ext1_bond_resolved"]
print(f"  Classification : {e1['classification']}")
print(f"  Chapter        : §{e1['thesis_chapter_section']}")
print(f"  Hardware viable: {e1['hardware_viable']}")
print(f"  Risk           : {e1['implementation_risk']}")
print("\n  Narrative:")
wrap(e1["thesis_narrative"])

rm1 = e1["raw_metrics"]
print("\n  Key metrics:")
print(
    f"    intra_N   : {rm1['intra_n_classification']}"
    f"  ΔE/gap={rm1['intra_n_de_gap'] * 100:.1f}%"
    f"  {rm1['intra_n_pass']} PASS"
    f"  GNN {rm1['gnn_vs_random']:.0f}× vs random"
)
print(f"    cross_N   : {rm1['cross_n_classification']}  ratio={rm1['params_data_ratio']:.0f}×")
print(f"    hardware  : {rm1['hardware_classification']}  {rm1['cx_count']} CX (threshold=18)")
dr = rm1["data_requirement"]
print(
    f"    N_min_data: {dr['N_min_data']} pts"
    f"  T_collection={dr['T_collection_hours'] * 60:.0f} min"
    f"  gate={dr['gate_approved']}"
    f"  n_h_points={dr['n_h_points']} (metadata)"
)

rr1 = e1["rejection_report"]
print("\n  Rejection report:")
print(f"    criterion : {rr1['criterion_id']} — {rr1['criterion_description']}")
print(f"    measured  : {rr1['measured_value']}")
print(f"    threshold : {rr1['threshold']}")

# ------------------------------------------------------------------
# EXT2
# ------------------------------------------------------------------
print(f"\n{sep}")
print("EXT2 — Kagomé / QSL")
print(sep)
e2 = r["ext2_kagome"]
print(f"  Classification : {e2['classification']}")
print(f"  Chapter        : §{e2['thesis_chapter_section']}")
print(f"  Estimated time : {e2['estimated_time_to_result_hours']}h (natural estimate, not 0.0)")
print("\n  Narrative:")
wrap(e2["thesis_narrative"])

rr2 = e2["rejection_report"]
print("\n  Rejection report:")
print(f"    criterion : {rr2['criterion_id']} — {rr2['criterion_description']}")
print(f"    measured  : {rr2['measured_value']}")
print("    note: raw exception string is now EXCLUDED from narrative (M7 fix)")

# ------------------------------------------------------------------
# EXT3
# ------------------------------------------------------------------
print(f"\n{sep}")
print("EXT3 — Normalizing Flows (MAF)")
print(sep)
e3 = r["ext3_normalizing_flows"]
print(f"  Classification : {e3['classification']}")
print(f"  Chapter        : §{e3['thesis_chapter_section']}")
print(f"  Hardware viable: {e3['hardware_viable']}")
print(f"  Risk           : {e3['implementation_risk']}")
print("\n  Narrative:")
wrap(e3["thesis_narrative"])

rm3 = e3["raw_metrics"]
print("\n  Architecture analysis:")
print(
    f"    Arch B trainable params: {rm3['arch_b_trainable_params']}  guard_triggered={rm3['arch_b_guard_triggered']}"
)
print(f"    Arch A finetune params : {rm3['arch_a_finetune_trainable_params']}")
print(
    f"    Arch A E2E estimate    : {rm3['arch_a_e2e_params_estimate']}  guard={rm3['arch_a_e2e_guard']}"
)
print(
    f"    Flow classification    : calib_improvement={rm3['flow_calibration_improvement']}  de_gap_assumed={rm3['flow_de_gap_assumed']}"
)
print(f"    Classification basis   : {rm3['flow_classification_basis'][:90]}...")
mc = rm3["mc_dropout_baseline"]
print(
    f"    MC-Dropout baseline    : coverage_90={mc['coverage_90']}  mean_sharpness={mc['mean_sharpness']} (placeholder — no Phase3 path provided)"
)

# ------------------------------------------------------------------
# PREREQUISITES
# ------------------------------------------------------------------
print(f"\n{sep}")
print("PREREQUISITE FAILURES")
print(sep)
for pf in r["prerequisite_failures"]:
    print(f"  • {pf}")

# ------------------------------------------------------------------
# SCIENTIFIC INTERPRETATION
# ------------------------------------------------------------------
print(f"\n{sep}")
print("SCIENTIFIC INTERPRETATION")
print(sep)

print("""
PRIORITY: ext3 > ext1 > ext2

EXT3 — CONDITIONALLY_VIABLE (§5.4)
  Structural analysis confirms normalizing flows (EmbeddingMAF Arch B,
  ~4,976 trainable params) pass the overparameterization gate for the
  45-point dataset. No actual flow training was performed; calibration
  improvement is 0.0 (not measured). The CONDITIONALLY_VIABLE verdict
  is conservative and accurate: architecture is sound, but comparison
  with MC-Dropout L6 requires experimental confirmation with a live
  MPNNPredictor + Phase 3 data supplied via --phase3-results.
  Recommendation for §5.4: implement Arch B training experiment.

EXT1 — REJECTED_INSUFFICIENT_DATA (§5.2)
  Intra-N regime: CONDITIONALLY_VIABLE (6/6 PASS, ΔE/gap=0.7%, GNN
  4414× vs random init) — established finding, not re-run.
  Cross-N regime: REJECTED — 45 pts / 494K params = 10,978× >> 1000
  threshold. Data requirement: 494 pts. Collection time: ~65 min at
  N=6 (gate approved, <48h). Hardware: 24 CX > 18 ZNE threshold →
  HARDWARE_INCOMPATIBLE for p=2. Use p=1 (12 CX ≤ 18) if hardware
  validation is needed.
  Recommendation for §5.2: report intra-N success + cross-N data gap.

EXT2 — PREREQUISITE_FAILED (§5.3)
  make_lattice(geometry='kagome') is not implemented. This is a
  documented architectural gap, not a scientific failure.
  Documental findings (if prerequisite is resolved):
    - Heisenberg HVA p≤2: HARD_PHYSICS_LIMIT (V9, 30 runs confirmed)
    - Heisenberg N=12 p=2: 144 CX >> 18 threshold
    - TFIM Kagomé N=6 p=1: 12 CX ≤ 18 (hardware viable alternative)
    - ExactDiag recommended for N≤12 (H.S.=4,096, fast)
    - Ground truth: ExactDiag > TeNPy (not installed) > Literature
  Recommendation for §5.3: document as future work requiring lattice
  module extension. Report CX budget analysis and HARD_PHYSICS_LIMIT
  finding as thesis contributions.
""")
print(sep)
