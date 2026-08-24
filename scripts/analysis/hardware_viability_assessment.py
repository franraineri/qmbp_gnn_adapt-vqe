#!/usr/bin/env python
"""Hardware Viability Assessment: Evaluates feasibility of quantum experiments.

Combines existing data (GT cache, NPZ, zoo models, DQPT trajectories) to assess:
1. Amortized efficiency (MPNN inference vs classical cost)
2. GNN fidelity by h-range (where predictions are hardware-viable)
3. Quench direction feasibility (which directions produce DQPTs)
4. Hardware readiness (circuit cost, error budget, QPU time estimate)
5. Model readiness & scalability score (zoo quality)
6. Go/No-Go gate (combines QPT + DQPT criteria)
7. Best viable path forward

Usage:
    .venv/bin/python scripts/analysis/hardware_viability_assessment.py
    .venv/bin/python scripts/analysis/hardware_viability_assessment.py --topology heavy_hex
    .venv/bin/python scripts/analysis/hardware_viability_assessment.py --save
    .venv/bin/python scripts/analysis/hardware_viability_assessment.py --full  # includes live circuit cost + QPU time
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Section 1: Amortized Efficiency
# ---------------------------------------------------------------------------

def assess_amortized_efficiency(topology: str = "heavy_hex") -> dict:
    """Measure MPNN inference cost vs classical ground truth cost.

    Uses:
    - GT cache entries count (proxy for classical compute invested)
    - NPZ training points (actual training data)
    - Zoo model n_training_points
    - MPNN inference time (measured live)
    """
    from qmbp_simulation.predictors.model_zoo import _load_manifest

    gt_path = ROOT / "data" / "ground_truth_cache.json"
    gt = json.load(open(gt_path))
    entries = gt.get("entries", {})

    # Count GT points for this topology
    topo_entries = {k: v for k, v in entries.items() if k.startswith(f"{topology}|")}
    n_gt_points = len(topo_entries)

    # Training NPZ points
    npz_dir = ROOT / "data" / "multi_n_training"
    training_points = 0
    training_files = []
    for f in sorted(npz_dir.glob(f"{topology}_*_p1.npz")):
        d = np.load(f)
        n_pts = len(d["h_values"])
        training_points += n_pts
        training_files.append({"file": f.name, "n_points": n_pts,
                               "h_range": [float(d["h_values"].min()), float(d["h_values"].max())]})

    # Extrapolation points (free inferences already done)
    ext_dir = ROOT / "data" / "large_n_extrapolation"
    extrapolation_inferences = 0
    for f in ext_dir.glob(f"{topology}_*_p1.npz"):
        d = np.load(f, allow_pickle=True)
        extrapolation_inferences += len(d["h_values"])

    # Zoo models
    manifest = _load_manifest()
    topo_models = [e for e in manifest if e.topology == topology]

    # Measure MPNN inference time
    inference_time_ms = _measure_mpnn_inference_time(topology)

    # Estimated classical cost per point (heuristic from method + N)
    # ED: ~0.1s for N<=14, DMRG: ~1-60s for N=16-60
    n_by_method = defaultdict(int)
    for k, v in topo_entries.items():
        method = v.get("method", "unknown")
        n_by_method[method] += 1

    # Amortization calculation
    # After training_points invested classically, every new prediction is free
    crossover_point = training_points  # After this many queries, GNN is cheaper

    return {
        "topology": topology,
        "gt_cache_points": n_gt_points,
        "training_points_invested": training_points,
        "training_files": training_files,
        "extrapolation_inferences_done": extrapolation_inferences,
        "zoo_models": len(topo_models),
        "methods_used": dict(n_by_method),
        "mpnn_inference_time_ms": inference_time_ms,
        "amortization_crossover": crossover_point,
        "total_free_predictions": extrapolation_inferences,
        "efficiency_ratio": (extrapolation_inferences / max(training_points, 1)),
    }


def _measure_mpnn_inference_time(topology: str) -> float:
    """Measure actual MPNN inference time in ms."""
    try:
        from qmbp_simulation.predictors.model_zoo import load_best_model_for
        import torch

        model, entry, _ = load_best_model_for(topology, p_layers=1)
        if model is None:
            return -1.0

        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice

        lattice = make_lattice(topology, 10, J=1.0, h=3.0)
        graph = build_unified_bond_resolved_graph(lattice, h_value=3.0, p_layers=1)

        # Warm up
        with torch.no_grad():
            for _ in range(5):
                model(graph)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                model(graph)
                times.append((time.perf_counter() - t0) * 1000)

        return float(np.median(times))
    except Exception as e:
        logger.warning(f"Could not measure inference time: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# Section 2: GNN Fidelity by h-range
# ---------------------------------------------------------------------------

def assess_gnn_fidelity(topology: str = "heavy_hex") -> dict:
    """Evaluate where GNN predictions are good enough for hardware.

    Reads extrapolation NPZ data and computes fidelity bounds from ΔE/gap.
    """
    ext_dir = ROOT / "data" / "large_n_extrapolation"
    results_by_n = {}

    for f in sorted(ext_dir.glob(f"{topology}_*_p1.npz")):
        n_str = f.stem.split("_N")[1].split("_p")[0]
        n = int(n_str)

        d = np.load(f, allow_pickle=True)
        h_values = d["h_values"]
        e_pred = d.get("e_vqe", d.get("e_pred", None))
        e_exact = d.get("e_exact", None)
        gaps = d.get("gaps", None)

        if e_pred is None or e_exact is None or gaps is None:
            continue

        points = []
        for i, h in enumerate(h_values):
            gap = float(gaps[i]) if gaps[i] and gaps[i] > 0 else 0.314
            de = abs(float(e_pred[i]) - float(e_exact[i]))
            de_gap = de / gap if gap > 0 else 999
            # Fidelity lower bound: F >= 1 - ΔE/gap
            f_bound = max(0.0, 1.0 - de_gap)
            points.append({
                "h": float(h), "de_gap": float(de_gap),
                "fidelity_bound": float(f_bound),
                "pass_5pct": de_gap < 0.05,
                "pass_10pct": de_gap < 0.10,
            })

        n_pass_5 = sum(1 for p in points if p["pass_5pct"])
        n_pass_10 = sum(1 for p in points if p["pass_10pct"])

        # Find h_viable_min: lowest h where pass_5pct holds
        passing_h = sorted([p["h"] for p in points if p["pass_5pct"]])
        h_viable_min = passing_h[0] if passing_h else None

        results_by_n[n] = {
            "n_points": len(points),
            "n_pass_5pct": n_pass_5,
            "n_pass_10pct": n_pass_10,
            "pass_rate_5pct": n_pass_5 / len(points) if points else 0,
            "h_range": [float(h_values.min()), float(h_values.max())],
            "h_viable_min": h_viable_min,
            "mean_fidelity_bound": float(np.mean([p["fidelity_bound"] for p in points])),
        }

    # Determine overall viable h range
    all_viable_mins = [v["h_viable_min"] for v in results_by_n.values() if v["h_viable_min"]]
    overall_h_viable_min = max(all_viable_mins) if all_viable_mins else None

    return {
        "topology": topology,
        "by_n": results_by_n,
        "overall_h_viable_min": overall_h_viable_min,
        "recommendation": (
            f"GNN predictions are hardware-viable for h >= {overall_h_viable_min:.1f}"
            if overall_h_viable_min else "Insufficient data"
        ),
    }


# ---------------------------------------------------------------------------
# Section 3: Quench Direction Feasibility
# ---------------------------------------------------------------------------

def assess_quench_directions(topology: str = "heavy_hex") -> dict:
    """Evaluate which quench directions produce DQPTs and are GNN-preparable.

    Reads existing DQPT trajectory NPZ files.
    """
    traj_dir = ROOT / "data" / "dqpt_trajectories"
    trajectories = []

    for f in sorted(traj_dir.glob(f"{topology}_N*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            L = d["loschmidt_echo"]
            h_pre = float(d["h_pre"]) if "h_pre" in d.files else None
            h_post = float(d["h_post"]) if "h_post" in d.files else None
            n_qubits = int(d["n_qubits"]) if "n_qubits" in d.files else None
            ct = d["critical_times"] if "critical_times" in d.files else np.array([])

            # Classify pattern
            if len(L) > 2:
                ups = sum(1 for i in range(1, len(L)) if L[i] > L[i - 1])
                is_oscillating = ups > len(L) * 0.2
            else:
                is_oscillating = False

            trajectories.append({
                "file": f.name,
                "n_qubits": n_qubits,
                "h_pre": h_pre,
                "h_post": h_post,
                "n_steps": len(L) - 1,
                "L_min": float(L.min()),
                "n_dqpts": len(ct),
                "is_oscillating": is_oscillating,
                "pattern": "oscillating (DQPT)" if is_oscillating else "monotone decay",
                "gnn_preparable": h_pre is not None and h_pre >= 2.5,
            })
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")

    # Classify by direction
    from_ordered = [t for t in trajectories if t["h_pre"] and t["h_pre"] < 1.5]
    from_paramagnetic = [t for t in trajectories if t["h_pre"] and t["h_pre"] >= 2.5]

    return {
        "topology": topology,
        "total_trajectories": len(trajectories),
        "from_ordered_phase": {
            "count": len(from_ordered),
            "dqpt_detected": sum(1 for t in from_ordered if t["n_dqpts"] > 0),
            "oscillating": sum(1 for t in from_ordered if t["is_oscillating"]),
            "gnn_preparable": False,
            "note": "DQPTs detected but GNN cannot prepare h<1.5 states",
        },
        "from_paramagnetic_phase": {
            "count": len(from_paramagnetic),
            "dqpt_detected": sum(1 for t in from_paramagnetic if t["n_dqpts"] > 0),
            "oscillating": sum(1 for t in from_paramagnetic if t["is_oscillating"]),
            "gnn_preparable": True,
            "note": "GNN can prepare these states but L(t) shows monotone decay (no oscillatory DQPTs)",
        },
        "trajectories": trajectories,
        "viable_direction": _determine_viable_direction(from_ordered, from_paramagnetic),
    }


def _determine_viable_direction(from_ordered, from_paramagnetic):
    """Determine the best viable quench direction."""
    # Check if paramagnetic direction has ANY oscillation at longer times
    param_oscillating = [t for t in from_paramagnetic if t["is_oscillating"]]

    if param_oscillating:
        return {
            "direction": "paramagnetic → ordered",
            "viable": True,
            "reason": "Oscillations detected from paramagnetic phase",
        }

    # Best path: use h=1.5-2.0 (borderline, GNN works with p>=2) → quench to h<0.5
    return {
        "direction": "h=1.5-2.0 → h=0.3-0.5 (recommended)",
        "viable": "partial",
        "reason": (
            "Standard DQPTs (h<1→h>1) require states GNN cannot prepare (h<1.5). "
            "Best compromise: prepare at h=1.5-2.0 (GNN viable with p>=2, bond-resolved to h=0.83) "
            "and quench to h=0.3-0.5. OR: use ED ground state for N<=22 and focus on QPU for dynamics."
        ),
    }


# ---------------------------------------------------------------------------
# Section 4: Hardware Readiness
# ---------------------------------------------------------------------------

def assess_hardware_readiness(topology: str = "heavy_hex") -> dict:
    """Check circuit cost and error budget for QPU execution."""
    cost_file = ROOT / "results" / "analysis" / f"circuit_cost_{topology}_N51.json"
    if not cost_file.exists():
        return {"error": f"No circuit cost data for {topology} N=51"}

    cost_data = json.load(open(cost_file))
    results = cost_data.get("results", [])

    viable_configs = []
    for r in results:
        if r.get("fits_qesem", False):
            viable_configs.append({
                "trotter_steps": r["trotter_steps"],
                "n_ecr": r["n_ecr_estimated"],
                "depth_2q": r["circuit_depth_2q"],
                "t2_ratio": r["t2_budget_ratio"],
                "error_budget": r["error_budget"],
            })

    return {
        "topology": topology,
        "n_qubits": cost_data.get("n_qubits", 51),
        "viable_trotter_configs": viable_configs,
        "max_trotter_steps_viable": max(c["trotter_steps"] for c in viable_configs) if viable_configs else 0,
        "hardware_params": cost_data.get("hardware_params", {}),
    }


# ---------------------------------------------------------------------------
# Section 5: Model Readiness & Scalability Score
# ---------------------------------------------------------------------------

def assess_model_readiness(topology: str = "heavy_hex") -> dict:
    """Evaluate zoo model deployment readiness and scalability.

    Uses compute_model_readiness from model_zoo and compute_scalability_score
    from analysis.metrics.
    """
    from qmbp_simulation.predictors.model_zoo import _load_manifest, compute_model_readiness

    manifest = _load_manifest()
    topo_models = [e for e in manifest if e.topology == topology]

    if not topo_models:
        return {"topology": topology, "error": "No zoo models for this topology"}

    readiness_results = []
    for entry in topo_models:
        try:
            readiness = compute_model_readiness(entry, n_target=20)
            readiness_results.append({
                "checkpoint": str(entry.checkpoint_file)[:60],
                "grade": readiness.get("grade", "?"),
                "readiness_score": readiness.get("readiness_score", 0),
                "recommendation": readiness.get("recommendation", "unknown"),
                "pass_rate": entry.pass_rate,
                "n_training_points": entry.n_training_points,
            })
        except Exception as e:
            readiness_results.append({
                "checkpoint": str(entry.checkpoint_file)[:60],
                "error": str(e),
            })

    # Scalability score
    scalability = None
    try:
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        # Find best model's metrics for scalability
        best = max(topo_models, key=lambda e: e.pass_rate)
        # Estimate n_max_viable and h_frontier from fidelity data
        ext_dir = ROOT / "data" / "large_n_extrapolation"
        n_max_viable = 10
        for f in ext_dir.glob(f"{topology}_*_p1.npz"):
            n_str = f.stem.split("_N")[1].split("_p")[0]
            n = int(n_str)
            d = np.load(f, allow_pickle=True)
            de_gaps = d.get("de_gaps", np.array([]))
            if len(de_gaps) > 0 and np.mean(de_gaps < 0.05) > 0.5:
                n_max_viable = max(n_max_viable, n)

        score, reason = compute_scalability_score(
            topology=topology,
            n_max_viable=n_max_viable,
            pass_rate_dual=best.pass_rate,
            h_frontier=2.5,
        )
        scalability = {"score": score, "reason": reason, "n_max_viable": n_max_viable}
    except Exception as e:
        scalability = {"error": str(e)}

    return {
        "topology": topology,
        "n_models": len(topo_models),
        "readiness": readiness_results,
        "best_grade": max((r.get("grade", "F") for r in readiness_results), default="F"),
        "scalability": scalability,
    }


# ---------------------------------------------------------------------------
# Section 6: Go/No-Go Gate (QPT + DQPT combined)
# ---------------------------------------------------------------------------

def assess_go_no_go(topology: str = "heavy_hex") -> dict:
    """Run the combined QPT + DQPT go/no-go evaluation.

    Uses validate_dqpt_results.compute_go_no_go which aggregates all criteria.
    """
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

    try:
        from validate_dqpt_results import compute_go_no_go
        result = compute_go_no_go(topology, p_layers=1)
        return {
            "topology": topology,
            "overall_go": result.get("overall_go", False),
            "n_passed": result.get("n_passed", 0),
            "n_total": result.get("n_total", 0),
            "blocking_issues": result.get("blocking_issues", []),
        }
    except Exception as e:
        return {"topology": topology, "error": str(e)}


# ---------------------------------------------------------------------------
# Section 7: Live Circuit Cost + QPU Time Budget
# ---------------------------------------------------------------------------

def assess_qpu_budget(topology: str = "heavy_hex", n_qubits: int = 51) -> dict:
    """Compute live circuit cost and QPU time estimate for target experiment.

    Uses circuit_cost_check.compute_circuit_cost and
    qpu_time_estimator.estimate_circuit_qpu_time.
    """
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    sys.path.insert(0, str(ROOT / "project_health" / "cli"))

    results = []
    try:
        from circuit_cost_check import compute_circuit_cost
        from qpu_time_estimator import estimate_circuit_qpu_time

        for steps in [10, 15, 20, 25, 30]:
            cost = compute_circuit_cost(
                topology=topology, n_qubits=n_qubits, p_layers=1,
                h_prep=3.0, h_quench=0.5, dt=0.1, trotter_steps=steps,
            )
            qpu_time = estimate_circuit_qpu_time(
                depth=cost.circuit_depth_2q,
                n_2q=cost.n_ecr_estimated,
                method="pea_balanced",
                shots=16384,
            )

            results.append({
                "trotter_steps": steps,
                "n_ecr": cost.n_ecr_estimated,
                "depth_2q": cost.circuit_depth_2q,
                "t2_ratio": cost.t2_budget_ratio,
                "fits_qesem": cost.fits_qesem,
                "error_budget": cost.error_budget,
                "qpu_time_per_circuit_s": qpu_time["total_s"],
                "qpu_time_per_h_point_min": qpu_time["total_s"] / 60,
            })
    except Exception as e:
        return {"topology": topology, "n_qubits": n_qubits, "error": str(e)}

    # Estimate total QPU budget for a full experiment
    n_h_points = 10  # Typical sweep
    viable = [r for r in results if r["fits_qesem"]]
    if viable:
        recommended = viable[len(viable) // 2]  # Middle option
        total_time_min = recommended["qpu_time_per_h_point_min"] * n_h_points
    else:
        recommended = None
        total_time_min = None

    return {
        "topology": topology,
        "n_qubits": n_qubits,
        "configs": results,
        "recommended_steps": recommended["trotter_steps"] if recommended else None,
        "estimated_total_qpu_min": total_time_min,
        "n_h_points_assumed": n_h_points,
    }


# ---------------------------------------------------------------------------
# Section 5 (original): Best Path Forward
# ---------------------------------------------------------------------------

def determine_best_path(efficiency, fidelity, quench, hardware) -> dict:
    """Synthesize all assessments into a recommended path."""
    recommendations = []
    blockers = []

    # Efficiency assessment
    if efficiency["efficiency_ratio"] > 2:
        recommendations.append({
            "priority": 1,
            "action": "Amortized Efficiency Paper",
            "description": (
                f"Already {efficiency['extrapolation_inferences_done']} free predictions from "
                f"{efficiency['training_points_invested']} training points. "
                f"Ratio: {efficiency['efficiency_ratio']:.1f}x. Publishable as ML-efficiency result."
            ),
            "effort": "2h (analysis script, data already exists)",
            "requires_hardware": False,
        })

    # Fidelity assessment
    h_min = fidelity.get("overall_h_viable_min")
    if h_min and h_min <= 3.0:
        recommendations.append({
            "priority": 2,
            "action": f"Deploy GNN at h >= {h_min:.1f} on QPU",
            "description": (
                f"GNN predictions pass dual criterion for h >= {h_min:.1f}. "
                "Run VQE warm-started by GNN on IBM Heron for N=10-20 heavy_hex. "
                "Compare wall-time vs cold-start VQE."
            ),
            "effort": "4h QPU time + analysis",
            "requires_hardware": True,
        })
    else:
        blockers.append("GNN fidelity insufficient at low h — limit to h >= 3.0")

    # Quench assessment
    quench_viable = quench.get("viable_direction", {})
    if quench_viable.get("viable") == "partial":
        recommendations.append({
            "priority": 3,
            "action": "Quench from h=1.5 (bond-resolved) or ED state",
            "description": quench_viable.get("reason", ""),
            "effort": "1 week (code + noiseless validation + QPU)",
            "requires_hardware": True,
        })

    # Hardware readiness
    max_steps = hardware.get("max_trotter_steps_viable", 0)
    if max_steps >= 15:
        recommendations.append({
            "priority": 4,
            "action": f"Dynamics on QPU: {max_steps} Trotter steps feasible",
            "description": (
                f"Circuit cost analysis shows N=51 heavy_hex with up to {max_steps} "
                "Trotter steps fits within QESEM error budget. "
                "This is the regime where IBM demonstrated quantum advantage."
            ),
            "effort": "Requires IBM Quantum Credits + QESEM access",
            "requires_hardware": True,
        })

    # Immediate actions (no hardware needed)
    recommendations.append({
        "priority": 0,
        "action": "χ-convergence Panel A+B plot (no hardware)",
        "description": (
            "Run MPS precision study at multiple χ for: "
            "(A) ground state evaluation (should converge at χ=64), "
            "(B) time evolution (should diverge at step ~10-15). "
            "This is the key thesis figure."
        ),
        "effort": "4-8h compute (noiseless MPS runs)",
        "requires_hardware": False,
    })

    return {
        "recommendations": sorted(recommendations, key=lambda x: x["priority"]),
        "blockers": blockers,
        "overall_viability": "VIABLE" if not blockers else "PARTIAL (blockers exist)",
        "best_immediate_action": recommendations[0]["action"] if recommendations else "None",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hardware Viability Assessment")
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--full", action="store_true", help="Include live circuit cost + QPU time (slower)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    topo = args.topology
    print(f"\n{'='*70}")
    print(f"  HARDWARE VIABILITY ASSESSMENT: {topo}")
    print(f"{'='*70}\n")

    # Run assessments
    print("1. Amortized Efficiency...")
    efficiency = assess_amortized_efficiency(topo)
    print(f"   Training invested: {efficiency['training_points_invested']} points")
    print(f"   Free predictions done: {efficiency['extrapolation_inferences_done']}")
    print(f"   Efficiency ratio: {efficiency['efficiency_ratio']:.1f}x")
    print(f"   MPNN inference: {efficiency['mpnn_inference_time_ms']:.2f} ms")

    print("\n2. GNN Fidelity by h-range...")
    fidelity = assess_gnn_fidelity(topo)
    print(f"   Overall viable h_min: {fidelity['overall_h_viable_min']}")
    for n, data in sorted(fidelity["by_n"].items()):
        print(f"   N={n:>3}: pass_5%={data['pass_rate_5pct']:.0%} ({data['n_pass_5pct']}/{data['n_points']}), "
              f"h_viable>={data['h_viable_min']}")

    print("\n3. Quench Direction Feasibility...")
    quench = assess_quench_directions(topo)
    print(f"   Total trajectories: {quench['total_trajectories']}")
    print(f"   From ordered (h<1.5): {quench['from_ordered_phase']['count']} "
          f"({quench['from_ordered_phase']['dqpt_detected']} with DQPTs)")
    print(f"   From paramagnetic (h>2.5): {quench['from_paramagnetic_phase']['count']} "
          f"({quench['from_paramagnetic_phase']['dqpt_detected']} with DQPTs)")
    print(f"   Viable direction: {quench['viable_direction']['direction']}")

    print("\n4. Hardware Readiness (static)...")
    hardware = assess_hardware_readiness(topo)
    if "error" not in hardware:
        print(f"   N={hardware['n_qubits']} qubits, max viable steps: {hardware['max_trotter_steps_viable']}")
        print(f"   QESEM-compatible configs: {len(hardware['viable_trotter_configs'])}")
    else:
        print(f"   {hardware['error']}")

    print("\n5. Model Readiness & Scalability...")
    model_readiness = assess_model_readiness(topo)
    if "error" not in model_readiness:
        print(f"   Zoo models: {model_readiness['n_models']}, best grade: {model_readiness['best_grade']}")
        for r in model_readiness["readiness"][:3]:
            if "grade" in r:
                print(f"     {r['checkpoint'][:45]:45s} grade={r['grade']} rec={r['recommendation']}")
        if model_readiness.get("scalability") and "score" in model_readiness["scalability"]:
            sc = model_readiness["scalability"]
            print(f"   Scalability: {sc['score']:.2f} ({sc['reason']}), N_max_viable={sc['n_max_viable']}")
    else:
        print(f"   {model_readiness.get('error', 'No data')}")

    print("\n6. Go/No-Go Gate (QPT + DQPT)...")
    go_no_go = assess_go_no_go(topo)
    if "error" not in go_no_go:
        status = "GO" if go_no_go["overall_go"] else "NO-GO"
        print(f"   Overall: {status} ({go_no_go['n_passed']}/{go_no_go['n_total']} criteria pass)")
        if go_no_go.get("blocking_issues"):
            for issue in go_no_go["blocking_issues"][:3]:
                print(f"   BLOCKER: {issue}")
    else:
        print(f"   {go_no_go.get('error', 'Unavailable')}")

    qpu_budget = {}
    if args.full:
        print("\n7. Live Circuit Cost + QPU Budget...")
        qpu_budget = assess_qpu_budget(topo, n_qubits=51)
        if "error" not in qpu_budget:
            print(f"   Recommended Trotter steps: {qpu_budget['recommended_steps']}")
            print(f"   Estimated total QPU time: {qpu_budget['estimated_total_qpu_min']:.1f} min "
                  f"({qpu_budget['n_h_points_assumed']} h-points)")
            print(f"   Configs evaluated:")
            for c in qpu_budget["configs"]:
                qesem = "✓" if c["fits_qesem"] else "✗"
                print(f"     {c['trotter_steps']:>2} steps: {c['n_ecr']} ECR, "
                      f"T2={c['t2_ratio']:.3f}, QPU={c['qpu_time_per_h_point_min']:.1f}min/pt {qesem}")
        else:
            print(f"   {qpu_budget['error']}")

    print(f"\n{'─'*70}")
    print("8. Best Path Forward...")
    path = determine_best_path(efficiency, fidelity, quench, hardware)
    print(f"   Overall: {path['overall_viability']}")
    if path["blockers"]:
        print(f"   Blockers: {path['blockers']}")
    print(f"\n   RECOMMENDATIONS (priority order):")
    for r in path["recommendations"]:
        hw = " [QPU]" if r["requires_hardware"] else " [LOCAL]"
        print(f"   P{r['priority']}{hw}: {r['action']}")
        print(f"       {r['description'][:100]}")
        print(f"       Effort: {r['effort']}")
        print()

    # Save
    if args.save:
        output = {
            "topology": topo,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "efficiency": efficiency,
            "fidelity": fidelity,
            "quench": quench,
            "hardware": hardware,
            "model_readiness": model_readiness,
            "go_no_go": go_no_go,
            "qpu_budget": qpu_budget if args.full else {},
            "path_forward": path,
        }
        # Remove non-serializable
        output["quench"].pop("trajectories", None)
        out_path = ROOT / "results" / "analysis" / f"viability_assessment_{topo}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
