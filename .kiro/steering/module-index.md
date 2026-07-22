# Module Index (auto-generated)

Compact catalog of all code modules. Use to find reusable functionality.
Run `python scripts/maintenance/generate_module_index.py` to refresh.

## Library (src/qmbp_simulation)

### `src/qmbp_simulation/` — Hybrid GNN-HVA Framework for Topological Phase Characterization.
Exports: HamiltonianBuilder, make_lattice, LatticeConfig, GroundTruthResult, VQEConfig, VQEResult, SUPPORTED_TOPOLOGIES, MAX_P_LAYERS, ClassicalSolver, HVACircuitBuilder, ExecutionBackend, NoiselessBackend +16


### `qsim/analysis/` — Analysis submodule — gradient analysis, diagnostics, landscape, and entanglement.
Exports: AlignmentReport, BaselineComparison, BaselineMetrics, ClusterResult, ClusterSolver, ComparativeMetrics, ComparisonResult, DiagnosticCollector, EntanglementAnalyzer, EntanglementResult, GradientAnalysisResult, GroundTruthValidationReport +48

- `qsim/analysis/circuit_visualizer.py` — Circuit visualization utilities for hardware deployment. — F:print_circuit,save_circuit_diagram,circuit_summary,print_circuit_comparison,transpiled_circuit_stats,compute_error_budget+8
- `qsim/analysis/comparative.py` — Comparative analysis and regime discovery for multi-model evaluation. — C:RegimeDiscoveryResult,ComparativeMetrics | F:find_h_min,classify_result,filter_by_threshold,find_minimum_viable_threshold,compute_cx_budget,classify_outcome+2
- `qsim/analysis/cross_n_validator.py` — Cross-N Prediction Validator — Three-level verification for zero-shot GNN predic — C:L1Result,L2Result,L3Result,CrossNValidationReport,CrossNValidator | F:preflight_cross_n
- `qsim/analysis/data_models.py` — Analysis Data Models — Dataclasses for analysis results. — C:GradientAnalysisResult,ComparisonResult,BaselineMetrics,BaselineComparison
- `qsim/analysis/diagnostics.py` — Pipeline Observability — Diagnostic metrics collection and structured logging. — C:DiagnosticCollector | F:configure_pipeline_logging | K:MIN_LAYOUTS
- `qsim/analysis/entanglement.py` — Entanglement analysis for ground state characterization. — C:EntanglementResult,EntanglementAnalyzer
- `qsim/analysis/extension_analyzer.py` — Main orchestrator for thesis extension analysis pipeline. — C:RejectionReportGenerator,PrerequisiteChecker,CalibrationComparator,OverparameterizationGuard,ThesisImpactReporter+6
- `qsim/analysis/extension_classifiers.py` — Pure threshold-based classification engine for thesis extension analysis. — C:ClassificationEngine
- `qsim/analysis/extension_models.py` — Data models and enumerations for thesis extension analysis. — C:ExtensionClassification,PrerequisiteFailedError,HardPhysicsLimitError,RejectionReport,ExtensionResult+1
- `qsim/analysis/extension_ranker.py` — Extension priority ranking for thesis extension analysis. — C:ExtensionScore,ExtensionPriorityRanker
- `qsim/analysis/flow_warmstart.py` — Flow-based warmstart manager for VQE parameter initialisation. — C:FlowWarmstartManager
- `qsim/analysis/gradient.py` — Weight Gradient Analysis — Unsupervised phase detection from MPNN weights. — C:WeightGradientAnalyzer | K:GRADIENT_CRITICAL_REGION,GRADIENT_PEAK_PROMINENCE
- `qsim/analysis/ground_truth_validator.py` — Ground Truth Validator — Phase 1 post-computation validation. — C:GroundTruthValidationReport,GroundTruthValidator
- `qsim/analysis/landscape.py` — Landscape Analysis — Hessian computation and trainability metrics. — F:compute_hessian,landscape_fluctuation
- `qsim/analysis/metrics.py` — Analysis Metrics — Pure computation helpers for pipeline diagnostics. — F:compute_snr,compute_theta_smoothness,compute_classification_confidence,compute_energy_decomposition,compute_fraction_near_gs
- `qsim/analysis/nlce.py` — NLCE — Numerical Linked-Cluster Expansion for 1D spin systems. — C:NLCEConfig,ClusterResult,NLCEResult,ClusterSolver,VQEClusterSolver+1 | F:tfim_analytical_energy_per_site,nlce_convergence_analysis
- `qsim/analysis/normalizing_flow.py` — Normalizing flow architectures for Ext3 — Flujos Normalizantes. — C:MaskedLinear,MAFLayer,FlowHead,EmbeddingMAF
- `qsim/analysis/theta_alignment.py` — Post-VQE theta alignment — eliminates parameter discontinuities. — C:AlignmentReport,OutlierReport,EnergyGuardReport | F:detect_jumps,align_theta_sweep,align_theta_array,detect_theta_outliers,filter_theta_outliers,cross_h_energy_guard
- `qsim/analysis/theta_validator.py` — Theta Validation Module — Post-prediction quality assurance for MPNN outputs. — C:BoundCheckResult,NumericalSanityResult,InterpolationResult,FidelityResult,GradientNormResult+4 | K:DEFAULT_BOUND_SIGMAS,DEFAULT_INTERPOLATION_THRESHOLD,DEFAULT_GRADIENT_THRESHOLD,DEFAULT_MC_DROPOUT_PASSES
- `qsim/analysis/vqe_validator.py` — VQE Result Validator — Comprehensive post-optimization validation. — C:Severity,ValidationIssue,VQEValidationReport,VQEValidator

### `qsim/circuits/` — Circuits submodule — HVA circuit construction and AQC compression.
Exports: HVACircuitBuilder, AQCCircuitCompressor, AQCCompressionConfig, AQCCompressionResult, CompressionValidation, AQCCompressionCache

- `qsim/circuits/aqc_compression.py` — AQC-Tensor Circuit Compression Module. — C:AQCCompressionConfig,AQCCompressionResult,CompressionValidation,AQCCircuitCompressor,AQCCompressionCache
- `qsim/circuits/hva.py` — HVA Circuit Builder — Hamiltonian Variational Ansatz with lattice-aware — C:HVACircuitBuilder | F:do_checks

### `qsim/execution/hardware/` — Hardware execution backend for IBM Quantum processors.
Exports: HardwareBackend, HardwareConfig, HardwareRunResult, SPSAConfig, MAPOMATIC_AVAILABLE, LayoutOptimizationResult, build_filtered_coupling_map, compute_layout_fidelity_cost, find_vf2_layouts, rank_backends, select_optimal_layouts, QPUCostEstimate +4

- `qsim/execution/hardware/backend.py` — HardwareBackend — IBM Runtime execution with PEA/GF ZNE (primary) and R² quality — C:HardwareBackend | K:ZNE_R2_QUALITY_THRESHOLD
- `qsim/execution/hardware/config.py` — Configuration dataclasses for hardware execution backend. — C:HardwareConfig,SPSAConfig,HardwareRunResult
- `qsim/execution/hardware/layout_optimizer.py` — Noise-aware layout optimization using VF2 subgraph isomorphism. — C:LayoutOptimizationResult | F:build_filtered_coupling_map,find_vf2_layouts,compute_layout_fidelity_cost,select_optimal_layouts,rank_backends
- `qsim/execution/hardware/observables.py` — Construction and extraction of per-site observables. — F:build_per_site_observables,map_observables_to_layout,extract_array_result
- `qsim/execution/hardware/persistence.py` — Persistence of hardware execution results. — F:save_run,save_partial_before_error,save_sweep_summary
- `qsim/execution/hardware/phase.py` — Phase classification — pure, deterministic, testable. — F:classify_phase
- `qsim/execution/hardware/preflight.py` — Preflight checks for hardware execution. — C:QPUCostEstimate,QPUThroughputProfile,SPSACostModel,QPUCostEstimateExtended | F:compute_mean_2q_error,compute_layout_2q_error,compute_mean_readout_error,compute_min_t1_t2,check_native_gate_support,run_preflight_checks+5
- `qsim/execution/hardware/qesem.py` — QESEM integration — Qedma's Qiskit Function for unbiased error mitigation. — C:QESEMResult | F:check_qesem_available,validate_qesem_submission,extrapolate_qet_wls,estimate_qesem_time,run_qesem_deployment,run_qesem_sweep | K:QESEM_AVAILABLE
- `qsim/execution/hardware/spsa.py` — SPSA refinement for hardware execution. — F:spsa_refinement
- `qsim/execution/hardware/submission.py` — Job submission with submit-all-then-collect pattern and retry logic. — F:select_layouts_for_hardware,submit_all_then_collect,wait_for_qpu_execution,build_estimator_options

### `qsim/execution/` — Execution submodule — quantum backend abstraction layer.
Exports: ExecutionBackend, HardwareBackend, HardwareConfig, HardwareRunResult, SPSAConfig, MitigationOptions, MPSBackend, NoiselessBackend, NoisyBackend, NoisyEstimatorConfig, LayoutSelection, ZNEResult +30

- `qsim/execution/backends.py` — Execution backends for quantum circuit evaluation. — C:MitigationOptions,ExecutionBackend,NoiselessBackend,NoisyBackend,HardwareBackend | F:select_backend,select_backend_with_topology_warning
- `qsim/execution/mitiq_utils.py` — Mitiq integration module — complementary error mitigation via Mitiq toolkit. — C:MitiqZNEResult,MitiqCDRResult,MitiqDDDZNEResult,MitiqPECResult,MitiqComparisonResult | F:is_mitiq_available,make_mitiq_executor,make_noiseless_executor,run_mitiq_zne,run_mitiq_cdr,run_mitiq_ddd_zne+2
- `qsim/execution/mps_backend.py` — MPS-based execution backend for VQE at N>22. — C:MPSBackend
- `qsim/execution/noisy_utils.py` — Noisy simulation utilities — layout selection, CES computation, ZNE. — C:NoisyEstimatorConfig,LayoutSelection,ZNEResult,ZNEDeploymentResult,GateFoldingZNEResult+8 | F:build_adjacency,find_layouts_bfs,compute_circuit_ces,select_layouts_by_circuit_ces,select_layouts_low_ces,noisy_estimate+14 | K:TWO_QUBIT_GATES

### `qsim/framework/` — Framework submodule — experiment engine, config, metrics, CLI, and benchmarking.
Exports: BaseExperiment, ExperimentConfig, SystemConfig, VQEConfig, MPNNConfig, AnalysisConfig, ExperimentMetrics, WarmColdComparison, StructuredLogger, ProgressReporter, ResultStore, CATEGORY_MAP +63

- `qsim/framework/__main__.py` — Allow running preflight checks via: python -m qmbp_simulation.framework.prefligh
- `qsim/framework/artifact_serializers.py` — Artifact serializers — lazy-import format handlers. — C:ArtifactSerializer,QPYSerializer,QASM3Serializer,TorchSerializer,NumpySerializer+2 | F:get_serializer,register_serializer
- `qsim/framework/artifact_store.py` — Artifact Store — versioned persistence of experiment artifacts. — C:ArtifactEntry,ManifestEntry,ArtifactCollector | F:load_manifest,load_artifact,find_artifacts_for_run,inspect_circuit_artifact,print_circuit_summary,save_circuit_png | K:MANIFEST_SCHEMA_VERSION,ARTIFACTS_SUFFIX
- `qsim/framework/base.py` — Abstract base class for all experiments. — C:BaseExperiment
- `qsim/framework/benchmarking.py` — Performance benchmarking suite for qmbp_simulation components. — C:BenchmarkResult,BenchmarkSuite
- `qsim/framework/cli.py` — Shared CLI argument parsing for scripts and experiments. — F:create_base_parser,add_system_args,add_sweep_args,add_vqe_args,add_mpnn_args,add_output_args+10
- `qsim/framework/config.py` — Experiment configuration dataclasses. — C:SystemConfig,VQEConfig,MPNNConfig,AnalysisConfig,ExperimentConfig
- `qsim/framework/criteria.py` — Canonical experiment success criteria and verdict logic. — F:compute_verdict
- `qsim/framework/logging.py` — Structured event logging for experiment execution. — C:ExperimentEvent,StructuredLogger,ProgressReporter
- `qsim/framework/metrics.py` — Experiment metrics for framework runs. — C:ExperimentMetrics,WarmColdComparison
- `qsim/framework/preflight.py` — Reusable pre-flight validation for pipeline variant configurations AND experimen — C:Severity,Issue,VariantSpec,PreflightReport,PreflightChecker+2 | F:get_valid_regime,get_regime_threshold,validate_regime_tables,specs_from_pipeline_variants,specs_from_json,specs_from_variant_runner+1
- `qsim/framework/presets.py` — Config Presets System — YAML-based experiment configuration. — F:load_preset,list_presets,preset_to_args
- `qsim/framework/result_index.py` — Result Index — lightweight metadata cache for fast experiment discovery. — C:ResultIndex
- `qsim/framework/result_io.py` — Standardized result saving and loading for experiments and pipelines. — F:build_experiment_id,generate_timestamp,build_result_envelope,save_experiment_result,save_pipeline_result,save_benchmark_result+5
- `qsim/framework/result_store.py` — Result storage, querying, and cross-experiment comparison. — C:ResultStore
- `qsim/framework/runner_base.py` — Standardized runner base classes for experiment and validation scripts. — C:Section,SectionResult,ValidationRunner,ExperimentRunner,VariantPipelineRunner+1 | F:resolve_project_root
- `qsim/framework/variant_runner.py` — Shared variant runner infrastructure for thesis pipeline validation. — C:PipelineVariant,RunResult,VariantRunner | F:extract_metrics_from_output,run_variant,create_variant_cli,run_variant_script

### `qsim/models/` — Models submodule — data models, Hamiltonians, constants, and model registry.
Exports: DEFAULT_SEEDS, DMRG_QUBIT_LIMIT, EXACT_DIAG_QUBIT_LIMIT, EXACT_GAP_QUBIT_LIMIT, MAX_P_LAYERS, MPS_DEFAULT_CHI_MAX, STATEVECTOR_MAX_N, SUPPORTED_VQE_METHODS, ModelSpec, SUPPORTED_TOPOLOGIES, DeployResult, GroundTruthResult +9

- `qsim/models/constants.py` — Physics constants for the GNN-HVA quantum simulation framework.
- `qsim/models/data_models.py` — Data models for the GNN-HVA quantum simulation framework. — C:LatticeConfig,GroundTruthResult,VQEConfig,OptimizationTrajectory,VQEResult+1
- `qsim/models/hamiltonian.py` — Hamiltonian Builder — Construct parameterized spin Hamiltonians for arbitrary — C:HamiltonianBuilder | F:generate_chain_1d,generate_ladder,generate_square,generate_triangular,generate_kagome,generate_heavy_hex+2
- `qsim/models/model_registry.py` — ModelRegistry — Centralized registry for spin model specifications. — F:register_model,get_model_spec,list_models
- `qsim/models/model_spec.py` — ModelSpec — Strategy pattern for model-agnostic pipeline execution. — C:ModelSpec

### `qsim/optimizers/` — Optimizers submodule — VQE and SPSA optimization.
Exports: VQEOptimizer, SPSAOptimizer, AdaptiveRestartConfig, SelectiveAscendingConfig, SelectiveAscendingReport, compute_adaptive_restarts, compute_restarts_for_sweep, select_suspicious_points

- `qsim/optimizers/spsa.py` — SPSA Optimizer — Simultaneous Perturbation Stochastic Approximation. — C:SPSAOptimizer
- `qsim/optimizers/sweep_strategies.py` — VQE Sweep Strategies — Adaptive restarts and selective ascending pass. — C:AdaptiveRestartConfig,SelectiveAscendingConfig,SelectiveAscendingReport | F:compute_adaptive_restarts,compute_restarts_for_sweep,select_suspicious_points
- `qsim/optimizers/vqe.py` — VQE Optimizer — Multi-start L-BFGS-B with diagnostic callbacks and trajectory — C:OptimizationCallback,VQEOptimizer

### `qsim/pipeline/` — Pipeline submodule — orchestration, dataset I/O, and QRC fallback.
Exports: PipelineRunner, load_phase12_dataset, save_phase12_dataset, run_exact_diag_sweep

- `qsim/pipeline/dataset_io.py` — Dataset I/O — Phase 1+2 dataset persistence with metadata validation. — F:get_library_versions,save_phase12_dataset,load_phase12_dataset,generate_nonuniform_h_grid | K:PIPELINE_VERSION,EXPECTED_COST_FUNCTION
- `qsim/pipeline/qrc.py` — QRC Pipeline — Quantum Reservoir Computing fallback route. — C:QRCPipeline
- `qsim/pipeline/runner.py` — Pipeline Runner — Orchestrates the full Phase 1 → 2 → 3 → 4 workflow. — C:PipelineRunner | F:run_exact_diag_sweep

### `qsim/predictors/` — Predictors submodule — MPNN parameter prediction + GNN-QEM error correction.
Exports: MPNNPredictor, build_graph_dataset, load_mpnn_checkpoint, save_mpnn_checkpoint, train_mpnn, BondResolvedMPNN, build_bond_resolved_graph, train_bond_resolved_mpnn, GNNQEMCorrector, GNNQEMConfig, QEMSample, QEMTrainResult +10

- `qsim/predictors/gnn_qem.py` — GNN-QEM — Graph Neural Network for Quantum Error Mitigation. — C:GNNQEMConfig,GNNQEMCorrector,QEMSample,QEMTrainResult,QEMCorrectionResult | F:build_qem_graph,build_qem_dataset,train_gnn_qem,correct_energy,save_qem_checkpoint,load_qem_checkpoint+3
- `qsim/predictors/mpnn.py` — MPNN Predictor — Lattice-agnostic parameter predictor via PyTorch Geometric — C:MPNNPredictor,BondResolvedMPNN | F:build_graph_dataset,train_mpnn,save_mpnn_checkpoint,load_mpnn_checkpoint,build_bond_resolved_graph,train_bond_resolved_mpnn | K:NNCONV_EDGE_MLP_HIDDEN

### `qsim/solvers/` — Solvers submodule — exact diagonalization and DMRG.
Exports: ClassicalSolver

- `qsim/solvers/classical.py` — Classical Solver — Ground truth generation via exact diagonalization or DMRG/TeN — C:ClassicalSolver

### `qsim/utils/` — Utils submodule — shared utilities with zero internal dependencies.
Exports: TimerResult, canonicalize_theta, filter_consistent_theta, json_dump, json_serialize, set_global_seed, timer

- `qsim/utils/helpers.py` — Shared utility functions — seeding, JSON serialization, and timing. — C:TimerResult | F:set_global_seed,json_serialize,json_dump,timer,canonicalize_theta,filter_consistent_theta

## Project Health (project_health/)

### `project_health/` — Project health checker — unified orchestration of Phase 4 analysis tools.
Exports: ActionItem, CoverageGap, EnergyDecompositionStats, HealthReport, ModelDistribution, MPNNQualityStats, TimingStats, VQEQualityStats

- `ph/__main__.py` — CLI entry point for the project health checker. — F:parse_args,main
- `ph/compare.py`

### `ph/analysis/coverage/` — Coverage scanning and gap analysis CLI.

- `ph/analysis/coverage/scan_coverage.py` — Comprehensive coverage scanner for all result types in the GNN-HVA project. — C:PipelineRecord,NoisyRecord,ExperimentRecord,FilterConfig | F:discover_variant_folders,scan_pipeline_results,scan_noisy_results,scan_experiment_results,print_section,report_pipeline_coverage+11 | K:ROOT,RESULTS,THESIS,EXPERIMENTS

### `ph/analysis/hardware/` — Hardware and mitigation analysis tools.

- `ph/analysis/hardware/hw_rehearsal_analyzer.py` — Analyze HW_REHEARSAL_V2 results — comprehensive post-run diagnostics. — F:parse_args,load_runs,analyze_single_run,compute_hardware_readiness,print_text_report,main
- `ph/analysis/hardware/hw_results_analyzer.py` — Analyze hardware mitigation benchmark results. — F:load_results,check_circuit_metrics,check_qpu_time,check_energy,main | K:ROOT,HW_DIR,SIM_DIR,EXPECTED
- `ph/analysis/hardware/layout_optimizer_analyzer.py` — Layout Optimizer Analyzer — Mapomatic VF2 Integration Results. — C:LayoutSelectionRecord,BenchmarkResult,LayoutOptimizerReport | F:run_benchmark,analyze,main
- `ph/analysis/hardware/mitigation_benchmark_analyzer.py` — Mitigation Benchmark Analyzer. — C:MitigationBenchmarkAnalyzer | F:main | K:DEFAULT_RESULTS_DIR,REQUIRED_SECTIONS,REQUIRED_METADATA_KEYS,REQUIRED_RESULTS_KEYS
- `ph/analysis/hardware/mitiq_analyzer.py` — Mitiq Integration Analyzer. — C:MitiqMethodResult,MitiqComparisonSummary,MitiqAnalysisReport | F:scan_mitiq_results,format_report,format_thesis_table,get_mitiq_health_summary,run_statistical_analysis,main | K:RESULTS_ROOT,REHEARSAL_DIR,MITIQ_DIR,HARDWARE_DIR
- `ph/analysis/hardware/post_execution_validator.py` — Post-execution validator for hardware/benchmark runs. — C:Severity,Finding,ValidationReport | F:validate_envelope,validate_hardware_summary,validate_run,print_report,main | K:ROOT,BASE_CLOPS,REF_DEPTH,DEPTH_EXPONENT
- `ph/analysis/hardware/transpilation_analyzer.py` — Analyze transpilation metrics from benchmark results. — F:scan_transpilation_stats,format_transpilation_table,format_opt_level_summary,main | K:RESULTS_DIR
- `ph/analysis/hardware/validate_qet.py` — QET (Quasi-probabilistic Error Tuning) Post-Execution Validator. — C:QETValidationIssue,QETValidationReport | F:validate_qet_result,print_report,validate_file,main | K:DE_GAP_THRESHOLD

### `ph/analysis/models/` — GNN/MPNN/AQC model analysis tools.

- `ph/analysis/models/aqc_tensor_analyzer.py` — AQC-Tensor Compression Analyzer. — C:POCSummary,CrossTopologySummary,ComparisonSummary,AQCTensorReport | F:analyze,print_report,print_thesis_table,run_statistical_analysis,print_statistical_report,get_aqc_health_summary+1
- `ph/analysis/models/gnn_qem_analyzer.py` — Analyze GNN-QEM results: compare with PEA-ZNE baseline, identify validation gaps
- `ph/analysis/models/mpnn_eval_analyzer.py` — MPNN Evaluation Analyzer — processes HW_REHEARSAL_V3 section 10-14 results. — C:WarmstartResult,LOOCVResult,LandscapeResult,InterpExtrapResult,NoisyEvalResult+6 | F:parse_warmstart,parse_loo_cv,parse_landscape,parse_interp_extrap,parse_noisy_eval,parse_scaling_with_n+9 | K:ROOT,DEFAULT_RESULTS_DIR,V3_EXP_ID,V3_DIR_PATTERN

### `ph/analysis/scaling/` — MPS scaling and flow analysis tools.

- `ph/analysis/scaling/flow_warmstart_analyzer.py` — Flow Warmstart & σ_flow Extension Analyzer. — C:FlowWarmstartSummary,BondResolvedSummary,SigmaFlowBoostSummary,Ext1bSummary,FlowExtensionReport | F:scan_flow_extension_results,format_report,main
- `ph/analysis/scaling/scaling_analyzer.py` — Scaling experiment analyzer — processes MPS scaling results. — C:ScalingPointResult,ScalingRunSummary,ScalingLawValidation,CrossNComparison,ScalingReport | F:scan_scaling_results,parse_scaling_run,validate_scaling_law,detect_anomalies,build_cross_n_comparison,generate_report+2 | K:ROOT,DEFAULT_SCALING_DIR
- `ph/analysis/scaling/scaling_extensions_analyzer.py` — Scaling Extensions Analyzer — processes E5 results. — C:BondDimResult,VQEConvergenceResult,HEComparisonResult,NLCEValidationResult,ScalingExtensionsSummary | F:scan_e5_results,parse_bond_dim,parse_vqe_convergence,parse_he_comparison,parse_nlce_result,analyze_e5_results+6 | K:ROOT,DEFAULT_RESULTS_DIR

### `ph/analysis/thesis/` — Thesis compilation tools (tables, figures, summaries).

- `ph/analysis/thesis/heisenberg_summary.py` — Heisenberg XXZ experiment summary and cross-N comparison. — F:find_heisenberg_folders,enrich_with_heisenberg_data,print_summary,print_scaling_comparison,export_json,main | K:ROOT,RESULTS
- `ph/analysis/thesis/thesis_figures.py` — Thesis-level global figures — cross-experiment, aggregated, publication-ready. — C:FigureConfig | F:register_thesis_figure,fig_global_de_gap_distribution,fig_scaling_law_comprehensive,fig_topology_performance_violin,fig_pea_vs_gf_comparison,fig_gnn_qem_summary_panel+17 | K:ROOT,RESULTS_DIR,DEFAULT_OUTPUT_DIR,THESIS_RC
- `ph/analysis/thesis/thesis_tables_compiler.py` — Thesis Tables Compiler — auto-generates global thesis tables from live data. — C:TableSpec,TablesReport | F:register_table,compile_tables,parse_args,main | K:ROOT,RESULTS_DIR

### `ph/analysis/validation/` — Validation and verification tools.

- `ph/analysis/validation/affine_overshoot_auditor.py` — Audit Affine Overshoot Frequency — Close coverage gap G8. — F:scan_experiment_file,main | K:ZNE_EXPERIMENT_DIRS
- `ph/analysis/validation/audit_findings.py` — Deep audit of key thesis findings against raw data. — F:audit_f2_pea_zne,audit_f3_scaling_law,audit_f4_gnn_qem,audit_f5_cross_n,audit_f8_pea_triangular,audit_f9_gnn_not_composable+24 | K:RESULTS,DOCS,ANALYSIS
- `ph/analysis/validation/sanity_check.py` — Sanity check for analysis results — validates data integrity and physics. — C:CheckResult,SanityReport | F:register_check,check_theta_trajectories_exist,check_pca_results_exist,check_derivative_results_exist,check_theta_pca_physics,check_theta_derivative_physics+13 | K:ROOT,RAW_DATA_DIR
- `ph/analysis/validation/thesis_findings_validator.py` — Thesis Findings Validator — corroborates all key findings against raw data. — C:EvidenceStrength,StatisticalEvidence,FindingValidation,ValidationReport | F:register_finding,run_validation,parse_args,main | K:ROOT,RESULTS_DIR
- `ph/analysis/validation/validate_s_series.py` — Validation analyses for S-series experiment results. — F:validation_1_s1_a3_consistency,validation_2_cft_scaling,validation_3_s4_extra_seeds,validation_4_s6_bootstrap,validation_5_s2_vs_g5,main | K:ROOT
- `ph/analysis/validation/verify_results.py` — Post-verification analysis — checks pipeline results against defined specs. — C:VerificationResult,GroupConclusion,VerificationReport | F:parse_pass_criteria,evaluate_criteria,classify_de_gap,scan_results_directory,analyze_verification,format_report_text+2 | K:ROOT,DEFAULT_RESULTS_DIR

### `ph/analysis/` — Project health analysis tools — organized by domain.

- `ph/analysis/diagnose.py` — Automated failure diagnosis for GNN-HVA pipeline results. — C:RootCause,DeploymentPoint,Diagnosis | F:parse_pipeline_run,classify_root_causes,scan_folder,scan_all_thesis,report_diagnoses,main | K:ROOT,RESULTS,THESIS,THETA_SMOOTHNESS_CHAIN_BREAK
- `ph/analysis/gnn_qem_analyzer.py`
- `ph/analysis/layout_optimizer_analyzer.py` — Backward-compatible re-export (moved to hardware subpackage).
- `ph/analysis/mitigation_benchmark_analyzer.py` — Backward-compatible re-export.
- `ph/analysis/mitiq_analyzer.py`
- `ph/analysis/noiseless_model_comparison.py` — Noiseless Model Comparison — extracts structured metrics from per-h analysis rep — C:PerHPoint,SectionMetrics,RunResult,ModelSummary,TopologySummary+2 | F:parse_markdown_report,build_model_summaries,build_topology_summaries,build_comparison_report,format_summary,format_table_markdown+5 | K:ROOT,RE_HEADER,RE_FILE,RE_ELAPSED
- `ph/analysis/noiseless_pipeline_analyzer.py` — Noiseless Pipeline Analyzer — processes exact statevector experiment results. — C:NoiselessRunSummary,TopologyComparison,NoiselessReport | F:scan_noiseless_results,parse_run,detect_anomalies,build_comparisons,generate_report,format_report_text+3 | K:ROOT,DEFAULT_RESULTS_DIR
- `ph/analysis/post_execution_validator.py` — Alias: project_health.analysis.post_execution_validator -> .hardware.post_execut
- `ph/analysis/sanity_check.py` — Backward-compatible re-export (moved to validation subpackage).
- `ph/analysis/statistical_tests.py` — Reusable statistical tests for experiment validation. — F:paired_ttest,improvement_rate,effect_size_cohens_d

### `ph/cli/` — CLI tools for ad-hoc analysis and comparison.

- `ph/cli/analyze_data_quality.py` — Data Quality Analyzer — Index health check and garbage detection. — F:classify_entry,analyze_index_quality,print_report,main | K:ROOT,GARBAGE_EXPERIMENT_IDS
- `ph/cli/compare.py` — Cross-experiment and pipeline result comparison CLI. — F:parse_args,main
- `ph/cli/inspect_noiseless_run.py` — Inspect a noiseless pipeline result JSON — per-h-point breakdown. — F:find_latest_run,load_run,extract_config,extract_section_summary,extract_deploy_perpoint,extract_vqe_summary+6 | K:ROOT,RESULTS_DIR
- `ph/cli/inspect_results.py` — Inspect benchmark results — transpiled circuit properties & pre-submission audit — F:find_result_file,load_envelope,format_value,build_config_report,compute_averages,print_report+2 | K:DEFAULT_CONFIGS,DEFAULT_H_VALUES,DEFAULT_SEED,DEFAULT_MODE
- `ph/cli/qpu_time_estimator.py` — QPU time estimation per circuit using actual transpiled circuit properties. — F:compute_effective_clops,estimate_circuit_qpu_time,parse_args,main | K:BASE_CLOPS,REF_DEPTH,DEPTH_EXPONENT,SHOTS
- `ph/cli/query_index.py` — Query the result index for fast experiment discovery. — F:parse_args,main | K:ROOT

### `ph/core/` — Core engine modules for project health checker.
Exports: HealthReport, ActionItem, CoverageGap, Priority

- `ph/core/coverage.py` — Coverage gap detection and advanced metric computation. — F:detect_coverage_gaps,derive_actions,compute_noiseless_stats,compute_noiseless_by_topology,compute_noisy_stats,compute_vqe_quality+4
- `ph/core/engine.py` — Core health check engine — orchestrates scan, analysis, and reporting. — F:run_health_check
- `ph/core/models.py` — Data models for the project health report. — C:Priority,GapType,CoverageGap,ActionItem,ExperimentSummary+6
- `ph/core/reporter.py` — Output formatters for the health report. — F:generate_timestamped_filename,format_text,format_json,format_markdown
- `ph/core/state.py` — Persistence layer for tracking "new since last run" delta. — F:load_previous_state,save_current_state,detect_delta,detect_new_results,detect_removed_results | K:DEFAULT_STATE_FILE

### `ph/digest/` — Result digest package — lightweight, no heavy dependencies.
Exports: CrossTopologyResult, ExperimentResult, ModeComparisonResult, N120SweepResult, NoiselessResult, NoisyResult, ResultScanner, ScalingResult

- `ph/digest/__main__.py` — CLI entry point for the digest tool. — F:apply_filters,parse_args,main | K:SORT_KEYS_NOISELESS,SORT_KEYS_NOISY,SORT_KEYS_EXPERIMENT
- `ph/digest/formatters.py` — Formatters for the result digest system. — F:format_noiseless_text,format_noisy_text,format_experiment_text,format_noiseless_grouped,format_noisy_grouped,format_markdown+6
- `ph/digest/models.py` — Data models for the result digest system. — C:NoiselessResult,NoisyResult,ExperimentResult,ScalingResult,ModeComparisonResult+2
- `ph/digest/scanner.py` — Result scanner — discovers and parses all result files into typed objects. — C:ResultScanner

## Experiments (experiments/)

### `experiments/` — Experiment scripts for the GNN-HVA framework.


### `exp/generalization/` — Generalization experiments: cross-model and cross-regime tests.

- `exp/generalization/exp_comparative_analysis.py` — CA1: Comparative Analysis — TFIM vs Heisenberg Pipeline. — C:ComparativeAnalysisExperiment
- `exp/generalization/exp_e4_longitudinal.py` — E4: TFIM with Longitudinal Field (2D Phase Diagram). — C:ExperimentE4 | F:build_tfim_longitudinal,exact_diag_sparse
- `exp/generalization/exp_e4b_longitudinal_hva_extended.py` — E4b: TFIM + Longitudinal Field with Extended HVA (RZ layer). — C:ExperimentE4b | K:G_VALUES,H_VALUES,VQE_RESTARTS,VQE_MAXITER
- `exp/generalization/exp_e4c_frustrated_tfim.py` — E4c: Frustrated TFIM (J1-J2) with NNN HVA. — C:ExperimentE4c | K:J2_VALUES,H_VALUES,VQE_RESTARTS,VQE_MAXITER
- `exp/generalization/exp_regime_discovery.py` — RD1: Heisenberg XXZ Regime Discovery. — C:RegimeDiscoveryExperiment

### `exp/hardware/` — Hardware and noisy simulation experiments.


### `exp/helpers/` — Reusable technique modules for experiments.

- `exp/helpers/active_learning.py` — E3: Active Learning for Optimal h-Grid Selection. — F:compute_ensemble_uncertainty,max_variance_acquisition,expected_improvement_acquisition,select_next_point,should_stop
- `exp/helpers/analytical_init.py` — B1: Analytical Initial Guess from Perturbation Theory. — F:analytical_init_p1,analytical_init_p2,validate_analytical_init
- `exp/helpers/dypp.py` — F1: Dynamic Parameter Prediction (DyPP) for VQE Acceleration. — F:dypp_linear,dypp_quadratic,dypp_predict,evaluate_dypp_quality
- `exp/helpers/graph_utils.py` — Shared graph construction utilities for experiments. — F:build_experiment_dataset,predict_theta,predict_theta_batch
- `exp/helpers/hessian_restart.py` — B4: Hessian-Guided Adaptive Restart Strategy. — F:hessian_guided_vqe,standard_multistart_vqe
- `exp/helpers/parameter_freezing.py` — B2: TITAN-Style Parameter Freezing for HVA. — F:analyze_parameter_activity,frozen_vqe
- `exp/helpers/physics_loss.py` — C1: Physics-Informed MPNN Loss. — C:PhysicsInformedLoss | F:evaluate_energy_batch,select_eval_subset
- `exp/helpers/sign_equivariant.py` — C3: Sign-Equivariant MPNN for Z2 Symmetry. — C:SignInvariantLoss | F:canonicalize_sign,canonicalize_dataset,detect_sign_inconsistency

### `exp/landscape/` — Landscape analysis experiments: energy landscape characterization.

- `exp/landscape/exp_f1_dypp.py` — F1: Dynamic Parameter Prediction (DyPP) for VQE Acceleration. — C:ExperimentF1
- `exp/landscape/exp_f3_fluctuation.py` — F3: Landscape Fluctuation Analysis for Valid Regime Prediction. — C:ExperimentF3
- `exp/landscape/exp_s3_landscape_n20.py` — S3: Landscape Analysis at N=20 (F3 + B4 Extension). — C:ExperimentS3

### `exp/optimization/` — Optimization experiments: VQE techniques and parameter strategies.

- `exp/optimization/exp_b1_analytical.py` — B1: Analytical Initial Guess from Perturbation Theory. — C:ExperimentB1
- `exp/optimization/exp_b2_freezing.py` — B2: TITAN-Style Parameter Freezing for HVA p=2. — C:ExperimentB2
- `exp/optimization/exp_b4_hessian.py` — B4: Hessian-Guided Adaptive Restarts. — C:ExperimentB4
- `exp/optimization/exp_c3_sign.py` — C3: Sign Canonicalization for p=1 N=20 MPNN Deployment. — C:ExperimentC3
- `exp/optimization/exp_g4_condition_restarts.py` — G4: Condition Number vs Required Restarts. — C:ExperimentG4

### `exp/predictor/` — Predictor experiments: MPNN training and deployment strategies.

- `exp/predictor/exp_c1_physics_loss.py` — C1: Physics-Informed MPNN Loss. — C:ExperimentC1
- `exp/predictor/exp_d1_weight_space.py` — D1: Unsupervised Phase Detection from MPNN Weight Space. — C:ExperimentD1
- `exp/predictor/exp_e3_active.py` — E3: Active Learning for Optimal h-Grid Selection. — C:ExperimentE3
- `exp/predictor/exp_g1_data_efficiency.py` — G1: Data Efficiency Curve — Minimum VQE points for MPNN deployment. — C:ExperimentG1
- `exp/predictor/exp_g2_ensemble_calibration.py` — G2: MPNN Ensemble Uncertainty Calibration. — C:ExperimentG2
- `exp/predictor/exp_g5_cross_seed.py` — G5: Cross-Seed Generalization. — C:ExperimentG5
- `exp/predictor/exp_s2_cross_topology.py` — S2: Cross-Topology Transfer Learning. — C:ExperimentS2
- `exp/predictor/exp_s4_data_efficiency_n10.py` — S4: Data Efficiency at N=10 (Extension of G1). — C:ExperimentS4 | K:FULL_H_GRID,K_VALUES
- `exp/predictor/exp_s6_mc_dropout_uq.py` — S6: MC-Dropout Uncertainty Quantification (Fix for G2). — C:ExperimentS6 | K:N_MC_SAMPLES

### `exp/scaling/` — Scaling experiments: finite-size scaling laws and system-size studies.

- `exp/scaling/exp_a3_scaling_law.py` — A3: Finite-Size Scaling of the Valid Regime Boundary. — C:ExperimentA3
- `exp/scaling/exp_g3_n20_optimized.py` — G3: N=20 p=2 Optimized Pipeline (Capstone Experiment). — C:ExperimentG3
- `exp/scaling/exp_s1_entanglement_scaling.py` — S1: Entanglement Entropy vs Valid Regime Boundary. — C:ExperimentS1 | F:compute_entanglement_entropy
- `exp/scaling/exp_s5_n20_p1_pipeline.py` — S5: Full Pipeline N=20 p=1 with MPNN (not interpolation). — C:ExperimentS5 | K:H_TRAIN,H_TEST
- `exp/scaling/exp_s8_d1_finite_size_scaling.py` — S8: Finite-Size Scaling of h_c via Weight-Space Phase Detection (D1). — C:ExperimentS8 | F:fss_model | K:N_VALUES,H_TRAIN,H_PROBE,EPSILON
- `exp/scaling/exp_s8b_mpnn_finite_size_scaling.py` — S8b: Finite-Size Scaling of h_c via MPNN Weight-Space Gradients. — C:ExperimentS8b | F:fss_model | K:N_VALUES,H_TRAIN,H_PROBE,MPNN_EPOCHS

## Runners (scripts/experiment_runners/)

### `s/experiment_runners/cross_topology/` — Cross-topology transfer experiment runners.

- `s/experiment_runners/cross_topology/helpers.py` — Shared utilities for cross-topology transfer experiments. — C:SourceData,ValidationReport,MLPBaseline | F:detect_format,load_source_data,filter_source_data,load_source_data_filtered,validate_training_data,validate_predictions_sanity+9
- `s/experiment_runners/cross_topology/run_ablation.py` — Ablation study: GNN vs MLP vs Scipy + BatchNorm comparison. — F:get_target_h_values,compute_spearman_correlation,evaluate_gnn_predictor,evaluate_mlp_predictor,evaluate_scipy_predictor,run_predictor_comparison+4 | K:DEFAULT_SOURCE_DIRS,NORM_TYPES
- `s/experiment_runners/cross_topology/run_cross_n_validation.py` — Within-topology cross-N validation for triangular and heavy_hex. — F:get_test_h_values,find_source_file,run_single_seed,run_topology_validation,main
- `s/experiment_runners/cross_topology/run_cross_topology.py` — Cross-topology transfer experiment: train on topology A, predict topology B. — F:get_target_h_values,run_single_direction,aggregate_seed_results,main | K:DEFAULT_SOURCE_DIRS
- `s/experiment_runners/cross_topology/run_cross_topology_noisy.py` — Cross-Topology Transfer + Noisy PEA Validation. — C:CrossTopologyNoisyRunner | K:NOISE_FACTORS,ZNE_SHOTS,SEED,N_CANDIDATE_LAYOUTS
- `s/experiment_runners/cross_topology/run_orchestrator.py` — Full experiment orchestration for cross-topology transfer pipeline. — F:run_step,check_data_exists,find_result_files,check_budget,run_orchestrator,main | K:TOTAL_BUDGET_S,BUDGET_WARNING_THRESHOLD
- `s/experiment_runners/cross_topology/run_vqe_data_gen.py` — VQE data generation for cross-topology transfer experiments. — F:get_h_values,run_vqe_sweep,get_output_path,get_expected_path_pattern,main

### Standalone scripts

- `s/experiment_runners/aqc_tensor/run_aqc_cross_topology.py` — AQC-Tensor Cross-Topology Validation — Phase 3. — F:run_single_compression,main | K:ROOT,N_QUBITS,P_LAYERS_TARGET,BOND_DIM
- `s/experiment_runners/aqc_tensor/run_aqc_poc.py` — AQC-Tensor Proof of Concept — Phase 1 Validation. — F:run_poc,main | K:ROOT
- `s/experiment_runners/aqc_tensor/run_aqc_vs_direct.py` — AQC-Tensor vs Direct p=1 Comparison — Quantify expressibility benefit. — F:run_comparison,main | K:ROOT
- `s/experiment_runners/bond_resolved/run_bond_resolved_cross_n.py` — B4: Bond-Resolved Cross-N GNN Necessity Proof. — C:BondResolvedCrossNRunner | K:N_QUBITS,P_LAYERS,TOPOLOGY,SEED
- `s/experiment_runners/bond_resolved/run_bond_resolved_cross_n_transfer.py` — B5: Bond-Resolved Cross-N Transfer — Train N=40, Predict N=60/80. — C:BondResolvedCrossNTransferRunner | K:TRAIN_N,P_LAYERS,TOPOLOGY,SEED
- `s/experiment_runners/bond_resolved/run_bond_resolved_scaling.py` — Bond-Resolved HVA Scaling Suite — 2D Geometry + MPNN + Noisy Simulation. — C:BondResolvedScalingRunner
- `s/experiment_runners/bond_resolved/run_bond_resolved_validation.py` — Bond-Resolved HVA Validation Suite. — C:BondResolvedValidationRunner
- `s/experiment_runners/bond_resolved/run_e3_bond_resolved_scaling.py` — E3: Bond-Resolved HVA at N=40 — Scaling Validation. — C:E3BondResolvedScalingRunner | K:DEFAULT_N_QUBITS,DEFAULT_TOPOLOGY,P_LAYERS,SEED
- `s/experiment_runners/bond_resolved/run_n16_square_dmrg2d.py` — N=16 Square (4x4) Bond-Resolved HVA with DMRG 2D ground truth. — C:N16SquareDMRG2DRunner
- `s/experiment_runners/bond_resolved/run_scaling_extensions.py` — Scaling Extensions Suite — N=120 Bond Dimension + VQE + HE Comparison + NLCE. — C:ScalingExtensionsRunner | F:analytical_theta_x | K:N_BOND_DIM,CHI_VALUES,STRATEGY,SEED
- `s/experiment_runners/gnn_exp/run_gnn_qem_ablation_no_enoisy.py` — GNN-QEM Ablation T1 — Remove E_noisy from context vector. — C:MLPContextOnly | F:augment,zero_out_enoisy_in_dataset,evaluate_model,train_variant,linear_no_enoisy,main
- `s/experiment_runners/gnn_exp/run_gnn_qem_ablation_suite.py` — GNN-QEM Ablation Suite — Validate that the GNN learns from graph structure. — C:MLPContextOnly | F:augment,evaluate_on_test,train_and_eval,run_v1_mlp_ablation,run_v2_shuffled_edges,run_v3_multi_seed+2
- `s/experiment_runners/gnn_exp/run_gnn_qem_cross_topology.py` — GNN-QEM Cross-Topology Validation — Test generalization to heavy_hex N=10. — F:main
- `s/experiment_runners/gnn_exp/run_gnn_qem_post_zne_validation.py` — GNN-QEM Post-ZNE Validation — Does GNN-QEM help AFTER PEA-ZNE? — F:run_vqe_sweep,main | K:TOPOLOGY,N_QUBITS,P_LAYERS,H_VALUES
- `s/experiment_runners/gnn_exp/run_gnn_qem_training.py` — GNN-QEM Training Pipeline — Generate data, train, evaluate, save results. — F:main
- `s/experiment_runners/gnn_exp/run_gnn_qem_vqe_realistic.py` — GNN-QEM with VQE-Optimized θ — Realistic Error Regime + Circuit Selection Mode. — F:generate_vqe_data,augment,zero_enoisy_in_dataset,evaluate,main | K:TRAIN_TOPOLOGIES,TEST_TOPOLOGY,N_QUBITS_TRAIN,N_QUBITS_TEST
- `s/experiment_runners/hardware/_sanity_check_envelope.py` — Quick sanity check: import the persistence module and verify functions.
- `s/experiment_runners/hardware/benchmark_configs.py` — Benchmark configuration registry for mitigation experiments (C0-C18). — C:BenchmarkConfig | K:VALID_DD_SEQUENCES,MITIQ_METHODS
- `s/experiment_runners/hardware/run_full_deployment_pipeline.py` — Full deployment pipeline: V3 rehearsal → IBM Torino deployment. — F:find_latest_rehearsal_json,main | K:REHEARSAL_DIR,SCRIPT_DIR
- `s/experiment_runners/hardware/run_hardware_mitigation_flow.py` — Hardware Mitigation Flow — Orchestrated 3-step deployment. — F:log_step,run_command,check_credentials,check_dependencies,parse_smoke_results,step_0_preflight+7 | K:ROOT,PYTHON,BENCHMARK_SCRIPT,REHEARSAL_SCRIPT
- `s/experiment_runners/hardware/run_ibm_deployment.py` — IBM Torino QPU Deployment — Tiered Execution with Calibration-First Strategy. — C:TierMetrics | F:check_credentials,load_sigma_flow_from_rehearsal,build_hardware_config,recompute_budget_from_measurement,prepare_mpnn_predictions,prepare_aqc_compressed_circuit+7 | K:TOPOLOGY,N_QUBITS,P_LAYERS,MODEL
- `s/experiment_runners/hardware/run_mitigation_benchmark.py` — Mitigation Benchmark Runner — systematic evaluation of 19 configs. — F:append_to_manifest,compute_derived_circuit_stats,apply_affine_on_raw,route_execution,parse_args,resolve_configs+4 | K:BENCHMARK_VERSION,RESULTS_BASE,MANIFEST_PATH
- `s/experiment_runners/noise_zne_gf_pea/run_gf_zne_batch.py` — Batch runner: Gate-Folding ZNE comparison across topologies. — C:RunConfig | F:run_single_config,run_comparison_analysis,parse_args,main | K:ROOT,COMPARISON_SCRIPT,PYTHON,RESULTS_DIR
- `s/experiment_runners/noise_zne_gf_pea/run_gf_zne_comparison.py` — Gate-Folding ZNE vs CES-ZNE — Direct Comparison Experiment. — C:GFZNEComparisonRunner | K:DEFAULT_TOPOLOGY,DEFAULT_N_QUBITS,DEFAULT_P_LAYERS,SEEDS
- `s/experiment_runners/noise_zne_gf_pea/run_pea_cross_topology_dense.py` — PEA Cross-Topology Dense Validation — 5 seeds × 6 h-points × 4 topologies. — C:PEACrossTopologyDenseRunner | K:NOISE_FACTORS,ZNE_SHOTS,SEEDS,N_CANDIDATE_LAYOUTS
- `s/experiment_runners/noise_zne_gf_pea/run_pea_full_pipeline.py` — PEA-ZNE Full Pipeline Validation — MPNN Predict + Mitigate + Classify. — C:PEAFullPipelineRunner | K:TOPOLOGY,N_QUBITS,P_LAYERS,H_TRAIN
- `s/experiment_runners/noise_zne_gf_pea/run_pea_hardware_readiness.py` — PEA-ZNE Hardware Readiness — Scalability & Realism Assessment. — C:PEAHardwareReadinessRunner | K:DEFAULT_TOPOLOGY,DEFAULT_N_QUBITS,DEFAULT_P_LAYERS,H_TEST_HARDWARE
- `s/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py` — PEA-ZNE at N=40/50 — Validates PEA extrapolation at hardware scale. — C:PEAScalingRunner | K:NOISE_FACTORS,ZNE_SHOTS,SEED,MPS_CHI
- `s/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py` — PEA-ZNE Triangular Validation — Close coverage gap G6. — C:PEATriangularRunner | K:NOISE_FACTORS,ZNE_SHOTS,N_CANDIDATE_LAYOUTS,DEFAULT_TOPOLOGY
- `s/experiment_runners/noise_zne_gf_pea/run_pea_zne_validation.py` — PEA-ZNE Comprehensive Validation — Multi-topology, Multi-seed. — C:PEAZNEValidationRunner | K:DEFAULT_TOPOLOGY,DEFAULT_N_QUBITS,DEFAULT_P_LAYERS,SEEDS
- `s/experiment_runners/noise_zne_gf_pea/run_zne_3way_comparison.py` — 3-Way ZNE Comparison: CES-ZNE vs Gate-Folding vs PEA. — C:ZNE3WayComparisonRunner | K:DEFAULT_TOPOLOGY,DEFAULT_N_QUBITS,DEFAULT_P_LAYERS,H_TEST_VALUES
- `s/experiment_runners/noise_zne_gf_pea/run_zne_cross_topology_validation.py` — ZNE Cross-Topology Validation — Definitive PEA vs GF-ZNE Comparison. — C:ZNECrossTopologyRunner | K:NOISE_FACTORS,ZNE_SHOTS,N_CANDIDATE_LAYOUTS,CONFIGS
- `s/experiment_runners/noiseless/run_mc_dropout_uq.py` — MC-Dropout Uncertainty Quantification — Robust Version. — C:MCDropoutRunner | K:MC_CONFIGS
- `s/experiment_runners/noiseless/run_noiseless_cross_n.py` — Noiseless Cross-N Generalization — MPNN predicts θ for unseen system sizes. — C:NoiselessCrossNRunner | K:DEFAULT_TRAIN_SIZES,DEFAULT_TARGET_N,DEFAULT_P,DEFAULT_TOPOLOGY
- `s/experiment_runners/noiseless/run_noiseless_pipeline.py` — Noiseless Pipeline Runner — Exact statevector experiments (N ≤ 22). — C:NoiselessPipelineRunner | K:DEFAULT_N,DEFAULT_P,DEFAULT_TOPOLOGY,DEFAULT_MODEL
- `s/experiment_runners/scaling/run_cross_n_warmstart_eval.py` — Cross-N Warm-Start Evaluation — heavy_hex p=1 scaling to N=100. — C:CrossNWarmstartEvalRunner | K:DEFAULT_TRAIN_SIZES,DEFAULT_TARGET_N,DEFAULT_TOPOLOGY,DEFAULT_MODEL
- `s/experiment_runners/scaling/run_scaling_phase3_mpnn.py` — MPS Scaling Phase 3+4 — MPNN Training + Deployment at N>30. — C:MPSScalingPhase3Runner | F:canonicalize_theta,load_theta_from_result | K:DEFAULT_HIDDEN_DIM,DEFAULT_N_LAYERS,DEFAULT_N_EPOCHS,DEFAULT_PATIENCE
- `s/experiment_runners/scaling/run_scaling_validation.py` — MPS Scaling Validation — DMRG ground truth + MPS-VQE at N>30. — C:MPSScalingValidationRunner | K:DEFAULT_N,DEFAULT_TOPOLOGY,DEFAULT_MODEL,DEFAULT_P
- `s/experiment_runners/t1_exp/run_t1a_dense_j2.py` — Tier 1A Dense: MPNN Generalization Across 8 J₂ Values (Dense Grid). — C:DenseJ2Runner | K:N_QUBITS,P_LAYERS,TOPOLOGY,SEEDS
- `s/experiment_runners/t1_exp/run_t1a_mpnn_2d_predictor.py` — Tier 1A: MPNN Generalization Across J₂ Values (2D Predictor). — C:MPNN2DPredictorRunner | K:N_QUBITS,P_LAYERS,TOPOLOGY,SEEDS
- `s/experiment_runners/t1_exp/run_t1b_longitudinal_zne.py` — Tier 1B: Noisy Simulation with FakeTorino — TFIM+longitudinal, p=1, ZNE. — C:LongitudinalZNERunner | K:N_QUBITS,P_LAYERS,TOPOLOGY,G_DEFAULT
- `s/experiment_runners/t1_exp/run_t1c_d1_frustrated.py` — Tier 1C: Weight-Space Phase Detection (D1) for Frustrated TFIM. — C:D1FrustratedRunner | K:N_QUBITS,P_LAYERS,TOPOLOGY,SEEDS
- `s/experiment_runners/topology_exp/run_thesis_variants-chain_1d.py` — Exhaustive pipeline variant runner for thesis validation. — F:build_noiseless_variants,build_extended_variants,build_noisy_variants | K:DEFAULT_N_QUBITS
- `s/experiment_runners/topology_exp/run_thesis_variants-heavy_hex.py` — Exhaustive pipeline variant runner — N=10, p=2, HEAVY-HEX topology. — F:build_noiseless_variants,build_noisy_variants,build_extended_variants,main | K:DEFAULT_N_QUBITS,P_LAYERS,TOPOLOGY,DEFAULT_SEED
- `s/experiment_runners/topology_exp/run_thesis_variants-heisenberg.py` — Exhaustive pipeline variant runner — Heisenberg XXZ model experiments. — F:build_noiseless_variants,build_noisy_variants,build_extended_variants,main | K:DEFAULT_N_QUBITS,P_LAYERS,TOPOLOGY,DEFAULT_SEED
- `s/experiment_runners/topology_exp/run_thesis_variants-ladder.py` — Exhaustive pipeline variant runner — N=10, p=2, LADDER topology. — F:build_noiseless_variants,build_noisy_variants,build_extended_variants | K:DEFAULT_N_QUBITS,P_LAYERS,TOPOLOGY,BASE_H_VALUES
- `s/experiment_runners/topology_exp/run_thesis_variants-triangular.py` — Exhaustive pipeline variant runner — N=10, p=2, TRIANGULAR topology. — F:build_noiseless_variants,build_noisy_variants,build_extended_variants | K:DEFAULT_N_QUBITS,P_LAYERS,TOPOLOGY,DEFAULT_SEED
- `s/experiment_runners/transpiler_exploration/check_fair_comparison.py` — Fair comparison: same CES-selected layout, compare circuit representations. — K:H
- `s/experiment_runners/transpiler_exploration/run_transpiler_comparison.py` — Transpiler & Noise Suppression Exploration — Pre-Hardware Validation. — C:TranspilerResult,TranspilerExplorationRunner | K:N_QUBITS,P_LAYERS,TOPOLOGY,H_TEST
- `s/experiment_runners/analyse_thesis_extensions.py` — Analyse and print a structured report of thesis extension results. — F:wrap | K:ROOT
- `s/experiment_runners/run_ext1_intra_n_p1.py` — Ext1b: Standalone p=1 revalidation for CONDITIONALLY_VIABLE h-points. — C:Ext1bP1ValidationRunner | K:CONDITIONALLY_VIABLE
- `s/experiment_runners/run_frustrated_tfim_validation.py` — Frustrated TFIM Full Execution Suite. — F:section_1,section_2,section_3,section_4,section_5,section_6+2
- `s/experiment_runners/run_hardware_rehearsal_v2.py` — Hardware Deployment Rehearsal V2 — PEA/GF/Adaptive ZNE on FakeKingston. — C:HardwareRehearsalV2 | K:DEFAULT_TOPOLOGY,DEFAULT_N_QUBITS,DEFAULT_P_LAYERS,DEFAULT_MODEL
- `s/experiment_runners/run_hardware_rehearsal_v3.py` — Hardware Deployment Rehearsal V3 — V2 + MPNN Evaluation Suite. — C:HardwareRehearsalV3 | K:SPEEDUP_THRESHOLD,DEFAULT_N_VQE_BENCH_RESTARTS,DEFAULT_MAXITER_REFINE,LOO_PASS_RATE_THRESHOLD
- `s/experiment_runners/run_mps_pseudo_hardware.py` — MPS Pseudo-Hardware Validation — Tensor Network Noise Proxy. — C:MPSPseudoHardwareRunner | K:SEEDS,P_LAYERS,VQE_RESTARTS,VQE_MAXITER
- `s/experiment_runners/run_p1_pipeline_variants_r2.py` — p=1 pipeline variant runner — Round 2 (corrected h_test + complementary). — F:build_noiseless_variants,build_noisy_variants,build_extended_variants,main | K:DEFAULT_N_QUBITS,SEEDS,PIPELINE_SCRIPT
- `s/experiment_runners/run_verification_plan.py` — Verification Plan — Systematic validation of thesis findings. — F:build_tier1_variants,build_tier2_variants,build_tier3_variants,main | K:PIPELINE_SCRIPT,OUTPUT_BASE,SEEDS,P1_VALID_REGIME

## Scripts (scripts/)

### Standalone scripts

- `s/analysis/analyze_all_phase3.py` — Analyze Phase3 MPNN scaling results with full diagnostics. — F:h_min_scaling_law,parse_args,load_phase3_result,find_results,detect_anomalies,validate_scaling_law+5 | K:ROOT,DEFAULT_DIR,SCALING_LAW_OFFSET,SCALING_LAW_COEFF
- `s/analysis/check_delta_e_by_topo_p.py` — Check how |ΔE| (absolute, NOT divided by gap) behaves for harder topologies — F:extract_deploy_points,get_config,main | K:BASE,RESULTS_DIR
- `s/analysis/check_matrix_gaps.py` — Check what (N, p) combinations have h_expr data and identify gaps. — F:parse_args,main | K:ROOT,DEFAULT_BASE,DEFAULT_TARGET_NS,DEFAULT_TARGET_PS
- `s/analysis/compute_h_frontier.py` — Compute precise h_frontier values via linear interpolation. — F:parse_args,interpolate_frontier,main | K:ROOT,DEFAULT_BASE,THRESHOLD
- `s/analysis/compute_h_frontier_all.py` — Compute h_frontier matrix across ALL topologies and models. — F:parse_args,interpolate_frontier,scan_results,print_matrix,main | K:ROOT,THRESHOLD
- `s/analysis/compute_h_frontier_models.py` — Compute precise h_frontier via linear interpolation for model exploration. — F:compute_frontier | K:ROOT,RESULTS_DIR,DE_GAP_THRESHOLD,H_CRITICAL
- `s/analysis/compute_h_frontier_topologies.py` — Compute h_frontier for TFIM_longitudinal across all topologies, N=10, P=2..8. — F:compute_frontier | K:ROOT,RESULTS_DIR,DE_GAP_THRESHOLD
- `s/analysis/extract_delta_e_fidelity.py` — Extract ΔE (absolute) and Fidelity metrics from noiseless pipeline runs. — C:PointMetrics,RunSummary | F:extract_from_run,scan_all_runs,scan_cross_n,format_table,main
- `s/analysis/extract_theta_trajectories.py` — Task 2.1: Extract θ_opt(h) trajectories from ALL pipeline/scaling results. — F:extract_trajectory,scan_results,scan_scaling_results,filter_best_trajectories,main | K:ROOT,RESULTS_DIR,SCALING_DIR,OUTPUT_FILE
- `s/analysis/reanalyze_p2_filtered.py` — Re-analyze p=1-4 with h >= 1.3 filter (same regime as definitive runs).
- `s/analysis/theta_derivative_analysis.py` — Task 3: ∂θ/∂h Derivative vs D1 Weight Gradient Comparison. — F:load_d1_gradients,compute_theta_derivative,compute_correlation,generate_figure,parse_args,main | K:ROOT,THETA_FILE,D1_DIR,OUTPUT_RESULTS
- `s/analysis/theta_pca_phase_detection.py` — Task 2.2: PCA and clustering analysis of θ_opt(h) for unsupervised phase detecti — F:load_trajectories,analyze_trajectory,generate_figure,parse_args,run_scaling_analysis,main | K:ROOT,INPUT_FILE,OUTPUT_RESULTS,FIGURES_DIR
- `s/analysis/verify_hva_periodicity.py` — Verify HVA circuit parameter periodicities numerically. — K:H
- `s/benchmarks/benchmark.py` — Performance benchmarking CLI for qmbp_simulation. — F:parse_args,main
- `s/benchmarks/benchmark_pea_parallel.py` — Benchmark PEA parallel vs sequential noise factor execution. — F:benchmark_measurement_only,benchmark_full_pea,main
- `s/benchmarks/benchmark_pea_performance.py` — Benchmark PEA-ZNE performance: measures per-component timing. — F:benchmark_pea,main
- `s/hardware/analyze_qesem_error_detail.py` — Detailed per-observable error comparison: noisy vs QESEM mitigated. — K:RESULT_PATH,H
- `s/hardware/analyze_qesem_result.py` — Analyze recovered QESEM result against exact values. — K:H
- `s/hardware/analyze_qesem_tier1.py` — Analyze QESEM Tier-1 result (h=3.5) from recovered JSON. — K:RESULT_PATH,H,H4
- `s/hardware/complete_tier0_from_qpu.py` — Complete Tier 0 result from QPU data that was successfully executed. — K:ROOT,E_ZNE,E_STD,E_EXACT
- `s/hardware/convert_qesem_to_hwresult.py` — Convert recovered QESEM JSON results into HardwareRunResult format. — F:convert_qesem_recovered,main | K:PROJECT,EXPECTED_LABEL,DE_GAP_THRESHOLD
- `s/hardware/estimate_qesem_budget.py` — QESEM QPU Budget Estimator. — F:build_test_circuit,main | K:TOPOLOGY,N_QUBITS,P_LAYERS,H_VALUES
- `s/hardware/hardware.py` — Unified hardware deployment CLI — single entry point for all QPU operations. — F:cmd_cost,cmd_preflight,cmd_rehearsal,cmd_analyze,build_parser,main
- `s/hardware/preflight_hw.py` — Pre-QPU execution preflight check — run BEFORE every hardware deployment. — F:ok,fail,warn | K:ROOT
- `s/hardware/print_tier0_circuit.py` — Print the EXACT circuit used in Tier 0 QPU execution (2026-06-14). — K:ROOT
- `s/hardware/qesem_error_analysis.py` — Análisis de errores reales vs mitigados en ejecuciones QESEM. — K:E_EXACT_H4,GAP_H4
- `s/hardware/recover_job_result.py` — Recover and analyze results from a completed IBM Runtime job. — F:save_raw_job_output,connect_service,retrieve_job,main
- `s/hardware/recover_qesem_job.py` — Recover QESEM job results from IBM Qiskit Functions catalog. — F:main
- `s/hardware/recover_qesem_jobs.py` — Recover results from completed QESEM (Qiskit Functions) jobs. — F:connect_catalog,list_recent_jobs,recover_qesem_job,main
- `s/validation/compare_resource_estimation.py` — Compare & validate Qiskit ResourceEstimation integration. — F:section_1_consistency_check,section_2_cross_topology,section_3_error_budget,section_4_depth_2q_vs_opt_level,section_5_logical_vs_transpiled,main
- `s/validation/test_batch_hw_path.py` — Test the batch hardware path with FakeTorino as mock backend. — K:ROOT
- `s/validation/test_pea_config.py` — Quick end-to-end test of the new PEA configuration in fake_backend mode. — K:H
- `s/validation/validate_gnn_qem_post_zne.py` — Validate GNN-QEM on POST-ZNE residuals (realistic deployment scenario). — F:generate_post_zne_samples,main
- `s/validation/validate_warm_start_e3.py` — Quick validation: warm-start from uniform params vs cold-start failure. — K:N,P,TOPOLOGY,H

## Maintenance (scripts/maintenance/)

### Standalone scripts

- `s/maintenance/audit_pipeline_consistency.py` — Audit the energy correction pipeline for additional potential bugs. — F:section,warn,ok | K:ROOT,N_ISSUES
- `s/maintenance/generate_module_index.py` — Generate a compact module index for .kiro/steering/module-index.md. — C:ModuleEntry,PackageEntry | F:extract_module,extract_package,scan_directory,format_index,main | K:PROJECT_ROOT,DEFAULT_OUTPUT,SCAN_DIRS,SKIP_FILES
- `s/maintenance/generate_presets_from_index.py` — Auto-generate config presets from successful ResultIndex entries. — F:generate_preset_yaml,main | K:ROOT,PRESETS_DIR
- `s/maintenance/md_index.py` — Generate a rich markdown index/TOC from a directory of .md files. — C:Section,FileStats,DocEntry | F:extract_metadata,format_full,format_list,format_table,main
- `s/maintenance/organize_results.py` — Organize experiment results into the hierarchical folder structure. — F:extract_config_from_file,organize_flat_dirs,archive_failed,main | K:ROOT,RESULTS_ROOT,ARCHIVE_ROOT,ARCHIVE_MODELS
- `s/maintenance/scan_new_runs.py` — Scan and analyse noiseless experiment results. — C:RunMetrics,GroupStats | F:resolve_dirs,load_runs,parse_run,group_runs,compute_group_stats,print_header+8 | K:ROOT,DEFAULT_EXPERIMENT_DIRS
- `s/maintenance/update_project_status.py` — Auto-generate .kiro/steering/project-status.md from the result index. — F:main | K:ROOT

## Notebooks

### Standalone scripts

- `nb/test_notebooks.py` — Test all 3 notebooks end-to-end (script mode, no Jupyter required). — F:test_notebook_01,test_notebook_02,test_notebook_03 | K:ROOT
- `nb/viz_helpers.py` — Visualization helpers for demo notebooks. — F:draw_circuit,draw_circuit_stats,draw_hva_structure,draw_lattice,draw_topology_comparison,draw_pipeline_diagram+3
