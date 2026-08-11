# Module Index (auto-generated)

Legend: C=class F=func K=const. Base import: `from qmbp_simulation.<module> import ...`
Run `python scripts/maintenance/generate_module_index.py` to refresh.

## Quick Lookup

| Need | Module | Symbols |
|---|---|---|
| Build Hamiltonian | models.hamiltonian | HamiltonianBuilder, make_lattice |
| HVA circuit | circuits.hva | HVACircuitBuilder |
| VQE optimize | optimizers.vqe | VQEOptimizer |
| Noiseless eval | execution.backends | NoiselessBackend, select_backend |
| MPS eval (N>22) | execution.mps_backend | MPSBackend |
| Cache evals | execution.eval_cache | CachedBackend, EvalCache |
| Ground truth | solvers.classical | ClassicalSolver |
| GT cache (disk) | solvers.ground_truth_cache | GroundTruthCache |
| Train MPNN | predictors.mpnn | MPNNPredictor, train_mpnn |
| UnifiedMPNN (cross-N) | predictors.unified_mpnn | UnifiedMPNN |
| Model zoo | predictors.model_zoo | load_pretrained, register_checkpoint |
| Accelerated pipeline | pipeline.accelerated | AcceleratedVQE |
| Full pipeline | pipeline.runner | PipelineRunner |
| Validate θ | analysis.theta_validator | ThetaValidator |
| Deploy stats | analysis.metrics | compute_deploy_summary |
| θ alignment | analysis.theta_alignment | align_theta_array |
| Noisy ZNE | execution.noisy_utils | run_pea_zne, run_gate_folding_zne |
| Hardware QPU | execution.hardware.backend | HardwareBackend |
| Runner base | framework.runner_base | ValidationRunner, Section |
| Result I/O | framework.result_io | save_experiment_result |
| CLI args | framework.cli | create_base_parser |
| Quality predict | analysis.quality_predictor | QualityPredictor |
| JSON serialize | utils.helpers | json_serialize, json_dump |
| Canonicalize θ | utils.helpers | canonicalize_theta |
| Model spec | models.model_registry | get_model_spec |


## Library (src/qmbp_simulation)

### src/qmbp_simulation/ (0)
  ↳ HamiltonianBuilder make_lattice LatticeConfig GroundTruthResult VQEConfig VQEResult SUPPORTED_TOPOLOGIES MAX_P_LAYERS +20


### qsim/analysis/ (22)
  ↳ AlignmentReport BaselineComparison BaselineMetrics ClusterResult ClusterSolver ComparativeMetrics ComparisonResult DiagnosticCollector +56

  circuit_visualizer         VIS   F:print_circuit,save_circuit_diagram,circuit_summary+11
  comparative                ANAL  C:RegimeDiscoveryResult,ComparativeMetrics | F:find_h_min,classify_result,filter_by_threshold+5
  cross_n_validator          VAL   C:L1Result,L2Result,L3Result+2 | F:preflight_cross_n
  data_models                MODEL C:GradientAnalysisResult,ComparisonResult,BaselineMetrics+1
  diagnostics                CORE  C:DiagnosticCollector | F:configure_pipeline_logging
  entanglement               ANAL  C:EntanglementResult,EntanglementAnalyzer
  extension_analyzer         PIPE  C:RejectionReportGenerator,PrerequisiteChecker,CalibrationComparator+8
  extension_classifiers      ANAL  C:ClassificationEngine
  extension_models           ANAL  C:ExtensionClassification,PrerequisiteFailedError,HardPhysicsLimitError+3
  extension_ranker           ANAL  C:ExtensionScore,ExtensionPriorityRanker
  flow_multishot             PRED  C:MultiShotResult,FlowMultiShotPredictor
  flow_warmstart             OPT   C:FlowWarmstartManager
  gradient                   PRED  C:WeightGradientAnalyzer
  ground_truth_validator     VAL   C:GroundTruthValidationReport,GroundTruthValidator
  landscape                  CORE  F:compute_hessian,landscape_fluctuation
  metrics                    CORE  F:is_point_failure,identify_failures,compute_refinement_priority+22
  nlce                             C:NLCEConfig,ClusterResult,NLCEResult+3 | F:tfim_analytical_energy_per_site,nlce_convergence_analysis
  normalizing_flow                 C:MaskedLinear,MAFLayer,FlowHead+1
  quality_predictor          PRED  C:PredictionReport,QualityPredictor
  theta_alignment            POST  C:AlignmentReport,OutlierReport,EnergyGuardReport | F:detect_jumps,align_theta_sweep,align_theta_array+3
  theta_validator            VAL   C:BoundCheckResult,NumericalSanityResult,InterpolationResult+6
  vqe_validator              VAL   C:Severity,ValidationIssue,VQEValidationReport+1

### qsim/circuits/ (2)
  ↳ HVACircuitBuilder AQCCircuitCompressor AQCCompressionConfig AQCCompressionResult CompressionValidation AQCCompressionCache

  aqc_compression            CIRC  C:AQCCompressionConfig,AQCCompressionResult,CompressionValidation+2
  hva                        MODEL C:HVACircuitBuilder | F:do_checks

### qsim/execution/hardware/ (10)
  ↳ HardwareBackend HardwareConfig HardwareRunResult SPSAConfig MAPOMATIC_AVAILABLE LayoutOptimizationResult build_filtered_coupling_map compute_layout_fidelity_cost +8

  backend                    EXEC  C:HardwareBackend
  config                     CFG   C:HardwareConfig,SPSAConfig,HardwareRunResult
  layout_optimizer           OPT   C:LayoutOptimizationResult | F:build_filtered_coupling_map,find_vf2_layouts,compute_layout_fidelity_cost+2
  observables                      F:build_per_site_observables,map_observables_to_layout,extract_array_result
  persistence                IO    F:save_run,save_partial_before_error,save_sweep_summary
  phase                      TEST  F:classify_phase
  preflight                        C:QPUCostEstimate,QPUThroughputProfile,SPSACostModel+1 | F:compute_mean_2q_error,compute_layout_2q_error,compute_mean_readout_error+8
  qesem                            C:QESEMResult | F:check_qesem_available,validate_qesem_submission,extrapolate_qet_wls+3
  spsa                       OPT   F:spsa_refinement
  submission                 DIAG  F:select_layouts_for_hardware,submit_all_then_collect,wait_for_qpu_execution+1

### qsim/execution/ (5)
  ↳ ExecutionBackend HardwareBackend HardwareConfig HardwareRunResult SPSAConfig MitigationOptions MPSBackend NoiselessBackend +35

  backends                   CIRC  C:MitigationOptions,ExecutionBackend,NoiselessBackend+3 | F:select_backend,select_backend_with_topology_warning
  eval_cache                 CACHE C:EvalCache,CachedBackend
  mitiq_utils                CORE  C:MitiqZNEResult,MitiqCDRResult,MitiqDDDZNEResult+2 | F:is_mitiq_available,make_mitiq_executor,make_noiseless_executor+5
  mps_backend                EXEC  C:MPSBackend
  noisy_utils                EXEC  C:NoisyEstimatorConfig,LayoutSelection,ZNEResult+10 | F:build_adjacency,find_layouts_bfs,compute_circuit_ces+17

### qsim/framework/ (17)
  ↳ BaseExperiment ExperimentConfig SystemConfig VQEConfig MPNNConfig AnalysisConfig ExperimentMetrics WarmColdComparison +68

  __main__                   CLI   
  artifact_serializers       IO    C:ArtifactSerializer,QPYSerializer,QASM3Serializer+4 | F:get_serializer,register_serializer
  artifact_store             IO    C:ArtifactEntry,ManifestEntry,ArtifactCollector | F:load_manifest,load_artifact,find_artifacts_for_run+3
  base                             C:BaseExperiment
  benchmarking               BENCH C:BenchmarkResult,BenchmarkSuite
  cli                        CLI   F:create_base_parser,add_system_args,add_sweep_args+15
  config                     CFG   C:SystemConfig,VQEConfig,MPNNConfig+2
  criteria                   DIAG  F:compute_verdict
  logging                    DIAG  C:ExperimentEvent,StructuredLogger,ProgressReporter
  metrics                    CORE  C:ExperimentMetrics,WarmColdComparison
  preflight                  VAL   C:Severity,Issue,VariantSpec+4 | F:get_valid_regime,get_regime_threshold,validate_regime_tables+4
  presets                    CFG   F:load_preset,list_presets,preset_to_args
  result_index               CACHE C:ResultIndex
  result_io                  IO    F:build_experiment_id,generate_timestamp,build_result_envelope+10
  result_store               ANAL  C:ResultStore
  runner_base                VAL   C:Section,SectionResult,ValidationRunner+3 | F:resolve_project_root
  variant_runner             VAL   C:PipelineVariant,RunResult,VariantRunner | F:extract_metrics_from_output,run_variant,create_variant_cli+1

### qsim/models/ (5)
  ↳ DEFAULT_SEEDS DMRG_QUBIT_LIMIT EXACT_DIAG_QUBIT_LIMIT EXACT_GAP_QUBIT_LIMIT MAX_P_LAYERS MPS_DEFAULT_CHI_MAX STATEVECTOR_MAX_N SUPPORTED_VQE_METHODS +13

  constants                  CFG   
  data_models                PRED  C:LatticeConfig,GroundTruthResult,VQEConfig+3
  hamiltonian                MODEL C:HamiltonianBuilder | F:generate_chain_1d,generate_ladder,generate_square+5
  model_registry             MODEL F:register_model,get_model_spec,list_models
  model_spec                 MODEL C:ModelSpec

### qsim/optimizers/ (3)
  ↳ VQEOptimizer SPSAOptimizer AdaptiveRestartConfig SelectiveAscendingConfig SelectiveAscendingReport compute_adaptive_restarts compute_restarts_for_sweep select_suspicious_points

  spsa                       OPT   C:SPSAOptimizer
  sweep_strategies           OPT   C:AdaptiveRestartConfig,SelectiveAscendingConfig,SelectiveAscendingReport | F:compute_adaptive_restarts,compute_restarts_for_sweep,select_suspicious_points
  vqe                        OPT   C:OptimizationCallback,VQEOptimizer

### qsim/pipeline/ (4)
  ↳ PipelineRunner load_phase12_dataset save_phase12_dataset run_exact_diag_sweep run_accelerated AcceleratedVQE AcceleratedConfig AcceleratedResult

  accelerated                OPT   C:AcceleratedResult,AcceleratedConfig,AcceleratedVQE
  dataset_io                 VAL   F:get_library_versions,save_phase12_dataset,load_phase12_dataset+1
  qrc                        PIPE  C:QRCPipeline
  runner                     PIPE  C:PipelineRunner | F:run_accelerated,run_exact_diag_sweep

### qsim/predictors/external_benchmarks/ (2)
  ↳ VQEzyInstance VQEzyDataset load_vqezy_tfi load_vqezy_xyz reconstruct_tfi_hamiltonian BenchmarkResult InstanceResult VQEzyBenchmarkEvaluator

  benchmark_evaluator        BENCH C:InstanceResult,BenchmarkResult,VQEzyBenchmarkEvaluator
  vqezy_loader               IO    C:VQEzyInstance,VQEzyDataset | F:load_vqezy_tfi,load_vqezy_xyz,reconstruct_tfi_hamiltonian+1

### qsim/predictors/ (6)
  ↳ MPNNPredictor build_graph_dataset load_mpnn_checkpoint save_mpnn_checkpoint train_mpnn BondResolvedMPNN build_bond_resolved_graph train_bond_resolved_mpnn +47

  gnn_qem                    PRED  C:GNNQEMConfig,GNNQEMCorrector,QEMSample+5 | F:build_qem_graph,build_qem_dataset,train_gnn_qem+14
  model_zoo                  CFG   C:ZooEntry | F:list_pretrained,load_pretrained,load_best_for_cross_n+7
  mpnn                       PRED  C:MPNNPredictor,BondResolvedMPNN | F:predict_theta,build_graph_dataset,train_mpnn+4
  multi_n_aggregator               C:MultiNAggregator
  unified_graph              PRED  F:build_unified_bond_resolved_graph,build_unified_dataset,validate_unified_graph+1
  unified_mpnn               PRED  C:UnifiedMPNN | F:train_unified_mpnn,fine_tune_unified_mpnn,should_retrain+2

### qsim/solvers/ (2)
  ↳ ClassicalSolver

  classical                  SOLVE C:ClassicalSolver
  ground_truth_cache         CACHE C:GroundTruthCache

### qsim/utils/ (1)
  ↳ TimerResult canonicalize_theta filter_consistent_theta json_dump json_serialize set_global_seed timer

  helpers                    IO    C:TimerResult | F:set_global_seed,json_serialize,json_dump+3

## Project Health (project_health/)

### project_health/ (2)
  ↳ ActionItem CoverageGap EnergyDecompositionStats HealthReport ModelDistribution MPNNQualityStats TimingStats VQEQualityStats

  __main__                   CLI   F:parse_args,main
  compare                    ANAL  

### ph/analysis/coverage/ (1)

  scan_coverage              PRED  C:PipelineRecord,NoisyRecord,ExperimentRecord+1 | F:discover_variant_folders,scan_pipeline_results,scan_noisy_results+14

### ph/analysis/hardware/ (8)

  hw_rehearsal_analyzer      DIAG  F:parse_args,load_runs,analyze_single_run+3
  hw_results_analyzer        BENCH F:load_results,check_circuit_metrics,check_qpu_time+2
  layout_optimizer_analyzer  OPT   C:LayoutSelectionRecord,BenchmarkResult,LayoutOptimizerReport | F:run_benchmark,analyze,main
  mitigation_benchmark_analyzer BENCH C:MitigationBenchmarkAnalyzer | F:main
  mitiq_analyzer             ANAL  C:MitiqMethodResult,MitiqComparisonSummary,MitiqAnalysisReport | F:scan_mitiq_results,format_report,format_thesis_table+3
  post_execution_validator   VAL   C:Severity,Finding,ValidationReport | F:validate_envelope,validate_hardware_summary,validate_run+2
  transpilation_analyzer     BENCH F:scan_transpilation_stats,format_transpilation_table,format_opt_level_summary+1
  validate_qet               VAL   C:QETValidationIssue,QETValidationReport | F:validate_qet_result,print_report,validate_file+1

### ph/analysis/models/ (3)

  aqc_tensor_analyzer        CIRC  C:POCSummary,CrossTopologySummary,ComparisonSummary+1 | F:analyze,print_report,print_thesis_table+4
  gnn_qem_analyzer           VAL   
  mpnn_eval_analyzer         PRED  C:WarmstartResult,LOOCVResult,LandscapeResult+8 | F:parse_warmstart,parse_loo_cv,parse_landscape+12

### ph/analysis/scaling/ (3)

  flow_warmstart_analyzer    ANAL  C:FlowWarmstartSummary,BondResolvedSummary,SigmaFlowBoostSummary+2 | F:scan_flow_extension_results,format_report,main
  scaling_analyzer           ANAL  C:ScalingPointResult,ScalingRunSummary,ScalingLawValidation+2 | F:scan_scaling_results,parse_scaling_run,validate_scaling_law+5
  scaling_extensions_analyzer ANAL  C:BondDimResult,VQEConvergenceResult,HEComparisonResult+2 | F:scan_e5_results,parse_bond_dim,parse_vqe_convergence+9

### ph/analysis/thesis/ (3)

  heisenberg_summary         ANAL  F:find_heisenberg_folders,enrich_with_heisenberg_data,print_summary+3
  thesis_figures             VIS   C:FigureConfig | F:register_thesis_figure,fig_global_de_gap_distribution,fig_scaling_law_comprehensive+20
  thesis_tables_compiler           C:TableSpec,TablesReport | F:register_table,compile_tables,parse_args+1

### ph/analysis/validation/ (6)

  affine_overshoot_auditor   VAL   F:scan_experiment_file,main
  audit_findings             VAL   F:audit_f2_pea_zne,audit_f3_scaling_law,audit_f4_gnn_qem+27
  sanity_check               VAL   C:CheckResult,SanityReport | F:register_check,check_theta_trajectories_exist,check_pca_results_exist+16
  thesis_findings_validator  VAL   C:EvidenceStrength,StatisticalEvidence,FindingValidation+1 | F:register_finding,run_validation,parse_args+1
  validate_s_series          VAL   F:validation_1_s1_a3_consistency,validation_2_cft_scaling,validation_3_s4_extra_seeds+3
  verify_results             PIPE  C:VerificationResult,GroupConclusion,VerificationReport | F:parse_pass_criteria,evaluate_criteria,classify_de_gap+5

### ph/analysis/ (11)

  accelerated_cross_n_analyzer PIPE  C:CrossNAnalysis,AcceleratedReport,LargeNResult | F:scan_results,analyze_cross_n_result,format_report+3
  diagnose                   PRED  C:RootCause,DeploymentPoint,Diagnosis | F:parse_pipeline_run,classify_root_causes,scan_folder+3
  gnn_qem_analyzer           PRED  
  layout_optimizer_analyzer  OPT   
  mitigation_benchmark_analyzer BENCH 
  mitiq_analyzer             ANAL  
  noiseless_model_comparison CORE  C:PerHPoint,SectionMetrics,RunResult+4 | F:parse_markdown_report,build_model_summaries,build_topology_summaries+8
  noiseless_pipeline_analyzer PIPE  C:NoiselessRunSummary,TopologyComparison,NoiselessReport | F:scan_noiseless_results,parse_run,detect_anomalies+6
  post_execution_validator   VAL   
  sanity_check               VAL   
  statistical_tests          TEST  F:paired_ttest,improvement_rate,effect_size_cohens_d

### ph/cli/ (6)

  analyze_data_quality       ANAL  F:classify_entry,analyze_index_quality,print_report+1
  compare                    CLI   F:parse_args,main
  inspect_noiseless_run      PIPE  F:find_latest_run,load_run,extract_config+9
  inspect_results            VAL   F:find_result_file,load_envelope,format_value+5
  qpu_time_estimator         CIRC  F:compute_effective_clops,estimate_circuit_qpu_time,parse_args+1
  query_index                      F:parse_args,main

### ph/core/ (5)
  ↳ HealthReport ActionItem CoverageGap Priority

  coverage                   CORE  F:detect_coverage_gaps,derive_actions,compute_noiseless_stats+7
  engine                     ANAL  F:run_health_check
  models                     ANAL  C:Priority,GapType,CoverageGap+8
  reporter                   ANAL  F:generate_timestamped_filename,format_text,format_json+1
  state                      IO    F:load_previous_state,save_current_state,detect_delta+2

### ph/digest/ (4)
  ↳ CrossTopologyResult ExperimentResult ModeComparisonResult N120SweepResult NoiselessResult NoisyResult ResultScanner ScalingResult

  __main__                   CLI   F:apply_filters,parse_args,main
  formatters                       F:format_noiseless_text,format_noisy_text,format_experiment_text+9
  models                           C:NoiselessResult,NoisyResult,ExperimentResult+4
  scanner                          C:ResultScanner

## Experiments (experiments/)

### experiments/ (0)


### exp/generalization/ (5)

  exp_comparative_analysis   PIPE  C:ComparativeAnalysisExperiment
  exp_e4_longitudinal              C:ExperimentE4 | F:build_tfim_longitudinal,exact_diag_sparse
  exp_e4b_longitudinal_hva_extended CIRC  C:ExperimentE4b
  exp_e4c_frustrated_tfim    CIRC  C:ExperimentE4c
  exp_regime_discovery             C:RegimeDiscoveryExperiment

### exp/hardware/ (0)


### exp/helpers/ (9)

  active_learning            OPT   F:compute_ensemble_uncertainty,max_variance_acquisition,expected_improvement_acquisition+2
  analytical_init            ANAL  F:analytical_init_p1,analytical_init_p2,validate_analytical_init
  dypp                       PRED  F:dypp_linear,dypp_quadratic,dypp_predict+1
  graph_utils                CORE  F:build_experiment_dataset,predict_theta,predict_theta_batch+4
  hessian_restart                  F:hessian_guided_vqe,standard_multistart_vqe
  parameter_freezing         CIRC  F:analyze_parameter_activity,frozen_vqe
  physics_loss               PRED  C:PhysicsInformedLoss | F:evaluate_energy_batch,select_eval_subset
  scaling_utils              CORE  F:fit_power_law,compute_transpilation_metrics,evaluate_at_multiple_chi+1
  sign_equivariant           PRED  C:SignInvariantLoss | F:canonicalize_sign,canonicalize_dataset,detect_sign_inconsistency

### exp/landscape/ (3)

  exp_f1_dypp                PRED  C:ExperimentF1
  exp_f3_fluctuation         VAL   C:ExperimentF3
  exp_s3_landscape_n20       ANAL  C:ExperimentS3

### exp/optimization/ (5)

  exp_b1_analytical          ANAL  C:ExperimentB1
  exp_b2_freezing            CIRC  C:ExperimentB2
  exp_b4_hessian                   C:ExperimentB4
  exp_c3_sign                PRED  C:ExperimentC3
  exp_g4_condition_restarts        C:ExperimentG4

### exp/predictor/ (9)

  exp_c1_physics_loss        PRED  C:ExperimentC1
  exp_d1_weight_space        PRED  C:ExperimentD1
  exp_e3_active              OPT   C:ExperimentE3
  exp_g1_data_efficiency     PRED  C:ExperimentG1
  exp_g2_ensemble_calibration PRED  C:ExperimentG2
  exp_g5_cross_seed                C:ExperimentG5
  exp_s2_cross_topology      DIAG  C:ExperimentS2
  exp_s4_data_efficiency_n10       C:ExperimentS4
  exp_s6_mc_dropout_uq             C:ExperimentS6

### exp/scaling/ (6)

  exp_a3_scaling_law         VAL   C:ExperimentA3
  exp_g3_n20_optimized       OPT   C:ExperimentG3
  exp_s1_entanglement_scaling VAL   C:ExperimentS1 | F:compute_entanglement_entropy
  exp_s5_n20_p1_pipeline     PRED  C:ExperimentS5
  exp_s8_d1_finite_size_scaling       C:ExperimentS8 | F:fss_model
  exp_s8b_mpnn_finite_size_scaling PRED  C:ExperimentS8b | F:fss_model

## Runners (scripts/experiment_runners/)

### s/experiment_runners/cross_topology/ (7)

  helpers                    CORE  C:SourceData,ValidationReport,MLPBaseline | F:detect_format,load_source_data,filter_source_data+12
  run_ablation               PRED  F:get_target_h_values,compute_spearman_correlation,evaluate_gnn_predictor+7
  run_cross_n_validation     VAL   F:get_test_h_values,find_source_file,run_single_seed+2
  run_cross_topology         PRED  F:get_target_h_values,run_single_direction,aggregate_seed_results+1
  run_cross_topology_noisy   VAL   C:CrossTopologyNoisyRunner
  run_orchestrator           DIAG  F:run_step,check_data_exists,find_result_files+3
  run_vqe_data_gen           DIAG  F:get_h_values,run_vqe_sweep,get_output_path+2

### s/experiment_runners/noise_aware/ (3)

  run_mpnn_warmstart_accelerator PRED  C:MPNNWarmstartAccelerator
  run_noise_aware_comparison PRED  C:NoiseAwareComparisonRunner
  run_unified_mpnn_benchmark BENCH C:UnifiedMPNNBenchmark

### Standalone scripts

  run_aqc_cross_topology     VAL   F:run_single_compression,main
  run_aqc_poc                VAL   F:run_poc,main
  run_aqc_vs_direct          CIRC  F:run_comparison,main
  run_accelerated_cross_n    PRED  C:AcceleratedCrossNRunner
  run_bond_resolved_cross_n  PRED  C:BondResolvedCrossNRunner
  run_bond_resolved_cross_n_transfer PRED  C:BondResolvedCrossNTransferRunner
  run_bond_resolved_scaling  PRED  C:BondResolvedScalingRunner
  run_bond_resolved_validation VAL   C:BondResolvedValidationRunner
  run_e3_bond_resolved_scaling VAL   C:E3BondResolvedScalingRunner
  run_n16_square_dmrg2d      CIRC  C:N16SquareDMRG2DRunner
  run_scaling_extensions     ANAL  C:ScalingExtensionsRunner | F:analytical_theta_x
  gen_qem_v2_data_ladder     PRED  F:main
  gen_qem_v2_data_ladder_n10 PRED  F:main
  run_gnn_qem_ablation_no_enoisy PRED  C:MLPContextOnly | F:augment,zero_out_enoisy_in_dataset,evaluate_model+3
  run_gnn_qem_ablation_suite VAL   C:MLPContextOnly | F:augment,evaluate_on_test,train_and_eval+5
  run_gnn_qem_cross_topology TEST  F:main
  run_gnn_qem_post_zne_validation VAL   F:run_vqe_sweep,main
  run_gnn_qem_training       IO    F:main
  run_gnn_qem_v2_training    PRED  F:parse_args,generate_or_load_data,augment_samples+6
  run_gnn_qem_vqe_realistic  OPT   F:generate_vqe_data,augment,zero_enoisy_in_dataset+2
  _sanity_check_envelope     IO    
  benchmark_configs          BENCH C:BenchmarkConfig
  run_full_deployment_pipeline PIPE  F:find_latest_rehearsal_json,main
  run_hardware_mitigation_flow       F:log_step,run_command,check_credentials+10
  run_ibm_deployment               C:TierMetrics | F:check_credentials,load_sigma_flow_from_rehearsal,build_hardware_config+10
  run_mitigation_benchmark   BENCH F:append_to_manifest,compute_derived_circuit_stats,apply_affine_on_raw+7
  run_parametric_deployment  CORE  C:DeploymentConfig,ParametricDeployment | F:build_parser,main
  run_gf_zne_batch           DIAG  C:RunConfig | F:run_single_config,run_comparison_analysis,parse_args+1
  run_gf_zne_comparison      ANAL  C:GFZNEComparisonRunner
  run_pea_cross_topology_dense VAL   C:PEACrossTopologyDenseRunner
  run_pea_full_pipeline      VAL   C:PEAFullPipelineRunner
  run_pea_hardware_readiness       C:PEAHardwareReadinessRunner
  run_pea_scaling_n40        VAL   C:PEAScalingRunner
  run_pea_triangular_validation VAL   C:PEATriangularRunner
  run_pea_zne_validation     VAL   C:PEAZNEValidationRunner
  run_zne_3way_comparison    ANAL  C:ZNE3WayComparisonRunner
  run_zne_cross_topology_validation VAL   C:ZNECrossTopologyRunner
  run_mc_dropout_uq                C:MCDropoutRunner
  run_noiseless_cross_n      PRED  C:NoiselessCrossNRunner
  run_noiseless_pipeline     PIPE  C:NoiselessPipelineRunner
  run_cross_n_warmstart_eval       C:CrossNWarmstartEvalRunner
  run_large_n_extrapolation  TEST  C:LargeNExtrapolationRunner | F:load_extrapolation_npz,compute_extrapolation_summary
  run_mps_precision_study    DIAG  C:MPSPrecisionStudyRunner
  run_qpu_time_scaling       CIRC  C:QPUTimeScalingRunner
  run_scaling_phase3_mpnn    PRED  C:MPSScalingPhase3Runner | F:canonicalize_theta,load_theta_from_result
  run_scaling_validation     VAL   C:MPSScalingValidationRunner
  run_t1a_dense_j2           PRED  C:DenseJ2Runner
  run_t1a_mpnn_2d_predictor  PRED  C:MPNN2DPredictorRunner
  run_t1b_longitudinal_zne   EXEC  C:LongitudinalZNERunner
  run_t1c_d1_frustrated            C:D1FrustratedRunner
  run_thesis_variants-chain_1d VAL   F:build_noiseless_variants,build_extended_variants,build_noisy_variants
  run_thesis_variants-heavy_hex DIAG  F:build_noiseless_variants,build_noisy_variants,build_extended_variants+1
  run_thesis_variants-heisenberg PIPE  F:build_noiseless_variants,build_noisy_variants,build_extended_variants+1
  run_thesis_variants-ladder DIAG  F:build_noiseless_variants,build_noisy_variants,build_extended_variants
  run_thesis_variants-triangular DIAG  F:build_noiseless_variants,build_noisy_variants,build_extended_variants
  check_fair_comparison      CIRC  K:H
  run_transpiler_comparison  VAL   C:TranspilerResult,TranspilerExplorationRunner
  analyse_thesis_extensions  ANAL  F:wrap
  run_ext1_intra_n_p1        VAL   C:Ext1bP1ValidationRunner
  run_frustrated_tfim_validation VAL   F:section_1,section_2,section_3+5
  run_hardware_rehearsal_v2        C:HardwareRehearsalV2
  run_hardware_rehearsal_v3  PRED  C:HardwareRehearsalV3
  run_mps_pseudo_hardware    VAL   C:MPSPseudoHardwareRunner
  run_p1_pipeline_variants_r2 TEST  F:build_noiseless_variants,build_noisy_variants,build_extended_variants+1
  run_verification_plan      VAL   F:build_tier1_variants,build_tier2_variants,build_tier3_variants+1

## Scripts (scripts/)

### Standalone scripts

  _test_dmrg_2d_heavy_hex    TEST  C:HeavyHexTFIM | F:run_dmrg_2d
  _test_dmrg_chi_convergence_large_n TEST  K:ROOT
  _test_dmrg_exact           TEST  K:ROOT
  _test_dmrg_vs_vqe_scaling  TEST  F:run_vqe_best,main
  _test_vqe_expressibility   TEST  K:ROOT,N
  analyze_all_phase3         PRED  F:h_min_scaling_law,parse_args,load_phase3_result+8
  benchmark_vqezy            VAL   F:parse_args,find_vqezy_dataset,main
  campaign_extractor         CORE  F:find_latest_result,extract_a1_qpu_scaling,extract_a5_mps_precision+3
  check_delta_e_by_topo_p    DIAG  F:extract_deploy_points,get_config,main
  check_matrix_gaps                F:parse_args,main
  compute_h_frontier               F:parse_args,interpolate_frontier,main
  compute_h_frontier_all     DIAG  F:parse_args,interpolate_frontier,scan_results+4
  compute_h_frontier_models        F:compute_frontier
  compute_h_frontier_topologies DIAG  F:compute_frontier
  dmrg_vs_exact_comparison   SOLVE F:run_comparison,main
  extract_delta_e_fidelity   CORE  C:PointMetrics,RunSummary | F:extract_from_run,scan_all_runs,scan_cross_n+2
  extract_theta_trajectories PIPE  F:extract_trajectory,scan_results,scan_scaling_results+2
  noise_aware_extractor      PRED  F:find_latest_result,extract_section1,extract_section2+8
  precision_frontier_study         F:run_study,main
  quick_noisy_comparison     EXEC  F:run_comparison,main
  reanalyze_p2_filtered      ANAL  
  theta_derivative_analysis  ANAL  F:load_d1_gradients,compute_theta_derivative,compute_correlation+3
  theta_pca_phase_detection  ANAL  F:load_trajectories,analyze_trajectory,generate_figure+3
  validate_vqezy_robustness  VAL   F:build_lattice_and_circuit,load_training_data,evaluate_mpnn_on_vqezy+4
  verify_hva_periodicity     CIRC  K:H
  benchmark                  BENCH F:parse_args,main
  benchmark_pea_parallel     BENCH F:benchmark_measurement_only,benchmark_full_pea,main
  benchmark_pea_performance  BENCH F:benchmark_pea,main
  analyze_qesem_error_detail EXEC  K:RESULT_PATH,H
  analyze_qesem_result       ANAL  K:H
  analyze_qesem_tier1        ANAL  K:RESULT_PATH,H
  complete_tier0_from_qpu          K:ROOT,E_ZNE
  convert_qesem_to_hwresult        F:convert_qesem_recovered,main
  estimate_qesem_budget            F:build_test_circuit,main
  hardware                   CLI   F:cmd_cost,cmd_preflight,cmd_rehearsal+3
  preflight_hw                     F:ok,fail,warn
  print_ladder_circuit       IO    K:ROOT
  print_tier0_circuit        CIRC  K:ROOT
  qesem_error_analysis       ANAL  K:E_EXACT_H4,GAP_H4
  recover_job_result         ANAL  F:save_raw_job_output,connect_service,retrieve_job+1
  recover_qesem_job          DIAG  F:main
  recover_qesem_jobs               F:connect_catalog,list_recent_jobs,recover_qesem_job+1
  compare_resource_estimation VAL   F:section_1_consistency_check,section_2_cross_topology,section_3_error_budget+3
  test_batch_hw_path         TEST  K:ROOT
  test_pea_config            TEST  K:H
  validate_gnn_qem_post_zne  VAL   F:generate_post_zne_samples,main
  validate_warm_start_e3     VAL   K:N,P

## Maintenance (scripts/maintenance/)

### Standalone scripts

  analyze_ladder             ANAL  K:ROOT
  analyze_topology_results   DIAG  K:ROOT,DATA
  audit_pipeline_consistency VAL   F:section,warn,ok
  cleanup_repo               CACHE C:CleanupReport | F:is_protected,scan_named_dirs,scan_cache_dirs+9
  generate_module_index            C:ModuleEntry,PackageEntry | F:extract_module,extract_package,scan_directory+2
  generate_presets_from_index CFG   F:generate_preset_yaml,main
  generate_scaling_report    PRED  F:load_dashboard,compute_quality_tier_breakdown,format_report_text+1
  inspect_data_stores        CACHE F:main
  md_index                         C:Section,FileStats,DocEntry | F:extract_metadata,format_full,format_list+2
  organize_results                 F:extract_config_from_file,organize_flat_dirs,archive_failed+1
  quick_health_check         PRED  F:parse_args,check_model_zoo,check_training_data+4
  run_full_validation        VAL   F:step_1_regenerate_dashboard,step_2_quality_tier_analysis,step_3_training_readiness+5
  scan_new_runs              ANAL  C:RunMetrics,GroupStats | F:resolve_dirs,load_runs,parse_run+11
  update_cross_n_coverage    PIPE  F:load_dashboard,compute_quality_tier_breakdown,generate_quality_tier_table+16
  update_project_status            F:main
  upgrade_npz_quality_tiers        F:compute_quality_tier,upgrade_npz_file,main
  verify_steerings                 C:Issue,VerificationReport | F:estimate_tokens,parse_front_matter,get_body+23

## Notebooks

### Standalone scripts

  test_notebooks             TEST  F:test_notebook_01,test_notebook_02,test_notebook_03
  viz_helpers                VIS   F:draw_circuit,draw_circuit_stats,draw_hva_structure+6
