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


### qsim/analysis/ (25)
  ↳ AlignmentReport BaselineComparison BaselineMetrics ClusterResult ClusterSolver ComparativeMetrics ComparisonResult DiagnosticCollector +65

  circuit_visualizer         VIS   F:print_circuit,save_circuit_diagram,circuit_summary+11
  comparative                ANAL  C:RegimeDiscoveryResult,ComparativeMetrics | F:find_h_min,classify_result,filter_by_threshold+5
  constants                  CFG   F:compute_quality_score,grade_from_score
  cross_n_validator          VAL   C:L1Result,L2Result,L3Result+2 | F:preflight_cross_n,build_l1_from_precomputed,quick_cross_n_report
  data_models                MODEL C:GradientAnalysisResult,ComparisonResult,BaselineMetrics+1
  diagnostics                CORE  C:DiagnosticCollector | F:configure_pipeline_logging
  entanglement               ANAL  C:EntanglementResult,EntanglementAnalyzer
  evaluation_report          ANAL  F:validate_metrics,generate_comparison_table,generate_evaluation_report+1
  extension_analyzer         PIPE  C:RejectionReportGenerator,PrerequisiteChecker,CalibrationComparator+8
  extension_classifiers      ANAL  C:ClassificationEngine
  extension_models           ANAL  C:ExtensionClassification,PrerequisiteFailedError,HardPhysicsLimitError+3
  extension_ranker           ANAL  C:ExtensionScore,ExtensionPriorityRanker
  failures_tests             TEST  C:FailureDiagnostic | F:diagnose_gap_masking,diagnose_contaminated_training,diagnose_generalization_failure+4
  flow_multishot             PRED  C:MultiShotResult,FlowMultiShotPredictor
  flow_warmstart             OPT   C:FlowWarmstartManager
  gradient                   PRED  C:WeightGradientAnalyzer
  ground_truth_validator     VAL   C:GroundTruthValidationReport,GroundTruthValidator
  landscape                  CORE  F:compute_hessian,landscape_fluctuation
  metrics                    CORE  C:PointClassification | F:is_point_failure,identify_failures,classify_point_failure+34
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

### qsim/framework/ (18)
  ↳ BaseExperiment ExperimentConfig SystemConfig VQEConfig MPNNConfig AnalysisConfig ExperimentMetrics WarmColdComparison +76

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
  quality_profile                  C:QualityProfile | F:compute_quality_profile,format_quality_summary,format_per_h_status+2
  result_index               CACHE C:ResultIndex
  result_io                  IO    F:build_experiment_id,generate_timestamp,build_result_envelope+17
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
  ↳ AcceleratedVQE AcceleratedConfig AcceleratedResult load_phase12_dataset save_phase12_dataset PipelineRunner run_exact_diag_sweep run_accelerated

  accelerated                OPT   C:AcceleratedResult,AcceleratedConfig,AcceleratedVQE
  dataset_io                 VAL   F:get_library_versions,save_phase12_dataset,load_phase12_dataset+2
  qrc                        PIPE  C:QRCPipeline
  runner                     PIPE  

### qsim/predictors/external_benchmarks/ (2)
  ↳ VQEzyInstance VQEzyDataset load_vqezy_tfi load_vqezy_xyz reconstruct_tfi_hamiltonian BenchmarkResult InstanceResult VQEzyBenchmarkEvaluator

  benchmark_evaluator        BENCH C:InstanceResult,BenchmarkResult,VQEzyBenchmarkEvaluator
  vqezy_loader               IO    C:VQEzyInstance,VQEzyDataset | F:load_vqezy_tfi,load_vqezy_xyz,reconstruct_tfi_hamiltonian+1

### qsim/predictors/ (7)
  ↳ MPNNPredictor build_graph_dataset load_mpnn_checkpoint save_mpnn_checkpoint train_mpnn BondResolvedMPNN build_bond_resolved_graph train_bond_resolved_mpnn +60

  gnn_qem                    PRED  C:GNNQEMConfig,GNNQEMCorrector,QEMSample+5 | F:build_qem_graph,build_qem_dataset,train_gnn_qem+14
  model_registry_db          PRED  C:TrainingMetrics,ModelArchitectureConfig,OptimizerConfig+8
  model_zoo                  CFG   C:ZooEntry | F:compute_model_readiness,compute_training_quality_score,refresh_zoo_quality_scores+15
  mpnn                       PRED  C:MPNNPredictor,BondResolvedMPNN | F:predict_theta,build_graph_dataset,train_mpnn+4
  multi_n_aggregator               C:MultiNAggregator
  unified_graph              PRED  F:build_unified_bond_resolved_graph,build_unified_dataset,validate_unified_graph+1
  unified_mpnn               PRED  C:UnifiedMPNN | F:train_unified_mpnn,fine_tune_unified_mpnn,should_retrain+3

### qsim/solvers/ (2)
  ↳ ClassicalSolver

  classical                  SOLVE C:ClassicalSolver
  ground_truth_cache         CACHE C:GroundTruthCache

### qsim/utils/ (1)
  ↳ BatchWriteMixin TimerResult atomic_savez augment_theta_symmetries canonicalize_theta filter_consistent_theta json_dump json_serialize +3

  helpers                    IO    C:TimerResult,BatchWriteMixin | F:set_global_seed,json_serialize,json_dump+6

## Project Health (project_health/)

### project_health/ (1)
  ↳ ActionItem CoverageGap EnergyDecompositionStats HealthReport ModelDistribution MPNNQualityStats TimingStats VQEQualityStats

  __main__                   CLI   F:parse_args,main

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

### ph/analysis/scaling/ (1)

  scaling_analyzer           ANAL  C:ScalingPointResult,ScalingRunSummary,ScalingLawValidation+2 | F:scan_scaling_results,parse_scaling_run,validate_scaling_law+5

### ph/analysis/thesis/ (3)

  heisenberg_summary         ANAL  F:find_heisenberg_folders,enrich_with_heisenberg_data,print_summary+3
  thesis_figures             VIS   C:FigureConfig | F:register_thesis_figure,fig_global_de_gap_distribution,fig_scaling_law_comprehensive+20
  thesis_tables_compiler           C:TableSpec,TablesReport | F:register_table,compile_tables,parse_args+1

### ph/analysis/validation/ (6)

  affine_overshoot_auditor   VAL   F:scan_experiment_file,main
  audit_findings             VAL   F:audit_f2_pea_zne,audit_f3_scaling_law,audit_f4_gnn_qem+27
  pipeline_consistency       VAL   F:section,warn,ok+1
  sanity_check               VAL   C:CheckResult,SanityReport | F:register_check,check_theta_trajectories_exist,check_pca_results_exist+16
  thesis_findings_validator  VAL   C:EvidenceStrength,StatisticalEvidence,FindingValidation+1 | F:register_finding,run_validation,parse_args+1
  verify_results             PIPE  C:VerificationResult,GroupConclusion,VerificationReport | F:parse_pass_criteria,evaluate_criteria,classify_de_gap+5

### ph/analysis/ (6)

  accelerated_cross_n_analyzer PIPE  C:CrossNAnalysis,AcceleratedReport,LargeNResult | F:scan_results,analyze_cross_n_result,format_report+5
  diagnose                   PRED  C:RootCause,DeploymentPoint,Diagnosis | F:parse_pipeline_run,classify_root_causes,scan_folder+3
  gnn_qem_analyzer           PRED  
  noiseless_model_comparison CORE  C:PerHPoint,SectionMetrics,RunResult+4 | F:parse_markdown_report,build_model_summaries,build_topology_summaries+8
  noiseless_pipeline_analyzer PIPE  C:NoiselessRunSummary,TopologyComparison,NoiselessReport | F:scan_noiseless_results,parse_run,detect_anomalies+6
  statistical_tests          TEST  F:paired_ttest,improvement_rate,effect_size_cohens_d+4

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

## Runners (scripts/experiment_runners/)

### s/experiment_runners/cross_topology/ (2)

  helpers                    CORE  C:SourceData,ValidationReport,MLPBaseline | F:detect_format,load_source_data,filter_source_data+12
  run_vqe_data_gen           DIAG  F:get_h_values,run_vqe_sweep,get_output_path+2

### Standalone scripts

  run_accelerated_cross_n    PRED  C:AcceleratedCrossNRunner
  run_bond_resolved_validation VAL   C:BondResolvedValidationRunner
  run_n16_square_dmrg2d      CIRC  C:N16SquareDMRG2DRunner
  run_scaling_extensions     ANAL  C:ScalingExtensionsRunner | F:analytical_theta_x
  gen_qem_v2_data_ladder     PRED  F:main
  gen_qem_v2_data_ladder_n10 PRED  F:main
  run_gnn_qem_ablation_no_enoisy PRED  C:MLPContextOnly | F:augment,zero_out_enoisy_in_dataset,evaluate_model+3
  run_gnn_qem_ablation_suite VAL   C:MLPContextOnly | F:augment,evaluate_on_test,train_and_eval+5
  run_gnn_qem_post_zne_validation VAL   F:run_vqe_sweep,main
  run_gnn_qem_training       IO    F:main
  run_gnn_qem_v2_training    PRED  F:parse_args,generate_or_load_data,augment_samples+6
  _sanity_check_envelope     IO  
  benchmark_configs          BENCH C:BenchmarkConfig
  run_full_deployment_pipeline PIPE  F:find_latest_rehearsal_json,main
  run_hardware_mitigation_flow       F:log_step,run_command,check_credentials+10
  run_ibm_deployment               C:TierMetrics | F:check_credentials,load_sigma_flow_from_rehearsal,build_hardware_config+10
  run_mitigation_benchmark   BENCH F:append_to_manifest,compute_derived_circuit_stats,apply_affine_on_raw+7
  run_parametric_deployment  CORE  C:DeploymentConfig,ParametricDeployment | F:build_parser,main
  run_pea_cross_topology_dense VAL   C:PEACrossTopologyDenseRunner
  run_pea_full_pipeline      VAL   C:PEAFullPipelineRunner
  run_pea_hardware_readiness       C:PEAHardwareReadinessRunner
  run_pea_scaling_n40        VAL   C:PEAScalingRunner
  run_noiseless_cross_n      PRED  C:NoiselessCrossNRunner
  run_noiseless_pipeline     PIPE  C:NoiselessPipelineRunner
  run_cross_n_warmstart_eval       C:CrossNWarmstartEvalRunner
  run_large_n_extrapolation  TEST  C:LargeNExtrapolationRunner | F:load_extrapolation_npz,compute_extrapolation_summary
  run_mps_precision_study    DIAG  C:MPSPrecisionStudyRunner
  run_qpu_time_scaling       CIRC  C:QPUTimeScalingRunner
  run_scaling_phase3_mpnn    PRED  C:MPSScalingPhase3Runner | F:canonicalize_theta,load_theta_from_result
  run_scaling_validation     VAL   C:MPSScalingValidationRunner
  run_hardware_rehearsal_v2        C:HardwareRehearsalV2
  run_hardware_rehearsal_v3  PRED  C:HardwareRehearsalV3
  run_mps_pseudo_hardware    VAL   C:MPSPseudoHardwareRunner
  run_verification_plan      VAL   F:build_tier1_variants,build_tier2_variants,build_tier3_variants+1

## Scripts (scripts/)

### Standalone scripts

  _test_dmrg_2d_heavy_hex    TEST  C:HeavyHexTFIM | F:run_dmrg_2d
  _test_dmrg_chi_convergence_large_n TEST  K:ROOT
  _test_dmrg_exact           TEST  K:ROOT
  _test_dmrg_vs_vqe_scaling  TEST  F:run_vqe_best,main
  _test_vqe_expressibility   TEST  K:ROOT,N
  analyze_all_phase3         PRED  F:h_min_scaling_law,parse_args,load_phase3_result+8
  analyze_extrapolation_runs ANAL  F:parse_args,load_and_filter_runs,extract_metrics+2
  benchmark_vqezy            VAL   F:parse_args,find_vqezy_dataset,main
  campaign_extractor         CORE  F:find_latest_result,extract_a1_qpu_scaling,extract_a5_mps_precision+3
  check_delta_e_by_topo_p    DIAG  F:extract_deploy_points,get_config,main
  check_matrix_gaps                F:parse_args,main
  compile_multiseed_report   ANAL  F:extract_metrics,main
  compute_h_frontier               F:parse_args,interpolate_frontier,main
  compute_h_frontier_all     DIAG  F:parse_args,interpolate_frontier,scan_results+4
  compute_h_frontier_models        F:compute_frontier
  compute_h_frontier_topologies DIAG  F:compute_frontier
  dmrg_vs_exact_comparison   SOLVE F:run_comparison,main
  evaluate_zoo_models              F:parse_args,evaluate_model,format_markdown_report+1
  extract_delta_e_fidelity   CORE  C:PointMetrics,RunSummary | F:extract_from_run,scan_all_runs,scan_cross_n+2
  extract_theta_trajectories PIPE  F:extract_trajectory,scan_results,scan_scaling_results+2
  noise_aware_extractor      PRED  F:find_latest_result,extract_section1,extract_section2+8
  precision_frontier_study         F:run_study,main
  quick_noisy_comparison     EXEC  F:run_comparison,main
  reanalyze_p2_filtered      ANAL  
  regenerate_eval_reports    ANAL  F:parse_args,load_npz_as_per_h_results,load_baseline_summary+3
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

  analyze_topology_results   DIAG  K:ROOT,DATA
  audit_and_fix_model_zoo    VAL   F:audit_manifest,audit_registry,audit_consistency+7
  check_zoo_coherence              F:check_coherence,main
  generate_presets_from_index CFG   F:generate_preset_yaml,main
  generate_scaling_report    PRED  F:load_dashboard,compute_quality_tier_breakdown,format_report_text+1
  inspect_data_stores        CACHE F:main
  query_model_registry       MODEL F:cmd_list,cmd_get,cmd_summary+17
  quick_health_check         PRED  F:parse_args,check_model_zoo,check_training_data+5
  reevaluate_zoo_models            F:evaluate_npz_quality,main
  run_full_validation        VAL   F:step_1_regenerate_dashboard,step_2_quality_tier_analysis,step_3_training_readiness+5
  update_cross_n_coverage    PIPE  F:load_dashboard,compute_quality_tier_breakdown,generate_quality_tier_table+19
  update_project_status            F:main
  upgrade_npz_quality_tiers        F:compute_quality_tier_for_npz,upgrade_single_npz,main

## Notebooks

### Standalone scripts

  test_notebooks             TEST  F:test_notebook_01,test_notebook_02,test_notebook_03
  viz_helpers                VIS   F:draw_circuit,draw_circuit_stats,draw_hva_structure+6
