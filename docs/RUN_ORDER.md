# Research run-order map

This repository preserves the original flat `src/` layout because many scripts use project-root-relative paths and `src.*` imports.

## Phase 1
Core generation/training/evaluation:
- `data.py`
- `model.py`
- `train.py`
- `evaluate.py`
- `run_experiment.py`
- `run_experiment_1b.py`
- `run_experiment_1c.py`
- `generate_ood_data.py`
- `generate_severity_data.py`
- `analyze_phase1.py`
- `run_phase1_confirmation.py`
- `analyze_phase1_confirmation.py`

## Phase 2
- `augmentation.py`
- `audit_augmentation.py`
- `test_augmentation.py`
- `run_experiment_2_pilot.py`
- `analyze_experiment_2_pilot.py`
- `analyze_phase2_interactions.py`
- `run_phase2_confirmation.py`
- `analyze_phase2_confirmation.py`

## Phase 3
- `model_phase3.py`
- `train_phase3.py`
- `test_phase3_models.py`
- `test_phase3_m0_equivalence.py`
- `test_phase3_training_equivalence.py`
- `run_experiment_3_pilot.py`
- `analyze_phase3_interactions.py`
- `run_phase3_confirmation.py`
- `analyze_phase3_confirmation.py`

## Phase 4
- `run_experiment_4_pilot.py`
- `analyze_phase4_interactions.py`
- `analyze_phase4_diagnostics.py`
- `run_phase4_confirmation.py`
- `analyze_phase4_confirmation.py`

## Phase 5
- `run_experiment_5_pilot.py`
- `check_phase5_analysis_integrity.py`
- `diagnose_phase5_h55_permutation.py`
- `analyze_phase5_effect_patterns.py`
- `run_phase5_confirmation.py`
- `analyze_phase5_confirmation.py`

## Phase 6
- `generator_v2.py`
- `validate_generator_v2.py`
- `augmentation_phase6.py`
- `test_augmentation_phase6.py`
- `run_experiment_6_pilot.py`
- `run_phase6_confirmation.py`
- `analyze_phase6_confirmation.py`
- `compare_phase5_phase6.py`

## Phase 7
Audit/reconstruction sequence:
- `audit_phase7_real_data.py`
- `inspect_phase7_schema.py`
- `build_phase7_domain_inventory.py`
- `build_phase7_environment_linkage.py`
- `inspect_phase7_portable_layout.py`
- `audit_phase7_temporal_integrity.py`
- `inspect_phase7_temporal_layout.py`
- `reconstruct_phase7_temporal_history.py`
- `inspect_phase7_exp2_date_anomaly.py`
- `audit_phase7_domain_identity.py`
- `inspect_phase7_exp1_domain_layout.py`
- `build_phase7_corrected_domain_linkage_v2.py`
- `inspect_phase7_exp2_missing_block.py`
- `build_phase7_plant_dataset_v3.py`

Modeling/final analyses:
- `run_phase7_ood_modeling.py`
- `analyze_phase7_matched_ood.py`
- `analyze_phase7_severity_sensitivity.py`

Earlier v1/v2 reconstruction scripts are retained for provenance but should not replace the final v2/v3 files above.
