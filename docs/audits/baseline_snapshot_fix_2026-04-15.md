# Baseline Snapshot Fix (2026-04-15)

## 1. Problem

`baseline_20260415_deterministic_two_step_v1` was registered against a lineage-only authority root:

- `data/results/20260415_23c14f0/18_dev15_deterministic_two_step_diag_with_combined_binding_v3`

That path is documented in checked audits and runner defaults, but it is not materialized in the current workspace. As a result, the baseline registry resolved to an abstract lineage description rather than to a real reproducible artifact root.

## 2. Chosen materialized source

Best available local materialized equivalent:

- primary Step1/binding source:
  - `data/results/20260415_targeted_core_a_repair_codepath_v2`
- associated Step2 child:
  - `data/results/20260415_8a2502a/02_deterministic_step2_baseline`

Why this pair was chosen:

- both directories exist locally
- Step1 source contains:
  - `final_formulation_table_v1.tsv`
  - `final_output_decision_trace_v1.tsv`
  - `formulation_relation_records_v1.tsv`
  - `resolved_relation_fields_v1.tsv`
  - `table_row_binding_resolved_v1.tsv`
  - `formulation_parameter_binding_resolved_v1.tsv`
  - `RUN_CONTEXT.md`
- associated Step2 child contains:
  - `step2_value_backfill_table_v1.tsv`
  - `step2_value_backfill_summary_v1.md`
  - `RUN_CONTEXT.md`

No single local run directory contained the full originally registered combined-binding lineage plus materialized Layer2/Layer3 compare outputs.

## 3. Snapshot location

- `data/results/frozen_baselines/20260415_deterministic_two_step_v1`

## 4. Manifest and registry corrections

Updated baseline primary authority to the frozen snapshot:

- registry `authority_root`:
  - `data/results/frozen_baselines/20260415_deterministic_two_step_v1`
- registry `primary_lineage_root`:
  - `data/results/20260415_targeted_core_a_repair_codepath_v2`

Updated manifest fields:

- `authority_root`
- `source_run_dir`
- `source_type`
- `reproducibility_status`
- `source_artifacts`
- `artifact_chain`
- `lineage_chain`

Added reproducibility fields:

- `"source_run_dir": "data/results/frozen_baselines/20260415_deterministic_two_step_v1"`
- `"source_type": "frozen_snapshot"`
- `"reproducibility_status": "fully_materialized"`

## 5. Snapshot contents

Copied live artifacts:

- `final_formulation_table_v1.tsv`
- `step2_value_backfill_table_v1.tsv`
- `step2_value_backfill_summary_v1.md`
- `final_output_decision_trace_v1.tsv`
- `formulation_relation_records_v1.tsv`
- `resolved_relation_fields_v1.tsv`
- `table_row_binding_resolved_v1.tsv`
- `table_row_binding_summary_v1.md`
- `formulation_parameter_binding_resolved_v1.tsv`
- `formulation_parameter_binding_summary_v1.md`
- `RUN_CONTEXT.md`

Added snapshot companion governance notes:

- `SNAPSHOT_PROVENANCE.md`
- `layer2_compare_summary_v1.md`
- `layer3_compare_summary_v1.md`

The Layer2 and Layer3 summary files are explicitly marked as frozen snapshot companion summaries because no materialized live-run compare summaries existed in the chosen local source lineage.

## 6. Validation

Commands run:

```powershell
python src/utils/baseline_registry_v1.py validate
python src/utils/baseline_registry_v1.py list
python src/utils/baseline_registry_v1.py show --query 20260415
```

Validation result:

- registry validation passed
- both baselines on `2026-04-15` resolve cleanly
- the deterministic two-step baseline now points to a real existing authority root

## 7. Result

The baseline is now reproducible in the governed sense:

- registry resolves to a real path
- referenced snapshot artifacts exist
- future reuse no longer depends on the missing `20260415_23c14f0` run directory
- lineage-only references remain as metadata, not as the primary baseline authority
