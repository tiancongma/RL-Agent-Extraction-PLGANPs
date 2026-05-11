# Objective

Bring the remaining same-lineage top-level run groups for `0950`, `1049`, and `1207` into compliance with the single-parent run-lineage containment policy.

# Target lineages cleaned

- `run_20260313_0950_f4912f3_*`
- `run_20260313_1049_f4912f3_*`
- `run_20260313_1207_f4912f3_*`

# Chosen parent runs

- `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`
  - kept top-level because it is the benchmark-valid baseline lineage
- `run_20260313_1049_f4912f3_dev15_current_merged_stage5_relation_provenance_v1`
  - kept top-level because it is the latest human-facing relation-provenance output
- `run_20260313_1207_f4912f3_dev3_doe_integration_stage5_v1`
  - kept top-level because it is the downstream validation result for the three-paper DOE integration lineage

# Old sibling runs per lineage

## 0950 lineage

- `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`
- `run_20260313_0950_f4912f3_dev15_remaining10_current_stage2_extraction_v1`
- `run_20260313_0950_f4912f3_dev15_remaining10_current_stage5_closure_v1`

## 1049 lineage

- `run_20260313_1049_f4912f3_dev15_current_merged_stage3_relation_v1`
- `run_20260313_1049_f4912f3_dev15_current_merged_stage5_relation_provenance_v1`

## 1207 lineage

- `run_20260313_1207_f4912f3_dev3_doe_integration_stage2_v1`
- `run_20260313_1207_f4912f3_dev3_doe_integration_stage3_v1`
- `run_20260313_1207_f4912f3_dev3_doe_integration_stage5_v1`

# New containment layout

Each cleaned lineage now uses the same layout:

- parent run remains top-level
- child runs live under:
  - `lineage/children/<ordered_role>/<child_run_id>/`
- parent lineage includes:
  - `lineage/lineage_mapping.tsv`
  - `lineage/lineage_manifest.md`
  - `lineage/cleanup_report.md`

# Mapping artifact paths

- 0950 lineage:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/lineage_mapping.tsv`
- 1049 lineage:
  - `data/results/run_20260313_1049_f4912f3_dev15_current_merged_stage5_relation_provenance_v1/lineage/lineage_mapping.tsv`
- 1207 lineage:
  - `data/results/run_20260313_1207_f4912f3_dev3_doe_integration_stage5_v1/lineage/lineage_mapping.tsv`
- 1235 template lineage retained from the previous cleanup wave:
  - `data/results/run_20260313_1235_f4912f3_dev15_current_merged_benchmark_doe_v1/lineage/lineage_mapping.tsv`

# Remaining exceptions

- The post-cleanup lineage audit reports no remaining top-level sibling groups for the target timestamp-hash families.
- Older historical lineages outside `0950`, `1049`, `1207`, and `1235` were not moved in this cleanup wave.
- Child run `RUN_CONTEXT.md` files now include a current-location note, but historical prose elsewhere may still mention the former top-level paths.

# Whether the repo now substantially conforms for these groups

Yes.

The `0950`, `1049`, `1207`, and `1235` lineages now each have one clear parent directory, child runs are nested and mapped, and the post-cleanup audit no longer flags these groups as top-level sibling sprawl.

# Post-cleanup audit reference

- `data/results/run_20260313_1235_f4912f3_dev15_current_merged_benchmark_doe_v1/lineage/top_level_lineage_audit_post_cleanup.md`
