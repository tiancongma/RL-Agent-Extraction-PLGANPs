# Results Lineage Normalization Pass

## Objective

Perform one controlled normalization pass for top-level `data/results/run_*`
directories so that each detected same-lineage run family has exactly one
top-level parent run directory.

## Lineages detected under the global normalization rule

The normalization rule used in this pass was:

- same date
- same short commit
- same full suffix after the short commit

Under that rule, two top-level sibling lineages required normalization.

### Lineage 1

- Family key: `20260312 / 455ac37 / targeted5_stage2_regression_v1`
- Parent chosen: `run_20260312_1030_455ac37_targeted5_stage2_regression_v1`
- Children moved:
  - `run_20260312_1031_455ac37_targeted5_stage2_regression_v1`
  - `run_20260312_1253_455ac37_targeted5_stage2_regression_v1`
  - `run_20260312_1321_455ac37_targeted5_stage2_regression_v1`

### Lineage 2

- Family key: `20260312 / 455ac37 / minimal_final_output_v1`
- Parent chosen: `run_20260312_1610_455ac37_minimal_final_output_v1`
- Children moved:
  - `run_20260312_1614_455ac37_minimal_final_output_v1`
  - `run_20260312_1617_455ac37_minimal_final_output_v1`
  - `run_20260312_1619_455ac37_minimal_final_output_v1`
  - `run_20260312_1624_455ac37_minimal_final_output_v1`
  - `run_20260312_1629_455ac37_minimal_final_output_v1`
  - `run_20260312_1636_455ac37_minimal_final_output_v1`

## Parent lineage artifacts created

### `run_20260312_1030_455ac37_targeted5_stage2_regression_v1`

- `lineage/children/`
- `lineage/lineage_mapping.tsv`
- `lineage/lineage_manifest.md`
- `lineage/cleanup_report.md`

### `run_20260312_1610_455ac37_minimal_final_output_v1`

- `lineage/children/`
- `lineage/lineage_mapping.tsv`
- `lineage/lineage_manifest.md`
- `lineage/cleanup_report.md`

## Directories unaffected

The following top-level `run_*` directories were left unchanged because they
were not same-lineage siblings under the exact rule used for this pass, were
already-contained parents, or were frozen historical exceptions that this pass
did not modify:

- `run_20260201_0927_bb13267_sample20`
- `run_20260310_dev15_remaining12_synthmethod`
- `run_20260310_dev15_remaining12_synthmethod_merged`
- `run_20260310_v7pilot3r3fixparse_synthmethod`
- `run_20260312_1633_455ac37_targeted5_full_pipeline_benchmark_v1`
- `run_20260312_1723_455ac37_downstream_only_from_existing_stage2_v1`
- `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`
- `run_20260313_1049_f4912f3_dev15_current_merged_stage5_relation_provenance_v1`
- `run_20260313_1151_f4912f3_ufxx9wxe_doe_row_recovery_v1`
- `run_20260313_1152_f4912f3_ufxx9wxe_doe_row_recovery_v2`
- `run_20260313_1153_f4912f3_ufxx9wxe_doe_row_recovery_v3`
- `run_20260313_1154_f4912f3_ufxx9wxe_doe_row_recovery_v4`
- `run_20260313_1157_f4912f3_ufxx9wxe_doe_row_recovery_v5`
- `run_20260313_1207_f4912f3_dev3_doe_integration_stage5_v1`
- `run_20260313_1235_f4912f3_dev15_current_merged_benchmark_doe_v1`

## Ambiguous cases

- The `UFXX9WXE` DOE recovery sequence (`v1` through `v5`) was left unchanged.
  Under the exact rule for this pass, those runs do not share the same full
  suffix because the version token is part of the suffix.
- The `20260310` historical top-level runs were left unchanged because this pass
  did not modify frozen historical exceptions or archival compatibility paths.

## Counts

- lineages normalized: `2`
- runs moved: `9`
- runs left unchanged at top level: `17`

## Post-pass state

- Each detected same-lineage family under the exact normalization rule now has
  one top-level parent run directory.
- Moved child runs now live under:
  - `parent_run/lineage/children/<child_run_dir>/`
- No files were deleted.
- No pipeline logic changed.

---

## Pass 2: Broad Family Container Rule

### Objective

Apply the broader containment rule:

- one top-level parent per `(YYYYMMDD, short_commit)` family
- ignore differences in time, suffix, purpose, stage, or version tag when
  deciding top-level containment

### Families detected

1. `20260201 / bb13267`
   - top-level runs:
     - `run_20260201_0927_bb13267_sample20`
   - action:
     - unchanged, single-run family
2. `20260312 / 455ac37`
   - parent chosen:
     - `run_20260312_1030_455ac37_targeted5_stage2_regression_v1`
   - children moved in this pass:
     - `run_20260312_1610_455ac37_minimal_final_output_v1`
     - `run_20260312_1633_455ac37_targeted5_full_pipeline_benchmark_v1`
     - `run_20260312_1723_455ac37_downstream_only_from_existing_stage2_v1`
3. `20260313 / f4912f3`
   - parent chosen:
     - `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`
   - children moved in this pass:
     - `run_20260313_1049_f4912f3_dev15_current_merged_stage5_relation_provenance_v1`
     - `run_20260313_1151_f4912f3_ufxx9wxe_doe_row_recovery_v1`
     - `run_20260313_1152_f4912f3_ufxx9wxe_doe_row_recovery_v2`
     - `run_20260313_1153_f4912f3_ufxx9wxe_doe_row_recovery_v3`
     - `run_20260313_1154_f4912f3_ufxx9wxe_doe_row_recovery_v4`
     - `run_20260313_1157_f4912f3_ufxx9wxe_doe_row_recovery_v5`
     - `run_20260313_1207_f4912f3_dev3_doe_integration_stage5_v1`
     - `run_20260313_1235_f4912f3_dev15_current_merged_benchmark_doe_v1`

### Unaffected top-level runs

- `run_20260201_0927_bb13267_sample20`
- `run_20260310_dev15_remaining12_synthmethod`
- `run_20260310_dev15_remaining12_synthmethod_merged`
- `run_20260310_v7pilot3r3fixparse_synthmethod`
- `run_20260312_1030_455ac37_targeted5_stage2_regression_v1`
- `run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1`

### Ambiguous cases

- `run_20260310_dev15_remaining12_synthmethod`
- `run_20260310_dev15_remaining12_synthmethod_merged`
- `run_20260310_v7pilot3r3fixparse_synthmethod`

These were not normalized under the broad family rule because they do not
expose a parseable `run_<date>_<time>_<short_commit>_...` structure and remain
frozen historical exceptions in current governance.

### Counts for pass 2

- families normalized: `2`
- runs moved: `11`
- runs left unchanged: `6`

### Post-pass state

- Top-level `run_*` containment is now governed by the broader `(date,
  short_commit)` family container rule.
- Narrower semantic sub-lineages remain preserved inside those parents as child
  directories with their own lineage packs.
