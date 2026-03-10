# DEV3 Grouping Debug: Post-Processing vs DOE Decisions

Date: 2026-03-09

## Scope
Target papers:
- `7ZS858NS` (`10.1021/acsomega.0c00111`) - expected GT rows: 1
- `UFXX9WXE` (`10.1155/2014/156010`) - expected GT rows: 26
- `5ZXYABSU` (`10.2147/ijn.s130908`) - expected GT rows: 9

Two historical decisions under audit:
1. Post-processing/test-condition differences should **not** create new formulation instances.
2. DOE/optimization variables outside narrow fixed schema may still define real distinct formulations.

## Pipeline Path Used (current/latest normal path)

Latest run pointer found in:
- `runs/latest.txt` -> `run_20260228_1634_d37c1f3_goren2025_control_no_tablefirst_goren2025_step1dev_v1`

Latest extraction/grouping artifacts used as source:
- Extraction output: `data/results/run_20260228_1634_d37c1f3_goren2025_control_no_tablefirst_goren2025_step1dev_v1/step1_dev/weak_labels__gemini.tsv`
- Grouping stage script rerun on subset: `src/stage5_benchmark/run_formulation_core_signature_v1.py`
- Grouping engine: `src/stage5_benchmark/formulation_core_signature_v1.py`

Why this path:
- It is the latest active deterministic grouping stage used in normal pipeline outputs.
- It uses extraction output rows and runs formulation identity construction/merge gates (`A/B/C`), not the manual skeleton scaffold.

## Input Subset Used

Created subset files:
- Manifest: `data/results/run_20260309_1632_aa0bb8a_dev3_grouping_debug/dev3_grouping_debug_v1/dev3_manifest_v1.tsv`
- Extraction rows: `data/results/run_20260309_1632_aa0bb8a_dev3_grouping_debug/dev3_grouping_debug_v1/weak_labels__gemini__dev3.tsv`

Run command executed:
```bash
py -3 src/stage5_benchmark/run_formulation_core_signature_v1.py \
  --run-id run_20260309_1632_aa0bb8a_dev3_grouping_debug \
  --out-subdir dev3_grouping_debug_v1 \
  --input-tsv data/results/run_20260309_1632_aa0bb8a_dev3_grouping_debug/dev3_grouping_debug_v1/weak_labels__gemini__dev3.tsv
```

Output grouping source for projection:
- `data/results/run_20260309_1632_aa0bb8a_dev3_grouping_debug/dev3_grouping_debug_v1/formulation_core_signature_v1/instance_assignment_v1.tsv`

## GT Source Used

For this targeted DEV3 debug pass, GT rows were fixed from the established expected counts provided for these three papers and stored as:
- `data/results/run_20260309_1632_aa0bb8a_dev3_grouping_debug/dev3_grouping_debug_v1/dev3_manual_gt_expected_rows.tsv`

Rows:
- `7ZS858NS` -> 1
- `UFXX9WXE` -> 26
- `5ZXYABSU` -> 9

## Predicted Skeleton Projection

Projection file (one row per predicted formulation instance):
- `project/debug/dev3_predicted_formulation_skeleton.tsv`

Columns:
- `paper_key`
- `doi`
- `predicted_formulation_id`
- `predicted_instance_source_type`
- `predicted_identity_basis`
- `notes`

## Comparison Output

Comparison table:
- `project/debug/dev3_predicted_formulation_skeleton_comparison.tsv`

Summary:

| paper_key | doi | GT_rows | predicted_rows | ratio | error_type |
|---|---|---:|---:|---:|---|
| 7ZS858NS | 10.1021/acsomega.0c00111 | 1 | 7 | 7.0 | over-segmentation |
| UFXX9WXE | 10.1155/2014/156010 | 26 | 8 | 0.3077 | under-segmentation |
| 5ZXYABSU | 10.2147/ijn.s130908 | 9 | 9 | 1.0 | correct |

## Per-Paper Diagnosis and Decision Audit

### 1) 7ZS858NS (post-processing over-segmentation case)
- Pipeline likely treated freeze-drying/cryoprotectant variants as separate formulations (IDs 1..7).
- Evidence: weak-label notes include post-processing variants (sucrose/mannitol freeze-drying) and grouping retained separate rows.
- Decision 1 (post-processing should not split): **Not preserved**.
- Decision 2 (DOE/generic variable should define real formulations): **Not directly exercised** here.
- Likely failure stage: **formulation identity construction / grouping** (explicit instance retention without post-processing collapse rule).

### 2) UFXX9WXE (DOE under-segmentation case)
- Pipeline produced 8 predicted instances (IDs 1..8) vs GT 26.
- Build log from grouping run shows DOE trace disabled: `doe_trace_enabled=False`, `count_rows_with_doe_signature=0`.
- Decision 1 (post-processing non-split): **Not the dominant issue** in this paper.
- Decision 2 (DOE/generic-variable identity must be preserved): **Not preserved**.
- Likely failure stage: **extraction + formulation identity construction** (upstream extraction under-enumerates DOE runs; grouping has no DOE signature input to recover missing runs).

### 3) 5ZXYABSU (table-driven control case)
- Pipeline produced 9 predicted instances and matches GT 9.
- Decision 1 (post-processing non-split): **Preserved in effect** for this case.
- Decision 2 (DOE/generic-variable identity): **No clear violation observed** in this case.
- Likely status: **stable under current table/label-driven grouping behavior**.

## Explicit Check of the Two Decisions

| paper_key | Decision 1 preserved? | Decision 2 preserved? | Primary failing stage (if any) |
|---|---|---|---|
| 7ZS858NS | No | Not applicable / not tested | grouping / formulation identity construction |
| UFXX9WXE | Not primary issue | No | extraction + formulation identity construction |
| 5ZXYABSU | Yes (for this case) | Yes (for this case) | none |

## Recommended Next Engineering Action (no code changes in this task)

Add a deterministic paper-level identity policy layer before final grouping count export:
1. Post-processing suppression rule family for non-core condition variants (freeze-dry/storage/release/test-only changes) when core synthesis anchors are unchanged.
2. DOE-preservation augmentation that consumes DOE/generic-variable evidence (or decoded factor traces where available) into identity signatures for papers like `UFXX9WXE`.
3. Run the same DEV3 triage after each change to track regressions on both decisions simultaneously.

