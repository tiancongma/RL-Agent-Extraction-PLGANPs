# DEV15 Segmentation Closure Validation

## FACTS

- run_dir:
  - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation`
- comparable prior full DEV15 replay:
  - `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation`
- scripts used:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
  - `src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py`
  - `src/utils/update_run_context_with_feature_activation_v1.py`
- manifest used:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- legacy replay source:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses`
- exact command:
```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'
$env:STAGE2_INPUT_EVIDENCE_PACKING_MODE='ordered_blocks'
$env:STAGE2_TABLE_MODE='summary'
$env:STAGE2_TABLE_SUMMARY_FIRST_COLUMN_ENHANCEMENT='1'
python src/stage2_sampling_labels/run_stage2_composite_v1.py --run-dir data/results/20260411_312d44b/06_dev15_segmentation_closure_validation --manifest-tsv data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv --source-mode legacy_llm_replay --legacy-raw-responses-dir data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses --llm-backend gemini --model gemini-2.5-flash
```
- total papers processed:
  - `15`
- technical pass/fail counts:
  - `15 pass / 0 fail`
  - contract report:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_semantic_authority_contract_report_v1.json`
    - `status=pass`
- design pass/fail counts:
  - `4 pass / 11 fail`
- design-pass papers:
  - `WIVUCMYG`
  - `QLYKLPKT`
  - `UFXX9WXE`
  - `WFDTQ4VX`
- papers with regression vs prior comparable full DEV15 replay:
  - `5ZXYABSU`
  - `L3H2RS2H`
  - `5GIF3D8W`
  - `7ZS858NS`
  - `BB3JUVW7`
  - `BXCV5XWB`
  - `INMUTV7L`
  - `PA3SPZ28`
  - `RHMJWZX8`
  - `V99GKZEI`
  - `YGA8VQKU`
- papers newly fixed vs prior comparable full DEV15 replay:
  - none
- candidate count distribution summary:
  - min: `46`
  - median: `101`
  - mean: `116.40`
  - max: `235`
  - exact distribution:
    - `46:1`
    - `61:1`
    - `63:1`
    - `64:1`
    - `71:1`
    - `82:1`
    - `90:1`
    - `101:1`
    - `102:1`
    - `118:1`
    - `141:1`
    - `152:1`
    - `197:1`
    - `223:1`
    - `235:1`
- table_blocks distribution summary:
  - min: `0`
  - median: `1`
  - mean: `1.60`
  - max: `3`
  - exact distribution:
    - `0:3`
    - `1:5`
    - `2:2`
    - `3:5`
- stable table_blocks decreases vs prior comparable replay:
  - `5ZXYABSU: 4 -> 1`
  - `L3H2RS2H: 4 -> 1`
  - `5GIF3D8W: 4 -> 0`
  - `7ZS858NS: 4 -> 1`
  - `BXCV5XWB: 4 -> 0`
  - `INMUTV7L: 4 -> 2`
  - `PA3SPZ28: 4 -> 1`
  - `QLYKLPKT: 4 -> 3`
  - `RHMJWZX8: 4 -> 1`
  - `UFXX9WXE: 4 -> 3`
  - `WFDTQ4VX: 4 -> 3`
  - `YGA8VQKU: 4 -> 3`
  - `V99GKZEI: 1 -> 0`
- papers still missing key roles after refinement:
  - `5ZXYABSU`: `OPTIMIZATION_RESULT`
  - `L3H2RS2H`: `OPTIMIZATION_RESULT`
  - `5GIF3D8W`: `EXPERIMENTAL_DESIGN, VARIABLE_TABLE, FORMULATION_TABLE`
  - `7ZS858NS`: `MATERIALS, EXPERIMENTAL_DESIGN, VARIABLE_TABLE`
  - `BB3JUVW7`: `MATERIALS`
  - `BXCV5XWB`: `PREPARATION_METHOD, EXPERIMENTAL_DESIGN, VARIABLE_TABLE, FORMULATION_TABLE, OPTIMIZATION_RESULT`
  - `INMUTV7L`: `EXPERIMENTAL_DESIGN, VARIABLE_TABLE`
  - `PA3SPZ28`: `OPTIMIZATION_RESULT`
  - `RHMJWZX8`: `EXPERIMENTAL_DESIGN, VARIABLE_TABLE`
  - `V99GKZEI`: `MATERIALS, EXPERIMENTAL_DESIGN, VARIABLE_TABLE, FORMULATION_TABLE, OPTIMIZATION_RESULT`
  - `YGA8VQKU`: `EXPERIMENTAL_DESIGN, OPTIMIZATION_RESULT`
- run classification:
  - `diagnostic-only, not benchmark-valid final output`

## INFERENCE

- Stage2 segmentation is not clean enough at DEV15 scope to treat this refinement cycle as closed.
- The current failure set is not limited to the original inline-table gap.
- The strongest remaining pattern is mixed:
  - selector/evidence:
    - `5GIF3D8W`
    - `7ZS858NS`
    - `BB3JUVW7`
    - `BXCV5XWB`
    - `INMUTV7L`
    - `PA3SPZ28`
    - `RHMJWZX8`
    - `YGA8VQKU`
  - segmentation:
    - `5ZXYABSU`
    - `L3H2RS2H`
  - table extraction quality:
    - `V99GKZEI`
- There is evidence of regression:
  - only `4/15` papers satisfy current design contract
  - `3/15` papers have `table_blocks=0`
  - multiple papers retain table candidates but canonical evidence still drops table/design roles
- There is no strong evidence of over-splitting at DEV15 scale:
  - candidate counts are elevated but not explosively inflated
  - the dominant problem is missing role coverage, not runaway candidate volume

## UNCERTAINTY

- The prior comparable run is not perfectly contract-equivalent:
  - `data/results/20260410_a165cd1/02_dev15_s2_2_replay_validation/semantic_stage2_objects/evidence_blocks/<paper>/evidence_blocks_v1.json`
  - records `design_status.overall=pass`
  - but does not persist `selected_roles` or `missing_or_weak_roles`
  - so prior-vs-current design pass/fail is directionally useful but not a like-for-like role-coverage audit
- This run alone does not separate:
  - selector scoring weakness
  - weak table-summary quality
  - latent Stage2 semantic-object suppression after evidence assembly
- No downstream Stage3/Stage5 replay was run here, so no benchmark-valid closure claim can be made.

## CLOSURE JUDGMENT

- `segmentation_not_fixed`
- why:
  - full DEV15 replay still has `11/15` design fails
  - stable failure surface is broader than the original `5ZXYABSU` inline-table gap
  - several papers with abundant table candidates still fail to materialize required canonical evidence roles
  - full-scope validation is not clean enough to justify docs or memory closure updates

## COPY BACK TO CHATGPT — REMAINING BLOCKERS PACKAGE

### Remaining failing papers

- `5ZXYABSU`
  - blocker: `segmentation`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/5ZXYABSU/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/5ZXYABSU/candidate_blocks_v1.json`
      - copy: `5ZXYABSU__candidate_paragraph__06`, `5ZXYABSU__candidate_paragraph__21`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=5ZXYABSU`

- `L3H2RS2H`
  - blocker: `segmentation`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/L3H2RS2H/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/L3H2RS2H/candidate_blocks_v1.json`
      - copy: first `FORMULATION_RESULT`-like and `OPTIMIZATION_RESULT`-like paragraph candidates
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=L3H2RS2H`

- `5GIF3D8W`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/5GIF3D8W/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/5GIF3D8W/candidate_blocks_v1.json`
      - copy: `5GIF3D8W__candidate_table__01`, `__02`, `__03`
    - `data/cleaned/content_goren_2025/tables/5GIF3D8W/5GIF3D8W__table_01__pdf_table.csv`
    - `data/cleaned/content_goren_2025/tables/5GIF3D8W/5GIF3D8W__table_04__pdf_table.csv`
      - copy: first 20-40 lines
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=5GIF3D8W`

- `7ZS858NS`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/7ZS858NS/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/7ZS858NS/candidate_blocks_v1.json`
      - copy: first 3 table candidates plus first `materials` paragraph candidate
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=7ZS858NS`

- `BB3JUVW7`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/BB3JUVW7/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/BB3JUVW7/candidate_blocks_v1.json`
      - copy: first 2 `section_kind=materials` paragraph candidates and both table candidates
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=BB3JUVW7`

- `BXCV5XWB`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/BXCV5XWB/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/BXCV5XWB/candidate_blocks_v1.json`
      - copy: `BXCV5XWB__candidate_table__01`, `__02`, `__03`
    - `data/cleaned/content_goren_2025/tables/BXCV5XWB/BXCV5XWB__table_01__pdf_table.csv`
    - `data/cleaned/content_goren_2025/tables/BXCV5XWB/BXCV5XWB__table_02__pdf_table.csv`
      - copy: first 20-40 lines
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=BXCV5XWB`

- `INMUTV7L`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/INMUTV7L/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/INMUTV7L/candidate_blocks_v1.json`
      - copy: first 3 table candidates and first `experimental_design`-like paragraph if present
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=INMUTV7L`

- `PA3SPZ28`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/PA3SPZ28/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/PA3SPZ28/candidate_blocks_v1.json`
      - copy: first 3 table candidates and first optimization-like paragraph candidate
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=PA3SPZ28`

- `RHMJWZX8`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/RHMJWZX8/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/RHMJWZX8/candidate_blocks_v1.json`
      - copy: first 3 table candidates and any `section_kind=experimental_design` candidate
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=RHMJWZX8`

- `V99GKZEI`
  - blocker: `table extraction quality`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/V99GKZEI/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/V99GKZEI/candidate_blocks_v1.json`
      - copy: `V99GKZEI__candidate_table__01` plus first 3 prose candidates
    - `data/cleaned/content_goren_2025/tables/V99GKZEI/V99GKZEI__table_01__html_table.csv`
      - copy: first 20-40 lines
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=V99GKZEI`

- `YGA8VQKU`
  - blocker: `selector/evidence`
  - evidence:
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/evidence_blocks/YGA8VQKU/evidence_blocks_v1.json`
      - copy: `coverage_summary`, `selected_roles`, `missing_or_weak_roles`, `design_status`
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/semantic_stage2_objects/candidate_blocks/YGA8VQKU/candidate_blocks_v1.json`
      - copy: `YGA8VQKU__candidate_table__01`, `__02`, `__03`, plus first `section_kind=experimental_design` paragraph candidate
    - `data/results/20260411_312d44b/06_dev15_segmentation_closure_validation/analysis/stage2_prompt_preview_v1.tsv`
      - copy row: `document_key=YGA8VQKU`
