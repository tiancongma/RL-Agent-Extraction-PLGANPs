# Memory Failure Trace v2

## Scope
- Task scope was limited to memory tooling and validation.
- No Stage2, Stage3, or Stage5 pipeline semantics were modified.
- Validation rules were preserved; no silent bypass was added.

## Reproduced Commands
1. `python3 src/utils/build_mem_v1.py`
2. `python3 src/utils/check_mem_v1.py`

## Raw Error Capture

### Command 1
```text
mem_dir=/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/mem/v1
status=initialized
mode=rebuild
refreshing=dec.tsv,err.tsv,idx.tsv,lin.tsv,prm.tsv,run.tsv
Traceback (most recent call last):
  File "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py", line 752, in <module>
    raise SystemExit(main())
  File "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py", line 705, in main
    run_rows, lin_rows = build_run_rows(sources)
  File "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py", line 348, in build_run_rows
    parent_run = parent_run_from_path(path)
  File "/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py", line 329, in parent_run_from_path
    for ancestor in path.parents[1:]:
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/pathlib.py", line 634, in __getitem__
    if idx < 0 or idx >= len(self):
TypeError: '<' not supported between instances of 'slice' and 'int'
```

### Command 2
Initial `check_mem_v1.py` emitted `1332` `ERROR:` lines.

All initial errors had the same concrete form:
```text
ERROR: Missing source file referenced by <table>: <source_file>
```

Representative exact lines from the initial output:
```text
ERROR: Missing source file referenced by idx.tsv: data/results/run_20260312_1030_455ac37_targeted5_stage2_regression_v1/RUN_CONTEXT.md
ERROR: Missing source file referenced by idx.tsv: data/results/run_20260331_1156_03e5d25_threepaper_stage2_v2_live_gemini_structural_eval_v2/RUN_CONTEXT.md
ERROR: Missing source file referenced by idx.tsv: data/results/20260401_5d9f4e6/09_dev15_count_validation/RUN_CONTEXT.md
ERROR: Missing source file referenced by idx.tsv: data/results/20260407_ab12cd3/21_qlyk_table_selection_fix_validation_v3/RUN_CONTEXT.md
ERROR: Missing source file referenced by err.tsv: data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/42_dev15_v2_l3_value_gt_annotation_v4_repaired/run_20260320_1738_f54824a_dev15_v2_l3_value_gt_annotation_v4_repaired_v1/RUN_CONTEXT.md
ERROR: Missing source file referenced by err.tsv: data/results/run_20260411_1415_9507f0a_rules_only_comparator_fulltext_v1/RUN_CONTEXT.md
```

## Failure Classification

### F1. `build_mem_v1.py` crash
- Failure type: `path parsing bug`
- Exact code path: [src/utils/build_mem_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py:377)
- Root cause:
  - `parent_run_from_path` sliced `Path.parents` with `path.parents[1:]`.
  - `Path.parents` is not sliceable in Python 3.9, so rebuild aborted before any run-lineage normalization could complete.

### F2. `build_mem_v1.py` latent run-layout mismatch uncovered during trace
- Failure type: `legacy run path mismatch`
- Exact code path: [src/utils/build_mem_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py:270), [src/utils/build_mem_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py:402)
- Root cause:
  - the builder only recognized legacy `run_*` IDs in free text before repair
  - current governed sources are mostly v2 bucket/child `RUN_CONTEXT.md` files
  - some historical legacy IDs also use 6-digit time tokens
  - without explicit run-id parsing, v2 child runs and some legacy child runs would be skipped or mis-parented

### F3. `check_mem_v1.py` initial missing-source failure
- Failure type: `missing source file`
- Exact code path: [src/utils/check_mem_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/check_mem_v1.py:76)
- Root cause:
  - the pre-rebuild `data/mem/v1/*.tsv` tables referenced `RUN_CONTEXT.md` paths that are not present in the current checkout
  - classification of all `1332` missing references from `HEAD:data/mem/v1/*.tsv`:
    - `historical_missing_source`: `0`
    - `stale_memory_entry`: `1332`
    - `path_normalization_bug`: `0`
    - `invalid_contract_reference`: `0`
- Governed handling applied:
  - stale memory entries were removed by rebuilding memory from the current governed source set
  - no validator suppression was added

### F4. Post-fix intermediate validation failure
- Failure type: `schema inconsistency`
- Exact code path: [src/utils/build_mem_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/utils/build_mem_v1.py:441)
- Root cause:
  - some v2 bucket parents (`20260413_6971522`, `20260417_385b6e1`, `20260418_3579206`) have child `RUN_CONTEXT.md` files but no bucket-root `RUN_CONTEXT.md`
  - lineage rows were correctly derived, but corresponding parent bucket run rows were absent from `run.tsv`
  - `check_mem_v1.py` then failed on `lin.tsv parent_run missing from run.tsv`
- Governed handling applied:
  - synthetic parent bucket rows were materialized only for explicit v2 bucket parents
  - each synthetic bucket row is sourced from an existing child `RUN_CONTEXT.md`
  - non-v2 missing parents still fail loudly

## Source-Reference Classification

### Initial missing-reference inventory from `HEAD:data/mem/v1/*.tsv`
- total missing source references: `1332`
- `historical_missing_source`: `0`
- `stale_memory_entry`: `1332`
- `path_normalization_bug`: `0`
- `invalid_contract_reference`: `0`

Interpretation:
- the validator was correct to fail
- the problem was stale memory state, not a broken existence check

## Function Behavior After Patch

### Legacy path
```text
run_id=run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1
parent=
```

### New v2 child path
```text
run_id=01_s2_2
parent=20260418_9538ec2
```

### Invalid path
```text
ValueError: RUN_CONTEXT lacks supported run_id
```

## Repair Summary
- `parent_run_from_path` now uses explicit ancestor inspection plus explicit ancestor path tokens only.
- run-id parsing now supports:
  - legacy `run_*` IDs with 4-digit or 6-digit time tokens
  - v2 bucket IDs `YYYYMMDD_<hash>`
  - v2 child IDs `NN_<cue>` / `NNN_<cue>`
- v2 bucket parents without bucket-root `RUN_CONTEXT.md` now receive explicit synthetic `run.tsv` rows sourced from child lineage evidence.
- No validation rule was weakened.
