# Post-Reopen Row Emission Repair Report

## Executive Conclusion

The bounded post-reopen row-emission repair succeeded for the two targeted
positive cases and tightened the failure surface for the negative case.

Using the maintained `S2-7` runner on the bounded validation lineage:

- before:
  - `data/results/20260421_bf6c1a2/05_s2_7`
- after:
  - `data/results/20260421_bf6c1a2/11_s2_7_row_emission_repair_v5`

observed changes were:

- `UFXX9WXE`
  - `table_row_expansion_v1` now emits `26` rows from reopened `Table 2`
  - total Stage2 rows: `2 -> 28`
- `WIVUCMYG`
  - DOE recovery now emits `26` rows from `F1..F26`
  - coded-table decode is active:
    - `coding_table_used = Table 5`
    - `run_table_used = Table 1`
    - `decode_status = decoded`
  - total Stage2 rows: `3 -> 29`
- `5GIF3D8W`
  - still emits `0` governed recovery rows
  - failure is now more specific and correct:
    - `table_row_expansion skip_reason = insufficient_explicit_row_labels`

No obvious row explosion was observed in the bounded validation:

- `UFXX9WXE` emitted `26` rows from an explicit 26-row formulation table
- `WIVUCMYG` emitted `26` rows from an explicit `F1..F26` DOE run matrix

## What Was Repaired

The patch addressed four localized post-reopen blockers:

1. DOE numbered-row parsing
   - added support for common explicit formulation labels:
     - bare digits
     - digit plus punctuation
     - `F<number>`
     - `f<number>`

2. Bounded coded DOE decode
   - restored a governed two-table decode path when artifacts support it
   - for `WIVUCMYG`, the emitter now decodes:
     - run table: `Table 1`
     - coding table: `Table 5`

3. Non-DOE multi-variable formulation-table expansion
   - removed the hard dependence on `len(varying_variables) == 1` as the only
     usable row-emission path
   - explicit row-bearing formulation tables can now emit direct rows from the
     reopened authority payload when the table structure itself is sufficient

4. DOE boundary fallback
   - DOE-scoped tables no longer suppress the table emitter automatically when
     DOE emission produced zero rows
   - fallback is now explicit and audited

## Which Files Changed

- [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py)
- [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py)
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)

## DOE Numbered-Row Parser Changes

File:
- [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py)

Changes:
- replaced the numeric-only row-label parser with a bounded label parser that
  also accepts `F`-style formulation labels
- extended coded-level normalization to accept exact coded values written as
  decimal text such as `2.0`
- preserved the conservative table requirement that the row-bearing table must
  still look like an explicit DOE/design table rather than inventing unseen
  rows

Observed effect:
- `WIVUCMYG Table 1` now emits `26` DOE rows from explicit `F1..F26` labels

## DOE Coded-Table Decode Changes

Files:
- [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py)
- [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py)

Changes:
- added a bounded decode resolver that:
  - detects coded factor columns in the reopened run table
  - searches the reopened normalized payload family for a compatible coding
    table
  - decodes coded values only when factor mapping is unambiguous
- emitted audit fields:
  - `coding_table_used`
  - `run_table_used`
  - `decode_status`
  - `decode_failure_reason`

Observed effect:
- `WIVUCMYG`
  - `coding_table_used = Table 5`
  - `run_table_used = Table 1`
  - `decode_status = decoded`

## Non-DOE Multi-Variable Table Gate Changes

File:
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)

Changes:
- kept the old single-variable selection path as a fallback
- added a new mainline direct row-emission path that:
  - reads explicit formulation rows directly from reopened authority payloads
  - derives variable columns from table-local header structure
  - emits row-local assignments when the table is clearly row-bearing and
    formulation-row-like
- stopped using `unsupported_varying_variable_count:*` as the only outcome for
  multi-variable formulation tables

Observed effect:
- `UFXX9WXE Table 2` now emits `26` rows despite
  `varying_variable_count = 2`

## DOE Boundary Fallback Changes

Files:
- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)

Changes:
- `table_row_expansion_v1` now receives DOE summary state from the same
  projection pass
- when DOE scope exists:
  - if DOE rows were emitted, table fallback is blocked explicitly
  - if DOE rows were not emitted, table fallback is allowed and audited
- added audit fields:
  - `doe_path_attempted`
  - `doe_rows_emitted`
  - `fell_back_to_table_expansion`
  - `fallback_reason`

Observed effect:
- `UFXX9WXE`
  - `fell_back_to_table_expansion = yes`
  - `fallback_reason = doe_emitted_zero_rows`
  - fallback succeeded on `Table 2`
- `WIVUCMYG`
  - DOE succeeded, so table path is now blocked with the more correct
    `blocked_by_successful_doe_emission`
- `5GIF3D8W`
  - fallback was attempted
  - failure is now `insufficient_explicit_row_labels`

## Before/After Bounded Validation

Validation source:
- `data/results/20260421_bf6c1a2/04_s2_6`

Validation outputs:
- before:
  - `data/results/20260421_bf6c1a2/05_s2_7`
- final after:
  - `data/results/20260421_bf6c1a2/11_s2_7_row_emission_repair_v5`

### UFXX9WXE

- before:
  - DOE rows emitted: `0`
  - table rows emitted: `0`
  - Stage2 total rows: `2`
- after:
  - DOE rows emitted: `0`
  - table rows emitted: `26`
  - Stage2 total rows: `28`
- key after-state:
  - `Table 2`
  - `rows_emitted = 26`
  - `fell_back_to_table_expansion = yes`

### WIVUCMYG

- before:
  - DOE rows emitted: `0`
  - table rows emitted: `0`
  - Stage2 total rows: `3`
- after:
  - DOE rows emitted: `26`
  - table rows emitted: `0`
  - Stage2 total rows: `29`
- key after-state:
  - DOE rows came from explicit `F1..F26`
  - `decode_status = decoded`
  - `coding_table_used = Table 5`
  - `run_table_used = Table 1`

### 5GIF3D8W

- before:
  - DOE rows emitted: `0`
  - table rows emitted: `0`
  - Stage2 total rows: `2`
- after:
  - DOE rows emitted: `0`
  - table rows emitted: `0`
  - Stage2 total rows: `2`
- key after-state:
  - no false fallback emission
  - more specific post-reopen failure:
    - `insufficient_explicit_row_labels`

## Residual Risks

- `UFXX9WXE` direct table expansion currently emits bounded row-bearing rows,
  but the compatibility-field mapping remains selective:
  - `plga_mass_mg` is populated
  - `drug_feed_amount_text` is populated
  - not every leading design column is yet mapped into a legacy core field
- `WIVUCMYG Table 6` still does not emit rows because it is a selected
  measurement table without row-local formulation-variable columns
- the new multi-variable table path is still intentionally conservative:
  - it requires explicit row labels
  - it requires recoverable pre-measurement variable columns
  - it does not invent missing per-row variable assignments

## What Is Still Not Solved

- this patch does not solve all DOE/table row-emission cases
- it does not repair papers where the semantically authorized table is not a
  row-enumerable run matrix
- it does not reconstruct row-level variables from optimized summary tables
- it does not change Stage3 or Stage5 behavior
- it does not attempt broad downstream deduplication between DOE and later
  sequential-child tables

The remaining clearly unsolved bounded target is:
- `5GIF3D8W`
  - explicit reopen works
  - fallback is now lawful and auditable
  - but no explicit row-bearing formulation matrix exists for governed row
    emission from the authorized table
