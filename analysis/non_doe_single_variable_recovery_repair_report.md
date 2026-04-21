# Non-DOE Single-Variable Recovery Repair Report

## Executive Conclusion

The bounded repair succeeded for the general non-DOE failure family represented
by `5GIF3D8W`.

Using the maintained `S2-7` runner on the bounded validation lineage:

- before:
  - `data/results/20260421_bf6c1a2/11_s2_7_row_emission_repair_v5`
- after:
  - `data/results/20260421_bf6c1a2/16_s2_7_non_doe_single_variable_repair_v5`

the repaired path now does two reusable things:

1. recovers explicit formulation-bearing anchor rows from a column-oriented
   optimized/reference table
2. recovers independent single-variable formulation families only when the
   source text gives all of the following explicitly:
   - the variable axis
   - the tested levels
   - the held-constant baseline context
   - a one-parameter-at-a-time contract

Observed bounded outcome:

- `5GIF3D8W`
  - table-row expansion: `0 -> 13`
  - explicit anchor rows: `4`
  - single-variable rows: `9`
- `UFXX9WXE` guard
  - unchanged at `26`
  - no non-DOE single-variable rows emitted
- `WIVUCMYG` secondary quiet check
  - unchanged non-DOE path
  - DOE remains the governing emitted path

No Cartesian explosion was introduced. The new rows for `5GIF3D8W` come from:

- one explicit optimized formulation table
- three independently supported single-variable families

## General Failure Family Definition

This repair targets non-DOE formulation studies with all of the following
properties:

1. an explicit formulation-bearing baseline / optimized / reference table is
   present
2. that table may be column-oriented rather than row-oriented
3. the paper also reports one-variable-at-a-time formulation exploration
4. the explored variable groups are independently supported by explicit text
   rather than by a DOE run matrix
5. those groups must not be recovered through Cartesian expansion
6. recovery is legal only when the paper explicitly supports:
   - which variable changes
   - which levels were tested
   - which baseline conditions were held constant

This is broader than `5GIF3D8W`, but narrower than generic prose mining.

## Why 5GIF3D8W Is Only a Diagnostic Anchor

`5GIF3D8W` is the paper that made the family easy to see:

- `Table 4` is a real formulation-bearing optimized table, but it is
  column-oriented
- the source text explicitly states:
  - `polymer amount (25, 50, 100, and 200 mg)`
  - `concentration of stabilizer (0.5, 0.75, 1.0, and 2.0% w/v)`
  - `etoposide amount (2.5, 5, 10, and 20 mg)`
  - `Only one parameter was changed in each series of experiments`
- the optimized baseline preparation is also explicit:
  - polymer `50 mg`
  - etoposide `5 mg`
  - stabilizer `1.0% w/v`

The patch does not use the paper key or any expected count. It uses only the
general contract above.

## What Reusable Logic Was Added

Changed files:

- [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)

Reusable additions:

1. column-oriented explicit table-anchor recovery
   - detects formulation-bearing tables where formulations are columns and
     measured properties are rows
   - emits formulation rows from header-column identity, not from
     hallucinated combinations

2. bounded non-DOE single-variable recovery
   - requires:
     - `has_variable_sweep = true`
     - explicit one-parameter-at-a-time wording
     - explicit tested level lists tied to named variables
     - explicit held-constant baseline assignments
     - successful explicit anchor recovery first
   - emits one independent family per supported variable axis
   - skips the baseline-equivalent level so the recovery does not duplicate the
     anchor formulation needlessly

3. passive semantic-signals preservation into compatibility projection
   - needed because the normalization layer was dropping `semantic_signals`
     before `table_row_expansion_v1` could read them

4. execution-ledger expansion
   - adds audit fields for:
     - `explicit_table_rows_emitted`
     - `non_doe_single_variable_groups_detected`
     - `single_variable_recovery_attempted`
     - `single_variable_rows_emitted`
     - `single_variable_recovery_source_type`
     - `single_variable_recovery_failure_reason`
     - `held_constant_context_source`
     - `variable_axis_detected`

## Explicit Table-Anchor Recovery Result

For `5GIF3D8W`, the repaired table path now treats `Table 4` as a real
formulation anchor table even though it is column-oriented.

Recovered anchor rows:

- `PLGA 50/50 / Empty`
- `PLGA 50/50 / Drug loaded`
- `PLGA 75/25 / Empty`
- `PLGA 75/25 / Drug loaded`

This is a general table-orientation repair, not a special case for one paper.

## Single-Variable Recovery Result

For `5GIF3D8W`, the new non-DOE sweep path detects three supported groups:

- `polymer amount`
- `concentration of stabilizer`
- `etoposide amount`

Emitted rows:

- polymer family:
  - `25 mg`
  - `100 mg`
  - `200 mg`
- stabilizer family:
  - `0.5 % w/v`
  - `0.75 % w/v`
  - `2.0% w/v`
- etoposide family:
  - `2.5 mg`
  - `10 mg`
  - `20 mg`

Held-constant context source:

- `source_text_baseline_clause`

Single-variable recovery source type:

- `explicit_narrative_single_variable_contract`

This remains bounded:

- no cross-family combinations
- no guessed fixed values
- no figure OCR
- no broad prose regex mining beyond the explicit contract window

## Guard-Paper Result

Guard paper:

- `UFXX9WXE`

Reason for selection:

- it already has a legitimate non-DOE table-expansion path
- it should not trigger the new non-DOE single-variable recovery unless the
  same explicit one-parameter-at-a-time contract exists

Observed result:

- total rows unchanged: `26 -> 26`
- `single_variable_recovery_attempted = yes`
- `single_variable_rows_emitted = 0`
- `single_variable_recovery_failure_reason = single_variable_contract_not_found`

This is the intended quiet behavior.

Secondary quiet check:

- `WIVUCMYG`
- DOE remained successful and continued to block non-DOE fallback:
  - `skip_reason = blocked_by_successful_doe_emission`

## Before/After Bounded Validation

Validation source:

- `data/results/20260421_bf6c1a2/04_s2_6`

Final validation output:

- `data/results/20260421_bf6c1a2/16_s2_7_non_doe_single_variable_repair_v5`

### 5GIF3D8W

- before:
  - table rows emitted: `0`
  - Stage2 total rows: `2`
- after:
  - explicit table-anchor rows emitted: `4`
  - non-DOE single-variable groups detected: `3`
  - single-variable rows emitted: `9`
  - table-row expansion total: `13`
  - Stage2 total rows: `15`

### UFXX9WXE

- before:
  - table-row expansion total: `26`
- after:
  - table-row expansion total: `26`
  - no single-variable rows emitted

### WIVUCMYG

- before:
  - DOE rows emitted: `26`
  - non-DOE table rows emitted: `0`
- after:
  - DOE rows emitted: `26`
  - non-DOE table rows emitted: `0`

## Residual Limitations

- `5GIF3D8W` anchor rows still represent formulation identity mainly through
  `identity_variables_json`; they do not yet populate every legacy core field
  because the optimized table is characterization-heavy.
- the single-variable recovery path currently uses explicit narrative text,
  not generic figure mining.
- the path requires an explicit one-parameter-at-a-time contract and explicit
  baseline assignments; papers without that support will still fail loudly.
- the patch does not touch Stage3 or Stage5.

## Risks / Scope Boundaries

- the new path is intentionally conservative and will miss papers where the
  sweep family is only implicit.
- it is not a generic caption miner or OCR extractor.
- it is not a DOE substitute.
- it does not construct unsupported combinations across variable groups.
- it does not override the existing DOE path; it only adds a bounded non-DOE
  family recovery when DOE did not emit rows and the non-DOE contract is
  explicit.
