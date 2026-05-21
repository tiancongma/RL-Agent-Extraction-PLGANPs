# Extensible Studied-Variable Carrythrough Repair Plan

Date: 2026-05-20

## Purpose

Repair the final-table handling of paper-specific studied variables that are
not part of the stable core field catalog.

If a paper studies a formulation/process variable, dropping that variable makes
the paper's experimental design uninterpretable. At the same time, these
variables must not all become sparse global core columns because different
papers study different variables.

The repair introduces a stable extensible JSON surface rather than expanding
the core schema.

## Boundary

This repair must not:

- add every paper-specific variable as a fixed core column
- use GT rows to backfill system outputs
- create new formulation rows
- redefine Stage2 semantic authority
- report benchmark-valid performance from partial outputs

This repair may:

- preserve row-local studied variables already present in lawful upstream
  artifacts
- parse explicitly reported row-local variable phrases in reviewer/master-table
  surfaces
- carry unknown but structured variables into a final-table JSON column

## Target Surface

Add `studied_variables_json` to Stage5 final outputs and reviewer UI cards.

Each item should carry:

- `variable_name`
- `variable_family`
- `value`
- `unit`
- `scope`
- `source`
- `evidence_text`

Example for `2RNHC2M5`:

```json
{
  "variable_name": "homogenization_speed",
  "variable_family": "homogenization_speed_rpm",
  "value": "23000",
  "unit": "rpm",
  "scope": "formulation_row",
  "source": "row_identity_description",
  "evidence_text": "Box-Behnken design standard order 8; PVA 50 mg/mL, PLGA 8.13 mg/mL, homogenization speed 23000 rpm"
}
```

## Implementation Steps

1. Add Stage5 helper logic to build `studied_variables_json` from:
   - `table_row_variable_assignments_json`
   - `identity_variables_json`
   - `shared_parameters_json`
   - row-local descriptive text such as `row_identity_description`

2. Preserve arbitrary non-core variables in `studied_variables_json` rather than
   forcing them into fixed columns.

3. Add GPT35/reviewer UI support so `studied_variables_json` and row-local
   phrases display as Value Review fields.

4. Add regression tests for `2RNHC2M5`-style `Homogenization Speed (rpm)`.

5. Run focused tests and an actual GPT35 UI smoke test.

## Acceptance Criteria

- `homogenization speed 23000 rpm` is visible as
  `homogenization_speed_rpm` in Value Review.
- Stage5 can emit `studied_variables_json` without changing the core field
  catalog.
- Existing unknown/shared relation fields remain preserved for audit.
- Focused tests pass.
