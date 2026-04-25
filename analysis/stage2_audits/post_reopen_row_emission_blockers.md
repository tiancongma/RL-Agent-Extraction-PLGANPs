# Post-Reopen Row Emission Blockers

## Executive Conclusion

The explicit authority reopen patch worked. For the bounded lineage
`data/results/20260421_bf6c1a2`, the target emitters now reopen the correct
`S2-2 normalized_table_payloads` and bind them successfully.

The remaining zero-row problem starts **after** reopen and splits into two
separate blocker families:

1. **DOE emitter blocker**
   - `doe_row_expansion_function_unit_v1` binds the reopened payload, then
     hands it to `enumerate_numbered_doe_candidates_for_explicit_tables(...)`.
   - That enumerator only accepts tables with explicit **digit-numbered**
     formulation rows and at least `min_numbered_rows = 8`.
   - This fails differently by paper:
     - `UFXX9WXE`: the authorized DOE table is the factor-level Box-Behnken
       levels table, not a numbered run table.
     - `WIVUCMYG`: the authorized DOE table is row-bearing, but its rows are
       `F1`, `F2`, ... rather than bare digits, so the numeric row parser
       rejects them.
     - `5GIF3D8W`: the authorized DOE table is an optimized-formulation
       characterization table, not a DOE run table with explicit numbered rows.

2. **Non-DOE table emitter blocker**
   - `table_row_expansion_v1` is independently blocked.
   - For `UFXX9WXE` and `WIVUCMYG`, the reopened non-DOE tables are found, but
     emission stops at `len(varying_variables) != 1`.
   - For `5GIF3D8W`, the only table scope is DOE-scoped, so the non-DOE table
     emitter stops at `blocked_by_doe_boundary`.

So the blocker is **not** authority access anymore. It is a combination of:

- DOE row-shape / parser mismatch in the DOE emitter
- too-strict non-DOE varying-variable gate in the table emitter
- paper-specific scope mismatch where the semantically authorized DOE table does
  not actually contain explicit row-enumerable DOE runs

## Per-Paper Emission Trace

### UFXX9WXE

DOE emitter:

- reopened payload:
  - `Table 1`
  - `data/results/20260421_bf6c1a2/00_prellm/semantic_stage2_objects/normalized_table_payloads/UFXX9WXE/payloads/UFXX9WXE__table_10__pdf_table__normalized.csv`
- row-bearing content present:
  - yes, but it is a Box-Behnken factor-level table
  - by inspection it has no explicit formulation/run rows after the leading
    enumerator row is stripped
- last point where material is clearly present:
  - `resolve_authorized_doe_targets(...)` in
    [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:271)
  - binding resolves successfully to the normalized payload
- first point where generation is blocked:
  - `explicit_table_candidate(...)` in
    [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:297)
  - specifically `numbered_idx = first_numbered_row_index(rows)` then
    `if numbered_idx is None: return None` at line `311`
- reason:
  - the reopened Table 1 is not an explicit numbered DOE run table
- classification:
  - `table scope mismatch`

Non-DOE table emitter:

- reopened payload:
  - `Table 2`
  - `.../UFXX9WXE__table_13__pdf_table__normalized.csv`
- row-bearing content present:
  - yes
  - by inspection about `25` digit-numbered rows are visible in the normalized
    payload
- last point where material is clearly present:
  - payload resolved in `run_table_row_expansion(...)` at
    [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:876)
- first point where generation is blocked:
  - `if len(varying_variables) != 1` at
    [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:908)
- actual condition:
  - `varying_variable_count = 2`
  - `polymer concentration (w/v)|surfactant concentration (w/v)`
- classification:
  - `varying-variable-count gate too strict`

### WIVUCMYG

DOE emitter:

- reopened payload:
  - `Table 1`
  - `data/results/20260421_bf6c1a2/00_prellm/semantic_stage2_objects/normalized_table_payloads/WIVUCMYG/payloads/WIVUCMYG__table_01__html_table__normalized.csv`
- row-bearing content present:
  - yes
  - by inspection roughly `26` row-bearing formulation rows exist
  - their first-column labels are `F1 ... F26`, not bare digits
- last point where material is clearly present:
  - `resolve_authorized_doe_targets(...)` in
    [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:271)
- first point where generation is blocked:
  - `row_is_numbered(row)` via `parse_formulation_number(...)` in
    [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:166)
  - only `(\d{1,3})\.?` is accepted
  - therefore `F1`, `F2`, ... do not count as numbered rows
  - `first_numbered_row_index(rows)` returns `None`, then
    `explicit_table_candidate(...)` exits at line `311`
- classification:
  - `payload parsing mismatch`

Non-DOE table emitter:

- reopened payload:
  - `Table 6`
  - `.../WIVUCMYG__table_06__html_table__normalized.csv`
- row-bearing content present:
  - yes
  - by inspection about `3` formulation rows are visible: `F11`, `F19`, `F20`
- last point where material is clearly present:
  - payload resolved in `run_table_row_expansion(...)` at
    [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:876)
- first point where generation is blocked:
  - `if len(varying_variables) != 1` at
    [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:908)
- actual condition:
  - `varying_variable_count = 4`
  - `Pranoprofen (PF) concentration|PLGA concentration|PVA concentration|pH value`
- classification:
  - `varying-variable-count gate too strict`

### 5GIF3D8W

DOE emitter:

- reopened payload:
  - `Table 4`
  - `data/results/20260421_bf6c1a2/00_prellm/semantic_stage2_objects/normalized_table_payloads/5GIF3D8W/payloads/5GIF3D8W__table_04__pdf_table__normalized.csv`
- row-bearing content present:
  - yes, but it is an optimized-formulation characterization table
  - by inspection it does not contain explicit DOE run rows
- last point where material is clearly present:
  - `resolve_authorized_doe_targets(...)` in
    [doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py:271)
- first point where generation is blocked:
  - `explicit_table_candidate(...)` in
    [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:297)
  - `first_numbered_row_index(rows)` returns `None` and the function exits at
    line `311`
- classification:
  - `table scope mismatch`

Non-DOE table emitter:

- reopened payload:
  - the table scope remains `Table 4`
- row-bearing content present:
  - yes, but the non-DOE emitter never uses it for row generation
- last point where material is clearly present:
  - scope iteration begins in `run_table_row_expansion(...)`
- first point where generation is blocked:
  - DOE boundary guard:
    `if bool(boundary.get("is_doe")): skip_reason = "blocked_by_doe_boundary"`
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:871)
- reason:
  - the only authorized table scope is DOE-scoped, so the non-DOE emitter is
    intentionally bypassed
- classification:
  - `DOE eligibility gate too strict` for the non-DOE emitter path

## Shared vs Paper-Specific Blockers

Shared blockers:

- `table_row_expansion_v1` requires exactly one varying variable:
  - `if len(varying_variables) != 1`
  - this blocks both `UFXX9WXE` and `WIVUCMYG`
- `doe_row_expansion_function_unit_v1` delegates to an enumerator that expects
  explicit numbered DOE rows with a numeric first cell and at least `8` rows

Paper-specific blockers:

- `UFXX9WXE`
  - DOE target is a factor-level Box-Behnken table, not an explicit run table
- `WIVUCMYG`
  - DOE target is correct and row-bearing, but uses `F`-coded row labels that
    the numeric parser rejects
- `5GIF3D8W`
  - DOE target is an optimized-characterization table, not an explicit DOE run
    table
  - the non-DOE emitter is separately blocked because the only scope is still
    marked DOE

## Dominant Blocker Ranking

1. **DOE explicit-row parser / selector assumptions**
   - impact: blocks DOE recovery for all three primary papers
   - responsible locations:
     - `parse_formulation_number(...)`
     - `first_numbered_row_index(...)`
     - `explicit_table_candidate(...)`
     - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:166)
     - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:187)
     - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:297)

2. **Non-DOE table-expansion exact-one-varying-variable gate**
   - impact: blocks the reopened sequential-child tables for `UFXX9WXE` and
     `WIVUCMYG`
   - responsible location:
     - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:908)

3. **DOE boundary block inside the non-DOE table emitter**
   - impact: blocks the only scope for `5GIF3D8W` on the non-DOE path
   - responsible location:
     - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:871)

## Exact Code Locations Responsible

- DOE emitter invoked from:
  - [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:1332)
- Non-DOE table emitter invoked from:
  - [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:1353)

Post-reopen blocker branches:

- Numeric DOE row parser:
  - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:166)
- First numbered-row finder:
  - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:187)
- DOE table candidate acceptance:
  - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:297)
- DOE explicit-target loop:
  - [build_numbered_doe_row_candidates_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py:657)
- Table-emitter DOE boundary block:
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:871)
- Table-emitter varying-variable-count gate:
  - [table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py:908)

## FACTS

- In the bounded lineage, DOE recovery summaries now show:
  - `binding_success = yes`
  - `normalized_payload_used = yes`
  - `reopen_source_type = normalized_table_payloads_explicit`
- `UFXX9WXE` DOE target is `Table 1`; non-DOE table scope is `Table 2`
- `WIVUCMYG` DOE target is `Table 1`; non-DOE table scope is `Table 6`
- `5GIF3D8W` DOE target is `Table 4`; no non-DOE formulation scope survives
- `WIVUCMYG Table 1` contains `F1 ... F26` row labels by direct payload
  inspection
- `UFXX9WXE Table 1` and `5GIF3D8W Table 4` do not contain explicit run rows
  acceptable to the current DOE enumerator
- `UFXX9WXE Table 2` and `WIVUCMYG Table 6` resolve successfully but are
  blocked in `table_row_expansion_v1` before assignment extraction

## INFERENCES

- The dominant remaining problem is emitter logic, not authority reopen
- `UFXX9WXE` and `5GIF3D8W` would still fail DOE emission even with a more
  permissive numeric parser, because the semantically authorized DOE table does
  not appear to contain explicit DOE run rows
- `WIVUCMYG` is the clearest case where reopened row-bearing material exists
  but the DOE parser rejects the row-label format
- The non-DOE sequential-child path for `UFXX9WXE` and `WIVUCMYG` is blocked by
  a shared exact-one-varying-variable assumption rather than by payload absence

## UNCERTAINTIES

- If the table-expansion varying-variable-count gate were relaxed, later gates
  such as `missing_candidate_values` might still block some rows
- For `UFXX9WXE`, it is possible that a different table than the authorized
  DOE `Table 1` contains the true row-enumerable DOE runs, but this audit did
  not broaden beyond the bounded lineage’s surviving scopes
- For `5GIF3D8W`, the DOE scope itself may be semantically accurate at a high
  level while still being unusable for explicit row enumeration

## NOT RECOMMENDED YET

- Do not change authority reopen again; that path is no longer the blocker
- Do not broaden into selector or prompt auditing; those are upstream of the
  current failure
- Do not propose Stage3 or Stage5 changes from this evidence
- Do not collapse all failures into a single “DOE broken” statement; the
  bounded lineage shows both DOE-emitter and non-DOE-emitter blocks, and they
  differ by paper
