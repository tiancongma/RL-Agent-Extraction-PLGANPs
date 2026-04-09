# INMUTV7L Explanation Audit

## Scope

This is an explanation-only audit based on governed existing artifacts. No code, prompts, contracts, or function units were changed.

## Authoritative Sources Used

DEV15 audit outputs:

- `docs/dev15_llm_capability_audit_v1.tsv`
- `docs/dev15_llm_capability_audit_summary.md`

Maintained Stage2 DEV15 coverage run:

- run dir: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration`
- semantic summary: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- semantic objects: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
- raw response: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_stage2_objects/raw_responses/INMUTV7L__stage2_v2_raw_response.json`
- completed Stage2 surface: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- projection trace: `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration/semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`

Paper table evidence:

- `data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_15__pdf_table.csv`
- `data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_06__pdf_table.csv`
- `data/cleaned/content_goren_2025/text/INMUTV7L.pdf.txt`

No paper-specific downstream diagnostic note was found inside the April 2 DEV15 Stage2 run beyond the normal semantic summary and compatibility trace surfaces.

## A. Exact Audit Classification

From `docs/dev15_llm_capability_audit_v1.tsv`:

- `overall_status`: `needs_new_function_unit`
- `primary_gap_type`: `missing_relation_marking`
- `notes`: `Stage2 names Formulation 1-12 and detects surfactant and PEG factors, but it does not bind those factors to the numbered variants strongly enough for reliable downstream reconstruction.`

From `docs/dev15_llm_capability_audit_summary.md`:

- `INMUTV7L` is the clearest example under the summary sentence:
  - `Numbered variant studies are still fragile when the LLM emits formulation numbers without binding the row-level factor assignments back onto those numbered instances.`

## B. Exact Evidence That Caused The Classification

The governing evidence is the mismatch between what the paper table contains and what current Stage2 keeps row-bound.

Paper-side evidence:

- `INMUTV7L__table_15__pdf_table.csv` is a 12-row formulation table with:
  - formulation number
  - polymer block
  - surfactant
  - particle size
  - PDI
  - zeta potential
  - EE
- Example rows:
  - `1 ... PVA ... 234.1 ± 0.5 ... 93.4`
  - `2 ... Tween80 ... 146.0 ± 0.6 ... 87.5`
  - `3 ... Lutrol ... 159.5 ± 0.8 ... 85.1`
  - rows 4-6 under `PLGA-5%`
  - rows 7-9 under `PLGA 10%`
  - rows 10-12 under `PLGA 15%`

Raw Stage2 semantic evidence:

- `semantic_stage2_v2_summary.tsv` reports:
  - `formulation_count=14`
  - `variable_count=23`
  - `relation_hint_count=15`
  - `multi_component_formulation_count=1`
- The raw response creates:
  - `FC003` through `FC014` for `Formulation 1` through `Formulation 12`
  - identity signals for formulation numbers
  - general factor candidates for surfactant type and PEG percentage
- But the same raw response explicitly says for `FC003` to `FC013`:
  - `Specific composition details ... are not fully enumerated`
- It only binds a row-specific surfactant and measurements to `FC014` (`Formulation 12 with Lutrol`).

Completed Stage2 evidence:

- In `weak_labels__v7pilot_r3_fixparse.tsv`:
  - `FC003` to `FC013` have `identity_variables_json` containing only formulation number
  - `polymer_name_raw`, `surfactant_name_value_text`, and all measurements are blank for those rows
  - only `FC014` carries row-bound measurement values
- `FC001` becomes a compressed aggregate row with:
  - `polymer_name_raw = PLGA | PEG`
  - `surfactant_name_value_text = PVA | Tween 80® | Lutrol F68`
  - this captures the study-level factor set, not per-row formulation identity
- The projection trace confirms this:
  - `FC001` polymer and surfactant fields are `compressed`
  - `FC003` and `FC004` rows show polymer, surfactant, drug, and measurements as `unavailable`

That is why the audit called this a relation-marking problem rather than a pure variable-detection problem.

## C. Does The Paper Actually Contain A Clean Row-Wise Formulation Table?

Yes.

The best evidence is `data/cleaned/goren_2025/tables/INMUTV7L/INMUTV7L__table_15__pdf_table.csv`, which is structurally clear:

- rows are formulation numbers `1` through `12`
- polymer identity is given in grouped blocks:
  - `PLGA 503 H`
  - `PLGA-5%`
  - `PLGA 10%`
  - `PLGA 15%`
- within each polymer block, surfactant identity cycles across:
  - `PVA`
  - `Tween80`
  - `Lutrol`
- each row has direct measurements:
  - size
  - PDI
  - zeta potential
  - EE

So the table shape itself is clear. The main complexity is that the polymer label is expressed as a block header row spanning the next three formulation rows, not repeated in every individual row cell.

## D. What Current Stage2 Represents Correctly vs Incorrectly

### Row identity

Partially correct.

- Stage2 does create formulation instances `FC003` to `FC014` for `Formulation 1` to `Formulation 12`.
- In the completed Stage2 surface, those rows retain formulation-number identity in `identity_variables_json`.

### Polymer identity

Mostly incorrect at row level.

- The paper table clearly groups rows into four polymer blocks.
- Current Stage2 does not attach those polymer-block identities to rows `FC003` to `FC014`.
- Instead, the completed Stage2 surface compresses polymer identity into the general parent row `FC001` as `PLGA | PEG`.

### Surfactant identity

Mostly incorrect at row level.

- The paper table clearly gives one surfactant per formulation row.
- Current Stage2 keeps surfactant choices mostly at the general study row `FC001` as `PVA | Tween 80® | Lutrol F68`.
- Only `FC014` is row-bound to `Lutrol`.

### Which variables differ across rows

Detected, but not represented as formulation-universe structure.

- Raw Stage2 detects the changing variables:
  - surfactant type
  - PEG percentage
- But those variables are mostly attached to `FC001` as shared factor inventory rather than distributed across the numbered formulation rows.

### Relation between table rows and formulation instances

Weak.

- The numbered rows exist as instances.
- But the relation from row number to polymer block and row number to surfactant assignment is not preserved for most rows.
- That is the key failure for downstream deterministic use.

## E. Precise Failure Mode

Closest category:

- `factor-to-row relation weak`

Justification:

- Row identities are not missing. They exist as `Formulation 1` through `Formulation 12`.
- Variables are not missing either. Surfactant type and PEG percentage are detected.
- The main failure is that current Stage2 does not bind those detected varying factors onto the numbered formulation rows in a way the compatibility projection can preserve.

Secondary description that is also true:

- `variable changes detected but not represented as formulation-universe structure`

But the sharpest primary description remains:

- the factor-to-row relation is weak

## F. Root Cause Interpretation

Best supported interpretation:

- `2. the current LLM output underrepresents an otherwise clear table`
- with a secondary downstream consequence:
  - `4. existing function units are insufficient for this pattern`

Why this is not mainly paper ambiguity:

- the table itself is clear enough to support row-wise reconstruction
- the important identities are present in the table
- the ambiguity comes from grouped polymer headers, not from missing row structure

Why this is not mainly contract strictness:

- the current contract can already consume row-wise formulation tables when the Stage2 output actually binds row identities to row-level components and variables
- here the completed Stage2 rows are too sparse because the raw semantic representation did not preserve the row-to-factor linkage strongly enough

Why the audit still said `needs_new_function_unit`:

- under current maintained behavior, there is no existing structured handler that safely reconstructs the grouped-header table pattern once the Stage2 output has collapsed row-level polymer and surfactant assignments into a general parent row
- so the blocker is not that the paper is too hard
- it is that the current Stage2 representation is only partially adequate for this table pattern, and no existing maintained unit repairs that pattern downstream

## Bottom Line

`INMUTV7L` was not classified as needing further work because the paper lacks a formulation table. It was classified that way because the paper has a usable 12-row formulation table, but the current Stage2 output keeps most of the table logic only as:

- numbered row identities
- general study-level factor inventory

and does not preserve the decisive row-bound links:

- row number -> polymer block
- row number -> surfactant choice
- row number -> row measurements

for most of the rows.
