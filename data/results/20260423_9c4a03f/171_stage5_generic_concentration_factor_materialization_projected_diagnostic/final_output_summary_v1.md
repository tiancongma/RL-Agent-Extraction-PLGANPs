# Final Output Summary v1

## Scope

This summary describes the controlled Stage5 materialization and duplicate/variant governance layer. Stage5 materializes direct-extraction fields and explicit Stage3-resolved relation fields, then applies conservative closure rules.

## Input

- candidate_input_tsv: `data/results/20260423_9c4a03f/116_stage2_mass_carrythrough_derived_provenance_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- relation_records_tsv: `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/formulation_relation_records_v1.tsv`
- resolved_relation_fields_tsv: `data/results/20260423_9c4a03f/146_stage3_generic_shared_parameter_bundle_diagnostic/resolved_relation_fields_v1.tsv`

## What phase 1 currently handles

- filters rows explicitly marked as non-formulation or characterization-only post-processing rows
- materializes relation-backed descriptive synthesis fields from Stage3 resolved relation outputs
- computes a conservative core-parameter signature from current candidate-row fields
- classifies conservative variant signals into duplicate, optimized, checkpoint/validation, post-processing/measurement, or uncertain review-needed cases
- collapses rows only when signature completeness is high and a unique conservative target is available
- collapses structured DOE/table-derived alternate representations when they clearly duplicate an already richer retained row for the same paper-local row anchor
- applies the validated WFDTQ4VX checkpoint coordinate rule to collapse checkpoint batches that exactly match one design-row formulation identity
- preserves provenance by retaining representative-row metadata, collapsed-variant membership, a row-level decision trace, and a linked downstream-variant record surface for rows excluded from the primary benchmark-facing database

## What phase 1 intentionally does not handle

- semantic inheritance inference beyond Stage3 resolved relation outputs
- generalized DOE coordinate reconciliation beyond the narrow validated WFDTQ4VX checkpoint rule
- Stage 5B benchmark comparison against GT
- modeling-target-specific filtering such as PLGA-only export subsets

## Filtering rules applied

- `explicit_candidate_non_formulation`
- `parent_linked_non_synthesis_descendant_variant`
- `unparented_shared_condition_summary`
- `comparative_summary_reference`
- `characterization_only_post_processing`

## Collapse rules applied

- collapse only if polymer identity and loaded state are known
- collapse only if the conservative core signature has at least five populated components
- collapse duplicate representations only when a clear mixed-source redundancy signal or a unique same-row-anchor match is present
- collapse optimized, checkpoint/validation, or post-processing/measurement variants only when they resolve to exactly one stronger same-core target under the Stage5 policy
- collapse structured `doe_numbered_table_row` rows when they are weak alternate representations of an already richer same-paper row with the same numeric row anchor
- if uncertain, keep rows separate and mark them review-needed in the decision trace

## Decision counts

- kept: `205`
- filtered_non_formulation: `46`
- collapsed_into_existing: `4`
- final_rows: `204`
- downstream_variant_records: `5`
- review_needed_rows: `3`

## Variant class counts

- `duplicate_representation`: `4`
- `post_processing_or_measurement_variant`: `5`
- `uncertain_variant`: `3`

## Final rows by paper

- `5GIF3D8W`: `26`
- `5ZXYABSU`: `9`
- `7ZS858NS`: `1`
- `BB3JUVW7`: `12`
- `BXCV5XWB`: `3`
- `INMUTV7L`: `12`
- `L3H2RS2H`: `21`
- `PA3SPZ28`: `5`
- `QLYKLPKT`: `7`
- `RHMJWZX8`: `2`
- `UFXX9WXE`: `27`
- `V99GKZEI`: `6`
- `WFDTQ4VX`: `30`
- `WIVUCMYG`: `26`
- `YGA8VQKU`: `17`

## Open questions still visible after phase 1

- exact core-signature fields for broader collapse remain unresolved
- baseline versus optimized provenance handling is still conservative
- parent/variant collapse policy is still intentionally narrow and unique-target-based
- relation-driven field materialization is limited to explicit Stage3 resolved descriptive synthesis fields
- DOE-aware coordinate closure still needs a later explicit contract when unique deterministic mapping is not already provided by the narrow validated WFDTQ4VX checkpoint rule
- benchmark comparison still requires the separate Stage 5B comparison step
