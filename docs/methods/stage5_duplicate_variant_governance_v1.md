# Stage5 Duplicate And Variant Governance v1

## Scope

This note defines the controlled Stage5 governance layer for benchmark-facing
final output.

The layer operates only inside the active final-output builder:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`

It does not redesign Stage2 extraction, Stage3 relation materialization, or the
benchmark comparison node.

## Purpose

Stage5 must prevent benchmark-facing over-counting when multiple candidate rows
are only alternate representations or downstream variants of the same
formulation identity.

The governed responsibility is:

1. detect conservative equivalence cases within one paper
2. classify the variant relationship
3. retain one benchmark-facing row
4. preserve traceability for collapsed rows

## Variant Classes

The first governed version supports only these classes:

- `duplicate_representation`
- `optimized_variant`
- `checkpoint_or_validation_variant`
- `post_processing_or_measurement_variant`
- `uncertain_variant`

No broader taxonomy is introduced here.

Provenance labels such as baseline, optimized, checkpoint, or post-processing
do not by themselves define final formulation identity.

## Retention Policy

Retention remains conservative.

- Filter rows explicitly marked as non-formulation or characterization-only
  post-processing rows.
- Collapse `duplicate_representation` rows only when the overlap is explicit.
  Current supported cases are:
  - clear mixed-source same-core overlap
  - weak numbered-DOE/table alternate rows that match exactly one richer row by
    paper-local row anchor
- Collapse `optimized_variant`, `checkpoint_or_validation_variant`, and
  `post_processing_or_measurement_variant` only when Stage5 finds exactly one
  stronger same-core target inside the same paper under the conservative core
  signature policy.
- If Stage5 cannot identify one unique safe target, keep the row separate and
  mark it `review_needed = yes`.

This means Stage5 now owns duplicate / variant governance for benchmark-facing
final output, but it still avoids broad automatic merging.

## Non-Goals

This layer does not:

- perform broad rule-heavy formulation reconstruction
- decode full DOE coordinate structure when Stage5 lacks a unique deterministic
  target
- override Stage2 row discovery responsibilities
- turn Stage3 provenance into a full closure engine
- introduce modeling-specific export policy

## Traceability Requirements

The Stage5 decision trace must expose, per source row:

- keep / collapse / filter outcome
- `variant_class`
- `variant_signal`
- `equivalence_group_id`
- target final formulation id
- decision rule
- decision reason
- retention reason
- collapse reason
- review-needed flag

The final formulation table must preserve collapsed-variant membership so that a
retained benchmark row can be traced back to all collapsed source rows.

## Validation Status

Validated replay run:

- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/04_stage5_variant_governance_replay/run_20260313_2002_c4eccc8_dev15_stage5_variant_governance_replay_v1`

Confirmed replay outcomes:

- `UFXX9WXE` remained `27` vs GT `26`, so the DOE recovery benefit was preserved.
- `INMUTV7L` remained corrected at `12` vs GT `12`.
- `BXCV5XWB` changed from `9` to `7` through two conservative `post_processing_or_measurement_variant` same-core collapses.
- No other papers changed relative to the previous replay.
- No fresh LLM calls were made.

## Current Limits

The current governed layer remains deliberately narrow.

- `WFDTQ4VX` checkpoint / validation cases are classified but are not
  auto-collapsed in Stage5 without a unique deterministic target.
- Optimized / baseline handling remains unique-target-only.
- Parent / variant inheritance is still not relation-driven in Stage5.
- Ambiguous cases remain `uncertain_variant` with `review_needed = yes`.
