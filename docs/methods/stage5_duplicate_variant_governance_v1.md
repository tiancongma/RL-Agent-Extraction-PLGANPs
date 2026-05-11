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

## Identity Constraints

Stage5 also applies benchmark-facing identity constraints.

A row is included in the final formulation table only when it represents an
independently reported formulation instance. A row is excluded when it only
references a parent formulation identity or only summarizes shared/comparative
context without introducing a separate formulation instance.

### Constraint 1: Parent-linked non-synthesis descendants are not independent identities

When a row is parent-linked and explicitly marked as a non-synthesis
descendant, Stage5 treats that row as a formulation-referencing variant rather
than a new benchmark-facing formulation identity when the row is clearly in
control, characterization, post-processing, or downstream evaluation context.

Safeguard:

- `post_processing` is not by itself a universal exclusion signal.
- If a parent-linked row is still a sweep-style `variant_formulation` member,
  Stage5 must not auto-filter it solely because `post_processing` appears in
  the tags.
- Those ambiguous rows must instead fall through to the conservative
  duplicate/variant governance path, where they are kept unless a unique safe
  collapse target exists.

Why this safeguard exists:

- Some papers encode table-native formulation members as parent-linked
  non-synthesis variants with `post_processing` tags even though the paper and
  Layer 2 benchmark authority still treat them as benchmark-facing formulation
  identities.
- `BB3JUVW7` is the validated regression anchor for this failure class, but the
  safeguard is defined by row semantics rather than by paper key.

Conceptual definition:

- the parent row carries the benchmark-facing formulation identity
- the descendant row carries downstream usage or measurement context
- the descendant is therefore excluded from final identity closure

Additional helper-descendant safeguard:

- Stage5 must also filter parent-linked helper descendants when existing
  downstream-visible signals already show that the row is a control,
  characterization, or assay/helper derivative even if Stage2 routing tags
  regressed.
- This safeguard intentionally keeps Stage2 frozen for the `BXCV5XWB`-class
  failure. The rule uses existing preserved signals such as:
  - `formulation_role`
  - `payload_state`
  - `instance_context_tags`
  - parent linkage
  - raw label text and change-description text that indicate blank/no-drug or
    model-drug substitution behavior
- The rule is not keyword-only and is not paper-specific. It requires a parent
  link plus helper-descendant semantics that do not define an independent
  synthesis identity.

### Constraint 2: Shared/comparative summaries are not independent identities

When a row is unparented but is explicitly tagged as a shared-condition summary
or as a comparative-study summary reference, Stage5 treats that row as context
for formulation interpretation rather than as an independently reported
formulation instance.

Conceptual definition:

- shared-condition blocks describe preparation context reused across multiple
  formulations
- comparative-study summary references describe a comparison surface, not a new
  formulation identity
- these rows are therefore excluded from final identity closure

These constraints are part of the current `DEV15_v2` Stage5 identity contract.
They are system behavior definitions, not paper-specific patches.

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

Additional regression protection:

- `src/stage5_benchmark/validate_stage5_descendant_filter_regression_v1.py`
- This deterministic checker reruns Stage5 against frozen artifact inputs and
  asserts both:
  - `BB3JUVW7` retains all `12` benchmark-facing final rows
  - `BXCV5XWB` retains only the `3` KGN benchmark-facing rows
  - `RHMJWZX8` drops its parent-linked empty-control helper row
  - known descendant/control rows from existing blocker material remain
    filtered
  - `WIVUCMYG` remains stable at `26`

## Current Limits

The current governed layer remains deliberately narrow.

- `WFDTQ4VX` checkpoint / validation cases now support one narrow
  benchmark-valid collapse rule: checkpoint batches are collapsed when the
  validated coordinate-signature mapping shows that the batch matches exactly
  one existing design-row formulation identity.
- Generalized DOE coordinate reconciliation is still not implemented in Stage5.
- Optimized / baseline handling remains unique-target-only.
- Parent / variant inheritance is still not relation-driven in Stage5.
- Ambiguous cases remain `uncertain_variant` with `review_needed = yes`.
