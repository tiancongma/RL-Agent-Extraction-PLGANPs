# Snapshot: 2026-03-31 Stage2 Composite Contract Correction

## Problem

The repository accumulated a Stage2 architecture and evaluation mismatch.

Observed drift:

- Stage2 was described in some places as LLM semantic discovery only.
- The deterministic post-LLM compatibility completion step was described as a
  separate bridge rather than as part of Stage2.
- The three-paper live Gemini comparison slice was interpreted too strongly as
  a Stage2 judgment even though it compared only the raw semantic-discovery
  intermediate.
- Scope-specific wrappers risked implying alternative Stage2 definitions.

This was a governance problem, not evidence that "LLM failed."

The misalignment was that the evaluation object did not match the composite
Stage2 contract.

## Correction

Stage2 is now defined as one composite stage consisting of:

1. LLM semantic discovery
2. deterministic post-LLM completion

No new numbered stage was introduced.

The official stage chain remains:

- Stage2 -> Stage3 -> Stage5

Interpretation rule:

- raw semantic objects are an internal Stage2 intermediate
- the completed Stage2 artifact is the only authoritative Stage2 output for:
  - Stage3 consumption
  - Stage2 structural evaluation

## Resulting execution rule

The one governed Stage2 execution entrypoint is:

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`

Its internal graph is fixed:

1. `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
2. `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Scope variation is allowed only through inputs/config such as:

- manifest
- paper keys
- source mode
- llm backend
- model
- max text chars

Special wrapper scripts must not define Stage2.

## Audit note: corrected Stage2-relevant script roles

- governed Stage2 entrypoint:
  - `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- internal semantic extraction substep:
  - `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- internal deterministic post-LLM completion substep:
  - `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- deterministic fallback/comparator extractor only:
  - `src/stage2_sampling_labels/emit_semantic_objects_from_cleaned_papers_v1.py`
- deprecated legacy wide-row fallback only:
  - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- historical three-paper extractor name retained as compatibility shim only:
  - `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py`
- comparison-only wrapper:
  - `src/utils/run_threepaper_stage2_v2_comparison.py`
- semantic-intermediate comparison helper only:
  - `src/analysis/build_stage2_v2_threepaper_comparison_pack.py`

## Correct interpretation of the recent live Gemini three-paper slice

Run:

- `run_20260331_1156_03e5d25_threepaper_stage2_v2_live_gemini_structural_eval_v2`

Facts preserved:

- the live run surfaced useful semantic-intermediate findings on:
  - formulation boundary behavior
  - component preservation
  - expression richness
  - variable detection
  - ambiguity handling
- the live semantic intermediate showed real boundary instability on the three
  target papers

Corrected interpretation:

- the original three-paper comparison targeted the raw semantic-discovery
  intermediate
- it did not evaluate the completed Stage2 artifact as the authoritative
  Stage2 object
- therefore it is valid as semantic-intermediate diagnostic evidence only
- it is not, by itself, a final go/no-go judgment on the completed composite
  Stage2 contract

Explicit skipped-step note for the original interpretation:

- the original live three-paper evaluation compared raw semantic output against
  GT/comparator surfaces without treating deterministic post-LLM completion as
  the evaluation object

## Execution and evaluation rule after correction

- If the question is "how good is the LLM semantic-discovery intermediate?",
  raw semantic comparison is acceptable as diagnostic-only work.
- If the question is "how good is Stage2 as the governed production stage?",
  evaluation must target the completed Stage2 artifact after deterministic
  post-LLM completion.

## Non-change statement

- No new numbered stage was added.
- Stage3 still resolves logic, inheritance, and relation among completed
  Stage2 outputs.
- Stage5 still materializes final benchmark-valid outputs only.
- `data/results/ACTIVE_RUN.json` was not changed by this correction snapshot.
