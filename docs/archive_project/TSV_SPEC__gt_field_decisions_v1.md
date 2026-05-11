# Field-level GT Template Specification (v1)

This document defines the output of `build_gt_template_from_conflict_queue.py`.

Goal
- Human-in-the-loop ground truth (GT) is defined as *field-level decisions*,
  not enforcing a complete formulation-level truth table.
- Only conflicting fields (detected by cross-model disagreement) are routed to humans.
- Missing/ambiguous literature evidence must be labeled `unclear` rather than guessed.

## Input

`formulations_conflict_queue.tsv` produced by `multi_model_consensus_vote.py`.

Required columns:
- `key`
- `formulation_id`
- `model1`, `model2` (the model names compared)
- `conflict_fields` (semicolon-separated field names)

Recommended columns (used when present):
- `preferred_model`
- per-field values: `<field>_model1`, `<field>_model2`
- evidence blocks (main and optionally per-model):
  - `evidence_section_main`
  - `evidence_span_text_main`
  - `evidence_span_start_main`
  - `evidence_span_end_main`
  - `evidence_method_main`
  - `evidence_quality_main`
  - (optional) `evidence_*_model1`, `evidence_*_model2`

## Output: `gt_field_decisions.tsv`

One row per `(key, formulation_id, field_name)`.

Core columns:
- `key`
- `formulation_id`
- `field_name`
- `model1`, `model2`, `preferred_model`
- `value_model1`, `value_model2`

Evidence columns (always included as *_main; per-model optional):
- `evidence_section_main`
- `evidence_span_text_main`
- `evidence_span_start_main`
- `evidence_span_end_main`
- `evidence_method_main`
- `evidence_quality_main`
- optional: `evidence_*_model1`, `evidence_*_model2`

Human annotation columns:
- `gt_decision` (required): one of
  - `accept_model1`
  - `accept_model2`
  - `reject_both`
  - `unclear`
- `gt_value_text` (optional):
  - used when `reject_both` and you can confidently type a corrected value supported by evidence
  - otherwise leave empty
- `gt_notes` (optional): short rationale (unit mismatch, ambiguous text, table unclear, etc.)

## Recommended annotation policy

For each row:
1) Read `evidence_span_text_main` first.
2) Compare `value_model1` vs `value_model2`.
3) Choose:
   - `accept_model1` or `accept_model2` if evidence supports one clearly
   - `reject_both` if both are unsupported by evidence
   - `unclear` if evidence is insufficient/ambiguous

This design minimizes manual intervention while preserving auditability and scientific defensibility.
