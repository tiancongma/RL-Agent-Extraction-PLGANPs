# Final Formulation Audit-Ready Export V1

## Context

The current Stage 5 final formulation table already preserves several useful
provenance fields, but it does not expose an explicit human-audit layer with
review tiers or review-priority flags.

Confirmed repository evidence:

- `final_formulation_table_v1.tsv` carries indirect provenance fields such as:
  - `instance_confidence`
  - `candidate_source`
  - `instance_evidence_region_type`
  - `supporting_evidence_refs`
  - `source_candidate_sources`
  - `source_candidate_count`
  - `final_output_rule`
- `final_output_decision_trace_v1.tsv` adds closure-rule context but does not
  itself assign a reviewer-facing confidence tier.

## Decision

Add a postprocessing export, not a pipeline rewrite.

The audit-ready table is derived from existing final-output and provenance
artifacts and does not change scientific extraction or benchmark logic.

## Export Contract

Script:

- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`

Primary output:

- `final_formulation_table_audit_ready_v1.tsv`

The export reuses existing provenance signals and materializes:

- `confidence_tier`
- `review_priority`
- `needs_review_reason`
- compact evidence locator fields
- blank reviewer action columns

## Confidence Logic

- `tier1_structured_row`
  - explicit numbered DOE or similarly structured table row
  - table-derived evidence
  - stable row anchor and table identifier
- `tier2_table_derived`
  - table-based evidence exists, but the row is less explicit than a stable
    numbered row
- `tier3_llm_with_evidence`
  - LLM-extracted row with usable provenance and evidence, but weaker than
    deterministic structured-table grounding
- `tier4_review_required`
  - weak or ambiguous provenance
  - unknown evidence region
  - full-text-window-only support
  - figure sweep or similarly indirect derivation

## Intended Use

This export is a review surface for fast manual triage. It should help auditors
sort final rows by structural grounding without reopening all Stage 2, Stage 3,
or Stage 5 artifacts.
