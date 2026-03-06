# Weak Labels v7 Schema Design (2026-03-06)

## Current v6 Schema Summary
Current extraction behavior in `auto_extract_weak_labels_v6.py` is:
- top-level JSON: `formulations` (array) + `paper_notes`,
- each formulation: `id`, `fields` (flat object), `notes`,
- flattened TSV output with one row per extracted formulation candidate,
- row-level evidence contract only:
  `evidence_section`, `evidence_span_text`, `evidence_span_start`, `evidence_span_end`, `evidence_method`, `evidence_quality`.

v6 is a strong baseline for extraction completeness and deterministic downstream processing, but semantic structure is still under-specified for instance-level arbitration.

## Diagnosed Semantic Gaps in v6
1. Shared-vs-instance ambiguity:
- v6 does not explicitly label whether a value is global, table-header-scoped, row-scoped, or inherited.

2. Field membership ambiguity:
- v6 stores a single field value without explicit confidence for membership to the current formulation instance.

3. Evidence region under-typing:
- evidence is row-level and not consistently typed as table row, table column header, methods-global statement, etc.

4. Formulation role under-specification:
- v6 does not explicitly encode whether a row is baseline/control/optimized/variant/comparative.

5. Instance confidence under-specification:
- v6 does not provide an explicit confidence score for the formulation instance itself, separate from field-level uncertainty.

## Design Goals for v7
- Keep extraction JSON machine-parseable and prompt-friendly.
- Preserve deterministic downstream arbitration authority.
- Add LLM-side semantic typing so downstream semantic repair logic can shrink.
- Maintain compatibility with staged rollout (v7 JSON can still be flattened to TSV).
- Improve auditability by typing evidence region and semantic scope per field.

## Proposed Top-Level JSON Structure
```json
{
  "schema_version": "weak_labels_v7",
  "paper_id": "zotero_key_or_doc_key",
  "paper_notes": "string or null",
  "extraction_notes": "string or null",
  "formulations": [
    {
      "...": "see formulation object"
    }
  ]
}
```

## Proposed Formulation Object Structure
```json
{
  "formulation_id": "string_or_int",
  "formulation_role": "baseline|control|optimized|variant|comparative|characterization_only|unknown",
  "instance_confidence": "high|medium|low",
  "instance_evidence": {
    "evidence_region_type": "table_row|table_block|methods_sentence|results_sentence|mixed|unknown",
    "evidence_span_text": "string or null",
    "evidence_span_start": 0,
    "evidence_span_end": 0,
    "evidence_section": "string or null"
  },
  "fields": {
    "plga_mw_kDa": {
      "...": "see field object"
    }
  },
  "notes": "string or null"
}
```

## Proposed Field Object Structure
```json
{
  "value": "string|number|null",
  "value_text": "original text or null",
  "unit": "string or null",
  "scope": "instance_specific|shared_within_table|shared_within_paper|unknown",
  "membership_confidence": "high|medium|low",
  "evidence_region_type": "table_cell|table_row|table_header|table_caption|methods_sentence|results_sentence|unknown",
  "evidence": {
    "evidence_section": "string or null",
    "evidence_span_text": "string or null",
    "evidence_span_start": 0,
    "evidence_span_end": 0
  },
  "missing_reason": "not_reported|ambiguous|conflicting|not_applicable|unknown"
}
```

## New Concept Definitions

### scope
`scope` describes semantic applicability of a field value:
- `instance_specific`: applies only to the current formulation instance.
- `shared_within_table`: shared across rows/conditions in a table context.
- `shared_within_paper`: global condition/baseline for multiple formulations in the paper.
- `unknown`: insufficient evidence to classify.

### membership_confidence
`membership_confidence` is field-level confidence that the field belongs to the current formulation instance.
It does not represent numeric correctness; it represents semantic assignment confidence.

### evidence_region_type
`evidence_region_type` captures the structural source region of evidence (table header, table row, methods sentence, etc.) so downstream rules do not need to infer region type repeatedly.

### formulation_role
`formulation_role` identifies experiment role semantics for a formulation candidate:
baseline/control/optimized/variant/comparative/characterization-only/unknown.

### instance_confidence
`instance_confidence` is formulation-level confidence that the row is a valid distinct instance, independent of individual field confidence.

## Mapping from v6 to v7

### v6 top-level to v7 top-level
- `paper_notes` -> unchanged.
- `formulations[]` -> unchanged container, but object schema expanded.

### v6 formulation object to v7 formulation object
- v6 `id` -> v7 `formulation_id`.
- v6 `notes` -> v7 `notes`.
- new in v7: `formulation_role`, `instance_confidence`, `instance_evidence`.

### v6 field scalar to v7 field object
- v6 `fields.<name> = scalar` becomes v7 `fields.<name>.value`.
- if v6 has text+value+unit variants (e.g., surfactant concentration), map directly:
  - `surfactant_concentration_text` -> `fields.surfactant_concentration.value_text`,
  - `surfactant_concentration_value` -> `fields.surfactant_concentration.value`,
  - `surfactant_concentration_unit` -> `fields.surfactant_concentration.unit`.
- new in v7: `scope`, `membership_confidence`, `evidence_region_type`, per-field `evidence`, `missing_reason`.

### v6 row-level evidence to v7 evidence
- v6 row-level evidence fields become fallback values for:
  - `formulation.instance_evidence`,
  - and each field lacking field-level evidence object.

## Downstream Rule Families Expected to Be Reduced If v7 Is Adopted
- formulation grouping/regrouping based on inferred semantic signatures,
- global baseline inheritance rules that infer shared method constants,
- condition-instance key reconstruction from partial text/table cues,
- drug/surfactant semantic normalization used to recover membership assignments,
- repeated shared-vs-instance semantic repair in alignment pipelines.

Deterministic numeric arbitration, derivation, and export remain required.

## What v7 still does not delegate to the LLM
- final numeric arbitration across conflicting evidence values,
- deterministic evidence token gating and tolerance-based numeric checks,
- deterministic derivation math and unit normalization policies,
- release-time schema composition and database export contracts,
- PLGA-only publication filter at database/export layer.

## Implementation Notes for Future Work
- v7 is a target schema contract, not implemented in runtime scripts in this step.
- staged rollout should support:
  1. v7 JSON emission,
  2. compatibility flattening to current TSV columns,
  3. incremental downstream adoption of `scope`, `membership_confidence`, `evidence_region_type`, `formulation_role`, `instance_confidence`.
