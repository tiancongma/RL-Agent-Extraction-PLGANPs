# Layer 3 Field GT Protocol V1

## Purpose

This document defines the next active GT layer after the DEV15 reviewed-boundary
freeze.

Layer 3 evaluates field-level correctness for formulation rows that are already
accepted by Layer 2. It is not a restart of row-boundary review.

Primary goals:

- normalize extracted field values into auditable canonical forms
- distinguish reported values from derived values
- validate field-level consistency and derivation rules
- support benchmarkable field-level comparison without changing formulation-row
  identity

## Why This Lives In `docs/methods/`

Existing repo convention separates:

- `docs/snapshots/` for stage-state freeze points
- `docs/methods/` for reusable engineering and evaluation method definitions

Layer 3 needs a reusable method spec, not just a freeze note, so it belongs in
`docs/methods/`.

## Existing Layer 3-Relevant Assets

### Asset Inventory And Triage

| Asset | Current role | Triage | Notes |
|---|---|---|---|
| `src/stage5_benchmark/run_derivation_v1.py` | deterministic value derivation from extraction outputs | `update_and_reuse` | Useful base for Layer 3 derivation traces, but defaults and field set are tied to historical Goren overlap work. |
| `src/stage5_benchmark/run_projection_to_curated_v1.py` | projection of derived fields into benchmark columns | `update_and_reuse` | Reusable projection pattern; needs DEV15/Layer 2 row linkage instead of historical curated overlap defaults. |
| `src/stage5_benchmark/run_alignment_eval_v1.py` | field-level alignment modes (`strict`, `relaxed`, `canonicalized`) | `update_and_reuse` | Strong starting point for Layer 3 evaluation policy; needs active row linkage and Layer 3 field catalog. |
| `src/stage5_benchmark/run_alignment_eval_core_v1.py` | core-level reuse wrapper around alignment eval | `reference_only` | Helpful pattern, but Layer 3 is row-level-on-frozen-Layer-2, not core-collapse evaluation. |
| `src/stage5_benchmark/run_projection_core_to_curated_v1.py` | schema_v1 core projection | `reference_only` | Useful for legacy schema understanding, not the preferred Layer 3 entry surface. |
| `src/stage5_benchmark/run_evidence_token_qc_v1.py` | evidence-token gating and numeric support QC | `update_and_reuse` | Valuable as a pre-compare guard and field review prioritizer. |
| `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py` | review-ready Stage 5 export with provenance tiers | `active_reuse` | Good seed surface for Layer 3 review packs because it already exposes audit tiers without changing benchmark behavior. |
| `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py` | active Layer 2 workbook builder | `update_and_reuse` | Workbook structure and editable/reference split should be reused for Layer 3 workbook design. |
| `src/stage5_benchmark/build_field_gt_review_workbook_v1.py` | active Layer 3 workbook builder | `active_reuse` | Current reviewer-facing XLSX seed for field GT; keeps reviewer columns compact, adds readable helper labels, and only keeps evidence when it directly supports the seeded extracted value. |
| `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py` | human-evidence audit workbook builder | `update_and_reuse` | Strong evidence-review surface for field-level work; likely better as a supporting Layer 3 audit pack than as the GT authority itself. |
| `src/stage5_benchmark/audit_evidence_resolver_v1.py` | resolver for text/table evidence and ownership checks | `active_reuse` | Good reusable evidence locator for Layer 3 review and QC. |
| `src/stage5_benchmark/build_two_table_schema_v2.py` | deterministic split between formulation core and measurements | `reference_only` | Useful as schema inspiration for separating core identity vs measurement fields. |
| `src/stage5_benchmark/build_two_table_schema_v3.py` | richer deterministic schema with DOE-aware splitting | `reference_only` | Useful for field catalog and derived-field candidates; not the immediate Layer 3 authority object. |
| `data/cleaned/labels/manual/gt_field_decisions__run_20260201_0927_bb13267_sample20.tsv` | prior field-level human GT decisions | `reference_only` | Important prior art for field decision columns, but based on cross-model conflict arbitration rather than frozen Layer 2 rows. |
| `docs/archive_project/TSV_SPEC__gt_field_decisions_v1.md` | archived field-level GT template spec | `reference_only` | Useful seed for decision enums and evidence columns; needs modernization for Layer 2 row linkage and normalization traces. |
| `data/cleaned/labels/manual/dev_v7pilot3_field_mapping_audit.xlsx` | JSON/TSV field-mapping audit workbook | `reference_only` | Shows useful audit-review structure for parser-to-TSV correctness. |
| `data/cleaned/labels/manual/doi_10.1016_j.ejpb.2004.09.002_value_flow_audit*.xlsx` | per-DOI value-flow audit workbooks | `reference_only` | Good examples of raw -> parsed -> TSV field flow analysis. |
| `archive/code/benchmark_specific_audit_report/build_v7pilot_field_mapping_audit.py` | field-mapping audit workbook builder | `reference_only` | Historical but useful as a template source. |
| `archive/code/benchmark_specific_audit_report/build_doi_value_flow_audit_v7pilot.py` | raw-to-TSV value flow audit | `reference_only` | Historical diagnostic reference; should not be revived directly as active runtime. |
| `data/results/historical_non_compliant_runs/.../benchmark_goren_2025/derived_values.tsv` and related outputs | historical field derivation/eval artifacts | `reference_only` | Strong evidence that partial Layer 3 work exists; outputs are historical and non-compliant but structurally informative. |

### Reuse Summary

Directly reusable now:

- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/audit_evidence_resolver_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

### Current Layer 3 Workbook Semantics

The active Layer 3 review workbook currently uses these reviewer-facing rules:

- keep the original `formulation_id` plus two helper identity columns:
  - `formulation_label_stage5`
  - `formulation_label_params`
- keep article-native formulation identifiers visible as reviewer aids:
  - `article_formulation_id`
  - `article_formulation_label`
- when GT-skeleton mode is active, prefer a unique canonical article-native
  match from the latest audit-ready export before accepting historical
  scaffold fallback
- never replace the canonical system identity with article-native labels
- keep Layer 2 paper-risk metadata visible on the review surface when
  `analysis/paper_risk_assessment.tsv` is provided:
  - `paper_risk_level`
  - `layer3_inclusion_flag`
- keep the review sheet narrow enough for manual review by freezing the
  identity block before `field_name`
- if `extracted_value` is blank, leave `evidence_text` blank and mark
  `evidence_source_type=blank_value`
- if a table-derived field cannot be supported by row-local table evidence,
  mark `evidence_source_type=unresolved_table` and do not fall back to text
- if a text-derived field cannot be supported by a value-matching sentence,
  mark `evidence_source_type=unsupported_text` and do not keep misleading
  paragraph text
- preserve the closest available evidence anchor separately from direct support:
  - `evidence_text` remains reserved for direct value support
  - `evidence_anchor_text` may carry the closest reviewer-useful source anchor
    when direct support is missing
  - `evidence_anchor_text` must remain blank when the only available anchor is
    a broad row-level span with no field-local relationship to the seeded field
  - `evidence_status_detail` distinguishes:
    - `supported`
    - `unsupported_text`
    - `unresolved_table`
    - `derived_without_direct_text`
    - `missing_evidence_anchor`
- apply field-aware support checks for structured fields so `LA/GA` and
  `polymer_MW` are not marked supported by unrelated numeric text
- when a field is surfaced as `relation_resolved`, preserve the Stage 3
  relation-resolution provenance in the seed/reference layer instead of
  reusing representative row-level anchors as if they were direct support
- do not surface polymer grade/product-code text as if it were a molecular-
  weight value in the `polymer_MW` review row
- mark raw-mass concentration rows with `normalization_pending` instead of
  implying a safe normalized concentration
- keep risk metadata review-only:
  - risk labels may guide Layer 3 review priority
  - risk labels must not change frozen formulation identity or benchmark counts
- emit an alignment-resolution audit TSV whenever GT-skeleton rows are built so
  direct canonical overrides versus scaffold fallback stay reviewer-auditable

### Layer 3 Evidence Handoff Contract

#### Purpose

The Layer 3 workbook must faithfully represent field-level values together with
their evidence support status and provenance.

It must not introduce weaker or misleading evidence representations than the
stronger evidence/QC safeguards already present upstream in the repository.

#### Scope

This contract applies to reviewer-facing Layer 3 outputs, including:

- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- any future Layer 3 audit/export builders that surface field-level evidence

This contract does not modify:

- Stage 2 extraction semantics
- Stage 3 relation semantics
- Stage 5 identity logic or final row membership
- benchmark-valid outputs

#### Required Behaviors

Hard constraints:

- field-local evidence must be preferred over row-level representative spans
- if no field-local evidence exists:
  - `evidence_anchor_text` must be empty
  - `evidence_status_detail` must reflect `missing_evidence_anchor` or an
    equally explicit no-anchor state
- broad non-local text must not be surfaced as field evidence
- structured fields must not be validated using generic numeric-token matching
- field-aware matching must be used for:
  - `LA/GA`
  - `polymer_MW`
  - other structured numeric fields added later
- relation-resolved values must carry provenance such as:
  - `relation_resolution_rule`
  - `relation_resolution_confidence`
  - `relation_resolution_source_ids`
- reviewer-facing outputs must preserve:
  - `extracted_value`
  - `evidence_text` only when direct support exists
  - `evidence_status_detail`
  - provenance metadata

#### Prohibited Behaviors

- treating row-level `supporting_evidence_refs` as implicit field-level support
- displaying unrelated or weak-context spans as `evidence_anchor_text`
- marking structured fields as supported based only on shared numeric tokens
- silently replacing stronger upstream evidence/QC classifications with weaker
  local workbook heuristics

#### Relationship To Upstream QC

If stronger evidence/QC artifacts exist, such as
`src/stage5_benchmark/run_evidence_token_qc_v1.py`, Layer 3 must:

- either consume them directly
- or avoid contradicting them

Workbook-layer logic must not degrade upstream evidence classification or
present weaker row-level heuristics as stronger field support.

#### Golden Regression Cases

The contract is regression-protected by:

- golden case definitions:
  - `docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv`
- lightweight validator:
  - `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py`

Minimum golden behaviors:

- `5GIF3D8W`, `polymer_MW=10.0 kDa`
  - extracted value present
  - empty `evidence_anchor_text`
  - `evidence_status_detail=missing_evidence_anchor`
  - non-empty `relation_resolution_rule`
- `5GIF3D8W`, `LA/GA=75/25`
  - must not be supported by unrelated text containing `50 mg`
- `5ZXYABSU`, polymer grade vs `polymer_MW`
  - grade text must not surface as `polymer_MW`
  - reviewer warning must stay explicit
- broad row-level span only
  - must not surface as `evidence_anchor_text`

Best candidates to update and reuse for an active Layer 3 path:

- `src/stage5_benchmark/run_derivation_v1.py`
- `src/stage5_benchmark/run_projection_to_curated_v1.py`
- `src/stage5_benchmark/run_alignment_eval_v1.py`
- `src/stage5_benchmark/run_evidence_token_qc_v1.py`
- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_audit_pack_human_evidence_v1.py`

Historical references only:

- archived `gt_field_decisions` spec
- archived field-mapping/value-flow audit builders
- historical Goren benchmark outputs under `data/results/historical_non_compliant_runs/`

### Cleanup Decision

No cleanup is performed in this pass.

Reason:

- the candidate Layer 3 assets are still useful for reference or future
  promotion
- several are historical but not clearly superseded by an active Layer 3 path
  yet
- removing them now would increase ambiguity rather than reduce it

## Proposed Layer 3 Authority Model

### Authority Boundary

Layer 3 authority starts from frozen Layer 2 formulation rows.

The primary row authority is:

- reviewed-boundary-accepted Stage 5 final formulation instances

Layer 3 must not:

- add rows
- remove rows
- change family identity
- change variant retention behavior

### Recommended GT Object

One row per:

- `(paper_key, frozen_final_formulation_id, field_name)`

This keeps Layer 3 anchored to the frozen Layer 2 row object while allowing
independent field decisions and normalization traces.

## Proposed Layer 3 GT Schema

Recommended core columns:

- `paper_key`
- `doi`
- `frozen_final_formulation_id`
- `frozen_family_id`
- `frozen_parent_core_row_id`
- `field_name`
- `raw_extracted_value`
- `raw_extracted_value_text`
- `source_unit_raw`
- `normalized_value`
- `normalized_unit`
- `normalization_method`
- `normalization_rule_id`
- `reported_vs_derived_status`
- `derivation_status`
- `derivation_formula_or_rule`
- `derivation_inputs`
- `evidence_source_type`
- `evidence_locator`
- `evidence_quote`
- `field_decision_status`
- `comparison_mode`
- `tolerance_rule`
- `not_reported_status`
- `not_applicable_status`
- `benchmark_field_include`
- `manual_override_value`
- `manual_override_reason`
- `review_notes`

Recommended decision enums:

- `field_decision_status`
  - `accept_extracted`
  - `accept_normalized_equivalent`
  - `accept_derived`
  - `reject_extracted`
  - `not_reported`
  - `not_applicable`
  - `unclear`
- `reported_vs_derived_status`
  - `reported_raw`
  - `reported_normalized`
  - `derived_from_reported`
  - `inferred_unsupported`
- `derivation_status`
  - `none`
  - `safe_deterministic`
  - `blocked_context_insufficient`
  - `manual_override`
- `benchmark_field_include`
  - `yes`
  - `no`

## Proposed Evaluation Protocol

### 1. Exact-Match Fields

Use exact match after conservative text normalization for:

- categorical controlled-vocabulary fields
- explicit identifiers and anchor-like categorical values
- boolean or enum-style statuses

Examples:

- `preparation_method`
- `emulsion_structure`
- `polymer_identity`
- `surfactant_name_normalized`
- `organic_solvent_normalized`

### 2. Normalized-Equivalence Fields

Use equivalence after semantic and unit normalization for:

- polymer names
- solvent names
- surfactant names
- formulation method labels
- ratios expressed in equivalent canonical form

Examples:

- `Poloxamer 188` == `Pluronic F68` only when controlled vocabulary explicitly
  maps them
- `50:50` == `1.0` only when the field is a ratio field with an approved ratio
  normalization rule

### 3. Tolerance-Based Numeric Fields

Use numeric tolerance only after canonical-unit conversion.

Recommended classes:

- exact numeric after conversion for formulation-defining inputs where units are
  explicit and conversion is unambiguous
- tolerance-based numeric for measured outputs

Examples:

- `size_nm`
- `pdi`
- `zeta_mV`
- `encapsulation_efficiency_percent`
- `loading_content_percent`

Recommended policy:

- preserve source value and source unit
- compare only normalized canonical values
- record the tolerance rule used per field

### 4. Derived-Field Validation

Layer 3 should distinguish:

- directly reported fields
- normalized reported fields
- safely derived fields

Derived-field evaluation should check:

- whether derivation inputs are present
- whether the derivation rule is approved
- whether the derived output is internally consistent with stored source values

### 5. Consistency Checks Across Fields

Run deterministic cross-field checks before final scoring:

- drug/polymer ratio consistency against reported masses
- LA/GA fraction consistency against ratio text
- phase-ratio consistency against emulsion structure fields
- solvent/preparation-method compatibility flags
- blank-control rows should not carry drug-loaded-only metrics without explicit
  justification

### 6. Missing / Not-Reported / Not-Applicable

Keep these separate:

- `not_reported`
  - the paper does not report the field for that row
- `not_applicable`
  - the field is not meaningful for that row
- `unclear`
  - evidence exists but is ambiguous or insufficient

These statuses must not be collapsed into an empty string.

## Unit Normalization Policy

Direct unit conversion is allowed only when:

- source unit is explicit
- target canonical unit is defined for the field
- conversion does not require hidden formulation context

Direct unit conversion is forbidden when:

- concentration vs absolute amount ambiguity is unresolved
- mass ratio context is missing
- phase-volume denominator is unclear
- molecular-weight representation mixes intrinsic viscosity, grade name, and MW
  without a safe mapping rule

Required stored columns:

- `source_unit_raw`
- `normalized_unit`
- `normalization_method`
- `normalization_rule_id`

## Semantic Normalization Policy

Semantic normalization should use controlled mappings with traceable rule IDs.

Recommended controlled-vocabulary areas:

- polymer identity
- surfactant identity
- solvent identity
- preparation method
- emulsion structure

Examples:

- polymer:
  - `poly(D,L-lactide-co-glycolide)` -> `PLGA`
  - `PEG-PLGA` remains distinct from `PLGA`
- surfactant:
  - `PVA` -> `polyvinyl alcohol`
  - `Poloxamer 188` and `Pluronic F68` map only if the rule registry declares
    them equivalent
- solvent:
  - `DCM` -> `dichloromethane`
  - `ethyl acetate` remains distinct from `acetone`
- preparation method:
  - `solvent displacement` -> `nanoprecipitation`
  - `double emulsion solvent evaporation` -> `double_emulsion` plus
    subordinate route or method trace

## Formula / Derivation Policy

Primary derived-field candidates:

- `drug_to_polymer_mass_ratio`
- `la_fraction`
- `ga_fraction`
- normalized polymer MW lower/upper bounds
- canonical surfactant concentration units

Required evidence before derivation:

- both numerator and denominator inputs are present and attributable to the same
  frozen Layer 2 formulation row
- units are explicit and compatible
- derivation rule is registered and deterministic

Reported vs derived distinction:

- if the paper reports the target value directly, keep it as reported
- if the target value is produced by deterministic math from reported inputs,
  mark it as derived
- if the value requires assumption-heavy inference, do not derive it

## Minimal Governance-Compliant Implementation Plan

Smallest justified next-step file set:

- keep this method spec in `docs/methods/`
- use the active Layer 3 review workbook builder
  `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`, which reuses the
  existing Layer 2 workbook layout pattern from
  `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- update, rather than replace, the existing Stage 5 field tools:
  - `run_derivation_v1.py`
  - `run_projection_to_curated_v1.py`
  - `run_alignment_eval_v1.py`
  - `run_evidence_token_qc_v1.py`
- seed future Layer 3 review from:
  - `final_formulation_table_audit_ready_v1.tsv`
  - reviewed-boundary-accepted rows only

Not recommended now:

- creating a brand-new top-level Layer 3 code namespace
- reviving archived conflict-queue GT tooling as the active runtime path
- creating a parallel field GT object detached from frozen Layer 2 row IDs

## Recommended Next Execution Step

Start Layer 3 by building a run-scoped field-review seed surface from the frozen
reviewed-boundary benchmark object:

1. filter to reviewed-boundary-accepted final rows
2. export an audit-ready field seed using current Stage 5 provenance
3. define the active Layer 3 field catalog and canonical units
4. generate the initial review workbook from the frozen final table using the
   active builder
5. only then wire deterministic field-level evaluation against the reviewed
   Layer 3 authority

## Initial Layer 3 Workbook Field Catalog

The current active workbook seed includes compact review rows for:

- `polymer_MW`
- `LA/GA`
- `drug_polymer_ratio`
- `surfactant_name`
- `surfactant_concentration`
- `solvent`
- `particle_size`
- `EE`
- `LC`
- `preparation_method`
- `emulsion_structure`

Current omission:

- `phase_ratio` is not seeded in the initial workbook because the frozen Stage 5
  final table does not yet carry a safe explicit phase-ratio field for all
  rows.

## Layer 3 Evidence Binding Rule

The active Layer 3 workbook uses a strict evidence-binding policy:

- if a field is table-derived, evidence is accepted only from row-local
  `table_cell` or `table_row` references already attached to the frozen Stage 5
  row
- if no valid row-local table evidence matches the field value, set
  `evidence_source_type = unresolved_table`
- do not fall back from a table-derived field to an arbitrary paragraph span
- text-derived fields should prefer a sentence containing the extracted value
  rather than a generic full-span paragraph
- structured fields such as `LA/GA` and `polymer_MW` must use field-aware
  support checks rather than generic numeric-token-only matching
- if a direct supporting anchor cannot be justified from existing frozen
  artifacts, keep `evidence_anchor_text` blank and let
  `evidence_status_detail` carry the downgrade
- if a field was materialized as `relation_resolved`, carry Stage 3
  relation-resolution metadata into the seed/reference export and do not treat
  representative row-level spans as field-local direct support by default
