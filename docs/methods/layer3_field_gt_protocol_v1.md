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

## Architectural Role

Layer 3 is not only an evaluation artifact.

It is also part of the governed production audit and governance layer around
the frozen formulation database.

Current interpretation:

- the benchmark-valid endpoint remains `final_formulation_table_v1.tsv`
- reviewer-facing Layer 3 outputs remain downstream audit surfaces and must not
  mutate benchmark-valid outputs
- the preferred reviewer entry object is a formulation row
- value review is a formulation-row credibility audit, not an independent field
  exercise detached from formulation identity
- formulation existence and identity audit comes first
- value credibility audit depends on structure correctness and often exposes
  structure mistakes indirectly
- current repo capability is partially present but not yet unified into one
  formulation-centered audit system contract

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
- for first-week explicit-value progress reporting and modeling-readiness summaries, do not score `surfactant_name` and `stabilizer_name` as separate primary fields when the paper uses them interchangeably for aqueous-phase emulsion/stabilization agents such as PVA, Pluronic, Tween, or related protective colloids
  - report the primary role-tolerant field as `emulsifier_stabilizer_name`
  - form the role-tolerant value from the union of `surfactant_name` and `stabilizer_name` on both GT and system sides
  - treat cross-role matches such as GT `stabilizer_name=PVA` vs system `surfactant_name=PVA` as a correct role-tolerant value match, not a value extraction error
  - retain the article-native source role (`surfactant`, `stabilizer`, `emulsifier`, `protective colloid`, etc.) as provenance / reviewer metadata rather than using it as the first-week primary score split
  - only split roles in downstream review when a paper explicitly reports multiple distinct surface-active excipients with different functions
- for DOE-factor variables whose article table reports coded factor levels, first-week explicit extraction may score the article-native coded value as the raw field value; decode from coded level to physical value belongs to the later derived/calculation layer and must retain provenance linking coded value, factor-level table, and decoded numeric value
- for ratio variables surfaced in formulation labels or row identity tokens, comparison may bind the article-native ratio token from labels such as `Drug:Polymer ratio 1:20`, `PLGA:ITZ ratio=10:1`, or compact table labels like `1:20 / 50:50`; this is a generic label-token rebinding rule, not a paper-specific override
- when a ratio label explicitly names both materials, left/right material order is part of the value semantics; `PLGA:ITZ 5:1` and `ITZ:PLGA 5:1` must not be scored as a match just because the numeric token is identical
- QLYKLPKT evidence note for future review: the source preparation text states that various amounts of ITZ were dissolved in 1% w/v PLGA acetone solutions to obtain `PLGA:ITZ (w/w)` ratios of `5:1`, `10:1`, and `15:1`; Table 2 is titled `Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios` and reports rows `5:1`, `10:1`, and `15:1` with EE values `61 ±4`, `72±1`, and `73±1`
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

Linked review-layer rule:

- formulation existence and identity review is the upstream human-audit layer
- value credibility review is the downstream human-audit layer
- if a formulation row is structurally wrong, field-level value review on that
  row is secondary and may be misleading

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

## Current DEV15 Layer 3 Schema Freeze

The current governed DEV15 Layer 3 authority file is:

- `data/cleaned/gt_authority/v1/dev15_layer3_values.tsv`

This surface is treated as a formulation-row-aligned wide GT table derived from
Layer 2 row authority plus manually calibrated value backfill.

Interpretation rules:

- Layer 3 GT inherits its formulation universe from Layer 2 and must not reopen
  row-boundary decisions.
- Layer 3 GT stores only values explicitly reported in paper text or tables.
- Values obtained only by calculation, algebraic derivation, or unsupported
  inference must remain blank in GT.
- Manual calibration may correct extraction-facing representation, alignment, or
  normalization issues, but it must not promote unreported values into GT.

### Layer 3 Numeric Backfill And Compare Responsibility Freeze

Layer 3 value work must follow the repository-wide semantic-versus-deterministic
boundary.

Hard role split:

- LLM-side extraction and semantic discovery remain responsible for:
  - formulation instance boundaries
  - field-role assignment
  - shared-vs-instance-specific interpretation
  - relation and inheritance cues
  - semantic table scope and formulation-bearing value hints
- deterministic downstream layers remain responsible for:
  - numeric evidence binding
  - current-row identity alignment
  - relation-resolved carry-through
  - canonicalization and unit normalization
  - explicit derivation rules
  - compare, QC, and reviewer-facing audit surfaces

Therefore Layer 3 numeric backfill must not operate as a second semantic
extractor.

Allowed Layer 3/backfill behavior:

- bind values onto already frozen formulation identities
- use canonical current-system identity surfaces plus advisory scaffold or bridge
  artifacts to align GT rows to current rows
- normalize directly reported values into auditable canonical forms
- preserve provenance for relation-resolved or derived values
- emit alignment-resolution and evidence-status audit surfaces when identity or
  support remains uncertain

Prohibited behavior:

- using Layer 3 compare/backfill rules to create new formulation rows or redefine
  the formulation universe
- treating heuristic compare-side matching as new semantic discovery authority
- freely inferring missing values without explicit deterministic evidence-binding
  support
- presenting derived or relation-resolved values as if they were directly
  reported paper values
- allowing compare-side bridge logic to grow into an unbounded semantic repair
  system

Current architectural interpretation:

- numeric backfill should rely primarily on deterministic evidence binding,
  normalization, and audit over frozen formulation identities
- LLM outputs may provide semantic anchors and candidate ownership hints, but
  they are not the final benchmark-facing numeric authority by themselves
- if recurring Layer 3 failures require increasingly semantic downstream repair,
  treat that pattern as upstream extraction-schema backlog rather than licensing
  indefinite compare-side rule growth

### Frozen Field Groups

For current-system comparison and debugging, fields are frozen into three
classes.

#### 1. Core fixed fields

These are named schema fields whose compare behavior should remain stable across
papers.

Identity / alignment anchor fields:

- `paper_key`
- `doi`
- `gt_formulation_id`
- `family_id`
- `parent_core`
- `variant_role`
- `benchmark_default_include`
- `formulation_label`
- `seed_pred_representative_source_formulation_id`
- `gt_row_decision`

Core composition and condition fields:

- `polymer_name`
- `polymer_grade`
- `polymer_mw_raw`
- `polymer_mw_kDa`
- `la_ga_ratio_raw`
- `la_ga_ratio_normalized`
- `polymer_mass_mg`
- `polymer_concentration_value`
- `polymer_concentration_unit`
- `polymer_concentration_phase`
- `polymer_to_solvent_ratio_raw`
- `polymer_to_drug_ratio_raw`
- `drug_name`
- `drug_mass_mg`
- `drug_concentration_value`
- `drug_concentration_unit`
- `drug_to_polymer_ratio_raw`
- `surfactant_name`
- `surfactant_mass_mg`
- `surfactant_concentration_value`
- `surfactant_concentration_unit`
- `stabilizer_name`
- `helper_material_name`

Core process fields:

- `method_type`
- `solvent_name`
- `co_solvent_name`
- `W1_volume_mL`
- `O_volume_mL`
- `W2_volume_mL`
- `external_aqueous_phase_volume_mL`
- `internal_aqueous_phase_volume_mL`
- `phase_ratio_raw`
- `sonication_time_s`
- `homogenization_time_min`
- `stirring_time_h`
- `evaporation_time_h`
- `centrifugation_g`
- `centrifugation_time_min`

Core measured output fields:

- `ee_percent`
- `lc_percent`
- `dl_percent`
- `particle_size_nm`
- `pdi`
- `zeta_mV`

#### 2. Named extensible variable fields

These are semantically named, paper-class-dependent variables that must not be
collapsed into anonymous slots such as `new_variable_1`.

Current governed member:

- `pH_raw`

Current decision for `pH_raw`:

- retain it as a named extensible variable field
- do not rename it to an anonymous variable slot
- do not yet elevate it to the same benchmark-first status as globally common
  fixed fields such as `particle_size_nm`, `ee_percent`, or `zeta_mV`
- compare it separately from the core fixed field group

Reason:

- current DEV15 coverage is limited and paper-class-specific rather than global
- `pH` already has stable semantic meaning and should remain explicitly named
- in some papers it behaves like an identity-bearing or DOE variable rather
  than a universally expected PLGA core field

#### 3. Provenance / reviewer-only fields

These must remain available for audit and debugging but should not be counted as
field-value benchmark targets.

- `value_source_type`
- `candidate_notes`

### Fields explicitly not benchmark-ready as current fixed targets

The following current columns exist in the wide Layer 3 GT but should not be
framed as current-system benchmark priorities until coverage and extraction
contracts are clarified:

- `helper_material_name`
- `co_solvent_name`
- `W1_volume_mL`
- `internal_aqueous_phase_volume_mL`
- `evaporation_time_h`
- `homogenization_time_min`
- any future sparse paper-local variable added only for one or two papers

These fields may still be reviewed manually and may remain part of the GT row
surface, but they should not drive the first round of recall/accuracy tuning.

## Frozen Layer 3 Compare Contract

### Compare unit

The atomic compare unit is one field cell:

- `(paper_key, gt_formulation_id, field_name)`

Layer 3 compare must not use paper-level or row-summary-only scoring as the
primary benchmark surface.

### Alignment rule

Layer 3 compare inherits formulation alignment from the frozen Layer 2 / Stage5
row universe.

Rules:

- do not reopen formulation-boundary review at Layer 3
- compare only after a stable formulation-row mapping exists
- if row alignment is uncertain, mark the value comparison blocked rather than
  silently comparing against a guessed row

### Value-source rule

For GT:

- only explicitly reported values belong in GT
- calculated or inferred values must remain blank

For system-side compare:

- compare against the system's extracted or source-backed resolved value surface
- track provenance for any relation-resolved value
- do not silently score derivation-only convenience fills as reported-value
  successes

### Paper-local explicit measured-output evidence notes

These notes do not redefine GT by themselves. They document source-backed field semantics that may be used during compare-side debugging of first-week explicit extraction.

Reference rule for future uploads:
- append newly user-supplied paper excerpts, tables, and numeric spans to this section so later debugging can reuse one governed reference surface before inspecting compare outputs
- treat this section as debugging evidence and field-semantics reference, not as automatic GT or system-success promotion

- `5ZXYABSU`
  - User-supplied Table 1 (`Nanoparticle formulations developed`) lists formulation ingredients for `NPR1..NPG3`.
  - User-supplied Table 2 (`Characteristics of the nanoparticle formulations prepared`) reports `Mean particle size`, `ζ-Potential`, and `Encapsulation efficiency ± standard deviation (%)`.
  - No separate `Drug content (%)`, `Drug loading (%)`, or `Loading content (%)` column is present in the supplied table surface.
  - Therefore remaining `lc_percent` misses for this paper should not be force-recovered from those tables without a separate explicit source span.

- `5GIF3D8W`
  - User-supplied optimized-formulation table explicitly separates `Drug content (%)` from `EEc (%)`.
  - Drug-loaded optimized rows report `Drug content (%)` values `1.04 ± 0.06`, `1.14 ± 0.02`, `1.45 ± 0.11`, and `1.44 ± 0.09` for PLGA `50/50`, `75/25`, `85/15`, and `PCL` respectively.
  - User-supplied narrative text states that `percent drug content was low for all batches with a maximum of around 1.45 for PLGA 85/15 and PCL`, confirming the field semantics as explicit drug-content reporting rather than later numeric inference.
  - Compare-side `lc_percent` recovery is allowed only when the aligned system row preserves the same optimized formulation identity and the value is directly supported by the row evidence surface.
  - Current diagnostic boundary: `G036` and `G038` still map to aligned Stage5 rows whose evidence surface is narrative-only `optimized_family_anchor_completion`, not the explicit optimized-table metric tail. Keep these as `missing_in_system` until upstream row evidence materialization restores the direct `Drug content (%)` span.

- `BB3JUVW7`
  - User-supplied materials/methods text establishes two distinct preparation paths in the same paper:
    - `artemether loaded PLGA nanospheres` via standard nanoprecipitation
    - `artemether loaded PLGA nanorods` produced by stretching nanoparticle-embedded polymeric films
  - Source-backed ingredient and formulation constants from the supplied text:
    - artemether `5 mg or 10 mg`
    - PLGA `75 mg` in the nanosphere organic phase
    - acetone `5–15 mL`
    - aqueous PVA `0.25–1% w/v`
    - organic phase added under `400 rpm` stirring
    - nanosphere centrifugation at `20,000 rpm for 40 min`
    - freeze-drying uses trehalose `10% w/v`
  - The nanorod-preparation text establishes process-condition variables not interchangeable with the nanosphere table, including film-forming solution `PVA 5% w/v`, `glycerol 2.5% w/v`, aqueous dispersion `1 mL` equivalent to `10 mg artemether`, film solution `10 mL`, drying `24 h`, stretching rate `10 mm/min`, liquefaction media `acetone` or `silicon oil (65 °C)`, incubation `15 min`, and stretching extent `2x–4x`.
  - User-supplied Table 1 (`Particle size, PDI, %EE, %DL and zeta potential of the artemether loaded nanospheres`) explicitly reports nanosphere fields:
    - composition columns `Artemether (mg)`, `PLGA (mg)`, `PVA (mg)`, `Acetone (mL)`, `Aqueous phase (mL)`
    - measured outputs `Particle size (nm)`, `PDI`, `Zeta potential (mV)`, `%EE`, `%DL`
  - The supplied Table 1 rows confirm explicit nanosphere examples such as:
    - `5 / 75 / 75 / 5 / 15 -> 190.2 ± 18.0 nm, 0.06 ± 0.01, −8.0 ± 0.58 mV, 78.5 ± 1.8 %EE, 4.9 ± 0.1 %DL`
    - `10 / 75 / 150 / 15 / 30 -> 129.3 ± 3.6 nm, 0.06 ± 0.01, −7.4 ± 0.69 mV, 86.9 ± 0.2 %EE, 10.2 ± 0.0 %DL`
  - User-supplied Table 2 (`Physicochemical parameters of nanorods obtained by varying the process conditions`) explicitly reports nanorod fields:
    - process columns `Film thickness (µm)`, `PLGA type (lactide:glycolide)`, `Extent of stretching`, `Liquefaction method`, `Incubation period (min)`
    - measured outputs `Major axis (nm)`, `Minor axis (nm)`, `AR`, `Feret’s diameter (nm)`, `Minor Feret’s diameter (nm)`, `Drug content (µg/mg)`
  - The supplied Table 2 rows confirm explicit nanorod examples such as:
    - `100 / 75:25 / 4x / Acetone / 15 -> 234.1 ± 61.7 major axis, 61.3 ± 8.7 minor axis, AR 3.8 ± 0.8, drug content 1.8 µg/mg`
    - `100 / 75:25 / 4x / Heat / 15 -> 510.7 ± 114.6 major axis, 102.0 ± 23.7 minor axis, AR 5.1 ± 0.9, drug content 3.2 µg/mg`
  - Debugging implication: this paper contains two formulation families with different field surfaces; nanosphere `%DL` and `%EE` must not be confused with nanorod `Drug content (µg/mg)`, and nanorod process-condition rows should be analyzed against the nanorod table rather than forced into the nanosphere composition schema.

- `BXCV5XWB`
  - User-supplied materials text confirms the benchmark-relevant core materials and units:
    - `KGN`
    - `HA sodium salt (MW 39 kDa)`
    - `PLGA (1:1 d,l-lactic to glycolic acid, MW 15.1 kDa)`
    - `PEG-bis-amine (MW 2 kDa)`
    - `PVA (MW 89–98 kDa)`
    - solvents including `ACN`, `DCM`, `methanol`, `diethyl ether`, `DMSO`
  - User-supplied fabrication text establishes a shared nanoprecipitation core for `PLGA` and `PLGA–PEG` particles:
    - polymer mass `50 mg`
    - `ACN 10 mL`
    - precipitation into `0.2% w/v PVA in water (100 mL)`
    - addition rate `0.5 mL/min`
    - optional payload `KGN or FITC (5 mg)` loaded by inclusion in the polymer solution before precipitation
    - stirred at `60 °C for 4 h`
    - collected by centrifugation `15,000 rpm, 20 min, RT (20 °C)` and washed three times
  - The supplied text also establishes a second conjugation step specific to `PLGA–PEG–HA` nanoparticles:
    - HA de-salted against `0.1 M HCl`
    - HA solubilized in `0.01 M MES buffer (5 mg/mL, pH 5.5)`
    - activated with `EDC` and `sulfo-NHS` at `1:5:5 molar ratio` for `2 h` at RT under `N2`
    - blank or KGN-loaded `PLGA–PEG` nanoparticles added at `3:1 w/w HA:nanoparticle`
    - stirred `24 h` at RT under `N2`
    - collected by centrifugation `15,000 rpm, 20 min, RT` and washed three times
  - User-supplied Table 2 (`KGN-loaded nanoparticle properties`) explicitly reports the three formulation identities `PLGA`, `PLGA–PEG`, and `PLGA–PEG–HA` with output fields:
    - `Particle diameter (nm)` with both `DLS` and `TEM`
    - `PDI (AU)`
    - `Zeta (ζ) potential (mV)`
    - `Encapsulation efficiency (%)`
    - `Drug loading (mg KGN/mg nanoparticles)`
    - `HA content (mg HA/mg nanoparticles)`
  - The supplied rows confirm explicit examples:
    - `PLGA -> DLS 166.63 ± 4.48 nm, TEM 84.4 ± 7.2 nm, PDI 0.282 ± 0.023, ζ −33.1 ± 1.6 mV, EE 62.0 ± 3.6, drug loading 0.467 ± 0.192, HA content n/a`
    - `PLGA–PEG -> DLS 297.32 ± 4.55 nm, TEM 164.2 ± 37.5 nm, PDI 0.236 ± 0.014, ζ 11.2 ± 0.3 mV, EE 70.5 ± 4.8, drug loading 0.128 ± 0.026, HA content n/a`
    - `PLGA–PEG–HA -> DLS 507.01 ± 12.03 nm, TEM 182.1 ± 44.9 nm, PDI 0.293 ± 0.021, ζ −28.5 ± 0.9 mV, EE 55.3 ± 11.8, drug loading 0.156 ± 0.033, HA content 0.357 ± 0.086`
  - Debugging implication: `FITC` appears in the fabrication paragraph only as an optional payload variant for `FITC-loaded formulations`; it should not be promoted into GT as an independent benchmark formulation unless the paper reports a separate result-bearing formulation row for it. The main KGN-loaded benchmark-facing rows are the three Table 2 nanoparticle identities above.

### Required compare statuses

Each compare cell must resolve to exactly one primary status:

- `missing_in_system`
- `present_and_match`
- `present_but_mismatch`
- `extra_in_system`
- `blocked_alignment`
- `not_reported_in_gt`

Recommended interpretation:

- `missing_in_system`
  - GT cell is non-empty and system cell is empty
- `present_and_match`
  - GT cell is non-empty, system cell is non-empty, and the selected compare
    mode says they match
- `present_but_mismatch`
  - GT cell is non-empty, system cell is non-empty, and compare says mismatch
- `extra_in_system`
  - GT cell is empty and system cell is non-empty
- `blocked_alignment`
  - row mapping or prerequisite alignment legality failed
- `not_reported_in_gt`
  - GT cell is empty by design and should not be counted as a missed recall item

### Frozen compare groups

All compare outputs must report metrics separately for:

- `core_fixed_fields`
- `named_extensible_variables`

Current contract:

- `pH_raw` belongs only to `named_extensible_variables`
- it must not be merged into core fixed field summaries
- it must not be hidden inside anonymous variable buckets

### Minimum compare outputs

A governed Layer 3 compare run should emit at least:

- one cell-level TSV
- one summary TSV
- one error-bucket TSV

Recommended names:

- `layer3_value_compare_cells_v1.tsv`
- `layer3_value_compare_summary_v1.tsv`
- `layer3_value_error_buckets_v1.tsv`

Minimum cell-level columns:

- `paper_key`
- `gt_formulation_id`
- `matched_system_formulation_id`
- `field_name`
- `field_group`
- `gt_value_raw`
- `system_value_raw`
- `compare_status`
- `strict_match`
- `relaxed_match`
- `canonicalized_match`
- `error_bucket`
- `system_value_source_type`
- `evidence_status_detail`

Minimum summary outputs:

- overall value recall
- overall conditional accuracy
- overall extra-value rate
- per-field recall
- per-field strict / relaxed / canonicalized accuracy
- separate summaries for `core_fixed_fields` and `named_extensible_variables`
- per-paper breakdown for high-risk debugging

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

## Frozen Layer 3 Metric Definitions

### 1. Value recall

Definition:

- denominator: GT cells with non-empty reported values
- numerator: those cells whose aligned system cell is non-empty

This metric answers:

- did the system recover a reported value at all?

### 2. Conditional accuracy

Definition:

- denominator: aligned cells where both GT and system are non-empty
- numerator: cells judged matching under the selected compare mode

All governed compare outputs should report at least:

- `strict`
- `relaxed`
- `canonicalized`

### 3. Extra-value rate

Definition:

- cells where GT is empty but system is non-empty

This metric is required because Layer 3 GT excludes calculation-only or
unsupported inferred values.

### 4. Error buckets

Every mismatch or miss should be assigned an error bucket.

Recommended minimum buckets:

- `missing_value`
- `unsupported_text`
- `unresolved_table`
- `normalization_mismatch`
- `numeric_extraction_mismatch`
- `field_mapping_mismatch`
- `derived_value_leakage`
- `blocked_alignment`

These buckets are the primary debugging surface for deciding whether the next
repair belongs in Stage2 extraction, Stage3 relation resolution, Stage5 field
materialization, evidence binding, or normalization.

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


## Diagnostic Repair Notes

### 2026-04-27 — Stage5 shared preparation solvent carrythrough

- Repair pattern: `PAT_STAGE5_SHARED_PREPARATION_SOLVENT_CARRYTHROUGH_V1`.
- Failure class: `shared_preparation_solvent_not_materialized_to_rows`.
- Earliest failing boundary: Stage5 final-output materialization. Source/original preparation text can contain one unique organic solvent in preparation/dissolution/organic-phase context, while DOE/table rows are retained in the final output with blank `organic_solvent_*` surfaces; Layer3 compare then reports `solvent_name` as `missing_in_system`.
- Generic rule added in `src/stage5_benchmark/build_minimal_final_output_v1.py`: when the row-local `organic_solvent` bundle is blank, fill it only from a unique source-text preparation-context solvent. The rule does not overwrite row-local values, rejects chromatography/mobile-phase/extraction/assay/calibration/materials-list-only contexts, and leaves ambiguous multi-solvent contexts blank.
- Regression tests added in `tests/test_compare_layer3_values_v1.py`: positive unique preparation-solvent carrythrough and negative ambiguous/non-preparation solvent guard. Full related suite passed: `python3 -m unittest tests.test_compare_layer3_values_v1` → `Ran 91 tests ... OK`.
- Diagnostic artifacts:
  - Stage5 replay: `data/results/20260423_9c4a03f/62_stage5_shared_preparation_solvent_diagnostic/`
  - after compare: `data/results/20260423_9c4a03f/64_layer3_compare_shared_preparation_solvent_after/`
  - run context: `data/results/20260423_9c4a03f/62_stage5_shared_preparation_solvent_diagnostic/RUN_CONTEXT.md`
- Diagnostic effect under locked DEV15 GT/scope/alignment: `solvent_name` present_and_match `46 -> 124`, missing_in_system `102 -> 24`, extra_in_system unchanged `2 -> 2`; G2 recall `52.42% -> 62.26%`; total compare error rows `2495 -> 2371`.
- Boundary caveat: these are diagnostic-only artifacts and do not update `data/results/ACTIVE_RUN.json`. The after lineage also contains the prior Stage5 global polymer-material LA:GA carrythrough; the solvent-specific direct effect is the 78 `solvent_name` `missing_in_system -> present_and_match` cells.

## User-Provided Original Source Excerpts For Field-GT Debugging

### INMUTV7L

材料制备段落：

> “2.1. Preparation of Polymeric Nanoparticles
> PLGA nanoparticles (NPs) containing DXI were prepared by using the solvent displacement method [10,11]. In summary, 90 mg of PLGA (Boehringuer Ingelheim®, Ingelheim am Rhein, Germany) and 5 mg of dexibuprofen (Amadis Chemical®, Hangzhou, Zhejiang, China) were weighed and dissolved in 5 mL of acetone. The organic solution was added dropwise into 10 mL of an aqueous surfactant solution at pH 3.5 under magnetic stirring. Following this, the acetone was evaporated under reduced pressure. Various optimized concentrations of surfactants were used (PVA 0.5%, Tween 80® 0.3% and Lutrol F68 (1%). NPs prepared using PVA were centrifuged at 15,000 rpm for 30 min in order to remove excess of PVA. Empty PLGA NPs were prepared using the same protocol but without dexibuprofen.”

表格：

> “Table 1. Characterization of the different formulations developed.
> Formulation Number	Polymer Used	Surfactant	Average Size (nm)	Polydispersity Index (PI)	Zeta Potential (ZP, mV)	EE (%)
> 1	PLGA 503 H	PVA	234.1 ± 0.5	0.081 ± 0.009	−12.2 ± 1.3	93.4
> 2	Tween80®	146.0 ± 0.6	0.054 ± 0.008	−25.2 ± 0.6	87.5
> 3	Lutrol	159.5 ± 0.8	0.058 ± 0.021	−26.0 ± 0.1	85.1
> 4	PLGA-5%	PVA	167.1 ± 1.1	0.080 ± 0.012	−11.8 ± 0.9	95.0
> 5	Tween80®	138.4 ± 1.3	0.072 ± 0.015	−14.1 ± 1.1	91.5
> 6	Lutrol	154.2 ± 1.9	0.063 ± 0.015	−18.7 ± 1.4	93.8
> 7	PLGA 10%	PVA	140.9 ± 1.0	0.055 ± 0.023	−16.7 ± 0.7	99.0
> 8	Tween80®	119.2 ± 1.0	0.074 ± 0.008	−21.2 ± 0.6	99.2
> 9	Lutrol	120.7 ± 0.8	0.071 ± 0.008	−23.1 ± 1.8	91.5
> 10	PLGA 15%	PVA	156.4 ± 0.8	0.078 ± 0.008	−16.2 ± 0.7	92.2
> 11	Tween80®	143.0 ± 0.5	0.062 ± 0.006	−21.4 ± 0.8	93.4
> 12	Lutrol	155.2 ± 1.1	0.076 ± 0.012	−22.5 ± 0.5	94.0”

表格前面的段落：

> “3.1. Characterization of the Nanocarriers
> The formulations were characterized as seen in Table 1. The formulations were optimized using PLGA-PEG 5% for each surfactant used (Lutrol, PVA and Tween80®). Therefore, four different polymers and three of the most commonly used surfactants were assessed. Different PLGA-PEG triblocks were used.
> ”

### BB3JUVW7

原文段落，材料和两种制备方法：

> “2. Materials and methods
> 2.1. Materials
> Artemether (>98.0%) was purchased from Tokyo Chemical Industry Co. Ltd. (Tokyo, Japan). Poly(lactic-co-glycolic) acid (PLGA) (50:50, MW 30,000–60,000) and (75:25, MW 4000–15,000), polyvinyl alcohol (PVA, MW 9000–10,000) were purchased from Sigma-Aldrich Chemicals Company (Missouri, United States). Glycerol was purchased from S D Fine-Chem Ltd (Mumbai, India). Silicon oil was purchased from RankemTM (Bangalore, India). Trehalose was purchased from Spectrochem (Mumbai, India). Emplura® grade acetone was procured from Merck Life Science Pvt. Ltd. (Mumbai, India). Water was obtained from Milli-Q system (Millipore GmbH, Germany). All other chemicals, solvents, and reagents utilized were either HPLC or analytical grade.
> 2.2. Methods
> 2.2.1. Preparation of artemether loaded PLGA nanospheres
> Artemether loaded PLGA nanospheres were prepared by the standard nanoprecipitation method. Briefly, PLGA (75 mg) and artemether (5 mg or 10 mg) were dissolved in acetone (5–15 ml) to prepare the organic phase. The aqueous phase was prepared by dissolving the PVA (molecular weight 9000–10,000) (0.25–1% w/v) in Milli-Q water at 65 ⁰C. The organic phase was added drop-wise to the aqueous phase under constant stirring (400 rpm). Thereafter, the organic phase was evaporated using a rotary vacuum evaporator (Buchi Rotavapor, Switzerland) at 40 ⁰C under reduced pressure. After the evaporation of the organic phase, the aqueous dispersion was centrifuged at 20,000 rpm for 40 min to separate the artemether loaded PLGA nanospheres. The obtained pellet was redispersed in 1 ml of Milli-Q water by sonication. Nanospheres were characterized for particle size, polydispersity index (PDI), % entrapment efficiency, particle shape and % drug loading. The artemether loaded PLGA nanospheres were freeze-dried using trehalose (10% w/v) as a cryoprotectant and stored at 4⁰C until further use.
> 2.2.2. Preparation of artemether loaded PLGA nanorods
> Artemether loaded PLGA nanorods were prepared as reported previously by Champion et. al. with modification (Champion et al., 2007). Briefly, a film-forming solution was prepared by dissolving PVA (5% w/v) and glycerol (2.5% w/v) in Milli-Q water. Thereafter, 1 ml of aqueous dispersion of artemether loaded PLGA nanospheres (equivalent to 10 mg of artemether) was added to the above film-forming solution (10 ml) under stirring which was then added to a 6 × 6 (cm) glass mould and dried at room temperature for 24 h to form the artemether loaded PLGA nanospheres embedded polymeric film. The film was stretched by using an in-house fabricated film stretching apparatus (Fig. 1) in one dimension at the rate of 10 mm/min in acetone or silicon oil (65°C). In the case of acetone, the film was immersed in acetone for 15 min, removed, and stretched in the air. While in case of heat stretching, the film was immersed in hot silicon oil, for 15 min and stretched while still in the oil. The extent of stretching was varied from 2- to 4-fold of the initial length of the film. After stretching, the film was removed from the apparatus and dissolved in Mill-Q-water. The particles were washed with water to remove the PVA absorbed onto the particle surface and separated by centrifugation. The particles were freeze-dried and stored at 4°C until further use.”

表格：

> “Table 1. Particle size, PDI, %EE, %DL and zeta potential of the artemether loaded nanospheres.
>
> Composition	Particle size (nm)	PDI	Zeta potential (mV)	%EE	% DL
> Artemether (mg)	PLGA (mg)	PVA (mg)	Acetone (mL)	Aqueous phase (mL)
> 5	75	75	5	15	190.2 ± 18.0	0.06 ± 0.01	−8.0 ± 0.58	78.5 ± 1.8	4.9 ± 0.1
> 5	75	150	5	15	214.3 ± 6.2	0.238	−1.8 ± 0.22	80.0 ± 0.1	5.0 ± 0.0
> 10	75	300	10	30	196.8 ± 1.1	0.214	−2.2 ± 0.16	78.3 ± 0.0	9.2 ± 0.0
> 10	75	150	10	30	245.5 ± 44.5	0.183	−3.2 ± 0.24	80.6 ± 0.4	9.5 ± 0.0
> 10	75	150	15	30	129.3 ± 3.6	0.06 ± 0.01	−7.4 ± 0.69	86.9 ± 0.2	10.2 ± 0.0
> ”

> “Table 2. Physicochemical parameters of nanorods obtained by varying the process conditions.
>
> Process conditions	Major axis (nm)	Minor axis (nm)	AR	Feret’s diameter (nm)	Minor Feret’s diameter (nm)	Drug content (µg/mg)
> Film thickness (µm)	PLGA type (lactide: glycolide)	Extent of stretching	Liquefaction method	Incubation period (min)
> 100	75:25	4x	Acetone	15	234.1 ± 61.7	61.3 ± 8.7	3.8 ± 0.8	237.2 ± 61.9	77.8 ± 13.3	1.8
> 150	75:25	4x	Acetone	15	295.1 ± 64.9	58.1 ± 12.0	5.1 ± 0.8	265.4 ± 66.3	59.6 ± 15.2	1.4
> 100	50:50	4x	Acetone	15	211.3 ± 44.1	61.1 ± 11.6	3.5 ± 0.7	198.1 ± 37.5	67.7 ± 10.4	2.6
> 100	75:25	2x	Acetone	15	128.1 ± 23.1	61.5 ± 10.3	2.1 ± 0.3	127.5 ± 22.9	61.3 ± 10.4	1.6
> 100	75:25	4x	Heat*	5	241.7 ± 76.3	140.1 ± 31.3	1.7 ± 0.3	294.0 ± 85.8	157.6 ± 32.8	0.9
> 100	75:25	2x	Heat*	15	329.6 ± 79.3	92.2 ± 14.9	3.6 ± 0.7	343.0 ± 79.7	98.9 ± 17.8	0.8
> 100	75:25	4x	Heat*	15	510.7 ± 114.6	102.0 ± 23.7	5.1 ± 0.9	559.5 ± 102.8	119.3 ± 28.8	3.2”

### BXCV5XWB

原文的材料和制备段落：

> “Materials
> KGN was purchased from Abcam (Cambridge, MA). HA sodium salt (molecular weight (MW) 39 kDa) was obtained from Lifecore Biomedical (Chaska, MN). PLGA (1:1 d, l-lactic to glycolic acid, MW 15.1 kDa), PEG-bis-amine (PEG-bis-NH2 MW 2 kDa), poly(vinyl alcohol) (PVA, MW 89–98 kDa), N-hydroxysulfosuccinimide (sulfo-NHS), N,N′-dicyclohexylcarbodiimide (DCC), N-(3-dimethylaminopropyl)-N′-ethylcarbodiimide hydrochloride (EDC), diethyl ether, methanol, dichloromethane (DCM), acetonitrile (ACN), 2-(N-morpholino)ethanesulfonic acid (MES), fluorescein isothiocyanate (FITC), hexadecyltrimethylammonium bromide (CTAB), phosphate buffered saline (PBS), dimethyl sulfoxide (DMSO), dexamethasone, l-proline, ascorbate-2-phosphate, alcian blue, sodium acetate, ethylenediaminetetraacetic acid disodium salt (EDTA), l-cysteine hydrochloric acid (HCl), trifluoroacetic acid (TFA), papain, disodium phosphate, and a Kaiser test kit were obtained from Sigma-Aldrich (St. Louis, MO). Deuterated chloroform was purchased from Cambridge Isotope Laboratories, Inc. (Andover, MA). UranyLess stain solution was purchased from Electron Microscopy Sciences (Hatfield, PA). Cell Counting Kit-8 (CCK-8) was purchased from Dojindo Molecular Technologies (Tokyo, Japan). High and low-glucose Dulbecco’s modified Eagle’s medium (DMEM) and fetal bovine serum (FBS) were purchased from Gibco-BRL (Grand Island, NY). Penicillin–streptomycin and trypsin–EDTA were purchased by Caisson Labs (Smithfield, UT). l-Glutamine, sodium pyruvate, TGF-β1 recombinant human protein, Quant-iT™ PicoGreen™ dsDNA assay kit and insulin/transferrin/selenium (ITS+) premix were purchased from Fisher Scientific (Hampton, NH). A Blyscan sulfated glycosaminoglycan (sGAG) assay was purchased from Biocolor Ltd (Carrickfergus, UK). Bone marrow-derived hMSCs (33 year old female) were purchased from Lonza (Hopkinton, MA). All chemicals were of analytical reagent quality or high performance liquid chromatography (HPLC) grade. Ultrapure water (18.2 MΩ cm Milli-Q, Millipore Sigma, Billerica, MA) was utilized in all experiments requiring water.
>
> Fabrication of PLGA, PLGA–PEG, and PLGA–PEG–HA Nanoparticles
> The PLGA–PEG copolymer was synthesized as previously reported.38 All nanoparticles were prepared using nanoprecipitation. PLGA or PLGA–PEG (50 mg) was dissolved in ACN (10 mL) and added to 0.2% w/v PVA in water (100 mL) at 0.5 mL/min. KGN or FITC (5 mg) were included in the polymer solution prior to precipitation for KGN- or FITC-loaded formulations. This solution was stirred at 60 °C for 4 h before collecting the nanoparticles by centrifugation [15,000 rpm, 20 min, room temperature (RT, 20 °C)] and washing three times with water.
>
> For PLGA–PEG–HA nanoparticles, HA was de-salted by dialyzing against 0.1 M HCl. De-salted HA was solubilized in 0.01 M MES buffer (5 mg/mL, pH 5.5) and activated using EDC and sulfo-NHS (1:5:5 molar ratio) for 2 h at RT under N2. Blank or KGN-loaded PLGA–PEG nanoparticles in MES buffer were added to the activated HA solution (3:1 w/w HA:nanoparticle), and stirred for 24 h at RT under N2. The nanoparticles were collected by centrifugation (15,000 rpm, 20 min, RT) and washed three times.”

具体表格：

> “Table 2 KGN-loaded nanoparticle properties.
> From: Effects of Nanoparticle Properties on Kartogenin Delivery and Interactions with Mesenchymal Stem Cells
>
> KGN-loaded nanoparticles
>
> Particle diameter (nm)
>
> PDI (AU)a
>
> Zeta (ζ) potential (mV)c
>
> Encapsulation efficiency (%)d
>
> Drug loading (mg KGN/mg nanoparticles)d
>
> HA content (mg HA/mg nanoparticles)e
>
> DLSa
>
> TEMb
>
> PLGA
>
> 166.63 ± 4.48
>
> 84.4 ± 7.2
>
> 0.282 ± 0.023
>
> − 33.1 ± 1.6
>
> 62.0 ± 3.6
>
> 0.467 ± 0.192
>
> n/a
>
> PLGA–PEG
>
> 297.32 ± 4.55
>
> 164.2 ± 37.5
>
> 0.236 ± 0.014
>
> 11.2 ± 0.3
>
> 70.5 ± 4.8
>
> 0.128 ± 0.026
>
> n/a
>
> PLGA–PEG–HA
>
> 507.01 ± 12.03
>
> 182.1 ± 44.9
>
> 0.293 ± 0.021
>
> − 28.5 ± 0.9
>
> 55.3 ± 11.8
>
> 0.156 ± 0.033
>
> 0.357 ± 0.086
>
> n/a not applicable
> aHydrodynamic diameter determined by DLS
> bDry diameter determined by TEM
> cMeasured using a Zetasizer Nano-ZS equipped with a standard capillary electrophoresis cell
> dDetermined by HPLC
> eDetermined by CTAB assay”

### L3H2RS2H

段落原文：

> “2. Materials and methods
> 2.1. Materials
> Xanthone (XAN), PLGA (50:50) MW 50 000–75 000, Pluronic F-68, phosphate buffered saline tablets and soybean lecithin (40% purity by thin-layer chromatography) were purchased from Sigma-Aldrich Química (Sintra, Portugal). Threalose dihydrate was purchased from Fluka (Sintra, Portugal). 3-Methoxyxanthone (3-MeOXAN) was synthesised in our laboratory according to Fernandes et al. [22]. Myritol 318 (caprilic/capric acid triglyceride) was kindly supplied by Henkel (Lisboa, Portugal). HPLC-grade methanol and acetonitrile were obtained from Merck (Lisboa, Portugal). Water was purified by reverse osmosis Milli-Q system (Millipore®, Lisboa, Portugal). Other chemicals were of analytical grade.
> 2.2. Preparation of nanospheres
> Nanospheres containing XAN or 3-MeOXAN were prepared by the solvent displacement technique described by Fessi et al. [18]. Briefly, an organic solution of PLGA (63 mg) and different amounts of XAN or 3-MeOXAN in acetone (10 mL) was poured, under moderate magnetic stirring, into 10 mL of an aqueous solution of Pluronic® F-68 0.25% (w/v). Following 5 min of stirring, the volume of nanosphere dispersion was concentrated to 10 mL under reduced pressure. Separation of non-encapsulated compounds was performed by centrifugation at 2000×g for 30 min. (Heraeus Sepatec centrifuge, Porto, Portugal) after solubilization of a certain amount of threalose dihydrate for achieving a 5% (w/v) concentration. The supernatant was discarded and the pellet containing the nanospheres was redispersed in water to complete the initial volume of nanosphere dispersion submitted to centrifugation. Empty nanospheres were prepared according to the same procedure but omitting the xanthones in the organic phase.”

> “2.4. Preparation of nanocapsules
> XAN and 3-MeOXAN-containing PLGA nanocapsules were prepared by the interfacial polymer deposition technique described by Fessi et al. [19]. Briefly, about 50 mg of polymer and 100 mg of soybean lecithin were dissolved in 10 mL of acetone. Different amounts of XAN or 3-MeOXAN were dissolved in 0.5–0.6 mL of Myritol® 318 and the solution obtained was added to the previously prepared acetonic solution. The final solution was poured into 20 mL of an aqueous solution of Pluronic® F-68 0.5% (w/v) under moderate stirring, leading to the formation of the nanocapsules. Then, acetone was removed under vacuum and the colloidal dispersion of nanocapsules was concentrated to 5–10 mL by evaporation under reduced pressure. The amount of non-encapsulated xanthones (either XAN or 3-MeOXAN) was separated by ultrafiltration/centrifugation technique [19] using centrifugal filter devices (Centricon YM-50, Millipore®, Lisboa, Portugal) at 4000×g for 2 h (Beckman UL-80 ultracentrifuge, Albertville, USA). Empty nanocapsules were prepared according to the same procedure but omitting the xanthones in the organic phase.
> XAN and 3-MeOXAN nanoemulsions were prepared as nanocapsules, omitting the polymer in the organic phase.”

表格：

> “Table 1. Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanospheres
>
> Theoretical concentration (μg/mL)	XAN nanospheres	3-MeOXAN nanospheres
> Final concentration (μg/mL)	Encapsulation efficiency (%)	Final concentration (μg/mL)	Encapsulation efficiency (%)
> 50	13.0±1.1	26.1±2.1	19.0±0.6	38.1±1.1
> 60	20.0±2.4	33.0±4.1	24.9±4.6	41.5±7.6
> 70	Crystals of XAN	ND	Crystals of 3-MeOXAN	ND
> 80	Crystals of XAN	ND	Crystals of 3-MeOXAN	ND
> Values express the mean results±SD values of three different batches. ND, not determined.“

> ”Table 2. Mean diameter, polydispersity index (PI) and zeta potential (ζ) of PLGA empty and loaded nanospheres
>
> Empty Cell	Emptynanospheres	XAN nanospheresa	3-MeOXAN nanospheresb
> Diameter (nm)	154±6	164±8	164±9
> PI	0.06±0.03	0.06±0.03	0.06±0.01
> ζ (mV)	−36.2±5.2	−38.9±1.3	−36.0±3.0
> Values express the mean results±SD values of three different batches.
> a
> XAN nanosphere with theoretical concentration of 60 μg/mL.
> b
> 3-MeOXAN nanospheres with theoretical concentration of 60 μg/mL.“

> ”Table 3. Encapsulation parameters of XAN and 3-MeOXAN in PLGA nanocapsules
>
> XAN nanocapsules	3-MeOXAN nanocapsules
> Theoretical concentration (μg/mL)	XAN/Myritol 318% (w/v)	Final concentration (μg/mL)	Encapsulation efficiency (%)	Theoretical concentration (μg/mL)	3-MeOXAN/Myritol 318% (w/v)	Final concentration (μg/mL)	Encapsulation efficiency (%)
> 200	0.4	178±21	89±11	1000	2.0	887±51	89±5
> 400	0.8	342±18	85±5	1200	2.4	918±9	77±1
> 600	1.2	529±57	88±9	1400	2.8	1162±80	83±6
> 700	1.4	Crystals of XAN	ND	1600	3.2	Crystals of 3-MeOXAN	ND
> 800	1.6	Crystals of XAN	ND				
> Values express the mean results±SD values of three different batches. ND, not determined.“

> ”Table 4. Mean diameter, polydispersity index (PI) and zeta potential (ζ) of empty and loaded PLGA nanocapsules
>
> Empty Cell	Empty nanocapsules	XAN nanocapsulesa	3-MeOXAN nanocapsulesb
> Diameter (nm)	274±3	278±15	280±19
> PI	0.455±0.130	0.412±0.051	0.376±0.104
> ζ (mV)	−40.9±5.9	−39.1±0.7	−39.5±4.7
> Values express the mean results±SD values of three different batches.
> a
> XAN nanocapsules with theoretical concentration of 600 μg/mL.
> b
> 3-MeOXAN nanocapsules with theoretical concentration of 1400 μg/mL.
> Table 5. Mean diameter, polydispersity index (PI), zeta potential (ζ) and incorporation parameters of various nanocapsule formulations: empty nanocapsules (0.6 mL Myritol 318 and without xanthones), XAN-loaded nanocapsules (0.6 mL Myritol 318, XAN theoretical concentration of 1440 μg/mL) and 3-MeOXAN-loaded nanocapsules (0.6 mL Myritol 318, 3-MeOXAN theoretical concentration of 3360 μg/mL)
>
> Sample	Theoretical concentration (μg/mL)	Final concentration (μg/mL) [n]	Encapsulation efficiency (%)	Diameter (nm) [n]	PI	ζ (mV) [n]
> Empty nanocapsules	–	–	–	261±17 [5]	0.48±0.06	−36.3±4.3 [5]
> XAN nanocapsules	1440	1173±100 [7]	82±7	273±18 [5]	0.48±0.05	−36.4±9.3 [5]
> 3-MeOXAN nanocapsules	3360	2780±238 [7]	83±7	271±16 [5]	0.43±0.03	−41.8±5.4 [5]
> “

### PA3SPZ28

段落原文：

> “Materials
> PLGA 50:50 (PURASORB PDLG 5010) was obtained as a gift from Purac Biomaterials. FITC and 4′,6-diamidino-2-phenylindole (DAPI) were purchased from Sigma-Aldrich. Vitamin E TPGS NF Grade was gifted by Antares Health Products. HPLC grade acetone was purchased from Spectrochem (Mumbai, India). All other solvents and chemicals were of analytical grade and procured from Merck, India. 99MoO4 − was purchased from Bhabha Atomic Research Centre (Mumbai, India) and 99mTcO4 − was extracted from a 5 N NaOH solution of 99MoO4 − by butan-2-one. ECIL gamma counter (Model LV 4755) obtained from ECIL was used for measuring tissue and organ radioactivities. Scintigraphic imaging studies of animals were done in GE Infina Gamma camera equipped with Xeleris work station.
>
> Preparation of GAR-loaded nanoparticles (GAR-NPs)
> Nanoparticles were prepared by the nanoprecipitation method20. Briefly, an organic solution of PLGA (50 mg) and GAR (5 mg) in acetone (10 ml) was added to an aqueous vitamin E TPGS solution (20 ml, 0.03% w/v) under magnetic stirring at room temperature. The solvent was allowed to evaporate overnight. The suspension obtained was filtered (Whatman filter paper 1) to remove any precipitate and centrifuged at 18,000 rpm at 4 °C (Sorvall RC 5 Plus). The supernatant containing the free drug was discarded; the pellet obtained was washed 2–3 times with distilled water and lyophilised (VirTis, USA) for 48 h to get a free flowing powder.
>
> Drug free nanoparticles were prepared according to the same procedure. Fluorescent nanoparticles were prepared by replacing GAR with FITC (5 mg).“

表格：

> “Table1 Characterizaton of GAR-NPs.
> From: Garcinol loaded vitamin E TPGS emulsified PLGA nanoparticles: preparation, physicochemical characterization, in vitro and in vivo studies
>
> Parameters
>
> Nanoprecipitation method
>
> After storage at 4 °C for 3 months
>
> Drug:Polymer ratio
>
> 1:20
>
> 1:10
>
> 1:6.66
>
> 1:10
>
> PLGA grade
>
> 50:50
>
> 50:50
>
> 50:50
>
> 50:50
>
> Size (nm)
>
> 90.21 ± 2.2
>
> 88.05 ± 2.7
>
> 95.45 ± 2.4
>
> 100 ± 4.2
>
> PDI
>
> 0.212 ± 0.07
>
> 0.170 ± 0.05
>
> 0.237 ± 0.09
>
> 0.295 ± 0.08
>
> Zeta potential (mV)
>
> −30 ± 2.2
>
> −28.1 ± 2.1
>
> −33 ± 2.4
>
> −27.2 ± 2.2
>
> Drug loading (DL, %)
>
> 4.12 ± 1.9
>
> 9.25 ± 2.8
>
> 5.29 ± 2.1
>
>  
> Encapsulation efficiency (EE, %)
>
> 53.43 ± 2.8
>
> 88.32 ± 3.3
>
> 60.61 ± 3.5
>
>  ”

### QLYKLPKT

段落原文：

> “Materials
> PLGA (polylactic-glycolic acid ratio: 50:50, molecular weight 30,000–60,000 Da), poloxamer 188 (Pluronic® F-68), ITZ, Dulbecco’s phosphate buffered saline (Life Technologies, Carlsbad CA, USA), dextrose, sucrose, mannitol, acetone, acetonitrile, and HP-β-CD were all purchased from Sigma-Aldrich (St Louis, MO, USA). Sporanox® was purchased from Walgreens (Deerfield IL, USA). Deionized water (Thermo Fisher Scientific, Waltham, MA, USA) was used throughout the study to prepare the solution and mobile phase.
>
> Preparation of PLGA-ITZ-NS
> The unloaded PLGA nanoparticles and the PLGA-ITZ-NS were synthesized by a nanoprecipitation method.Citation16 In addition, a 1% (w/v) PLGA solution was prepared by dissolving the PLGA (50:50) in acetone. To get an optimal concentration of the surfactant used to prepare PLGA nanoparticles, 5 mL of PLGA (1% w/v) solution was slowly added to four aqueous solutions (50 mL of each) containing 2.5, 3, 4, and 10 mg/mL of nonionic surfactant (poloxamer 188) with stirring. After the optimal surfactant concentration had been determined, various amounts of ITZ were dissolved into three solutions (5 mL of each) of 1% (w/v) PLGA in acetone to obtain PLGA:ITZ (w/w) ratios of 5:1, 10:1, and 15:1, respectively. The organic phase was added at a constant rate of 0.3 mL per minute with stirring to 50 mL of the aqueous phase, containing an optimal concentration of poloxamer 188. The resulting mixture turned milky instantaneously because of the formation of NS by the solvent displacement and polymer deposition. Acetone was removed by evaporation under vacuum at 60°C for 6 hours; about 30 mL of PLGA-ITZ-NS suspension was obtained. Finally, the suspension was filtered and centrifuged to remove extra free drug and surfactant.”

> “To discover the optimal concentration of the stabilizer used in the NS formulation with the nanoprecipitation method the PLGA concentration (1% w/v) was kept constant, and the four different poloxamer 188 concentrations were used to prepare unloaded PLGA nanoparticles. As shown in Table 1, the zeta potentials of all the four preparations were between −22 mV and −28 mV, which can prevent aggregation by repulse particles with each other. When the poloxamer 188 concentration was 3 mg/mL, the optimal PLGA-ITZ-NS formulation was obtained with the smallest particle size (189±5 nm) and PDI (0.08±0.02). Therefore, 3 mg/mL of poloxamer 188 was chosen for the preparation of PLGA-ITZ-NS during the whole study.
>
> The properties of PLGA-ITZ-NS synthesized with different PLGA: ITZ (w/w) initial ratios were summarized in Table 2. Again, all the preparations had negative zeta potentials, which can provide steric stability to the particles. PLGA-ITZ-NS, with a ratio of 5:1 and 10:1, gave similar small particle sizes, but the latter had lower PDI and higher EE, which were more preferable. NS, with a ratio of 15:1, provided similar EE as NS with its ratio of 10:1, but the particle size was larger. Therefore, the PLGA-ITZ prepared with an initial ratio of 10:1 was chosen as the optimal formulation when considering particle size, PDI and EE comprehensively, and was utilized for the formulation of all the following studies.”

表格：

> “Table 1 Physicochemical properties of unloaded PLGA nanoparticles stabilized by different concentrations of poloxamer 188
> Poloxamer 188 (mg/mL)	Particle size (nm)	PDI	Zeta potential (mV)
> 2.5	299±33	0.45±0.08	−28±1
> 3	189±5	0.08±0.02	−22±0
> 4	232±19	0.07±0.04	−24±0
> 10	481±41	0.29±0.03	−23±1
> Note: 3 mg/mL was selected as optimal surfactant concentration to prepare PLGA-ITZ-NS for the remaining studies.
>
> Abbreviations: PLGA, poly(lactic-co-glycolic acid); PDI, polydispersity index; PLGA-ITZ-NS, itraconazole-loaded poly(lactic-co-glycolic acid) nanospheres.”

> “Table 2 Physicochemical properties of PLGA-ITZ-NS with different PLGA:ITZ initial ratios
> PLGA:ITZ (w/w)	Particle size (nm)	PDI	Zeta potential (mV)	EE%
> 5:1	175±5	0.41 ±0.03	−17±1	61 ±4
> 10:1	178±6	0.19±0.03	−20±1	72±1
> 15:1	191±2	0.14±0.02	−30±1	73±1
> Note: Ratio of 10:1 was then selected to prepare PLGA-ITZ-NS for the remaining studies.
>
> Abbreviations: PLGA-ITZ-NS, itraconazole-loaded poly(lactic-co-glycolic acid) nanospheres; PLGA, poly(lactic-co-glycolic acid); ITZ, itraconazole; PDI, polydispersity index; EE, encapsulation efficiency.
>
> “

### RHMJWZX8

段落：

> “Materials
> Substances and reagents
> The PUE standard (Batch No. 110752-200912) was acquired from the National Institutes for Food and Drug Control (Beijing, China). The AP sample (purity ≥ 98%) was a gift from Shandong Academy of Medical Sciences (Jinan, China). PLGA (lactide/glycolide = 50 : 50, molecular weight = 10 000) was obtained from Jinan Daigang Biomaterial Co., Ltd. (Jinan, China). Polysorbate 80 (Tween 80) was purchased from Sinopharm Chemical Reagent Co., Ltd. (Shanghai, China). Methanol and acetonitrile of chromatographic grade were supplied by the Fisher Company (Fair Lawn, NJ, USA). The dialysis membrane was obtained from the Viskase Company, Inc. (Darien, IL, USA). All other chemicals and reagents used were of analytical grade.”

> “Preparation and characterization of AP-PLGA-NPs
> AP-PLGA-NPs were prepared by a solvent diffusion methodology. Briefly, 18 mg of PLGA and 7 mg of AP were dissolved in 1 ml of acetone to form an organic phase; then the organic phase was poured into 4 ml of stirred aqueous phase (containing 1% polysorbate 80). The formed O/W emulsion was stirred continuously at 800 rpm at 40°C for 12 h with a magnetic stir bar, and the final NP suspension was subsequently filtered through a membrane (0.8-μm pore size) to remove the non-incorporated AP.27, 28
>
> The preparation method was optimized with a homogeneous design combined with response surface methodology. The PLGA concentration, the volume ratio of the aqueous phase to the oil phase and the AP feeding amount were selected as the factors and were evaluated according to U*12 (1210) homogeneous design. The encapsulation efficiency (EE) and drug loading (DL) of AP-PLGA-NPs, which were taken as the response variables, were determined with a centrifugation ultrafiltration method.29“

> “A schematic diagram showing the preparation of the NPs is illustrated in Figure 2a. The prepared NPs were spherical and uniform in shape under TEM (Figure 2b), with a particle size of 145.0 ± 1.3 nm, a polydispersity index of 0.153 and a zeta potential of −14.81 ± 1.39 mV. The optimized EE and DL values were 90.51 ± 0.28% and 17.07 ± 0.33%, respectively. The in-vitro drug release profile (Figure 3) of AP from the PLGA-NPs showed an initial burst release followed by a sustained release, which was consistent with the Higuchi's diffusion mechanism, while the crude drug release followed first-order kinetic model.”

> “Zeta potential can influence physical stability of a colloidal dispersion. High negative charge of zeta potential indicates that the electrostatic repulsion between particles will prevent the aggregation of the spheres and thus stabilize particle suspensions.26 The zeta potential of AP-PLGA-NPs was found to be −14.81 ± 1.39 mV, whereas that of empty NPs was −36.13 ± 3.35 mV. Negative value of zeta potential might result from the negative charge of PLGA.38”

### UFXX9WXE

段落：

> “. Materials and Methods
> 2.1. Materials
> Poly (D, L-lactide-co-glycolic acid) (PLGA) 50 : 50 (molecular weight 30,000–60,000) and poloxamer 407 were purchased from Sigma-Aldrich, St. Louis, USA. Lorazepam was purchased from R L Fine Chem., Bangalore, India. HPLC grade acetone and water were purchased from Fisher Scientific, Mumbai, India. All other solvents were of HPLC grade.
>
> 2.2. Experimental Design
> Box-Behnken design was employed for constructing polynomial model for optimization of Lzp-PLGA-NPs keeping 4 independent and 2 dependent variables using Design Expert (version 8.0.0, Stat-Ease Inc., Minneapolis, Minnesota). Box-Behnken design was selected for the study as it generates fewer runs with 4 independent variables. The independent and dependent variables are listed in Table 1. The polynomial equation generated by the experimental design is as follows:”

> “2.3. Nanoparticles Preparation
> Lzp-PLGA-NPs were prepared using emulsion solvent evaporation (nanoprecipitation) method. During the process, the organic phase was prepared by dissolving accurately weighed PLGA and lorazepam in acetone as organic solvent. The organic phase was then added drop wise at the rate of 1ml/min into an aqueous phase containing surfactant (poloxamer 407) dissolved in water as aqueous solvent. The nanoparticles suspension was kept under continuous stirring at 300 rpm (RPM preoptimized, data not shown) for 3 h at 30°C to allow the complete evaporation of acetone, leaving behind the colloidal suspension of Lzp-PLGA-NPs in aqueous phase.
> ”

> “3.4. Data Analysis and Optimization
> The optimum Lzp-PLGA-NPs formulation was selected by applying constraints on the dependent factors as shown in Table 1. Point prediction of the Design Expert software was used to determine the optimized NPs on the basis of closeness of desirability factor close to 1, which predicted the optimized process parameters to be X1 10 mg/mL, X2 9.42 mg/mL, X3 10, and X4 4.5 mg/mL with predicted values of responses Y1 170.5 d·nm and Y2 86.81%. The optimized formulation was developed and characterized for z-average and % drug entrapment. The experimental value for responses Y1 168.2 d·nm with PDI 0.08 and Y2 83.8% of optimized formulation was found in good agreement with the predicted values generated by the RSM and the result assures the validity of RSM model.
>
> The percentage drug loading of optimized Lzp-PLGA-NPs was calculated using (3) and it was found to be 8.7%.”

表格：

> “Table 1. Independent and dependent variables levels in Box-Behnken design.
>  	Levels
> −1	0	1
> Independent variables	 	 	 
> X1 = polymer concentration (w/v)	10	35	60
> X2 = surfactant concentration (w/v)	2	8.50	15
> X3 = aqueous/organic phase ratio (v/v)	2	6	10
> X4 = drug concentration (w/v)	1	3	5
>   
> Dependent Variables:	Constraints
> Y1 = z-average (d·nm)	Minimize
> Y2 = % drug entrapment	Maximize”

> “Table 2. Effect of independent process variables on dependent variable.
> Formulation	PLGA mg/mL	Poloxamer mg/mL	w/o phase volume ratio	Drug conc. Mg/mL	
> z-Average d·nm
>
> (±SD)
>
> % Drug entrapment
>
> (±SD)
>
> PDI
>
> (±SD)
>
> 1.	35	2	6	1	211 ± 0.11	70 ± 1.3	0.183 ± 0.002
> 2.	35	2	6	5	220 ± 0.8	88.48 ± 0.8	0.150 ± 0.003
> 3.	10	8.50	10	3	176 ± 0.5	83 ± 0.5	0.048 ± 0.001
> 4.	35	8.50	2	1	177 ± 1.2	71 ± 1.5	0.17 ± 0.004
> 5.	10	2	6	3	205 ± 0.9	81 ± 0.7	0.315 ± 0.003
> 6.	10	8.50	6	5	177 ± 1.6	83.5 ± 0.5	0.110 ± 0.002
> 7.	10	8.50	2	3	184 ± 1.5	75 ± 0.35	0.078 ± 0.002
> 8.	35	8.50	6	3	197 ± 0.5	86.6 ± 0.65	0.112 ± 0.004
> 9.	60	8.50	6	1	271 ± 0.8	76 ± 0.22	0.24 ± 0.001
> 10.	35	15	6	5	192 ± 1.4	84.3 ± 0.35	0.19 ± 0.001
> 11.	10	15	6	3	177 ± 0.6	80 ± 2.1	0.04 ± 0.003
> 12.	60	2	6	3	318 ± 1.2	90.1 ± 0.8	0.441 ± 0.002
> 13.	35	15	10	3	191 ± 1.5	83.5 ± 1	0.17 ± 0.003
> 14.	60	15	6	3	228 ± 0.5	82 ± 0.4	0.15 ± 0.001
> 15.	60	8.50	2	3	241 ± 0.4	88 ± 0.85	0.309 ± 0.005
> 16.	35	15	6	1	182.5 ± 0.5	66.4 ± 0.2	0.09 ± 0.002
> 17.	35	2	2	3	215 ± 0.7	87.83 ± 0.1	0.15 ± 0.005
> 18.	60	8.50	6	5	261 ± 0.5	89 ± 1.7	0.20 ± 0.001
> 19.	35	8.50	2	5	193 ± 1.1	88 ± 1.5	0.10 ± 0.002
> 20.	10	8.50	6	1	167 ± 0.8	65.5 ± 1.1	0.21 ± 0.005
> 21.	35	8.50	10	1	192 ± 1.7	69 ± 0.6	0.28 ± 0.001
> 22.	35	2	10	3	241 ± 1.5	88 ± 1.5	0.21 ± 0.003
> 23.	35	8.50	10	5	202 ± 1.2	87 ± 1	0.19 ± 0.002
> 24.	35	15	2	3	186 ± 1.5	84 ± 0.8	0.15 ± 0.002
> 25.	60	8.50	10	3	283 ± 0.7	88 ± 1.4	0.15 ± 0.006
> 26.	35	8.50	6	3	193 ± 0.5	85.12 ± 0.7	0.102 ± 0.004“

### V99GKZEI

段落：

> “2.1 Materials
> Methylene Blue (tetramethylthionine chloride, MB, MW = 319.86 a.m.u.) was received as gift sample from A.C.E.F. S.p.A. (Fiorenzuola d’Arda, Italy). Polylactic-co-glycolic acid (PLGA, RG502H, 50 : 50; MW range 7000–17 000), polyoxyethylene sorbitan monoleate (Tween® 80), Pluronic® F68, Span® 80 and all other chemicals and solvents were of analytical grade and obtained from Sigma-Aldrich (Milano, Italy). Heptakis(2-O-oligo(ethylene oxide)-6-hexylthio)-β-CyD (SC6OH, MW (nEO = 32, EO = ethylene oxide) = 3250 a.m.u) was synthesized according to general procedures.34 All dispersions used for spectroscopic characterizations were prepared in pure microfiltered water (Galenica Senese, Siena, Italy). Deionized, double distilled water was used throughout the study. All solvents were filtered through 0.22 μm Millipore® GSWP filters (Bedford, USA).”

> “2.3 Preparation of NPs
> MB loaded-PLGA/SC6OH NPs were prepared by the nanoprecipitation/solvent displacement method. Briefly, MB (0.5 mg), PLGA (20 mg) and different amounts of SC6OH (10, 20, 30 and 40 mg) were dissolved in acetone (6 ml). The organic phase was poured into 20 ml of aqueous solution under magnetic stirring, thus forming a milky colloidal suspension. The suspensions were left to stir overnight to eliminate the organic solvent.37 NPs were then purified by centrifugation at 5000 rpm for 15 min, collecting and centrifuging again the supernatants at 18 000 rpm for 30 min with a Beckman Optima™ XL-100K centrifuge. The supernatants were eliminated. The pellets were re-suspended in 1 ml of water, and then freeze-dried (VirtTis Benchtop K Instrument, SP 127 Scientific, USA) after addition of trehalose (5% w/v) as a lyoprotectant agent. MB loaded-PLGA NPs were prepared dissolving MB (0.5 mg), PLGA (20 mg) and Tween 80® (0.5% w/v) in acetone (6 ml) and adopting the previous procedure.
>
> Additional MB loaded-PLGA NPs were obtained by the W/O/W emulsion/solvent evaporation method by dissolution of MB (0.5 mg) in 1 ml of water. This solution was added dropwise to 5 ml of an acetone/dichloromethane mixture (1 : 1, v/v) containing PLGA (20 mg) and Span® 80 (0.5%, w/v), under sonication over an ice bath for 2 min. This first emulsion was added dropwise to 50 ml of an aqueous solution containing Pluronic® F68 (3%, w/v) under sonication for 1 min.38 The colloidal suspension was stirred overnight to eliminate the organic solvent, and then was subjected to the same procedure described above (purification by centrifugation and lyophilization in presence of trehalose).“

表格：

> ”Table 1 Overall properties of NPs (sizes), polydispersity index (P.I.), yield percentage and encapsulation efficiency percentage (E.E.%) of MB loaded-PLGA NPs prepared in the presence and in the absence of SC6OH. SD was calculated from at least three different batches
> NPs compositiona	Sizes (nm) ± S.D.	P. I. ± S.D.	Yield% ± S.D.	D.C.% ± S.D.	E.E.% ± S.D.
> MB loaded-PLGAb	220 ± 4	0.19 ± 0.02	43.21 ± 2.69	0.52 ± 0.19	3.12 ± 1.12
> MB loaded-PLGA (W/O/W)c	266 ± 5	0.40 ± 0.10	39.98 ± 6.32	1.13 ± 0.26	6.75 ± 1.54
> MB loaded-PLGA/SC6OH10b	230 ± 6	0.13 ± 0.06	36.12 ± 3.21	5.02 ± 0.43	30.09 ± 2.58
> MB loaded-PLGA/SC6OH20b	221 ± 2	0.23 ± 0.08	40.58 ± 1.89	4.86 ± 0.38	29.14 ± 2.31
> MB loaded-PLGA/SC6OH30b	226 ± 5	0.15 ± 0.04	34.14 ± 5.61	5.43 ± 0.61	32.56 ± 3.65
> MB loaded-PLGA/SC6OH40b	201 ± 2	0.16 ± 0.08	40.85 ± 4.01	9.65 ± 0.31	57.89 ± 1.86
> a The amounts of MB and PLGA were always maintained at 0.5 mg and 20 mg, respectively. The amount of SC6OH was changed from 10 to 40 mg. b NPs were prepared with the nanoprecipitation/solvent displacement method using acetone as the organic phase. c NPs were prepared with the W/O/W emulsion/solvent evaporation method using a mixture of acetone/dichloromethane (1 : 1, v/v) as the organic phase; Tween 80 and Pluronic F68® were used as surfactants.“

### WFDTQ4VX

段落：

> “Materials
> Lopinavir was obtained as a gift sample from Aurobindo Pharmaceuticals, Hyderabad, India. Poly (lactide-co-glycolide) PLGA (50:50) (inherent viscosity 0.2 dl/g) was gift sample from Purac Biomaterials, The Netherlands. Pluronic F 68 was obtained as gift sample from BASF, Germany. Brij 35 was purchased from Sigma Aldrich, Germany. Caco-2 cells were obtained from National Centre for cell line studies (NCCS), Pune, India. DMEM medium, penicillin–streptomycin solution, Trypsin-EDTA solution, fetal bovine serum (FBS) and Hank’s balanced salt solution (HBSS) were purchased from Himedia, Mumbai, India. All other chemicals used were of analytical grade. The 24-well Transwell inserts were purchased from Nunc, Roskilde, Denmark. The 6-, 24- and 96-well plates were purchased from Costar Corning, NY, USA. MTT assay dye was purchased from Himedia, Mumbai, India.
>
> Formulation of lopinavir-loaded PLGA NPs
> NPs were prepared using solvent diffusion (nanoprecipitation) method (Fessi et al., Citation1989). The NPs were prepared by dissolving PLGA (25 mg) and drug (15 mg) in 2.5 ml of acetone. The organic phase was added at the rate of 0.5 ml/min into 5 ml of aqueous phase containing 0.25% w/v Pluronic F68 with continuous stirring on magnetic stirrer at room temperature (Shah et al., Citation2014). Stirring was continued until the complete evaporation of organic solvent. The NPs suspension was centrifuged at 25 000 g for 30 min at 4 °C (3K30, Sigma Centrifuge, Osterode, Germany), supernatant was alienated and NPs were collected. Nanoparticle suspension was lyophilized using sucrose as cryoprotectant (1:1).“

> “Optimization and characterization of NPs
> Twenty seven batches of lopinavir-loaded PLGA NPs were prepared by 33 factorial design. Drug concentration, polymer concentration and surfactant concentration were the major independent factors selected on the basis of preliminary optimization. The coded and actual values of the formulation parameters are mentioned in Table 1. The particle size (PS) and entrapment efficiency (EE) obtained at various levels of three independent variables (X1, X2 and X3) were subjected to multiple regressions to yield second order polynomial equations (EquationEquation 1 and Equation2, full model). The main effects of X1, X2 and X3 represent the average result of changing one variable at a time from its low to high value. The interactions (X1X2, X1X3, X2X3 and X1X2X3) show how the PS and EE changes when two or more variables were simultaneously changed. The PS and EE of total 27 batches showed a wide variation from 126.6 ± 4.16 to 237.1 ± 1.37 nm and 21.03 ± 2.28 to 95.7 ± 2.43%, respectively (Table 2)”

> “Optimized NPs with desirability criteria
> From the results, the optimum levels of independent variables were screened by multiple regression analysis. Our desirability criteria were maximum entrapment with minimum particle size (less than 200 nm). Since PS and EE were taken into consideration simultaneously, the batch with smallest particle size of 126.6 ± 4.16 nm exhibited EE near to 22% (at X1 = −1, X2 = −1.0, X3 = −1.0) while that with highest EE of 95.7 ± 2.43% produced particle size greater than 200 nm (at X1 = 1, X2 = 0.0 X3 = −1). Hence, the optimum formulation with EE 93.03 ± 1.27% and particle size 142.1 ± 2.13 nm (Figure 3) found at 1.0, −1, and −1 levels of X1, X2 and X3, respectively, was selected.”

表格：

> ”Table 1. Factorial design parameters and experimental conditions.
> Levels used, Actual (coded)
> Factors	Low (−1)	Medium (0)	High (+1)
> X1 – Drug Concentration  in organic phase(%w/v)	0.2	0.3	0.4
> X2 – Polymer concentration  in organic phase (%w/v)	1	2	3
> X3 – Surfactant concentration	0.50%	0.75%	1.0%
> “

> ”Table 2. Full factorial design layout of lopinavir loaded NPs showing the effect of independent variables X1 (Drug concentration), X2 (Polymer concentration) and X3 (surfactant concentration) on responses PS (Particle size) and EE (entrapment efficiency).
> Sr. No.	X1	X2	X3	Y1Table Footnotea (EE, %)	Y2Table Footnotea (PS, nm)
> 1	−1	−1	−1	36.5 ± 2.21	126.6 ± 4.16
> 2	−1	−1	0	29.4 ± 1.05	131.3 ± 6.13
> 3	−1	−1	1	21.3 ± 2.28	140.8 ± 1.41
> 4	−1	0	−1	50.9 ± 1.86	127.3 ± 2.68
> 5	−1	0	0	48.7 ± 2.34	136.3 ± 4.53
> 6	−1	0	1	38.2 ± 2.43	139.7 ± 2.68
> 7	−1	1	−1	51.1 ± 1.49	159.7 ± 3.72
> 8	−1	1	0	57.3 ± 1.19	163.5 ± 3.91
> 9	−1	1	1	40.3 ± 0.71	176.9 ± 1.69
> 10	0	−1	−1	61.6 ± 2.56	144.1 ± 2.13
> 11	0	−1	0	55.4 ± 1.34	148.2 ± 1.98
> 12	0	−1	1	47.4 ± 1.04	152.6 ± 6.23
> 13	0	0	−1	83.3 ± 1.42	155.6 ± 4.61
> 14	0	0	0	73.2 ± 1.35	159.6 ± 3.73
> 15	0	0	1	63.1 ± 2.17	165.2 ± 7.26
> 16	0	1	−1	91.4 ± 2.43	162.6 ± 2.34
> 17	0	1	0	84.6 ± 2.57	169.7 ± 3.87
> 18	0	1	1	77.5 ± 1.93	173.8 ± 4.23
> 19	1	−1	−1	93.3 ± 1.27	142.1 ± 2.13
> 20	1	−1	0	91.4 ± 1.86	163.3 ± 2.43
> 21	1	−1	1	87.6 ± 1.76	175.5 ± 3.57
> 22	1	0	−1	95.7 ± 2.43	214.8 ± 3.89
> 23	1	0	0	91.4 ± 1.67	231.3 ± 2.59
> 24	1	0	1	75.3 ± 2.38	233.6 ± 1.73
> 25	1	1	−1	93.8 ± 1.26	207.4 ± 2.75
> 26	1	1	0	85.1 ± 2.76	237.1 ± 1.37
> 27	1	1	1	71.1 ± 1.43	236.1 ± 3.26
> aValues are represented as mean ± SD.“

> ”Table 7. Check point analysis, t-test analysis and NE determination for lopinavir-loaded NPs.
> PS (nm)	EE (%)
> Batch No.	X1	X2	X3	Observed	Predicted	Observed	Predicted
> Checkpoint batches with their predicted and measured values of PS and EE
>  1	−1 (0.2%)	0.0 (2%)	0.5 (62.5mg)	152.63	148.94	44.86	45.72
>  2	0 (0.4%)	0.0 (2%)	0.5 (62.5 mg)	167.96	163.23	66.86	67.10
>  3	1 (0.6%)	−1 (1%)	0.0 (50 mg)	189.24	184.4	88.43	87.03
>  tcalculated	0.0068	0.8956
>  ttabulated	2.9199	2.9199
>  NE	0.04507	0.0251“

### WIVUCMYG

段落：

> “Materials
> Pranoprofen and Oftalar® were kindly supplied by Alcon Cusi (Barcelona, Spain); PLGA Resomer® 753S was obtained from Boehringer Ingelheim (Ingelheim, Germany); PVA with 90% hydrolization was obtained from Sigma–Aldrich (St. Louis, Missouri). The purified water used in all the experiments was obtained from a MilliQ System. All the other chemicals and reagents used in the study were of analytical grade.
>
> Methods
> Preparation and Optimization of PLGA NPs
> The NPs were obtained by the solvent displacement technique described by Fessi et al.14 PLGA (80–100 mg) and PF (0–20 mg) were dissolved in 5 mL of acetone. This organic phase was poured, under moderate stirring into 10 mL of an aqueous solution of PVA (5–25 mg/mL) adjusted to the desired pH value (2.5–6.5). The acetone was then evaporated, and the NPs dispersed were concentrated to 10 mL under reduced pressure (Bϋchi B-480 Flawil, Switzerland).
>
> A factorial design is frequently used to plan research because it provides maximum information, requiring the minimum number of experiments.15 A four factor, five-level central composite rotatable design 24 + principal was used to study the main effects and interactions of four factors on average particle size (Z Ave), polydispersity index (PI), ZP, and entrapment efficiency (EE). This central composite design consisted of three groups of design points, including two-level factorial axial and center design points. The factors or independent variables studied were PF concentration (cPF), PVA concentration (cPVA), PLGA concentration (cPLGA), and aqueous phase pH. They were studied at five different levels coded as −α, −1, 0, 1, and +α. The value of alpha (2) was calculated to meet the design rotatability (Table 1).”

表格：

> “Table 1. Factors and their Corresponding Levels in Experimental Design
> Factor	−2	−1	0	+1	+2
> cPF (mg/mL)	0.00	0.50	1.00	1.50	2.00
> cPVA (mg/mL)	5.00	10.00	15.00	20.00	25.00
> cPLGA (mg/mL)	8.00	8.50	9.00	9.50	10.00
> Aqueous phase pH	2.50	3.50	4.50	5.50	6.50”

> “Table 2. Coded Values and Measured Responses of the Four Factors: cPF, cPVA, cPLGA, and Aqueous Phase pH
> Coded Levels of Factors	Measured Responses
> Factorial	cPF (mg/mL)	cPVA (mg/mL)	cPLGA (mg/mL)	pH	Mean Size (nm) ± SD	Polidispersity Index ± SD	Zeta Potential (mV) ± SD	Entrapment Efficiency (%) ± SD
> F1	−1	−1	−1	−1	566.50 ± 8.05	0.286 ± 0.02	−8.28 ± 0.71	87.69 ± 0.13
> F2	1	−1	−1	−1	597.60 ± 3.12	0.335 ± 0.01	−8.17 ± 0.32	90.26 ± 1.18
> F3	−1	1	−1	−1	605.00 ± 6.28	0.256 ± 0.01	−6.00 ± 0.48	82.57 ± 2.01
> F4	1	1	−1	−1	611.20 ± 7.21	0.368 ± 0.02	−6.80 ± 0.18	90.85 ± 1.25
> F5	−1	−1	1	−1	539.00 ± 8.11	0.227 ± 0.03	−9.27 ± 0.31	85.14 ± 1.82
> F6	1	−1	1	−1	532.10 ± 6.70	0.231 ± 0.03	−7.31 ± 0.07	89.29 ± 0.51
> F7	−1	1	1	−1	784.30 ± 5.63	0.391 ± 0.02	−5.91 ± 0.06	89.93 ± 1.98
> F8	1	1	1	−1	641.50 ± 6.26	0.363 ± 0.01	−5.99 ± 0.36	89.01 ± 0.56
> F9	1	1	1	1	348.30 ± 2.12	0.106 ± 0.03	−7.50 ± 0.16	66.50 ± 0.07
> F10	−1	1	1	1	280.10 ± 1.50	0.098 ± 0.03	−7.92 ± 0.15	49.79 ± 0.62
> F11	1	−1	1	1	324.30 ± 4.03	0.091 ± 0.01	−7.41 ± 0.56	80.02 ± 1.33
> F12	−1	−1	1	1	265.40 ± 5.75	0.073 ± 0.02	−9.61 ± 0.31	62.05 ± 2.18
> F13	1	1	−1	1	408.90 ± 4.67	0.216 ± 0.01	−6.68 ± 0.15	45.54 ± 0.23
> F14	−1	1	−1	1	318.40 ± 3.11	0.207 ± 0.02	−7.44 ± 0.19	39.79 ± 1.77
> F15	1	−1	−1	1	327.60 ± 1.91	0.095 ± 0.01	−7.22 ± 0.54	69.51 ± 0.32
> F16	−1	−1	−1	1	269.90 ± 1.25	0.096 ± 0.01	−11.20 ± 0.53	72.74 ± 1.15
> F17	2.0	0	0	0	433.90 ± 5.15	0.224 ± 0.01	−6.97 ± 0.14	77.23 ± 0.52
> F18	−2.0	0	0	0	326.70 ± 2.78	0.013 ± 0.03	−8.54 ± 0.12	00.00 ± 0.00
> F19	0	2.0	0	0	368.20 ± 2.65	0.097 ± 0.01	−6.25 ± 0.34	80.29 ± 1.25
> F20	0	−2.0	0	0	343.80 ± 2.74	0.085 ± 0.03	−8.50 ± 0.61	80.34 ± 0.32
> F21	0	0	2.0	0	421.00 ± 3.99	0.153 ± 0.02	−7.63 ± 0.15	77.15 ± 1.29
> F22	0	0	−2.0	0	340.40 ± 4.66	0.118 ± 0.01	−8.06 ± 0.30	68.48 ± 2.45
> F23	0	0	0	2.0	321.40 ± 1.47	0.224 ± 0.01	−12.92 ± 0.29	58.70 ± 0.25
> F24	0	0	0	−2.0	611.80 ± 3.29	0.301 ± 0.04	−2.15 ± 0.13	86.39 ± 1.36
> F25	0	0	0	0	354.60 ± 2.62	0.146 ± 0.02	−6.90 ± 0.15	62.01 ± 0.85
> F26	0	0	0	0	357.80 ± 3.40	0.147 ± 0.02	−6.58 ± 0.22	62.09 ± 1.30”

### YGA8VQKU

段落：

> “2.1. Materials
> The poly(lactic-co-glycolic) acid (PLGA) polymers Resomer® RG756S and Resomer® RG753S composed of lactide:glycolide 75:25 molar ratio with an inherent viscosity of 0.32–0.44 and 0.7–1.1 dL/g, respectively, were obtained from Boehringer Ingelheim (Ingelheim, Germany). Poloxamer 188 (Lutrol® F68) was a gift from BASF (Barcelona, Spain). Flurbiprofen was purchased from Sigma (St. Louis, EUA). Double distilled water was used after filtration in a Millipore® system home supplied. All other reagents used were of analytical grade.
> 2.2. Methods
> 2.2.1. Production of PLGA nanospheres
> Nanospheres composed of PLGA 75:25 of different viscosity (0.32–0.44 dL/g and 0.7–1.1 dL/g) containing FB, were produced by the solvent displacement technique described by Fessi et al. [2]. An organic solution of 90 mg of polymer in 25 mL of acetone containing FB (1.5 mg/mL) was poured, under moderate stirring, into 50 mL of an aqueous surfactant solution (10 mg/mL of Poloxamer 188) adjusted to pH 3.5, a value previously assessed by factorial design. The ratio between polymer and surfactant was assessed beforehand by factorial design to obtain the highest entrapment efficiency (94.60%, m/m) and the lowest mean particle size (232.8 ± 1.93 nm), which were shown to be suitable for ocular drug delivery [25]. The resulting colloidal suspension was kept under stirring for 5 min. Finally, the acetone was evaporated and the volume of nanospheres suspension was concentrated to 10 mL under reduced pressure in a rota-vapour (Bücchi R-114, Switzerland).”

> “2.2.2. Experimental design
> For the present work, a three factor, five-level central composite rotatable design 23 + star [26] was selected to study the effect of FB concentration in the organic phase (cFB), of poloxamer 188 concentration in the aqueous phase (cP188) and of the pH value of the aqueous phase (pH), on the mean particle size, zeta potential, and entrapment efficiency of PLGA nanospheres. An initial 23 full factorial design was created, providing the upper (+1) and lower (−1) level values for each evaluated parameter (cFB, cP188, pH) (Table 1). A total of 23 experiments were required (factorial points, Table 2). Effects and interactions between factors were calculated. To determine the effect of a particular factor x (Ex), the following equation was applied:”

> “Physicochemical properties and EE of the developed nanospheres were compared to a previously developed similar system composed of a polymer of higher viscosity [24]. Table 6 depicts the main differences between both nanospheres. Slightly differences were observed in the morfometry of both samples, resulting in higher sizes of nanospheres made with the higher viscosity polymer. However, both formulations revealed an adequate size for ocular administration without inducing corneal irritancy [31]. The measured ZP was considered appropriate to provide long-term stability. No significant differences were detected for the EE.”

表格：

> “Table 1. Initial 23 full factorial design, providing the upper (+1) and lower (−1) level values for each evaluated factor. cFB, concentration of flurbiprofen (mg/mL); cP188, concentration of poloxamer 188 (mg/mL).
>
> Evaluated factors	−1.68	−1	0	+1	1.68
> cFB (mg/mL)	0.16	0.50	1.00	1.50	1.84
> cP188 (mg/mL)	6.60	10.00	15.00	20.00	23.40
> pH of water phase	2.82	3.50	4.50	5.50	6.18
> Table 2. Coded level and measured responses of the evaluated factors: FB concentration (cFB), P188 concentration (cP188) and pH of the aqueous phase (pH).
>
> Coded levels of factors	Measured responses
> Empty Cell	cFB (mg/mL)	cP188 (mg/mL)	pH	Mean size (nm) ± SD	EE (%) ± SD	Zeta potential (mV) ± SD
> Factorial points
> F1	−1	−1	−1	240.00 ± 15.90	76.37 ± 0.46	−22.43 ± 0.40
> F2	1	−1	−1	205.30 ± 2.52	97.75 ± 0.01	−24.60 ± 0.95
> F3	−1	1	−1	236.00 ± 3.46	77.36 ± 0.05	−23.90 ± 0.30
> F4	1	1	−1	186.00 ± 3.61	87.11 ± 0.31	−23.63 ± 0.81
> F5	−1	−1	1	182.00 ± 1.00	52.68 ± 0.66	−27.40 ± 0.75
> F6	1	−1	1	203.30 ± 4.51	77.75 ± 0.07	−30.60 ± 0.75
> F7	−1	1	1	183.00 ± 1.00	59.54 ± 0.42	−27.46 ± 0.61
> F8	1	1	1	175.70 ± 1.53	74.38 ± 0.25	−27.00 ± 0.60
>
> Axial points
> F9	1.68	0	0	179.30 ± 2.31	85.18 ± 0.05	−25.80 ± 0.44
> F10	−1.68	0	0	203.30 ± 4.51	38.46 ± 0.01	−28.73 ± 0.80
> F11	0	1.68	0	166.00 ± 1.73	66.69 ± 0.47	−24.77 ± 0.42
> F12	0	−1.68	0	183.00 ± 1.00	63.76 ± 0.39	−20.33 ± 3.79
> F13	0	0	1.68	185.00 ± 1.73	65.34 ± 0.28	−23.90 ± 0.79
> F14	0	0	−1.68	201.70 ± 2.52	82.52 ± 0.02	−16.33 ± 0.55
>
> Centre points
> F15	0	0	0	182.30 ± 1.15	68.22 ± 0.27	−24.93 ± 0.06
> F16	0	0	0	179.00 ± 0.00	64.94 ± 0.23	−25.03 ± 0.57
> “”

> “Table 6. Mean particle size, zeta potential and entrapment efficiency of FB-loaded nanospheres produced with low viscosity PLGA in comparison to nanospheres produced with high viscosity PLGA [24].
>
> Empty Cell	Mean size (nm) ± SD	Zeta potential (mV) ± SD	Entrapment efficiency (%) ± SD
> Nanospheres produced with low viscosity PLGA (0.32–0.44 dL/g)	205.30 ± 2.52	−24.60 ± 0.95	97.75 ± 0.01
> Nanospheres produced with high viscosity PLGA (PLGA 0.7–1.1 dL/g)	232.80 ± 1.93	−25.79 ± 1.17	94.60 ± 0.42“

### 7ZS858NS

段落：

> “2.1. Materials
> Mometasone furoate (MF) was purchased from Acros Organics (New Jersey, USA). PLGA-Purasorb PDLG 5010 (50:50) with an inherent viscosity midpoint of 1 dL/g was kindly donated by Corbion Purac (Gorinchem, the Netherlands). Solutol HS 15 (Macrogol 15 hydroxystearate) was kindly supplied by BASF (Ludwigshafen, Germany). Sodium dodecyl sulfate (SDS) was purchased from Biological Industries (Beit HaEmek, Israel). Acetone and methanol were purchased from Sigma-Aldrich (Rehovot, Israel). Spectra/Por Biotech 1.1 dialysis membranes with a molecular weight cutoff (MWCO) of 8000 Da were purchased from Spectrum Medical Industries (Houston, Texas, USA).
> 2.2. Preparation of MF NPs
> MF-loaded nanoparticles were prepared using the nanoprecipitation method (13) with modification. The organic phase, consisting of 2 mg of MF and 6 mg of PLGA in 1 mL of acetone, was rapidly poured into 2 mL of aqueous solution containing 0.1% (w/v) Solutol HS 15. The suspension was stirred at 900 rpm for 24 h to allow complete evaporation of the organic solvent, and the formulation volume was adjusted with water to 2 mL (Figure2). Complete evaporation was confirmed by weighting the glass vial before addition of the organic phase and after the evaporation process. Then, the formulation was centrifuged for 1 min at 3000 rpm to discard debris. The supernatant was then transferred to a new tube for further investigation.”

表格：

> “Table 1. Physicochemical Properties, Encapsulation Efficiency, and Drug Loading Content of MF NPs (n = 3, mean ± s.d.)
> formulation	mean diameter (nm)	polydispersity index (PDI)	zeta potential (mV)	encapsulation efficiency (%)a	drug loading content (%)b
> MF NPs	117 ± 13	0.26 ± 0.02	–32 ± 1.2	90 ± 2.1	22.4 ± 0.5
> aEncapsulation efficiency (%) = (amount of drug in nanoparticles/amount of drug fed initially) × 100.
> bDrug loading content (%) = [amount of drug/(amount of drug + amount of polymer)] × 100.”

### 5ZXYABSU

段落：

> “Materials and methods
> Rhodamine (Rh)-123 was supplied by Sigma-Aldrich (St Louis, MO, USA) and Gat sesquihydrate (1-cyclopropyl-6-fluoro-8-methoxy-7-[3-methylpiperazin-1-yl]-4-oxo-1,4-dihydroquinoline-3-carboxylic acid sesquihydrate) by Santa Cruz Biotechnology (Dallas, TX, USA). PLGA with a ratio of 50:50 PLGA (Resomer® RG 502) was purchased from Evonik (Essen, Germany). Polysorbate 80 was obtained from PanReac (Barcelona, Spain). Labrafil® M 1944 CS (PEG-5 oleate FDA IIG), was supplied by Gattefossé (Lyon, France). Polyvinyl alcohol (PVA; molecular weight 30,000–70,000 Da) was obtained from Sigma-Aldrich. Others reagents used were of analytical grade and provided by Merck (Darmstadt, Germany). Distilled and deionized water (Milli-Q; EMD Millipore, Billerica, MA, USA) was used in the preparation of all buffers and solutions.
>
> Preparation and characterization of NPs
> Rhodamine-loaded PLGA NPs
> Rh-loaded PLGA NPs (NPR1; Table 1) were prepared by nanoprecipitation using an acetone–water system. Briefly, 50 mg of PLGA and 2.5 mg Rh were dissolved in 4 mL acetone and mixed by vortexing. This mixture was added dropwise into 12 mL of 1% PVA under continuous stirring for 15 minutes. The resulting suspension was then evaporated in a rotary evaporator (Rotavapor-R; Büchi Labortechnik AG, Flawil, Switzerland) to remove acetone completely (2 hours, 25°C, and 70 mbar). The NP suspension obtained was then washed with distilled water three times and centrifuged (Avanti J-301; Beckman Coulter, Brea, CA, USA) at 15,000 rpm for 30 minutes to remove PVA. Finally, the dispersed solution was freeze-dried for 24 hours with sucrose as cryoprotectant (Flexi-Dry MP™; FTS Systems, Stone Ridge, NY, USA).
>
> Table 1 Nanoparticle formulations developed
>
> Download CSVDisplay Table
> Rh-loaded PLGA–polysorbate 80 NPs were prepared by the nanoprecipitation method indicated previously using 12 mL of 0.5% PVA and 0.5% polysorbate 80 as the external phase (NPR2; Table 1). All NP formulations were prepared in triplicate. Rh-loaded PLGA–Labrafil NPs were prepared using the same protocol, but incorporating 3.5 mg Labrafil into the inner phase of the emulsion (NPR3). When Labrafil was incorporated, the desiccation process was performed under vacuum.
>
> Gatifloxacin-loaded PLGA NPs
> Gat-loaded PLGA NPs, Gat-loaded PLGA–polysorbate 80 NPs, and Gat-loaded PLGA–Labrafil NPs were prepared using the same procedure but incorporating 5 mg of Gat into the inner phase. Table 1 shows the different formulations prepared. In all cases, blank NPs were prepared.“

表格：

> “Table 1 Nanoparticle formulations developed
> Formulation	Rhodamine
> (mg)	Gatifloxacin
> (mg)	Polysorbate
> 80 (%)	Labrafil
> (mg)
> NPR1	2.5	–	–	–
> NPR2	2.5	–	1	–
> NPR3	2.5	–	–	3.5
> NPB1	–	–	–	–
> NPB2	–	–	1	–
> NPB3	–	–	–	3.5
> NPG1	–	5	–	–
> NPG2	–	5	1	–
> NPG3	–	5	–	3.5”

> “Table 2 Characteristics of the nanoparticle formulations prepared
> Formulation	Mean particle size ± standard deviation (nm)	ζ-Potential ± standard deviation (mV)	Encapsulation efficiency ± standard deviation (%)
> NPR1	234.8±4.3	–	55.3±0.4
> NPR2	194.9±5.7	–	51.3±0.3
> NPR3	237.8±11	–	60.9±0.4
> NPB1	150.5±5.1	−23±1.3	–
> NPB2	98.9±9.6	−19.1±1.1	–
> NPB3	156.3±6.1	−17.3±1	–
> NPG1	176.6±11.6	−18.6±0.4	34.1±0.1
> NPG2	176.5±2.9	−20.1±1.1	28.2±0.2
> NPG3	182.9±2.5	−19.3±0.8	10.4±1.1”

### 5GIF3D8W

段落：

> “Materials
> Polylactide-co-glycolide 50/50 (PLGA 50/50, molecular weight of 10,000), poly(ε -caprolactone) (PCL) (molecular weight of 40,000), and Pluronic F 68 (F68) were procured from Sigma-Aldrich Chemicals, USA. PLGA 75/25 (Purasorb PDLGA® 75/25, molecular weight of 10,000) and PLGA 85/15 (Purasorb PDLGA® 85/15, molecular weight of 10,000) were generously gifted by Purac, The Netherlands. Etoposide was gift from Dabur research foundation, Sahibabad, India. Triple distilled water was used in the preparation of nanoparticles. All other materials were of analytical grade (Spectrochem, Mumbai, India) and used as received.
>
> Preparation of Nanoparticles
> Nanoparticles were prepared by nanoprecipitation (CitationFessi et al. 1989; CitationPeltonen et al. 2002) and emulsion solvent evaporation methods (CitationLeroux et al. 1995) using PLGA polymers and PCL, respectively. These methods were modified according to present requirement. The optimized formulation was prepared using nanoprecipitation method as follows: polymer (50 mg) and etoposide (5 mg) were dissolved in acetone. Dichloromethane was used to dissolve PCL and etoposide in case of emulsion solvent evaporation method. Resulting organic phase was added slowly under moderate magnetic stirring into triple distilled water (TDW) containing F68 as stabilizer (1.0% w/v). This aqueous phase immediately turns milky with formation of nanoparticle dispersion. The organic solvent was then removed by evaporating under magnetic stirring or under reduced pressure at 35°C for approximately 1 hr (Rotavapor, Buchi, Switzerland). Entire dispersion was centrifuged at 14,000 rpm at 25°C for 10 min (Cooling Compufuge, Remi, Mumbai) in three cycles. Supernatant was analyzed for free drug content and the sediment constituting nanoparticles was freeze-dried. For freeze-drying, prefreezing of samples was done at –20°C for 20 hr, then the flasks were connected to freeze-drier (Maxi Dry Lyo, Heto, Germany) under vacuum (1 mbar, –110°C). The process was continued till free-flowing powder was obtained.
>
> Different formulation variables like polymer amount (25, 50, 100, and 200 mg), concentration of stabilizer (0.5, 0.75, 1.0, and 2.0% w/v), and etoposide amount (2.5, 5, 10, and 20 mg) were varied and the effect on size, zeta potential, and entrapment efficiency was studied. Only one parameter was changed in each series of experiments. Drug free nanoparticles were prepared by the same methods without the addition of etoposide and characterized.
> ”

> “Nanoprecipitation and emulsion solvent evaporation methods produced fine dispersions without any agglomeration. In case of formulations prepared with PLGA-copolymers, particle size found to vary depending on type of PLGA polymer and stabilizer used. It was found that change in composition of lactide and glycolide content in PLGA polymers influenced size of the nanoparticles. For optimized formulation, containing stabilizer 1.0%, PLGA 50/50 produced nanoparticles of 91.8 ± 0.74 nm, whereas at same stabilizer concentrations, PLGA 75/25 and PLGA 85/15 produced nanoparticles of size 103.7 ± 0.18 nm and 105.1 ± 0.38 nm, respectively. PCL leads to the formation of bigger particles (257.2 ± 0.96 nm) when compared with size of nanoparticles prepared by PLGA-copolymers. On the basis of preliminary studies, amount of drug, amount of polymer, and concentration of stabilizer were optimized to 5 mg, 50 mg, and 1.0% w/v. Formulation characters of the optimized formulations prepared with PLGA-copolymers and PCL are given in Table 1. Size distribution profiles of formulations prepared with PLGA 50/50 and PCL are given in Figure 1A and 1B, respectively. They show the narrow size distribution of the particles. Transmission electron microscopy photographs of nanoparticles show that all the nanoparticles are round, smooth without showing any agglomeration (Figure 2A–2D), and dispersion does not contain detectable free drug crystals. Atomic force microscopy images (Figure 3A and 3B) show that the surface of the particles is smooth and they are spherical. Percent recovery of the formulations prepared with these polymers was very good and was in the range of 85.99–96.71. Entrapment efficiency ranging from 57 to 80% was obtained for the formulations, and the percent drug content was low for all batches with a maximum of around 1.45 for PLGA 85/15 and PCL.”

> “Stabilizer Concentration
> Increase in concentration of F68 from 0.5 to 1.0% w/v made significant decrease in size of nanoparticles prepared with all the polymers in the study. Above 1.0% w/v of F68, there was no significant decrease in size of the particles. Therefore, 1.0% w/v of F68 was considered as optimum for the preparation of nanoparticles and used for optimized products. The effect of stabilizer concentration on the mean size of particles is shown in Figure 6. An optimum concentration of stabilizer leads to a reduced size of nanoparticles. Insufficient amount of stabilizer is unable to cover the dispersed nanoparticles completely and fails in stabilizing and causes aggregation leading to larger nanoparticles (CitationFeng and Huang 2001). In the solvent evaporation and nanoprecipitation methods, formation and stabilization of particles are crucial factors. The amount of surfactant plays an important role in the protection of particles because it can avoid the agglomeration of particles. There was no difference in size of the particles as the concentration of stabilizer increased to 2% w/v indicating excess stabilizer. Further increase in concentration of stabilizer increased particle size because of the viscosity of the aqueous phase due to higher stabilizer concentration at specific stirring speed (CitationMurakami et al. 1999). Polydispersity index values of the formulations prepared are less than 0.9 for all formulations prepared in the study indicating narrow and homogenous size distribution. Particles prepared with F68 as stabilizer have significant negative surface charge ranging around −33.38 to −18.91 mV for PLGA-copolymers and −33.14 to −23.62 mV for PCL. It was observed that increase in stabilizer concentration changed the zeta potential toward positive value.”

> “An increase in the polymer amount affected the morphology, particle size, polydispersity, zeta potential, drug content, and encapsulation efficiency of the nanoparticles. Initially on increase from 25 to 50 mg, there was no significant increase in mean size of the particles but polydispersity index was slightly increased. Nanoparticles prepared with 25 and 50 mg of polymer formed milky dispersions without any agglomerates. For formulations with PLGA-copolymers, increase in amount to 100 mg increased the size (167.8 and 235.4 nm for 100 and 200 mg, respectively, for PLGA 50/50) and there was formation of small agglomerates. But in formulations prepared with PCL, agglomeration started with 100 mg and when increased to 200 mg, agglomeration was very high leading to precipitation of polymer into large flakes. This is because aqueous phase containing stabilizer was not able to hold this amount of polymer and making to precipitate into agglomerates. Increase in particle size was observed with an increase in amount of polymers by other authors also (CitationOgawa et al. 1988; CitationQuintanar-Guerrero et al. 1996; CitationMurakami et al. 1999; CitationKwon et al. 2001; CitationChorny et al. 2002; CitationBudhian, Siegel, and Winey 2007). Increase in the polymer concentration led to increase in viscosity of the organic phase, and this resulted in poor dispersibility of the organic phase into the aqueous phase. Coarse dispersions were obtained at higher polymer concentrations, which lead to the formation of bigger particles during the diffusion process. This can also be due to the insufficient amount of stabilizer present in the aqueous phase for that particular polymer amount. Entrapment efficiency was influenced by the amount of polymer present in the formulation. When polymer amount was increased from 25 to 200 mg, there was increase in entrapment efficiency for batches prepared with PLGA-copolymers and PCL. For batches prepared with PCL, entrapment efficiency was increased with polymer content from 25 to 50 mg. Increasing content of polymer above 50 mg leads to precipitation and aggregate formation. Therefore, entrapment efficiency, drug content, and recovery for these preparations were not determined.
>
> Amount of Drug
> Maintaining a constant initial mass of polymers (50 mg), the mass of etoposide was varied between 2.5 and 20 mg. It was observed that the increase in the amount of etoposide from 2.5 to 10 mg increased the nanoparticle mean diameter from 82.7 to 92.4 nm for PLGA 50/50 and 221.4 to 255.7 nm for PCL. The reason might be, when less amount of etoposide was taken initially, smaller particles were produced after evaporation of organic solvent, whereas if the amount of etoposide added initially is increased, the mean particle size increases because of the high solid content after evaporation. The encapsulation efficiency was increased for all batches when etoposide amount was increased from 2.5 to 10 mg, beyond which there was no effect. This could be due to inadequate amount of polymer present in the system and was not sufficient to entrap the drug inside the matrix.”

表格：

> “TABLE 1  Formulation characters for the optimized nanoparticle formulations
> PLGA 50/50 (Mean ± SD)	PLGA 75/25 (Mean ± SD)
> Empty	Drug loaded	Empty	Drug loaded
> Diameter (nm)	87.2 ± 0.25	91.8 ± 2.74	96.9 ± 1.06	103.7 ± 2.98
> PIa	0.14 ± 0.01	0.13 ± 0.01	0.12 ± 0.01	0.14 ± 0.01
> ZPb (mV)	−18.3 ± 0.52	−21.23 ± 1.04	−17.2 ± 0.51	−28.06 ± 0.39
> Recovery (%)	90.28 ± 0.88	91.14 ± 0.28	94.02 ± 1.33	95.39 ± 0.91
> Drug content (%)	—	1.04 ± 0.06	—	1.14 ± 0.02
> EEc (%)	—	57.64 ± 0.97	—	66.11 ± 0.72
> TABLE 1  Formulation characters for the optimized nanoparticle formulations
> PLGA 85/15 (Mean ± SD)	PCL (Mean ± SD)
> Empty	Drug loaded	Empty	Drug loaded
> Diameter (nm)	106.1 ± 2.07	105.1 ± 2.38	254.1 ± 1.00	257.2 ± 3.96
> PIa	0.09 ± 0.03	0.11 ± 0.01	0.13 ± 0.01	0.10 ± 0.01
> ZPb (mV)	−22.3 ± 1.18	−29.41 ± 0.58	−28.1 ± 0.81	−27.70 ± 0.19
> Recovery (%)	91.28 ± 1.11	89.28 ± 0.58	88.39 ± 1.42	91.34 ± 0.87
> Drug content (%)	—	1.45 ± 0.11	—	1.44 ± 0.09
> EEc (%)	—	78.99 ± 1.04	—	80.15 ± 1.01
> aPolydispersity index.
>
> bZeta potential.
>
> cEncapsulation efficiency; each value is mean of three independent determinations.”



### 2026-04-27 — Stage5 shared loaded-drug identity carrythrough

Diagnostic repair pattern: `PAT_STAGE5_SHARED_DRUG_IDENTITY_CARRYTHROUGH_V1`.

Problem class: Stage5 final-output materialization did not carry unique article-global loaded drug identity into DOE/table rows when row-local Stage2/Stage3 surfaces retained coded variable/result rows but blank `drug_name` bundles. This was visible in the G1 explicit-value view as high `drug_name` missingness for WIVUCMYG, WFDTQ4VX, and YGA8VQKU.

First missing boundary: Stage5 final-table materialization. Stage2/Stage3 had already authorized formulation rows; the missing field was a source-backed shared value not materialized into blank row bundles.

Repair rule: extract a unique loaded-drug identity from source-backed title/abstract/methods lead surfaces; resolve abbreviations such as `PF` or `FB` when source text binds them to a full drug name; fill only blank `drug_name` bundles in non-empty/non-blank rows; never overwrite row-local values; skip coded NP-family labels such as `NPG1`/`NPR1`/`NPB1`; ignore helper dye candidates such as coumarin/FITC/fluorescein.

Validation artifacts:

- Stage5 diagnostic replay: `data/results/20260423_9c4a03f/65_stage5_shared_drug_identity_diagnostic/`
- Baseline diagnostic compare: `data/results/20260423_9c4a03f/66_layer3_compare_shared_drug_identity_baseline/`
- After diagnostic compare: `data/results/20260423_9c4a03f/67_layer3_compare_shared_drug_identity_after/`
- Run context: `data/results/20260423_9c4a03f/65_stage5_shared_drug_identity_diagnostic/RUN_CONTEXT.md`

Diagnostic effect, not benchmark-valid final evidence:

- `drug_name` missing_in_system: 86 -> 25
- `drug_name` present_and_match: 56 -> 117
- `drug_name` present_but_mismatch: 11 -> 11
- WIVUCMYG: missing 26 -> 0; match 0 -> 26
- WFDTQ4VX: missing 18 -> 0; match 0 -> 18
- YGA8VQKU: missing 17 -> 0; match 0 -> 17
- G1 recall: 60.91% -> 70.15%
- G1 conditional accuracy: 85.57% -> 87.47%
- G1 correct-value recall: 52.12% -> 61.36%
- total diagnostic compare error rows: 2371 -> 2310

Unit verification: `python3 -m unittest tests.test_compare_layer3_values_v1` passed `Ran 96 tests ... OK`.

### 2026-04-27 — Stage5 DOE factor emulsifier-name carrythrough diagnostic

Repair pattern: `PAT_STAGE5_DOE_FACTOR_EMULSIFIER_NAME_CARRYTHROUGH_V1`

Diagnostic-only lineage:

- Source run: `data/results/20260423_9c4a03f`
- Stage5 diagnostic output: `data/results/20260423_9c4a03f/68_stage5_emulsifier_factor_name_diagnostic/`
- Layer3 diagnostic compare: `data/results/20260423_9c4a03f/69_layer3_compare_emulsifier_factor_name_after/`
- Locked GT authority: `data/cleaned/gt_authority/v1/dev15_layer3_values.tsv`

Failure class:

- After the shared loaded-drug repair, the highest G2 residual field class was `emulsifier_stabilizer_name`.
- The first failure boundary was Stage5 final-output materialization, not GT or compare scoring: DOE/coded table rows preserved numeric concentration values such as `cPVA=...` or `cP188=...`, but `surfactant_name_value_text` remained blank even when the source text explicitly defined the coded factor.

Implemented generic rule:

- Parse row-local coded factor labels from `change_descriptions`, `identity_variables_json`, and `identity_variables`.
- Parse source-backed factor definitions such as:
  - `cPVA`, `PVA concentration`
  - `cP188`, `concentration of poloxamer 188`
- Fill the blank `surfactant_name` bundle only when exactly one row factor maps to exactly one supported emulsifier/stabilizer material name.
- Do not overwrite row-local `surfactant_name`.
- Do not infer or decode concentration values; existing row-local concentration values are preserved unchanged.
- Ambiguous or unmapped factors remain blank.

Validation:

- Unit test command: `python3 -m unittest tests.test_compare_layer3_values_v1`
- Unit test result: `Ran 98 tests ... OK`
- Diagnostic compare impact versus `67_layer3_compare_shared_drug_identity_after`:
  - `emulsifier_stabilizer_name missing_in_system`: `73 -> 31`
  - `emulsifier_stabilizer_name present_and_match`: `54 -> 96`
  - `emulsifier_stabilizer_name present_but_mismatch`: `26 -> 26`
  - new mismatches introduced: `0`
  - `WIVUCMYG`: `missing_in_system 26 -> 0`, `present_and_match 0 -> 26`
  - `YGA8VQKU`: `missing_in_system 16 -> 0`, `present_and_match 0 -> 16`
  - G2 recall: `62.26% -> 65.17%`
  - G2 conditional accuracy: `89.88% -> 90.33%`
  - G2 correct-value recall: `55.96% -> 58.86%`
  - compare CLI error rows: `2310 -> 2268`

Residual note:

- `WFDTQ4VX` still has unresolved surfactant-name misses for `X3` coded surfactant concentration rows because the cleaned source text available to the rule does not expose an explicit `X3 -> Pluronic F68` definition. This should not be filled by a paper-local override or by guessing from a material list; it requires stronger source evidence or a separate generic table-definition recovery path.

