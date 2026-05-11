# S5-3 Opportunity-Driven Value Backfill V1 Design Plan

> **For Hermes:** This is a design contract and implementation plan for the first version of S5-3 value backfill. Do not treat this file as evidence that runtime behavior already exists. Future implementation should follow repository governance and use maintained entrypoints only after scripts are implemented, tested, and registered.

**Goal:** Define the first-version S5-3 logic for deciding which already-empty row-field positions in already-admitted formulation rows should be submitted to an LLM for direct-value adjudication, how evidence should be assembled, how prompts should be grouped, what the LLM must output, and how audit evidence must be preserved.

**Architecture:** S5-3 is a post-S5-2 residual value-opportunity layer. S5-2 and S5-3 both operate only inside formulation boundaries that have already been assigned upstream. They are value-fill layers only: they must not create formulations, remove formulations, split/merge formulation boundaries, review row membership, or overwrite `final_formulation_table_v1.tsv` by default. Task assignment is source/evidence/opportunity-driven, not blank-schema-slot-driven, and is limited to empty/unresolved target fields.

**Status:** design-only, diagnostic-development. No live LLM call is authorized by this plan.

---

## 1. Governing principles

S5-3 must support large-scale literature processing where no GT table exists to say which values should or should not be present. Because PLGA literature is heterogeneous, empty fields are normal and can be correct. Therefore:

- Empty final-table cells are not S5-3 tasks by themselves.
- Numeric mentions in source text are not values by themselves.
- S5-3 task assignment must be driven by source/evidence opportunity signals plus row/field applicability.
- S5-3 may provide a controlled LLM fallback for S5-2 mechanical materialization misses, but the fallback must remain trace-labeled and downstream-risk-gated.
- LLM output is a candidate, not authority.
- Direct quote/table-cell locator, source locator, row-scope rationale, field boundary, and direct/derived classification are mandatory.
- `not_reported_in_evidence`, `wrong_context`, `scope_unclear`, and `conflict_requires_review` are valid outputs.
- S5-3 must not review or replace already-filled values. If evidence suggests a conflict with an already-filled value, S5-3 may emit only a conflict hint sidecar; overwrite/review belongs to a separate downstream risk/review workflow.
- S5-4/S5-risk controls authority and risk; S5-6 initially links sidecars only.
- S5-3 runs only after S5-2 exists for the lineage; no S5-2 means no S5-3.

S5-3 should be understood as:

```text
source evidence indicates a reportable value opportunity
+ deterministic S5-2 did not already resolve the empty target field
+ field is applicable to the already-admitted row
+ absence is not normal/negative
=> ask LLM for a direct quoted/table-cell candidate or explicit no-value/conflict status
```

---

## 2. Runtime boundary

S5-3 is downstream of:

- completed Stage2 semantic authority surface;
- Stage3 relation artifacts;
- Stage5 fixed row universe from S5-1/S5-2, with S5-2 already completed for the lineage;
- Stage2 evidence surfaces from S2-2, especially `evidence_blocks_v1.json`, `normalized_table_payloads_v1.json`, and any execution-grade table-cell/grid authority available for the run.

S5-3 must not:

- rediscover formulation rows;
- create row membership;
- remove row membership;
- infer dense database completion;
- treat GT as value authority;
- treat system final-table values as source authority;
- overwrite Stage5 final table by default;
- review or replace non-empty target fields;
- use `latest`, mtime, parent fallback, or glob selection for input lineage.

All inputs must be explicit paths or resolved through a governed active data-source contract before execution.

### 2.1 Relationship to Stage5 Evidence Binding

Evidence Binding Pack work and S5-3 are both Stage5-side audit/value-layer systems. They should not become two unrelated evidence stacks. The intended relationship is:

```text
Evidence Binding = post-Stage5 frozen row/value audit and risk explanation.
S5-3 = pre-LLM direct-value opportunity adjudication and candidate generation.
```

They have different timing and outputs, but they should share Stage5 infrastructure wherever the semantics match.

Reusable from Evidence Binding into S5-3:

- authority resolution discipline from `resolve_evidence_binding_authority_v1.py`;
- exact artifact manifest style and `RUN_CONTEXT.md` lineage recording;
- alias-conflict failure behavior before evidence interpretation;
- assignment-path vocabulary where it describes value lineage, such as `stage3_relation_resolved`, `shared_method_context`, `direct_same_table_row`, `parent_inheritance`, and `doe_factor_decode`;
- risk/status vocabulary ideas such as `missing_evidence_anchor`, `relation_path_missing`, `unresolved_table`, `source_surface_missing`, `conflict`, and `normalization_pending`;
- validator/golden-case style for protecting evidence contracts;
- workbook/review integration pattern where sidecars are explicit inputs and legacy mode must be explicit.

Not reusable as-is:

- current Evidence Binding packs explain already-frozen final-table values; they do not decide which empty/unresolved row-fields deserve LLM adjudication;
- current Evidence Binding packs are not yet a complete S5-3 prompt evidence packet, because S5-3 must include source excerpts, candidate source signals, prior Stage2/LLM semantic signals, Stage3 relation context, S5-2 value/failure traces, and negative-context evidence before prompt rendering;
- Evidence Binding risk assessment consumes frozen packs and must not be used to re-resolve S5-3 evidence.

Design rule: S5-3d evidence packet assembly may reuse Evidence Binding authority/lineage/taxonomy helpers, but it must emit its own pre-LLM packet manifest/items and must not treat post-Stage5 Evidence Binding risk outputs as source authority. Evidence Binding and prior LLM/Stage3 artifacts may be lineage/semantic evidence when labeled by source type, but original paper source/table evidence remains required for accepted direct values.

---

## 3. First-version S5-3 substructure

The first version should be split into auditable, separately runnable surfaces. The purpose of the split is that a reviewer can inspect why a row-field was selected before seeing any prompt, then inspect exactly which evidence was assembled before any live LLM call, then inspect candidate parsing and authority validation independently.

```text
S5-3a field responsibility map
S5-3b source/value signal detection
S5-3c opportunity assignment / dispatch decision
S5-3d evidence packet assembly
S5-3e prompt packet construction
S5-3f live/direct-value candidate extraction
S5-4 direct-value authority validation
S5-6 sidecar manifest/link integration
```

The two most important trust boundaries are:

```text
S5-3c decides WHAT to ask the LLM.
S5-3d decides WHAT EVIDENCE the LLM is allowed to see.
```

These must be separate outputs. A task should not be submitted merely because evidence exists, and evidence should not be silently bundled merely because a task exists. Both steps need their own audit rows, input hashes, source refs, rejection reasons, and summary counts.

S5-5 derived values remain separate and are not part of direct S5-3 adjudication. S5-3/S5-4 must not perform `%w/v × mL -> mg`, `mg/mL × mL -> mg`, ratio × known mass, concentration × volume, unit-conversion-as-derivation, or formula-based inferred values. Those belong only in S5-5 with separate provenance.

---

## 4. S5-3a field responsibility map

Create or generate a field responsibility map before task assignment.

Proposed artifact:

```text
s5_3_field_responsibility_map_v1.tsv
```

Required columns:

```text
field_name
field_family
default_owner
s5_3_allowed
s5_3_allowed_modes
mechanical_priority
requires_direct_quote
allows_shared_scope
allows_derived_value
negative_context_risks
notes
```

Example semantics:

```text
particle_size_nm
field_family = particle_characterization
default_owner = s5_2_table_cell
s5_3_allowed = yes
s5_3_allowed_modes = mechanical_fallback,source_numeric_signal
mechanical_priority = high
requires_direct_quote = yes
allows_shared_scope = conditional
allows_derived_value = no
negative_context_risks = scale_bar,filter_pore_size,membrane_pore_size,thickness,assay_condition
```

The map must distinguish:

- S5-2 mechanical/table-cell owner fields;
- shared/prose/global fields where semantic scope binding is expected;
- fields that should never be handled by direct S5-3;
- fields requiring review rather than LLM backfill.

First-version field scope should be an explicit EE-modeling-focused whitelist rather than an automatic all-schema scan. Initial allowed field list:

```text
phase_ratio_raw
pH_raw
la_ga_ratio_normalized
la_ga_ratio_raw
polymer_concentration_value
polymer_concentration_unit
solvent_name
drug_name
emulsifier_stabilizer_name
particle_size_nm
ee_percent
emulsifier_stabilizer_concentration_unit
emulsifier_stabilizer_concentration_value
drug_mass_mg
polymer_mass_mg
```

Notes:

- The whitelist is motivated by EE-modeling readiness and known recall gaps, not by GT-driven task generation.
- `emulsifier_stabilizer_name` has known mismatch/accuracy concerns, but S5-3 first version may only fill empty values; mismatch cleanup belongs to risk/review, not S5-3.
- High-existing-recall fields such as `phase_ratio_raw`, `pH_raw`, and `la_ga_ratio_normalized` may remain in scope only for empty cells with evidence opportunity signals; their good coverage does not imply dense completion.

---

## 5. S5-3b source signal detection

S5-3b should perform high-recall, targeted signal detection over governed Stage1/Stage2/Stage5 evidence surfaces. Because Stage2 has already performed whole-paper semantic understanding, S5-3b should not blindly include every evidence block or full clean text in prompts. Instead it should use rule/search-based retrieval, table/header/unit patterns, Stage3 row/group clues, and S5-2 traces to select relevant source neighborhoods. Clean text is lawful evidence and may be searched/used directly, but selected snippets must be recorded with explicit source refs and hashes.

Proposed artifact:

```text
s5_3_source_value_signals_v1.tsv
```

Required columns:

```text
signal_id
paper_key
source_ref
source_surface_type
section_label
table_id
row_hint
field_family
raw_expression
normalized_expression_type
unit_hint
context_window
signal_level
negative_context_flags
candidate_row_refs
reason
```

Allowed `source_surface_type` values:

```text
stage2_evidence_block
normalized_table_payload
table_cell_grid
table_caption
table_note
clean_text_section
s2_2_candidate_context
```

Suggested `signal_level` values:

```text
0 no signal
1 paper-level weak signal
2 section/table-level signal
3 row-local or shared-scope strong signal
```

Numeric examples are only field-family signals:

```text
100 nm  -> possible particle_size_nm, with risks scale_bar/filter/pore/thickness
93.4 %  -> possible ee_percent/loading/concentration/release, context required
-26 mV  -> possible zeta_potential_mv
50:50   -> possible la_ga_ratio
10 mg   -> possible drug_mass_mg/polymer_mass_mg/dose, context required
```

First-version detection should favor recall over precision (`strategy=A`): collect plausible numeric/name/unit signals broadly, because lightweight models and small repeated prompts can adjudicate them later. High-confidence negative contexts may suppress tasks; lower-confidence negative contexts should be retained as warning flags for S5-3d/S5-3e.

---

## 6. S5-3c opportunity assignment

S5-3c combines final-row state, field responsibility, completed S5-2 output/trace, and source signals into a task table. If S5-2 has not run for the lineage, S5-3 must not run.

Proposed artifact:

```text
s5_3_value_opportunity_tasks_v1.tsv
```

Required columns:

```text
task_id
paper_key
final_row_id
source_formulation_id
raw_formulation_label
field_name
field_family
current_value_state
field_applicability
default_owner
trigger_type
source_signal_level
source_signal_refs
s5_2_trace_status
mechanical_failure_reason
negative_context_flags
dispatch_decision
task_priority
prompt_group_id
reason
```

Decision rule:

```text
LLM task =
  admitted Stage5 row
  AND current target field is empty/unresolved
  AND field applicable to this row
  AND source signal present
  AND completed S5-2 did not already resolve it
  AND NOT negative absence/wrong-context dominated
```

`dispatch_decision` controlled vocabulary:

```text
no_task__already_filled
no_task__field_not_applicable
no_task__no_source_signal
no_task__negative_absence
no_task__deterministic_repair_preferred
task__s5_2_mechanical_fallback
task__source_numeric_signal_adjudication
task__shared_prose_value_adjudication
task__table_note_scope_adjudication
conflict_hint__already_filled_or_scope_conflict
```

Main trigger types:

### 6.1 S5-2 mechanical fallback

Used when the field is normally S5-2-owned, S5-2 has already run for the lineage, S5-2 did not fill the empty target field, and table/grid/header/cell/source trace suggests that a direct value may exist.

Examples:

```text
Table has Size (nm) column and a row-local numeric cell, but final particle_size_nm is empty.
Table footnote/caption defines units or shared condition, but deterministic materializer did not bind it.
Header alias/row binding/table geometry was too messy for current S5-2 rules.
```

This is not permission for LLM to overwrite S5-2 or any non-empty value. It is a trace-labeled empty-field fallback candidate flow.

### 6.2 S2-2/source numeric signal adjudication

Used when S2-2 evidence contains field-family numeric signals that are absent from the final row-field and require context adjudication.

Example:

```text
Source contains "100 nm" near formulation evidence; final table lacks particle_size_nm.
```

This is only an opportunity signal. The LLM must decide whether the number directly reports the target field for the target row, or whether it is wrong context such as scale bar, filter pore size, membrane pore size, thickness, equipment condition, assay condition, or unrelated result.

### 6.3 Shared prose/table-note value adjudication

Used when values appear in preparation prose, global/shared statements, table notes, captions, or footnotes and may apply to multiple rows only if the scope is explicitly supported.

---

## 7. S5-3d evidence packet assembly

S5-3d is a separate evidence-pack step between task assignment and prompt construction. It builds the complete, reviewable evidence packet for each `prompt_group_id` or task cluster before any prompt text is rendered or live LLM call is made.

Because S5-3d and Evidence Binding are both Stage5 evidence-sidecar systems, S5-3d should reuse Evidence Binding's authority-resolution and lineage discipline rather than inventing another path resolver. The S5-3d packet is nevertheless a distinct pre-LLM object: it is assembled to support an LLM adjudication task, while Evidence Binding packs explain already-frozen Stage5 row/value assignments after the fact.

Proposed artifacts:

```text
s5_3_evidence_packet_manifest_v1.tsv
s5_3_evidence_packet_items_v1.tsv
evidence_packets/<evidence_packet_id>.json
evidence_packets/<evidence_packet_id>.txt
```

`S5-3c` decides which row-field tasks are eligible. `S5-3d` decides which evidence is relevant and lawful to show the LLM. These decisions must be independently inspectable. In particular, a row-field task may be created but held from prompting if evidence assembly cannot produce enough lawful context.

Required packet-level columns for `s5_3_evidence_packet_manifest_v1.tsv`:

```text
evidence_packet_id
paper_key
prompt_group_id
row_ids
task_ids
field_families
trigger_types
authority_manifest_path
evidence_binding_reused_components
source_surface_count
stage2_signal_count
stage3_relation_ref_count
s5_2_trace_ref_count
table_authority_ref_count
evidence_char_count
input_hash
packet_status
hold_reason
```

Required item-level columns for `s5_3_evidence_packet_items_v1.tsv`:

```text
evidence_item_id
evidence_packet_id
paper_key
item_role
source_layer
source_ref
source_locator
source_hash
evidence_binding_assignment_path
evidence_binding_status_hint
related_task_ids
related_row_ids
related_field_names
include_decision
include_reason
risk_flags
text_or_payload_excerpt
```

Allowed `item_role` values:

```text
source_anchor_quote
stage2_evidence_block
stage2_llm_signal_or_candidate
stage2_raw_response_excerpt
normalized_table_payload_excerpt
table_cell_grid_excerpt
table_caption_or_note
stage3_relation_graph_excerpt
stage3_resolved_field_trace
s5_2_value_trace
s5_2_failure_trace
evidence_binding_authority_manifest
evidence_binding_assignment_hint
evidence_binding_status_hint
row_identity_context
negative_context_warning
```

Evidence packet contents should include, when relevant and available:

1. **Original source text/table evidence**
   - Mainline S5-3 evidence packets must use the corresponding paper's governed Stage1/Stage2 source surfaces: clean text, Stage2 evidence blocks, normalized table payloads, table-cell grids, captions, notes, row/column neighborhoods, and source locators.
   - They must not depend on manually uploaded source excerpts, GT-derived snippets, or DEV15-only source anchors, because scaled extraction over hundreds of papers has no GT and no manually uploaded original excerpts.
   - Manual source-anchor sections such as `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864` are diagnostic/debug authorities for DEV15 field-GT audits only, not mainline S5-3 extraction inputs.

2. **Earlier LLM semantic signals**
   - Stage2 LLM candidate/formulation signals, semantic object snippets, raw-response locators, or parsed candidate rows that explain why this row/field is plausible.
   - These are valid labeled evidence items for lineage, row identity, and semantic attention because Stage3 relation context is itself downstream of earlier LLM signals. They must be labeled as LLM/semantic evidence, not original paper source. Accepted direct values still require source quote/table-cell evidence unless a later governed rule explicitly allows another authority class.

3. **Stage3 relation context**
   - Relation graph excerpts, parent/child links, shared formulation groups, inherited relation fields, and resolved-field provenance needed to judge whether a value is row-local, shared-group, or not applicable.
   - This context constrains scope; it must not create new source values.

4. **S5-2 trace context**
   - Existing filled values for non-target fields needed for row identity.
   - S5-2 trace or failure reason for target fields, especially for mechanical fallback.
   - Candidate cell/header/row binding info when S5-2 likely missed a table-cell value.

5. **Evidence Binding lineage/taxonomy context**
   - Reuse Stage5 Evidence Binding authority manifests and, where available, assignment-path/status hints for the same row/field family.
   - These hints may guide evidence selection and audit labeling, but they are not source authority and cannot replace direct source quote/table evidence for S5-4 acceptance.

6. **Negative-context evidence**
   - Nearby text/table rows indicating scale bars, filter pore size, membrane pore size, assay conditions, release medium, non-PLGA comparators, blank controls, equipment parameters, or other wrong-context risks.

Packet status controlled vocabulary:

```text
packet_ready_for_prompt
hold__no_direct_source_evidence
hold__stage3_scope_conflict
hold__evidence_too_large_split_required
hold__source_locator_missing
hold__review_required_before_prompt
```

The evidence packet is the main audit object for trust. Evidence items should be reusable: the same source snippet/table row/Stage3 relation may support multiple tasks or prompt groups, so item identity and hashes should be stable and referenced rather than duplicated when possible. A reviewer should be able to answer these questions from S5-3d outputs without seeing the LLM response:

```text
Why is this row-field being asked?
Which source quote/table/cell is the possible value signal?
What earlier LLM/Stage2 signal caused attention to this field?
What Stage3 relation or inheritance context controls scope?
What S5-2 trace explains why deterministic materialization did not already fill it?
What wrong-context risks were included for the LLM to assess?
```

---

## 8. Prompt grouping policy

S5-3e prompts should be grouped by:

```text
one paper
+ compatible evidence packet
+ compatible field family
+ small row chunk
```

Default prompt shape:

```text
1 paper
1-3 admitted rows
6-12 opportunity fields
<= 24 row-field tasks
<= 20k-30k evidence characters
```

Rescue prompt shape for complex cases:

```text
1 row
4-8 fields
one field family or one evidence packet
```

Field-family prompt shape for shared evidence:

```text
multiple rows
1 field family
only fields covered by the same table/prose evidence packet
```

Hard prohibitions:

- Do not submit one paper with all rows and all schema fields.
- Do not submit blank fields that lack source evidence signals.
- Do not submit already-filled target fields; at most emit a separate conflict hint sidecar outside the LLM task list.
- Do not mix unrelated field families unless the evidence packet itself is shared and compact.
- Do not rely on model memory or external knowledge.

Recommended field families:

```text
particle_characterization: particle_size_nm
encapsulation_loading: ee_percent, drug_mass_mg
composition_material: drug_name, emulsifier_stabilizer_name, la_ga_ratio_raw, la_ga_ratio_normalized
composition_amount: polymer_mass_mg, polymer_concentration_value, polymer_concentration_unit, emulsifier_stabilizer_concentration_value, emulsifier_stabilizer_concentration_unit
process_condition: solvent_name, phase_ratio_raw, pH_raw
```

---

## 9. Prompt packet structure

Proposed prompt packet artifacts:

```text
prompts/s5_3_prompt_audit_v1.tsv
prompts/prompt_packets/<prompt_group_id>.json
prompts/prompt_packets/<prompt_group_id>.txt
```

Each prompt should contain these sections:

### 8.1 Task header

State that the task is S5-3 direct-value opportunity adjudication, not dense extraction or schema completion.

Required instructions:

```text
Do not infer.
Do not complete blank schema fields.
Do not use external knowledge.
A numeric mention is only a candidate signal, not a value.
Return not_reported_in_evidence when evidence does not directly support a value.
```

### 8.2 Row identity context

Include only admitted Stage5 rows in the prompt chunk.

Required row context:

```text
row_id
paper_key
source_formulation_id
raw_formulation_label
normalized_formulation_label, if available
formulation_role
parent/group context, if available
known non-target row context needed for scope binding
```

Target fields must be empty/unresolved. There is no target-field state other than empty/unresolved in S5-3 prompts. Do not include GT or expected answers; scaled extraction has no GT. Do not include a target field's existing system value because non-empty fields are not S5-3 tasks.

### 8.3 Target row-field tasks

List only opportunity tasks from `s5_3_value_opportunity_tasks_v1.tsv`.

For each task:

```text
task_id
row_id
field_name
field_family
trigger_type
why_targeted
allowed_value_type
negative_context_risks
```

### 8.4 Candidate source signals

Include S5-3a signals relevant to the prompt.

For each signal:

```text
signal_id
raw_expression
field_family_guess
source_ref
context_type
signal_strength
negative_flags
```

### 8.5 Source evidence packet

Use governed Stage1/Stage2/Stage5 evidence surfaces only, with targeted retrieval rather than whole-paper dumping.

Text evidence should be delimited:

```text
BEGIN_EVIDENCE_BLOCK id=<id> source=<path-or-ref> section=<section>
...
END_EVIDENCE_BLOCK
```

Table evidence should be structured:

```text
TABLE_ID
CAPTION
HEADER_PATH
RELEVANT_ROWS
NOTES
TABLE_CELL_GRID_REF, when available
```

Full numeric table rows should be included only when they are relevant to the targeted rows/field family. Prompt-facing summaries are not execution-grade table authority; locators must point back to normalized/table grid surfaces when available.

### 8.6 Decision rules

Required acceptance rules:

```text
A direct value is acceptable only if:
1. It is directly stated in provided evidence.
2. The source quote supports the value and unit.
3. The row or shared-group scope includes the target row.
4. The value belongs to the target field, not a wrong-context numeric mention.
5. The value is direct. Derived reasoning, formula conversion, or unit-conversion-as-derivation belongs to S5-5 only.
```

Required uncertainty behavior:

```text
If uncertain, return scope_unclear or conflict_requires_review; this flags downstream risk/review but is not itself S5-3 review.
If the value is present but belongs to another context, return value_present_wrong_context.
If the evidence does not report the value for this row, return not_reported_in_evidence.
```

### 8.7 Output schema

Use JSON array. Return exactly one object per row-field `task_id`.

Required object schema:

```json
{
  "task_id": "",
  "row_id": "",
  "field_name": "",
  "candidate_status": "direct_value_found | not_reported_in_evidence | not_applicable_to_row | value_present_wrong_context | scope_unclear | conflict_requires_review",
  "candidate_value_raw": "",
  "candidate_value_normalized": "",
  "candidate_unit": "",
  "direct_or_derived": "direct | none",
  "source_quote": "",
  "source_ref": "",
  "source_locator": "",
  "scope_type": "row_local | shared_group | paper_global | unclear | none",
  "scope_rationale": "",
  "negative_context_assessed": [],
  "confidence": "high | medium | low"
}
```

`candidate_value_normalized` is limited to direct textual/canonical normalization of the reported value. It must not contain S5-5 derivations or formula/unit-conversion outputs.

### 8.8 Coverage checklist

Prompt must require:

```text
Return exactly one object for each task_id.
Do not add unlisted fields.
Do not omit no-value tasks.
Do not return values without source_quote.
```

---

## 10. S5-3f candidate output

Proposed artifact:

```text
s5_3_llm_value_candidates_v1.tsv
```

Required columns:

```text
candidate_id
task_id
paper_key
final_row_id
field_name
candidate_status
candidate_value_raw
candidate_value_normalized
candidate_unit
direct_or_derived
source_quote
source_ref
source_locator
scope_type
scope_rationale
row_binding_rationale
confidence
negative_context_assessed
model_name
provider
prompt_id
raw_response_path
```

Also write:

```text
s5_3_llm_response_parse_errors_v1.tsv
raw_responses/<prompt_id>.json
```

---

## 11. Evidence sidecar

Proposed artifact:

```text
s5_3_value_evidence_sidecar_v1.tsv
```

Required columns:

```text
candidate_id
task_id
paper_key
field_name
source_quote
quote_start_or_locator
quote_end_or_locator
source_surface_type
source_file
source_block_id
table_id
table_row_index
table_column_header
evidence_hash
evidence_context_before
evidence_context_after
```

For table values, evidence must trace to table payload/grid row/column/header/caption/note when available. For prose values, evidence must trace to evidence block, section, source locator, and quote context.

---

## 12. S5-4 authority validation

S5-4 validates S5-3 candidates for direct source support. It does not use GT or current system final-table values as source authority, does not review already-filled values, and does not perform high-risk adjudication beyond emitting flags/decisions for the downstream risk assessment system.

Proposed artifact:

```text
s5_4_value_authority_decisions_v1.tsv
```

Required columns:

```text
candidate_id
task_id
decision
decision_rule
field_allowed
has_direct_quote
quote_supports_value
quote_supports_unit
row_scope_supported
direct_not_derived
negative_context_cleared
conflict_status
accepted_value_raw
accepted_value_normalized
accepted_unit
risk_flag_reason
```

Controlled decisions:

```text
accept_direct_value
reject_no_quote
reject_wrong_context
reject_field_not_allowed
reject_derived_in_direct_layer
reject_scope_unsupported
flag_conflict_for_risk_assessment
flag_ambiguous_for_risk_assessment
```

Accepted direct candidates may later be linked only through an explicit governed integration mode. S5-4 validation is not a high-risk review workflow and does not review already-filled values; risk assessment controls high-risk accepted/rejected/candidate outputs in later Stage5 sidecars. Any merge or overlay table must be an explicitly named diagnostic overlay, not a mutation of `final_formulation_table_v1.tsv`.

---

## 13. S5-6 sidecar integration

First-version integration is manifest/link only.

Proposed artifact:

```text
stage5_value_layer_sidecar_manifest_v1.tsv
stage5_value_layer_sidecar_manifest_v1.json
```

Required manifest fields:

```text
s5_3_tasks_path
s5_3_signals_path
s5_3_evidence_packet_manifest_path
s5_3_evidence_packet_items_path
s5_3_candidates_path
s5_3_evidence_sidecar_path
s5_4_decisions_path
s5_4_accepted_values_path
row_membership_changed=no
final_table_modified=no
benchmark_valid=no
```

Do not merge S5-4 accepted values into `final_formulation_table_v1.tsv` by default. A future overlay/merge mode must be explicit, diagnostic-labeled, and handled outside S5-3 task generation.

---

## 14. Run/audit summary

Every S5-3 diagnostic execution must write:

```text
RUN_CONTEXT.md
s5_3_task_assignment_summary_v1.json
s5_3_task_assignment_summary_v1.tsv
```

Summary metrics:

```text
total_empty_cells_seen
total_source_signals
total_opportunities
total_evidence_packets
evidence_packet_count_by_status
evidence_packet_item_count_by_role
task_count_by_trigger_type
no_task_count_by_reason
mechanical_fallback_count
source_numeric_signal_count
shared_prose_count
conflict_hint_count
risk_flag_count
paper_count
field_count
prompt_count
model_name
provider
live_llm_calls
benchmark_valid=no
```

`RUN_CONTEXT.md` must record exact input paths, source authority, model/provider, prompt count, raw response directory, candidate output paths, authority-validation paths, and diagnostic-only status.

---

## 15. Proposed run directory layout

For a future diagnostic child execution root:

```text
<run_dir>/<NN_s5_3_value_backfill>/
  RUN_CONTEXT.md

  s5_3_field_responsibility_map_v1.tsv
  s5_3_source_value_signals_v1.tsv
  s5_3_value_opportunity_tasks_v1.tsv
  s5_3_task_assignment_summary_v1.json
  s5_3_task_assignment_summary_v1.tsv

  evidence/
    s5_3_evidence_packet_manifest_v1.tsv
    s5_3_evidence_packet_items_v1.tsv
    evidence_packets/
      <evidence_packet_id>.txt
      <evidence_packet_id>.json

  prompts/
    s5_3_prompt_audit_v1.tsv
    prompt_packets/
      <prompt_group_id>.txt
      <prompt_group_id>.json

  raw_responses/
    <prompt_group_id>.json

  candidates/
    s5_3_llm_value_candidates_v1.tsv
    s5_3_value_evidence_sidecar_v1.tsv
    s5_3_llm_response_parse_errors_v1.tsv

  authority/
    s5_4_value_authority_decisions_v1.tsv
    s5_4_accepted_direct_values_v1.tsv
    s5_4_rejected_value_candidates_v1.tsv
    s5_4_value_risk_flags_v1.tsv

  integration/
    stage5_value_layer_sidecar_manifest_v1.tsv
    stage5_value_layer_sidecar_manifest_v1.json
```

---

## 16. Implementation tasks for first runtime version

### Task 1: Add no-live S5-3a responsibility-map loader

**Objective:** Define field ownership and S5-3 allowed modes without making LLM calls.

**Files:**

- Create or modify a Stage5 helper under `src/stage5_benchmark/` only after this design is accepted.
- Add tests under `tests/test_s5_value_layer_contract_v1.py` or a new focused test file.

**Verification:** Unit tests should prove that S5-2-owned mechanical fields can still be S5-3 eligible only under trace-labeled `mechanical_fallback` or `source_numeric_signal` modes.

### Task 2: Add source-signal detector over Stage2 evidence surfaces

**Objective:** Detect field-family signals from governed Stage2 clean evidence blocks and normalized table/table-cell authority.

**Verification:** Tests should cover `100 nm` as `particle_characterization` signal while retaining wrong-context risk flags for scale bar/filter/pore/thickness contexts.

### Task 3: Add opportunity assignment table

**Objective:** Combine admitted final rows, empty/unresolved fields, field applicability, S5-2 trace, and source signals into `s5_3_value_opportunity_tasks_v1.tsv`.

**Verification:** Tests should prove blank cells with no source signal become `no_task__no_source_signal`, while S5-2 trace misses become `task__s5_2_mechanical_fallback` only when evidence exists.

### Task 4: Add evidence packet assembler

**Objective:** Build auditable evidence packets from source paragraphs/tables, prior Stage2/LLM semantic signals, Stage3 relation context, S5-2 traces, and negative-context evidence before prompt rendering. Reuse Stage5 Evidence Binding authority-resolution, lineage-manifest, assignment-path/status-taxonomy, and validator patterns where semantics match, while keeping S5-3d as a distinct pre-LLM packet builder.

**Verification:** Tests should enforce item-level source refs/hashes, authority-manifest reuse recording, Evidence Binding assignment/status hints where applicable, packet status decisions, inclusion of Stage3/S5-2 context when required, source evidence delimiters, and hold statuses when lawful evidence is insufficient.

### Task 5: Add prompt packet builder

**Objective:** Render bounded prompt packets from approved evidence packets, grouped by paper, evidence packet, field family, and 1-3 row chunks.

**Verification:** Tests should enforce max row-field tasks, no already-filled target fields, exact one output object per task instruction, and source evidence delimiters.

### Task 6: Add or adapt S5-3f runner for structured candidate JSON

**Objective:** Parse model responses into candidate and evidence sidecar TSVs.

**Verification:** Use fixture raw responses before any live calls. Prove `not_reported_in_evidence` rows are retained as coverage, not dropped.

### Task 7: Extend S5-4 validation for opportunity-driven candidates

**Objective:** Validate direct quotes, field boundaries, scope, directness, and wrong-context flags.

**Verification:** Tests should reject quote-less values, derived values in direct layer, wrong-context numeric mentions, and scope-unsupported shared values.

### Task 8: Add S5-6 manifest/link integration only

**Objective:** Link sidecars into Stage5 output directories without changing row membership or final-table values.

**Verification:** Tests should prove `row_membership_changed=no`, `final_table_modified=no`, and `benchmark_valid=no` in the manifest.

---

## 17. First smoke recommendation after implementation

Before any large run:

1. Run S5-3a/S5-3b/S5-3c no-live task assignment on a small explicit lineage.
2. Inspect task table manually for a known S5-2 mechanical miss and one S2-2 numeric signal case.
3. Build S5-3d evidence packets without prompts and audit source refs, Stage2/LLM signals, Stage3 relation context, S5-2 trace, and negative-context warnings.
4. Render S5-3e prompt packets from only `packet_ready_for_prompt` evidence packets.
5. Audit prompt cleanliness and delimiters.
6. Stop before live LLM calls in the first implementation pass because S5-2 must exist first and dry-run audit surfaces need review before calls.
7. After user approval in a later pass, run exactly one live prompt using an explicitly selected lightweight model, then immediately run S5-4/risk validation.
8. Keep all outputs diagnostic-only.

---

## 18. Open questions for next design iteration

- Confirm exact column names for the EE-modeling-focused responsibility-map whitelist in the active final-table schema.
- Which Stage2/clean-text/table sidecar paths are canonical for high-recall signal search in the first accepted S5-2 lineage?
- What exact S5-2 output/trace surface will S5-3 consume after S5-2 is implemented?
- Should prompt grouping prioritize field family or table/evidence packet first when they disagree?
- What risk-assessment sidecar should consume S5-3/S5-4 outputs for high-risk handling?
- When, if ever, should S5-4 accepted values be merged into a separate diagnostic overlay rather than linked as sidecars?
