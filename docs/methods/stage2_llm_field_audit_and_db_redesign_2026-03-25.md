# Stage2 LLM Field Audit And DB Redesign (2026-03-25)

## Scope

This note is an audit-first design review of the current active Stage2 LLM
extraction contract and a proposal for a cleaner downstream database schema.

Authoritative reading basis used in this audit:

- `AGENTS.md`
- `project/ACTIVE_PIPELINE_FLOW.md`
- `project/2_ARCHITECTURE.md`
- `project/4_DECISIONS_LOG.md`
- `project/PIPELINE_SCRIPT_MAP.md`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `docs/methods/llm_deterministic_responsibility_contract_2026-03-06.md`
- `docs/methods/weak_labels_v7_schema_design_2026-03-06.md`
- `docs/methods/v7pilot_r3_fixparse_input_assembly_audit_2026-03-10.md`
- `docs/methods/v7pilot_r3_fixparse_synthesis_method_block_2026-03-10.md`
- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Authoritative artifact source resolution:

- `data/results/ACTIVE_RUN.json`
- resolved active run root:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1`
- resolved Stage2 authority artifact:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`
- matching Stage2 JSONL:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.jsonl`
- matching raw responses:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/`

Comparison-only evidence used to detect code/artifact drift:

- current extractor code in `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- later validation artifact:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/lineage/children/49_stage2_components_shadow_validation/run_20260324_1432_76e3ce2_dev15_stage2_components_shadow_validation_v1/full_stage2_shadow/weak_labels__v7pilot_r3_fixparse.tsv`

## Executive Summary

The current Stage2 contract is wider than the active pipeline really needs for
interpretable formulation-to-EE modeling. The LLM is doing semantic extraction,
but it is also being asked to do coarse evidence typing, conflict narration,
shared-vs-instance ownership, and some inheritance/arbitration work that the
governed architecture says should be deterministic.

The repo evidence supports the claim that Stage2 currently carries too much
evidence-binding responsibility, but only partially in the strict sense:

- supported:
  - raw responses and Stage2 fields show that the LLM is asked to emit
    `supporting_evidence_refs`, `instance_evidence`, field `scope`,
    `membership_confidence`, and field `evidence_region_type`
  - observed raw responses also narrate conflict resolution and value
    prioritization in `paper_notes`
- not fully supported:
  - precise token-safe evidence binding, audit pack generation, numeric QC, and
    table ownership still live downstream in deterministic tooling

Assessment:

- claim status: `partially supported, leaning supported`

Main redesign recommendation:

- keep the LLM focused on formulation identity, component semantics, phase
  semantics, process semantics, and outcome extraction
- move evidence binding, normalization, derivation, and audit-grade pointer
  construction into deterministic post-processing
- replace the current wide-row Stage2 mental model with a normalized database
  schema built around:
  - `document`
  - `formulation_identity`
  - `formulation_component`
  - `formulation_process`
  - `formulation_phase`
  - `formulation_measurement`
  - `evidence_binding`
  - optional `modeling_view`

No active pipeline code was modified in this audit.

## A. Current Active Stage2 Field Inventory

### Directly observed active Stage2 TSV contract

Observed in the active authoritative Stage2 TSV:

- 119 columns total
- 23 top-level row columns
- 16 logical field groups, each flattened into 6 suffix columns:
  - `_value`
  - `_value_text`
  - `_scope`
  - `_membership_confidence`
  - `_evidence_region_type`
  - `_missing_reason`

Observed top-level columns in the active authoritative TSV:

```text
key
doi
model
local_instance_id
formulation_id
raw_formulation_label
polymer_identity
polymer_name_raw
instance_kind
parent_instance_id
change_descriptions
change_role
instance_context_tags
change_context_tags
supporting_evidence_refs
formulation_role
instance_confidence
candidate_source
instance_evidence_region_type
evidence_section
evidence_span_text
evidence_span_start
evidence_span_end
```

Observed logical Stage2 field groups in the active authoritative TSV:

```text
emul_type
emul_method
la_ga_ratio
plga_mw_kDa
plga_mass_mg
surfactant_name
surfactant_concentration_text
pva_conc_percent
organic_solvent
drug_name
drug_feed_amount_text
size_nm
pdi
zeta_mV
encapsulation_efficiency_percent
loading_content_percent
```

### Directly observed current extractor code contract

Observed in current `build_output_columns()`:

- 124 columns total
- same 23 top-level row columns as the active TSV
- plus 3 reconciliation columns:
  - `instance_kind_raw`
  - `instance_kind_inferred`
  - `instance_kind_reconciliation_note`
- same 16 logical field groups, but with canonical `polymer_mw_kDa` naming in
  code rather than active-artifact `plga_mw_kDa`
- plus deterministic enrichment fields:
  - `preparation_method`
  - `emulsion_structure`

### Directly observed raw JSON response structure

Observed in raw LLM responses:

- top-level:
  - `schema_version`
  - `paper_notes`
  - `formulations`
- per formulation:
  - `formulation_id`
  - `raw_formulation_label`
  - `instance_kind`
  - `parent_instance_id`
  - `change_descriptions`
  - `change_role`
  - `supporting_evidence_refs`
  - `instance_context_tags`
  - `change_context_tags`
  - `formulation_role`
  - `instance_confidence`
  - `fields`
- per field object:
  - `value`
  - `value_text`
  - `scope`
  - `membership_confidence`
  - `evidence_region_type`
  - optional `missing_reason`

### Directly observed schema drift and mismatch

Direct observations:

1. Current extractor code emits 124 columns, but the active authoritative Stage2
   TSV has 119 columns.
2. Current extractor code expects canonical `polymer_mw_kDa_*`, but the active
   authoritative Stage2 TSV still carries legacy `plga_mw_kDa_*`.
3. Current extractor code emits:
   - `instance_kind_raw`
   - `instance_kind_inferred`
   - `instance_kind_reconciliation_note`
   - `preparation_method`
   - `emulsion_structure`
   but those columns are absent from the active authoritative Stage2 TSV.
4. Raw LLM responses still use `plga_mw_kDa` in observed papers, so the
   polymer-MW canonical rename is incomplete upstream, not just in old TSVs.

Interpretation:

- fact:
  - code and authoritative artifact do not currently expose the same Stage2
    schema surface
- inference:
  - the repository is in a transition state where the extractor contract has
    evolved faster than the currently pinned benchmark-valid Stage2 artifact

## B. Field Classification

| Field or field group | Classification | Notes |
|---|---|---|
| `formulation_id`, `local_instance_id`, `raw_formulation_label` | formulation identity | row identity and paper-local label surface |
| `instance_kind`, `instance_kind_raw`, `instance_kind_inferred`, `instance_kind_reconciliation_note` | formulation identity | semantic routing and later reconciliation metadata |
| `parent_instance_id` | formulation identity | parent-child or family linkage |
| `formulation_role` | formulation identity | baseline, variant, control, comparative, characterization-only |
| `candidate_source` | formulation identity / provenance | distinguishes `llm_extracted`, `doe_numbered_table_row`, `figure_variable_sweep` |
| `change_descriptions`, `change_role`, `instance_context_tags`, `change_context_tags` | process / identity semantics | identity-defining vs non-synthesis change context |
| `polymer_identity`, `polymer_name_raw` | material / identity | polymer family aid outside the flat field object set |
| `emul_type`, `emul_method`, `preparation_method`, `emulsion_structure` | process / phase / formulation type | method and emulsion-route semantics |
| `la_ga_ratio`, `polymer_mw_kDa` or `plga_mw_kDa`, `plga_mass_mg` | material / component | polymer property and amount surface |
| `surfactant_name`, `surfactant_concentration_text`, `pva_conc_percent`, `organic_solvent` | material / component / phase | component and phase-bearing formulation inputs |
| `drug_name`, `drug_feed_amount_text` | material / component | drug identity and amount |
| `size_nm`, `pdi`, `zeta_mV`, `encapsulation_efficiency_percent`, `loading_content_percent` | outcome / measurement | downstream modeling targets and outputs |
| `_value_text` columns | raw extracted values | closest preserved author wording |
| `_value` columns | semi-normalized extracted values | still Stage2-produced and not yet deterministic canonical values |
| `_scope` columns | ownership / process semantics | instance-specific vs shared semantics |
| `_membership_confidence` columns | semantic uncertainty metadata | LLM judgment about field-to-row ownership |
| `_evidence_region_type` columns | provenance / evidence hints | coarse location typing, not audit-grade binding |
| `_missing_reason` columns | extraction-state metadata | non-reporting, ambiguity, or conflict signals |
| `instance_evidence_region_type`, `evidence_section`, `evidence_span_text`, `evidence_span_start`, `evidence_span_end` | provenance / evidence hints | row-level evidence hints only |
| `supporting_evidence_refs` | provenance / evidence hints | overloaded mixed bag of support claims, locators, and fallback spans |
| `paper_notes` in raw JSONL | provenance / arbitration narrative | observed conflict narration and value prioritization |

## C. Which Stage2 Fields Are Actually Consumed Downstream

### Stage3 relation builder

Directly observed active uses in `src/stage3_relation/build_formulation_relation_artifacts_v1.py`:

- top-level routing and provenance:
  - `key`, `doi`, `model`, `local_instance_id`, `formulation_id`
  - `raw_formulation_label`
  - `instance_kind`
  - `parent_instance_id`
  - `change_descriptions`
  - `change_role`
  - `instance_context_tags`
  - `change_context_tags`
  - `supporting_evidence_refs`
  - `formulation_role`
  - `instance_confidence`
  - `candidate_source`
  - `instance_evidence_region_type`
  - `evidence_section`
  - `evidence_span_text`
  - `evidence_span_start`
  - `evidence_span_end`
- dynamic field extraction from all `_value` columns
- explicit extras:
  - `polymer_identity`
  - `polymer_name_raw`
  - `preparation_method` when present
- direct deterministic canonical alias support:
  - `plga_mw_kDa` -> `polymer_mw_kDa`

What Stage3 appears to actually rely on most:

- identity and routing fields
- field raw values
- field scope
- field evidence-region type
- row-level evidence section/snippet

What Stage3 does not appear to use materially:

- field membership confidence
- field missing reason
- measurement outcome fields for relation logic except as passive membership rows

### Stage5 final-output builder

Directly observed active uses in `src/stage5_benchmark/build_minimal_final_output_v1.py`:

- identity and collapse logic:
  - `formulation_id`
  - `raw_formulation_label`
  - `instance_kind`
  - `parent_instance_id`
  - `change_role`
  - `instance_context_tags`
  - `change_context_tags`
  - `candidate_source`
  - `formulation_role`
  - `polymer_identity`
  - `polymer_name_raw`
- overlap/collapse signatures:
  - `la_ga_ratio_value`
  - `drug_feed_amount_text_value`
  - `plga_mass_mg_value`
  - `surfactant_name_value`
  - `surfactant_concentration_text_value`
  - `organic_solvent_value`
  - `pva_conc_percent_value`
- evidence-adjacent heuristics:
  - `evidence_span_text`
  - `supporting_evidence_refs`
- relation-resolved fill targets:
  - `polymer_mw_kDa`
  - `surfactant_name`
  - `organic_solvent`
  - `preparation_method`

What Stage5 core logic does not materially use:

- field membership confidence
- field missing reason
- row-level span start/end
- most measurement output fields for keep/drop/collapse logic

### Downstream usage summary table

| Field family | Stage3 | Stage5 | Comment |
|---|---|---|---|
| identity/routing fields | yes | yes | core operational surface |
| raw material value fields | yes | yes | core operational surface |
| process fields | yes | partly | `preparation_method` is relation-resolved fill target |
| measurement fields | weakly | mostly passive carry-through | useful for output, not for identity logic |
| field `scope` | yes | limited | Stage3 uses it for shared-field reasoning |
| field `membership_confidence` | no direct active use found | no direct active use found | audit-only at present |
| field `missing_reason` | no direct active use found | no direct active use found | audit-only at present |
| row-level evidence span fields | yes | limited | Stage3 relation provenance, minor Stage5 heuristics |
| `supporting_evidence_refs` | carried | minor heuristic use | not an identity-critical Stage3/5 dependency |

## D. Evidence-Binding Responsibilities Currently Assigned To The LLM

The following responsibilities are currently pushed into Stage2 in ways that
look heavier than the architecture intent:

1. Row-level evidence hint production
   - `instance_evidence_region_type`
   - `evidence_section`
   - `evidence_span_text`
   - `evidence_span_start`
   - `evidence_span_end`
2. Free-form support packing
   - `supporting_evidence_refs`
   - observed as a mixture of table labels, block labels, text snippets, and
     fallback structured objects
3. Shared-vs-instance ownership
   - per-field `scope`
4. Field-to-row ownership confidence
   - per-field `membership_confidence`
5. Coarse evidence region typing per field
   - per-field `evidence_region_type`
6. Conflict arbitration in raw `paper_notes`
   - directly observed in raw responses
   - example behaviors:
     - "Prioritized Table 1 value"
     - "Prioritized the specific value from the Table 2 summary"
     - inferred shared PLGA mass across formulations
7. Implicit inheritance decisions
   - raw responses explicitly state inherited shared parameters in some papers

### Why this matters

The governed architecture says:

- Stage2 should own semantic extraction
- deterministic layers should own evidence binding, normalization, derivation,
  and audit

Observed current state:

- Stage2 is already carrying semantic extraction plus a substantial amount of
  evidence-oriented metadata and arbitration narration
- the resulting evidence surface is often too coarse to be truly audit-grade

Observed artifact quality signal from the active run:

- `supporting_evidence_refs` populated in all 269 rows
- `instance_evidence_region_type`:
  - `unknown` for 189 rows
  - `table_row` for 50 rows
  - `results_sentence` for 30 rows
- `evidence_section`:
  - usually `full_text_window` for LLM-extracted rows

Interpretation:

- the LLM is doing evidence work
- but the evidence product is often not precise enough to justify keeping that
  work in the LLM contract

## E. Recommendation: What Stays In LLM vs Moves To Deterministic

### Keep in Stage2 LLM output

- formulation identity and boundaries
- parent/variant semantics
- change semantics
- multi-component extraction
- phase semantics
- formulation/process semantics
- measurement extraction
- coarse source hint only:
  - `source_hint_type`
  - `source_region_type`
  - `source_locator_text`

### Move out of Stage2 into deterministic post-processing

- exact evidence binding
- table-row and table-cell ownership
- text span anchoring
- conflict arbitration between duplicated evidence values
- unit normalization and conversions
- derived values
- component canonicalization
- phase canonicalization
- final database assembly
- release-ready modeling projections

### Recommended minimal Stage2 contract

If the project goal is interpretable formulation-to-EE modeling rather than
maximal evidence narration, the minimal LLM-side contract should be:

1. formulation identity object
   - `formulation_id`
   - `raw_formulation_label`
   - `parent_instance_id`
   - `instance_kind`
   - `formulation_role`
   - `identity_confidence`
   - `candidate_source`
   - `change_descriptions`
2. component objects
   - `component_name_raw`
   - `component_role_raw`
   - `phase_raw`
   - `amount_expression_raw`
3. process objects
   - `preparation_method_raw`
   - `emulsion_structure_raw`
   - optional local process notes
4. measurement objects
   - `measurement_name_raw`
   - `measurement_value_raw`
   - `measurement_unit_raw`
   - optional statistic qualifier
5. coarse provenance hints only
   - `source_region_type`
   - `source_locator_text`

Not recommended in the minimal LLM contract:

- exact span offsets
- field-level evidence ownership
- field-level evidence binding tables
- per-field membership confidence for every scalar
- free-form conflict-arbitration notes as a required output

## F. Proposed Database Schema

### Design rules

- one semantic fact per table row
- raw, normalized, and derived values separated
- evidence binding separated from semantic extraction
- formulation identity remains authoritative
- multi-component formulations represented explicitly
- phase-aware formulations represented explicitly
- units preserved before conversion

### Table overview

#### `document`

One row per paper.

Purpose:

- store stable paper identity and source asset pointers

#### `formulation_identity`

One row per benchmark-facing or candidate formulation identity.

Purpose:

- authoritative formulation identity layer
- parent/variant structure
- paper-local label preservation

#### `formulation_component`

One row per component within a formulation.

Purpose:

- represent polymers, drugs, surfactants, solvents, oils, co-solvents, and
  additives without wide-row flattening

#### `formulation_process`

One row per process fact or process parameter tied to a formulation.

Purpose:

- represent preparation route and explicit process parameters without embedding
  them in material rows

#### `formulation_phase`

One row per phase in a formulation.

Purpose:

- make W1/O/W2 and related variants explicit

#### `formulation_measurement`

One row per extracted outcome or measurement.

Purpose:

- separate formulation identity from outcomes such as size, PDI, zeta, EE, and
  loading content

#### `evidence_binding`

One row per deterministic or manual evidence-binding event.

Purpose:

- bind a target field in a target record to auditable text/table evidence

#### Optional `modeling_view`

A deterministic projection for modeling.

Purpose:

- produce one row per formulation with normalized modeling features and targets

## Raw vs Normalized vs Derived

### Raw fields

Definition:

- closest preserved author wording or source-surface wording

### Normalized fields

Definition:

- deterministic canonical forms derived without changing the preserved raw text

### Derived fields

Definition:

- computed values that did not appear directly in the source wording

Hard rule recommendation:

- raw fields must never be overwritten by normalized or derived values

## Migration Strategy

### Phase 1: non-disruptive adapter layer

1. Freeze the active Stage2 benchmark-facing TSV and JSONL as source authority.
2. Build a deterministic adapter that reads current Stage2 outputs and writes
   `db_v2` tables without changing Stage2, Stage3, or Stage5.
3. Populate:
   - `document` from manifest and Stage2 metadata
   - `formulation_identity` from top-level Stage2 row fields
   - `formulation_measurement` from current measurement field groups
4. Populate `evidence_binding` only as coarse bindings initially:
   - `binding_status = coarse`
   - preserve current Stage2 evidence hints without pretending they are exact

### Phase 2: component and phase normalization

1. Use Stage2 component shadow outputs where available as the initial
   `formulation_component` bootstrap source.
2. Keep unresolved rows explicit instead of forcing one-component-one-object
   certainty from the current wide-row Stage2 artifact.
3. Build deterministic phase canonicalization into `formulation_phase`.

### Phase 3: deterministic evidence-binding layer

1. Bind table rows/cells and text spans deterministically.
2. Write binding results into `evidence_binding`.
3. Treat Stage2 evidence hints as routing hints, not final authority.

### Phase 4: future Stage2 contract cleanup

1. Update the Stage2 JSON prompt/contract to emit the minimal semantic schema.
2. Keep a backward-compatible TSV flattener for Stage3/Stage5 transition work.
3. Only change downstream consumers after the adapter and compatibility layer
   are stable.

## Safe No-Regret Refactor Order

1. Build the deterministic Stage2-to-`db_v2` adapter from the current frozen
   Stage2 TSV/JSONL.
2. Materialize `formulation_measurement` and `formulation_identity` first.
3. Bootstrap `formulation_component` from the current component shadow sidecar,
   preserving unresolved statuses.
4. Add deterministic `evidence_binding` over frozen records.
5. Introduce a compatibility view that can still emit the legacy wide-row
   Stage2-style surface if downstream scripts need it.
6. Only then simplify the LLM prompt/contract.

## Recommended Next Implementation Step

Implement a deterministic Stage2-to-`db_v2` adapter that reads the current
authoritative Stage2 TSV plus JSONL and writes:

- `document`
- `formulation_identity`
- `formulation_measurement`
- coarse `evidence_binding`

This is the smallest high-signal step because it:

- does not change benchmark-valid pipeline behavior
- exposes exactly which current Stage2 fields are missing or overloaded
- creates a stable landing zone for later component and phase work

## Final Assessment Of The User Hypothesis

Hypothesis:

- the current pipeline may be asking the LLM to care too much about evidence
  binding, while evidence binding is not the LLM's comparative advantage

Assessment:

- `partially supported, leaning supported`

Evidence supporting the claim:

- Stage2 asks the LLM to emit evidence refs, evidence region types, ownership
  scope, membership confidence, and conflict-arbitration notes
- raw responses directly narrate evidence prioritization and inheritance
- active Stage2 evidence outputs are often coarse (`unknown`,
  `full_text_window`) rather than audit-grade

Evidence limiting the claim:

- exact evidence binding, numeric token QC, and audit-pack resolution remain
  downstream deterministic responsibilities in the active pipeline

Bottom line:

- Stage2 is not doing full deterministic evidence binding
- but it is already doing too much coarse evidence-binding and arbitration work
  for the value it returns
