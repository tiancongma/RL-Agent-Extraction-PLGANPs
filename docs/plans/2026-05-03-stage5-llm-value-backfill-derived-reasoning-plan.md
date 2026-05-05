# Stage5 LLM-Assisted Value Backfill And Derived Reasoning Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Extend the existing Stage5 internal hierarchy without adding a new coarse stage, so fixed formulation rows can receive deterministic direct values, LLM-assisted direct value candidates, audited merge decisions, and separately-provenanced derived/calculated values.

**Architecture:** Stage2 remains responsible for LLM semantic discovery and formulation-boundary authorization. Stage3 remains deterministic relation materialization. Stage5 remains the only final-output namespace, but its internal substeps become explicit: S5-1 fixed-row intake, S5-2 deterministic direct materialization, S5-3 LLM-assisted direct value candidate extraction, S5-4 value authority validation/merge, S5-5 derived reasoning, and S5-6 final table closure. All substeps consume fixed, declared input files and emit auditable TSV/JSON sidecars before any downstream comparison.

**Tech Stack:** Python 3, TSV/JSON artifacts, existing `src/stage5_benchmark/` namespace, existing `data/results/ACTIVE_RUN.json` authority resolution, existing cleaned text/table assets, optional LLM provider only inside S5-3.

---

## Non-Negotiable Requirements

1. Do not create a new coarse runtime stage (`Stage6`, `Stage7`, or a second Stage5 namespace).
2. Keep all implementation under the existing `src/stage5_benchmark/` namespace unless governance explicitly approves a different Stage5-local path later.
3. Do not alter formulation membership in S5-3, S5-4, or S5-5.
4. S5-3 may propose values only for already-fixed Stage5 formulation rows.
5. S5-3 direct-value candidates must include source evidence and a declared scope.
6. S5-5 derived values must never enter current direct-value compare metrics.
7. Every executable substep must accept exact input file paths or resolve them through `data/results/ACTIVE_RUN.json`; no recency, glob-first, latest-child, or parent fallback resolution is allowed.
8. Every run directory must include `RUN_CONTEXT.md` or equivalent metadata sidecars that record exact inputs, code entrypoint, boundary class, benchmark-valid status, and output paths.
9. Original GT files must never be overwritten by automated Stage5 value layers.

---

## Stage5 Internal Contract

### S5-1 Fixed-row candidate intake

**Purpose:** Resolve completed Stage2 and Stage3 artifacts and construct the fixed candidate row universe that downstream Stage5 substeps may annotate but not redefine.

**Inputs:**
- Completed Stage2 weak-label TSV, exact path.
- Stage3 `formulation_relation_records_v1.tsv`, exact path.
- Stage3 `resolved_relation_fields_v1.tsv`, exact path.
- Scope manifest TSV, exact path.

**Outputs:**
- Stage5 fixed row universe in memory and then final candidate materialization surfaces.
- Provenance fields tying each row to Stage2/Stage3 source authority.

**Boundary rule:** mainline Stage5 internal intermediate.

### S5-2 Deterministic direct materialization

**Purpose:** Apply rules that are source-faithful and direct-evidence-only over fixed rows. Current working principle after boundaryfix14: S5-2 is the main deterministic value-materialization layer, and S5-3 LLM is only a post-S5-2 gap-filler. S5-3 must not repeat whole-table or mechanical row-local value extraction when S5-2 can bind the value from preserved table structure, row-local assignments, or unique scoped source text.

**Accepted diagnostic baseline for S5-2 work:**
- baseline lineage: boundaryfix14
- Stage2: `data/results/20260423_9c4a03f/273_stage2_full_replay_tabledict_boundaryfix14_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- Stage3 relation records: `data/results/20260423_9c4a03f/274_stage3_tabledict_boundaryfix14_diagnostic/relation_artifacts/formulation_relation_records_v1.tsv`
- Stage3 resolved fields: `data/results/20260423_9c4a03f/274_stage3_tabledict_boundaryfix14_diagnostic/relation_artifacts/resolved_relation_fields_v1.tsv`
- Stage5 no-S5-3 final table: `data/results/20260423_9c4a03f/275_stage5_tabledict_boundaryfix14_no_s53_diagnostic/final_formulation_table_v1.tsv`
- final count compare: `data/results/20260423_9c4a03f/277_compare_final_tabledict_boundaryfix14_diagnostic/final_table_vs_gt_counts.tsv`
- Layer3 compare: `data/results/20260423_9c4a03f/278_layer3_compare_tabledict_boundaryfix14_diagnostic/layer3_value_compare_summary_v1.tsv`

Boundaryfix14 baseline metrics:
- DEV15 final counts all match current Layer1 GT.
- Key preserved counts: `PA3SPZ28 = 3/3`, `5GIF3D8W = 26/26`, `INMUTV7L = 12/12`, `QLYKLPKT = 7/7`, `WFDTQ4VX = 30/30`, `WIVUCMYG = 26/26`, `YGA8VQKU = 17/17`.
- Layer3 core baseline: `system_nonempty_on_gt_cells = 2086`, `value_recall = 0.668804`, `conditional_accuracy_strict = 0.610259`, `conditional_accuracy_relaxed = 0.933845`, `extra_in_system_cells = 30`, `risk_review_queue_rows = 1599`.

**Allowed work:**
- DOE/table row materialization already authorized upstream.
- Row-local table cell binding.
- Full-table recovery is coordinate-preserving execution authority, not a visual
  cleanup surface. It must preserve blank placeholder columns/cells whenever they
  carry header/value geometry, and S5-2 should consume this preserved table/grid
  contract for mechanical values.
- Summary table rendering is separate from full-table authority. Summary views
  may compress rows/columns, select only the first row/first columns, or otherwise
  reduce prompt burden for the LLM, but they are semantic-facing only and must
  not feed numeric materialization, table-row expansion, or header/value binding.
- Row-local table source-CSV/header rebinding when the final row preserves a unique table-row locator/snippet and the source CSV header has exactly one unambiguous target metric column. This is compatibility fallback debt, not the preferred long-term source once the preserved full-table/grid contract is fully consumed downstream.
- Group/continuation label carry-down already preserved in Stage2/table structure, surfaced in Stage5 without inventing new rows.
- Source-backed shared direct carrythrough when scope is unique.
- Method paragraph scoped carrythrough for direct preparation constants such as drug identity, polymer mass, organic solvent volume, external aqueous volume, and pH only when uniqueness and row eligibility are proven.
- Value/unit split when the source cell directly contains both.
- Filtering/normalization/identity guardrails already governed by Stage5.

**Forbidden work:**
- Donor fill.
- New formulation discovery.
- Assumption-based inference.
- Derived arithmetic.
- LLM calls.
- Using GT values, S5-3 candidates, or system-output agreement as value authority.
- Letting helper/control/sequential-child tables create or alter the fixed row universe.

**Immediate S5-2 mechanical-value repair queue:**
1. Row-local table-column binding for fields that are visibly mechanical table cells: `pdi`, `ee_percent`, `particle_size_nm`, `zeta_mV`, `surfactant_name`, `emulsifier_stabilizer_name`.
2. Group label carry-down into final row surfaces for structured table groups such as `PLGA 503 H`, `PLGA-5%`, `PLGA 10%`, and `PLGA 15%`.
3. Source-scoped preparation carrythrough for direct constants such as `drug_name`, `polymer_mass_mg`, `O_volume_mL`, `external_aqueous_phase_volume_mL`, and `pH_raw`, respecting row-local authority above scoped carrythrough.
4. After each S5-2 patch, rebuild Stage5 from the exact boundaryfix14 Stage2/Stage3 inputs and compare against the boundaryfix14 metrics above. `final_count_nonmatches` must remain `0`; the key preserved counts must remain unchanged; Layer3 coverage should improve or at minimum not regress.
5. Only after this queue is exhausted should S5-3a `target_mode=missing_system` be regenerated; remaining targets are the true LLM gap-fill surface.

**Outputs:**
- `s5_2_rule_direct_values_v1.tsv` when split out of the current builder, or equivalent columns in `final_output_decision_trace_v1.tsv` until decomposition occurs.
- Rule decision trace rows for every materialized value.

### S5-3 LLM-assisted direct value candidate extraction

**Purpose:** Let an LLM read local evidence for already-fixed formulation rows and propose direct Layer3 value candidates with quotes and scope labels.

**Inputs:**
- Exact `final_formulation_table_v1.tsv` or fixed-row pre-final candidate TSV from the same Stage5 run.
- Exact `final_output_decision_trace_v1.tsv`.
- Exact scope manifest TSV.
- Exact Stage1 cleaned text path(s): `data/cleaned/content/text/<paper_key>.*.txt` as resolved from the manifest/scope, not by ad hoc search.
- Exact Stage1/S2 table payload paths, preferring S2 preserved `normalized_table_payloads_v1.json` when available and falling back only through declared Stage1 table paths.
- Optional exact Zotero attachment path when recorded in the source inventory.

**Prompt contract:**
- Input is a fixed row list plus bounded source evidence.
- The LLM must not create or delete rows.
- The LLM must not calculate values.
- The LLM must classify each candidate as `direct`, `derived`, `ambiguous`, or `absent`.
- The LLM must include `source_quote`, `source_table_id` or text location, and `evidence_scope`.

**Outputs:**
- `s5_3_llm_direct_value_candidates_v1.tsv`
- `s5_3_llm_direct_value_evidence_sidecar_v1.tsv`
- `s5_3_llm_raw_responses/<paper_key>__value_backfill_raw_response.json`
- `s5_3_llm_prompt_audit_v1.tsv`
- `RUN_CONTEXT.md`

**Candidate TSV minimum columns:**
- `paper_key`
- `formulation_id`
- `field_name`
- `value_text`
- `unit_text`
- `raw_cell_text`
- `direct_or_derived`
- `evidence_type`
- `evidence_scope`
- `source_file`
- `source_table_id`
- `source_row_id`
- `source_quote`
- `confidence`
- `needs_review`
- `llm_model`
- `prompt_hash`
- `input_artifact_hash`

**Boundary rule:** diagnostic/supporting candidate boundary until validated by S5-4; not a final-output boundary.

### S5-4 Value authority validation and merge

**Purpose:** Merge S5-2 deterministic values and S5-3 LLM candidates without letting either bypass authority rules.

**Authority order:**
1. Row-local direct source cell binding.
2. Typed row-local assignment/direct table binding.
3. Unique table-scoped direct value with row binding.
4. Unique paper/global direct preparation/material constant with eligible-row proof.
5. LLM direct candidate accepted only when it satisfies the same evidence/scope requirements.
6. Derived candidates are rejected from direct final fields and passed to S5-5 or review.

**Validator decisions:**
- `accepted_direct`
- `rejected_derived_for_direct_layer`
- `rejected_missing_quote`
- `rejected_scope_ambiguous`
- `rejected_conflict_with_higher_authority`
- `review_needed`

**Outputs:**
- `s5_4_value_authority_decisions_v1.tsv`
- `s5_4_accepted_direct_values_v1.tsv`
- `s5_4_rejected_value_candidates_v1.tsv`
- `s5_4_value_review_queue_v1.tsv`
- conflict/audit summary JSON

**Boundary rule:** Stage5 internal validation boundary; accepted direct values may feed S5-6.

### S5-5 Derived reasoning / calculated value materialization

**Purpose:** Compute values that are not directly reported but are mathematically derivable from accepted direct inputs.

**Allowed formula families:**
- `%w/v × mL -> mg`
- `mg/mL × mL -> mg`
- concentration × volume -> mass
- ratio × known mass -> missing endpoint mass
- unit conversion between compatible units

**Inputs:**
- S5-4 accepted direct values.
- Frozen final row identities.
- Formula registry file or in-code registry with explicit formula IDs.

**Outputs:**
- `s5_5_derived_values_v1.tsv`
- `s5_5_derived_provenance_v1.tsv`
- `s5_5_derived_review_queue_v1.tsv`

**Derived TSV minimum columns:**
- `paper_key`
- `formulation_id`
- `target_field_name`
- `derived_value`
- `derived_unit`
- `formula_id`
- `formula_expression`
- `input_field_names`
- `input_values`
- `input_source_provenance`
- `eligible_for_direct_compare` (always `no`)
- `eligible_for_derived_compare`
- `needs_review`

**Boundary rule:** derived sidecar boundary; never current direct compare authority.

### S5-6 Final table closure and audit export

**Purpose:** Emit the benchmark-facing final formulation table and linked sidecars.

**Outputs:**
- `final_formulation_table_v1.tsv`
- `downstream_variant_records_v1.tsv`
- `final_output_decision_trace_v1.tsv`
- `final_output_summary_v1.md`
- value-layer sidecars from S5-2 through S5-5 when enabled

**Boundary rule:** `final_formulation_table_v1.tsv` remains the primary Stage5 final output. Direct and derived sidecars must be clearly separated in downstream compare/reporting.

---

## Implementation Tasks

### Task 1: Governance documentation update

**Objective:** Record the new Stage5 internal design without changing coarse pipeline stages.

**Files:**
- Modify: `project/2_ARCHITECTURE.md`
- Modify: `project/ACTIVE_PIPELINE_FLOW.md`
- Modify: `project/ACTIVE_PIPELINE_RUNBOOK.md`
- Modify: `project/PIPELINE_SCRIPT_MAP.md`
- Modify: `project/4_DECISIONS_LOG.md`

**Steps:**
1. Replace the old three-part Stage5 substep description with S5-1 through S5-6.
2. Add the rule that S5-3/S5-4/S5-5 cannot change formulation membership.
3. Add the direct-vs-derived separation rule.
4. Add the exact input pinning rule for S5-3 and S5-5.
5. Verify no new project governance file is created.

**Verification:**
Run `python3 - <<'PY' ...` to scan these files for the new substep labels and for any accidental tool-display truncation marker.

### Task 2: Register planned Stage5 value-layer surfaces as design, not maintained scripts

**Objective:** Avoid pretending future scripts exist while still documenting planned file contracts.

**Files:**
- Modify: `project/PIPELINE_SCRIPT_MAP.md`
- Do not modify: `docs/maintained_script_surface.tsv` until scripts exist.

**Steps:**
1. Add a Stage5 planned decomposition note after the active Stage5 entrypoint row or under Production Path Notes.
2. State that `build_minimal_final_output_v1.py` remains the only current maintained final-table entrypoint until S5-3/S5-5 runners are implemented and registered.

**Verification:**
Check `docs/maintained_script_surface.tsv` has no fake entries for nonexistent S5-3/S5-5 scripts.

### Task 3: Implement S5-3 diagnostic runner skeleton

**Objective:** Create a non-mainline runner that can resolve exact inputs, write prompt/candidate placeholders, and fail safely before any live LLM provider is configured.

**Files:**
- Create: `src/stage5_benchmark/build_s5_3_llm_direct_value_candidates_v1.py`
- Create/Modify tests: `tests/test_s5_value_layer_contract_v1.py`

**Expected behavior:**
- CLI accepts explicit `--final-table-tsv`, `--decision-trace-tsv`, `--scope-manifest-tsv`, `--out-dir`.
- Optional `--source-inventory-tsv` pins source text/table/PDF paths.
- Default mode writes empty candidate/audit files plus `RUN_CONTEXT.md` and exits with a clear message when no LLM backend is configured.
- It must not infer source paths by latest directory or glob-first search.

**Verification:**
- `python3 -m unittest tests.test_s5_value_layer_contract_v1`
- Confirm output files exist and carry exact source path metadata.

### Task 4: Implement S5-4 validator skeleton

**Objective:** Validate candidate rows against direct/derived/scope evidence requirements before merge.

**Files:**
- Create: `src/stage5_benchmark/validate_s5_value_candidates_v1.py`
- Modify tests: `tests/test_s5_value_layer_contract_v1.py`

**Expected behavior:**
- Accepts `s5_3_llm_direct_value_candidates_v1.tsv` and optional `s5_2_rule_direct_values_v1.tsv`.
- Rejects `direct_or_derived=derived` from direct layer.
- Rejects missing `source_quote` for direct candidates.
- Marks ambiguous scope as review-needed.
- Writes accepted/rejected/review TSVs.

**Verification:**
- Unit tests for accepted direct, derived rejection, missing quote rejection, ambiguous scope review.

### Task 5: Implement S5-5 derived reasoning skeleton

**Objective:** Add a derived-value sidecar builder that supports formula-family registration without affecting direct compare.

**Files:**
- Create: `src/stage5_benchmark/build_s5_5_derived_values_v1.py`
- Modify tests: `tests/test_s5_value_layer_contract_v1.py`

**Expected behavior:**
- Reads S5-4 accepted direct values.
- Supports first formula family `%w/v × mL -> mg` only after unit tests specify exact behavior.
- Writes derived sidecar rows with `eligible_for_direct_compare=no`.

**Verification:**
- Unit test that derived mass is computed and flagged ineligible for direct compare.
- Unit test that insufficient inputs produce review/no output, not guessed values.

### Task 6: Integrate into Stage5 only after skeleton validation

**Objective:** Connect optional S5-3/S5-4/S5-5 sidecars to Stage5 final-table production without changing default benchmark outputs.

**Files:**
- Modify: `src/stage5_benchmark/build_minimal_final_output_v1.py` only after prior skeletons pass.

**Expected behavior:**
- Default Stage5 output remains unchanged when value-layer inputs are absent.
- Optional CLI flags may consume S5-4 accepted direct sidecar and S5-5 derived sidecar.
- Direct sidecar values can appear in final table only after S5-4 acceptance.
- Derived sidecar values remain separate unless a future derived-aware output schema is explicitly requested.

**Verification:**
- Existing `tests.test_compare_layer3_values_v1` remains passing.
- A no-sidecar Stage5 replay produces unchanged row count.
- A synthetic sidecar test proves direct accepted value merge does not alter row membership.

---

## Immediate Execution Scope For This Session

This session executes Task 1 and Task 2 only. Tasks 3-6 are implementation backlog after the governance contract is reviewed, because they create new scripts and runtime behavior.
