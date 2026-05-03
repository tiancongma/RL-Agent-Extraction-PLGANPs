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

**Purpose:** Apply rules that are source-faithful and direct-evidence-only over fixed rows.

**Allowed work:**
- DOE/table row materialization already authorized upstream.
- Row-local table cell binding.
- Source-backed shared direct carrythrough when scope is unique.
- Value/unit split when the source cell directly contains both.
- Filtering/normalization/identity guardrails already governed by Stage5.

**Forbidden work:**
- Donor fill.
- New formulation discovery.
- Assumption-based inference.
- Derived arithmetic.
- LLM calls.

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
