# Authority Reopen Option B Implementation Spec

## Executive Conclusion

This spec defines the **smallest additive implementation** needed to upgrade the current `C_PARTIAL_REOPEN` behavior into a stable explicit authority reopen contract.

The target remains:

- `S2-2 normalized_table_payloads` is the normal downstream row-bearing source of truth
- `S2-4a` remains summary-only
- `S2-5` semantic output may be incomplete
- downstream deterministic expansion may reopen earlier authority surfaces after `S2-5` authorization
- reopen must use explicit stable handles, not `source_raw_response_path` guessing

The minimal implementation strategy is:

1. Write explicit authority-root metadata at the S2-2 authority artifact boundary.
2. Carry that metadata forward passively through existing maintained per-paper metadata surfaces.
3. Attach the explicit authority handle and stable table locators to `S2-5` semantic documents.
4. Switch `S2-7` function units to resolve row-bearing authority from those explicit fields.
5. Retire raw-response-derived reopen as the mainline behavior.

## Locked Design Target

The implementation target is **Option B**:

- explicit authority reopen contract
- stable handles to earlier S2-2 authority surfaces
- no explicit full-payload downstream handoff
- no Stage1 tables as the normal downstream row-bearing source

This is intentionally not:

- a Stage2 redesign
- an S2-5 payload stuffing design
- a selector change
- a prompt-content change

## Minimal Field Contract

### 1. Final minimal required document-level fields

Required in the downstream semantic document:

- `authority_run_dir`
  - exact Stage2 run directory that owns the authoritative S2-2 payload family
- `authority_payload_root`
  - exact path to:
    `semantic_stage2_objects/normalized_table_payloads`

### 2. Optional document-level field

Optional but not strictly required:

- `authority_payload_family`
  - fixed value such as `normalized_table_payloads_v1`

Recommendation:

- **do not require** `authority_payload_family` for the minimal patch

Reason:

- `authority_payload_root` already identifies the family
- the payload manifest already carries `contract_version`
- adding a second constant family field is redundant for stable reopen

### 3. Final minimal required table-locator fields

At the table-scope / authorized-table level, the minimal stable locator is:

- existing `table_id`
- `source_table_asset_id`
- `source_table_reference`

### 4. Fields that are not required for the minimal patch

Not required:

- duplicated `paper_key` inside every table locator
  - already available at document level as `document_key`
- opaque `payload_id`
  - unnecessary if `table_id + source_table_asset_id + source_table_reference` already identify the authority row
- `source_raw_response_path` as an authority handle
  - must remain provenance only, not reopen logic

### 5. Recommended representation

Document-level:

- `authority_run_dir`
- `authority_payload_root`

Scope-level:

- continue to keep existing `table_id`
- add:
  - `source_table_asset_id`
  - `source_table_reference`

On `semantic_scope_declarations`, add a parallel explicit locator family:

- existing:
  - `table_scope_refs: ["Table 1", ...]`
- new:
  - `table_scope_locators: [{"table_id": "...", "source_table_asset_id": "...", "source_table_reference": "..."}]`

Why this is minimal:

- preserves backward compatibility with current `table_scope_refs`
- adds explicit stable execution locators without redesigning the semantic scope structure

## Earliest Write Points

### A. Earliest lawful writer for document-level authority fields

Earliest lawful writer:

- `extract_semantic_stage2_objects_v2.py`
- inside S2-2 normalized payload artifact creation

Current relevant code:

- preserved authority manifest wrapper returned near:
  [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:6318)

Required future write:

- add top-level fields to the normalized payload artifact wrapper:
  - `authority_run_dir`
  - `authority_payload_root`

Why here:

- this is the first place where the authoritative S2-2 payload family is concretely known
- it keeps the authority pointer bound to the actual authority artifact, not reconstructed later

### B. Earliest lawful writer for table locator fields

Earliest lawful writer already exists:

- `extract_semantic_stage2_objects_v2.py`
- preserved payload entries already write:
  - `table_id`
  - `source_table_reference`
  - `source_table_asset_id`

Current lines:
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py:6241)

Required future change:

- no new semantic meaning here
- only harden these fields as required members of the downstream reopen contract

### C. First pass-through writer into the frozen LLM path

Recommended earliest carry point:

- `run_stage2_s2_4a_prompt_construction_v1.py` / prompt record metadata

Why:

- `S2-4a` already has the canonical source evidence path
- it can carry the authority root metadata passively without changing prompt text

Required behavior:

- add `authority_run_dir` and `authority_payload_root` to per-paper prompt metadata artifacts
- do **not** add them to the prompt text itself

### D. First per-paper bridge into raw-response lineage

Recommended pass-through writer:

- `run_stage2_s2_4b_live_llm_call_v1.py`
- request metadata

Why:

- `S2-5` currently works from raw responses
- request metadata is already the natural per-paper bridge beside raw responses

Required behavior:

- copy `authority_run_dir` and `authority_payload_root` from the frozen prompt metadata into request metadata

## Preservation Path

### 1. S2-2

Must carry:

- `authority_run_dir`
- `authority_payload_root`
- per-entry:
  - `table_id`
  - `source_table_asset_id`
  - `source_table_reference`

### 2. S2-4a

Prompt text:

- unchanged

Prompt artifact metadata:

- passively preserve:
  - `authority_run_dir`
  - `authority_payload_root`

Why relevant:

- not needed for prompt legality
- needed as the clean maintained bridge into `S2-4b`

### 3. S2-4b

Raw response payload:

- unchanged

Request metadata:

- passively preserve:
  - `authority_run_dir`
  - `authority_payload_root`

### 4. S2-5

This is the critical stage for semantic-document carry-through.

Required behavior:

- when converting raw responses to Stage2 semantic documents:
  - read sibling request metadata
  - attach:
    - `authority_run_dir`
    - `authority_payload_root`
  - enrich LLM-authorized table scopes / scope declarations with:
    - `source_table_asset_id`
    - `source_table_reference`
    - or `table_scope_locators` for declarations

Why here:

- `S2-5 semantic_jsonl` is the only required semantic input to `S2-6` and `S2-7`
- this is the narrowest place to make downstream reopen explicit without changing the prompt contract

### 5. S2-6

Minimal requirement:

- preserve passively by reading/writing the same semantic JSONL path only

Recommended minimal validation:

- none required for the first implementation patch

Optional later validation:

- if a document contains `row_enumeration_required=yes`, assert presence of:
  - `authority_run_dir`
  - `authority_payload_root`

### 6. S2-7

Must read:

- `authority_run_dir`
- `authority_payload_root`
- `table_scope_locators` or equivalent scope-level locator fields

`S2-7` should not need any new CLI input if the semantic JSONL contains these explicit handles.

## Read Path Changes

### A. `run_stage2_s2_7_compatibility_projection_v1.py`

Required change:

- no runner-input redesign required
- continue consuming `S2-5 semantic_jsonl`
- no explicit new CLI flags necessary in the minimal patch

This file likely changes only if:

- run context or audit reporting needs to mention the explicit authority reopen contract

### B. `build_stage2_compatibility_projection_v1.py`

Required change:

- preserve the new document-level authority fields when normalizing the semantic document into the compatibility view
- preserve scope-level locator fields or `table_scope_locators`

Current analogous preservation area:
- [build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py:679)

### C. `function_units/doe_row_expansion_function_unit_v1.py`

Required change:

- replace mainline `_resolve_semantic_stage2_root(document)` behavior
- primary resolution path must become:
  1. read `authority_payload_root`
  2. open `<authority_payload_root>/<paper_key>/normalized_table_payloads_v1.json`
  3. resolve authorized targets using:
     - `table_id`
     - then `source_table_asset_id`
     - then `source_table_reference`

Legacy behavior:

- `source_raw_response_path` derived root must be retired as the mainline reopen mechanism

### D. `table_row_expansion_v1.py`

Required change:

- same change pattern as DOE expansion
- load normalized payloads from explicit `authority_payload_root`
- resolve scope by stable locator fields, not by raw-response-derived root

## Legacy Behavior to Retire

Retire from the mainline reopen contract:

- implicit reopen from `source_raw_response_path`
- parent/root guessing based on `raw_responses/`
- assuming the raw-response lineage root co-locates normalized payloads

Retirement rule:

- this behavior should no longer be the default execution path

Minimal compatibility allowance:

- if temporary backward compatibility is needed, retain the raw-response-derived logic only as an explicitly labeled legacy fallback branch with:
  - `reopen_source_type = legacy_raw_response_derived`
  - `reopen_resolution_status = fallback_resolved` or `fallback_failed`

But for the main implementation target, the reopen path should be explicit-handle first and contract-governed.

## Failure and Audit Contract

### Required failure labels

Minimum required:

- `authority_root_missing`
- `payload_locator_missing`
- `authorized_target_unresolved`
- `multiple_candidate_payloads`

Recommended additional explicit labels:

- `semantic_authorized_but_row_bearing_source_unavailable`
- `lineage_root_mismatch`
- `stale_handle_into_wrong_lineage`

### Required audit outputs

Minimum required audit fields:

- `reopen_source_type`
- `reopen_resolution_status`
- `reopen_failure_reason`
- `normalized_payload_used`

Minimum recommended additional fields:

- `reopen_authority_run_dir`
- `reopen_authority_root`
- `reopen_payload_locator`
- `semantic_scope_ref`
- `table_scope_ref`

### Meaning

- `reopen_source_type`
  - should be `normalized_table_payloads` for the new mainline
- `normalized_payload_used`
  - yes/no
- `reopen_resolution_status`
  - `resolved`, `unresolved`, `ambiguous`, `failed`
- `reopen_failure_reason`
  - one of the typed failure labels

## Minimal Code Touch Set

### Must change

- [src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)
  - add document-level authority metadata at S2-2 payload wrapper
  - add carry-through support into S2-5 semantic document creation
  - add scope-level locator propagation
- [src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py)
  - passive prompt-record metadata carry of authority root fields
- [src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py)
  - copy authority root metadata into request metadata
- [src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py)
  - load per-paper request metadata and pass explicit authority fields into semantic document construction
- [src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py)
  - preserve new fields into the compatibility view
- [src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py)
  - read explicit authority root + locator
  - retire raw-response-derived mainline reopen
- [src/stage2_sampling_labels/table_row_expansion_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/table_row_expansion_v1.py)
  - same as DOE unit

### Likely change

- [src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/validate_stage2_semantic_authority_contract_v1.py)
  - if adding minimal authority-handle validation
- [src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py)
  - only if report summaries should surface missing authority handles
- [src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py)
  - only if run-context wording or top-level summary fields need to reflect the new explicit contract

### Should not change

- selector logic
- `S2-4a` prompt text
- `S2-2` evidence selection semantics
- Stage1 table asset generation
- Stage3
- Stage5
- `ACTIVE_RUN.json`

## Stepwise Implementation Order

1. **Lock field names**
   - finalize:
     - `authority_run_dir`
     - `authority_payload_root`
     - `source_table_asset_id`
     - `source_table_reference`
     - `table_scope_locators`

2. **Write authority metadata at S2-2**
   - add document-level authority root fields to the normalized payload manifest wrapper
   - harden per-entry locator fields as required contract members

3. **Carry authority metadata through the frozen live-call path**
   - S2-4a prompt record metadata: passive carry only
   - S2-4b request metadata: passive carry only

4. **Attach explicit handles to S2-5 semantic documents**
   - load request metadata during semantic parsing
   - write document-level authority fields
   - enrich table-scope declarations with stable locators

5. **Preserve handles through compatibility projection**
   - keep document-level authority fields
   - keep scope-level locators

6. **Switch readers in DOE/table expansion**
   - primary open path: explicit `authority_payload_root`
   - primary resolution: locator fields
   - stop using `source_raw_response_path` as the mainline authority root

7. **Add failure and audit outputs**
   - emit typed reopen failures
   - emit resolution status and source type

8. **Optional validator tightening**
   - if desired, make `S2-6` fail early for missing authority handles on row-enumeration-required scopes

## FACTS

- `S2-2 normalized_table_payloads` already carries `table_id`, `source_table_asset_id`, and `source_table_reference`.
- `S2-5` currently writes `source_raw_response_path`, `source_text_path`, and `source_table_files`.
- `S2-7` function units currently derive the reopen root from `source_raw_response_path`.
- `S2-6` currently validates semantic JSONL only and can preserve new fields passively.

## INFERENCES

- The smallest stable fix is to propagate an explicit authority root and stable scope locators into semantic JSONL.
- The main engineering gap is not missing row-bearing authority data; it is missing explicit handoff of how to reopen it.
- Existing request metadata is the narrowest maintained bridge from frozen prompts/live calls into semantic parsing.

## UNCERTAINTIES

- Whether `authority_payload_family` is worth keeping as an explicit constant field remains optional.
- Whether `S2-6` should validate authority handles in the first patch or only pass them through can be decided separately.

## NOT IN THIS PATCH

- no prompt text changes
- no selector redesign
- no Stage1 fallback broadening
- no Stage3 or Stage5 changes
- no explicit full-payload downstream handoff redesign

