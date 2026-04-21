# Authority Reopen Option B Patch Report

## What Changed

This patch implements the approved minimal Option B authority reopen contract for
Stage2 downstream expansion.

The patch is intentionally narrow:

- `S2-2 normalized_table_payloads` remains the sole normal downstream
  row-bearing source of truth.
- `S2-4a` remains summary-only.
- `S2-7` no longer depends on `source_raw_response_path` as its mainline reopen
  mechanism.
- explicit authority handles are written once upstream, carried passively
  through the frozen LLM boundary, and consumed only by the Stage2 completion
  readers.

## Files Changed

- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
- `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`
- `src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`
- `src/stage2_sampling_labels/table_row_expansion_v1.py`

No selector logic, prompt text, Stage3, Stage5, or `ACTIVE_RUN.json` was
changed.

## Fields Added

Document-level:

- `authority_run_dir`
- `authority_payload_root`

Scope-level:

- existing `table_id` preserved
- `source_table_asset_id`
- `source_table_reference`
- additive `table_scope_locators`

Typed reopen audit fields:

- `reopen_source_type`
- `reopen_resolution_status`
- `reopen_failure_reason`
- `normalized_payload_used`

## Where Fields Are Written

### S2-2

In `extract_semantic_stage2_objects_v2.py`:

- `build_evidence_blocks_artifact(...)`
  - now writes top-level `authority_run_dir` and `authority_payload_root`
    into `evidence_blocks_v1.json`
- `build_normalized_table_payload_artifact(...)`
  - now writes top-level `authority_run_dir` and `authority_payload_root`
    into `normalized_table_payloads_v1.json`

### S2-4a / S2-4b pass-through

- `run_stage2_s2_4a_prompt_construction_v1.py`
  - now preserves `authority_run_dir` and `authority_payload_root` in frozen
    prompt rows and audit rows
- `run_stage2_s2_4b_live_llm_call_v1.py`
  - now preserves `authority_run_dir` and `authority_payload_root` in request
    metadata sidecars

### S2-5 semantic documents

- `run_stage2_s2_5_semantic_parsing_v1.py`
  - now reads sibling request metadata and passes authority metadata into the
    semantic document normalizer
- `extract_semantic_stage2_objects_v2.py`
  - live/replay semantic document builders now accept authority metadata
  - finalized semantic docs now carry:
    - `authority_run_dir`
    - `authority_payload_root`
    - `table_scope_locators`
    - `source_table_asset_id`
    - `source_table_reference`
    where resolvable from the preserved normalized payload authority surface

### S2-7 compatibility view

- `build_stage2_compatibility_projection_v1.py`
  - now preserves `authority_run_dir` and `authority_payload_root`
  - now preserves table locator surfaces in both shrunken and canonical
    normalization branches

## Where Fields Are Consumed

### DOE expansion

`src/stage2_sampling_labels/function_units/doe_row_expansion_function_unit_v1.py`

Mainline behavior now:

1. read `authority_payload_root`
2. open `<authority_payload_root>/<paper_key>/normalized_table_payloads_v1.json`
3. resolve payloads by explicit table locator data
4. emit typed reopen audit fields

### Table expansion

`src/stage2_sampling_labels/table_row_expansion_v1.py`

Mainline behavior now:

1. read `authority_payload_root`
2. open `<authority_payload_root>/<paper_key>/normalized_table_payloads_v1.json`
3. resolve payloads by explicit table locator data
4. emit typed reopen audit fields in the projection summary

## Legacy `source_raw_response_path` Reopen

Legacy raw-response-derived reopen was **retired as the mainline behavior**.

Current status:

- explicit authority reopen is primary
- raw-response-derived reopen remains only as a backward-compatibility fallback
  for older semantic artifacts that do not carry the new authority fields
- when that fallback is used, `reopen_source_type` is set to
  `legacy_raw_response_derived`

The bounded validation below exercised the explicit path, not the legacy
fallback.

## Bounded Validation

Validation lineage root:

- `data/results/20260421_bf6c1a2`

Maintained entrypoints used:

1. `src/stage2_sampling_labels/run_stage2_composite_v1.py --stop-before-live-call`
2. `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
3. `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
4. `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
5. `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
6. `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`

Target papers:

- `UFXX9WXE`
- `WIVUCMYG`
- `5GIF3D8W`

Reason the narrow pre-LLM refresh was required:

- the previously accepted `20260421_3579206/09_selector_contract_dev15_prellm`
  artifacts were created before this patch and therefore did not yet contain
  `authority_run_dir` / `authority_payload_root`

### Before vs After

`UFXX9WXE`

- before:
  - `binding_success = no`
  - `normalized_payload_used = no`
  - `skip_reason = authorized_target_unresolved`
- after:
  - `binding_success = yes`
  - `normalized_payload_used = yes`
  - `reopen_source_type = normalized_table_payloads_explicit`
  - `reopen_resolution_status = resolved`
  - `resolved_execution_target` points to the normalized payload CSV
  - `skip_reason = no_rows_emitted`

`WIVUCMYG`

- before:
  - `binding_success = no`
  - `normalized_payload_used = no`
  - `skip_reason = authorized_target_unresolved`
- after:
  - `binding_success = yes`
  - `normalized_payload_used = yes`
  - `reopen_source_type = normalized_table_payloads_explicit`
  - `reopen_resolution_status = resolved`
  - `resolved_execution_target` points to the normalized payload CSV
  - `skip_reason = no_rows_emitted`

`5GIF3D8W`

- before:
  - `binding_success = no`
  - `normalized_payload_used = no`
  - `skip_reason = authorized_target_unresolved`
- after:
  - `binding_success = yes`
  - `normalized_payload_used = yes`
  - `reopen_source_type = normalized_table_payloads_explicit`
  - `reopen_resolution_status = resolved`
  - `resolved_execution_target` points to the normalized payload CSV
  - `skip_reason = no_rows_emitted`

### Row Counts

Stage2 completed row counts in the bounded validation did **not** increase:

- `UFXX9WXE`: `2 -> 2`
- `WIVUCMYG`: `3 -> 3`
- `5GIF3D8W`: `2 -> 2`

Interpretation:

- explicit authority reopen now works
- the remaining zero-growth problem is no longer authority access
- the next unsolved issue is downstream row enumeration or materialization from
  the resolved normalized payload surface

## Risks / Residual Limitations

- older lineages without the new authority metadata still require the legacy
  fallback path
- bounded validation confirms stable payload resolution, but not broader count
  recovery
- `feature_activation_report_v2.tsv` and `execution_ledger_v2.tsv` still expose
  a compact governed surface; the new reopen audit fields are visible in
  `compatibility_projection_summary_v1.json` rather than every flattened audit
  TSV

## Still Not Solved

- DOE row expansion still emitted `0` rows for the bounded target papers
  even after successful explicit payload reopen
- non-DOE table expansion still emitted `0` rows for the bounded target papers
  where downstream scope logic remained blocked by DOE boundaries or unsupported
  varying-variable counts
- this patch does **not** repair the enumerator or materializer logic itself
- this patch does **not** change Stage3 or Stage5 behavior
