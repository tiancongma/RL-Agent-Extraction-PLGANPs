# Stage2 Replacement Compatibility Adapter (2026-03-25)

## Purpose

This note records the first deterministic compatibility adapter for the true
Stage2 replacement effort.

Observed repo fact:
- Active benchmark runtime still flows through the legacy wide-row Stage2
  surface into Stage3 and Stage5.
- The approved replacement direction is semantic-object Stage2 output, not
  continued expansion of the fixed-slot Stage2 contract.

Decision in this pass:
- Add a deterministic transitional adapter that reads semantic-object Stage2
  payloads and projects them back into the legacy wide-row Stage2 surface.

Non-decision:
- This adapter does not make the semantic-object contract the active benchmark
  runtime by itself.

## Why The Adapter Is Needed

- Stage3 and Stage5 currently depend on the legacy Stage2 wide-row surface.
- We want the replacement Stage2 to become operational without forcing another
  round of fixed-slot-first LLM design.
- A deterministic bridge keeps the new semantic core clean while preserving
  current downstream compatibility during migration.

## Inputs

Primary input:
- semantic-object Stage2 JSONL payloads with these object families:
  - `formulation_identity_candidate`
  - `component_candidate`
  - `phase_candidate`
  - `process_step_candidate`
  - `variable_or_factor_candidate`
  - `measurement_candidate`
  - `relation_cue`
  - `evidence_handoff`

Implementation:
- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

## Outputs

Compatibility artifacts:
- `weak_labels__v7pilot_r3_fixparse.tsv`
- `weak_labels__v7pilot_r3_fixparse.jsonl`
- `compatibility_projection_trace_v1.tsv`
- `compatibility_projection_summary_v1.json`

Contract artifact:
- `data/db/db_v2/stage2_replacement_compatibility_projection_contract.tsv`

## Projection Rules

Direct projection:
- formulation identity fields from `formulation_identity_candidate`
- component names into legacy component slots
- process names into `emul_method` and `preparation_method`
- measurement values by deterministic measurement-name matching
- coarse evidence locator fields from `evidence_handoff`

Derived projection:
- `polymer_identity` from polymer component names
- `emul_type` from phase candidates and coarse factor hints
- `*_scope` from repeated row values within one document
- `*_membership_confidence` from projection status, not from new LLM output

Unavailable or partial projection:
- fields remain blank when the replacement objects do not contain enough
  deterministic information
- `*_missing_reason` records that the value was not projectable from the
  current replacement payload
- exact evidence ownership binding is intentionally not invented here

## Transitional Limits

- Multiple same-role components are compressed into pipe-delimited legacy text
  values when necessary for downstream compatibility.
- The adapter does not normalize units into canonical scientific forms.
- The adapter does not perform final evidence arbitration.
- Legacy field names such as `plga_mass_mg` remain only as transitional output
  names for Stage3 and Stage5 compatibility.

## Governance Status

- Transitional deterministic support infrastructure
- Not an active benchmark entrypoint by itself
- Compatible with the approved Stage2 replacement architecture direction
- Historical results under `data/results/` remain untouched by this method note

## Stage2 JSON Sanitation Patch (2026-04)

### Problem

- Recent live Stage2 runs began failing at the strict JSON parse boundary with
  `JSONDecodeError` after the model response was already received.
- The failing raw responses were malformed or truncated at the tail rather than
  failing at transport time.

### Root Cause

- The maintained live Stage2 path now uses a stricter JSON object contract.
- Ordered evidence packing and table-heavy prompts increased response size and
  made tail truncation more likely on harder papers.

### Solution: Path 1

- Apply a conservative sanitation layer before strict parse.
- Perform only mechanical cleanup:
  - trim leading/trailing code fences
  - trim obvious leading junk before the first JSON start
  - attempt a narrow balanced-close repair only when the document is clearly
    truncated at the tail
- Do not invent missing semantic content.
- Preserve the original raw response artifact and write an auditable sanitation
  sidecar when repair occurs.

### Validation

- Offline reuse validation succeeded on the previously failing responses for
  `WIVUCMYG` and `BB3JUVW7`.
- The same sanitation path preserved successful examples unchanged for
  `UFXX9WXE`, `5ZXYABSU`, and `5GIF3D8W`.
- Live confirmation on `BB3JUVW7` succeeded with `parse_stage=balanced_close`,
  and the downstream Stage3, Stage5, and GT comparison surfaces all completed.

### Scope Limitation

- This patch only addresses tail-truncation or mechanical JSON contamination.
- It does not attempt broad tolerant parsing or semantic reconstruction.
- Internal JSON corruption that cannot be repaired by a narrow balanced-close
  rule remains outside Path 1.

### Solution: Path 2

- Historical fixparse fallback basis:
  - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py::safe_json_load`
- Repair behavior:
  - strip code fences
  - attempt a broad brace-window extraction from the cleaned string
  - if that fails, fall back to a default empty weak-label schema object
- Governance interpretation:
  - diagnostic-only fallback, not mainline authority
  - broader than syntax-only repair
  - should be treated as `mixed_or_unclear` rather than a promoted governed fallback
- Validation on the frozen DEV15 raw baseline:
  - Path 1 produced no failures, so Path 2 was not exercised on this baseline
  - the historical basis remains useful for comparison and failure-localization work only

## Stage2 Schema-Aware Raw Rehydration Mode (2026-04)

### Problem

- The governed raw Stage2 freeze boundary preserved current live-v2 raw
  responses but could not legally return to mainline because the maintained
  replay branch expected legacy raw-response structure and emitted zero
  formulations on the frozen DEV15 baseline.

### Solution

- Extend the maintained replay branch in
  `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py` so it
  detects current live-v2 raw-response payloads directly.
- When the saved raw response already contains the current Stage2 v2 object
  families, normalize that payload into the maintained Stage2 semantic
  intermediate and then pass it through the maintained deterministic
  compatibility projection.
- Keep legacy replay support intact for older saved raw-response payloads.

### Governance Status

- Maintained replay/rehydration support inside the existing composite Stage2
  entrypoint.
- The raw-response freeze remains `diagnostic_boundary` by itself.
- The lawful Stage3 upstream boundary is still the completed Stage2 artifact
  re-emitted by the maintained composite Stage2 replay path.

### Validation

- DEV15 frozen raw baseline:
  - `data/results/20260402_5c1e7a4/01_dev15_raw_llm_freeze/frozen_raw_responses`
- Maintained rehydration run:
  - `data/results/20260402_5c1e7a4/12_dev15_stage2_livev2_raw_rehydration`
- Result:
  - `15/15` papers produced nonzero completed Stage2 rows
  - `131` completed Stage2 rows were emitted
  - the resulting completed Stage2 artifact is again lawful upstream input for
    Stage3
