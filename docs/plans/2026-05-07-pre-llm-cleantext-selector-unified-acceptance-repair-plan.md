# Pre-LLM Clean Text / Evidence Selector Unified Acceptance Repair Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task when delegating; keep writes serial and verify specialist findings before treating them as final.

**Goal:** Define and enforce diagnostic acceptance standards for clean text, unified PDF/HTML structure/table authority, and Stage2 evidence selector before any full-corpus live LLM calls.

**Architecture:** Stage1 emits a source-agnostic downstream interface: flattened clean text plus optional structured blocks/sections plus table/cell authority sidecars. PDF/Marker and HTML parsers may retain provenance (`source_type`, `parser`, `parser_variant`, hashes), but downstream Stage2 consumes the same logical surfaces. Stage2 S2-2 remains a pre-LLM evidence-construction boundary: selector ranks and packs semantic-facing evidence while preserving non-noise authority; it must not use user-uploaded excerpts as runtime evidence or force equality to those excerpts.

**Tech Stack:** Python 3, TSV/JSON diagnostic artifacts, `unittest`, maintained Stage1/Stage2 entrypoints under `src/stage1_cleaning/` and `src/stage2_sampling_labels/`.

---

## Governance and boundaries

- Startup governance read before this plan: `project/0_PROJECT_CHARTER.md`, `project/1_REQUIREMENTS.md`, `project/2_ARCHITECTURE.md`, `project/PIPELINE_SCRIPT_MAP.md`, `project/ACTIVE_PIPELINE_FLOW.md`, `project/ACTIVE_PIPELINE_RUNBOOK.md`, `project/FILE_NAMING_AND_VERSIONING.md`, `project/ACTIVE_DATA_SOURCE_CONTRACT.md`.
- This plan is a non-governance repair plan under `docs/plans/`; it does not create new `project/` files.
- All outputs from these audits are `diagnostic_only=yes`, `benchmark_valid=no`.
- Do not update `data/results/ACTIVE_RUN.json` in this work.
- Do not run live LLM calls until a specific batch passes the pre-live acceptance gate and the user gives explicit batch-level approval.
- User-provided source excerpts at `docs/methods/layer3_field_gt_protocol_v1.md:1098-1864` are audit anchors only. The acceptance standard uses differences between anchors and system clean/evidence surfaces to classify failure boundaries; it does not force runtime output to match those snippets and does not inject snippets into Stage2.

## Acceptance standard v1

A paper/batch is live-LLM eligible only when promoted rows pass all applicable checks below. Failed rows go to hold/review with first-failure boundary and repair target.

### Mainline integration completion gate

A repair is not complete if it only passes local tests, parser bakeoff, synthetic smoke, or a run-local `data/results/.../targeted_manifest.tsv`. Before any repair is marked completed, record its status as one of:

- `mainline_integrated`: reachable from maintained entrypoints and hydrated into the authoritative data surface consumed by future baselines.
- `maintained_code_only_not_hydrated`: code path is reachable, but authoritative data/index surfaces are not rebuilt yet.
- `diagnostic_only_not_mainline`: useful diagnostic artifact/smoke only; future baselines will not benefit.

For Stage1/Stage2 sidecar repairs, `mainline_integrated` requires governed hydration into `data/cleaned/index/manifest_current.tsv` or a governed index that maintained Stage2 refreshes from, plus a no-live downstream replay showing explicit `loaded`/`consumed` status. This gate prevents fixes from being lost when the next baseline is run.

### A. Clean text / structure acceptance

Required per paper:

1. `text_path` exists and is readable.
2. `text_source_type` / `source_type` is recorded (`PDF`, `HTML`, or compatible normalized value).
3. A unified structure sidecar is loaded or explicitly marked `not_available` with reason. The downstream shape must be source-agnostic:
   - `blocks[]` with text, block type, section id/label/kind, noise class/reason, source/provenance.
   - `sections[]` with section id/label/kind/path and block span.
4. Method/preparation body and formulation/result/EE signals are visible in clean text or preserved table authority when expected from candidate evidence.
5. Source-anchor visibility differences are classified as diagnostics (`stage1_clean_text_visibility`, `stage1_table_authority_visibility`, etc.), not used as direct extraction authority.

### B. Table/cell authority acceptance

Required for selected or preserved table evidence:

1. Every selected table summary has a backing full authority artifact:
   - `normalized_table_payloads_v1.json`, payload CSV/grid, or Stage1 table-cell sidecar consumed into the payload.
2. PDF/Marker and HTML must converge to the same downstream logical interface: table id, caption, row/col coordinates, raw/normalized cell text, header/row labels when available, source/provenance fields, and warning metadata.
3. HTML-native DOM cells should be used as first-class table-cell authority for HTML sources when available; CSV summaries alone are insufficient for final promotion if DOM cells exist but are not connected.
4. Marker PDF table/cell sidecars must be frozen/reused from source/parser-hash-bound Stage1 artifacts before promotion. If Marker is unstable for a paper, mark it explicitly and use current fallback without pretending Marker is active.
5. Only confirmed pure noise may be excluded; non-noise tables must remain preserved for authority even if not selected for prompt.

### C. Evidence selector acceptance

Required before live prompt promotion:

1. Selector is a ranker/packer, not an authority filter: `selector_authority_filter_violations=0`.
2. Candidate EE/loading/result signal must be represented in selected evidence when present, or the row is held with `selected_evidence_missing_ee_or_loading_signal`.
3. Candidate preparation/material core must be represented in selected evidence when present, or the row is held with `missing_preparation_core` / `evidence_selection_missing_preparation_core`.
4. Selected evidence must not be dominated by release, PK, tissue, reference, license, docking, or unrelated assay tail noise.
5. Prompt preview must have no `oversized` rows and all promoted rows must have `s2_3_ready_overall=pass`.
6. Prompt summaries are semantic-facing only. Numeric completeness is not required in the prompt, but lossy summaries must have full-table authority backing.

### D. Batch-level live eligibility

A live manifest may include only rows where:

- `pre_llm_acceptance_status=pass_for_live_llm`
- no critical failure reasons are present;
- sidecar status is `loaded` / `consumed` / `explicitly_not_applicable`, not silently missing;
- exact dryrun run directory and input manifest are recorded.

## Repair work packages

### Task 1: Add a unified pre-LLM acceptance gate script

**Objective:** Materialize the above standard as a diagnostic script that can consume existing dryrun artifacts and emit pass/hold manifests plus first-failure reasons.

**Files:**
- Create: `src/stage2_sampling_labels/audit_pre_llm_acceptance_gate_v1.py`
- Create/modify test: `tests/test_pre_llm_acceptance_gate_v1.py`

**Verification:**

```bash
PYTHONPATH=. python3 -m unittest tests.test_pre_llm_acceptance_gate_v1
python3 -m py_compile src/stage2_sampling_labels/audit_pre_llm_acceptance_gate_v1.py
```

### Task 2: Extend acceptance gate to check unified Stage1 sidecar status

**Objective:** Require explicit Stage1 structure/table-cell sidecar status fields and classify silent missing sidecars as hold/review, while allowing explicit not-applicable states.

**Files:**
- Modify: `src/stage2_sampling_labels/audit_pre_llm_acceptance_gate_v1.py`
- Modify: `tests/test_pre_llm_acceptance_gate_v1.py`

**Verification:** same as Task 1.

### Task 3: Connect Stage2 manifests to unified structure/table-cell paths

**Objective:** Ensure targeted manifests or explicit roots can propagate `structure_path` and table-cell sidecar references for both HTML and PDF/Marker sources.

**Files to inspect/modify after tests:**
- `src/stage1_cleaning/build_run_input_contract_v1.py`
- `src/stage1_cleaning/derive_target_manifest_v1.py` if needed
- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- `src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py`
- relevant tests under `tests/test_run_input_contract_v1.py`, `tests/test_stage2_stage1_sidecar_consumption_v1.py`

### Task 4: Promote HTML-native DOM table-cell sidecars as first-class Stage1 authority

**Objective:** For HTML sources, freeze `extract_html_native_table_cells(...)` output into a stable sidecar root/index and allow Stage2 to consume it through the same table-cell interface used for Marker cells.

**Files to inspect/modify after tests:**
- `src/stage1_cleaning/pdf2clean.py`
- `src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py` or a maintained Stage1 helper if appropriate
- `tests/test_stage1_html_native_table_extraction_v1.py`
- `tests/test_stage1_cell_sidecar_v1.py`

### Task 5: Hydrate Marker/HTML improvements into the unique authoritative manifest

**Objective:** Make the Stage1 improvements part of the governed mainline data surface, not only diagnostic bakeoff outputs. Generate/freeze source-hash/parser-version-bound Stage1 sidecar assets for HTML-native DOM cells and promoted Marker/PDF table cells, then hydrate those asset bindings into the unique authoritative manifest:

- `data/cleaned/index/manifest_current.tsv`

`manifest_current.tsv` must remain the only corpus-level authority table. Sidecar inventories such as `stage1_table_cells_manifest_v1.tsv` are asset indexes only; they may feed hydration but must not become competing manifests.

**Required manifest fields after hydration:**
- `structure_path`
- `structure_available`
- `stage1_table_cell_sidecar_path`
- `stage1_table_cell_sidecar_available`
- source/provenance fields sufficient to distinguish HTML-native, Marker, and fallback parser surfaces without changing the downstream logical interface.

**Files to inspect/modify after tests:**
- `src/stage1_cleaning/hydrate_manifest_v1.py`
- `src/stage1_cleaning/build_run_input_contract_v1.py`
- `src/stage1_cleaning/run_stage1_parser_bakeoff_v1.py` or promoted maintained Stage1 helper
- `data/cleaned/index/manifest_current.tsv` only through governed hydration, not ad hoc editing
- relevant tests under `tests/test_run_input_contract_v1.py`, `tests/test_stage1_parser_bakeoff_v1.py`, and hydration-specific tests if present/added.

**Acceptance:** Stage2 scope manifests derived from `manifest_current.tsv` carry the hydrated HTML/Marker/PDF sidecar bindings without run-local synthetic overrides, and no-live Stage2 dryrun records explicit `stage1_structure_sidecar_status` and `stage1_cell_sidecar_status` values.

### Task 6: Run no-live smoke on regenerated Batch001 dryrun artifacts

**Objective:** After mainline hydration, regenerate the no-live Stage2 dryrun artifacts from `manifest_current.tsv`-derived scope manifests and generate a diagnostic acceptance report over Batch001 without spending LLM calls.

**Expected outputs:**
- run-scoped diagnostic outputs under a child `analysis/pre_llm_acceptance_gate_v1/`
- pass/hold manifests under `analysis/ee_modeling/` only after exact input lineage is recorded.
- a before/after delta showing whether first failures moved away from `stage1_structure_sidecar_missing` / `stage1_table_cell_sidecar_missing` toward more specific selector/clean-text issues.

## Progress ledger

Use `docs/plans/2026-05-07-pre-llm-cleantext-selector-unified-acceptance-progress.tsv`.

Columns: `task_id`, `status`, `test_status`, `notes`, `updated_at`.
