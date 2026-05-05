# Evidence Binding Pack and Risk Assessment Integration Plan

> **For Hermes:** Use subagent-driven-development only for read-only audits or isolated review. Keep writes serial in main/default. Follow strict PLGA governance and use maintained entrypoints only.

**Goal:** Add a post-Stage5 audit sidecar that explains each frozen final-table row/value assignment chain, then build a separate risk assessment from that sidecar without changing rows or values.

**Architecture:** Evidence Binding Pack is downstream of the frozen Stage5 final table. It consumes authority-resolved Stage5, Stage3, Stage2, table/text provenance and optional S5-3/S5-4/S5-5/Layer3 compare sidecars. It never creates rows, creates values, replaces the final table, or upgrades value-only matches into supporting evidence. Risk assessment is a separate consumer of frozen packs, and workbook rendering is a later display layer.

**Tech Stack:** Python 3 standard library TSV/JSONL tooling, existing `src.utils.active_data_source` resolver, existing Stage5 workbook/evidence handoff validator, `unittest`.

---

## Revised Scope and Corrections to the Draft

The draft direction is correct. The implementation must narrow a few points to avoid governance drift:

1. **Start with authority resolution, not binding logic.** Existing `ACTIVE_RUN.json` already contains conflicting legacy/top-level aliases. A strict gate must detect semantic alias conflicts and require either canonical `authoritative_terminal_files` keys or an explicit `--authority-field` override.
2. **Do not create another evidence contract root.** The binding contract is a downstream extension of `docs/methods/layer3_field_gt_protocol_v1.md` and the existing Layer3 evidence handoff validator/golden cases.
3. **Do not mix pack building and risk rules.** The pack builder records factual assignment chains and status taxonomy only. Risk assessment consumes frozen packs and emits risk levels/reasons.
4. **Do not force workbook migration until pack/risk surfaces are validated.** The workbook should later fail loudly when pack/risk paths are missing, except in explicit `--legacy-evidence-mode`.
5. **Diagnostic validation measures status distributions, not GT improvement.** This sidecar is not a benchmark-valid output and should not be interpreted as improving recall.

---

## Phase 0: Authority Resolution Gate

**Objective:** Create a strict audit/gate script that resolves and freezes all required inputs before evidence binding.

**Files:**
- Create: `src/stage5_benchmark/resolve_evidence_binding_authority_v1.py`
- Create/modify tests: `tests/test_evidence_binding_authority_v1.py`

**Required resolved artifacts:**
- `stage5_final_table_tsv`
- `stage5_decision_trace_tsv`
- `stage3_relation_records_tsv`
- `resolved_relation_fields_tsv`
- `stage2_compatibility_tsv`
- `stage2_projection_trace_tsv` when present/required by selected mode
- cleaned text root / `key2txt.tsv`
- Stage2 table authority root
- `layer3_compare_cells_tsv`
- `layer3_risk_review_queue_tsv`

**Rules:**
- Prefer `authoritative_terminal_files` exact keys.
- Do not use latest-by-sort, glob-first, parent fallback, or mtime.
- If alias group members point to different paths, fail unless the user supplies an explicit `--authority-field semantic_name=pointer_key` override.
- Print and write the resolved run directory plus exact source paths.
- Write only under a run-scoped child directory and include `RUN_CONTEXT.md`.

**Validation:**
- Unit test: no conflict -> manifest rows emitted.
- Unit test: conflicting aliases -> fail with `authority_alias_conflict`.
- Unit test: explicit override -> selected canonical path recorded with conflict warning.

---

## Phase 1: Run-Scoped Contract Audit

**Objective:** Inspect current evidence consumers and prove actual behavior, not design intent.

**Files:**
- Extend Phase 0 script or create after Phase 0: `src/stage5_benchmark/audit_evidence_binding_contract_v1.py`
- Output under child run dir:
  - `analysis/evidence_binding_contract_audit_v1.tsv`
  - `analysis/evidence_binding_contract_audit_v1.md`

**Audit dimensions:**
- Layer3 workbook seed rows.
- Audit-ready export.
- Layer3 risk queue / compare queue.
- Existing evidence handoff validator and golden cases.
- Classify evidence surface as `direct_evidence`, `broad_anchor`, `stage3_relation_provenance`, `legacy_fallback`, or `not_consumed`.

**Validation:**
- A contract audit run on current ACTIVE_RUN must print all resolved paths and write `RUN_CONTEXT.md`.
- It must label output `diagnostic-only, not benchmark-valid final output`.

---

## Phase 2: Contract Integration

**Objective:** Integrate methodology docs without creating a competing authority contract.

**Files:**
- Append to: `docs/methods/layer3_field_gt_protocol_v1.md`
- Optional only if necessary: `docs/methods/evidence_binding_contract_v1.md`

**Rules:**
- If a separate contract doc is created, its first paragraph must say it is a downstream extension of Layer3 evidence handoff and must cite:
  - `docs/methods/layer3_field_gt_protocol_v1.md`
  - `docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv`
  - `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py`

---

## Phase 3: Binding Pack Builder

**Objective:** Build factual row/field binding packs from frozen inputs.

**Files:**
- Create: `src/stage5_benchmark/build_evidence_binding_packs_v1.py`
- Create/modify tests: `tests/test_evidence_binding_packs_v1.py`

**Inputs:**
- Required: authority manifest from Phase 0 or equivalent explicit paths.
- Required resolved artifacts: final table, decision trace, Stage3 relation records, resolved relation fields, Stage2 compatibility TSV, projection trace if available, Stage2 table/evidence authority, cleaned text/evidence blocks.
- Optional: S5-3 candidates, S5-4 accepted/rejected/review sidecars, S5-5 derived provenance sidecars, Layer3 compare cells/risk queue.

**Outputs:**
- `evidence_binding_packs_v1.jsonl`
- `evidence_binding_field_summary_v1.tsv`
- `evidence_binding_formulation_summary_v1.tsv`
- metadata JSON and `RUN_CONTEXT.md`

**Non-goals:**
- No risk levels.
- No workbook rendering.
- No row/value creation.
- No upgrade of `value_only_match` into support.

---

## Phase 4: Status Taxonomy

**Objective:** Freeze the pack status vocabulary before risk rules depend on it.

**Required statuses:**
- `direct_supported`
- `relation_supported`
- `derived_supported`
- `identity_only_match`
- `value_only_match`
- `ambiguous_assignment`
- `unresolved_table`
- `unsupported_text`
- `blank_value`
- `conflict`
- `normalization_pending`
- `relation_path_missing`
- `missing_evidence_anchor`
- `derived_without_direct_text`
- `raw_value_supported_normalization_pending`
- `coded_value_supported_decode_pending`
- `role_tolerant_supported`
- `source_surface_missing`
- `authority_alias_conflict`

**Rule:** DOE coded factors that have row/table support but unresolved decode must use `coded_value_supported_decode_pending`, not generic `normalization_pending`.

---

## Phase 5: Relation-Visible Assignment Path

**Objective:** Every field pack must expose how the final value was assigned.

**Assignment paths:**
- `direct_same_table_row`
- `stage3_relation_resolved`
- `parent_inheritance`
- `shared_method_context`
- `selection_marker`
- `doe_factor_decode`
- `sequential_optimization_link`
- `derived_from_row_values`

**Hard rule:** If the final value came from Stage3 resolved fields but the pack cannot return the relation path, status must be `relation_path_missing`, not supported.

---

## Phase 6: Independent Risk Assessment

**Objective:** Build risk outputs from frozen packs only.

**Files:**
- Create: `src/stage5_benchmark/build_evidence_binding_risk_assessment_v1.py`
- Tests: `tests/test_evidence_binding_risk_assessment_v1.py`

**Outputs:**
- `evidence_binding_field_risk_v1.tsv`
- `evidence_binding_formulation_risk_v1.tsv`
- `evidence_binding_paper_risk_v1.tsv`

**Rules:**
- Consume only `evidence_binding_packs_v1.jsonl` plus optional compare/risk queue inputs.
- Align risk vocabulary with existing `run_layer3_cross_audit_v1.py` where possible.
- Do not re-resolve evidence from source files.

---

## Phase 7: Workbook Integration With Explicit Mode

**Objective:** Make workbook rendering consume validated packs/risk outputs.

**Files:**
- Modify: `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

**Rules:**
- Default mode requires `evidence_binding_packs_v1.jsonl` and risk TSVs.
- Missing pack/risk output fails loudly.
- Add `--legacy-evidence-mode` for old behavior.
- Workbook metadata must record:
  - `evidence_binding_pack_path`
  - `evidence_binding_risk_path`
  - authority-resolved final table path

---

## Phase 8: Golden Cases and Validator

**Objective:** Ensure evidence-binding work cannot regress existing evidence hardening.

**Files:**
- Extend: `docs/methods/layer3_evidence_handoff_golden_cases_v1.tsv`
- Modify: `src/stage5_benchmark/validate_layer3_evidence_contract_v1.py`

**Required retained behaviors:**
- broad row anchor suppression
- polymer grade not MW
- generic numeric mismatch rejection
- relation provenance visibility when direct anchor is missing

**Validation:**
- Validator must be able to read old golden cases and optional pack/risk expectations.

---

## Phase 9: Bounded Diagnostic Validation

**Objective:** Validate on representative papers without claiming benchmark validity.

**Paper set:**
- `UFXX9WXE`
- `WFDTQ4VX`
- `5GIF3D8W`
- `QLYKLPKT`
- `INMUTV7L`
- one ordinary direct-table paper selected by exact source manifest, not by recency

**Metrics:**
- Distribution by field and paper for:
  - `direct_supported`
  - `relation_supported`
  - `value_only_match`
  - `ambiguous_assignment`
  - `relation_path_missing`
  - `coded_value_supported_decode_pending`

**Output label:**
- `diagnostic-only, not benchmark-valid final output`.

---

## Phase 10: Governance Update

**Objective:** Register implemented surfaces only after they exist and pass validation.

**Files:**
- `docs/src_script_registry.tsv`
- `docs/maintained_script_surface.tsv`
- `project/ACTIVE_PIPELINE_RUNBOOK.md`
- `project/2_ARCHITECTURE.md`
- `README.md`
- `project/4_DECISIONS_LOG.md`

**Rules:**
- Do not create new files under `project/`.
- Register scripts as supporting diagnostic/audit surfaces until fully validated.
- Documentation must not imply runtime behavior before code and tests exist.

---

## First Implementation Slice

Start with Phase 0 only:
1. Write failing tests for authority alias conflict, override, and manifest emission.
2. Implement `resolve_evidence_binding_authority_v1.py` with strict pointer resolution.
3. Run targeted tests and py_compile.
4. Run the script against current ACTIVE_RUN into a new diagnostic child directory and inspect whether alias conflicts are correctly reported.
5. Only after Phase 0 is stable, proceed to Phase 1 audit.
