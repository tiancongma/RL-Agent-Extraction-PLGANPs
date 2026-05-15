# Mainline Pipeline Execution Checklist

Status: execution/audit checklist, not a governance authority document
Scope starts at the existing canonical manifest plus local PDF/HTML assets. Raw Zotero intake and manifest creation before `data/cleaned/index/manifest_current.tsv` are intentionally out of scope for this checklist version.
Current phase: diagnosis-baseline extraction. GT comparison remains a diagnosis tool, not a separate execution stage.

## 1. Purpose

This document defines a scriptable, run-bound acceptance checklist for PLGA mainline pipeline executions. It is used after selecting a target run or candidate lineage to verify that:

- the run consumed the unique canonical manifest and an explicit scope selection;
- local PDF/HTML/clean-text/table assets were resolved through governed path logic;
- the Python/Marker/HTML/LLM execution environment was sufficient and recorded without leaking secrets;
- table CSV and normalized table payloads were consumable enough for downstream deterministic stages;
- S2-2 preserved full table authority while S2-4a exposed only compact structural table summaries;
- S2-4b model/backend/request configuration was explicit and reproducible;
- S2-5/S2-6/S2-7 preserved semantic authority and reattached deterministic table/value authority without asking the LLM to carry numeric tables;
- Stage3 and Stage5 consumed lawful upstream boundaries and did not recreate formulation membership outside the authorized scope;
- diagnostic compare / ACTIVE_RUN pointer checks, when applicable, remain diagnosis-baseline checks only.

## 2. Target-run binding rule

Every checklist execution must bind to one explicit target:

1. an explicit run directory, or
2. the repository authority pointer in `data/results/ACTIVE_RUN.json`.

The checklist executor must not choose a target by latest directory, modification time, lexical sorting, parent fallback, glob-first matching, or unstated memory.

A checklist result must record the target-run identity and the run parameters, or exact pointers to the run parameters. At minimum the result must include:

- `checklist_target_run_id`
- `checklist_target_run_dir`
- `authority_resolution_mode` (`explicit_path` or `ACTIVE_RUN.json`)
- selected scope artifact(s) and selection parameters/tags
- every consumed input file path
- every executed script path
- every script's relevant CLI arguments/configuration
- environment descriptor path(s)
- env file path(s) used for secret import, without secret values
- S2-4b/S5-3 model/backend/request parameters when live LLM calls are involved
- output run directories and terminal artifact paths
- downstream diagnostic compare paths, if run

Recommended checklist result outputs:

```text
data/results/<run_bucket>/<audit_child>/mainline_execution_checklist_results_v1.tsv
data/results/<run_bucket>/<audit_child>/mainline_execution_checklist_summary_v1.json
data/results/<run_bucket>/<audit_child>/RUN_CONTEXT.md
```

Recommended result TSV columns:

```text
check_id	stage	gate_level	status	target_run_dir	input_paths	output_paths	script_paths	parameter_refs	evidence_paths	failure_boundary	notes
```

Allowed `gate_level` values:

- `hard_gate`: failure blocks continuation or diagnosis-baseline pointer use for the checked target.
- `soft_audit`: failure does not block by itself but must be recorded.
- `warning`: execution may continue only with a clear limitation label.
- `record_only`: implementation is planned or optional; absence is recorded but does not block this checklist version.

Allowed `status` values:

- `pass`
- `fail`
- `warning`
- `not_applicable`
- `record_only_missing`
- `not_checked`

## 3. Global hard prohibitions

These prohibitions apply to all checklist stages:

- Do not infer authority from latest path, mtime, folder name, or glob order.
- Do not treat a run-scoped subset as a competing canonical manifest.
- Do not hard-code or record API keys, tokens, or secret values. Record only env file paths and secret-presence booleans.
- Do not claim complete PDF extraction if Marker was not available and executed or if frozen Marker artifacts were not consumed.
- Do not let Stage2 pre-LLM rules decide semantic table importance, formulation membership, or table role.
- Do not put full numeric table dumps into S2-4a prompts as execution authority.
- Do not treat S2-4a table summaries as numeric/materialization authority.
- Do not require the LLM to transmit full numeric table contents, table-cell coordinates, or execution locators.
- Do not allow S2-7 to expand rows from LLM memory alone; expansion must return to S2-2 full table/value authority through deterministic sidecars or alias-equivalent locators.
- Do not let Stage3 rediscover chemistry from prose.
- Do not let Stage5 create/split/merge formulation rows unless an explicit governed identity rule records and justifies it.
- Do not use S5-3 as database completion or blank-slot filling.
- Do not use this checklist to certify outputs. This checklist is diagnosis/execution/audit only.

## 4. Threshold defaults

Until replaced by stricter stage-specific thresholds, any key table-consumption metric below `50%` over the selected run scope is a hard gate failure unless the run explicitly declares the missing source type as out of scope.

Key table-consumption metrics include:

- resolved source asset coverage;
- parseable CSV/table payload coverage;
- non-empty table coverage;
- header-presence coverage;
- first-column or row-identity signal coverage for table-like assets;
- source table/caption/title locator coverage;
- normalized payload to full-table authority backlink coverage;
- Stage2 evidence summary to full authority pointer coverage;
- S2-7 projected rows with applicable table/payload/cell locator coverage.

The checklist result must record numerator, denominator, percentage, threshold, and failed paper/table identifiers for every thresholded metric.

## 5. Checklist items

### S1-0 Stage1 preflight, authority, and execution environment

#### CHECK-S1-0-001: Active source and parameter-source resolution

- Gate: `hard_gate`
- Purpose: prevent silent use of the wrong source run, parameter source, or Stage1 scope.
- Required records:
  - `active_source_resolution_method`
  - `parameter_source_run`
  - `parameter_source_run_context`
  - `parameter_source_files`
  - `scope_source_manifest`
  - `ACTIVE_RUN_snapshot_path` when `ACTIVE_RUN.json` is consulted
  - explicit CLI/user-provided run path when used
  - resolved run bucket and child run
- Pass criteria:
  - Source and parameter authority are resolved only from an explicit user/CLI path or `data/results/ACTIVE_RUN.json`.
  - The result records exact source files and parameter-reference files.
  - No source is selected by latest directory, lexical sort order, modification time, parent fallback, glob-first matching, or unstated memory.
- Failure boundary: `active_source_resolution_failure`

#### CHECK-S1-0-002: Maintained Stage1 entrypoint selection

- Gate: `hard_gate`
- Purpose: prevent legacy/diagnostic scripts from standing in for maintained Stage1 execution.
- Required records:
  - selected entrypoint script(s)
  - status/class from `project/PIPELINE_SCRIPT_MAP.md` and/or `docs/maintained_script_surface.tsv`
  - runbook reference
  - script role and declared inputs/outputs
- Pass criteria:
  - Execution-facing Stage1 work uses maintained active/supporting entrypoints.
  - Diagnostic, wrapper-only, legacy, deprecated, or archive scripts are not selected unless explicitly requested by the user.
- Failure boundary: `unmaintained_stage1_entrypoint_selection`

#### CHECK-S1-0-003: Runtime environment executable preflight

- Gate: `hard_gate`
- Purpose: prevent the prior failure mode where an agent silently skipped work because Python/Marker/dependencies were absent or incorrectly assumed unusable.
- Required records:
  - attempted Python commands (`python`, `python3`, or explicit venv path)
  - resolved Python executable and version
  - platform and working directory
  - venv/conda/uv environment path when applicable
  - package import smoke-test results and exit codes
  - command exit codes for required executables
- Minimum package smoke checks when in scope:
  - `pandas`
  - `fitz` / PyMuPDF when PDF text paths are inspected or generated
  - `bs4` and `lxml` when HTML path inspection or HTML-sidecar reading is in scope
  - `trafilatura` when complete HTML enhancement/extraction is requested or claimed; the smoke test must actually `import trafilatura`, not merely locate the package spec
  - `lxml_html_clean` or equivalent `lxml[html-clean]` support when `trafilatura` is requested or claimed, because modern `lxml` can install HTML-cleaning support as a separate package
  - HTML enhancement dependency failures are a hard gate when complete HTML enhancement is requested or claimed; otherwise they are recorded HTML-readiness warnings and the run must label any BeautifulSoup-only fallback as partial HTML enhancement
  - `openpyxl` when workbook/audit surfaces are involved
  - `unittest` and only `pytest` when the checked workflow requires it
- Pass criteria:
  - At least one Python executable is actually runnable and recorded.
  - Required imports/commands for the declared run stages are actually executed as smoke tests.
  - Optional enhancement dependencies that are not required by the declared Stage1 mode may fail only if the run records the limitation and does not claim that enhancement as complete.
  - If `python` is unavailable but `python3` is available, record that distinction and use the resolved executable; do not fail or assume silently.
- Failure boundary: `runtime_environment_preflight_failure`

#### CHECK-S1-0-004: Marker executable preflight if Marker enhancement is requested

- Gate: `hard_gate` when Marker enhancement is requested or claimed; otherwise `not_applicable`
- Purpose: Marker is additive enhancement, but requested/claimed Marker work must be backed by a real executable preflight.
- Required records:
  - `marker_requested`
  - Marker executable path, normally `.venv_marker/bin/marker_single` or an explicitly declared equivalent
  - Marker venv path
  - Marker help/version/smoke command and exit code
  - Marker timeout policy
  - Marker availability status
- Pass criteria:
  - If the run claims Marker PDF enhancement, the Marker executable must exist and its smoke command must pass before execution or frozen Marker artifacts must be explicitly consumed.
  - If Marker is unavailable, the run must stop or record `compatibility_text_only_user_approved` after explicit user approval; it must not silently downgrade.
- Failure boundary: `marker_preflight_failure`

### S1-1 Canonical manifest and scope contract

#### CHECK-S1-1-001: Canonical manifest uniqueness

- Gate: `hard_gate`
- Inputs: `data/cleaned/index/manifest_current.tsv`
- Required records:
  - canonical manifest path
  - row count and schema columns
  - checksum or immutable snapshot pointer
  - manifest selection reason
- Pass criteria:
  - `manifest_current.tsv` exists and remains the unique canonical manifest for this checklist scope.
  - Run-scoped subsets are recorded as scope artifacts or selection outputs, not as alternative canonical manifests.
  - Archived/bad manifests are not used.
- Failure boundary: `S1_manifest_scope_contract_failure`

#### CHECK-S1-1-002: Scope selection reproducibility

- Gate: `hard_gate`
- Inputs: canonical manifest plus run scope selection fields.
- Required evidence:
  - `run_input_scope_v1.tsv`, `run_input_manifest_snapshot_v1.tsv`, `run_input_contract_v1.json`, or alias-equivalent artifacts.
  - `RUN_CONTEXT.md` fields recording selection reason and scope parameters.
- Pass criteria:
  - Every paper processed downstream can be traced to exactly one canonical manifest row.
  - Every row records why it entered scope.
  - `paper_key`, DOI/title identity, and source provenance are preserved.
  - Scope is not produced by directory scanning or implicit latest/glob logic.
- Failure boundary: `S1_scope_selection_unreproducible`

#### CHECK-S1-1-003: Manifest row path health and bad-manifest exclusion

- Gate: `hard_gate` for canonical input resolution; row-level missing assets are `warning`
- Required per-row records:
  - `paper_key`, DOI/title identity
  - manifest PDF/HTML/text path fields when present
  - resolved PDF/HTML/text paths
  - file existence, file size, and readability status
  - missing/unreadable reason
- Pass criteria:
  - The canonical manifest itself is not an archived/bad manifest.
  - Row-level missing/unreadable assets are explicitly recorded and are not silently dropped.
  - Path health checks are resolved through governed Stage1 resolver logic rather than raw ad hoc path tests.
- Failure boundary: `manifest_path_health_failure`

### S1-2 Source asset path resolution, hydration, fusion, and table-sidecar readiness

#### CHECK-S1-2-001: PDF/HTML path resolver compliance

- Gate: `hard_gate`
- Inputs:
  - `manifest_current.tsv`
  - PDF/HTML attachment references
  - `.local_stage1_paths.json` or equivalent local path mapping, if used
- Scriptable check:
  - Resolve manifest attachment paths with the maintained Stage1 resolver, not raw `os.path.exists` on stored strings.
  - Record original manifest path and resolved local path.
- Pass criteria:
  - Each in-scope row has at least one declared source asset path or a loud missing-source status.
  - Resolved existing PDF/HTML coverage is recorded by source type.
  - Missing source assets do not silently fall back to unrelated files.
- Failure boundary: `S1_source_path_resolution_failure`

#### CHECK-S1-2-002: PDF/HTML/current clean-text hydration

- Gate: `warning` for row-level missing/failed source assets; `hard_gate` for silent fallback or missing hydration surface
- Required fields or alias-equivalent records:
  - `paper_key`
  - current clean text path and length/status
  - PDF path/status
  - HTML path/status
  - source/dataset lineage fields
  - missing reason fields
- Pass criteria:
  - Stage1 records per-row current text/PDF/HTML availability.
  - Missing current text, PDF, or HTML is allowed as `warning` when explicitly recorded.
  - Missing rows must be handled by the Stage2 admission filter before Stage2 execution.
  - Stage2 is not allowed to discover text paths by ad hoc directory search or stale `key2txt.tsv` fallback.
- Failure boundary: `S1_asset_hydration_silent_fallback`

#### CHECK-S1-2-003: Multi-source selection/fusion declaration

- Gate: `hard_gate`
- Applies when PDF and HTML, or current text plus Marker/fusion sidecars, are both available.
- Required records:
  - fusion strategy
  - current text role
  - Marker text/structure role
  - HTML text/table role
  - table sidecar role
  - source priority and append/replace policy
- Pass criteria:
  - The run records which source view was consumed or produced for Stage2.
  - Current clean text remains the compatibility base unless explicitly superseded by governed policy.
  - Marker and HTML sources are preserved additively where available.
  - Fusion does not overwrite raw/current clean text.
- Failure boundary: `S1_source_fusion_undeclared`

#### CHECK-S1-2-004: Local source resolver audit

- Gate: `hard_gate` for resolver compliance; row-level unresolved assets are `warning`
- Purpose: cover recurring local Zotero and dual-source path-mapping failures.
- Required records:
  - raw Zotero/manifest PDF and HTML paths
  - resolved local PDF and HTML paths
  - resolver decision/status
  - dual-source availability and priority/fusion decision
- Pass criteria:
  - Local path remapping is explicit and auditable.
  - PDF/HTML dual-source availability is recorded.
  - Scripts do not modify the canonical manifest as an implicit path-repair side effect.
- Failure boundary: `local_source_resolver_audit_failure`

#### CHECK-S1-2-005: Table sidecar and table-directory discoverability

- Gate: `warning` at Stage1; `hard_gate` if Stage2 table-authority mode is declared for the checked handoff
- Required per-row records:
  - table directory/path
  - table directory existence
  - table file count
  - HTML table-sidecar count
  - Marker table artifact count when available
  - parseable/unparseable table counts when checked
  - table asset status
- Pass criteria:
  - Table sidecar absence is recorded as a limitation, not silently hidden.
  - If Stage2 declares table-authority consumption, the corresponding Stage1 or S2-2 authority path must exist or be explicitly unavailable before Stage2.
- Failure boundary: `table_asset_discoverability_failure`

### S1-3 Marker, HTML, and source-coverage recording

#### CHECK-S1-3-001: Marker PDF extraction execution record

- Gate: `warning` for per-file Marker failure/timeout; `hard_gate` when Marker execution is claimed without evidence
- Required records:
  - Marker requested/consumed status
  - PDF scope count
  - Marker attempted/success/failed/timeout/skipped counts
  - Marker output paths and error/log paths
  - timeout seconds and config hash where available
- Pass criteria:
  - If Marker is requested, attempted rows and failed/timeout rows are explicitly recorded.
  - Marker failure/timeout does not block Stage1 by itself under the current policy; it is a recorded warning and must propagate to Stage2 admission metadata.
  - A run does not claim Marker enhancement without output/metadata evidence.
- Failure boundary: `marker_execution_claim_without_evidence`

#### CHECK-S1-3-002: Marker/source coverage label truthfulness

- Gate: `hard_gate`
- Allowed labels include:
  - `compatibility_text_only`
  - `compatibility_text_only_user_approved`
  - `current_clean_text_plus_marker_subset`
  - `marker_attempted_all_pdfs`
  - `marker_successful_subset`
  - `marker_enhanced_subset`
  - `complete_marker_pdf_coverage`
- Pass criteria:
  - Source-coverage labels match actual attempted/success/failure counts.
  - If Marker success is below the PDF scope denominator, the run must not claim complete Marker PDF coverage.
  - If Marker was not executed or frozen Marker artifacts were not consumed, the run must not claim Marker-enhanced PDF coverage.
- Failure boundary: `source_coverage_label_misrepresentation`

#### CHECK-S1-3-003: HTML enhancement readiness and coverage

- Gate: `warning` unless complete HTML enhancement is explicitly required; `hard_gate` for silent fallback or false coverage claims
- Required records:
  - HTML scope count
  - HTML available/resolved/parsed counts
  - HTML parse-failed/missing counts
  - HTML table-sidecar count
  - HTML text-output count
  - HTML structure-sidecar count
  - parser outcome counts split by `trafilatura_native`, `trafilatura_plus_beautifulsoup_supplement`, `beautifulsoup_fallback`, and parse failure
  - exact dependency smoke-test results for `trafilatura`, `lxml_html_clean` / `lxml[html-clean]`, `bs4`, and `lxml`
  - missing/failure reasons, including import errors and fallback reasons
- Pass criteria:
  - HTML parser dependencies are checked by real imports when HTML is in scope.
  - If complete HTML enhancement is requested or claimed, `trafilatura` plus its HTML-cleaning dependency must import successfully, and fallback-only parser outcomes must not be reported as complete HTML enhancement.
  - HTML contribution is visible in the hydrated/unified surface or explicitly marked unavailable.
  - There is no silent fallback from declared HTML-enhanced paths to stale text paths.
  - Parser outcome distributions and fallback reasons are recorded so a run can distinguish source-quality limitations from environment/dependency failures.
- Failure boundary: `html_enhancement_unrecorded_or_silent_fallback`

#### CHECK-S1-3-004: Source coverage summary

- Gate: `hard_gate` for summary presence; row-level source failures are `warning`
- Required summary metrics:
  - total scope rows
  - current text available/missing
  - PDF available/missing
  - Marker attempted/success/failed/timeout/skipped
  - HTML available/parsed/failed/missing
  - table assets available/missing
  - Stage2 admissible/excluded counts when the Stage2 admission filter is prepared
  - warning and hard-gate failure counts
- Pass criteria:
  - Denominators are recorded for every source type in scope.
  - Failed/missing rows are not silently excluded from summaries.
- Failure boundary: `stage1_coverage_summary_missing`

#### CHECK-S1-3-005: LLM environment and parameter recording

- Gate: `hard_gate` for live LLM runs; `not_applicable` for no-live Stage1 runs.
- Required records:
  - backend/provider name
  - exact model name
  - response format, max tokens, timeout, retries, max parallel requests, inter-request sleep, temperature, streaming mode where applicable
  - env file path used for secrets, if any
  - secret presence booleans, never secret values
- Pass criteria:
  - Live-call parameters are recorded at the owning live-call boundary.
  - API keys or secret values are not written to run artifacts.
- Failure boundary: `llm_parameter_or_secret_recording_failure`

### S1-4 Stage1 run reproducibility and output governance

#### CHECK-S1-4-001: Run reproducibility specification

- Gate: `hard_gate`
- Required evidence: stage-local `RUN_CONTEXT.md` or alias-equivalent run specification.
- Required records:
  - run purpose and run type
  - exact starting inputs
  - exact script execution order and script paths
  - parameters
  - intermediate artifacts
  - final outputs
  - declared status and intended downstream use
- Pass criteria:
  - Stage1 run directories are documented with reproducibility-grade context.
  - No undocumented Stage1 result directory is left as a downstream candidate.
- Failure boundary: `stage1_run_context_missing_or_incomplete`

#### CHECK-S1-4-002: Environment descriptor artifact

- Gate: `hard_gate`
- Recommended artifact: `analysis/stage1_environment_descriptor_v1.json` or audit-child alias.
- Required content:
  - Python executable/version/platform/working directory
  - venv/environment identity
  - package import checks and exit codes
  - Marker executable path and smoke command result when Marker is requested/claimed
  - dependency check timestamp
  - no secret values
- Pass criteria:
  - Environment checks are persisted as a run artifact, not only described in conversation.
- Failure boundary: `stage1_environment_descriptor_missing`

#### CHECK-S1-4-003: No overwrite / governed output path compliance

- Gate: `hard_gate`
- Pass criteria:
  - Stage1 run outputs remain under governed `data/results/...` child paths or approved reusable `data/cleaned/...` Stage1 asset paths.
  - The run does not overwrite `data/cleaned/index/manifest_current.tsv` unless explicitly performing governed manifest construction.
  - New project governance documents are not created by the checklist/audit.
  - Run outputs do not masquerade as canonical source indexes.
- Failure boundary: `stage1_output_path_governance_failure`

### S1-S2 Stage1-to-Stage2 admission and handoff checks

These checks may be materialized by the Stage1 audit, but they are enforced before Stage2 execution. They exist so Stage1 warnings are not lost when Stage2 begins.

#### CHECK-S1-S2-001: Refreshed Stage2 text/table binding from current Stage1 authority

- Gate: `hard_gate` before Stage2
- Required per-row records:
  - `paper_key`
  - Stage1 unified/source row id
  - Stage1 text path
  - Stage1 table authority/sidecar path
  - Stage2 text path
  - Stage2 table path
  - binding source/status and refresh timestamp
- Pass criteria:
  - Stage2 scope is generated from the current Stage1 authority surface, not copied unchanged from stale `targeted_manifest.tsv`, stale `key2txt.tsv`, or old table paths.
  - Every Stage2 input path traces back to the Stage1 output selected for this lineage.
- Failure boundary: `stale_stage1_asset_binding_in_stage2_scope`

#### CHECK-S1-S2-002: Explicit Stage2 admission scope filter

- Gate: `hard_gate` before Stage2
- Required artifact: `stage2_admission_scope_v1.tsv` or alias-equivalent.
- Required fields:
  - `paper_key`, DOI/title when available
  - `has_current_text`, `has_marker_text`, `marker_status`
  - `has_html`, `html_status`
  - `has_table_sidecar`
  - Stage1 warning flags
  - `stage2_admission_decision`
  - `stage2_admission_reason`
  - `text_path_for_stage2`
  - `table_authority_path_for_stage2`
  - `source_limitation_label`
- Allowed decisions:
  - `admit_full_source`
  - `admit_current_text_only`
  - `admit_with_marker_enhancement`
  - `admit_with_html_enhancement`
  - `admit_with_source_limitations`
  - `exclude_missing_text`
  - `exclude_unreadable_source`
  - `exclude_policy_out_of_scope`
- Pass criteria:
  - Stage2 does not blindly consume the entire Stage1 full scope.
  - Marker/HTML warnings propagate as limitation labels.
  - Missing text rows are excluded unless another lawful text source is explicitly selected.
- Failure boundary: `stage2_admission_filter_missing`

#### CHECK-S1-S2-003: Text window and truncation policy declaration

- Gate: `hard_gate` before live Stage2
- Required records:
  - source text path
  - raw text length
  - `max_chars`
  - section-aware vs front-slice/windowing mode
  - whether truncation was applied
  - post-window text length
  - truncation reason
- Pass criteria:
  - The actual Stage2 LLM-visible source window is declared before any live LLM call.
  - Truncated text is not described as complete source coverage.
- Failure boundary: `stage2_text_window_policy_missing`

#### CHECK-S1-S2-004: Raw source vs denoised projection separation

- Gate: `hard_gate` if S2-1b denoise projection is used
- Required records:
  - raw/current source text path
  - denoised projection path
  - denoise rule set
  - deleted-span count and audit path
  - Stage2 consumed text path
  - raw source preservation status
- Pass criteria:
  - Raw/current clean text remains preserved as audit authority.
  - Denoised projection is Stage2-internal and is not named `clean_text`, `source_text`, or `final_text`.
  - Denoising deletes only high-confidence noise and does not perform semantic formulation discovery.
- Failure boundary: `raw_source_denoised_projection_conflation`

#### CHECK-S1-S2-005: Stage2 source limitation propagation

- Gate: `hard_gate` before Stage2 summary or downstream claims
- Required propagated flags:
  - `missing_current_text`
  - `marker_failed`
  - `marker_timeout`
  - `marker_not_requested`
  - `html_missing`
  - `html_parse_failed`
  - `table_sidecar_missing`
  - `text_only_compatibility_mode`
  - `partial_marker_enhancement`
  - `partial_html_enhancement`
- Pass criteria:
  - Stage2 scope and `RUN_CONTEXT.md` preserve Stage1 limitation labels.
  - Downstream reporting does not turn a partial source run into a complete-source claim.
- Failure boundary: `stage1_limitation_not_propagated_to_stage2`

#### CHECK-S1-S2-006: Stage2 prompt/input source contract preview

- Gate: `hard_gate` before live Stage2
- Required artifact: `stage2_input_source_preview_v1.tsv` or alias-equivalent.
- Required fields:
  - `paper_key`
  - text path and text length
  - table authority path
  - Marker/HTML enhancement used flags
  - source limitation label
  - prompt/evidence-block source
  - ready-for-Stage2-live boolean
  - blocking reason
- Pass criteria:
  - Every paper's Stage2 input source is inspectable before live calls.
  - No unknown, stale, or silently defaulted source path enters Stage2.
- Failure boundary: `stage2_input_source_preview_missing`

#### CHECK-S1-S2-007: Enhanced structure downstream-consumption proof

- Gate: `hard_gate` before claiming Marker/HTML structure enhancement was consumed by downstream stages; `warning` when only Stage1 asset readiness is being audited.
- Purpose: distinguish asset existence from actual downstream use. Stage1 `structure_path`, Marker fusion sidecars, HTML structure sidecars, and table-cell sidecars are not by themselves proof that Stage2, Stage3, or Stage5 consumed the enhanced structure.
- Required artifacts when Stage2 is run or downstream consumption is claimed:
  - the exact Stage1 hydrated/unified manifest surface selected for Stage2
  - `stage2_admission_scope_v1.tsv` or alias-equivalent with `has_marker_text`, `marker_status`, `has_html`, `html_status`, `has_table_sidecar`, `stage2_admission_decision`, `source_limitation_label`, `text_path_for_stage2`, and `table_authority_path_for_stage2`
  - `stage2_input_source_preview_v1.tsv` or alias-equivalent before live calls
  - `analysis/stage1_source_consumption_audit_v1.tsv`
  - `analysis/stage1_source_consumption_summary_v1.json`
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `analysis/table_authority_validation_v1.tsv`
  - `RUN_CONTEXT.md` recording the consumed Stage1 surface and limitation labels
- Required fields in consumption audit:
  - `paper_key`
  - `text_path`
  - `source_clean_text_type`
  - `marker_consumed`
  - `html_consumed`
  - `structure_path`
  - `stage1_structure_source`
  - `fallback_reason`
  - `stage1_table_cell_sidecar_path`
  - `stage1_table_cell_sidecar_consumed`
  - `stage1_table_cell_sidecar_available`
- Pass criteria:
  - Every claimed Marker/HTML-enhanced paper has an explicit Stage2-bound row whose text and structure paths trace back to the selected Stage1 authority surface.
  - `marker_consumed=yes` / `html_consumed=yes` are present only when the Stage2-bound text or structure surface actually uses those enhancements.
  - `stage1_table_cell_sidecar_consumed=yes` is backed by an existing sidecar path or an explicit unavailable/limitation label.
  - S2-2 full-table authority artifacts exist for admitted papers, and downstream row/table materialization resolves through `normalized_table_payloads_v1.json` / coordinate-grid authority rather than raw Stage1 CSVs or prompt summaries.
  - Stage3/Stage5 claims reference the completed Stage2 artifact as the lawful boundary, not Stage1 sidecars directly.
- Failure boundary: `enhanced_structure_downstream_consumption_unproven`

### S2-1 Scope resolution

#### CHECK-S2-1-001: Stage2 scope context persistence

- Gate: `hard_gate`
- Recommended output:
  - `semantic_stage2_objects/scope_context/<paper_key>/paper_scope_context_v1.json`
  - alias-equivalent scope context artifacts are acceptable.
- Pass criteria:
  - Each paper's Stage2-visible clean text, table assets, paper metadata, run lineage, and source lineage are recorded.
  - This step performs resolution only: no ranking, selection, summary, or semantic interpretation.
- Failure boundary: `S2_scope_context_missing_or_semanticized`

#### CHECK-S2-1-002: Stage2 pre-run frozen input contract

- Gate: `hard_gate`
- Pass criteria:
  - `paper_key`, `text_path`, table asset references, and source lineage are frozen before S2-2.
  - Stage2 does not refresh rows from unrelated global indexes in a way that discards the explicit unified/hydrated surface.
- Failure boundary: `S2_input_contract_drift`

### S2-2a Candidate segmentation and full-table authority preservation

#### CHECK-S2-2A-001: Candidate segmentation artifacts

- Gate: `hard_gate`
- Required outputs or alias-equivalent artifacts:
  - `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
  - `analysis/candidate_segmentation_debug_v1.tsv`
- Pass criteria:
  - Candidate text/table evidence is segmented with source references.
  - Only confirmed pure noise is irreversibly deleted.
  - No semantic table-importance veto or formulation membership decision happens here.
- Failure boundary: `S2_candidate_segmentation_contract_failure`

#### CHECK-S2-2A-002: Full-table authority preservation

- Gate: `hard_gate`
- Required outputs or alias-equivalent artifacts:
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
  - `semantic_stage2_objects/normalized_table_payloads/<paper_key>/payloads/*.csv`
  - `table_cell_grid_v1.tsv` or `table_cell_grid_v1.jsonl` or equivalent coordinate grid
  - `analysis/table_authority_validation_v1.tsv`
- Pass criteria:
  - Non-noise tables are preserved as execution-grade authority even if not selected for prompt.
  - Stable table IDs, row IDs, column/header information, source locators, and table-local references exist where recoverable.
  - If geometry/header recovery fails, the table is explicitly marked `unrecoverable` with a reason.
  - Key table-consumption metrics meet the default 50% threshold.
- Failure boundary: `S2_full_table_authority_loss`

#### CHECK-S2-2A-003: CSV/table payload consumability

- Gate: `hard_gate`
- Scriptable check:
  - Parse all in-scope CSV/payload files.
  - Compute parseable, non-empty, header-present, first-column-present, all-blank-row, all-blank-column, and locator coverage metrics.
- Pass criteria:
  - No file-level parsing failures are silent.
  - Critical coverage metrics are at least 50% unless source type is explicitly out of scope.
  - Large clusters of blank/error tables are listed with paper/table IDs.
- Failure boundary: `table_payload_consumability_failure`

### S2-2b Selector prioritization and evidence block construction

#### CHECK-S2-2B-001: Evidence block construction

- Gate: `hard_gate`
- Required output:
  - `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- Pass criteria:
  - Evidence blocks include materials, preparation, formulation, table-summary, result/characterization evidence where available.
  - Each selected prompt-facing evidence block can point back to full authority records where relevant.
  - The selector records selected-for-prompt vs preserved-for-authority status.
- Failure boundary: `S2_evidence_blocks_missing_or_untraceable`

#### CHECK-S2-2B-002: Selector authority limit

- Gate: `hard_gate`
- Pass criteria:
  - Selector acts as ranker/packer, not formulation-row authority.
  - Evidence blocks may be compact/lossy, but full preserved authority remains available downstream.
  - Feature activation, selection mode, technical status, and design status are recorded.
- Failure boundary: `S2_selector_semantic_overreach`

### S2-3 Prompt assembly

#### CHECK-S2-3-001: Prompt assembly input restriction

- Gate: `hard_gate`
- Inputs:
  - persisted `evidence_blocks_v1.json` only.
- Pass criteria:
  - Prompt assembly does not reread clean text.
  - Prompt assembly does not rerank/rescore evidence.
  - Prompt assembly does not use full numeric tables as execution authority.
- Failure boundary: `S2_prompt_assembly_input_drift`

#### CHECK-S2-3-002: Prompt preview artifact

- Gate: `hard_gate`
- Required output:
  - `analysis/stage2_prompt_preview_v1.tsv` or alias-equivalent prompt preview.
- Pass criteria:
  - Prompt preview is derived from persisted evidence blocks.
  - Preview records enough evidence to audit prompt visibility before live calls.
- Failure boundary: `S2_prompt_preview_missing`

### S2-4a Prompt construction freeze

#### CHECK-S2-4A-001: No-live prompt freeze boundary

- Gate: `hard_gate`
- Required outputs:
  - `analysis/s2_4a_prompt_template_v1.txt`
  - `analysis/s2_4a_prompts_v1.jsonl`
  - `analysis/s2_4a_prompt_audit_v1.tsv`
  - stage-local `RUN_CONTEXT.md`
- Pass criteria:
  - No live LLM call occurs in S2-4a.
  - Future live calls consume these frozen prompts or a new governed prompt-freeze lineage.
- Failure boundary: `S2_4a_live_call_or_missing_freeze`

#### CHECK-S2-4A-002: Hard table-summary contract

- Gate: `hard_gate`
- Pass criteria for each table summary exposed to S2-4a:
  - complete header representation is present;
  - first column / row-identity column is present when recoverable;
  - at most two sample data rows are present, excluding header rows;
  - if fewer than two sample rows exist, all available rows may be shown;
  - full numeric table dumps are absent;
  - table summary is structural prompt visibility only, not numeric authority.
- Failure boundary: `S2_4a_table_summary_contract_failure`

#### CHECK-S2-4A-003: No pre-LLM semantic labels

- Gate: `hard_gate`
- Pass criteria:
  - Pre-LLM artifacts may include structural/ranking/observability metadata.
  - They must not pre-label formulation semantic truth, final row membership, semantic table role, or LLM-like classification.
- Failure boundary: `S2_4a_pre_llm_semantic_leakage`

### S2-4b Live LLM semantic discovery

#### CHECK-S2-4B-001: Frozen prompt consumption

- Gate: `hard_gate`
- Inputs:
  - S2-4a frozen prompts.
- Pass criteria:
  - Live LLM runner consumes the exact governed S2-4a prompts.
  - If prompts differ, a new S2-4a lineage is recorded.
- Failure boundary: `S2_4b_prompt_lineage_mismatch`

#### CHECK-S2-4B-002: Raw response and request metadata persistence

- Gate: `hard_gate`
- Required outputs:
  - `raw_responses/<paper_key>__stage2_v2_raw_response.json`
  - `request_metadata/`
  - `analysis/s2_4b_request_summary_v1.tsv`
  - stage-local `RUN_CONTEXT.md`
- Pass criteria:
  - Each expected paper has success/failure status.
  - Model/backend/request parameters are recorded.
  - Secret values are not recorded.
  - LLM output is treated as semantic signal/authorization, not table/value authority.
- Failure boundary: `S2_4b_raw_response_or_metadata_failure`

### S2-5 Semantic parsing

#### CHECK-S2-5-001: Semantic parser fidelity

- Gate: `hard_gate`
- Required outputs:
  - `semantic_stage2_objects/semantic_stage2_v2_objects.jsonl`
  - `semantic_stage2_objects/semantic_stage2_v2_summary.tsv`
- Pass criteria:
  - Parser does not add new semantic decisions.
  - Parser preserves table scopes, shared semantics, relation cues, and evidence references present in raw responses.
  - If a signal exists in raw response but disappears in parsed objects, failure is assigned to S2-5.
- Failure boundary: `S2_5_semantic_signal_drop`

### S2-5b Semantic signal to S2-2 authority reattachment sidecar

#### CHECK-S2-5B-001: Authority reattachment surface

- Gate: `record_only` for this checklist version; becomes `hard_gate` after implementation is registered.
- Recommended outputs:
  - `semantic_stage2_objects/authority_reattachment/<paper_key>/semantic_authority_reattachment_v1.json`
  - `analysis/semantic_authority_reattachment_audit_v1.tsv`
- Current compatibility rule:
  - If no explicit S2-5b file exists, record `record_only_missing` unless alias-equivalent S2-7 trace / `table_cell_bindings_json` / authority sidecar proves deterministic reattachment.
- Pass-equivalent criteria:
  - LLM semantic table/scope signals resolve to stable S2-2 table authority records.
  - Ambiguity, conflicts, unresolved targets, and selected authority records are recorded.
  - Reattachment does not create semantic authorization; it only binds existing semantic signals to preserved authority.
- Failure boundary when enforced: `S2_5b_authority_reattachment_failure`

### S2-6 Contract validation

#### CHECK-S2-6-001: Stage2 semantic-authority contract report

- Gate: `hard_gate`
- Required output:
  - `analysis/stage2_semantic_authority_contract_report_v1.json`
- Pass criteria:
  - Validates provenance completeness and Stage2 authority legality.
  - Accepts alias-equivalent authority reattachment surfaces only when they are explicit and auditable.
  - Reports unresolved reattachment as target-resolution failure.
  - Does not silently project downstream when authority legality fails.
- Failure boundary: `S2_6_contract_validation_failure`

### S2-7 Compatibility projection and Stage3 handoff

#### CHECK-S2-7-001: Lawful Stage2 completion artifact

- Gate: `hard_gate`
- Required outputs:
  - `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
  - `semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.jsonl`
  - `semantic_to_widerow_adapter/compatibility_projection_trace_v1.tsv`
  - `semantic_to_widerow_adapter/compatibility_projection_summary_v1.json`
- Pass criteria:
  - S2-7 consumes S2-6-passing semantic objects and S2-2/S2-5b authority locators.
  - Completed weak-label TSV is a lawful Stage3 resume boundary.
  - Raw responses and raw semantic objects alone are not treated as lawful Stage3 inputs.
- Failure boundary: `S2_7_completion_artifact_failure`

#### CHECK-S2-7-002: Authorized table expansion through full authority

- Gate: `hard_gate`
- Pass criteria:
  - S2-7 may use LLM semantic authorization to decide that a table/scope should be expanded.
  - It must return to S2-2 full table/value authority to perform deterministic row expansion.
  - Applicable expanded rows preserve row-level binding surfaces such as `table_cell_bindings_json`, table ID, payload path, cell locator, or alias-equivalent locators.
  - Row/value losses are classified as semantic signal missing, authority reattachment unresolved, deterministic row expansion failure, or source table unrecoverable.
- Row-count note:
  - Row count may be recorded for diagnostics, but matching GT is not a generic hard gate and must not block full-manifest runs with no GT.
- Failure boundary: `S2_7_authorized_expansion_without_authority`

### S3-1 Relation materialization

#### CHECK-S3-1-001: Relation artifacts from completed Stage2 only

- Gate: `hard_gate`
- Inputs:
  - completed Stage2 weak-label TSV/JSONL.
  - scope/run metadata.
- Required outputs:
  - `formulation_relation_records_v1.tsv`
  - `formulation_logic_graph_v1.jsonl`
  - `formulation_relation_summary_v1.tsv`
- Pass criteria:
  - Stage3 does not call LLM.
  - Stage3 does not reread prose to rediscover missing chemistry.
  - Relation construction remains within Stage2-authorized scope.
- Failure boundary: `S3_relation_materialization_scope_drift`

### S3-2 Relation resolution and shared carrythrough

#### CHECK-S3-2-001: Shared carrythrough legality

- Gate: `hard_gate`
- Required output:
  - `resolved_relation_fields_v1.tsv`
- Pass criteria:
  - Parent/child inheritance, selected-condition inheritance, and shared preparation fields are source-backed, scope-aware, and entity-bound.
  - Stage3 does not perform cross-table Cartesian reconstruction.
  - Values never entering Stage2 weak labels or shared semantic surfaces are not rediscovered by Stage3.
- Failure boundary: `S3_shared_carrythrough_illegal`

### S4 Candidate diagnostics and review surfaces

#### CHECK-S4-1-001: Diagnostic-only classification

- Gate: `soft_audit`
- Pass criteria:
  - Candidate diagnostics, review workbooks, paper risk queues, and optional GT/review references are labeled diagnostic/reviewer-facing.
  - They are not downstream execution authority unless an explicit maintained contract says so.
- Failure boundary: `S4_diagnostic_surface_misclassified`

### S5-1 Fixed-row candidate intake

#### CHECK-S5-1-001: Fixed row universe

- Gate: `hard_gate`
- Inputs:
  - completed Stage2 weak-label TSV.
  - Stage3 relation records.
  - Stage3 resolved relation fields.
- Pass criteria:
  - Stage5 freezes the formulation-row universe before value layers.
  - Stage5 value layers attach values only to admitted rows.
  - Stage5 does not split, merge, or create rows unless an explicit governed identity rule records it.
  - Any count change is classified in decision trace.
- Failure boundary: `S5_row_universe_drift`

### S5-2 Deterministic direct materialization

#### CHECK-S5-2-001: Deterministic direct value path

- Gate: `hard_gate`
- Inputs:
  - fixed row universe.
  - Stage2 compatibility row fields.
  - S2-5b/S2-7 reattached authority locators.
  - `table_cell_bindings_json` or alias-equivalent locators.
  - Stage3 resolved relation fields.
- Pass criteria:
  - S5-2 consumes S2-2 full-table authority through S2-5b/S2-7 locators.
  - Prompt summaries are not numeric authority.
  - No donor-fill, assumption-fill, or direct mass-from-concentration conversion is written to direct fields.
  - If S2 authority contains a value but S5-2 cannot materialize it, the deterministic consumer path is the first repair target before S5-3.
- Failure boundary: `S5_2_deterministic_materialization_failure`

### S5-3 LLM-assisted residual direct-value extraction

#### CHECK-S5-3-001: Residual scope legality

- Gate: `hard_gate` if S5-3 is executed; `not_applicable` otherwise.
- Inputs:
  - fixed Stage5 rows.
  - audited residual direct-value gaps.
  - upstream source/evidence artifacts already governed.
- Pass criteria:
  - S5-3 scope is source-observability/residual-evidence driven, not blank-schema-slot driven.
  - S5-3 does not replace missing S5-2 deterministic consumption for table/simple shared values.
  - Outputs record direct-value candidates, evidence quote, scope, direct/derived classification, prompt hash, model identity, request parameters, validation status, and env file path without secrets.
- Failure boundary: `S5_3_residual_scope_or_metadata_failure`

### S5-4 Value authority validation and merge

#### CHECK-S5-4-001: Direct-value merge legality

- Gate: `hard_gate` if S5-4 is executed; `not_applicable` otherwise.
- Pass criteria:
  - Direct evidence, entity binding, scope, conflicts, and type compatibility are validated.
  - Row-local direct evidence outranks shared constants and LLM candidates.
  - Ambiguous, conflict-bearing, or quote-less values are rejected or routed to review.
  - `present_but_mismatch` and `blocked_alignment` are treated as review/alignment categories, not ordinary fill targets.
- Failure boundary: `S5_4_value_authority_merge_failure`

### S5-5 Derived reasoning sidecars

#### CHECK-S5-5-001: Derived sidecar separation

- Gate: `hard_gate` if derived values are produced; `not_applicable` otherwise.
- Pass criteria:
  - Derived/calculated fields are written to sidecars with formula IDs and input provenance.
  - Derived values do not overwrite direct fields.
  - Direct-field comparison or direct-field audit does not consume derived sidecars unless an explicit governed exception says so.
- Failure boundary: `S5_5_derived_value_contamination`

### S5-6 Final table closure and audit export

#### CHECK-S5-6-001: Final table and decision trace

- Gate: `hard_gate`
- Required outputs:
  - `final_formulation_table_v1.tsv`
  - `final_output_decision_trace_v1.tsv`
  - `final_output_summary_v1.md`
  - optional downstream/audit sidecars
- Pass criteria:
  - Final table has one output row per admitted formulation according to the frozen row universe and governed identity rules.
  - Decision trace explains retained/excluded rows and materialized values.
  - Output summary records row counts and diagnosis-baseline status for current-phase runs.
- Failure boundary: `S5_final_table_closure_failure`

### Diagnostic compare and ACTIVE_RUN pointer checks

#### CHECK-DIAG-001: Diagnostic compare classification

- Gate: `soft_audit`
- Applies only if diagnostic compare is in scope.
- Pass criteria:
  - Compare consumes explicit final table, declared scope, and explicit diagnostic reference inputs where applicable.
  - Compare mode is recorded as diagnostic.
  - The checklist does not require GT for full-manifest runs and does not define certification status.
- Failure boundary: `diagnostic_compare_misclassified`

#### CHECK-ACTIVE-001: ACTIVE_RUN pointer consistency

- Gate: `hard_gate` when the selected run is represented as current ACTIVE_RUN; `not_applicable` otherwise.
- Inputs:
  - `data/results/ACTIVE_RUN.json`
- Pass criteria:
  - Stage2/Stage3/Stage5/compare aliases point to the selected lineage's exact terminal artifacts.
  - No mixed stale/new aliases for the same semantic artifact.
  - Every referenced file exists.
  - `diagnosis_baseline=true` and current phase labels remain diagnostic when GT comparison is present.
  - Pointer update type is recorded as diagnosis-baseline authority pointer update.
- Failure boundary: `ACTIVE_RUN_pointer_inconsistent`

## 6. Current ACTIVE_RUN example snapshot

This appendix is an example target binding for the current diagnosis baseline at checklist creation time. Future checklist executions must bind to their own explicit target-run metadata/result outputs rather than editing this appendix as a run log.

Authority pointer:

```text
data/results/ACTIVE_RUN.json
```

Current target:

```text
active_run_id: 20260423_9c4a03f
active_run_dir: data/results/20260423_9c4a03f
diagnosis_baseline: true
compare_mode: diagnostic
```

Current terminal chain:

```text
S2-4a prompt freeze:
  data/results/20260423_9c4a03f/449_stage2_dev15_s2_4a_baseline_runnable_current_diagnostic/

S2-4b raw responses:
  data/results/20260423_9c4a03f/452_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_4b_assembled_diagnostic/

S2-5 semantic parsing:
  data/results/20260423_9c4a03f/453_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_5_diagnostic/

S2-6 contract validation:
  data/results/20260423_9c4a03f/454_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_6_diagnostic/

S2-7 completed Stage2 compatibility projection:
  data/results/20260423_9c4a03f/504_stage2_s2_7_source_excerpt_empty_pair_repair_diagnostic/

Stage3 relation materialization:
  data/results/20260423_9c4a03f/505_stage3_source_excerpt_empty_pair_repair_diagnostic/

Stage5 final closure:
  data/results/20260423_9c4a03f/508_stage5_source_excerpt_empty_pair_stage5_retention_repair_diagnostic/

Diagnostic count compare:
  data/results/20260423_9c4a03f/509_compare_source_excerpt_empty_pair_stage5_retention_repair_diagnostic/
```

Current main outputs:

```text
stage2_compatibility_tsv:
  data/results/20260423_9c4a03f/504_stage2_s2_7_source_excerpt_empty_pair_repair_diagnostic/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv

stage2_projection_summary_json:
  data/results/20260423_9c4a03f/504_stage2_s2_7_source_excerpt_empty_pair_repair_diagnostic/semantic_to_widerow_adapter/compatibility_projection_summary_v1.json

stage3_relation_records_tsv:
  data/results/20260423_9c4a03f/505_stage3_source_excerpt_empty_pair_repair_diagnostic/formulation_relation_records_v1.tsv

stage3_resolved_relation_fields_tsv:
  data/results/20260423_9c4a03f/505_stage3_source_excerpt_empty_pair_repair_diagnostic/resolved_relation_fields_v1.tsv

stage5_final_table_tsv:
  data/results/20260423_9c4a03f/508_stage5_source_excerpt_empty_pair_stage5_retention_repair_diagnostic/final_formulation_table_v1.tsv

stage5_decision_trace_tsv:
  data/results/20260423_9c4a03f/508_stage5_source_excerpt_empty_pair_stage5_retention_repair_diagnostic/final_output_decision_trace_v1.tsv

compare_counts_by_doi_tsv:
  data/results/20260423_9c4a03f/509_compare_source_excerpt_empty_pair_stage5_retention_repair_diagnostic/final_table_vs_gt_counts_by_doi.tsv
```

Current diagnostic count summary:

```text
papers: 15
matched_papers: 15
mismatched_papers: 0
total_gt_rows: 202
total_pred_rows: 202
final_table_rows: 202
```

Current known run-parameter pointers from `ACTIVE_RUN.json`:

```text
scope_manifest_tsv:
  data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv

stage2_support_scope_current_text_tsv:
  data/results/20260423_9c4a03f/374_stage2_dev15_v4_flash_tablefix_prellm_diagnostic/dev15_scope_current_text.tsv

stage2_s2_2_normalized_table_payloads_root:
  data/results/20260423_9c4a03f/416_stage2_dev15_cleantext_current_s2_4a_diagnostic/semantic_stage2_objects/normalized_table_payloads

stage2_s2_2_evidence_blocks_root:
  data/results/20260423_9c4a03f/416_stage2_dev15_cleantext_current_s2_4a_diagnostic/semantic_stage2_objects/evidence_blocks

stage2_s2_2_candidate_blocks_root:
  data/results/20260423_9c4a03f/416_stage2_dev15_cleantext_current_s2_4a_diagnostic/semantic_stage2_objects/candidate_blocks

stage2_s2_2_table_authority_validation_tsv:
  data/results/20260423_9c4a03f/416_stage2_dev15_cleantext_current_s2_4a_diagnostic/analysis/table_authority_validation_v1.tsv

stage2_s2_4a_prompt_audit_tsv:
  data/results/20260423_9c4a03f/449_stage2_dev15_s2_4a_baseline_runnable_current_diagnostic/analysis/s2_4a_prompt_audit_v1.tsv

stage2_s2_4b_request_summary_tsv:
  data/results/20260423_9c4a03f/452_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_4b_assembled_diagnostic/analysis/s2_4b_request_summary_v1.tsv

stage2_contract_report_json:
  data/results/20260423_9c4a03f/454_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_6_diagnostic/analysis/stage2_semantic_authority_contract_report_v1.json

stage2_authority_reattachment_sidecar:
  data/results/20260423_9c4a03f/453_stage2_dev15_s2_4a449_deepseek_v4_flash_s2_5_diagnostic/semantic_stage2_objects/authority_reattachment_sidecar_v1.json
```

Current example live-call lineage note:

```text
S2-4b backend/model: DeepSeek / deepseek-v4-flash, per run-local request metadata and RUN_CONTEXT files.
API keys: must be imported from the operator's env file/process environment; secret values must not be recorded in checklist outputs.
```

## 7. Implementation note for future automated checker

A future automated checker should implement this document as data-driven checks, not as prose matching. It should:

1. resolve target authority explicitly;
2. parse `ACTIVE_RUN.json`, `RUN_CONTEXT.md`, machine-readable run metadata, request summaries, and stage summaries;
3. inspect existence and schema of required artifacts;
4. compute thresholded table/CSV metrics;
5. emit one row per `CHECK-*` item;
6. keep missing planned surfaces as `record_only_missing` where this document says so;
7. fail loudly on hard-gate failures;
8. never record secret values;
9. never emit or require certification status.
