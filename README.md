# RL-Agent-Extraction-PLGANPs

This repository implements a governed, auditable pipeline for extracting
nanoparticle formulation records from literature and converting them into final
per-formulation structured outputs.

The active system is not just a prompt collection and not just a field
extraction stack. It is a Stage 0 to Stage 5 pipeline with explicit provenance:

1. Zotero-derived raw corpus intake
2. manifest and cleaned-content construction
3. composite Stage2 extraction:
   - LLM semantic discovery
   - deterministic post-LLM completion inside Stage2
4. deterministic formulation relation materialization
5. candidate-level diagnostics
6. final formulation-table closure and final-table benchmark comparison

The benchmark-valid system result is the Stage 5 final formulation table and its
GT-comparison outputs. Intermediate artifacts may be used for diagnosis, but
they are not official benchmark outputs.

Current phase note:

- the repository is currently operating in diagnostic development mode
- current baselines are diagnostic baselines
- benchmark mode is currently disabled

The repository also supports a deterministic post-comparison Layer 2 risk
stratification artifact for downstream Layer 3 audit planning. That risk layer
is metadata only and does not alter benchmark-valid final outputs.

## Canonical Pipeline

The authoritative manual reproduction document is:

- [ACTIVE_PIPELINE_FLOW.md](project/ACTIVE_PIPELINE_FLOW.md)

That file defines the canonical Stage 0 to Stage 5 path, the exact stage
completion artifacts, the active scripts for each stage, and the allowed
incremental-reuse rules.

## Active Data Source Authority

Current `data/results` workflows must resolve their source artifacts through the
repository authority contract:

- [ACTIVE_DATA_SOURCE_CONTRACT.md](project/ACTIVE_DATA_SOURCE_CONTRACT.md)
- machine-readable pointer:
  - `data/results/ACTIVE_RUN.json`

Current authority promotion note:

- `ACTIVE_RUN.json` now points to the semantic Stage2 mainline lineage
  `run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1`.
- Historical legacy-extractor runs remain preserved under `data/results/` for
  auditability, but they are no longer the active repository authority.

The repository must not determine the active source by directory recency,
modification time, parent fallback, or glob-first matching. Use an explicit
CLI source such as `--run-dir`, or the declared authority pointer, or fail
loudly.

## Boundary Governance

Pipeline debugging may pause, branch, and replay only at explicit boundary
classes described in the active pipeline flow and runbook.

The current governed classes are:

- `internal_intermediate`
- `diagnostic_boundary`
- `mainline_resume_boundary`
- `benchmark_terminal_boundary`

Raw Stage2 freeze baselines remain diagnostic boundaries unless they also
preserve the authoritative completed Stage2 artifact required by Stage3.
Current maintained replay support allows the governed composite Stage2
entrypoint to rehydrate a frozen current live-v2 raw-response set into the
authoritative completed Stage2 artifact without making new LLM calls, but the
raw-response freeze by itself remains diagnostic-only.

## Active Stage Namespaces

The active runtime stage directories are:

- `src/stage0_relevance/`
- `src/stage1_cleaning/`
- `src/stage2_sampling_labels/`
- `src/stage3_relation/`
- `src/stage4_eval/`
- `src/stage5_benchmark/`

The reserved non-runtime reference namespace is:

- `src/stage3_gt/`

There is no active Stage 6 or Stage 7 namespace. Historical or retired methods
live outside `src/` under `archive/`.

## Repository Layout

- `project/`
  Governance and authoritative pipeline definitions.
- `src/`
  Active engineering code and stable stage-local tools only.
- `archive/`
  Historical methods, retired code, and delete-candidate quarantine.
- `data/`
  Raw inputs, cleaned assets, manual labels, and run artifacts.
- `docs/`
  Supporting documentation, audits, registries, and governance support files.

## Current Runtime Contract

The canonical path starts from Zotero-derived raw records under
`data/raw/zotero/` and ends at:

- `final_formulation_table_v1.tsv`
- `downstream_variant_records_v1.tsv`
- `final_table_vs_gt_counts.tsv`
- `final_table_vs_gt_summary.md`

Current active Stage 5 scripts:

- `src/stage5_benchmark/build_minimal_final_output_v1.py`
- `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Stage 5 is a materialization layer. It must not perform semantic inference.
Benchmark-valid Stage 5 closure now depends on both Stage 3 relation records and
Stage 3 resolved relation fields.
The primary benchmark-facing formulation table excludes downstream/post-processing
variants unless they are independently reported formulation identities; those
excluded rows are preserved in the governed linked lower-level surface
`downstream_variant_records_v1.tsv`.

Current Stage 5 supporting review export:

- `src/stage5_benchmark/build_boundary_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`

Current active Stage 3 script:

- `src/stage3_relation/build_formulation_relation_artifacts_v1.py`

Current maintained Stage 2 execution surfaces:

- `src/stage2_sampling_labels/run_stage2_composite_v1.py`
  - governed coarse-grained Stage2 wrapper and lawful replay/rehydration path
    into the authoritative completed Stage2 artifact
- `src/stage2_sampling_labels/run_stage2_s2_4a_prompt_construction_v1.py`
  - dedicated maintained frozen `S2-4a` prompt-construction boundary
- `src/stage2_sampling_labels/run_stage2_s2_4b_live_llm_call_v1.py`
  - dedicated maintained frozen `S2-4b` live-call boundary
- `src/stage2_sampling_labels/run_stage2_s2_5_semantic_parsing_v1.py`
  - dedicated maintained frozen `S2-5` semantic-parsing boundary that writes
    semantic-intermediate artifacts only
- `src/stage2_sampling_labels/run_stage2_s2_6_contract_validation_v1.py`
  - dedicated maintained frozen `S2-6` contract-validation boundary that
    consumes `S2-5` semantic intermediates and writes validation artifacts only
- `src/stage2_sampling_labels/run_stage2_s2_7_compatibility_projection_v1.py`
  - dedicated maintained frozen `S2-7` compatibility-projection boundary that
    consumes a passing `S2-6` validation surface and writes the completed
    Stage2 artifact only

Stage2 authority reminder:

- After the 2026-03-30 architecture freeze, deterministic semantic emitters are
  fallback, comparator, migration-support, or diagnostic infrastructure only.
- Stage2 is a composite stage consisting of LLM semantic discovery followed by
  deterministic post-LLM completion.
- Operationally, Stage2 now includes fine-grained maintained execution
  surfaces for frozen substeps, while the composite runner remains the coarse
  maintained wrapper and the lawful replay/rehydration path for completed
  Stage2 authority.
- No formulation candidate may enter authoritative Stage2 output unless it is
  traceable to `llm_semantic_discovery` or to an explicitly declared governed
  fallback semantic source.
- Deterministic DOE row expansion is preserved, but in normal
  `llm_first_composite` mode it is lawful only within LLM-declared DOE scope.
- The pre-LLM selector no longer owns semantic table truth. Its maintained
  contract is conservative denoising, minimum evidence coverage, and bounded
  packing only.
- Irreversible pre-LLM table removal is governed by a confirmed-noise-only
  rule: if a table is not confirmed pure noise, it must remain preserved in
  the pre-LLM authority surface.
- All `S2-4a` table evidence remains summary-only. Full tables are preserved in
  `S2-2` authority artifacts for deterministic execution, but they are not
  allowed back into the LLM-facing prompt surface.
- For low-ambiguity non-DOE `full_formulation` tables, deterministic row
  enumeration may happen after LLM table authorization from preserved `S2-2`
  normalized payload authority; it does not require LLM row-level output.
- When multiple plausible table summaries are present, the LLM owns semantic
  table scoping and must decide which surfaces are formulation-bearing or
  downstream/result-only.
- The maintained summary contract is neutral across preserved tables. The main
  residual risk is lossy summary compression, not primary-table promotion.
- Within summary-only table evidence, header / column schema plus first-column
  row identity surfaces are the primary structural contract. Sample rows are
  optional aids only.
- Authority reopen handles such as `authority_run_dir`,
  `authority_payload_root`, and table-scope locators are deterministic
  execution-side metadata. They must not depend on the LLM to generate or
  transmit them and may be reattached by governed sidecar logic during replay.
- If the LLM-facing input is unchanged, downstream deterministic contract
  changes should prefer replay from frozen raw responses over fresh live calls,
  provided execution-side metadata can be lawfully reattached.
- The governed `S2-4a` audit is split into three layers:
  - Hard Gate for pre-LLM legality/readiness only
  - Feature Activation Audit for artifact-backed repaired-feature activation
  - Calibration Review only for known-answer semantic checks
- Only the completed Stage2 artifact is authoritative for downstream Stage3
  consumption and Stage2 structural evaluation.
- The completed Stage2 artifact remains authoritative for downstream Stage3
  consumption whether produced through the coarse composite wrapper or the
  dedicated frozen `S2-7` runner.
- The governed three-paper comparison slice
  `src/utils/run_threepaper_stage2_v2_comparison.py` is a Stage2-only
  semantic-intermediate architecture-enforcement experiment for `WIVUCMYG`,
  `UFXX9WXE`, and `5GIF3D8W`. It does not replace `ACTIVE_RUN` or promote a new
  authority path.

Current active compatibility bridge:

- `src/stage2_sampling_labels/build_stage2_compatibility_projection_v1.py`

Current active Stage 4 diagnostic script:

- `src/stage4_eval/eval_weak_labels_v7pilot3.py`

Deprecated legacy Stage 2 fallback:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`

Current Layer 3 field-audit support surfaces:

- `src/stage5_benchmark/export_final_formulation_audit_ready_v1.py`
- `src/stage5_benchmark/build_field_gt_review_workbook_v1.py`
- `src/stage5_benchmark/build_layer2_risk_assessment_v1.py`

These are post-comparison review surfaces only. They do not alter frozen Stage
2, Stage 3, or Stage 5 semantics, and they do not change benchmark-valid final
outputs.

## Governance

Before changing code or pipeline documentation, read:

- [AGENTS.md](AGENTS.md)
- [2_ARCHITECTURE.md](project/2_ARCHITECTURE.md)
- [PIPELINE_SCRIPT_MAP.md](project/PIPELINE_SCRIPT_MAP.md)
- [ACTIVE_PIPELINE_FLOW.md](project/ACTIVE_PIPELINE_FLOW.md)
- [ACTIVE_PIPELINE_RUNBOOK.md](project/ACTIVE_PIPELINE_RUNBOOK.md)

## Reproducibility

Every `data/results/run_*` directory must contain a reproducibility-grade
`RUN_CONTEXT.md`.

Future maintained writer default:

- new maintained write surfaces under `data/results/` now default to the
  MDEC084 v2 bucket/child layout:
  - `data/results/YYYYMMDD_<short_hash>/NN_<cue>/`
- explicit legacy `run_*` creation remains compatibility-only and must be
  requested explicitly with a legacy `--run-id`

Accepted run types are:

- `intermediate_diagnostic_run`
- `component_regression_run`
- `full_pipeline_benchmark_run`

Only `full_pipeline_benchmark_run` may report official GT results, and only
when the evaluation object is the Stage 5 final formulation table.
## Evidence Binding audit sidecars

The repository includes diagnostic Evidence Binding sidecars downstream of the Stage 5 frozen final table. These sidecars are for audit and review planning only:

- `resolve_evidence_binding_authority_v1.py` locks the authority-resolved final table and related provenance surfaces.
- `audit_evidence_binding_contract_v1.py` audits which current surfaces expose direct evidence, broad anchors, Stage3 relation provenance, or legacy fallback metadata.
- `build_evidence_binding_packs_v1.py` emits frozen row/field binding packs without creating rows or values.
- `build_evidence_binding_risk_assessment_v1.py` consumes frozen packs and emits field/formulation/paper risk sidecars.
- `build_field_gt_review_workbook_v1.py` requires pack/risk inputs by default; `--legacy-evidence-mode` is required for explicit fallback.

Evidence Binding outputs are diagnostic-only and must not be reported as benchmark-valid final outputs.

