# S2-4a Audit Standard

This document defines the governed audit standard for the frozen `S2-4a`
prompt-construction boundary.

Boundary statement:

- Hard Gate checks legality and readiness only.
- Feature Activation Audit checks whether repaired capabilities are actually on
  in the current artifacts.
- Calibration Review checks semantic correctness on known-answer papers only.

Hard Gate must not adjudicate true semantic primary-table meaning. Calibration
Review may do that, but only on known-answer papers and only as calibration.

Why this split exists:

- to prevent checklist overreach
- to prevent hidden semantic adjudication by deterministic rules
- to ensure repaired functions are not merely configured in code but actually
  active in run artifacts

Scope:

- this standard evaluates the prompt-ready evidence pack produced at `S2-4a`
- this standard does not modify Stage1, downstream `S2-5` / `S2-6` / `S2-7`,
  or the execution-grade full-table authority preserved in `S2-2`
- this standard is reusable across ordinary readiness checks, feature
  activation review, and targeted calibration review, but those three uses are
  intentionally separated

## Layer A — Hard Gate

Purpose:

- determine whether a paper may legally and safely cross the `S2-4a` boundary
  into `S2-4b`
- evaluate prompt-readiness and contract compliance only
- avoid semantic truth claims such as which table is the true formulation core

### A1. Minimum evidence contract

Check:

- Hard Gate passes the minimum evidence contract when any one of the following
  sufficiency paths is satisfied:
  - Path 1:
    at least one formulation-bearing table summary survives into the LLM-facing
    evidence pack
  - Path 2:
    preparation or method evidence is present and strong table-adjacent
    formulation description survives in text
  - Path 3:
    preparation or method evidence is present and an explicit formulation
    definition survives in text

Path details:

- Path 1 is the primary sufficiency path.
- Path 1 does not require materials evidence.
- Path 1 does not require additional supporting or discussion context.
- Path 2 is a no-table fallback for papers where a usable formulation table
  summary does not survive.
- Path 3 is a rare text-only fallback for papers that define the formulation
  explicitly in text.

Path 1 interpretation:

- a formulation-bearing table summary is a structured summary surface for a
  multi-row formulation table, not a purely narrative block
- Hard Gate may ask whether at least one such summary surface survived
- Hard Gate must not decide which surviving table is the true semantic core

Optional evidence:

- materials evidence is optional soft-support
- supporting context is optional soft-support
- absence of materials or supporting context must not cause Hard Gate failure
- Hard Gate must not require multiple evidence families or evidence-rich
  coverage when minimum formulation sufficiency is already satisfied

Interpretation notes:

- this layer may ask whether a minimally sufficient formulation surface
  survived into the pack
- this layer must not ask which surviving table is semantically the real core

### A2. Summary-only table contract

Check:

- all LLM-facing table evidence is summary-only
- no inline full-table leakage survives
- no raw table dump appears in the prompt
- governed summary fields only are used, such as caption/title, key columns,
  units, row-label preview, sample rows, and important footnotes
- table-derived `inline_table_text` may exist as an intermediate diagnostic
  surface upstream, but it is not lawful as final prompt-facing table evidence

### A3. Selector boundary compliance

Check:

- selector did not semantically veto ambiguous candidate tables
- `must_include` table summaries are preserved
- `hard_drop` is limited to high-confidence garbage only
- `optional_context` remains bounded

Interpretation notes:

- this layer may judge whether selector behavior violated the selector
  contract
- this layer must not declare which preserved table should have been the true
  semantic primary surface

### A4. Prompt legality and boundedness

Check:

- no prompt inflation
- no duplicated oversized blocks
- no broken ordering contract
- evidence-pack structure is valid for the frozen prompt boundary

### A5. Hard-gate failure labels

Use only legality and readiness labels such as:

- `table_missing`
- `evidence_underselected`
- `summary_contract_violation`
- `prompt_inflation`
- `hard_drop_overreach`
- `selector_boundary_violation`
- `candidate_table_quality_failure`

Hard Gate output:

- `PASS` or `FAIL`
- failure labels
- bounded readiness reason
- no semantic truth claim such as "`Table 1` is definitely the real primary
  table"

## Layer B — Feature Activation Audit

Purpose:

- verify that important repaired or maintained capabilities are actually active
  in the current run artifacts
- evaluate artifact-backed activation, not code presence or configuration
  presence alone
- separate capability activation from semantic correctness

General rule:

- if activation is observable from current artifacts, record `PASS` or `FAIL`
- if the run does not provide enough evidence, record `UNKNOWN`
- do not infer activation from repository code alone

Primary artifact surfaces:

- `semantic_stage2_objects/candidate_blocks/<paper_key>/candidate_blocks_v1.json`
- `semantic_stage2_objects/normalized_table_payloads/<paper_key>/normalized_table_payloads_v1.json`
- `semantic_stage2_objects/evidence_blocks/<paper_key>/evidence_blocks_v1.json`
- `analysis/stage2_prompt_preview_v1.tsv`
- `analysis/s2_4a_prompt_audit_v1.tsv`
- `analysis/feature_activation_report_v1.tsv`
- run-local `RUN_CONTEXT.md`

### B1. Table recovery and table repair activation

Expected behavior:

- repaired summary exists when expected
- `repair_primary_source` is populated when authoritative repair was used
- `repair_actions` is populated when repair intervened
- first-column preservation appears when applicable
- `selector_readiness` shows a usable summary state when the repaired summary
  is expected to be usable

Artifact-backed evidence:

- `normalized_table_payloads_v1.json`
- payload-level repaired summary metadata

### B2. Summary-first behavior activation

Expected behavior:

- `S2-4a` uses summary-only table surfaces
- no fallback to full-table prompt mode appears in prompt artifacts
- table summaries are actually consumed by the prompt

Artifact-backed evidence:

- `s2_4a_prompt_audit_v1.tsv`
- frozen `s2_4a_prompts_v1.jsonl`
- `evidence_blocks_v1.json`

### B3. Ordered evidence packing activation

Expected behavior:

- evidence blocks appear in governed order
- method, materials, table, and supporting blocks comply with the current
  maintained contract
- no regression to arbitrary ordering appears in the canonical evidence pack

Artifact-backed evidence:

- `evidence_blocks_v1.json`
- `stage2_prompt_preview_v1.tsv`
- `s2_4a_prompt_audit_v1.tsv`

### B4. Raw-prefix removal activation

Expected behavior:

- legacy raw-prefix style blocks do not leak into the current frozen prompt
  surface
- the prompt uses the canonical evidence pack only

Artifact-backed evidence:

- `s2_4a_prompt_audit_v1.tsv`
- frozen `s2_4a_prompts_v1.jsonl`

### B5. Duplicate suppression activation

Expected behavior:

- near-duplicate method, materials, or table surfaces are not repeatedly
  packed
- no obvious duplicated large evidence block survives into the frozen prompt

Artifact-backed evidence:

- `evidence_blocks_v1.json`
- selector-debug metadata inside the evidence artifact
- `feature_activation_report_v1.tsv`

Interpretation note:

- if a run contains no duplicate case, the activation state may be
  `not_invoked` or `UNKNOWN`; do not claim positive activation merely because
  the code path exists

### B6. Selector contract activation

Expected behavior:

- `must_include` tables are preserved
- `optional_context` does not displace `must_include`
- no semantic primary-table forcing appears in prompt ordering

Artifact-backed evidence:

- `normalized_table_payloads_v1.json`
- `evidence_blocks_v1.json`
- `s2_4a_prompt_audit_v1.tsv`

### B7. Other maintained repaired features documented in governance

The `S2-4a` feature-activation audit must also consult the governed feature
surfaces and record any relevant maintained Stage2 features as:

- `feature_name`
- expected behavior
- artifact-backed evidence
- `PASS` / `FAIL` / `UNKNOWN`

Governing sources:

- `project/feature_units/feature_unit_registry.json`
- `project/feature_units/feature_intervention_matrix.tsv`
- `project/feature_units/FEATURE_UNIT_GOVERNANCE.md`
- run-local `analysis/feature_activation_report_v1.tsv`

Examples already relevant to `S2-4a`:

- `s2_candidate_section_aware_split`
- `s2_candidate_table_isolation`
- `s2_candidate_noise_filtering`
- `s2_2_evidence_artifact_contract`
- `s2_2_design_success_split`
- `s2_2_prompt_preview_derived_from_evidence_artifact`
- `s2_2_evidence_priority_selection`
- `s2_2_duplicate_table_suppression`
- `stage2_input_evidence_packing` when that optional execution mode is enabled

Feature Activation output:

- one row per audited feature
- expected behavior
- artifact-backed evidence
- `PASS` / `FAIL` / `UNKNOWN`

## Layer C — Calibration Review Only

Purpose:

- review semantic correctness on known-answer or manually understood papers
- support targeted repairs and post-change regression review
- isolate semantic correctness review from the hard gate

Calibration-only rule:

- this layer is not the general readiness gate for ordinary pipeline runs
- this layer may make semantic truth claims on known papers
- this layer must explicitly say that it is calibration-only

Allowed semantic review questions:

- which table is actually formulation-bearing
- which table is downstream, result-only, or non-core
- whether bundled summary-only evidence is semantically adequate for the known
  paper
- whether the current table-scoping prompt framing matches the human reference

Calibration-paper examples:

- `INMUTV7L`
- `V99GKZEI`
- `L3H2RS2H`
- `QLYKLPKT`

Allowed calibration judgments:

- expected formulation-bearing table or table bundle
- expected downstream or non-core table surfaces
- whether the bundled summaries are semantically adequate
- whether the current result looks improved, stable, or regressed relative to a
  known baseline

Calibration Review output:

- paper key
- calibration expectation
- current semantic adequacy judgment
- calibration notes
- explicit note that this is not the universal hard gate for ordinary runs
