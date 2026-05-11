# Layer 3 Cross-Audit v1

## Purpose

This method defines a post-annotation Layer 3 audit surface for the compact
value workbook.

The goal is not to repair workbook values automatically. The goal is to
produce a human-review risk report that highlights workbook cells whose current
values may be unsupported, derived, directionally inverted, contaminated, or
otherwise unsafe to trust as GT without manual confirmation.

## Architectural Role

This method is not only an evaluation helper.

It is part of the reviewer-facing production audit and governance layer around
the frozen formulation database.

Current interpretation:

- the benchmark-valid endpoint remains `final_formulation_table_v1.tsv`
- this cross-audit remains a downstream report-only surface and must not
  mutate benchmark-valid outputs
- the preferred reviewer entry object is still the formulation row
- this cell-level report serves the value-credibility layer of review
- it depends on upstream formulation existence and identity correctness
- many apparent cell-level value problems are projections of structure or
  identity errors rather than isolated value mistakes
- current repo capability is partially present but not yet unified into one
  formulation-centered audit system contract

## Governing Principle

Layer 3 retains only explicit source-supported values.

Hard rule:

- only explicit support from the current paper's cleaned text or cleaned tables
  counts as support
- computed, normalized, inferred, inherited, or inverted values do not count as
  direct support
- model agreement does not create truth
- blank or unloaded formulations must preserve null drug fields under the
  existing null-drug rule

## Scope

This method applies to reviewer-facing Layer 3 value workbooks and downstream
audit reports.

It does not modify:

- Stage 2 extraction semantics
- Stage 3 relation semantics
- Stage 5 final-table identity closure
- benchmark-valid outputs
- workbook contents

## Entrypoint

- `src/stage5_benchmark/run_layer3_cross_audit_v1.py`

Subcommands:

- `rule-audit`
- `export-model-tasks`
- `execute-models`
- `merge`
- `run`

## Inputs

Minimum runtime inputs:

- value workbook:
  - `value_gt_annotation_workbook_representation_repaired_v4.xlsx`
- cleaned text assets
- cleaned table CSV assets

Authority resolution:

- prefer explicit `--workbook` or `--run-dir`
- otherwise resolve from `data/results/ACTIVE_RUN.json`
- do not infer sources from newest-directory heuristics

## Output Contract

The merged report emits one row per flagged cell with these columns:

- `paper_id`
- `formulation_id`
- `field_name`
- `current_value`
- `risk_level`
- `risk_type`
- `source_of_flag`
- `reason`
- `evidence_status`
- `evidence_snippet`
- `source_paths`

Primary outputs:

- run-scoped TSV:
  - `data/results/<run_id>/analysis/layer3_gt_cross_audit_report_v4.tsv`
- run-scoped high-risk subset:
  - `data/results/<run_id>/analysis/layer3_gt_cross_audit_report_v4_high_risk.tsv`
- human-readable markdown summary:
  - `project/layer3_gt_cross_audit_report_v4.md`

Supporting outputs:

- deterministic rule flags TSV/JSON
- Gemini task JSONL
- NVIDIA task JSONL
- task-row TSV
- optional partial backend TSVs written after each model batch

## Risk Types

The framework uses these normalized risk types:

- `unsupported_value`
- `derived_value`
- `direction_mismatch`
- `unit_or_normalization_only`
- `blank_should_be_null`
- `cross_paper_contamination`
- `inheritance_contamination`
- `ambiguity`

## Deterministic Rule Audit

The rule audit scans non-empty workbook cells field by field and flags cells
when at least one of these conditions holds:

- the current value is not found explicitly in cleaned text or cleaned tables
- a ratio is represented only through an inverse or alternate direction
- a numeric value appears only after unit or formatting transformation
- a ratio appears computable from component numbers but is not explicitly stated
- a blank, empty, unloaded, placebo, drug-free, or no-drug row carries a
  non-null `drug_name`
- a value appears only in reference-like text
- the same value appears in sibling/family rows without row-local support
- support is weak, partial, or conflicts across multiple mentions

Risk-level policy:

- `high`: clear null-rule violation, cross-paper contamination, strong
  inheritance contamination, or unsupported high-stakes value
- `medium`: derived or ambiguous mapping risk
- `low`: normalization-only or formatting-only risk

## Gemini And NVIDIA Auditor Tasks

The model layer is auditor-only.

They must not:

- fill missing values
- normalize values into support
- compute ratios and call them supported
- override deterministic report output silently

They may only return flagged rows that match the shared report schema.

Execution mode:

- `run` and `execute-models` perform live Gemini and/or NVIDIA calls when not
  skipped
- execution is currently synchronous and per-cell
- `batch_size` governs progress logging and partial-write cadence, not
  multi-cell API bundling
- model execution must be bounded during smoke tests or debugging with:
  - `--max-candidates`
  - `--max-gemini-calls`
  - `--max-nvidia-calls`
  - `--request-timeout-seconds`
  - `--max-retries`
  - `--write-partial-every-batch`

Safety controls:

- `--rules-only` merges a rule-only report without attempting model calls
- `--skip-gemini` and `--skip-nvidia` disable individual backends
- missing backend credentials fail fast for that backend and write an empty
  result TSV instead of hanging silently
- partial backend results are checkpointed after each batch when requested

Prompt contract:

- review the current workbook cell plus local text/table snippets
- flag only if suspicious
- return an empty array when no flag is warranted

## Merge And Adjudication

The merge step combines:

- deterministic rule flags
- optional Gemini audit rows
- optional NVIDIA audit rows

Merge policy:

- one output row per flagged cell
- `source_of_flag` becomes `rule`, `gemini`, `nvidia`, or `multiple`
- if multiple sources flag the same cell, priority is elevated by one risk
  level step
- risk type resolves by governed severity precedence rather than majority vote
- reasons, snippets, and source paths are retained in merged form

## Model Governance

Gemini and NVIDIA are auditors only, not GT editors.

They are supporting review signals. They do not create benchmark-valid truth,
and they do not override the explicit-support rule.

## Example Usage

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx
```

Fast rules-only smoke test:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py run `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --rules-only
```

Gemini-only bounded smoke test:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py execute-models `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-nvidia `
  --max-candidates 10 `
  --max-gemini-calls 10 `
  --batch-size 5 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch
```

NVIDIA-only bounded smoke test:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py execute-models `
  --workbook data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4.xlsx `
  --skip-gemini `
  --max-candidates 10 `
  --max-nvidia-calls 10 `
  --batch-size 5 `
  --request-timeout-seconds 30 `
  --max-retries 0 `
  --write-partial-every-batch
```

Explicit merge with externally generated model rows:

```powershell
python src/stage5_benchmark/run_layer3_cross_audit_v1.py merge `
  --run-dir data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1 `
  --rule-tsv data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/analysis/layer3_cross_audit_rule_flags_v4.tsv `
  --gemini-tsv path/to/gemini_audit_rows.tsv `
  --nvidia-tsv path/to/nvidia_audit_rows.tsv
```

## Relationship To Existing Layer 3 Rules

This method preserves and depends on the existing blank/unloaded null-drug
rule.

It extends the reviewer-surface safeguards by adding a report-only
cross-audit layer that surfaces unsupported or suspicious workbook cells before
manual GT correction.

Dependency note:

- this cross-audit is the downstream value-credibility layer
- it should be interpreted together with formulation existence and identity
  review, not as an independent truth object
