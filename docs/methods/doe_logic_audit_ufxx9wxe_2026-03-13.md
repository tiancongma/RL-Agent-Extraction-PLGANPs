# Objective

Audit the repository for existing DOE-related logic, record the confirmed `UFXX9WXE` failure mode in engineering-useful terms, and determine whether a new implementation is actually needed.

# Confirmed UFXX9WXE failure mode

- Paper key: `UFXX9WXE`
- DOI: `10.1155/2014/156010`
- Confirmed paper type: DOE-style optimization paper
- Manual confirmation: the source paper contains a table with `26` explicitly numbered formulations
- Current active DEV15 lineage:
  - Stage2 extracted about `5` candidate formulations
  - Stage5 produced `4` benchmark-facing final rows
  - GT count is `26`

Engineering interpretation:

- This is a confirmed Stage2 under-enumeration failure.
- The failure specifically concerns numbered DOE/design-table rows that were not enumerated as Stage2 formulation candidates.
- This is not primarily a Stage3 relation failure.
- This is not primarily a Stage5 collapse failure.
- Downstream stages cannot reconstruct formulation rows that were never enumerated upstream.

Primary evidence:

- [paper_diagnostic_summary.tsv](../../data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_diagnostic_summary.tsv)
- [paper_audit_pack.md](../../data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_audit_pack.md)
- [project/4_DECISIONS_LOG.md](../../project/4_DECISIONS_LOG.md)

# Existing DOE-related logic found

## 1. Active Stage2 prompt-side DOE handling

File:

- [src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py](../../src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py)

What exists:

- DOE/table-heavy keyword detection in input text.
- Prompt instructions that explicitly tell the model to enumerate design-matrix or sweep rows row by row.
- Table-heavy prompt hints such as:
  - DOE, Box-Behnken, response-surface, or parameter-sweep rows can still be formulation rows.
  - Enumerate formulation candidates row by row before abstraction.
  - Each distinct table row should become its own formulation candidate.

What does not exist here:

- No deterministic row enumerator for numbered DOE tables.
- No deterministic decoding of factor levels into explicit formulation rows before or during Stage2.
- No guarantee that all numbered table rows become Stage2 candidates.

Conclusion:

- Active Stage2 contains prompt-side intent for DOE row enumeration, but not a deterministic general DOE table enumerator.

## 2. Active Stage1 table discovery and extraction support

Files:

- [src/stage1_cleaning/find_html_table_candidates_v1.py](../../src/stage1_cleaning/find_html_table_candidates_v1.py)
- [src/stage1_cleaning/extract_tables_for_keys_v1.py](../../src/stage1_cleaning/extract_tables_for_keys_v1.py)

What exists:

- HTML table candidate discovery and table extraction helpers.
- Some DOE-adjacent keyword recognition such as `factorial`, `design`, `coded`, and similar table-quality cues.

What does not exist here:

- No formulation-specific DOE row enumeration logic.
- No deterministic conversion from numbered design-table rows into Stage2 candidate formulation rows.

Conclusion:

- Active Stage1 can help surface tables, but it does not solve the missing-row enumeration problem.

## 3. Active Stage3 relation logic

Files:

- [src/stage3_relation/build_formulation_relation_artifacts_v1.py](../../src/stage3_relation/build_formulation_relation_artifacts_v1.py)
- [src/stage3_relation/run_formulation_relation_artifacts_v1.py](../../src/stage3_relation/run_formulation_relation_artifacts_v1.py)

What exists:

- Deterministic grouping of existing Stage2 candidates into method groups, shared fields, and variation axes.

What does not exist here:

- No mechanism to invent or recover missing formulation rows when Stage2 did not emit them.
- No table parser or DOE design-matrix row generator.

Conclusion:

- Stage3 is not the right place to solve missing DOE row enumeration.

## 4. Active Stage4 DOE-specific evaluation logic

File:

- [src/stage4_eval/eval_weak_labels_v7pilot3.py](../../src/stage4_eval/eval_weak_labels_v7pilot3.py)

What exists:

- A narrow DOI-specific DOE reconciliation rule for `WFDTQ4VX`.
- It parses factor levels and coded checkpoint rows to reconcile evaluation coordinates for that one paper.

What it does not do:

- It does not generalize numbered DOE row extraction across papers.
- It does not create missing Stage2 formulation candidates.

Conclusion:

- This is active DOE-related logic, but it is downstream evaluation logic and paper-specific.

## 5. Stage5 branch-active DOE derivation and schema logic

Files:

- [src/stage5_benchmark/derive_doe_coded_factors_v1.py](../../src/stage5_benchmark/derive_doe_coded_factors_v1.py)
- [src/stage5_benchmark/run_derivation_v1.py](../../src/stage5_benchmark/run_derivation_v1.py)
- [src/stage5_benchmark/build_two_table_schema_v3.py](../../src/stage5_benchmark/build_two_table_schema_v3.py)
- [src/stage5_benchmark/formulation_core_signature_v1.py](../../src/stage5_benchmark/formulation_core_signature_v1.py)
- [src/stage5_benchmark/export_full_database_v1.py](../../src/stage5_benchmark/export_full_database_v1.py)

What exists:

- Deterministic DOE coded/decoded factor derivation from source tables.
- DOE factor-row outputs such as `doe_factor_rows.tsv` and `doe_decode_diagnostics.tsv`.
- Schema splitting and factor-signature logic that can use DOE decoded/coded values when the derivation layer succeeds.

What it does not do in the current canonical path:

- It is classified in the registry as `branch_active`, not `mainline_active`.
- It is not the current canonical Stage0 to Stage5 benchmark path described in the active runbook.
- It augments downstream benchmark/schema logic; it does not serve as the active general Stage2 enumerator for DOE table rows.

Conclusion:

- Useful DOE-related logic already exists, but it sits in a branch/supporting benchmark path and does not close the confirmed UFXX9WXE Stage2 gap by itself.

## 6. Historical or archived DOE logic

Files:

- [archive/code/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py](../../archive/code/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py)

What exists:

- Historical experimental DOE reconciliation logic overlapping with the active `WFDTQ4VX` evaluation rule.

Conclusion:

- Historical code exists as a reference and should stay historical.

# Active vs historical status

Active mainline:

- `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- `src/stage3_relation/*` relation layer

Branch-active or supporting, not current mainline:

- `src/stage5_benchmark/derive_doe_coded_factors_v1.py`
- `src/stage5_benchmark/run_derivation_v1.py`
- `src/stage5_benchmark/build_two_table_schema_v3.py`
- `src/stage5_benchmark/build_doe_signature_injection_audit_pack_v1.py`

Historical / archive:

- `archive/code/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py`
- older archived weak-label and reconstruction paths under `archive/code/`

# Duplicate / overlapping functionality check

Confirmed overlap:

- `WFDTQ4VX` DOE coordinate reconciliation exists both as active logic in [src/stage4_eval/eval_weak_labels_v7pilot3.py](../../src/stage4_eval/eval_weak_labels_v7pilot3.py) and as historical experimental reference in [archive/code/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py](../../archive/code/benchmark_specific_audit_report/test_doe_coordinate_reconciliation_v1.py).
- Stage5 has multiple DOE-adjacent supporting tools that partially overlap around decoded factors, DOE signatures, and audit reporting.

Interpretation:

- There is overlap in downstream DOE handling and audit support.
- There is not an active duplicate general-purpose DOE row enumerator in Stage2.
- The repository already has more than one downstream DOE helper, so any future implementation should avoid adding another downstream-only script that duplicates branch-active derivation behavior.

# Recommended intervention point

The confirmed UFXX9WXE gap is upstream:

- The missing functionality belongs before Stage3 and before Stage5 closure.
- The smallest correct intervention point is the Stage2 extraction boundary, or the Stage1-to-Stage2 seam if explicit table-row assets are needed to support deterministic enumeration.

Why:

- Stage3 can only relate candidates that already exist.
- Stage5 can only retain, collapse, decode, or export candidates that already exist.
- Missing numbered DOE rows must be enumerated before relation inference and before final closure.

# Whether a new script is needed

Yes, a real gap is confirmed, but no implementation should start from this audit alone without deliberate reuse planning.

Reason:

- Existing active Stage2 logic is prompt-side only for DOE row enumeration.
- Existing Stage5 DOE derivation logic is useful but downstream, branch-active, and not a substitute for upstream candidate creation.
- Existing active Stage4 DOE logic is paper-specific evaluation support and not a reusable enumeration layer.

Therefore:

- A new implementation is eventually needed if the project wants reliable recovery of numbered DOE table rows in the active canonical path.
- That implementation should not be another Stage4 or Stage5 patch.

# If a new script is needed, what exact stage it should belong to and why

Smallest justified next step:

- Stage placement: Stage2 extraction support, with optional use of Stage1 table assets
- Intended role: deterministic DOE table row enumerator for explicitly numbered design/formulation tables before relation inference

Exact intended inputs:

- dataset-scoped extracted paper table assets from Stage1 when available
- Stage2 scope manifest or paper selection metadata
- optionally the cleaned paper text for fallback evidence anchoring

Exact intended outputs:

- a run-scoped deterministic DOE row artifact under `data/results/<run_id>/...`
- row-level formulation candidates or candidate augmentations that the active Stage2 extractor can consume or merge before writing the weak-label TSV

Relation to existing scripts:

- Reuse Stage1 table assets instead of building a new table extractor.
- Reuse knowledge from `src/stage5_benchmark/derive_doe_coded_factors_v1.py` for coded-level and factor-level decoding where applicable.
- Do not duplicate Stage4 `WFDTQ4VX` evaluation-specific reconciliation logic.

Why existing scripts cannot already do it:

- Active Stage2 does not deterministically enumerate numbered DOE rows.
- Stage3 cannot create missing candidates.
- Stage5 DOE derivation and schema tools are downstream and branch-active; they operate after candidate extraction and do not repair missing Stage2 rows in the mainline path.

# Bottom line

- The confirmed UFXX9WXE problem is a real upstream gap.
- The repository already contains DOE-aware logic, but mostly downstream, branch-active, or paper-specific.
- New implementation is justified only at the Stage2 boundary, and only if it explicitly reuses existing table assets and downstream DOE-decoding knowledge instead of duplicating them.

# Implemented minimal upstream capability

As of 2026-03-13, the smallest justified upstream fix is now implemented:

- [src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py](../../src/stage2_sampling_labels/build_numbered_doe_row_candidates_v1.py)
- additive integration inside [src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py](../../src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py)

Implemented scope:

- deterministically detect explicit numbered DOE or design-table rows from Stage1 table CSV assets
- emit `numbered_doe_row_candidates_v1.tsv` and `numbered_doe_row_candidates_summary_v1.tsv`
- merge missing numbered rows additively into the Stage2 candidate stream
- expose a regression threshold via `--expected-min-recovered`

Still not implemented:

- broad coded-level DOE decoding for all papers
- prose-only DOE reconstruction without explicit numbered rows
- downstream generalized DOE collapse

# DOE numbered-table Stage2 strategy

## Context

`UFXX9WXE` is now the confirmed reference case for strong numbered DOE-table
under-enumeration in the current DEV15 family.

The operational question is no longer whether the paper contains enough source
structure. The repo now contains direct evidence that the critical numbered
table was already available before Stage2 and that the baseline miss happened
at the Stage2 interpretation boundary.

## Confirmed repository evidence

- HTML table assets were unavailable for this paper:
  - `data/cleaned/goren_2025/tables/UFXX9WXE/tables_manifest.json` records
    `html_found = false`.
- PDF table assets were available and sufficient:
  - the same manifest records `pdf_found = true`, `n_tables_pdf_extracted = 18`,
    and `preferred_table_source = pdf`.
- The explicit numbered `1` to `26` DOE structure was already present in the
  Stage1 PDF table asset:
  - `data/cleaned/goren_2025/tables/UFXX9WXE/UFXX9WXE__table_13__pdf_table.csv`
- The baseline Stage2 miss came from interpretation over a truncated full-text
  window rather than structured table consumption:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/08_UFXX9WXE_10.1155_2014_156010.txt`
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/analysis/paper_diagnostic_summary.tsv`
- DOE recovery succeeded by deterministic enumeration over the already-available
  PDF table asset:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/run_20260313_1157_f4912f3_ufxx9wxe_doe_row_recovery_v5/numbered_doe_row_candidates_v1/numbered_doe_row_candidates_summary_v1.tsv`

## Decision

For strong DOE-style numbered tables, deterministic Stage2 enumeration is the
primary row-discovery mechanism.

The LLM should not be the primary mechanism for row counting or row discovery
when a structured Stage1 table asset already exposes stable numbered row
anchors and row-wise formulation structure.

In strong-structure cases, the LLM acts as judge, not counter.

## Triggering conditions

Use `deterministic_enumeration` when all of the following are true:

- a structured Stage1 table asset exists
- the table exposes stable row anchors such as numbered rows
- the table shows row-wise formulation or design structure
- safety guards do not indicate obvious misfire risk

Use `hybrid_enumeration_review` when:

- deterministic row anchors exist
- but table irregularity or row ambiguity is high enough that emitted rows
  should carry explicit review-oriented provenance

Use `llm_discovery_only` when:

- no reliable structured table asset exists
- or the paper does not expose stable row-wise formulation structure suitable
  for deterministic enumeration

## LLM role

In strong numbered-table cases, the LLM supports semantic judgment tasks:

- confirm that a detected table is a formulation or design table
- validate whether enumerated rows are true formulation rows
- map column semantics into the Stage2 schema
- identify exceptional rows such as optimized, control, summary, validation,
  or non-formulation rows

This is narrower than using the LLM as the main row counter.

## Non-goals / current exclusions

- no broad DEV15-wide DOE rollout decision is made here
- no claim that all DOE papers should immediately switch to deterministic
  enumeration
- no generalized coded-level DOE decoding strategy is declared here
- no downstream Stage3 or Stage5 patch is treated as the primary fix for this
  failure class

The current full DEV15 DOE rebuild remains useful as integration evidence, but
it is not baseline-ready because broader regressions remain in that lineage.

## Immediate next step

The next justified optimization target is a bounded Stage2 fix and validation
path for `UFXX9WXE`-class strong numbered DOE tables.

The immediate engineering goal is not broad benchmark rollout. It is to make
the Stage2 boundary reliably prefer deterministic numbered-row recovery in
strong-structure cases while keeping the scope narrow enough to audit regressions.

## Bounded UFXX validation result

Validation run:

- `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/lineage/children/run_20260313_1526_f4912f3_ufxx_only_stage2_doe_validation_v1`

Patched behavior:

- deterministic numbered DOE rows are now emitted even when overlapping numeric
  labels already exist as `llm_extracted` rows
- overlapping numeric `llm_extracted` rows are dropped in favor of explicit
  structured-table `doe_numbered_table_row` rows

Observed bounded replay result:

- baseline Stage2 candidate count: `5`
- validation Stage2 candidate count: `28`
- deterministic numbered DOE candidates: `26`
- baseline final row count: `4`
- validation final row count: `28`
- GT row count: `26`

What changed in the active path:

- `1.`, `2.`, and `3.` are no longer retained as `llm_extracted` rows with
  `full_text_window` evidence only
- the numbered DOE rows now carry:
  - `candidate_source = doe_numbered_table_row`
  - `evidence_section = UFXX9WXE__numbered_doe_table_01`
  - `instance_evidence_region_type = table_row`
- the enumerator artifact records overlap with the former LLM rows through
  `existing_stage2_match`

Interpretation:

- The bounded UFXX replay succeeded for the intended Stage2 target.
- The improvement came from structured-table consumption at the Stage2
  boundary, not from an opaque LLM-only row-counting behavior.
- The remaining `+2` at final output is due to two retained non-table LLM rows
  (`Optimized_Lzp_PLGA_NPs` and `F10`), not due to fabricated extra numbered
  DOE rows.

Current conclusion:

- This validates the narrow Stage2 merge preference for strong numbered DOE
  tables.
- It still does not authorize broad DEV15-wide DOE rollout because residual
  row-governance questions remain outside the numbered-table recovery itself.
