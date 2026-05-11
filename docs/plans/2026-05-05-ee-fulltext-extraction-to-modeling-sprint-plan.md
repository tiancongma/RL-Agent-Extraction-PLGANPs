# 2026-05-05 EE Full-Text Extraction to Modeling Sprint Plan

Status: active sprint plan, diagnostic/modeling-facing. This is not a governance document and does not modify `project/` authority.

Current timestamp: 2026-05-05 21:50:44 EDT

## User decisions recorded

1. Scope: first run all papers with downloaded/or locally available full text, but audit clean text because text extraction may have lost information.
2. Live LLM is allowed this week only with explicit batch-level user approval immediately before each live run, and only after a pre-live evidence-selector gate to avoid invalid calls where noisy evidence packs dominate the prompt.
3. Modeling target is locked to EE / encapsulation or entrapment efficiency.
4. Dataset stratification is accepted: Tier A high-confidence modeling rows, Tier B broad LLM-extracted rows, plus hold/review queues.
5. New plan/progress artifacts may be created outside `project/`.

## Strategic objective

Move from DEV15 field-by-field repair to a modeling-sprint production line:

- this week: broad EE-centered extraction over locally available full-text papers;
- this week: pre-live clean-text and evidence-selector gates before spending LLM calls;
- this week: produce broad extracted dataset, Tier A candidate dataset, and human audit queue;
- next week: start EE modeling with Tier A, with Tier B used for sensitivity and coverage analysis.

This sprint intentionally does not wait for full Layer3 completeness or benchmark-valid status. Outputs must be labeled diagnostic/modeling-sprint until a benchmark-valid lineage is established.

## Corpus scope already materialized

Generated files:

- `analysis/ee_modeling/2026w19_downloaded_fulltext_cleantext_scope_audit_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_summary_v1.md`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_25_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch002_50_manifest_v1.tsv`
- `analysis/ee_modeling/2026w19_ee_extraction_scope_batch003_100_manifest_v1.tsv`

Scope audit result:

- Existing keyed clean text files audited: 432 before manifest deduplication.
- Deduplicated Stage2 scope manifest: 382 papers.
- Initial included/review-first scope before dedupe: 426 rows.
- Priority distribution before dedupe:
  - P0 EE keyword: 414
  - P1 EE keyword weak formulation: 3
  - P2 PLGA/result without explicit EE keyword: 10
  - P3 low EE signal: 5
- Clean text status before dedupe:
  - usable_for_dryrun: 416
  - short_or_truncated_review: 14
  - weak_body_signal_review: 2

Interpretation: the project has enough locally available clean text to start broad extraction now. `manifest_current.tsv` undercounts text availability; the sprint manifest is built from existing keyed clean-text surfaces instead.

## Batch 001 dry run already executed

Dry run command used `--stop-before-live-call` and wrote:

`data/results/20260505_c1ad6ca/01_ee_sprint_batch001_25_dryrun/`

Key artifacts:

- `analysis/stage2_prompt_preview_v1.tsv`
- `analysis/table_selection_debug_v1.json`
- `analysis/candidate_segmentation_debug_v1.tsv`
- `semantic_stage2_objects/candidate_blocks/`
- `semantic_stage2_objects/evidence_blocks/`
- `analysis/pre_llm_evidence_sufficiency_audit_v1/`
- `analysis/pre_llm_ee_noise_gate_v1/`

The dry run intentionally stopped before live calls. Its `success_count=0 failure_count=25` is expected for raw-response completion because live calls were not made; the useful artifacts are the prompt/evidence/debug outputs.

## Pre-live gate result for Batch 001

`analysis/pre_llm_ee_noise_gate_v1/pre_llm_ee_noise_gate_summary_v1.json`:

- 25 rows audited
- pass for live LLM: 14
- hold for selector/clean-text review: 11

Gate reasons:

- oversized_prompt: 2
- tail_noise_with_weak_ee_signal: 4
- selected_evidence_missing_ee_or_loading_signal: 6
- missing_preparation_core: 2

Live-pass manifest:

`analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_live_gatepass_14_manifest_v1.tsv`

Hold/review manifest:

`analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_hold_for_review_11_manifest_v1.tsv`

## Pre-live LLM hard gate

A paper/batch may proceed to live LLM only when:

1. clean text is not short/truncated and has method or result body signal;
2. selected evidence contains EE / entrapment / encapsulation / loading signal when candidate evidence has it;
3. selected evidence is not dominated by release, PK, tissue distribution, cell uptake, references, copyright/license, or unrelated biological assay noise;
4. prompt is not oversized by current health policy, unless manually approved for a specific high-value paper;
5. preparation/material evidence is not missing for a formulation-bearing paper unless the paper is being run only for target/output discovery.

Failing papers are not discarded. They move to a hold/review queue for selector repair, clean-text repair, or manual approval.

## Modeling target and tiers

Primary target:

- `ee_percent` / encapsulation efficiency / entrapment efficiency.

Tier A high-confidence modeling rows require:

- paper identity and DOI/source provenance;
- formulation identity clear;
- EE value present and alignable to a formulation instance;
- drug identity present;
- polymer identity or PLGA-dominant carrier identity present;
- preparation method class present or inferable;
- no high-risk target/formulation alignment flag;
- evidence locator or selected evidence block preserved.

Tier B broad LLM-extracted rows include:

- rows with EE but missing one or more important predictors;
- rows with medium/high evidence or alignment risk;
- rows useful for coverage reporting, sensitivity analysis, and audit queue generation.

Do not silently mix Tier B into primary modeling.

## Minimum predictors for next-week EE modeling

Training-blocking/core fields:

- `paper_key`
- DOI/source identity
- stable formulation ID
- formulation label
- `ee_percent`
- EE evidence locator or selected evidence block
- drug name
- polymer/carrier identity
- preparation method class
- risk flags

High-value predictors, not all required for first model:

- polymer MW / LA:GA ratio
- polymer mass or concentration
- drug mass or drug/polymer ratio
- surfactant/stabilizer name and concentration
- organic solvent/co-solvent
- phase volumes or organic/aqueous ratio

Auxiliary outputs for characterization-assisted model only:

- particle size
- PDI
- zeta potential
- LC/DL

The main model should be design-only and exclude post-characterization outputs as predictors. A second characterization-assisted model may include them and must be labeled accordingly.

## Immediate execution sequence

1. Run live LLM only for Batch 001 gate-pass manifest.
2. Run Stage2 completion/compatibility for that live batch.
3. Run Stage3 and Stage5 maintained entrypoints for the live batch if Stage2 completes lawfully.
4. Produce an EE modeling-readiness summary for live Batch 001.
5. In parallel, inspect the 11 held papers and decide whether to repair selector, repair clean text, or approve manual live run.
6. Expand to Batch 002 only after Batch 001 live output has acceptable EE/formulation extraction coverage and no systemic prompt-noise failure.

## Non-goals for this sprint

- Do not claim benchmark-valid extraction performance from this sprint.
- Do not fix every Layer3 residual before broad extraction.
- Do not block EE modeling on size/PDI/zeta completeness.
- Do not use GT protocol excerpts or user snippets as Stage2 runtime evidence.
- Do not add paper-specific runtime rules.

## Acceptance by end of week

Minimum acceptable output:

- broad EE-centered extraction over a substantial locally available full-text subset;
- explicit pass/hold evidence-selector gate for every live batch;
- Tier A candidate dataset and Tier B broad dataset;
- human audit queue prioritized by target ambiguity and model leverage;
- reproducible run directories with `RUN_CONTEXT.md`;
- next-week design-only EE modeling can start.
