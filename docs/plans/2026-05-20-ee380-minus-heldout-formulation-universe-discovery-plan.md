# EE380 Minus Heldout Formulation-Universe Discovery Plan

Date: 2026-05-20

## Purpose

Run a high-recall formulation-universe discovery pass on the EE campaign papers
that are not part of the existing DEV15 diagnostic set and not part of the
GPT35/EE35 manual-upload set.

This is a diagnostic Stage2-supporting workflow. It does not create a completed
Stage2 artifact, does not feed Stage3 or Stage5, does not update
`data/results/ACTIVE_RUN.json`, and is not benchmark-valid final output.

## Scope Contract

Authoritative EE article scope:

- `data/results/20260511_b069802/14_ee_paper_level_selection_labels/ee_article_scope_selected_keys_v1.txt`
- expected unique EE article-scope papers: 380

Heldout exclusions:

- DEV15 keys from `data/results/20260511_b069802/123_stage1_dev15_goldline_table_authority_replay/analysis/dev15_keys.tsv`
- GPT35/EE35 keys from `data/results/20260511_b069802/16_ee35_upload_sources_for_gpt_web/upload_source_index_v1.tsv`

Observed count note:

- DEV15 contributes 15 keys inside EE380.
- GPT35/EE35 contributes 34 keys inside EE380 because `GWWNCC35` is in the
  GPT35/EE35 upload set but is not in the EE380 campaign key set.
- Therefore the strict EE380-minus-heldout scope is 331 papers, not 330.
- The workflow must preserve this count truth rather than deleting an extra
  paper to force a nominal count.

Scope artifact:

- `data/results/20260511_b069802/262_formulation_universe_ee380_minus_dev15_gpt35_scope/targeted_manifest.tsv`
- `data/results/20260511_b069802/262_formulation_universe_ee380_minus_dev15_gpt35_scope/analysis/scope_summary_v1.json`

## Execution Steps

1. Build the explicit scope manifest from the EE380 campaign key list, excluding
   DEV15 and GPT35/EE35 heldout keys. All later runs consume only this manifest.

2. Run prompt-only formulation-universe discovery over the scope. This validates
   source package creation, prompt construction, clean-text availability, and
   table-surface observability without making live model calls.

3. Run DeepSeek formulation-universe discovery over the same scope with the
   EE35 prompt and model semantics. The semantic task and prompt shape must not
   be changed. Operational parameters such as timeout and max output tokens may
   be adjusted only to prevent transport truncation or timeout.

4. Retry failures only by paper key. Retry categories include timeout, empty
   response, JSON parse failure, and likely output truncation. Retries may raise
   timeout or output-token limits but must not change the semantic row-creation
   rules.

5. Build GT-free risk flags. Risk signals include high model row count, DOE or
   matrix-heavy source signals, article-native label surplus, large exclusion
   ledgers, unresolved review candidates, source truncation, table truncation,
   and retry/parse-failure history.

## Boundary Rule

Row creation authority for downstream value binding belongs to this controlled
formulation-universe phase only after review and explicit promotion. Later value
extraction may fill existing rows or raise suspected missing rows to an audit
queue, but it must not freely create new formulation rows.
