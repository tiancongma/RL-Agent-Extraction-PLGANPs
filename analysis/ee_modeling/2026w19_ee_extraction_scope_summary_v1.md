# 2026W19 EE Extraction Scope / Clean Text Audit v1

- Existing keyed clean text files audited: **432**
- Included in sprint scope manifest (`yes` or `review_first`): **426**
- Stage2 scope manifest: `analysis/ee_modeling/2026w19_ee_extraction_scope_manifest_v1.tsv`
- First smoke batch: `analysis/ee_modeling/2026w19_ee_extraction_scope_batch001_25_manifest_v1.tsv`
- Second batch candidate: `analysis/ee_modeling/2026w19_ee_extraction_scope_batch002_50_manifest_v1.tsv`
- Third batch candidate: `analysis/ee_modeling/2026w19_ee_extraction_scope_batch003_100_manifest_v1.tsv`

## Priority distribution

- P0_ee_keyword: 414
- P2_plga_result_no_ee_keyword: 10
- P3_available_text_low_ee_signal: 5
- P1_ee_keyword_weak_formulation: 3

## Include distribution

- yes: 415
- review_first: 11
- no_low_priority: 6

## Clean text audit status

- usable_for_dryrun: 416
- short_or_truncated_review: 14
- weak_body_signal_review: 2

## Gate rule

Before any live LLM batch, run Stage2 with `--stop-before-live-call`, then inspect evidence blocks/prompt audit for:

1. EE/result evidence included when candidate has EE signals.
2. No large PK/release/tissue/reference/license noise takeover.
3. Clean text is not short/truncated and contains method/result body signals.

This scope is diagnostic/modeling-sprint input, not benchmark-valid evidence.
