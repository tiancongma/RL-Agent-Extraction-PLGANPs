# S2-3 Prompt Input Audit

Comparison surfaces used:

- Baseline prompt preview:
  - `data/results/20260417_385b6e1/09_dev15_stage2_baseline_repaired_contractfix_v1/analysis/stage2_prompt_preview_v1.tsv`
- Current prompt preview:
  - `data/results/20260418_9538ec2/02_s2_3/analysis/stage2_prompt_preview_v1.tsv`
- Current evidence blocks:
  - `data/results/20260418_9538ec2/01_s2_2/semantic_stage2_objects/evidence_blocks/<paper>/evidence_blocks_v1.json`

## QLYKLPKT

- input type:
  - `mixed`
  - dominant current behavior: `full_table + selected narrative bridge blocks`
- table completeness:
  - `partial`
- risk level:
  - `high`

Notes:

- Baseline prompt layout:
  - `raw_prefix_then_table_excerpts`
- Current prompt layout:
  - `ordered_controlled_evidence_pack`
- Current selector uses `role_aware_selector_v1`, but the evidence pack still includes noisy table material.
- Current prompt length remains very large:
  - `55442`

## UFXX9WXE

- input type:
  - `mixed`
  - dominant current behavior: `full_table + selected narrative bridge blocks`
- table completeness:
  - `partial`
- risk level:
  - `medium`

Notes:

- Baseline prompt layout:
  - `raw_prefix_then_table_excerpts`
- Current prompt layout:
  - `ordered_controlled_evidence_pack`
- Current prompt is evidence-only and not marked truncated in the prompt preview.
- This is the strongest current example of a useful shift from broad fallback to structured evidence-pack prompting.

## WFDTQ4VX

- input type:
  - `mixed`
  - dominant current behavior: `full_table + multiple narrative blocks`
- table completeness:
  - `partial`
- risk level:
  - `high`

Notes:

- Baseline prompt layout:
  - `raw_prefix_then_table_excerpts`
- Current prompt layout:
  - `ordered_controlled_evidence_pack`
- Current prompt is evidence-only but still very large:
  - `57617`
- The shift to evidence-pack prompting did not produce a clean WFDTQ4VX formulation input surface.

## V99GKZEI

- input type:
  - `mixed`
  - dominant current behavior: `raw-prefix text + table excerpt`
- table completeness:
  - `partial`
- risk level:
  - `high`

Notes:

- Current selector remains `sorted_csv_first_4`
- Current prompt layout remains:
  - `raw_prefix_then_table_excerpts`
- Prompt preview explicitly records:
  - `uses_evidence_pack_only = no`
  - `truncation_detected = yes`
  - `prompt_length = 103485`
- This is the clearest remaining summary/fallback-style prompt path among the audited cases.

## Cross-paper conclusion

- None of the audited papers use `summary` table mode.
- The real shift is not `summary_table -> full_table`.
- The shift is:
  - baseline: `raw_prefix + first four table excerpts`
  - current top-risk papers: `evidence-pack with full-table excerpts plus selected narrative blocks`
- Current status by paper:
  - `QLYKLPKT`: evidence-pack, but still noisy
  - `UFXX9WXE`: evidence-pack, cleaner, medium risk
  - `WFDTQ4VX`: evidence-pack, still high risk
  - `V99GKZEI`: still fallback/raw-prefix style with truncation

## Bottom line

- Current S2-3 inputs are mostly `table-first mixed packs`, not summary tables.
- The exception in practice is `V99GKZEI`, which still behaves like the old fallback prompt path.

## Caveat

- Diagnostic-only.
- Peer review recommended before action.
