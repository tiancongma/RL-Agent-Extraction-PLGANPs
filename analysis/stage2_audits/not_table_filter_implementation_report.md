# NOT_TABLE Filter Implementation Report

## Executive conclusion

The bounded implementation succeeded **partially**.

- High-confidence pseudo-table removal is now active in `S2-2b` and is auditable in candidate/debug surfaces.
- Real anchor tables remained preserved for `WFDTQ4VX`, `5ZXYABSU`, and `UFXX9WXE`.
- Prompt shrinkage was material for `5ZXYABSU` and modest for `WFDTQ4VX`.
- `UFXX9WXE` did not shrink under this narrow rule, which means its remaining prompt breadth is not dominated by high-confidence NOT_TABLE cases.
- Guard papers stayed stable in this bounded rerun.

Validation lineages:
- pre-LLM validation: [07_s2_4a_not_table_validation_rerun2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/07_s2_4a_not_table_validation_rerun2)
- frozen prompt boundary: [08_s2_4a_not_table_prompt_rerun2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/08_s2_4a_not_table_prompt_rerun2)

## What code changed

Changed file:
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Main change:
- Added a narrow `evaluate_high_confidence_not_table_surface(...)` rule family shared by candidate-side and payload-side hard-drop logic.
- Added audit fields: `not_table_filter_applied`, `not_table_reason`, `preserved_due_to_ambiguity`, `confirmed_noise_yes_no`, and `not_table_weak_signals`.

## Exact NOT_TABLE rule family implemented

Allowed high-confidence removals now include only surfaces with strong non-table evidence such as:
- bibliography/reference-like rows that dominate the block
- figure-caption-like prose blocks
- OCR `(cid:...)` garbage blocks with no recoverable tabular anchors
- strongly prose-dominated narrative fragments with no compact row anchors and no header/schema signal
- existing front-matter / journal-page / review-surfaces already covered by the confirmed-noise policy

What this rule intentionally does **not** remove:
- weak but real tables
- messy OCR tables that still preserve compact row anchors or header keywords
- characterization-heavy but still table-like surfaces
- degraded DOE or formulation tables
- ambiguous mixed table/text surfaces

## Before/after anchor results

### WFDTQ4VX

- prompt size: `20261 -> 19288`
- table count: `16 -> 15`
- real key tables preserved: `yes`
- pseudo tables removed: `yes`
- confirmed NOT_TABLE drops:
  - `WFDTQ4VX__candidate_table__10` -> `caption_or_narrative_fragment_surface`
  - `WFDTQ4VX__candidate_table__16` -> `caption_or_narrative_fragment_surface`
- note: Prompt shrank slightly; real DOE tables remained, but several pseudo-table-like surfaces still remain. Removed: WFDTQ4VX__table_16__pdf_table.csv, WFDTQ4VX__table_17__pdf_table.csv. Added: WFDTQ4VX__table_08__pdf_table.csv.

### 5ZXYABSU

- prompt size: `16790 -> 14909`
- table count: `14 -> 12`
- real key tables preserved: `yes`
- pseudo tables removed: `yes`
- confirmed NOT_TABLE drops:
  - `5ZXYABSU__candidate_table__01` -> `caption_or_narrative_fragment_surface`
  - `5ZXYABSU__candidate_table__11` -> `front_matter_or_bibliographic_surface`
  - `5ZXYABSU__candidate_table__12` -> `caption_or_narrative_fragment_surface`
- note: Prompt shrank materially; key formulation tables remained and obvious pseudo-tables were removed. Removed: 5ZXYABSU__table_09__pdf_table.csv, 5ZXYABSU__table_10__pdf_table.csv.

### UFXX9WXE

- prompt size: `19458 -> 19458`
- table count: `17 -> 17`
- real key tables preserved: `yes`
- pseudo tables removed: `no`
- note: Key Box-Behnken tables remained, but this narrow NOT_TABLE filter did not shrink the prompt pack.

## Guard results

- `INMUTV7L`: prompt size `13036 -> 13036`, table count `11 -> 11`, key tables preserved `yes`.
- `WIVUCMYG`: prompt size `9805 -> 9805`, table count `6 -> 6`, key tables preserved `yes`.
- `5GIF3D8W`: prompt size `15291 -> 15291`, table count `7 -> 7`, key tables preserved `yes`.

## Residual risks

- The rule is intentionally conservative, so prompt bloat is reduced only where a pseudo-table is high-confidence non-table.
- `WFDTQ4VX` still carries several ambiguous paragraph-like surfaces because they do not yet meet the high-confidence NOT_TABLE threshold.
- `UFXX9WXE` did not shrink at all, which means a future compaction rule would need to target neutral summary breadth rather than confirmed-noise removal.

## What is still NOT solved

- This patch does not solve neutral prompt compaction.
- It does not touch prompt semantics, selector ranking, DOE logic, simple-table enumeration, non-DOE sweep recovery, Stage3, or Stage5.
- It only removes high-confidence NOT_TABLE pseudo-tables at the pre-LLM preservation boundary.

