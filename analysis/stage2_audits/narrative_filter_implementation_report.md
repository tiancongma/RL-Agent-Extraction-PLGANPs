# Narrative Filter Implementation Report

## Executive summary

This bounded implementation succeeded **partially**.

- A new high-confidence structural `narrative/metadata` removal layer is now active pre-LLM alongside the existing `NOT_TABLE` filter.
- The new layer remains separate from semantic importance logic and only activates on structural narrative/metadata surfaces with no stable table structure.
- All anchor tables remained preserved in the bounded rerun.
- Additional prompt shrinkage was observed only for `UFXX9WXE`, where one author/affiliation front-matter pseudo-table was removed.
- `WFDTQ4VX`, `5ZXYABSU`, and all three guards remained stable but saw no further shrinkage beyond the earlier `NOT_TABLE` pass.

Validation lineages:
- pre-LLM validation: [11_s2_4a_narrative_filter_validation_rerun](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/11_s2_4a_narrative_filter_validation_rerun)
- frozen prompt boundary: [12_s2_4a_narrative_filter_prompt_rerun](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/12_s2_4a_narrative_filter_prompt_rerun)

## What rule was added

Changed file:
- [extract_semantic_stage2_objects_v2.py](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/src/stage2_sampling_labels/extract_semantic_stage2_objects_v2.py)

Main change:
- Added `evaluate_high_confidence_narrative_metadata_surface(...)` as a structural pre-LLM filter family.
- The rule activates only when a surface is narrative/metadata dominated and simultaneously lacks repeated row pattern, stable column/header structure, row identifiers, variable-structure signals, and compact numeric-column patterns.
- Added audit fields: `narrative_filter_applied`, `narrative_filter_reason`, `preserved_due_to_structure`, and `confirmed_narrative_yes_no`.

## Difference vs NOT_TABLE filter

- `NOT_TABLE` remains the broader confirmed-noise gate for obvious non-table fragments such as bibliography fragments, figure-caption fragments, and OCR garbage.
- The new narrative layer is narrower in intent: it targets prose/metadata blocks that still slipped through because they looked cleaner than OCR garbage but still lacked true table structure.
- The new layer does **not** delete weak-but-real tables, numeric tables, DOE/design tables, formulation tables, or mixed table/text blocks with recoverable structure.

## Anchor results

### WFDTQ4VX

- preserved key tables: `Table 12`, `Table 13`
- prompt size: `19288 -> 19288`
- table count: `15 -> 15`
- additional narrative removals: none
- result: stable, no over-cleaning, no extra shrinkage

### 5ZXYABSU

- preserved key tables: `Table 1`, `Table 2`
- prompt size: `14909 -> 14909`
- table count: `12 -> 12`
- additional narrative removals: none
- result: stable, no over-cleaning, no extra shrinkage

### UFXX9WXE

- preserved key tables: `Table 10`, `Table 13`
- prompt size: `19458 -> 18683`
- table count: `17 -> 16`
- confirmed narrative removal:
  - `UFXX9WXE__table_01__pdf_table.csv` -> `author_affiliation_metadata_surface`
- result: modest improvement without losing the key Box-Behnken tables

## Guard results

- `INMUTV7L`: stable, no prompt delta
- `WIVUCMYG`: stable, no prompt delta
- `5GIF3D8W`: stable, no prompt delta

## Residual issues

- The new narrative layer improved only `UFXX9WXE` in this bounded slice.
- `WFDTQ4VX` and `5ZXYABSU` did not shrink further because their remaining pseudo-table breadth is still either already covered by the existing `NOT_TABLE` filter or still ambiguous under a high-confidence structural standard.
- `UFXX9WXE` still contains several narrative/mixed pseudo-table surfaces that did not cross the confirmed-drop threshold.

## What is NOT solved

- This patch does not solve neutral prompt compaction.
- It does not touch prompt semantics, selector ranking, DOE logic, simple-table enumeration, non-DOE sweep recovery, Stage3, Stage5, or `ACTIVE_RUN.json`.
- It does not prove full-DEV15 behavior; this is bounded to the six-paper validation slice.
