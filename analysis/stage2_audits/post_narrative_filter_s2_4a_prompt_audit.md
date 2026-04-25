# Post-Narrative Filter S2-4a Prompt Audit

## Executive summary

This is a bounded six-paper post-filter audit, not a full DEV15 freeze.

Main outcome:
- the new narrative/metadata layer produced **additional prompt shrinkage only for `UFXX9WXE`**
- all anchor tables remained preserved
- no prompt truncation or summary-contract breakage was observed
- overall result is **partial success**, not a full prompt-bloat fix

Validation lineages:
- baseline comparison prompt lineage: [08_s2_4a_not_table_prompt_rerun2](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/08_s2_4a_not_table_prompt_rerun2)
- new prompt lineage: [12_s2_4a_narrative_filter_prompt_rerun](/Users/yeshanwang/projects/RL-Agent-Extraction-PLGANPs/data/results/20260421_9c4a03f/12_s2_4a_narrative_filter_prompt_rerun)

## Before vs after distribution

- prompt size min / median / max before: `9805 / 15100.0 / 19458`
- prompt size min / median / max after: `9805 / 14909.0 / 19288`
- table count min / median / max before: `6 / 11.5 / 17`
- table count min / median / max after: `6 / 11.5 / 16`

## Before vs after prompt size

- `WFDTQ4VX`: `19288 -> 19288` chars; table blocks `15 -> 15`
- `5ZXYABSU`: `14909 -> 14909` chars; table blocks `12 -> 12`
- `UFXX9WXE`: `19458 -> 18683` chars; table blocks `17 -> 16`
- `INMUTV7L`: `13036 -> 13036` chars; table blocks `11 -> 11`
- `WIVUCMYG`: `9805 -> 9805` chars; table blocks `6 -> 6`
- `5GIF3D8W`: `15291 -> 15291` chars; table blocks `7 -> 7`

## Per-paper classification

| paper_key | before_prompt_size | after_prompt_size | before_table_count | after_table_count | real_tables_preserved | narrative_removed | classification | notes |
|---|---:|---:|---:|---:|---|---|---|---|
| `WFDTQ4VX` | `19288` | `19288` | `15` | `15` | `yes` | `no` | `NEEDS_COMPACTION` | prompt_status=pass; truncation_detected=no |
| `5ZXYABSU` | `14909` | `14909` | `12` | `12` | `yes` | `no` | `LARGE_BUT_SAFE` | prompt_status=pass; truncation_detected=no |
| `UFXX9WXE` | `19458` | `18683` | `17` | `16` | `yes` | `yes` | `NEEDS_COMPACTION` | Removed: UFXX9WXE__table_01__pdf_table.csv Narrative drops: UFXX9WXE__table_01__pdf_table.csv (author_affiliation_metadata_surface) prompt_status=pass; truncation_detected=no |
| `INMUTV7L` | `13036` | `13036` | `11` | `11` | `yes` | `no` | `LARGE_BUT_SAFE` | Narrative drops: INMUTV7L__table_04__pdf_table.csv (author_affiliation_metadata_surface) prompt_status=pass; truncation_detected=no |
| `WIVUCMYG` | `9805` | `9805` | `6` | `6` | `yes` | `no` | `OK` | prompt_status=pass; truncation_detected=no |
| `5GIF3D8W` | `15291` | `15291` | `7` | `7` | `yes` | `no` | `LARGE_BUT_SAFE` | prompt_status=pass; truncation_detected=no |

## Whether key tables are preserved

- `WFDTQ4VX`: `Table 12` and `Table 13` remained present
- `5ZXYABSU`: `Table 1` and `Table 2` remained present
- `UFXX9WXE`: `Table 10` and `Table 13` remained present

## Whether pseudo tables were removed

- `UFXX9WXE`: yes
  - removed `UFXX9WXE__table_01__pdf_table.csv`
  - candidate audit reason: `author_affiliation_metadata_surface`
- `WFDTQ4VX`: no additional removals beyond the prior NOT_TABLE baseline
- `5ZXYABSU`: no additional removals beyond the prior NOT_TABLE baseline
- guards: unchanged

## Structure quality

From the new prompt audit:
- every paper still reports `all_selected_blocks_included=yes`
- every paper still reports `truncation_detected=no`
- every paper still reports `status=pass`

Interpretation:
- the narrative layer did not damage the prompt renderer contract
- row identity/header structure remains governed by the unchanged summary path
- no new prompt-layer primary-table bias was introduced

## Whether prompt bloat materially improved

- `UFXX9WXE`: modestly improved
- `WFDTQ4VX`: no additional improvement in this pass
- `5ZXYABSU`: no additional improvement in this pass
- guards: intentionally unchanged

## Whether any real table seems wrongly removed

- No wrongful anchor-table removal was observed in the bounded slice.
- The only new removal in this pass was the `UFXX9WXE` author/affiliation front-matter surface.

## FACTS

- The new pre-LLM validation lineage is `11_s2_4a_narrative_filter_validation_rerun`.
- The new frozen prompt lineage is `12_s2_4a_narrative_filter_prompt_rerun`.
- Additional prompt shrinkage occurred only on `UFXX9WXE`.
- The new confirmed narrative removal recorded in the candidate audit was `UFXX9WXE__table_01__pdf_table.csv -> author_affiliation_metadata_surface`.
- All six papers still passed the frozen prompt audit with no truncation detected.

## INFERENCES

- The structural narrative/metadata rule is safe on the bounded slice, but still very conservative.
- Remaining prompt breadth is still dominated by ambiguous mixed surfaces rather than by the easiest narrative/metadata debris.

## UNCERTAINTIES

- This audit is bounded to the six-paper slice and does not establish full-DEV15 behavior.
- The prompt benefit is measured by character counts and table-block counts, not by downstream extraction quality.
