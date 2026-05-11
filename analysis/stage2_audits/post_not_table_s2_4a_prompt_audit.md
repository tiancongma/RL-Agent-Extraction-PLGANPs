# Post-NOT_TABLE S2-4a Prompt Audit

## Executive summary

This is a bounded six-paper post-filter audit, not a full DEV15 freeze.

Main outcome:
- the new NOT_TABLE filter reduces prompt bloat **without observed loss of anchor tables**
- improvement is **material but partial**
- row-identity/header structure remains intact in the prompt surface because the prompt contract and renderer were not changed

## Before vs after distribution

- prompt size min / median / max before: `9805 / 16040.5 / 20261`
- prompt size min / median / max after: `9805 / 15100.0 / 19458`
- table count min / median / max before: `6 / 12.5 / 17`
- table count min / median / max after: `6 / 11.5 / 17`

## Before vs after prompt size

- `WFDTQ4VX`: `20261 -> 19288` chars; table blocks `16 -> 15`
- `5ZXYABSU`: `16790 -> 14909` chars; table blocks `14 -> 12`
- `UFXX9WXE`: `19458 -> 19458` chars; table blocks `17 -> 17`
- `INMUTV7L`: `13036 -> 13036` chars; table blocks `11 -> 11`
- `WIVUCMYG`: `9805 -> 9805` chars; table blocks `6 -> 6`
- `5GIF3D8W`: `15291 -> 15291` chars; table blocks `7 -> 7`

## Per-paper classification

| paper_key | before_prompt_size | after_prompt_size | before_table_count | after_table_count | real_key_tables_preserved | pseudo_tables_removed | classification | notes |
|---|---:|---:|---:|---:|---|---|---|---|
| `WFDTQ4VX` | `20261` | `19288` | `16` | `15` | `yes` | `yes` | `NEEDS_COMPACTION` | Prompt shrank slightly; real DOE tables remained, but several pseudo-table-like surfaces still remain. Removed: WFDTQ4VX__table_16__pdf_table.csv, WFDTQ4VX__table_17__pdf_table.csv. Added: WFDTQ4VX__table_08__pdf_table.csv. |
| `5ZXYABSU` | `16790` | `14909` | `14` | `12` | `yes` | `yes` | `LARGE_BUT_SAFE` | Prompt shrank materially; key formulation tables remained and obvious pseudo-tables were removed. Removed: 5ZXYABSU__table_09__pdf_table.csv, 5ZXYABSU__table_10__pdf_table.csv. |
| `UFXX9WXE` | `19458` | `19458` | `17` | `17` | `yes` | `no` | `NEEDS_COMPACTION` | Key Box-Behnken tables remained, but this narrow NOT_TABLE filter did not shrink the prompt pack. |
| `INMUTV7L` | `13036` | `13036` | `11` | `11` | `yes` | `no` | `LARGE_BUT_SAFE` | Guard stayed stable with no prompt change. |
| `WIVUCMYG` | `9805` | `9805` | `6` | `6` | `yes` | `no` | `OK` | DOE guard stayed unchanged and compact. |
| `5GIF3D8W` | `15291` | `15291` | `7` | `7` | `yes` | `no` | `LARGE_BUT_SAFE` | Guard stayed stable with no prompt change. |

## Structure quality (header + row identity)

Evidence from the new frozen prompt audit:
- `WFDTQ4VX`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`
- `5ZXYABSU`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`
- `UFXX9WXE`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`
- `INMUTV7L`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`
- `WIVUCMYG`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`
- `5GIF3D8W`: `all_selected_blocks_included=yes`, `truncation_detected=no`, `status=pass`

Interpretation:
- no prompt truncation was introduced by the filter
- no prompt-layer primary-table bias was reintroduced
- row identity and header/schema preservation remain governed by the existing summary contract, which this patch did not alter

## Whether prompt bloat materially improved

- `5ZXYABSU`: yes, from `16790 -> 14909` chars and `14 -> 12` table blocks
- `WFDTQ4VX`: yes, modestly, from `20261 -> 19288` chars and `16 -> 15` table blocks
- `UFXX9WXE`: no, unchanged at `19458` chars and `17` table blocks
- guards: unchanged, which is acceptable because the filter is deliberately narrow

## Whether any real table seems wrongly removed

- No wrongful anchor-table removal was observed in the bounded slice.
- `WFDTQ4VX` retained real DOE-bearing tables `Table 12` and `Table 13`.
- `5ZXYABSU` retained real key tables `Table 1` and `Table 2`.
- `UFXX9WXE` retained real Box-Behnken tables `Table 10` and `Table 13`.

## FACTS

- The new pre-LLM bounded validation lineage is `07_s2_4a_not_table_validation_rerun2`.
- The new frozen prompt lineage is `08_s2_4a_not_table_prompt_rerun2`.
- Prompt shrinkage occurred only on `WFDTQ4VX` and `5ZXYABSU`.
- Confirmed NOT_TABLE drops in the bounded slice were recorded for candidate tables under `WFDTQ4VX`, `5ZXYABSU`, and `INMUTV7L`.
- `UFXX9WXE`, `WIVUCMYG`, and `5GIF3D8W` had no prompt-table removals in this bounded rerun.

## INFERENCES

- The narrow filter is catching only the most obvious pseudo-table cases, which is why the shrinkage is safe but incomplete.
- Remaining prompt breadth is now dominated by ambiguous or weak-but-preserved tables rather than high-confidence NOT_TABLE noise.

## UNCERTAINTIES

- This audit is bounded to the six-paper slice and does not prove full-DEV15 behavior.
- Prompt quality is still measured mostly by frozen audit fields and table-count/size deltas rather than model-token cost.

