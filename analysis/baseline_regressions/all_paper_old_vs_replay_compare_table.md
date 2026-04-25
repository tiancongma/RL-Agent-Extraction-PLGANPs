# All-Paper Old vs Replay Compare Table

## Executive Summary

- Scope size: `15` papers.
- Old diagnostic baseline: `data/results/20260421_43ed145`.
- New replay baseline: `data/results/20260421_c8f4b61`.
- Total final rows: `35 -> 117`.
- Matched papers: `1 -> 1`.
- Both compare surfaces use the same raw columns: `final_table_count`, `gt_count`, `count_diff`, `comparison_status`.
- Normalization used here:
  - `old_final_rows` / `new_final_rows` come from `final_table_count`.
  - `gt_rows` comes from `gt_count`.
  - `old_delta_vs_gt` / `new_delta_vs_gt` come from `count_diff`.
  - `old_missing_vs_gt`, `old_extra_vs_gt`, `new_missing_vs_gt`, and `new_extra_vs_gt` are derived deterministically from final count vs GT count because the compare TSV does not provide them directly.

## Focus List

- Biggest improvements:
  - `UFXX9WXE`: `2 -> 28` rows (`change=26`), status `under -> over`.
  - `WIVUCMYG`: `3 -> 29` rows (`change=26`), status `under -> over`.
  - `YGA8VQKU`: `3 -> 19` rows (`change=16`), status `under -> over`.
  - `5GIF3D8W`: `1 -> 11` rows (`change=10`), status `under -> under`.
  - `PA3SPZ28`: `1 -> 4` rows (`change=3`), status `under -> under`.
- Unchanged papers: `5ZXYABSU`, `7ZS858NS`, `BB3JUVW7`, `BXCV5XWB`, `INMUTV7L`, `L3H2RS2H`, `QLYKLPKT`, `RHMJWZX8`, `WFDTQ4VX`
- Papers still severely under GT in the replay (`new_delta_vs_gt <= -10`): `5GIF3D8W` (-15), `L3H2RS2H` (-16), `WFDTQ4VX` (-27)

## Full Table Sorted By Largest Absolute Row-Count Change

| paper_key | old_final_rows | new_final_rows | gt_rows | old_delta_vs_gt | new_delta_vs_gt | row_change | old_compare_status | new_compare_status | changed_yes_no | old_missing_vs_gt | old_extra_vs_gt | new_missing_vs_gt | new_extra_vs_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UFXX9WXE | 2 | 28 | 27 | -25 | 1 | 26 | under | over | yes | 25 | 0 | 0 | 1 |
| WIVUCMYG | 3 | 29 | 26 | -23 | 3 | 26 | under | over | yes | 23 | 0 | 0 | 3 |
| YGA8VQKU | 3 | 19 | 17 | -14 | 2 | 16 | under | over | yes | 14 | 0 | 0 | 2 |
| 5GIF3D8W | 1 | 11 | 26 | -25 | -15 | 10 | under | under | yes | 25 | 0 | 15 | 0 |
| PA3SPZ28 | 1 | 4 | 5 | -4 | -1 | 3 | under | under | yes | 4 | 0 | 1 | 0 |
| V99GKZEI | 3 | 4 | 6 | -3 | -2 | 1 | under | under | yes | 3 | 0 | 2 | 0 |
| 5ZXYABSU | 1 | 1 | 9 | -8 | -8 | 0 | under | under | no | 8 | 0 | 8 | 0 |
| 7ZS858NS | 1 | 1 | 1 | 0 | 0 | 0 | match | match | no | 0 | 0 | 0 | 0 |
| BB3JUVW7 | 3 | 3 | 12 | -9 | -9 | 0 | under | under | no | 9 | 0 | 9 | 0 |
| BXCV5XWB | 1 | 1 | 9 | -8 | -8 | 0 | under | under | no | 8 | 0 | 8 | 0 |
| INMUTV7L | 3 | 3 | 12 | -9 | -9 | 0 | under | under | no | 9 | 0 | 9 | 0 |
| L3H2RS2H | 5 | 5 | 21 | -16 | -16 | 0 | under | under | no | 16 | 0 | 16 | 0 |
| QLYKLPKT | 2 | 2 | 7 | -5 | -5 | 0 | under | under | no | 5 | 0 | 5 | 0 |
| RHMJWZX8 | 3 | 3 | 2 | 1 | 1 | 0 | over | over | no | 0 | 1 | 0 | 1 |
| WFDTQ4VX | 3 | 3 | 30 | -27 | -27 | 0 | under | under | no | 27 | 0 | 27 | 0 |

## Full Table Sorted By paper_key

| paper_key | old_final_rows | new_final_rows | gt_rows | old_delta_vs_gt | new_delta_vs_gt | row_change | old_compare_status | new_compare_status | changed_yes_no | old_missing_vs_gt | old_extra_vs_gt | new_missing_vs_gt | new_extra_vs_gt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5GIF3D8W | 1 | 11 | 26 | -25 | -15 | 10 | under | under | yes | 25 | 0 | 15 | 0 |
| 5ZXYABSU | 1 | 1 | 9 | -8 | -8 | 0 | under | under | no | 8 | 0 | 8 | 0 |
| 7ZS858NS | 1 | 1 | 1 | 0 | 0 | 0 | match | match | no | 0 | 0 | 0 | 0 |
| BB3JUVW7 | 3 | 3 | 12 | -9 | -9 | 0 | under | under | no | 9 | 0 | 9 | 0 |
| BXCV5XWB | 1 | 1 | 9 | -8 | -8 | 0 | under | under | no | 8 | 0 | 8 | 0 |
| INMUTV7L | 3 | 3 | 12 | -9 | -9 | 0 | under | under | no | 9 | 0 | 9 | 0 |
| L3H2RS2H | 5 | 5 | 21 | -16 | -16 | 0 | under | under | no | 16 | 0 | 16 | 0 |
| PA3SPZ28 | 1 | 4 | 5 | -4 | -1 | 3 | under | under | yes | 4 | 0 | 1 | 0 |
| QLYKLPKT | 2 | 2 | 7 | -5 | -5 | 0 | under | under | no | 5 | 0 | 5 | 0 |
| RHMJWZX8 | 3 | 3 | 2 | 1 | 1 | 0 | over | over | no | 0 | 1 | 0 | 1 |
| UFXX9WXE | 2 | 28 | 27 | -25 | 1 | 26 | under | over | yes | 25 | 0 | 0 | 1 |
| V99GKZEI | 3 | 4 | 6 | -3 | -2 | 1 | under | under | yes | 3 | 0 | 2 | 0 |
| WFDTQ4VX | 3 | 3 | 30 | -27 | -27 | 0 | under | under | no | 27 | 0 | 27 | 0 |
| WIVUCMYG | 3 | 29 | 26 | -23 | 3 | 26 | under | over | yes | 23 | 0 | 0 | 3 |
| YGA8VQKU | 3 | 19 | 17 | -14 | 2 | 16 | under | over | yes | 14 | 0 | 0 | 2 |
