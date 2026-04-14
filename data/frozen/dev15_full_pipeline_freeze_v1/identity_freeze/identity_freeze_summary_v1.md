# Identity Freeze Summary v1

IDENTITY_FREEZE_RULE_V1 check at the Stage5 post-materialization boundary.

## Inputs
- upstream identity scaffold: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260412_8517d36\18_full_pipeline_benchmark_dev15_v1\audit\layer2_identity_scaffold_binding_v1\layer2_identity_scaffold_rows_v1.tsv`
- Stage5 final table: `C:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs\data\results\20260412_8517d36\18_full_pipeline_benchmark_dev15_v1\final_formulation_table_v1.tsv`
- paper scope: `5GIF3D8W, 5ZXYABSU, 7ZS858NS, BB3JUVW7, BXCV5XWB, INMUTV7L, L3H2RS2H, PA3SPZ28, QLYKLPKT, RHMJWZX8, UFXX9WXE, V99GKZEI, WFDTQ4VX, WIVUCMYG, YGA8VQKU`

## Rule
- formulation count must remain invariant after identity freeze
- formulation membership must remain invariant after identity freeze
- downstream stages may attach, resolve, and derive fields only
- downstream stages must not implicitly split or merge formulations

## Results
| paper_key | upstream_identity_count | final_table_count | selected_binding_count | row_count_drift_detected | identity_reassignment_detected | unresolved_scaffold_rows | ambiguous_scaffold_rows | violation | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5GIF3D8W | 26 | 1 | 0 | yes | yes | 26 | 0 | yes | fail |
| 5ZXYABSU | 9 | 7 | 0 | yes | yes | 9 | 0 | yes | fail |
| 7ZS858NS | 1 | 1 | 0 | no | yes | 1 | 0 | yes | fail |
| BB3JUVW7 | 12 | 17 | 0 | yes | yes | 12 | 0 | yes | fail |
| BXCV5XWB | 9 | 3 | 0 | yes | yes | 9 | 0 | yes | fail |
| INMUTV7L | 12 | 2 | 0 | yes | yes | 12 | 0 | yes | fail |
| L3H2RS2H | 21 | 4 | 0 | yes | yes | 21 | 0 | yes | fail |
| PA3SPZ28 | 5 | 2 | 0 | yes | yes | 5 | 0 | yes | fail |
| QLYKLPKT | 7 | 2 | 0 | yes | yes | 7 | 0 | yes | fail |
| RHMJWZX8 | 2 | 2 | 0 | no | yes | 2 | 0 | yes | fail |
| UFXX9WXE | 27 | 2 | 0 | yes | yes | 27 | 0 | yes | fail |
| V99GKZEI | 6 | 5 | 0 | yes | yes | 6 | 0 | yes | fail |
| WFDTQ4VX | 30 | 2 | 0 | yes | yes | 30 | 0 | yes | fail |
| WIVUCMYG | 26 | 9 | 0 | yes | yes | 26 | 0 | yes | fail |
| YGA8VQKU | 17 | 4 | 0 | yes | yes | 17 | 0 | yes | fail |
