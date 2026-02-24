# Step2 EE Table Support Regression

Run: `run_20260219_1623_780eb83_goren18_weaklabels_v1`

## Per-Doc Counts Before vs After
| doc_key   | phase   |   n_formulation_core |   n_EE_value_raw_nonempty |   n_ee_supported |
|:----------|:--------|---------------------:|--------------------------:|-----------------:|
| WIVUCMYG  | before  |                   28 |                        26 |                0 |
| WIVUCMYG  | after   |                   28 |                        26 |               26 |
| WFDTQ4VX  | before  |                   10 |                        27 |                5 |
| WFDTQ4VX  | after   |                   10 |                        27 |                6 |

## EE Fail Reason Distribution (WIVUCMYG, WFDTQ4VX)
### WIVUCMYG
| ee_fail_reason         |   after |   before |
|:-----------------------|--------:|---------:|
| header_context_missing |       0 |        2 |
| missing_span           |       3 |        3 |
| numeric_anchor_fail    |       0 |       24 |
| supported_or_na        |      26 |        0 |

### WFDTQ4VX
| ee_fail_reason      |   after |   before |
|:--------------------|--------:|---------:|
| missing_span        |       1 |        1 |
| numeric_anchor_fail |      21 |       22 |
| supported_or_na     |       6 |        5 |

## UFXX9WXE Core Count Guardrail
- `schema_v2` formulation_core count for `UFXX9WXE`: **26** (expected unchanged at 26).

## Artifacts
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/step2_ee_support_summary__before_after.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/step2_ee_fail_reason_distribution__before_after.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/ee_support_debug__WIVUCMYG__before_legacy.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/ee_support_debug__WIVUCMYG__after_table_block.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/ee_support_debug__WFDTQ4VX__before_legacy.tsv`
- `data/results/run_20260219_1623_780eb83_goren18_weaklabels_v1/debug/ee_support_debug__WFDTQ4VX__after_table_block.tsv`
