# DEV15 GT Authority v1

This directory contains the frozen machine-readable DEV15 GT authority surfaces.

## Source workbooks

- Layer2 source workbook: `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/boundary_gt_review_v1/boundary_gt_review_workbook_v1.xlsx`
- Layer3 source workbook: `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/value_gt_annotation_workbook_representation_repaired_v4_with_pH.xlsx`

## Derivation rules

- Layer1 `dev15_layer1_gt_counts.tsv`: group accepted Layer2 rows by `paper_key` and count rows where `gt_row_decision = include_gt`.
- Layer2 `dev15_layer2_identity.tsv`: copy accepted GT formulation rows from `review_gt_rows` where `gt_row_decision = include_gt`.
- Layer3 `dev15_layer3_values.tsv`: export the `value_gt_annotation` sheet from the approved Layer3 value workbook as TSV.

## Authority rule

- These files are frozen GT authority for DEV15.
- GT-consuming workflows must resolve these files from contract, not by directory scan, heuristic discovery, timestamp, or filename similarity.
- Replacing these files by ad hoc workbook selection, repo scanning, or heuristic path inference is forbidden.

