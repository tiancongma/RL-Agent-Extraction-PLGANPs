# DEV-15 Review Workbook v1

Purpose:
Build a human-review workbook for formulation-instance extraction on the fixed DEV-15 formulation skeleton benchmark.

This workbook is intended for manual inspection of instance-boundary errors, especially:
- under-segmentation, where multiple GT formulations were collapsed
- over-segmentation, where extra predicted formulation candidates were emitted
- contamination by predicted rows that are not valid formulation instances

Input artifacts used:
- `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
- `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10.tsv`
- `data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10/per_doi_formulation_instance_summary.tsv`
- `data/cleaned/labels/manual/formulation_instance_pilot3_eval_synthmethod_2026-03-10/per_doi_formulation_instance_summary.tsv`
- latest matching `weak_labels__v7pilot_r3_fixparse.tsv` files discovered under `data/results/` for:
  - the tuned 3-paper set
  - the remaining 12-paper DEV expansion set

Workbook output:
- `data/results/dev15_review/dev15_instance_review_v1.xlsx`

Sheet definitions:
- `paper_summary`: one row per DEV paper with GT count, predicted formulation count, count difference, error type, and summary notes.
- `predicted_instances`: one row per predicted instance row from the latest weak-label outputs for the DEV papers. This sheet includes instance identity, instance kind, parent instance id, an evidence locator, a short evidence snippet, and an empty `reviewer_decision` column for manual triage.
- `review_queue`: subset of `paper_summary` restricted to non-exact papers. This is the default queue for manual inspection.

How the reviewer should inspect instances:
1. Start with `review_queue` to identify papers with count mismatches.
2. Filter `predicted_instances` by `zotero_key` for the paper under review.
3. Inspect `instance_kind`, `parent_instance_id`, and `evidence_snippet` together to decide whether the predicted row is a valid formulation instance, a merged formulation family, or an invalid extra row.
4. Record a manual decision in `reviewer_decision`, for example `valid_formulation`, `not_a_formulation`, or `needs_second_pass`.
5. Use the paper-level count difference to prioritize the highest-risk papers first.
