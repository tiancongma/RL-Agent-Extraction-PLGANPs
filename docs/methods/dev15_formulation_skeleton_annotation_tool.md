# DEV15 Formulation Skeleton Annotation Tool

## Purpose
This toolchain builds a lightweight, auditable DEV-only review workflow for formulation skeleton ground truth.

Scope:
- confirm formulation count per paper
- keep deterministic formulation IDs
- review source type and minimal locator notes

Non-scope:
- full field-level parameter annotation
- final publication benchmark release

## Inputs
- DEV manifest (auto-detected by default):
  - `data/cleaned/goren_2025/index/splits/dev_manifest_v1.tsv`
- Optional candidate TSV (if available from extraction outputs)

## Outputs
- Candidate scaffold files:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/dev15_formulation_candidates.tsv`
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/candidates/<paper_key>__<doi_slug>__candidates.jsonl`
- Review workbook:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx`
- Validation outputs:
  - `dev15_formulation_skeleton_validation_report_v1.json`
  - `dev15_formulation_skeleton_validation_issues_v1.tsv`
- Final skeleton export:
  - `dev15_formulation_skeleton_gt_v1.tsv`

## Workbook Design (Low-Typing)
- Main sheet: `review_formulations`
- Lookup sheet: `dropdown_options` (hidden)
- Guidance sheet: `instructions`
- Optional metadata sheet: `source_summary`

Reviewer mostly uses dropdowns for:
- `source_type`
- `formulation_exists_gt`
- `formulation_boundary_confidence`
- `review_status`

Prefilled:
- paper metadata (`paper_key`, `doi`, `paper_title`)
- deterministic IDs (`{paper_key}_F01`, `{paper_key}_F02`, ...)
- candidate rows (or default blank rows per paper)

Built-in review helpers:
- freeze panes + filter
- missing-dropdown highlighting
- uncertain/second-pass highlighting
- duplicate formulation ID highlighting per paper
- helper formula columns for incomplete/duplicate status
- protected non-editable columns

## Review Workflow
1. Build scaffold + workbook.
2. Reviewer updates dropdown fields and optional notes.
3. Run validator to produce issue report.
4. Export confirmed rows to clean TSV.

## Commands
```bash
python src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py --overwrite

python src/stage3_gt/validate_dev15_formulation_skeleton_review_v1.py \
  --xlsx data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx

python src/stage3_gt/export_dev15_formulation_skeleton_gt_v1.py \
  --xlsx data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1.xlsx
```

Optional:
```bash
python src/stage3_gt/build_dev15_formulation_skeleton_review_v1.py \
  --candidate-tsv <path_to_candidate_tsv> \
  --default-rows-per-paper 3 \
  --overwrite
```

