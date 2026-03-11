# Active Pipeline Flow

This file is the script-level execution contract for the current active DEV-15 formulation-instance workflow.

Scope:
- current Stage2 extractor,
- current Stage4 evaluator,
- current reviewer workbook builder,
- current fixed input artifacts and output artifacts,
- actual script order and actual command shapes.

It is not a methodology note and it is not a historical repository inventory.

## Preconditions

- Fixed GT workbook:
  - `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`
- Fixed split manifests:
  - `data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv`
  - `data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv`
- Cleaned text assets already exist under:
  - `data/cleaned/goren_2025/text/...`
  - some scripts also tolerate legacy-compatible text paths under `data/cleaned/content_goren_2025/text/...`

## Step Table

| step_id | stage | script_path | status | purpose | primary_inputs | primary_outputs | upstream_step | downstream_step | canonical_for | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| DEV15_01 | Stage2 | `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | ACTIVE_MAINLINE | Extract formulation-instance weak labels for the fixed tuned 3-paper DEV subset. | `data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv`; cleaned text assets; model access | `data/results/run_20260310_v7pilot3r3fixparse_synthmethod/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`; `...jsonl` | preexisting DEV-15 benchmark assets | DEV15_02 | tuned_3paper extraction refresh | Output directory and run id are example values from the current documented path. |
| DEV15_02 | Stage4 | `src/stage4_eval/eval_weak_labels_v7pilot3.py` | ACTIVE_MAINLINE | Evaluate the fixed tuned 3-paper subset against the DEV formulation-skeleton GT. | tuned-3 weak-label TSV from DEV15_01; `data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv`; GT workbook | `data/cleaned/labels/manual/formulation_instance_pilot3_eval_synthmethod_2026-03-10/per_doi_formulation_instance_summary.tsv`; `.../predicted_instance_rows.tsv`; `docs/methods/formulation_instance_pilot3_eval_synthmethod_2026-03-10.md` | DEV15_01 | DEV15_05 | tuned_3paper evaluation | Current full DEV-15 view reuses this tuned-3 summary as an input artifact. |
| DEV15_03 | Stage2 | `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py` | ACTIVE_MAINLINE | Extract formulation-instance weak labels for the remaining 12 DEV papers. | `data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv`; cleaned text assets; model access | `data/results/run_20260310_dev15_remaining12_synthmethod_merged/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`; `...jsonl` | preexisting DEV-15 benchmark assets | DEV15_04 | remaining_12paper extraction refresh | Output directory and run id are example values from the current documented path. |
| DEV15_04 | Stage4 | `src/stage4_eval/eval_weak_labels_v7pilot3.py` | ACTIVE_MAINLINE | Evaluate the remaining 12 DEV papers and apply the current deterministic WFDTQ4VX DoE coordinate reconciliation. | remaining-12 weak-label TSV from DEV15_03; `data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv`; GT workbook | `data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10_reconciled/per_doi_formulation_instance_summary.tsv`; `.../predicted_instance_rows.tsv`; `docs/methods/formulation_instance_remaining12_eval_2026-03-10_reconciled.md` | DEV15_03 | DEV15_05 | remaining_12paper evaluation | WFDTQ4VX reconciliation is integrated here. Raw and reconciled counts are both preserved in the per-DOI summary. |
| DEV15_05 | Stage4 | `src/stage4_eval/build_dev15_review_workbook_v1.py` | ACTIVE_SUPPORTING | Build the reviewer workbook from the combined DEV-15 evaluation artifact and predicted instance rows. | `data/cleaned/labels/manual/dev15_formulation_skeleton/dev15_formulation_skeleton_review_v1_fixed.xlsx`; `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10.tsv` `NEEDS_CONFIRMATION`; remaining-12 per-DOI summary; tuned-3 per-DOI summary; latest matching weak-label TSVs | `data/results/dev15_review/dev15_instance_review_v1.xlsx` | DEV15_02 and DEV15_04 plus combined DEV artifact assembly `NEEDS_CONFIRMATION` | none | reviewer workbook generation | `INFERRED_FROM_SCRIPT`: this script currently points to the non-reconciled combined TSV. Active runbook names `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv` as the official artifact. There is no canonical checked-in builder script for the combined DEV-15 TSV yet. |

## Final Ordered Execution Sequence

1. Run Stage2 extraction for the fixed tuned 3-paper manifest with `auto_extract_weak_labels_v7pilot_r3_fixparse.py`.
2. Run Stage4 evaluation for the fixed tuned 3-paper manifest with `eval_weak_labels_v7pilot3.py`.
3. Run Stage2 extraction for the remaining 12-paper manifest with `auto_extract_weak_labels_v7pilot_r3_fixparse.py`.
4. Run Stage4 evaluation for the remaining 12-paper manifest with `eval_weak_labels_v7pilot3.py`.
5. Build or confirm the combined DEV-15 TSV artifact.
   - `NEEDS_CONFIRMATION`: there is no canonical checked-in builder script for the full combined DEV-15 TSV.
6. Run `build_dev15_review_workbook_v1.py` to generate the reviewer workbook.

## Current Default DEV-15 Commands

### Step DEV15_01

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'; python src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py --manifest-tsv data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv --model gemini-2.5-flash --max-items 3 --max-chars 50000 --out-dir data/results/run_20260310_v7pilot3r3fixparse_synthmethod/weak_labels_v7pilot_r3_fixparse --verbose
```

### Step DEV15_02

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'; python src/stage4_eval/eval_weak_labels_v7pilot3.py --pilot-tsv data/results/run_20260310_v7pilot3r3fixparse_synthmethod/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv --pilot-manifest data/cleaned/goren_2025/index/splits/dev_manifest_v7pilot3_2026-03-06.tsv --summary-md docs/methods/formulation_instance_pilot3_eval_synthmethod_2026-03-10.md --out-dir data/cleaned/labels/manual/formulation_instance_pilot3_eval_synthmethod_2026-03-10
```

### Step DEV15_03

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'; python src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py --manifest-tsv data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv --model gemini-2.5-flash --max-items 12 --max-chars 50000 --out-dir data/results/run_20260310_dev15_remaining12_synthmethod_merged/weak_labels_v7pilot_r3_fixparse --verbose
```

### Step DEV15_04

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'; python src/stage4_eval/eval_weak_labels_v7pilot3.py --pilot-tsv data/results/run_20260310_dev15_remaining12_synthmethod_merged/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv --pilot-manifest data/cleaned/goren_2025/index/splits/dev_manifest_remaining12_2026-03-10.tsv --summary-md docs/methods/formulation_instance_remaining12_eval_2026-03-10_reconciled.md --out-dir data/cleaned/labels/manual/formulation_instance_remaining12_eval_2026-03-10_reconciled
```

### Step DEV15_05

```powershell
$env:PYTHONPATH='c:\Users\tianc\Downloads\GitHub\RL-Agent-Extraction-PLGANPs'; python src/stage4_eval/build_dev15_review_workbook_v1.py
```

## Fields Marked NEEDS_CONFIRMATION

- Combined DEV-15 TSV builder step between evaluation and workbook generation:
  - no canonical checked-in builder script currently documented
- `src/stage4_eval/build_dev15_review_workbook_v1.py` input combined TSV:
  - script constant points to `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10.tsv`
  - active runbook names `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv` as the official current artifact

---

## Consolidated DEV Split Registry

This section consolidates the active DEV/TEST split registry content.

### DEV vs TEST Policy

- DEV splits are used for iterative pipeline improvement.
- TEST splits must exclude all registered DEV keys for the same dataset.

### Registered Dataset Split: `goren_2025`

- `split_name`: `dev_v1`
- `dev_keys_file`: `data/cleaned/goren_2025/index/splits/dev_keys_v1.tsv`
- `dev_manifest_file`: `data/cleaned/goren_2025/index/splits/dev_manifest_v1.tsv`
- `dev_coverage_file`: `data/cleaned/goren_2025/index/splits/dev_tables_extraction_coverage_v1.tsv`
- `selection_rule`: include all `html_found=True`, then fill with `pdf_found=True && html_found=False`
- `seed`: `13`

### Registered DEV Keys

- `5GIF3D8W`
- `5ZXYABSU`
- `7ZS858NS`
- `BB3JUVW7`
- `BXCV5XWB`
- `INMUTV7L`
- `L3H2RS2H`
- `PA3SPZ28`
- `QLYKLPKT`
- `RHMJWZX8`
- `UFXX9WXE`
- `V99GKZEI`
- `WFDTQ4VX`
- `WIVUCMYG`
- `YGA8VQKU`

### Enforcement Requirement

- TEST split builders must require an exclusion-keys input or load the registered DEV keys automatically in strict mode.
