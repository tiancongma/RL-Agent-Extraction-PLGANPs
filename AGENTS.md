# Repository Agent Entrypoint

## Project Overview

This repository builds a structured PLGA nanoparticle formulation dataset from scientific literature.

The current workflow is formulation-instance-centered:
- Stage2 extracts formulation candidates from cleaned paper text and tables.
- Stage4 evaluates formulation counts and boundaries against the DEV-15 benchmark.
- Reviewer-facing workbooks and audit artifacts are produced from the evaluated outputs.

## Agent Read Order

Read these files before doing work:

1. `project/ACTIVE_PIPELINE_RUNBOOK.md`
2. `project/ACTIVE_PIPELINE_RUNBOOK.md`
3. `project/PIPELINE_SCRIPT_MAP.md`
4. `project/4_DECISIONS_LOG.md`
5. `docs/tool_index.md`

## Current Active DEV-15 Path

- Stage2 extractor:
  - `src/stage2_sampling_labels/auto_extract_weak_labels_v7pilot_r3_fixparse.py`
- Stage4 evaluator:
  - `src/stage4_eval/eval_weak_labels_v7pilot3.py`
- Reviewer workbook builder:
  - `src/stage4_eval/build_dev15_review_workbook_v1.py`
- Experimental DoE reconciliation validator:
  - `src/stage4_eval/test_doe_coordinate_reconciliation_v1.py`
- Official DEV-15 reconciled artifact:
  - `data/cleaned/labels/manual/formulation_instance_dev15_combined_eval_2026-03-10_reconciled.tsv`

## Agent Rules

- Prefer `ACTIVE_MAINLINE` scripts listed in `project/ACTIVE_PIPELINE_RUNBOOK.md`.
- Do not select scripts based only on filename similarity or suffix order.
- Do not modify Stage2 extraction unless the task explicitly requests it.
- Stage4 reconciliation rules must remain deterministic and auditable.
- Treat scripts not listed as `ACTIVE_MAINLINE` as non-default entrypoints unless the task explicitly requires them.
