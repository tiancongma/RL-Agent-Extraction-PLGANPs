# Cross-Machine Sync Strategy

This repository's Mac sync profile is governed by the Stage1 authority rule and
the active data-source contract.

## What To Sync

- `src/`
- `project/`
- `docs/`
- `data/cleaned/index/`
- `data/cleaned/content/`
- `data/cleaned/goren_2025/index/`
- `data/cleaned/goren_2025/tables/`
- `data/cleaned/labels/manual/`
- `data/cleaned/gt_authority/`
- `data/mem/v1/`
- `data/results/ACTIVE_RUN.json`
- the contract-pinned minimal freeze under:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/`
  - keep only:
    - `RUN_CONTEXT.md`
    - `dev15_scope.tsv`
    - `semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
    - `semantic_stage2_objects/semantic_stage2_objects_summary_v1.json`
    - `semantic_to_widerow_adapter/`
    - `formulation_relation_v1/`
    - `final_formulation_table_v1.tsv`
    - `final_output_decision_trace_v1.tsv`
    - `final_output_summary_v1.md`

## What To Exclude

- all other `data/results/run_*`
- child lineage history not explicitly pinned by `data/results/ACTIVE_RUN.json`
- Stage4 diagnostic outputs
- ad hoc `analysis/` run outputs
- diagnostic-only runs
- raw source corpora under `data/raw/`
- local caches, backups, and regenerated exports

## Why This Split

Stage1 cleaned assets are the authoritative reusable upstream inputs.
Stage2 through Stage5 outputs are reproducible and should not be treated as
required production inputs. A small freeze is kept only to support debugging,
comparison, and agent continuity on another machine.

## Reproducing On A Clean Mac

1. Clone the repository and install the Python environment used by this repo.
2. Confirm the Stage1 authorities exist:
   - `data/cleaned/index/manifest_current.tsv`
   - `data/cleaned/index/key2txt.tsv`
   - `data/cleaned/content/`
   - `data/cleaned/goren_2025/tables/`
3. Query memory before deep investigation:
   - `python src/utils/mem_bootstrap_v1.py --query "your task here"`
   - or `python src/utils/query_mem_v1.py --query "your task here"`
4. Run Stage2 from maintained entrypoints only, using Stage1 assets as input:
   - `python src/stage2_sampling_labels/run_stage2_composite_v1.py ...`
5. Regenerate Stage3 through Stage5 from maintained entrypoints as needed.

## Regeneration Rule

- Stage2: `src/stage2_sampling_labels/run_stage2_composite_v1.py`
- Stage3: `src/stage3_relation/build_formulation_relation_artifacts_v1.py`
- Stage5: `src/stage5_benchmark/build_minimal_final_output_v1.py`
- GT compare: `src/stage5_benchmark/compare_final_table_to_gt_v1.py`

Do not infer active sources by recency. Use explicit CLI paths or
`data/results/ACTIVE_RUN.json`.

## Future Agent Instruction

Always query `data/mem/v1/` before reading raw files for complex debugging,
regression investigation, run comparison, GT mismatch analysis, or lineage
tracing.
