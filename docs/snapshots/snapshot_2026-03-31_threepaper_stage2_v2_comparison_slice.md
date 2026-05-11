# Snapshot: 2026-03-31 Three-Paper Stage2 v2 Comparison Slice

## Purpose

This snapshot records the implementation of a governed three-paper Stage2 v2
comparison slice for:

- `WIVUCMYG`
- `UFXX9WXE`
- `5GIF3D8W`

The slice exists to enforce the frozen Stage2 architecture split:

- LLM owns open semantic discovery in Stage2
- deterministic layers own downstream relation resolution, inheritance,
  normalization, filtering, and materialization
- deterministic Stage2 semantic replacement remains comparator or fallback
  scope only

This slice does not replace `ACTIVE_RUN` and is not a promoted authority path.

## Implemented scripts

- `src/stage2_sampling_labels/extract_semantic_stage2_v2_threepaper.py`
  - emits object-first Stage2 v2 artifacts only
  - supports truthful replay from saved historical raw LLM responses
  - does not emit final wide rows or perform downstream relation resolution
- `src/analysis/build_stage2_v2_threepaper_comparison_pack.py`
  - compares the new Stage2 v2 slice against:
    - current deterministic semantic active-run artifacts
    - current deterministic compatibility wide-row active-run artifacts
    - historical legacy Stage2 wide-row comparator artifacts
- `src/utils/run_threepaper_stage2_v2_comparison.py`
  - resolves current sources explicitly
  - writes a run-scoped `RUN_CONTEXT.md`
  - produces a reproducible three-paper comparison run without changing
    `data/results/ACTIVE_RUN.json`

## Reproducible run

- run id:
  - `run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1`
- run root:
  - `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/`
- execution mode used in this snapshot:
  - `legacy_llm_replay`

Resolved source artifacts:

- active source run:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/`
- active scope manifest:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/dev15_scope.tsv`
- active deterministic semantic JSONL:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_stage2_objects/semantic_stage2_objects_v1.jsonl`
- active deterministic compatibility TSV:
  - `data/results/run_20260329_1753_63b0c8d_dev15_identity_variable_preservation_exp_v1/semantic_to_widerow_adapter/weak_labels__v7pilot_r3_fixparse.tsv`
- historical replay raw responses:
  - `data/results/run_20260313_0950_f4912f3_dev15_current_merged_benchmark_v1/weak_labels_v7pilot_r3_fixparse/raw_responses/`
- historical legacy comparator TSV:
  - `data/results/run_20260314_1206_076995e_dev15_deterministic_refresh_no_llm_v1/weak_labels_v7pilot_r3_fixparse/weak_labels__v7pilot_r3_fixparse.tsv`

## Output surfaces

Stage2 v2 semantic outputs:

- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/semantic_stage2_v2/semantic_stage2_v2_objects.jsonl`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/semantic_stage2_v2/semantic_stage2_v2_summary.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/semantic_stage2_v2/raw_responses/`

Comparison pack outputs:

- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/paper_level_counts.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/boundary_surface.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/variable_retention.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/measurement_retention.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/comparison_summary.tsv`
- `data/results/run_20260331_1015_03e5d25_threepaper_stage2_v2_comparison_v1/analysis/stage2_v2_threepaper_comparison/comparison_report.md`

## Guardrail statement

- This is a Stage2-only comparison surface.
- It is `diagnostic-only, not benchmark-valid final output`.
- It must not be used as formal GT benchmark evidence for the current system.
- Future promotion would require broader governed evidence beyond this fixed
  three-paper slice.
